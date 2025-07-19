import os 
import math 

from tqdm.auto import tqdm

from einops import rearrange, reduce
from einops.layers.torch import Rearrange

from inspect import isfunction

import torch
from torch import nn, Tensor
import torch.nn.functional as F
import pytorch3d.transforms as transforms 

from m2d.model.transformer_module import Decoder 
from m2d.utils.projection import perspective_projection, perspective_projection_from_cam 
from m2d.data.amass_motion3d_dataset import normalize_pose2d, normalize_pose2d_w_grad, de_normalize_pose2d, run_smplx_model, quat_fk_torch 
from m2d.data.utils_multi_view_2d_motion import extract_smplx2openpose_jnts19_torch, project_3d_random_view

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def linear_beta_schedule(timesteps):
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype = torch.float64)

def cosine_beta_schedule(timesteps, s = 0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype = torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)

# sinusoidal positional embeds

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class LearnedSinusoidalPosEmb(nn.Module):
    """ following @crowsonkb 's lead with learned sinusoidal pos emb """
    """ https://github.com/crowsonkb/v-diffusion-jax/blob/master/diffusion/models/danbooru_128.py#L8 """

    def __init__(self, dim):
        super().__init__()
        assert (dim % 2) == 0
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim))

    def forward(self, x):
        x = rearrange(x, 'b -> b 1')
        freqs = x * rearrange(self.weights, 'd -> 1 d') * 2 * math.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim = -1)
        fouriered = torch.cat((x, fouriered), dim = -1)
        return fouriered

class Transformer2DLiftingDiffusionModel(nn.Module):
    def __init__(
        self,
        opt, 
        d_input_feats,
        d_feats,
        d_model,
        n_dec_layers,
        n_head,
        d_k,
        d_v,
        d_out, 
        max_timesteps,
    ):
        super().__init__()
        
        self.d_feats = d_feats 
        self.d_model = d_model
        self.n_head = n_head
        self.n_dec_layers = n_dec_layers
        self.d_k = d_k 
        self.d_v = d_v 
        self.d_out = d_out 
        self.max_timesteps = max_timesteps 

        self.use_smpl_rep = opt.use_smpl_rep 
        self.use_jnts_in_cam = opt.use_jnts_in_cam 
        self.use_decomposed_jnts_root_in_cam = opt.use_decomposed_jnts_root_in_cam 
        self.add_2d_motion_condition = opt.add_2d_motion_condition 

        self.add_cam_pred = opt.add_cam_pred 

        d_input = d_input_feats 
        
        # Input: BS X D X T 
        # Output: BS X T X D'
        self.motion_transformer = Decoder(d_feats=d_input, d_model=self.d_model, \
            n_layers=self.n_dec_layers, n_head=self.n_head, d_k=self.d_k, d_v=self.d_v, \
            max_timesteps=self.max_timesteps, use_full_attention=True)  

        self.linear_out = nn.Linear(self.d_model, self.d_out)

        if self.add_cam_pred:
            self.cam_linear_out = nn.Linear(self.d_model, 12) # 9 + 3 

        # For noise level t embedding
        dim = 64
        learned_sinusoidal_dim = 16
        time_dim = dim * 4

        learned_sinusoidal_cond = False
        self.learned_sinusoidal_cond = learned_sinusoidal_cond

        if learned_sinusoidal_cond:
            sinu_pos_emb = LearnedSinusoidalPosEmb(learned_sinusoidal_dim)
            fourier_dim = learned_sinusoidal_dim + 1
        else:
            sinu_pos_emb = SinusoidalPosEmb(dim)
            fourier_dim = dim

        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, d_model)
        )

    def forward(self, src, noise_t, cam_extrinsic, \
        condition=None, padding_mask=None, ds=None, \
        betas=None, gender=None, rest_human_offsets=None):
        # src: BS X T X D
        # cam_extrinsic: BS X 4 X 4 
        # noise_t: int 

        if condition is not None:
            src = torch.cat((src, condition), dim=-1)
       
        noise_t_embed = self.time_mlp(noise_t) # BS X d_model 
        noise_t_embed = noise_t_embed[:, None, :] # BS X 1 X d_model 

        bs = src.shape[0]
        num_steps = src.shape[1] + 1

        if padding_mask is None:
            # In training, no need for masking 
            padding_mask = torch.ones(bs, 1, num_steps).to(src.device).bool() # BS X 1 X timesteps

        # Get position vec for position-wise embedding
        pos_vec = torch.arange(num_steps)+1 # timesteps
        pos_vec = pos_vec[None, None, :].to(src.device).repeat(bs, 1, 1) # BS X 1 X timesteps

        data_input = src.transpose(1, 2).detach() # BS X D X T 
        feat_pred, _ = self.motion_transformer(data_input, padding_mask, pos_vec, \
                obj_embedding=noise_t_embed)
        
        output = self.linear_out(feat_pred[:, 1:]) # BS X T X D (18*3)

        if not self.use_smpl_rep:
            # Project 3D joints to 2D for supervision.
            window_size = self.max_timesteps - 1  
            if self.use_decomposed_jnts_root_in_cam:
                pred_root_traj_in_cam = output[:, :, :3] # BS X T X 3 
                pred_local_jpos_in_cam = output[:, :, 3:] # BS X T X (J*3) 

                # Denormalize 
                pred_local_jpos_in_cam = ds.de_normalize_local_jpos_in_cam_min_max(pred_local_jpos_in_cam)
                pred_root_traj_in_cam = ds.de_normalize_root_traj_in_cam_min_max(pred_root_traj_in_cam)

                pred_local_jpos_in_cam = pred_local_jpos_in_cam.reshape(bs, window_size, -1, 3) # BS X T X J X 3 
                pred_root_traj_in_cam = pred_root_traj_in_cam[:, :, None, :]
                denormed_pred_jnts3d = pred_local_jpos_in_cam + pred_root_traj_in_cam # BS X T X J X 3 
                denormed_pred_jnts3d = denormed_pred_jnts3d.reshape(bs*window_size, -1, 3) # (BS*T) X J X 3 
            elif self.use_jnts_in_cam:
                denormed_pred_jnts3d = ds.de_normalize_jnts3d_in_cam_min_max(output) 
                denormed_pred_jnts3d = denormed_pred_jnts3d.reshape(bs*window_size, -1, 3) # (BS*T) X 18 X 3 
            else: # Global 3D joint positions 
                pred_jnts3d = output.reshape(bs, window_size, -1, 3) # BS X T X 18 X 3 
                pred_jnts3d = pred_jnts3d.reshape(bs*window_size, -1, 3) # (BS*T) X 18 X 3 

                # Denormalize 3D positions to do projection 
                denormed_pred_jnts3d = ds.de_normalize_jpos3d_min_max(pred_jnts3d.reshape(-1, 54)) # (BS*T) X 54 
                denormed_pred_jnts3d = denormed_pred_jnts3d.reshape(-1, 18, 3) # (BS*T) X 18 X 3

            # Prepare camera extrinsic 
            if self.add_cam_pred:
                cam_pred_res = self.cam_linear_out(feat_pred[:, 1, :]) # BS X 9 
                # cam_rot6d = cam_pred_res[:, :6] # BS X 6 
                # cam_rot_mat = transforms.rotation_6d_to_matrix(cam_rot6d) # BS X 3 X 3 
                cam_rot_mat = cam_pred_res[:, :9].reshape(-1, 3, 3) # BS X 3 X 3 
                cam_trans = cam_pred_res[:, -3:] # BS X 3 
                cam_rot_mat = cam_rot_mat[:, None, :, :].repeat(1, window_size, 1, 1).reshape(-1, 3, 3) # BS X T X 3 X 3  
                cam_trans = cam_trans[:, None, :].repeat(1, window_size, 1).reshape(-1, 3)
            else:
                cam_rot_mat = cam_extrinsic[:, :3, :3][:, None, :, :].repeat(1, window_size, \
                            1, 1).reshape(-1, 3, 3) # (BS*T) X 3 X 3 
                cam_trans = cam_extrinsic[:, :3, -1][:, None, :].repeat(1, window_size, \
                            1).reshape(-1, 3) # (BS*T) X 3 

            # Currently, don't need perspective projection 
            # pred_jnts2d = denormed_pred_jnts3d[:, :, :2]
            # pred_jnts2d = pred_jnts2d.reshape(bs, window_size, -1, 2)

            if self.use_jnts_in_cam or self.use_decomposed_jnts_root_in_cam:
                pred_jnts2d = perspective_projection_from_cam(denormed_pred_jnts3d)
            else:
                pred_jnts2d = perspective_projection(denormed_pred_jnts3d, cam_rot_mat, cam_trans) # (BS*T) X 18 X 2 
            
            pred_jnts2d = pred_jnts2d.reshape(bs, window_size, -1, 2)

            # Normalize to [-1, 1] 
            normalized_pred_jnts2d = normalize_pose2d(pred_jnts2d) # BS X T X 18 X 2 
            normalized_pred_jnts2d = normalized_pred_jnts2d.reshape(bs, window_size, -1) # BS X T X (18*2)
        
            normalized_pred_jnts2d.clamp_(-1., 1.)
        else: # Hmm, we don't have betas for Pamela dataset. Predict skeleton length? 
            window_size = self.max_timesteps - 1  

            normalized_root_trans = output[:, :, :3] # BS X T X 3 
            local_rot_6d = output[:, :, 3:3+22*6] # BS X T X (22*6) 
            local_rot_6d = local_rot_6d.reshape(bs, -1, 22, 6) # BS X T X 22 X 6 

            # Run SMPL model to get 3D joints 
            root_trans = ds.de_normalize_root_jpos_min_max(normalized_root_trans) # BS X T X 3 

            local_rot_mat = transforms.rotation_6d_to_matrix(local_rot_6d) # BS X T X 22 X 3 X 3 
           
            # For finetuning on YouTube, using random rest_human_offsets from AMASS
            if rest_human_offsets.shape[0] < bs:
                rep_rest_human_offsets = rest_human_offsets[0:1].repeat(bs-rest_human_offsets.shape[0], 1, 1) 
                padded_rest_human_offsets = torch.cat((rest_human_offsets, \
                                    rep_rest_human_offsets), dim=0) 
                curr_seq_local_jpos = padded_rest_human_offsets[:, None, :, :].repeat(1, \
                        local_rot_mat.shape[1], 1, 1).cuda() # BS X T X 24 X 3 
            elif rest_human_offsets.shape[0] > bs: 
                curr_seq_local_jpos = rest_human_offsets[:bs][:, None, :, :].repeat(1, \
                        local_rot_mat.shape[1], 1, 1).cuda() # BS X T X 24 X 3 
            else:
                curr_seq_local_jpos = rest_human_offsets[:, None, :, :].repeat(1, \
                        local_rot_mat.shape[1], 1, 1).cuda() # BS X T X 24 X 3  
            curr_seq_local_jpos[:, :, 0, :] = root_trans # BS X T X 3 

            local_rot_mat = local_rot_mat.reshape(-1, 22, 3, 3) # (BS*T) X 22 X 3 X 3 
            curr_seq_local_jpos = curr_seq_local_jpos.reshape(-1, 24, 3) # (BS*T) X 24 X 3 
            _, human_jnts = quat_fk_torch(local_rot_mat, curr_seq_local_jpos)
            # human_jnts: (BS*T) X 24 X 3 

            # actually 13 joints since other joints are vertices. 
            smpl2openpose_idx_jnts13 = [12, 17, 19, 21, 16, 18, 20, 2, 5, 8, 1, 4, 7] 
            smplx_jnts18 = human_jnts[:, smpl2openpose_idx_jnts13, :] # (BS*T) X 13 X 3 

            # curr_joint_seq, curr_vert_seq, mesh_faces = \
            #     run_smplx_model(root_trans, local_pose_aa, betas, \
            #     gender, ds.bm_dict) 
            # # BS X T X 24 X 3 
            # curr_joint_seq = curr_joint_seq.reshape(bs*window_size, -1, 3) # (BS*T) X 24 X 3 
            # curr_vert_seq = curr_vert_seq.reshape(bs*window_size, -1, 3) # (BS*T) X Nv X 3 

            # smplx2openpose_jnts19 = extract_smplx2openpose_jnts19_torch(curr_joint_seq, \
            #         curr_vert_seq) 
            # smplx_jnts18 = torch.cat((smplx2openpose_jnts19[:, :8], \
            #         smplx2openpose_jnts19[:, 9:]), dim=1) # (BS*T) X 18 X 3 

            # Prepare camera extrinsic 
            if self.add_cam_pred:
                cam_pred_res = self.cam_linear_out(feat_pred[:, 1, :]) # BS X 9 
                # cam_rot6d = cam_pred_res[:, :6] # BS X 6 
                # cam_rot_mat = transforms.rotation_6d_to_matrix(cam_rot6d) # BS X 3 X 3 
                cam_rot_mat = cam_pred_res[:, :9].reshape(-1, 3, 3) # BS X 3 X 3 
                cam_trans = cam_pred_res[:, -3:] # BS X 3 
                cam_rot_mat = cam_rot_mat[:, None, :, :].repeat(1, window_size, 1, 1).reshape(-1, 3, 3) # BS X T X 3 X 3  
                cam_trans = cam_trans[:, None, :].repeat(1, window_size, 1).reshape(-1, 3)
            else:
                cam_rot_mat = cam_extrinsic[:, :3, :3][:, None, :, :].repeat(1, window_size, 1, 1).reshape(-1, 3, 3) # (BS*T) X 3 X 3 
                cam_trans = cam_extrinsic[:, :3, -1][:, None, :].repeat(1, window_size, 1).reshape(-1, 3) # (BS*T) X 3 

            pred_jnts2d = perspective_projection(smplx_jnts18, cam_rot_mat, cam_trans) # (BS*T) X 18 X 2 
            pred_jnts2d = pred_jnts2d.reshape(bs, window_size, -1, 2)

            # Normalize to [-1, 1] 
            normalized_pred_jnts2d = normalize_pose2d(pred_jnts2d) # BS X T X 18 X 2 
            normalized_pred_jnts2d = normalized_pred_jnts2d.reshape(bs, window_size, -1) # BS X T X (18*2)

            normalized_pred_jnts2d.clamp_(-1., 1.)

        # return normalized_pred_jnts2d, pred_jnts3d.reshape(bs, window_size, -1, 3) 
        return normalized_pred_jnts2d, output, cam_rot_mat, cam_trans  

class CondGaussianDiffusion(nn.Module):
    def __init__(
        self,
        opt,
        d_feats,
        d_model,
        n_head,
        n_dec_layers,
        d_k,
        d_v,
        max_timesteps,
        out_dim,
        timesteps = 1000,
        loss_type = 'l1',
        objective = 'pred_noise',
        beta_schedule = 'cosine',
        p2_loss_weight_gamma = 0., # p2 loss weight, from https://arxiv.org/abs/2204.00227 - 0 is equivalent to weight of 1 across time - 1. is recommended
        p2_loss_weight_k = 1,
        batch_size=None,
        input_first_human_pose=False, 
    ):
        super().__init__()

        if input_first_human_pose or opt.add_2d_motion_condition:
            d_input_feats = 2*d_feats
        else:
            d_input_feats = d_feats 

        self.use_smpl_rep = opt.use_smpl_rep 
        self.use_decomposed_jnts_root_in_cam = opt.use_decomposed_jnts_root_in_cam 
        if self.use_smpl_rep:
            d_out = 3 + 22 * 6 
        elif self.use_decomposed_jnts_root_in_cam:
            d_out = 3 + 18 * 3 
        else:
            d_out = 18 * 3 

        self.add_sds_loss = opt.add_sds_loss 
        
        self.add_cam_pred = opt.add_cam_pred 
        
        self.denoise_fn = Transformer2DLiftingDiffusionModel(opt, d_input_feats=d_input_feats, \
                    d_feats=d_feats, d_model=d_model, n_head=n_head, \
                    d_k=d_k, d_v=d_v, n_dec_layers=n_dec_layers, d_out=d_out, \
                    max_timesteps=max_timesteps) 
        # Input condition and noisy motion, noise level t, predict gt motion
        
        self.objective = objective

        self.seq_len = max_timesteps - 1 
        self.out_dim = out_dim 

        if beta_schedule == 'linear':
            betas = linear_beta_schedule(timesteps)
        elif beta_schedule == 'cosine':
            betas = cosine_beta_schedule(timesteps)
        else:
            raise ValueError(f'unknown beta schedule {beta_schedule}')

        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value = 1.)

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.loss_type = loss_type

        # helper function to register buffer from float64 to float32

        register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32))

        register_buffer('betas', betas)
        register_buffer('alphas_cumprod', alphas_cumprod)
        register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others

        register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)

        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)

        register_buffer('posterior_variance', posterior_variance)

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain

        register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min =1e-20)))
        register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

        # calculate p2 reweighting

        register_buffer('p2_loss_weight', (p2_loss_weight_k + alphas_cumprod / (1 - alphas_cumprod)) ** -p2_loss_weight_gamma)

    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )
    
    def predict_noise_from_start(self, x_t, t, x_start):
        return (
            (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x_start) /
            (extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape))
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, t, cam_extrinsic, x_cond, padding_mask, clip_denoised, ds, \
        betas=None, gender=None, rest_human_offsets=None):
        # x_all = torch.cat((x, x_cond), dim=-1)
        # model_output = self.denoise_fn(x_all, t)

        model_output, pred_j3d, cam_rot_mat, cam_trans = self.denoise_fn(x, t, cam_extrinsic, x_cond, padding_mask, \
            ds=ds, betas=betas, gender=gender, rest_human_offsets=rest_human_offsets)

        if self.objective == 'pred_noise':
            x_start = self.predict_start_from_noise(x, t = t, noise = model_output)
        elif self.objective == 'pred_x0':
            x_start = model_output
        else:
            raise ValueError(f'unknown objective {self.objective}')

        if clip_denoised:
            x_start.clamp_(-1., 1.)

        model_mean, posterior_variance, posterior_log_variance = \
            self.q_posterior(x_start=x_start, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance, pred_j3d 

    @torch.no_grad()
    def p_sample(self, x, t, cam_extrinsic, x_cond=None, padding_mask=None, \
        clip_denoised=True, ds=None, betas=None, gender=None, rest_human_offsets=None):
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance, pred_j3d = self.p_mean_variance(x=x, t=t, cam_extrinsic=cam_extrinsic, \
            x_cond=x_cond, padding_mask=padding_mask, clip_denoised=clip_denoised, ds=ds, \
            betas=betas, gender=gender, rest_human_offsets=rest_human_offsets)
        noise = torch.randn_like(x)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise, pred_j3d 

    @torch.no_grad()
    def p_sample_loop(self, shape, x_start, cam_extrinsic, x_cond=None, padding_mask=None, \
        ds=None, betas=None, gender=None, rest_human_offsets=None):
        device = self.betas.device

        b = shape[0]
        x = torch.randn(shape, device=device)

        for i in tqdm(reversed(range(0, self.num_timesteps)), desc='sampling loop time step', total=self.num_timesteps):
            x, pred_j3d = self.p_sample(x, torch.full((b,), i, device=device, dtype=torch.long), cam_extrinsic, \
                x_cond, padding_mask=padding_mask, ds=ds, \
                betas=betas, gender=gender, rest_human_offsets=rest_human_offsets)    

        return x, pred_j3d # BS X T X D, BS X T X 18 X 3 

    @torch.no_grad()
    def sample(self, x_start, cam_extrinsic, cond_mask=None, padding_mask=None, ds=None, \
        betas=None, gender=None, rest_human_offsets=None): 
        # naive conditional sampling by replacing the noisy prediction with input target data. 
        self.denoise_fn.eval() 
       
        if cond_mask is not None:
            x_pose_cond = x_start * (1. - cond_mask)
            x_cond = x_pose_cond 
        else:
            x_cond = None 

        sample_res, pred_j3d = self.p_sample_loop(x_start.shape, \
                x_start, cam_extrinsic, x_cond, padding_mask, ds=ds, \
                betas=betas, gender=gender, rest_human_offsets=rest_human_offsets)
        # BS X T X D
            
        self.denoise_fn.train()

        return sample_res, pred_j3d   

    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    @property
    def loss_fn(self):
        if self.loss_type == 'l1':
            return F.l1_loss
        elif self.loss_type == 'l2':
            return F.mse_loss
        else:
            raise ValueError(f'invalid loss type {self.loss_type}')

    def get_sds_loss(self, pred_j3d, padding_mask, diffusion_2d_trainer):
        # pred_j3d: BS X T X J X 3 

        bs = pred_j3d.shape[0]
        t = torch.randint(0, self.num_timesteps, (bs,), device=pred_j3d.device).long()

        # Random sample novel view 2D pose 
        pred_j2d, vis_2d_mask, novel_cam_extrinsic = project_3d_random_view(pred_j3d) 
        # BS X T X D (J*2), BS X T, BS X T X 4 X 4  

        # Normalize pred_j2d 
        curr_seq_len = pred_j2d.shape[1]
        pred_j2d = normalize_pose2d_w_grad(pred_j2d.reshape(bs, curr_seq_len, -1, 2))
        pred_j2d = pred_j2d.reshape(bs, curr_seq_len, -1)

        noise = torch.randn_like(pred_j2d) 
        noise_pred_j2d = diffusion_2d_trainer.ema.ema_model.q_sample(x_start=pred_j2d, t=t, noise=noise)

        x_start_pred = diffusion_2d_trainer.ema.ema_model.denoise_fn(noise_pred_j2d, t, \
            padding_mask=padding_mask)
        noise_pred = self.predict_noise_from_start(noise_pred_j2d, t, x_start=x_start_pred)

        grad = extract(self.p2_loss_weight, t, noise_pred.shape) * (noise_pred - noise)
        grad = torch.nan_to_num(grad)

        targets = (pred_j2d - grad).detach()
        loss = 0.5 * F.mse_loss(pred_j2d.float(), targets, \
            reduction='sum') / pred_j2d.shape[0]

        return loss 

    def p_losses(self, x_start, t, cam_extrinsic, gt_j3d, x_cond=None, noise=None, \
            padding_mask=None, ds=None, betas=None, gender=None, rest_human_offsets=None):
        # x_start: BS X T X D
        # cam_extrinsic: BS X 4 X 4 
        # x_cond: BS X T X D_cond
        # padding_mask: BS X 1 X T 
        b, timesteps, d_input = x_start.shape # BS X T X D(3+n_joints*4)
        noise = default(noise, lambda: torch.randn_like(x_start))

        if cam_extrinsic.shape[0] < b:
            padding_cam = cam_extrinsic[0:1].repeat(b-cam_extrinsic.shape[0], 1, 1) 
            cam_extrinsic = torch.cat((cam_extrinsic, padding_cam), dim=0)
        elif cam_extrinsic.shape[0] > b:
            cam_extrinsic = cam_extrinsic[:b] 

        if gt_j3d.shape[0] < b:
            padding_j3d = gt_j3d[0:1].repeat(b-gt_j3d.shape[0], 1, 1) 
            gt_j3d = torch.cat((gt_j3d, padding_j3d), dim=0) 
        elif gt_j3d.shape[0] > b:
            gt_j3d = gt_j3d[:b] 

        x = self.q_sample(x_start=x_start, t=t, noise=noise) # noisy motion in noise level t. 
            
        model_out, pred_j3d, cam_rot_mat, cam_trans = self.denoise_fn(x, t, cam_extrinsic, x_cond, padding_mask, \
                    ds=ds, betas=betas, gender=gender, rest_human_offsets=rest_human_offsets)

        # Add SDS loss for projected 2D motion from novel viewpoints 
        if self.add_sds_loss:
            loss_sds = self.get_sds_loss(pred_j3d) 

        if self.objective == 'pred_noise':
            target = noise
        elif self.objective == 'pred_x0':
            target = x_start
        else:
            raise ValueError(f'unknown objective {self.objective}')

        if padding_mask is not None:
            loss = self.loss_fn(model_out, target, reduction = 'none') * padding_mask[:, 0, 1:][:, :, None]
            loss_3d = self.loss_fn(pred_j3d.reshape(b, timesteps, -1), \
                gt_j3d.reshape(b, timesteps, -1), reduction = 'none') * padding_mask[:, 0, 1:][:, :, None]
           
            gt_cam_rot_mat = cam_extrinsic[:, :3, :3][:, None, :, :].repeat(1, \
                        x.shape[1], 1, 1) # BS X 3 X 3 -> BS X T X 3 X 3 
            gt_cam_trans = cam_extrinsic[:, :3, 3].repeat(1, x.shape[1], 1) # BS X 3 -> BS X T X 3  
            loss_cam_rot = self.loss_fn(cam_rot_mat.reshape(b, timesteps, -1), \
                        gt_cam_rot_mat.reshape(b, timesteps, -1), reduction='none') * padding_mask[:, 0, 1:][:, :, None]
            loss_cam_trans = self.loss_fn(cam_trans.reshape(b, timesteps, -1), \
                        gt_cam_trans.reshape(b, timesteps, -1), reduction='none') * padding_mask[:, 0, 1:][:, :, None]
        else:
            loss = self.loss_fn(model_out, target, reduction = 'none') # BS X T X D 
            loss_3d = self.loss_fn(pred_j3d.reshape(b, timesteps, -1), \
                gt_j3d.reshape(b, timesteps, -1), reduction = 'none')
            
            
            gt_cam_rot_mat = cam_extrinsic[:, :3, :3][:, None, :, :].repeat(1, \
                        x.shape[1], 1, 1) # BS X 3 X 3 -> BS X T X 3 X 3 
            gt_cam_trans = cam_extrinsic[:, :3, 3].repeat(1, x.shape[1], 1) # BS X 3 -> BS X T X 3  
            loss_cam_rot = self.loss_fn(cam_rot_mat.reshape(b, timesteps, -1), \
                        gt_cam_rot_mat.reshape(b, timesteps, -1), reduction='none')
            loss_cam_trans = self.loss_fn(cam_trans.reshape(b, timesteps, -1), \
                        gt_cam_trans.reshape(b, timesteps, -1), reduction='none')

        loss = reduce(loss, 'b ... -> b (...)', 'mean') # BS X (T*D)
        loss = loss * extract(self.p2_loss_weight, t, loss.shape)

        loss_3d = reduce(loss_3d, 'b ... -> b (...)', 'mean') # BS X (T*D)
        loss_3d = loss_3d * extract(self.p2_loss_weight, t, loss_3d.shape)

        loss_cam_rot = reduce(loss_cam_rot, 'b ... -> b (...)', 'mean') # BS X (T*D)
        loss_cam_rot = loss_cam_rot * extract(self.p2_loss_weight, t, loss_cam_rot.shape)

        loss_cam_trans = reduce(loss_cam_trans, 'b ... -> b (...)', 'mean') # BS X (T*D)
        loss_cam_trans = loss_cam_trans * extract(self.p2_loss_weight, t, loss_cam_trans.shape)
        
        return loss.mean(), loss_3d.mean(), loss_cam_rot.mean(), loss_cam_trans.mean()  

    def forward(self, x_start, cam_extrinsic, gt_j3d, cond_mask=None, padding_mask=None, ds=None, \
        betas=None, gender=None, rest_human_offsets=None):
        # x_start: BS X T X D 
        # cam_extrinsic: BS X 4 X 4 
        # gt_j3d: BS X T X 18 X 3, normalized (in global space) 
        # or gt_j3d: BS X T X D(18*3) unnormalized/ BS X T X D(3+18*3) unnormalized in camera coord
        bs = x_start.shape[0] 
        t = torch.randint(0, self.num_timesteps, (bs,), device=x_start.device).long()

        if cond_mask is not None:
            x_pose_cond = x_start * (1. - cond_mask) 
            x_cond = x_pose_cond 
        else:
            x_cond = None 
        
        curr_loss, curr_loss_3d, curr_loss_cam_rot, curr_loss_cam_trans = self.p_losses(x_start, t, cam_extrinsic, gt_j3d, \
            x_cond=x_cond, padding_mask=padding_mask, ds=ds, betas=betas, gender=gender, \
            rest_human_offsets=rest_human_offsets)  

        return curr_loss, curr_loss_3d, curr_loss_cam_rot, curr_loss_cam_trans  
        