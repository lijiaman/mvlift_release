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

from m2d.model.transformer_module import Decoder, MultiViewDecoder 

from m2d.data.utils_2d_pose_normalize import de_normalize_pose2d, normalize_pose2d, normalize_pose2d_w_grad  
from m2d.data.utils_multi_view_2d_motion import perspective_projection, cal_min_seq_len_from_mask_torch 
from m2d.vis.vis_jnts import plot_3d_motion, gen_2d_motion_vis, reverse_perspective_projection 

from m2d.utils.projection import camera_to_world_jnts3d

from m2d.model.smpl_fit2joints3d import SMPLRegressor
from m2d.model.smpl_aligner import SMLPFitter 

from m2d.model.smal_fitter import SmalFitter 
from m2d.model.smal_fitter_2d import SmalFitter2D 

# from m2d.model.smpl_aligner_w_vposer import SMLPVPoserAligner 

from torch.optim import Adam

from datetime import datetime

import random 

import numpy as np 
from itertools import combinations

import trimesh 

import wandb 

from torch.optim.lr_scheduler import MultiStepLR

from m2d.model.smpl_w_vposer_fitter import SmplVPoserFitter
from m2d.model.smpl_w_vposer_fitter_2d import SmplVPoser2DFitter

from m2d.model.object_fitter import ObjectFitter 
from m2d.model.object_fitter_2d import ObjectFitter2D 


import copy 

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
        
class TransformerDiffusionModel(nn.Module):
    def __init__(
        self,
        d_input_feats,
        d_feats,
        d_model,
        n_dec_layers,
        n_head,
        d_k,
        d_v,
        max_timesteps,
    ):
        super().__init__()
        
        self.d_feats = d_feats 
        self.d_model = d_model
        self.n_head = n_head
        self.n_dec_layers = n_dec_layers
        self.d_k = d_k 
        self.d_v = d_v 
        self.max_timesteps = max_timesteps 

        # Input: BS X D X T 
        # Output: BS X T X D'
        self.motion_transformer = Decoder(d_feats=d_input_feats, d_model=self.d_model, \
            n_layers=self.n_dec_layers, n_head=self.n_head, d_k=self.d_k, d_v=self.d_v, \
            max_timesteps=self.max_timesteps, use_full_attention=True)  

        self.linear_out = nn.Linear(self.d_model, self.d_feats)

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

        gender = "male" 
        # self.skel_model = SKEL(gender).cuda() 

    def forward(self, src, noise_t, condition=None, \
        padding_mask=None, language_embedding=None):
        # src: BS X T X D
        # noise_t: int 
        if condition is not None:
            src = torch.cat((src, condition), dim=-1)
       
        noise_t_embed = self.time_mlp(noise_t) # BS X d_model 
        if language_embedding is not None:
            noise_t_embed += language_embedding # BS X d_model 
        noise_t_embed = noise_t_embed[:, None, :] # BS X 1 X d_model 

        bs = src.shape[0]
        num_steps = src.shape[1] + 1

        if padding_mask is None:
            # In training, no need for masking 
            padding_mask = torch.ones(bs, 1, num_steps).to(src.device).bool() # BS X 1 X timesteps

        # Get position vec for position-wise embedding
        pos_vec = torch.arange(num_steps)+1 # timesteps
        pos_vec = pos_vec[None, None, :].to(src.device).repeat(bs, 1, 1) # BS X 1 X timesteps

        data_input = src.transpose(1, 2) # BS X D X T 
       
        feat_pred, _ = self.motion_transformer(data_input, padding_mask, pos_vec, \
                obj_embedding=noise_t_embed)
        
        output = self.linear_out(feat_pred[:, 1:]) # BS X T X D

        return output # predicted noise, the same size as the input 

class TransformerMultiViewDiffusionModel(nn.Module):
    def __init__(
        self,
        d_input_feats,
        d_feats,
        d_model,
        n_dec_layers,
        n_head,
        d_k,
        d_v,
        max_timesteps,
        num_views=6, 
    ):
        super().__init__()
        
        self.d_feats = d_feats 
        self.d_model = d_model
        self.n_head = n_head
        self.n_dec_layers = n_dec_layers
        self.d_k = d_k 
        self.d_v = d_v 
        self.max_timesteps = max_timesteps 

        self.num_views = num_views 

        # Input: BS X K X D X T 
        # Output: BS X K X T X D'
        self.motion_transformer = MultiViewDecoder(d_feats=d_input_feats, d_model=self.d_model, \
            n_layers=self.n_dec_layers, n_head=self.n_head, d_k=self.d_k, d_v=self.d_v, \
            max_timesteps=self.max_timesteps, use_full_attention=True, num_views=self.num_views)  

        self.linear_out = nn.Linear(self.d_model, self.d_feats)

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

    def forward(self, src, noise_t, condition=None, \
        padding_mask=None, language_embedding=None):
        # src: (BS*K) X T X D
        # noise_t: int 
        if condition is not None:
            src = torch.cat((src, condition), dim=-1)
       
        noise_t_embed = self.time_mlp(noise_t) # BS X d_model 
        if language_embedding is not None:
            noise_t_embed += language_embedding # BS X d_model 
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
        
        output = self.linear_out(feat_pred[:, 1:]) # BS X T X D

        # output.clamp_(-1., 1.)

        return output # predicted noise, the same size as the input 

class MultiViewCondGaussianDiffusion(nn.Module):
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
        input_line_cond=False, 
        num_views=6, 
        use_animal_data=False,
        use_bb_data=False, 
        use_omomo_data=False,
        omomo_object_name="largebox", 
    ):
        super().__init__()

        self.opt = opt 

        self.use_animal_data = use_animal_data 

        self.use_omomo_data = use_omomo_data 

        d_input_feats = d_feats * 2 # Input reference 2d pose sequence as condition 
        self.input_line_cond = input_line_cond
        if self.input_line_cond:
            d_input_feats += 18*3 

        self.num_views = num_views 
        self.denoise_fn = TransformerMultiViewDiffusionModel(d_input_feats=d_input_feats, d_feats=d_feats, \
                    d_model=d_model, n_head=n_head, \
                    d_k=d_k, d_v=d_v, n_dec_layers=n_dec_layers, max_timesteps=max_timesteps, num_views=self.num_views) 
        # Input condition and noisy motion, noise level t, predict gt motion
        
        self.objective = objective

        self.seq_len = max_timesteps - 1 
        self.out_dim = out_dim 

        self.clip_encoder = nn.Sequential(
            nn.Linear(in_features=512, out_features=d_model),
            )

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

        self.use_animal_data = use_animal_data 
        self.use_omomo_data = use_omomo_data 

        
        self.fitter = SmplVPoserFitter()

        self.fitter_2d = SmplVPoser2DFitter() 

        if self.use_animal_data:
            self.smal_fitter = SmalFitter() 

            self.smal_fitter2d = SmalFitter2D() 
        elif self.use_omomo_data:
            self.object_fitter = ObjectFitter(object_name=omomo_object_name)

            self.object_fitter_2d = ObjectFitter2D(object_name=omomo_object_name)

    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, t, x_cond, padding_mask, language_embedding, clip_denoised):
        model_output = self.denoise_fn(x, t, x_cond, padding_mask, \
            language_embedding=language_embedding)

        if self.objective == 'pred_noise':
            x_start = self.predict_start_from_noise(x, t = t, noise = model_output)
        elif self.objective == 'pred_x0':
            x_start = model_output
        else:
            raise ValueError(f'unknown objective {self.objective}')

        if clip_denoised:
            x_start.clamp_(-1., 1.)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_start, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance

    # @torch.no_grad()
    def p_sample(self, x, t, x_cond=None, padding_mask=None, \
        language_embedding=None, clip_denoised=True):
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance = self.p_mean_variance(x=x, t=t, x_cond=x_cond, \
            padding_mask=padding_mask, language_embedding=language_embedding, \
            clip_denoised=clip_denoised)
        noise = torch.randn_like(x)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    # @torch.no_grad()
    def p_sample_loop(self, shape, x_start, x_cond=None, \
        padding_mask=None, language_embedding=None):
        device = self.betas.device

        b = shape[0]
        x = torch.randn(shape, device=device)

        for i in tqdm(reversed(range(0, self.num_timesteps)), desc='sampling loop time step', total=self.num_timesteps):
            x = self.p_sample(x, torch.full((b,), i, device=device, dtype=torch.long), \
                x_cond, padding_mask=padding_mask, language_embedding=language_embedding)    

        return x # (BS*K) X T X D

    # @torch.no_grad()
    def sample(self, x_start, cond_data, padding_mask=None, \
        language_input=None, input_line_cond=None):
        # naive conditional sampling by replacing the noisy prediction with input target data. 
        self.denoise_fn.eval() 
        self.clip_encoder.eval()
       
        x_cond = cond_data 

        language_embedding = None 

        sample_res = self.p_sample_loop(x_start.shape, \
                x_start, x_cond, padding_mask, language_embedding=language_embedding)
        # BS X T X D
            
        self.denoise_fn.train()
        self.clip_encoder.train() 

        return sample_res  

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

    def compute_joint2line_distance_loss(self, pred_joints, line_coeffs, padding_mask, compute_for_each_view=False):
        """
        Computes the distance from predicted joint positions to their corresponding lines.

        Parameters:
        - pred_joints (torch.Tensor): Predicted joint positions with shape (BS x T x J x 2)
        - line_coeffs (torch.Tensor): Line coefficients with shape (BS x T x J x 3)

        Returns:
        - torch.Tensor: Mean distance loss across all batches, timesteps, and joints.
        """

        if pred_joints.shape[-1] != 2:
            bs, seq_len, _ = pred_joints.shape 
            pred_joints = pred_joints.reshape(bs, seq_len, -1, 2) 

        # Extract line coefficients
        a = line_coeffs[..., 0]  # Shape: (BS x T x J)
        b = line_coeffs[..., 1]  # Shape: (BS x T x J)
        c = line_coeffs[..., 2]  # Shape: (BS x T x J)

        # Extract joint coordinates
        x = pred_joints[..., 0]  # Shape: (BS x T x J)
        y = pred_joints[..., 1]  # Shape: (BS x T x J)

        # Calculate absolute distance using the line equation ax + by + c
        distances = torch.abs(a * x + b * y + c)  # Shape: (BS x T x J)

        distances= distances * padding_mask[:, 0, 1:][:, :, None] # K X T X 18 

        if compute_for_each_view:
            # mean_distance_loss = distances.mean(dim=1).mean(dim=1) # K 

            mean_distance_loss = distances.mean(dim=2) # K X T 

            return mean_distance_loss 
        else:
       
            # Calculate mean distance as the loss
            mean_distance_loss = distances.mean()

            return mean_distance_loss

    def p_losses(self, x_start, t, x_cond=None, noise=None, padding_mask=None, \
        language_embedding=None, input_line_cond=None):
        # x_start: (BS*K) X T X D
        # x_cond: (BS*K) X T X D_cond
        # padding_mask: (BS*K) X 1 X T 
        b, timesteps, d_input = x_start.shape # (BS*K) X T X D
        noise = default(noise, lambda: torch.randn_like(x_start))

        x = self.q_sample(x_start=x_start, t=t, noise=noise) # noisy motion in noise level t. 
            
        model_out = self.denoise_fn(x, t, x_cond, padding_mask, \
            language_embedding=language_embedding)

        if self.objective == 'pred_noise':
            target = noise
        elif self.objective == 'pred_x0':
            target = x_start
        else:
            raise ValueError(f'unknown objective {self.objective}')

        if padding_mask is not None:
            loss = self.loss_fn(model_out, target, \
                reduction = 'none') * padding_mask[:, 0, 1:][:, :, None] 
        else:
            loss = self.loss_fn(model_out, target, reduction = 'none') # BS X T X D 

        loss = reduce(loss, 'b ... -> b (...)', 'mean') # BS X (T*D)

        loss = loss * extract(self.p2_loss_weight, t, loss.shape)

        # if self.input_line_cond: 
            
        #     curr_line_cond = x_cond[:, :, -18*3:].reshape(x_cond.shape[0], x_cond.shape[1], 18, 3) # (BS*K) X T X 18 X 3
            
        #     loss_line = self.compute_joint2line_distance_loss(model_out, curr_line_cond, padding_mask)
        
        #     return loss.mean(), loss_line 
        
        return loss.mean()

    def forward(self, x_start, cond_data, padding_mask=None, \
        language_input=None, input_line_cond=None, use_cfg=False, \
        cond_mask_prob=0.25):
        # x_start: (BS*K) X T X D(18*2)
        # cond_data: (BS*K) X T X D(18*2) 
        bs = x_start.shape[0] 
        t = torch.randint(0, self.num_timesteps, (bs,), device=x_start.device).long()

        x_cond = cond_data 

        language_embedding = None 
        
        curr_loss = self.p_losses(x_start, t, x_cond=x_cond, padding_mask=padding_mask, \
            language_embedding=language_embedding, \
            input_line_cond=input_line_cond)

        return curr_loss

    def opt_3d_w_vposer_w_joints3d_input(self, target_jnts3d, center_all_jnts2d=False, for_interaction=False):
        # target_jnts3d: T X 18 X 3/BS X T X 18 X 3 
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # fitter = SmplVPoserFitter()

        batch_size = target_jnts3d.shape[0] 
        # smpl_param, fixed_scale_seq = self.fitter.init_param(target_jnts3d.reshape(-1, 18, 3), \
        #                         batch_size=batch_size) 

        smpl_param = self.fitter.init_param(target_jnts3d.reshape(-1, 18, 3), \
                                batch_size=batch_size) 
      
      
        smpl_param = {'orient': torch.tensor(smpl_param['orient'], dtype=torch.float32, device=device, requires_grad=True),
                    'trans': torch.tensor(smpl_param['trans'], dtype=torch.float32, device=device, requires_grad=True),
                    'latent': torch.zeros((batch_size*target_jnts3d.shape[1], 32), dtype=torch.float32, device=device, requires_grad=True),
                    'scaling': torch.tensor(smpl_param['scaling'], dtype=torch.float32, device=device, requires_grad=True)}

        optimizer = torch.optim.Adam(smpl_param.values(), lr=1e-2)
        # optimizer = torch.optim.Adam(smpl_param.values(), lr=1e-3)

        # if fixed_scale_seq is not None:
        #     target_jnts3d = torch.from_numpy(target_jnts3d).to(device) * torch.from_numpy(fixed_scale_seq).float().to(device)[:, None, None] 
        # else:
        target_jnts3d = torch.from_numpy(target_jnts3d).to(device)

        target_jnts3d = target_jnts3d.reshape(-1, 18, 3) # (BS*T) X 18 X 3 
        ori_smpl_param, smpl_param, smpl_faces = self.fitter.solve(smpl_param=smpl_param, optimizer=optimizer, \
            closure=self.fitter.gen_closure(optimizer, smpl_param, target_jnts3d, batch_size=batch_size), \
            iter_max=1000) # Original 1000 , test initial smpl global orientation 

        ori_smpl_param_list = []
        smpl_param_list = []
        keypoints3d_list = [] 
        scale_val_list = [] 

        for bs_idx in range(batch_size):
            # Force scale to 1 
            curr_ori_smpl_param = {} 
            curr_ori_smpl_param['orient'] = ori_smpl_param['orient'].reshape(batch_size, -1, 3)[bs_idx] # T X 3 
            curr_ori_smpl_param['trans'] = ori_smpl_param['trans'].reshape(batch_size, -1, 3)[bs_idx] # T X 3 
            curr_ori_smpl_param['latent'] = ori_smpl_param['latent'].reshape(batch_size, -1, 32)[bs_idx] # T X 32
            curr_ori_smpl_param['scaling'] = ori_smpl_param['scaling'].reshape(batch_size, -1)[bs_idx] # 1 

            ori_smpl_param_list.append(curr_ori_smpl_param) 

            curr_smpl_param = {} 
            curr_smpl_param['smpl_trans'] = smpl_param['smpl_trans'].reshape(batch_size, -1, 3)[bs_idx] # T X 3 
            curr_smpl_param['smpl_poses'] = smpl_param['smpl_poses'].reshape(batch_size, -1, 24, 3)[bs_idx] # T X 24 X 3
            curr_smpl_param['smpl_scaling'] = smpl_param['smpl_scaling'].reshape(batch_size, -1)[bs_idx] # 1 

            if for_interaction:
                scale_val = curr_smpl_param['smpl_scaling'][0]
                
                keypoints3d = target_jnts3d.reshape(batch_size, -1, 18, 3)[bs_idx] 

                smpl_param_list.append(curr_smpl_param) 

                keypoints3d_list.append(keypoints3d) 
                scale_val_list.append(scale_val) 

            else:
                scale_val = curr_smpl_param['smpl_scaling'][0]
                curr_smpl_param['smpl_scaling'] = np.asarray([1.])

                aligned_smpl_trans = curr_smpl_param['smpl_trans']
                aligned_smpl_trans /= scale_val 
                
                keypoints3d = target_jnts3d.reshape(batch_size, -1, 18, 3)[bs_idx]/scale_val 

                curr_smpl_param['smpl_trans'] = aligned_smpl_trans 
                smpl_param_list.append(curr_smpl_param) 

                keypoints3d_list.append(keypoints3d) 
                scale_val_list.append(scale_val) 

        # return ori_smpl_param, smpl_param, keypoints3d.detach().cpu().numpy(), smpl_faces, scale_val   
        return ori_smpl_param_list, smpl_param_list, keypoints3d_list, smpl_faces, scale_val_list 

    def opt_3d_object_w_joints3d_input(self, target_jnts3d):
        # target_jnts3d: BS X T X 5 X 3 
        # obj_rest_verts: N_v X 3 

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        batch_size = target_jnts3d.shape[0] 

        obj_param = self.object_fitter.init_param(target_jnts3d.reshape(-1, 5, 3), batch_size=batch_size) 

        obj_param['obj_orient'] = torch.from_numpy(obj_param['obj_orient']).float().to(device)
        obj_param['obj_orient'].requires_grad = True 
        obj_param['obj_trans'] = torch.from_numpy(obj_param['obj_trans']).float().to(device)
        obj_param['obj_trans'].requires_grad = True
        obj_param['obj_scaling'] = torch.from_numpy(obj_param['obj_scaling']).float().to(device)
        obj_param['obj_scaling'].requires_grad = True 

        optimizer = torch.optim.Adam(obj_param.values(), lr=1e-2)
      
        target_jnts3d = torch.from_numpy(target_jnts3d).to(device)

        target_jnts3d = target_jnts3d.reshape(-1, 5, 3) # (BS*T) X 5 X 3 
        ori_obj_param, obj_param, obj_verts, obj_faces = self.object_fitter.solve(obj_param=obj_param, optimizer=optimizer, \
            closure=self.object_fitter.gen_closure(optimizer, obj_param, target_jnts3d, batch_size=batch_size), \
            iter_max=500, batch_size=batch_size) # Original 1000 , test initial smpl global orientation 

        # obj_verts is the vertices that applied scale, not the scale==1 

        ori_obj_param_list = []
        obj_param_list = []
        keypoints3d_list = []
        scale_val_list = []

        obj_verts_list = [] 

        for bs_idx in range(batch_size):
            # Force scale to 1 
            curr_ori_smpl_param = {} 
            curr_ori_smpl_param['obj_orient'] = ori_obj_param['obj_orient'].reshape(batch_size, -1, 3)[bs_idx] # T X 3 
            curr_ori_smpl_param['obj_trans'] = ori_obj_param['obj_trans'].reshape(batch_size, -1, 3)[bs_idx] # T X 3 
            curr_ori_smpl_param['obj_scaling'] = ori_obj_param['obj_scaling'].reshape(batch_size, -1)[bs_idx] # 1 

            ori_obj_param_list.append(curr_ori_smpl_param) 

            curr_obj_param = {} 
            curr_obj_param['obj_trans'] = obj_param['obj_trans'].reshape(batch_size, -1, 3)[bs_idx] # T X 3 
            curr_obj_param['obj_orient'] = obj_param['obj_orient'].reshape(batch_size, -1, 24, 3)[bs_idx] # T X 24 X 3
            curr_obj_param['obj_scaling'] = obj_param['obj_scaling'].reshape(batch_size, -1)[bs_idx] # 1 

            scale_val = curr_obj_param['obj_scaling'][0]
            curr_obj_param['obj_scaling'] = np.asarray([1.])

            aligned_obj_trans = curr_obj_param['obj_trans']
            aligned_obj_trans /= scale_val 
            
            keypoints3d = target_jnts3d.reshape(batch_size, -1, 5, 3)[bs_idx]/scale_val 

            curr_obj_verts = obj_verts.reshape(batch_size, -1, obj_verts.shape[-2], 3)[bs_idx]/scale_val # T X N_v X 3

            curr_obj_param['obj_trans'] = aligned_obj_trans 
            obj_param_list.append(curr_obj_param) 

            keypoints3d_list.append(keypoints3d) 
            scale_val_list.append(scale_val) 

            obj_verts_list.append(curr_obj_verts)

        return ori_obj_param_list, obj_param_list, keypoints3d_list, obj_verts_list, obj_faces, scale_val_list  

    def opt_3d_object_w_joints2d_input(self, target_jnts2d, cam_rot_mat, cam_trans):
        # target_jnts3d: BS X T X 5 X 2 
        # obj_rest_verts: N_v X 3 
        # cam_rot_mat: BS X T X 3 X 3 
        # cam_trans: BS X T X 3 

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        batch_size = target_jnts2d.shape[0] 

        obj_param = self.object_fitter_2d.init_param(target_jnts2d.reshape(-1, 5, 2), batch_size=batch_size) 

        optimizer = torch.optim.Adam(obj_param.values(), lr=1e-2)
      
        target_jnts2d = torch.from_numpy(target_jnts2d).to(device)

        target_jnts2d = target_jnts2d.reshape(-1, 5, 2) # (BS*T) X 5 X 2
        cam_rot_mat = cam_rot_mat.reshape(-1, 3, 3) # (BS*T) X 3 X 3
        cam_trans = cam_trans.reshape(-1, 3) # (BS*T) X 3 
        ori_obj_param, obj_param, obj_kpts, obj_verts, obj_faces = self.object_fitter_2d.solve(obj_param=obj_param, optimizer=optimizer, \
            closure=self.object_fitter_2d.gen_closure(optimizer, obj_param, target_jnts2d, cam_rot_mat, cam_trans, batch_size=batch_size), \
            iter_max=500, batch_size=batch_size) # Original 1000 , test initial smpl global orientation 

        ori_obj_param_list = []
        obj_param_list = []
        keypoints3d_list = []
        scale_val_list = []

        obj_verts_list = [] 

        for bs_idx in range(batch_size):
            # Force scale to 1 
            curr_ori_smpl_param = {} 
            curr_ori_smpl_param['obj_orient'] = ori_obj_param['obj_orient'].reshape(batch_size, -1, 3)[bs_idx] # T X 3 
            curr_ori_smpl_param['obj_trans'] = ori_obj_param['obj_trans'].reshape(batch_size, -1, 3)[bs_idx] # T X 3 
            curr_ori_smpl_param['obj_scaling'] = ori_obj_param['obj_scaling'].reshape(batch_size, -1)[bs_idx] # 1 

            ori_obj_param_list.append(curr_ori_smpl_param) 

            curr_obj_param = {} 
            curr_obj_param['obj_trans'] = obj_param['obj_trans'].reshape(batch_size, -1, 3)[bs_idx] # T X 3 
            curr_obj_param['obj_orient'] = obj_param['obj_orient'].reshape(batch_size, -1, 24, 3)[bs_idx] # T X 24 X 3
            curr_obj_param['obj_scaling'] = obj_param['obj_scaling'].reshape(batch_size, -1)[bs_idx] # 1 

            scale_val = curr_obj_param['obj_scaling'][0]
            curr_obj_param['obj_scaling'] = np.asarray([1.])

            aligned_obj_trans = curr_obj_param['obj_trans']
            aligned_obj_trans /= scale_val 
            
            keypoints3d = obj_kpts.reshape(batch_size, -1, 5, 3)[bs_idx]/scale_val 

            curr_obj_verts = obj_verts.reshape(batch_size, -1, obj_verts.shape[-2], 3)[bs_idx]/scale_val # T X N_v X 3

            curr_obj_param['obj_trans'] = aligned_obj_trans 
            obj_param_list.append(curr_obj_param) 

            keypoints3d_list.append(keypoints3d) 
            scale_val_list.append(scale_val) 

            obj_verts_list.append(curr_obj_verts)

        return ori_obj_param_list, obj_param_list, keypoints3d_list, obj_verts_list, obj_faces, scale_val_list  

    def get_projection_from_motion3d(self, num_views, cam_rot_mat, cam_trans, fk_jnts3d):
        # Project the optimized 3D motion to 2D 
        reprojected_ori_jnts2d_list = [] 
        reprojected_jnts2d_list = [] 

        for view_idx in range(num_views):
            _, _, reprojected_ori_jnts2d = perspective_projection(
                    fk_jnts3d, cam_rot_mat[view_idx], \
                    cam_trans[view_idx])

            reprojected_ori_jnts2d_list.append(reprojected_ori_jnts2d.clone())
            reprojected_jnts2d = normalize_pose2d_w_grad(reprojected_ori_jnts2d)
            
            reprojected_jnts2d_list.append(reprojected_jnts2d)

        reprojected_ori_jnts2d_list = torch.stack(reprojected_ori_jnts2d_list) # K X T X 18 X 2 
        reprojected_jnts2d_list = torch.stack(reprojected_jnts2d_list) # K X T X 18 X 2 

        seq_len = reprojected_jnts2d_list.shape[1]
        reprojected_jnts2d_list = reprojected_jnts2d_list.reshape(num_views, seq_len, -1) # K X T X (18*2)

        return reprojected_jnts2d_list, reprojected_ori_jnts2d_list, fk_jnts3d  
        # K X T X (18*2), K X T X 18 X 2, T X 18 X 3  

    def opt_3d_w_multi_view_2d_res(self, line_cond_2d_res, cam_extrinsic, \
        padding_mask=None):
        # line_cond_2d_res: BS X K X T X (18*2) 
        # cam_extrinsic: BS X K X 4 X 4
        # padding_mask: (BS*K) X 1 X 121 
        # actual_seq_len: a value 120 
        # input_line_cond: BS X K X T X 18 X 3
        # x_start: BS X K X T X (18*2) 
            
        # Prepare for optimization during denoising step 
        shape = line_cond_2d_res.shape 
        batch, device, total_timesteps = \
        shape[0], self.betas.device, self.num_timesteps

        num_views = line_cond_2d_res.shape[1]
        seq_len = line_cond_2d_res.shape[2]

        padding_mask = padding_mask.reshape(batch, num_views, 1, -1) # BS X K X 1 X 121

        cam_rot_mat = cam_extrinsic[:, :, :3, :3].unsqueeze(2).repeat(1, 1, seq_len, 1, 1) # BS X K X T X 3 X 3 
        cam_trans = cam_extrinsic[:, :, :3, -1].unsqueeze(2).repeat(1, 1, seq_len, 1) # BS X K X T X 3 
       
        # Initialize 3D pose
        if self.use_animal_data:
            num_joints = 17 
        elif self.use_omomo_data:
            num_joints = 18 + 5 
        else:
            num_joints = 18 
        self.motion_3d = torch.zeros(batch, seq_len, num_joints, 3).to(line_cond_2d_res.device) # BS X T X 18 X 3 
        self.motion_3d.requires_grad_(True) 
            
        self.optimizer_3d = Adam([self.motion_3d], lr=0.01)

        self.scheduler_3d = MultiStepLR(self.optimizer_3d, \
                milestones=[5000], gamma=0.1)

        opt_iterations = 500
        for opt_iter in range(opt_iterations):
            self.optimizer_3d.zero_grad()

            # Project 3D to 2D pose sequences 
            batch_reprojected_jnts2d_list = []
            final_jnts3d_list = []
            for bs_idx in range(batch):
                reprojected_jnts2d_list, reprojected_ori_jnts2d_list, final_jnts3d = \
                    self.get_projection_from_motion3d(num_views, cam_rot_mat[bs_idx], cam_trans[bs_idx], \
                    self.motion_3d[bs_idx])
                
                batch_reprojected_jnts2d_list.append(reprojected_jnts2d_list) 
                final_jnts3d_list.append(final_jnts3d)

            batch_reprojected_jnts2d_list = torch.stack(batch_reprojected_jnts2d_list, dim=0) # BS X K X T X (18*2) 
            final_jnts3d_list = torch.stack(final_jnts3d_list, dim=0) # BS X T X 18 X 3 
                
            loss_projection = F.mse_loss(batch_reprojected_jnts2d_list, \
                                line_cond_2d_res, \
                                reduction='none') * padding_mask[:, :, 0, 1:][:, :, :, None]

            loss = loss_projection.mean() 
            if opt_iter % 20 == 0:
                print("Optimization step:{0}, loss:{1}".format(opt_iter, \
                    loss.item()))
            
            loss.backward()

            self.optimizer_3d.step() 
            self.scheduler_3d.step()
       
        return batch_reprojected_jnts2d_list, final_jnts3d_list

    def opt_3d_w_vposer_w_joints2d_reprojection(self, target_jnts2d, cam_rot_mat, cam_trans):
        # target_jnts2d: BS X T X J X 2 
        # cam_rot_mat: BS X T X 3 X 3 
        # cam_trans: BS X T X 3 

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        batch_size = target_jnts2d.shape[0] 
        seq_len = target_jnts2d.shape[1] 

        init_target_jnts2d = target_jnts2d.clone()[:, 0:1].repeat(1, seq_len, 1, 1) # BS X 1 X 18 X 2 
        init_cam_rot_mat = cam_rot_mat[:, 0:1].repeat(1, seq_len, 1, 1) # BS X 1 X 3 X 3 
        init_cam_trans = cam_trans[:, 0:1].repeat(1, seq_len, 1) # BS X 1 X 3 

        init_smpl_param = self.fitter_2d.init_param(init_target_jnts2d.reshape(-1, 18, 2), \
                                batch_size=batch_size)
      
        # Stage1 optimization 
        init_smpl_param = {'orient': torch.tensor(init_smpl_param['orient'], dtype=torch.float32, device=device, requires_grad=True),
                    'trans': torch.tensor(init_smpl_param['trans'], dtype=torch.float32, device=device, requires_grad=False),
                    'latent': torch.tensor(init_smpl_param['latent'], dtype=torch.float32, device=device, requires_grad=False),
                    'scaling': torch.tensor(init_smpl_param['scaling'], dtype=torch.float32, device=device, requires_grad=True)}

        optimizer = torch.optim.Adam(init_smpl_param.values(), lr=1e-2)

        init_target_jnts2d = init_target_jnts2d.reshape(-1, 18, 2).detach()  # (BS*1) X 18 X 2 
        init_cam_rot_mat = init_cam_rot_mat.reshape(-1, 3, 3).detach() # (BS*1) X 3 X 3 
        init_cam_trans = init_cam_trans.reshape(-1, 3).detach() # (BS*1) X 3 

        target_jnts2d = target_jnts2d.reshape(-1, 18, 2).detach()  # (BS*T) X 18 X 2 
        cam_rot_mat = cam_rot_mat.reshape(-1, 3, 3).detach() # (BS*T) X 3 X 3 
        cam_trans = cam_trans.reshape(-1, 3).detach() # (BS*T) X 3 

        ori_smpl_param, smpl_param, smpl_faces = self.fitter_2d.solve(smpl_param=init_smpl_param, optimizer=optimizer, \
            closure=self.fitter_2d.gen_closure(optimizer, init_smpl_param, init_target_jnts2d, init_cam_rot_mat, init_cam_trans, batch_size=batch_size), \
            iter_max=3000) # Original 1000 , test initial smpl global orientation 

        # # Stage2 optimization 
        smpl_param = {'orient': torch.tensor(ori_smpl_param['orient'], dtype=torch.float32, device=device, requires_grad=True),
                    'trans': torch.tensor(ori_smpl_param['trans'], dtype=torch.float32, device=device, requires_grad=True),
                    'latent': torch.tensor(ori_smpl_param['latent'], dtype=torch.float32, device=device, requires_grad=False),
                    'scaling': torch.tensor(ori_smpl_param['scaling'], dtype=torch.float32, device=device, requires_grad=False)}

        optimizer = torch.optim.Adam(smpl_param.values(), lr=1e-2)

        ori_smpl_param, smpl_param, smpl_faces = self.fitter_2d.solve(smpl_param=smpl_param, optimizer=optimizer, \
            closure=self.fitter_2d.gen_closure(optimizer, smpl_param, target_jnts2d, cam_rot_mat, cam_trans, batch_size=batch_size), \
            iter_max=1000) # Original 1000 , test initial smpl global orientation 

        # # Stage3 optimization 
        smpl_param = {'orient': torch.tensor(ori_smpl_param['orient'], dtype=torch.float32, device=device, requires_grad=True),
                    'trans': torch.tensor(ori_smpl_param['trans'], dtype=torch.float32, device=device, requires_grad=True),
                    'latent': torch.tensor(ori_smpl_param['latent'], dtype=torch.float32, device=device, requires_grad=True),
                    'scaling': torch.tensor(ori_smpl_param['scaling'], dtype=torch.float32, device=device, requires_grad=False)}

        optimizer = torch.optim.Adam(smpl_param.values(), lr=1e-2)

        ori_smpl_param, smpl_param, smpl_faces = self.fitter_2d.solve(smpl_param=smpl_param, optimizer=optimizer, \
            closure=self.fitter_2d.gen_closure(optimizer, smpl_param, target_jnts2d, cam_rot_mat, cam_trans, batch_size=batch_size), \
            iter_max=1000) # Original 1000 , test initial smpl global orientation 

        # Save results to list 
        ori_smpl_param_list = []
        smpl_param_list = []
        scale_val_list = [] 

        for bs_idx in range(batch_size):
            # Force scale to 1 
            curr_ori_smpl_param = {} 
            curr_ori_smpl_param['orient'] = ori_smpl_param['orient'].reshape(batch_size, -1, 3)[bs_idx] # T X 3 
            curr_ori_smpl_param['trans'] = ori_smpl_param['trans'].reshape(batch_size, -1, 3)[bs_idx] # T X 3 
            curr_ori_smpl_param['latent'] = ori_smpl_param['latent'].reshape(batch_size, -1, 32)[bs_idx] # T X 32
            curr_ori_smpl_param['scaling'] = ori_smpl_param['scaling'].reshape(batch_size, -1)[bs_idx] # 1 

            ori_smpl_param_list.append(curr_ori_smpl_param) 

            curr_smpl_param = {} 
            curr_smpl_param['smpl_trans'] = smpl_param['smpl_trans'].reshape(batch_size, -1, 3)[bs_idx] # T X 3 
            curr_smpl_param['smpl_poses'] = smpl_param['smpl_poses'].reshape(batch_size, -1, 24, 3)[bs_idx] # T X 24 X 3
            curr_smpl_param['smpl_scaling'] = smpl_param['smpl_scaling'].reshape(batch_size, -1)[bs_idx] # 1 

            scale_val = curr_smpl_param['smpl_scaling'][0]
            curr_smpl_param['smpl_scaling'] = np.asarray([1.])

            aligned_smpl_trans = curr_smpl_param['smpl_trans']
            aligned_smpl_trans /= scale_val 

            curr_smpl_param['smpl_trans'] = aligned_smpl_trans 
            smpl_param_list.append(curr_smpl_param) 

            scale_val_list.append(scale_val) 

        return ori_smpl_param_list, smpl_param_list, smpl_faces, scale_val_list  

    def opt_3d_w_smal_animal_w_joints3d_input(self, target_jnts3d):
        # target_jnts3d: BS X T  X J X 3 
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        batch_size = target_jnts3d.shape[0] 
        smal_param = self.smal_fitter.init_param(target_jnts3d.reshape(-1, 17, 3), \
                            batch_size=batch_size) 
        # smpl_param {} 
        # betas: BS X 20 
        # log_beta_scales: BS X 6
        # global_rotation: (BS*T) X 3
        # trans: (BS*T) X 3
        # joint_rotations: (BS*T) X 34 X 3 

        optimizer = torch.optim.Adam(smal_param.values(), lr=1e-2)

        target_jnts3d = torch.from_numpy(target_jnts3d).to(device).float() # BS X T X J X 3 
        smal_param, smal_faces = self.smal_fitter.solve(smal_param=smal_param, optimizer=optimizer, \
            closure=self.smal_fitter.gen_closure(optimizer, smal_param, target_jnts3d.reshape(-1, 17, 3), batch_size=batch_size), \
            iter_max=1000) # Original 1000 , test initial smpl global orientation 

        # Enforce mid spiine to zeros. 
        # mid_bone_idx_in_34 = 7 
        # smal_param['joint_rotations'][:, mid_bone_idx_in_34, :] = \
        #     torch.zeros(smal_param['joint_rotations'].shape[0], 3).to(smal_param['joint_rotations'].device) # T X 34 X 3 

        force_spine_zero = True 
        if force_spine_zero:
            smal_param['joint_rotations'].requires_grad = False
            # spine_idx_list = [1, 2, 3, 4, 5, 6] # spine 
            # spine_idx_list = [1, 2, 3, 4, 5, 25, 26, 27, 28, 29, 30, 31, 10, 14, 20, 24]

            spine_idx_list = [25, 26, 27, 28, 29, 30, 31]
            # smal_param['joint_rotations'][:, 7, :] = torch.zeros(smal_param['joint_rotations'].shape[0], 3).to(smal_param['joint_rotations'].device) # T X 34 X 3
            # smal_param['joint_rotations'][:, 8, :] = torch.zeros(smal_param['joint_rotations'].shape[0], 3).to(smal_param['joint_rotations'].device) # T X 34 X 3
            # smal_param['joint_rotations'][:, 9, :] = torch.zeros(smal_param['joint_rotations'].shape[0], 3).to(smal_param['joint_rotations'].device) # T X 34 X 3 

            for tmp_s_idx in spine_idx_list:
                smal_param['joint_rotations'][:, tmp_s_idx, :] = torch.zeros(smal_param['joint_rotations'].shape[0], 3).to(smal_param['joint_rotations'].device)

        # Get optimized joint and vertices 
        smal_opt_jnts3d, smal_opt_verts, _ = self.smal_fitter.smal_forward(smal_param, batch_size=batch_size) 

        smal_opt_jnts3d_17 = smal_opt_jnts3d[:, self.smal_fitter.smal2xpose_idx_list] # (BS*T) X 17 X 3 
        smal_opt_jnts3d_17 = smal_opt_jnts3d_17.reshape(batch_size, -1, 17, 3) # BS X T X 17 X 3
        smal_opt_verts = smal_opt_verts.reshape(batch_size, -1, smal_opt_verts.shape[-2], 3) # BS X T X N_v X 3 

        #  smal_param.keys() dict_keys(['betas', 'log_beta_scales', 'global_rotation', 'trans', 'joint_rotations'])
        ori_smal_param_list = []
        smal_param_list = []
        scale_val_list = []

        

        for bs_idx in range(batch_size):
            curr_ori_smal_param = {} 
            curr_ori_smal_param['global_rotation'] = smal_param['global_rotation'].reshape(batch_size, -1, 3)[bs_idx].detach().cpu().numpy() # T X 3 
            curr_ori_smal_param['joint_rotations'] = smal_param['joint_rotations'].reshape(batch_size, -1, 17, 3)[bs_idx].detach().cpu().numpy() # T X 17 X 3
            curr_ori_smal_param['trans'] = smal_param['trans'].reshape(batch_size, -1, 3)[bs_idx].detach().cpu().numpy() # T X 3 
            curr_ori_smal_param['betas'] = smal_param['betas'].reshape(batch_size, -1)[bs_idx].detach().cpu().numpy() # 20  
            curr_ori_smal_param['log_beta_scales'] = smal_param['log_beta_scales'].reshape(batch_size, -1)[bs_idx].detach().cpu().numpy() # 6 

            curr_smal_param = {} 
            curr_smal_param['global_rotation'] = smal_param['global_rotation'].reshape(batch_size, -1, 3)[bs_idx].detach().cpu().numpy() # T X 3 
            curr_smal_param['joint_rotations'] = smal_param['joint_rotations'].reshape(batch_size, -1, 17, 3)[bs_idx].detach().cpu().numpy() # T X 17 X 3
            curr_smal_param['trans'] = smal_param['trans'].reshape(batch_size, -1, 3)[bs_idx].detach().cpu().numpy() # T X 3 
            curr_smal_param['betas'] = smal_param['betas'].reshape(batch_size, -1)[bs_idx].detach().cpu().numpy() # 20  
            curr_smal_param['log_beta_scales'] = smal_param['log_beta_scales'].reshape(batch_size, -1)[bs_idx].detach().cpu().numpy() # 6 

            scale_val = smal_param['scaling'][bs_idx].detach().cpu().numpy()[0] 
            curr_smal_param['scaling'] = np.asarray([1.])

            aligned_smal_trans = curr_smal_param['trans']
            aligned_smal_trans /= scale_val 
            curr_smal_param['trans'] = aligned_smal_trans 

            smal_opt_jnts3d_17[bs_idx] /= scale_val 
            smal_opt_verts[bs_idx] /= scale_val

            smal_param_list.append(curr_smal_param) 
            scale_val_list.append(scale_val) 

        return smal_param_list, smal_faces, scale_val_list, smal_opt_jnts3d_17, smal_opt_verts 

    def opt_3d_w_smal_animal_w_joints2d_reprojection(self, target_jnts2d, cam_rot_mat, cam_trans):
        # target_jnts3d: BS X T  X J X 3 
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        batch_size = target_jnts2d.shape[0] 
        seq_len = target_jnts2d.shape[1] 

        init_target_jnts2d = target_jnts2d.clone()[:, 0:1].repeat(1, seq_len, 1, 1) # BS X 1 X 18 X 2 
        init_cam_rot_mat = cam_rot_mat[:, 0:1].repeat(1, seq_len, 1, 1) # BS X 1 X 3 X 3 
        init_cam_trans = cam_trans[:, 0:1].repeat(1, seq_len, 1) # BS X 1 X 3 

        init_smpl_param = self.smal_fitter2d.init_param(init_target_jnts2d.reshape(-1, 17, 2), \
                                batch_size=batch_size)
      
        # Stage1 optimization 
        optimizer = torch.optim.Adam(init_smpl_param.values(), lr=1e-2)

        init_target_jnts2d = init_target_jnts2d.reshape(-1, 17, 2).detach()  # (BS*1) X 18 X 2 
        init_cam_rot_mat = init_cam_rot_mat.reshape(-1, 3, 3).detach() # (BS*1) X 3 X 3 
        init_cam_trans = init_cam_trans.reshape(-1, 3).detach() # (BS*1) X 3 

        target_jnts2d = target_jnts2d.reshape(-1, 17, 2).detach()  # (BS*T) X 18 X 2 
        cam_rot_mat = cam_rot_mat.reshape(-1, 3, 3).detach() # (BS*T) X 3 X 3 
        cam_trans = cam_trans.reshape(-1, 3).detach() # (BS*T) X 3 

        smpl_param, smpl_faces = self.smal_fitter2d.solve(smal_param=init_smpl_param, \
            optimizer=optimizer, \
            closure=self.smal_fitter2d.gen_closure(optimizer, init_smpl_param, init_target_jnts2d, \
            init_cam_rot_mat, init_cam_trans, batch_size=batch_size), \
            iter_max=1000) # Original 1000 , test initial smpl global orientation 

        # # Stage2 optimization 
        smpl_param = {'global_rotation': torch.tensor(smpl_param['global_rotation'].detach().cpu().numpy(), dtype=torch.float32, device=device, requires_grad=True),
                    'trans': torch.tensor(smpl_param['trans'].detach().cpu().numpy(), dtype=torch.float32, device=device, requires_grad=True),
                    'joint_rotations': torch.tensor(smpl_param['joint_rotations'].detach().cpu().numpy(), dtype=torch.float32, device=device, requires_grad=False),
                    'scaling': torch.tensor(smpl_param['scaling'].detach().cpu().numpy(), dtype=torch.float32, device=device, requires_grad=False), 
                    'betas': smpl_param['betas'], 
                    'log_beta_scales': smpl_param['log_beta_scales']}

        optimizer = torch.optim.Adam(smpl_param.values(), lr=1e-2)

        smpl_param, smpl_faces = self.smal_fitter2d.solve(smal_param=smpl_param, optimizer=optimizer, \
            closure=self.smal_fitter2d.gen_closure(optimizer, smpl_param, target_jnts2d, cam_rot_mat, cam_trans, batch_size=batch_size), \
            iter_max=1000) # Original 1000 , test initial smpl global orientation 

        # # Stage3 optimization 
        smpl_param = {'global_rotation': torch.tensor(smpl_param['global_rotation'].detach().cpu().numpy(), dtype=torch.float32, device=device, requires_grad=True),
                    'trans': torch.tensor(smpl_param['trans'].detach().cpu().numpy(), dtype=torch.float32, device=device, requires_grad=True),
                    'joint_rotations': torch.tensor(smpl_param['joint_rotations'].detach().cpu().numpy(), dtype=torch.float32, device=device, requires_grad=True),
                    'scaling': torch.tensor(smpl_param['scaling'].detach().cpu().numpy(), dtype=torch.float32, device=device, requires_grad=False), 
                    'betas': smpl_param['betas'], 
                    'log_beta_scales': smpl_param['log_beta_scales']}

        optimizer = torch.optim.Adam(smpl_param.values(), lr=1e-2)

        smpl_param, smpl_faces = self.smal_fitter2d.solve(smal_param=smpl_param, optimizer=optimizer, \
            closure=self.smal_fitter2d.gen_closure(optimizer, smpl_param, target_jnts2d, cam_rot_mat, cam_trans, batch_size=batch_size), \
            iter_max=1000) # Original 1000 , test initial smpl global orientation 


        force_spine_zero = True 
        if force_spine_zero:
            smpl_param['joint_rotations'].requires_grad = False
            smpl_param['joint_rotations'][:, 7, :] = torch.zeros(smpl_param['joint_rotations'].shape[0], 3).to(smpl_param['joint_rotations'].device) # T X 34 X 3
            smpl_param['joint_rotations'][:, 8, :] = torch.zeros(smpl_param['joint_rotations'].shape[0], 3).to(smpl_param['joint_rotations'].device) # T X 34 X 3
            smpl_param['joint_rotations'][:, 9, :] = torch.zeros(smpl_param['joint_rotations'].shape[0], 3).to(smpl_param['joint_rotations'].device) # T X 34 X 3 


        # Get optimized joint and vertices 
        smal_opt_jnts3d, smal_opt_verts, _ = self.smal_fitter2d.smal_forward(smpl_param, batch_size=batch_size) 

        smal_opt_jnts3d_17 = smal_opt_jnts3d[:, self.smal_fitter2d.smal2xpose_idx_list] # (BS*T) X 17 X 3 
        smal_opt_jnts3d_17 = smal_opt_jnts3d_17.reshape(batch_size, -1, 17, 3) # BS X T X 17 X 3
        smal_opt_verts = smal_opt_verts.reshape(batch_size, -1, smal_opt_verts.shape[-2], 3) # BS X T X N_v X 3 

        #  smal_param.keys() dict_keys(['betas', 'log_beta_scales', 'global_rotation', 'trans', 'joint_rotations'])
        ori_smal_param_list = []
        smal_param_list = []
        scale_val_list = []

        for bs_idx in range(batch_size):
            curr_ori_smal_param = {} 
            curr_ori_smal_param['global_rotation'] = smpl_param['global_rotation'].reshape(batch_size, -1, 3)[bs_idx].detach().cpu().numpy() # T X 3 
            curr_ori_smal_param['joint_rotations'] = smpl_param['joint_rotations'].reshape(batch_size, -1, 17, 3)[bs_idx].detach().cpu().numpy() # T X 17 X 3
            curr_ori_smal_param['trans'] = smpl_param['trans'].reshape(batch_size, -1, 3)[bs_idx].detach().cpu().numpy() # T X 3 
            curr_ori_smal_param['betas'] = smpl_param['betas'].reshape(batch_size, -1)[bs_idx].detach().cpu().numpy() # 20  
            curr_ori_smal_param['log_beta_scales'] = smpl_param['log_beta_scales'].reshape(batch_size, -1)[bs_idx].detach().cpu().numpy() # 6 

            curr_smal_param = {} 
            curr_smal_param['global_rotation'] = smpl_param['global_rotation'].reshape(batch_size, -1, 3)[bs_idx].detach().cpu().numpy() # T X 3 
            curr_smal_param['joint_rotations'] = smpl_param['joint_rotations'].reshape(batch_size, -1, 17, 3)[bs_idx].detach().cpu().numpy() # T X 17 X 3
            curr_smal_param['trans'] = smpl_param['trans'].reshape(batch_size, -1, 3)[bs_idx].detach().cpu().numpy() # T X 3 
            curr_smal_param['betas'] = smpl_param['betas'].reshape(batch_size, -1)[bs_idx].detach().cpu().numpy() # 20  
            curr_smal_param['log_beta_scales'] = smpl_param['log_beta_scales'].reshape(batch_size, -1)[bs_idx].detach().cpu().numpy() # 6 
           
            scale_val = smpl_param['scaling'][bs_idx].detach().cpu().numpy()[0] 
            curr_smal_param['scaling'] = np.asarray([1.])

            aligned_smal_trans = curr_smal_param['trans']
            aligned_smal_trans /= scale_val 
            curr_smal_param['trans'] = aligned_smal_trans 

            smal_opt_jnts3d_17[bs_idx] /= scale_val 
            smal_opt_verts[bs_idx] /= scale_val

            smal_param_list.append(curr_smal_param) 
            scale_val_list.append(scale_val) 

        return smal_param_list, smpl_faces, scale_val_list, smal_opt_jnts3d_17, smal_opt_verts 

   
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
        train_2d_diffusion_w_line_cond=False, 
        use_animal_data=False,
        use_omomo_data=False,
        omomo_object_name="largebox", 
    ):
        super().__init__()

        self.opt = opt 

        self.use_animal_data = use_animal_data 
        self.use_omomo_data = use_omomo_data 
        if self.use_animal_data:
            num_joints = 17 
        elif self.use_omomo_data:
            num_joints = 18 + 5 
        else:
            num_joints = 18 

        
        d_input_feats = d_feats

        if train_2d_diffusion_w_line_cond:
            d_input_feats += num_joints*3 

        self.denoise_fn = TransformerDiffusionModel(d_input_feats=d_input_feats, d_feats=d_feats, \
                    d_model=d_model, n_head=n_head, \
                    d_k=d_k, d_v=d_v, n_dec_layers=n_dec_layers, max_timesteps=max_timesteps) 
        # Input condition and noisy motion, noise level t, predict gt motion
        
        self.objective = objective

        self.seq_len = max_timesteps - 1 
        self.out_dim = out_dim 

        self.clip_encoder = nn.Sequential(
            nn.Linear(in_features=512, out_features=d_model),
            )

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

        self.fitter = SmplVPoserFitter()
        self.fitter_2d = SmplVPoser2DFitter() 

        if use_animal_data:
            self.smal_fitter = SmalFitter() 
            self.smal_fitter2d = SmalFitter2D() 
        elif use_omomo_data:
            self.object_fitter = ObjectFitter(object_name=omomo_object_name)
            # self.object_fitter_2d = ObjectFitter2D(object_name=omomo_object_name)  

    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, t, x_cond, padding_mask, language_embedding, clip_denoised, \
                        use_cfg=False, cfg_scale=3., x_cond_for_uncond_gen=None):
        # x_all = torch.cat((x, x_cond), dim=-1)
        # model_output = self.denoise_fn(x_all, t)

        if use_cfg:
            cond_output = self.denoise_fn(x, t, x_cond, padding_mask, \
                language_embedding=language_embedding)
            uncond_output =  self.denoise_fn(x, t, x_cond_for_uncond_gen, padding_mask, \
                language_embedding=language_embedding)
            model_output = uncond_output + cfg_scale * (cond_output - uncond_output) 

            model_output.clamp_(-1., 1.)
        else:
            model_output = self.denoise_fn(x, t, x_cond, padding_mask, \
                language_embedding=language_embedding)

        if self.objective == 'pred_noise':
            x_start = self.predict_start_from_noise(x, t = t, noise = model_output)
        elif self.objective == 'pred_x0':
            x_start = model_output
        else:
            raise ValueError(f'unknown objective {self.objective}')

        if clip_denoised:
            x_start.clamp_(-1., 1.)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_start, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance

    # @torch.no_grad()
    def p_sample(self, x, t, x_cond=None, padding_mask=None, \
        language_embedding=None, clip_denoised=True, use_cfg=False, cfg_scale=3., x_cond_for_uncond_gen=None):
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance = self.p_mean_variance(x=x, t=t, x_cond=x_cond, \
            padding_mask=padding_mask, language_embedding=language_embedding, \
            clip_denoised=clip_denoised, use_cfg=use_cfg, cfg_scale=cfg_scale, x_cond_for_uncond_gen=x_cond_for_uncond_gen)
        noise = torch.randn_like(x)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    # @torch.no_grad()
    def p_sample_loop(self, shape, x_start, x_cond=None, \
        padding_mask=None, language_embedding=None):
        device = self.betas.device

        b = shape[0]
        x = torch.randn(shape, device=device)

        for i in tqdm(reversed(range(0, self.num_timesteps)), desc='sampling loop time step', total=self.num_timesteps):
            x = self.p_sample(x, torch.full((b,), i, device=device, dtype=torch.long), \
                x_cond, padding_mask=padding_mask, language_embedding=language_embedding)    

        return x # BS X T X D

    # @torch.no_grad()
    def sample(self, x_start, cond_mask=None, padding_mask=None, \
        language_input=None, input_line_cond=None, ref_2d_seq_cond=None):
        # naive conditional sampling by replacing the noisy prediction with input target data. 
        self.denoise_fn.eval() 
        self.clip_encoder.eval()
       
        if cond_mask is not None:
            x_pose_cond = x_start * (1. - cond_mask)
            x_cond = x_pose_cond 
        else:
            x_cond = None 

        if input_line_cond is not None:
            if x_cond is not None:
                x_cond = torch.cat((input_line_cond.reshape(x_start.shape[0], x_start.shape[1], -1), \
                    x_cond), dim=-1) 
            else:
                x_cond = input_line_cond.reshape(x_start.shape[0], x_start.shape[1], -1) 

        if language_input is not None:
            language_embedding = self.clip_encoder(language_input)
        else:
            language_embedding = None 

        if ref_2d_seq_cond is not None:
            x_cond = torch.cat((ref_2d_seq_cond, x_cond), dim=-1) 

        sample_res = self.p_sample_loop(x_start.shape, \
                x_start, x_cond, padding_mask, language_embedding=language_embedding)
        # BS X T X D
            
        self.denoise_fn.train()
        self.clip_encoder.train() 

        return sample_res  

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

    def p_losses(self, x_start, t, x_cond=None, noise=None, padding_mask=None, \
        motion_mask=None, language_embedding=None, input_line_cond=None):
        # x_start: BS X T X D
        # x_cond: BS X T X D_cond
        # padding_mask: BS X 1 X T 
        b, timesteps, d_input = x_start.shape # BS X T X D(3+n_joints*4)
        noise = default(noise, lambda: torch.randn_like(x_start))

        x = self.q_sample(x_start=x_start, t=t, noise=noise) # noisy motion in noise level t. 
            
        model_out = self.denoise_fn(x, t, x_cond, padding_mask, \
            language_embedding=language_embedding)

        if self.objective == 'pred_noise':
            target = noise
        elif self.objective == 'pred_x0':
            target = x_start
        else:
            raise ValueError(f'unknown objective {self.objective}')

        if padding_mask is not None:
            loss = self.loss_fn(model_out, target, \
                reduction = 'none') * padding_mask[:, 0, 1:][:, :, None] * motion_mask
        else:
            loss = self.loss_fn(model_out, target, reduction = 'none') * motion_mask # BS X T X D 

        loss = reduce(loss, 'b ... -> b (...)', 'mean') # BS X (T*D)

        loss = loss * extract(self.p2_loss_weight, t, loss.shape)

        if input_line_cond is not None:
            loss_line = self.compute_joint2line_distance_loss(model_out, input_line_cond, padding_mask)
        
            return loss.mean(), loss_line 
        
        return loss.mean()

    def compute_joint2line_distance_loss(self, pred_joints, line_coeffs, padding_mask, compute_for_each_view=False):
        """
        Computes the distance from predicted joint positions to their corresponding lines.

        Parameters:
        - pred_joints (torch.Tensor): Predicted joint positions with shape (BS x T x J x 2)
        - line_coeffs (torch.Tensor): Line coefficients with shape (BS x T x J x 3)

        Returns:
        - torch.Tensor: Mean distance loss across all batches, timesteps, and joints.
        """

        if pred_joints.shape[-1] != 2:
            bs, seq_len, _ = pred_joints.shape 
            pred_joints = pred_joints.reshape(bs, seq_len, -1, 2) 

        # Extract line coefficients
        a = line_coeffs[..., 0]  # Shape: (BS x T x J)
        b = line_coeffs[..., 1]  # Shape: (BS x T x J)
        c = line_coeffs[..., 2]  # Shape: (BS x T x J)

        # Extract joint coordinates
        x = pred_joints[..., 0]  # Shape: (BS x T x J)
        y = pred_joints[..., 1]  # Shape: (BS x T x J)

        # Calculate absolute distance using the line equation ax + by + c
        distances = torch.abs(a * x + b * y + c)  # Shape: (BS x T x J)

        distances= distances * padding_mask[:, 0, 1:][:, :, None] # K X T X 18 

        if compute_for_each_view:
            # mean_distance_loss = distances.mean(dim=1).mean(dim=1) # K 

            mean_distance_loss = distances.mean(dim=2) # K X T 

            return mean_distance_loss 
        else:
       
            # Calculate mean distance as the loss
            mean_distance_loss = distances.mean()

            return mean_distance_loss

    def forward(self, x_start, cond_mask=None, padding_mask=None, \
        motion_mask=None, language_input=None, input_line_cond=None, use_cfg=False, \
        cond_mask_prob=0.25, ref_2d_seq_cond=None):
        # x_start: BS X T X D 
        # ori_x_cond: BS X T X D' 
        # input_line_cond: BS X T X J X 3 
        # ref_2d_seq_cond: BS X T X D(J*2) 
        bs = x_start.shape[0] 
        t = torch.randint(0, self.num_timesteps, (bs,), device=x_start.device).long()

        if use_cfg:
            for_all_cond_mask = torch.bernoulli(torch.ones(bs).to(x_start.device) * cond_mask_prob) 

        if cond_mask is not None:
            if use_cfg:
                x_pose_cond = x_start * (1. - cond_mask) * (1 - for_all_cond_mask[:, None, None])
                x_cond = x_pose_cond 
            else:
                x_pose_cond = x_start * (1. - cond_mask)
                x_cond = x_pose_cond 
        else:
            x_cond = None 

        if input_line_cond is not None:
            if use_cfg:
                input_line_cond = input_line_cond * (1 - for_all_cond_mask[:, None, None, None])
                
            if x_cond is not None:
                x_cond = torch.cat((input_line_cond.reshape(bs, x_start.shape[1], -1), \
                    x_cond), dim=-1) 
            else:
                x_cond = input_line_cond.reshape(bs, x_start.shape[1], -1) 

        if ref_2d_seq_cond is not None:
            x_cond = torch.cat((ref_2d_seq_cond, x_cond), dim=-1)

        if language_input is not None:
            language_embedding = self.clip_encoder(language_input) # BS X d_model 
        else:
            language_embedding = None 
        
        curr_loss = self.p_losses(x_start, t, x_cond=x_cond, padding_mask=padding_mask, \
            motion_mask=motion_mask, language_embedding=language_embedding, \
            input_line_cond=input_line_cond)

        return curr_loss

    def get_projection_from_motion3d(self, num_views, cam_rot_mat, cam_trans, fk_jnts3d):
        # Project the optimized 3D motion to 2D 
        reprojected_ori_jnts2d_list = [] 
        reprojected_jnts2d_list = [] 

        for view_idx in range(num_views):
            _, _, reprojected_ori_jnts2d = perspective_projection(
                    fk_jnts3d, cam_rot_mat[view_idx], \
                    cam_trans[view_idx])

            reprojected_ori_jnts2d_list.append(reprojected_ori_jnts2d.clone())
            reprojected_jnts2d = normalize_pose2d_w_grad(reprojected_ori_jnts2d)
            
            reprojected_jnts2d_list.append(reprojected_jnts2d)

        reprojected_ori_jnts2d_list = torch.stack(reprojected_ori_jnts2d_list) # K X T X 18 X 2 
        reprojected_jnts2d_list = torch.stack(reprojected_jnts2d_list) # K X T X 18 X 2 

        seq_len = reprojected_jnts2d_list.shape[1]
        reprojected_jnts2d_list = reprojected_jnts2d_list.reshape(num_views, seq_len, -1) # K X T X (18*2)

        return reprojected_jnts2d_list, reprojected_ori_jnts2d_list, fk_jnts3d  
        # K X T X (18*2), K X T X 18 X 2, T X 18 X 3  

    def predict_noise_from_start(self, x_t, t, x_start):
        return (
            (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x_start) /
            (extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape))
        )

    def compute_sds_loss(self, pred_j2d, t, x_cond=None, padding_mask=None, \
            use_cfg=False, cfg_scale=3, x_cond_for_uncond_gen=None, language_embedding=None):
        # Normalized pred_j2d: K X T X (18*2)

        t = torch.full((pred_j2d.shape[0],), t, device=pred_j2d.device, dtype=torch.long)

        noise = torch.randn_like(pred_j2d) 
        noise_pred_j2d = self.q_sample(x_start=pred_j2d, t=t, noise=noise)

        if use_cfg:
            uncond_x_start_pred = self.denoise_fn(noise_pred_j2d, t, x_cond_for_uncond_gen, \
                padding_mask=padding_mask)
            
            cond_x_start_pred = self.denoise_fn(noise_pred_j2d, t, x_cond, \
                padding_mask=padding_mask)

            x_start_pred = uncond_x_start_pred + \
                    (cfg_scale * (cond_x_start_pred - uncond_x_start_pred))
        else:
            x_start_pred = self.denoise_fn(noise_pred_j2d, t, x_cond, \
                padding_mask=padding_mask, language_embedding=language_embedding)

        noise_pred = self.predict_noise_from_start(noise_pred_j2d, t, x_start=x_start_pred)

        grad = extract(self.p2_loss_weight, t, noise_pred.shape) * (noise_pred - noise)
        grad = torch.nan_to_num(grad)

        targets = (pred_j2d - grad).detach()
        # loss_sds = 0.5 * F.mse_loss(pred_j2d.float(), targets, \
        #     reduction='sum') / pred_j2d.shape[0]

        loss_sds = (0.5 * F.mse_loss(pred_j2d.float(), targets, \
            reduction='none') * padding_mask[:, 0, 1:][:, :, None]).sum() / pred_j2d.shape[0]

        # if t[0] < 5:
        #     print("loss sds:{0}".format((0.5 * F.mse_loss(pred_j2d.float(), targets, \
        #         reduction='none') * padding_mask[:, 0, 1:][:, :, None]).sum(dim=1).sum(dim=1)))

        #     import pdb 
        #     pdb.set_trace() 

        return loss_sds 

    def ddim_sample(self, x_start, cam_extrinsic=None, init_pose_3d=None, \
        cond_mask=None, padding_mask=None, language_input=None, rest_human_offsets=None):
        self.denoise_fn.eval() 
        self.clip_encoder.eval()

        shape = x_start.shape 
        batch, device, total_timesteps, sampling_timesteps, eta = \
        shape[0], self.betas.device, self.num_timesteps, 50, 1

        times = torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1)   # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:])) # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

        x = torch.randn(shape, device = device)

        if cond_mask is not None:
            x_pose_cond = x_start * (1. - cond_mask)
            x_cond = x_pose_cond 
        else:
            x_cond = None 

        if language_input is not None:
            language_embedding = self.clip_encoder(language_input)
        else:
            language_embedding = None 

        x_start = None

        for time, time_next in tqdm(time_pairs, desc = 'sampling loop time step'):
            time_cond = torch.full((batch,), time, device=device, dtype=torch.long)
            x_start = self.denoise_fn(x, time_cond, x_cond, padding_mask, \
                    language_embedding=language_embedding)

            pred_noise = self.predict_noise_from_start(x, time_cond, \
                x_start=x_start)

            if time_next < 0:
                x = x_start
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]

            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()

            noise = torch.randn_like(x)

            x = x_start * alpha_next.sqrt() + \
                c * pred_noise + \
                sigma * noise

        # x: K X T X (18*2)

        # Visualize projected 2D 
        vis_debug = True 
        if vis_debug:
            num_views = x.shape[0]

            ori_jnts2d = de_normalize_pose2d(x.reshape(num_views, -1, 18, 2)) # K X T X 18 X 2  

            tmp_debug_folder = "./tmp_debug_ddim_sampling"
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            tmp_debug_folder = os.path.join(tmp_debug_folder, timestamp)
            if not os.path.exists(tmp_debug_folder):
                os.makedirs(tmp_debug_folder)

            for view_idx in range(num_views):
                dest_vid_path = os.path.join(tmp_debug_folder, \
                    "view_idx_"+str(view_idx)+"_2d.mp4")

                gen_2d_motion_vis(ori_jnts2d[view_idx].detach().cpu().numpy(), \
                    dest_vid_path, use_smpl_jnts13=False)

        self.denoise_fn.train()
        self.clip_encoder.train() 

        return x

    def opt_paired_2d_w_sds_loss_and_jnts2line_dist_loss(self, x_start, cam_extrinsic, input_line_cond=None, \
        cond_mask=None, padding_mask=None, compute_epi_lines_func=None, init_jnts2d_seq=None):
        # x_start: K X T X (18*2) 
        # cam_extrinsic: K X 4 X 4 
        # input_line_cond: K X T X 18 X 3 
        # cond_mask: None 
        # padding_mask: K X 1 X 121 

        # x_start: BS X K X T X (18*2) 
        # cam_extrinsic: BS X K X 4 X 4 
        # input_line_cond: BS X K X T X 18 X 3 
        # cond_mask: None 
        # padding_mask: (BS*K) X 1 X 121 
        # init_jnts2d_seq: BS X K X T X (18*2)  
    
        self.denoise_fn.eval() 
        self.clip_encoder.eval()

        shape = x_start.shape 
        batch, device, total_timesteps = \
        shape[0], self.betas.device, self.num_timesteps
       
        if cond_mask is not None:
            x_pose_cond = x_start * (1. - cond_mask)
            x_cond = x_pose_cond 
        else:
            x_cond = None 

        if input_line_cond is not None:
            if x_cond is not None:
                x_cond = torch.cat((input_line_cond.reshape(x_start.shape[0], x_start.shape[1], -1), \
                    x_cond), dim=-1) 
            else:
                x_cond = input_line_cond.reshape(x_start.shape[0], x_start.shape[1], x_start.shape[2], -1) # BS X K X T X (18*3) 

        num_opt_views = x_start.shape[1]-1 
        num_steps = x_start.shape[2] 
        if init_jnts2d_seq is not None:
            self.motion_2d = init_jnts2d_seq 
            self.motion_2d.requires_grad_(True)
            self.optimizer_2d = Adam([self.motion_2d], lr=0.0005)
        else:
            if self.use_animal_data:
                num_joints = 17
            elif self.use_omomo_data:
                num_joints = 18 + 5 
            else:
                num_joints = 18 
            self.motion_2d = torch.zeros(batch, num_opt_views, x_start.shape[2], num_joints*2).to(x_start.device) # BS X (K-1) X T X (18*2)
            self.motion_2d.requires_grad_(True)
            self.optimizer_2d = Adam([self.motion_2d], lr=0.001)
       
        self.scheduler_2d = MultiStepLR(self.optimizer_2d, \
            milestones=[5000], gamma=0.1)

        curr_x_cond = x_cond[:, 1:] # BS X (K-1) X T X (18*3)

        padding_mask = padding_mask.reshape(batch, x_start.shape[1], 1, -1) # BS X K X 1 X 121 
        curr_padding_mask = padding_mask[:, 1:] # BS X (K-1) X 1 X 121
      
        opt_iterations = 5000
        # opt_iterations = 5
        for opt_iter in range(opt_iterations):
            self.optimizer_2d.zero_grad()

            denoise_step = random.randint(1, 999)

            loss_sds = self.compute_sds_loss(self.motion_2d.reshape(batch*num_opt_views, num_steps, -1), denoise_step, \
                curr_x_cond.reshape(batch*num_opt_views, num_steps, -1), \
                padding_mask=curr_padding_mask.reshape(batch*num_opt_views, 1, curr_padding_mask.shape[-1]))
 
            loss_jnts2line_a2b = torch.zeros(1).to(x_start.device)
            loss_jnts2line_b2a = torch.zeros(1).to(x_start.device) 
            loss_jnts2line_ref2a = torch.zeros(1).to(x_start.device)
            loss_jnts2line_ref2b = torch.zeros(1).to(x_start.device) 

            for tmp_idx in range(num_opt_views-1):
                target_epi_lines_in_b, _ = compute_epi_lines_func(self.motion_2d[:, tmp_idx], \
                                        reference_cam_extrinsic=cam_extrinsic[:, tmp_idx+1], \
                                        target_cam_extrinsic=cam_extrinsic[:, tmp_idx+2])
                loss_jnts2line_a2b += self.compute_joint2line_distance_loss(self.motion_2d[:, tmp_idx+1].reshape(batch, -1, num_joints, 2), \
                    target_epi_lines_in_b.detach(), padding_mask=padding_mask[:, tmp_idx+2])

                target_epi_lines_in_a, _ = compute_epi_lines_func(self.motion_2d[:, tmp_idx+1], \
                                        reference_cam_extrinsic=cam_extrinsic[:, tmp_idx+2], \
                                        target_cam_extrinsic=cam_extrinsic[:, tmp_idx+1])
                loss_jnts2line_b2a += self.compute_joint2line_distance_loss(self.motion_2d[:, tmp_idx].reshape(batch, -1, num_joints, 2), \
                    target_epi_lines_in_a.detach(), padding_mask=padding_mask[:, tmp_idx+1])

                loss_jnts2line_ref2a += self.compute_joint2line_distance_loss(self.motion_2d[:, tmp_idx].reshape(batch, -1, num_joints, 2), \
                    input_line_cond[:, tmp_idx+1], padding_mask=padding_mask[:, tmp_idx+1])

                loss_jnts2line_ref2b += self.compute_joint2line_distance_loss(self.motion_2d[:, tmp_idx+1].reshape(batch, -1, num_joints, 2), \
                    input_line_cond[:, tmp_idx+2], padding_mask=padding_mask[:, tmp_idx+2])

            # loss = loss_sds + (loss_jnts2line_a2b + loss_jnts2line_b2a) * 1e3 / (num_opt_views-1) + \
            #     (loss_jnts2line_ref2a + loss_jnts2line_ref2b) * 1e3 / (num_opt_views-1)

            # loss = loss_sds
            # loss = loss_sds + (loss_jnts2line_a2b + loss_jnts2line_b2a) * 1e2 / (num_opt_views-1) + \
            #      (loss_jnts2line_ref2a + loss_jnts2line_ref2b) * 1e2 / (num_opt_views-1)

            # loss = loss_sds +  (loss_jnts2line_ref2a + loss_jnts2line_ref2b) * 1e2 / (num_opt_views-1)

            loss = loss_sds + (loss_jnts2line_a2b + loss_jnts2line_b2a) * 1e3 / (num_opt_views-1) + \
                (loss_jnts2line_ref2a + loss_jnts2line_ref2b) * 1e3 / (num_opt_views-1)
            
            if denoise_step == 5:
                print("Optimization step:{0}, noise level:{1}, loss_sds:{2}, \
                    loss_jnts2line_a2b:{3}, loss_jnts2line_b2a:{4}".format(opt_iter, \
                    denoise_step, loss_sds.item(), loss_jnts2line_a2b.item(), loss_jnts2line_b2a.item()))
                
                # print("Loss in reference view of view a:{0},view b:{1}".format(loss_jnts2line_a2ref.item(), \
                #                                                                loss_jnts2line_b2ref.item()))

                print("Loss in view a jnts2line:{0},view b:{1}".format(loss_jnts2line_ref2a.item(), \
                                                            loss_jnts2line_ref2b.item()))
           
            loss.backward()

            self.optimizer_2d.step() 
            self.scheduler_2d.step() 

        self.denoise_fn.train()
        self.clip_encoder.train() 

        return self.motion_2d 

    def opt_3d_w_smal_animal_w_joints3d_input(self, target_jnts3d):
        # target_jnts3d: BS X T  X J X 3 
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        batch_size = target_jnts3d.shape[0] 
        smal_param = self.smal_fitter.init_param(target_jnts3d.reshape(-1, 17, 3), \
                            batch_size=batch_size) 
        # smpl_param {} 
        # betas: BS X 20 
        # log_beta_scales: BS X 6
        # global_rotation: (BS*T) X 3
        # trans: (BS*T) X 3
        # joint_rotations: (BS*T) X 34 X 3 

        optimizer = torch.optim.Adam(smal_param.values(), lr=1e-2)

        target_jnts3d = torch.from_numpy(target_jnts3d).to(device).float() # BS X T X J X 3 
        smal_param, smal_faces = self.smal_fitter.solve(smal_param=smal_param, optimizer=optimizer, \
            closure=self.smal_fitter.gen_closure(optimizer, smal_param, target_jnts3d.reshape(-1, 17, 3), batch_size=batch_size), \
            iter_max=1000) # Original 1000 , test initial smpl global orientation 

        # import pdb 
        # pdb.set_trace() 

        # Get optimized joint and vertices 
        smal_opt_jnts3d, smal_opt_verts, _ = self.smal_fitter.smal_forward(smal_param, batch_size=batch_size) 

        smal_opt_jnts3d_17 = smal_opt_jnts3d[:, self.smal_fitter.smal2xpose_idx_list] # (BS*T) X 17 X 3 
        smal_opt_jnts3d_17 = smal_opt_jnts3d_17.reshape(batch_size, -1, 17, 3) # BS X T X 17 X 3
        smal_opt_verts = smal_opt_verts.reshape(batch_size, -1, smal_opt_verts.shape[-2], 3) # BS X T X N_v X 3 

        #  smal_param.keys() dict_keys(['betas', 'log_beta_scales', 'global_rotation', 'trans', 'joint_rotations'])
        ori_smal_param_list = []
        smal_param_list = []
        scale_val_list = []

        for bs_idx in range(batch_size):
            curr_ori_smal_param = {} 
            curr_ori_smal_param['global_rotation'] = smal_param['global_rotation'].reshape(batch_size, -1, 3)[bs_idx].detach().cpu().numpy() # T X 3 
            curr_ori_smal_param['joint_rotations'] = smal_param['joint_rotations'].reshape(batch_size, -1, 17, 3)[bs_idx].detach().cpu().numpy() # T X 17 X 3
            curr_ori_smal_param['trans'] = smal_param['trans'].reshape(batch_size, -1, 3)[bs_idx].detach().cpu().numpy() # T X 3 
            curr_ori_smal_param['betas'] = smal_param['betas'].reshape(batch_size, -1)[bs_idx].detach().cpu().numpy() # 20  
            curr_ori_smal_param['log_beta_scales'] = smal_param['log_beta_scales'].reshape(batch_size, -1)[bs_idx].detach().cpu().numpy() # 6 

            curr_smal_param = {} 
            curr_smal_param['global_rotation'] = smal_param['global_rotation'].reshape(batch_size, -1, 3)[bs_idx].detach().cpu().numpy() # T X 3 
            curr_smal_param['joint_rotations'] = smal_param['joint_rotations'].reshape(batch_size, -1, 17, 3)[bs_idx].detach().cpu().numpy() # T X 17 X 3
            curr_smal_param['trans'] = smal_param['trans'].reshape(batch_size, -1, 3)[bs_idx].detach().cpu().numpy() # T X 3 
            curr_smal_param['betas'] = smal_param['betas'].reshape(batch_size, -1)[bs_idx].detach().cpu().numpy() # 20  
            curr_smal_param['log_beta_scales'] = smal_param['log_beta_scales'].reshape(batch_size, -1)[bs_idx].detach().cpu().numpy() # 6 
           
            scale_val = smal_param['scaling'][bs_idx].detach().cpu().numpy()[0] 
            curr_smal_param['scaling'] = np.asarray([1.])

            aligned_smal_trans = curr_smal_param['trans']
            aligned_smal_trans /= scale_val 
            curr_smal_param['trans'] = aligned_smal_trans 

            smal_opt_jnts3d_17[bs_idx] /= scale_val 
            smal_opt_verts[bs_idx] /= scale_val

            smal_param_list.append(curr_smal_param) 

            scale_val_list.append(scale_val)

        return smal_param_list, smal_faces, scale_val_list, smal_opt_jnts3d_17, smal_opt_verts 

    def opt_3d_w_smal_animal_w_joints2d_reprojection(self, target_jnts2d, cam_rot_mat, cam_trans):
        # target_jnts3d: BS X T  X J X 3 
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        batch_size = target_jnts2d.shape[0] 
        seq_len = target_jnts2d.shape[1] 

        init_target_jnts2d = target_jnts2d.clone()[:, 0:1].repeat(1, seq_len, 1, 1) # BS X 1 X 18 X 2 
        init_cam_rot_mat = cam_rot_mat[:, 0:1].repeat(1, seq_len, 1, 1) # BS X 1 X 3 X 3 
        init_cam_trans = cam_trans[:, 0:1].repeat(1, seq_len, 1) # BS X 1 X 3 

        init_smpl_param = self.smal_fitter2d.init_param(init_target_jnts2d.reshape(-1, 17, 2), \
                                batch_size=batch_size)
      
        # Stage1 optimization 
        optimizer = torch.optim.Adam(init_smpl_param.values(), lr=1e-2)

        init_target_jnts2d = init_target_jnts2d.reshape(-1, 17, 2).detach()  # (BS*1) X 18 X 2 
        init_cam_rot_mat = init_cam_rot_mat.reshape(-1, 3, 3).detach() # (BS*1) X 3 X 3 
        init_cam_trans = init_cam_trans.reshape(-1, 3).detach() # (BS*1) X 3 

        target_jnts2d = target_jnts2d.reshape(-1, 17, 2).detach()  # (BS*T) X 18 X 2 
        cam_rot_mat = cam_rot_mat.reshape(-1, 3, 3).detach() # (BS*T) X 3 X 3 
        cam_trans = cam_trans.reshape(-1, 3).detach() # (BS*T) X 3 

        smpl_param, smpl_faces = self.smal_fitter2d.solve(smal_param=init_smpl_param, \
            optimizer=optimizer, \
            closure=self.smal_fitter2d.gen_closure(optimizer, init_smpl_param, init_target_jnts2d, \
            init_cam_rot_mat, init_cam_trans, batch_size=batch_size), \
            iter_max=1000) # Original 1000 , test initial smpl global orientation 

        # # Stage2 optimization 
        smpl_param = {'global_rotation': torch.tensor(smpl_param['global_rotation'].detach().cpu().numpy(), dtype=torch.float32, device=device, requires_grad=True),
                    'trans': torch.tensor(smpl_param['trans'].detach().cpu().numpy(), dtype=torch.float32, device=device, requires_grad=True),
                    'joint_rotations': torch.tensor(smpl_param['joint_rotations'].detach().cpu().numpy(), dtype=torch.float32, device=device, requires_grad=False),
                    'scaling': torch.tensor(smpl_param['scaling'].detach().cpu().numpy(), dtype=torch.float32, device=device, requires_grad=False), 
                    'betas': smpl_param['betas'], 
                    'log_beta_scales': smpl_param['log_beta_scales']}

        optimizer = torch.optim.Adam(smpl_param.values(), lr=1e-2)

        smpl_param, smpl_faces = self.smal_fitter2d.solve(smal_param=smpl_param, optimizer=optimizer, \
            closure=self.smal_fitter2d.gen_closure(optimizer, smpl_param, target_jnts2d, cam_rot_mat, cam_trans, batch_size=batch_size), \
            iter_max=1000) # Original 1000 , test initial smpl global orientation 

        # # Stage3 optimization 
        smpl_param = {'global_rotation': torch.tensor(smpl_param['global_rotation'].detach().cpu().numpy(), dtype=torch.float32, device=device, requires_grad=True),
                    'trans': torch.tensor(smpl_param['trans'].detach().cpu().numpy(), dtype=torch.float32, device=device, requires_grad=True),
                    'joint_rotations': torch.tensor(smpl_param['joint_rotations'].detach().cpu().numpy(), dtype=torch.float32, device=device, requires_grad=True),
                    'scaling': torch.tensor(smpl_param['scaling'].detach().cpu().numpy(), dtype=torch.float32, device=device, requires_grad=False), 
                    'betas': smpl_param['betas'], 
                    'log_beta_scales': smpl_param['log_beta_scales']}

        optimizer = torch.optim.Adam(smpl_param.values(), lr=1e-2)

        smpl_param, smpl_faces = self.smal_fitter2d.solve(smal_param=smpl_param, optimizer=optimizer, \
            closure=self.smal_fitter2d.gen_closure(optimizer, smpl_param, target_jnts2d, cam_rot_mat, cam_trans, batch_size=batch_size), \
            iter_max=1000) # Original 1000 , test initial smpl global orientation 

        # Get optimized joint and vertices 
        smal_opt_jnts3d, smal_opt_verts, _ = self.smal_fitter2d.smal_forward(smpl_param, batch_size=batch_size) 

        smal_opt_jnts3d_17 = smal_opt_jnts3d[:, self.smal_fitter2d.smal2xpose_idx_list] # (BS*T) X 17 X 3 
        smal_opt_jnts3d_17 = smal_opt_jnts3d_17.reshape(batch_size, -1, 17, 3) # BS X T X 17 X 3
        smal_opt_verts = smal_opt_verts.reshape(batch_size, -1, smal_opt_verts.shape[-2], 3) # BS X T X N_v X 3 

        #  smal_param.keys() dict_keys(['betas', 'log_beta_scales', 'global_rotation', 'trans', 'joint_rotations'])
        ori_smal_param_list = []
        smal_param_list = []
        scale_val_list = []

        for bs_idx in range(batch_size):
            curr_ori_smal_param = {} 
            curr_ori_smal_param['global_rotation'] = smpl_param['global_rotation'].reshape(batch_size, -1, 3)[bs_idx].detach().cpu().numpy() # T X 3 
            curr_ori_smal_param['joint_rotations'] = smpl_param['joint_rotations'].reshape(batch_size, -1, 17, 3)[bs_idx].detach().cpu().numpy() # T X 17 X 3
            curr_ori_smal_param['trans'] = smpl_param['trans'].reshape(batch_size, -1, 3)[bs_idx].detach().cpu().numpy() # T X 3 
            curr_ori_smal_param['betas'] = smpl_param['betas'].reshape(batch_size, -1)[bs_idx].detach().cpu().numpy() # 20  
            curr_ori_smal_param['log_beta_scales'] = smpl_param['log_beta_scales'].reshape(batch_size, -1)[bs_idx].detach().cpu().numpy() # 6 

            curr_smal_param = {} 
            curr_smal_param['global_rotation'] = smpl_param['global_rotation'].reshape(batch_size, -1, 3)[bs_idx].detach().cpu().numpy() # T X 3 
            curr_smal_param['joint_rotations'] = smpl_param['joint_rotations'].reshape(batch_size, -1, 17, 3)[bs_idx].detach().cpu().numpy() # T X 17 X 3
            curr_smal_param['trans'] = smpl_param['trans'].reshape(batch_size, -1, 3)[bs_idx].detach().cpu().numpy() # T X 3 
            curr_smal_param['betas'] = smpl_param['betas'].reshape(batch_size, -1)[bs_idx].detach().cpu().numpy() # 20  
            curr_smal_param['log_beta_scales'] = smpl_param['log_beta_scales'].reshape(batch_size, -1)[bs_idx].detach().cpu().numpy() # 6 
           
            scale_val = smpl_param['scaling'][bs_idx].detach().cpu().numpy()[0] 
            curr_smal_param['scaling'] = np.asarray([1.])

            aligned_smal_trans = curr_smal_param['trans']
            aligned_smal_trans /= scale_val 
            curr_smal_param['trans'] = aligned_smal_trans 

            smal_opt_jnts3d_17[bs_idx] /= scale_val 
            smal_opt_verts[bs_idx] /= scale_val

            smal_param_list.append(curr_smal_param) 
            scale_val_list.append(scale_val) 

        return smal_param_list, smpl_faces, scale_val_list, smal_opt_jnts3d_17, smal_opt_verts 

    def opt_2d_w_sds_loss_for_debug(self, x_start, input_line_cond=None, \
        cond_mask=None, padding_mask=None):

        # val_data[1:], \
        # input_line_cond=epi_line_conditions, \
        # cond_mask=cond_mask[1:], padding_mask=padding_mask[1:]

        self.denoise_fn.eval() 
        self.clip_encoder.eval()

        shape = x_start.shape 
        batch, device, total_timesteps = \
        shape[0], self.betas.device, self.num_timesteps

        x = torch.randn(shape, device = device)
       
        if cond_mask is not None:
            x_pose_cond = x_start * (1. - cond_mask)
            x_cond = x_pose_cond 
        else:
            x_cond = None 

        if input_line_cond is not None:
            if x_cond is not None:
                x_cond = torch.cat((input_line_cond.reshape(x_start.shape[0], x_start.shape[1], -1), \
                    x_cond), dim=-1) 
            else:
                x_cond = input_line_cond.reshape(x_start.shape[0], x_start.shape[1], -1)

        vis_debug = False 
        if vis_debug:
            tmp_debug_folder = "./tmp_opt_amass_pose2d_w_sds"

            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            tmp_debug_folder = os.path.join(tmp_debug_folder, timestamp)
            if not os.path.exists(tmp_debug_folder):
                os.makedirs(tmp_debug_folder)
      
        self.motion_2d = torch.zeros(x_start.shape[0], x_start.shape[1], 18*2).to(x_start.device)
        self.motion_2d.requires_grad_(True)
        self.optimizer_2d = Adam([self.motion_2d], lr=0.0005)

        num_views = x_start.shape[0]
        seq_len = x_start.shape[1]
      
        opt_iterations = 50000
        for opt_iter in range(opt_iterations):
            self.optimizer_2d.zero_grad()

            denoise_step = random.randint(1, 999)

            loss_sds = self.compute_sds_loss(self.motion_2d, denoise_step, \
                x_cond, padding_mask=padding_mask)
        
            loss = loss_sds
            if denoise_step < 20:
                print("Optimization step:{0}, noise level:{1}, loss_sds:{2}".format(opt_iter, \
                    denoise_step, loss_sds.item()))
           
            loss.backward()

            self.optimizer_2d.step() 

        if vis_debug:
            # Visualize projected 2D 
            normalized_opt_motion2d_for_vis = torch.cat((x_start[0:1], self.motion_2d), dim=0)
            opt_motion2d_for_vis = de_normalize_pose2d(normalized_opt_motion2d_for_vis.reshape(-1, \
                            x_start.shape[1], 18, 2))
            for view_idx in range(num_views+1):
                dest_vid_path = os.path.join(tmp_debug_folder, \
                    "view_idx_"+str(view_idx)+"_2d.mp4")
                
                gen_2d_motion_vis(opt_motion2d_for_vis[view_idx].detach().cpu().numpy(), \
                    dest_vid_path)

        self.denoise_fn.train()
        self.clip_encoder.train() 

        return self.motion_2d 

    def opt_3d_w_multi_view_2d_res(self, line_cond_2d_res, cam_extrinsic, \
        padding_mask=None, actual_seq_len=None, input_line_cond=None, \
        x_start=None, cond_mask=None, language_input=None):
        # line_cond_2d_res: BS X K X T X (18*2) 
        # cam_extrinsic: BS X K X 4 X 4
        # padding_mask: (BS*K) X 1 X 121 
        # actual_seq_len: a value 120 
        # input_line_cond: BS X K X T X 18 X 3
        # x_start: BS X K X T X (18*2) 
            
        # Prepare for optimization during denoising step 
        shape = x_start.shape 
        batch, device, total_timesteps = \
        shape[0], self.betas.device, self.num_timesteps
    
        if cond_mask is not None:
            x_pose_cond = x_start * (1. - cond_mask)
            x_cond = x_pose_cond 
        else:
            x_cond = None 

        if language_input is not None:
            language_embedding = self.clip_encoder(language_input)
        else:
            language_embedding = None 

        if input_line_cond is not None:
            if x_cond is not None:
                x_cond = torch.cat((input_line_cond.reshape(x_start.shape[0], x_start.shape[1], -1), \
                    x_cond), dim=-1) 
            else:
                x_cond = input_line_cond.reshape(x_start.shape[0], x_start.shape[1], x_start.shape[2], -1) # BS X K X T X (18*3)
      
        num_views = line_cond_2d_res.shape[1]
        seq_len = line_cond_2d_res.shape[2]

        padding_mask = padding_mask.reshape(batch, num_views, 1, -1) # BS X K X 1 X 121

        cam_rot_mat = cam_extrinsic[:, :, :3, :3].unsqueeze(2).repeat(1, 1, seq_len, 1, 1) # BS X K X T X 3 X 3 
        cam_trans = cam_extrinsic[:, :, :3, -1].unsqueeze(2).repeat(1, 1, seq_len, 1) # BS X K X T X 3 
       
        # Initialize 3D pose
        if self.use_animal_data:
            num_joints = 17
        elif self.use_omomo_data:
            num_joints = 18 + 5 
        else:
            num_joints = 18
        self.motion_3d = torch.zeros(batch, seq_len, num_joints, 3).to(line_cond_2d_res.device) # BS X T X 18 X 3 
        self.motion_3d.requires_grad_(True) 

        param_for_opt_list = []
        param_for_opt_list.append(self.motion_3d)
            
        self.optimizer_3d = Adam(param_for_opt_list, lr=0.01)

        self.scheduler_3d = MultiStepLR(self.optimizer_3d, \
                milestones=[5000], gamma=0.1)


        opt_iterations = 500
        # opt_iterations = 10000
        # opt_iterations = 1 
        # opt_iterations = 1000 
        for opt_iter in range(opt_iterations):
            self.optimizer_3d.zero_grad()

            # Project 3D to 2D pose sequences 
            batch_reprojected_jnts2d_list = []
            final_jnts3d_list = []
            for bs_idx in range(batch):
                reprojected_jnts2d_list, reprojected_ori_jnts2d_list, final_jnts3d = \
                    self.get_projection_from_motion3d(num_views, cam_rot_mat[bs_idx], cam_trans[bs_idx], \
                    self.motion_3d[bs_idx])
                
                batch_reprojected_jnts2d_list.append(reprojected_jnts2d_list) 
                final_jnts3d_list.append(final_jnts3d)

            batch_reprojected_jnts2d_list = torch.stack(batch_reprojected_jnts2d_list, dim=0) # BS X K X T X (18*2) 
            final_jnts3d_list = torch.stack(final_jnts3d_list, dim=0) # BS X T X 18 X 3 

            loss_projection = F.mse_loss(batch_reprojected_jnts2d_list, \
                                line_cond_2d_res, \
                                reduction='none') * padding_mask[:, :, 0, 1:][:, :, :, None]

            loss = loss_projection.mean() 
            if opt_iter % 20 == 0:
                print("Optimization step:{0}, loss:{1}".format(opt_iter, \
                    loss.item()))
            
            loss.backward()

            self.optimizer_3d.step() 
            self.scheduler_3d.step()

        # reprojected_jnts2d_list, reprojected_ori_jnts2d_list, final_jnts3d = \
        #         self.get_projection_from_motion3d(num_views, cam_rot_mat, cam_trans, \
        #         self.motion_3d)
       
        return batch_reprojected_jnts2d_list, final_jnts3d_list, self.motion_3d
        # return batch_reprojected_jnts2d_list, final_jnts3d_list, self.motion_3d
        # BS X K X T X (18*2), BS X T X 18 X 3, BS X T X 18 X 3 

    def opt_3d_w_multi_view_2d_res_w_fixed_local_pose(self, line_cond_2d_res, cam_extrinsic, \
        local_jpos3d, padding_mask=None):
        # line_cond_2d_res: BS X K X T X (18*2) 
        # cam_extrinsic: BS X K X 4 X 4
        # padding_mask: (BS*K) X 1 X 121 
        # local_jpos3d: BS X T X 18 X 3 
            
        # Prepare for optimization during denoising step 
        shape = line_cond_2d_res.shape 
        batch, device, total_timesteps = \
        shape[0], self.betas.device, self.num_timesteps
          
        num_views = line_cond_2d_res.shape[1]
        seq_len = line_cond_2d_res.shape[2]

        padding_mask = padding_mask.reshape(batch, num_views, 1, -1) # BS X K X 1 X 121

        cam_rot_mat = cam_extrinsic[:, :, :3, :3].unsqueeze(2).repeat(1, 1, seq_len, 1, 1) # BS X K X T X 3 X 3 
        cam_trans = cam_extrinsic[:, :, :3, -1].unsqueeze(2).repeat(1, 1, seq_len, 1) # BS X K X T X 3 
       
        # Initialize 3D pose
        self.motion_3d = torch.zeros(batch, seq_len, 1, 3).to(line_cond_2d_res.device) # BS X T X 1 X 3 
        self.motion_3d.requires_grad_(True) 

        self.seq_scale = torch.ones(batch, 1, 1, 1).to(line_cond_2d_res.device) 
        self.seq_scale.requires_grad_(True) 
            
        self.optimizer_3d = Adam([self.motion_3d, self.seq_scale], lr=0.01)

        self.scheduler_3d = MultiStepLR(self.optimizer_3d, \
                milestones=[5000], gamma=0.1)

        opt_iterations = 500
        for opt_iter in range(opt_iterations):
            self.optimizer_3d.zero_grad()

            # Project 3D to 2D pose sequences 
            batch_reprojected_jnts2d_list = []
            final_jnts3d_list = []
            for bs_idx in range(batch):
                reprojected_jnts2d_list, reprojected_ori_jnts2d_list, final_jnts3d = \
                    self.get_projection_from_motion3d(num_views, cam_rot_mat[bs_idx], cam_trans[bs_idx], \
                    (self.motion_3d[bs_idx]+local_jpos3d[bs_idx])*self.seq_scale[bs_idx])
                
                batch_reprojected_jnts2d_list.append(reprojected_jnts2d_list) 
                final_jnts3d_list.append(final_jnts3d)

            batch_reprojected_jnts2d_list = torch.stack(batch_reprojected_jnts2d_list, dim=0) # BS X K X T X (18*2) 
            final_jnts3d_list = torch.stack(final_jnts3d_list, dim=0) # BS X T X 18 X 3 
               
            loss_projection = F.mse_loss(batch_reprojected_jnts2d_list, \
                                line_cond_2d_res, \
                                reduction='none') * padding_mask[:, :, 0, 1:][:, :, :, None]

            loss = loss_projection.mean() 
            if opt_iter % 20 == 0:
                print("Optimization step:{0}, loss:{1}".format(opt_iter, \
                    loss.item()))
            
            loss.backward()

            self.optimizer_3d.step() 
            self.scheduler_3d.step()
       
        # return batch_reprojected_jnts2d_list, final_jnts3d_list, (self.motion_3d+local_jpos3d)*self.seq_scale 
        return batch_reprojected_jnts2d_list, final_jnts3d_list/self.seq_scale, self.motion_3d+local_jpos3d 

        # return batch_reprojected_jnts2d_list, final_jnts3d_list, self.motion_3d
        # BS X K X T X (18*2), BS X T X 18 X 3, BS X T X 18 X 3 

        # return line_cond_2d_res*curr_scale_view, final_jnts3d, self.motion_3d # For debug scale 

    def opt_3d_w_vposer_w_joints3d_input(self, target_jnts3d, center_all_jnts2d=False):
        # target_jnts3d: T X 18 X 3/BS X T X 18 X 3 
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # fitter = SmplVPoserFitter()

        batch_size = target_jnts3d.shape[0] 
        smpl_param = self.fitter.init_param(target_jnts3d.reshape(-1, 18, 3), \
                                center_all_jnts2d=center_all_jnts2d, batch_size=batch_size) 
      
        smpl_param = {'orient': torch.tensor(smpl_param['orient'], dtype=torch.float32, device=device, requires_grad=True),
                    'trans': torch.tensor(smpl_param['trans'], dtype=torch.float32, device=device, requires_grad=True),
                    'latent': torch.zeros((batch_size*target_jnts3d.shape[1], 32), dtype=torch.float32, device=device, requires_grad=True),
                    'scaling': torch.tensor(smpl_param['scaling'], dtype=torch.float32, device=device, requires_grad=True)}

        optimizer = torch.optim.Adam(smpl_param.values(), lr=1e-2)
        # optimizer = torch.optim.Adam(smpl_param.values(), lr=1e-3)

        # if fixed_scale_seq is not None:
        #     target_jnts3d = torch.from_numpy(target_jnts3d).to(device) * torch.from_numpy(fixed_scale_seq).float().to(device)[:, None, None] 
        # else:
        target_jnts3d = torch.from_numpy(target_jnts3d).to(device)

        target_jnts3d = target_jnts3d.reshape(-1, 18, 3) # (BS*T) X 18 X 3 
        ori_smpl_param, smpl_param, smpl_faces = self.fitter.solve(smpl_param=smpl_param, optimizer=optimizer, \
            closure=self.fitter.gen_closure(optimizer, smpl_param, target_jnts3d, batch_size=batch_size), \
            iter_max=1000) # Original 1000 , test initial smpl global orientation 

        ori_smpl_param_list = []
        smpl_param_list = []
        keypoints3d_list = [] 
        scale_val_list = [] 

        for bs_idx in range(batch_size):
            # Force scale to 1 
            curr_ori_smpl_param = {} 
            curr_ori_smpl_param['orient'] = ori_smpl_param['orient'].reshape(batch_size, -1, 3)[bs_idx] # T X 3 
            curr_ori_smpl_param['trans'] = ori_smpl_param['trans'].reshape(batch_size, -1, 3)[bs_idx] # T X 3 
            curr_ori_smpl_param['latent'] = ori_smpl_param['latent'].reshape(batch_size, -1, 32)[bs_idx] # T X 32
            curr_ori_smpl_param['scaling'] = ori_smpl_param['scaling'].reshape(batch_size, -1)[bs_idx] # 1 

            ori_smpl_param_list.append(curr_ori_smpl_param) 

            curr_smpl_param = {} 
            curr_smpl_param['smpl_trans'] = smpl_param['smpl_trans'].reshape(batch_size, -1, 3)[bs_idx] # T X 3 
            curr_smpl_param['smpl_poses'] = smpl_param['smpl_poses'].reshape(batch_size, -1, 24, 3)[bs_idx] # T X 24 X 3
            curr_smpl_param['smpl_scaling'] = smpl_param['smpl_scaling'].reshape(batch_size, -1)[bs_idx] # 1 

            scale_val = curr_smpl_param['smpl_scaling'][0]
            curr_smpl_param['smpl_scaling'] = np.asarray([1.])

            aligned_smpl_trans = curr_smpl_param['smpl_trans']
            aligned_smpl_trans /= scale_val 
            
            keypoints3d = target_jnts3d.reshape(batch_size, -1, 18, 3)[bs_idx]/scale_val 

            curr_smpl_param['smpl_trans'] = aligned_smpl_trans 
            smpl_param_list.append(curr_smpl_param) 

            keypoints3d_list.append(keypoints3d) 
            scale_val_list.append(scale_val) 

        # return ori_smpl_param, smpl_param, keypoints3d.detach().cpu().numpy(), smpl_faces, scale_val   
        return ori_smpl_param_list, smpl_param_list, keypoints3d_list, smpl_faces, scale_val_list  

    def opt_3d_object_w_joints3d_input(self, target_jnts3d):
        # target_jnts3d: BS X T X 5 X 3 
        # obj_rest_verts: N_v X 3 

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        batch_size = target_jnts3d.shape[0] 

        obj_param = self.object_fitter.init_param(target_jnts3d.reshape(-1, 5, 3), batch_size=batch_size) 

        obj_param['obj_orient'] = torch.from_numpy(obj_param['obj_orient']).float().to(device)
        obj_param['obj_orient'].requires_grad = True 
        obj_param['obj_trans'] = torch.from_numpy(obj_param['obj_trans']).float().to(device)
        obj_param['obj_trans'].requires_grad = True
        obj_param['obj_scaling'] = torch.from_numpy(obj_param['obj_scaling']).float().to(device)
        obj_param['obj_scaling'].requires_grad = True 

        optimizer = torch.optim.Adam(obj_param.values(), lr=1e-2)
      
        target_jnts3d = torch.from_numpy(target_jnts3d).to(device)

        target_jnts3d = target_jnts3d.reshape(-1, 5, 3) # (BS*T) X 5 X 3 
        ori_obj_param, obj_param, obj_verts, obj_faces = self.object_fitter.solve(obj_param=obj_param, optimizer=optimizer, \
            closure=self.object_fitter.gen_closure(optimizer, obj_param, target_jnts3d, batch_size=batch_size), \
            iter_max=500, batch_size=batch_size) # Original 1000 , test initial smpl global orientation 

        ori_obj_param_list = []
        obj_param_list = []
        keypoints3d_list = []
        scale_val_list = []

        obj_verts_list = [] 

        for bs_idx in range(batch_size):
            # Force scale to 1 
            curr_ori_smpl_param = {} 
            curr_ori_smpl_param['obj_orient'] = ori_obj_param['obj_orient'].reshape(batch_size, -1, 3)[bs_idx] # T X 3 
            curr_ori_smpl_param['obj_trans'] = ori_obj_param['obj_trans'].reshape(batch_size, -1, 3)[bs_idx] # T X 3 
            curr_ori_smpl_param['obj_scaling'] = ori_obj_param['obj_scaling'].reshape(batch_size, -1)[bs_idx] # 1 

            ori_obj_param_list.append(curr_ori_smpl_param) 

            curr_obj_param = {} 
            curr_obj_param['obj_trans'] = obj_param['obj_trans'].reshape(batch_size, -1, 3)[bs_idx] # T X 3 
            curr_obj_param['obj_orient'] = obj_param['obj_orient'].reshape(batch_size, -1, 24, 3)[bs_idx] # T X 24 X 3
            curr_obj_param['obj_scaling'] = obj_param['obj_scaling'].reshape(batch_size, -1)[bs_idx] # 1 

            scale_val = curr_obj_param['obj_scaling'][0]
            curr_obj_param['obj_scaling'] = np.asarray([1.])

            aligned_obj_trans = curr_obj_param['obj_trans']
            aligned_obj_trans /= scale_val 
            
            keypoints3d = target_jnts3d.reshape(batch_size, -1, 5, 3)[bs_idx]/scale_val 

            curr_obj_verts = obj_verts.reshape(batch_size, -1, obj_verts.shape[-2], 3)[bs_idx]/scale_val # T X N_v X 3

            curr_obj_param['obj_trans'] = aligned_obj_trans 
            obj_param_list.append(curr_obj_param) 

            keypoints3d_list.append(keypoints3d) 
            scale_val_list.append(scale_val) 

            obj_verts_list.append(curr_obj_verts)

        return ori_obj_param_list, obj_param_list, keypoints3d_list, obj_verts_list, obj_faces, scale_val_list   
   
    # def opt_3d_by_multi_step_recon_loss(self, x_start, cam_extrinsic, \
    #     cond_mask=None, padding_mask=None, use_sds_loss=True):

    #     self.denoise_fn.eval() 
    #     self.clip_encoder.eval()

    #     shape = x_start.shape 
    #     batch, device, total_timesteps, sampling_timesteps, eta = \
    #     shape[0], self.betas.device, self.num_timesteps, 50, 1

    #     times = torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1)   # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
    #     times = list(reversed(times.int().tolist()))
    #     time_pairs = list(zip(times[:-1], times[1:])) # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

    #     x = torch.randn(shape, device = device)
       
    #     if cond_mask is not None:
    #         x_pose_cond = x_start * (1. - cond_mask)
    #         x_cond = x_pose_cond 
    #     else:
    #         x_cond = None 

    #     language_embedding = None 

    #     # Initialize 3D pose
    #     self.motion_3d = torch.zeros(x_start.shape[1], 18, 3).to(x_start.device)
    #     self.root_trans_3d = torch.zeros(x_start.shape[1], 1, 3).to(x_start.device)

    #     self.motion_3d.requires_grad_(True)
    #     self.root_trans_3d.requires_grad_(True) 
    #     self.optimizer_3d = Adam([self.motion_3d, self.root_trans_3d], lr=0.001)

    #     num_views = x_start.shape[0]
    #     seq_len = x_start.shape[1]
    #     cam_rot_mat = cam_extrinsic[:, :3, :3].unsqueeze(1).repeat(1, seq_len, 1, 1) # K X T X 3 X 3 
    #     cam_trans = cam_extrinsic[:, :3, -1].unsqueeze(1).repeat(1, seq_len, 1) # K X T X 3 

    #     opt_iterations = 3000
    #     t_max = 1000 
    #     for opt_iter in range(opt_iterations):
    #         self.optimizer_3d.zero_grad()

    #         # Project 3D to 2D pose sequences 
    #         reprojected_jnts2d_list, reprojected_ori_jnts2d_list, _ = \
    #             self.get_projection_from_motion3d(num_views, cam_rot_mat, cam_trans, self.motion_3d)

    #         # Sample a t in [t_min, t_max]
    #         t_min = times[-2] + (times[0]-times[-2]) * (1 - opt_iter/opt_iterations)
    #         t_min = int(t_min) 

    #         denoise_step = random.randint(t_min, t_max) 

    #         # Add noise in noise level t to the projected 2D pose sequences 
    #         # Run K steps to get the predicted clean x in noise level 0 
    #         # Compute the reconstruction loss between the projected 2D poses and the clean x0 
    #         if use_sds_loss:
    #             denoise_step = random.randint(1, 999)
    #             loss_sds = self.compute_sds_loss(reprojected_jnts2d_list, denoise_step, \
    #                 x_cond, padding_mask=padding_mask)
    #         else:
    #             loss_recon = self.compute_multi_step_recon_loss(reprojected_jnts2d_list, \
    #                 denoise_step, x_cond, padding_mask=padding_mask)
                
    #         if use_sds_loss:
    #             loss = loss_sds 
    #             print("Optimization step:{0}, loss_sds:{1}".format(opt_iter, \
    #                 loss_sds.item()))
    #         else:
    #             loss = loss_recon 
    #             print("Optimization step:{0}, loss_recon:{1}".format(opt_iter, \
    #                 loss_recon.item()))

    #         loss.backward()

    #         self.optimizer_3d.step() 

    #     reprojected_jnts2d_list, reprojected_ori_jnts2d_list, final_jnts3d = \
    #             self.get_projection_from_motion3d(num_views, cam_rot_mat, cam_trans, self.motion_3d)

    #     self.denoise_fn.train()
    #     self.clip_encoder.train() 

    #     return reprojected_jnts2d_list, final_jnts3d 

    def opt_3d_by_multi_step_recon_loss(self, x_start, cam_extrinsic, \
        input_line_cond=None, \
        cond_mask=None, padding_mask=None, actual_seq_len=None, use_sds_loss=True):
    
        self.denoise_fn.eval() 
        self.clip_encoder.eval()

        shape = x_start.shape 
        batch, device, total_timesteps, sampling_timesteps, eta = \
        shape[0], self.betas.device, self.num_timesteps, 50, 1

        times = torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1)   # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:])) # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

        x = torch.randn(shape, device = device)
       
        if cond_mask is not None:
            x_pose_cond = x_start * (1. - cond_mask)
            x_cond = x_pose_cond 
        else:
            x_cond = None 

        if input_line_cond is not None:
            if x_cond is not None:
                x_cond = torch.cat((input_line_cond.reshape(x_start.shape[0], x_start.shape[1], -1), \
                    x_cond), dim=-1) 
            else:
                x_cond = input_line_cond.reshape(x_start.shape[0], x_start.shape[1], -1) 

        else:
            x_cond = None 
       
        language_embedding = None 

        # Initialize 3D pose
        
        self.motion_3d = torch.zeros(x_start.shape[1], 18, 3).to(x_start.device)
        
        # self.root_trans_3d = torch.zeros(x_start.shape[1], 1, 3).to(x_start.device)

        self.motion_3d.requires_grad_(True)
        # self.root_trans_3d.requires_grad_(True) 
        # self.optimizer_3d = Adam([self.motion_3d, self.root_trans_3d], lr=0.0) # Loss is about 0.03
        # self.optimizer_3d = Adam([self.motion_3d, self.root_trans_3d], lr=0.1)
        # self.optimizer_3d = Adam([self.motion_3d, self.root_trans_3d], lr=0.0001)
        self.optimizer_3d = Adam([self.motion_3d], lr=0.01)
        # self.optimizer_3d = Adam([self.motion_3d], lr=0.0)

        num_views = x_start.shape[0]
        seq_len = x_start.shape[1]
        cam_rot_mat = cam_extrinsic[:, :3, :3].unsqueeze(1).repeat(1, seq_len, 1, 1) # K X T X 3 X 3 
        cam_trans = cam_extrinsic[:, :3, -1].unsqueeze(1).repeat(1, seq_len, 1) # K X T X 3 

        opt_iterations = 3000
        t_max = 1000 
        # t_max = 200 
        for opt_iter in range(opt_iterations):
            self.optimizer_3d.zero_grad()

            # Project 3D to 2D pose sequences 
            reprojected_jnts2d_list, reprojected_ori_jnts2d_list, _ = \
                self.get_projection_from_motion3d(num_views, cam_rot_mat, cam_trans, self.motion_3d)

            # Sample a t in [t_min, t_max]
            t_min = times[-2] + (times[0]-times[-2]) * (1 - opt_iter/opt_iterations)
            t_min = int(t_min) 

            denoise_step = random.randint(t_min, t_max) 

            # Add noise in noise level t to the projected 2D pose sequences 
            # Run K steps to get the predicted clean x in noise level 0 
            # Compute the reconstruction loss between the projected 2D poses and the clean x0 
            if use_sds_loss:
                denoise_step = random.randint(1, 999)
                loss_sds = self.compute_sds_loss(reprojected_jnts2d_list, denoise_step, \
                    x_cond, padding_mask=padding_mask)
            else:
                loss_recon = self.compute_multi_step_recon_loss(reprojected_jnts2d_list, \
                    denoise_step, x_cond, padding_mask=padding_mask)
                
            if use_sds_loss:
                # loss = loss_sds 
               
                loss_projection = F.mse_loss(reprojected_jnts2d_list[0:1], \
                                x_start[0:1], \
                                reduction='none') * padding_mask[0:1, 0, 1:][:, :, None]

                loss = loss_projection.mean() * 1e3 + loss_sds 

                if denoise_step < 5:
                    print("Optimization step:{0}, loss_sds:{1}, loss projection:{2}".format(opt_iter, \
                        loss_sds.item(), loss_projection.mean().item()))
            else:
                loss = loss_recon 
                print("Optimization step:{0}, loss_recon:{1}".format(opt_iter, \
                    loss_recon.item()))

            loss.backward()

            self.optimizer_3d.step() 

        reprojected_jnts2d_list, reprojected_ori_jnts2d_list, final_jnts3d = \
                self.get_projection_from_motion3d(num_views, cam_rot_mat, cam_trans, self.motion_3d)

        # final_jnts3d, final_skel_pose, skel_faces, unscaled_ori_opt_jnts3d = \
        #     self.SMPL_fitting_w_jpos3d(final_jnts3d.detach().cpu().numpy()[:actual_seq_len])

        # if vis_debug:
        #     # Visualize projected 2D 
        #     for view_idx in range(num_views):
        #         dest_vid_path = os.path.join(tmp_debug_folder, \
        #             "view_idx_"+str(view_idx)+"_2d.mp4")

        #         gen_2d_motion_vis(reprojected_ori_jnts2d_list[view_idx].detach().cpu().numpy(), \
        #             dest_vid_path)

        #     # Visualize optimized 3D 
        #     # dest_vid_3d_path = os.path.join(tmp_debug_folder, \
        #     #     "denoise_ddim_opt_3d_local.mp4")
        #     # plot_3d_motion(dest_vid_3d_path, self.motion_3d.detach().cpu().numpy(), \
        #     #     use_smpl_jnts13=use_smpl_rep) 

        #     dest_vid_3d_path_global = os.path.join(tmp_debug_folder, \
        #         "denoise_ddim_opt_3d_global.mp4")
        #     plot_3d_motion(dest_vid_3d_path_global, \
        #         final_jnts3d.detach().cpu().numpy()) 

            # converted_jnts3d = self.fit_smpl_to_jnts3d(bm_dict)
            # dest_vid_3d_path_global_smpl = os.path.join(tmp_debug_folder, \
            #     "denoise_ddim_opt_3d_global_smpl_converted.mp4")
            # plot_3d_motion(dest_vid_3d_path_global_smpl, \
            #     converted_jnts3d.detach().cpu().numpy()) 

        self.denoise_fn.train()
        self.clip_encoder.train() 

        return reprojected_jnts2d_list, final_jnts3d 
        # return reprojected_jnts2d_list, final_jnts3d, final_skel_pose, skel_faces, self.motion_3d