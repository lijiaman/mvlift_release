import torch
from torch import nn

import torch.nn.functional as F

from einops import rearrange, reduce
from einops.layers.torch import Rearrange

from m2d.model.transformer_module import Decoder, MultiViewDecoder 

import numpy as np 

class VAE(nn.Module):
    def __init__(self, encoder, decoder, kl_loss_weight=0.0):
        super().__init__()
       
        self.encoder = encoder
        self.decoder = decoder

        self.kl_loss_weight = kl_loss_weight 

        self.window = 120 

    def kl_loss(self, logvar, mu):
        loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)

        return loss.mean()

    def reparametrize(self, pred_mean, pred_logvar):
        random_z = torch.randn_like(pred_mean)
        vae_z = random_z * torch.exp(0.5 * pred_logvar)
        vae_z = vae_z + pred_mean
        return vae_z

    def forward(self, motion, seq_len):
        # motion: BS X T X J X 2  
        # seq_len: BS 

        motion = motion.reshape(motion.shape[0], motion.shape[1], -1) # BS X T X (J*2)

        # Generate padding mask for encoder 
        enc_seq_len = seq_len + 2 # BS, + 1 since we need additional timestep for noise level 
        tmp_mask = torch.arange(self.window+2).expand(motion.shape[0], \
        self.window+2) < enc_seq_len[:, None].repeat(1, self.window+2) # BS X (T+2) 
        enc_padding_mask = tmp_mask[:, None, :].to(motion.device) # BS X 1 X (T+2)

        mu, logvar = self.encoder(motion, enc_padding_mask) # BS X D, BS X D 

        # Get sampled latent vector 
        latent = self.reparametrize(mu, logvar) # BS X D 


        # Generate padding mask for decoder 
        dec_seq_len = seq_len + 1 # BS, + 1 since we need additional timestep for latent vector
        tmp_mask = torch.arange(self.window+1).expand(motion.shape[0], \
        self.window+1) < dec_seq_len[:, None].repeat(1, self.window+1) # BS X (T+1)
        dec_padding_mask = tmp_mask[:, None, :].to(motion.device) # BS X 1 X (T+1)
        
        recon = self.decoder(latent, dec_padding_mask) # BS X T X D 

        loss_kl = self.kl_loss(logvar, mu) 
        loss_recon = F.mse_loss(recon, motion, reduction='none') * dec_padding_mask[:, 0, 1:][:, :, None] # BS X T X D 

        loss_recon = reduce(loss_recon, 'b ... -> b (...)', 'mean') # BS X (T*D)
        loss_recon = loss_recon.mean() 

        loss = loss_recon + self.kl_loss_weight * loss_kl 

        return loss, loss_recon, loss_kl 

    def reconstruct(self, motion, seq_len):
        motion = motion.reshape(motion.shape[0], motion.shape[1], -1) # BS X T X (J*2)

        # Generate padding mask for encoder 
        enc_seq_len = seq_len + 2 # BS, + 1 since we need additional timestep for noise level 
        tmp_mask = torch.arange(self.window+2).expand(motion.shape[0], \
        self.window+2) < enc_seq_len[:, None].repeat(1, self.window+2) # BS X (T+2) 
        enc_padding_mask = tmp_mask[:, None, :].to(motion.device) # BS X 1 X (T+2)

        mu, logvar = self.encoder(motion, enc_padding_mask) # BS X D, BS X D 
        
        latent = mu
        
        # Generate padding mask for decoder 
        dec_seq_len = seq_len + 1 # BS, + 1 since we need additional timestep for latent vector
        tmp_mask = torch.arange(self.window+1).expand(motion.shape[0], \
        self.window+1) < dec_seq_len[:, None].repeat(1, self.window+1) # BS X (T+1)
        dec_padding_mask = tmp_mask[:, None, :].to(motion.device) # BS X 1 X (T+1)
        
        recon = self.decoder(latent, dec_padding_mask) # BS X T X D 

        return recon 

    def get_latent(self, motion, seq_len):
        # Generate padding mask for encoder 
        enc_seq_len = seq_len + 2 # BS, + 1 since we need additional timestep for noise level 
        tmp_mask = torch.arange(self.window+1).expand(motion.shape[0], \
        self.window+2) < actual_enc_seq_len[:, None].repeat(1, self.window+2) # BS X (T+2) 
        enc_padding_mask = tmp_mask[:, None, :].to(motion.device) # BS X 1 X (T+2)

        mu, logvar = self.encoder(motion, end_padding_mask) # BS X D, BS X D 
        
        latent = mu

        return latent 

    def sample_motion(self, latent, seq_len):
        # Generate padding mask for decoder 
        dec_seq_len = seq_len + 1 # BS, + 1 since we need additional timestep for latent vector
        tmp_mask = torch.arange(self.window+1).expand(latent.shape[0], \
        self.window+1) < dec_seq_len[:, None].repeat(1, self.window+1) # BS X (T+1)
        dec_padding_mask = tmp_mask[:, None, :].to(latent.device) # BS X 1 X (T+1)
        
        recon = self.decoder(latent, dec_padding_mask) # BS X T X D 

        return recon 

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


class TransformerEncoder(nn.Module):
    def __init__(self,):
        super().__init__()

        num_joints = 18 
        self.d_feats = num_joints * 2  
        self.d_model = 512
        self.n_head = 4
        self.n_dec_layers = 2
        self.d_k = 256
        self.d_v = 256 
        self.max_timesteps = 120  

        self.mu_query = nn.Parameter(torch.randn([self.d_model]))
        self.sigma_query = nn.Parameter(torch.randn([self.d_model]))

        # Input: BS X D X T 
        # Output: BS X T X D'
        self.motion_transformer = Decoder(d_feats=self.d_feats, d_model=self.d_model, \
            n_layers=self.n_dec_layers, n_head=self.n_head, d_k=self.d_k, d_v=self.d_v, \
            max_timesteps=self.max_timesteps, use_full_attention=True)  

        self.linear_out = nn.Linear(self.d_model, self.d_model)

        # For noise level t embedding
        dim = 64
        learned_sinusoidal_dim = 16
        time_dim = dim * 4

        learned_sinusoidal_cond = False
        self.learned_sinusoidal_cond = learned_sinusoidal_cond

        sinu_pos_emb = SinusoidalPosEmb(dim)
        fourier_dim = dim

    def forward(self, src, padding_mask=None):
        # src: BS X T X D
        # padding_mask: BS X 1 X (T+2) 

        bs = src.shape[0]
        num_steps = src.shape[1] + 2 # add 2 for mu and sigma 

        if padding_mask is None:
            padding_mask = torch.ones(bs, 1, num_steps).to(src.device).bool() # BS X 1 X timesteps

        # Get position vec for position-wise embedding
        pos_vec = torch.arange(num_steps)+1 # timesteps
        pos_vec = pos_vec[None, None, :].to(src.device).repeat(bs, 1, 1) # BS X 1 X timesteps

        data_input = src.transpose(1, 2) # BS X D X T 
       
        latent_embedding = torch.cat((self.mu_query[None], self.sigma_query[None]), dim=0)[None].repeat(bs, 1, 1) # BS X 2 X D 
        feat_pred, _ = self.motion_transformer(data_input, padding_mask, pos_vec, \
                obj_embedding=latent_embedding) 
        
        output = self.linear_out(feat_pred[:, :2]) # BS X 2 X D
        pred_mu = output[:, 0] # BS X D 
        pred_sigma = output[:, 1] # BS X D 

        return pred_mu, pred_sigma # predicted noise, the same size as the input 

class TransformerDecoder(nn.Module):
    def __init__(self,):
        super().__init__()
        
        num_joints = 18 
        self.d_feats = num_joints * 2  
        self.d_model = 512
        self.n_head = 4
        self.n_dec_layers = 2
        self.d_k = 256
        self.d_v = 256 
        self.max_timesteps = 120  

        # Input: BS X D X T 
        # Output: BS X T X D'
        self.motion_transformer = Decoder(d_feats=self.d_feats, d_model=self.d_model, \
            n_layers=self.n_dec_layers, n_head=self.n_head, d_k=self.d_k, d_v=self.d_v, \
            max_timesteps=self.max_timesteps, use_full_attention=True)  

        self.linear_out = nn.Linear(self.d_model, self.d_feats)

        # For noise level t embedding
        dim = 64
        learned_sinusoidal_dim = 16
        time_dim = dim * 4

        learned_sinusoidal_cond = False
        self.learned_sinusoidal_cond = learned_sinusoidal_cond

        sinu_pos_emb = SinusoidalPosEmb(dim)
        fourier_dim = dim

    def forward(self, latent, padding_mask=None):
        # latent: BS X D 
        # src: BS X T X D

        src = torch.zeros(latent.shape[0], self.max_timesteps, self.d_feats).to(latent.device) # BS X T X D 

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
                obj_embedding=latent[:, None, :])
        
        output = self.linear_out(feat_pred[:, 1:]) # BS X T X D

        return output # reconstructed motion, the same size as the input 
