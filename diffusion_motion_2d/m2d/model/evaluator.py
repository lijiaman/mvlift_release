import torch
from torch import nn

import numpy as np 

# from model.mdm import InputProcess, OutputProcess, PositionalEncoding

# KL_DIV_LOSS_WEIGHT = 0.00001

KL_DIV_LOSS_WEIGHT = 0.0

def kl_div_loss(mu, logvar, **kwargs):
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

mse = torch.nn.MSELoss()


class InputProcess(nn.Module):
    def __init__(self, input_feats, latent_dim):
        super().__init__()
        self.input_feats = input_feats
        self.latent_dim = latent_dim
        self.poseEmbedding = nn.Linear(self.input_feats, self.latent_dim)

    def forward(self, x):
        bs, njoints, nfeats, nframes = x.shape
        x = x.permute((3, 0, 1, 2)).reshape(nframes, bs, njoints * nfeats)

        x = self.poseEmbedding(x)  # [seqlen, bs, d]
        return x


class OutputProcess(nn.Module):
    def __init__(self, input_feats, latent_dim, njoints, nfeats):
        super().__init__()
        self.input_feats = input_feats
        self.latent_dim = latent_dim
        self.njoints = njoints
        self.nfeats = nfeats
        self.poseFinal = nn.Linear(self.latent_dim, self.input_feats)

    def forward(self, output):
        nframes, bs, d = output.shape
        output = self.poseFinal(output)  # [seqlen, bs, 150]
        output = output.reshape(nframes, bs, self.njoints, self.nfeats)
        output = output.permute(1, 2, 3, 0)  # [bs, njoints, nfeats, nframes]
        return output


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer("pe", pe)

    def forward(self, x):
        # not used in the final model
        x = x + self.pe[: x.shape[0], :]
        return self.dropout(x)


class VAE(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
       
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, motion, seq_len):
        mu, sigma = self.encoder(motion, seq_len)
        latent = mu + torch.exp(sigma / 2) * torch.randn_like(mu)
        recon = self.decoder(latent, seq_len)
        return {"mu": mu, "sigma": sigma, "recon_motion": recon}

    def compute_loss(self, motion, seq_len):
        res = self.forward(motion, seq_len)
        mu, logvar, recon_motion = res["mu"], res["sigma"], res["recon_motion"]
        loss_recon = mse(recon_motion, motion)
        loss_kl = kl_div_loss(mu, logvar)
        loss = loss_recon + KL_DIV_LOSS_WEIGHT * loss_kl
        return loss, loss_recon, loss_kl 

    def reconstruct(self, motion, seq_len):
        mu, sigma = self.encoder(motion, seq_len)
        latent = mu
        recon = self.decoder(latent, seq_len)
        return recon 

def get_seq_mask(lengths):
    lengths = lengths.view(-1, 1)  # [bs, 1]
    positions = torch.arange(lengths.max(), device=lengths.device).view(1, -1)  # [1, nframes+1]
    return positions >= lengths  # [nframes+1, bs]


class TransformerEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.njoints = 18
        self.nfeats = 2 

        self.latent_dim = 256
        self.num_layers = 6 
        self.num_heads = 4 
        self.ff_size = 1024 
        self.dropout = 0.1 
        self.activation = "gelu" 

        self.set_arch()

    def set_arch(self):
        self.input_dim = self.njoints * self.nfeats
       
        self.input_process = InputProcess(self.input_dim, self.latent_dim)

        self.sequence_pos_encoder = PositionalEncoding(self.latent_dim, self.dropout)

        seqTransEncoderLayer = nn.TransformerEncoderLayer(d_model=self.latent_dim, nhead=self.num_heads, dim_feedforward=self.ff_size, dropout=self.dropout, activation=self.activation)
        self.seqTransEncoder = nn.TransformerEncoder(seqTransEncoderLayer, num_layers=self.num_layers)

        self.mu_query = nn.Parameter(torch.randn([self.latent_dim]))
        self.sigma_query = nn.Parameter(torch.randn([self.latent_dim]))

    def forward(self, x, seq_len):
        # seq_len: BS 
        bs, njoints, nfeats, nframes = x.shape  # [bs, njoints, nfeats, nframes]
        assert njoints == self.njoints and nfeats == self.nfeats
        x = self.input_process(x)  # [nframes, bs, d]
        x = torch.cat((self.mu_query.expand(x[[0]].shape), self.sigma_query.expand(x[[0]].shape), x), axis=0)
        x = self.sequence_pos_encoder(x)  # [nframes+2, bs, d]

        # create a bigger mask, to allow to attend to mu and sigma
        x = self.seqTransEncoder(x, src_key_padding_mask=get_seq_mask(seq_len + 2))
        mu = x[0]
        logvar = x[1]

        return mu, logvar

class TransformerDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.njoints = 18 
        self.nfeats = 2

        self.latent_dim = 256
        self.num_layers = 6 
        self.num_heads = 4 
        self.ff_size = 1024 
        self.dropout = 0.1 
        self.activation = "gelu" 

        self.set_arch()
        

    def set_arch(self):
        self.input_dim = self.njoints * self.nfeats

        self.output_process = OutputProcess(self.input_dim, self.latent_dim, self.njoints, self.nfeats)

        self.sequence_pos_encoder = PositionalEncoding(self.latent_dim, self.dropout)

        seqTransDecoderLayer = nn.TransformerDecoderLayer(d_model=self.latent_dim, nhead=self.num_heads, dim_feedforward=self.ff_size, \
            dropout=self.dropout, activation=self.activation)
        self.seqTransDecoder = nn.TransformerDecoder(seqTransDecoderLayer, num_layers=self.num_layers)

    def forward(self, latent, seq_len):
        bs, latent_dim = latent.shape  # [bs, d]
        nframes = seq_len.max()
        assert latent_dim == self.latent_dim
        x = torch.zeros([nframes, bs, self.latent_dim]).to(latent.device)  # [nframes, bs, d]
        x = self.sequence_pos_encoder(x)  # [nframes, bs, d]

        x = self.seqTransDecoder(tgt=x, memory=latent.unsqueeze(0), tgt_key_padding_mask=get_seq_mask(seq_len))
        x = self.output_process(x)  # [nframes, bs, nfeats*njoints]

        return x
