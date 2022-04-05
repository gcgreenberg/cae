import numpy as np
import scipy
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys

sys.path.append('/home/gcgreen2/neurips_comp/cae')
import config
par = config.par

# TORCH MODEL

# We can directly use these models.
class Mish(nn.Module):
    def __init__(self): 
        super().__init__()

    def forward(self,x):
        return x*torch.tanh(F.softplus(x))
    
class Layer(nn.Module):
    def __init__(self, dim1, dim2):
        super(Layer, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(dim1, dim2),  
            nn.BatchNorm1d(dim2),
            Mish())
        
    def forward(self, x):
        return self.layer(x)

# For no conditional VAE
class Encoder(nn.Module):
    def __init__(self, in_dim, h1_dim, h2_dim, z_dim):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
#             nn.Dropout(0.1),
            Layer(in_dim+4, h1_dim),
#             nn.Dropout(0.1),
            Layer(h1_dim, h2_dim))
        self.mu = nn.Linear(h2_dim, z_dim)
        self.logvar = nn.Linear(h2_dim, z_dim)

    def forward(self, x):
        x = self.encoder(x)
        return self.mu(x), self.logvar(x)

# For no conditional VAE
class Decoder(nn.Module):
    def __init__(self, z_dim, h2_dim, h1_dim, out_dim):
        super(Decoder, self).__init__()
        self.decoder = nn.Sequential(
            Layer(z_dim+4, h2_dim),
#             nn.Dropout(0.1),
            Layer(h2_dim, h1_dim),
            nn.Linear(h1_dim,out_dim))

    def forward(self, x):
        return self.decoder(x)

# For conditional VAE      
class Encoder_old(nn.Module):
    def __init__(self, in_dim, h1_dim, h2_dim, z_dim):
        super(Encoder_old, self).__init__()
        self.encoder = nn.Sequential(
#             nn.Dropout(0.1),
            Layer(in_dim, h1_dim),
#             nn.Dropout(0.1),
            Layer(h1_dim, h2_dim))
        self.mu = nn.Linear(h2_dim, z_dim)
        self.logvar = nn.Linear(h2_dim, z_dim)

    def forward(self, x):
        x = self.encoder(x)
        return self.mu(x), self.logvar(x)

# For conditional VAE 
class Decoder_old(nn.Module):
    def __init__(self, z_dim, h2_dim, h1_dim, out_dim):
        super(Decoder_old, self).__init__()
        self.decoder = nn.Sequential(
            Layer(z_dim, h2_dim),
#             nn.Dropout(0.1),
            Layer(h2_dim, h1_dim),
            nn.Linear(h1_dim,out_dim))

    def forward(self, x):
        return self.decoder(x)

# class Encoder(nn.Module):
#     def __init__(self, in_dim, h1_dim, h2_dim, z_dim):
#         super(Encoder, self).__init__()
#         self.encoder = nn.Sequential(
#             Layer(in_dim, h1_dim),
#             Layer(h1_dim, h2_dim))
#         self.mu = nn.Linear(h2_dim, z_dim)
#         self.logvar = nn.Linear(h2_dim, z_dim)

#     def forward(self, x):
#         x = self.encoder(x)
#         return self.mu(x), self.logvar(x)

# class Decoder(nn.Module):
#     def __init__(self, z_dim, h2_dim, h1_dim, out_dim):
#         super(Decoder, self).__init__()
#         self.decoder = nn.Sequential(
#             Layer(z_dim, h2_dim),
#             Layer(h2_dim, h1_dim),
#             nn.Linear(h1_dim,out_dim))

#     def forward(self, x):
#         return self.decoder(x)

class CAE_old(nn.Module):
    def __init__(self, in_dim, h1_dim, h2_dim, z_dim, **args):
        super(CAE_old, self).__init__()
        self.enc_mod1 = Encoder_old(in_dim, h1_dim, h2_dim, z_dim)
        self.dec_mod1 = Decoder_old(z_dim, h2_dim, h1_dim, in_dim)
        self.enc_mod2 = Encoder_old(in_dim, h1_dim, h2_dim, z_dim)
        self.dec_mod2 = Decoder_old(z_dim, h2_dim, h1_dim, in_dim)
        
    def reparam(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)
    
    # conditional
    def forward(self, mod1, mod2, label):
        mu_mod1, logvar_mod1 = self.enc_mod1(mod1)
        z_mod1 = self.reparam(mu_mod1, logvar_mod1)
        
        mu_mod2, logvar_mod2 = self.enc_mod2(mod2)
        z_mod2 = self.reparam(mu_mod2, logvar_mod2)
        
        mod2_recon = self.dec_mod2(z_mod1)
        mod1_recon = self.dec_mod1(z_mod2)
        return mod1_recon, mod2_recon, mu_mod1, logvar_mod1, mu_mod2, logvar_mod2, z_mod1, z_mod2
    
class CAE(nn.Module):
    def __init__(self, in_dim, h1_dim, h2_dim, z_dim, **args):
        super(CAE, self).__init__()
        self.enc_mod1 = Encoder(in_dim, h1_dim, h2_dim, z_dim)
        self.dec_mod1 = Decoder(z_dim, h2_dim, h1_dim, in_dim)
        self.enc_mod2 = Encoder(in_dim, h1_dim, h2_dim, z_dim)
        self.dec_mod2 = Decoder(z_dim, h2_dim, h1_dim, in_dim)
        
    def reparam(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)
    
    # conditional
    def forward(self, mod1, mod2, label):
        mu_mod1, logvar_mod1 = self.enc_mod1(torch.cat((mod1,label), 1))
        z_mod1 = self.reparam(mu_mod1, logvar_mod1)
        
        mu_mod2, logvar_mod2 = self.enc_mod2(torch.cat((mod2,label), 1))
        z_mod2 = self.reparam(mu_mod2, logvar_mod2)
        
        mod2_recon = self.dec_mod2(torch.cat((z_mod1,label),1))
        mod1_recon = self.dec_mod1(torch.cat((z_mod2,label),1))
        return mod1_recon, mod2_recon, mu_mod1, logvar_mod1, mu_mod2, logvar_mod2, z_mod1, z_mod2
    
class CAE_avg(CAE):
    def __init__(self, in_dim, h1_dim, h2_dim, z_dim, **args):
        super(CAE_avg, self).__init__(in_dim=in_dim, h1_dim=h1_dim, h2_dim=h2_dim, z_dim=z_dim)
    
    def forward(self, mod1, mod2, label):
        mu_mod1, logvar_mod1 = self.enc_mod1(torch.cat((mod1,label), 1))
        z_mod1 = self.reparam(mu_mod1, logvar_mod1)
        
        mu_mod2, logvar_mod2 = self.enc_mod2(torch.cat((mod2,label), 1))
        z_mod2 = self.reparam(mu_mod2, logvar_mod2)
        
        z = 1/2 * (z_mod1+z_mod2)
        mod2_recon = self.dec_mod2(torch.cat((z,label),1))
        mod1_recon = self.dec_mod1(torch.cat((z,label),1))
        return mod1_recon, mod2_recon, mu_mod1, logvar_mod1, mu_mod2, logvar_mod2, z_mod1, z_mod2
    
class CAE_linear_comb(CAE):
    def __init__(self, in_dim, h1_dim, h2_dim, z_dim, **args):
        super(CAE_linear_comb, self).__init__(in_dim=in_dim, h1_dim=h1_dim, h2_dim=h2_dim, z_dim=z_dim)
        self.linear_comb = nn.Linear(2*z_dim, z_dim)
    
    def forward(self, mod1, mod2, label):
        mu_mod1, logvar_mod1 = self.enc_mod1(torch.cat((mod1,label), 1))
        z_mod1 = self.reparam(mu_mod1, logvar_mod1)
        
        mu_mod2, logvar_mod2 = self.enc_mod2(torch.cat((mod2,label), 1))
        z_mod2 = self.reparam(mu_mod2, logvar_mod2)
        
        z_concat = torch.cat((z_mod1, z_mod2), 1) # concat the features dimension (-> # cells x 2*z_dim)
        z = self.linear_comb(z_concat)
        
        z = torch.cat((z, label), 1)
        
        mod2_recon = self.dec_mod2(z)
        mod1_recon = self.dec_mod1(z)
        return mod1_recon, mod2_recon, mu_mod1, logvar_mod1, mu_mod2, logvar_mod2, z_mod1, z_mod2
        
class CAE_avg_old(CAE_old):
    def __init__(self, in_dim, h1_dim, h2_dim, z_dim, **args):
        super(CAE_avg_old, self).__init__(in_dim=in_dim, h1_dim=h1_dim, h2_dim=h2_dim, z_dim=z_dim)
    
    def forward(self, mod1, mod2, label):
        mu_mod1, logvar_mod1 = self.enc_mod1(mod1)
        z_mod1 = self.reparam(mu_mod1, logvar_mod1)
        
        mu_mod2, logvar_mod2 = self.enc_mod2(mod2)
        z_mod2 = self.reparam(mu_mod2, logvar_mod2)
        
        z = 1/2 * (z_mod1+z_mod2)
        mod2_recon = self.dec_mod2(z)
        mod1_recon = self.dec_mod1(z)
        return mod1_recon, mod2_recon, mu_mod1, logvar_mod1, mu_mod2, logvar_mod2, z_mod1, z_mod2
        
class CAE_linear_comb_old(CAE_old):
    def __init__(self, in_dim, h1_dim, h2_dim, z_dim, **args):
        super(CAE_linear_comb_old, self).__init__(in_dim=in_dim, h1_dim=h1_dim, h2_dim=h2_dim, z_dim=z_dim)
        self.linear_comb = nn.Linear(2*z_dim, z_dim)
    
    def forward(self, mod1, mod2, label):
        mu_mod1, logvar_mod1 = self.enc_mod1(mod1)
        z_mod1 = self.reparam(mu_mod1, logvar_mod1)
        
        mu_mod2, logvar_mod2 = self.enc_mod2(mod2)
        z_mod2 = self.reparam(mu_mod2, logvar_mod2)
        
        z_concat = torch.cat((z_mod1, z_mod2), 1) # concat the features dimension (-> # cells x 2*z_dim)
        z = self.linear_comb(z_concat)
        mod2_recon = self.dec_mod2(z)
        mod1_recon = self.dec_mod1(z)
        return mod1_recon, mod2_recon, mu_mod1, logvar_mod1, mu_mod2, logvar_mod2, z_mod1, z_mod2