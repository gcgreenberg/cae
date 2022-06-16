import loompy
from numba import jit
from collections import Counter

import imap  #used for feature detected
import numpy as np
import pandas as pd
import scanpy as sc
import os
from skmisc.loess import loess
import sklearn.preprocessing as preprocessing

import torch.autograd
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_s 

from sklearn import preprocessing
from torch.utils.data import DataLoader
import torch.utils.data as data_utils

from metrics import calculate_metrics

import scib 
import random

from annoy import AnnoyIndex

class Mish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,x):
        return x*torch.tanh(F.softplus(x))
        
class VectorQuantizer(nn.Module):
    """
    VQ-VAE layer: Input any tensor to be quantized. 
    Args:
        embedding_dim (int): the dimensionality of the tensors in the
          quantized space. Inputs to the modules must be in this format as well.
        num_embeddings (int): the number of vectors in the quantized space.
        commitment_cost (float): scalar which controls the weighting of the loss terms (see
          equation 4 in the paper - this variable is Beta).
    """
    def __init__(self, embedding_dim, num_embeddings, commitment_cost):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost
        
        # initialize embeddings
        self.embeddings = nn.Embedding(self.num_embeddings, self.embedding_dim)
        
    def forward(self, x):
        flat_x = x.reshape(-1, self.embedding_dim)
        
        # Use index to find embeddings in the latent space
        encoding_indices = self.get_code_indices(flat_x)
        quantized = self.quantize(encoding_indices)
        quantized = quantized.view_as(x) 
        
        # embedding loss: move the embeddings towards the encoder's output
        q_latent_loss = F.mse_loss(quantized, x.detach())
        # commitment loss
        e_latent_loss = F.mse_loss(x, quantized.detach())
        loss = q_latent_loss + self.commitment_cost * e_latent_loss

        # Straight Through Estimator
        quantized = x + (quantized - x).detach()
    
        return quantized, loss
    
    def get_code_indices(self, flat_x):
        # compute L2 distance
        distances = (
            torch.sum(flat_x ** 2, dim=1, keepdim=True) +
            torch.sum(self.embeddings.weight ** 2, dim=1) -
            2. * torch.matmul(flat_x, self.embeddings.weight.t())
        ) # [N, M]
        encoding_indices = torch.argmin(distances, dim=1) # [N,]
        return encoding_indices
    
    def quantize(self, encoding_indices):
        """Returns embedding tensor for a batch of indices."""
        return self.embeddings(encoding_indices)      
        
    
class Encoder(nn.Module):
    """Encoder of VQ-VAE"""
    
    def __init__(self, in_dim=2500, latent_dim=16):
        super().__init__()
        self.in_dim = in_dim
        self.latent_dim = latent_dim
        
        self.enc = nn.Sequential(
                    nn.Linear(self.in_dim, 1024),
                    nn.BatchNorm1d(1024),
                    Mish(),
                    nn.Linear(1024, 512),
                    nn.BatchNorm1d(512),
                    Mish(),
                    nn.Linear(512, 256),
                    nn.BatchNorm1d(256),
                    Mish(),
                    nn.Linear(256, latent_dim),
        )
        
    def forward(self, x):
        return self.enc(x)
    

class Decoder(nn.Module):
    """Decoder of VQ-VAE"""
    
    def __init__(self, out_dim=2500, latent_dim=16):
        super().__init__()
        self.out_dim = out_dim
        self.latent_dim = latent_dim
        
        self.dec= nn.Sequential(
                    nn.Linear(latent_dim, 256),
                    nn.BatchNorm1d(256),
                    Mish(),
                    nn.Linear(256, 512),
                    nn.BatchNorm1d(512),
                    Mish(),
                    nn.Linear(512, 1024),
                    nn.BatchNorm1d(1024),
                    Mish(),
                    nn.Linear(1024, self.out_dim),
        )
        
    def forward(self, x):
        return self.dec(x)
        
        
        
class VQVAE(nn.Module):
    """VQ-VAE"""
    
    def __init__(self, in_dim, embedding_dim, num_embeddings, data_variance1, data_variance2,
                 commitment_cost=0.25, lambda_z = 10):
        super().__init__()
        self.in_dim = in_dim
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.data_variance1 = data_variance1
        self.data_variance2 = data_variance2
        
        self.encoder_gexatac = Encoder(in_dim, embedding_dim)
        self.vq_layer_gexatac = VectorQuantizer(embedding_dim, num_embeddings, commitment_cost)
        self.decoder_gexatac = Decoder(in_dim, embedding_dim)
        
        self.encoder_atacgex = Encoder(in_dim, embedding_dim)
        self.vq_layer_atacgex = VectorQuantizer(embedding_dim, num_embeddings, commitment_cost)
        self.decoder_atacgex = Decoder(in_dim, embedding_dim)
        
        self.lambda_z = lambda_z
        
    def forward(self, x, y):
        z = self.encoder_gexatac(x)
        if not self.training:
            e = self.vq_layer(z)
            x_recon = self.decoder(e)
            return e, x_recon
        
        e, e_q_loss = self.vq_layer_gexatac(z)
        x_recon = self.decoder_gexatac(e)
    
        recon_loss = F.mse_loss(x_recon, y) / self.data_variance1 #atac
        
        z1 = self.encoder_atacgex(y)
        if not self.training:
            e = self.vq_layer(z)
            x_recon = self.decoder(e)
            return e, x_recon
        
        e1, e_q_loss1 = self.vq_layer_atacgex(z1)
        y_recon = self.decoder_gexatac(e1)
    
        recon_loss1 = F.mse_loss(y_recon, x) / self.data_variance2 #gex       
        
        print(F.mse_loss(z,z1))
        print(e_q_loss + recon_loss)
        return e_q_loss + recon_loss +e_q_loss1+ recon_loss1 + self.lambda_z*F.mse_loss(z,z1)