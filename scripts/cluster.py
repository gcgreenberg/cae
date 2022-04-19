import numpy as np
from os.path import join
import os
from tqdm import tqdm
from sklearn.manifold import TSNE
import umap
import sys
import importlib
import torch
import anndata as ad
import scanpy as sc


PROJ_DIR = '/home/gcgreen2/neurips_comp'
DATA_DIR = join(PROJ_DIR, 'data', 'multiome')
OUT_DIR = os.getcwd()

sys.path.append(join(PROJ_DIR,'cae'))
from scripts import models, utils, task3_metrics as t3


with open(join(OUT_DIR,'config.py'), 'r') as fh:
    lines = fh.read()
    eval(compile(lines, '<string>', 'exec'))
    

mod1_pca = np.load(files['mod1_pca'])
mod2_pca = np.load(files['mod2_pca'])

adata_mod1.obsm['X_pca'] = mod1_pca
adata_mod2.obsm['X_pca'] = mod2_pca

adata_mod1.obs['mod'] = 'mod1'
adata_mod2.obs['mod'] = 'mod2'

model = eval(utils.model_str(par))
model.load_state_dict(torch.load(files['model']))


X,Y = [torch.FloatTensor(data.obsm['X_pca']) for data in [adata_mod1,adata_mod2]]
_,_,Mu_mod1,_,Mu_mod2,_,_,_ = [x.detach().numpy() for x in model(X,Y)]


# adata_mod1.obsm['Z'] = Z_mod1
# adata_mod2.obsm['Z'] = Z_mod2
adata_mod1.obsm['Z_mu'] = Mu_mod1
adata_mod2.obsm['Z_mu'] = Mu_mod2
adata_mod1.obsm['Z_mu_avg'] = 1/2 * (Mu_mod1+Mu_mod2)
# adata_mod1.obsm['Z_var'] = Logvar_mod1
# adata_mod2.obsm['Z_var'] = Logvar_mod2
# adata_mod1.obsm['Z_2d'] = Z_mod1[:,:2]
# adata_mod2.obsm['Z_2d'] = Z_mod2[:,:2]


clust_path = join(OUT_DIR, 'clustering.npy')
sc.pp.neighbors(adata_mod1, n_pcs=0, use_rep='Z_mu_avg')
sc.tl.louvain(adata_mod1)
np.save(clust_path, adata.obs['louvain'])






