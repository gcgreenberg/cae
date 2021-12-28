import numpy as np
import scipy
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys
import anndata as ad
from sklearn.model_selection import train_test_split
from sklearn.decomposition import TruncatedSVD

sys.path.append('/home/gcgreen2/neurips_comp/cae')
from scripts import models, utils
import config

par = config.par
files = config.files

# Init
OUT_DIR = config.OUT_DIR
utils.init_out_dir(OUT_DIR)
LOG = files['log']
utils.logheader(LOG, par)

# Load h5 files
adata_mod1 = ad.read_h5ad(par['data_mod1'])
adata_mod2 = ad.read_h5ad(par['data_mod2'])
utils.log(LOG, 'loaded datasets')

# Perform PCA
utils.log(LOG, 'performing PCA on mod1 data')
X = TruncatedSVD(par['n_pcs']).fit_transform(adata_mod1.X)
np.save(files['mod1_pca'], X)
utils.log(LOG, 'mod1 PCA data saved')

utils.log(LOG, 'performing PCA on mod2 data')
Y = TruncatedSVD(par['n_pcs']).fit_transform(adata_mod2.X)
np.save(files['mod2_pca'], X)
utils.log(LOG, 'mod2 PCA data saved')

# Split test and train data
assert len(X) == len(Y)
X_train, X_test, Y_train, Y_test, idx_train, idx_test = \
    train_test_split(X, Y, range(len(X)), train_size=par['pct_train'])
np.savetxt(files['idx_train'], idx_train, fmt='%d')
np.savetxt(files['idx_test'], idx_test, fmt='%d')
utils.log(LOG, 'split into test and train data and stored indices')

# Init model, etc
model = eval(utils.model_str(par))
torch.save(model.state_dict(), files['model'])



