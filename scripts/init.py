import numpy as np
import scipy
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys
import shutil
import anndata as ad
from sklearn.model_selection import train_test_split
from sklearn.decomposition import TruncatedSVD

CAE_DIR = '/home/gcgreen2/neurips_comp/cae'                                      
sys.path.append(CAE_DIR)
from scripts import models, utils

def init(config):
    par = config.par
    files = config.files

    # Init output dir
    os.makedirs(par['out_dir'], exist_ok=True)
    shutil.copyfile(files['config'], files['config_cp'])
    
    # Init logfile
    LOG = files['log']
    if os.path.exists(LOG): os.remove(LOG)
    utils.logheader(LOG, par)

    # Load h5 files
    adata_mod1 = ad.read_h5ad(par['data_mod1'])
    adata_mod2 = ad.read_h5ad(par['data_mod2'])
    utils.log(LOG, 'loaded datasets')

    # Perform PCA
    if not os.path.exists(files['mod1_pca']):
        utils.log(LOG, 'performing PCA on mod1 data')
        X = TruncatedSVD(par['n_pcs']).fit_transform(adata_mod1.X)
        np.save(files['mod1_pca'], X)
        utils.log(LOG, 'mod1 PCA data saved')
    else: 
        X = np.load(files['mod1_pca'])
        utils.log(LOG, 'mod1 PCA already exists. skipping computation')

    if not os.path.exists(files['mod2_pca']):
        utils.log(LOG, 'performing PCA on mod2 data')
        Y = TruncatedSVD(par['n_pcs']).fit_transform(adata_mod2.X)
        np.save(files['mod2_pca'], X)
        utils.log(LOG, 'mod2 PCA data saved')
    else: 
        Y = np.load(files['mod2_pca'])
        utils.log(LOG, 'mod2 PCA already exists. skipping computation')


    # Split test and train data
    assert len(X) == len(Y)
    idx_train, idx_test = train_test_split(range(len(X)), train_size=par['pct_train'])
    np.savetxt(files['idx_train'], idx_train, fmt='%d')
    np.savetxt(files['idx_test'], idx_test, fmt='%d')
    utils.log(LOG, 'split data into test and train and stored indices')

    # Init model, etc
    model = eval(utils.model_str(par))
    torch.save(model.state_dict(), files['model'])



