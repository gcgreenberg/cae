import numpy as np
import scipy
import torch
import torch.nn as nn
import torch.nn.functional as F
import os


from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import euclidean_distances
from scipy.special import logsumexp
from scipy.sparse import csr_matrix

import sys
sys.path.append('/home/gcgreen2/neurips_comp/cae')
from scripts import models, utils


par = {
    'data_mod1': 'sample_data/openproblems_bmmc_multiome_starter/openproblems_bmmc_multiome_starter.train_mod1.h5ad',
    'data_mod2': 'sample_data/openproblems_bmmc_multiome_starter/openproblems_bmmc_multiome_starter.train_mod2.h5ad',
    'model': 'CAE',
    'pct_train': 0.8,
    'n_pcs': 500,
    'n_epochs': 200,
    'batch' = 128,
    'lr' = 0.001,
    'lambda_mod2' = 1,
    'lambda_latent' = 1000
}


# Init
par['out_dir'] = utils.init_out_dir(par) 
LOG = os.path.join(par['out_dir'], 'log')
utils.logheader(LOG, par)


# Load h5 files
adata_mod1 = ad.read_h5ad(par['data_mod1'])
adata_mod2 = ad.read_h5ad(par['data_mod2'])
utils.log(LOG, 'loaded datasets')


# Perform PCA
utils.log(LOG, 'performing PCA')
X = TruncatedSVD(par['n_pcs']).fit_transform(adata_mod1.X)
np.save(os.path.join(DATA_DIR, 'mod1_pca.npy'.format(par['n_pcs'])), X)
utils.log(LOG, 'performed PCA on mod1 data and saved')

Y = TruncatedSVD(par['n_pcs']).fit_transform(adata_mod2.X)
np.save(os.path.join(DATA_DIR, 'mod2_pca.npy'.format(par['n_pcs'])), X)
utils.log(LOG, 'performed PCA on mod2 data and saved')


# Split test and train data
assert len(X) == len(Y)
X_train, X_test, Y_train, Y_test, idx_train, idx_test = \
    train_test_split(X, Y, range(len(X)), train_size=par['pct_train'])
X_train, Y_train, X_test, Y_test = [torch.FloatTensor(x) for x in [X_train, Y_train, X_test, Y_test]]
np.savetxt(os.path.join(par['out_dir'],'idx_train.txt'), idx_train, fmt='%d')
np.savetxt(os.path.join(par['out_dir'],'idx_test.txt'), idx_test, fmt='%d')
utils.log(LOG, 'split into test and train data. stored indices')


# Init model, etc
model = eval(par['model'] + '(in_dim={})'.format(par['n_pcs']))
optimizer = torch.optim.Adam(model.parameters(), lr=par['lr'])
model.train()
smth_loss = nn.SmoothL1Loss()
mse_loss = nn.MSELoss()


# Training
for epoch in range(1,par['n_epochs']+1):
    if print_epoch(epoch): 
        utils.log(LOG, 'epoch: {}'.format(epoch))
        X_recon, Y_recon, Z_mod1, Z_mod2 = model(X_train, Y_train)
        error_X = smth_loss(X_recon, X_train)
        error_Y = lambda_atac * smth_loss(Y_recon, Y_train)
        error_Z = lambda_latent * mse_loss(model.z_gex, model.z_atac)
        utils.log(utils.errors_str(error_X, error_Y, error_Z)

    epoch_batch_idx = np.arange(len(X_train)); np.random.shuffle(train_idx)
    for iter in range(0,len(X_train),batch):
        X_batch = X_train[epoch_batch_idx[iter:iter+batch]]
        Y_batch = Y_train[epoch_batch_idx[iter:iter+batch]]

        optimizer.zero_grad()
        X_recon, Y_recon = model(X_batch, Y_batch)
        z_X = model.z_gex
        z_Y = model.z_atac

        error_X = smth_loss(X_recon, X_train_batch)
        error_Y = lambda_atac * smth_loss(Y_recon, Y_train_batch)
        error_Z = lambda_latent * euc_loss(model.z_gex, model.z_atac)
        error = error_X + error_Y + error_Z

        error.backward()
        optimizer.step()

# testing

def get_sigma(dists, k=10):
    sigma = np.sort(dists, axis=1)
    sigma = sigma[:,:k].flatten()
    sigma = np.sqrt(np.mean(np.square(sigma)))
    return sigma

def dist2prob(row, k, sigma):
    idx = np.argsort(row)
    row[idx[k:]] = 0
    probs = row[idx[:k]]
    probs = -np.square(probs)/(2*sigma**2)
    probs = 1/np.sqrt(2*np.pi*sigma**2)*np.exp(probs - logsumexp(probs))
    probs /= sum(probs)
    row[idx[:k]] = probs
    return row

def kNN_probs(dists, k): 
    XY_dists = dists.copy()
    X2Y = np.apply_along_axis(dist2prob, 1, XY_dists, k, get_sigma(XY_dists,k))
    return X2Y

model.eval()
Z_gex_test = model.enc_gex(X_test).detach().numpy()
Z_atac_test = model.enc_atac(Y_test).detach().numpy()

k = par['n_neighbors']
dists = euclidean_distances(Z_gex_test, Z_atac_test)
pairing_matrix = kNN_probs(dists, k)
pairing_matrix = csr_matrix(pairing_matrix)

logging.info('write prediction output')
out = ad.AnnData(
    X=pairing_matrix,
    uns={
        "dataset_id": input_train_mod1.uns["dataset_id"],
        "method_id": method_id
    }
)
out.write_h5ad(par['output'], compression="gzip")


#####################################################################

