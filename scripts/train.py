import numpy as np
import scipy
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys
import anndata as ad
# from sklearn.model_selection import train_test_split
from sklearn.decomposition import TruncatedSVD

sys.path.append('/home/gcgreen2/neurips_comp/cae')
from scripts import models, utils

device = torch.device('cpu')

def train(config):
    par = config.par
    files = config.files
    LOG = files['log']
    utils.log(LOG, 'beginning training procedure', newline=True)

    # Load PCA data 
    X = np.load(files['mod1_pca'])
    Y = np.load(files['mod2_pca'])
    X,Y = [torch.FloatTensor(data).to(device) for data in [X, Y]]
    utils.log(LOG, 'PCA data loaded')

    # Split test and train data
#     idx_train = np.loadtxt(files['idx_train'], dtype=int)
#     idx_test = np.loadtxt(files['idx_test'], dtype=int)
#     X_train, Y_train = [torch.FloatTensor(data[idx_train]) for data in [X, Y]]
#     X_test, Y_test = [torch.FloatTensor(data[idx_test]) for data in [X, Y]]
#     utils.log(LOG, 'data split into test and train')

    # Init model, etc
    model = eval(utils.model_str(par))
    model.load_state_dict(torch.load(files['model']))
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=par['lr'], weight_decay=1e-4)
    smth_loss = nn.SmoothL1Loss()
    mse_loss = nn.MSELoss()
    kl_loss = lambda mu, logvar: torch.mean(-0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp(), dim = 1), dim = 0)
    #1/2 * torch.mean(mu**2 + logvar.exp() - 1 - logvar)


    def calc_errors(X, Y):
        X_recon, Y_recon, Mu_mod1, Logvar_mod1, Mu_mod2, Logvar_mod2, Z_mod1, Z_mod2 = model(X, Y)
        error_X = par['lambda_mod1'] * smth_loss(X_recon, X)
        error_Y = par['lambda_mod2'] * smth_loss(Y_recon, Y)
        error_Z = par['lambda_latent'] * mse_loss(Z_mod1, Z_mod2) # if par['model']=='CAE' else torch.zeros_like(error_X)
        KL = par['lambda_kl'] * (kl_loss(Mu_mod1, Logvar_mod1) + kl_loss(Mu_mod2, Logvar_mod2))
        reg_Z = par['lambda_reg'] * (torch.mean(torch.square(Z_mod1)) + torch.mean(torch.square(Z_mod2)))
        return error_X, error_Y, error_Z, KL, reg_Z

    # Training
    model.train()
    for epoch in range(1,par['n_epochs']+1):
        if utils.print_epoch(epoch):
            utils.log(LOG, 'epoch: {}'.format(epoch), newline=True)
            utils.log(LOG, 'losses:  ' + utils.errors_str(*calc_errors(X, Y)))
        torch.save(model.state_dict(), files['model'])

        batch_idx = np.arange(len(X)); np.random.shuffle(batch_idx)
        for iter in range(0,len(batch_idx),par['batch']):
            X_batch = X[batch_idx[iter:iter+par['batch']]]
            Y_batch = Y[batch_idx[iter:iter+par['batch']]]

            optimizer.zero_grad()
            error_X, error_Y, error_Z, KL, reg_Z = calc_errors(X_batch, Y_batch)
            error = error_X + error_Y + error_Z + KL + reg_Z
            error.backward()
            optimizer.step()

    # Save model
    torch.save(model.state_dict(), files['model'])


