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

device = torch.device('cuda')

def train_vae(config):
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

    smth_loss = nn.SmoothL1Loss()
    mse_loss = nn.MSELoss()
    kl_loss = lambda mu, logvar: torch.mean(-0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp(), dim = 1), dim = 0)
     
    
    # Init model, etc
    for data,mod in [(X,'mod1'),(Y,'mod2')]:
        model = eval(utils.model_str(par))
        model_path = os.path.join(par['out_dir'], 'model_'+mod+'.torch')
        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=par['lr'], weight_decay=1e-4)
        
        def calc_errors(X_true):
            X_recon, Mu, Logvar, Z = model(X_true)
            error_X = par['lambda_'+mod] * smth_loss(X_recon, X_true)
            KL = par['lambda_kl_'+mod] * kl_loss(Mu, Logvar)
            reg_Z = par['lambda_reg'] * torch.mean(torch.pow(Z,2))
            return error_X, KL, reg_Z

        # Training mod1
        model.train()
        for epoch in range(1,par['n_epochs']+1):
            if utils.print_epoch(epoch):
                utils.log(LOG, 'epoch: {}'.format(epoch), newline=True)
                utils.log(LOG, 'losses:  ' + utils.errors_str_vae(*calc_errors(data)))
            torch.save(model.state_dict(), model_path)

            batch_idx = np.arange(len(data)); np.random.shuffle(batch_idx)
            for iter in range(0,len(batch_idx),par['batch']):
                data_batch = data[batch_idx[iter:iter+par['batch']]]

                optimizer.zero_grad()
                error_X, KL, reg_Z = calc_errors(data_batch)
                error = error_X + KL + reg_Z
                error.backward()
                optimizer.step()

        # Save model
        model.eval()
        torch.save(model.state_dict(), model_path)
        _, Mu, _, _ = model(data)
        np.save(files[mod+'_z'], Mu.cpu().detach().numpy())


