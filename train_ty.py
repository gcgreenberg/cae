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

sys.path.append('/gpfs/ysm/home/tl688/scrnahpc/cae') #please modify this path
import models 
import utils

device = torch.device('cuda')

def train(config):
    par = config.par
    files = config.files
    LOG = files['log']
    utils.log(LOG, 'beginning training procedure', newline=True)

    # load cell type lables 
    adata = ad.read_h5ad('multiome_gex_processed_training.h5ad')
    celltype = list(adata.obs['cell_type'])
    celltype_unique = list(set(celltype))
    dict_celltype = {} 
    for i in range(len(celltype_unique)):
      dict_celltype[celltype_unique[i]] = i
    
    embd = nn.Embedding(len(celltype_unique), 4)
    # Load PCA data 
    X = np.load(files['mod1_pca'])
    Y = np.load(files['mod2_pca'])
    
    X,Y = [torch.FloatTensor(data).to(device) for data in [X, Y]]
    
    celltype_info = torch.LongTensor([dict_celltype[celltype[k]] for k in np.arange(len(X))])
    celltype_total1 = torch.FloatTensor(embd(celltype_info).detach_().numpy()).to(device).detach_()
    
    
    #X0 = torch.cat((X, celltype_total1), 1)
    #Y0 = torch.cat((Y, celltype_total1), 1)
    
    utils.log(LOG, 'PCA data loaded')

    # Split test and train data
#     idx_train = np.loadtxt(files['idx_train'], dtype=int)
#     idx_test = np.loadtxt(files['idx_test'], dtype=int)
#     X_train, Y_train = [torch.FloatTensor(data[idx_train]) for data in [X, Y]]
#     X_test, Y_test = [torch.FloatTensor(data[idx_test]) for data in [X, Y]]
#     utils.log(LOG, 'data split into test and train')

    # Init model, etc
    model = eval(utils.model_str(par))
    #model.load_state_dict(torch.load(files['model']))
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=par['lr'], weight_decay=1e-4)
    smth_loss = nn.SmoothL1Loss().to(device)
    mse_loss = nn.MSELoss().to(device)
    kl_loss = lambda mu, logvar: torch.mean(-0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp(), dim = 1), dim = 0)


    def calc_errors(t1, t2, label):
        X_recon, Y_recon, Mu_mod1, Logvar_mod1, Mu_mod2, Logvar_mod2, Z_mod1, Z_mod2 = model(t1, t2, label)
        error_X = par['lambda_mod1'] * smth_loss(X_recon, t1)
        error_Y = par['lambda_mod2'] * smth_loss(Y_recon, t2)
        error_Z = par['lambda_latent'] * mse_loss(Z_mod1, Z_mod2) # if par['model']=='CAE' else torch.zeros_like(error_X)
        #error_Z = par['lambda_latent']*torch.sum(1-F.cosine_similarity(Z_mod1, Z_mod2, dim=1)).mean().cuda() # if par['model']=='CAE' else torch.zeros_like(error_X)
        KL = par['lambda_kl'] * (kl_loss(Mu_mod1, Logvar_mod1) + kl_loss(Mu_mod2, Logvar_mod2))
        reg_Z = par['lambda_reg'] * (torch.mean(torch.square(Z_mod1)) + torch.mean(torch.square(Z_mod2)))
        return error_X, error_Y, error_Z, KL, reg_Z

    # Training
    model.train()
    for epoch in range(1,par['n_epochs']+1):
        #if utils.print_epoch(epoch):
            #utils.log(LOG, 'epoch: {}'.format(epoch), newline=True)
            #utils.log(LOG, 'losses:  ' + utils.errors_str(*calc_errors(X, Y)))
        torch.save(model.state_dict(), files['model'])

        batch_idx = np.arange(len(X)); 
        #np.random.shuffle(batch_idx)
        for iter_step in range(0,len(batch_idx),par['batch']):
            X_batch = X[batch_idx[iter_step:iter_step+par['batch']]]
            Y_batch = Y[batch_idx[iter_step:iter_step+par['batch']]]
            celltype_batch = celltype_total1[batch_idx[iter_step:iter_step+par['batch']]]

            optimizer.zero_grad()
            error_X, error_Y, error_Z, KL, reg_Z = calc_errors(X_batch, Y_batch, celltype_batch)
            print(error_X, error_Y, error_Z, KL)
            error0 = error_X + error_Y + error_Z + KL + reg_Z
            error0.backward()
            optimizer.step()

    # Save model
    model.eval()
    torch.save(model.state_dict(), files['model'])
    X_recon, Y_recon, Mu_mod1, Logvar_mod1, Mu_mod2, Logvar_mod2, Z_mod1, Z_mod2 = model(X, Y, celltype_total1)
    final_set = np.vstack([Z_mod1.detach().cpu().numpy(),Z_mod2.detach().cpu().numpy()])
    np.save('output_dim.npy', final_set)
    
    


