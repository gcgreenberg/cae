'''
No dropout
'''
from os.path import join, exists

PROJ_DIR = '/home/gcgreen2/neurips_comp'
DATA_DIR = join(PROJ_DIR, 'data', 'multiome')
OUT_DIR = join(PROJ_DIR, 'out', '4-18-vae2')

par = {
    'out_dir': OUT_DIR,
    'data_mod1': join(DATA_DIR,'multiome_gex_processed_training.h5ad'),
    'data_mod2': join(DATA_DIR,'multiome_atac_processed_training.h5ad'),
    'model': 'VAE',
    'n_pcs': 2500,
    'h1_dim': 256,
    'h2_dim': 64,
    'z_dim': 16,
#     'pct_train': 0.8,
    'n_epochs': 400,
    'batch' : 128,
    'lr' : 0.0001,
    'lambda_mod1' : 1,
    'lambda_mod2' : 1,
    'lambda_kl_mod1': 0.0001,
    'lambda_kl_mod2': 1,
    'lambda_reg' : 0
}
par['in_dim'] = par['n_pcs']

files = {
    'log': join(OUT_DIR, 'log.txt'),
    'mod1_pca': join(DATA_DIR, 'pca_2500.npy'),
    'mod2_pca': join(DATA_DIR, 'lsi_2500.npy'),
    'model': join(OUT_DIR, 'model.torch'),
#     'idx_train': join(OUT_DIR,'idx_train.txt'),
#     'idx_test': join(OUT_DIR,'idx_test.txt'),
    'config': join(PROJ_DIR, 'cae/config_vae.py'),
    'config_cp': join(OUT_DIR, 'config_vae.py'),
    'eval_nb': join(PROJ_DIR, 'cae/eval-single.ipynb'),
    'eval_cp': join(OUT_DIR, 'eval-single.ipynb'),
    'mod1_z': join(OUT_DIR, 'mod1_z.npy'),
    'mod2_z': join(OUT_DIR, 'mod2_z.npy')
}






