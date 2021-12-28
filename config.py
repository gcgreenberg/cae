from os.path import join

PROJ_DIR = '/home/gcgreen2/neurips_comp'
DATA_DIR = join(PROJ_DIR, 'data')
OUT_DIR = join(PROJ_DIR, 'out', '12-27b')

par = {
    'out_dir': OUT_DIR,
    'data_mod1': join(DATA_DIR,'multiome/multiome_gex_processed_training.h5ad'),
    'data_mod2': join(DATA_DIR,'multiome/multiome_atac_processed_training.h5ad'),
    'model': 'CAE',
    'h1_dim': 128,
    'h2_dim': 32,
    'z_dim': 8,
    'pct_train': 0.8,
    'n_pcs': 2000,
    'n_epochs': 200,
    'batch' : 128,
    'lr' : 0.001,
    'lambda_mod2' : 1,
    'lambda_latent' : 1000,
    'lambda_reg' : 0.1
}
par['in_dim'] = par['n_pcs']

files = {
    'log': join(OUT_DIR, 'log'),
    'mod1_pca': join(OUT_DIR, 'mod1_pca.npy'),
    'mod2_pca': join(OUT_DIR, 'mod2_pca.npy'),
    'model': join(OUT_DIR, 'model.torch'),
    'idx_train': join(OUT_DIR,'idx_train.txt'),
    'idx_test': join(OUT_DIR,'idx_test.txt')
}





