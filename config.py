from os.path import join

PROJ_DIR = '/home/gcgreen2/neurips_comp'
DATA_DIR = join(PROJ_DIR, 'data', 'multiome')
OUT_DIR = join(PROJ_DIR, 'out', '1-3')

par = {
    'out_dir': OUT_DIR,
    'data_mod1': join(DATA_DIR,'multiome_gex_processed_training.h5ad'),
    'data_mod2': join(DATA_DIR,'multiome_atac_processed_training.h5ad'),
    'model': 'CAE',
    'n_pcs': 2000,
    'h1_dim': 128,
    'h2_dim': 32,
    'z_dim': 8,
    'pct_train': 0.8,
    'n_epochs': 200,
    'batch' : 128,
    'lr' : 0.001,
    'lambda_mod1' : 1,
    'lambda_mod2' : 1,
    'lambda_latent' : 100,
    'lambda_reg' : 0.01
}
par['in_dim'] = par['n_pcs']

files = {
    'log': join(OUT_DIR, 'log'),
    'mod1_pca': join(DATA_DIR, 'pca', 'mod1_pca_{}.npy'.format(par['n_pcs'])),
    'mod2_pca': join(DATA_DIR, 'pca', 'mod2_pca_{}.npy'.format(par['n_pcs'])),
    'model': join(OUT_DIR, 'model.torch'),
    'idx_train': join(OUT_DIR,'idx_train.txt'),
    'idx_test': join(OUT_DIR,'idx_test.txt'),
    'config': join(PROJ_DIR, 'cae/config.py'),
    'config_cp': join(OUT_DIR, 'config.py')
}





