import scanpy as sc
import pandas as pd
import numpy as np
import scipy

from sklearn.utils.extmath import randomized_svd
#.A1 means to dense
#For ATAC
def lsi_transform(adata: sc.AnnData, n_comp=50, n_peaks=30000):
    top_idx = set(np.argsort(adata.X.sum(axis=0).A1)[-n_peaks:])
    features = adata.var_names.tolist()
    X = adata[:, features].layers["counts"]
    idf = X.shape[0] / X.sum(axis=0).A1
    if scipy.sparse.issparse(X):
        tf = X.multiply(1 / X.sum(axis=1))
        X = tf.multiply(idf)
        X = X.multiply(1e4 / X.sum(axis=1))
    else:
        tf = X / X.sum(axis=1, keepdims=True)
        X = tf * idf
        X = X * (1e4 / X.sum(axis=1, keepdims=True))
    X = np.log1p(X)
    u, s, vh = randomized_svd(X, n_comp, n_iter=15, random_state=0)
    X_lsi = X @ vh.T / s
    X_lsi -= X_lsi.mean(axis=1, keepdims=True)
    X_lsi /= X_lsi.std(axis=1, ddof=1, keepdims=True)
    adata.obsm["X_lsi"] = X_lsi
    return adata

#For rna seq
def pca_transform(adata: sc.AnnData, n_comp = 50):
    adata_new = sc.AnnData(adata_gex.X)
    sc.pp.scale(adata_new)
    sc.tl.pca(adata_new, n_comp)
    adata.obsm['X_pca'] = adata_new.obsm['X_pca']
    return adata