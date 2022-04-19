import scanpy as sc
import pandas as pd
import numpy as np
import scipy
import anndata as ad
import sys
from os.path import join

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


if __name__=='__main__':
    lsi, pca, atac, gex, n_comp = sys.argv[1:]
    n_comp = int(n_comp)
    adata_atac = ad.read_h5ad(atac)
    adata_gex = ad.read_h5ad(gex)
    adata_atac = lsi_transform(adata_atac, n_comp)
    adata_gex = pca_transform(adata_gex, n_comp)
    X_lsi = adata_atac.obsm['X_lsi']
    X_pca = adata_gex.obsm['X_pca']
    np.save(lsi, X_lsi)
    np.save(pca, X_pca)

