import numpy as np
import pandas as pd
import scanpy as sc
import scib
from sklearn.metrics.cluster import completeness_score

#adata_result: our output

#adata_raw: benchmark method

#All the datasets I mentioned above can be generated by task3 prediciton code.
#nmi rate
def get_nmi(adata):
    print('Preprocessing')
    sc.pp.neighbors(adata, use_rep='X_emb')
    print('Compute score')
    score = scib.me.nmi(adata, group1='cluster', group2='cell_type')
    return score


#cell type asw
def get_cell_type_ASW(adata):
    return scib.me.silhouette(adata, group_key='cell_type', embed='X_emb')

def get_louvain_ASW(adata):
    return scib.me.silhouette(adata, group_key='cluster', embed='X_emb')

def get_completeness(adata):
    return completeness_score(adata.obs['cell_type'], adata.obs['cluster'])

def evaluation_task3(adata):
    adata.obsm['X_emb'] = adata.obsm['embeddings']

    nmi = get_nmi(adata)
    cell_type_asw = get_cell_type_ASW(adata)
    louvain_asw = get_louvain_ASW(adata)
    completeness = get_completeness(adata)

    print('cell type rate')
    print('nmi:',nmi, '    celltype asw:',cell_type_asw, '       louvain_asw:',louvain_asw, '          completeness:',completeness)
    
    
    
    