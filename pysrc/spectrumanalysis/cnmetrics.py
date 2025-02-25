import numpy as np
import pandas as pd

import scgenome


def calculate_variance_metrics(adata, allele_specific=False):
    agg_layers = {
        'copy': np.nanmean,
        'state': np.nanmedian,
    }

    diff_layers = ['state', 'copy']

    if allele_specific:
        agg_layers.update({
            'alleleA': np.nansum,
            'alleleB': np.nansum,
            'totalcounts': np.nansum,
            'B': np.nanmedian,
            'A': np.nanmedian,
        })

        diff_layers.extend(['BAF', 'A', 'B'])

    adata_clusters = scgenome.tl.aggregate_clusters(adata, agg_X=np.sum, agg_layers=agg_layers)

    if allele_specific:
        with np.errstate(divide='ignore', invalid='ignore'):
            adata_clusters.layers['BAF'] = adata_clusters.layers['alleleB'] / adata_clusters.layers['totalcounts']

    for layer_name in diff_layers:
        cell_data = adata.layers[layer_name]
        cluster_data = adata_clusters[adata.obs['cluster_id'].astype(str).values, :].layers[layer_name]

        # Cluster cell differences
        adata.layers[layer_name + '_abs_diff'] = np.absolute(cell_data - cluster_data)
        adata.layers[layer_name + '_sq_diff'] = np.square(cell_data - cluster_data)

        # Per bin means
        adata.var['mean_' + layer_name + '_abs_diff'] = np.array(np.nanmean(adata.layers[layer_name + '_abs_diff'], axis=0))
        adata.var['mean_' + layer_name + '_sq_diff'] = np.array(np.nanmean(adata.layers[layer_name + '_sq_diff'], axis=0))

        # Per cell means
        adata.obs['mean_' + layer_name + '_abs_diff'] = np.array(np.nanmean(adata.layers[layer_name + '_abs_diff'], axis=1))
        adata.obs['mean_' + layer_name + '_sq_diff'] = np.array(np.nanmean(adata.layers[layer_name + '_sq_diff'], axis=1))

    adata.layers['copy_state_abs_diff'] = np.absolute(adata.layers['copy'] - adata.layers['state'])
    adata.layers['copy_state_sq_diff'] = np.square(adata.layers['copy'] - adata.layers['state'])

    # Per bin means
    adata.var['mean_copy_state_abs_diff'] = np.array(np.nanmean(adata.layers['copy_state_abs_diff'], axis=0))
    adata.var['mean_copy_state_sq_diff'] = np.array(np.nanmean(adata.layers['copy_state_sq_diff'], axis=0))

    # Per cell means
    adata.obs['mean_copy_state_abs_diff'] = np.array(np.nanmean(adata.layers['copy_state_abs_diff'], axis=1))
    adata.obs['mean_copy_state_sq_diff'] = np.array(np.nanmean(adata.layers['copy_state_sq_diff'], axis=1))

    if allele_specific:
        adata.layers['BAF_signals'] = adata.layers['B'] / adata.layers['state']

        adata.layers['BAF_ideal_abs_diff'] = np.absolute(adata.layers['BAF'] - adata.layers['BAF_signals'])
        adata.layers['BAF_ideal_sq_diff'] = np.square(adata.layers['BAF'] - adata.layers['BAF_signals'])

        # Per bin means
        adata.var['mean_BAF_ideal_abs_diff'] = np.array(np.nanmean(adata.layers['BAF_ideal_abs_diff'], axis=0))
        adata.var['mean_BAF_ideal_sq_diff'] = np.array(np.nanmean(adata.layers['BAF_ideal_sq_diff'], axis=0))

        # Per cell means
        adata.obs['mean_BAF_ideal_abs_diff'] = np.array(np.nanmean(adata.layers['BAF_ideal_abs_diff'], axis=1))
        adata.obs['mean_BAF_ideal_sq_diff'] = np.array(np.nanmean(adata.layers['BAF_ideal_sq_diff'], axis=1))

    return adata


def calculate_local_baf_variance(adata):
    adata = adata.copy()

    agg_X = np.sum

    agg_layers = {
        'alleleA': np.nansum,
        'alleleB': np.nansum,
        'totalcounts': np.nansum,
    }

    adata.obs['all_cells'] = 1
    adata_pseudobulk = scgenome.tl.aggregate_clusters(adata, agg_X, agg_layers, cluster_col='all_cells')
    with np.errstate(divide='ignore', invalid='ignore'):
        adata_pseudobulk.layers['BAF'] = adata_pseudobulk.layers['alleleB'] / adata_pseudobulk.layers['totalcounts']

    baf = adata_pseudobulk.var[['chr', 'start', 'end']].copy()
    baf['BAF'] = np.array(adata_pseudobulk.layers['BAF'][0])
    baf = baf.sort_values(['chr', 'start']).dropna()

    indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=4)
    fwd = baf['BAF'].rolling(window=indexer, min_periods=1).var()
    rev = baf['BAF'].iloc[::-1].rolling(window=indexer, min_periods=1).var().iloc[::-1]
    baf['local_baf_variance'] = np.minimum(np.nan_to_num(fwd.values, nan=np.inf), np.nan_to_num(rev.values, nan=np.inf))

    adata.var['local_baf_variance'] = baf['local_baf_variance']

    return adata

