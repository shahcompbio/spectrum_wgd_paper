import numpy as np

import scgenome


def aggregate_cna_adata(
        adata,
        cluster_label,
    ):

    agg_X = np.sum

    agg_layers = {
        'copy': np.nanmean,
        'state': np.nanmedian,
        'alleleA': np.nansum,
        'alleleB': np.nansum,
        'totalcounts': np.nansum,
        'A': np.nanmedian,
        'B': np.nanmedian,
    }

    agg_obs = {
        'n_wgd': lambda a: np.nanmean(a.astype(float)),
    }

    adata_clusters = scgenome.tl.aggregate_clusters(adata, agg_X, agg_layers, agg_obs, cluster_col=cluster_label)
    adata_clusters.layers['state'] = adata_clusters.layers['state'].round()
    adata_clusters.layers['A'] = adata_clusters.layers['A'].round()
    adata_clusters.layers['B'] = adata_clusters.layers['B'].round()
    adata_clusters.layers['BAF'] = np.full(adata_clusters.shape, np.nan)
    adata_clusters.layers['BAF'] = np.divide(adata_clusters.layers['alleleB'], adata_clusters.layers['totalcounts'], out=adata_clusters.layers['BAF'], where=adata_clusters.layers['totalcounts'] != 0)
    adata_clusters.layers['minor'] = np.minimum(adata_clusters.layers['A'], adata_clusters.layers['B'])
    adata_clusters.obs['n_wgd'] = adata_clusters.obs['n_wgd'].round().astype(int)

    return adata_clusters
