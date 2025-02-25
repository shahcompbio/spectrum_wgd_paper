import numpy as np
import pandas as pd


def compute_cn_changepoint_matrix(adata):
    """ Compute copy number changepoints in each chromosome and in each cell.
    """
    # Remove poor gc regions
    adata_filtered = adata[:, adata.var['gc'] > 0]

    # Filter high copy regions
    mean_bin_copy = np.array(adata_filtered.layers['copy'].mean(axis=0))
    adata_filtered = adata_filtered[:, mean_bin_copy < 10]

    changepoint_matrix = []

    for _, df in adata_filtered.var.groupby('chr'):
        chr_adata = adata_filtered[:, df.index]

        # Remove poor gc regions
        chr_adata = chr_adata[:, chr_adata.var['gc'] > 0]

        # CN transitions / changepoints using diff, sort first!
        chr_adata = chr_adata[:, chr_adata.var.sort_values(['chr', 'start']).index]
        is_changepoint = (np.diff(chr_adata.layers['state'], axis=1) != 0)

        # Create a changepoint matrix
        is_changepoint_df = pd.DataFrame(is_changepoint, index=chr_adata.obs.index, columns=chr_adata.var.index[:-1]).T
        changepoint_matrix.append(is_changepoint_df)

    changepoint_matrix = pd.concat(changepoint_matrix)

    return changepoint_matrix


