import logging
import sys
import pandas as pd
import anndata as ad
import numpy as np
import click
import yaml


metric = 'state_total_segment_is_diff_threshold_4'

@click.command()
@click.argument('adata_filename')
@click.argument('distances_filename')
@click.argument('nnd_filename')
def compute_cell_nnd(
        adata_filename,
        distances_filename,
        nnd_filename,
    ):

    adata = ad.read_h5ad(adata_filename)
    n_bins = adata.shape[1]

    distances = pd.read_csv(distances_filename, dtype={'cell_id_1': 'category', 'cell_id_2': 'category'})
    distances = distances[distances['cell_id_1'] != distances['cell_id_2']]
    distances = distances.loc[distances.groupby('cell_id_1', observed=True)[metric].idxmin(), ['cell_id_1', 'cell_id_2', metric]]
    cell_distances = distances.rename(columns={'cell_id_1': 'cell_id'})
    cell_distances['nnd'] = (cell_distances[metric]) / n_bins

    cell_distances.to_csv(nnd_filename, index=False, compression={'method':'gzip', 'mtime':0, 'compresslevel':9})


if __name__ == "__main__":
    logging.basicConfig(stream=sys.stderr, level=logging.INFO)
    compute_cell_nnd()

