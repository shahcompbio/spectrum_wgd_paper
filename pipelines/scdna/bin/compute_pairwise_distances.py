# Compute pairwise distances between cells

import logging
import sys
import pandas as pd
import numpy as np
import anndata as ad
import click
from tqdm import tqdm


import spectrumanalysis.distances


@click.command()
@click.argument('adata_filename')
@click.argument('cell_ids_filename')
@click.argument('output_filename')
def compute_pairwise_distances(
        adata_filename,
        cell_ids_filename,
        output_filename,
    ):

    adata = ad.read(adata_filename)

    cell_ids = pd.read_csv(cell_ids_filename, header=None)[0]

    # Sort adata by segment chromosome, start position, required by distance computation
    adata = adata[:, adata.var.sort_values(['chr', 'start']).index].copy()

    distances = []
    for cell_id in tqdm(cell_ids):
        if cell_id not in adata.obs.index:
            raise ValueError(f'cell id {cell_id} not in adata')
        
        for cell_id2 in adata.obs.index:
            distances.append({
                'cell_id_1': cell_id,
                'cell_id_2': cell_id2,
                'copy_mean_sq_distance': spectrumanalysis.distances.compute_mean_sq_diff(adata, adata, cell_id, cell_id2, 'copy'),
                'copy_mean_sq_wgd_distance': spectrumanalysis.distances.compute_mean_sq_diff_wgd(adata, adata, cell_id, cell_id2, 'copy'),
                'state_mean_is_diff_distance': spectrumanalysis.distances.compute_mean_is_diff(adata, adata, cell_id, cell_id2, 'state'),
                'copy_mean_is_diff_wgd_distance': spectrumanalysis.distances.compute_mean_is_diff_wgd(adata, adata, cell_id, cell_id2, 'state'),
                'state_largest_segment_is_diff_distance': spectrumanalysis.distances.compute_largest_segment_is_diff(adata, adata, cell_id, cell_id2, 'state'),
                'state_total_segment_is_diff_threshold_4': spectrumanalysis.distances.compute_total_segment_is_diff_threshold(adata, adata, cell_id, cell_id2, 'state', 4),
                'state_total_segment_is_diff_threshold_10': spectrumanalysis.distances.compute_total_segment_is_diff_threshold(adata, adata, cell_id, cell_id2, 'state', 10),
                'state_total_segment_is_diff_threshold_20': spectrumanalysis.distances.compute_total_segment_is_diff_threshold(adata, adata, cell_id, cell_id2, 'state', 20),
            })
    distances = pd.DataFrame(distances)

    distances.to_csv(output_filename, index=False, compression={'method':'gzip', 'mtime':0, 'compresslevel':9})


if __name__ == "__main__":
    logging.basicConfig(stream=sys.stderr, level=logging.INFO)
    compute_pairwise_distances()

