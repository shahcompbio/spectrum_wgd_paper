import logging
import sys
import pandas as pd
import anndata as ad
import numpy as np
import click
import yaml

import spectrumanalysis.amps


@click.command()
@click.argument('adata_filename')
@click.argument('cell_info_filename')
@click.argument('hlamp_events_filename')
def compute_amplifications(
        adata_filename,
        cell_info_filename,
        hlamp_events_filename,
    ):

    adata = ad.read_h5ad(adata_filename)

    cell_info = pd.read_csv(cell_info_filename)
    cell_info = cell_info[(cell_info['include_cell'] == True)]
    adata = adata[adata.obs.index.isin(cell_info['cell_id'])].copy()

    cell_hlamps = spectrumanalysis.amps.compute_hlamp_events(
        adata, copy_norm_threshold=3., copy_context_threshold=None)

    cell_hlamps = cell_hlamps.merge(adata.obs.reset_index()[['cell_id', 'patient_id']])

    cell_hlamps.to_csv(hlamp_events_filename, index=False)


if __name__ == "__main__":
    logging.basicConfig(stream=sys.stderr, level=logging.INFO)
    compute_amplifications()

