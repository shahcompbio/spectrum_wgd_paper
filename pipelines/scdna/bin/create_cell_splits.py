#!/juno/work/shah/users/mcphera1/repos/spectrumanalysis/analysis/notebooks/scdna/venv/bin/python

# TODO
###!/usr/bin/env python

import logging
import sys
import pandas as pd
import numpy as np
import anndata as ad
import click


@click.command()
@click.argument('adata_filename')
@click.argument('n_split', type=int)
@click.argument('cell_ids_filenames', nargs=-1)
def create_cell_split(
        adata_filename,
        n_split,
        cell_ids_filenames,
    ):

    adata = ad.read(adata_filename)

    # Evenly split cells into n_split groups
    cell_ids = adata.obs.index.values
    cell_ids_split = np.array_split(cell_ids, n_split)

    # Write cell ids to files
    for cell_ids_filename, cell_ids in zip(cell_ids_filenames, cell_ids_split):
        pd.Series(cell_ids).to_csv(cell_ids_filename, index=False, header=False)


if __name__ == "__main__":
    logging.basicConfig(stream=sys.stderr, level=logging.INFO)
    create_cell_split()

