import logging
import sys
import pandas as pd
import numpy as np
import anndata as ad
import click


@click.command()
@click.argument('obs_table_filename')
@click.argument('adata_filenames', nargs=-1)
def concatenate_anndata_obs(
        obs_table_filename,
        adata_filenames,
    ):

    obs_table = []

    for adata_filename in adata_filenames:
        obs_table.append(ad.read_h5ad(adata_filename).obs.copy())

    obs_table = pd.concat(obs_table)

    assert obs_table.index.name == 'cell_id'

    obs_table.to_csv(obs_table_filename, index=True, compression={'method':'gzip', 'mtime':0, 'compresslevel':9})


if __name__ == "__main__":
    logging.basicConfig(stream=sys.stderr, level=logging.INFO)
    concatenate_anndata_obs()

