import logging
import sys
import anndata as ad
import click


@click.command()
@click.argument('adata_filename')
@click.argument('obs_table_filename')
def concatenate_anndata_obs(
        adata_filename,
        obs_table_filename
    ):

    adata = ad.read_h5ad(adata_filename)
    obs_table = adata.obs
    assert obs_table.index.name == 'cell_id'

    obs_table.to_csv(obs_table_filename, index=True, compression={'method':'gzip', 'mtime':0, 'compresslevel':9})


if __name__ == "__main__":
    logging.basicConfig(stream=sys.stderr, level=logging.INFO)
    concatenate_anndata_obs()

