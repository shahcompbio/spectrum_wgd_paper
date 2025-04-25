import click
import pandas as pd
import anndata as ad
import numpy as np
import logging
import sys
import os
from scipy.sparse import load_npz
import scipy
from scgenome.tools.ranges import dataframe_to_pyranges


def create_sbmclone_adata(sbmclone_outdir):

    sbmclone_alt = load_npz(os.path.join(sbmclone_outdir, 'alt_counts.npz'))
    sbmclone_ref = load_npz(os.path.join(sbmclone_outdir, 'ref_counts.npz'))

    sbmclone_cells = pd.read_csv(os.path.join(sbmclone_outdir, 'cells.csv'))
    sbmclone_snvs = pd.read_csv(os.path.join(sbmclone_outdir, 'snvs.csv'), dtype={'chromosome': str})

    sbmclone_cells = sbmclone_cells.set_index('cell_id')
    sbmclone_cells.index.name = 'brief_cell_id'

    sbmclone_snvs.index = (
        sbmclone_snvs['chromosome'] + ':' +
        sbmclone_snvs['position'].astype(str) + ':' +
        sbmclone_snvs['ref'] + '>' +
        sbmclone_snvs['alt'])
    sbmclone_snvs.index.name = 'snv_id'

    adata = ad.AnnData(
        obs=sbmclone_cells,
        var=sbmclone_snvs,
        layers={
            'alt': sbmclone_alt,
            'ref': sbmclone_ref,
        },
        dtype=int)

    adata.layers['total'] = adata.layers['alt'] + adata.layers['ref']

    adata.obs['sbmclone_cluster_id'] = adata.obs['block_assignment']

    return adata


def add_cn_bin(adata, cn_bins):

    assert 'bin' in cn_bins

    # Convert from HMMCopy 1-based end included to 1-based end excluded pyranges
    cn_ranges = dataframe_to_pyranges(cn_bins)
    cn_ranges = cn_ranges.assign('End', lambda df: df['End'] + 1)

    snv_ranges = dataframe_to_pyranges(adata.var.reset_index().assign(
        chr=lambda df: df['chromosome'], start=lambda df: df['position'], end=lambda df: df['position']))
    snv_ranges = snv_ranges.assign('End', lambda df: df['End'] + 1) # TODO: is this necessary, are we 1-based or 0-based?

    i1 = snv_ranges.intersect(cn_ranges).as_df()
    i2 = cn_ranges.intersect(snv_ranges).as_df()
    intersection = i1.merge(i2)

    adata.var['cn_bin'] = intersection.set_index('snv_id')['bin']

    assert not adata.var['cn_bin'].isnull().any()

    return adata


@click.command()
@click.option('--signals_adata')
@click.option('--sbmclone_outdir')
@click.option('--output')
def main(signals_adata, sbmclone_outdir, output):

    adata = create_sbmclone_adata(sbmclone_outdir)
    adata = adata[:, adata.var['chromosome'] != 'Y'].copy()

    cn_adata = ad.read_h5ad(signals_adata)

    adata = add_cn_bin(adata, cn_adata.var.reset_index())

    # Add full cell id to adata
    adata.obs['cell_id'] = cn_adata.obs.reset_index().set_index('brief_cell_id').loc[adata.obs.index, 'cell_id']
    assert not adata.obs['cell_id'].isnull().any()

    # Add copy number layers to adata
    adata.layers['state'] = cn_adata[adata.obs['cell_id'].values, adata.var['cn_bin'].values].layers['state']
    adata.layers['A'] = cn_adata[adata.obs['cell_id'].values, adata.var['cn_bin'].values].layers['A']
    adata.layers['B'] = cn_adata[adata.obs['cell_id'].values, adata.var['cn_bin'].values].layers['B']
    adata.obs = adata.obs.reset_index().set_index('cell_id')
    adata.write(output)


if __name__ == "__main__":
    logging.basicConfig(stream=sys.stderr, level=logging.INFO)
    main()
