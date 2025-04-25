#!/usr/bin/env python

import logging
import sys
import pandas as pd
import numpy as np
import click

import scgenome


@click.command()
@click.argument('hmmcopy_metrics_filename')
@click.argument('hmmcopy_reads_filename')
@click.argument('patient_id')
@click.argument('sample_id')
@click.argument('aliquot_id')
@click.argument('adata_filename')
def create_hmmcopy_anndata(
        hmmcopy_reads_filename,
        hmmcopy_metrics_filename,
        patient_id,
        sample_id,
        aliquot_id,
        adata_filename,
    ):

    hmmcopy_reads_dtype = {
        'cell_id': 'category',
        'chr': 'category',
    }

    hmmcopy_reads = pd.read_csv(hmmcopy_reads_filename, dtype=hmmcopy_reads_dtype)

    hmmcopy_metrics_dtype = {
        'library_id': 'category',
    }

    hmmcopy_metrics = pd.read_csv(hmmcopy_metrics_filename, dtype=hmmcopy_metrics_dtype)
    hmmcopy_metrics['patient_id'] = patient_id
    hmmcopy_metrics['sample_id'] = sample_id
    hmmcopy_metrics['aliquot_id'] = aliquot_id

    # Add sample id to cell id
    hmmcopy_metrics['brief_cell_id'] = hmmcopy_metrics['cell_id']
    hmmcopy_metrics['cell_id'] = (sample_id + '-' + hmmcopy_metrics['brief_cell_id'].astype(str)).astype('category')
    hmmcopy_reads['brief_cell_id'] = hmmcopy_reads['cell_id']
    hmmcopy_reads['cell_id'] = (sample_id + '-' + hmmcopy_reads['brief_cell_id'].astype(str)).astype('category')

    def check_same(a, b):
        a, b = set(a), set(b)
        if a != b:
            raise ValueError(f'mismatch, a-b: {a-b}, b-a: {b-a}')
    check_same(hmmcopy_reads['cell_id'].values, hmmcopy_metrics['cell_id'].values)

    X_column = 'reads'

    layers_columns = [
        'copy',
        'state',
    ]

    bin_metrics_columns = [
        'chr',
        'start',
        'end',
        'gc',
    ]

    bin_metrics_data = hmmcopy_reads[bin_metrics_columns].drop_duplicates()

    # Create anndata from hmmcopy input
    adata = scgenome.pp.create_cn_anndata(
        hmmcopy_reads,
        X_column=X_column,
        layers_columns=layers_columns,
        cell_metrics_data=hmmcopy_metrics,
        bin_metrics_data=bin_metrics_data,
    )

    # Check: need to have bin as index name for var...
    assert adata.var.index.name == 'bin'

    # Add cytoband
    adata.var = scgenome.tl.add_cyto_giemsa_stain(adata.var)

    # Add chromosome arm
    adata.var['arm'] = None
    adata.var.loc[adata.var['cyto_band_name'].str.startswith('p'), 'arm'] = 'p'
    adata.var.loc[adata.var['cyto_band_name'].str.startswith('q'), 'arm'] = 'q'
    assert adata.var['arm'].notnull().any()

    # Remove Y chromosome
    adata = adata[:, adata.var['chr'] != 'Y'].copy()

    # GC < 0 is actually missing values
    adata.var.loc[adata.var['gc'] < 0, 'gc'] = np.NaN

    adata.write(adata_filename)


if __name__ == "__main__":
    logging.basicConfig(stream=sys.stderr, level=logging.INFO)
    create_hmmcopy_anndata()

