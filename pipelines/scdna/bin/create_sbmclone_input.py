import logging
import sys
from scipy.sparse import csr_matrix, save_npz
import pandas as pd
import numpy as np
import click
import pickle
import os

from spectrumanalysis.sparsematrix import create_sparse_matrices


def prepare_input(counts_files, filter_files, cell_list):
    snv_counts = pd.concat([
        pd.read_csv(cf, dtype={'chromosome': str}) for cf in counts_files], ignore_index=True)
    snv_counts = snv_counts.drop_duplicates()

    # Filter SNVs not classified as PASS
    if len(filter_files) > 0:
        filter_results = pd.concat([
            pd.read_table(f, dtype={'chrm': str}) for f in filter_files])
        filter_results = filter_results[filter_results['result'] == 'PASS']
        filter_results = filter_results.rename(columns={
            'chrm': 'chromosome',
            'pos': 'position',
            'ref_allele': 'ref',
            'alt_allele': 'alt',
        })
        filter_results = filter_results[['chromosome', 'position', 'ref', 'alt']].drop_duplicates()

        snv_counts = snv_counts.merge(filter_results)

    # Filter cells not in the input cell list if it exists
    if cell_list is not None:
        filtered_cells = set([c.strip() for c in open(cell_list, 'r').readlines()])
        snv_counts = snv_counts[snv_counts['cell_id'].isin(filtered_cells)]

    # Create genotype column for sbmclone
    snv_counts['genotype'] = np.where(snv_counts['alt_count'] > 0, 1, 0)

    # Create csr matrices
    snv_counts['snv_idx'] = snv_counts.groupby(['chromosome', 'position', 'ref', 'alt'], observed=True).ngroup()
    snv_counts['cell_id'] = snv_counts['cell_id'].astype('category')

    matrices, index, columns = create_sparse_matrices(
        snv_counts, 'cell_id', 'snv_idx', ['alt_count', 'ref_count', 'genotype'])

    # Create chromosome, position, ref, alt index from snv_idx
    snv_info = snv_counts[['chromosome', 'position', 'ref', 'alt', 'snv_idx']].drop_duplicates().set_index('snv_idx')
    snv_info_columns = snv_info.loc[columns, :].set_index(['chromosome', 'position', 'ref', 'alt']).index

    return matrices['genotype'], matrices['ref_count'], matrices['alt_count'], index, snv_info_columns


@click.command()
@click.option('--counts_files', required = True, multiple = True, type = str)
@click.option('--filter_files', required = False, multiple = True, type = str)
@click.option('--cell_list', required = True, type = str)
@click.option('--output_matrix', required = True, type = str)
@click.option('--output_refcounts', required = True, type = str)
@click.option('--output_altcounts', required = True, type = str)
@click.option('--output_metadata', required = True, type = str)
def create_sbmclone_input(
        counts_files,
        filter_files,
        output_matrix,
        output_altcounts,
        output_refcounts,
        output_metadata,
        cell_list
    ):
    M, refM, altM, row_labels, col_labels = prepare_input(counts_files, filter_files, cell_list)
    save_npz(output_matrix, M)
    save_npz(output_refcounts, refM)
    save_npz(output_altcounts, altM)

    with open(output_metadata, 'wb') as f:
        pickle.dump((row_labels, col_labels), f)


if __name__ == "__main__":
    logging.basicConfig(stream=sys.stderr, level=logging.INFO)
    create_sbmclone_input()
