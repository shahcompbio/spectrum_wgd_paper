import pickle
import click
import pandas as pd
import anndata as ad
import numpy as np
import logging
import sys
from gzip import GzipFile
from io import TextIOWrapper

import spectrumanalysis.ultrametric.preprocessing
import spectrumanalysis.wgd

@click.command()
@click.argument('tree_filename')
@click.argument('assignments_filename')
@click.argument('snv_adata_filename')
@click.argument('adata_filename')
@click.argument('adata_clusters_filename')
@click.argument('output_tree_filename')
@click.argument('output_branch_info_filename')
@click.argument('output_cluster_info_filename')
@click.argument('output_cell_info_filename')
def tree_snv_postprocessing(
        tree_filename,
        assignments_filename,
        snv_adata_filename,
        adata_filename,
        adata_clusters_filename,
        output_tree_filename,
        output_branch_info_filename,
        output_cluster_info_filename,
        output_cell_info_filename,
    ):
    
    # Read in required files
    tree = pickle.load(open(tree_filename, 'rb'))
    tree_assignments = pd.read_csv(assignments_filename, dtype={'chromosome': 'str'})
    snv_adata = ad.read_h5ad(snv_adata_filename)
    adata = ad.read_h5ad(adata_filename)
    adata_clusters = ad.read(adata_clusters_filename)

    # Filter CN adata and add smbclone cluster id
    adata = adata[snv_adata.obs.index]
    adata.obs = adata.obs.merge(snv_adata.obs[['sbmclone_cluster_id']], left_index=True, right_index=True, how='left')
    assert not adata.obs['sbmclone_cluster_id'].isnull().any()

    # TODO: this is copied around lots
    block2leaf = {}
    for l in tree.get_terminals():
        for b in l.name.lstrip('clone_').split('/'):
            block2leaf[int(b)] = l.name.lstrip('clone_') # TODO: why?
    adata.obs['leaf_id'] = adata.obs.sbmclone_cluster_id.map(block2leaf)

    # Kludge: add is_wgd
    tree_assignments = tree_assignments.merge(pd.Series({a.name: a.is_wgd for a in tree.find_clades()}, name='is_wgd').rename_axis('clade').reset_index())

    tree = spectrumanalysis.ultrametric.preprocessing.preprocess_ultrametric_tree(tree)
    
    branch_info = spectrumanalysis.ultrametric.preprocessing.generate_snv_counts(tree_assignments, tree)
    snv_subsets = spectrumanalysis.ultrametric.preprocessing.generate_snv_genotype_subsets_table(adata_clusters, tree)
    opportunity = snv_subsets.groupby('branch_segment')['opportunity'].sum()

    cluster_info = adata_clusters.obs
    cluster_info['branch_segment'] = {a.cluster_id: a.name for a in tree.get_terminals()}

    branch_info['opportunity'] = opportunity

    # TODO
    sample_info = pd.read_csv('../../metadata/tables/sequencing_scdna.tsv', sep='\t')
    sample_info = sample_info.drop(['sample_id'], axis=1).rename(columns={'spectrum_sample_id': 'sample_id'})

    # Add sample information
    for site_col in ['tumor_site', 'tumor_megasite']:
        site_map = sample_info.drop_duplicates(['sample_id', site_col])[['sample_id', site_col]].dropna().set_index('sample_id')[site_col]
        adata.obs[site_col] = adata.obs['sample_id'].map(site_map)

    # Add filtered n_wgd counts
    patient_cells = adata.obs.reset_index()
    n_wgd_counts = (
        patient_cells
            .groupby(['leaf_id', 'n_wgd'])
            .size().unstack(fill_value=0)
            .T.reindex([0, 1, 2], fill_value=0).T
            .rename(columns=lambda a: f'n_wgd_{a}'))
    cluster_info = cluster_info.merge(n_wgd_counts, left_index=True, right_index=True, how='left')

    # Add total snv counts
    for snv_type_suffix in ('', '_age'):
        branch_info[f'expected_snv_count{snv_type_suffix}_per_gb'] = branch_info[f'snv_count{snv_type_suffix}'] / (branch_info['opportunity'])
        cluster_info[f'sum_snv_count{snv_type_suffix}_per_gb'] = branch_info.loc[tree.clade.name, f'expected_snv_count{snv_type_suffix}_per_gb']
        for leaf in tree.get_terminals():
            for clade in tree.get_path(leaf.name):
                cluster_info.loc[leaf.cluster_id, f'sum_snv_count{snv_type_suffix}_per_gb'] += branch_info.loc[clade.name, f'expected_snv_count{snv_type_suffix}_per_gb']

    # Add is_wgd to branch table
    branch_info['is_wgd'] = False
    for clade in tree.find_clades():
        branch_info.loc[clade.name, 'is_wgd'] = clade.is_wgd

    with open(output_tree_filename, 'wb') as f:
        pickle.dump(tree, f)
    branch_info.to_csv(output_branch_info_filename, index=True, compression={'method':'gzip', 'mtime':0, 'compresslevel':9})
    cluster_info.to_csv(output_cluster_info_filename, index=True, compression={'method':'gzip', 'mtime':0, 'compresslevel':9})
    adata.obs[['leaf_id', 'n_wgd', 'tumor_site']].to_csv(output_cell_info_filename, index=True, compression={'method':'gzip', 'mtime':0, 'compresslevel':9})


if __name__ == "__main__":
    logging.basicConfig(stream=sys.stderr, level=logging.INFO)
    tree_snv_postprocessing()
