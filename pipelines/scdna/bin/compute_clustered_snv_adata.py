import click
import pandas as pd
import anndata as ad
import numpy as np
import logging
import sys
import pickle
import scgenome
import scipy
from Bio.Phylo.BaseTree import Clade

def find_clade(T, clade_name, expected=True):
    my_clade = [a for a in T.find_clades(clade_name)]
    if expected:
        assert len(my_clade) == 1, f"Expected 1 clade matching {clade_name}, found {len(my_clade)}"
        my_clade = my_clade[0]
        return my_clade
    else:
        assert len(my_clade) == 0, f"Expected 0 clades matching {clade_name}, found {len(my_clade)}"
        return None

def add_wgd_tree(patient_id, T, adata_cn_clusters):
    for clade in T.find_clades():
        clade.is_wgd = False

    if patient_id.endswith('025'):
        print(T.get_terminals())
        # 2 WGDs
        clade0 = find_clade(T, 'clone_0')
        clade1 = find_clade(T, 'clone_1')
        clone01_mca = T.common_ancestor(clade0, clade1)
        clone01_mca.is_wgd = True

        # clade 2 should be dropped for being too small after removing 2x WGD cells
        clade2 = find_clade(T, 'clone_2', expected=False)
        clade3 = find_clade(T, 'clone_3')
        clone23_mca = clade3
        clone23_mca.is_wgd = True

    elif patient_id.endswith('045'):
        clade0 = find_clade(T, 'clone_0')
        clade0.is_wgd = True
        clade2 = find_clade(T, 'clone_2')
        clade2.is_wgd = True
        clade1 = find_clade(T, 'clone_1')
        clade3 = find_clade(T, 'clone_3')

        internal_clades = [int(x.name.split('_')[1]) for x in T.find_clades() if x.name.startswith('internal')]
        if len(internal_clades) > 0:
            new_name = f'internal_{max(internal_clades) + 1}'
        else:
            new_name = 'internal_0'

        mca_13 = T.common_ancestor(clade1, clade3)
        if len(mca_13.clades) > 2:
            # add dummy parent to split WGD events
            new_parent = Clade(1)
            new_parent.clades = [clade1, clade3]
            new_parent.is_wgd=True
            new_parent.mutations=''
            new_parent.name = new_name
            mca_13.clades.remove(clade1)
            mca_13.clades.remove(clade3)
            mca_13.clades.append(new_parent)

    elif patient_id.endswith('081'):
        print(T.get_terminals())
        assert adata_cn_clusters.obs['n_wgd']['0'] == 1
        assert adata_cn_clusters.obs['n_wgd']['1'] == 0
        assert adata_cn_clusters.obs['n_wgd']['2'] == 0
        assert adata_cn_clusters.obs['n_wgd']['3'] == 0
        clade0 = [a for a in T.find_clades('clone_0')]
        assert len(clade0) == 1, clade0
        clade0 = clade0[0]
        clade0.is_wgd = True

    else:
        assert (adata_cn_clusters.obs['n_wgd'] == 0).all() or (adata_cn_clusters.obs['n_wgd'] == 1).all()
        T.clade.is_wgd = (adata_cn_clusters.obs['n_wgd'] == 1).all()


def count_wgd(clade, n_wgd):
    if clade.is_wgd:
        clade.n_wgd = n_wgd + 1
    else:
        clade.n_wgd = n_wgd
    for child in clade.clades:
        count_wgd(child, clade.n_wgd)


@click.command()
@click.option('--adata_cna')
@click.option('--adata_snv')
@click.option('--tree_filename')
@click.option('--cell_info_filename')
@click.option('--patient_id')
@click.option('--output_cn')
@click.option('--output_snv')
@click.option('--output_pruned_tree')
@click.option('--min_clone_size', type=int, default=20, required=False)
@click.option('--across_clones_homogeneity_threshold', type=float, default=0.9, required=False)
@click.option('--within_clone_homogeneity_threshold', type=float, default=0.8, required=False)
def main(adata_cna, adata_snv, tree_filename, cell_info_filename, patient_id, output_cn, output_snv, output_pruned_tree, 
         min_clone_size, across_clones_homogeneity_threshold, within_clone_homogeneity_threshold):

    cell_info = pd.read_csv(cell_info_filename)
    cell_info['haploid_depth'] = cell_info['coverage_depth'] / cell_info['ploidy']

    adata = ad.read_h5ad(adata_cna)
    adata.obs = adata.obs.merge(cell_info.set_index('cell_id')[['include_cell', 'multipolar', 'haploid_depth']], left_index=True, right_index=True, how='left')
    assert not adata.obs['multipolar'].isnull().any()

    snv_adata = ad.read_h5ad(adata_snv)
    snv_adata.obs = snv_adata.obs.merge(cell_info.set_index('cell_id')[['include_cell', 'multipolar', 'n_wgd']], left_index=True, right_index=True, how='left')
    assert not snv_adata.obs['multipolar'].isnull().any()

    tree = pickle.load(open(tree_filename, 'rb'))

    adata = adata[snv_adata.obs.index]
    adata.obs = adata.obs.merge(snv_adata.obs[['sbmclone_cluster_id']], left_index=True, right_index=True, how='left')
    assert not adata.obs['sbmclone_cluster_id'].isnull().any()

    # Wrangle CN anndata, identify bins with compatible cn, filter clones
    # 

    # Add the modal wgd state for each sbmclone
    adata.obs['n_wgd_mode'] = adata.obs.groupby('sbmclone_cluster_id')['n_wgd'].transform(lambda x: x.mode()[0])

    # Add leaf id to copy number anndata
    # Since multiple clones could have been combined into one leaf
    # of the clone tree, there is not necessarily a one to one mapping
    # of leaves to clones
    block2leaf = {}
    for l in tree.get_terminals():
        for b in l.name.lstrip('clone_').split('/'):
            block2leaf[int(b)] = l.name.lstrip('clone_') # TODO: why?
    adata.obs['leaf_id'] = adata.obs.sbmclone_cluster_id.map(block2leaf)

    # Filter anndata for high quality non-multipolar cells
    # with n_wgd<=1 and n_wgd equal to the modal n_wgd
    adata = adata[(
        adata.obs.include_cell &
        ~adata.obs.multipolar &
        (adata.obs.n_wgd.astype(int) <= 1) &
        adata.obs.n_wgd == adata.obs.n_wgd_mode)].copy()

    # Threshold on size of clone
    adata.obs['leaf_size'] = adata.obs.groupby('leaf_id').transform('size')
    adata = adata[adata.obs['leaf_size'] >= min_clone_size]

    # Aggregate the copy number based on the leaf id
    adata_cn_clusters = scgenome.tl.aggregate_clusters(
        adata,
        agg_layers={
            'A': 'median',
            'B': 'median',
            'state': 'median',
        },
        cluster_col='leaf_id')
    adata_cn_clusters.obs.index = adata_cn_clusters.obs.index.astype(str)

    # Add per clone statistics for the frequency of the median state
    for layer in ['state', 'A', 'B']:
        adata.layers[f'clone_median_{layer}'] = adata_cn_clusters.to_df(layer).loc[adata.obs['leaf_id'].values, :]
        adata.layers[f'is_eq_clone_median_{layer}'] = adata.layers[layer] == adata.layers[f'clone_median_{layer}']
        adata.var[f'is_eq_clone_median_{layer}'] = np.nanmean(adata.layers[f'is_eq_clone_median_{layer}'], axis=0)

    # Redo aggregation of the copy number and include per clone stats for frequency of median state
    adata_cn_clusters = scgenome.tl.aggregate_clusters(
        adata,
        agg_layers={
            'A': 'median',
            'B': 'median',
            'state': 'median',
            'is_eq_clone_median_state': 'mean',
            'is_eq_clone_median_A': 'mean',
            'is_eq_clone_median_B': 'mean',
        },
        agg_obs={
            'n_wgd': 'median',
            'haploid_depth': 'sum',
        },
        cluster_col='leaf_id')
    adata_cn_clusters.obs.index = adata_cn_clusters.obs.index.astype(str)
    adata_cn_clusters.obs['n_wgd'] = adata_cn_clusters.obs['n_wgd'].round()

    # Calculate and threshold for homogeneous copy number within each clone
    adata_cn_clusters.var['is_homogenous_cn'] = (
        (adata_cn_clusters.var['is_eq_clone_median_A'] > across_clones_homogeneity_threshold) &
        (adata_cn_clusters.var['is_eq_clone_median_B'] > across_clones_homogeneity_threshold) &
        (adata_cn_clusters.layers['is_eq_clone_median_A'] > within_clone_homogeneity_threshold).all(axis=0) &
        (adata_cn_clusters.layers['is_eq_clone_median_B'] > within_clone_homogeneity_threshold).all(axis=0))

    # Compatible states for WGD1 and WGD0, major/minor for the snv tree model
    compatible_cn_types = {
        '1:0': [{'n_wgd': 1, 'A': 1, 'B': 0}, {'n_wgd': 0, 'A': 1, 'B': 0}],
        '2:0': [{'n_wgd': 1, 'A': 2, 'B': 0}, {'n_wgd': 0, 'A': 1, 'B': 0}],
        '1:1': [{'n_wgd': 1, 'A': 1, 'B': 1}, {'n_wgd': 0, 'A': 1, 'B': 1}],
        '2:1': [{'n_wgd': 1, 'A': 2, 'B': 1}, {'n_wgd': 0, 'A': 1, 'B': 1}],
        '2:2': [{'n_wgd': 1, 'A': 2, 'B': 2}, {'n_wgd': 0, 'A': 1, 'B': 1}],
    }

    # Check for compatibility and assign
    adata_cn_clusters.var['snv_type'] = 'incompatible'
    for name, cn_states in compatible_cn_types.items():
        cn_states = pd.DataFrame(cn_states).set_index('n_wgd')
        clone_maj = adata_cn_clusters.obs['n_wgd'].map(cn_states['A'])
        clone_min = adata_cn_clusters.obs['n_wgd'].map(cn_states['B'])
        is_compatible = (
            (adata_cn_clusters.layers['A'] == clone_maj.values[:, np.newaxis]) &
            (adata_cn_clusters.layers['B'] == clone_min.values[:, np.newaxis]))
        bin_is_compatible = np.all(is_compatible, axis=0)
        adata_cn_clusters.var.loc[bin_is_compatible, 'snv_type'] = name

    # Wrangle tree, prune based on removed clusters
    #

    # Prune clones from the tree if they were removed due to size
    remaining_leaves = adata_cn_clusters.obs.index
    tree = scgenome.tl.prune_leaves(tree, lambda a: a.name.lstrip('clone_') not in remaining_leaves)
    # Merge branches
    def merge_branches(parent, child):
        return {
            'name': child.name,
            'branch_length': 1,
            'mutations': ','.join(filter(lambda a: a, [parent.mutations, child.mutations])),
        }
    tree = scgenome.tl.aggregate_tree_branches(tree, f_merge=merge_branches)

    # Manually add WGD events to the tree
    add_wgd_tree(patient_id, tree, adata_cn_clusters)

    # Recursively add n_wgd to each clade
    count_wgd(tree.clade, 0)

    # Wrangle SNV anndata, filter based on cn adatas and merge bin information
    #

    # Filter snv adata similarly to cn adata
    snv_adata = snv_adata[adata.obs.index]

    # Aggregate snv counts
    snv_adata.obs['leaf_id'] = snv_adata.obs.sbmclone_cluster_id.map(block2leaf)
    adata_clusters = scgenome.tl.aggregate_clusters(
        snv_adata,
        agg_layers={
            'alt': 'sum',
            'ref': 'sum',
            'A': 'median',
            'B': 'median',
            'state': 'median',
        },
        cluster_col='leaf_id')

    # Filter snv cluster data similar to copy number
    adata_clusters = adata_clusters[adata_cn_clusters.obs.index].copy()

    # Additional layers
    adata_clusters.layers['vaf'] = adata_clusters.layers['alt'] / (adata_clusters.layers['ref'] + adata_clusters.layers['alt'])
    adata_clusters.layers['ref_count'] = adata_clusters.layers['ref']
    adata_clusters.layers['alt_count'] = adata_clusters.layers['alt']
    adata_clusters.layers['total_count'] = adata_clusters.layers['ref'] + adata_clusters.layers['alt']

    # Add information from cn bin analysis
    adata_clusters.var['snv_type'] = adata_cn_clusters.var.loc[adata_clusters.var['cn_bin'], 'snv_type'].values
    adata_clusters.var['is_homogenous_cn'] = adata_cn_clusters.var.loc[adata_clusters.var['cn_bin'], 'is_homogenous_cn'].values

    adata_cn_clusters.write(output_cn)
    adata_clusters.write(output_snv)
    with open(output_pruned_tree, 'wb') as f:
        pickle.dump(tree, f)


if __name__ == "__main__":
    logging.basicConfig(stream=sys.stderr, level=logging.INFO)
    main()
