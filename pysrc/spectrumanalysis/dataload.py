import os
import pandas as pd
import numpy as np
import anndata as ad
import Bio
import pickle

import scgenome

import spectrumanalysis.wgd
import spectrumanalysis.ultrametric.preprocessing


def load_medicc(cn_filename, tree_filename, input_filename=None):
    medicc_input_dtype = {
        'sample_id': 'category',
        'chrom': 'category',
        'start': int,
        'end': int,
        'chr': 'category',
        'original_sample_id': 'category',
        'original_library_id': 'category',
        'cn_a': int,
        'cn_b': int,
    }

    if input_filename is not None and os.path.exists(input_filename):
        medicc_input = pd.read_csv(input_filename, sep='\t', dtype=medicc_input_dtype)

        if 'sample_id.1' in medicc_input.columns:
            cell_info = medicc_input.rename(columns={
                'sample_id': 'cell_id',
                'sample_id.1': 'sample_id',
            })[['cell_id', 'sample_id']].drop_duplicates()

        else:
            cell_info = medicc_input.rename(columns={
                'sample_id': 'cell_id',
                'original_sample_id': 'sample_id',
                'original_library_id': 'library_id',
            })[['cell_id', 'sample_id', 'library_id']].drop_duplicates()

    else:
        cell_info = pd.read_csv(cn_filename, sep='\t')
        cell_info = cell_info.rename(columns={
            'sample_id': 'cell_id'})[['cell_id']].drop_duplicates()
        cell_info['sample_id'] = cell_info['cell_id'].str.rsplit('-', expand=True, n=3)[0]
        cell_info['library_id'] = cell_info['cell_id'].str.rsplit('-', expand=True, n=3)[1]

    adata = scgenome.pp.load_cn.read_medicc2_cn(cn_filename, allele_specific=False)

    def load_tree(newick_filename):
        tree = Bio.Phylo.read(newick_filename, 'newick')
        scgenome.tl.prune_leaves(tree, lambda a: a.name == 'diploid')
        return tree

    tree = load_tree(tree_filename)

    adata.obs['sample_id'] = cell_info.set_index('cell_id')['sample_id']
    
    return tree, adata


def load_medicc_tree(newick_filename):
    tree = Bio.Phylo.read(newick_filename, 'newick')

    # Remove the artificial diploid leaf
    tree = scgenome.tl.prune_leaves(tree, lambda a: a.name == 'diploid')

    # Merge the two root branch resulting from removing the diploid leaf
    def f_merge(parent, child):
        return {
            'name': parent.name if parent.name else child.name,
            'branch_length': parent.branch_length + child.branch_length,
        }
    tree = scgenome.tl.aggregate_tree_branches(tree, f_merge)

    return tree


def load_medicc_as(cn_filename, tree_filename, events_filename, input_filename=None):
    medicc_input_dtype = {
        'sample_id': 'category',
        'chrom': 'category',
        'start': int,
        'end': int,
        'chr': 'category',
        'original_sample_id': 'category',
        'original_library_id': 'category',
        'cn_a': int,
        'cn_b': int,
    }

    if input_filename is not None and os.path.exists(input_filename):
        medicc_input = pd.read_csv(input_filename, sep='\t', dtype=medicc_input_dtype)

        cell_info = medicc_input.rename(columns={
            'sample_id': 'cell_id',
            'original_sample_id': 'sample_id',
            'original_library_id': 'library_id',
        })[['cell_id', 'sample_id', 'library_id']].drop_duplicates()

    else:
        cell_info = pd.read_csv(cn_filename, sep='\t')
        cell_info = cell_info.rename(columns={
            'sample_id': 'cell_id'})[['cell_id']].drop_duplicates()
        cell_info['sample_id'] = cell_info['cell_id'].str.rsplit('-', expand=True, n=3)[0]
        cell_info['library_id'] = cell_info['cell_id'].str.rsplit('-', expand=True, n=3)[1]

    adata = scgenome.pp.load_cn.read_medicc2_cn(cn_filename, allele_specific=True)

    tree = load_medicc_tree(tree_filename)

    root = tree.clade.name
    adata.obs['is_root'] = False
    adata.obs.loc[root, 'is_root'] = True

    leaves = [a.name for a in tree.get_terminals()]
    adata.obs['is_cell'] = False
    adata.obs.loc[leaves, 'is_cell'] = True

    adata.obs['is_internal'] = (~adata.obs['is_root']) & (~adata.obs['is_cell'])

    adata.obs['sample_id'] = cell_info.set_index('cell_id')['sample_id']

    events = pd.read_csv(events_filename, sep='\t', dtype=medicc_input_dtype)

    return tree, adata, events


def load_signals(hscn_filename):
    signals_dtype = {
        'cell_id': 'category',
        'chr': 'category',
        'state_AS_phased': 'category',
        'state_AS': 'category',
        'phase': 'category',
        'state_phase': 'category',
    }

    hscn = pd.read_csv(hscn_filename, dtype=signals_dtype)

    adata = scgenome.pp.convert_dlp_signals(hscn, hscn[['cell_id']].drop_duplicates())

    return adata


class MissingDataError(Exception):
    pass


def load_clone_tree_data(patient_id, project_dir, version='v5'):
    """ Load clone tree data for a patient.
    """

    tree_filename = f'{project_dir}/tree_snv/inputs/{patient_id}_clones_pruned.pickle'
    tree_assignments_filename = f'{project_dir}/tree_snv/outputs/{patient_id}_general_snv_tree_assignment.csv'
    adata_clusters_filename = f'{project_dir}/tree_snv/inputs/{patient_id}_cna_clustered.h5'
    snv_adata_filename = f'{project_dir}/sbmclone/sbmclone_{patient_id}_snv.h5'
    adata_filename = f'{project_dir}/preprocessing/signals/signals_{patient_id}.h5'
    cell_info_filename = f'{project_dir}/preprocessing/summary/filtered_cell_table.csv.gz'

    if not os.path.exists(tree_filename):
        raise MissingDataError(f'tree missing for patient {patient_id}: {tree_filename}')

    if not os.path.exists(adata_clusters_filename):
        raise MissingDataError(f'adata missing for patient {patient_id}: {adata_clusters_filename}')

    if not os.path.exists(tree_assignments_filename):
        raise MissingDataError(f'tree assignments missing for patient {patient_id}: {tree_assignments_filename}')

    tree_assignments = pd.read_csv(tree_assignments_filename, dtype={'chromosome': 'str'})
    if tree_assignments.empty:
        raise MissingDataError()
    
    # Read in required files
    tree = pickle.load(open(tree_filename, 'rb'))
    snv_adata = ad.read_h5ad(snv_adata_filename)
    adata = ad.read_h5ad(adata_filename)
    adata_clusters = ad.read(adata_clusters_filename)
    cell_info = pd.read_csv(cell_info_filename).set_index('cell_id')

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

    # Add wgd info to the 
    cell_info = spectrumanalysis.wgd.classify_subclonal_wgd(cell_info).loc[adata.obs.index]
    adata.obs['majority_n_wgd'] = cell_info['majority_n_wgd']
    adata.obs['subclonal_wgd'] = cell_info['subclonal_wgd']
    adata.obs['multipolar'] = cell_info['multipolar']

    # HACK
    if 'snv_id' not in tree_assignments:
        tree_assignments['snv_id'] = tree_assignments['snv_id_signature']

    # Kludge: add is_wgd
    tree_assignments = tree_assignments.merge(pd.Series({a.name: a.is_wgd for a in tree.find_clades()}, name='is_wgd').rename_axis('clade').reset_index())

    tree = spectrumanalysis.ultrametric.preprocessing.preprocess_ultrametric_tree(tree)
    
    branch_info = spectrumanalysis.ultrametric.preprocessing.generate_snv_counts(tree_assignments, tree)
    snv_subsets = spectrumanalysis.ultrametric.preprocessing.generate_snv_genotype_subsets_table(adata_clusters, tree)
    opportunity = snv_subsets.groupby('branch_segment')['opportunity'].sum()

    cluster_info = adata_clusters.obs
    cluster_info['branch_segment'] = {a.cluster_id: a.name for a in tree.get_terminals()}

    branch_info['opportunity'] = opportunity

    return {
        'tree': tree,
        'tree_assignments': tree_assignments,
        'adata': adata,
        'adata_clusters': adata_clusters,
        'snv_adata': snv_adata,
        'branch_info': branch_info,
        'cluster_info': cluster_info,
    }


def load_filtered_cna_adata(project_dir, patient_id):
    cell_info = pd.read_csv(f'{project_dir}/preprocessing/summary/filtered_cell_table.csv.gz')
    cell_info = cell_info[(cell_info['include_cell'] == True)]
    cell_info = spectrumanalysis.wgd.classify_subclonal_wgd(cell_info)
    cell_info = cell_info.set_index('cell_id')

    sbmclone_cell_info = ad.read_h5ad(f'{project_dir}/sbmclone/sbmclone_{patient_id}_snv.h5').obs

    adata = ad.read_h5ad(f'{project_dir}/preprocessing/signals/signals_{patient_id}.h5')
    adata = adata[adata.obs.index.isin(cell_info.index)].copy()
    adata.obs['majority_n_wgd'] = cell_info['majority_n_wgd']
    adata.obs['subclonal_wgd'] = cell_info['subclonal_wgd']
    adata.obs['multipolar'] = cell_info['multipolar']
    adata.obs['sbmclone_cluster_id'] = sbmclone_cell_info['sbmclone_cluster_id'].astype(str)

    return adata



def load_wgd_ar(project_dir, patient_id, use_sankoff_ar=False, diploid_parent=False, wgd_clade_cell_threshold=30):

    prefix = f'{project_dir}/postprocessing/sankoff_ar'
    filename = f'{prefix}/{patient_id}/sankoff_ar_{patient_id}.h5'
    tree_filename = f'{prefix}/{patient_id}/sankoff_ar_tree_{patient_id}.pickle'
    events_filename = f'{prefix}/greedy_events/{patient_id}/sankoff_ar_{patient_id}_copynumber_events_df.tsv'

    adata = ad.read_h5ad(filename)

    with open(tree_filename, 'rb') as f:
        tree = pickle.load(f)

    events = pd.read_csv(events_filename, sep='\t', dtype={'chr': str}, low_memory=False)

    if use_sankoff_ar:
        cn_field_suffix = '_2'
    else:
        cn_field_suffix = ''

    # Pre post WGD changes that minimize total changes
    n_states = int(max(adata.layers['cn_a' + cn_field_suffix].max(), adata.layers['cn_b' + cn_field_suffix].max()) + 1)
    pre_post_changes = spectrumanalysis.phylocn.calculate_pre_post_changes(n_states)

    # Get parent names for recomputing cn changes pre/post WGD
    parent_names = {tree.clade.name: 'diploid'}
    for clade in tree.find_clades():
        for child in clade.clades:
            parent_names[child.name] = clade.name

    # Identify large WGD clades
    wgd_clades = []
    for a in tree.find_clades():
        if a.wgd:
            if diploid_parent:
                parent_clade = 'diploid'
            else:
                parent_clade = parent_names[a.name]
            wgd_clades.append({'name': a.name, 'wgd': a.wgd, 'parent': parent_clade, 'n_leaves': a.count_terminals()})
    wgd_clades = pd.DataFrame(wgd_clades)
    wgd_clades = wgd_clades[wgd_clades['n_leaves'] >= wgd_clade_cell_threshold]

    events['wgd_clade'] = events['obs_id']
    events = events[events['wgd_clade'].isin(wgd_clades['name'].values)]
    events = events[events['kind'] != 'wgd']
    events.loc[events['region'].isin(['p-arm', 'q-arm']), 'region'] = 'arm'
    assert events['kind'].isin(['loss', 'gain']).all()

    wgd_obs = pd.concat([wgd_clades.assign(timing='pre'), wgd_clades.assign(timing='post')])
    wgd_obs = wgd_obs.set_index(wgd_obs['name'] + '_' + wgd_obs['timing'])

    # Generate pre/post WGD anndata
    wgd_changes = ad.AnnData(
        obs=wgd_obs,
        var=adata.var,
    )
    wgd_changes.layers['cn_a'] = np.zeros(wgd_changes.shape)
    wgd_changes.layers['cn_b'] = np.zeros(wgd_changes.shape)
    wgd_changes.layers['cn_a_pre'] = np.zeros(wgd_changes.shape)
    wgd_changes.layers['cn_b_pre'] = np.zeros(wgd_changes.shape)

    for idx, row in wgd_clades.iterrows():
        wgd_clade = row['name']
        parent_clade = row['parent']

        for allele in ['cn_a', 'cn_b']:

            parent_cn = np.array(adata[parent_clade, :].layers[f'{allele}{cn_field_suffix}'][0])
            child_cn = np.array(adata[wgd_clade, :].layers[f'{allele}{cn_field_suffix}'][0])

            for timing in ['pre', 'post']:
                wgd_clade_id = wgd_clade + '_' + timing
                wgd_clade_idx = wgd_changes.obs.index.get_loc(wgd_clade_id)

                wgd_changes.layers[allele][wgd_clade_idx, :] = pre_post_changes.loc[pd.IndexSlice[zip(parent_cn, child_cn)], timing].values

                if timing == 'pre':
                    wgd_changes.layers[allele + '_pre'][wgd_clade_idx, :] = parent_cn + wgd_changes.layers[allele][wgd_clade_idx, :]

    return {
        'adata': adata,
        'wgd_clades': wgd_clades,
        'wgd_changes': wgd_changes,
        'events': events,
    }
