import pandas as pd
import seaborn as sns
import scgenome
import panel as pn
import sys
import click
import os
import anndata as ad
from Bio import Phylo
from copy import deepcopy
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import patches
import tqdm
import pickle
import numpy as np
import itertools
import logging
import click
import spectrumanalysis.plots
import vetica.mpl
import yaml
pn.extension()

n_wgd_colors = {0:mcolors.to_hex((197/255, 197/255, 197/255)),
              1:mcolors.to_hex((252/255, 130/255, 79/255)),
              2:mcolors.to_hex((170/255, 0, 0/255))}

def color_tree_wgd(events, tree, patient_id):
    wgd_cmap = {1:mcolors.cnames['lightsalmon'], 
                2:mcolors.cnames['red'], 
                3:mcolors.cnames['magenta'],
               4:mcolors.cnames['darkviolet']}
    wgd_nodes = set(events[events.type == 'wgd'].sample_id.values)
    
    # Patient specific fixes
    if patient_id == 'SPECTRUM-OV-002':
        wgd_nodes = wgd_nodes.union(['internal30'])
    elif patient_id == 'SPECTRUM-OV-003':
        wgd_nodes = wgd_nodes.union(['internal44'])
    elif patient_id == 'SPECTRUM-OV-014':
        wgd_nodes = wgd_nodes.union(['internal66'])
    elif patient_id == 'SPECTRUM-OV-024':
        wgd_nodes = wgd_nodes.union(['internal10'])
    elif patient_id == 'SPECTRUM-OV-036':
        wgd_nodes = wgd_nodes.union(['internal482'])
    elif patient_id == 'SPECTRUM-OV-044':
        wgd_nodes = wgd_nodes.union(['internal657'])
    elif patient_id == 'SPECTRUM-OV-051':
        wgd_nodes = wgd_nodes.union(['internal1'])
    elif patient_id == 'SPECTRUM-OV-052':
        wgd_nodes = wgd_nodes.union(['internal94'])
    elif patient_id == 'SPECTRUM-OV-071':
        wgd_nodes = wgd_nodes.union(['internal72'])
    elif patient_id == 'SPECTRUM-OV-083':
        wgd_nodes = wgd_nodes.union(['internal20'])

    # Count the # WGD events affecting each branch
    n_wgds = {}
    for wgd_branch in wgd_nodes:
        wgd_clade = [a for a in tree.find_clades(wgd_branch)]
        assert len(wgd_clade) == 1, (wgd_branch, len(wgd_clade))
        wgd_clade = wgd_clade[0]

        for node in wgd_clade.get_terminals():
            if node.name not in n_wgds:
                n_wgds[node.name] = 0
            n_wgds[node.name] += 1
        for node in wgd_clade.get_nonterminals():
            if node.name not in n_wgds:
                n_wgds[node.name] = 0
            n_wgds[node.name] += 1
        wgd_clade.color = wgd_cmap[n_wgds[wgd_clade.name]]
        #print(f'assigning color [{wgd_cmap[n_wgds[wgd_clade.name]]}] to branch [{wgd_clade.name}]')
    return n_wgds

def count_wgd(clade, n_wgd):
    if clade.is_wgd:
        clade.n_wgd = n_wgd + 1
    else:
        clade.n_wgd = n_wgd
    for child in clade.clades:
        count_wgd(child, clade.n_wgd)

def assign_plot_locations(tree):
    """
    Assign plotting locations to clades
    
    Parameters:
    - tree: A Bio.Phylo tree object
    
    Returns:
    - tree: A Bio.Phylo tree object with assigned values
    """

    def assign_branch_pos(clade, counter):
        """
        Recursive function to traverse the tree and assign values.
        """
        # Base case: if this is a leaf, assign the next value and return it
        if clade.is_terminal():
            clade.branch_pos = next(counter)
            return clade.branch_pos
        
        # Recursive case: assign the average of the child values
        child_values = [assign_branch_pos(child, counter) for child in clade]
        average_value = float(sum(child_values)) / float(len(child_values))
        clade.branch_pos = average_value
        return average_value
    
    assign_branch_pos(tree.clade, itertools.count())

    def assign_branch_start(clade, branch_start):
        """
        Recursive function to traverse the tree and assign values.
        """
        clade.branch_start = branch_start
        for child in clade:
            assign_branch_start(child, branch_start + clade.branch_length)
    
    assign_branch_start(tree.clade, 0)

    return tree

def draw_branch_wgd(ax, clade, bar_height=0.25):
    if clade.is_wgd:
        length1 = clade.branch_length * clade.wgd_fraction
        length2 = clade.branch_length * (1. - clade.wgd_fraction)
        rect1 = patches.Rectangle((clade.branch_start, clade.branch_pos-bar_height/2.), length1, bar_height, 
                                  linewidth=0, edgecolor='none', facecolor=n_wgd_colors[clade.n_wgd-1])
        ax.add_patch(rect1)
        rect2 = patches.Rectangle((clade.branch_start + length1, clade.branch_pos-bar_height/2.), length2, bar_height, 
                                  linewidth=0, edgecolor='none', facecolor=n_wgd_colors[clade.n_wgd])
        ax.add_patch(rect2)
        ax.scatter([clade.branch_start + length1], [clade.branch_pos+bar_height], marker='v', color='darkorange')
    else:
        rect = patches.Rectangle(
            (clade.branch_start, clade.branch_pos-bar_height/2.), clade.branch_length, bar_height, 
            linewidth=0, edgecolor='none', facecolor=n_wgd_colors[clade.n_wgd])
        ax.add_patch(rect)

def draw_branch_links(ax, clade, bar_height=0.25):
    if not clade.is_terminal():
        child_pos = [child.branch_pos for child in clade.clades]
        ax.plot(
            [clade.branch_start + clade.branch_length, clade.branch_start + clade.branch_length],
            [min(child_pos)-bar_height/2., max(child_pos)+bar_height/2.], color='k', ls=':')

def draw_leaf_tri_size(ax, clade, bar_height=0.25, max_height=2.):
    if clade.is_terminal():
        expansion_height = bar_height + max(0.1, clade.cell_fraction) * (max_height - bar_height) # bar_height to 1.5

        # Transform to create a regular shaped triangle
        height = (ax.transData.transform([0, expansion_height]) - ax.transData.transform([0, 0]))[1]
        length = (ax.transData.inverted().transform([height, 0]) - ax.transData.inverted().transform([0, 0]))[0]

        branch_end = clade.branch_start+clade.branch_length
        branch_pos_bottom = clade.branch_pos-bar_height/2.
        branch_pos_top = clade.branch_pos+bar_height/2.

        vertices = [
            [branch_end, branch_pos_bottom],
            [branch_end, branch_pos_top],
            [branch_end + length, branch_pos_top + expansion_height / 2],
            [branch_end + length, branch_pos_bottom - expansion_height / 2],
        ]
        tri = patches.Polygon(vertices, linewidth=1, edgecolor='0.25', facecolor='0.25')
        ax.add_patch(tri)

def is_apobec_snv(ref_base, alt_base, trinucleotide_context):
    """
    Classify a SNV as APOBEC-induced based on its substitution type and trinucleotide context.
    This function also accounts for the reverse complement context.

    Parameters:
    - ref_base: The reference base (e.g., 'C').
    - alt_base: The alternate base (e.g., 'T').
    - trinucleotide_context: The trinucleotide context (e.g., 'TCA').

    Returns:
    - True if the SNV is APOBEC-induced, False otherwise.
    """

    # Check if the substitution is a C-to-T or G-to-A transition
    is_c_to_t = ref_base.upper() == 'C' and alt_base.upper() == 'T'
    is_g_to_a = ref_base.upper() == 'G' and alt_base.upper() == 'A'

    # Check if the substitution is a C-to-G or G-to-C transition
    is_c_to_g = ref_base.upper() == 'C' and alt_base.upper() == 'G'
    is_g_to_c = ref_base.upper() == 'G' and alt_base.upper() == 'C'

    # Check if the trinucleotide context fits the TpCpX pattern on the forward strand
    is_tpctx_forward = trinucleotide_context[1].upper() == 'C' and trinucleotide_context[0].upper() == 'T'

    # Check if the trinucleotide context fits the RpGpX pattern on the reverse strand (where R is A or G)
    is_tpctx_reverse = trinucleotide_context[1].upper() == 'G' and trinucleotide_context[2].upper() == 'A'

    # APOBEC-induced mutations are C-to-T or C-to-G in TpCpX context or reverse complement
    return ((is_c_to_t or is_c_to_g) and is_tpctx_forward) or ((is_g_to_a or is_g_to_c) and is_tpctx_reverse)

def is_c_to_t_in_cpg_context(ref_base, alt_base, trinucleotide_context):
    """
    This function checks if a single nucleotide variant (SNV) is a C to T mutation
    in a CpG context or its reverse complement G to A in a CpG context.
    
    Parameters:
    ref_base (str): The reference nucleotide
    alt_base (str): The alternate nucleotide
    trinucleotide_context (str): The trinucleotide context of the SNV (string of 3 nucleotides)
    
    Returns:
    bool: True if the mutation is a C to T mutation in a CpG context or a G to A mutation
          in a CpG context on the reverse strand, False otherwise.
    """
    
    # Check if the mutation is C to T in a CpG context on the forward strand
    if ref_base == 'C' and alt_base == 'T':
        if len(trinucleotide_context) == 3 and trinucleotide_context[1] == 'C' and trinucleotide_context[2] == 'G':
            return True

    # Check if the mutation is G to A in a CpG context on the reverse strand
    if ref_base == 'G' and alt_base == 'A':
        if len(trinucleotide_context) == 3 and trinucleotide_context[0] == 'C' and trinucleotide_context[1] == 'G':
            return True
    
    return False

def draw_branch_wgd_fraction(ax, clade, bar_height=0.25):
    bars = []
    if clade.is_wgd:
        start = clade.branch_start
        length = clade.branch_length * clade.wgd_fraction
        bars.append({'start': start, 'length': length, 'color': n_wgd_colors[clade.n_wgd-1]})

        start += length
        length = clade.branch_length * (1. - clade.wgd_fraction)
        bars.append({'start': start, 'length': length, 'color': n_wgd_colors[clade.n_wgd]})

    else:
        bars.append({'start': clade.branch_start, 'length': clade.branch_length, 'color': n_wgd_colors[clade.n_wgd]})

    for bar in bars:
        rect = patches.Rectangle(
            (bar['start'], clade.branch_pos-bar_height/2.), bar['length'], bar_height, 
            linewidth=0, edgecolor='none', facecolor=bar['color'])
        ax.add_patch(rect)

def draw_branch_wgd_event(ax, clade, bar_height=0.25):
    if clade.is_wgd:
        length1 = clade.branch_length * clade.wgd_fraction
        ax.scatter([clade.branch_start + length1], [clade.branch_pos+bar_height], marker='v', color='darkorange')
       
def draw_branch_wgd_apobec_fraction(ax, clade, apobec_fraction, bar_height=0.25):
    bars = []
    if clade.is_wgd:
        start = clade.branch_start
        length = clade.branch_length * clade.wgd_fraction * apobec_fraction.get((clade.name, 'prewgd'), 0)
        bars.append({'start': start, 'length': length, 'color': 'r'})

        start += length
        length = clade.branch_length * clade.wgd_fraction * (1. - apobec_fraction.get((clade.name, 'prewgd'), 0))
        bars.append({'start': start, 'length': length, 'color': '0.75'})

        start += length
        length = clade.branch_length * (1. - clade.wgd_fraction) * apobec_fraction.get((clade.name, 'postwgd'), 0)
        bars.append({'start': start, 'length': length, 'color': 'r'})

        start += length
        length = clade.branch_length * (1. - clade.wgd_fraction) * (1. - apobec_fraction.get((clade.name, 'postwgd'), 0))
        bars.append({'start': start, 'length': length, 'color': '0.75'})

    else:
        start = clade.branch_start
        length = clade.branch_length * apobec_fraction.get((clade.name, 'none'), 0)
        bars.append({'start': start, 'length': length, 'color': 'r'})

        start += length
        length = clade.branch_length * (1. - apobec_fraction.get((clade.name, 'none'), 0))
        bars.append({'start': start, 'length': length, 'color': '0.75'})

    for bar in bars:
        rect = patches.Rectangle(
            (bar['start'], clade.branch_pos-bar_height/2.), bar['length'], bar_height, 
            linewidth=0, edgecolor='none', facecolor=bar['color'])
        ax.add_patch(rect)

def draw_apobec_fraction(ax, clade, bar_height=0.25):
    if clade.is_wgd:        
        length1 = clade.branch_length * clade.wgd_fraction
        length2 = clade.branch_length * (1. - clade.wgd_fraction)
        rect1 = patches.Rectangle((clade.branch_start, clade.branch_pos-bar_height/2.), length1, bar_height, 
                                  linewidth=0, edgecolor='none', facecolor=n_wgd_colors[clade.n_wgd-1])
        ax.add_patch(rect1)
        rect2 = patches.Rectangle((clade.branch_start + length1, clade.branch_pos-bar_height/2.), length2, bar_height, 
                                  linewidth=0, edgecolor='none', facecolor=n_wgd_colors[clade.n_wgd])
        ax.add_patch(rect2)
        ax.scatter([clade.branch_start + length1], [clade.branch_pos+bar_height], marker='v', color='darkorange')

    else:
        rect = patches.Rectangle(
            (clade.branch_start, clade.branch_pos-bar_height/2.), clade.branch_length, bar_height, 
            linewidth=0, edgecolor='none', facecolor=n_wgd_colors[clade.n_wgd])
        ax.add_patch(rect)

    length1 = clade.branch_length * clade.apobec_fraction
    length2 = clade.branch_length * (1. - clade.apobec_fraction)
    rect1 = patches.Rectangle((clade.branch_start, clade.branch_pos-bar_height/2.), length1, bar_height, 
                              linewidth=0, edgecolor='none', facecolor='r')
    ax.add_patch(rect1)
    rect2 = patches.Rectangle((clade.branch_start + length1, clade.branch_pos-bar_height/2.), length2, bar_height, 
                              linewidth=0, edgecolor='none', facecolor='0.75')
    ax.add_patch(rect2)

def draw_doubletime_tree(tree, patient_id, snv_types, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 1), dpi=150)
    
    for clade in tree.find_clades():
        draw_branch_wgd(ax, clade)
        draw_branch_links(ax, clade)
    
    yticks = list(zip(*[(clade.branch_pos, f'{clade.name.replace("_", " ")}, n={clade.cell_count}') for clade in tree.get_terminals()]))
    ax.set_yticks(*yticks)
    ax.yaxis.tick_right()
    sns.despine(trim=True, left=True, right=False)
    ax.yaxis.tick_right()
    ax.yaxis.set_ticks_position('right')
    ax.set_xlabel('# SNVs')
    ax.set_title(f'{patient_id}\n SNV types: {snv_types}', fontsize=10)
    
    
    for clade in tree.find_clades():
        draw_leaf_tri_size(ax, clade, bar_height=0)
    
    
    legend_elements = [patches.Patch(color=n_wgd_colors[0], label='0'),
                       patches.Patch(color=n_wgd_colors[1], label='1'),
                       patches.Patch(color=n_wgd_colors[2], label='2')]
    legend_1 = ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.5, 1), frameon=False, fontsize=8, title='#WGD')

def draw_doubletime_tree_apobec(tree, data, patient_id, ax=None):
    apobec_fraction = data[['snv', 'clade', 'wgd_timing', 'is_apobec']].drop_duplicates().groupby(['clade', 'wgd_timing'])['is_apobec'].mean()
    apobec_counts = data[['snv', 'clade', 'wgd_timing', 'is_apobec']].drop_duplicates().groupby(['clade', 'wgd_timing', 'is_apobec']).size()

    
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 1), dpi=150)
    
    for clade in tree.find_clades():
        draw_branch_wgd_apobec_fraction(ax, clade, apobec_fraction)
        draw_branch_links(ax, clade)
        draw_branch_wgd_event(ax, clade)
    
    yticks = list(zip(*[(clade.branch_pos, f'{clade.name.replace("_", " ")}, n={clade.cell_count}') for clade in tree.get_terminals()]))
    ax.set_yticks(*yticks)
    ax.yaxis.tick_right()
    sns.despine(trim=True, left=True, right=False)
    ax.yaxis.tick_right()
    ax.yaxis.set_ticks_position('right')
    ax.set_xlabel('# SNVs')
    
    for clade in tree.find_clades():
        draw_leaf_tri_size(ax, clade, bar_height=0)
    
    ax.set_title(f'Patient {patient_id} APOBEC')

def draw_doubletime_tree_cpg(tree, data, patient_id, snv_types, ax=None):
    cpg_tree = deepcopy(tree)
    for clade in cpg_tree.find_clades():
        clade_df = data[(data.clade == clade.name) & (data.is_cpg)]
        
        clade.branch_length = len(clade_df.snv_id.unique())
        
        if clade.is_wgd:
            cntr = clade_df.wgd_timing.value_counts().reindex(['prewgd', 'postwgd'])
            if np.isnan(cntr['prewgd']):
                cntr['prewgd'] = 0
            clade.wgd_fraction = 2 * cntr['prewgd'] / (2 * cntr['prewgd'] + cntr['postwgd'])

    
    #Phylo.draw(cpg_tree)
    assign_plot_locations(cpg_tree)

    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 1), dpi=150)
    
    for clade in cpg_tree.find_clades():
        draw_branch_wgd(ax, clade)
        draw_branch_links(ax, clade)
    
    yticks = list(zip(*[(clade.branch_pos, f'{clade.name.replace("_", " ")}, n={clade.cell_count}') for clade in tree.get_terminals()]))
    ax.set_yticks(*yticks)
    ax.yaxis.tick_right()
    sns.despine(trim=True, left=True, right=False)
    ax.yaxis.tick_right()
    ax.yaxis.set_ticks_position('right')
    ax.set_xlabel('# SNVs')
    ax.set_title(f'{patient_id} CpG\n SNV types: {snv_types}', fontsize=10)
    
    
    for clade in cpg_tree.find_clades():
        draw_leaf_tri_size(ax, clade, bar_height=0)
    
    
    legend_elements = [patches.Patch(color=n_wgd_colors[0], label='0'),
                       patches.Patch(color=n_wgd_colors[1], label='1'),
                       patches.Patch(color=n_wgd_colors[2], label='2')]
    legend_1 = ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.5, 1), frameon=False, fontsize=8, title='#WGD')

def load_doubletime(patients, pipeline_outputs, min_total_counts_perblock):
    doubletime_trees = {}
    doubletime_data = {}
    all_snv_types = {}

    for patient_id in tqdm.tqdm(patients):
        try:
            adata_filename = f'{pipeline_outputs}/tree_snv/inputs/{patient_id}_general_clone_adata.h5'
            tree_filename = f'{pipeline_outputs}/tree_snv/inputs/{patient_id}_clones_pruned.pickle'
            table_filename = f'{pipeline_outputs}/tree_snv/outputs/{patient_id}_general_snv_tree_assignment.csv'
            
            adata = ad.read_h5ad(adata_filename)
            tree = pickle.load(open(tree_filename, 'rb'))
            data = pd.read_csv(table_filename)
        except FileNotFoundError as e:
            print(f"Missing tree_snv outputs for patient {patient_id}:", e)
            continue

        if adata.shape[0] == 0:
            print("Empty anndata for patient", patient_id)
            continue
        
        adata.var['min_total_count'] = np.array(adata.layers['total_count'].min(axis=0))
        adata = adata[:, adata.var['min_total_count'] >= min_total_counts_perblock]
                
        # filter adata based on tree
        clones = []
        for leaf in tree.get_terminals():
            clones.append(leaf.name.replace('clone_', ''))
        
        adata = adata[clones].copy()
        
        clone_sizes = adata.obs['cluster_size'].copy()
        clone_sizes.index = [f'clone_{a}' for a in clone_sizes.index]
        
        for clade in tree.find_clades():
            clade_df = data[data.clade == clade.name]
            cntr = clade_df.wgd_timing.value_counts()
            
            clade.branch_length = len(clade_df.snv_id.unique())
        
            if 'prewgd' in cntr.index:
                assert 'postwgd' in cntr.index
                clade.wgd_fraction = 2 * cntr['prewgd'] / (2 * cntr['prewgd'] + cntr['postwgd'])
            if clade.is_terminal():
                clade.cell_count = clone_sizes.loc[clade.name]
                clade.cell_fraction = clone_sizes.loc[clade.name] / clone_sizes.sum()
        
        assign_plot_locations(tree)
        snv_types = sorted(data.ascn.unique())

        doubletime_trees[patient_id] = tree
        doubletime_data[patient_id] = data
        all_snv_types[patient_id] = snv_types

    for p, tree in doubletime_trees.items():
        for branch in tree.find_clades():
            if branch.is_wgd:
                branch.color = n_wgd_colors[branch.n_wgd]
    
    return doubletime_trees, doubletime_data, all_snv_types

@click.command()
@click.argument('pipeline_outputs')
@click.argument('spectrumanalysis_repo_dir')
@click.argument('medicc2_output_dir')
@click.argument('medicc2_suffix')
@click.argument('infercnv_plot_dir')
@click.argument('output_filename')
@click.option('--has_doubletime', is_flag=True)
@click.option('--has_sbmclone', is_flag=True)
@click.option('--min_total_counts_perblock', type=int, default=2)
def main(pipeline_outputs, spectrumanalysis_repo_dir, medicc2_output_dir, medicc2_suffix, infercnv_plot_dir, output_filename,
         has_doubletime, has_sbmclone, min_total_counts_perblock):
    ## HARDCODED PATHS ## TODO ## WARNING
    patient_vafs_dir = '/data1/shahs3/users/myersm2/repos/spectrum-figures/figures/final/patient-pairwise-vafs/'
    patient_evo_categories = pd.read_csv('/data1/shahs3/users/myersm2/repos/spectrum-figures/tables/patient_evolution_category.csv').set_index('patient_id')['evolution_category']

    colors_file = os.path.join(spectrumanalysis_repo_dir, 'config/colors.yaml')
    colors_yaml = yaml.safe_load(open(colors_file, 'r').read())

    sigs_file = os.path.join(spectrumanalysis_repo_dir, 'annotations/mutational_signatures.tsv')
    sigs = pd.read_table(sigs_file).set_index('patient_id')

    sample_info = pd.read_table(os.path.join(spectrumanalysis_repo_dir, 'metadata/tables/sequencing_scdna.tsv'))
    sample_info = sample_info.drop(['sample_id'], axis=1).rename(columns={'spectrum_sample_id': 'sample_id'})

    cell_info_file = os.path.join(pipeline_outputs, 'preprocessing/summary/filtered_cell_table.csv.gz')
    signals_adata_stem = os.path.join(pipeline_outputs, 'preprocessing/signals')

    ### load data
    # cell info
    cell_info = pd.read_csv(cell_info_file)

    # signals anndatas
    signals_adatas = {}
    for f in sorted(os.listdir(signals_adata_stem)):
        if '.h5' not in f:
            continue
            
        p = f.split('_')[1].split('.')[0]
        adata = ad.read_h5ad(os.path.join(signals_adata_stem, f))
        orig_order = adata.obs.index
        exclusive_cols = set(cell_info.columns) - set(adata.obs.columns)
        adata.obs = adata.obs.merge(cell_info[list(exclusive_cols)], left_index=True, right_on='cell_id').set_index('cell_id').loc[orig_order]
        signals_adatas[p] = adata
    patients = sorted(signals_adatas.keys())
    
    # MEDICC2 trees
    all_trees = {}
    all_events = {}
    for p in sorted(patients):
        all_events[p] = pd.read_table(os.path.join(medicc2_output_dir, f'{p}{medicc2_suffix}',  f'{p}{medicc2_suffix}_copynumber_events_df.tsv'))
        all_trees[p] = Phylo.read(os.path.join(medicc2_output_dir, f'{p}{medicc2_suffix}', f'{p}{medicc2_suffix}_final_tree.new'), 'newick')
        color_tree_wgd(all_events[p], all_trees[p], p)

    if has_sbmclone:
        # sbmclone anndatas
        sbmclone_dir = os.path.join(pipeline_outputs, 'sbmclone') 
        sbmclone_adatas = {}
        for p in patients:
            sbmclone_adatas[p] = ad.read_h5ad(os.path.join(sbmclone_dir, f'sbmclone_{p}_snv.h5'))

    if has_doubletime:
        # load doubleTime objects
        doubletime_trees, doubletime_data, all_snv_types = load_doubletime(patients, pipeline_outputs, min_total_counts_perblock)

    ### plot rconnect
    message = """
        This page shows supplementary data visualization for the 41 HGSOC patients described in "Ongoing genome doubling promotes evolvability and immune dysregulation in ovarian cancer." Only those cells that pass filtering are shown (for DLP+ scWGS, only non-divergent cells were included in the MEDICC2 tree, so divergent cells are omitted).

        Tabs:
        * **total**: total copy-number profiles
        * **allelic imbalance**: cell genomes colored by the direction and extent of allelic imbalance
        * **SBMClone matrix**: cell-by-SNV matrix ordered by SBMClone-inferred blocks, where 1-entries are black (downsampled for visualization)
        * **SBMClone densities**: proportion of 1-entries for each combination of clone and SNV block
        * **doubleTime**: for patients included in doubleTime analysis, the top panel shows the basic clone tree, the middle panel shows the clone tree with raw SNV counts as branch lengths, and the bottom panel shows the clone tree with C>T CpG SNV counts as branch lengths. Neither set of branch lengths was corrected for sensitivity as in the trees shown in the main figure.
        * **scRNA inferCNV**: for patients with matched scRNA-seq data, this tab shows inferCNV-inferred copy-number profiles annotated with tumor site, clone, and patient-level WGD classification.

        Cell annotations:
        * **tumor_site**: anatomical site from which cells were obtained for sequencing
        * **n_wgd**: number of WGD events ancestral to the cell according to the WGD classification heuristic
        * **sbmclone_cluster_id**: SBMClone clone ID to which the cell was assigned

        Patient annotations:

        * **WGD-high**  (>50\% WGD cells) or **WGD-low** (<15\% WGD cells)  depending on the overall prevalence of cells with at least 1 WGD
        * **Evolutionary category**: Truncal WGD (all cells share 1 ancestral WGD), Parallel WGD (multiple WGD clones), Subclonal WGD (WGD clone and 0xWGD cells both present), or Unexpanded WGD (no evident WGD clone)
        * **Signature**: FBI, HRD-Dup, HRD-Del, or Undetermined
        * **BRCA status** indicating WT (wild type), gBRCA1/2 (germline mutation), or sBRCA1/2 (somatic mutation)

        In DLP+ scWGS tabs, MEDICC2 branches are colored by the number of WGD events ancestral to the branch according to MEDICC2. This may differ from the annotation on the right which shows the WGD heuristic used in the paper.
    """
    column = pn.Column(scroll=True)
    md = pn.pane.Markdown(message, width = 800)
    column.append(md)

    pane_dpi = 175 # originally 144
    pfig_height = 800 # originally 800

    annot_fields = ['tumor_site', 'n_wgd']
    if has_sbmclone:
        annot_fields += ['sbmclone_cluster_id']

    i = 0
    for p, adata in sorted(signals_adatas.items()):
        tree = deepcopy(all_trees[p])
        # keep track of the root branch length before pruning and restore it after pruning
        nondiploid = [a for a in tree.root.clades if a.name != 'diploid']
        assert len(nondiploid) == 1
        nondiploid = nondiploid[0]
        rootlength = nondiploid.branch_length
        tree.prune('diploid')
        tree.root.branch_length = rootlength

        events = all_events[p]
        my_info = cell_info[cell_info.include_cell & (cell_info.patient_id == p)]
        adata = adata.copy()
        if has_sbmclone:
            adata.obs = adata.obs.merge(sbmclone_adatas[p].obs[['sbmclone_cluster_id']], how = 'left', left_index=True, right_index=True)
            adata.obs.sbmclone_cluster_id = adata.obs.sbmclone_cluster_id.fillna(-1).astype(int).astype('category')
        adata.obs = adata.obs.set_index('brief_cell_id')
        adata.obs['tumor_site'] = adata.obs['sample_id'].map(sample_info.dropna(subset=['sample_id', 'tumor_site']).drop_duplicates(['sample_id', 'tumor_site']).set_index('sample_id')['tumor_site'])

        adata = adata[my_info.brief_cell_id]
        tree_cells = set([a.name for a in tree.get_terminals()])
        adata_cells = set(adata.obs.index)
        print(p, len(tree_cells), len(adata_cells), len(tree_cells.intersection(adata_cells)))

        if len(tree_cells - adata_cells) > 0:
            print(f"Pruning {len(tree_cells - adata_cells)} cells present in the adata but not in the tree")
            for c in tree_cells - adata_cells:
                tree.prune(c)
        plot_cells = sorted(tree_cells.intersection(adata_cells))

        tree_color = color_tree_wgd(events, tree, p)
        adata = adata[plot_cells].copy()
        adata.obs['n_wgd_medicc'] = adata.obs.index.map(tree_color).fillna(0).astype(int)
        adata.obs['wgd_diff'] = adata.obs.n_wgd - adata.obs.n_wgd_medicc
        adata.obs.n_wgd = adata.obs.n_wgd.astype('category')
        print(adata.obs[['n_wgd', 'n_wgd_medicc']].value_counts())

        tabs = pn.Tabs()
        fig = plt.figure(figsize=(12, 8), dpi = pane_dpi)
        hm = scgenome.pl.plot_cell_cn_matrix_fig(
            adata,
            tree = tree,
            annotation_fields = annot_fields, layer_name = 'state', fig = fig,
            annotation_cmap={'tumor_site':{k:v for k,v in colors_yaml['tumor_site'].items() if k in adata.obs['tumor_site'].unique()}, 
                            'n_wgd':{k:v for k,v in n_wgd_colors.items() if k in adata.obs.n_wgd.unique()}, 
                            'sbmclone_cluster_id':'tab10'}
        )

        pfig = pn.pane.Matplotlib(fig, dpi=pane_dpi, tight=True)
        pfig.height = pfig_height
        tabs.append(('total', pfig))
        plt.close()

        adata = spectrumanalysis.plots.add_allele_state_layer(adata)

        fig = plt.figure(figsize=(12, 8), dpi = pane_dpi)
        scgenome.pl.plot_cell_cn_matrix_fig(
            adata,
            tree=tree,
            layer_name='allele_state',
            raw=True,
            cmap=spectrumanalysis.plots.allele_state_cmap,
            fig=fig,
            annotation_fields=annot_fields,
            annotation_cmap={'tumor_site':{k:v for k,v in colors_yaml['tumor_site'].items() if k in adata.obs['tumor_site'].unique()}, 
                            'n_wgd':{k:v for k,v in n_wgd_colors.items() if k in adata.obs.n_wgd.unique()}, 
                            'sbmclone_cluster_id':'tab10'}
        )
        pfig = pn.pane.Matplotlib(fig, dpi=pane_dpi, tight=True)
        pfig.height = pfig_height
        tabs.append(('allelic imbalance', pfig))
        plt.close()
        
        if has_sbmclone:
            tabs.append(('SBMClone matrix', pn.panel(os.path.join(pipeline_outputs, 'sbmclone', p, 'matrix_fig.png'), height = pfig_height)))
            tabs.append(('SBMClone densities', pn.panel(os.path.join(pipeline_outputs, 'sbmclone', p, 'density_fig.png'), height = pfig_height)))

        if os.path.exists(os.path.join(patient_vafs_dir, f'vafs_{p}.png')):
            tabs.append(('pairwise VAFs', pn.panel(os.path.join(patient_vafs_dir, f'vafs_{p}.png'), height = pfig_height)))

        if has_doubletime and p in doubletime_trees:
            fig = plt.figure(figsize=(12, 8), dpi = pane_dpi, layout='constrained')
            gs = fig.add_gridspec(3, 2)
            ax1 = fig.add_subplot(gs[0, :])
            Phylo.draw(doubletime_trees[p], axes=ax1)
            ax1.set_title("doubleTime tree")

            if p in doubletime_trees:
                draw_doubletime_tree(doubletime_trees[p], p, all_snv_types[p], ax = fig.add_subplot(gs[1, :]))    
                draw_doubletime_tree_cpg(doubletime_trees[p], doubletime_data[p], p, all_snv_types[p], ax = fig.add_subplot(gs[2, :]))
            plt.tight_layout()
            
            pfig = pn.pane.Matplotlib(fig, dpi=pane_dpi, tight=True)
            pfig.height = pfig_height
            tabs.append(('doubleTime', pfig))
            plt.close()
        
        if os.path.exists(os.path.join(infercnv_plot_dir, f'{p}_infercnv_heatmap.png')):
            tabs.append(('scRNA inferCNV', pn.panel(os.path.join(infercnv_plot_dir, f'{p}_infercnv_heatmap.png'), height = int(pfig_height*0.8))))

        column.append(pn.Card(tabs, title = f'{p} ({adata.shape[0]} cells, WGD-{"high" if (adata.obs.n_wgd.astype(int) > 0).mean() > 0.5 else "low"}, ' +
                              f'{patient_evo_categories[p]}, {sigs.loc[p, "consensus_signature"]}, {sigs.loc[p, "BRCA_gene_mutation_status"].replace("Wildtype", "BRCA WT")})'))

    from bokeh.resources import INLINE
    column.save(output_filename, resources=INLINE)



if __name__ == "__main__":
    logging.basicConfig(stream=sys.stderr, level=logging.INFO)
    main()