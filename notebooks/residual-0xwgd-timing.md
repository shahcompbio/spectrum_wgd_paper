---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.16.1
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

```python
import pandas as pd
import numpy as np
import anndata as ad
import os
import seaborn as sns
import matplotlib.pyplot as plt
import tqdm
import scgenome 

from sklearn.metrics import adjusted_rand_score
from IPython.display import Image
import matplotlib.cm as cm
import Bio.Phylo
from copy import deepcopy
import pickle
import itertools
import matplotlib.colors as mcolors
```

```python
pipeline_outputs = pipeline_dir # path to root directory of scWGS pipeline outputs
min_total_counts_perblock = 2
patients = [a.split('_')[0] for a in os.listdir(os.path.join(pipeline_outputs, 'tree_snv/outputs'))]
```

# load doubleTime data
* get CpG mutations
* count proportion CpG before WGD vs. after on path from root to leaves

```python

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

```

```python
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

```

```python
doubletime_trees, doubletime_data, all_snv_types = load_doubletime(patients, pipeline_outputs, min_total_counts_perblock)
```

```python
doubletime_data['SPECTRUM-OV-081']
```

# compute proportion of time before WGD

```python
doubletime_trees
```

```python

```

```python

```

```python

```

```python

```

# load hmmcopy tables to get total sequenced cells

```python
hmmcopy_table = pd.read_csv('../pipelines/scdna/inputs/hmmcopy_table.csv')
aliquot2cells = {}
all_depth = []
all_breadth = []
for _, r in hmmcopy_table.iterrows():
    hmmcopy = pd.read_csv(r.MONDRIAN_HMMCOPY_metrics)
    aliquot2cells[r.isabl_aliquot_id] = len(hmmcopy)
    all_depth.extend(hmmcopy.coverage_depth.values)
    all_breadth.extend(hmmcopy.coverage_breadth.values)
```

```python
sample2cells = {s:sum([aliquot2cells[a] for a in sdf.isabl_aliquot_id]) for s, sdf in hmmcopy_table.groupby('isabl_sample_id')}
```

# generate sample-level DLP summary tables
* filtering counts
* sequencing metrics (all cells and filtered only)


```python
cell_info = pd.read_csv(os.path.join(pipeline_outputs, 'preprocessing/summary/filtered_cell_table.csv.gz'))
# reformat cell filtering fields to be boolean
cell_info['filter_passing'] = cell_info.include_cell
cell_info['is_doublet'] = cell_info.is_doublet != 'No'
cell_info['too_short_135_segment'] = cell_info.longest_135_segment < 20
# fix definition of aberrant normal to be specific rather than including all normal-classified cells
cell_info['is_aberrant_normal_cell'] = np.logical_xor(cell_info.is_aberrant_normal_cell, cell_info.is_normal)
orig_cell_info = cell_info.copy()


# apply cell filtering thresholds successively in the order described in the paper
cell_filtering_fields = ['is_normal', 'is_aberrant_normal_cell', 'is_s_phase_thresholds', 'is_doublet', 'too_short_135_segment']
cell_filtering_values = [cell_info[c].values for c in cell_filtering_fields]
cell_info['is_s_phase_thresholds'] = np.logical_and(cell_info.is_s_phase_thresholds, ~np.any(cell_filtering_values[:2], axis = 0))
cell_info['is_doublet'] = np.logical_and(cell_info.is_doublet, ~np.any(cell_filtering_values[:3], axis = 0))
cell_info['too_short_135_segment'] = np.logical_and(cell_info.too_short_135_segment, ~np.any(cell_filtering_values[:4], axis = 0))
```

```python
cell_info.filter_passing.sum()
```

```python
mean_fields = ['total_mapped_reads', 'coverage_depth', 'coverage_breadth', 'quality', 'ploidy', 'fraction_loh', 'breakpoints']
sum_fields = ['is_normal', 'is_aberrant_normal_cell', 'is_s_phase_thresholds', 'is_doublet', 'too_short_135_segment', 'filter_passing']

name_mapping = {'is_s_phase_thresholds':'sphase_cells',
                'is_normal':'normal_cells',
                'is_aberrant_normal_cell':'aberrant_normal_cells',
                'is_doublet':'doublets',
                'filter_passing':'filter_passing_cells',
                'multipolar':'divergent_cells',
                'too_short_135_segment':'too_short_135_segment_cells'
               }

# collect all cell counts
df = cell_info[['patient_id', 'sample_id']].drop_duplicates().reset_index(drop=True)
df['total_sequenced_cells'] = df.sample_id.map(sample2cells)
df = df.merge(cell_info.groupby('sample_id').size().reset_index().rename(columns={0:'high_quality_cells'}), how = 'left')
df['low_quality_cells'] = df.total_sequenced_cells - df.high_quality_cells 
for field in sum_fields:
    new_field = cell_info.groupby('sample_id')[field].sum().reset_index()
    df = df.merge(new_field, how = 'left')

# count the number of cells in each WGD state
wgds = cell_info[cell_info.include_cell][['sample_id', 'n_wgd']].value_counts().reset_index().pivot(index='sample_id', columns ='n_wgd')
wgds.columns = [f'{x}xWGD_cells' for x in wgds.columns.get_level_values(1)]
wgds = wgds.fillna(0).astype(int).reset_index()
df = df.merge(wgds, how = 'left')

# for divergent cells, only count those cells that passed filtering
df = df.merge(cell_info[cell_info.include_cell].groupby('sample_id').multipolar.sum().reset_index(), how = 'left')

# count the number of divergent cells in each WGD state
wgds_divergent = cell_info[cell_info.include_cell & cell_info.multipolar][['sample_id', 'n_wgd']].value_counts().reset_index().pivot(index='sample_id', columns ='n_wgd')
wgds_divergent.columns = [f'{x}xWGD_divergent_cells' for x in wgds_divergent.columns.get_level_values(1)]
wgds_divergent = wgds_divergent.fillna(0).astype(int).reset_index()
df = df.merge(wgds_divergent, how = 'left')

# average over sequencing statistics for highquality cells (those in cell_info table)
for field in mean_fields:
    new_field = cell_info.groupby('sample_id')[field].mean().reset_index()
    new_field.columns = [new_field.columns[0], 'highquality_mean_' + new_field.columns[1]]
    df = df.merge(new_field, how = 'left')
# average over sequencing statistics for 
for field in mean_fields:
    new_field = cell_info[cell_info.include_cell].groupby('sample_id')[field].mean().reset_index()
    new_field.columns = [new_field.columns[0], 'filtered_mean_' + new_field.columns[1]]
    df = df.merge(new_field, how = 'left')

df = df.drop(columns = ['high_quality_cells'])
df = df.rename(columns=name_mapping)

```

```python
df.filter_passing_cells.sum()
```

```python
cell_info.groupby('sample_id').filter_passing.sum().sum()
```

```python
assert len(orig_cell_info[orig_cell_info.include_cell]) == df.filter_passing_cells.sum()

assert np.array_equal(df.total_sequenced_cells, 
                      df.low_quality_cells + df.normal_cells + df.aberrant_normal_cells + df.sphase_cells
                      + df.doublets + df.too_short_135_segment_cells + df.filter_passing_cells)
```

```python
len(df)
```

```python
df.iloc[0]
```

```python
oc = orig_cell_info[orig_cell_info.sample_id == 'SPECTRUM-OV-002_S1_INFRACOLIC_OMENTUM']

for i, c in enumerate(cell_filtering_fields):
    print(c, np.logical_and(oc[c], ~oc[cell_filtering_fields[:i]].any(axis=1)).sum())
```

# write table

```python
df.to_csv('../../tables/dlp_sample_summary.csv')
```

# summary numbers

```python
df.total_sequenced_cells.sum()
```

```python
np.median(all_depth)
```

```python
np.median(all_breadth)
```

```python
df.groupby('patient_id').total_sequenced_cells.sum().median()
```

```python
df.total_sequenced_cells.mean()
```

```python
df.filter_passing_cells.sum()
```

```python
df.columns
```

```python
# count WGD states for filtered cells
patient_wgd_counts = df.groupby('patient_id').aggregate('sum')[['0xWGD_cells', '1xWGD_cells', '2xWGD_cells', '0xWGD_divergent_cells']]
patient_wgd_props = (patient_wgd_counts.T / np.sum(patient_wgd_counts, axis = 1).values).T

patient_wgd_counts.head()
```

```python
# number of patients with >1 WGD state
((patient_wgd_counts > 0).sum(axis=1) > 1).sum()
```

```python
# number of patients with the majority state representing over 85% of cells
np.max(patient_wgd_props, axis = 1) > 0.85
```

```python
patient_wgd_counts['nondivergent_0xwgd'] = (patient_wgd_counts['0xWGD_cells'] - patient_wgd_counts['0xWGD_divergent_cells']).astype(int)
```

```python
# WGD-high patients with extant 0xWGD cells
patient_wgd_counts[(patient_wgd_counts['0xWGD_cells'] > 0) & (patient_wgd_props['1xWGD_cells'] > 0.5)]
```

```python
patient_wgd_props[(patient_wgd_counts['0xWGD_cells'] > 0) & (patient_wgd_props['1xWGD_cells'] > 0.5)]
```

```python
patient_wgd_counts[(patient_wgd_counts['1xWGD_cells'] + patient_wgd_counts['2xWGD_cells'] == 0)]
```

```python
(patient_wgd_counts['1xWGD_cells'] + patient_wgd_counts['2xWGD_cells'])
```

## specific patients mentioned

```python
df[df.patient_id == 'SPECTRUM-OV-081'][['sample_id', '0xWGD_cells']]
```

```python
patient_wgd_counts.loc['SPECTRUM-OV-045']
```

```python
patient_wgd_counts.loc['SPECTRUM-OV-006']
```

```python
patient_wgd_counts.loc['SPECTRUM-OV-081']
```

## divergent cell counts


```python
divergent_counts = df.groupby('patient_id').aggregate('sum')[['divergent_cells', 'filter_passing_cells']]
(divergent_counts.divergent_cells > 0).sum()
```

```python
# divergent cell counts
(divergent_counts.divergent_cells / divergent_counts.filter_passing_cells).mean()
```

## count non-majority nWGD cells

```python
# sbmclone anndatas
sbmclone_dir = os.path.join(pipeline_outputs, 'sbmclone') 
sbmclone_adatas = {}
for p in tqdm.tqdm(df.patient_id.unique()):
    sbmclone_adatas[p] = ad.read_h5ad(os.path.join(sbmclone_dir, f'sbmclone_{p}_snv.h5'))

```

```python
for p in sbmclone_adatas.keys():
    adata = sbmclone_adatas[p]
    combined_table = adata.obs.merge(cell_info, how='left', left_index=True, right_on='cell_id')
    combined_table['n_wgd_mode'] = combined_table.groupby('sbmclone_cluster_id')['n_wgd'].transform(lambda x: x.mode()[0])
    my_cells = combined_table[(combined_table.n_wgd == 0) &( combined_table.n_wgd_mode > 0)]
    if len(my_cells) > 0:
        print(p, my_cells[['sbmclone_cluster_id', 'n_wgd']].value_counts())
```

# how many small clones are there?

```python
clonedfs[0].columns[1:]
```

```python
['patient_id'] + list(clonedfs[0].columns[:-1])
```

```python
clonedfs = []
wgd_cols = ['0xWGD', '1xWGD', '2xWGD']
for p in sbmclone_adatas.keys():
    adata = sbmclone_adatas[p]
    clonedf = adata.obs.groupby('sbmclone_cluster_id').size().reset_index()
    clonedf['sbmclone_cluster_id'] = p + '_' + clonedf.sbmclone_cluster_id.astype(str)

    celldf = adata.obs.merge(cell_info[cell_info.include_cell], left_index=True, right_on='cell_id', how = 'inner')
    wgd_counts = celldf[['sbmclone_cluster_id', 'n_wgd']].value_counts().reset_index().pivot(index='sbmclone_cluster_id', columns='n_wgd').fillna(0).astype(int)
    wgd_counts.columns = [f'{i}xWGD' for c, i in wgd_counts.columns]
    for c in wgd_cols:
        if c not in wgd_counts.columns:
            wgd_counts[c] = 0
    wgd_counts = wgd_counts.reset_index()
    wgd_counts['sbmclone_cluster_id'] = p + '_' + wgd_counts.sbmclone_cluster_id.astype(str)
    clonedf = clonedf.merge(wgd_counts[['sbmclone_cluster_id'] + wgd_cols] , on = 'sbmclone_cluster_id')
    clonedf['patient_id'] = p

    clonedfs.append(clonedf)
clonedf = pd.concat(clonedfs)[['patient_id'] + list(clonedfs[0].columns[:-1])].rename(columns={0:'n_cells'}).reset_index(drop=True)
```

# quantify +1 WGD cells vs. SBMClone clones

```python
n_morewgd = []
clonedf['morewgd'] = 0
for _, r in clonedf.iterrows():
    majority = np.argmax(r.iloc[3:])
    n_morewgd.append(r.iloc[3 + majority + 1:].sum())
clonedf['morewgd'] = n_morewgd
```

## how many patients have any "morewgd" cells

```python
(clonedf.groupby('patient_id').morewgd.sum() > 0).sum(), len(clonedf.patient_id.unique())
```

## how many patients with >1 clone have "morewgd" cells in >1 clone

```python
pdf[pdf.morewgd > 0]
```

```python
multiclone_patients = []
multiclone_morewgd_patients = []
for p, pdf in clonedf.groupby('patient_id'):
    if len(pdf) > 1:
        multiclone_patients.append(p)
        if len(pdf[pdf.morewgd > 0]) > 1:
            multiclone_morewgd_patients.append(p)
len(multiclone_morewgd_patients), len(multiclone_patients)
```

```python
set(multiclone_patients) - set(multiclone_morewgd_patients)
```

```python
multiclone_morewgd_patients
```

# look at +1 WGD vs. sample

```python
sample_nwgd = cell_info[['patient_id', 'sample_id', 'n_wgd']].value_counts().reset_index().pivot(index=['patient_id', 'sample_id'], columns='n_wgd').fillna(0).astype(int).reset_index()
sample_nwgd.columns = [f'{a[1]}xWGD' if a[1] != '' else a[0] for a in sample_nwgd.columns.values]

n_morewgd = []
sample_nwgd['morewgd'] = 0
for _, r in sample_nwgd.iterrows():
    majority = np.argmax(r.iloc[2:])
    n_morewgd.append(r.iloc[2 + majority + 1:].sum())
sample_nwgd['morewgd'] = n_morewgd
sample_nwgd
```

```python
patients_multisite = []
patients_multisite_extrawgd = []
for p, pdf in sample_nwgd.groupby('patient_id'):
    if len(pdf) > 1:
        patients_multisite.append(p)
        if (pdf.morewgd > 0).sum() > 1:
            patients_multisite_extrawgd.append(p)
len(patients_multisite_extrawgd), len(patients_multisite)
```

# check 2xWGD cells in 025
count the number of SNVs exclusive to cluster 2

```python
adata = sbmclone_adatas['SPECTRUM-OV-025']
combined_table = adata.obs.merge(cell_info, how='left', left_index=True, right_on='cell_id')
combined_table['n_wgd_mode'] = combined_table.groupby('sbmclone_cluster_id')['n_wgd'].transform(lambda x: x.mode()[0])
```

```python
combined_table[['sbmclone_cluster_id', 'n_wgd']].value_counts()
```

```python
adata
```

```python
clone_adata = scgenome.tl.aggregate_clusters(adata, cluster_col='block_assignment', agg_layers={'alt':'sum', 'ref':'sum', 'state':'median', 'total':'sum'})
```

```python
clone_adata.obs
```

```python
exclusive = np.where(np.logical_and(np.all(clone_adata[[0, 1, 3]].layers['alt'] == 0, axis = 0), clone_adata[2].layers['alt'] > 0))[1]
len(exclusive)
```

## try looking only at the 2xWGD cellsm

```python
non2x_cells = combined_table[(combined_table.block_assignment == 2) & (combined_table.n_wgd != 2)].cell_id.values
```

```python
adata.obs['block_assignment2'] = adata.obs.block_assignment
adata.obs.loc[non2x_cells, 'block_assignment2'] = 3
clone_adata2 = scgenome.tl.aggregate_clusters(adata, cluster_col='block_assignment2', agg_layers={'alt':'sum', 'ref':'sum', 'state':'median', 'total':'sum'})
```

```python
exclusive = np.where(np.logical_and(np.all(clone_adata2[[0, 1, 3]].layers['alt'] == 0, axis = 0), clone_adata2[2].layers['alt'] > 0))[1]
len(exclusive)
```

```python

```
