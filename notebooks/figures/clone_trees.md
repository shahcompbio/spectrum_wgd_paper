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
    display_name: Python Spectrum
    language: python
    name: python_spectrum
---

```python

import os
import anndata as ad
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import Bio.Phylo
import io
import pickle
import tqdm
import itertools
import matplotlib
import yaml
import vetica.mpl


import scgenome

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import spectrumanalysis.phylo
import spectrumanalysis.cnevents
import spectrumanalysis.clonetreeplot

project_dir = os.environ['SPECTRUM_PROJECT_DIR']

colors_dict = yaml.safe_load(open('../../../../config/colors.yaml', 'r'))

```

```python

patient_info = pd.read_csv('../../../../metadata/tables/patients.tsv', sep='\t')

```

```python

patient_clone_trees = {}

for patient_id in tqdm.tqdm(patient_info['patient_id'].unique()):
    tree_filename = f'{project_dir}/tree_snv/postprocessing/{patient_id}_clones_pruned.pickle'
    branch_info_filename = f'{project_dir}/tree_snv/postprocessing/{patient_id}_branch_info.csv.gz'
    cluster_info_filename = f'{project_dir}/tree_snv/postprocessing/{patient_id}_cluster_info.csv.gz'
    cell_info_filename = f'{project_dir}/tree_snv/postprocessing/{patient_id}_cell_info.csv.gz'

    if not os.path.exists(tree_filename):
        continue

    patient_clone_trees[patient_id] = {}
    patient_clone_trees[patient_id]['tree'] = pickle.load(open(tree_filename, 'rb'))
    patient_clone_trees[patient_id]['branch_info'] = pd.read_csv(branch_info_filename).set_index('branch_segment')
    patient_clone_trees[patient_id]['cluster_info'] = pd.read_csv(cluster_info_filename, dtype={'leaf_id': str}).set_index('leaf_id')
    patient_clone_trees[patient_id]['cell_info'] = pd.read_csv(cell_info_filename, dtype={'leaf_id': str}).set_index('cell_id')

```

```python

# Annotate mean sum of mutations to leaves and number of independent WGD

patient_info['mean_sum_snv_count_per_gb'] = np.NaN
patient_info['mean_sum_snv_count_age_per_gb'] = np.NaN
patient_info['n_cells'] = 0
patient_info['num_indep_wgd'] = -1

for patient_id in patient_clone_trees:
    tree = patient_clone_trees[patient_id]['tree']
    cluster_info = patient_clone_trees[patient_id]['cluster_info']
    branch_info = patient_clone_trees[patient_id]['branch_info']

    patient_info.loc[patient_info['patient_id'] == patient_id, 'mean_sum_snv_count_per_gb'] = cluster_info['sum_snv_count_per_gb'].mean()
    patient_info.loc[patient_info['patient_id'] == patient_id, 'mean_sum_snv_count_age_per_gb'] = cluster_info['sum_snv_count_age_per_gb'].mean()
    patient_info.loc[patient_info['patient_id'] == patient_id, 'n_cells'] = cluster_info['cluster_size'].sum()
    patient_info.loc[patient_info['patient_id'] == patient_id, 'num_indep_wgd'] = branch_info['is_wgd'].sum()

```

```python

# determine patient order

# patient_info['num_indep_wgd_group'] = patient_info['num_indep_wgd'].map({0: '0 WGD', 1: '1 WGD'}).fillna('>1 WGD')
# patient_info['num_indep_wgd_group'] = pd.Categorical(
#     patient_info['num_indep_wgd_group'],
#     categories=['0 WGD', '>1 WGD', '1 WGD'], ordered=True)

subclonal_wgd_patients = [
    'SPECTRUM-OV-081',
    'SPECTRUM-OV-075',
    'SPECTRUM-OV-139',
    'SPECTRUM-OV-006',
    'SPECTRUM-OV-031',
]

wgd_group_name = 'clonal_wgd'
wgd_group_name = 'parallel_wgd'
wgd_group_name = 'subclonal_wgd'
wgd_group_name = 'nonclonal_wgd'

if wgd_group_name == 'clonal_wgd':
    patient_ids = patient_info.loc[
        (patient_info['num_indep_wgd'] == 1) &
        (~patient_info['patient_id'].isin(subclonal_wgd_patients)), 'patient_id'].values

elif wgd_group_name == 'nonclonal_wgd':
    patient_ids = patient_info.loc[
        (patient_info['num_indep_wgd'] == 0) &
        (~patient_info['patient_id'].isin(subclonal_wgd_patients)), 'patient_id'].values

elif wgd_group_name == 'parallel_wgd':
    patient_ids = patient_info.loc[
        (patient_info['num_indep_wgd'] > 1) &
        (~patient_info['patient_id'].isin(subclonal_wgd_patients)), 'patient_id'].values

elif wgd_group_name == 'subclonal_wgd':
    patient_ids = patient_info.loc[
        (patient_info['patient_id'].isin(subclonal_wgd_patients)), 'patient_id'].values

sort_order = ['mean_sum_snv_count_age_per_gb']
patient_order = patient_info[
    (patient_info['patient_id'].isin(patient_ids))
].sort_values(sort_order)['patient_id'].values

patient_order = list(reversed(patient_order))

```

```python

site_hue_order = ['Right Adnexa', 'Left Adnexa', 'Omentum', 'Bowel', 'Peritoneum', 'Other']

snv_type_suffix = ''

# Uncomment for age specific mutations
snv_type_suffix = '_age'

logscalewgd = True

height_ratios = []
for patient_id in patient_order:
    height_ratios.append(patient_clone_trees[patient_id]['tree'].count_terminals())
patient_count = len(height_ratios)

fig, axes = plt.subplots(
    nrows=patient_count, ncols=3,
    height_ratios=height_ratios,
    width_ratios=[10, 2, 2],
    sharex='col', sharey='row',
    figsize=(3, patient_count/2), dpi=300,
)

plt_idx = 0
for patient_id in patient_order:
    tree = patient_clone_trees[patient_id]['tree']
    cluster_info = patient_clone_trees[patient_id]['cluster_info']
    branch_info = patient_clone_trees[patient_id]['branch_info']
    patient_cell_info = patient_clone_trees[patient_id]['cell_info']
    plot_patient_id = patient_id.replace('SPECTRUM-', '')

    n_wgd_order = [2, 1, 0]

    patient_age = patient_info.set_index('patient_id')['patient_age_at_diagnosis'][patient_id]

    branch_info['expected_snv_count_per_gb'] = branch_info['snv_count'+snv_type_suffix] / (branch_info['opportunity']) # snv_count_age

    n_wgd_counts = (
        patient_cell_info
            .groupby(['leaf_id', 'n_wgd']).size().rename('cell_count')
            .unstack(fill_value=0).reindex(columns=[0, 1, 2], fill_value=0))
    n_wgd_fractions = (n_wgd_counts.T / n_wgd_counts.sum(axis=1)).T

    site_fractions = (
        patient_cell_info
            .groupby(['leaf_id', 'tumor_site']).size().rename('cell_count')
            .unstack(fill_value=0).reindex(columns=site_hue_order, fill_value=0))
    site_fractions = (site_fractions.T / site_fractions.sum(axis=1)).T

    ax = axes[plt_idx, 0]
    spectrumanalysis.clonetreeplot.plot_clone_tree(tree, branch_info['expected_snv_count_per_gb'], cluster_info['cluster_size'], ax=ax)
    ax.set_ylabel(f'{plot_patient_id}', rotation=150, fontsize=8, labelpad=4, ha='left', va='center')
    ax.get_legend().remove()

    # Extract location of terminal branches in plot
    cluster_info['branch_pos'] = -1
    for t in tree.get_terminals():
        cluster_info.loc[t.cluster_id, 'branch_pos'] = t.branch_pos

    # Add branch locations to additional matrices
    n_wgd_fractions['branch_pos'] = cluster_info['branch_pos']
    n_wgd_fractions = n_wgd_fractions.sort_values('branch_pos')
    site_fractions['branch_pos'] = cluster_info['branch_pos']
    site_fractions = site_fractions.sort_values('branch_pos')

    ax.spines[['right', 'top', 'left']].set_visible(False)
    ax.tick_params('y', length=0, width=0, which='major')
    if plt_idx < len(patient_order) - 1:
        ax.spines.bottom.set_visible(False)
        ax.tick_params('x', length=0, width=0, which='major')
    ax.tick_params(axis='y', labelright=False)

    # Subclonal WGD
    ax = axes[plt_idx, 1]
    left_accum = None
    for n_wgd in n_wgd_order:
        # Rasterized to avoid svg issues with log scale
        ax.barh(width=n_wgd_fractions[n_wgd], y=n_wgd_fractions['branch_pos'], left=left_accum, color=colors_dict['wgd_multiplicity'][n_wgd], rasterized=True)
        if left_accum is None:
            left_accum = n_wgd_fractions[n_wgd]
        else:
            left_accum += n_wgd_fractions[n_wgd]
    ax.set_xlim((0.001, 1.))
    ax.set_xscale('log')
    ax.invert_xaxis()
    ax.spines.left.set_visible(False)
    ax.spines.top.set_visible(False)
    ax.set_ylabel('')
    ax.set_yticklabels([])
    ax.tick_params('y', length=0, width=0, which='minor')
    ax.tick_params('y', length=0, width=0, which='major')

    if plt_idx < len(patient_order) - 1:
        ax.spines.bottom.set_visible(False)
        ax.tick_params('x', length=0, width=0, which='major')
        ax.tick_params('x', length=0, width=0, which='minor')
        ax.set_xlabel('')

    # Sample counts
    ax = axes[plt_idx, 2]
    left_accum = None
    for site_name in site_hue_order:
        ax.barh(width=site_fractions[site_name], y=site_fractions['branch_pos'], left=left_accum, color=colors_dict['tumor_site'][site_name])
        if left_accum is None:
            left_accum = site_fractions[site_name]
        else:
            left_accum += site_fractions[site_name]
    ax.set_yticklabels(site_fractions.index, minor=False, rotation=90, ha='center', va='center', fontsize=6)
    ax.yaxis.tick_right()
    ax.yaxis.set_ticks_position('right')
    ax.tick_params('y', which='major', right=True, pad=5)
    ax.set_xlim(0, 1)
    ax.invert_xaxis()
    ax.spines.left.set_visible(False)
    ax.spines.top.set_visible(False)
    ax.set_ylabel('')

    if plt_idx < len(patient_order) - 1:
        ax.spines.bottom.set_visible(False)
        ax.tick_params('x', length=0, width=0, which='major')
        ax.tick_params('x', length=0, width=0, which='minor')
        ax.set_xlabel('')

    plt_idx += 1

ax = axes[-1, 0]
if snv_type_suffix == '':
    ax.set_xlabel('Mut / GB', fontsize=8, labelpad=10, rotation=90)
else:
    ax.set_xlabel('C>T CpG\nMut / GB', fontsize=8, labelpad=10, rotation=90)
ax.axis('on')
ax.tick_params('x', labelsize=8, rotation=90, which='major')
ax.spines.bottom.set_visible(True)
ax.spines.bottom.set_position(('outward', 10))
ax.spines.bottom.set_bounds(0, ax.get_xlim()[1])

ax = axes[-1, 1]
ax.spines.bottom.set_position(('outward', 10))
ax.tick_params('x', labelsize=8, rotation=90, which='major')
ax.set_xlabel('Fraction\n#WGD', fontsize=8, labelpad=10, rotation=90)

ax = axes[-1, 2]
ax.spines.bottom.set_position(('outward', 10))
ax.tick_params('x', labelsize=8, rotation=90, which='major')
ax.set_xlabel('Fraction\nCells', fontsize=8, labelpad=10, rotation=90)

xlim = axes[0, 0].get_xlim()
plt_idx = 0
for patient_id in patient_order:
    ax = axes[plt_idx, 0]
    cluster_info = patient_clone_trees[patient_id]['cluster_info']
    for idx, row in cluster_info.iterrows():
        ax.plot(
            [row['sum_snv_count'+snv_type_suffix+'_per_gb'], xlim[1]],
            [row['branch_pos'], row['branch_pos']],
            color='0.8', ls=':', lw=0.5, zorder=-100)
    plt_idx += 1
axes[0, 0].set_xlim(xlim)

fig.subplots_adjust(wspace=.1)

fig.savefig(f'../../../../figures/figure2/snv_tree_{wgd_group_name}.svg', bbox_inches='tight', metadata={'Date': None})

```

```python

```
