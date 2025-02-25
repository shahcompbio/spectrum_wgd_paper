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

import scgenome

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import spectrumanalysis.phylo
import spectrumanalysis.cnevents
import spectrumanalysis.dataload

project_dir = os.environ['SPECTRUM_PROJECT_DIR']

colors_dict = yaml.safe_load(open('../../../../config/colors.yaml', 'r'))

patient_info = pd.read_csv('../../../../metadata/tables/patients.tsv', sep='\t')

cell_info = pd.read_csv(os.path.join(project_dir, f'preprocessing/summary/filtered_cell_table.csv.gz'))
cell_info = cell_info[cell_info['include_cell']]

cell_info = spectrumanalysis.wgd.classify_subclonal_wgd(cell_info)

```

```python

patient_clone_trees = {}

for patient_id in tqdm.tqdm(cell_info['patient_id'].unique()):
    try:
        patient_clone_trees[patient_id] = spectrumanalysis.dataload.load_clone_tree_data(patient_id, project_dir)
    except spectrumanalysis.dataload.MissingDataError:
        pass

```

```python

def get_root_branches(clade):
    yield clade.name
    if len(clade.clades) == 1:
        for name in get_root_branches(clade.clades[0]):
            yield name

cohort_cluster_info = []
cohort_branch_info = []
cohort_sample_info = []

for patient_id in list(patient_clone_trees.keys()):
    tree = patient_clone_trees[patient_id]['tree']
    cluster_info = patient_clone_trees[patient_id]['cluster_info']
    branch_info = patient_clone_trees[patient_id]['branch_info']
    adata = patient_clone_trees[patient_id]['adata']

    cluster_info['age'] = patient_info.set_index('patient_id').loc[patient_id, 'patient_age_at_diagnosis']

    for snv_type_suffix in ('', '_age'):
        branch_info[f'snv_count{snv_type_suffix}_per_gb'] = branch_info[f'snv_count{snv_type_suffix}'] / (branch_info['opportunity'])

        # Root branch
        cluster_info[f'snv_count_root{snv_type_suffix}_per_gb'] = branch_info.loc[list(get_root_branches(tree.clade)), f'snv_count{snv_type_suffix}_per_gb'].sum()

        # SNVs
        cluster_info[f'snv_count{snv_type_suffix}_per_gb_since_birth'] = 0
        for leaf in tree.get_terminals():
            for clade in tree.get_path(leaf.name):
                cluster_info.loc[leaf.cluster_id, f'snv_count{snv_type_suffix}_per_gb_since_birth'] += branch_info.loc[clade.name, f'snv_count{snv_type_suffix}_per_gb']
            cluster_info.loc[leaf.cluster_id, f'snv_count{snv_type_suffix}_per_gb_since_birth'] += branch_info.loc[tree.clade.name, f'snv_count{snv_type_suffix}_per_gb']

        # SNVs since WGD
        cluster_info[f'snv_count{snv_type_suffix}_per_gb_since_wgd'] = 0
        cluster_info['is_wgd'] = False
        for clade in tree.find_clades():
            if not clade.is_wgd:
                continue
            for leaf in clade.get_terminals():
                cluster_info.loc[leaf.cluster_id, 'is_wgd'] = True
                for subclade in clade.get_path(leaf.name):
                    cluster_info.loc[leaf.cluster_id, f'snv_count{snv_type_suffix}_per_gb_since_wgd'] += branch_info.loc[subclade.name, f'snv_count{snv_type_suffix}_per_gb']

        # SNVs prior to WGD
        cluster_info[f'snv_count{snv_type_suffix}_per_gb_to_wgd'] = 0
        for clade in tree.find_clades():
            if not clade.is_wgd:
                continue
            snv_count_to_wgd = 0
            for clade2 in tree.get_path(clade.name):
                snv_count_to_wgd += branch_info.loc[clade2.name, f'snv_count{snv_type_suffix}_per_gb']
            # Important! get_path does not include the root clade
            snv_count_to_wgd += branch_info.loc[tree.clade.name, f'snv_count{snv_type_suffix}_per_gb'] 
            for leaf in clade.get_terminals():
                cluster_info.loc[leaf.cluster_id, f'snv_count{snv_type_suffix}_per_gb_to_wgd'] = snv_count_to_wgd

    sample_info = (
        adata.obs
            .groupby(['leaf_id', 'sample_id']).size()
            .rename('n_cells').reset_index(level=1))
    sample_info = sample_info.merge(cluster_info['cluster_size'], left_index=True, right_index=True, how='right')
    # assert (sample_info.groupby(level=0)['n_cells'].transform('sum') == sample_info['cluster_size']).all()

    cohort_cluster_info.append(cluster_info.assign(patient_id=patient_id))
    cohort_branch_info.append(branch_info.assign(patient_id=patient_id))
    cohort_sample_info.append(sample_info.assign(patient_id=patient_id))

cohort_cluster_info = pd.concat(cohort_cluster_info).reset_index()
cohort_branch_info = pd.concat(cohort_branch_info).reset_index()
cohort_sample_info = pd.concat(cohort_sample_info).reset_index()

```

```python

cohort_cluster_info.to_csv('../../../../results/tables/snv_tree/snv_leaf_table.csv', index=False)
cohort_branch_info.to_csv('../../../../results/tables/snv_tree/snv_branch_table.csv', index=False)
cohort_sample_info.to_csv('../../../../results/tables/snv_tree/snv_sample_table.csv', index=False)

```

```python

```
