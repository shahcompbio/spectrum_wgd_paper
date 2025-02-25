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
import tqdm
import yaml
import numpy as np
import pandas as pd
import anndata as ad
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib

import vetica.mpl

import spectrumanalysis.wgd

colors_dict = yaml.safe_load(open('../../../../config/colors.yaml', 'r'))

```

```python

project_dir = os.environ['SPECTRUM_PROJECT_DIR']

cell_info = pd.read_csv(f'{project_dir}/preprocessing/summary/filtered_cell_table.csv.gz')

scdna_metadata = pd.read_csv('../../../../metadata/tables/sequencing_scdna.tsv', sep='\t')
scdna_metadata['sample_id'] = scdna_metadata['spectrum_sample_id']
cell_info = cell_info.merge(scdna_metadata[['sample_id', 'tumor_site']].drop_duplicates(), how='left')

```

```python

# Fraction WGD
fraction_wgd = pd.read_csv('../../../../annotations/fraction_wgd_class.csv')
fraction_wgd.groupby('wgd_class').size(), (fraction_wgd['wgd_class'] == 'WGD-high').mean()

```

```python

# Number of patients
len(cell_info['patient_id'].unique())

```

```python

# Number of cells
cell_info.shape[0]

```

```python

# Median cells per patient
cell_info.groupby('patient_id').size().median()

```

```python

# Median coverage breadth of all cells
cell_info['coverage_breadth'].median()

```

```python

plot_data = cell_info.groupby(['patient_id', 'tumor_site']).size().unstack(fill_value=0)
plot_data.index = plot_data.index.str.replace('SPECTRUM-', '')

site_hue_order = ['Right Adnexa', 'Left Adnexa', 'Omentum', 'Bowel', 'Peritoneum', 'Other']

fig = plt.figure(figsize=(8, 2), dpi=150)
ax = plt.gca()

lower_accum = None
for site_name in site_hue_order:
    ax.bar(height=plot_data[site_name], x=plot_data.index, bottom=lower_accum, color=colors_dict['tumor_site'][site_name])
    if lower_accum is None:
        lower_accum = plot_data[site_name]
    else:
        lower_accum += plot_data[site_name]

sns.despine(ax=ax)
ax.tick_params(axis='x', rotation=90)
ax.set_xlabel('Patient ID')
ax.set_xlim((-0.75, plot_data.shape[0]))
ax.set_ylabel('# Cells')
ax.set_xlabel('Patient ID')

fig.savefig(f'../../../../figures/edfigure2/cell_counts.svg', bbox_inches='tight', metadata={'Date': None})


```

```python

plot_data = cell_info.copy()
plot_data['patient_id'] = plot_data['patient_id'].str.replace('SPECTRUM-', '')

fig = plt.figure(figsize=(8, 2), dpi=150)
ax = sns.boxplot(x='patient_id', y='coverage_depth', data=plot_data, linewidth=1, fliersize=1, color='k', fill=False)
plt.setp(ax.lines, color='k')
sns.despine(ax=ax)
ax.tick_params(axis='x', rotation=90)
ax.set_xlabel('Patient ID')
ax.set_ylabel('Coverage Depth')

fig.savefig(f'../../../../figures/edfigure2/coverage_depths.svg', bbox_inches='tight', metadata={'Date': None})

```

```python

cell_info['cell_qc'] = 'Tumor'
cell_info.loc[(cell_info['is_normal'] == True) | (cell_info['is_aberrant_normal_cell'] == True), 'cell_qc'] = 'Non-tumor'
cell_info.loc[(cell_info['is_s_phase_thresholds'] == True), 'cell_qc'] = 'S-phase'
cell_info.loc[(cell_info['is_doublet'] != 'No'), 'cell_qc'] = 'Doublet'

plot_data = cell_info.groupby(['patient_id', 'cell_qc']).size().unstack(fill_value=0)
plot_data = plot_data[['Doublet', 'S-phase', 'Non-tumor', 'Tumor']]
plot_data = (plot_data.T / plot_data.sum(axis=1)).T
plot_data.index = plot_data.index.str.replace('SPECTRUM-', '')

fig = plt.figure(figsize=(8, 2), dpi=150)
ax = plt.gca()
plot_data.plot.bar(ax=ax, stacked=True, width=0.8, color=matplotlib.colormaps['Accent'].colors[-6:])
sns.despine(ax=ax)
ax.tick_params(axis='x', rotation=90)
ax.set_xlabel('Patient ID')
sns.move_legend(ax, loc='upper left', bbox_to_anchor=(1., 1.), title='Cell QC', markerscale=2, frameon=False)

fig.savefig(f'../../../../figures/edfigure2/fraction_qc.svg', bbox_inches='tight', metadata={'Date': None})

```

```python

# Total number of cancer cells

(cell_info['cell_qc'] == 'Tumor').sum()

```

```python

scrna = pd.read_csv(os.path.join(project_dir, 'analyses/scrna/merged_all_cells.csv.gz'))

```

```python

# Fraction of patients with scRNA
len(set(cell_info['patient_id'].unique()).intersection(set(scrna['patient_id'].unique()))) / len(set(cell_info['patient_id'].unique()))

```

```python

# Fraction of patients with IF

mn_rate_data = pd.read_csv(os.path.join(project_dir, 'analyses/if/mn_rates/ignacio_mn_rates.csv'))

len(set(cell_info['patient_id'].unique()).intersection(set(mn_rate_data['patient_id'].unique()))) / len(set(cell_info['patient_id'].unique()))

```

```python

n_wgd_counts = cell_info[cell_info['cell_qc'] == 'Tumor'].groupby('n_wgd').size()
n_wgd_counts / n_wgd_counts.sum()

```

```python

snv_leaf_table = pd.read_csv('../../../../results/tables/snv_tree/snv_leaf_table.csv')
snv_leaf_table['relative_wgd_age'] = snv_leaf_table['snv_count_age_per_gb_to_wgd'] / snv_leaf_table['snv_count_root_age_per_gb']

wgds_table = snv_leaf_table.query('is_wgd')[['patient_id', 'relative_wgd_age']]
wgds_table.query('relative_wgd_age > 0.5')['patient_id'].unique()

```

```python

```
