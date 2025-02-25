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

import anndata as ad
import pandas as pd
import scgenome
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import numpy as np
import yaml
import vetica.mpl

import spectrumanalysis.dataload
import spectrumanalysis.plots

project_dir = os.environ['SPECTRUM_PROJECT_DIR']

colors_dict = yaml.safe_load(open('../../../../config/colors.yaml', 'r'))

cell_info = pd.read_csv(f'{project_dir}/preprocessing/summary/filtered_cell_table.csv.gz')
cell_info = cell_info[cell_info['include_cell']]

patient_id = 'SPECTRUM-OV-081'
adata = spectrumanalysis.dataload.load_filtered_cna_adata(project_dir, patient_id)

```

```python

plot_data = cell_info.query(f'patient_id == "{patient_id}"')

f, ax = plt.subplots(figsize=(1, 3), dpi=300)
# ax.set_yscale("log")

sns.boxplot(
    x='n_wgd', y='nnd', hue='n_wgd',
    dodge=False, data=plot_data, order=[0, 1], fliersize=0,
    palette=colors_dict['wgd_multiplicity'])
sns.stripplot(
    x='n_wgd', y='nnd', hue='n_wgd',
    dodge=False, data=plot_data, order=[0, 1], size=3,
    linewidth=0.5, palette=colors_dict['wgd_multiplicity'])
ax.get_legend().remove()
ax.set_xlabel('# WGD')
ax.set_ylabel('Fraction genome different\ncompared with most similar cell')
sns.despine()

```

```python

plot_data = cell_info.query(f'patient_id == "{patient_id}"')

f, ax = plt.subplots(figsize=(3, 3), dpi=300)

sns.scatterplot(
    ax=ax, y='nnd', x='ploidy', hue='n_wgd', data=plot_data,
    s=10, linewidth=0.5, edgecolor='k', palette=colors_dict['wgd_multiplicity'])
sns.despine(trim=True)
ax.set_ylabel('Fraction genome different\ncompared with most similar cell')
ax.set_xlabel('Ploidy')
sns.move_legend(
    ax, 'upper left', prop={'size': 8}, markerscale=2, bbox_to_anchor=(0.8, 0.7),
    labelspacing=0.4, handletextpad=0, columnspacing=0.5,
    ncol=1, title='#WGD', title_fontsize=10, frameon=False)

```

```python

n_wgds = cell_info.query(f'patient_id == "{patient_id}"')['n_wgd'].value_counts()
n_wgds / n_wgds.sum()

```

```python

plot_data = cell_info.query(f'patient_id == "{patient_id}"').copy()

plot_data['label'] = 'nWGD'
plot_data.loc[plot_data['n_wgd'] == 1, 'label'] = 'WGD'
plot_data.loc[plot_data['multipolar'], 'label'] = 'Divergent'

fig, ax = plt.subplots(figsize=(3, 3), dpi=300)

sns.scatterplot(
    ax=ax, y='nnd', x='ploidy', hue='label', hue_order=['nWGD', 'WGD', 'Divergent'], data=plot_data,
    s=10, linewidth=0.5, edgecolor='k')
sns.despine(trim=True)
ax.set_ylabel('Fraction genome different\ncompared with most similar cell')
ax.set_xlabel('Ploidy')
ax.set_title('OV-081')
sns.move_legend(
    ax, 'upper left', prop={'size': 8}, markerscale=2, bbox_to_anchor=(0.8, 0.7),
    labelspacing=0.4, handletextpad=0, columnspacing=0.5,
    ncol=1, title='Divergent', title_fontsize=10, frameon=False)

fig.savefig(f'../../../../figures/edfigure4/ploidy_nnd_{patient_id}.svg', bbox_inches='tight', metadata={'Date': None})

```

```python

import scgenome
np.random.seed(10)

diploid = plot_data.query('is_wgd == 0').sample(20)['cell_id'].values

fig, axes = plt.subplots(
    nrows=3, ncols=1, height_ratios=[1, 1, 1], figsize=(4, 2), dpi=300, sharex=True)

g = scgenome.pl.plot_cell_cn_matrix(
    adata[diploid],
    ax=axes[0],
    cell_order_fields=['cell_order'],
    style='white',
)
axes[0].set_xlabel('')
axes[0].set_ylabel('non-WGD', rotation=0, ha='right', va='center', fontsize=8)
axes[0].set_title('OV-081', fontsize=10)

stable_cells = plot_data.query(f'is_wgd == 1 and nnd < 0.3').sample(20)['cell_id'].values

g = scgenome.pl.plot_cell_cn_matrix(
    adata[stable_cells],
    ax=axes[1],
    cell_order_fields=['cell_order'],
    style='white',
)
axes[1].set_xlabel('')
axes[1].set_ylabel('WGD', rotation=0, ha='right', va='center', fontsize=8)

unstable_cells = plot_data.query(f'is_wgd == 1 and nnd > 0.3').sample(20)['cell_id'].values

g = scgenome.pl.plot_cell_cn_matrix(
    adata[unstable_cells],
    ax=axes[2],
    cell_order_fields=['cell_order'],
    style='white',
)
axes[2].set_ylabel('Divergent', rotation=0, ha='right', va='center', fontsize=8)

plt.subplots_adjust(hspace=0.15)

spectrumanalysis.plots.remove_xticklabels(axes[2], ['14', '16', '18', '20', '22'])

fig.savefig(f'../../../../figures/edfigure4/nwgd_wgd_divergent_{patient_id}.svg', bbox_inches='tight', metadata={'Date': None})

```

```python

```

```python

```
