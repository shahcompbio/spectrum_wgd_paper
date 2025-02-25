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

import scgenome
import os
import yaml
import anndata as ad
import numpy as np
import pandas as pd
import Bio.Phylo
import io
import pickle
import vetica.mpl

import seaborn as sns
import matplotlib.pyplot as plt

import spectrumanalysis.plots

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

project_dir = os.environ['SPECTRUM_PROJECT_DIR']

```

```python

cell_info = pd.read_csv(f'{project_dir}/preprocessing/summary/filtered_cell_table.csv.gz')
cell_info = cell_info[cell_info['include_cell']]

colors_dict = yaml.safe_load(open('../../../../config/colors.yaml', 'r'))

```

```python

patient_id = 'SPECTRUM-OV-025'

filename = f'{project_dir}/sbmclone/sbmclone_{patient_id}_snv.h5'
adata = ad.read(filename)
adata = adata[adata.obs.index.isin(cell_info['cell_id'].values)]
adata.obs['n_wgd'] = cell_info.set_index('cell_id')['n_wgd']
assert not adata.obs['n_wgd'].isnull().any()

```

```python

adata.layers['is_present'] = (adata.layers['alt'] > 0) * 1

block_means = (adata.to_df('is_present')
    .set_index(adata.obs['sbmclone_cluster_id'], append=True)
    .groupby(level=1).mean().T
    .set_index(adata.var['block_assignment'], append=True)
    .groupby(level=1).mean().T)

adata.layers['block_means'] = block_means.loc[adata.obs['sbmclone_cluster_id'].values, adata.var['block_assignment'].values]

```

```python

pd.Series(np.array(adata.layers['block_means']).flatten()).hist()

adata.layers['block_present'] = adata.layers['block_means'] > 0.02

```

```python

# Remove absent everywhere

adata.var['n_block_present'] = adata.layers['block_present'].sum(axis=0)
adata = adata[:, adata.var['n_block_present'] > 0]

```

```python

adata.var.groupby('block_assignment').size()

```

```python

adata.obs.groupby(['sbmclone_cluster_id', 'n_wgd']).size()

```

```python

# Downsample cells
# adata = adata[adata.obs.sample(frac=0.05).index, adata.var.sample(frac=0.05).index]

adata = adata[
    adata.obs.sort_values(['sbmclone_cluster_id', 'n_wgd'], ascending=(True, True)).index,
    adata.var.sort_values('block_assignment').index]

```

```python

import matplotlib
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

cluster_ids = sorted(adata.obs['sbmclone_cluster_id'].unique())
cluster_colors = dict(zip(cluster_ids, colors_dict['clone_id'].values()))

adata_plot = adata
height = 3

fig, axes = plt.subplots(
    nrows=1, ncols=3, width_ratios=[40, 1, 1],
    figsize=(4, 3), dpi=300, sharex=False)

fig_cbar, bar_ax = plt.subplots(figsize=(0.2, 1))

ax = axes[0]
sns.heatmap(
    adata_plot.to_df('block_means'),
    cmap='Greys',
    vmin=-0.01,
    ax=ax,
    xticklabels=False,
    yticklabels=False,
    cbar=True,
    cbar_ax=bar_ax,
    cbar_kws=dict(use_gridspec=False, ticks=[0, 0.05, 0.1], location='right'),
    rasterized=True,
)
bar_ax.set_title('Fraction SNVs\ndetected', fontsize=6)
bar_ax.tick_params(axis='y', which='major', labelsize=6)
bar_ax.set_ylim((0, bar_ax.get_ylim()[1]))
for loc in ['bottom', 'top', 'right', 'left']:
    ax.spines[loc].set_visible(False)
    # ax.spines[loc].set_linewidth(0.25)
ax.set_ylabel('Cell', rotation=0, ha='right')
ax.set_xlabel('SNV')

clone_boundaries = np.where(adata_plot.obs['sbmclone_cluster_id'].diff() != 0)[0]
for b in clone_boundaries[:]:
    ax.axhline(b, linewidth=0.5, color='w')

block_boundaries = np.where(adata_plot.var['block_assignment'].diff() != 0)[0]
for b in block_boundaries[1:]:
    ax.axvline(b, linewidth=0.5, color='w')

ax = axes[1]
scgenome.plotting.heatmap._plot_categorical_annotation(
    adata_plot.obs[['sbmclone_cluster_id']].copy().values,
    ax=ax,
    ax_legend=ax,
    title='',
    horizontal=False,
    cmap=cluster_colors,
)
ax.minorticks_off()
ax.tick_params(axis='x', which='major', bottom=False)
ax.set_yticks([])
ax.set_xticks([0])
ax.set_xticklabels(['Cell cluster'])
for loc in ['bottom', 'top', 'right', 'left']:
    ax.spines[loc].set_visible(False)
sns.move_legend(ax, loc='lower left', bbox_to_anchor=(5., 0.75), fontsize=6, title='SBMClone\nCell cluster', ncols=1, markerscale=0.2, frameon=False)

ax = axes[2]
scgenome.plotting.heatmap._plot_categorical_annotation(
    adata_plot.obs[['n_wgd']].copy().values,
    ax=ax,
    ax_legend=ax,
    title='',
    horizontal=False,
    cmap=colors_dict['wgd_multiplicity'],
)
ax.minorticks_off()
ax.tick_params(axis='x', which='major', bottom=False)
ax.set_yticks([])
ax.set_xticks([0])
ax.set_xticklabels(['WGD'])
for loc in ['bottom', 'top', 'right', 'left']:
    ax.spines[loc].set_visible(False)
sns.move_legend(ax, loc='lower left', bbox_to_anchor=(4.5, 0.5), fontsize=6, title='WGD', ncols=1, markerscale=0.2, frameon=False)

plt.subplots_adjust(wspace=0.05, hspace=0.05)

fig.savefig(f'../../../../figures/edfigure3/sbmclone_{patient_id}.svg', bbox_inches='tight', metadata={'Date': None})

fig_cbar.savefig(f'../../../../figures/edfigure3/sbmclone_{patient_id}_cbar.svg', bbox_inches='tight', metadata={'Date': None})

```

```python

import spectrumanalysis.plots

spectrumanalysis.plots.plot_feature_colors_legends({'#WGD': colors_dict['wgd_multiplicity']})

```

```python

plot_data = adata_plot.obs.groupby(['sbmclone_cluster_id', 'n_wgd']).size().unstack()
plot_data = plot_data.reindex(columns=colors_dict['wgd_multiplicity'].keys())
plot_data = (plot_data.T / plot_data.sum(axis=1)).T

fig, ax = plt.subplots(figsize=(2, 1.5))
ax = plot_data.plot.bar(ax=ax, stacked=True, color=colors_dict['wgd_multiplicity'])
ax.set_xlabel('Clone')
ax.set_ylabel('Fraction')
ax.get_legend().remove()

spectrumanalysis.plots.style_barplot(ax)

```

```python

```
