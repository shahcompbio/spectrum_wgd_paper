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
import scgenome
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import numpy as np
import tqdm
import vetica.mpl
import os

import spectrumanalysis.cnmetrics

project_dir = os.environ['SPECTRUM_PROJECT_DIR']

patient_id = 'SPECTRUM-OV-081'
version = 'v5'

adata = ad.read(os.path.join(project_dir, f'pipeline_outputs/v5/postprocessing/aggregated_anndatas/{patient_id}_cna.h5'))

adata = adata[
    (adata.obs['is_normal'] == False) &
    (adata.obs['is_aberrant_normal_cell'] == False) &
    (adata.obs['is_doublet'] == 'No') &
    (adata.obs['is_s_phase_thresholds'] == False) &
    (adata.obs['multiplet'] == False) &
    (adata.obs['multipolar'] == False)].copy()

# No n_WGD = 2 for this analysis
adata = adata[adata.obs['n_wgd'] <= 1]

noisy_cells = pd.read_csv(os.path.join(project_dir, f'analyses/check_wgd/all_cells_obs.csv'))
noisy_cells = noisy_cells.query('longest_135_segment >= 20')
adata = adata[adata.obs['brief_cell_id'].isin(noisy_cells['brief_cell_id'].values)]

```

```python

cluster_label = 'sbmclone_cluster_id'

adata.obs['cluster__n_wgd'] = adata.obs[cluster_label].astype(str) + '/' + adata.obs['n_wgd'].astype(str)

agg_X = np.sum

agg_layers = {
    'copy': np.nanmean,
    'state': np.nanmedian,
    'alleleA': np.nansum,
    'alleleB': np.nansum,
    'totalcounts': np.nansum,
    'Min': np.nanmedian,
    'Maj': np.nanmedian,
}

agg_obs = {
    'n_wgd': lambda a: np.nanmean(a.astype(float)),
}

adata_clusters = scgenome.tl.aggregate_clusters(adata, agg_X, agg_layers, agg_obs, cluster_col='cluster__n_wgd')
adata_clusters.layers['state'] = adata_clusters.layers['state'].round()
adata_clusters.layers['Min'] = adata_clusters.layers['Min'].round()
adata_clusters.layers['Maj'] = adata_clusters.layers['Maj'].round()
adata_clusters.layers['BAF'] = adata_clusters.layers['alleleB'] / adata_clusters.layers['totalcounts']
adata_clusters.layers['minor'] = np.minimum(adata_clusters.layers['Min'], adata_clusters.layers['Maj'])
adata_clusters.obs['n_wgd'] = adata_clusters.obs['n_wgd'].round().astype(int)

adata_clusters.obs

```

```python

adata_snvs = ad.read(os.path.join(project_dir, f'pipeline_outputs/v5/postprocessing/aggregated_anndatas/{patient_id}_snv.h5'))
snv_data = (
    adata_snvs
        .to_df('alt').set_index(adata_snvs.obs['sbmclone_cluster_id'], append=True)
        .T.set_index(adata_snvs.var['block_assignment'], append=True).T)
snv_matrix = snv_data.groupby(level=1).mean().T.groupby(level=1).mean()

(snv_matrix > 0.025) * 1

```

```python

adata_snvs.var.groupby('block_assignment').size()

```

```python

adata.obs.groupby(['sample_id', 'n_wgd']).size()

```

```python

adata_wgd = adata[adata.obs.query('is_wgd == 1').index].copy()

layers = ['state', 'copy', 'Min', 'Maj']

for layer in layers:
    adata_wgd.layers[f'wgd_diff_{layer}'] = np.empty(adata_wgd.layers[layer].shape)
    adata_wgd.layers[f'wgd_diff_{layer}'][:] = np.NaN

for cell_id, cluster_id in adata_wgd.obs[cluster_label].items():
    cell_idx = adata_wgd.obs.index.get_loc(cell_id)
    for layer in layers:
        cell_diff = adata_wgd[cell_id].layers[layer][0] - 2 * adata_clusters[f'{cluster_id}/0'].layers[layer][0]
        adata_wgd.layers[f'wgd_diff_{layer}'][cell_idx, :] = cell_diff

adata_wgd.layers['wgd_diff_imbalance'] = np.clip(adata_wgd.layers['wgd_diff_Maj'] - adata_wgd.layers['wgd_diff_Min'], -2, 2)

```

```python

cluster_ids = adata.obs[cluster_label].unique()

cmap = plt.cm.tab10_r
cluster_colors = cmap(np.linspace(0, 1, len(cluster_ids)))
cluster_colors = dict(zip(cluster_ids, cluster_colors))

chromosomes = adata_wgd.var['chr'].unique()
# chromosomes = ['6', '11', '15', '17']
chromosomes = ['11', '6', '17']

adata_wgd_plot = adata_wgd[(adata_wgd.obs['is_wgd'] == 1),  (adata_wgd.var['has_allele_cn']) & (adata_wgd.var['chr'].isin(chromosomes))].copy()

# Sort on a subset of chromosomes for aesthetics
adata_wgd_plot_sorted = scgenome.tl.sort_cells(adata_wgd_plot[:, (adata_wgd_plot.var['chr'].isin(['6', '17']))], layer_name='state')
adata_wgd_plot.obs['cell_order'] = adata_wgd_plot_sorted.obs['cell_order']

adata_nwgd_plot = adata[(adata.obs['is_wgd'] == 0), (adata.var['has_allele_cn']) & (adata.var['chr'].isin(chromosomes))]

# Sort on a subset of chromosomes for aesthetics
adata_nwgd_plot_sorted = scgenome.tl.sort_cells(adata_nwgd_plot[:, (adata_nwgd_plot.var['chr'].isin(chromosomes))], layer_name='state')
adata_nwgd_plot.obs['cell_order'] = adata_nwgd_plot_sorted.obs['cell_order']

nwgd_cells = adata_nwgd_plot.shape[0] / 2
wgd_cells = adata_wgd_plot.shape[0] / 2


fig, axes = plt.subplots(
    nrows=4, ncols=2,
    height_ratios=[10, nwgd_cells, wgd_cells, wgd_cells],
    width_ratios=[20, 1],
    sharex='col', sharey='row',
    figsize=(2.5, 4), dpi=150)

axes[0, 1].axis('off')

ax = axes[0, 0]
scgenome.plotting.heatmap._plot_categorical_annotation(
    adata_wgd_plot.var[['cyto_band_giemsa_stain']].copy().values.T,
    ax=ax,
    ax_legend=None,
    title='',
    horizontal=True,
    cmap=scgenome.plotting.heatmap.cyto_band_giemsa_stain_colors)
ax.minorticks_off()
ax.tick_params(axis='x', which='major', bottom=False)
ax.set_yticks([])
ax.spines[:].set_linewidth(0.5)

ax = axes[1, 0]
g = scgenome.pl.plot_cell_cn_matrix(
    adata_nwgd_plot[adata_nwgd_plot.obs[cluster_label].isin(cluster_ids)],
    layer_name='state',
    ax=ax,
    cell_order_fields=[cluster_label, 'cell_order'],
    style='white',
)
ax.set_ylabel('nWGD cells\nCN', size=6, rotation=0, ha='right', va='center')
ax.tick_params(axis='x', which='major', bottom=False)

ax = axes[1, 1]
scgenome.plotting.heatmap._plot_categorical_annotation(
    adata_nwgd_plot[g['adata'].obs.index].obs[[cluster_label]].copy().values,
    ax=ax,
    ax_legend=None,
    title='',
    horizontal=False,
    cmap=cluster_colors)
ax.minorticks_off()
ax.tick_params(axis='x', which='major', bottom=False)
ax.set_yticks([])
ax.spines[:].set_visible(False)

ax = axes[2, 0]
g = scgenome.pl.plot_cell_cn_matrix(
    adata_wgd_plot[adata_wgd_plot.obs[cluster_label].isin(cluster_ids)],
    layer_name='state',
    ax=ax,
    cell_order_fields=[cluster_label, 'cell_order'],
    style='white',
)
ax.set_ylabel('WGD cells\nCN', size=6, rotation=0, ha='right', va='center')
ax.tick_params(axis='x', which='major', bottom=False)

ax = axes[2, 1]
scgenome.plotting.heatmap._plot_categorical_annotation(
    adata_wgd_plot[g['adata'].obs.index].obs[[cluster_label]].copy().values,
    ax=ax,
    ax_legend=None,
    title='',
    horizontal=False,
    cmap=cluster_colors)
ax.minorticks_off()
ax.tick_params(axis='x', which='major', bottom=False)
ax.set_yticks([])
ax.spines[:].set_visible(False)

ax = axes[3, 0]
g = scgenome.pl.plot_cell_cn_matrix(
    adata_wgd_plot[adata_wgd_plot.obs[cluster_label].isin(cluster_ids)],
    layer_name='wgd_diff_state',
    ax=ax,
    raw=True,
    cmap='coolwarm',
    vmin=-2,
    vmax=2,
    cell_order_fields=[cluster_label, 'cell_order'],
    style='white',
)
ax.set_ylabel('WGD cells\npost-WGD ∆CN', size=6, rotation=0, ha='right', va='center')

ax = axes[3, 1]
scgenome.plotting.heatmap._plot_categorical_annotation(
    adata_wgd_plot[g['adata'].obs.index].obs[[cluster_label]].copy().values,
    ax=ax,
    ax_legend=None,
    title='',
    horizontal=False,
    cmap=cluster_colors)
ax.set_xticklabels(['Clone'])
ax.spines[:].set_visible(False)

plt.subplots_adjust(hspace=0.051, wspace=0.051)

```

```python

```
