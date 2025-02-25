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

import yaml
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
import spectrumanalysis.dataload
import spectrumanalysis.utils
import spectrumanalysis.plots


project_dir = os.environ['SPECTRUM_PROJECT_DIR']

patient_id = 'SPECTRUM-OV-139'

adata = spectrumanalysis.dataload.load_filtered_cna_adata(project_dir, patient_id)

cell_info = pd.read_csv(f'{project_dir}/preprocessing/summary/filtered_cell_table.csv.gz')
cell_info = cell_info[(cell_info['include_cell'] == True) & (cell_info['multipolar'] == False)]

adata = adata[adata.obs.index.intersection(cell_info['cell_id'].values)]

colors_dict = yaml.safe_load(open('../../../../config/colors.yaml', 'r'))

```

```python

adata = adata[:, adata.var['has_allele_cn']]

```

```python

cluster_label = 'sbmclone_cluster_id'

adata_clusters = spectrumanalysis.utils.aggregate_cna_adata(adata, cluster_label)

```

```python

adata.obs.groupby([cluster_label, 'n_wgd']).size().unstack(fill_value=0)

```

```python

import spectrumanalysis.phylocn

# Pre post WGD changes that minimize total changes
n_states = int(adata.layers['state'].max() + 1)
pre_post_changes = spectrumanalysis.phylocn.calculate_pre_post_changes(n_states)

```

```python

adata_wgd = adata[adata.obs.query('is_wgd == 1').index].copy()

layers = ['state']

for layer in layers:
    for wgd_timing in ['pre', 'post']:
        adata_wgd.layers[f'{wgd_timing}_wgd_diff_{layer}'] = np.empty(adata_wgd.layers[layer].shape)
        adata_wgd.layers[f'{wgd_timing}_wgd_diff_{layer}'][:] = np.NaN

for cell_id, cluster_id in adata_wgd.obs[cluster_label].items():
    cell_idx = adata_wgd.obs.index.get_loc(cell_id)
    for layer in layers:
        cell_cn = adata_wgd[cell_id].layers[layer][0]
        cluster_cn = adata_clusters[cluster_id].layers[layer][0].toarray()

        # HACK: set bins with cluster cn 0 to have cluster cn 1
        cluster_cn[cluster_cn == 0] = 1
        
        for wgd_timing in ['pre', 'post']:
            cn_change = pre_post_changes.loc[pd.IndexSlice[zip(cluster_cn, cell_cn)], wgd_timing].values
            adata_wgd.layers[f'{wgd_timing}_wgd_diff_{layer}'][cell_idx, :] = cn_change

```

```python

adata_wgd_subclone = scgenome.tl.cluster_cells(
    adata_wgd[(adata_wgd.obs['n_wgd'] == 1), (adata_wgd.var['has_allele_cn']) & (adata_wgd.var['chr'].isin(['1']))],
    layer_name='copy', max_k=16)

fig = plt.figure(figsize=(4, 16))
g = scgenome.pl.plot_cell_cn_matrix_fig(
    adata_wgd_subclone,
    fig=fig,
    layer_name='state',
    cell_order_fields=['cluster_id'],
    annotation_fields=['cluster_id'],
    style='white',
    show_cell_ids=True
)

g['adata'].obs.index

```

# Manual step!

Select the WGD cluster with the post-wgd changes that point to a shared clonal history


```python

wgd_subclone_cell_ids = adata_wgd_subclone.obs[adata_wgd_subclone.obs['cluster_id'].isin(['1'])].index

adata_wgd.obs['is_wgd_subclone'] = False
adata_wgd.obs.loc[wgd_subclone_cell_ids, 'is_wgd_subclone'] = True

adata_wgd.obs['is_wgd_subclone'].sum()

```

```python

adata_wgd.obs.loc[wgd_subclone_cell_ids].reset_index()[['cell_id']].to_csv(f'../../../../annotations/wgd_subclones/wgd_subclones_{patient_id}.csv', index=False)

```

```python

chromosomes = [str(a) for a in range(1, 23)] + ['X']
scgenome.refgenome.set_genome_version('hg19', chromosomes=chromosomes, plot_chromosomes=chromosomes)

cluster_ids = sorted(adata.obs[cluster_label].unique())
cluster_colors = dict(zip(cluster_ids, colors_dict['clone_id'].values()))

chromosomes = adata_wgd.var['chr'].unique()
chromosomes = ['1', '4', '15', 'X']

adata_wgd_plot = adata_wgd[(adata_wgd.obs['n_wgd'] == 1),  (adata_wgd.var['has_allele_cn']) & (adata_wgd.var['chr'].isin(chromosomes))].copy()

# Sort on just chromosome 'X' for aesthetics
adata_wgd_plot_sorted = scgenome.tl.sort_cells(adata_wgd_plot[:, (adata_wgd_plot.var['chr'].isin(['4', 'X']))], layer_name='state')
adata_wgd_plot.obs['cell_order'] = adata_wgd_plot_sorted.obs['cell_order']

adata_nwgd_plot = adata[(adata.obs['n_wgd'] == 0), (adata.var['has_allele_cn']) & (adata.var['chr'].isin(chromosomes))]

# Sort on just chromosome 'X' for aesthetics
adata_nwgd_plot_sorted = scgenome.tl.sort_cells(adata_nwgd_plot[:, (adata_nwgd_plot.var['chr'].isin(['4', 'X']))], layer_name='state')
adata_nwgd_plot.obs['cell_order'] = adata_nwgd_plot_sorted.obs['cell_order']

nwgd_cells = adata_nwgd_plot.shape[0]
wgd_cells = adata_wgd_plot.shape[0]

fig_legend, ax_legend = plt.subplots()
spectrumanalysis.plots.generate_color_legend(
    cluster_colors, order=sorted(cluster_colors.keys()), ax=ax_legend, title='SBMClone\nCell Cluster')

fig, axes = plt.subplots(
    nrows=4, ncols=2,
    height_ratios=[10, nwgd_cells / 2, wgd_cells / 2, wgd_cells / 2],
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
ax.set_ylabel('Cyto. band', size=6, rotation=0, ha='right', va='center')

ax = axes[1, 0]
g = scgenome.pl.plot_cell_cn_matrix(
    adata_nwgd_plot[adata_nwgd_plot.obs[cluster_label].isin(cluster_ids)],
    layer_name='state',
    ax=ax,
    cell_order_fields=[cluster_label, 'cell_order'],
    style='white',
)
nwgd_order = g['adata'].obs.index
ax.set_ylabel(f'nWGD cells\nCN (n = {nwgd_cells})', size=6, rotation=0, ha='right', va='center')
ax.tick_params(axis='x', which='major', bottom=False)

ax = axes[1, 1]
scgenome.plotting.heatmap._plot_categorical_annotation(
    adata_nwgd_plot[nwgd_order].obs[[cluster_label]].copy().values,
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
    cell_order_fields=[cluster_label, 'is_wgd_subclone', 'cell_order'],
    style='white',
)
wgd_order = g['adata'].obs.index
ax.set_ylabel(f'WGD cells\nCN (n = {wgd_cells})', size=6, rotation=0, ha='right', va='center')
ax.tick_params(axis='x', which='major', bottom=False)

ax = axes[2, 1]
scgenome.plotting.heatmap._plot_categorical_annotation(
    adata_wgd_plot[wgd_order].obs[[cluster_label]].copy().values,
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
    adata_wgd_plot[wgd_order],
    layer_name='post_wgd_diff_state',
    ax=ax,
    raw=True,
    cmap='coolwarm',
    vmin=-2,
    vmax=2,
    style='white',
)
ax.set_ylabel('WGD cells\npost-WGD ∆CN', size=6, rotation=0, ha='right', va='center')

ax = axes[3, 1]
scgenome.plotting.heatmap._plot_categorical_annotation(
    adata_wgd_plot[wgd_order].obs[[cluster_label]].copy().values,
    ax=ax,
    ax_legend=None,
    title='',
    horizontal=False,
    cmap=cluster_colors)
ax.set_xticklabels(['Clone'])
ax.spines[:].set_visible(False)

plt.suptitle(patient_id.replace('SPECTRUM-', ''), size=10, y=0.93)

plt.subplots_adjust(hspace=0.051, wspace=0.051)

fig.savefig(f'../../../../figures/edfigure3/wgd_clone_{patient_id}.svg', bbox_inches='tight', metadata={'Date': None})

fig_legend.savefig(f'../../../../figures/edfigure3/wgd_clone_legend_{patient_id}.svg', bbox_inches='tight', metadata={'Date': None})

```

# Compute cn changes pre and post wgd


```python

adata_wgd.var['subclone_post_wgd_diff_state'] = np.median(adata_wgd[wgd_subclone_cell_ids].layers['post_wgd_diff_state'], axis=0).astype(int)
adata_wgd.var['subclone_pre_wgd_diff_state'] = np.median(adata_wgd[wgd_subclone_cell_ids].layers['pre_wgd_diff_state'], axis=0).astype(int)

plt.figure(figsize=(4, 1))
scgenome.pl.plot_profile(
    adata_wgd.var,
    y='subclone_post_wgd_diff_state',
)

plt.figure(figsize=(4, 1))
scgenome.pl.plot_profile(
    adata_wgd.var,
    y='subclone_pre_wgd_diff_state',
)

```

```python

import spectrumanalysis.cnevents

spectrumanalysis.cnevents.annotate_bins(adata_wgd)

wgd_clone_events = []
for wgd_timing in ['pre', 'post']:
    cn_change = adata_wgd.var.rename(columns={f'subclone_{wgd_timing}_wgd_diff_state': 'cn_change'})
    for event in spectrumanalysis.cnevents.classify_segments(cn_change):
        event['timing_wgd'] = wgd_timing
        wgd_clone_events.append(event)
wgd_clone_events = pd.DataFrame(wgd_clone_events)
wgd_clone_events['patient_id'] = patient_id
wgd_clone_events['cell_count'] = len(wgd_subclone_cell_ids)

```

```python

sns.catplot(x='kind', col='region', hue='timing_wgd', data=wgd_clone_events, kind='count')

```

```python

# Save post-wgd events
wgd_clone_events.to_csv(f'../../../../results/tables/wgd_subclone_events/wgd_subclone_events_{patient_id}.csv', index=False)

```

```python

```
