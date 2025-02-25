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
import numpy as np
import scgenome
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import tqdm

project_dir = os.environ['SPECTRUM_PROJECT_DIR']

version = 'v5'

patient_id = 'SPECTRUM-OV-026'

```

```python

adata = ad.read(os.path.join(project_dir, f'pipeline_outputs/v5/postprocessing/aggregated_anndatas/{patient_id}_cna.h5'))

adata = adata[
    (adata.obs['is_normal'] == False) &
    (adata.obs['is_aberrant_normal_cell'] == False) &
    (adata.obs['is_doublet'] == 'No') &
    (adata.obs['is_s_phase_thresholds'] == False) &
    (adata.obs['multiplet'] == False) &
    (adata.obs['multipolar'] == False)].copy()

```

```python

cluster_label = 'sbmclone_cluster_id'

agg_X = np.sum

agg_layers = {
    'copy': np.nanmean,
    'state': np.nanmedian,
    'alleleA': np.nansum,
    'alleleB': np.nansum,
    'totalcounts': np.nansum,
    'Min': np.nanmedian,
    'Maj': np.nanmedian,
    'A': np.nanmedian,
    'B': np.nanmedian,
}

agg_obs = {
    'is_wgd': lambda a: np.nanmean(a.astype(float)),
    'n_wgd': lambda a: np.nanmean(a.astype(float)),
}

adata_reclusters = scgenome.tl.aggregate_clusters(
    adata, agg_X, agg_layers, agg_obs, cluster_col=cluster_label)
adata_reclusters.layers['BAF'] = adata_reclusters.layers['alleleB'] / adata_reclusters.layers['totalcounts']
adata_reclusters.layers['state'] = adata_reclusters.layers['state'].round()
adata_reclusters.layers['Min'] = adata_reclusters.layers['Min'].round()
adata_reclusters.layers['Maj'] = adata_reclusters.layers['Maj'].round()

```

```python

from scgenome.plotting.cn import genome_axis_plot, setup_genome_xaxis_ticks, setup_genome_xaxis_lims

def plot_ascn_profile(adata, cell_id, ax=None, chromosome=None, start=None, end=None, tick_args={}, **kwargs):

    ascn_state_colors = {
        'A-Gained': '#53AFC0',
        'A-Hom': '#025767',
        'A-LOH': '#025767',
        'B-Gained': '#FF9E41',
        'B-Hom': '#A75200',
        'B-LOH': '#9ECAE1',
        'Balanced': '#999999',
    }

    plot_data = scgenome.tl.get_obs_data(
        adata,
        cell_id,
        layer_names=['copy', 'BAF', 'state', 'A', 'B']
    )

    plot_data['ascn_state'] = 'Balanced'
    plot_data.loc[plot_data['A'] > plot_data['B'], 'ascn_state'] = 'A-Gained'
    plot_data.loc[plot_data['B'] > plot_data['A'], 'ascn_state'] = 'B-Gained'
    plot_data.loc[plot_data['B'] == 0, 'ascn_state'] = 'A-Hom'
    plot_data.loc[plot_data['A'] == 0, 'ascn_state'] = 'B-Hom'

    genome_axis_plot(
        plot_data,
        sns.scatterplot,
        ('start',),
        x='start',
        y='BAF',
        hue='ascn_state',
        hue_order=['Balanced', 'A-Gained', 'B-Gained', 'A-Hom', 'B-Hom'],
        palette=ascn_state_colors,
        ax=ax,
        linewidth=0,
        **kwargs)

    setup_genome_xaxis_ticks(ax, chromosome=chromosome, start=start, end=end, **tick_args)

    setup_genome_xaxis_lims(ax, chromosome=chromosome, start=start, end=end)

    plt.ylabel('BAF')

    sns.move_legend(
        ax, 'upper left', prop={'size': 8}, markerscale=1, bbox_to_anchor=(1, 1),
        labelspacing=0.4, handletextpad=0, columnspacing=0.5,
        ncol=1, title='AS CN state', title_fontsize=10, frameon=False)

    ax.spines[['right', 'top']].set_visible(False)

    if chromosome is not None:
        ax.set_xlabel(f'Chr. {chromosome}')

    else:
        ax.set_xlabel('Chromosome')

```

```python


def setup_chromosome_facets(chromosomes, nrows=1, **kwargs):

    column_widths = scgenome.refgenome.info.chromosome_info.set_index('chr')['chromosome_length'].loc[chromosomes].values
    
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=len(chromosomes),
        width_ratios=column_widths,
        sharex='col',
        **kwargs)

    for row_idx in range(nrows):
        for col_idx in range(len(chromosomes)):
            ax = axes[row_idx, col_idx]
            if col_idx > 0:
                ax.yaxis.set_visible(False)
                ax.spines['left'].set_visible(False)
            
    plt.subplots_adjust(wspace=0.1, hspace=0.1)

    return fig, axes


def plot_chromosome_facets(row_idx, chromosomes, plot_function, *args, **kwargs):
    for col_idx, chromosome in enumerate(chromosomes):
        ax = axes[row_idx, col_idx]
        plot_function(*args, ax=ax, chromosome=chromosome, **kwargs)
        if (col_idx != len(chromosomes) - 1) or (row_idx != 0):
            if ax.get_legend():
                ax.get_legend().remove()


def plot_isloh_shaded(adata, obs_id, ax, chromosome=None):
    plot_data = scgenome.tl.get_obs_data(
        adata,
        obs_id,
        layer_names=['A', 'B']
    )

    plot_data['is_loh'] = (plot_data['A'] == 0) | (plot_data['B'] == 0)
    
    plot_data = plot_data[plot_data['chr'] == chromosome]

    def shade_loh_regions(data, ax=None):
        if ax is None:
            ax = plt.gca()
        for start, end in data.query('is_loh')[['start', 'end']].values:
            ax.fill_between([start, end], -0.05, 1, color='0.95')

    genome_axis_plot(
        plot_data,
        shade_loh_regions,
        ('start', 'end'),
        ax=ax)


# Clone A
cell_id = 'SPECTRUM-OV-026_S1_LEFT_OVARY-128742A-R30-C46'
cluster_id = adata.obs.loc[cell_id, cluster_label]

chromosomes = ['2', '11', '12']
fig, axes = setup_chromosome_facets(chromosomes, nrows=2, figsize=(5, 3), dpi=150, sharey='row')

plot_chromosome_facets(0, chromosomes, plot_isloh_shaded, adata_reclusters, cluster_id)
plot_chromosome_facets(0, chromosomes, plot_ascn_profile, adata_reclusters, cluster_id, s=10, tick_args=dict(major_spacing=100e6, minor_spacing=10e6))
axes[0, 0].set_ylabel(f'Clone A', rotation=0, ha='right')
axes[0, 0].set_ylim((-0.05, 1))

plot_chromosome_facets(1, chromosomes, plot_isloh_shaded, adata, cell_id)
plot_chromosome_facets(1, chromosomes, plot_ascn_profile, adata, cell_id, s=10, tick_args=dict(major_spacing=100e6, minor_spacing=10e6))
axes[1, 0].set_ylabel(f'WGD Cell 1', rotation=0, ha='right')
axes[1, 0].set_ylim((-0.05, 1))

```

```python

chromosomes = list(str(a) for a in range(1, 23)) + ['X']
scgenome.refgenome.set_genome_version('hg19', chromosomes=chromosomes, plot_chromosomes=chromosomes)

fig, axes = plt.subplots(
    nrows=2, figsize=(6, 3), dpi=150, sharex=True, sharey=False)

ax = axes[0]
scgenome.pl.plot_cn_profile(
    adata_reclusters,
    cluster_id,
    value_layer_name='copy',
    state_layer_name='state',
    s=10,
    ax=ax,
)
ax.get_legend().remove()
ax.set_ylim((0, 10))
ax.set_ylabel(f'Clone A', rotation=0, ha='right')

ax = axes[1]
scgenome.pl.plot_cn_profile(
    adata,
    cell_id,
    value_layer_name='copy',
    state_layer_name='state',
    s=10,
    ax=ax,
)
ax.get_legend().remove()
ax.set_ylim((0, 10))
ax.set_ylabel(f'WGD Cell 1', rotation=0, ha='right')

```

```python

# Clone B
cell_id = 'SPECTRUM-OV-026_S1_BOWEL-128674A-R20-C42'
cluster_id = adata.obs.loc[cell_id, cluster_label]

chromosomes = ['2', '11', '12']
fig, axes = setup_chromosome_facets(chromosomes, nrows=2, figsize=(5, 3), dpi=150, sharey='row')

plot_chromosome_facets(0, chromosomes, plot_isloh_shaded, adata_reclusters, cluster_id)
plot_chromosome_facets(0, chromosomes, plot_ascn_profile, adata_reclusters, cluster_id, s=10, tick_args=dict(major_spacing=100e6, minor_spacing=10e6))
axes[0, 0].set_ylabel(f'Clone B', rotation=0, ha='right')
axes[0, 0].set_ylim((-0.05, 1))

plot_chromosome_facets(1, chromosomes, plot_isloh_shaded, adata, cell_id)
plot_chromosome_facets(1, chromosomes, plot_ascn_profile, adata, cell_id, s=10, tick_args=dict(major_spacing=100e6, minor_spacing=10e6))
axes[1, 0].set_ylabel(f'WGD Cell 2', rotation=0, ha='right')
axes[1, 0].set_ylim((-0.05, 1))

```

```python

chromosomes = list(str(a) for a in range(1, 23)) + ['X']
scgenome.refgenome.set_genome_version('hg19', chromosomes=chromosomes, plot_chromosomes=chromosomes)

fig, axes = plt.subplots(
    nrows=2, figsize=(6, 3), dpi=150, sharex=True, sharey=False)

ax = axes[0]
scgenome.pl.plot_cn_profile(
    adata_reclusters,
    cluster_id,
    value_layer_name='copy',
    state_layer_name='state',
    s=10,
    ax=ax,
)
ax.get_legend().remove()
ax.set_ylim((0, 10))
ax.set_ylabel(f'Clone B', rotation=0, ha='right')

ax = axes[1]
scgenome.pl.plot_cn_profile(
    adata,
    cell_id,
    value_layer_name='copy',
    state_layer_name='state',
    s=10,
    ax=ax,
)
ax.get_legend().remove()
ax.set_ylim((0, 10))
ax.set_ylabel(f'WGD Cell 2', rotation=0, ha='right')

```

```python

plt.figure(figsize=(1, 1), dpi=300)
scgenome.pl.cn_legend(plt.gca(), frameon=False, loc=2, bbox_to_anchor=(0., 1.), title='Copy Number')
plt.axis('off')
    
```

```python

layers = ['Min', 'Maj']

adata_wgd = adata[adata.obs.query('n_wgd == 1').index].copy()

distances = []
for obs_id_1 in tqdm.tqdm(adata_wgd.obs.index):
    for obs_id_2 in adata_reclusters.obs.index:
        distance = 0
        for layer in layers:
            distance += np.nanmean(
                adata_wgd[obs_id_1, adata_wgd.var['has_allele_cn']].layers[layer][0] != 2 * adata_reclusters[obs_id_2, adata_reclusters.var['has_allele_cn']].layers[layer][0])
        distances.append({
            'cell_id': obs_id_1,
            'cluster_id': obs_id_2,
            'distance': float(distance) / float(len(layers)),
        })
distances = pd.DataFrame(distances)

wgd_v_nowgd_distances = distances.set_index(['cell_id', 'cluster_id'])['distance'].unstack()

wgd_v_nowgd_distances_norm = (wgd_v_nowgd_distances.T / wgd_v_nowgd_distances.sum(axis=1)).T

```

```python

# wgd_v_nowgd_distances = wgd_v_nowgd_distances[wgd_v_nowgd_distances.max(axis=1) < 0.5]
wgd_v_nowgd_distances

```

```python

import spectrumanalysis.plots

wgd_cell_clusters = adata.obs.loc[wgd_v_nowgd_distances.index, [cluster_label]]

wgd_cell_clusters[cluster_label] = wgd_cell_clusters[cluster_label].astype(str).astype(
    pd.CategoricalDtype(categories=adata_reclusters.obs.index.values, ordered=True))

feature_colors, attribute_to_color = spectrumanalysis.plots.get_feature_colors(
    wgd_cell_clusters,
    {
        cluster_label: 'tab20',
    })

cluster_order = list(sorted(wgd_v_nowgd_distances.columns))
wgd_cell_order = wgd_cell_clusters.sort_values(cluster_label).index

g = sns.clustermap(
    wgd_v_nowgd_distances.loc[wgd_cell_order, cluster_order],
    col_cluster=True,
    row_cluster=False,
    cmap='Blues',
    vmin=wgd_v_nowgd_distances.min().min(),
    vmax=wgd_v_nowgd_distances.max().max(),
    z_score=None,
    standard_scale=None,
    figsize=(2, 4),
    yticklabels=False,
    row_colors=feature_colors.loc[wgd_cell_order],
    cbar_pos=(1.15, 0.7, 0.025, 0.08))

g.ax_heatmap.yaxis.set_label_position('left')
g.ax_heatmap.set_ylabel('WGD cell', labelpad=10)
g.ax_heatmap.set_xlabel('nWGD clone')
g.ax_cbar.set_title('Fraction genome\ndifferent', fontsize=10, pad=10)
g.ax_row_colors.set_xticks([])

ax_legend = g.fig.add_axes(g.ax_heatmap.get_position())
ax_legend.axis('off')
for attribute, color in attribute_to_color[cluster_label].items():
    ax_legend.bar(0, 0, color=color, label=attribute, linewidth=0)
ax_legend.legend(loc='upper left', ncols=1, bbox_to_anchor=(1.05, 0.65), frameon=False, title='SBMClone Block')

g.fig.suptitle(patient_id.replace('SPECTRUM-', ''), y=1.05, x=0.75)

```

```python

```

```python

```
