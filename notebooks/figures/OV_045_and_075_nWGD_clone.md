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
import yaml

import spectrumanalysis.cnmetrics
import spectrumanalysis.dataload
import spectrumanalysis.utils
import spectrumanalysis.plots


chromosomes = list(str(a) for a in range(1, 23)) + ['X']
scgenome.refgenome.set_genome_version('hg19', chromosomes=chromosomes, plot_chromosomes=chromosomes)

project_dir = os.environ['SPECTRUM_PROJECT_DIR']

patient_id = 'SPECTRUM-OV-125'

site_hue_order = ['Right Adnexa', 'Left Adnexa', 'Omentum', 'Bowel', 'Peritoneum', 'Other']

colors_dict = yaml.safe_load(open('../../../../config/colors.yaml', 'r'))

adata = spectrumanalysis.dataload.load_filtered_cna_adata(project_dir, patient_id)
adata = adata[(adata.obs['multipolar'] == False)]

# No n_WGD = 2 for this analysis
adata = adata[adata.obs['n_wgd'] <= 1]

```

```python

adata.obs.groupby(['sample_id', 'n_wgd']).size().unstack()

```

```python

adata.obs['n_wgd'].value_counts() / adata.shape[0]

```

```python

# Add sample information

sample_info = pd.read_csv('../../../../metadata/tables/sequencing_scdna.tsv', sep='\t')
sample_info = sample_info.drop(['sample_id'], axis=1).rename(columns={'spectrum_sample_id': 'sample_id'})

for site_col in ['tumor_site', 'tumor_megasite']:
    site_map = sample_info.drop_duplicates(['sample_id', site_col])[['sample_id', site_col]].dropna().set_index('sample_id')[site_col]
    adata.obs[site_col] = adata.obs['sample_id'].map(site_map)

```

```python

cluster_label = 'sbmclone_cluster_id'

adata.obs['cluster__n_wgd'] = adata.obs[cluster_label].astype(str) + '/' + adata.obs['n_wgd'].astype(str)

adata_clusters = spectrumanalysis.utils.aggregate_cna_adata(adata, 'cluster__n_wgd')
adata_clusters = adata_clusters[adata_clusters.obs['cluster_size'] >= 5]
adata_clusters = spectrumanalysis.plots.add_allele_state_layer(adata_clusters)
adata_clusters.obs['cluster_id'] = adata_clusters.obs.reset_index()['cluster__n_wgd'].str.split('/', expand=True)[0].values

adata_clusters.obs

```

```python

fig, ax = plt.subplots(figsize=(6, 1.5), dpi=150)

cluster__n_wgd = adata_clusters.obs.sort_values(['n_wgd', 'cluster_size'], ascending=(True, False)).index[0]

cluster_size = adata_clusters.obs.loc[cluster__n_wgd, 'cluster_size']

g = scgenome.pl.plot_cn_profile(
    adata_clusters,
    cluster__n_wgd,
    state_layer_name='state',
    value_layer_name='copy',
    squashy=True,
    ax=ax)
cluster_id, n_wgd = cluster__n_wgd.split('/')

if n_wgd == '0':
    name = 'nWGD Clone'
else:
    name = f'WGD Clone {cluster_id}'

ax.get_legend().remove()
ax.set_title(f'{name}\nn={cluster_size}', rotation=0, ha='center', va='center')

scgenome.plotting.cn.setup_genome_xaxis_ticks(
    ax, chromosome_names=dict(zip(
        [str(a) for a in range(1, 23)] + ['X'],
        ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '', '13', '', '15', '', '', '18', '', '', '21', '', 'X'])))

```

```python

scgenome.refgenome.info.chromosomes = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '', '15', '', '17', '', '19', '', '21', '', 'X', 'Y']
scgenome.refgenome.info.plot_chromosomes = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '', '15', '', '17', '', '19', '', '21', '', 'X', '']

adata_clusters = scgenome.tl.sort_cells(adata_clusters, layer_name='state')

fig, axes = plt.subplots(
    ncols=4,
    width_ratios=[4, 4, 1, 1],
    sharey=True,
    figsize=(8, .15 * adata_clusters.shape[0]), dpi=300)

ax = axes[0]
g = scgenome.pl.plot_cell_cn_matrix(
    adata_clusters,
    layer_name='state',
    cell_order_fields=['n_wgd', 'cell_order'],
    ax=ax,
    style='white',
)
ax.set_xlabel('Chromosome', fontsize='8')

ax = axes[1]
scgenome.pl.plot_cell_cn_matrix(
    adata_clusters[g['adata'].obs.index],
    layer_name='allele_state',
    raw=True,
    cmap=spectrumanalysis.plots.allele_state_cmap,
    ax=ax,
    style='white',
)
ax.set_xlabel('Chromosome', fontsize='8')
ax.tick_params(axis='y', length=0, width=0, which='major')

cluster_info = adata_clusters.obs.loc[g['adata'].obs.index]
cluster_info['idx'] = range(cluster_info.shape[0])

def plot_stacked_barh(ax, data, order, colors, y_column=None):
    left_accum = None
    for cat in order:
        if y_column is None:
            y = data.index
        else:
            y = data[y_column]
        ax.barh(
            width=data[cat],
            y=y,
            left=left_accum,
            color=colors[cat],
            label=cat)
        if left_accum is None:
            left_accum = data[cat]
        else:
            left_accum += data[cat]

ax = axes[2]
plot_data = adata.obs.groupby(['cluster__n_wgd', 'tumor_site']).size().rename('cell_count').reset_index()
plot_data = plot_data.merge(cluster_info.reset_index()[['cluster__n_wgd', 'idx']])
plot_data = plot_data.set_index(['idx', 'tumor_site'])['cell_count'].unstack()
plot_data = (plot_data.T / plot_data.sum(axis=1)).T
ylim = ax.get_ylim()
plot_data = plot_data.reindex(columns=site_hue_order, fill_value=0)
plot_stacked_barh(ax, plot_data, site_hue_order, colors_dict['tumor_site'])
# plot_data[site_hue_order].plot.barh(ax=ax, stacked=True, color=colors_dict['tumor_site'], width=0.8)
ax.set_yticks([], minor=True) # Why do i need this pandas ???
ax.set_ylim(ylim)
ax.set_xticks([], minor=True)
ax.set_yticks(range(len(plot_data.shape)))
ax.set_xlim(0, 1)
ax.spines.right.set_visible(False)
ax.spines.top.set_visible(False)
ax.set_ylabel('')
ax.tick_params(axis='y', length=0, width=0, which='major')
ax.tick_params(axis='x', which='major', labelsize=6, pad=2)
ax.set_xlabel('Site Fraction', fontsize='8')

ax = axes[3]
ax.set_xscale('log')
if patient_id == 'SPECTRUM-OV-125':
    ax.set_xlim((5, 100))
ax.barh(y='idx', left=.5, width='cluster_size', data=cluster_info, color='0.75')
ax.tick_params(axis='x', which='major', labelsize=6, pad=2)
ax.tick_params(axis='y', length=0, width=0, which='major')
ax.set_xlabel('Num. Cells', fontsize='8')
sns.despine(ax=ax, trim=False)

ax = axes[0]
labels = []
for idx, row in g['adata'].obs.iterrows():
    if row['n_wgd'] == 0:
        labels.append(f'{row["cluster_id"]} (non-WGD)')
    else:
        labels.append(f'{row["cluster_id"]}')
ax.set_yticks(ticks=range(len(labels)), labels=labels, fontsize=6)
ax.tick_params(axis='y', length=2, width=1, which='major')

if patient_id == 'SPECTRUM-OV-125':
    ax.set_ylabel('SBMClone\ncell cluster', fontsize='8', rotation=0, ha='right', va='center', labelpad=10)
    plt.suptitle(patient_id.replace('SPECTRUM-', ''), fontsize=8, y=1.3)
else:
    ax.set_ylabel('SBMClone\ncell cluster', fontsize='8', rotation=0, ha='right', va='center', labelpad=-10)
    plt.suptitle(patient_id.replace('SPECTRUM-', ''), fontsize=8, y=1.1)

plt.subplots_adjust(wspace=0.05)

fig.savefig(f'../../../../figures/edfigure3/nwgd_clone_{patient_id}.svg', bbox_inches='tight', metadata={'Date': None})

```

```python

```

```python

```
