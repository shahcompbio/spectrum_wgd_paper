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
import scgenome
import os
import anndata as ad
import numpy as np
import pandas as pd
import Bio.Phylo
import io
import pickle
import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import vetica.mpl

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import spectrumanalysis.plots
import spectrumanalysis.dataload

project_dir = os.environ['SPECTRUM_PROJECT_DIR']

colors_dict = yaml.safe_load(open('../../../../config/colors.yaml', 'r'))

chromosomes = list(str(a) for a in range(1, 23)) + ['X']

scgenome.refgenome.set_genome_version('hg19', chromosomes=chromosomes, plot_chromosomes=chromosomes)

patient_id = 'SPECTRUM-OV-045'
adata = spectrumanalysis.dataload.load_filtered_cna_adata(project_dir, patient_id)

```

```python

sample_info = pd.read_csv('../../../../metadata/tables/sequencing_scdna.tsv', sep='\t')
sample_info = sample_info.drop(['sample_id'], axis=1).rename(columns={'spectrum_sample_id': 'sample_id'})

adata.obs['tumor_site'] = adata.obs['sample_id'].map(sample_info.dropna(subset=['sample_id', 'tumor_site']).drop_duplicates(['sample_id', 'tumor_site']).set_index('sample_id')['tumor_site'])
adata.obs['#WGD'] = adata.obs['n_wgd'].astype('category')
adata.obs['Sample'] = adata.obs['tumor_site']

adata = spectrumanalysis.plots.add_allele_state_layer(adata)

```

```python

fig, axes = plt.subplots(ncols=4, width_ratios=[2, 2, 0.05, 0.05], figsize=(8, 3), dpi=300)

cell_ids = adata.obs.query('n_wgd == 1').sample(200).index
cell_ids = cell_ids.union(adata.obs.query('n_wgd == 0').index)
cell_ids = cell_ids.union(adata.obs.query('n_wgd == 2').index)

example_cells = [
    'SPECTRUM-OV-045_S1_RIGHT_OVARY-A98245B-R45-C58',
    'SPECTRUM-OV-045_S1_PELVIC_IMPLANT-A108838B-R46-C51',
    'SPECTRUM-OV-045_S1_PELVIC_IMPLANT-A108838B-R55-C28',
]

cell_ids = cell_ids.union(example_cells)

adata.obs['Sample'] = adata.obs['tumor_site']

ax = axes[0]
g = scgenome.pl.plot_cell_cn_matrix(
    adata[cell_ids, adata.var['has_allele_cn']],
    layer_name='state',
    ax=ax,
    cell_order_fields=['#WGD', 'Sample', 'cell_order'],
    style='white',
)
ax.xaxis.set_tick_params(width=0.5)
ax.set_xlabel('')
spectrumanalysis.plots.remove_xticklabels(ax, ['14', '16', '18', '20', '22'])

ax = axes[1]
scgenome.pl.plot_cell_cn_matrix(
    adata[g['adata'].obs.index, adata.var['has_allele_cn']],
    layer_name='allele_state',
    raw=True,
    cmap=spectrumanalysis.plots.allele_state_cmap,
    ax=ax,
    style='white',
    rasterized=True,
)
ax.xaxis.set_tick_params(width=0.5)
ax.set_xlabel('')
spectrumanalysis.plots.remove_xticklabels(ax, ['14', '16', '18', '20', '22'])

ax = axes[2]
scgenome.plotting.heatmap._plot_categorical_annotation(
    adata[g['adata'].obs.index, adata.var['has_allele_cn']].obs[['#WGD']].copy().values,
    ax=ax,
    ax_legend=None,
    title='',
    horizontal=False,
    cmap=colors_dict['wgd_multiplicity'])
ax.set_xticklabels(['#WGD'])
ax.set_yticks([])
[i.set_visible(False) for i in ax.spines.values()]
ax.xaxis.set_tick_params(width=0.5)

ax = axes[3]
scgenome.plotting.heatmap._plot_categorical_annotation(
    adata[g['adata'].obs.index, adata.var['has_allele_cn']].obs[['Sample']].copy().values,
    ax=ax,
    ax_legend=None,
    title='',
    horizontal=False,
    cmap={k: colors_dict['tumor_site'][k] for k in adata.obs['Sample'].unique()})
ax.set_xticklabels(['Sample'])
ax.set_yticks([])
[i.set_visible(False) for i in ax.spines.values()]
ax.xaxis.set_tick_params(width=0.5)

fig.suptitle(f'Patient OV-045 (n={len(cell_ids)} cells)', fontsize=8, y=0.95)

plt.subplots_adjust(wspace=0.05)

ax = axes[0]
for cell_id in example_cells:
    y = g['adata'].obs.index.tolist().index(cell_id)
    ax.arrow(-100, y, 80, 0, head_width=2, head_length=20, linewidth=1)

fig.savefig('../../../../figures/figure1/OV045_heatmap.svg', bbox_inches='tight', metadata={'Date': None})

fig, ax = plt.subplots()
spectrumanalysis.plots.generate_color_legend(scgenome.plotting.cn_colors.color_reference, ax=ax, title='Total CN', ncols=3)
fig.savefig('../../../../figures/figure1/cn.svg', metadata={'Date': None})

fig, ax = plt.subplots()
spectrumanalysis.plots.generate_color_legend(spectrumanalysis.plots.allele_state_colors, ax=ax, title='Allele CN')
fig.savefig('../../../../figures/figure1/ascn.svg', metadata={'Date': None})

fig, ax = plt.subplots()
spectrumanalysis.plots.generate_color_legend(colors_dict['wgd_multiplicity'], ax=ax, title='#WGD')
fig.savefig('../../../../figures/figure1/wgd.svg', metadata={'Date': None})

fig, ax = plt.subplots()
spectrumanalysis.plots.generate_color_legend({k: colors_dict['tumor_site'][k] for k in adata.obs['Sample'].unique()}, ax=ax, title='Sample')
fig.savefig('../../../../figures/figure1/site.svg', metadata={'Date': None})

```

```python

adata.obs.groupby('n_wgd').size()

```

```python

adata.obs.loc[cell_ids].groupby('n_wgd').size()

```

```python

for cell_id in example_cells:
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(5, 2), sharex=True, dpi=150)
    _ = spectrumanalysis.plots.pretty_cell_ascn(adata[:, adata.var['has_allele_cn']], cell_id, fig=fig, axes=axes, rasterized=True)
    axes[0].get_legend().remove()
    axes[1].get_legend().remove()
    fig.savefig(f'../../../../figures/figure1/cell_{cell_id}.svg', bbox_inches='tight', metadata={'Date': None})

```

```python

```

```python

```
