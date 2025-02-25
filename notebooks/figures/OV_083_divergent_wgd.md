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

plt.rcParams['svg.fonttype'] = 'none'

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import spectrumanalysis.plots
import spectrumanalysis.dataload

project_dir = os.environ['SPECTRUM_PROJECT_DIR']

colors_dict = yaml.safe_load(open('../../../../config/colors.yaml', 'r'))

chromosomes = list(str(a) for a in range(1, 23)) + ['X']

scgenome.refgenome.set_genome_version('hg19', chromosomes=chromosomes, plot_chromosomes=chromosomes)

```

```python

patient_id = 'SPECTRUM-OV-083'

cn_filename = f'{project_dir}/medicc/output/{patient_id}__neither/{patient_id}__neither_final_cn_profiles.tsv'
tree_filename = f'{project_dir}/medicc/output/{patient_id}__neither/{patient_id}__neither_final_tree.new'
events_filename = f'{project_dir}/medicc/output/{patient_id}__neither/{patient_id}__neither_copynumber_events_df.tsv'

tree, adata, events = spectrumanalysis.dataload.load_medicc_as(cn_filename, tree_filename, events_filename)

adata_cna = spectrumanalysis.dataload.load_filtered_cna_adata(project_dir, patient_id)

adata = scgenome.tl.rebin_regular(adata, 500000, outer_join=False, agg_X=None, agg_var=None, agg_layers={'cn_a': np.nanmedian, 'cn_b': np.nanmedian})
adata.layers['state'] = adata.layers['cn_a'] + adata.layers['cn_b']

```

```python

tree2, adata2 = scgenome.tl.align_cn_tree(tree, adata)

fig = plt.figure(figsize=(5, 3), dpi=300)
g = scgenome.pl.plot_cell_cn_matrix_fig(
    adata2[:, adata.var['chr'] == '6'],
    tree=tree2,
    layer_name='cn_a',
    fig=fig,
    style='white',
)
spectrumanalysis.plots.remove_xticklabels(g['heatmap_ax'], ['14', '16', '18', '20', '22'])
g['heatmap_ax'].xaxis.set_tick_params(width=.5)

g['tree_ax'].set_xticks([0, 50])
g['tree_ax'].spines['bottom'].set_bounds(0, 50)
g['tree_ax'].spines['bottom'].set_position(('axes', -0.02))
g['tree_ax'].spines['bottom'].set_linewidth(.5)
g['tree_ax'].xaxis.set_tick_params(width=.5)

fig = plt.figure(figsize=(5, 3), dpi=300)
g = scgenome.pl.plot_cell_cn_matrix_fig(
    adata2[:, adata.var['chr'] == '6'],
    tree=tree2,
    layer_name='cn_b',
    fig=fig,
    style='white',
)
spectrumanalysis.plots.remove_xticklabels(g['heatmap_ax'], ['14', '16', '18', '20', '22'])
g['heatmap_ax'].xaxis.set_tick_params(width=.5)

g['tree_ax'].set_xticks([0, 50])
g['tree_ax'].spines['bottom'].set_bounds(0, 50)
g['tree_ax'].spines['bottom'].set_position(('axes', -0.02))
g['tree_ax'].spines['bottom'].set_linewidth(.5)
g['tree_ax'].xaxis.set_tick_params(width=.5)

```

```python

plt.figure(figsize=(8, 2))
ax = plt.gca()
spectrumanalysis.plots.plot_cn_rect(
    adata,
    obs_id=tree2.clade.name,
    ax=ax,
    y='cn_b',
    hue=None,
    color='b',
    rect_kws=dict(height=0.3),
    offset=-0.1,
)

spectrumanalysis.plots.plot_cn_rect(
    adata,
    obs_id=tree2.clade.name,
    ax=ax,
    y='cn_a',
    hue=None,
    color='r',
    rect_kws=dict(height=0.3),
    offset=0.1,
)

ax.set_ylim((-0.5, 8.5))
ax.grid(visible=True, which='major', axis='y')


plt.figure(figsize=(8, 2))
ax = plt.gca()
spectrumanalysis.plots.plot_cn_rect(
    adata,
    obs_id=tree2.clade.clades[0].name,
    ax=ax,
    y='cn_b',
    hue=None,
    color='b',
    rect_kws=dict(height=0.3),
    offset=-0.1,
)

spectrumanalysis.plots.plot_cn_rect(
    adata,
    obs_id=tree2.clade.clades[0].name,
    ax=ax,
    y='cn_a',
    hue=None,
    color='r',
    rect_kws=dict(height=0.3),
    offset=0.1,
)

ax.set_ylim((-0.5, 8.5))
ax.grid(visible=True, which='major', axis='y')



plt.figure(figsize=(8, 2))
ax = plt.gca()
spectrumanalysis.plots.plot_cn_rect(
    adata,
    obs_id=tree2.clade.clades[1].name,
    ax=ax,
    y='cn_b',
    hue=None,
    color='b',
    rect_kws=dict(height=0.3),
    offset=-0.1,
)

spectrumanalysis.plots.plot_cn_rect(
    adata,
    obs_id=tree2.clade.clades[1].name,
    ax=ax,
    y='cn_a',
    hue=None,
    color='r',
    rect_kws=dict(height=0.3),
    offset=0.1,
)

ax.set_ylim((-0.5, 8.5))
ax.grid(visible=True, which='major', axis='y')



```

```python

clade1 = [a.name for a in tree2.clade.clades[0].get_terminals()]
fig = spectrumanalysis.plots.pretty_pseudobulk_ascn(adata_cna[adata_cna.obs['brief_cell_id'].isin(clade1)])
fig.savefig(f'../../../../figures/figure4/patient_{patient_id}_clade1.svg', bbox_inches='tight', metadata={'Date': None})

clade2 = [a.name for a in tree2.clade.clades[1].get_terminals()]
fig = spectrumanalysis.plots.pretty_pseudobulk_ascn(adata_cna[adata_cna.obs['brief_cell_id'].isin(clade2)])
fig.savefig(f'../../../../figures/figure4/patient_{patient_id}_clade2.svg', bbox_inches='tight', metadata={'Date': None})

```

```python

plt.figure(figsize=(8, 2))
ax = plt.gca()
spectrumanalysis.plots.plot_cn_rect(
    adata,
    obs_id=tree2.clade.name,
    ax=ax,
    y='cn_a',
    hue='cn_a',
)

plt.figure(figsize=(8, 2))
ax = plt.gca()
spectrumanalysis.plots.plot_cn_rect(
    adata,
    obs_id=tree2.clade.name,
    ax=ax,
    y='cn_b',
    hue='cn_b',
)


plt.figure(figsize=(8, 2))
ax = plt.gca()
spectrumanalysis.plots.plot_cn_rect(
    adata,
    obs_id=tree2.clade.clades[0].name,
    ax=ax,
    y='cn_a',
    hue='cn_a',
)

plt.figure(figsize=(8, 2))
ax = plt.gca()
spectrumanalysis.plots.plot_cn_rect(
    adata,
    obs_id=tree2.clade.clades[0].name,
    ax=ax,
    y='cn_b',
    hue='cn_b',
)


plt.figure(figsize=(8, 2))
ax = plt.gca()
spectrumanalysis.plots.plot_cn_rect(
    adata,
    obs_id=tree2.clade.clades[1].name,
    ax=ax,
    y='cn_a',
    hue='cn_a',
)

plt.figure(figsize=(8, 2))
ax = plt.gca()
spectrumanalysis.plots.plot_cn_rect(
    adata,
    obs_id=tree2.clade.clades[1].name,
    ax=ax,
    y='cn_b',
    hue='cn_b',
)

```

```python

fig = spectrumanalysis.plots.pretty_pseudobulk_tcn(adata)

```

```python

fig = spectrumanalysis.plots.pretty_cell_tcn(adata, vestigial_cell_id)

```

```python

```
