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

patient_id = 'SPECTRUM-OV-014'

adata = spectrumanalysis.dataload.load_filtered_cna_adata(project_dir, patient_id)
adata.obs = adata.obs.reset_index().set_index('brief_cell_id')

tree_filename = f'{project_dir}/medicc/output/{patient_id}__neither/{patient_id}__neither_final_tree.new'
tree = spectrumanalysis.dataload.load_medicc_tree(tree_filename)

```

```python

fig = spectrumanalysis.plots.pretty_pseudobulk_tcn(adata)

fig.savefig('../../../../figures/OV_014_vestigial_cell/OV_014_vestigial_cell_pseudobulk.svg', metadata={'Date': None})

```

```python

tree2, adata2 = scgenome.tl.align_cn_tree(tree, adata)

fig = plt.figure(figsize=(5, 3), dpi=300)
g = scgenome.pl.plot_cell_cn_matrix_fig(
    adata2,
    tree=tree2,
    layer_name='state',
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

vestigial_cell_id = g['adata'].obs.index[-1]

g['fig'].savefig('../../../../figures/OV_014_vestigial_cell/OV_014_vestigial_cell_tree_heatmap.svg', metadata={'Date': None})

```

```python

fig = spectrumanalysis.plots.pretty_cell_tcn(adata, vestigial_cell_id)

fig.savefig('../../../../figures/OV_014_vestigial_cell/OV_014_vestigial_cell_cn_profile.svg', metadata={'Date': None})

```

```python

```
