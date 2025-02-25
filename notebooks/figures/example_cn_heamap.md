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

```

```python

patient_id = 'SPECTRUM-OV-110'

filename = f'{project_dir}/postprocessing/sankoff_ar/{patient_id}/sankoff_ar_{patient_id}.h5'
adata = ad.read(filename)

tree_filename = f'{project_dir}/postprocessing/sankoff_ar/{patient_id}/sankoff_ar_tree_{patient_id}.pickle'
with open(tree_filename, 'rb') as f:
    tree = pickle.load(f)

```

```python

# adata2.obs = adata2.obs.set_index('brief_cell_id')
tree, adata = scgenome.tl.align_cn_tree(tree, adata)
adata.layers['state'] = adata.layers['cn_a_2'] + adata.layers['cn_b_2']

for clade in tree.find_clades():
    if clade.wgd:
        clade.color = 'r'
    else:
        clade.color = 'k'

fig = plt.figure(figsize=(8, 4), dpi=300)
g = scgenome.pl.plot_cell_cn_matrix_fig(
    adata,
    tree=tree,
    layer_name='state',
    raw=False,
    fig=fig,
    style='white',
)
spectrumanalysis.plots.remove_xticklabels(g['heatmap_ax'], ['14', '16', '18', '20', '22'])

phylo_order = g['adata'].obs.index

fig.savefig(f'../../../../figures/edfigure4/example_tree_heatmap_{patient_id}.svg', bbox_inches='tight', metadata={'Date': None})

```

```python

colors = []
losses = np.zeros((5, 5))
gains = np.zeros((5, 5))
for a in range(-2, 2+1):
    for b in range(-2, 2+1):
        gain = max(0, a) + max(0, b)
        loss = -min(0, a) - min(0, b)
        gain = gain / 3.
        loss = loss / 3.
        grey_mix = 0.05
        color = np.array([
            np.clip(0.8 * (1-loss) + 0.2 - grey_mix, a_min=0, a_max=1),
            np.clip(0.8 * min(1-loss, 1-gain) + 0.2 - grey_mix, a_min=0, a_max=1),
            np.clip(0.8 * (1-gain) + 0.2 - grey_mix, a_min=0, a_max=1)])
        # color = np.array([
        #     np.clip(0.4 * gain + 0.6, a_min=0, a_max=1),
        #     np.clip(0.6, a_min=0, a_max=1),
        #     np.clip(0.4 * loss + 0.6, a_min=0, a_max=1)])
        colors.append({'a': a, 'b': b, 'color': color})
colors = pd.DataFrame(colors).set_index(['a', 'b'])

colors_mat = colors['color'].unstack()

plot_colors = np.zeros(colors_mat.shape + (3,))
for i in range(plot_colors.shape[0]):
    for j in range(plot_colors.shape[1]):
        plot_colors[i, j, :] = np.array(colors_mat.values[i, j])

fig, ax = plt.subplots(figsize=(1.5, 1.5), dpi=150)
ax.imshow(plot_colors[::-1, :])
ax.set_xticks(range(5), colors_mat.columns.values)
ax.set_yticks(range(5), colors_mat.index.values[::-1])
ax.set_xlabel('A')
ax.set_ylabel('B', rotation=0, ha='right', va='center')
ax.set_xticks(np.arange(6) - 0.5, minor=True)
ax.set_yticks(np.arange(6) - 0.5, minor=True)
ax.grid(which='minor', color='w')
ax.spines[:].set_color('w')
ax.tick_params(axis='both', left=False, right=False, length=0, which='minor')
ax.set_title('CN Change')

fig.savefig(f'../../../../figures/edfigure4/example_tree_cnchange_{patient_id}_legend.svg', bbox_inches='tight', metadata={'Date': None})

```

```python

cn_a = adata.layers['cn_a_2_change']
cn_b = adata.layers['cn_b_2_change']

cn_a[cn_a > 2] = 2
cn_a[cn_a < -2] = -2

cn_b[cn_b > 2] = 2
cn_b[cn_b < -2] = -2

allele_change_color = np.zeros(adata.shape + (3,))

for a in range(-2, 2+1):
    for b in range(-2, 2+1):
        allele_change_color[(cn_a == a) & (cn_b == b)] = colors.loc[(a, b), 'color']

adata.layers['allele_change_color'] = allele_change_color

```

```python

fig = plt.figure(figsize=(5, 4), dpi=300)
g = scgenome.pl.plot_cell_cn_matrix_fig(
    adata[phylo_order],
    layer_name='allele_change_color',
    raw=True,
    fig=fig,
    style='white',
    vmin=0,
    vmax=len(colors),
)
spectrumanalysis.plots.remove_xticklabels(g['heatmap_ax'], ['14', '16', '18', '20', '22'])

g['legend_info']['ax_legend'].remove()
g['legend_info']['axins'].remove()

fig.savefig(f'../../../../figures/edfigure4/example_tree_cnchange_{patient_id}.svg', bbox_inches='tight', metadata={'Date': None})

```

```python

```
