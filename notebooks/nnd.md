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

import os
import glob
import anndata as ad
import pandas as pd
import numpy as np
import Bio
from tqdm import tqdm
import yaml

import vetica.mpl
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns

import scgenome

import spectrumanalysis.wgd
import spectrumanalysis.stats

colors_dict = yaml.safe_load(open('../../../../config/colors.yaml', 'r'))
project_dir = os.environ['SPECTRUM_PROJECT_DIR']

fraction_wgd = pd.read_csv('../../../../annotations/fraction_wgd_class.csv')

```

```python

cell_info = pd.read_csv(f'{project_dir}/preprocessing/summary/filtered_cell_table.csv.gz')
cell_info = cell_info[(cell_info['include_cell'] == True)]

cell_info = spectrumanalysis.wgd.classify_subclonal_wgd(cell_info)

cell_info = cell_info.merge(fraction_wgd)

```

```python

metric = 'nnd'
plot_data = cell_info.copy()

plot_data2 = plot_data.groupby(['patient_id', 'wgd_class', 'n_wgd']).agg(
    metric=(metric, 'mean'),
    n_cells=(metric, 'size')).reset_index()
plot_data2 = plot_data2[plot_data2['n_cells'] >= 10]
plot_data2[metric] = plot_data2['metric']

fig, ax = plt.subplots(figsize=(1, 2), dpi=150)

sns.boxplot(
    ax=ax, x='wgd_class', y=metric, hue='n_wgd',
    dodge=True, data=plot_data2,
    fliersize=1,
    order=['WGD-low', 'WGD-high'],
    palette=colors_dict['wgd_multiplicity'])

ax.set_ylabel('Fraction genome different\ncompared with most similar cell')
ax.set_xlabel('')
for label in ax.get_xticklabels():
    label.set_rotation(60)
    label.set_ha('right')
    label.set_rotation_mode('anchor')
sns.move_legend(
    ax, 'lower left', prop={'size': 8}, markerscale=0.5, bbox_to_anchor=(1.0, 0.5),
    ncol=1, title='#WGD', title_fontsize=10, frameon=False)
sns.despine()

mwu_stats = spectrumanalysis.stats.mwu_tests(plot_data2, ['wgd_class', 'n_wgd'], metric)
mwu_stats = spectrumanalysis.stats.fdr_correction(mwu_stats)

pairs = [
    (('WGD-low', 0), ('WGD-low', 1)),
    (('WGD-high', 1), ('WGD-high', 2)),
    (('WGD-high', 1), ('WGD-low', 1)),
]

def get_position(wgd_class, n_wgd):
    if wgd_class == 'WGD-high':
        x = 1
    else:
        x = 0
    if n_wgd == 0:
        x -= 0.25
    elif n_wgd == 2:
        x += 0.25
    return x

y_pos = 1.0
for pair in pairs:
    stat_row = mwu_stats.set_index([
        'wgd_class_1', 'n_wgd_1',
        'wgd_class_2', 'n_wgd_2',
    ]).loc[pair[0][0], pair[0][1], pair[1][0], pair[1][1]]
    x_pos_1 = get_position(*pair[0])
    x_pos_2 = get_position(*pair[1])
    spectrumanalysis.stats.add_significance_line(ax, stat_row['significance_corrected'], x_pos_1, x_pos_2, y_pos)
    y_pos += 0.08

fig.savefig('../../../../figures/figure3/nnd_wgd.svg', bbox_inches='tight', metadata={'Date': None})

```

```python

import spectrumanalysis.dataload

# 0WGD cells
patient_id = "SPECTRUM-OV-046"
cell_1 = 'SPECTRUM-OV-046_S1_INFRACOLIC_OMENTUM-A108761B-R65-C26'

# 1WGD cells
# patient_id = "SPECTRUM-OV-075"
# cell_1 = 'SPECTRUM-OV-075_S1_LEFT_FALLOPIAN_TUBE-128687A-R13-C16'

adata = spectrumanalysis.dataload.load_filtered_cna_adata(project_dir, patient_id)

```

```python

metric = 'state_total_segment_is_diff_threshold_4'

distances_filename = f'{project_dir}/preprocessing/pairwise_distance/pairwise_distance_{patient_id}.csv.gz'
distances = pd.read_csv(distances_filename, dtype={'cell_id_1': 'category', 'cell_id_2': 'category'})
distances = distances[distances['cell_id_1'].isin(cell_info['cell_id'].values)]
distances = distances[distances['cell_id_2'].isin(cell_info.query('multipolar == False')['cell_id'].values)]
distances = distances[distances['cell_id_1'] != distances['cell_id_2']]
d = distances.loc[distances.groupby('cell_id_1', observed=True)[metric].idxmin(), ['cell_id_1', 'cell_id_2', metric]]
d.columns = ['obs_id_1', 'obs_id_2', metric]

cell_2 = d.loc[d['obs_id_1'] == cell_1, 'obs_id_2'].values[0]

```

```python

from scgenome.plotting.cn import genome_axis_plot, setup_genome_xaxis_ticks

def plot_shaded(adata, var_col, ax):
    plot_data = adata.var.copy()

    def shade_regions(data, ax=None):
        if ax is None:
            ax = plt.gca()
        for start, end in data.query(var_col)[['start', 'end']].values:
            ax.fill_between([start, end], -0.05, 10, color='0.95')

    genome_axis_plot(
        plot_data,
        shade_regions,
        ('start', 'end'),
        ax=ax)

```

```python

adata.var['cell_is_diff'] = np.array(adata[cell_1].layers['state'] != adata[cell_2].layers['state'])[0]

chromosomes = list(str(a) for a in range(1, 23)) + ['X']
plot_chromosomes = {a: a for a in chromosomes}
plot_chromosomes['16'] = ''
plot_chromosomes['18'] = ''
plot_chromosomes['20'] = ''
plot_chromosomes['22'] = ''

fig, axes = plt.subplots(
    nrows=2, figsize=(6, 3), dpi=150, sharex=True, sharey=False)

ax = axes[0]
plot_shaded(adata, 'cell_is_diff', ax)
scgenome.pl.plot_cn_profile(
    adata[:, adata.var['has_allele_cn']],
    cell_1,
    value_layer_name='copy',
    state_layer_name='state',
    s=4,
    ax=ax,
    rasterized=True,
)
ax.get_legend().remove()
ax.set_ylim((0, 10))
ax.set_ylabel(f'Cell 1', rotation=0, ha='right')
setup_genome_xaxis_ticks(ax, chromosome_names=plot_chromosomes)

ax = axes[1]
plot_shaded(adata, 'cell_is_diff', ax)
scgenome.pl.plot_cn_profile(
    adata[:, adata.var['has_allele_cn']],
    cell_2,
    value_layer_name='copy',
    state_layer_name='state',
    s=4,
    ax=ax,
    rasterized=True,
)
ax.get_legend().remove()
ax.set_ylim((0, 10))
ax.set_ylabel(f'Cell 2', rotation=0, ha='right')
setup_genome_xaxis_ticks(ax, chromosome_names=plot_chromosomes)

fig.savefig(f'../../../../figures/edfigure4/cell_{cell_1}.svg', bbox_inches='tight', metadata={'Date': None})

```

```python

```
