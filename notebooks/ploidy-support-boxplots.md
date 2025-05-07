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
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

```python
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import anndata as ad

import tqdm
from scipy.stats import linregress, mannwhitneyu
import pickle
from yaml import safe_load
import matplotlib.colors as mcolors
from datetime import datetime
import statsmodels.api as sm
import statsmodels.formula.api as smf
import spectrumanalysis.stats
import vetica.mpl
import matplotlib
matplotlib.rcParams['svg.fonttype'] = 'none'
```

```python
pipeline_outputs = pipeline_dir # path to root directory of scWGS pipeline outputs
spectrumanalysis_repo = '.'
colors_yaml = safe_load(open(os.path.join(spectrumanalysis_repo, 'config/colors.yaml'), 'r').read())
wgd_colors = {0:mcolors.to_hex((197/255, 197/255, 197/255)),
              1:mcolors.to_hex((252/255, 130/255, 79/255)),
              2:mcolors.to_hex((170/255, 0, 0/255))}

data_dir = 'repos/spectrum-figures/compute-read-overlaps/output'

```

# read overlaps

```python
cell_info = pd.read_csv('filtered_cell_table_withoverlaps.csv.gz')
cell_info = cell_info[cell_info.include_cell].copy()
```

```python
from matplotlib.transforms import blended_transform_factory
def add_significance_line(ax, sig, x_pos_1, x_pos_2, y_pos):
    """
    Add a significance line with text annotation to a plot.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes object to add the significance line to.
    sig : str
        The text annotation to display above the line.
    x_pos_1 : float
        The x-coordinate of the start of the line.
    x_pos_2 : float
        The x-coordinate of the end of the line.
    y_pos : float
        The y-coordinate of the line.
    """
    transform = blended_transform_factory(ax.transData, ax.transAxes)
    line = plt.Line2D([x_pos_1, x_pos_2], [y_pos, y_pos], transform=transform, color='k', linestyle='-', linewidth=1)
    lineL = plt.Line2D([x_pos_1, x_pos_1], [y_pos, y_pos-0.02], transform=transform, color='k', linestyle='-', linewidth=1)
    lineR = plt.Line2D([x_pos_2, x_pos_2], [y_pos, y_pos-0.02], transform=transform, color='k', linestyle='-', linewidth=1)
    ax.get_figure().add_artist(line)
    ax.get_figure().add_artist(lineL)
    ax.get_figure().add_artist(lineR)
    y_pos_offset = 0

    plt.text((x_pos_1+x_pos_2)/2, y_pos+y_pos_offset, sig, ha='center', va='bottom', fontsize=8, transform=transform)
```

```python
def boxplot_field(table, field, ylabel=None, figsize=(1.5, 3), dpi=150):
    result01 = mannwhitneyu(table[~table[field].isna() & (table.n_wgd == 0)][field].values, table[~table[field].isna() & (table.n_wgd == 1)][field].values)
    p01 = result01.pvalue
    significance01 = spectrumanalysis.stats.get_significance_string(p01)
    
    result12 = mannwhitneyu(table[~table[field].isna() & (table.n_wgd == 2)][field].values, table[~table[field].isna() & (table.n_wgd == 1)][field].values)
    p12 = result12.pvalue
    significance12 = spectrumanalysis.stats.get_significance_string(p12)
    
    fig, ax = plt.subplots(figsize=figsize, dpi = dpi)
    
    sns.boxplot(data=table[~table[field].isna()], x = 'n_wgd', y = field, hue='n_wgd',
                palette={i:wgd_colors[i] for i in sorted(table.n_wgd.unique())}, dodge=False, fliersize=1)
    add_significance_line(ax, significance01, 0, 1, 1.05)
    add_significance_line(ax, significance12, 1, 2, 1.1)
    ax.get_legend().remove()
    sns.despine(ax=ax)

    if ylabel is not None:
        plt.ylabel(ylabel)
    
    plt.xlabel("#WGD")
    return fig
```

```python
cell_info[['include_cell', 'n_wgd']].value_counts()
```

```python
f = boxplot_field(cell_info[cell_info.include_cell], field='fraction_overlapping_reads', ylabel = 'Average Fraction of Overlapping Reads')
plt.savefig(os.path.join(spectrumanalysis_repo, 'figures/edfigure2/read_overlaps.svg'))
```

# cell diameter

```python
f = boxplot_field(cell_info[cell_info.include_cell], field='Diameter', ylabel = 'Cell Diameter')
plt.savefig(os.path.join(spectrumanalysis_repo, 'figures/edfigure2/cell_diameter.svg'))
```

# mtdna copy number

<!-- #raw -->
# initial version using Minsoo's copy number calls
mtdna = pd.read_table('users/kimm/project/scDNA/bam/SPECTRUM/cell_info/allcellsfile5.tsv')
mtdna['log10_mt_copynumber'] = np.log10(mtdna.mt_copynumber)
print(len(cell_info), len(mtdna), len(cell_info.merge(mtdna, on='cell_id')))

f = boxplot_field(cell_info.merge(mtdna, on = 'cell_id'), field='log10_mt_copynumber', ylabel = 'log10(mtDNA Copy Number)')
plt.savefig(os.path.join(spectrumanalysis_repo, 'figures/edfigure2/mtdna_copynumber.svg'))
<!-- #endraw -->

```python
mtdna_results
```

```python
mtdna_results = pd.read_csv('my_mtdna_table.csv.gz')

f = boxplot_field(mtdna_results[mtdna_results.include_cell], field='log10_mtdna_copynumber', ylabel = 'log10(mtDNA Copy Number)')
plt.savefig(os.path.join(spectrumanalysis_repo, 'figures/edfigure2/mtdna_copynumber.svg'))
```

```python

```
