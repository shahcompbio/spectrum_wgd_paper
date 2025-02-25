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
import seaborn as sns

import scgenome

chromosomes = [str(a) for a in range(1, 23)] + ['X']

colors_dict = yaml.safe_load(open('../../../../config/colors.yaml', 'r'))

project_dir = os.environ['SPECTRUM_PROJECT_DIR']

```

```python

cell_info = pd.read_csv(f'{project_dir}/preprocessing/summary/filtered_cell_table.csv.gz')
cell_info = cell_info[cell_info['include_cell']]

cell_info.head()

```

```python

fig = plt.figure(figsize=(2, 2), dpi=300)
ax = plt.gca()
sns.scatterplot(ax=ax, x='cn_gt2', y='mean_allele_diff', hue='n_wgd', data=cell_info, s=3, lw=0, alpha=1, palette=colors_dict['wgd_multiplicity'], rasterized=True)
ax.axvline(0.5, lw=0.5, ls=':', color='k')
ax.set_ylim((0, 3.55))
ax.set_ylabel('Mean major - minor CN')
ax.set_xlabel('Frac. genome with\nmajor CN ≥ 2')
legend = ax.legend(loc='upper right', bbox_to_anchor=(0.25, 1), title='#WGD', frameon=False, fontsize=6, markerscale=3)
legend.set_title('#WGD', prop={'size': 8})
sns.despine(trim=True, offset=10)
ax.text(.6, 0.05, f'n = {len(cell_info):,}', transform=ax.transAxes)

fig.savefig('../../../../figures/edfigure2/wgd_classify_1.svg', bbox_inches='tight', metadata={'Date': None})

```

```python

fig = plt.figure(figsize=(2, 2), dpi=300)
ax = plt.gca()
sns.scatterplot(ax=ax, x='cn_gt3', y='mean_allele_diff', hue='n_wgd', data=cell_info, s=3, lw=0, alpha=1, palette=colors_dict['wgd_multiplicity'], rasterized=True)
ax.axvline(0.5, lw=0.5, ls=':', color='k')
ax.set_ylim((0, 3.55))
ax.set_ylabel('Mean major - minor CN')
ax.set_xlabel('Frac. genome with\nmajor CN ≥ 3')
legend = ax.legend(loc='upper right', bbox_to_anchor=(0.25, 1), title='#WGD', frameon=False, fontsize=6, markerscale=3)
legend.set_title('#WGD', prop={'size': 8})
sns.despine(trim=True, offset=10)

fig.savefig('../../../../figures/edfigure2/wgd_classify_2.svg', bbox_inches='tight', metadata={'Date': None})

```

```python

cell_info['is_wgd_tracerx'] = cell_info['cn_gt2'] > 0.5

cell_info.groupby(['is_wgd', 'is_wgd_tracerx']).size().unstack()

```

```python

```

```python

```
