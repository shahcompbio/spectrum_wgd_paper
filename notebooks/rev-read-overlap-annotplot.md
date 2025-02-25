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
from scipy.stats import linregress
import pickle
from yaml import safe_load
import matplotlib.colors as mcolors
from datetime import datetime

```

# NOTE: DEPENDENCIES
This notebook to generate the read overlaps figure requires `statannotations`, which requires an earlier version of `seaborn`

Last run with:
* `python` 3.9.20
* `statannotations` 0.6.0
* `seaborn` 0.11.2
* `pandas` 1.5.3


# load data

```python
from statannotations.Annotator import Annotator
```

```python
pipeline_outputs = '/data1/shahs3/users/myersm2/repos/spectrum_wgd_data5'
colors_yaml = safe_load(open('/data1/shahs3/users/myersm2/repos/spectrumanalysis/config/colors.yaml', 'r').read())
wgd_colors = {0:mcolors.to_hex((197/255, 197/255, 197/255)),
              1:mcolors.to_hex((252/255, 130/255, 79/255)),
              2:mcolors.to_hex((170/255, 0, 0/255))}

data_dir = '/data1/shahs3/users/myersm2/repos/spectrum-figures/compute-read-overlaps/output'

```

```python
# merge read overlaps by aliquot into cell table from the pipeline

cell_info = pd.read_csv(os.path.join(pipeline_outputs, 'preprocessing/summary/filtered_cell_table.csv.gz'))
table_cells = set(cell_info.cell_id)
cell_info['fraction_overlapping_reads'] = None
cell_info = cell_info.set_index('cell_id')

for f in sorted(os.listdir(data_dir)):
    if not f.endswith('.yaml') and f.split('.')[0] in set(cell_info.aliquot_id):
        df = pd.read_csv(os.path.join(data_dir, f), dtype={'chr':str})
        celldf = df.groupby('cell_id').fraction_overlapping_reads.mean().reset_index() 
        prefix = '_'.join(f.split('_')[:-2]).replace('_UNSORTED', '').replace('_CD45N', '').replace('_NUCLEI', '').replace('_NEW', '').replace('_OLD', '')
        celldf.cell_id =  prefix + '-' + celldf.cell_id
        celldf['in_table'] = celldf.cell_id.isin(table_cells)
        print(datetime.now(), f'{f.split(".")[0]}: {celldf.in_table.sum()}/{len(celldf)}')

        celldf2 = celldf[celldf.in_table]
        cell_info.loc[celldf2.cell_id, 'fraction_overlapping_reads'] = celldf2.fraction_overlapping_reads.values
```

```python
cell_info.to_csv('filtered_cell_table_withoverlaps.csv.gz', compression={'method':'gzip', 'mtime':0, 'compresslevel':9})
```

```python
cell_info = pd.read_csv('filtered_cell_table_withoverlaps.csv.gz')
```

```python
min_cells = 10

fig = plt.figure(figsize=(12,9), dpi = 300)
j = 0
for i, (p, pdf) in enumerate(sorted(cell_info.groupby('patient_id'))):
    pdf = pdf[pdf.include_cell & ~pdf.fraction_overlapping_reads.isna()]
    wgd_categories = [r['index'] for _, r in pdf.n_wgd.value_counts().reset_index().iterrows() if r['n_wgd'] >= min_cells]
    pdf = pdf[pdf.n_wgd.isin(wgd_categories)].copy()
    if len(wgd_categories) <= 1:
        print('only 1 WGD category for patient', p)
        continue

    pdf.fraction_overlapping_reads = pdf.fraction_overlapping_reads.astype(float)
    
    plt.subplot(4, 6, j + 1)
    ax = plt.gca()
    sns.boxplot(data=pdf, x = 'n_wgd', y = 'fraction_overlapping_reads', hue='n_wgd',
                palette={i:wgd_colors[i] for i in sorted(pdf.n_wgd.unique())}, dodge=False)
    sns.stripplot(data=pdf, x = 'n_wgd', y = 'fraction_overlapping_reads', linewidth=0.2, s=3.5, hue='n_wgd',
                palette={i:wgd_colors[i] for i in sorted(pdf.n_wgd.unique())}, dodge=False)
    plt.title(p)

    pairs = [(a, b) for a in pdf.n_wgd.unique() for b in pdf.n_wgd.unique() if a < b]
    if len(pairs) > 0:
        annotator = Annotator(ax, pairs, data=pdf, x='n_wgd', y='fraction_overlapping_reads')
        annotator.configure(test='t-test_ind', text_format='star')
        annotator.apply_and_annotate()
    plt.gca().get_legend().remove()
    plt.ylabel('')
    j += 1

fig.supylabel("Average fraction of overlapping reads")
plt.tight_layout()

```

# divide each cell by coverage and summarize in 1 plot?

```python
plt.figure(dpi=300)
my_df = cell_info[cell_info.include_cell & ~cell_info.fraction_overlapping_reads.isna()].copy()
my_df['fraction_overlapping_norm'] = my_df.fraction_overlapping_reads / my_df.coverage_depth
sns.boxplot(data=my_df, x='n_wgd', y='fraction_overlapping_reads', palette={i:wgd_colors[i] for i in sorted(my_df.n_wgd.unique())})


pairs = [(a, b) for a in my_df.n_wgd.unique() for b in my_df.n_wgd.unique() if a < b]
if len(pairs) > 0:
    annotator = Annotator(plt.gca(), pairs, data=my_df, x='n_wgd', y='fraction_overlapping_reads')
    annotator.configure(test='t-test_ind', text_format='star')
    annotator.apply_and_annotate()
j += 1
```

```python
plt.figure(dpi=300)
sns.boxplot(data=my_df, x='n_wgd', y='fraction_overlapping_norm', palette={i:wgd_colors[i] for i in sorted(my_df.n_wgd.unique())})

pairs = [(a, b) for a in my_df.n_wgd.unique() for b in my_df.n_wgd.unique() if a < b]
if len(pairs) > 0:
    annotator = Annotator(plt.gca(), pairs, data=my_df, x='n_wgd', y='fraction_overlapping_norm')
    annotator.configure(test='t-test_ind', text_format='star')
    annotator.apply_and_annotate()

plt.ylabel("Mean frac. overlapping reads / coverage")
j += 1
```

```python
min_cells = 10

fig = plt.figure(figsize=(12,9), dpi = 300)
j = 0
for i, (p, pdf) in enumerate(sorted(cell_info.groupby('patient_id'))):
    pdf = pdf[pdf.include_cell & ~pdf.fraction_overlapping_reads.isna()]
    wgd_categories = [r['index'] for _, r in pdf.n_wgd.value_counts().reset_index().iterrows() if r['n_wgd'] >= min_cells]
    pdf = pdf[pdf.n_wgd.isin(wgd_categories)].copy()
    if len(wgd_categories) <= 1:
        print('only 1 WGD category for patient', p)
        continue

    pdf.fraction_overlapping_reads = pdf.fraction_overlapping_reads.astype(float)
    
    plt.subplot(4, 6, j + 1)
    ax = plt.gca()
    sns.boxplot(data=pdf, x = 'n_wgd', y = 'fraction_overlapping_reads', hue='n_wgd',
                palette={i:wgd_colors[i] for i in sorted(pdf.n_wgd.unique())}, dodge=False)
    sns.stripplot(data=pdf, x = 'n_wgd', y = 'fraction_overlapping_reads', linewidth=0.2, s=3.5, hue='n_wgd',
                palette={i:wgd_colors[i] for i in sorted(pdf.n_wgd.unique())}, dodge=False)
    plt.title(p)

    pairs = [(a, b) for a in pdf.n_wgd.unique() for b in pdf.n_wgd.unique() if a < b]
    if len(pairs) > 0:
        annotator = Annotator(ax, pairs, data=pdf, x='n_wgd', y='fraction_overlapping_reads')
        annotator.configure(test='t-test_ind', text_format='star')
        annotator.apply_and_annotate()
    plt.gca().get_legend().remove()
    plt.ylabel('')
    j += 1

fig.supylabel("Average fraction of overlapping reads")
plt.tight_layout()

```
