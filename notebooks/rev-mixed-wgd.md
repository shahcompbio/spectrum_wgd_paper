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
import pandas as pd
import anndata as ad
import numpy as np
import os
import pickle
import seaborn as sns
import re
import tqdm
from Bio import Phylo
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix
from scgenome.pl.cn_colors import color_reference
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
```

```python
pipeline_outputs = '/data1/shahs3/users/myersm2/repos/spectrum_wgd_data5'

```

```python
cell_info = pd.read_csv(os.path.join(pipeline_outputs, 'preprocessing/summary/filtered_cell_table.csv.gz'))
cell_info = cell_info[cell_info.include_cell & ~cell_info.is_normal & ~cell_info.is_s_phase_thresholds].copy()
cell_info['patient_id'] = cell_info.cell_id.str.slice(0, 15)
```

<!-- #raw -->


adatas = {}

signals_dir = os.path.join(pipeline_outputs, 'preprocessing', 'signals')
for f in tqdm.tqdm(os.listdir(signals_dir)):
    p = f.split('_')[1].split('.')[0]
    adatas[p] = ad.read_h5ad(os.path.join(signals_dir, f))

# sbmclone anndatas
sbmclone_dir = os.path.join(pipeline_outputs, 'sbmclone') 
sbmclone_adatas = {}
for p in tqdm.tqdm(adatas.keys()):
    sbmclone_adatas[p] = ad.read_h5ad(os.path.join(sbmclone_dir, f'sbmclone_{p}_snv.h5'))


for p, adata in adatas.items():
    adata = adata[adata.obs.index.isin(cell_info.cell_id)].copy()
    sbmclone_adata = sbmclone_adatas[p]
    adata.obs['sbmclone_cluster_id'] = sbmclone_adata.obs.loc[adata.obs.index, 'block_assignment']
    adatas[p] = adata
<!-- #endraw -->

```python
rows = []
for p, pdf in cell_info.groupby('patient_id'):
    cntr = pdf.n_wgd.value_counts()
    
    row = {}
    row['patient_id'] = p
    row['n_cells_total'] = len(pdf)
    row['n_cells_0wgd'] = cntr[0] if 0 in cntr else 0
    row['n_cells_1wgd'] = cntr[1] if 1 in cntr else 0
    row['n_cells_2wgd'] = cntr[2] if 2 in cntr else 0
    rows.append(row)

df = pd.DataFrame(rows)
df = df.sort_values(by = 'patient_id').reset_index(drop = True)
df['n_cells_wgd'] = (df.n_cells_1wgd + df.n_cells_2wgd)

df['prop_0wgd'] = df.n_cells_0wgd / df.n_cells_total
df['prop_1wgd'] = df.n_cells_1wgd / df.n_cells_total
df['prop_2wgd'] = df.n_cells_2wgd / df.n_cells_total
df['prop_wgd'] = df.n_cells_wgd / df.n_cells_total
```

```python
wgd_colors = {0:'#C5C5C5', 1:'#FC824F', 2:'#AA0000'}
df['<=1wgd'] = df.prop_0wgd + df.prop_1wgd
```

```python
sns.barplot(data=df, x = 'patient_id', y=1, color=wgd_colors[2])
xlocs, xlabels = plt.xticks()
plt.xticks(xlocs, [a.get_text()[-3:] for a in xlabels], rotation=90)

sns.barplot(data=df, x = 'patient_id', y="<=1wgd", color=wgd_colors[1])
xlocs, xlabels = plt.xticks()
plt.xticks(xlocs, [a.get_text()[-3:] for a in xlabels], rotation=90)

sns.barplot(data=df, x = 'patient_id', y="prop_0wgd", color=wgd_colors[0])
xlocs, xlabels = plt.xticks()
plt.xticks(xlocs, [a.get_text()[-3:] for a in xlabels], rotation=90)


plt.legend(handles=[mpatches.Patch(color=wgd_colors[2], label='2 WGD'),
                    mpatches.Patch(color=wgd_colors[1], label='1 WGD'),
                    mpatches.Patch(color=wgd_colors[0], label='nWGD')],
          bbox_to_anchor=(1, 1))
plt.ylabel("Proportion of cells")
```

```python
arr = df[['n_cells_0wgd', 'n_cells_1wgd', 'n_cells_2wgd']].values.copy()
arr[np.arange(arr.shape[0]), np.argmax(arr, axis = 1)] = 0
df['n_cells_non_dominant'] = np.max(arr, axis = 1)
```

```python
plt.figure(figsize=(12, 5), dpi = 200)

plt.subplot(1, 2, 1)
sns.barplot(data=df, x = 'patient_id', y='n_cells_total', color='lightgrey')
xlocs, xlabels = plt.xticks()
plt.xticks(xlocs, [a.get_text()[-3:] for a in xlabels], rotation=90)

sns.barplot(data=df, x = 'patient_id', y="n_cells_non_dominant", color='purple')
xlocs, xlabels = plt.xticks()
plt.xticks(xlocs, [a.get_text()[-3:] for a in xlabels], rotation=90)

plt.legend(handles=[mpatches.Patch(color='lightgrey', label='Total'),
                    mpatches.Patch(color='purple', label='Non-dominant WGD state')],
          bbox_to_anchor=(1, 1), loc='upper right')
plt.ylabel("Number of cells")

plt.subplot(1, 2, 2)
sns.barplot(data=df, x = 'patient_id', y="n_cells_non_dominant", color='purple')
xlocs, xlabels = plt.xticks()
plt.xticks(xlocs, [a.get_text()[-3:] for a in xlabels], rotation=90)

plt.ylabel("Number of cells in non-dominant WGD state")
plt.tight_layout()
```

```python
len(df)
```

```python
(df.n_cells_non_dominant >= 2).sum()
```

```python
(df.n_cells_non_dominant >= 5).sum()
```

```python
(df.n_cells_non_dominant >= 10).sum()
```

```python
(df.n_cells_non_dominant >= 25).sum()
```

# look at proportion WGD across the cohort to validate 2-class distinction

```python
sns.histplot(df, x='prop_wgd', bins = 20)
plt.xlabel("Proportion of cells with >=1 WGD")
plt.ylabel("Patients")
```

```python
df.prop_wgd[df.prop_wgd < 0.5].max()
```

```python
df.prop_wgd[df.prop_wgd > 0.75].min()
```

```python
df[(df.prop_wgd > 0.5) & (df.prop_wgd < 0.75)]
```

# make CDF plot

```python
nonplurality_cells = {}
for p, pdf in cell_info.groupby('patient_id'):
    pdf = pdf[pdf.include_cell]
    cntr = pdf.n_wgd.value_counts()
    nonplurality_cells[p] = cntr[1:].sum()
```

```python
len(nonplurality_cells)
```

```python
plt.figure(dpi = 300)
vals = np.array(list(nonplurality_cells.values()))
xs = np.arange(0, np.max(vals))
ys = [(vals > x).sum() for x in xs]
plt.plot(xs, ys)
plt.xlabel("Non-plurality WGD state cells")
plt.ylabel("Patients with at least X nonplurality cells")
```

```python

```
