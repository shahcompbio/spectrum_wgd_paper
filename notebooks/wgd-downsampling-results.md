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
import matplotlib.pyplot as plt
import seaborn as sns
import scgenome
import yaml
from datetime import datetime
import os
import anndata as ad
from Bio import Phylo
from copy import deepcopy
import warnings
import tqdm
from scipy.stats import binom
import numpy as np
import vetica.mpl

os.environ['ISABL_API_URL'] = 'https://isabl.shahlab.mskcc.org/api/v1'
import isabl_cli as ii
from sklearn.metrics import confusion_matrix
```

```python
pipeline_outputs = 'pipeline_dir/'
downsampling_repo = 'repos/spectrum-rev-downsampling'
colors = yaml.safe_load(open('../config/colors.yaml', 'r').read())
```

```python
cohort = pd.read_csv(os.path.join(downsampling_repo, 'top10.csv'))
max_n_cells = 50000
coverage_proportions = [0.1, .25, 0.5, 0.75, 0.9]
n_repetitions = 5

datasets = []
aliquot2cells = {}
patient2aliquots = {}
paramsets = set()

for _, r in cohort.iterrows():
    cell_list = [a.strip() for a in open(r.cell_list).readlines()]
    if len(cell_list) > max_n_cells:
        assert False, "exceeded max number of cells when I'm trying to include all cells"
        np.random.seed(0)
        cell_list = np.random.choice(cell_list, size=max_n_cells, replace=False)
    aliquot2cells[r.aliquot_id] = cell_list

    if r.patient_id not in patient2aliquots:
        patient2aliquots[r.patient_id] = []
    patient2aliquots[r.patient_id].append(r.aliquot_id)

    for coverage_proportion in coverage_proportions:
        covprop = str(int(100 * coverage_proportion))
        for i in range(n_repetitions):
            paramsets.add((r.patient_id, covprop, i))
            datasets.append(f'{r.aliquot_id}_{covprop}_{i}')

```

```python
datadir = os.path.join(downsampling_repo, 'simdata')
temp_dir = os.path.join(downsampling_repo, 'temp')
outdir = os.path.join(downsampling_repo, 'output')
cell_list_dir = os.path.join(downsampling_repo, 'cell-lists')
patients = sorted(set([a.split('_')[0] for a in os.listdir(outdir)]))
```

```python
signals_dir = 'users/myersm2/spectrum-dlp-pipeline/v5.2/preprocessing/signals'
adatas0 = {}
for f in tqdm.tqdm(os.listdir(signals_dir)):
    p = f.split('.')[0].split('_')[1]
    if p in patients:
        adatas0[p] = ad.read_h5ad(os.path.join(signals_dir, f))
        adatas0[p].obs = adatas0[p].obs.set_index('brief_cell_id')
```

# load results

```python
rows = []
cells = {}

for f in tqdm.tqdm([f for f in os.listdir(outdir) if f.endswith('.csv')]):
    row = {}
    row['patient'], prop, seed, _, _ = f.split('.')[0].split('_')
    row['prop'] = int(prop) / 100
    row['seed'] = int(seed)

    orig_adata = adatas0[row['patient']]

    df = pd.read_csv(os.path.join(outdir, f))
    row['orig_mean_coverage'] = orig_adata[df.cell_id].obs.coverage_depth.mean()
    row['mean_coverage'] = orig_adata[df.cell_id].obs.coverage_depth.mean() * int(prop)/100
    
    row['n_cells'] = len(df)
    cells[row['patient'], row['prop'], row['seed']] = df.cell_id.values

    cmat = df.merge(orig_adata[df.cell_id.values].obs.n_wgd, left_on='cell_id', right_index=True)[['n_wgd_x', 'n_wgd_y']].value_counts()    
    for i in range(3):
        for j in range(3):
            row[f'true{i}_inf{j}'] = cmat[i, j] if (i,j) in cmat else 0
            
    cntr = orig_adata[df.cell_id].obs.n_wgd.value_counts()
    row['true_0wgd'] = int(cntr[0]) if 0 in cntr else 0
    row['true_1wgd'] = int(cntr[1]) if 1 in cntr else 0
    row['true_2wgd'] = int(cntr[2]) if 2 in cntr else 0

    cntr = df.n_wgd.value_counts()
    row['inf_0wgd'] = int(cntr[0]) if 0 in cntr else 0
    row['inf_1wgd'] = int(cntr[1]) if 1 in cntr else 0
    row['inf_2wgd'] = int(cntr[2]) if 2 in cntr else 0
    
    rows.append(row)

    # add a 100% accurate row
    gt_row = row.copy()
    gt_row['inf_0wgd'] = row['true_0wgd']
    gt_row['inf_1wgd'] = row['true_1wgd']
    gt_row['inf_2wgd'] = row['true_2wgd']
    gt_row['true0_inf0'] = row['true_0wgd']
    gt_row['true0_inf1'] = 0
    gt_row['true0_inf2'] = 0
    gt_row['true1_inf0'] = 0
    gt_row['true1_inf1'] = row['true_1wgd']
    gt_row['true1_inf2'] = 0
    gt_row['true2_inf0'] = 0
    gt_row['true2_inf1'] = 0
    gt_row['true2_inf2'] = row['true_2wgd']
    gt_row['seed'] = 0
    gt_row['prop'] = 1
    rows.append(gt_row)

result = pd.DataFrame(rows).drop_duplicates()
result = result.fillna(0)
```

```python
result['accuracy'] = (result.true0_inf0 + result.true1_inf1 + result.true2_inf2) / result.n_cells
result['tp'] = (result.true1_inf1 + result.true2_inf2 + result.true2_inf1 + result.true1_inf2).astype(int)
result['tn'] = (result.true0_inf0).astype(int)
result['fp'] = (result.true0_inf1 + result.true0_inf2).astype(int)
result['fn'] = (result.true1_inf0 + result.true2_inf0).astype(int)
result['recall'] = result.tp / (result.tp + result.fn)
result['fp_rate'] = result.fp / result.n_cells

result['precision'] = result.tp / (result.tp + result.fp)
result['n_true_wgd'] = result.true1_inf1
result = result.sort_values(by = ['patient', 'prop', 'seed'])
```

# check WGD status for different cells

```python
plt.figure(figsize=(14,5), dpi = 150)
plt.subplot(1, 3, 1)
sns.lineplot(data=result, x='prop', y='accuracy', hue='patient', marker = '.')
plt.ylim(0, 1.05)
plt.xlabel("Prop. of original coverage")

plt.subplot(1, 3, 2)
sns.lineplot(data=result, x='prop', y='precision', hue='patient', marker = '.')
plt.ylim(0, 1.05)
plt.xlabel("Prop. of original coverage")

plt.subplot(1, 3, 3)
sns.lineplot(data=result, x='prop', y='recall', hue='patient', marker = '.')
plt.ylim(0, 1.05)
plt.xlabel("Prop. of original coverage")

plt.tight_layout()
```

```python
result[result.precision < 0.92].iloc[:, :20]
```

```python
plt.figure(figsize=(14,5), dpi = 150)
plt.subplot(1, 3, 1)
sns.scatterplot(data=result, x='mean_coverage', y='accuracy', hue='patient')
plt.ylim(0, 1.05)
plt.xlabel("Mean per-cell coverage")

plt.subplot(1, 3, 2)
sns.scatterplot(data=result, x='mean_coverage', y='precision', hue='patient')
plt.ylim(0, 1.05)
plt.xlabel("Mean per-cell coverage")

plt.subplot(1, 3, 3)
sns.scatterplot(data=result, x='mean_coverage', y='recall', hue='patient')
plt.ylim(0, 1.05)
plt.xlabel("Mean per-cell coverage")

plt.tight_layout()
```

```python
plt.figure(figsize=(8,4), dpi=300)
plt.subplot(1,2,1)
sns.lineplot(data=result, x='prop', y='fp', hue='patient')
plt.ylabel("False positive WGD cells")

plt.subplot(1,2,2)

sns.scatterplot(data=result, x='mean_coverage', y='fp', hue='patient')
plt.xlabel("Mean per-cell coverage")

plt.ylabel("False positive WGD cells")
plt.tight_layout()
```

```python
plt.figure(figsize=(10,4), dpi = 300)
plt.subplot(1, 2, 1)
sns.scatterplot(data=result, x='mean_coverage', y='accuracy', hue='patient', size = 'n_cells')
plt.ylim(0, 1.05)
plt.xlabel("Mean per-cell coverage")
plt.legend(*plt.gca().get_legend_handles_labels(), bbox_to_anchor=(1,1))

plt.subplot(1, 2, 2)
sns.scatterplot(data=result, x='mean_coverage', y='fp', hue='patient', size = 'n_cells')
plt.xlabel("Mean per-cell coverage")
plt.ylabel("False positive WGD cells")
plt.gca().get_legend().remove()

plt.tight_layout()
plt.subplot(1, 2, 1)
plt.text(-0.15, 1.05, 'A', weight='heavy', fontsize=30, transform=plt.gca().transAxes)

plt.subplot(1, 2, 2)
plt.text(-0.15, 1.05, 'B', weight='heavy', fontsize=30, transform=plt.gca().transAxes)

```

```python
set([p for p in result.patient if p not in colors['patient_id'].keys()])
```

```python
result.mean_coverage.min()
```

```python
result['has_fp_wgd'] = result.fp > 0
sns.kdeplot(data=result, x ='mean_coverage', hue='has_fp_wgd')
```

```python
result['multiple_fp'] = result.fp > 1
```

```python
result['has_fp_wgd'] = result.fp > 0
sns.kdeplot(data=result, x ='mean_coverage', hue='multiple_fp')
```

```python
result[result.has_fp_wgd].mean_coverage.max()
```

```python
result[result.multiple_fp].mean_coverage.max()
```

```python
plt.figure(figsize=(10,4), dpi = 150)
plt.subplot(1, 2, 1)
sns.lineplot(data=result, x='prop', y='accuracy', hue='patient')
plt.ylim(0, 1.05)
plt.xlabel("Proportion of original coverage")
plt.ylabel("Accuracy")
plt.legend(*plt.gca().get_legend_handles_labels(), bbox_to_anchor=(1,1))

plt.subplot(1, 2, 2)
sns.lineplot(data=result, x='prop', y='fp', hue='patient')
plt.xlabel("Proportion of original coverage")
plt.ylabel("False positive WGD cells")
plt.gca().get_legend().remove()

plt.tight_layout()
```

# look at mixed WGD statuses

```python
result['gt_mixed_wgd'] = (result[['true_0wgd', 'true_1wgd', 'true_2wgd']] > 0).sum(axis = 1) > 1
result['gt_nonplurality_cells'] = result[['true_0wgd', 'true_1wgd', 'true_2wgd']].sum(axis = 1) - result[['true_0wgd', 'true_1wgd', 'true_2wgd']].max(axis = 1)

result['inf_mixed_wgd'] = (result[['inf_0wgd', 'inf_1wgd', 'inf_2wgd']] > 0).sum(axis = 1) > 1
result['inf_nonplurality_cells'] = result[['inf_0wgd', 'inf_1wgd', 'inf_2wgd']].sum(axis = 1) - result[['inf_0wgd', 'inf_1wgd', 'inf_2wgd']].max(axis = 1)
```

```python
result[['gt_mixed_wgd', 'inf_mixed_wgd']].value_counts()
```

```python
result[~result.gt_mixed_wgd & result.inf_mixed_wgd].patient.unique()
```

```python
result[result.patient == 'SPECTRUM-OV-065'][['gt_mixed_wgd', 'inf_mixed_wgd']].value_counts()
```

```python
result[(result.patient == 'SPECTRUM-OV-065') & result.inf_mixed_wgd].iloc[:, :15]
```

```python
result[(result.patient == 'SPECTRUM-OV-065') & result.inf_mixed_wgd].inf_nonplurality_cells
```

```python
result[result.patient == 'SPECTRUM-OV-065'][['gt_mixed_wgd', 'inf_mixed_wgd']].inf_mixed_wgd.mean()
```

```python
result['gt_maj_wgd'] = result.true_1wgd + result.true_2wgd > result.true_0wgd
```

```python
sns.jointplot(data=result, x = 'gt_nonplurality_cells', y='inf_nonplurality_cells', hue = 'prop')

```

```python
result['n_nonplurality_diff'] = result.inf_nonplurality_cells - result.gt_nonplurality_cells
result['n_nonplurality_ratio'] = result.inf_nonplurality_cells / result.gt_nonplurality_cells

```

```python
sns.scatterplot(data=result, x='mean_coverage', y = 'n_nonplurality_diff', hue = 'patient')
```

# check coverage in original data

```python
cell_info = pd.read_csv(os.path.join(pipeline_outputs, 'preprocessing/summary/filtered_cell_table.csv.gz'))
```

```python
cell_info[cell_info.include_cell].groupby('patient_id').coverage_depth.mean()
```

```python
cell_info[cell_info.include_cell].groupby('patient_id').coverage_depth.mean().hist(bins=20)
```

```python
cell_info[cell_info.include_cell].groupby('patient_id').coverage_depth.mean().min()
```

```python
len(cell_info.patient_id.unique())
```

```python
sorted(cell_info[cell_info.include_cell].groupby('patient_id').coverage_depth.mean())
```
