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
import numpy as np
import anndata as ad
import os
import seaborn as sns
import matplotlib.pyplot as plt
import tqdm
import scgenome 
import scipy
from sklearn.metrics import confusion_matrix
```

```python
agg_dir = 'pipeline_dir/sbmclone'
adatas = {}

for f in tqdm.tqdm(os.listdir(agg_dir)):
    if f.endswith('snv.h5'):
        p = f.split('_')[1]
        adatas[p] = ad.read_h5ad(os.path.join(agg_dir, f))
```

```python
def get_binary(adata, binarization_threshold = 0.01):
    density = np.zeros((len(adata.obs.sbmclone_cluster_id.unique()), len(adata.var.block_assignment.unique())))
    for rb, ridx in adata.obs.groupby('sbmclone_cluster_id'):
        for cb, cidx in adata.var.groupby('block_assignment'):
            density[int(rb), int(cb)] = np.minimum(adata[ridx.index, cidx.index].layers['alt'].todense(), 1).sum() / (len(ridx) * len(cidx))
    
    binary_density = density.copy()
    binary_density[density < binarization_threshold] = 0
    binary_density[density > binarization_threshold] = 1
    return density, binary_density

```

```python
c = 0
Ds = {}
Bs = {}
for p, adata in adatas.items():
    if len(adata.obs.sbmclone_cluster_id.unique()) > 1:
        D, B = get_binary(adatas[p])
        block_sums = {}
        block_densities = {}
        block_sizes = adata.var.groupby('block_assignment').size()
        for cl, var in adata.var.groupby('block_assignment'):
            block_sums[cl] = np.array(adata[:, var.index].layers['alt'].sum(axis=1)).flatten()
            block_densities[cl] = block_sums[cl] / len(var)
        Ds[p] = D
        Bs[p] = B
```

```python
len(Ds)
```

```python
plt.figure(figsize=(12,16), dpi = 200)
panels_per_row = 4
panels_per_col = 8
for i, (p, D) in enumerate(sorted(Ds.items())):
    plt.subplot(panels_per_col, panels_per_row, i + 1)
    assert np.sum(D[:, -1]) == 0
    D = D[:, :-1]
    
    sns.heatmap(D, annot=D*100, fmt = '.2f')
    plt.title(p)
    plt.xlabel("SNV blocks")
    plt.ylabel("Clones")
plt.tight_layout()
```

