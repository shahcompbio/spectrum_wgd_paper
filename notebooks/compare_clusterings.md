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

from sklearn.metrics import adjusted_rand_score
from IPython.display import Image
import matplotlib.cm as cm
```

# compare SBMClone clustering to other clusterings of cells

```python
pipeline_outputs = pipeline_dir # path to root directory of scWGS pipeline outputs

```

```python
cell_info = pd.read_csv(os.path.join(pipeline_outputs, 'preprocessing/summary/filtered_cell_table.csv.gz'))
cell_info = cell_info[cell_info.include_cell & ~cell_info.is_normal & ~cell_info.is_s_phase_thresholds].copy()

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
```

```python

```

```python
smallest_clone_size = []
for p, adata in sorted(adatas.items()):
    if adata.obs.is_wgd.astype(int).mean() > 0.5:
        print(p, np.min(adata.obs.sbmclone_cluster_id.value_counts()))
        smallest_clone_size.append(np.min(adata.obs.sbmclone_cluster_id.value_counts()))
```

```python
plt.hist(smallest_clone_size, bins=50)
```

```python
aris = []
for p, adata in adatas.items():
    sbmclone = adata.obs.sbmclone_cluster_id
    cluster = adata.obs.cluster_id
    
    aris.append({'patient_id':p,
                'n_blocks':len(sbmclone.unique()),
                 'n_clusters':len(cluster.unique()),
                 'ari_cluster':adjusted_rand_score(sbmclone, cluster)})
    
aris = pd.DataFrame(aris)
```

# try using silhouette score
as in https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html

```python
from sklearn.metrics import silhouette_samples
```

```python
len([a for a in adatas.values() if len(a.obs.sbmclone_cluster_id.unique()) > 1])
```

```python
n_rows = 7
n_cols = 5
fig, axes = plt.subplots(n_rows, n_cols, dpi = 300, figsize = (9, 12))

idx = 0
for p, adata in sorted(adatas.items()):
    if len(adata.obs.sbmclone_cluster_id.unique()) > 1:
        ax1 = axes[int(idx // n_cols)][idx % n_cols]

        profiles = np.concatenate([adata[:, adata.var.has_allele_cn].layers['A'],
                adata[:, adata.var.has_allele_cn].layers['B']], axis = 1)
        
        sample_silhouette_values = silhouette_samples(profiles, adata.obs.sbmclone_cluster_id)
        cluster_labels = adata.obs.sbmclone_cluster_id.values
        n_clusters = len(adata.obs.sbmclone_cluster_id.unique())
                
        y_lower = 10
        for i in range(n_clusters):
            # Aggregate the silhouette scores for samples belonging to
            # cluster i, and sort them
            ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]
        
            ith_cluster_silhouette_values.sort()
        
            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i
        
            color = cm.nipy_spectral(float(i) / n_clusters)
            ax1.fill_betweenx(
                np.arange(y_lower, y_upper),
                0,
                ith_cluster_silhouette_values,
                facecolor=color,
                edgecolor=color,
                alpha=0.7,
            )
        
            # Label the silhouette plots with their cluster numbers at the middle
            ax1.text(-0.06, y_lower + 0.5 * size_cluster_i, str(i))
        
            # Compute the new y_lower for next plot
            y_lower = y_upper + 10  # 10 for the 0 samples
        ax1.set_title(p)
        ax1.set_xlabel("Silhouette score")

        idx += 1

for i in range(idx, n_rows * n_cols):
    axes[int(i // n_cols)][i % n_cols].set_visible(False)

plt.suptitle("Euclidean distance, copy-number profiles")

plt.tight_layout()
```

```python

```

## try using cell densities for silhouette

```python
for p, adata in tqdm.tqdm(sbmclone_adatas.items()):
    adata = sbmclone_adatas[p]
    
    cell_densities = {}
    for bl, df in adata.var.groupby('block_assignment'):
        cell_densities[bl] = np.array(adata[:, df.index].layers['alt'].mean(axis=1)).flatten()
    adata.obsm['block_density'] = pd.DataFrame(cell_densities).set_index(adata.obs.index)
    

```

```python

fig, axes = plt.subplots(8, 4, dpi = 300, figsize = (10, 20))
idx = 0
for p, adata in sorted(sbmclone_adatas.items()):
    if len(adata.obs.sbmclone_cluster_id.unique()) > 1:
        ax1 = axes[int(idx // 4)][idx % 4]
    
        sample_silhouette_values = silhouette_samples(adata.layers['alt'], adata.obs.sbmclone_cluster_id)
        cluster_labels = adata.obs.sbmclone_cluster_id.values
        n_clusters = len(adata.obs.sbmclone_cluster_id.unique())
                
        y_lower = 10
        for i in range(n_clusters):
            # Aggregate the silhouette scores for samples belonging to
            # cluster i, and sort them
            ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]
        
            ith_cluster_silhouette_values.sort()
        
            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i
        
            color = cm.nipy_spectral(float(i) / n_clusters)
            ax1.fill_betweenx(
                np.arange(y_lower, y_upper),
                0,
                ith_cluster_silhouette_values,
                facecolor=color,
                edgecolor=color,
                alpha=0.7,
            )
        
            # Label the silhouette plots with their cluster numbers at the middle
            ax1.text(0, y_lower + 0.5 * size_cluster_i, str(i))
        
            # Compute the new y_lower for next plot
            y_lower = y_upper + 10  # 10 for the 0 samples
        ax1.set_title(p)
        ax1.set_xlabel("Silhouette score")

        idx += 1
        
plt.suptitle("Euclidean distance, raw alt counts")

plt.tight_layout()
```

```python
n_rows = 7
n_cols = 5
fig, axes = plt.subplots(n_rows, n_cols, dpi = 300, figsize = (9, 12))

idx = 0
for p, adata in sorted(sbmclone_adatas.items()):
    if len(adata.obs.sbmclone_cluster_id.unique()) > 1:
        ax1 = axes[int(idx // n_cols)][idx % n_cols]
    
        sample_silhouette_values = silhouette_samples(adata.obsm['block_density'], adata.obs.sbmclone_cluster_id)
        cluster_labels = adata.obs.sbmclone_cluster_id.values
        n_clusters = len(adata.obs.sbmclone_cluster_id.unique())
                
        y_lower = 10
        for i in range(n_clusters):
            # Aggregate the silhouette scores for samples belonging to
            # cluster i, and sort them
            ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]
        
            ith_cluster_silhouette_values.sort()
        
            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i
        
            color = cm.nipy_spectral(float(i) / n_clusters)
            ax1.fill_betweenx(
                np.arange(y_lower, y_upper),
                0,
                ith_cluster_silhouette_values,
                facecolor=color,
                edgecolor=color,
                alpha=0.7,
            )
        
            # Label the silhouette plots with their cluster numbers at the middle
            ax1.text(0, y_lower + 0.5 * size_cluster_i, str(i))
        
            # Compute the new y_lower for next plot
            y_lower = y_upper + 10  # 10 for the 0 samples
        ax1.set_title(p)
        ax1.set_xlabel("Silhouette score")

        idx += 1

for i in range(idx, n_rows * n_cols):
    axes[int(i // n_cols)][i % n_cols].set_visible(False)


plt.suptitle("Euclidean distance, block densities")
plt.tight_layout()
```

## try intersection/union distance

```python
Ds = {}
for p, adata in tqdm.tqdm(sorted(sbmclone_adatas.items())):
    D = np.zeros((len(adata), len(adata)))
    snvs = {}
    for i in range(1, len(adata)):
        for j in range(i):
            if i in snvs:
                snvs_i = snvs[i]
            else:
                snvs_i = set(np.where(adata.layers['alt'][i].toarray() > 0)[1])
                snvs[i] = snvs_i
            if j in snvs:
                snvs_j = snvs[j]
            else:
                snvs_j = set(np.where(adata.layers['alt'][j].toarray()  > 0)[1])
                snvs[j] = snvs_j

            D[i][j] = len(snvs_i.intersection(snvs_j)) / len(snvs_i.union(snvs_j))
            D[j][i] = D[i][j]
        Ds[p] = D
```

```python

fig, axes = plt.subplots(8, 4, dpi = 300, figsize = (10, 20))
idx = 0
for p, adata in sorted(sbmclone_adatas.items()):
    if len(adata.obs.sbmclone_cluster_id.unique()) > 1:
        ax1 = axes[int(idx // 4)][idx % 4]
    
        sample_silhouette_values = silhouette_samples(Ds[p], adata.obs.sbmclone_cluster_id,
                                                     metric='precomputed')
        cluster_labels = adata.obs.sbmclone_cluster_id.values
        n_clusters = len(adata.obs.sbmclone_cluster_id.unique())
                
        y_lower = 10
        for i in range(n_clusters):
            # Aggregate the silhouette scores for samples belonging to
            # cluster i, and sort them
            ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]
        
            ith_cluster_silhouette_values.sort()
        
            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i
        
            color = cm.nipy_spectral(float(i) / n_clusters)
            ax1.fill_betweenx(
                np.arange(y_lower, y_upper),
                0,
                ith_cluster_silhouette_values,
                facecolor=color,
                edgecolor=color,
                alpha=0.7,
            )
        
            # Label the silhouette plots with their cluster numbers at the middle
            ax1.text(0, y_lower + 0.5 * size_cluster_i, str(i))
        
            # Compute the new y_lower for next plot
            y_lower = y_upper + 10  # 10 for the 0 samples
        ax1.set_title(p)
        ax1.set_xlabel("Silhouette score")

        idx += 1
plt.suptitle("Intersection/union of detected SNVs")
plt.tight_layout()
```

```python

```
