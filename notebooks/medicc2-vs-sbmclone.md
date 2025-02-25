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
import Bio.Phylo
from copy import deepcopy
```

# compare SBMClone clustering to other clusterings of cells

```python
pipeline_outputs = '/data1/shahs3/users/myersm2/repos/spectrum_wgd_data5'

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
trees = {}
for p in adatas.keys():
    trees[p] = Bio.Phylo.read(
        os.path.join(pipeline_outputs, f'medicc/output/{p}__neither/{p}__neither_final_tree.new'), 'newick')

```

```python
p = 'SPECTRUM-OV-002'
tree = trees[p]
adata = adatas[p]

all_cells = set([c.name for c in tree.get_terminals()]) - set(['diploid'])
```

```python
def check_ari(clade):    
    ingroup = set([a.name for a in clade.get_terminals()])
    outgroup = all_cells - ingroup
    
    sbmclone_clustering = []
    tree_clustering = []
    for _, r in adata.obs.iterrows():
        tree_clustering.append(1 if r.brief_cell_id in ingroup else 0)
        sbmclone_clustering.append(r.sbmclone_cluster_id)
    return adjusted_rand_score(sbmclone_clustering, tree_clustering)

```

```python
def check_purity(clade):
    my_cells = [c.name for c in clade.get_terminals()]
    vc = adata[adata.obs.brief_cell_id.isin(my_cells)].obs.sbmclone_cluster_id.value_counts()
    props = vc / vc.sum()
    return props
```

```python
props.reset_index()
```

```python
max_purity_clades = {}
for c in tqdm.tqdm(tree.find_clades()):
    props = check_purity(c).reset_index()
    for _, r in props.iterrows():
        cl = r.sbmclone_cluster_id
        pur = r['count']
        if cl not in max_purity_clades or max_purity_clades[cl] < pur:
            max_purity_clades[cl] = pur
```

```python
max_purity_clades
```

# find the tree ordering that maximizes ARI and return the ARI

```python
def rebalance_tree(tree, adata):
    to_visit = set([tree.root])
    while len(to_visit) > 0:
        c = to_visit.pop()
        if len(c.clades) > 1:
            # check proportion clone1 in each subtree
            weight_l = adata.obs.loc[[a.name for a in c.clades[0].get_terminals()], 'sbmclone_cluster_id'].astype(int).mean()
            weight_r = adata.obs.loc[[a.name for a in c.clades[1].get_terminals()], 'sbmclone_cluster_id'].astype(int).mean()
            n_l = len(c.clades[0].get_terminals())
            n_r = len(c.clades[1].get_terminals())
            
            if n_l > 1 and weight_l > weight_r:
                c.clades = [c.clades[1], c.clades[0]]
            to_visit.add(c.clades[0])
            to_visit.add(c.clades[1])
```

```python
def tree_ari(tree, adata):
    tree = deepcopy(tree)
    tree.prune('diploid')

    adata0 = adata.copy()
    adata0.obs = adata0.obs.set_index('brief_cell_id')
    rebalance_tree(tree, adata0)
        
    cells = [a.name for a in tree.get_terminals()]
    tree_cl = adata0.obs.loc[cells, 'sbmclone_cluster_id'].values
    cl1 = sorted(tree_cl)
    return adjusted_rand_score(tree_cl, cl1)
```

```python
aris = {}
for p, tree in tqdm.tqdm(trees.items()):
    adata = adatas[p]

    aris[p] = tree_ari(tree, adata)
```

# analyze ARI in context of other features

```python
sigs = pd.read_table('/data1/shahs3/users/myersm2/repos/spectrumanalysis/annotations/mutational_signatures.tsv').set_index('patient_id')
```

```python
df = pd.DataFrame()
df['patient'], df['ari'] = zip(*sorted(aris.items()))
df['prop_wgd'] = cell_info.groupby('patient_id').is_wgd.mean().loc[df.patient].values
df['n_clones'] = [len(adatas[p].obs.sbmclone_cluster_id.unique()) for p in df.patient]
df['signature'] = sigs.loc[df.patient, 'consensus_signature'].values
```

```python
plt.figure(figsize=(8,4), dpi = 200)
plt.subplot(1, 2, 1)
sns.boxplot(data=df, x = 'n_clones', y = 'ari', zorder=-1)
sns.stripplot(data=df, x = 'n_clones', y = 'ari', zorder=10, linewidth=0.5, edgecolor='k')
plt.ylabel("ARI")
plt.xlabel("Number of SBMClone clones")

plt.subplot(1, 2, 2)
sns.scatterplot(df, x = 'prop_wgd', y='ari', hue = 'signature')
plt.ylabel("ARI")
plt.xlabel("Proportion of WGD cells")
plt.suptitle("MEDICC2 tree vs. SBMClone clones")
plt.tight_layout()
```

```python

```

```python
df[df.n_clones > 1].ari.hist()
```

```python
df[df.n_clones > 1].ari.mean(), df[df.n_clones > 1].ari.std()
```

```python
df.to_csv('medicc2_tree_vs_sbmclone_clones.csv', index=False)
```

```python

```

```python

```

```python

```

```python

```

```python

```

```python

```
