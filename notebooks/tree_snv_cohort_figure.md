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

```python tags=["remove-cell"]
import itertools
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import anndata as ad
import Bio.Phylo
import numpy as np
from copy import deepcopy
import wgs_analysis.snvs.mutsig
import sys

import os
import tqdm
```
```python
pipeline_outputs = pipeline_dir # path to root directory of scWGS pipeline outputs

```

# put together plots for many patients

```python
datas = {}
adatas = {}
trees = {}

patients = []
for f in tqdm.tqdm(os.listdir(os.path.join(pipeline_outputs, 'tree_snv', 'outputs'))):
    p = f.split('_')[0]
    datas[p] = pd.read_csv(os.path.join(pipeline_outputs, 'tree_snv', 'outputs', f))
    patients.append(p)
    
for patient_id in tqdm.tqdm(patients):
    adatas[patient_id] = ad.read_h5ad(os.path.join(pipeline_outputs, 'tree_snv', 'inputs', f'{patient_id}_general_clone_adata.h5'))
    trees[patient_id] = pickle.load(open(os.path.join(pipeline_outputs, 'tree_snv', 'inputs', f'{patient_id}_clones_pruned.pickle'), 'rb'))
    
```

```python
for p in patients:
    data = datas[p]
    g = sns.FacetGrid(col='cn_state_a', data=data, sharey=False, hue='ascn')
    g.map_dataframe(sns.histplot, x='vaf', bins=20, binrange=(0, 1))
    g.add_legend()
    plt.suptitle(p)
    plt.tight_layout()
    break
```

```python
# "ascn" is really just a key -- need to reason about the expected CN state in each leaf depending on WGD status

[c for c in trees[p].find_clades()]
```

```python
clade2wgd = {c.name:c.n_wgd for c in trees[p].find_clades()}
clade2wgd['none'] = 0
```

```python
data['n_wgd'] = data.clade.map(clade2wgd)
```

```python
compatible_cn_types = {
    '1:0': {1:{'Maj': 1, 'Min': 0}, 0:{'Maj': 1, 'Min': 0}},
    '2:0': {1:{'Maj': 2, 'Min': 0}, 0:{'Maj': 1, 'Min': 0}},
    '1:1': {1:{'Maj': 1, 'Min': 1}, 0:{'Maj': 1, 'Min': 1}},
    '2:1': {1:{'Maj': 2, 'Min': 1}, 0:{'Maj': 1, 'Min': 1}},
    '2:2': {1:{'Maj': 2, 'Min': 2}, 0:{'Maj': 1, 'Min': 1}},
}
```

```python
data['clade_cn_a'] = [compatible_cn_types[r.ascn][r.n_wgd]['Maj'] for _, r in data.iterrows()]
data['clade_cn_b'] = [compatible_cn_types[r.ascn][r.n_wgd]['Min'] for _, r in data.iterrows()]
data['clade_ascn'] = data.clade_cn_a.astype(str) + ':' + data.clade_cn_b.astype(str)
data['clade_tcn'] = data.clade_cn_a + data.clade_cn_b
```

```python
#data['tcn'] = data.ascn.str.split(':', expand=True).astype(int).sum(axis=1)
data['expected_vaf'] = data.cn_state_a / data.clade_tcn
```

```python
data[['ascn', 'cn_state_a', 'cn_state_b']].value_counts()
```

```python
data[['cn_state_idx', 'cn_state_a', 'cn_state_b', 'ascn', 'clade']].value_counts()
```

```python
g = sns.FacetGrid(col='expected_vaf', data=data, sharey=False, hue='clade_ascn')
g.map_dataframe(sns.histplot, x='vaf', bins=20, binrange=(0, 1))
g.add_legend()
plt.suptitle(f'{p} new')
plt.tight_layout()
```

```python
# try applying to more patients
```

```python
compatible_cn_types = {
    '1:0': {1:{'Maj': 1, 'Min': 0}, 0:{'Maj': 1, 'Min': 0}},
    '2:0': {1:{'Maj': 2, 'Min': 0}, 0:{'Maj': 1, 'Min': 0}},
    '1:1': {1:{'Maj': 1, 'Min': 1}, 0:{'Maj': 1, 'Min': 1}},
    '2:1': {1:{'Maj': 2, 'Min': 1}, 0:{'Maj': 1, 'Min': 1}},
    '2:2': {1:{'Maj': 2, 'Min': 2}, 0:{'Maj': 1, 'Min': 1}},
}
```

```python
for p, data in sorted(datas.items()):
    if len(data) == 0:
        continue 
        
    clade2wgd = {c.name:c.n_wgd for c in trees[p].find_clades()}
    clade2wgd['none'] = 0
    
    data['n_wgd'] = data.clade.map(clade2wgd)
    data['clade_cn_a'] = [compatible_cn_types[r.ascn][r.n_wgd]['Maj'] for _, r in data.iterrows()]
    data['clade_cn_b'] = [compatible_cn_types[r.ascn][r.n_wgd]['Min'] for _, r in data.iterrows()]
    data['clade_ascn'] = data.clade_cn_a.astype(str) + ':' + data.clade_cn_b.astype(str)
    data['clade_tcn'] = data.clade_cn_a + data.clade_cn_b
    
    #data['tcn'] = data.ascn.str.split(':', expand=True).astype(int).sum(axis=1)
    data['expected_vaf'] = np.round((np.minimum(data.cn_state_a, data.clade_cn_a) / data.clade_tcn), 2)

    g = sns.FacetGrid(col='expected_vaf', data=data, sharey=False, hue='clade_ascn')
    g.map_dataframe(sns.histplot, x='vaf', bins=20, binrange=(0, 1))
    for ax in g.axes[0]: 
        ymin, ymax = ax.get_ylim()
        xval = float(ax.get_title().split()[-1])
        if xval == 0.67:
            xval = [0.33, 0.67]
        ax.vlines(ymin=ymin, ymax=ymax, x=xval, colors = 'k')
    
    g.add_legend()
    plt.suptitle(f'{p} new')
    plt.tight_layout()
    
```

```python
for p, data in sorted(datas.items()):
    if len(data) == 0:
        continue 
        
    clade2wgd = {c.name:c.n_wgd for c in trees[p].find_clades()}
    clade2wgd['none'] = 0
    
    data['n_wgd'] = data.clade.map(clade2wgd)
    data['clade_cn_a'] = [compatible_cn_types[r.ascn][r.n_wgd]['Maj'] for _, r in data.iterrows()]
    data['clade_cn_b'] = [compatible_cn_types[r.ascn][r.n_wgd]['Min'] for _, r in data.iterrows()]
    data['clade_ascn'] = data.clade_cn_a.astype(str) + ':' + data.clade_cn_b.astype(str)
    data['clade_tcn'] = data.clade_cn_a + data.clade_cn_b
    
    #data['tcn'] = data.ascn.str.split(':', expand=True).astype(int).sum(axis=1)
    data['expected_vaf'] = np.round((np.minimum(data.cn_state_a, data.clade_cn_a) / data.clade_tcn), 2)

    g = sns.FacetGrid(col='expected_vaf', data=data, sharey=False, hue='clade_ascn')
    g.map_dataframe(sns.scatterplot, x='vaf', y = 'total_counts')
    for ax in g.axes[0]: 
        ymin, ymax = ax.get_ylim()
        xval = float(ax.get_title().split()[-1])
        if xval == 0.67:
            xval = [0.33, 0.67]
        ax.vlines(ymin=ymin, ymax=ymax, x=xval, colors = 'k')
    g.add_legend()
    plt.suptitle(f'{p} new')
    plt.tight_layout()
    
```

```python
p = 'SPECTRUM-OV-025'
data = datas[p]

clade2wgd = {c.name:c.n_wgd for c in trees[p].find_clades()}
clade2wgd['none'] = 0

data['n_wgd'] = data.clade.map(clade2wgd)
data['clade_cn_a'] = [compatible_cn_types[r.ascn][r.n_wgd]['Maj'] for _, r in data.iterrows()]
data['clade_cn_b'] = [compatible_cn_types[r.ascn][r.n_wgd]['Min'] for _, r in data.iterrows()]
data['clade_ascn'] = data.clade_cn_a.astype(str) + ':' + data.clade_cn_b.astype(str)
data['clade_tcn'] = data.clade_cn_a + data.clade_cn_b

#data['tcn'] = data.ascn.str.split(':', expand=True).astype(int).sum(axis=1)
data['expected_vaf'] = np.round((np.minimum(data.cn_state_a, data.clade_cn_a) / data.clade_tcn), 2)

g = sns.FacetGrid(col='expected_vaf', data=data, sharey=False, hue='clade_ascn')
g.map_dataframe(sns.scatterplot, x='vaf', y = 'total_counts')
for ax in g.axes[0]: 
    ymin, ymax = ax.get_ylim()
    xval = float(ax.get_title().split()[-1])
    if xval == 0.67:
        xval = [0.33, 0.67]
    ax.vlines(ymin=ymin, ymax=ymax, x=xval, colors = 'k')
g.add_legend()
plt.suptitle(f'{p} new')
plt.tight_layout()

```

# assess branch assignments

```python
tree = trees[p]
data = datas[p]
```

```python
Bio.Phylo.draw(tree)
```

```python
for p in sorted(trees.keys()):
    tree = trees[p]
    data = datas[p]
    if len(data) == 0:
        continue

    internal2leaves = {internal.name:set([a.name for a in [c for c in tree.find_clades(internal.name)][0].get_terminals()])
                       for internal in tree.get_nonterminals()}
    for c in tree.get_terminals():
        internal2leaves[c.name] = set([c.name])
    internal2leaves['none'] = set()

    data['snv_present'] = np.logical_or(data.clade == data.leaf, 
                                        [(r.leaf in internal2leaves[r.clade]) for _, r in data.iterrows()])

    
    g = sns.FacetGrid(col='snv_present', data=data, sharey=True, hue='leaf', aspect=0.7)
    g.map_dataframe(sns.histplot, x='vaf', bins=20, binrange=(0, 1))
    g.add_legend(bbox_to_anchor=(1.2,0.5))
    plt.suptitle(p)
    plt.tight_layout()
```

```python
for p in sorted(trees.keys()):
    tree = trees[p]
    data = datas[p]
    if len(data) == 0:
        continue

    internal2leaves = {internal.name:set([a.name for a in [c for c in tree.find_clades(internal.name)][0].get_terminals()])
                       for internal in tree.get_nonterminals()}
    for c in tree.get_terminals():
        internal2leaves[c.name] = set([c.name])
    internal2leaves['none'] = set()

    data['snv_present'] = np.logical_or(data.clade == data.leaf, 
                                        [(r.leaf in internal2leaves[r.clade]) for _, r in data.iterrows()])

    
    g = sns.FacetGrid(col='snv_present', data=data, sharey=True, hue='leaf', aspect=0.7)
    g.map_dataframe(sns.scatterplot, x='vaf', y='total_counts')
    g.add_legend(bbox_to_anchor=(1.2,0.5))
    plt.suptitle(p)
    plt.tight_layout()
```

```python
combined_data = []
for p, data in sorted(datas.items()):
    data['patient_id'] = p
    combined_data.append(data)
combined_data = pd.concat(combined_data)
```

```python
combined_data
```

```python
g = sns.FacetGrid(col='patient_id', data=combined_data[combined_data.snv_present], 
                  sharey=False, hue='leaf', aspect=1, col_wrap=6)
g.map_dataframe(sns.histplot, x='vaf', bins=20, binrange=(0, 1))
#plt.tight_layout()
```

```python
g = sns.FacetGrid(col='patient_id', data=combined_data[~combined_data.snv_present], 
                  sharey=False, hue='leaf', aspect=1, col_wrap=6)
g.map_dataframe(sns.histplot, x='vaf', bins=20, binrange=(0, 1))
#plt.suptitle("VAF for SNVs inferred to be absent in corresponding leaf")
#plt.tight_layout()
```

<!-- #raw -->
combined_data.to_csv('doubletime_combined_data.csv.gz', index=False)
<!-- #endraw -->

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
