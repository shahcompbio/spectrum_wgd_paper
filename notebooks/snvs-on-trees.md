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

```

```python
tree = trees['SPECTRUM-OV-025']
[c for c in tree.find_clades()]
```

```python

```

<!-- #raw -->

    #data['tcn'] = data.ascn.str.split(':', expand=True).astype(int).sum(axis=1)
    #data['expected_vaf'] = np.round((np.minimum(data.cn_state_a, data.clade_cn_a) / data.clade_tcn), 2)
    # variant copy number is 1 except for SNVs that are assigned before a WGD event
    data['variant_cn'] = 1
    
    # pre-WGD mutations have 2 variant copies
    for leaf in tree.get_terminals():
        clone = leaf.name
        path = [tree.root] + tree.get_path(clone)
        prewgd_branches = []
        wgd_branch = None
        i = 0
        while i < len(path) and not path[i].is_wgd:
            prewgd_branches.append(path[i])
            i += 1
        if i < len(path):
            wgd_branch = path[i]
    
        for c in prewgd_branches:
            data.loc[(data.leaf == clone) & (data.clade == c.name), 'variant_cn'] = 2
        data.loc[(data.leaf == clone) & (data.clade == wgd_branch.name) & (data.wgd_timing == 'prewgd'), 'variant_cn'] = 2
    data.variant_cn = np.minimum(np.maximum(data.cn_state_a, data.cn_state_b), data.variant_cn)    
    data['expected_vaf'] = np.round((data.variant_cn/ data.clade_tcn), 2)
    
<!-- #endraw -->

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
    clade2wgd = {c.name:c.n_wgd for c in trees[p].find_clades()}
    clade2wgd['none'] = 0
    
    data['n_wgd'] = data.clade.map(clade2wgd)
    data['clade_cn_a'] = [compatible_cn_types[r.ascn][r.n_wgd]['Maj'] for _, r in data.iterrows()]
    data['clade_cn_b'] = [compatible_cn_types[r.ascn][r.n_wgd]['Min'] for _, r in data.iterrows()]
    data['clade_ascn'] = data.clade_cn_a.astype(str) + ':' + data.clade_cn_b.astype(str)
    data['clade_tcn'] = data.clade_cn_a + data.clade_cn_b
    data['expected_vaf'] = np.round((np.minimum(data.cn_state_a, data.clade_cn_a) / data.clade_tcn), 2)

    #data['tcn'] = data.ascn.str.split(':', expand=True).astype(int).sum(axis=1)
    #data['expected_vaf'] = np.round(data.cn_state_a / data.tcn, 2)
    data['ASCN'] = data['ascn']
    data['present'] = data.expected_vaf > 0

```

# show present vs. absent for each clade

```python
i = 0
for p, data in sorted(datas.items()):
    if len(data.clade.unique()) <= 2:
        continue
    
    data['ASCN'] = data['ascn']
    data['present'] = data.expected_vaf > 0
    
    tree = trees[p]
    g = sns.FacetGrid(col='clade', row='present', data=data[data.clade != 'none'], sharey=False, hue='ASCN',
                 height=2.5, aspect=1)
    g.map_dataframe(sns.histplot, x='vaf', bins=20, binrange=(0, 1))
    g.set_titles(template='{col_name}')
    g._axes[0][0].set_ylabel('Absent SNVs')
    g._axes[1][0].set_ylabel('Present SNVs')
    g.fig.supylabel('Count')
    col2title = {}
    for i, a in enumerate(g._axes[0]):
        t = a.get_title()
        col2title[i] = t
        if t == 'internal_0':
            title = 'root'
        elif t.startswith('internal'):
            clade = [c for c in tree.find_clades(t)][0]
            title = '/'.join([c.name for c in clade.get_terminals()])
        else:
            title = t
        a.set_title(f'{title} (n={len(data[data.present & (data.clade == t)].snv_id.unique())})')
    for a in g._axes[1]:
        a.set_title('')
    labels2handles = {}
    for row in g._axes:
        for a in row[1:]:
            a.set_ylabel('')
        for a in row:
            handles, labels = a.get_legend_handles_labels()
            for l, h in zip(labels, handles):
                labels2handles[l] = h
    all_labels, all_handles = zip(*sorted(labels2handles.items()))
    g._axes[0][0].legend(title='ASCN', handles=all_handles, labels=all_labels)
    plt.suptitle(p)
    plt.tight_layout()

    i += 1
    if i > 3:
        break

```

# final version
* clades split by WGD status
* expected VAF shown with dotted line

```python
from matplotlib.backends.backend_pdf import PdfPages
```

```python
i = 0
with PdfPages('../../figures/final/snvs-by-clade.pdf') as pdf:
            
    for p, data in sorted(datas.items()):
        data = data[data.clade != 'none'].copy()
        data['ASCN'] = data['ascn']
        data['present'] = data.expected_vaf > 0

        # add WGD timing to clade
        data.loc[data.wgd_timing != 'none', 'clade'] = data.loc[data.wgd_timing != 'none', 'clade'] + '/' + data.loc[data.wgd_timing != 'none', 'wgd_timing']
        g = sns.FacetGrid(col='leaf', row='clade', data=data, sharey=False, hue='ASCN',
                         height=4, aspect=1)
        g.map_dataframe(sns.histplot, x='vaf', bins=20, binrange=(0, 1))
    
        labels2handles = {}
        for row in g._axes:
            for ax in row:    
                handles, labels = ax.get_legend_handles_labels()
                for l, h in zip(labels, handles):
                    labels2handles[l] = h
    
        added_legend = False
        for row in g._axes:
            for ax in row:
                all0_evaf = True
                t = ax.get_title()
                leaf = t.split()[-1]
                clade = t.split()[2]
        
                ax.set_title(f"{clade} SNVs in {leaf}")
                my_snvs = data[(data.leaf == leaf) & (data.clade == clade)]
                
                _, ymax = ax.get_ylim()
                for ascn, df in my_snvs.groupby('ascn'):
                    evaf = df.expected_vaf.unique()
                    if (evaf > 0).any():
                        all0_evaf = False
                    if len(evaf) == 1 and evaf == 0.67 and 'prewgd' in clade:
                        evaf = [0.33, 0.67]
                    ax.vlines(x = evaf, ymin=0, ymax=ymax, 
                              color=labels2handles[ascn][0].get_facecolor(), linestyle = ':',
                             linewidth=2)
                if all0_evaf and not added_legend:
                    legend_ax = ax
                    added_legend = True
        if not added_legend:
            legend_ax = ax
                    
        legend_ax.legend(title='ASCN')
        plt.tight_layout()
        plt.suptitle(p, va='bottom', y = 1.02)
        pdf.savefig(bbox_inches='tight', pad_inches=0.5)
        #plt.close()
                            
    
        i += 1
        if i > 5:
            pass
            #break
        #break
```
