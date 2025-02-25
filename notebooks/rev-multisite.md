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
import scgenome
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import anndata as ad

import tqdm
from scipy.stats import linregress
import pickle
from yaml import safe_load
import matplotlib.colors as mcolors
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score
from collections import Counter

```

```python
pipeline_outputs = '/data1/shahs3/users/myersm2/repos/spectrum_wgd_data5'
```

```python
colors_dict = safe_load(open('/data1/shahs3/users/myersm2/repos/spectrumanalysis/config/colors.yaml', 'r').read())
sigs = pd.read_table('/data1/shahs3/users/myersm2/repos/spectrumanalysis/annotations/mutational_signatures.tsv').set_index('patient_id').consensus_signature
wgd_colors = {0:mcolors.to_hex((197/255, 197/255, 197/255)),
              1:mcolors.to_hex((252/255, 130/255, 79/255)),
              2:mcolors.to_hex((170/255, 0, 0/255))}
sitemap = {
    'INFRACOLIC_OMENTUM':'Om',
    'PELVIC_PERITONEUM':'Ptn',
    'RIGHT_ADNEXA':'RAdnx',
    'LEFT_ADNEXA':'LAdnx',
    'RIGHT_OVARY':'ROv',
    'LEFT_OVARY':'LOv',
    'PELVIC_IMPLANT':'Ptn',
    'BOWEL':'Bwl',
    'INFRARENAL_LYMPH_NODE':'LN',
    'RIGHT_OMENTUM':'ROm',
    'RIGHT_FALLOPIAN_TUBE':'RFT',
    'LEFT_FALLOPIAN_TUBE':'LFT',
    'RIGHT_OVARY_AND_FALLOPIAN_TUBE':'ROv_RFT',
    'SMALL_BOWEL_MESENTERY':'SBwlMes',
    'RIGHT_UPPER_ABDOMEN':'RUpAb',
}
```

# multisite analysis

```python
cell_info = pd.read_csv(os.path.join(pipeline_outputs, 'preprocessing/summary/filtered_cell_table.csv.gz'))
agg_dir = os.path.join(pipeline_outputs, 'postprocessing/aggregated_anndatas')
signals_dir = os.path.join(pipeline_outputs, 'preprocessing/signals')
```

```python
multipolar_nnd_threshold = 0.31734256961919605


cell_info = pd.read_csv(os.path.join(pipeline_outputs, 'preprocessing/summary/filtered_cell_table.csv.gz'))
cell_info = cell_info[cell_info.include_cell & ~cell_info.is_normal & 
~cell_info.is_s_phase_thresholds].copy()
```

```python
signals_dir = os.path.join(pipeline_outputs, 'preprocessing', 'signals')
signals_adatas = {}
for f in tqdm.tqdm(os.listdir(signals_dir)):
    p = f.split('_')[1].split('.')[0]
    if p.endswith('077') or p.endswith('116'):
        continue
    adata = ad.read_h5ad(os.path.join(signals_dir, f))
    adata.obs['site'] = ['_'.join(a.split('_')[2:]) for a in adata.obs.sample_id]
    adata.obs['brief-site'] = adata.obs.site.map(sitemap)

    assert adata.obs['brief-site'].isna().sum() == 0, f
    signals_adatas[p] = adata


# sbmclone anndatas
sbmclone_dir = os.path.join(pipeline_outputs, 'sbmclone') 
sbmclone_adatas = {}
for p in tqdm.tqdm(signals_adatas.keys()):
    sbmclone_adatas[p] = ad.read_h5ad(os.path.join(sbmclone_dir, f'sbmclone_{p}_snv.h5'))

for p, adata in signals_adatas.items():
    adata = adata[adata.obs.index.isin(cell_info.cell_id)].copy()
    sbmclone_adata = sbmclone_adatas[p]
    adata.obs['sbmclone_cluster_id'] = sbmclone_adata.obs.loc[adata.obs.index, 'block_assignment']
    signals_adatas[p] = adata
```

## which patients have multiple sites?

```python
multisite = {p:adata for p,adata in signals_adatas.items() if len(adata.obs.site.unique()) > 1}
len(multisite)
```

```python
sorted(multisite.keys())
```

## multi-sample vs multi-site vs. multi-aliquot

```python
for p, adata in sorted(multisite.items()):
    adata.obs['site'] = ['_'.join(a.split('_')[2:]) for a in adata.obs.sample_id]
    print(adata.obs[['sample_id', 'aliquot_id', 'site']].value_counts())
    print()
```

# how does WGD vary across sites?

```python
all_ns = {}
all_props = {}
n_sites = {}
all_sites = {}

rows = []

fig = plt.figure(figsize=(12, 12), dpi = 200)
for i, (p, adata) in enumerate(sorted(multisite.items())):
    plt.subplot(4, 6, i + 1)
    wgd_states = adata.obs.n_wgd.unique()
    xs = np.arange(len(adata.obs.site.unique()))
    sites = []
    sample_ids = []
    ns = {}
    props = {}
    for a, adf in adata.obs.groupby('brief-site', observed=True):
        sites.append(a)
        assert len(adf.sample_id.unique()) == 1
        sample_ids.append(adf.sample_id.unique()[0])
        cntr = adf.n_wgd.value_counts()
        for k in wgd_states:
            if k not in ns:
                ns[k] = []
                props[k] = []
            
            ns[k].append(cntr[k] if k in cntr else 0)
            props[k].append(cntr[k]/len(adf) if k in cntr else 0)
        
    bottom = np.zeros(xs.shape[0])
    for k,heights in sorted(props.items()):
        plt.bar(xs, heights, bottom=bottom, label=k, facecolor=wgd_colors[k])
        bottom += heights
    plt.ylim(0, 1)
    plt.legend(title='nWGD')
    plt.title(p)
    plt.xticks(xs, sites)
    all_sites[p] = sites
    all_ns[p] = ns
    all_props[p] = props
    n_sites[p] = len(sites)

    for i, s in enumerate(sites):
        for n_wgd, val in ns.items():
            rows.append({'patient_id':p, 'sample_id':sample_ids[i],
                         'site':s, 'n_wgd':n_wgd, 'num_cells':val[i], 'prop_cells_in_site':props[n_wgd][i]})
fig.supxlabel("Anatomical site")
fig.supylabel("Proportion of cells")
plt.tight_layout()
```

```python
all_props['SPECTRUM-OV-002']
```

```python
props
```

```python
ns
```

```python
cntr
```

```python
rows = []

for i, (p, adata) in enumerate(sorted(signals_adatas.items())):
    wgd_states = adata.obs.n_wgd.unique()
    xs = np.arange(len(adata.obs.site.unique()))
    sites = []
    sample_ids = []
    ns = {}
    props = {}
    for a, adf in adata.obs.groupby('brief-site', observed=True):
        sites.append(a)
        assert len(adf.sample_id.unique()) == 1
        sample_ids.append(adf.sample_id.unique()[0])
        cntr = adf.n_wgd.value_counts()
        for k in wgd_states:
            if k not in ns:
                ns[k] = []
                props[k] = []
            
            ns[k].append(cntr[k] if k in cntr else 0)
            props[k].append(cntr[k]/len(adf) if k in cntr else 0)
        
    all_sites[p] = sites
    all_ns[p] = ns
    all_props[p] = props
    n_sites[p] = len(sites)

    for i, s in enumerate(sites):
        for n_wgd, val in ns.items():
            rows.append({'patient_id':p, 'sample_id':sample_ids[i],
                         'site':s, 'n_wgd':n_wgd, 'num_cells':val[i], 'prop_cells_in_site':props[n_wgd][i]})
```

```python
longdf = pd.DataFrame(rows)
longdf.to_csv('../../data/multisite_wgd.csv', index=False)
```

```python

fig = plt.figure(figsize=(16, 12), dpi = 200)
for i, (p, ns) in enumerate(sorted(all_ns.items())):
    plt.subplot(6, 7, i + 1)
    xs = np.arange(len(ns[list(ns.keys())[0]]))
    bottom = np.zeros(xs.shape[0])
    for k,heights in sorted(ns.items()):
        plt.bar(xs, heights, bottom=bottom, label=k, facecolor=wgd_colors[k])
        bottom += heights
    plt.xticks(xs, all_sites[p])
    plt.legend(title='nWGD')
    plt.title(p)
fig.supxlabel("Anatomical site")
fig.supylabel("Number of cells")
plt.tight_layout()
```

## look for mixed WGD across sites

```python
multisite_mixed = []
allsite_mixed = []
has3_states = []
has3_states_multisite = []

for p in multisite.keys():
    props = all_props[p]

    # look for mixed WGD across multiple sites
    m = np.array(list(props.values()))
    assert np.allclose(1, np.sum(m, axis = 0))
    assert m.shape[0] > 1
    
    mixed = np.max(m, axis = 0) < 1
    if sum(mixed) > 1:
        multisite_mixed.append(p)
    if sum(mixed) == m.shape[1]:
        allsite_mixed.append(p)
    if m.shape[0] == 3:
        has3_states.append(p)
        if np.all(m > 0, axis = 0).sum() > 1:
            has3_states_multisite.append(p)
```

```python
allsite_mixed
```

```python
len(multisite_mixed), len(allsite_mixed)
```

```python
Counter([n_sites[p] for p in allsite_mixed])
```

```python
len(has3_states), len(has3_states_multisite)
```

```python
fig = plt.figure(figsize=(12,12), dpi = 200)
for i, p in enumerate(multisite_mixed):
    plt.subplot(5, 4, i + 1)
    props = all_props[p]
    ns = all_ns[p]

    m1 = np.array(list(props.values()))[::-1]
    m2 = np.array(list(ns.values()))[::-1]

    doublemixed = np.sum(m1 > 0, axis = 1) > 2
    print(p, sum(doublemixed), m1.shape[1])

    sites = all_sites[p]
    sns.heatmap(m1, annot=m1)
    plt.xticks(np.arange(len(sites)) + 0.5, sites)
    plt.yticks(np.arange(m1.shape[0])[::-1] + 0.5, sorted(ns.keys()))

    plt.title(p)

fig.supxlabel("Anatomical site")
fig.supylabel("nWGD")
plt.tight_layout()
```

# [not included -- seems irrelevant and uninteresting] how do SBMClone clones vary across sites and WGD states?

```python
adata = ad.read_h5ad('/data1/shahs3/users/myersm2/spectrum-dlp-pipeline/v5.2/postprocessing/sankoff_ar/greedy_events/cohort_all_events_bysite.h5')
```

```python
all_ns = {}
all_props = {}
n_samples = {}

fig = plt.figure(figsize=(12, 12), dpi = 200)
for i, (p, adata) in enumerate(sorted(multisite.items())):
    plt.subplot(5, 5, i + 1)
    df = adata.obs.groupby(['sbmclone_cluster_id', 'brief-site', 'n_wgd'], observed=True).size().unstack()
    sns.heatmap(df, annot=df)
    plt.xlabel('')
    plt.ylabel('')

fig.supxlabel("nWGD")
fig.supylabel("SBMClone Cluster ID - Anatomical Site")
plt.tight_layout()
```

```python
all_ns = {}
all_props = {}
n_samples = {}

fig = plt.figure(figsize=(12, 12), dpi = 200)
for i, (p, adata) in enumerate(sorted(multisite.items())):
    plt.subplot(5, 5, i + 1)
    df = adata.obs.groupby(['sbmclone_cluster_id', 'brief-site'], observed=True).size().unstack()
    sns.heatmap(df, annot=df, cmap = 'viridis')
    plt.xlabel('')
    plt.ylabel('')

fig.supxlabel("Anatomical site")
fig.supylabel("SBMClone Cluster ID")
plt.tight_layout()
```

```python
all_ns = {}
all_props = {}
n_samples = {}

fig = plt.figure(figsize=(12, 12), dpi = 200)
for i, (p, adata) in enumerate(sorted(multisite.items())):
    plt.subplot(5, 5, i + 1)
    df = adata.obs.groupby(['sbmclone_cluster_id', 'brief-site'], observed=True).size().unstack()
    bottom = np.zeros(df.shape[1])
    xs = np.arange(df.shape[1])
    for _, r in df.iterrows():
        plt.bar(xs, r.values, bottom=bottom, label=r.name)
        bottom += r.values
    plt.legend(title='SBMClone cluster')
    plt.xlabel('')
    plt.ylabel('')
    plt.title(p)

fig.supxlabel("Anatomical site")
fig.supylabel("SBMClone Cluster ID")
plt.tight_layout()
```

# simplify metrics for SBMClone and CN heterogeneity within vs across sites


## SBMClone: mutual information between site and clone

```python
rows = []
for i, (p, adata) in enumerate(sorted(multisite.items())):
    df = adata.obs.groupby(['sbmclone_cluster_id', 'brief-site'], observed=True).size().unstack()
    if np.min(df.shape) <= 1:
        continue
    
    d = {}
    # SNV-based heterogeneity across sites
    x = adata.obs.sbmclone_cluster_id
    y = adata.obs['brief-site'].astype('category').cat.codes

    d['patient_id'] = p
    d['n_clones'] = df.shape[0]
    d['n_sites'] = df.shape[1]
    d['consensus_signature'] = sigs[p]
    d['ami'] = adjusted_mutual_info_score(x, y)
    d['ari'] = adjusted_rand_score(x, y)
    rows.append(d)
summary_df = pd.DataFrame(rows)
```

```python
plt.figure(figsize=(8,4), dpi = 250)
plt.subplot(1, 2, 1)
sns.swarmplot(data=summary_df, x='n_sites', y = 'ami', hue='consensus_signature', palette=colors_dict['consensus_signature'])
plt.ylim(-0.05, 1.05)
plt.ylabel("Adjusted mutual information")
plt.legend(bbox_to_anchor=(1,1))

plt.subplot(1, 2, 2)
sns.swarmplot(data=summary_df, x='n_clones', y = 'ami', hue='consensus_signature', palette=colors_dict['consensus_signature'])
plt.ylim(-0.05, 1.05)
plt.ylabel("Adjusted mutual information")

plt.suptitle("Anatomical site vs. SBMClone clone")
plt.tight_layout()
plt.gca().get_legend().remove()
```

## CN heterogeneity: just use pairwise distance

```python
multisite.keys()
```

```python
len(signals_adatas)
```

```python
signals_adatas.keys()
```

```python
p
```

```python
rows = []
for i, (p, _) in tqdm.tqdm(enumerate(sorted(multisite.items()))):
    adata = signals_adatas[p]
    included_cells = set(cell_info.set_index('cell_id').loc[adata.obs.index].query('include_cell').reset_index().cell_id)

    pwd = pd.read_csv(os.path.join(pipeline_outputs, f'preprocessing/pairwise_distance/pairwise_distance_{p}.csv.gz'))
    pwd = pwd[pwd.cell_id_1.isin(included_cells) & pwd.cell_id_2.isin(included_cells)].copy()
    
    pwd['site1'] = pwd.cell_id_1.map(adata.obs['brief-site'])
    pwd['site2'] = pwd.cell_id_2.map(adata.obs['brief-site'])
    pwd['same_site'] = pwd.site1 == pwd.site2
    
    pwd.groupby(['same_site']).copy_mean_sq_distance.mean()
    
    d = {}
    d['patient_id'] = p
    d['consensus_signature'] = sigs[p]
    d['n_cells'] = len(included_cells)
    d['n_sites'] = len(adata.obs['brief-site'].unique())
    d['distances_same'] = pwd[pwd.same_site].copy_mean_sq_distance.mean()    
    d['distances_diff'] = pwd[~pwd.same_site].copy_mean_sq_distance.mean()

    d['all_distances_same'] = pwd[pwd.same_site].copy_mean_sq_distance
    d['all_distances_diff'] = pwd[~pwd.same_site].copy_mean_sq_distance

    
    rows.append(d)
summary_df2 = pd.DataFrame(rows)
```

```python

p = 'SPECTRUM-OV-139'

nnd = pd.read_csv(os.path.join(pipeline_outputs, f'preprocessing/pairwise_distance/cell_nnd_{p}.csv.gz'))
pwd = pd.read_csv(os.path.join(pipeline_outputs, f'preprocessing/pairwise_distance/pairwise_distance_{p}.csv.gz'))
adata = signals_adatas[p]
```

```python
jp = sns.jointplot(data=summary_df2, x='distances_same', y='distances_diff', hue = 'consensus_signature', 
                   palette=colors_dict['consensus_signature'], xlim=(-0.1, 4.5), ylim=(-0.1, 4.5))
jp.ax_joint.set_xlabel("Within-site mean CN distance")
jp.ax_joint.set_ylabel("Between-site mean CN distance")
jp.ax_joint.plot([0, 4.5], [0, 4.5], 'k--')
```

<!-- #raw -->
plt.figure(figsize=(8,4), dpi = 250)
plt.subplot(1, 2, 1)
sns.swarmplot(data=summary_df2, x='n_sites', y = 'ami', hue='consensus_signature', palette=colors_dict['consensus_signature'])
plt.ylim(-0.05, 1.05)
plt.ylabel("Adjusted mutual information")
plt.legend(bbox_to_anchor=(1,1))

plt.subplot(1, 2, 2)
sns.swarmplot(data=summary_df2, x='n_cells', y = 'ami', hue='consensus_signature', palette=colors_dict['consensus_signature'])
plt.ylim(-0.05, 1.05)
plt.ylabel("Adjusted mutual information")

plt.suptitle("Anatomical site vs. SBMClone clone")
plt.tight_layout()
plt.gca().get_legend().remove()
<!-- #endraw -->

```python
np.sqrt(pwd.shape[0]), adata.shape
```

```python

```

```python

```

```python

```
