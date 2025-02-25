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

import scgenome
from matplotlib.collections import LineCollection
from matplotlib.patches import Rectangle
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
from datetime import datetime
from collections import Counter
from yaml import safe_load
```

```python
pipeline_outputs = '/data1/shahs3/users/myersm2/repos/spectrum_wgd_data5'
colors_yaml = safe_load(open('/data1/shahs3/users/myersm2/repos/spectrum-genomic-instability/resources/annotation/colors.yaml', 'r').read())
bootstrap_repetitions = 500


```

```python
doubletime_form = os.path.join(pipeline_outputs, 'tree_snv/outputs/{}_general_snv_tree_assignment.csv')
segments_form = os.path.join(pipeline_outputs, 'medicc/resegmented_cn/{}.csv.gz')

```

<!-- #raw -->
adata_dir = os.path.join(pipeline_outputs, 'postprocessing/aggregated_anndatas')

adatas = {}
clone_adatas = {}

for f in tqdm.tqdm(sorted(os.listdir(adata_dir))):
    p = f.split('_')[0]
    if f.endswith('_cna.h5'):
        adatas[p] = ad.read_h5ad(os.path.join(adata_dir, f))
    elif f.endswith('_cna_clustered.h5'):
        clone_adatas[p] = ad.read_h5ad(os.path.join(adata_dir, f))
        
    if len(adatas) == 1 and len(clone_adatas) == 1:
        break
<!-- #endraw -->

# get chromosome timing for a patient

```python
p = 'SPECTRUM-OV-002'
#adata = adatas[p]
#clone_adata = clone_adatas[p]
df = pd.read_csv(doubletime_form.format(p))

segments = pd.read_csv(segments_form.format(p), dtype={'chr':str})
segments = segments[['chr', 'start', 'end']].drop_duplicates()
```

```python
def calc_timing(n1, n2, ascn):
    if ascn == '2:1':
        return 3*n2 / (2 * n2 + n1)
    elif ascn == '2:2' or ascn == '2:0':
        return 2*n2 / (2 * n2 + n1)
```

```python
def estimate_timing(snvs, segments, bootstrap_repetitions=200):
    results = []
    
    for _, r in segments.iterrows():
        segment_snvs = snvs[(snvs.chromosome == r.chr) & 
                        (snvs.position >= r.start) & 
                        (snvs.position <= r.end) &
                        ~(snvs.ascn.isin(['1:0', '1:1'])) ]
        if len(segment_snvs) > 0:
            for ascn, my_snvs in segment_snvs.groupby('ascn'):
                assert len(my_snvs.ascn.unique()) == 1, my_snvs.ascn.unique()
                cntr = my_snvs.cn_state_a.value_counts()
                my_ascn = my_snvs.ascn.iloc[0]
                
                n1 = cntr[1] if 1 in cntr else 0
                n2 = cntr[2] if 2 in cntr else 0
    
                if n1 + n2 == 0 or my_ascn in ['1:0', '1:1']:
                    results.append({'chr':r.chr, 'start':r.start, 'end':r.end, 'ascn':my_ascn,
                            'n1':0, 'n2':0, 'timing':np.nan, 't_lo':np.nan, 't_hi':np.nan})
                    continue
                    
                t = calc_timing(n1, n2, my_ascn)
        
                bootstrap_replicates = []
                np.random.seed(0)
                for _ in range(bootstrap_repetitions):        
                    # resample SNVs with replacement
                    idx = np.random.choice(my_snvs.index, size=len(my_snvs), replace=True)
                    cntr_ = my_snvs.loc[idx].cn_state_a.value_counts()
                    n1_ = cntr_[1] if 1 in cntr_ else 0
                    n2_ = cntr_[2] if 2 in cntr_ else 0
        
                    if n1_ + n2_ == 0:
                        continue
                    bootstrap_replicates.append(calc_timing(n1_, n2_, my_ascn))
                                    
                results.append({'chr':r.chr, 'start':r.start, 'end':r.end, 'ascn':my_ascn,
                                'n1':n1, 'n2':n2, 'timing':t,
                                't_lo':np.percentile(bootstrap_replicates, 5),
                                't_hi':np.percentile(bootstrap_replicates, 95)})
        else:
            results.append({'chr':r.chr, 'start':r.start, 'end':r.end, 'ascn':np.nan,
                    'n1':0, 'n2':0, 'timing':np.nan, 't_lo':np.nan, 't_hi':np.nan})
    
    results = pd.DataFrame(results)
    results['color'] = [mcolors.to_hex(c) for c in 
                        plt.get_cmap('tab10')(results.ascn.astype('category').cat.codes)]
    return results
```

```python
def plot_results(results, title):
    plt.figure(figsize=(12,3), dpi = 150)
    
    chr_order = sorted(results.chr.unique(), key = lambda x:23 if x=='X' else int(x))
    prefix = 0
    xticklocs = []
    sep_locs = []
    for ch in chr_order:
        cdf = results[results.chr == ch]
        largest_pos = cdf.end.max()
        xticklocs.append(prefix + largest_pos/2)
        sep_locs.append(prefix + largest_pos)
        
        cdf = cdf.dropna()
        
        lc = LineCollection([ [[r.start+prefix, r.timing],[r.end+prefix, r.timing]] 
                             for _, r in cdf.iterrows()], colors=cdf.color)
        plt.gca().add_collection(lc)
    
        for _, r in cdf.iterrows():
            # add rectangle showing confidence interval
            rect = Rectangle((r.start + prefix, r.t_lo), r.end - r.start, r.t_hi - r.t_lo,
                             alpha = 0.5, facecolor=r.color)
            plt.gca().add_patch(rect)
    
        prefix += largest_pos
    plt.xlim(0, prefix)
    plt.xticks(xticklocs, chr_order)
    plt.xlabel("Chromosome")
    plt.ylabel("Timing (late-ness)")
    plt.vlines(ymin=0, ymax=1, x=sep_locs, colors='k', linewidth=1)
    plt.ylim(-0.05, 1.05)
    plt.title(title)

    legend_map = {r['ascn']:r['color'] for _, r in 
                  results[['ascn', 'color']].value_counts().reset_index().iterrows()}
    
    plt.legend(handles=[Line2D([0], [0], label=k, color=v, linestyle='-') 
                        for k,v in legend_map.items()],
              loc='upper left', bbox_to_anchor=(1,1), title='ascn')

```

```python
patients = sorted(set([a.split('_')[0] for a in os.listdir(os.path.join(pipeline_outputs, 'tree_snv/inputs'))]))
len(patients)
```

```python
all_results = {}
```

```python
for p in sorted(patients):
    print(datetime.now(), p)
    if p in all_results:
        continue
    snvs = pd.read_csv(doubletime_form.format(p))
    segments = pd.read_csv(segments_form.format(p), dtype={'chr':str})
    segments = segments[['chr', 'start', 'end']].drop_duplicates()

    all_results[p] = estimate_timing(snvs, segments)
```

```python
pd.read_csv(doubletime_form.format('SPECTRUM-OV-002')).cn_state_a.value_counts()
```

```python
for p, results in all_results.items():
    plot_results(results, p)
```

# look at a more intrinsic measure: how does doubleTime timing vary by chromosome?

```python
all_tbc = {}
overall_timing = {}

for p in tqdm.tqdm(sorted(patients)):
    if p.endswith('045') or p.endswith('025'):
        continue
    snvs = pd.read_csv(doubletime_form.format(p))
    segments = pd.read_csv(segments_form.format(p), dtype={'chr':str})

    my_snvs = snvs[snvs.wgd_timing != 'none']
    if len(my_snvs) == 0:
        continue

    #assert len(my_snvs.clade.unique()) == 1, p
    cntr = my_snvs.wgd_timing.value_counts()
    overall_timing[p] = 2 * cntr.prewgd / (2 * cntr.prewgd + cntr.postwgd)
    
    tbc = my_snvs[['chromosome', 'wgd_timing']].value_counts().unstack().fillna(0).astype(int)
    tbc['time'] = 2 * tbc.prewgd / (2 * tbc.prewgd + tbc.postwgd)
    tbc['total_snvs'] = tbc.prewgd + tbc.postwgd
    tbc['prop_pre'] = tbc.prewgd / tbc.total_snvs

    np.random.seed(0)
    ch2ci = {}
    for _, r in tbc.iterrows():
        n_pre = np.sum(np.random.random(size=(int(r.total_snvs), bootstrap_repetitions)) < r.prop_pre, axis = 0)
        n_post = r.total_snvs - n_pre
        simT = 2 * n_pre / (2 * n_pre + n_post)
        ch2ci[r.name] = np.percentile(simT, 5), np.percentile(simT, 95)

    cis = tbc.index.map(ch2ci)
    tbc['T_ci_5'] = cis.get_level_values(0)
    tbc['T_ci_95'] = cis.get_level_values(1)
    all_tbc[p] = tbc

```

```python
overall_timing
```

<!-- #raw -->
my_snvs
<!-- #endraw -->

```python
tbc
```

```python
my_snvs[['leaf', 'clade', 'wgd_timing']].value_counts()
```

```python
plt.figure(figsize=(12, 12), dpi = 150)
for i, (p, tbc) in enumerate(all_tbc.items()):
    plt.subplot(5, 5, i + 1)
    chr_order = sorted(tbc.index.unique(), key = lambda x:23 if x=='X' else int(x))
    tbc = tbc.loc[chr_order]


    plt.bar(tbc.index.astype(str), tbc.time, width=0.5)
    plt.errorbar(tbc.index.astype(str), tbc.time, yerr=(tbc.time-tbc.T_ci_5, tbc.T_ci_95-tbc.time), fmt='.', c = 'r', markersize=0,
                capthick=5)
    plt.axhline(overall_timing[p], linestyle = '--', color = 'k')

    # stagger x-tick labels to avoid overlapping
    for tick in plt.gca().xaxis.get_major_ticks()[1::2]:
        tick.set_pad(15)
        
    plt.ylim(-0.05, 1.05)
    plt.title(p[-6:])
    plt.ylabel("WGD timing")
    plt.xlabel("Chromosome")
plt.tight_layout()
```

## look into outliers


### looks like 052 chr18 is just LOH, maybe by deletions instead of by WGD?

```python
adata_052 = ad.read_h5ad(os.path.join(adata_dir, 'SPECTRUM-OV-052_cna.h5'))
```

```python
fig=plt.figure(figsize=(12,8), dpi = 300)
_ = scgenome.pl.plot_cell_cn_matrix_fig(adata_052, cell_order_fields=['sbmclone_cluster_id'], annotation_fields=['sbmclone_cluster_id'], fig=fig)
```

```python
fig=plt.figure(figsize=(12,8), dpi = 300)
_ = scgenome.pl.plot_cell_cn_matrix_fig(adata_052, cell_order_fields=['sbmclone_cluster_id'], annotation_fields=['sbmclone_cluster_id'], fig=fig,
                                       layer_name='Min')
```

```python

```

```python

```

```python

```
