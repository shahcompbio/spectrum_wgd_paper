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
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import pandas as pd
import numpy as np
from IPython.display import Image
import anndata as ad
import scgenome
from matplotlib.patches import Rectangle
import yaml
import anndata as ad
from Bio import Phylo
import json
from collections import Counter
from matplotlib import colors as mcolors
from scipy.spatial.distance import pdist, squareform
import Bio
from copy import deepcopy
import shutil
from scipy.sparse import load_npz
import matplotlib.colors as mcolors
import warnings
from time import perf_counter

from scipy.stats import binom, gaussian_kde, poisson
import scipy
from multiprocessing import Pool
from sklearn.metrics import confusion_matrix
from scipy.cluster.hierarchy import linkage, fcluster
import itertools
import math
import warnings
from tqdm import tqdm
```

```python
from IPython.display import Image
```

```python
pipeline_outputs = '/data1/shahs3/users/myersm2/repos/spectrum_wgd_data5'
exclude_patients = ['SPECTRUM-OV-024', 'SPECTRUM-OV-125']

```

```python
colors_dict = yaml.safe_load(open('/data1/shahs3/users/myersm2/repos/spectrumanalysis/config/colors.yaml', 'r'))
sbm_adata_stem = os.path.join(pipeline_outputs, 'sbmclone')
patients = [a.split('_')[0] for a in os.listdir(sbm_adata_stem)]
clustered_adata_form =  os.path.join(pipeline_outputs, 'tree_snv/inputs/{}_cna_clustered.h5')
clone_adata_form =  os.path.join(pipeline_outputs, 'tree_snv/inputs/{}_general_clone_adata.h5')
signals_adata_form =  os.path.join(pipeline_outputs, 'preprocessing/signals/signals_{}.h5')
patients = sorted(np.unique([a.split('_')[0] for a in os.listdir(sbm_adata_stem)]))
```

```python
cell_info = pd.read_csv(os.path.join(pipeline_outputs, 'preprocessing/summary/filtered_cell_table.csv.gz'))
```

```python
def get_blocks_adata(p, epsilon = 0.001):
    cn_adata = ad.read_h5ad(clustered_adata_form.format(p))
    cn_adata.var['is_cnloh'] = np.logical_and(cn_adata.var.is_homogenous_cn, np.logical_or(
        np.logical_and(np.all(cn_adata.layers['A'] == 2, axis = 0), np.all(cn_adata.layers['B'] == 0, axis = 0)),
        np.logical_and(np.all(cn_adata.layers['A'] == 0, axis = 0), np.all(cn_adata.layers['B'] == 2, axis = 0))))

    blocks_adata = ad.read_h5ad(clone_adata_form.format(p))

    blocks_adata.var['is_cnloh'] = cn_adata.var.loc[blocks_adata.var.cn_bin, 'is_cnloh'].values
    blocks_adata.layers['vaf'] = blocks_adata.layers['alt_count'] / np.maximum(1, blocks_adata.layers['total_count'])
    blocks_adata.layers['p_cn0'] = binom.logpmf(k=blocks_adata.layers['alt_count'], n=blocks_adata.layers['total_count'], p = epsilon)
    blocks_adata.layers['p_cn1'] = binom.logpmf(k=blocks_adata.layers['alt_count'], n=blocks_adata.layers['total_count'], p = 0.5)
    blocks_adata.layers['p_cn2'] = binom.logpmf(k=blocks_adata.layers['alt_count'], n=blocks_adata.layers['total_count'], p = 1-epsilon)
    blocks_adata = blocks_adata[:, blocks_adata.var['is_cnloh']]
    return blocks_adata

def get_partition(blocks_adata, partition, epsilon = 0.001, min_snvcov_reads = 2):
    assert len(partition) == len(blocks_adata), (len(partition), len(blocks_adata))
    partition = np.array(partition)
    part1_idx = np.where(partition == 1)[0]
    part2_idx = np.where(partition == 2)[0]
    assert len(part1_idx) > 0 and len(part2_idx) > 0, (part1_idx, part2_idx)
    
    new_blocks_adata = blocks_adata[:2].copy()
    new_blocks_adata.layers['alt_count'][0] = blocks_adata[part1_idx].layers['alt_count'].toarray().sum(axis = 0)
    new_blocks_adata.layers['ref'][0] = blocks_adata[part1_idx].layers['ref'].toarray().sum(axis = 0)
    new_blocks_adata.layers['total_count'][0] = blocks_adata[part1_idx].layers['total_count'].toarray().sum(axis = 0)
    new_blocks_adata.layers['B'][0] = np.median(blocks_adata[part1_idx].layers['B'], axis = 0)
    new_blocks_adata.layers['A'][0] = np.median(blocks_adata[part1_idx].layers['A'], axis = 0)
    new_blocks_adata.layers['state'][0] = np.median(blocks_adata[part1_idx].layers['state'], axis = 0)
    new_blocks_adata.obs['partition_size'] = [sum([blocks_adata.obs.cluster_size[i] for i in part1_idx]),
                                              sum([blocks_adata.obs.cluster_size[i] for i in part2_idx])]
    new_blocks_adata.obs['blocks'] = ['/'.join([blocks_adata.obs.iloc[a].name for a in l]) for l in [part1_idx, part2_idx]]
    
    new_blocks_adata.layers['alt_count'][1] = blocks_adata[part2_idx].layers['alt_count'].toarray().sum(axis = 0)
    new_blocks_adata.layers['ref'][1] = blocks_adata[part2_idx].layers['ref'].toarray().sum(axis = 0)
    new_blocks_adata.layers['total_count'][1] = blocks_adata[part2_idx].layers['total_count'].toarray().sum(axis = 0)
    new_blocks_adata.layers['B'][1] = np.median(blocks_adata[part2_idx].layers['B'], axis = 0)
    new_blocks_adata.layers['A'][1] = np.median(blocks_adata[part2_idx].layers['A'], axis = 0)
    new_blocks_adata.layers['state'][1] = np.median(blocks_adata[part2_idx].layers['state'], axis = 0)
    
    new_blocks_adata.layers['vaf'] = new_blocks_adata.layers['alt_count'] / (new_blocks_adata.layers['total_count'])
    
    # add layers for marginal probabilities of CN states
    new_blocks_adata.layers['p_cn0'] = binom.logpmf(k=new_blocks_adata.layers['alt_count'], n=new_blocks_adata.layers['total_count'], p = epsilon)
    new_blocks_adata.layers['p_cn1'] = binom.logpmf(k=new_blocks_adata.layers['alt_count'], n=new_blocks_adata.layers['total_count'], p = 0.5)
    new_blocks_adata.layers['p_cn2'] = binom.logpmf(k=new_blocks_adata.layers['alt_count'], n=new_blocks_adata.layers['total_count'], p = 1-epsilon)
    
    # remove columns with too few total counts
    valid_snvs = np.where(np.min(new_blocks_adata.layers['total_count'], axis = 0) >= min_snvcov_reads)[0]
    new_blocks_adata = new_blocks_adata[:, valid_snvs].copy()
    new_blocks_adata.obs = new_blocks_adata.obs.drop(columns = ['cluster_size'])
    
    return new_blocks_adata

wgd1_options = [
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1],
    [2, 2],
]

wgd2_options = [
    [0, 0],
    [0, 1],
    [0, 2],
    [1, 0],
    [2, 0],
    [2, 2],
]


def compute_ll(partition_adata, return_ml_genotypes = False):
    ll_wgd1 = np.zeros((partition_adata.shape[1], len(wgd1_options)))
    for idx, (cn1, cn2) in enumerate(wgd1_options):
        ll_wgd1[:, idx] = partition_adata.layers[f'p_cn{cn1}'][0] + partition_adata.layers[f'p_cn{cn2}'][1]
    marginal_ll_wgd1 = scipy.special.logsumexp(ll_wgd1, axis=1)

    ll_wgd2 = np.zeros((partition_adata.shape[1], len(wgd2_options)))
    for idx, (cn1, cn2) in enumerate(wgd2_options):
        ll_wgd2[:, idx] = partition_adata.layers[f'p_cn{cn1}'][0] + partition_adata.layers[f'p_cn{cn2}'][1]
    marginal_ll_wgd2 = scipy.special.logsumexp(ll_wgd2, axis=1)
    
    if return_ml_genotypes:
        geno1 = [wgd1_options[a] for a in np.argmax(ll_wgd1, axis = 1)]
        geno2 = [wgd2_options[a] for a in np.argmax(ll_wgd2, axis = 1)]
        return marginal_ll_wgd1, marginal_ll_wgd2, geno1, geno2
    else:
        return marginal_ll_wgd1, marginal_ll_wgd2
    

def compute_ll_numpy(probs1, probs2, return_ml_genotypes = False):
    assert np.array_equal(probs1.shape, probs2.shape)
    ll_wgd1 = np.zeros((probs1.shape[1], len(wgd1_options)))
    for idx, (cn1, cn2) in enumerate(wgd1_options):
        ll_wgd1[:, idx] = probs1[cn1] + probs2[cn2]
    marginal_ll_wgd1 = scipy.special.logsumexp(ll_wgd1, axis=1)

    ll_wgd2 = np.zeros((probs1.shape[1], len(wgd2_options)))
    for idx, (cn1, cn2) in enumerate(wgd2_options):
        ll_wgd2[:, idx] = probs1[cn1] + probs2[cn2]
    marginal_ll_wgd2 = scipy.special.logsumexp(ll_wgd2, axis=1)
    
    if return_ml_genotypes:
        geno1 = [wgd1_options[a] for a in np.argmax(ll_wgd1, axis = 1)]
        geno2 = [wgd2_options[a] for a in np.argmax(ll_wgd2, axis = 1)]
        return marginal_ll_wgd1, marginal_ll_wgd2, geno1, geno2
    else:
        return marginal_ll_wgd1, marginal_ll_wgd2
```

```python
def generate_null_resample(partition_adata, genotypes_1wgd, epsilon = 0.001, n_iter = 1000, return_values = False):
    P = np.clip(np.array(genotypes_1wgd).T / 2, a_min = epsilon, a_max = 1 - epsilon)
    n_snvs = partition_adata.shape[1]
    
    scores_resample = []
    if return_values:
        all_probs1 = []
        all_probs2 = []
    for i in range(n_iter):
        np.random.seed(i)

        # HACK: mask 0 entries with 1 and then deplete to get around binom.rvs issues
        sim_alts = binom.rvs(p = P, n = np.maximum(partition_adata.layers['total_count'], 1).astype(int))
        sim_alts = np.minimum(partition_adata.layers['total_count'], sim_alts)

        probs1 = np.zeros((3, n_snvs))
        probs1[0] = binom.logpmf(k=sim_alts[0], n=partition_adata.layers['total_count'][0], p = epsilon)
        probs1[1] = binom.logpmf(k=sim_alts[0], n=partition_adata.layers['total_count'][0], p = 0.5)
        probs1[2] = binom.logpmf(k=sim_alts[0], n=partition_adata.layers['total_count'][0], p = 1-epsilon)

        
        probs2 = np.zeros((3, n_snvs))
        probs2[0] = binom.logpmf(k=sim_alts[1], n=partition_adata.layers['total_count'][1], p = epsilon)
        probs2[1] = binom.logpmf(k=sim_alts[1], n=partition_adata.layers['total_count'][1], p = 0.5)
        probs2[2] = binom.logpmf(k=sim_alts[1], n=partition_adata.layers['total_count'][1], p = 1-epsilon)
        
        probs1, probs2 = compute_ll_numpy(probs1, probs2)
        scores_resample.append(probs2.sum() - probs1.sum())
        if return_values:
            all_probs1.append(probs1)
            all_probs2.append(probs2)
            
    if return_values:
        return scores_resample, np.array(all_probs1), np.array(all_probs2)
    else:
        return scores_resample
```

# run across cohort

```python
def enumerate_partitions(n, skip_reflection = True):
    if n == 2:
        yield np.array([1, 2])
    else:
        part = np.ones(n, dtype = int)
        a = np.arange(n)
        oddn = n if n % 2 == 1 else n - 1

        for k in range(1, int(n/2) + 1):
            last_j = int(math.factorial(n) / (math.factorial(k) * math.factorial(n - k))) - 1
            for j, idx in enumerate(itertools.combinations(a, k)):
                if skip_reflection:
                    if k > 1 and j == last_j:
                        continue
                my_part = part.copy()
                my_part[list(idx)] = 2
                yield my_part
```

```python
def run_patient_partitions(p, epsilon = 0.001, n_iter = 10000, min_snvcov_reads = 2,
                           sbm_adata_stem=sbm_adata_stem):
    
    try:
        sbm_adata = ad.read_h5ad(os.path.join(sbm_adata_stem, f'sbmclone_{p}_snv.h5'))
    except FileNotFoundError:
        print("missing files for patient:", p)
        return

    sbm_adata.obs = sbm_adata.obs.merge(cell_info[['cell_id', 'n_wgd']], left_index=True, right_on=['cell_id']).set_index('cell_id').loc[sbm_adata.obs.index]
    sbm_adata = sbm_adata[(sbm_adata.obs.n_wgd == 1)]
    
    results = {}
    
    for block_column in ['sbmclone_cluster_id']:#, 'neighbor_sbmclone_block']:        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            blocks_adata = get_blocks_adata(p)
        if blocks_adata.shape[1] == 0:
            print(f"Patient {p} has no clonal cnLOH SNVs")
            break
        
        n_blocks = len(blocks_adata)
        for partition in enumerate_partitions(n_blocks):
            result = {}
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                partition_adata = get_partition(blocks_adata, partition, epsilon = epsilon, min_snvcov_reads = min_snvcov_reads)
            if partition_adata.shape[1] == 0:
                print(f"Patient {p} has no clonal cnLOH SNVs with sufficient coverage for column {block_column} and partition {partition}")
                continue
            #print(datetime.now(), block_column, partition, blocks_adata.shape[1], partition_adata.shape[1])
    
        
            result['prob_1wgd'], result['prob_2wgd'], result['ml_geno_1wgd'], result['ml_geno_2wgd'] = compute_ll(partition_adata, return_ml_genotypes = True)
            result['score'] = result['prob_2wgd'].sum() - result['prob_1wgd'].sum()
            
            result['null_scores'] =  np.array(generate_null_resample(partition_adata, result['ml_geno_1wgd'], n_iter = n_iter))
            result['null_mean'] = np.mean(result['null_scores'])
            result['null_std'] = np.std(result['null_scores'])
            result['ll_zscore'] = (result['score'] - result['null_mean']) / result['null_std']
            result['pvalue'] = np.sum(result['null_scores'] > result['score']) / n_iter
            result['partition_adata'] = partition_adata
            result['blocks_adata'] = blocks_adata
            results[block_column, tuple(partition)] = result
    return results
```

<!-- #raw -->
shortlist = ['SPECTRUM-OV-025', 'SPECTRUM-OV-045', 'SPECTRUM-OV-046', 'SPECTRUM-OV-081']
<!-- #endraw -->

```python
patients = sorted(np.unique([a.split('_')[1] for a in os.listdir(sbm_adata_stem) if a.endswith('.h5')]))
for p in exclude_patients:
    if p in patients:
        patients.remove(p)
```

```python
len(patients)
```

```python
print(datetime.now())
with Pool(8) as p:
    all_results = p.map(run_patient_partitions, patients)
print(datetime.now())
```

```python
with open('indepwgdtest_results.pickle', 'wb') as f:
    pickle.dump(all_results, f)
```

```python
rows = []
for p, res in zip(patients, all_results):
    if res is None:
        print(p)
        continue
    for (col, partition), v in res.items():
        adata = v['partition_adata']
        n_snvs = adata.shape[1]
        size1, size2 = v['partition_adata'].obs.partition_size
        
        rows.append((p, col, partition, size1, size2, size1 + size2, n_snvs, v['prob_1wgd'].sum(), v['prob_2wgd'].sum(), v['pvalue'],
                    v['null_mean'], v['null_std']))
        
sbmdf = pd.DataFrame(rows, columns = ['patient', 'cluster_column', 'partition', 'n_cells_p1', 'n_cells_p2', 
                                      'n_cells_total', 'n_snvs', 'll_1wgd', 'll_2wgd', 'pvalue', 'null_mean', 'null_std'])
sbmdf['score'] = sbmdf.ll_2wgd - sbmdf.ll_1wgd
sbmdf['yval'] = -1 * np.log10(np.maximum(9e-5, sbmdf.pvalue))
sbmdf['zscore'] = (sbmdf.score - sbmdf.null_mean) / sbmdf.null_std
                              
best_scores = []
for p, df in sbmdf.groupby('patient'):
    best_row = df.iloc[df.score.argmax()]
    best_scores.append(best_row)
best_scores = pd.DataFrame(best_scores).reset_index(drop = True)
```

```python
len(patients)
```

```python
# 10 had uninformative SBMClone results (only 1 tumor clone identified), 9 had no clonal cnLOH SNVs
```

```python
len(sbmdf.patient.unique())
```

```python
sns.scatterplot(data=sbmdf, x = 'score', y = 'yval', hue = 'patient', size='n_cells_total')
plt.xlabel("LL(2WGD) - LL(1WGD)")
plt.ylabel("-log10(p-value)")
plt.legend(bbox_to_anchor=(1, 1))
plt.axvline(x=0, c = 'grey')
```

```python
wgddf = []
for p in patients:
    cntr = Counter(cell_info[cell_info.include_cell & (cell_info.patient_id == p)].n_wgd)
    v = {'patient':p}
    for i in range(4):
        if i in cntr:
            v[f'{i}_wgd'] = cntr[i]
        else:
            v[f'{i}_wgd'] = 0
    v['prop_wgd'] = np.sum([val for k,val in v.items() if k != 'patient' and int(k[0]) > 0]) / sum(cntr.values())
    wgddf.append(v)
wgddf = pd.DataFrame(wgddf).set_index('patient')
```

```python
len(wgddf[wgddf.prop_wgd > 0.9])
```

```python
len(wgddf)
```

```python
len(best_scores)
```

```python
best_scores['prop_wgd'] = best_scores.patient.map(lambda x:wgddf.loc[x].prop_wgd)
```

```python
sns.scatterplot(best_scores, x = 'score', y = 'yval', hue = 'patient', size = 'n_cells_total')
plt.xlabel("LL(2WGD) - LL(1WGD)")
plt.ylabel("-log10(p-value)")
plt.legend(bbox_to_anchor=(1, 1))
plt.axvline(x=0, c = 'grey')
```

```python
len(best_scores[best_scores.prop_wgd > 0.9])
```

```python
best_scores[best_scores.prop_wgd > 0.9]
```

```python
plt.figure(dpi = 200)
sns.scatterplot(best_scores[best_scores.prop_wgd > 0.9], x = 'score', y = 'yval', hue = 'patient', size = 'n_cells_total', palette = 'tab20')
plt.xlabel("LL(2WGD) - LL(1WGD)")
plt.ylabel("-log10(p-value)")
plt.legend(bbox_to_anchor=(1, 1))
plt.axvline(x=0, c = 'grey')
```

```python
plt.figure(dpi = 200)
sns.scatterplot(best_scores[best_scores.prop_wgd > 0.9], x = 'score', y = 'zscore', hue = 'patient', size = 'n_cells_total', palette = 'tab20')
plt.xlabel("LL(2WGD) - LL(1WGD)")
plt.ylabel("z-score")
plt.legend(bbox_to_anchor=(1, 1))
plt.axvline(x=0, c = 'grey')
plt.axhline(y=0, c='grey')
plt.axhline(y=3, c='blue', linewidth = 0.5, linestyle='--', label='z=+/-3')
plt.axhline(y=-3, c='blue', linewidth = 0.5, linestyle='--')

```

```python

best_scores[(best_scores.n_cells_total > 500) & (best_scores.n_snvs > 100)]
```

```python
best_scores[best_scores.patient.isin(['SPECTRUM-OV-036', 'SPECTRUM-OV-044', 'SPECTRUM-OV-052'])]
```

```python
best_scores[(best_scores.n_cells_p1 > 100) & (best_scores.n_cells_p2 > 100) & (best_scores.n_snvs > 100)]
```

```python
sbmdf.groupby('patient').max('score')
```

```python
real_results = []
for _, df in sbmdf[(sbmdf.score > 0) & (sbmdf.pvalue < 0.9)].groupby(by = 'patient'):
    real_results.append(df.iloc[df.score.argmax()])
real_results = pd.DataFrame(real_results)
real_results
```

```python
sbmdf[sbmdf.patient == 'SPECTRUM-OV-045'].sort_values(by = 'score', ascending=False)
```

```python

```

# run on all pairs of blocks for 045

```python
def run_patient_pairs(p, epsilon = 0.001, n_iter = 10000, min_snvcov_reads = 2,
                           sbm_adata_stem=sbm_adata_stem, sbm_adata_suffix='_snv.h5'):
    try:
        sbm_adata = ad.read_h5ad(os.path.join(sbm_adata_stem, f'sbmclone_{p}_snv.h5'))
    except FileNotFoundError:
        print("missing files for patient:", p)
        return

    sbm_adata.obs = sbm_adata.obs.merge(cell_info[['cell_id', 'n_wgd']], left_index=True, right_on=['cell_id']).set_index('cell_id').loc[sbm_adata.obs.index]
    sbm_adata = sbm_adata[(sbm_adata.obs.n_wgd == 1)]
    
    results = {}
    for block_column in ['sbmclone_cluster_id']:        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            blocks_adata = get_blocks_adata(p)
        if blocks_adata.shape[1] == 0:
            print(f"Patient {p} has no clonal cnLOH SNVs")
            break
        
        n_blocks = len(sbm_adata.obs[block_column].unique())
        which_blocks = sorted(sbm_adata.obs[block_column].unique())
        for p1 in range(n_blocks):
            for p2 in range(p1 + 1, n_blocks):
                result = {}

                partition_adata = blocks_adata[[which_blocks[p1], which_blocks[p2]]].copy()
                partition_adata = partition_adata[:, np.where(np.sum(partition_adata.layers['total_count'], axis = 0) > min_snvcov_reads)[0]]
                
                if partition_adata.shape[1] == 0:
                    print(f"Patient {p} has no clonal cnLOH SNVs with sufficient coverage for column {block_column} and partition {partition}")
                    continue

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    result['prob_1wgd'], result['prob_2wgd'], result['ml_geno_1wgd'], result['ml_geno_2wgd'] = compute_ll(partition_adata, return_ml_genotypes = True)
                result['score'] = result['prob_2wgd'].sum() - result['prob_1wgd'].sum()

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    result['null_scores'] =  np.array(generate_null_resample(partition_adata,result['ml_geno_1wgd'],  n_iter = n_iter))
                result['pvalue'] = np.sum(result['null_scores'] > result['score']) / n_iter
                result['partition_adata'] = partition_adata
                results[block_column, (p1, p2)] = result
    return results
```

```python
print(datetime.now())
pairs045 = run_patient_pairs('SPECTRUM-OV-045')
print(datetime.now())
```

```python
for k,v in pairs045.items():
    print(k, v['partition_adata'].shape[1], v['score'], v['pvalue'])
```

# reproduce my figures


## 045

```python
p = 'SPECTRUM-OV-045'
partition = (1, 1, 2, 1)
cluster_col = 'sbmclone_cluster_id'

cn_adata = ad.read_h5ad(clustered_adata_form.format(p))
my_results =  all_results[patients.index(p)][cluster_col, partition]
part_ad = my_results['partition_adata']
```

```python
plotdf = part_ad.var.copy()
plotdf['vaf1'] = part_ad.layers['vaf'][0]
plotdf['vaf2'] = part_ad.layers['vaf'][1]

plotdf['total1'] = part_ad.layers['total_count'][0]
plotdf['total2'] = part_ad.layers['total_count'][1]
plotdf['total_both'] = plotdf.total1 + plotdf.total2

plotdf['alt1'] = part_ad.layers['alt_count'][0]
plotdf['alt2'] = part_ad.layers['alt_count'][1]

plotdf['prob_1wgd'] = my_results['prob_1wgd']
plotdf['prob_2wgd'] = my_results['prob_2wgd']
plotdf['partial_score'] = plotdf.prob_2wgd - plotdf.prob_1wgd

plotdf['ml_geno_1wgd'] = my_results['ml_geno_1wgd']
plotdf['ml_geno_2wgd'] = my_results['ml_geno_2wgd']

plotdf['coloring'] = [(0 if a == b else
                      1 if a == [1,1] else 2) for a,b in zip(plotdf.ml_geno_1wgd, plotdf.ml_geno_2wgd)]
plotdf['Label'] = plotdf.coloring.map({0:'Other', 1:'Shared 1-copy', 2:'Exclusive 2-copy'}).astype(
    pd.api.types.CategoricalDtype(categories=['Shared 1-copy', 'Exclusive 2-copy', 'Other'], ordered=True))
```

```python
g = sns.JointGrid(xlim = [-0.1, 1.1], ylim = [-0.1, 1.1])
g.fig.set_dpi(300)
x3, y3, h3, s3 = plotdf.vaf1, plotdf.vaf2, plotdf.Label, plotdf.total_both/5
x2, y2, h2 = plotdf[plotdf.coloring > 0].vaf1, plotdf[plotdf.coloring > 0].vaf2, plotdf[plotdf.coloring > 0].Label
pal3 = ['blue', 'red', 'grey']
sns.scatterplot(x=x3, y=y3, sizes=s3, hue=h3, palette=pal3, ax=g.ax_joint, alpha = 0.2)
g.ax_joint.get_legend().set_title('')

sns.histplot(x=x2, hue=h2, palette=pal3, ax=g.ax_marg_x, bins = 20)
g.ax_marg_x.get_legend().remove()

sns.histplot(y=y2, hue=h2, palette=pal3, ax=g.ax_marg_y, bins = 20)
g.ax_marg_y.get_legend().remove()

g.ax_joint.set_xlabel('Clones 0/1/3 VAF')
g.ax_joint.set_ylabel('Clone 2 VAF')
```

```python
plotdf.Label.value_counts()
```

```python
plotdf.groupby('Label').total_both.sum()
```

```python
clone_adata = ad.read_h5ad(clustered_adata_form.format(p))
clone_adata.obs.cluster_size = clone_adata.obs.cluster_size.astype(int)
clone_adata.obs[cluster_col] = clone_adata.obs.index.astype(int)
clone_adata.obs['partition'] = clone_adata.obs[cluster_col].map(lambda x:partition[x])
```

```python
chr_start = cn_adata.var.reset_index().merge(scgenome.refgenome.info.chromosome_info[['chr', 'chr_index']], how='left')
if chr_start['chr_index'].isnull().any():
    chromosomes = cn_adata.var['chr'].astype(str).values
    raise ValueError(f'mismatching chromosomes {chromosomes} and {scgenome.refgenome.info.chromosomes}')
chr_start = chr_start[['start', 'chr_index']].values
genome_ordering = np.lexsort(chr_start.transpose())
mat_chrom_idxs = chr_start[genome_ordering][:, 1]
chrom_boundaries = np.array([0] + list(np.where(mat_chrom_idxs[1:] != mat_chrom_idxs[:-1])[0]) + [mat_chrom_idxs.shape[0] - 1])
chrom_sizes = chrom_boundaries[1:] - chrom_boundaries[:-1]
chrom_mids = chrom_boundaries[:-1] + chrom_sizes / 2
annot_labels = [str(i) for i in range(1, 23)] + ['X']
chr_bounds = {ch:(df.start.min(), df.end.max()) for ch, df in cn_adata.var.groupby('chr')}

# add indexing info to SNV table
plotdf['cn_bin'] = part_ad.var.cn_bin
plotdf['cn_bin_xidx'] = cn_adata.var.iloc[genome_ordering].index.get_indexer(plotdf.cn_bin) 
plotdf['cn_bin_prop'] = (plotdf.position % 5e5) / 5e5
plotdf['xpos'] = plotdf.cn_bin_xidx + plotdf.cn_bin_prop
```

```python
f, axes = plt.subplots(2, 2, figsize = (8,4), height_ratios = [1.5, 4], width_ratios = [19, 1],
                       dpi = 300)
axes[1][1].set_axis_off()
matfig = scgenome.pl.plot_cell_cn_matrix(clone_adata, cell_order_fields = ['partition'], ax=axes[1][0])

to_plot = plotdf[plotdf.partial_score.abs() > 1]
sm = axes[0][0].scatter(to_plot.xpos, (to_plot.vaf1 - to_plot.vaf2).abs(), c = to_plot.partial_score, 
                        cmap = 'bwr',  s = to_plot.total_both/15, linewidth = 0.2, edgecolor = 'k',
                         norm = mcolors.TwoSlopeNorm(vmin = -1 * to_plot.partial_score.abs().max(), 
                                                     vcenter = 0,
                                                    vmax = to_plot.partial_score.abs().max()))
axes[0][0].set_ylim(-0.1, 1.1)
for x in chrom_boundaries:
    axes[0][0].axvline(x = x, c = 'k', linewidth = 0.5)
axes[0][0].set_xticks([])
axes[0][0].set_ylabel("|VAF1-VAF2|")
axes[0][0].set_xlim(chrom_boundaries[0], chrom_boundaries[-1])
plt.colorbar(sm, cax=axes[0][1])
axes[1][0].axhline(y=len(clone_adata.obs.query('partition == 1')) - 0.5, c = 'k')
axes[1][0].set_yticks(np.arange(4), 
                      [f'{int(r[cluster_col])} ({int(r.cluster_size)} cells)' for _, r in 
                       clone_adata.obs.sort_values(by = ['partition', 'sbmclone_cluster_id']).iterrows()])
axes[1][0].set_ylabel("SBMClone block")
```

```python
f, axes = plt.subplots(3, 2, figsize = (8,4), height_ratios = [0.75, 0.75, 4], width_ratios = [19, 1],
                       dpi = 300)
axes[1][1].set_axis_off()
matfig = scgenome.pl.plot_cell_cn_matrix(clone_adata, cell_order_fields = ['partition'], ax=axes[2][0])

to_plot = plotdf[plotdf.partial_score.abs() > 1]
sm = axes[0][0].scatter(to_plot.xpos, to_plot.vaf1, c = to_plot.partial_score, 
                        cmap = 'bwr',  s = to_plot.total_both/15, linewidth = 0.2, edgecolor = 'k',
                         norm = mcolors.TwoSlopeNorm(vmin = -1 * to_plot.partial_score.abs().max(), 
                                                     vcenter = 0,
                                                    vmax = to_plot.partial_score.abs().max()))
axes[0][0].set_ylim(-0.1, 1.1)
for x in chrom_boundaries:
    axes[0][0].axvline(x = x, c = 'k', linewidth = 0.5)
axes[0][0].set_xticks([])
axes[0][0].set_ylabel("VAF1")
axes[0][0].set_xlim(chrom_boundaries[0], chrom_boundaries[-1])
plt.colorbar(sm, cax=axes[0][1])

sm = axes[1][0].scatter(to_plot.xpos, to_plot.vaf2, c = to_plot.partial_score, 
                        cmap = 'bwr',  s = to_plot.total_both/15, linewidth = 0.2, edgecolor = 'k',
                         norm = mcolors.TwoSlopeNorm(vmin = -1 * to_plot.partial_score.abs().max(), 
                                                     vcenter = 0,
                                                    vmax = to_plot.partial_score.abs().max()))
axes[1][0].set_ylim(-0.1, 1.1)
for x in chrom_boundaries:
    axes[1][0].axvline(x = x, c = 'k', linewidth = 0.5)
axes[1][0].set_xticks([])
axes[1][0].set_ylabel("VAF2")
axes[1][0].set_xlim(chrom_boundaries[0], chrom_boundaries[-1])

axes[2][0].axhline(y=len(clone_adata.obs.query('partition == 1')) - 0.5, c = 'k')
axes[2][0].set_yticks(np.arange(4), 
                      [f'{int(r[cluster_col])} ({int(r.cluster_size)} cells)' for _, r in 
                       clone_adata.obs.sort_values(by = ['partition', 'sbmclone_cluster_id']).iterrows()])
axes[2][0].set_ylabel("SBMClone block")
```

## 045 second split

```python
p = 'SPECTRUM-OV-045'
partition = (2, 1, 1, 1)
cluster_col = 'sbmclone_cluster_id'

my_results =  all_results[patients.index(p)][cluster_col, partition]
part_ad = my_results['partition_adata']
```

```python
plotdf = part_ad.var.copy()
plotdf['vaf1'] = part_ad.layers['vaf'][0]
plotdf['vaf2'] = part_ad.layers['vaf'][1]

plotdf['total1'] = part_ad.layers['total_count'][0]
plotdf['total2'] = part_ad.layers['total_count'][1]
plotdf['total_both'] = plotdf.total1 + plotdf.total2

plotdf['alt1'] = part_ad.layers['alt_count'][0]
plotdf['alt2'] = part_ad.layers['alt_count'][1]

plotdf['prob_1wgd'] = my_results['prob_1wgd']
plotdf['prob_2wgd'] = my_results['prob_2wgd']
plotdf['partial_score'] = plotdf.prob_2wgd - plotdf.prob_1wgd

plotdf['ml_geno_1wgd'] = my_results['ml_geno_1wgd']
plotdf['ml_geno_2wgd'] = my_results['ml_geno_2wgd']

plotdf['coloring'] = [(0 if a == b else
                      1 if a == [1,1] else 2) for a,b in zip(plotdf.ml_geno_1wgd, plotdf.ml_geno_2wgd)]
plotdf['Label'] = plotdf.coloring.map({0:'Other', 1:'Shared 1-copy', 2:'Exclusive 2-copy'}).astype(
    pd.api.types.CategoricalDtype(categories=['Shared 1-copy', 'Exclusive 2-copy', 'Other'], ordered=True))
```

```python
g = sns.JointGrid(xlim = [-0.1, 1.1], ylim = [-0.1, 1.1])
g.fig.set_dpi(300)

x3, y3, h3, s3 = plotdf.vaf1, plotdf.vaf2, plotdf.Label, plotdf.total_both/5
x2, y2, h2 = plotdf[plotdf.coloring > 0].vaf1, plotdf[plotdf.coloring > 0].vaf2, plotdf[plotdf.coloring > 0].Label
pal3 = ['blue', 'red', 'grey']
sns.scatterplot(x=x3, y=y3, sizes=s3, hue=h3, palette=pal3, ax=g.ax_joint, alpha = 0.2)
g.ax_joint.get_legend().set_title('')

sns.histplot(x=x2, hue=h2, palette=pal3, ax=g.ax_marg_x, bins = 20)
g.ax_marg_x.get_legend().remove()

sns.histplot(y=y2, hue=h2, palette=pal3, ax=g.ax_marg_y, bins = 20)
g.ax_marg_y.get_legend().remove()

g.ax_joint.set_xlabel('Clones 1/2/3 VAF')
g.ax_joint.set_ylabel('Clone 0 VAF')
```

```python
plotdf.Label.value_counts()
```

```python
plotdf.groupby('Label').total_both.sum()
```

```python
clone_adata = ad.read_h5ad(clustered_adata_form.format(p))
clone_adata.obs.cluster_size = clone_adata.obs.cluster_size.astype(int)
clone_adata.obs[cluster_col] = clone_adata.obs.index.astype(int)
clone_adata.obs['partition'] = clone_adata.obs[cluster_col].map(lambda x:partition[x])
```

```python
chr_start = cn_adata.var.reset_index().merge(scgenome.refgenome.info.chromosome_info[['chr', 'chr_index']], how='left')
if chr_start['chr_index'].isnull().any():
    chromosomes = cn_adata.var['chr'].astype(str).values
    raise ValueError(f'mismatching chromosomes {chromosomes} and {scgenome.refgenome.info.chromosomes}')
chr_start = chr_start[['start', 'chr_index']].values
genome_ordering = np.lexsort(chr_start.transpose())
mat_chrom_idxs = chr_start[genome_ordering][:, 1]
chrom_boundaries = np.array([0] + list(np.where(mat_chrom_idxs[1:] != mat_chrom_idxs[:-1])[0]) + [mat_chrom_idxs.shape[0] - 1])
chrom_sizes = chrom_boundaries[1:] - chrom_boundaries[:-1]
chrom_mids = chrom_boundaries[:-1] + chrom_sizes / 2
annot_labels = [str(i) for i in range(1, 23)] + ['X']
chr_bounds = {ch:(df.start.min(), df.end.max()) for ch, df in cn_adata.var.groupby('chr')}

# add indexing info to SNV table
plotdf['cn_bin'] = part_ad.var.cn_bin
plotdf['cn_bin_xidx'] = cn_adata.var.iloc[genome_ordering].index.get_indexer(plotdf.cn_bin) 
plotdf['cn_bin_prop'] = (plotdf.position % 5e5) / 5e5
plotdf['xpos'] = plotdf.cn_bin_xidx + plotdf.cn_bin_prop
```

```python
f, axes = plt.subplots(2, 2, figsize = (8,4), height_ratios = [1.5, 4], width_ratios = [19, 1],
                       dpi = 300)
axes[1][1].set_axis_off()
matfig = scgenome.pl.plot_cell_cn_matrix(clone_adata, cell_order_fields = ['partition'], ax=axes[1][0])

to_plot = plotdf[plotdf.partial_score.abs() > 1]
sm = axes[0][0].scatter(to_plot.xpos, (to_plot.vaf1 - to_plot.vaf2).abs(), c = to_plot.partial_score, 
                        cmap = 'bwr',  s = to_plot.total_both/15, linewidth = 0.2, edgecolor = 'k',
                         norm = mcolors.TwoSlopeNorm(vmin = -1 * to_plot.partial_score.abs().max(), 
                                                     vcenter = 0,
                                                    vmax = to_plot.partial_score.abs().max()))
axes[0][0].set_ylim(-0.1, 1.1)
for x in chrom_boundaries:
    axes[0][0].axvline(x = x, c = 'k', linewidth = 0.5)
axes[0][0].set_xticks([])
axes[0][0].set_ylabel("|VAF1-VAF2|")
axes[0][0].set_xlim(chrom_boundaries[0], chrom_boundaries[-1])
plt.colorbar(sm, cax=axes[0][1])
axes[1][0].axhline(y=len(clone_adata.obs.query('partition == 1')) - 0.5, c = 'k')
axes[1][0].set_yticks(np.arange(4), 
                      [f'{int(r[cluster_col])} ({(r.cluster_size)} cells)' for _, r in 
                       clone_adata.obs.sort_values(by = ['partition', 'sbmclone_cluster_id']).iterrows()])
axes[1][0].set_ylabel("SBMClone block")
```

```python

```

# incorporate Andrew figures also


## 045

```python
p = 'SPECTRUM-OV-045'
```

```python
sbm_adata = ad.read_h5ad(os.path.join(sbm_adata_stem, f'sbmclone_{p}_snv.h5'))
blocks_adata = get_blocks_adata(p)
blocks_adata.layers['ml_genotype'] = np.stack([blocks_adata.layers['p_cn0'], blocks_adata.layers['p_cn1'], blocks_adata.layers['p_cn2']]).argmax(axis = 0)

blocks_adata = blocks_adata[blocks_adata.obs.index != 'nan']
blocks_adata = blocks_adata[blocks_adata.obs.index.astype(int) >= 0]

clones = blocks_adata.obs.index

fig, axes = plt.subplots(nrows=len(clones)-1, ncols=len(clones)-1, figsize=(5, 5), dpi=150)

for block1, block2 in itertools.combinations(clones, 2):    
    ax = axes[int(block1), int(block2)-1]
    plot_data = pd.DataFrame(blocks_adata[[block1, block2]].layers['ml_genotype'].T
                            ).groupby([0, 1]).size().unstack(fill_value=0).iloc[::-1, :]
    vmax = max(plot_data.loc[0, 2], plot_data.loc[1, 1], plot_data.loc[2, 0])
    # plot_data.loc[0, 0] = np.NaN
    # plot_data.loc[2, 2] = np.NaN

    sns.heatmap(ax=ax, data=plot_data, vmax=vmax, cbar=False, annot=True, fmt='d', annot_kws={"fontsize":8})

    if block1 == clones[0]:
        ax.set_xlabel(f'clone {block2}')
        ax.xaxis.set_label_position('top')
    else:
        ax.set_xlabel('')
    if block2 == clones[-1]:
        ax.set_ylabel(f'clone {block1}', labelpad=15)
        ax.yaxis.set_label_position('right')
        ax.yaxis.label.set(rotation=270)
    else:
        ax.set_ylabel('')
    ax.tick_params(axis='y', labelrotation=0)
        
    if int(block1) != int(block2)-1:
        ax.set_xticks([])
        ax.set_yticks([])
    
for row in range(blocks_adata.obs.shape[0]-1):
    for col in range(blocks_adata.obs.shape[0]-1):
        ax = axes[row, col]
        if row > col:
            ax.set_xticks([])
            ax.set_yticks([])
            ax.axis('off')

plt.subplots_adjust(wspace=0.1, hspace=0.1)

```

```python
my_pair_adata = blocks_adata[['3', '2']].copy()
my_pair_adata.var['cn'] = ['/'.join([str(a) for a in g]) for g in my_pair_adata.layers['ml_genotype'].T]
```

```python
my_snvs = np.abs(2-my_pair_adata.layers['ml_genotype'].sum(axis = 0)) < 2
sns.scatterplot(x=my_pair_adata[:, my_snvs].layers['vaf'][0], 
                y=my_pair_adata[:, my_snvs].layers['vaf'][1], 
                size=np.min(my_pair_adata[:, my_snvs].layers['total_count'], axis = 0), 
                hue=my_pair_adata[:, my_snvs].var.cn)
sns.move_legend(plt.gca(), 'upper left', bbox_to_anchor=(1, 1), ncol=1, title=None, frameon=False)
sns.despine(trim=True)
```

```python
sns.kdeplot(x=my_pair_adata[:, my_snvs].layers['vaf'][0], 
            y=my_pair_adata[:, my_snvs].layers['vaf'][1], clip=((0, 1), (0, 1)), fill=True)#, hue='cn')
sns.despine(trim=True)

```

```python

snv_pair_utility = {
    '0/0': '0/0 (uninformative)',
    '0/1': '0/1 (uninformative)',
    '1/0': '1/0 (uninformative)',
    '2/2': '2/2 (uninformative)',
    '2/0': '2/0 (independent)',
    '0/2': '0/2 (independent)',
    '1/1': '1/1 (common)',
    '1/2': '1/2 (contradictory)',
    '2/1': '2/1 (contradictory)',
}

snv_pair_utility_colors = {
    '0/0 (uninformative)': '#8B5E3C',
    '0/1 (uninformative)': '#D7DF23',
    '1/0 (uninformative)': '#F7941D',
    '2/2 (uninformative)': '#2E3192',
    '2/0 (independent)': '#EF4136',
    '0/2 (independent)': '#006838',
    '1/1 (common)': '#27AAE1',
    '1/2 (contradictory)': '#8B5E3C',
    '2/1 (contradictory)': '#8B5E3C',
}
```

```python

clones = blocks_adata.obs.index
snv_classes = [
    '0/0 (uninformative)',
    '0/1 (uninformative)',
    '1/0 (uninformative)',
    '2/2 (uninformative)',
    '2/0 (independent)',
    '0/2 (independent)',
    '1/1 (common)',
    '1/2 (contradictory)',
    '2/1 (contradictory)',
]

fig, axes = plt.subplots(nrows=len(clones)-1, ncols=len(clones)-1, figsize=(4, 4), dpi=150, sharex=True, sharey=True)

for block1, block2 in itertools.combinations(clones, 2):
    my_pair_adata = blocks_adata[[block1, block2]].copy()
    my_pair_adata.var['cn'] = ['/'.join([str(a) for a in g]) for g in my_pair_adata.layers['ml_genotype'].T]
    my_pair_adata.var['SNV class'] = my_pair_adata.var.cn.map(snv_pair_utility)
    my_pair_adata = my_pair_adata[:, ~my_pair_adata.var.cn.isin(['0/0', '2/2'])]
    
    block1_idx, block2_idx = sorted([int(block1), int(block2)])
    ax = axes[block2_idx - 1, block1_idx]
    sns.kdeplot(ax=ax, 
                x=my_pair_adata.layers['vaf'][0], 
                y=my_pair_adata.layers['vaf'][1],
                fill=True, bw_adjust=0.5, color='slategrey')#, clip=((-0.1, 1.1), (-0.1, 1.1)))#bw_method=.1)#
    sns.scatterplot(ax=ax, 
                    x=my_pair_adata.layers['vaf'][0], 
                    y=my_pair_adata.layers['vaf'][1], 
                    hue=my_pair_adata.var['SNV class'], 
                    hue_order=snv_classes, s=100, alpha=0.2, linewidth=0, palette=snv_pair_utility_colors)

    if block1_idx == 0:
        ax.set_ylabel(f'clone {block2}', labelpad=10)
        
    if block2_idx == len(clones) - 1:
        ax.set_xlabel(f'clone {block1}', labelpad=10)

    ax.tick_params(axis='y', labelrotation=0)

    ax.set_xticks([0, 0.5, 1])
    ax.set_yticks([0, 0.5, 1])
    
    sns.despine(ax=ax, trim=True, offset=-5)
    if block1_idx == 0 and block2_idx == 1:
        sns.move_legend(
            ax, 'upper left',
            bbox_to_anchor=(2, 0.9), ncol=1, frameon=False,
            title='Variant copy number (x/y)',
            markerscale=0.5,
            prop={'size': 8}, title_fontsize=10,
            labelspacing=0.4, handletextpad=0, columnspacing=0.5,
        )
    else:
        if ax.get_legend():
            ax.get_legend().remove()
    
for row in range(len(clones)-1):
    for col in range(len(clones)-1):
        if col > row:
            ax = axes[row, col]
            ax.axis('off')

plt.subplots_adjust(wspace=0.02, hspace=0.02)

```

# new plots


## count mutally exclusive SNVs

```python
Counter([tuple(a) for a in blocks_adata.layers['ml_genotype'].T if not np.any(a == 1)])
```

```python
blocks_adata.var['is_mutual_exclusive'] = np.logical_and(~np.any(blocks_adata.layers['ml_genotype'] == 1, axis = 0),
                                                         np.sum(blocks_adata.layers['ml_genotype'], axis = 0) > 0)
```

```python
blocks_adata.var['in_clone0'] = blocks_adata.layers['ml_genotype'][0] > 0
blocks_adata.var['in_clone1'] = blocks_adata.layers['ml_genotype'][1] > 0
blocks_adata.var['in_clone2'] = blocks_adata.layers['ml_genotype'][2] > 0
blocks_adata.var['in_clone3'] = blocks_adata.layers['ml_genotype'][3] > 0
```

```python
blocks_adata.var['total_both'] = blocks_adata.layers['total_count'].sum(axis = 0)

```

```python
blocks_adata.var['n_containing_cells'] = blocks_adata.obs.cluster_size.values[1-
blocks_adata.var[['in_clone0', 'in_clone1', 'in_clone2', 'in_clone3']].values.astype(int)].sum(axis = 1).astype(int)
```

```python
fig, ax = plt.subplots(1, 3, width_ratios = [4, 1, 1], dpi = 300)
mydf = blocks_adata.var[blocks_adata.var.is_mutual_exclusive][['in_clone0', 'in_clone1', 'in_clone2', 'in_clone3']].value_counts().reset_index()
mydf = mydf.merge(blocks_adata.var[blocks_adata.var.is_mutual_exclusive].groupby(['in_clone0', 'in_clone1', 'in_clone2', 'in_clone3']).total_both.sum().reset_index())
mydf = mydf.merge(blocks_adata.var[blocks_adata.var.is_mutual_exclusive].groupby(['in_clone0', 'in_clone1', 'in_clone2', 'in_clone3']).n_containing_cells.mean().reset_index())
mydf = mydf.rename(columns={'count':'n_snvs', 'total_both':'total_reads'})
mydf.total_reads = mydf.total_reads.astype(int)
mydf.n_containing_cells = mydf.n_containing_cells.astype(int)

mydf = mydf.iloc[1:]

mydf = mydf.sort_values(by = 'total_reads', ascending=False)
n_categories = len(mydf)
sns.heatmap(mydf.iloc[:, :-3], annot=mydf.iloc[:, :-3].astype(str), fmt='s', ax=ax[0], cbar=False)
ax[0].set_yticks([], [])
ax[0].set_title("Clones containing SNV")

ax[1].barh(y=np.arange(len(mydf))[::-1], width=mydf.total_reads, height=0.5)
ax[1].set_yticks(np.arange(n_categories)[::-1], mydf.total_reads)
ax[1].set_title("Supporting\nreads")


ax[2].barh(y=np.arange(len(mydf))[::-1], width=mydf.n_snvs, height=0.5)
ax[2].set_yticks(np.arange(n_categories)[::-1], mydf.n_snvs)
ax[2].set_title("Num. SNVs")


ax[0].set_xticklabels(['Clone 0', 'Clone 1', 'Clone 2', 'Clone 3'])
plt.tight_layout()
```

```python

```
