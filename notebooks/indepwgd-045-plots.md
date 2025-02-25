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

<!-- #raw -->
print(datetime.now())
with Pool(8) as p:
    all_results = p.map(run_patient_partitions, patients)
print(datetime.now())
<!-- #endraw -->

<!-- #raw -->
with open('indepwgdtest_results.pickle', 'wb') as f:
    pickle.dump(all_results, f)
<!-- #endraw -->

```python
all_results = pickle.load(open('indepwgdtest_results.pickle', 'rb'))
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
real_results = []
for _, df in sbmdf[(sbmdf.score > 0) & (sbmdf.pvalue < 0.9)].groupby(by = 'patient'):
    real_results.append(df.iloc[df.score.argmax()])
real_results = pd.DataFrame(real_results)
real_results
```

```python
sbmdf[sbmdf.patient == 'SPECTRUM-OV-045'].sort_values(by = 'score', ascending=False)
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

# new-style VAF plots


## copied functions

```python
import met_brewer
colors = met_brewer.met_brew(name="Degas", n=6, brew_type="continuous")

snv_pair_utility = {
    '0/1': '0/1 (uninformative)',
    '1/0': '1/0 (uninformative)',
    '2/2': '2/2 (uninformative)',
    '2/0': '2/0 (independent)',
    '0/2': '0/2 (independent)',
    '1/1': '1/1 (common)',
}


snv_pair_utility_colors = {
    '0/1 (uninformative)':colors[2],
    '0/2 (independent)':colors[1],
    
    '1/0 (uninformative)':colors[4],
    '2/0 (independent)':colors[5],
    
    '1/1 (common)':colors[3],
    '2/2 (uninformative)':colors[0],
}

snv_classes = snv_pair_utility_colors.keys()


c2snv3 = {v:k for k,v in snv_pair_utility_colors.items()}

plt.figure(figsize=(1, 4), dpi = 150)
plt.scatter(np.zeros(len(colors)), np.arange(len(colors)), c = colors, s = 250)
plt.xticks([], [])
plt.yticks(np.arange(len(colors)), [c2snv3[colors[i]] for i in range(len(colors))])
plt.title("Colors")
```

```python

def compute_ml_genotypes_vec(alt1, tot1, alt2, tot2, epsilon = 0.01, return_genotypes = False):
    # allowed in either 1 or 2 WGDs
    prob00 = binom.logpmf(k=alt1, n=tot1, p=epsilon) + binom.logpmf(k=alt2, n=tot2, p=epsilon)
    prob10 = binom.logpmf(k=alt1, n=tot1, p=0.5) + binom.logpmf(k=alt2, n=tot2, p=epsilon)
    prob01 = binom.logpmf(k=alt1, n=tot1, p=epsilon) + binom.logpmf(k=alt2, n=tot2, p=0.5)
    prob22 = binom.logpmf(k=alt1, n=tot1, p=1 - epsilon) + binom.logpmf(k=alt2, n=tot2, p=1 - epsilon)

    # not allowed in either
    #prob12 = binom.logpmf(k=alt1, n=tot1, p=0.5) * binom.logpmf(k=alt2, n=tot2, p=1 - epsilon)
    #prob21 = binom.logpmf(k=alt1, n=tot1, p=1-epsilon) * binom.logpmf(k=alt2, n=tot2, p=0.5)

    # exclusive to 1 WGD (shared 1-copy post-WGD)
    prob11 = binom.logpmf(k=alt1, n=tot1, p=0.5) + binom.logpmf(k=alt2, n=tot2, p=0.5)

    # exclusive to 2 WGDs (distinct 2-copy pre-WGD)
    prob02 = binom.logpmf(k=alt1, n=tot1, p=epsilon) + binom.logpmf(k=alt2, n=tot2, p=1 - epsilon)
    prob20 = binom.logpmf(k=alt1, n=tot1, p=1 - epsilon) + binom.logpmf(k=alt2, n=tot2, p=epsilon)
    
    probs1 = np.vstack([prob00, prob10, prob01, prob11, prob22])
    probs2 = np.vstack([prob00, prob10, prob01, prob02, prob20, prob22])

    if return_genotypes:
        best_geno1 = np.argmax(probs1, axis = 0)
        best_geno2 = np.argmax(probs2, axis = 0)
        
        genos1 = [[0,0], [1,0], [0,1], [1,1], [2,2]]
        genos2 = [[0,0], [1,0], [0,1], [0,2], [2,0], [2,2]]
        g1 = [genos1[i] for i in best_geno1]
        g2 = [genos2[i] for i in best_geno2]
        cidx = np.arange(probs1.shape[1])
        
        return g1, g2, probs1[best_geno1, cidx], probs2[best_geno2, cidx]
    else:
        return np.max(probs1, axis = 0), np.max(probs2, axis = 0)


def get_block_snvs(blocks_adata, block_name, epsilon=0.01):
    block_snvs = scgenome.tl.get_obs_data(
        blocks_adata, block_name,
        var_columns=['chromosome', 'position', 'ref', 'alt', 'is_cnloh'],
        layer_names=['ref_count', 'alt_count', 'total_count', 'vaf'])
    block_snvs = block_snvs[block_snvs['total_count'] > min_snvcov_reads]
    block_snvs = block_snvs[block_snvs['is_cnloh']]
    
    block_snvs['p_cn0'] = binom.logpmf(k=block_snvs['alt_count'], n=block_snvs['total_count'], p=epsilon)
    block_snvs['p_cn1'] = binom.logpmf(k=block_snvs['alt_count'], n=block_snvs['total_count'], p=0.5)
    block_snvs['p_cn2'] = binom.logpmf(k=block_snvs['alt_count'], n=block_snvs['total_count'], p=1-epsilon)
    block_snvs['cn'] = block_snvs[['p_cn0', 'p_cn1', 'p_cn2']].idxmax(axis=1).str.replace('p_cn', '').astype(int)

    return block_snvs


def get_block_snvs_pairs(blocks_adata, block1, block2):
    block1_snvs = get_block_snvs(blocks_adata, block1)
    block2_snvs = get_block_snvs(blocks_adata, block2)

    if block1_snvs.shape[0] == 0 or block2_snvs.shape[0] == 0:
        return None
    
    block_snvs_pairs = block1_snvs.merge(block2_snvs, on=['chromosome', 'position', 'ref', 'alt'], suffixes=['_1', '_2'])

    g1, g2, p1, p2 = compute_ml_genotypes_vec(
        block_snvs_pairs['alt_count_1'], block_snvs_pairs['total_count_1'],
        block_snvs_pairs['alt_count_2'], block_snvs_pairs['total_count_2'],
        return_genotypes=True)

    g1 = np.array(g1)
    g2 = np.array(g2)
    
    block_snvs_pairs['wgd1_1'] = g1[:, 0]
    block_snvs_pairs['wgd1_2'] = g1[:, 1]

    block_snvs_pairs['wgd2_1'] = g2[:, 0]
    block_snvs_pairs['wgd2_2'] = g2[:, 1]

    block_snvs_pairs['p_wgd1'] = p1
    block_snvs_pairs['p_wgd2'] = p2

    block_snvs_pairs['cn'] = block_snvs_pairs['cn_1'].astype(str) + '/' + block_snvs_pairs['cn_2'].astype(str)
    
    return block_snvs_pairs

```

## adapt dataframe processing and apply to 1 pair at a time

```python

```

```python
def get_block_snvs(blocks_adata, block_name, epsilon=0.01):
    block_snvs = scgenome.tl.get_obs_data(
        blocks_adata, block_name,
        var_columns=['chromosome', 'position', 'ref', 'alt', 'is_cnloh'],
        layer_names=['ref_count', 'alt_count', 'total_count', 'vaf'])
    block_snvs = block_snvs[block_snvs['total_count'] > min_snvcov_reads]
    block_snvs = block_snvs[block_snvs['is_cnloh']]
    
    block_snvs['p_cn0'] = binom.logpmf(k=block_snvs['alt_count'], n=block_snvs['total_count'], p=epsilon)
    block_snvs['p_cn1'] = binom.logpmf(k=block_snvs['alt_count'], n=block_snvs['total_count'], p=0.5)
    block_snvs['p_cn2'] = binom.logpmf(k=block_snvs['alt_count'], n=block_snvs['total_count'], p=1-epsilon)
    block_snvs['cn'] = block_snvs[['p_cn0', 'p_cn1', 'p_cn2']].idxmax(axis=1).str.replace('p_cn', '').astype(int)

    return block_snvs


```

```python
margin = 0.05
hist_ratio = 0.2
ylim = [-margin, 1 + margin]
xlim = [-margin, 1 + margin]
ticks = [0, 0.5, 1]
dpi = 250

min_snvcov_reads = 10
```

```python
block_snvs_pairs['SNV class']
```

```python
def plot_pairwise_vafs(axes, block_snvs_pairs):
    sns.scatterplot(ax=axes[1][0], x='vaf_1', y='vaf_2', hue='SNV class', hue_order=snv_classes, data=block_snvs_pairs, palette=snv_pair_utility_colors,s=25, alpha=0.7, linewidth=0)
    sns.kdeplot(ax=axes[1][0], x='vaf_1', y='vaf_2', data=block_snvs_pairs, fill=True, bw_adjust=0.5, color='slategrey', zorder=-100, alpha=0.4, 
                clip=((0., 1.), (0., 1.)), levels=5)#, alpha=0.1, lw=1)#, clip=((-0.1, 1.1), (-0.1, 1.1)))#bw_method=.1)#
    axes[1][0].set_xlim(xlim)
    axes[1][0].set_ylim(ylim)
    axes[1][0].set_xticks(ticks)
    axes[1][0].set_yticks(ticks)
    
    # horizontal marginal histogram
    sns.histplot(ax=axes[0][0], x='vaf_1', hue='SNV class', hue_order=snv_classes, data=block_snvs_pairs, bins=10,
                 palette=snv_pair_utility_colors, legend=False)
    axes[0][0].set_xlabel('')
    axes[0][0].set_ylabel('')
    axes[0][0].set_xticklabels([])
    axes[0][0].set_xticks(axes[1][0].get_xticks())
    axes[0][0].set_xlim(xlim)
    #axes[0][0].set_yticklabels([])
    sns.despine()
    
    # vertical marginal histogram
    sns.histplot(ax=axes[1][1], y='vaf_2', hue='SNV class', hue_order=snv_classes, data=block_snvs_pairs, bins=10,
                 palette=snv_pair_utility_colors, legend=False)
    axes[1][1].set_xlabel('')
    axes[1][1].set_ylabel('')
    #axes[1][1].set_xticklabels([])
    axes[1][1].set_yticks(axes[1][0].get_yticks())
    axes[1][1].set_ylim(ylim)
    axes[1][1].set_yticklabels([])
    sns.despine()
    


```

```python
fig = plt.figure(figsize=(8, 4), dpi=dpi)
subfigs = fig.subfigures(1, 2)

### first subplot
p = 'SPECTRUM-OV-045'
partition = (2, 1, 1, 1)
cluster_col = 'sbmclone_cluster_id'

my_results =  all_results[patients.index(p)][cluster_col, partition]
part_ad = my_results['partition_adata']

block_snvs_pairs = get_block_snvs_pairs(part_ad, '0', '1')
block_snvs_pairs = block_snvs_pairs[~block_snvs_pairs['cn'].isin(['0/0', '1/2', '2/1'])]
block_snvs_pairs['SNV class'] = block_snvs_pairs['cn'].map(snv_pair_utility)
block_snvs_pairs = block_snvs_pairs[block_snvs_pairs['SNV class'] != 'Non-phylogenetic']

axes = subfigs[0].subplots(2, 2, width_ratios = [0.9, hist_ratio], height_ratios = [hist_ratio, 0.9])

plot_pairwise_vafs(axes, block_snvs_pairs)
sns.move_legend(
    axes[1][0], 'upper left',
    bbox_to_anchor=(0.9, 1.6), ncol=1, frameon=False,
    title='Variant copy number (X/Y)',
    markerscale=1.2,
    prop={'size': 8}, title_fontsize=10,
    labelspacing=0.4, handletextpad=0, columnspacing=0.5)

axes[1][0].set_xlabel('clones ' + part_ad.obs.loc['0', 'blocks'])
axes[1][0].set_ylabel('clone ' + part_ad.obs.loc['1', 'blocks'])
axes[0][1].set_visible(False)


### second subplot
p = 'SPECTRUM-OV-045'
partition = (1, 1, 2, 1)
cluster_col = 'sbmclone_cluster_id'

cn_adata = ad.read_h5ad(clustered_adata_form.format(p))
my_results =  all_results[patients.index(p)][cluster_col, partition]
part_ad = my_results['partition_adata']

block_snvs_pairs = get_block_snvs_pairs(part_ad, '0', '1')
block_snvs_pairs = block_snvs_pairs[~block_snvs_pairs['cn'].isin(['0/0', '1/2', '2/1'])]
block_snvs_pairs['SNV class'] = block_snvs_pairs['cn'].map(snv_pair_utility)
block_snvs_pairs = block_snvs_pairs[block_snvs_pairs['SNV class'] != 'Non-phylogenetic']


axes = subfigs[1].subplots(2, 2, width_ratios = [0.9, hist_ratio], height_ratios = [hist_ratio, 0.9])
plot_pairwise_vafs(axes, block_snvs_pairs)

axes[1][0].set_xlabel('clones ' + part_ad.obs.loc['0', 'blocks'])
axes[1][0].set_ylabel('clone ' + part_ad.obs.loc['1', 'blocks'])
axes[1][0].get_legend().remove()


axes[0][1].set_visible(False)

```

```python

```

# count mutally exclusive SNVs

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
