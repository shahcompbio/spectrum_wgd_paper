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
import tqdm
```

```python
pipeline_outputs = pipeline_dir # path to root directory of scWGS pipeline outputs
colors_dict = yaml.safe_load(open('repos/spectrum-genomic-instability/resources/annotation/colors.yaml', 'r'))

```

```python
sbm_adata_stem = os.path.join(pipeline_outputs, 'sbmclone')
patients = [a.split('_')[0] for a in os.listdir(sbm_adata_stem)]
sbm_adata_form = os.path.join(pipeline_outputs, 'sbmclone', 'sbmclone_{}_snv.h5')
sbm_adata_form = os.path.join(pipeline_outputs, 'sbmclone', 'sbmclone_{}_snv.h5')

clustered_adata_form =  os.path.join(pipeline_outputs, 'tree_snv/inputs/{}_cna_clustered.h5')
clone_adata_form =  os.path.join(pipeline_outputs, 'tree_snv/inputs/{}_general_clone_adata.h5')
signals_adata_form =  os.path.join(pipeline_outputs, 'preprocessing/signals/signals_{}.h5')
patients = sorted(np.unique([a.split('_')[0] for a in os.listdir(sbm_adata_stem)]))
```

```python
def get_blocks_adata(adata, p, epsilon = 0.001):
    blocks_adata = scgenome.tl.aggregate_clusters(
        adata[~adata.obs['sbmclone_cluster_id'].isna()],
        agg_layers={
            'alt': "sum",
            'ref': "sum",
            'A': "median",
            'B': "median",
            'state': "median",
            'total': "sum"
        },
        cluster_col='sbmclone_cluster_id')
    blocks_adata.layers['vaf'] = blocks_adata.layers['alt'] / np.maximum(1, blocks_adata.layers['total'])
    blocks_adata.layers['p_cn0'] = binom.logpmf(k=blocks_adata.layers['alt'], n=blocks_adata.layers['total'], p = epsilon)
    blocks_adata.layers['p_cn1'] = binom.logpmf(k=blocks_adata.layers['alt'], n=blocks_adata.layers['total'], p = 0.5)
    blocks_adata.layers['p_cn2'] = binom.logpmf(k=blocks_adata.layers['alt'], n=blocks_adata.layers['total'], p = 1-epsilon)
    blocks_adata.layers['alt_count'] = blocks_adata.layers['alt']
    blocks_adata.layers['ref_count'] = blocks_adata.layers['ref']
    blocks_adata.layers['total_count'] = blocks_adata.layers['total']
    
    cn_adata = ad.read_h5ad(clustered_adata_form.format(p))
    cn_adata.var['is_cnloh'] = np.logical_and(cn_adata.var.is_homogenous_cn, np.logical_or(
        np.logical_and(np.all(cn_adata.layers['A'] == 2, axis = 0), np.all(cn_adata.layers['B'] == 0, axis = 0)),
        np.logical_and(np.all(cn_adata.layers['A'] == 0, axis = 0), np.all(cn_adata.layers['B'] == 2, axis = 0))))
    
    blocks_adata.var['is_cnloh'] = cn_adata.var.loc[blocks_adata.var.cn_bin, 'is_cnloh'].values
    blocks_adata = blocks_adata[:, blocks_adata.var['is_cnloh']]
    return blocks_adata
```

```python



def get_partition(blocks_adata, partition, epsilon = 0.001, min_snvcov_reads = 2):
    #assert len(partition) == len(blocks_adata), (len(partition), len(blocks_adata))
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
    new_blocks_adata.obs['partition_size'] = [sum([blocks_adata.obs.cluster_size.iloc[i] for i in part1_idx]),
                                              sum([blocks_adata.obs.cluster_size.iloc[i] for i in part2_idx])]
    new_blocks_adata.obs['blocks'] = ['/'.join([blocks_adata.obs.iloc[a].name for a in l]) for l in [part1_idx, part2_idx]]
    
    new_blocks_adata.layers['alt_count'][1] = blocks_adata[part2_idx].layers['alt_count'].toarray().sum(axis = 0)
    new_blocks_adata.layers['ref'][1] = blocks_adata[part2_idx].layers['ref'].toarray().sum(axis = 0)
    new_blocks_adata.layers['total_count'][1] = blocks_adata[part2_idx].layers['total_count'].toarray().sum(axis = 0)
    new_blocks_adata.layers['B'][1] = np.median(blocks_adata[part2_idx].layers['B'], axis = 0)
    new_blocks_adata.layers['A'][1] = np.median(blocks_adata[part2_idx].layers['A'], axis = 0)
    new_blocks_adata.layers['state'][1] = np.median(blocks_adata[part2_idx].layers['state'], axis = 0)
    
    new_blocks_adata.layers['vaf'] = new_blocks_adata.layers['alt_count'] / np.maximum(1, new_blocks_adata.layers['total_count'])
    
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
            blocks_adata = get_blocks_adata(sbm_adata, p)
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

```python
def run_subsample_test(patient_id, sbm_adata, prop_cells, prop_reads, seed, partition, epsilon=0.001, min_snvcov_reads=2, n_iter=1000):
    """
        patient_id: SPECTRUM-OV-XXX
        sbm_adata: sbmclone anndata with rows corresponding to cells and columns corresponding to SNVs
        prop_cells: proportion of cells to sample
        prop_reads: proportion of reads to sample
        seed: RNG seed
        partition: ternary tuple of length equal to the number of unique clones in sbm_adata.
            partition[i] = 1 if the clone at index i should be in partition 1, 
            partition[i] = 2 if the clone at index i should be in partition 2,
            or partition[i] = 0 if the clone at index i should be excluded from the bipartition

    """
    result = {}

    my_adata = sbm_adata

    if np.any(np.array(partition) == 0):
        # sample cells from valid clones only
        all_clones = set(sorted(my_adata.obs.sbmclone_cluster_id.unique()))
        valid_clones = sorted(all_clones - set(np.where(np.array(partition) == 0)[0]))
        valid_cells = my_adata[my_adata.obs.sbmclone_cluster_id.isin(valid_clones)].obs.index
        
        np.random.seed(seed)
        my_cells = np.random.choice(valid_cells, size=int(prop_cells * len(valid_cells)), replace=False)
    
        # supplement with 1 cell from each invalid clone so that blocks_adata still has all clones
        for j in all_clones - set(valid_clones):
            my_cells = np.concatenate([my_cells, [my_adata[my_adata.obs.sbmclone_cluster_id == j].obs.index[0]]])
    
    else:
        if prop_cells < 1:
            np.random.seed(seed)
            my_cells = np.random.choice(my_adata.obs.index, size=int(prop_cells * my_adata.shape[0]), replace=False)
        else:
            my_cells = sbm_adata.obs.index
    
    my_adata = sbm_adata[my_cells].copy()    
    if prop_reads < 1:
        nz_idx_alt = np.where(my_adata.layers['alt'].toarray() > 0)
        my_adata.layers['alt'][nz_idx_alt] = binom.rvs(p=prop_reads, n=my_adata.layers['alt'][nz_idx_alt].astype(int))

        nz_idx_ref = np.where(my_adata.layers['ref'].toarray() > 0)
        my_adata.layers['ref'][nz_idx_ref] = binom.rvs(p=prop_reads, n=my_adata.layers['ref'][nz_idx_ref].astype(int))
        my_adata.layers['total'] = my_adata.layers['alt'] + my_adata.layers['ref']
    
    my_block_ad = get_blocks_adata(my_adata, patient_id).copy()    
    part1_idx = np.where(np.array(partition) == 1)[0]
    part2_idx = np.where(np.array(partition) == 2)[0]
    try:
        n_cells_part1 = my_block_ad.obs.cluster_size.iloc[part1_idx].sum()
    except IndexError:
        return {'blocks_adata':my_block_ad, 'n_cells_part1':-1, 'n_cells_part2':-1}
    try:
        n_cells_part2 = my_block_ad.obs.cluster_size.iloc[part2_idx].sum()
    except IndexError:
        return {'blocks_adata':my_block_ad, 'n_cells_part1':n_cells_part1, 'n_cells_part2':-1}

    if n_cells_part1 == 0 or n_cells_part2 == 0:
        return {'blocks_adata':my_block_ad, 'n_cells_part1':n_cells_part1, 'n_cells_part2':n_cells_part2}
    else:
        my_partition_ad = get_partition(my_block_ad, partition, epsilon = epsilon, min_snvcov_reads = min_snvcov_reads)
        result['prob_1wgd'], result['prob_2wgd'], result['ml_geno_1wgd'], result['ml_geno_2wgd'] = compute_ll(my_partition_ad, return_ml_genotypes = True)
        result['score'] = result['prob_2wgd'].sum() - result['prob_1wgd'].sum()
        
        result['null_scores'] =  np.array(generate_null_resample(my_partition_ad, result['ml_geno_1wgd'], n_iter = n_iter))
        result['pvalue'] = np.sum(result['null_scores'] > result['score']) / n_iter
        result['partition_adata'] = my_partition_ad
        result['blocks_adata'] = my_block_ad
        return result

def run_subsample_test_wrapper(params):
    return run_subsample_test(*params)
```

# take 025 and 045, subsample cells, and test sensitivity

<!-- #raw -->
finaldf025 = pd.read_csv('rev_indepwgd_test_025.csv')
finaldf045 = pd.read_csv('rev_indepwgd_test_045_2vsrest.csv')
finaldf045_2 = pd.read_csv('rev_indepwgd_test_045_1vs03.csv')

finaldf036 = pd.read_csv('rev_indepwgd_test_036.csv')
finaldf044 = pd.read_csv('rev_indepwgd_test_044.csv')
finaldf052 = pd.read_csv('rev_indepwgd_test_052.csv')
<!-- #endraw -->

## first, try subsampling cells symmetrically

```python
p = 'SPECTRUM-OV-025'
sbm_adata = ad.read_h5ad(sbm_adata_form.format(p))
partition = (1, 1, 2)

subsample_props = [0.01, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1]
n_repetitions = 10
```

<!-- #raw -->
my_ad = sbm_adata.copy()
nz_idx = np.where(my_ad.layers['alt'].toarray() > 0)
my_ad.layers['alt'][nz_idx] = binom.rvs(p=0.5, n=my_ad.layers['alt'][nz_idx].astype(int))
<!-- #endraw -->

```python
partition
```

# focus on cell subsampling only

```python
sbm_adata
```

```python
prop_cells_final = [0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.25, 0.5]
partition_025 = (1,1,2)
paramset_final_025 = [('SPECTRUM-OV-025', sbm_adata, propc, propr, seed, partition_025) for propr in [1] for propc in prop_cells_final for seed in range(n_repetitions)]
paramset_final_025.append(('SPECTRUM-OV-025', sbm_adata, 1, 1, 0, partition_025))
len(paramset_final_025)
```

```python
r_ = []
for params in paramset_final_025:
    r_.append(run_subsample_test_wrapper(params))
```

<!-- #raw -->
print(datetime.now())
subsample_results_025f = list(map(run_subsample_test_wrapper, paramset_final_025))
print(datetime.now())
<!-- #endraw -->

<!-- #raw -->
print(datetime.now())
with Pool(5) as pool:
    subsample_results_025f = pool.map(run_subsample_test_wrapper, paramset_final_025)

print(datetime.now())

<!-- #endraw -->

```python
rows025 = []
for params, result in zip(paramset_final_025, r_):
    row = {}
    _, _, row['prop_cells'], row['prop_reads'], row['seed'], _ = params
    if 'score' in result:
        row['score'] = result['score']
        row['pvalue'] = result['pvalue']
        row['null_mean'] = np.mean(result['null_scores'])
        row['null_std'] = np.std(result['null_scores'])
        row['n_cells_p1'], row['n_cells_p2'] = result['partition_adata'].obs.partition_size
    else:
        continue

    rows025.append(row)
    
finaldf025 = pd.DataFrame(rows025)
finaldf025['ll_zscore'] = np.abs((finaldf025.score - finaldf025.null_mean) / finaldf025.null_std)
finaldf025.loc[(finaldf025.score == 0) & (finaldf025.null_mean == 0) & (finaldf025.null_std == 0), 'pvalue'] = 1 # failure to run test
finaldf025['pvalue_zero'] = finaldf025.pvalue == 0

```

```python
p = 'SPECTRUM-OV-045'
sbm_adata_045 = ad.read_h5ad(os.path.join(sbm_adata_stem, f'sbmclone_{p}_snv.h5'))
```

```python
sbm_adata_045.obs.block_assignment.value_counts()
```

```python
partition_045 = (1, 1, 2, 1)
partition_045_2 = (1, 2, 0, 2)

paramset_final_045 = [('SPECTRUM-OV-045', sbm_adata_045, propc, 1, seed, partition_045) for propc in prop_cells_final for seed in range(n_repetitions)]
paramset_final_045_2 = [('SPECTRUM-OV-045', sbm_adata_045, propc, 1, seed, partition_045_2) for propc in prop_cells_final for seed in range(n_repetitions)]
len(paramset_final_045)
```

<!-- #raw -->
print(datetime.now())
with Pool(7) as pool:
    subsample_results_045f = pool.map(run_subsample_test_wrapper, paramset_final_045)

print(datetime.now())

<!-- #endraw -->

<!-- #raw -->
print(datetime.now())
subsample_results_045f = list(map(run_subsample_test_wrapper, paramset_final_045))

print(datetime.now())

<!-- #endraw -->

```python
rows045 = []
for params, result in zip(paramset_final_045, subsample_results_045f):
    row = {}
    _, _, row['prop_cells'], row['prop_reads'], row['seed'], _ = params
    if 'score' in result:
        row['score'] = result['score']
        row['pvalue'] = result['pvalue']
        row['null_mean'] = np.mean(result['null_scores'])
        row['null_std'] = np.std(result['null_scores'])
        row['n_cells_p1'], row['n_cells_p2'] = result['partition_adata'].obs.partition_size
    else:
        continue

    rows045.append(row)
    
finaldf045 = pd.DataFrame(rows045)
finaldf045['ll_zscore'] = np.abs((finaldf045.score - finaldf045.null_mean) / finaldf045.null_std)
finaldf045.loc[(finaldf045.score == 0) & (finaldf045.null_mean == 0) & (finaldf045.null_std == 0), 'pvalue'] = 1 # failure to run test
finaldf045['pvalue_zero'] = finaldf045.pvalue == 0

```

<!-- #raw -->
print(datetime.now())
with Pool(7) as pool:
    subsample_results_045_2f = pool.map(run_subsample_test_wrapper, paramset_final_045_2)

print(datetime.now())

<!-- #endraw -->

<!-- #raw -->
finaldf045.to_csv('rev_indepwgd_test_045_2vsrest.csv', index=False)
<!-- #endraw -->

```python
print(datetime.now())
subsample_results_045_2f = list(map(run_subsample_test_wrapper, paramset_final_045_2))
print(datetime.now())

```

```python
rows045_2 = []
for params, result in zip(paramset_final_045_2, subsample_results_045_2f):
    row = {}
    _, _, row['prop_cells'], row['prop_reads'], row['seed'], _ = params
    if 'score' in result:
        row['score'] = result['score']
        row['pvalue'] = result['pvalue']
        row['null_mean'] = np.mean(result['null_scores'])
        row['null_std'] = np.std(result['null_scores'])
        row['n_cells_p1'], row['n_cells_p2'] = result['partition_adata'].obs.partition_size
    else:
        continue

    rows045_2.append(row)
    
finaldf045_2 = pd.DataFrame(rows045_2)
finaldf045_2['ll_zscore'] = np.abs((finaldf045_2.score - finaldf045_2.null_mean) / finaldf045_2.null_std)
finaldf045_2.loc[(finaldf045_2.score == 0) & (finaldf045_2.null_mean == 0) & (finaldf045_2.null_std == 0), 'pvalue'] = 1 # failure to run test
finaldf045_2['pvalue_zero'] = finaldf045_2.pvalue == 0
finaldf045_2['n_cells'] = finaldf045_2.n_cells_p1 + finaldf045_2.n_cells_p2

```

<!-- #raw -->
finaldf045_2.to_csv('rev_indepwgd_test_045_0vs13.csv', index=False)
<!-- #endraw -->

```python
finaldf045_2 = pd.read_csv('rev_indepwgd_test_045_0vs13.csv')
```

```python
plt.figure(dpi=150, figsize=(15,5))
plt.subplot(1, 3, 1)
sns.boxplot(data=finaldf025, x='prop_cells', y='ll_zscore')
sns.stripplot(data=finaldf025, x='prop_cells', y='ll_zscore', s = 2)
plt.axhline(y=2.33, c='k', linestyle='--', label='z=2.33')
plt.yscale('log')
plt.xlabel("Proportion of cells")
plt.ylabel("z-score")
plt.ylim(0.05, 100)
plt.legend()
plt.title("OV-025")

plt.subplot(1, 3, 2)
sns.boxplot(data=finaldf045, x='prop_cells', y='ll_zscore')
sns.stripplot(data=finaldf045, x='prop_cells', y='ll_zscore', s = 2)
plt.axhline(y=2.33, c='k', linestyle='--', label='z=2.33')
plt.yscale('log')
plt.xlabel("Proportion of cells")
plt.ylabel("z-score")
plt.ylim(0.05, 100)
plt.legend()
plt.title("OV-045 clone2 vs rest")

plt.subplot(1, 3, 3)
sns.boxplot(data=finaldf045_2, x='prop_cells', y='ll_zscore')
sns.stripplot(data=finaldf045_2, x='prop_cells', y='ll_zscore', s = 2)
plt.axhline(y=2.33, c='k', linestyle='--', label='z=2.33')
plt.yscale('log')
plt.xlabel("Proportion of cells")
plt.ylabel("z-score")
plt.ylim(0.05, 100)
plt.legend()
plt.title("OV-045 clone0 vs. 1/3")
plt.tight_layout()
```

```python
plt.figure(dpi=150, figsize=(15,5))
plt.subplot(1, 3, 1)
sns.boxplot(data=finaldf025, x='prop_cells', y='pvalue')
sns.stripplot(data=finaldf025, x='prop_cells', y='pvalue', s = 2)
plt.xlabel("Proportion of cells")
plt.ylabel("pvalue")
plt.title("OV-025")

plt.subplot(1, 3, 2)
sns.boxplot(data=finaldf045, x='prop_cells', y='pvalue')
sns.stripplot(data=finaldf045, x='prop_cells', y='pvalue', s = 2)
plt.xlabel("Proportion of cells")
plt.ylabel("pvalue")
plt.title("OV-045 clone2 vs rest")

plt.subplot(1, 3, 3)
sns.boxplot(data=finaldf045_2, x='prop_cells', y='pvalue')
sns.stripplot(data=finaldf045_2, x='prop_cells', y='pvalue', s = 2)
plt.xlabel("Proportion of cells")
plt.ylabel("pvalue")
plt.title("OV-045 clone0 vs. 1/3")
plt.tight_layout()
```

```python
finaldf025['smaller_part_ncells'] = np.minimum(finaldf025.n_cells_p1, finaldf025.n_cells_p2)
finaldf045['smaller_part_ncells'] = np.minimum(finaldf045.n_cells_p1, finaldf045.n_cells_p2)
finaldf045_2['smaller_part_ncells'] = np.minimum(finaldf045_2.n_cells_p1, finaldf045_2.n_cells_p2)
```

```python
finaldf025['n_cells'] = finaldf025.n_cells_p1 + finaldf025.n_cells_p2
finaldf045['n_cells'] = finaldf045.n_cells_p1 + finaldf045.n_cells_p2
finaldf045_2['n_cells'] = finaldf045_2.n_cells_p1 + finaldf045_2.n_cells_p2

sns.lineplot(finaldf025, x='n_cells', y='ll_zscore', label = 'OV-025')
sns.lineplot(finaldf045, x='n_cells', y='ll_zscore', label = 'OV-045 clone2 vs 0/1/3')
sns.lineplot(finaldf045_2, x='n_cells', y='ll_zscore', label = 'OV-045 clone0 vs 1/3')
plt.axhline(y=2.33, c='k', linestyle='--', label='z=2.33')
plt.yscale('log')
```

```python
finaldf025
```

```python
plt.figure(dpi=150, figsize=(8,4))
plt.subplot(1, 2, 1)
finaldf025['n_cells'] = finaldf025.n_cells_p1 + finaldf025.n_cells_p2
finaldf045['n_cells'] = finaldf045.n_cells_p1 + finaldf045.n_cells_p2
finaldf045_2['n_cells'] = finaldf045_2.n_cells_p1 + finaldf045_2.n_cells_p2

sns.lineplot(finaldf025, marker='.', x='n_cells', y='ll_zscore', label = 'OV-025')
sns.lineplot(finaldf045, marker='.', x='n_cells', y='ll_zscore', label = 'OV-045 clone2 vs 0/1/3')
sns.lineplot(finaldf045_2, marker='.', x='n_cells', y='ll_zscore', label = 'OV-045 clone0 vs 1/3')
plt.axhline(y=2.33, c='k', linestyle='--', label='z=2.33')
plt.legend()
plt.yscale('log')
plt.xlabel("Total number of cells")
plt.ylabel("z-score")

plt.subplot(1, 2, 2)
sns.scatterplot(finaldf025, x='smaller_part_ncells', y='ll_zscore', label = 'OV-025')
sns.scatterplot(finaldf045, x='smaller_part_ncells', y='ll_zscore', label = 'OV-045 clone2 vs. 0/1/3')
sns.scatterplot(finaldf045_2, x='smaller_part_ncells', y='ll_zscore', label = 'OV-045 clone0 vs 1/3')
plt.axhline(y=2.33, c='k', linestyle='--', label='z=2.33')
plt.legend()
plt.yscale('log')
plt.xlabel("Number of cells in smaller partition")
plt.ylabel("z-score")
plt.tight_layout()
```

```python
finaldf025[finaldf025.ll_zscore < 2.33].smaller_part_ncells.max()
```

```python
finaldf045[finaldf045.ll_zscore < 2.33].smaller_part_ncells.max()
```

```python
finaldf045_2[finaldf045_2.ll_zscore < 2.33].smaller_part_ncells.max()
```

```python
finaldf045[finaldf045.ll_zscore < 2.33].n_cells.max()
```

```python
finaldf045_2[finaldf045_2.ll_zscore < 2.33].n_cells.max()
```

<!-- #raw -->
finaldf025.to_csv('rev_indepwgd_test_025.csv', index=False)
finaldf045.to_csv('rev_indepwgd_test_045_2vsrest.csv', index=False)
finaldf045_2.to_csv('rev_indepwgd_test_045_1vs03.csv', index=False)
<!-- #endraw -->

```python
sbm_adata_045.obs.sbmclone_cluster_id.value_counts()
```

# apply to 1-WGD patients to make sure we never get positive result

```python
# 036, 044, 052
sbm_adata_036 = ad.read_h5ad(os.path.join(sbm_adata_form.format('SPECTRUM-OV-036')))
sbm_adata_044 = ad.read_h5ad(os.path.join(sbm_adata_form.format('SPECTRUM-OV-044')))
sbm_adata_052 = ad.read_h5ad(os.path.join(sbm_adata_form.format('SPECTRUM-OV-052')))

# results from indep WGD test -- partition that maximizes score
partition_036 = (1, 1, 1, 2)
partition_044 = (1, 1, 1, 2)
partition_052 = (1, 2)

```

```python
paramset_036 = [('SPECTRUM-OV-036', sbm_adata_036, propc, 1, seed, partition_036) for propc in prop_cells_final for seed in range(n_repetitions)]
paramset_044 = [('SPECTRUM-OV-044', sbm_adata_044, propc, 1, seed, partition_044) for propc in prop_cells_final for seed in range(n_repetitions)]
paramset_052 = [('SPECTRUM-OV-052', sbm_adata_052, propc, 1, seed, partition_052) for propc in prop_cells_final for seed in range(n_repetitions)]

```

```python
print(datetime.now())
subsample_results_036f = list(map(run_subsample_test_wrapper, paramset_036))
print(datetime.now())

subsample_results_044f = list(map(run_subsample_test_wrapper, paramset_044))
print(datetime.now())

subsample_results_052f = list(map(run_subsample_test_wrapper, paramset_052))
print(datetime.now())
```

```python
def construct_result_df(paramset, results):
    rows = []
    for params, result in zip(paramset, results):
        row = {}
        _, _, row['prop_cells'], row['prop_reads'], row['seed'], _ = params
        if 'score' in result:
            row['score'] = result['score']
            row['pvalue'] = result['pvalue']
            row['null_mean'] = np.mean(result['null_scores'])
            row['null_std'] = np.std(result['null_scores'])
            row['n_cells_p1'], row['n_cells_p2'] = result['partition_adata'].obs.partition_size
        else:
            continue
    
        rows.append(row)
        
    df = pd.DataFrame(rows)
    df['ll_zscore'] = (df.score - df.null_mean) / df.null_std
    df.loc[(df.score == 0) & (df.null_mean == 0) & (df.null_std == 0), 'pvalue'] = 1 # failure to run test
    df['pvalue_zero'] = df.pvalue == 0
    return df
```

```python
finaldf036 = construct_result_df(paramset_036, subsample_results_036f)
finaldf044 = construct_result_df(paramset_044, subsample_results_044f)
finaldf052 = construct_result_df(paramset_052, subsample_results_052f)

```

```python
plt.figure(dpi=150, figsize=(15,5))
plt.subplot(1, 3, 1)
sns.boxplot(data=finaldf036, x='prop_cells', y='ll_zscore')
sns.stripplot(data=finaldf036, x='prop_cells', y='ll_zscore', s = 2)
plt.axhline(y=0, c='grey', label='z=0')
plt.xlabel("Proportion of cells")
plt.ylabel("z-score")
plt.legend()
plt.title("OV-036")

plt.subplot(1, 3, 2)
sns.boxplot(data=finaldf044, x='prop_cells', y='ll_zscore')
sns.stripplot(data=finaldf044, x='prop_cells', y='ll_zscore', s = 2)
plt.axhline(y=0, c='grey', label='z=0')
plt.xlabel("Proportion of cells")
plt.ylabel("z-score")
plt.legend()
plt.title("OV-044")

plt.subplot(1, 3, 3)
sns.boxplot(data=finaldf052, x='prop_cells', y='ll_zscore')
sns.stripplot(data=finaldf052, x='prop_cells', y='ll_zscore', s = 2)
plt.axhline(y=0, c='grey', label='z=0')
plt.xlabel("Proportion of cells")
plt.ylabel("z-score")
plt.legend()
plt.title("OV-052")
plt.suptitle("Downsampling cells for single-WGD patients")
plt.tight_layout()
```

```python
plt.figure(dpi=150, figsize=(15,5))
plt.subplot(1, 3, 1)
sns.boxplot(data=finaldf036, x='prop_cells', y='pvalue')
sns.stripplot(data=finaldf036, x='prop_cells', y='pvalue', s = 2)
plt.xlabel("Proportion of cells")
plt.ylabel("pvalue")
plt.title("OV-036")

plt.subplot(1, 3, 2)
sns.boxplot(data=finaldf044, x='prop_cells', y='pvalue')
sns.stripplot(data=finaldf044, x='prop_cells', y='pvalue', s = 2)
plt.xlabel("Proportion of cells")
plt.ylabel("pvalue")
plt.title("OV-044")

plt.subplot(1, 3, 3)
sns.boxplot(data=finaldf052, x='prop_cells', y='pvalue')
sns.stripplot(data=finaldf052, x='prop_cells', y='pvalue', s = 2)
plt.xlabel("Proportion of cells")
plt.ylabel("pvalue")
plt.title("OV-052")
plt.suptitle("Downsampling cells for single-WGD patients")
plt.tight_layout()
```

```python
sns.jointplot(finaldf052, x = 'pvalue', y = 'll_zscore', hue = 'prop_cells')
```

<!-- #raw -->
finaldf036.to_csv('rev_indepwgd_test_036.csv', index=False)
finaldf044.to_csv('rev_indepwgd_test_044.csv', index=False)
finaldf052.to_csv('rev_indepwgd_test_052.csv', index=False)
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
