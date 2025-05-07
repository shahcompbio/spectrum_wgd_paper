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
import anndata as ad
import numpy as np
import tqdm
import anndata as ad
import hashlib
from collections import Counter, defaultdict
import pickle
os.environ['ISABL_API_URL'] = 'https://isabl.shahlab.mskcc.org/api/v1'
import isabl_cli as ii
import json
import Bio.Phylo
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
import matplotlib.colors as mcolors
from functools import lru_cache
import scipy
import pysam
import vetica.mpl
import met_brewer
import matplotlib
matplotlib.rcParams['svg.fonttype'] = 'none'
```

```python
cohort = pd.read_csv('repos/downsampling_smk/quick_sample_table_both.csv')

orig_articull_stem = 'users/satasg/archive/2023/snv_filter/results/SPECTRUM/0817_updatedmodel_fullcohort/'
output_stem = 'users/myersm2/spectrum-snv-detection-downsampling'
cell_list_stem = '/juno/work/shah/users/myersm2/misseg/downsampling/quick_samples'

pipeline_outputs_dir = 'users/myersm2/spectrum-dlp-pipeline/v5.2/'
tree_snv_inputs_dir = pipeline_outputs_dir + '/tree_snv/inputs/'
```

```python
cell_info = pd.read_csv(os.path.join(pipeline_outputs_dir, 'preprocessing', 'summary', 'cell_table.csv.gz'))
```

# retrieve cell lists

```python
with open('../revisions/downsampling-cell-lists2.pickle', 'rb') as f:
    cell_lists2 = pickle.load(f)
```

# check for output files

```python
done_keys = set()
for _, r in cohort.iterrows():
    if os.path.exists(os.path.join(output_stem, r.hash, 'articull', 'result.tsv')):
        done_keys.add(r.hash)
len(cohort), len(done_keys)
```

# analyze comparing to SNVs in clonal SBMClone block

```python
articull_dtypes = {'chrm':str, 'pos':int, 'ref_allele':str, 'alt_ellele':str, 'result':str, 'prob_artifact':float}

orig_articull = {}
for p in cohort.patient.unique():
    orig_articull[p] = pd.read_table(os.path.join(orig_articull_stem, p[9:], 'result.tsv'), dtype = articull_dtypes)
```

```python
snv_adatas = {}
for p in tqdm.tqdm(cohort.patient.unique()):
    snv_adatas[p] = ad.read_h5ad(os.path.join(tree_snv_inputs_dir, f'{p}_general_clone_adata.h5'))
```

```python
bulk_vafs = np.sum(snv_adatas[p].layers['alt'], axis = 0) / np.maximum(1, np.sum(snv_adatas[p].layers['total_count'], axis = 0))
total_cov = np.sum(snv_adatas[p].layers['total_count'], axis = 0)
```

```python
def get_clonal_snvs(p, prop_cells_clonal = 0.9, interval_width = 0.1, min_snv_coverage=50):
    snv_adata = snv_adatas[p]
    modes, modecounts = scipy.stats.mode(snv_adata.layers['state'], axis = 0, keepdims = False)
    bulk_vafs = (np.sum(snv_adata.layers['alt'], axis = 0) / 
                 np.maximum(1, np.sum(snv_adata.layers['total_count'], axis = 0)))
    total_cov = np.sum(snv_adata.layers['total_count'], axis = 0)
    
    table = snv_adata.var.copy()
    table['modal_cn'] = modes
    table['modal_cn_freq'] = modecounts / snv_adata.shape[0]
    table['bulk_vaf'] = bulk_vafs
    table['total_cov'] = total_cov
    table['copy1_clonal'] = ((table.modal_cn == 1) & 
                             (table.modal_cn_freq > prop_cells_clonal) &
                             (table.total_cov > min_snv_coverage) &
                             (table.bulk_vaf > 1-interval_width))
    table['copy2_clonal'] = ((table.modal_cn == 2) & 
                             (table.modal_cn_freq > prop_cells_clonal) &
                             (table.total_cov > min_snv_coverage) &
                             (table.bulk_vaf < 0.5 + interval_width/2) & 
                            (table.bulk_vaf > 0.5 - interval_width/2))
    table['copy3_clonal'] = ((table.modal_cn == 3) & 
                             (table.modal_cn_freq > prop_cells_clonal) &
                             (table.total_cov > min_snv_coverage) &
                             (table.bulk_vaf < 1/3 + interval_width/2) & 
                            (table.bulk_vaf > 1/3 - interval_width/2))
    table['is_clonal'] = (table.copy1_clonal | table.copy2_clonal | table.copy3_clonal)
    
    # TODO: threshold on number of cells containing SNV?
    table['patient'] = p
    return table
```

```python
clonal_snvs =  get_clonal_snvs(p)
```

```python
clonal_snvs[clonal_snvs.is_clonal].modal_cn.mean()
```

```python
results = defaultdict(lambda:[])
joint_tables = {}
all_clonal_snvs = {}

outer_articull = {}

for _, r in tqdm.tqdm(cohort.iterrows()):
    if r.hash in done_keys:
        if r.hash == '63ce5f751e7d14378cc097f04a02bbffa2f0f87a':
            continue
        
        p = r.patient
        my_snvs = snv_adatas[p].var
        my_orig_articull = orig_articull[p]
        sample_result = pd.read_table(os.path.join(output_stem, r.hash, 'articull', 'result.tsv'), dtype = articull_dtypes)
        joint_articull = my_orig_articull.merge(sample_result, on = ['chrm', 'pos', 'ref_allele', 'alt_allele'],
                       suffixes = ['_orig', '_sample'])
        joint_articull['result_pair'] = joint_articull.result_orig + '_' + joint_articull.result_sample
        outer_table = my_orig_articull.merge(sample_result, on = ['chrm', 'pos', 'ref_allele', 'alt_allele'],
                       suffixes = ['_orig', '_sample'], how = 'outer')
        n_na_pass = len(outer_table[outer_table.result_orig.isna() & (outer_table.result_sample == 'PASS')])
        n_artifact_pass = len(outer_table[(outer_table.result_orig == 'ARTIFACT') & (outer_table.result_sample == 'PASS')])
        outer_articull[p, r.hash] = outer_table

        #clonal_blocks = np.where(np.all(Bs[p] == 1, axis = 0))[0]
        #clonal_snvs = my_snvs[my_snvs.block_assignment.isin(clonal_blocks)]
        if p not in all_clonal_snvs:
            all_clonal_snvs[p] =  get_clonal_snvs(p)

        clonal_snvs = all_clonal_snvs[p]
        clonal_snvs['snv_id'] = clonal_snvs.chromosome.astype(str) + '-' + clonal_snvs.position.astype(str) + ':' +  clonal_snvs.ref.astype(str) + '>'  + clonal_snvs.alt.astype(str)
        clonal_snvs = clonal_snvs[clonal_snvs.is_clonal]
            
        joint_table = my_snvs.merge(joint_articull, left_on = ['chromosome', 'position', 'ref', 'alt'],
              right_on = ['chrm', 'pos', 'ref_allele', 'alt_allele'])
        joint_table['snv_id'] = joint_table.chromosome + '-' + joint_table.position.astype(str) + ':' +  joint_table.ref + '>'  + joint_table.alt
        joint_table['is_clonal'] = joint_table.snv_id.isin(clonal_snvs.snv_id)
        
        results['patient'].append(p)
        results['hash'].append(r.hash)
        results['sample_size'].append(r.sample_size)
        results['orig_mutect_calls'].append(len(my_orig_articull))
        results['orig_articull_pass'].append(my_orig_articull.result.value_counts()['PASS'])
        results['sbmclone_snvs'].append(len(my_snvs))
        results['clonal_snvs'].append(len(clonal_snvs))
        results['sample_mutect_calls'].append(len(sample_result))
        results['sample_articull_pass'].append(sample_result.result.value_counts()['PASS'])
        results['sample_articull_pass_clonal'].append(len(
            joint_table[(joint_table.result_sample == 'PASS') & joint_table.is_clonal]))
        results['n_na_pass'].append(n_na_pass)
        results['n_artifact_pass'].append(n_artifact_pass)
        
        sample_cells = cell_lists2[r.hash]
        my_cells = cell_info.set_index('cell_id').loc[sample_cells]
        
        my_snv_adata = snv_adatas[p]

        joint_table['snv_id'] = joint_table.chromosome.astype(str) + '-' + joint_table.position.astype(str) + ':' + joint_table.alt_allele.astype(str) + '>' + joint_table.ref_allele.astype(str)
        my_snv_adata.var['snv_id'] = my_snv_adata.var.chromosome.astype(str) + '-' + my_snv_adata.var.position.astype(str) + ':' + my_snv_adata.var.alt.astype(str) + '>' + my_snv_adata.var.ref.astype(str)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            hapcov = (my_cells.coverage_depth / my_cells.ploidy).sum()

        results['haploid_coverage'].append(hapcov)
        
        joint_tables[r.hash] = joint_table
df = pd.DataFrame(results)
df['prop_recovered'] = df.sample_articull_pass_clonal / df.clonal_snvs
```

<!-- #raw -->
df.to_csv('clonal_snv_results.csv', index = False)
<!-- #endraw -->

```python
for p, acl in all_clonal_snvs.items():
    acl = acl[acl.is_clonal]
    print(p, acl.modal_cn.mean())
```

# fit logistic curve

```python
def logistic_curve(x, k=0.62355594, x0=4.1070332):
    return 0.92 / (1 + np.exp(-1 * k * (x - x0)))
```

```python
plt.figure(figsize=(2.5,2), dpi = 150)
xs = np.linspace(0, 60, 1000)
sns.scatterplot(data = df, x = 'haploid_coverage', y = 'prop_recovered', hue = 'patient', s=10,
                palette = 'Set2', edgecolor='k', linewidth=0.2)
sns.move_legend(
    plt.gca(), 'lower right',
    bbox_to_anchor=(1, 0), ncol=1, frameon=False,
    title='Patient',
    markerscale=2,
    prop={'size': 8}, title_fontsize=10,
    labelspacing=0.4, handletextpad=0, columnspacing=0.5)
#plt.plot(xs, logistic_curve(xs, *params))
plt.plot(xs, logistic_curve(xs), c = 'k')
plt.xlabel("Sample haploid coverage")
plt.ylabel("Prop. clonal SNVs recovered")
plt.savefig('../../figures/final/sensitivity_curve.svg', dpi = 150)
```

```python
plt.figure(figsize=(5, 4), dpi = 150)
xs = np.linspace(0, 60, 1000)
sns.scatterplot(data = df, x = 'haploid_coverage', y = 'prop_recovered', hue = 'patient', s=20,
                palette = 'Set2', edgecolor='k', linewidth=0.5)
sns.move_legend(
    plt.gca(), 'lower right',
    bbox_to_anchor=(1, 0), ncol=1, frameon=False,
    title='Patient',
    markerscale=1.5,
    prop={'size': 8}, title_fontsize=10,
    labelspacing=0.4, handletextpad=0, columnspacing=0.5)
#plt.plot(xs, logistic_curve(xs, *params))
plt.plot(xs, logistic_curve(xs), c = 'k')
plt.xlabel("Sample haploid coverage")
plt.ylabel("Prop. clonal SNVs recovered")
```
