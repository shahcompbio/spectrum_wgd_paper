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
import pickle
```

```python
pipeline_outputs = pipeline_dir # path to root directory of scWGS pipeline outputs
```

# load hmmcopy tables to get total sequenced cells

```python
hmmcopy_table = pd.read_csv('../pipelines/scdna/inputs/hmmcopy_table.csv')
aliquot2cells = {}
all_depth = []
all_breadth = []
for _, r in hmmcopy_table.iterrows():
    hmmcopy = pd.read_csv(r.MONDRIAN_HMMCOPY_metrics)
    aliquot2cells[r.isabl_aliquot_id] = len(hmmcopy)
    all_depth.extend(hmmcopy.coverage_depth.values)
    all_breadth.extend(hmmcopy.coverage_breadth.values)
```

```python
sample2cells = {s:sum([aliquot2cells[a] for a in sdf.isabl_aliquot_id]) for s, sdf in hmmcopy_table.groupby('isabl_sample_id')}
```

# generate sample-level DLP summary tables
* filtering counts
* sequencing metrics (all cells and filtered only)


```python
cell_info = pd.read_csv(os.path.join(pipeline_outputs, 'preprocessing/summary/filtered_cell_table.csv.gz'))
# reformat cell filtering fields to be boolean
cell_info['filter_passing'] = cell_info.include_cell
cell_info['is_doublet'] = cell_info.is_doublet != 'No'
cell_info['too_short_135_segment'] = cell_info.longest_135_segment < 20
# fix definition of aberrant normal to be specific rather than including all normal-classified cells
cell_info['is_aberrant_normal_cell'] = np.logical_xor(cell_info.is_aberrant_normal_cell, cell_info.is_normal)
orig_cell_info = cell_info.copy()


# apply cell filtering thresholds successively in the order described in the paper
cell_filtering_fields = ['is_normal', 'is_aberrant_normal_cell', 'is_s_phase_thresholds', 'is_doublet', 'too_short_135_segment']
cell_filtering_values = [cell_info[c].values for c in cell_filtering_fields]
cell_info['is_s_phase_thresholds'] = np.logical_and(cell_info.is_s_phase_thresholds, ~np.any(cell_filtering_values[:2], axis = 0))
cell_info['is_doublet'] = np.logical_and(cell_info.is_doublet, ~np.any(cell_filtering_values[:3], axis = 0))
cell_info['too_short_135_segment'] = np.logical_and(cell_info.too_short_135_segment, ~np.any(cell_filtering_values[:4], axis = 0))
```

```python
cell_info.filter_passing.sum()
```

```python
mean_fields = ['total_mapped_reads', 'coverage_depth', 'coverage_breadth', 'quality', 'ploidy', 'fraction_loh', 'breakpoints']
sum_fields = ['is_normal', 'is_aberrant_normal_cell', 'is_s_phase_thresholds', 'is_doublet', 'too_short_135_segment', 'filter_passing']

name_mapping = {'is_s_phase_thresholds':'sphase_cells',
                'is_normal':'normal_cells',
                'is_aberrant_normal_cell':'aberrant_normal_cells',
                'is_doublet':'doublets',
                'filter_passing':'filter_passing_cells',
                'multipolar':'divergent_cells',
                'too_short_135_segment':'too_short_135_segment_cells'
               }

# collect all cell counts
df = cell_info[['patient_id', 'sample_id']].drop_duplicates().reset_index(drop=True)
df['total_sequenced_cells'] = df.sample_id.map(sample2cells)
df = df.merge(cell_info.groupby('sample_id').size().reset_index().rename(columns={0:'high_quality_cells'}), how = 'left')
df['low_quality_cells'] = df.total_sequenced_cells - df.high_quality_cells 
for field in sum_fields:
    new_field = cell_info.groupby('sample_id')[field].sum().reset_index()
    df = df.merge(new_field, how = 'left')

# count the number of cells in each WGD state
wgds = cell_info[cell_info.include_cell][['sample_id', 'n_wgd']].value_counts().reset_index().pivot(index='sample_id', columns ='n_wgd')
wgds.columns = [f'{x}xWGD_cells' for x in wgds.columns.get_level_values(1)]
wgds = wgds.fillna(0).astype(int).reset_index()
df = df.merge(wgds, how = 'left')

# for divergent cells, only count those cells that passed filtering
df = df.merge(cell_info[cell_info.include_cell].groupby('sample_id').multipolar.sum().reset_index(), how = 'left')

# count the number of divergent cells in each WGD state
wgds_divergent = cell_info[cell_info.include_cell & cell_info.multipolar][['sample_id', 'n_wgd']].value_counts().reset_index().pivot(index='sample_id', columns ='n_wgd')
wgds_divergent.columns = [f'{x}xWGD_divergent_cells' for x in wgds_divergent.columns.get_level_values(1)]
wgds_divergent = wgds_divergent.fillna(0).astype(int).reset_index()
df = df.merge(wgds_divergent, how = 'left')

# average over sequencing statistics for highquality cells (those in cell_info table)
for field in mean_fields:
    new_field = cell_info.groupby('sample_id')[field].mean().reset_index()
    new_field.columns = [new_field.columns[0], 'highquality_mean_' + new_field.columns[1]]
    df = df.merge(new_field, how = 'left')
# average over sequencing statistics for 
for field in mean_fields:
    new_field = cell_info[cell_info.include_cell].groupby('sample_id')[field].mean().reset_index()
    new_field.columns = [new_field.columns[0], 'filtered_mean_' + new_field.columns[1]]
    df = df.merge(new_field, how = 'left')

df = df.drop(columns = ['high_quality_cells'])
df = df.rename(columns=name_mapping)

```

```python
df.filter_passing_cells.sum()
```

```python
cell_info.groupby('sample_id').filter_passing.sum().sum()
```

```python
assert len(orig_cell_info[orig_cell_info.include_cell]) == df.filter_passing_cells.sum()

assert np.array_equal(df.total_sequenced_cells, 
                      df.low_quality_cells + df.normal_cells + df.aberrant_normal_cells + df.sphase_cells
                      + df.doublets + df.too_short_135_segment_cells + df.filter_passing_cells)
```

```python
len(df)
```

```python
df.iloc[0]
```

```python
oc = orig_cell_info[orig_cell_info.sample_id == 'SPECTRUM-OV-002_S1_INFRACOLIC_OMENTUM']

for i, c in enumerate(cell_filtering_fields):
    print(c, np.logical_and(oc[c], ~oc[cell_filtering_fields[:i]].any(axis=1)).sum())
```

# write table

```python
df.to_csv('../../tables/dlp_sample_summary.csv')
```

# summary numbers

```python
df.total_sequenced_cells.sum()
```

```python
np.median(all_depth)
```

```python
np.median(all_breadth)
```

```python
df.groupby('patient_id').total_sequenced_cells.sum().median()
```

```python
df.total_sequenced_cells.mean()
```

```python
df.filter_passing_cells.sum()
```

```python
df.columns
```

```python
# count WGD states for filtered cells
patient_wgd_counts = df.groupby('patient_id').aggregate('sum')[['0xWGD_cells', '1xWGD_cells', '2xWGD_cells', '0xWGD_divergent_cells']]
patient_wgd_props = (patient_wgd_counts.T / np.sum(patient_wgd_counts, axis = 1).values).T

patient_wgd_counts.head()
```

```python
# number of patients with >1 WGD state
((patient_wgd_counts > 0).sum(axis=1) > 1).sum()
```

```python
# number of patients with the majority state representing over 85% of cells
np.max(patient_wgd_props, axis = 1) > 0.85
```

```python
patient_wgd_counts['nondivergent_0xwgd'] = (patient_wgd_counts['0xWGD_cells'] - patient_wgd_counts['0xWGD_divergent_cells']).astype(int)
```

```python
# WGD-high patients with extant 0xWGD cells
patient_wgd_counts[(patient_wgd_counts['0xWGD_cells'] > 0) & (patient_wgd_props['1xWGD_cells'] > 0.5)]
```

```python
patient_wgd_props[(patient_wgd_counts['0xWGD_cells'] > 0) & (patient_wgd_props['1xWGD_cells'] > 0.5)]
```

```python
patient_wgd_counts[(patient_wgd_counts['1xWGD_cells'] + patient_wgd_counts['2xWGD_cells'] == 0)]
```

```python
(patient_wgd_counts['1xWGD_cells'] + patient_wgd_counts['2xWGD_cells'])
```

## specific patients mentioned

```python
df[df.patient_id == 'SPECTRUM-OV-081'][['sample_id', '0xWGD_cells']]
```

```python
patient_wgd_counts.loc['SPECTRUM-OV-045']
```

```python
patient_wgd_counts.loc['SPECTRUM-OV-006']
```

```python
patient_wgd_counts.loc['SPECTRUM-OV-081']
```

## divergent cell counts


```python
divergent_counts = df.groupby('patient_id').aggregate('sum')[['divergent_cells', 'filter_passing_cells']]
(divergent_counts.divergent_cells > 0).sum()
```

```python
# divergent cell counts
(divergent_counts.divergent_cells / divergent_counts.filter_passing_cells).mean()
```

## count non-majority nWGD cells

```python
# sbmclone anndatas
sbmclone_dir = os.path.join(pipeline_outputs, 'sbmclone') 
sbmclone_adatas = {}
for p in tqdm.tqdm(df.patient_id.unique()):
    sbmclone_adatas[p] = ad.read_h5ad(os.path.join(sbmclone_dir, f'sbmclone_{p}_snv.h5'))

```

```python
for p in sbmclone_adatas.keys():
    adata = sbmclone_adatas[p]
    combined_table = adata.obs.merge(cell_info, how='left', left_index=True, right_on='cell_id')
    combined_table['n_wgd_mode'] = combined_table.groupby('sbmclone_cluster_id')['n_wgd'].transform(lambda x: x.mode()[0])
    my_cells = combined_table[(combined_table.n_wgd == 0) &( combined_table.n_wgd_mode > 0)]
    if len(my_cells) > 0:
        print(p, my_cells[['sbmclone_cluster_id', 'n_wgd']].value_counts())
```

# how many small clones are there?

```python
clonedfs = []
wgd_cols = ['0xWGD', '1xWGD', '2xWGD']
for p in sbmclone_adatas.keys():
    adata = sbmclone_adatas[p]
    clonedf = adata.obs.groupby('sbmclone_cluster_id').size().reset_index()
    clonedf['sbmclone_cluster_id'] = p + '_' + clonedf.sbmclone_cluster_id.astype(str)

    celldf = adata.obs.merge(cell_info[cell_info.include_cell], left_index=True, right_on='cell_id', how = 'inner')
    wgd_counts = celldf[['sbmclone_cluster_id', 'n_wgd']].value_counts().reset_index().pivot(index='sbmclone_cluster_id', columns='n_wgd').fillna(0).astype(int)
    wgd_counts.columns = [f'{i}xWGD' for c, i in wgd_counts.columns]
    for c in wgd_cols:
        if c not in wgd_counts.columns:
            wgd_counts[c] = 0
    wgd_counts = wgd_counts.reset_index()
    wgd_counts['sbmclone_cluster_id'] = p + '_' + wgd_counts.sbmclone_cluster_id.astype(str)
    clonedf = clonedf.merge(wgd_counts[['sbmclone_cluster_id'] + wgd_cols] , on = 'sbmclone_cluster_id')
    clonedf['patient_id'] = p

    clonedfs.append(clonedf)
clonedf = pd.concat(clonedfs)[['patient_id'] + list(clonedfs[0].columns[:-1])].rename(columns={0:'n_cells'}).reset_index(drop=True)
```

# quantify +1 WGD cells vs. SBMClone clones

```python
n_morewgd = []
clonedf['morewgd'] = 0
for _, r in clonedf.iterrows():
    majority = np.argmax(r.iloc[3:])
    n_morewgd.append(r.iloc[3 + majority + 1:].sum())
clonedf['morewgd'] = n_morewgd
```

## how many patients have any "morewgd" cells

```python
(clonedf.groupby('patient_id').morewgd.sum() > 0).sum(), len(clonedf.patient_id.unique())
```

## how many patients with >1 clone have "morewgd" cells in >1 clone

```python
multiclone_patients = []
multiclone_morewgd_patients = []
for p, pdf in clonedf.groupby('patient_id'):
    if len(pdf) > 1:
        multiclone_patients.append(p)
        if len(pdf[pdf.morewgd > 0]) > 1:
            multiclone_morewgd_patients.append(p)
len(multiclone_morewgd_patients), len(multiclone_patients)
```

```python
set(multiclone_patients) - set(multiclone_morewgd_patients)
```

```python
multiclone_morewgd_patients
```

# look at +1 WGD vs. sample

```python
sample_nwgd = cell_info[cell_info.include_cell][['patient_id', 'sample_id', 'n_wgd']].value_counts().reset_index().pivot(index=['patient_id', 'sample_id'], columns='n_wgd').fillna(0).astype(int).reset_index()
sample_nwgd.columns = [f'{a[1]}xWGD' if a[1] != '' else a[0] for a in sample_nwgd.columns.values]

n_morewgd = []
sample_nwgd['morewgd'] = 0
for _, r in sample_nwgd.iterrows():
    majority = np.argmax(r.iloc[2:])
    n_morewgd.append(r.iloc[2 + majority + 1:].sum())
sample_nwgd['morewgd'] = n_morewgd
sample_nwgd
```

```python
patients_multisite = []
patients_multisite_extrawgd = []
for p, pdf in sample_nwgd.groupby('patient_id'):
    if len(pdf) > 1:
        patients_multisite.append(p)
        if (pdf.morewgd > 0).sum() > 1:
            patients_multisite_extrawgd.append(p)
len(patients_multisite_extrawgd), len(patients_multisite)
```

# check 2xWGD cells in 025
count the number of SNVs exclusive to cluster 2

```python
adata = sbmclone_adatas['SPECTRUM-OV-025']
combined_table = adata.obs.merge(cell_info, how='left', left_index=True, right_on='cell_id')
combined_table['n_wgd_mode'] = combined_table.groupby('sbmclone_cluster_id')['n_wgd'].transform(lambda x: x.mode()[0])
```

```python
combined_table[['sbmclone_cluster_id', 'n_wgd']].value_counts()
```

```python
adata
```

```python
clone_adata = scgenome.tl.aggregate_clusters(adata, cluster_col='block_assignment', agg_layers={'alt':'sum', 'ref':'sum', 'state':'median', 'total':'sum'})
```

```python
clone_adata.obs
```

```python
exclusive = np.where(np.logical_and(np.all(clone_adata[[0, 1, 3]].layers['alt'] == 0, axis = 0), clone_adata[2].layers['alt'] > 0))[1]
len(exclusive)
```

## try looking only at the 2xWGD cells

```python
non2x_cells = combined_table[(combined_table.block_assignment == 2) & (combined_table.n_wgd != 2)].cell_id.values
```

```python
adata.obs['block_assignment2'] = adata.obs.block_assignment
adata.obs.loc[non2x_cells, 'block_assignment2'] = 3
clone_adata2 = scgenome.tl.aggregate_clusters(adata, cluster_col='block_assignment2', agg_layers={'alt':'sum', 'ref':'sum', 'state':'median', 'total':'sum'})
```

```python
exclusive = np.where(np.logical_and(np.all(clone_adata2[[0, 1, 3]].layers['alt'] == 0, axis = 0), clone_adata2[2].layers['alt'] > 0))[1]
len(exclusive)
```

# wgd evolution modes

```python
truncal = ['003', '129', '118', '049', '068', '052', '044', '071', '036', '002', '105', '133', '110', '051', '082', '014', '083', '008', '065', '107', '087']
parallel = ['025', '045']
subclonal = ['139', '075', '031', '081', '006']
unexpanded = ['037', '050', '080', '022', '046', '115', '009', '007', '026', '070', '004']
omitted = ['125', '024']
len(truncal), len(parallel), len(subclonal), len(unexpanded)
```

```python
patient2evo = {}
for p in truncal:
    patient2evo['SPECTRUM-OV-' + p] = 'Truncal WGD'
for p in parallel:
    patient2evo['SPECTRUM-OV-' + p] = 'Parallel WGD'
for p in subclonal:
    patient2evo['SPECTRUM-OV-' + p] = 'Subclonal WGD'
for p in unexpanded:
    patient2evo['SPECTRUM-OV-' + p] = 'Unexpanded WGD'
for p in omitted:
    patient2evo['SPECTRUM-OV-' + p] = 'omitted'

patient2evo = pd.DataFrame(sorted(patient2evo.items()), columns=['patient_id', 'evolution_category'])
patient2evo.to_csv('../../tables/patient_evolution_category.csv', index=False)
```

```python
patient2evo.set_index('patient_id').evolution_category
```

```python

```
