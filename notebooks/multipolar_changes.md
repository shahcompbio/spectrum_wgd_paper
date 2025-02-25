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
    display_name: Python Spectrum
    language: python
    name: python_spectrum
---

```python

import os
import glob
import anndata as ad
import pandas as pd
import numpy as np
import Bio
from tqdm import tqdm
import yaml

import vetica.mpl
import matplotlib.pyplot as plt
import seaborn as sns

import scgenome

import spectrumanalysis.wgd
import spectrumanalysis.phylocn
import spectrumanalysis.utils
import spectrumanalysis.dataload
import spectrumanalysis.cnevents

colors_dict = yaml.safe_load(open('../../../../config/colors.yaml', 'r'))
project_dir = os.environ['SPECTRUM_PROJECT_DIR']

```

```python

cell_info = pd.read_csv(f'{project_dir}/preprocessing/summary/filtered_cell_table.csv.gz')

# REVISIT: should we be calculating nnd across all cells including those that are not included?
cell_info = cell_info[(cell_info['include_cell'] == True)]

```

```python

def compute_multipolar_events(adata, cluster_label='sbmclone_cluster_id'):
    spectrumanalysis.cnevents.annotate_bins(adata)
    
    adata_clusters = spectrumanalysis.utils.aggregate_cna_adata(adata, cluster_label)
    
    # Pre post WGD changes that minimize total changes
    n_states = int(adata.layers['state'].max() + 1)
    pre_post_changes = spectrumanalysis.phylocn.calculate_pre_post_changes(n_states)
    
    layers = ['state']
    
    events = []
    
    for cell_id, cluster_id in adata[adata.obs['multipolar']].obs[cluster_label].items():
        cell_n_wgd = adata.obs.loc[cell_id, 'n_wgd']
        cluster_n_wgd = adata_clusters.obs.loc[cluster_id, 'n_wgd']
        
        cell_data = scgenome.tl.get_obs_data(adata, cell_id)
        cluster_data = scgenome.tl.get_obs_data(adata_clusters, cluster_id)
        cell_data = cell_data.merge(cluster_data[['chr', 'start', 'end'] + layers], on=['chr', 'start', 'end'], suffixes=('', '_cluster'))
    
        for layer in layers:
            if cell_n_wgd == cluster_n_wgd:
                cell_data['cn_change'] = cell_data[layer] - cell_data[layer+'_cluster']
                cell_data['cn_change'] = cell_data['cn_change'].astype(int)

                for event in spectrumanalysis.cnevents.classify_segments(cell_data):
                    event['cell_id'] = cell_id
                    event['cluster_id'] = cluster_id
                    event['timing_wgd'] = 'none'
                    events.append(event)
    
            elif cell_n_wgd == cluster_n_wgd + 1:
                cell_cn = cell_data[layer].values
                cluster_cn = cell_data[layer+'_cluster'].values
                
                # HACK
                assert np.mean(cluster_cn == 0) < 0.03, f"bins with cn=0 is {np.mean(cluster_cn == 0)}"
                cluster_cn[cluster_cn == 0] = 1
                
                for wgd_timing in ['pre', 'post']:
                    cell_data['cn_change'] = pre_post_changes.loc[pd.IndexSlice[zip(cluster_cn, cell_cn)], wgd_timing].values
                    cell_data['cn_change'] = cell_data['cn_change'].astype(int)

                    for event in spectrumanalysis.cnevents.classify_segments(cell_data):
                        event['cell_id'] = cell_id
                        event['cluster_id'] = cluster_id
                        event['timing_wgd'] = wgd_timing
                        events.append(event)
    
            else:
                print(f'filtering cell {cell_id}')
                
    events = pd.DataFrame(events)

    return events

```

```python

events = []
for patient_id in cell_info['patient_id'].unique():
    print(patient_id)
    adata = spectrumanalysis.dataload.load_filtered_cna_adata(project_dir, patient_id)
    events.append(compute_multipolar_events(adata).assign(patient_id=patient_id))
events = pd.concat(events)

```

```python

events['region'] = events['region'].map(lambda a: {'q-arm': 'arm', 'p-arm': 'arm'}.get(a, a))
event_counts = events.groupby(['patient_id', 'cell_id', 'cluster_id', 'region', 'kind', 'timing_wgd']).size().rename('count').reset_index()
event_counts = event_counts[event_counts['timing_wgd'] == 'post']
event_counts = event_counts[event_counts['region'] != 'segment']
event_counts

```

```python

event_counts.groupby(['cell_id', 'region', 'kind'])['count'].sum().unstack(level=[1, 2], fill_value=0).describe()

```

```python

event_counts.groupby(['cell_id'])['count'].sum().describe()

```

```python

sns.boxplot(data=event_counts, x='region', hue='kind', y='count')

```

```python

event_counts = (
    events[events['timing_wgd'] == 'pre']
        .groupby(['patient_id', 'cell_id', 'cluster_id', 'region', 'kind'])
        .size().rename('count').unstack(level=(3, 4), fill_value=0))

event_counts.reset_index().head(20)

```

```python

patient_id, cell_id, cluster_id = event_counts.iloc[18].name

adata = spectrumanalysis.dataload.load_filtered_cna_adata(project_dir, patient_id)
cluster_label='sbmclone_cluster_id'
adata_clusters = spectrumanalysis.utils.aggregate_cna_adata(adata, cluster_label)

chromosome = None#'2'

plt.figure(figsize=(14, 2))
scgenome.pl.plot_cn_profile(adata, obs_id=cell_id, value_layer_name='copy', state_layer_name='state', squashy=True, chromosome=chromosome)
plt.ylabel('Copy Number')
sns.despine()

plt.figure(figsize=(14, 2))
scgenome.pl.plot_cn_profile(adata_clusters, obs_id=cluster_id, value_layer_name='copy', state_layer_name='state', squashy=True, chromosome=chromosome)
plt.ylabel('Copy Number')
sns.despine()

adata.var['diff'] = np.array(adata[cell_id, :].layers['state'][0] - adata_clusters[cluster_id, :].layers['state'][0])
plt.figure(figsize=(14, 2))
scgenome.pl.plot_profile(adata.var, 'diff', chromosome=chromosome)
sns.despine()

```

```python

```
