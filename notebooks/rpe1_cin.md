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
import scgenome
import matplotlib.pyplot as plt
import pickle
import yaml
```

```python
colors_yaml = yaml.safe_load(open('/data1/shahs3/users/myersm2/repos/spectrumanalysis/config/colors.yaml', 'r').read())

sample_ids = [
    'RPE-1-p53-nocadazole',
    'RPE-1-p53-DMSO',
    'RPE-1-p53-reversine',
    'RPE-1_p30_control', # mixed WGD/non-WGD
]

hmmcopy_metrics = {
    'RPE-1-p53-nocadazole': '/data1/shahs3/isabl_data_lake/analyses/36/08/23608/results/128680A_metrics.csv.gz',
    'RPE-1-p53-DMSO': '/data1/shahs3/isabl_data_lake/analyses/39/27/23927/results/128676A_metrics.csv.gz',
    'RPE-1-p53-reversine': '/data1/shahs3/isabl_data_lake/analyses/39/28/23928/results/128676A_metrics.csv.gz',
    'RPE-1_p30_control': '/data1/shahs3/isabl_data_lake/analyses/10/45/41045/results/SHAH_H002194_T44_01_DLP01_hmmcopy_metrics.csv.gz',
}

hmmcopy_reads = {
    'RPE-1-p53-nocadazole': '/data1/shahs3/isabl_data_lake/analyses/36/04/23604/results/128680A_reads.csv.gz',
    'RPE-1-p53-DMSO': '/data1/shahs3/isabl_data_lake/analyses/39/21/23921/results/128676A_reads.csv.gz',
    'RPE-1-p53-reversine': '/data1/shahs3/isabl_data_lake/analyses/39/22/23922/results/128676A_reads.csv.gz',
    'RPE-1_p30_control': '/data1/shahs3/isabl_data_lake/analyses/10/45/41045/results/SHAH_H002194_T44_01_DLP01_hmmcopy_reads.csv.gz',
}

metrics = []
for sample_id in sample_ids:
    metrics_filename = hmmcopy_metrics[sample_id]
    metrics.append(pd.read_csv(metrics_filename, low_memory=False))
metrics = pd.concat(metrics)

```

```python

adata = []
for sample_id in sample_ids:
    adata.append(scgenome.pp.read_dlp_hmmcopy(hmmcopy_reads[sample_id], hmmcopy_metrics[sample_id]))
adata = scgenome.tl.ad_concat_cells(adata)
adata = adata[:, adata.var['chr'] != 'Y'].copy()

adata = adata[
    (~adata.obs['is_control']) &
    (adata.obs['quality'] > 0.75) &
    (~adata.obs['is_s_phase'])
]

```

```python

adata = scgenome.tl.sort_cells(adata, layer_name=['copy'])

adata = scgenome.tl.cluster_cells(
    adata,
    min_k=5,
    max_k=8,
    layer_name=['copy'],
    method='gmm_diag_bic',
)

fig = plt.figure(figsize=(5, 3), dpi=300)
g = scgenome.pl.plot_cell_cn_matrix_fig(
    adata[:, (adata.var['chr'] != 'Y') & (adata.var['gc'] > 0)],
    layer_name='state',
    fig=fig,
    style='white',
    cell_order_fields=['cluster_id', 'sample_id', 'cell_order'],
    annotation_fields=['sample_id', 'cluster_id', 'is_s_phase'],
)

```

```python
fig = plt.figure(figsize=(5, 3), dpi=300)
g = scgenome.pl.plot_cell_cn_matrix_fig(
    adata[:, (adata.var['chr'] != 'Y') & (adata.var['gc'] > 0)],
    layer_name='state',
    fig=fig,
    style='white',
    cell_order_fields=['sample_id', 'cell_order'],
    annotation_fields=['sample_id', 'cluster_id', 'is_s_phase'],
)

```

```python

# Select a cluster to be the low-cin cluster as reference
wgd_cluster = '1'
low_cin_cluster = '2'

```

```python

import numpy as np

agg_X = np.sum

agg_layers = {
    'copy': np.nanmean,
    'state': np.nanmedian,
}

agg_obs = {
    'is_s_phase': np.nanmean,
}

adata_clusters = scgenome.tl.aggregate_clusters(adata, agg_X, agg_layers, agg_obs, cluster_col='cluster_id')
adata_clusters.layers['state'] = adata_clusters.layers['state'].round()

```

```python

import spectrumanalysis.plots

adata_clusters = scgenome.tl.sort_cells(adata_clusters, layer_name=['copy'])

fig = plt.figure(figsize=(5, 0.4), dpi=300)
g = scgenome.pl.plot_cell_cn_matrix_fig(
    adata_clusters[[wgd_cluster, low_cin_cluster], (adata.var['chr'] != 'Y') & (adata.var['gc'] > 0)],
    layer_name='state',
    fig=fig,
    style='white',
    show_cell_ids=True,
)

g['heatmap_ax'].set_yticklabels(['WGD', 'non-WGD'], fontsize=6)

spectrumanalysis.plots.remove_xticklabels(g['heatmap_ax'], ['14', '16', '18', '20', '22'])
sns.move_legend(
    g['legend_info']['ax_legend'], 'upper left',
    bbox_to_anchor=(-0.1, -1), ncol=3, frameon=False,
    prop={'size': 8})

fig.savefig(f'../../figures/final/model/rpe_wgd_clone.svg', bbox_inches='tight', metadata={'Date': None})

```

<!-- #raw -->

import spectrumanalysis.cnevents
import spectrumanalysis.phylocn
import tqdm

def compute_events_from_baseline(adata, adata_clusters, baseline_cluster_id):
    spectrumanalysis.cnevents.annotate_bins(adata)

    adata.obs['treatment'] = pd.Series(adata.obs.index).str.split('-', expand=True)[0].values
    adata.obs.loc[adata.obs.treatment == '130018A', 'treatment'] = 'control'
    print(adata.obs.treatment.unique())
    
    # Pre post WGD changes that minimize total changes
    n_states = int(adata.layers['state'].max() + 1)
    pre_post_changes = spectrumanalysis.phylocn.calculate_pre_post_changes(n_states)
    
    events = []
    counts = {}
    for cell_id in tqdm.tqdm(adata.obs.index):
        cell_data = scgenome.tl.get_obs_data(adata, cell_id)
        cluster_data = scgenome.tl.get_obs_data(adata_clusters, baseline_cluster_id)
        cell_data = cell_data.merge(cluster_data[['chr', 'start', 'end', 'state']], on=['chr', 'start', 'end'], suffixes=('', '_cluster'))

        ## Calculate events for both a wgd and non-wgd scenario
        # calculate nwgd events
        nwgd_events = []
        cell_data['cn_change'] = cell_data['state'] - cell_data['state_cluster']
        cell_data['cn_change'] = cell_data['cn_change'].astype(int)

        for event in spectrumanalysis.cnevents.classify_segments(cell_data):
            event['cell_id'] = cell_id
            event['timing_wgd'] = 'none'
            event['is_wgd'] = False
            nwgd_events.append(event)

        # calculate wgd events
        wgd_events = []
        cell_cn = cell_data['state'].values
        cluster_cn = cell_data['state_cluster'].values
        
        # workaround: set 0 copy bins in the baseline cluster to copy 1 to avoid gain from 0
        assert np.mean(cluster_cn == 0) < 0.03, f"bins with cn=0 is {np.mean(cluster_cn == 0)}"
        cluster_cn[cluster_cn == 0] = 1
        
        for wgd_timing in ['pre', 'post']:
            cell_data['cn_change'] = pre_post_changes.loc[pd.IndexSlice[zip(cluster_cn, cell_cn)], wgd_timing].values
            cell_data['cn_change'] = cell_data['cn_change'].astype(int)

            for event in spectrumanalysis.cnevents.classify_segments(cell_data):
                event['cell_id'] = cell_id
                event['timing_wgd'] = wgd_timing
                event['is_wgd'] = True
                wgd_events.append(event)


        condition = adata.obs.loc[cell_id, 'treatment']
        
        # keep the event set with fewer overall events
        if len(wgd_events) < len(nwgd_events):
            if (condition, 'wgd') not in counts:
                counts[condition, 'wgd'] = 0
            counts[condition, 'wgd'] += 1
                
            events.extend(wgd_events)
        else:
            if (condition, 'nwgd') not in counts:
                counts[condition, 'nwgd'] = 0
            counts[condition, 'nwgd'] += 1
            
            events.extend(nwgd_events)
    
    events = pd.DataFrame(events)
    
    return events, counts

events, cellcounts = compute_events_from_baseline(adata[:, adata.var['gc'] > 0], adata_clusters[:, adata.var['gc'] > 0], low_cin_cluster)

<!-- #endraw -->

<!-- #raw -->
events.to_csv('rpe1_events.csv.gz', compression={'method':'gzip', 'mtime':0, 'compresslevel':9})
<!-- #endraw -->

<!-- #raw -->
with open('rpe1_cellcounts.pickle', 'wb') as f:
    pickle.dump(cellcounts, f)
<!-- #endraw -->

```python
counts = events[['condition', 'region', 'is_wgd', 'kind']].value_counts().reset_index()
```

```python
cellcounts
```

```python
events['condition'] = events.cell_id.str.split('-', expand=True)[0]
events.loc[events.condition == '130018A', 'condition'] = 'control'
events['region2'] = events.region.map({'chromosome':'chromosome', 'q-arm':'arm', 'p-arm':'arm', 'segment':'segment'})
event_counts = events[events.region != 'segment'][['condition', 'region2', 'kind', 'is_wgd']].value_counts().reset_index()
ncells = []
for _, r in event_counts.iterrows():
    ncells.append(cellcounts[r.condition, 'wgd' if r.is_wgd else 'nwgd'])
event_counts['n_cells'] = ncells
event_counts['rate'] = event_counts['count'] / event_counts.n_cells
event_counts['event_type'] = event_counts.region2 + '_' + event_counts.kind
event_counts['sample_id'] = event_counts.condition
```

```python
adata.obs['ploidy'] = np.nanmean(adata.layers['state'], axis = 1)
adata.obs['wgd_by_ploidy'] = adata.obs.ploidy > 2.5
adata.obs['wgd_by_events'] = adata.obs.index.isin(events[events.is_wgd].cell_id.unique())
adata.obs[['wgd_by_ploidy', 'wgd_by_events']].value_counts()
```

# plot formatted like Matthew's

```python
event_counts['WGD'] = event_counts.is_wgd.map({True:'WGD clone', False:'nWGD'})
plot_df = event_counts.copy()
plot_df['sample_id'] = plot_df.sample_id.map({'RPE_D':'RPE1-D', 'control':'RPE1-WGD', 'RPE_Noco':'RPE1-Noco', 'RPE_rev':'RPE1-Rev'})

event_types = ['chromosome_loss', 'chromosome_gain', 'arm_loss', 'arm_gain', ]
event_type_names = ['Chrom. loss', 'Chrom. gain', 'Arm loss', 'Arm gain']

import seaborn as sns

order = ['RPE1-D', 'RPE1-Noco', 'RPE1-Rev', 'RPE1-WGD']

g = sns.catplot(
    x='sample_id', row='event_type', y='rate', kind='bar', row_order=event_types + ['empty'], 
    data=plot_df, hue='WGD', sharey=False, aspect=1.8, height=1.5, order=order, dodge=True, hue_order=['nWGD', 'WGD clone'],
    palette={'WGD clone':colors_yaml['wgd_prevalence']['Prevalent WGD'], 'nWGD':colors_yaml['wgd_prevalence']['Rare WGD']})

sns.barplot(data=plot_df[['sample_id', 'WGD', 'n_cells']].drop_duplicates(), x = 'sample_id', y = 'n_cells', hue='WGD',  ax=g.axes.flatten()[-1], hue_order=['nWGD', 'WGD clone'],
    palette={'WGD clone':colors_yaml['wgd_prevalence']['Prevalent WGD'], 'nWGD':colors_yaml['wgd_prevalence']['Rare WGD']})


print(g.axes.flatten()[-1].get_xticklabels())
for idx, ax in enumerate(g.axes.flatten()):
    ax.set_xlabel('')
    ax.set_title('')
    if idx == 4:
        #ax.set_xticks([0, 1, 2, 3])
        for label in ax.get_xticklabels():
            label.set_rotation(60)
            label.set_ha('right')
            label.set_rotation_mode('anchor')  
        ax.set_ylabel("Num. cells")
        ax.get_legend().remove()
    else:
        ax.spines["bottom"].set_visible(False)
        ax.tick_params(bottom=False)
        ax.set_yscale('log')
        ax.set_ylim((0.02, 5))
        ax.set_ylabel(event_type_names[idx] + '\nper cell')
g.savefig(f'../../figures/final/model/rpe_barplot.svg', bbox_inches='tight', metadata={'Date': None})
```

```python
plot_df.pivot(index='event_type', columns=['sample_id', 'WGD'], values = 'rate')[order]
```

```python
plot_df['count'] / plot_df['n_cells']
```

```python
plot_df.rename(columns={'region2':'region'}).to_csv('../../tables/rpe1_rates.csv', index=False)
```


