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
import yaml
import vetica.mpl
import pickle
import matplotlib.colors as mcolors
```

```python
colors_yaml = yaml.safe_load(open('/data1/shahs3/users/myersm2/repos/spectrumanalysis/config/colors.yaml', 'r').read())
```

```python

sample_ids = [
    'FNE1_p53mut_DMSO',
    'FNE1_p53mut_Reversine',
    'FNE1_p53ko_px17_CC', # mixed WGD/non-WGD
]

hmmcopy_metrics = {
    'FNE1_p53mut_DMSO': '/data1/shahs3/isabl_data_lake/analyses/42/75/44275/results/SHAH_H003339_T05_01_DLP01_metrics.csv.gz',
    'FNE1_p53mut_Reversine': '/data1/shahs3/isabl_data_lake/analyses/42/76/44276/results/SHAH_H003339_T06_01_DLP01_metrics.csv.gz',
    'FNE1_p53ko_px17_CC': '/data1/shahs3/isabl_data_lake/analyses/51/46/45146/results/SHAH_H003339_T08_01_DLP01_metrics.csv.gz',
}

hmmcopy_reads = {
    'FNE1_p53mut_DMSO': '/data1/shahs3/isabl_data_lake/analyses/42/75/44275/results/SHAH_H003339_T05_01_DLP01_hmmcopy_reads.csv.gz',
    'FNE1_p53mut_Reversine': '/data1/shahs3/isabl_data_lake/analyses/42/76/44276/results/SHAH_H003339_T06_01_DLP01_hmmcopy_reads.csv.gz',
    'FNE1_p53ko_px17_CC': '/data1/shahs3/isabl_data_lake/analyses/51/46/45146/results/SHAH_H003339_T08_01_DLP01_hmmcopy_reads.csv.gz',
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
adata.shape
```

```python

adata = scgenome.tl.sort_cells(adata, layer_name=['copy'])

adata = scgenome.tl.cluster_cells(
    adata,
    min_k=10,
    max_k=20,
    layer_name=['copy'],
    method='gmm_diag_bic',
)

fig = plt.figure(figsize=(5, 8), dpi=300)
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
fig = plt.figure(figsize=(5, 8), dpi=300)
g = scgenome.pl.plot_cell_cn_matrix_fig(
    adata[:, (adata.var['chr'] != 'Y') & (adata.var['gc'] > 0)],
    layer_name='state',
    fig=fig,
    style='white',
    cell_order_fields=['sample_id', 'cluster_id', 'cell_order'],
    annotation_fields=['sample_id', 'cluster_id', 'is_s_phase'],
)

```

```python

# Select a cluster to be the low-cin cluster as reference
wgd_cluster = '4'
low_cin_cluster = '7'

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

# fix 0s in chr9 problematic region
adata_clusters.layers['state'][adata_clusters.obs.index.get_loc('7'), np.where(adata_clusters.var.chr == '9')[0]] = np.maximum(adata_clusters[adata_clusters.obs.index.get_loc('7'), adata_clusters.var.chr == '9'].layers['state'], 1)
```

```python
adata_clusters = scgenome.tl.sort_cells(adata_clusters, layer_name=['copy'])

```

```python
adata_clusters.obs
```

```python
g['axes']
```

```python

import spectrumanalysis.plots


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


fig.savefig(f'../../figures/final/model/fne_wgd_clone.svg', bbox_inches='tight', metadata={'Date': None})

```

<!-- #raw -->

import spectrumanalysis.cnevents
import spectrumanalysis.phylocn
import tqdm

def compute_events_from_baseline(adata, adata_clusters, baseline_cluster_id):
    spectrumanalysis.cnevents.annotate_bins(adata)
    
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


        condition = adata.obs.loc[cell_id, 'condition']
        
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
events.to_csv('fne1_events.csv.gz', compression={'method':'gzip', 'mtime':0, 'compresslevel':9})
<!-- #endraw -->

<!-- #raw -->
with open('fne1_cellcounts.pickle', 'wb') as f:
    pickle.dump(cellcounts, f)
<!-- #endraw -->

```python
adata.obs.condition.value_counts()
```

```python
events['condition'] = adata.obs.loc[events.cell_id, 'condition'].values
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
cellcounts
```

```python
event_types = ['chromosome_loss', 'chromosome_gain', 'arm_loss', 'arm_gain', ]
event_type_names = ['Chrom. loss', 'Chrom. gain', 'Arm loss', 'Arm gain']

import seaborn as sns

order = sample_ids

g = sns.catplot(
    x='sample_id', col='event_type', y='rate', kind='bar', col_order=event_types, 
    data=event_counts[~event_counts.is_wgd], sharey=True, aspect=.4, height=3, order=order, bottom=0.02)

for idx, ax in enumerate(g.axes.flatten()):
    ax.set_yscale('log')
    ax.set_ylim((0.02, 5))
    ax.set_xlabel('')
    for label in ax.get_xticklabels():
        label.set_rotation(60)
        label.set_ha('right')
        label.set_rotation_mode('anchor')
    ax.set_title(event_type_names[idx])
g.axes[0][0].set_ylabel("Event count per cell (nWGD)")
fig.savefig(f'../../figures/final/model/fne_nWGD_barplot.svg', bbox_inches='tight', metadata={'Date': None})
```

```python
plot_df.sample_id
```

```python
plot_df
```

```python
plot_df = event_counts.copy()
plot_df = plot_df[(plot_df.sample_id == 'FNE1_p53ko_px17_CC') | ~plot_df.is_wgd]
plot_df.loc[(plot_df.sample_id != 'FNE1_p53ko_px17_CC'), 'sample_id'] = plot_df.loc[(plot_df.sample_id != 'FNE1_p53ko_px17_CC'), 'sample_id'] + '_nWGD'
plot_df.loc[(plot_df.sample_id == 'FNE1_p53ko_px17_CC') & plot_df.is_wgd, 'sample_id'] = 'FNE1_p53ko_px17_CC_WGD'
plot_df.loc[(plot_df.sample_id == 'FNE1_p53ko_px17_CC') & ~plot_df.is_wgd, 'sample_id'] = 'FNE1_p53ko_px17_CC_nWGD'


event_types = ['chromosome_loss', 'chromosome_gain', 'arm_loss', 'arm_gain', ]
event_type_names = ['Chrom. loss', 'Chrom. gain', 'Arm loss', 'Arm gain']

import seaborn as sns

order = [x + '_nWGD' for x in sample_ids[:2]] + ['FNE1_p53ko_px17_CC_nWGD', 'FNE1_p53ko_px17_CC_WGD']

g = sns.catplot(
    x='sample_id', col='event_type', y='rate', kind='bar', col_order=event_types, 
    data=plot_df, sharey=True, aspect=.4, height=3, order=order, bottom=0.02)

for idx, ax in enumerate(g.axes.flatten()):
    ax.set_yscale('log')
    ax.set_ylim((0.02, 5))
    ax.set_xlabel('')
    for label in ax.get_xticklabels():
        label.set_rotation(60)
        label.set_ha('right')
        label.set_rotation_mode('anchor')
    ax.set_title(event_type_names[idx])
g.axes[0][0].set_ylabel("Event count per cell")
#fig.savefig(f'../../figures/final/model/fne_nWGD_barplot.svg', bbox_inches='tight', metadata={'Date': None})
```

```python
event_types = ['chromosome_loss', 'chromosome_gain', 'arm_loss', 'arm_gain', ]
event_type_names = ['Chrom. loss', 'Chrom. gain', 'Arm loss', 'Arm gain']

import seaborn as sns

order = sample_ids

g = sns.catplot(
    x='sample_id', col='event_type', y='rate', kind='bar', col_order=event_types, 
    data=event_counts[~event_counts.is_wgd], sharey=True, aspect=.4, height=3, order=order, bottom=0.02)

for idx, ax in enumerate(g.axes.flatten()):
    ax.set_yscale('log')
    ax.set_ylim((0.02, 5))
    ax.set_xlabel('')
    for label in ax.get_xticklabels():
        label.set_rotation(60)
        label.set_ha('right')
        label.set_rotation_mode('anchor')
    ax.set_title(event_type_names[idx])
g.axes[0][0].set_ylabel("Event count per cell (nWGD)")
#fig.savefig(f'../../figures/final/model/fne_nWGD_barplot.svg', bbox_inches='tight', metadata={'Date': None})
```

# compare ploidy>2.5 to wgd by events

```python
adata.obs['ploidy'] = np.nanmean(adata.layers['state'], axis = 1)
adata.obs['wgd_by_ploidy'] = adata.obs.ploidy > 2.5
adata.obs['wgd_by_events'] = adata.obs.index.isin(events[events.is_wgd].cell_id.unique())
adata.obs[['wgd_by_ploidy', 'wgd_by_events']].value_counts()
```

# try to make a plot in similar style to Matthew's

```python
event_types = ['chromosome_loss', 'chromosome_gain', 'arm_loss', 'arm_gain', ]
event_type_names = ['Chrom. loss', 'Chrom. gain', 'Arm loss', 'Arm gain']

import seaborn as sns

order = sample_ids

g = sns.catplot(
    x='sample_id', col='event_type', y='rate', kind='bar', col_order=event_types, 
    data=event_counts, hue='is_wgd', sharey=True, aspect=.4, height=3, order=order, 
    palette={True:colors_yaml['wgd_prevalence']['Prevalent WGD'], False:colors_yaml['wgd_prevalence']['Rare WGD']})

for idx, ax in enumerate(g.axes.flatten()):
    ax.set_yscale('log')
    ax.set_ylim((0.02, 5))
    ax.set_xlabel('')
    for label in ax.get_xticklabels():
        label.set_rotation(60)
        label.set_ha('right')
        label.set_rotation_mode('anchor')
    ax.set_title(event_type_names[idx])
g.axes[0][0].set_ylabel("Event count per cell (nWGD)")
#fig.savefig(f'../../figures/final/model/fne_nWGD_barplot.svg', bbox_inches='tight', metadata={'Date': None})
```

```python
    spectrumanalysis.cnevents.annotate_bins(adata)

```

```python
np.nanmean(adata[:, (adata.var.chr == '4')].layers['copy'], axis=1)
```

```python
adata.obs['4/1q'] = np.nanmean(adata[:, (adata.var.chr == '4')].layers['copy'], axis=1) / np.nanmean(adata[:, (adata.var.chr == '1') & (adata.var.arm == 'q')].layers['copy'], axis=1)
adata.obs['5/1q'] = np.nanmean(adata[:, (adata.var.chr == '5')].layers['copy'], axis=1) / np.nanmean(adata[:, (adata.var.chr == '1') & (adata.var.arm == 'q')].layers['copy'], axis=1)
adata.obs['4/3'] = np.nanmean(adata[:, (adata.var.chr == '4')].layers['copy'], axis=1) / np.nanmean(adata[:, (adata.var.chr == '3')].layers['copy'], axis=1)
adata.obs['5/3'] = np.nanmean(adata[:, (adata.var.chr == '5')].layers['copy'], axis=1) / np.nanmean(adata[:, (adata.var.chr == '3')].layers['copy'], axis=1)
```

```python
sns.scatterplot(adata.obs, x = '4/1q', y ='5/1q', hue = 'wgd_by_ploidy')
plt.xlim(0, 5)
plt.ylim(0, 5)
```

```python
sns.scatterplot(adata.obs, x = '4/3', y ='5/3', hue = 'wgd_by_ploidy')
plt.xlim(0, 5)
plt.ylim(0, 5)
```

```python
prop_wgd = {c:(cellcounts[c, 'wgd'] / (cellcounts[c, 'nwgd'] + cellcounts[c, 'wgd'])) for c in event_counts.condition.unique()}
prop_wgd = pd.DataFrame(prop_wgd.items(), columns = ['sample', 'prop_wgd'])
plot_df = event_counts.copy()
plot_df = plot_df[(plot_df.sample_id == 'FNE1_p53ko_px17_CC') | ~plot_df.is_wgd]

event_types = ['chromosome_loss', 'chromosome_gain', 'arm_loss', 'arm_gain', ]
event_type_names = ['Chrom. loss', 'Chrom. gain', 'Arm loss', 'Arm gain']

import seaborn as sns

order = sample_ids

g = sns.catplot(
    x='sample_id', row='event_type', y='rate', kind='bar', row_order=event_types + ['empty'], 
    data=plot_df, hue='is_wgd', sharey=False, aspect=1.8, height=1.5, order=order, dodge=True,
    palette={True:colors_yaml['wgd_prevalence']['Prevalent WGD'], False:colors_yaml['wgd_prevalence']['Rare WGD']})

frac_axis = g.axes.flatten()[-1]
sns.barplot(prop_wgd, x = 'sample', y = 'prop_wgd', facecolor = colors_yaml['wgd_prevalence']['Prevalent WGD'],
           ax=frac_axis)
frac_axis.get_legend().remove()

print(g.axes.flatten()[-1].get_xticklabels())
for idx, ax in enumerate(g.axes.flatten()):
    ax.set_xlabel('')
    ax.set_title('')
    if idx == 4:
        ax.set_xticks([0, 1, 2])
        for label in ax.get_xticklabels():
            label.set_rotation(60)
            label.set_ha('right')
            label.set_rotation_mode('anchor')  
        ax.set_ylabel("Prop.\nWGD")
        ax.set_ylim(0, 1)
    else:
        for p in ax.patches[:2]:
            p.set_width(p.get_width() * 2)
        ax.spines["bottom"].set_visible(False)
        ax.tick_params(bottom=False)
        ax.set_yscale('log')
        ax.set_ylim((0.02, 5))
        ax.set_ylabel(event_type_names[idx] + '\nper cell')
#fig.savefig(f'../../figures/final/model/fne_nWGD_barplot.svg', bbox_inches='tight', metadata={'Date': None})
```

```python
plot_df.sample_id.unique()
```

```python
prop_wgd = {c:(cellcounts[c, 'wgd'] / (cellcounts[c, 'nwgd'] + cellcounts[c, 'wgd'])) for c in event_counts.condition.unique()}
prop_wgd = pd.DataFrame(prop_wgd.items(), columns = ['sample', 'prop_wgd'])
plot_df = event_counts.copy()
plot_df = plot_df[(plot_df.sample_id == 'FNE1_p53ko_px17_CC') | ~plot_df.is_wgd]
plot_df['WGD'] = plot_df.is_wgd.map({True:'WGD clone', False:'nWGD'})
plot_df['sample_id'] = plot_df.sample_id.map({'FNE1_p53ko_px17_CC':'FNE1-WGD', 'FNE1_p53mut_Reversine':'FNE1-Rev',
                                              'FNE1_p53mut_DMSO':'FNE1-D'})

event_types = ['chromosome_loss', 'chromosome_gain', 'arm_loss', 'arm_gain', ]
event_type_names = ['Chrom. loss', 'Chrom. gain', 'Arm loss', 'Arm gain']

import seaborn as sns

order = ['FNE1-D', 'FNE1-Rev', 'FNE1-WGD']

g = sns.catplot(
    x='sample_id', row='event_type', y='rate', kind='bar', row_order=event_types + ['empty'], 
    data=plot_df, hue='WGD', sharey=False, aspect=1.8, height=1.5, order=order, dodge=True, hue_order=['nWGD', 'WGD clone'],
    palette={'WGD clone':colors_yaml['wgd_prevalence']['Prevalent WGD'], 'nWGD':colors_yaml['wgd_prevalence']['Rare WGD']})

sns.barplot(data=plot_df[['sample_id', 'WGD', 'n_cells']].drop_duplicates(), x = 'sample_id', y = 'n_cells', hue='WGD',  ax=g.axes.flatten()[-1], hue_order=['nWGD', 'WGD clone'],
    palette={'WGD clone':colors_yaml['wgd_prevalence']['Prevalent WGD'], 'nWGD':colors_yaml['wgd_prevalence']['Rare WGD']})


print(g.axes.flatten()[-1].get_xticklabels())
for idx, ax in enumerate(g.axes.flatten()):
    # manually widen singleton bars because sns.catplot doesn't work with so.Dodge(fill='empty')
    for p in ax.patches:
        if p.get_x() < 0.9 and p.get_facecolor() == (0.7637254901960786, 0.5843137254901961, 0.31078431372549, 1): # found this value empirically
            #p.set_x(p.get_x() - p.get_width())
            p.set_width(p.get_width() * 2)
            p.set_zorder(-1)

    ax.set_xlabel('')
    ax.set_title('')
    if idx == 4:
        ax.set_xticks([0, 1, 2])
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
g.savefig(f'../../figures/final/model/fne_barplot.svg', bbox_inches='tight', metadata={'Date': None})
```

```python
plot_df.rename(columns={'region2':'region'}).to_csv('../../tables/fne1_rates.csv', index=False)
```

```python
prop_wgd
```

# check STING copy number for Matthew

```python
# 5q31.2 138855118..138862343
sting_bin = adata.var[(adata.var.chr == '5') & (adata.var.start <= 138855118) & (adata.var.end >= 138862343)].index
adata[adata.obs.cluster_id == wgd_cluster, sting_bin].layers['copy'].mean()
```

```python
adata[adata.obs.cluster_id == wgd_cluster].obs.ploidy.mean()
```

```python
from IPython.display import SVG, display
display(SVG('/data1/shahs3/users/myersm2/repos/spectrum-figures/figures/final/model/fne_barplot.svg'))
```

```python
from IPython.display import SVG, display
display(SVG('/data1/shahs3/users/myersm2/repos/spectrum-figures/figures/final/model/rpe_barplot.svg'))
```

```python

```

```python

```
