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
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import yaml
import vetica.mpl

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import spectrumanalysis.wgd

project_dir = os.environ['SPECTRUM_PROJECT_DIR']

colors_dict = yaml.safe_load(open('../../../../config/colors.yaml', 'r'))

sigs = pd.read_table('../../../../annotations/mutational_signatures.tsv')

```

```python

cell_info = pd.read_csv(f'{project_dir}/preprocessing/summary/filtered_cell_table.csv.gz')
cell_info = cell_info[(cell_info['include_cell'] == True)]

multipolar_fraction = cell_info.groupby(['patient_id'])['multipolar'].mean().rename('multipolar_fraction').reset_index()

cell_info = spectrumanalysis.wgd.classify_subclonal_wgd(cell_info)
fraction_subclonal_wgd = cell_info.groupby('patient_id')['subclonal_wgd'].mean().rename('fraction_subclonal_wgd').reset_index()

snv_leaf_table = pd.read_csv('../../../../results/tables/snv_tree/snv_leaf_table.csv')

fraction_wgd = pd.read_csv('../../../../annotations/fraction_wgd_class.csv')
snv_leaf_table = snv_leaf_table.merge(fraction_wgd)

```

```python

import  scipy.stats

patient_snv_count_info = (snv_leaf_table
    .groupby(['patient_id', 'wgd_class']).agg({
        'snv_count_age_per_gb_to_wgd': np.mean,
        'snv_count_age_per_gb_since_birth': np.mean,
    }).reset_index())

print(scipy.stats.mannwhitneyu(
    patient_snv_count_info.loc[patient_snv_count_info['wgd_class'] == 'Prevalent WGD', 'snv_count_age_per_gb_since_birth'],
    patient_snv_count_info.loc[patient_snv_count_info['wgd_class'] == 'Rare WGD', 'snv_count_age_per_gb_since_birth'],
    alternative='greater'))

plot_data = patient_snv_count_info.melt(id_vars=['patient_id', 'wgd_class'], ignore_index=False, var_name='event', value_name='C>T CpG Mut / GB')

plot_data = plot_data[(plot_data['wgd_class'] == 'Prevalent WGD') | (plot_data['event'] != 'snv_count_age_per_gb_to_wgd')]

plot_data['event'] = plot_data['event'].map({
    'snv_count_age_per_gb_to_wgd': 'WGD',
    'snv_count_age_per_gb_since_birth': 'Diagnosis'})
plot_data

g = sns.displot(
    x='C>T CpG Mut / GB', hue='event', row='wgd_class', data=plot_data, common_norm=False, alpha=0.25,
    kind='hist', fill=True, rug=True, bins=16, rug_kws=dict(height=0.1, linewidth=2),# clip=(0, None),
    aspect=3., height=2, facet_kws=dict(sharey=False), row_order=['Rare WGD', 'Prevalent WGD'])
sns.despine(trim=True)
g.axes[0, 0].set_title('Rare WGD patients', y=.9)
g.axes[1, 0].set_title('Prevalent WGD patients', y=.9)
g.axes[1, 0].set_xlabel('C>T CpG Mut / GB mutation time from conception until event')
sns.move_legend(g, 'upper left', bbox_to_anchor=(0.65, 0.95), ncol=1, title='Event', frameon=False)

g.fig.savefig(f'../../../../figures/figure2/age_to_event.svg', metadata={'Date': None})

```

```python

patient_snv_count_info = (snv_leaf_table
    .groupby(['patient_id', 'is_wgd']).agg({
        'snv_count_age_per_gb_since_birth': np.mean,
    })).reset_index()

sns.stripplot(y='snv_count_age_per_gb_since_birth', x='is_wgd', data=patient_snv_count_info)

```

```python

patient_snv_count_info = (snv_leaf_table
    .groupby(['patient_id', 'is_wgd']).agg({
        'snv_count_root_age_per_gb': np.mean,
        'snv_count_age_per_gb_since_birth': np.mean,
    }))
patient_snv_count_info['mrca'] = patient_snv_count_info['snv_count_age_per_gb_since_birth'] - patient_snv_count_info['snv_count_root_age_per_gb']
patient_snv_count_info = patient_snv_count_info.reset_index()

patient_snv_count_info['is_wgd'] = patient_snv_count_info['is_wgd'].map({
    True: 'Prevalent WGD',
    False: 'Rare WGD',
})

plt.figure(figsize=(1.25, 3))
ax = sns.boxplot(x='is_wgd', y='mrca', hue='is_wgd', palette=colors_dict['wgd_prevalence'], data=patient_snv_count_info, fliersize=1)
ax.set_title('MRCA')
ax.set_xlabel('')
ax.set_ylabel('C>T CpG Mut / GB')
sns.despine()
for label in ax.get_xticklabels():
    label.set_rotation(60)
    label.set_ha('right')
    label.set_rotation_mode('anchor')

```

```python

patient_wgd_snv_count_info = (snv_leaf_table
    .query('is_wgd')
    .groupby('patient_id').agg({
        'snv_count_age_per_gb_to_wgd': np.mean,
        'snv_count_age_per_gb_since_wgd': np.mean,
        'snv_count_age_per_gb_since_birth': np.mean,
        'cluster_size': np.sum}).reset_index())
patient_wgd_snv_count_info['n_cells'] = patient_wgd_snv_count_info['cluster_size'].astype(float)
patient_wgd_snv_count_info['snv_count_age_per_gb_to_wgd_fraction'] = patient_wgd_snv_count_info['snv_count_age_per_gb_to_wgd'] / patient_wgd_snv_count_info['snv_count_age_per_gb_since_birth']
patient_wgd_snv_count_info.sort_values('snv_count_age_per_gb_to_wgd_fraction')

```

```python

patient_wgd_snv_count_info['snv_count_age_per_gb_to_wgd_fraction'].hist()

```

```python

patient_wgd_snv_count_info.shape[0], (patient_wgd_snv_count_info['snv_count_age_per_gb_to_wgd_fraction'] > 0.5).sum()

```

```python

event_types = [
    'chromosome_gain',
    'arm_gain',
    'segment_gain',
    'chromosome_loss',
    'arm_loss',
    'segment_loss',
]

event_rates_filename = os.path.join(f'{project_dir}/postprocessing/sankoff_ar/sankoff_ar_rates.tsv')
event_rates = pd.read_csv(event_rates_filename, sep='\t')

event_rates = event_rates.query(f'group_level == "patient" & normalized == False')
event_rates = event_rates.melt(id_vars=['patient_id'], value_vars=event_types, value_name='event_rate', var_name='class')

```

```python

import scipy

plot_data = (event_rates
    .merge(sigs)
    .merge(patient_wgd_snv_count_info))

# Yuck
# plot_data = plot_data[~plot_data['patient_id'].isin(['SPECTRUM-OV-065'])]

plot_data = plot_data[plot_data['n_cells'] > 100]

g = sns.lmplot(
    x='snv_count_age_per_gb_since_wgd', y='event_rate', col='class', col_wrap=6,
    data=plot_data, sharex=False, sharey=False, height=2.5, scatter_kws=dict(s=4))

def annotate(data, **kws):
    offset = 0
    r, p = scipy.stats.spearmanr(data['event_rate'], data['snv_count_age_per_gb_since_wgd'])
    ax = plt.gca()
    ax.text(.05, .8 - offset * .1, 'r={:.2f}, p={:.2g}'.format(r, p),
            transform=ax.transAxes, color=kws['color'])

g.map_dataframe(annotate)

plot_data.sort_values('snv_count_age_per_gb_since_wgd')

```

```python

plot_data = (event_rates
    .merge(sigs)
    .merge(patient_wgd_snv_count_info))

sns.catplot(x='consensus_signature', y='event_rate', hue='consensus_signature', col='class',
            data=plot_data, palette=colors_dict['consensus_signature'], kind='box', fliersize=0, sharey=False, aspect=.5)

```

```python

plot_data = (patient_wgd_snv_count_info
    .merge(sigs))

fig, ax = plt.subplots(figsize=(1, 3))

sns.boxplot(ax=ax, x='consensus_signature', y='snv_count_age_per_gb_to_wgd', hue='consensus_signature',
            data=plot_data, palette=colors_dict['consensus_signature'], fliersize=0)

sns.stripplot(ax=ax, x='consensus_signature', y='snv_count_age_per_gb_to_wgd', hue='consensus_signature',
              data=plot_data, linewidth=1, palette=colors_dict['consensus_signature'])

ax.set_xlabel('')
ax.set_ylabel('C>T CpG\nMut / GB')
for label in ax.get_xticklabels():
    label.set_rotation(60)
    label.set_ha('right')
    label.set_rotation_mode('anchor')

sns.despine()

```

```python

plot_data = (patient_wgd_snv_count_info
    .merge(sigs))

fig, ax = plt.subplots(figsize=(1, 3))

sns.boxplot(ax=ax, x='consensus_signature', y='snv_count_age_per_gb_since_wgd', hue='consensus_signature',
            data=plot_data, palette=colors_dict['consensus_signature'], fliersize=0)

sns.stripplot(ax=ax, x='consensus_signature', y='snv_count_age_per_gb_since_wgd', hue='consensus_signature',
              data=plot_data, linewidth=1, palette=colors_dict['consensus_signature'])

ax.set_xlabel('')
ax.set_ylabel('C>T CpG\nMut / GB')
for label in ax.get_xticklabels():
    label.set_rotation(60)
    label.set_ha('right')
    label.set_rotation_mode('anchor')

sns.despine()

```

```python

plot_data = (patient_wgd_snv_count_info
    .merge(sigs))

fig, ax = plt.subplots(figsize=(1, 3))

sns.boxplot(ax=ax, x='consensus_signature', y='snv_count_age_per_gb_since_birth', hue='consensus_signature',
            data=plot_data, palette=colors_dict['consensus_signature'], fliersize=0)

sns.stripplot(ax=ax, x='consensus_signature', y='snv_count_age_per_gb_since_birth', hue='consensus_signature',
              data=plot_data, linewidth=1, palette=colors_dict['consensus_signature'])

ax.set_xlabel('')
ax.set_ylabel('C>T CpG\nMut / GB')
for label in ax.get_xticklabels():
    label.set_rotation(60)
    label.set_ha('right')
    label.set_rotation_mode('anchor')

sns.despine()

```

```python

plot_data = (patient_wgd_snv_count_info
    .merge(sigs))

plot_data = plot_data[plot_data['n_cells'] > 100]

fig, ax = plt.subplots(figsize=(3, 3))

sns.scatterplot(ax=ax, x='snv_count_age_per_gb_since_birth', y='snv_count_age_per_gb_to_wgd', hue='consensus_signature',
            data=plot_data, palette=colors_dict['consensus_signature'])

ax.set_xlim((0, 210))
ax.set_xlabel('C>T CpG Mut / GB to Diagnosis')
ax.set_ylabel('C>T CpG Mut / GB to WGD')

sns.despine(trim=True, offset=10)
sns.move_legend(ax, 'upper left', bbox_to_anchor=(1.1, 1), title='Signature', frameon=False)

```

```python

plot_data = (patient_wgd_snv_count_info
    .merge(sigs))

plot_data = plot_data[plot_data['n_cells'] > 100]

fig, ax = plt.subplots(figsize=(3, 3))

sns.scatterplot(ax=ax, x='snv_count_age_per_gb_since_wgd', y='snv_count_age_per_gb_to_wgd', hue='consensus_signature',
            data=plot_data, palette=colors_dict['consensus_signature'])

ax.set_xlim((0, 200))
ax.set_xlabel('C>T CpG Mut / GB since WGD')
ax.set_ylabel('C>T CpG Mut / GB to WGD')

sns.despine(trim=True, offset=10)
sns.move_legend(ax, 'upper right', title='Signature', frameon=False)

```

```python

plot_data = (patient_wgd_snv_count_info
    .merge(multipolar_fraction))

plot_data = plot_data[plot_data['n_cells'] > 100]

fig, ax = plt.subplots(figsize=(3, 3))

sns.scatterplot(ax=ax, x='snv_count_age_per_gb_since_wgd', y='multipolar_fraction', data=plot_data)

```

```python

plot_data = (patient_wgd_snv_count_info
    .merge(sigs)
    .merge(fraction_subclonal_wgd))

plot_data = plot_data[plot_data['n_cells'] > 100]

fig, ax = plt.subplots(figsize=(3, 3))

sns.scatterplot(
    ax=ax, x='snv_count_age_per_gb_since_wgd', y='fraction_subclonal_wgd', hue='consensus_signature',
    data=plot_data, palette=colors_dict['consensus_signature'])

ax.set_xlim((0, 200))
ax.set_xlabel('C>T CpG Mut / GB since WGD')
ax.set_ylabel('Fraction Subclonal WGD')

sns.despine(trim=True, offset=10)
sns.move_legend(ax, 'upper right', title='Signature', frameon=False)

```

```python

plot_data = sigs.merge(fraction_subclonal_wgd)
plot_data = plot_data[~plot_data['patient_id'].isin(['SPECTRUM-OV-081'])]

fig, ax = plt.subplots(figsize=(1, 3))

sns.boxplot(ax=ax, x='consensus_signature', y='fraction_subclonal_wgd', hue='consensus_signature',
    data=plot_data, palette=colors_dict['consensus_signature'], fliersize=0)

sns.stripplot(
    ax=ax, x='consensus_signature', y='fraction_subclonal_wgd', hue='consensus_signature',
    data=plot_data, linewidth=1, palette=colors_dict['consensus_signature'])

ax.set_xlabel('')
ax.set_ylabel('Fraction Subclonal WGD')
for label in ax.get_xticklabels():
    label.set_rotation(60)
    label.set_ha('right')
    label.set_rotation_mode('anchor')

sns.despine()
plot_data.sort_values('fraction_subclonal_wgd')

```

```python

```
