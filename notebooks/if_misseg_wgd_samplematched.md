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

import pandas as pd
from scipy.spatial import KDTree
import tqdm
import glob
import os
import yaml
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
import anndata as ad
import pandas as pd

import spectrumanalysis.wgd

project_dir = os.environ['SPECTRUM_PROJECT_DIR']

```

```python

colors_dict = yaml.safe_load(open('../../../../config/colors.yaml', 'r'))

```

```python

cell_info = pd.read_csv(f'{project_dir}/preprocessing/summary/filtered_cell_table.csv.gz')
cell_info = cell_info[(cell_info['include_cell'] == True)]
cell_info = spectrumanalysis.wgd.classify_subclonal_wgd(cell_info)

```

```python

fraction_wgd = pd.read_csv('../../../../annotations/fraction_wgd_class.csv')
fraction_wgd.head()

```

```python

# Load Misseg events

event_rates_filename = f'{project_dir}/postprocessing/sankoff_ar/greedy_events/sankoff_ar_rates.tsv'
all_event_rates = pd.read_csv(event_rates_filename, sep='\t')

event_rates = all_event_rates.query(f'group_level == "sample" & normalized == False')
event_rates = event_rates.drop(['group_level', 'normalized', 'n_wgd', 'majority_n_wgd', 'subclonal_wgd'], axis=1)

```

```python

high_background = [
    'SPECTRUM-OV-003_S1_LEFT_UPPER_QUADRANT_cGAS_ENPP1_DAPI_R2',
    'SPECTRUM-OV-003_S1_PELVIC_PERITONEUM_cGAS_ENPP1_DAPI_R1',
    'SPECTRUM-OV-003_S1_RIGHT_ADNEXA_cGAS_ENPP1_DAPI_R1',
    'SPECTRUM-OV-003_S1_RIGHT_UPPER_QUADRANT_cGAS_ENPP1_DAPI_R1',
    'SPECTRUM-OV-025_S1_BOWEL_cGAS_ENPP1_DAPI_R1',
    'SPECTRUM-OV-036_S1_RIGHT_ADNEXA_cGAS_ENPP1_DAPI_R2',
    'SPECTRUM-OV-036_S1_INFRACOLIC_OMENTUM_cGAS_ENPP1_DAPI_R1',
]

```

```python

if_metadata = pd.read_csv('../../../../metadata/tables/if_slide.tsv', sep='\t')
if_metadata = if_metadata[if_metadata['panel'].str.contains('cGAS')]
if_metadata['slide_id'] = if_metadata['spectrum_sample_id']

scdna_metadata = pd.read_csv('../../../../metadata/tables/sequencing_scdna.tsv', sep='\t')
scdna_metadata['sample_id'] = scdna_metadata['spectrum_sample_id']

```

```python

# Load MN rates
remove_batches = [7, 8]
remove_high_background = True
quality_filter = ['High', 'Medium']

sample_mn_rates = pd.read_csv(f'{project_dir}/analyses/if/mn_rates/ignacio_mn_rates.csv')
sample_mn_rates = sample_mn_rates.merge(if_metadata[['slide_id', 'slide_qc']])
if quality_filter is not None:
    sample_mn_rates = sample_mn_rates[sample_mn_rates['slide_qc'].isin(quality_filter)]
if remove_high_background:
    sample_mn_rates = sample_mn_rates[~sample_mn_rates['slide_id'].isin(high_background)]
sample_mn_rates = sample_mn_rates[~sample_mn_rates['batch'].isin(remove_batches)]

sample_mn_rates = sample_mn_rates[sample_mn_rates['patient_id'].isin(event_rates['patient_id'].values)]

```


```python

# Sample matching table
matching_columns = ['patient_id', 'tumor_site', 'therapy']
matched_samples = pd.merge(
    if_metadata[matching_columns + ['slide_id']].drop_duplicates(),
    scdna_metadata[matching_columns + ['sample_id']].drop_duplicates(),
    on=matching_columns)

matched_samples = matched_samples[matched_samples['slide_id'].isin(sample_mn_rates['slide_id'].values)]
matched_samples = matched_samples[matched_samples['sample_id'].isin(event_rates['sample_id'].values)]

```

# MN/PN stats


```python

sample_mn_rates[['pn_count', 'mn_count']].sum()

```

```python

sample_mn_rates.shape[0]

```

```python

sample_mn_rates['patient_id'].unique().shape

```

```python

sample_mn_rates['mn_rate'].describe()

```

# Misseg MN rate correlations and assocation with WGD


```python

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

def scaler_transform(a):
    a = a.values[:, np.newaxis]
    return scaler.fit_transform(a)[:, 0]

sample_mn_rates['mn_rate_log'] = np.log(sample_mn_rates['mn_rate'] + 1e-6)
sample_mn_rates['mn_rate_zscore'] = sample_mn_rates.groupby(['batch'])['mn_rate'].transform(scaler_transform)
sample_mn_rates['mn_rate_log_zscore'] = sample_mn_rates.groupby(['batch'])['mn_rate_log'].transform(scaler_transform)

```

```python

import statsmodels.api as sm
import statsmodels.formula.api as smf

plot_data = (sample_mn_rates.merge(fraction_wgd))

plot_data['is_rare_wgd'] = (plot_data['wgd_class'] == 'Rare WGD') * 1

col = 'mn_rate_log_zscore'

fig, axes = plt.subplots(nrows=2, height_ratios=[2, 1], figsize=(1, 2.5), dpi=300, sharex=True)

ax = axes[0]
sns.boxplot(
    ax=ax, x='wgd_class', y=col, hue='wgd_class', palette=colors_dict['wgd_prevalence'],
    data=plot_data)
sns.stripplot(
    ax=ax, x='wgd_class', y=col, hue='wgd_class', palette=colors_dict['wgd_prevalence'],
    data=plot_data, color='k', linewidth=1)
ax.set_ylabel('MN rate log z-score')

model = smf.gee(f'is_rare_wgd ~ {col}', 'patient_id', plot_data, family=sm.families.Binomial())
result = model.fit()
print(result.summary())
p = result.pvalues[col]
coef = result.params[col]
ax.text(.1, 1.05, 'p={:.2g}'.format(p),
        transform=ax.transAxes)
sns.despine(ax=ax)

ax = axes[1]
sns.countplot(
    ax=ax, x='wgd_class', palette=colors_dict['wgd_prevalence'],
    data=plot_data)
for label in ax.get_xticklabels():
    label.set_rotation(45)
    label.set_ha('right')
    label.set_rotation_mode('anchor')
ax.set_ylabel('Count')
ax.set_xlabel('')
sns.despine(ax=ax)

```

```python

import statsmodels.api as sm
import statsmodels.formula.api as smf

plot_data = event_rates.melt(id_vars=['patient_id', 'sample_id', 'n_cells'], var_name='class', value_name='event_rate')

plot_data = plot_data.merge(matched_samples)

plot_data = (plot_data
    .merge(sample_mn_rates)
    .merge(fraction_wgd))

plot_data = plot_data[plot_data['n_cells'] >= 10]

event_type_names = {
    'chromosome_gain': 'Chrom. gain',
    'arm_gain': 'Arm gain',
    'segment_gain': 'Segment gain',
    'chromosome_loss': 'Chrom. loss',
    'arm_loss': 'Arm loss',
    'segment_loss': 'Segment loss',
}

plot_data['class'] = plot_data['class'].map(event_type_names)

misseg_cols = [
    'Chrom. gain', 'Arm gain', 'Segment gain',
    'Chrom. loss', 'Arm loss', 'Segment loss',
]

g = sns.FacetGrid(plot_data, col='class', col_order=misseg_cols, sharex=False, sharey=True)
g.fig.set_dpi(300)
g.fig.set_figwidth(10)
g.fig.set_figheight(3)

def annotate(data, **kws):
    offset = 0
    model = smf.mixedlm("mn_rate_log_zscore ~ event_rate", data, groups=data.reset_index()['patient_id'])
    result = model.fit()
    p = result.pvalues['event_rate']
    coef = result.params['event_rate']
    ax = plt.gca()
    sns.regplot(ax=ax, x='event_rate', y='mn_rate_log_zscore', data=data, scatter=False)
    sns.scatterplot(
        ax=ax, x='event_rate', y='mn_rate_log_zscore', hue='wgd_class', palette=colors_dict['wgd_prevalence'],
        s=20, linewidth=1, edgecolor='k', data=data)
    ax.text(.55, .9 - offset * .1, 'p={:.2g}'.format(p),
            transform=ax.transAxes, color='k')

g.map_dataframe(annotate)

g.set_titles(template='{col_name}')

g.axes[0, 0].set_ylabel('MN rate log z-score')
for ax in g.axes[0, :]:
    ax.set_xlabel('Count / cell')

```


```python

cell_info = pd.read_csv(f'{project_dir}/preprocessing/summary/filtered_cell_table.csv.gz')

fraction_multipolar = cell_info.groupby('sample_id')['multipolar'].mean().rename('fraction_multipolar').reset_index()

plot_data = fraction_multipolar.merge(matched_samples)

plot_data = (plot_data
    .merge(sample_mn_rates)
    .merge(fraction_wgd))

ax = plt.gca()
ax = plt.gca()
sns.regplot(ax=ax, x='fraction_multipolar', y='mn_rate_log_zscore', data=plot_data, scatter=False)
sns.scatterplot(
    ax=ax, x='fraction_multipolar', y='mn_rate_log_zscore', hue='wgd_class', palette=colors_dict['wgd_prevalence'],
    s=20, linewidth=1, edgecolor='k', data=plot_data)
sns.despine()

plot_data.sort_values('fraction_multipolar', ascending=False).head(10)

```

```python

wgd_event_rates = all_event_rates.query(f'group_level == "sample_wgd" & normalized == False').query('subclonal_wgd == False')
wgd_event_rates = wgd_event_rates.rename(columns=event_type_names)[['patient_id', 'sample_id', 'n_cells'] + misseg_cols]

wgd_fraction = cell_info.groupby(['patient_id', 'sample_id'])['subclonal_wgd'].mean().rename('fraction_subclonal_wgd').reset_index()

multipolar_fraction = cell_info.groupby(['patient_id', 'sample_id'])['multipolar'].mean().rename('fraction_multipolar').reset_index()

plot_data = wgd_event_rates.merge(wgd_fraction).merge(multipolar_fraction).merge(matched_samples).merge(sample_mn_rates)

plot_data = plot_data.rename(columns={
    'fraction_subclonal_wgd': 'Fraction +1 WGD',
    'fraction_multipolar': 'Fraction divergent',
    'mn_rate': 'Micronuclei rate',
})

plot_data = plot_data[plot_data['n_cells'] >= 100]

plot_data = plot_data[misseg_cols + ['Fraction +1 WGD', 'Fraction divergent', 'Micronuclei rate']].corr(method='spearman')

g = sns.clustermap(plot_data, vmin=0, vmax=1, cmap='Blues')
g.fig.set_figwidth(3)
g.fig.set_figheight(3)

```

```python

```
