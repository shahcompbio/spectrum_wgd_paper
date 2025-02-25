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
import vetica.mpl
import statsmodels.api as sm
import statsmodels.formula.api as smf

import spectrumanalysis.wgd
import spectrumanalysis.stats

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

tp53 = pd.read_csv('../../../../annotations/tp53_mutation_type.csv')
tp53['is_nonsense'] = tp53['Variant_Classification'].isin([
    'Frame_Shift_Del', 'Frame_Shift_Ins', 'Nonsense_Mutation'])

```

```python

# Load Misseg events

event_rates_filename = f'{project_dir}/postprocessing/sankoff_ar/sankoff_ar_rates.tsv'
all_event_rates = pd.read_csv(event_rates_filename, sep='\t')

event_rates = all_event_rates.query(f'group_level == "sample" & normalized == False')
event_rates = event_rates.drop(['group_level', 'normalized', 'n_wgd', 'majority_n_wgd', 'subclonal_wgd', 'aliquot_id'], axis=1)

# # Exclude subclonal wgd
# event_rates = all_event_rates.query(f'group_level == "sample_wgd" & normalized == False & subclonal_wgd == False')
# event_rates = event_rates.drop(['group_level', 'normalized', 'n_wgd', 'majority_n_wgd', 'subclonal_wgd', 'aliquot_id'], axis=1)

```

```python

# Load MN rates
sample_mn_rates = pd.read_csv(f'{project_dir}/if/mn_rate.csv')
# sample_mn_rates = sample_mn_rates[sample_mn_rates['patient_id'].isin(event_rates['patient_id'].values)]
sample_mn_rates['sample_id'] = sample_mn_rates['image'].str.replace('_cGAS_STING_p53_panCK_CD8_DAPI_R1', '')

```


```python

sample_mn_rates = sample_mn_rates.query('pn_count > 1000')

```

```python

sample_mn_rates = sample_mn_rates.merge(tp53, how='left', on='patient_id')
assert not sample_mn_rates['Variant_Classification'].isnull().any()

```

```python

sns.stripplot(y='Variant_Classification', x='tp53_mean', data=sample_mn_rates)

```

```python

sns.scatterplot(x='tp53_mean', y='cgas_mean', data=sample_mn_rates, hue='Variant_Classification')

```

```python

variant_classification = 'Splice_Site'
variant_classification = 'Missense_Mutation'
# variant_classification = 'In_Frame_Del'

plot_data = sample_mn_rates.query(f'Variant_Classification == "{variant_classification}"')

model = smf.mixedlm(f'tp53_mean ~ cgas_mean', plot_data, groups='patient_id')
result = model.fit()
print(result.summary(), result.pvalues['cgas_mean'])
p = result.pvalues['cgas_mean']

sns.lmplot(x='tp53_mean', y='cgas_mean', data=plot_data, height=3)
plt.title(f'{variant_classification} (p={p:2f})')

```

```python

sigs = pd.read_table('../../../../annotations/mutational_signatures.tsv')
plot_data = tp53.merge(sigs).merge(fraction_wgd)
plot_data['is_nonsense_frameshift'] = plot_data['Variant_Classification'].isin([
    'Frame_Shift_Del', 'Frame_Shift_Ins', 'Nonsense_Mutation'])
plot_data.groupby(['wgd_class', 'consensus_signature', 'is_nonsense_frameshift']).size().rename('num_patients').unstack(level=2, fill_value=0)

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

```python

sample_mn_rates['batch'].value_counts()

```

# Misseg MN rate correlations and assocation with WGD


```python

plot_data = (sample_mn_rates.merge(fraction_wgd)).merge(sigs[['patient_id', 'consensus_signature']], how='left')

stat, pvalue = scipy.stats.mannwhitneyu(
    plot_data.loc[plot_data['wgd_class'] == 'WGD-high', 'mn_rate'],
    plot_data.loc[plot_data['wgd_class'] == 'WGD-low', 'mn_rate'],
    alternative='greater')
print(stat, pvalue)

plot_data['is_rare_wgd'] = (plot_data['wgd_class'] == 'WGD-low') * 1
plot_data['is_hrd'] = plot_data['consensus_signature'].str.startswith('HRD') * 1
plot_data['log_mn_rate'] = np.log(plot_data['mn_rate'])

fig, axes = plt.subplots(nrows=2, height_ratios=[2, 1], figsize=(.6, 2.5), dpi=150, sharex=True)

ax = axes[0]
sns.boxplot(
    ax=ax, x='wgd_class', y='mn_rate', hue='wgd_class', palette=colors_dict['wgd_class'],
    order=('WGD-low', 'WGD-high'), data=plot_data, linewidth=1, fliersize=1)
ax.set_ylabel('MN rate')
ax.set_yscale('log')

model = smf.gee(
    f'log_mn_rate ~ is_rare_wgd', 'patient_id',
    plot_data, cov_struct=sm.cov_struct.Exchangeable(),
    family=sm.families.Gaussian())

result = model.fit()
print(result.summary())
p = result.pvalues['is_rare_wgd']
coef = result.params['is_rare_wgd']
significance = spectrumanalysis.stats.get_significance_string(p)
spectrumanalysis.stats.add_significance_line(ax, significance, 0, 1, 1.05)
sns.despine(ax=ax)

ax = axes[1]
sns.countplot(
    ax=ax, x='wgd_class', palette=colors_dict['wgd_class'],
    data=plot_data)
for label in ax.get_xticklabels():
    label.set_rotation(45)
    label.set_ha('right')
    label.set_rotation_mode('anchor')
ax.set_ylabel('Count')
ax.set_xlabel('')
sns.despine(ax=ax)

fig.savefig('../../../../figures/figure3/mn_rate_wgd.svg', bbox_inches='tight', metadata={'Date': None})

```

```python

fig, ax = plt.subplots(figsize=(3, 1.5), dpi=150, sharex=True)

ax = sns.boxplot(
    ax=ax, y='consensus_signature', x='mn_rate', hue='wgd_class', palette=colors_dict['wgd_class'],
    data=plot_data, linewidth=1, fliersize=1)
sns.move_legend(ax, loc='upper left', bbox_to_anchor=(1, 1))
sns.despine()

```

```python

plot_data['log_sting'] = np.log(plot_data['sting'])
plot_data['log_cgas'] = np.log(plot_data['cgas_mean'])

fig, axes = plt.subplots(nrows=2, height_ratios=[2, 1], figsize=(.6, 2.5), dpi=150, sharex=True)

ax = axes[0]
sns.boxplot(
    ax=ax, x='wgd_class', y='sting', hue='wgd_class', palette=colors_dict['wgd_class'],
    order=('WGD-low', 'WGD-high'), data=plot_data, linewidth=1, fliersize=1)
ax.set_ylabel('STING1')

model = smf.gee(
    f'log_sting ~ is_rare_wgd', 'patient_id',
    plot_data, cov_struct=sm.cov_struct.Exchangeable(),
    family=sm.families.Gaussian())
result = model.fit()
print(result.summary())
p = result.pvalues['is_rare_wgd']
coef = result.params['is_rare_wgd']
significance = spectrumanalysis.stats.get_significance_string(p)
spectrumanalysis.stats.add_significance_line(ax, significance, 0, 1, 1.05)
sns.despine(ax=ax)

ax = axes[1]
sns.countplot(
    ax=ax, x='wgd_class', palette=colors_dict['wgd_class'],
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

fig, ax = plt.subplots(figsize=(3, 1.5), dpi=150, sharex=True)

ax = sns.boxplot(
    ax=ax, y='consensus_signature', x='log_sting', hue='wgd_class', palette=colors_dict['wgd_class'],
    data=plot_data, linewidth=1, fliersize=1)
sns.move_legend(ax, loc='upper left', bbox_to_anchor=(1, 1))
sns.despine()

```

```python

import statsmodels.api as sm
import statsmodels.formula.api as smf

plot_data = sample_mn_rates.merge(fraction_wgd)

# plot_data = plot_data[~plot_data['image'].str.startswith('SPECTRUM-OV-070_S1_RIGHT_UPPER_QUADRANT')]

plot_data['log_mn_rate'] = np.log(plot_data['mn_rate'])
plot_data['log_sting'] = np.log(plot_data['sting'])

plot_data['is_nwgd'] = (plot_data['wgd_class'] == 'WGD-low') * 1

model = smf.mixedlm(f"log_sting ~ log_mn_rate * is_nwgd", plot_data, groups=plot_data.reset_index()['patient_id'])
result = model.fit()
print(result.summary())

fig, axes = plt.subplots(ncols=2, figsize=(6, 3), sharey=True)

offset = 1.5
idx = 0
for _wgd_class, data in plot_data.groupby('wgd_class'):
    ax = axes[idx]
    idx += 1
    print(_wgd_class)
    model = smf.mixedlm(f"log_sting ~ log_mn_rate", data, groups=data.reset_index()['patient_id'])
    result = model.fit()
    print(result.summary())
    p = result.pvalues['log_mn_rate']
    coef = result.params['log_mn_rate']
    sns.regplot(ax=ax, x='mn_rate', y='sting', data=data, scatter=False, line_kws=dict(color="k", linestyle=':', linewidth=1))
    sns.scatterplot(
        ax=ax, x='mn_rate', y='sting', hue='wgd_class', palette=colors_dict['wgd_class'],
        s=20, linewidth=0, edgecolor='k', data=data)
    ax.text(.2, .9, f'{_wgd_class}: p = {p:.2g}', transform=ax.transAxes, color='k')
    offset += 1
    ax.get_legend().remove()

sns.despine(trim=True)

```

```python

plot_data = (sample_mn_rates.merge(fraction_wgd))

plot_data['log_mn_rate'] = np.log(plot_data['mn_rate'])
plot_data['log_sting'] = np.log(plot_data['sting'])
plot_data['is_nwgd'] = (plot_data['wgd_class'] == 'WGD-low') * 1

plot_data = plot_data[plot_data['is_nwgd'] == 0]

fig, axes = plt.subplots(ncols=4, nrows=6, figsize=(8, 12), sharey=True)

idx = 0
for patient_id, data in plot_data.groupby('patient_id', observed=True):
    ax = axes.flatten()[idx]
    idx += 1
    sns.scatterplot(
        ax=ax, x='mn_rate', y='sting', color='0.75',
        s=20, linewidth=0, data=plot_data)
    sns.scatterplot(
        ax=ax, x='mn_rate', y='sting', color='r',
        s=20, linewidth=0, data=data)
    ax.set_title(patient_id)

plt.tight_layout()
sns.despine(trim=True)

```

```python

import statsmodels.api as sm
import statsmodels.formula.api as smf

plot_data = event_rates.melt(id_vars=['patient_id', 'sample_id', 'n_cells'], var_name='class', value_name='event_rate')
plot_data = plot_data.merge(sample_mn_rates, on=['patient_id', 'sample_id'])
plot_data = plot_data.merge(fraction_wgd)

plot_data = plot_data[plot_data['n_cells'] >= 20]
print(plot_data['patient_id'].unique().shape)

plot_data['log_mn_rate'] = np.log(plot_data['mn_rate'])
plot_data['is_nwgd'] = (plot_data['wgd_class'] == 'WGD-low') * 1

model = smf.mixedlm(f"log_mn_rate ~ event_rate * is_nwgd", plot_data, groups=plot_data.reset_index()['patient_id'])
result = model.fit()
print(result.summary())

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
g.fig.set_figwidth(8)
g.fig.set_figheight(2.5)

def annotate(data, **kws):
    offset = 1.5
    model = smf.mixedlm(f"log_mn_rate ~ event_rate", data, groups=data.reset_index()['patient_id'])
    result = model.fit()
    print(data['class'].iloc[0])
    print(result.summary())
    p = result.pvalues['event_rate']
    coef = result.params['event_rate']
    ax = plt.gca()
    sns.regplot(ax=ax, x='event_rate', y='mn_rate', data=data, scatter=False, line_kws=dict(color="k", linestyle=':', linewidth=1))
    sns.scatterplot(
        ax=ax, x='event_rate', y='mn_rate', hue='wgd_class', palette=colors_dict['wgd_class'],
        s=25, linewidth=0, edgecolor='k', data=data)
    ax.text(.35, .9 - offset * .1, f'p = {p:.2g}',
            transform=ax.transAxes, color='k')

g.map_dataframe(annotate)

g.set_titles(template='{col_name}')

g.axes[0, 0].set_ylabel('MN rate')
for ax in g.axes[0, :]:
    ax.set_xlabel('Count / cell')

plt.subplots_adjust(wspace=0.1)

g.fig.savefig('../../../../figures/figure3/mn_rate_misseg.svg', bbox_inches='tight', metadata={'Date': None})

```


```python

import statsmodels.api as sm
import statsmodels.formula.api as smf

plot_data = event_rates.melt(id_vars=['patient_id', 'sample_id', 'n_cells'], var_name='class', value_name='event_rate')
plot_data = plot_data.merge(sample_mn_rates)
plot_data = plot_data.merge(fraction_wgd)
plot_data['log_mn_rate'] = np.log(plot_data['mn_rate'])

plot_data = plot_data[plot_data['n_cells'] >= 20]
print(plot_data['patient_id'].unique().shape)

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
    all_data = data
    offset = 1.5
    for _wgd_class, data in all_data.groupby('wgd_class'):
        model = smf.mixedlm(f"log_mn_rate ~ event_rate", data, groups=data.reset_index()['patient_id'])
        result = model.fit()
        p = result.pvalues['event_rate']
        coef = result.params['event_rate']
        ax = plt.gca()
        sns.regplot(ax=ax, x='event_rate', y='mn_rate', data=data, scatter=False, line_kws=dict(color="k", linestyle=':', linewidth=1))
        sns.scatterplot(
            ax=ax, x='event_rate', y='mn_rate', hue='wgd_class', palette=colors_dict['wgd_class'],
            s=40, linewidth=0, edgecolor='k', data=data)
        ax.text(.55, .9 - offset * .1, f'p = {p:.2g}',
                transform=ax.transAxes, color=colors_dict['wgd_class'][_wgd_class])
        offset += 1

g.map_dataframe(annotate)

g.set_titles(template='{col_name}')

g.axes[0, 0].set_ylabel('MN rate')
for ax in g.axes[0, :]:
    ax.set_xlabel('Count / cell')

```

```python

nullisomy_rates = pd.read_csv('../../../../annotations/nullisomy_rates.csv')[['patient_id', 'nullisomy_arm_rate']]

cell_info = pd.read_csv(f'{project_dir}/preprocessing/summary/filtered_cell_table.csv.gz')
cell_info = cell_info[cell_info['include_cell']]
fraction_multipolar = cell_info.groupby(['patient_id']).agg(fraction_multipolar=('multipolar', 'mean')).reset_index()

plot_data = event_rates.merge(sample_mn_rates).merge(fraction_multipolar).merge(nullisomy_rates).merge(fraction_wgd)
plot_data['log_mn_rate'] = np.log(plot_data['mn_rate'])

plot_data = plot_data.query('wgd_class == "WGD-low"')

plot_data = plot_data.rename(columns=event_type_names)
plot_data = plot_data.rename(columns={
    'log_mn_rate': 'MN rate',
    'fraction_multipolar': 'Frac. divergent',
    'nullisomy_arm_rate': 'Arm. null rate',
    'sting': 'STING1',
    'cgas_mean': 'cGAS',
    'tp53_mean': 'TP53',
    'pn_count': 'Num. PN',
    'fraction_wgd': 'Frac. WGD',
})

plot_data = plot_data[plot_data['n_cells'] >= 20]

plot_data = plot_data[misseg_cols + [
    'MN rate',
    'Frac. divergent',
    'Arm. null rate',
    'STING1',
    'cGAS',
    'TP53',
    'Num. PN',
    'Frac. WGD',
]].corr(method='spearman')

g = sns.clustermap(plot_data, vmin=0, vmax=1, cmap='Blues')
g.fig.set_figwidth(3.5)
g.fig.set_figheight(3.5)

```

```python

```
