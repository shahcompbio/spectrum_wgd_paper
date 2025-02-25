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
import tqdm
import pandas as pd
import seaborn as sns
import numpy as np
import anndata as ad
import scgenome
import yaml
import matplotlib.pyplot as plt
import vetica.mpl

import spectrumanalysis.wgd
import spectrumanalysis.dataload
import spectrumanalysis.stats

```

```python

project_dir = os.environ['SPECTRUM_PROJECT_DIR']

cell_info = pd.read_csv(f'{project_dir}/preprocessing/summary/filtered_cell_table.csv.gz')
cell_info = cell_info[cell_info['include_cell']]

cell_info = spectrumanalysis.wgd.classify_subclonal_wgd(cell_info)
cell_info['additional_wgd'] = cell_info['n_wgd'] > cell_info['majority_n_wgd']
cell_info['non_multipolar_subclonal_wgd'] = ~cell_info['multipolar'] & cell_info['subclonal_wgd']

```

```python

fraction_multipolar = cell_info.groupby(['patient_id']).agg(
    n_multipolar=('multipolar', 'sum'),
    n_non_multipolar_subclonal_wgd=('non_multipolar_subclonal_wgd', 'sum'),
    n_additional_wgd=('additional_wgd', 'sum'),
    fraction_multipolar=('multipolar', 'mean'),
    fraction_non_multipolar_subclonal_wgd=('non_multipolar_subclonal_wgd', 'mean'),
    fraction_additional_wgd=('additional_wgd', 'mean'),
    n_cells=('multipolar', 'size'),
).reset_index()

```

```python

fraction_multipolar['fraction_multipolar'].describe(), (fraction_multipolar['fraction_multipolar'] > 0).sum()

```

```python

sigs = pd.read_table('../../../../annotations/mutational_signatures.tsv')
sigs = sigs.merge(cell_info[['patient_id']].drop_duplicates())

wgd_class = pd.read_csv('../../../../annotations/fraction_wgd_class.csv')

colors_dict = yaml.safe_load(open('../../../../config/colors.yaml', 'r'))

```
```python

from matplotlib.ticker import LogLocator, LogFormatter

plot_data = fraction_multipolar.copy()
plot_data.loc[plot_data['fraction_multipolar'] == 0, 'fraction_multipolar'] = 1e-4

fig, ax = plt.subplots(figsize=(1.5, 2))

sns.histplot(ax=ax, x='fraction_multipolar', data=plot_data, log_scale=10, bins=20, color='0.75')
sns.despine(ax=ax, trim=False)
ax.set_xlabel('Fraction divergent')
ax.spines['left'].set_bounds((0, 15))
ax.set_yticks(np.linspace(0, 15, 6))
ax.set_xlim((1e-4, 1))

# Major ticks at each power of 10
ax.xaxis.set_major_locator(LogLocator(base=10.0, numticks=15))

# Minor ticks at 1, 2, and 5 times each power of 10
ax.xaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(1, 10), numticks=15))

a = ax.get_xticklabels()
a[1] = '0'
_ = ax.set_xticklabels(a)

```

```python

import scipy.stats

plot_data = cell_info.groupby(['patient_id', 'n_wgd']).agg(
    n_multipolar=('multipolar', 'sum'),
    fraction_multipolar=('multipolar', 'mean'),
    n_cells=('multipolar', 'size'),
).reset_index()

plot_data = plot_data.merge(wgd_class).query('n_cells > 20')

order = ['WGD-low', 'WGD-high']

fig = plt.figure(figsize=(1., 2.), dpi=150)
ax = plt.gca()
sns.boxplot(
    ax=ax, x='wgd_class', y='fraction_multipolar', hue='n_wgd', palette=colors_dict['wgd_multiplicity'],
    order=order, data=plot_data, linewidth=1, fliersize=1)
ax.set_xlim((-0.6, 1.5))
sns.despine()
sns.move_legend(ax, loc='upper left', bbox_to_anchor=(1., 0.95), title='#WGD', ncols=1, frameon=False)

ax.set_xticklabels(
    order,
    rotation=60, 
    ha='right',
    rotation_mode='anchor')
ax.set_xlabel('')
ax.set_ylabel('Fraction divergent cells')

mwu_tests = spectrumanalysis.stats.mwu_tests(plot_data, ['wgd_class', 'n_wgd'], 'fraction_multipolar')
mwu_tests = mwu_tests.query('wgd_class_1 == wgd_class_2')

significance = mwu_tests.loc[
    (mwu_tests['wgd_class_1'] == 'WGD-low') &
    (mwu_tests['wgd_class_2'] == 'WGD-low') &
    (mwu_tests['n_wgd_1'] == 0) &
    (mwu_tests['n_wgd_2'] == 1),
    'significance'].values[0]
spectrumanalysis.stats.add_significance_line(ax, significance, -0.25, 0, 1.05)

significance = mwu_tests.loc[
    (mwu_tests['wgd_class_1'] == 'WGD-high') &
    (mwu_tests['wgd_class_2'] == 'WGD-high') &
    (mwu_tests['n_wgd_1'] == 1) &
    (mwu_tests['n_wgd_2'] == 2),
    'significance'].values[0]
spectrumanalysis.stats.add_significance_line(ax, significance, 1., 1.25, 1.05)


fig.savefig('../../../../figures/edfigure4/multipolar_wgd_class_nwgd.svg', bbox_inches='tight', metadata={'Date': None})

mwu_tests

```

```python

import scipy.stats

plot_data = fraction_multipolar.merge(wgd_class).query('n_cells > 50')

order = ['WGD-low', 'WGD-high']

fig = plt.figure(figsize=(.5, 2.), dpi=150)
ax = plt.gca()
sns.boxplot(
    ax=ax, x='wgd_class', y='fraction_multipolar', hue='wgd_class', palette=colors_dict['wgd_class'],
    order=order, data=plot_data, linewidth=1, fliersize=1)
ax.set_xlim((-0.6, 1.5))
sns.despine()
ax.set_xticklabels(
    order,
    rotation=60, 
    ha='right',
    rotation_mode='anchor')
ax.set_xlabel('')
ax.set_ylabel('Fraction divergent cells')

mwu_tests = spectrumanalysis.stats.mwu_tests(plot_data, ['wgd_class'], 'fraction_multipolar')
spectrumanalysis.stats.add_significance_line(ax, mwu_tests.iloc[0]['significance'], 0, 1, 1.05)

fig.savefig('../../../../figures/figure3/multipolar_wgd_class.svg', bbox_inches='tight', metadata={'Date': None})

plot_data.query('n_cells > 200').sort_values('fraction_multipolar', ascending=False).reset_index(drop=True)

```

## Exploration of multipolar and nullisomy


```python

nullisomy_multipolar_rates = pd.read_csv('../../../../annotations/nullisomy_multipolar_rates.csv')
nullisomy_multipolar_rates = nullisomy_multipolar_rates.merge(wgd_class)

plot_data = nullisomy_multipolar_rates[nullisomy_multipolar_rates['n_cells_group'] >= 10]

mwu_tests = spectrumanalysis.stats.mwu_tests(plot_data, ['wgd_class', 'multipolar'], 'nullisomy_arm_rate')

order = ['WGD-low', 'WGD-high']

palette = {
    'divergent': colors_dict['is_multipolar']['Yes'],
    'non-divergent': colors_dict['is_multipolar']['No'],
}

plot_data['multipolar'] = plot_data['multipolar'].map({True: 'divergent', False: 'non-divergent'})

g = sns.catplot(
    x='wgd_class', hue='multipolar', y='nullisomy_arm_rate', data=plot_data,
    kind='box', fliersize=1, palette=palette, order=order, height=2.5, aspect=.7,
)

ax = g.axes[0, 0]

sig_rare = mwu_tests.query('wgd_class_1 == wgd_class_2 and wgd_class_1 == "WGD-low"').iloc[0]['significance']
spectrumanalysis.stats.add_significance_line(ax, sig_rare, -.2, .2, 1.05)

sig_prevalent = mwu_tests.query('wgd_class_1 == wgd_class_2 and wgd_class_1 == "WGD-high"').iloc[0]['significance']
spectrumanalysis.stats.add_significance_line(ax, sig_rare, 1-.2, 1+.2, 1.05)

g.fig.set_dpi(150)
ax = g.axes[0, 0]
ax.set_yscale('log')
ax.set_ylabel('Arm nullisomy count per cell')
ax.set_xlabel('')
ax.set_xticklabels(
    order,
    rotation=60, 
    ha='right',
    rotation_mode='anchor')

sns.move_legend(g.fig, loc='upper left', bbox_to_anchor=(0.55, 0.5), title='', ncols=1, frameon=False)

g.fig.savefig('../../../../figures/edfigure4/arm_nullisomy.svg', bbox_inches='tight', metadata={'Date': None})

```

```python

plot_data = fraction_multipolar.merge(wgd_class)

plot_data['WGD class'] = plot_data['wgd_class']
plot_data['# cells'] = plot_data['n_cells']

fig, axes = plt.subplots(ncols=3, figsize=(6, 2), dpi=300, sharey=True)

ax = axes[0]
c = 'WGD-low'
sns.regplot(
    x='fraction_additional_wgd', y='fraction_multipolar', scatter=False, line_kws=dict(lw=1),
    data=plot_data.query(f'wgd_class == "{c}"'), color=colors_dict['wgd_class'][c],
    ax=ax)
sns.scatterplot(
    x='fraction_additional_wgd', y='fraction_multipolar', size='# cells',
    data=plot_data.query(f'wgd_class == "{c}"'), color=colors_dict['wgd_class'][c],
    ax=ax)
ax.set_ylim((-0.01, 0.2))
sns.despine(ax=ax, trim=True)
ax.set_xlabel('Fraction additional-WGD')
ax.set_ylabel('Fraction divergent')
ax.get_legend().remove()
ax.set_title('WGD-low')

ax = axes[1]
c = 'WGD-high'
sns.regplot(
    x='fraction_additional_wgd', y='fraction_multipolar', scatter=False, line_kws=dict(lw=1),
    data=plot_data.query(f'wgd_class == "{c}"'), color=colors_dict['wgd_class'][c],
    ax=ax)
sns.scatterplot(
    x='fraction_additional_wgd', y='fraction_multipolar', size='# cells',
    data=plot_data.query(f'wgd_class == "{c}"'), color=colors_dict['wgd_class'][c],
    ax=ax)
ax.set_ylim((-0.01, 0.2))
sns.despine(ax=ax, trim=True)
ax.set_xlabel('Fraction additional-WGD')
ax.set_ylabel('')
ax.get_legend().remove()
ax.set_title('WGD-high')

ax = axes[2]
c = 'WGD-high'
sns.regplot(
    x='fraction_additional_wgd', y='fraction_multipolar', scatter=False, line_kws=dict(lw=1),
    data=plot_data[~plot_data['patient_id'].isin(['SPECTRUM-OV-081', 'SPECTRUM-OV-125'])].query(f'wgd_class == "{c}"'), color=colors_dict['wgd_class'][c],
    ax=ax)
sns.scatterplot(
    x='fraction_additional_wgd', y='fraction_multipolar', size='# cells',
    data=plot_data[~plot_data['patient_id'].isin(['SPECTRUM-OV-081', 'SPECTRUM-OV-125'])].query(f'wgd_class == "{c}"'), color=colors_dict['wgd_class'][c],
    ax=ax)
sns.despine(ax=ax, trim=True)
ax.set_xlabel('Fraction additional-WGD')
ax.set_ylabel('')
sns.move_legend(ax, loc='upper left', bbox_to_anchor=(1, 1.2), frameon=False)
ax.set_title('WGD-high\nwithout 081 and 125')

scipy.stats.pearsonr(plot_data['fraction_non_multipolar_subclonal_wgd'], plot_data['fraction_multipolar'])

```

```python

plot_data = fraction_multipolar.merge(sigs)

import seaborn as sns

from statannotations.Annotator import Annotator

plt.figure()
sns.boxplot(x='consensus_signature', y='fraction_multipolar', data=plot_data)
sns.despine()

pairs = [
    ('FBI', 'HRD-Dup'),
    ('HRD-Del', 'HRD-Dup'),
    ('HRD-Del', 'FBI'),
]
order = ['FBI', 'HRD-Del', 'HRD-Dup', 'TD']

plt.figure(figsize=(3, 3), dpi=150)
ax = plt.gca()
sns.swarmplot(ax=ax, x='consensus_signature', y='fraction_multipolar', hue='n_cells', data=plot_data, order=order)
annotator = Annotator(ax, pairs, data=plot_data, x='consensus_signature', y='fraction_multipolar', order=order)
annotator.configure(test='Mann-Whitney', text_format='star', loc='outside')
annotator.apply_and_annotate()
sns.despine()
plt.gca().legend(loc='upper left', bbox_to_anchor=(1, 1), title='# Cells', frameon=False)

```


## WGD age by fraction multipolar


```python

snv_leaf_table = pd.read_csv('../../../../results/tables/snv_tree/snv_leaf_table.csv')
snv_leaf_table = snv_leaf_table[snv_leaf_table['is_wgd']]
age_since_wgd = snv_leaf_table.groupby(['patient_id'])['snv_count_age_per_gb_since_wgd'].mean().reset_index()
age_since_wgd.head()

```

```python

plot_data = fraction_multipolar.merge(age_since_wgd)

s, p = scipy.stats.spearmanr(
    plot_data['fraction_multipolar'],
    plot_data['snv_count_age_per_gb_since_wgd'])
print(s, p)

s, p = scipy.stats.spearmanr(
    plot_data.query('snv_count_age_per_gb_since_wgd < 150')['fraction_multipolar'],
    plot_data.query('snv_count_age_per_gb_since_wgd < 150')['snv_count_age_per_gb_since_wgd'])
print(s, p)

fig, ax = plt.subplots(figsize=(2, 2))

sns.regplot(
    ax=ax, x='snv_count_age_per_gb_since_wgd', y='fraction_multipolar', data=plot_data, order=2,
    scatter_kws=dict(s=10, color='k'), line_kws=dict(color="k", linestyle=':', linewidth=1))
ax.set_xlabel('C>T CpG Mut / GB\nsince WGD')
ax.set_ylabel('Fraction divergent')
sns.despine(trim=True)

fig, ax = plt.subplots(figsize=(1.5, 2))

sns.regplot(
    ax=ax, x='snv_count_age_per_gb_since_wgd', y='fraction_multipolar',
    data=plot_data.query('snv_count_age_per_gb_since_wgd < 150'), order=1,
    scatter_kws=dict(s=10, color='k'), line_kws=dict(color="k", linestyle=':', linewidth=1))
sns.scatterplot(
    ax=ax, x='snv_count_age_per_gb_since_wgd', y='fraction_multipolar',
    data=plot_data.query('snv_count_age_per_gb_since_wgd > 150'),
    s=10, facecolors='none', edgecolors='k', linewidth=1)
ax.text(50, 0.06, f'p = {p:.2g}')
ax.set_yticks([a for a in ax.get_yticks() if a >= 0])
ax.set_xlabel('C>T CpG Mut / GB\nsince WGD')
ax.set_ylabel('Fraction divergent')
sns.despine(trim=True)

fig.savefig('../../../../figures/edfigure4/multipolar_wgd_age.svg', bbox_inches='tight', metadata={'Date': None})

```

```python

```
