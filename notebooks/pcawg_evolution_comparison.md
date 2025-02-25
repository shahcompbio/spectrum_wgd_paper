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
import yaml
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats
import vetica.mpl


project_dir = os.environ['SPECTRUM_PROJECT_DIR']

colors_dict = yaml.safe_load(open('../../../../config/colors.yaml', 'r'))

data = pd.read_excel('../../../../external/pcawg_evolutionary_history/41586_2019_1907_MOESM13_ESM.xlsx', 'ExtendedDataFigure8b')
data = data.rename(columns={'Unnamed: 0': 'sample'})
data = data[data['tumour_type'] == 'Ovary-AdenoCa']
data

```

```python

chord_data = pd.read_excel('../../../../external/chord/41467_2020_19406_MOESM4_ESM.xlsx', 'CHORD')
chord_data = chord_data[chord_data['group'] == 'PCAWG']

data = data.merge(chord_data, on='sample', how='left')
data.columns

```

```python

wgd_data = pd.read_csv('../../../../external/pcawg_wgd_timing/2018-07-24-wgdMrcaTiming.txt', sep='\t')
wgd_data['sample'] = wgd_data['uuid']

data = data.merge(wgd_data, on=['sample', 'age'], how='left')
data.columns

```

```python

spectrum_data = pd.read_csv('../../../../results/tables/snv_tree/snv_leaf_table.csv')
fraction_wgd = pd.read_csv('../../../../annotations/fraction_wgd_class.csv')

spectrum_data = spectrum_data.merge(fraction_wgd)
spectrum_data['WGD'] = False
spectrum_data.loc[spectrum_data['wgd_class'] == 'Prevalent WGD', 'WGD'] = True

spectrum_data.groupby(['patient_id', 'WGD', 'age'])['snv_count_age_per_gb_since_birth'].mean().sort_values()

```

```python

plot_data = spectrum_data.groupby(['patient_id', 'WGD', 'age'])['snv_count_age_per_gb_since_birth'].mean().reset_index()
plot_data = plot_data.rename(columns={'snv_count_age_per_gb_since_birth': 'CpG.TpG.Gb'})

plot_data = pd.concat([
    plot_data.assign(Cohort='SPECTRUM'),
    data.assign(Cohort='PCAWG')])

fig, axes = plt.subplots(
    nrows=2,
    ncols=2,
    height_ratios=[0.2, 1],
    width_ratios=[1, 0.2],
    figsize=(3, 3),
    dpi=300,
    sharex='col',
    sharey='row')
axes[0, 1].axis('off')

ax = axes[1, 0]
sns.scatterplot(ax=ax, x='age', y='CpG.TpG.Gb', hue='Cohort', data=plot_data, s=10, palette='Set2')
sns.move_legend(ax, 'upper left', bbox_to_anchor=(1, 1.4), ncol=1, title='Cohort', frameon=False, markerscale=1)
ax.set_xlim((0, 100))
sns.despine(ax=ax, trim=True)

ax = axes[0, 0]
sns.boxplot(ax=ax, x='age', y='Cohort', hue='Cohort', data=plot_data, fliersize=1, palette='Set2')
ax.set_ylabel('')
ax.set_yticks([])
ax.tick_params(axis='x', length=0)
ax.spines[:].set_visible(False)

ax = axes[1, 1]
sns.boxplot(ax=ax, x='Cohort', y='CpG.TpG.Gb', hue='Cohort', data=plot_data, fliersize=1, palette='Set2')
ax.set_xlabel('')
ax.set_xticks([])
ax.tick_params(axis='x', which='major', rotation=90)
ax.tick_params(axis='y', length=0)
ax.spines[:].set_visible(False)

stat, pvalue = scipy.stats.mannwhitneyu(
    plot_data.loc[plot_data['Cohort'] == 'SPECTRUM', 'CpG.TpG.Gb'],
    plot_data.loc[plot_data['Cohort'] == 'PCAWG', 'CpG.TpG.Gb'],
    alternative='greater',
)
stat, pvalue

```

```python

plot_data = spectrum_data.groupby(['patient_id', 'age'])['snv_count_age_per_gb_since_birth'].mean().reset_index()
plot_data = plot_data.rename(columns={'snv_count_age_per_gb_since_birth': 'CpG.TpG.Gb'})

plot_data = pd.concat([
    plot_data.assign(Cohort='SPECTRUM'),
    data.assign(Cohort='PCAWG')])

g = sns.JointGrid(space=0.2, ratio=3)

sns.scatterplot(x='age', y='CpG.TpG.Gb', hue='Cohort', data=plot_data, palette='Set2', ax=g.ax_joint)
sns.move_legend(g.ax_joint, 'upper left', bbox_to_anchor=(0.05, 0.95), ncol=1, title='Cohort', frameon=False, markerscale=1.5)

sns.kdeplot(x='age', fill=True, ax=g.ax_marg_x, hue='Cohort', data=plot_data, palette='Set2', clip=(30, 90))
g.ax_marg_x.get_legend().remove()
g.ax_marg_x.get_yaxis().set_visible(False)

sns.kdeplot(y='CpG.TpG.Gb', fill=True, ax=g.ax_marg_y, hue='Cohort', data=plot_data, palette='Set2', clip=(0, 250))
g.ax_marg_y.get_legend().remove()
g.ax_marg_y.get_xaxis().set_visible(False)

sns.despine(trim=True, offset=2)
g.ax_marg_x.spines['left'].set_visible(False)
g.ax_marg_y.spines['bottom'].set_visible(False)

```

```python

plot_data2 = pd.concat([data.assign(cohort='PCAWG'), plot_data.assign(cohort='SPECTRUM')])

sns.boxplot(x='cohort', y='CpG.TpG.Gb', data=plot_data2, fliersize=0)
sns.stripplot(x='cohort', y='CpG.TpG.Gb', hue='cohort', dodge=False, linewidth=1, data=plot_data2)

```

```python

sns.boxplot(x='cohort', y='age', data=plot_data2, fliersize=0)
sns.stripplot(x='cohort', y='age', hue='cohort', dodge=False, linewidth=1, data=plot_data2)

```

```python

fig, ax = plt.subplots(figsize=(4, 4))
ax.set_ylim((0, 150))
ax.set_xlim((0, 110))
sns.regplot(ax=ax, x='age', y='CpG.TpG.Gb', data=data, truncate=False, scatter_kws={'s': 5})
sns.despine()

```

```python

sns.lmplot(x='age', y='CpG.TpG.Gb', data=data, hue='WGD', truncate=False, scatter_kws={'s': 5})

```

```python

palette = {
    'WGD': colors_dict['wgd_prevalence']['Prevalent WGD'],
    'nWGD': colors_dict['wgd_prevalence']['Rare WGD'],
}

plot_data = data.copy()

stat, pvalue = scipy.stats.mannwhitneyu(
    plot_data.loc[plot_data['WGD'], 'age'],
    plot_data.loc[~plot_data['WGD'], 'age'],
    alternative='greater',
)

print(stat, pvalue)

plot_data['wgd_class'] = 'nWGD'
plot_data.loc[plot_data['WGD'], 'wgd_class'] = 'WGD'

fig = plt.figure(figsize=(.75, 2), dpi=150)
ax = plt.gca()
sns.boxplot(ax=ax, x='wgd_class', y='age', hue='wgd_class', data=plot_data, palette=palette, order=['nWGD', 'WGD'], fliersize=4)
sns.despine()
ax.set_ylabel('Age at diagnosis')
ax.set_xlabel('')
ax.set_xticklabels(
    ax.get_xticklabels(), 
    rotation=60, 
    ha='right',  
    rotation_mode='anchor')
ax.text(.05, 1.05, 'p={:.2g}'.format(pvalue), transform=ax.transAxes)

fig.savefig('../../../../figures/edfigure2/pcawg_age_wgd.svg', bbox_inches='tight', metadata={'Date': None})

```

```python

import scipy.stats

palette = {
    'WGD': colors_dict['wgd_prevalence']['Prevalent WGD'],
    'nWGD': colors_dict['wgd_prevalence']['Rare WGD'],
}

plot_data = data.copy()

stat, pvalue = scipy.stats.mannwhitneyu(
    plot_data.loc[plot_data['WGD'], 'CpG.TpG.Gb'],
    plot_data.loc[~plot_data['WGD'], 'CpG.TpG.Gb'],
    alternative='greater',
)

print(stat, pvalue)

plot_data['wgd_class'] = 'nWGD'
plot_data.loc[plot_data['WGD'], 'wgd_class'] = 'WGD'

plt.figure(figsize=(1.5, 3), dpi=150)
ax = plt.gca()
sns.boxplot(ax=ax, x='wgd_class', y='CpG.TpG.Gb', hue='wgd_class', data=plot_data, palette=palette, fliersize=4)
sns.despine()
ax.set_xlabel('')
ax.text(.25, 1.05, 'p={:.2g}'.format(pvalue), transform=ax.transAxes)

```

```python

palette = {
    'HR Proficient': colors_dict['consensus_signature']['FBI'],
    'BRCA1 type': colors_dict['consensus_signature']['HRD-Dup'],
    'BRCA2 type': colors_dict['consensus_signature']['HRD-Del'],
}

plot_data = data[data['hr_status'].notnull()]
plot_data['is_wgd'] = plot_data['WGD'] == True

plot_data['hr_type'] = 'NA'
plot_data.loc[plot_data['hr_status'] == 'HR_proficient', 'hr_type'] = 'HR Proficient'
plot_data.loc[plot_data['hrd_type'] == 'BRCA1_type', 'hr_type'] = 'BRCA1 type'
plot_data.loc[plot_data['hrd_type'] == 'BRCA2_type', 'hr_type'] = 'BRCA2 type'

assert not plot_data['icgc_donor_id'].duplicated().any()

n_patients = plot_data.groupby(['hr_type']).size()

plot_data = plot_data.groupby(['hr_type'])['is_wgd'].mean().rename('Prevalent WGD').to_frame()
plot_data['Rare WGD'] = 1. - plot_data['Prevalent WGD']

order = ['BRCA1 type', 'BRCA2 type', 'HR Proficient']

fig = plt.figure(figsize=(1., 2), dpi=150)
ax = plt.gca()
plot_data.loc[order].plot.bar(ax=ax, stacked=True, color=colors_dict['wgd_prevalence'])

for i, c in enumerate(order):
    ax.text((i+.5)/len(order), 1., n_patients.loc[c], ha='center', transform=ax.transAxes)

ax.get_legend().remove()
ax.set_ylabel('Fraction PCAWG Patients')
ax.set_xlabel('')
ax.set_xticklabels(
    ax.get_xticklabels(), 
    rotation=60, 
    ha='right',  
    rotation_mode='anchor')
sns.despine(ax=ax)
ax.spines.left.set_bounds((0, 1))

fig.savefig('../../../../figures/edfigure2/pcawg_hr_wgd.svg', bbox_inches='tight', metadata={'Date': None})

```

```python

sns.boxplot(hue='WGD', y='age', x='hr_status', data=data)

```

```python

ax = sns.boxplot(hue='WGD', y='age', x='hrd_type', order=['none', 'BRCA1_type', 'BRCA2_type'], data=data)
sns.stripplot(hue='WGD', y='age', x='hrd_type', order=['none', 'BRCA1_type', 'BRCA2_type'], dodge=True, data=data, linewidth=1)
sns.move_legend(ax, 'upper left', bbox_to_anchor=(1, 1))

```

```python

plot_data = data.groupby(['hr_status', 'WGD']).size().rename('count').unstack()#.reset_index()
plot_data.plot.bar(stacked=True)
sns.despine()

```

```python

plot_data = data.groupby(['hrd_type', 'WGD']).size().rename('count').unstack()#.reset_index()
plot_data.plot.bar(stacked=True)
sns.despine()

```
