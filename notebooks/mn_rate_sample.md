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
import seaborn as sns
import vetica.mpl

colors_dict = yaml.safe_load(open('../../../config/colors.yaml', 'r'))

microns_per_pixel = 0.1625

project_dir = os.environ['SPECTRUM_PROJECT_DIR']

```

```python

fov_summary = pd.read_csv(f'{project_dir}/if/mn_rate_fov.csv')

fraction_wgd = pd.read_csv('../../../annotations/fraction_wgd_class.csv')
fov_summary = fov_summary.merge(fraction_wgd)

sigs = pd.read_table('../../../annotations/mutational_signatures.tsv')
fov_summary = fov_summary.merge(sigs)

```

```python

# Filter based on slide qc and inclusion
fov_summary = fov_summary[
    (fov_summary['slide_qc'].fillna('').isin(['High', 'Medium'])) &
    (fov_summary['genomic_instability_inclusion_status'] == 'Yes')
]

```

```python

fov_summary['pn_count'].hist(bins=100)

```

```python

sns.displot(x='mn_rate', row='wgd_class', data=fov_summary.query('pn_count > 1000'), log_scale=True, aspect=2, height=2)

```

```python

fov_summary.query('pn_count > 1000').sort_values('mn_rate', ascending=False)

```

```python

plot_data = fov_summary.query('pn_count > 1000').dropna(subset=['wgd_class']).copy()
plot_data['patient_id'] = plot_data['patient_id'].astype(str)
order = plot_data.groupby('patient_id')['mn_rate'].mean().sort_values(ascending=False).index

fig = plt.figure(figsize=(10, 3))
ax = sns.barplot(
    x='patient_id',
    y='mn_rate',
    hue='wgd_class',
    dodge=False,
    order=order,
    data=plot_data,
    width=0.75,
    palette=colors_dict['wgd_class'],
)
sns.despine()
ax.tick_params(axis='x', rotation=90)
# ax.set_xticks([])

```

```python

plot_data = fov_summary.query('pn_count > 1000').dropna(subset=['wgd_class']).copy()
order = plot_data.groupby('image', observed=True)['mn_rate'].mean().sort_values(ascending=False).index

fig = plt.figure(figsize=(10, 3))
ax = sns.boxplot(
    x='image',
    y='mn_rate',
    hue='wgd_class',
    dodge=False,
    order=order,
    data=plot_data,
    palette=colors_dict['wgd_class'],
    fliersize=1,
    linewidth=1,
    whiskerprops=dict(color="0.5"),     # Whisker color
    capprops=dict(color="0.5"),       # Cap color
    medianprops=dict(color="0.5"),    # Median line color
)
sns.despine()
ax.set_xticks([])
ax.set_ylabel('MN rate')
ax.set_xlabel('Image')
sns.move_legend(ax, loc='upper right', bbox_to_anchor=(1, 1), title='WGD status', ncols=1, markerscale=1, frameon=False)

```

```python

plot_data = fov_summary.query('pn_count > 1000').dropna(subset=['wgd_class']).copy()
order = plot_data.groupby('image', observed=True)['mn_rate'].mean().sort_values(ascending=False).index

fig = plt.figure(figsize=(10, 3))
ax = sns.stripplot(
    x='image',
    y='mn_rate',
    hue='wgd_class',
    dodge=False,
    order=order,
    data=plot_data,
    palette=colors_dict['wgd_class'],
    linewidth=0,
    s=1,
)
sns.despine()
ax.set_xticks([])
ax.set_ylabel('MN rate')
ax.set_xlabel('Image')
ax.set_yscale('log')
sns.move_legend(ax, loc='upper right', bbox_to_anchor=(1, 1), title='WGD status', ncols=1, markerscale=1, frameon=False)

```

```python

plot_data = fov_summary.query('pn_count > 1000').dropna(subset=['wgd_class']).copy()
order = plot_data.groupby('image', observed=True)['mn_rate'].mean().sort_values().index

fig = plt.figure(figsize=(10, 3))
ax = sns.barplot(
    x='image',
    y='mn_rate',
    hue='consensus_signature',
    dodge=False,
    order=order,
    data=plot_data,
    width=0.75,
    palette=colors_dict['consensus_signature'],
    errorbar=None,
)
sns.despine()
ax.set_xticks([])

```

```python

plot_data = fov_summary.query('pn_count > 1000').dropna(subset=['wgd_class']).copy()
order = plot_data.groupby('image', observed=True)['mn_rate'].mean().sort_values().index

fig = plt.figure(figsize=(10, 3))
ax = sns.barplot(
    x='image',
    y='mn_rate',
    hue='patient_id',
    dodge=False,
    order=order,
    data=plot_data,
    width=0.75,
    errorbar=None,
    palette='tab20',
)
sns.despine()
ax.set_xticks([])
sns.move_legend(ax, loc='upper left', bbox_to_anchor=(1., 1.), title='Patient ID', markerscale=2, frameon=False)

```

```python

import statsmodels.api as sm
import statsmodels.formula.api as smf

plot_data = fov_summary.query('pn_count > 1000')

plot_data['log_mn_rate'] = np.log(plot_data['mn_rate'])
plot_data['log_sting'] = np.log(plot_data['sting'])

plot_data['is_nwgd'] = (plot_data['wgd_class'] == 'WGD-low') * 1

model = smf.mixedlm(f"log_sting ~ log_mn_rate * is_nwgd", plot_data, groups=plot_data.reset_index()['image'])
result = model.fit()
print(result.summary())
print(result.params)
print(result.pvalues)

fig, axes = plt.subplots(ncols=2, figsize=(6, 3), sharey=True)

idx = 0
for wgd_class, data in plot_data.groupby('wgd_class'):
    ax = axes[idx]
    idx += 1
    print(wgd_class)
    model = smf.mixedlm(f"log_sting ~ log_mn_rate", data, groups=data.reset_index()['image'])
    result = model.fit()
    print(result.summary())
    print(result.params)
    print(result.pvalues)
    p = result.pvalues['log_mn_rate']
    coef = result.params['log_mn_rate']
    sns.regplot(ax=ax, x='mn_rate', y='sting', data=data, scatter=False, line_kws=dict(color="k", linestyle=':', linewidth=1))
    sns.scatterplot(
        ax=ax, x='mn_rate', y='sting', hue='wgd_class', palette=colors_dict['wgd_class'],
        s=20, linewidth=0, edgecolor='k', data=data)
    ax.text(.2, .9, f'{wgd_class}: p = {p:.2g}', transform=ax.transAxes, color='k')
    ax.get_legend().remove()

sns.despine(trim=True)

```

```python

import statsmodels.api as sm
import statsmodels.formula.api as smf

plot_data = fov_summary.query('pn_count > 1000')

plot_data['log_mn_rate'] = np.log(plot_data['mn_rate'])
plot_data['log_sting'] = np.log(plot_data['sting'])

plot_data['is_nwgd'] = (plot_data['wgd_class'] == 'WGD-low') * 1

fig, axes = plt.subplots(nrows=2, figsize=(3, 6), dpi=300, sharex=True, sharey=True)

for c, ax in zip(('WGD-low', 'WGD-high'), axes):
    sns.kdeplot(
        ax=ax, x='mn_rate', y='sting', color='0.5',
        row='wgd_class', kind='kde', data=plot_data.query(f'wgd_class == "{c}"'), linewidths=1, fill=True)
    ax.set_ylabel('STING1')
    ax.set_xlabel('MN rate')
    sns.scatterplot(
        ax=ax, x='mn_rate', y='sting', hue='wgd_class', palette=colors_dict['wgd_class'],
        data=plot_data.query(f'wgd_class == "{c}"'), s=1, linewidth=0)

sns.despine(trim=True)

```

```python

import statsmodels.api as sm
import statsmodels.formula.api as smf

plot_data = fov_summary.query('pn_count > 1000')

plot_data['log_mn_rate'] = np.log(plot_data['mn_rate'])
plot_data['log_sting'] = np.log(plot_data['sting'])

plot_data['is_nwgd'] = (plot_data['wgd_class'] == 'WGD-low') * 1

fig = plt.figure(figsize=(2, 2), dpi=300)
ax = plt.gca()

sns.kdeplot(
    ax=ax, x='mn_rate', y='sting', hue='wgd_class', palette=colors_dict['wgd_class'],
    row='wgd_class', kind='kde', data=plot_data, linewidths=0.5)

sns.scatterplot(
    ax=ax, x='mn_rate', y='sting', alpha=0.5, hue='wgd_class', palette=colors_dict['wgd_class'],
    data=plot_data, s=1, linewidth=0, rasterized=True)

idx = 0
for wgd_class, data in plot_data.groupby('wgd_class'):
    idx += 1
    print(wgd_class)
    model = smf.mixedlm(f"log_sting ~ log_mn_rate", data, groups=data.reset_index()['image'])
    result = model.fit()
    print(result.summary())
    print(result.params)
    print(result.pvalues)
    p = result.pvalues['log_mn_rate']
    coef = result.params['log_mn_rate']
    ax.text(
        .6, .6 - idx * .2, f'{wgd_class}:\ncoef = {coef:.2g}\np = {p:.2g}',
        transform=ax.transAxes, color=colors_dict['wgd_class'][wgd_class], fontsize=6)

ax.set_ylabel('STING1')
ax.set_xlabel('MN rate')
sns.move_legend(ax, 'upper left', bbox_to_anchor=(.25, 1.1), ncol=1, frameon=False, title='WGD class', markerscale=3)

sns.despine(trim=True)

fig.savefig('../../../figures/figure6/mn_rate_sting_wgd.svg', bbox_inches='tight', metadata={'Date': None})

```

```python

import statsmodels.api as sm
import statsmodels.formula.api as smf

plot_data = fov_summary.query('pn_count > 1000')

plot_data['log_mn_rate'] = np.log(plot_data['mn_rate'])
plot_data['log_sting'] = np.log(plot_data['sting'])

plot_data['is_nwgd'] = (plot_data['wgd_class'] == 'WGD-low') * 1

model = smf.mixedlm(f"log_sting ~ log_mn_rate * is_nwgd", plot_data, groups=plot_data.reset_index()['image'])
result = model.fit()
print(result.summary())

fig, axes = plt.subplots(ncols=6, nrows=6, figsize=(20, 20), sharey=True)

idx = 0
for (wgd_class, patient_id), data in sorted(plot_data.groupby(['wgd_class', 'patient_id'])):
    ax = axes.flatten()[idx]
    idx += 1
    sns.regplot(ax=ax, x='mn_rate', y='sting', data=data, scatter=False, line_kws=dict(color="k", linestyle=':', linewidth=1))
    sns.scatterplot(
        ax=ax, x='mn_rate', y='sting', hue='image',
        s=20, linewidth=0, edgecolor='k', data=data)
    ax.set_title(f'{patient_id}\n{wgd_class}')
    ax.get_legend().remove()

    model = smf.mixedlm(f"log_sting ~ log_mn_rate", data, groups=data.reset_index()['image'])
    try:
        result = model.fit()
    except Exception as e:
        print(f'{patient_id}, {e}')
        continue
    p = result.pvalues['log_mn_rate']
    coef = result.params['log_mn_rate']
    ax.text(.2, .9, f'{wgd_class}: p = {p:.2g}', transform=ax.transAxes, color='k')

sns.despine(trim=True)
plt.tight_layout()

```

```python

```

```python

```
