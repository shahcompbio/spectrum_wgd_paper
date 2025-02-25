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

colors_dict = yaml.safe_load(open('../../../../config/colors.yaml', 'r'))

```

```python

project_dir = os.environ['SPECTRUM_PROJECT_DIR']

cell_info = pd.read_csv(f'{project_dir}/preprocessing/summary/filtered_cell_table.csv.gz')
cell_info = cell_info[(cell_info['include_cell'] == True)]

```

```python

plot_data = cell_info.copy()
plot_data['median_ploidy'] = plot_data.groupby('patient_id')['ploidy'].transform('median')
plot_data['ploidy_norm'] = plot_data['ploidy'] / plot_data['median_ploidy']

sns.scatterplot(x='ploidy_norm', y='nnd', data=plot_data, s=1, linewidth=0)

```

```python

from scipy.stats import beta

data = cell_info['nnd'].values

# Fit a log-normal distribution
params = beta.fit(data + 0.0005, floc=0)

fig = plt.figure(figsize=(3, 2.5), dpi=300)
sns.histplot(x='nnd', data=cell_info, stat='density', lw=0, rasterized=True)
plt.plot(np.linspace(0, 0.6, 1000), beta.pdf(np.linspace(0, 0.6, 1000), *params), color='r', ls=':', label='Beta fit', rasterized=True)
plt.legend(markerscale=5, frameon=False, loc=(0.5, 0.8))
plt.yscale('log')
plt.xlabel('NND')
sns.despine()

fig.savefig('../../../../figures/edfigure4/beta_fit.svg', bbox_inches='tight', metadata={'Date': None})

# Generate theoretical quantiles
theoretical_quantiles = np.linspace(0.0001, 1-0.0001, len(data))
gamma_quantiles = beta.ppf(theoretical_quantiles, *params)

# Sort data to get empirical quantiles
sorted_data = np.sort(data)

threshold = beta.ppf(0.99, *params)

fig = plt.figure(figsize=(3, 3), dpi=300)
ax = plt.gca()
plt.plot(gamma_quantiles[sorted_data <= threshold], sorted_data[sorted_data <= threshold], 'o', color='b', markersize=1, label='Non-divergent', rasterized=True)
plt.plot(gamma_quantiles[sorted_data > threshold], sorted_data[sorted_data > threshold], 'o', color='r', markersize=1, label='Divergent', rasterized=True)
plt.plot([min(gamma_quantiles), max(gamma_quantiles)], [min(gamma_quantiles), max(gamma_quantiles)], 'k--', rasterized=True)
plt.xlabel('Theoretical NND Quantiles (Beta)')
plt.ylabel('Empirical NND Quantiles')
plt.title('Q-Q Plot of Data vs. Beta Distribution')
plt.legend(markerscale=5, frameon=False, loc=(0.5, 0.1))
sns.despine(trim=True)

fig.savefig('../../../../figures/figure3/beta_fit_qq.svg', bbox_inches='tight', metadata={'Date': None})

threshold

```

```python

g = sns.FacetGrid(col='patient_id', col_wrap=4, data=cell_info.merge(cell_info[['patient_id', 'cell_id']]), sharex=True, sharey=True)

def plot_nnd_ecdf(data, **kwargs):
    sns.ecdfplot(x='nnd', data=data)
    plt.axvline(threshold, ls=':', color='k')

g.map_dataframe(plot_nnd_ecdf, data=plot_data)

```

```python

g = sns.FacetGrid(col='patient_id', col_wrap=4, data=cell_info.merge(cell_info[['patient_id', 'cell_id', 'ploidy']]), sharex=True, sharey=True)

def plot_nnd_ploidy(data, **kwargs):
    sns.scatterplot(y='nnd', x='ploidy', data=data, s=5, linewidth=0)
    plt.axhline(threshold, ls=':', color='k')

g.map_dataframe(plot_nnd_ploidy, data=plot_data)

```

```python

plot_data = cell_info.merge(cell_info[['patient_id', 'cell_id', 'ploidy', 'n_wgd']])
plot_data['n_wgd'] = plot_data['n_wgd'].astype('category')

g = sns.FacetGrid(
    col='patient_id', col_wrap=5, col_order=['SPECTRUM-OV-004', 'SPECTRUM-OV-081', 'SPECTRUM-OV-045', 'SPECTRUM-OV-036', 'SPECTRUM-OV-024'],
    data=plot_data, sharex=True, sharey=True, height=3)

def plot_nnd_ploidy(data, **kwargs):
    sns.scatterplot(y='nnd', x='ploidy', hue='n_wgd', data=data, s=5, linewidth=0, palette=colors_dict['wgd_multiplicity'])
    plt.axhline(threshold, ls=':', color='k')

g.map_dataframe(plot_nnd_ploidy, data=plot_data)
g.set_titles(template='{col_name}')
g.axes[0].set_ylabel('Fraction genome different\ncompared with most similar cell')
g.set_xlabels('Ploidy')
g.axes[0].set_xlim((0, 6.5))

```

```python

```
