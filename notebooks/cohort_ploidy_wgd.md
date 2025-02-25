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
import yaml
import numpy as np
import pandas as pd
import anndata as ad
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats

# Statsmodels breaks environments
# from statannotations.Annotator import Annotator

import vetica.mpl

import spectrumanalysis.wgd

```

```python

project_dir = os.environ['SPECTRUM_PROJECT_DIR']

cell_info = pd.read_csv(f'{project_dir}/preprocessing/summary/filtered_cell_table.csv.gz')
cell_info = cell_info[cell_info['include_cell']]

cell_info = spectrumanalysis.wgd.classify_subclonal_wgd(cell_info)

colors_dict = yaml.safe_load(open('../../../../config/colors.yaml', 'r'))

```

```python

cell_info['subclonal_wgd'].sum(), cell_info['subclonal_wgd'].mean()

```

```python

n_wgd = cell_info.groupby(['patient_id', 'n_wgd']).size().unstack(fill_value = 0)

n_wgd_populations = pd.DataFrame({
    'n_wgd_populations': (n_wgd > 1).sum(axis=1),
    'n_cells': n_wgd.sum(axis=1),
})

n_wgd_populations.sort_values('n_cells')

```

```python

n_wgd_populations.query('n_cells > 200').shape

```

```python

import pandas as pd
sigs = pd.read_table('../../../../annotations/mutational_signatures.tsv')
sigs = sigs.merge(cell_info[['patient_id']].drop_duplicates())
sigs.head()

```

```python

fraction_wgd = pd.read_csv('../../../../annotations/fraction_wgd_class.csv')
fraction_wgd.head()

```

```python

sample_info = pd.read_csv('../../../../metadata/tables/sequencing_scdna.tsv', sep='\t')
sample_info = sample_info.drop(['sample_id'], axis=1).rename(columns={'spectrum_sample_id': 'sample_id'})
sample_info.head()

```

```python

patient_info = pd.read_csv('../../../../metadata/tables/patients.tsv', sep='\t')

```

```python

cols = ['tumor_megasite', 'tumor_supersite', 'tumor_site', 'tumor_subsite', 'tumor_type']
cell_info = cell_info.merge(sample_info[['sample_id'] + cols], how='left', on='sample_id')
assert not cell_info[cols[0]].isnull().any()
cell_info.head()

```

```python

from matplotlib.ticker import LogLocator, LogFormatter

plot_data = cell_info.groupby(['patient_id', 'majority_n_wgd'])['subclonal_wgd'].mean().rename('fraction_subclonal_wgd').reset_index()

plot_data.loc[plot_data['fraction_subclonal_wgd'] == 0, 'fraction_subclonal_wgd'] = 1e-4

fig, ax = plt.subplots(figsize=(2, 2))

sns.histplot(ax=ax, x='fraction_subclonal_wgd', data=plot_data, log_scale=10, bins=20, color='0.75')
sns.despine(ax=ax, trim=False)
ax.set_xlabel('Fraction subclonal WGD')
ax.spines['left'].set_bounds((0, 10))
ax.set_yticks(np.linspace(0, 10, 6))
ax.set_xlim((1e-4, 1))

# Major ticks at each power of 10
ax.xaxis.set_major_locator(LogLocator(base=10.0, numticks=15))

# Minor ticks at 1, 2, and 5 times each power of 10
ax.xaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(1, 10), numticks=15))

a = ax.get_xticklabels()
a[1] = '0'
ax.set_xticklabels(a)

plot_data['fraction_subclonal_wgd'].describe()

fig.savefig('../../../../figures/edfigure2/fraction_subclonal_wgd.svg', bbox_inches='tight', metadata={'Date': None})

```

```python

# Experimental

plot_data = cell_info.groupby(['patient_id', 'majority_n_wgd'])['subclonal_wgd'].mean().rename('fraction_subclonal_wgd').reset_index()

fig, ax = plt.subplots(figsize=(4, 2))

squash_coeff = 4
squash_fwd = lambda a: np.tanh(squash_coeff * a)
squash_rev = lambda a: np.arctanh(a) / squash_coeff

data_min, data_max = plot_data['fraction_subclonal_wgd'].min(), plot_data['fraction_subclonal_wgd'].max()

# Calculate bin edges in the display space
num_bins = 30
display_bin_edges = np.linspace(squash_fwd(data_min), squash_fwd(data_max), num_bins)

# Convert display bin edges back to the data space
data_bin_edges = squash_rev(display_bin_edges)

# Plotting
ax.hist(plot_data['fraction_subclonal_wgd'], bins=data_bin_edges, color='skyblue', edgecolor='black')
ax.set_xscale('function', functions=(squash_fwd, squash_rev))
ax.set_xlim((-0.02, 0.99))
sns.despine(trim=True)

```

```python

plot_data = cell_info.groupby(['patient_id', 'majority_n_wgd'])['subclonal_wgd'].mean().rename('fraction_subclonal_wgd').reset_index()

fig, ax = plt.subplots(figsize=(2, 2))

sns.boxplot(ax=ax, y='fraction_subclonal_wgd', x='majority_n_wgd', data=plot_data)
plt.yscale('log')
sns.despine(ax=ax, trim=False)

```

```python

plot_data = cell_info.groupby(['patient_id'])['subclonal_wgd'].mean().rename('fraction_subclonal_wgd').reset_index()
plot_data = plot_data[plot_data['fraction_subclonal_wgd'] < 0.4]

plt.figure(figsize=(3, 2))
sns.boxplot(x='consensus_signature', y='fraction_subclonal_wgd', data=plot_data.merge(sigs))

```

```python

plot_data = cell_info.groupby(['patient_id', 'tumor_megasite'])['subclonal_wgd'].mean().rename('fraction_subclonal_wgd').reset_index()
plot_data = plot_data[plot_data['fraction_subclonal_wgd'] < 0.4]

plt.figure(figsize=(2, 2))
sns.boxplot(x='tumor_megasite', y='fraction_subclonal_wgd', data=plot_data)

```

```python

fig = plt.figure(figsize=(1, 3), dpi=300)
ax = plt.gca()
sns.boxplot(
    ax=ax, x='n_wgd', y='Diameter', hue='n_wgd', dodge=False, data=cell_info,
    linewidth=1, palette=colors_dict['wgd_multiplicity'], fliersize=0)
ax.get_legend().remove()
ax.set_xlabel('#WGD')
ax.set_ylabel('Cell Diameter')

pairs = [(0, 1), (1, 2)]

# Uncomment for pvalue annotation
# annotator = Annotator(
#     ax, pairs, x='n_wgd', y='Diameter', dodge=False, data=cell_info)
# annotator.configure(test='Mann-Whitney', text_format='star', loc='inside', line_width=1)
# annotator.apply_and_annotate()

sns.despine(trim=True)

```

```python

import scipy.stats
import spectrumanalysis.stats

plot_data = cell_info.copy()
plot_data['patient_id'] = plot_data['patient_id'].str.replace('SPECTRUM-', '')

plot_data = plot_data.groupby(['patient_id', 'n_wgd']).agg(
    Diameter=('Diameter', 'mean'), n_cells_group=('Diameter', 'size')).reset_index()
plot_data = plot_data[plot_data['n_cells_group'] >= 20]

wgd_01 = plot_data.set_index(['patient_id', 'n_wgd'])['Diameter'].unstack().loc[:, [0, 1]].dropna()
s_01, p_01 = scipy.stats.wilcoxon(wgd_01[0], wgd_01[1], alternative='less')

wgd_12 = plot_data.set_index(['patient_id', 'n_wgd'])['Diameter'].unstack().loc[:, [0, 1]].dropna()
s_12, p_12 = scipy.stats.wilcoxon(wgd_12[0], wgd_12[1], alternative='less')

order = sorted(plot_data['patient_id'].unique())

palette = {a: '#44bb44' for a in order}
palette = {a: 'k' for a in order}

fig = plt.figure(figsize=(1, 3), dpi=300)
ax = plt.gca()
sns.boxplot(ax=ax, x='n_wgd', y='Diameter', hue='n_wgd', palette=colors_dict['wgd_multiplicity'], data=plot_data, fliersize=0)
sns.lineplot(ax=ax, x='n_wgd', y='Diameter', hue='patient_id', palette=palette, data=plot_data, zorder=999, linewidth=0.5, linestyle='--')
sns.scatterplot(ax=ax, x='n_wgd', y='Diameter', hue='n_wgd', palette=colors_dict['wgd_multiplicity'], data=plot_data, s=10, zorder=1000, edgecolor='k')
spectrumanalysis.stats.add_significance_line(ax, spectrumanalysis.stats.get_significance_string(p_01), 0, 1, 1.)
spectrumanalysis.stats.add_significance_line(ax, spectrumanalysis.stats.get_significance_string(p_12), 1, 2, 1.05)
ax.get_legend().remove()
ax.set_xlabel('#WGD')
ax.set_ylabel('Cell Diameter')
sns.despine(trim=True)

fig.savefig('../../../../figures/edfigure2/cell_size_wgd.svg', bbox_inches='tight', metadata={'Date': None})

```

```python

plot_data = cell_info.copy()
plot_data['patient_id'] = plot_data['patient_id'].str.replace('SPECTRUM-', '')

plot_data['n_cells_group'] = plot_data.groupby(['patient_id', 'n_wgd']).transform('size')
plot_data = plot_data[plot_data['n_cells_group'] >= 10]

order = sorted(plot_data['patient_id'].unique())

fig = plt.figure(figsize=(12, 3), dpi=300)
ax = plt.gca()
sns.boxplot(
    ax=ax, x='patient_id', y='Diameter', hue='n_wgd', dodge=True, data=plot_data,
    linewidth=1, palette=colors_dict['wgd_multiplicity'], fliersize=0)
ax.set_xticklabels(
    order, 
    rotation=60, 
    ha='right',  
    rotation_mode='anchor')
ax.set_xlabel('Patient')
ax.set_ylabel('Cell Diameter')
sns.despine(ax=ax)
sns.move_legend(ax, 'upper left', bbox_to_anchor=(0.94, 1.), title='#WGD', frameon=False)

```

```python

exclude_patients = ['SPECTRUM-OV-081', 'SPECTRUM-OV-125']#, 'SPECTRUM-OV-024']

plot_data = cell_info[~cell_info['patient_id'].isin(exclude_patients)].groupby(['patient_id', 'n_wgd', 'subclonal_wgd'])['Diameter'].mean().reset_index()

fig = plt.figure(figsize=(2, 3))
ax = plt.gca()
sns.boxplot(ax=ax, x='n_wgd', y='Diameter', hue='subclonal_wgd', dodge=True, linewidth=1, data=plot_data, palette=colors_dict['wgd_multiplicity'], fliersize=0)
sns.stripplot(ax=ax, x='n_wgd', y='Diameter', hue='subclonal_wgd', dodge=True, linewidth=1, data=plot_data, palette=colors_dict['wgd_multiplicity'], legend=False)
ax.set_xlabel('#WGD')
ax.set_ylabel('Cell Diameter')
sns.despine(trim=True)
sns.move_legend(ax, 'upper left', bbox_to_anchor=(1, 1), frameon=False)

plot_data = cell_info[~cell_info['patient_id'].isin(exclude_patients)].groupby(['patient_id', 'n_wgd', 'subclonal_wgd'])['ploidy'].mean().reset_index()

fig = plt.figure(figsize=(2, 3))
ax = plt.gca()
sns.boxplot(ax=ax, x='n_wgd', y='ploidy', hue='subclonal_wgd', dodge=True, linewidth=1, data=plot_data, palette=colors_dict['wgd_multiplicity'], fliersize=0)
sns.stripplot(ax=ax, x='n_wgd', y='ploidy', hue='subclonal_wgd', dodge=True, linewidth=1, data=plot_data, palette=colors_dict['wgd_multiplicity'], legend=False)
ax.set_xlabel('#WGD')
ax.set_ylabel('Ploidy')
ax.set_ylim(0, ax.get_ylim()[1])
sns.despine(trim=True)
sns.move_legend(ax, 'upper left', bbox_to_anchor=(1, 1), frameon=False)

```

```python

site_col = 'tumor_site'
site_col = 'tumor_megasite'

exclude_patients = ['SPECTRUM-OV-081', 'SPECTRUM-OV-125']

plot_data = cell_info[~cell_info['patient_id'].isin(exclude_patients)].groupby(['patient_id', site_col, 'subclonal_wgd']).size().rename('cell_count').unstack(fill_value=0)
plot_data = (plot_data.T / plot_data.sum(axis=1)).T
plot_data = plot_data[True].rename('fraction_subclonal_wgd').reset_index()

fig, ax = plt.subplots(figsize=(2, 3))
sns.boxplot(ax=ax, x=site_col, y='fraction_subclonal_wgd', color='w', data=plot_data)

pairs = [('Adnexa', 'Non-Adnexa')]

# Uncomment for pvalue annotation
# annotator = Annotator(
#     ax, pairs, x=site_col, y='fraction_subclonal_wgd', data=plot_data)
# annotator.configure(test='Mann-Whitney', text_format='star', loc='inside', line_width=1)
# annotator.apply_and_annotate()

sns.despine(trim=True)

```

```python

def setup_categorical_axis(ax, name, tick=0.5):
    ax.set_ylabel('')
    ax.set_xlabel('')
    ax.set_yticks([])
    ax.tick_params(
        axis='x',
        which='both',
        bottom=False,
        top=False,
        labelbottom=False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.set_yticks([tick])
    ax.set_yticklabels([name], rotation=0)

```

```python

import matplotlib.pyplot as plt
import seaborn as sns

plot_data = cell_info.copy()
plot_data['patient_id'] = plot_data['patient_id'].str.replace('SPECTRUM-', '')

wgd_plot_data = fraction_wgd.copy()
wgd_plot_data['patient_id'] = wgd_plot_data['patient_id'].str.replace('SPECTRUM-', '')

sigs_plot_data = sigs.copy()
sigs_plot_data['patient_id'] = sigs_plot_data['patient_id'].str.replace('SPECTRUM-', '')

fraction_subclonal_wgd = cell_info.groupby('patient_id')['subclonal_wgd'].mean().rename('fraction_subclonal_wgd').reset_index()
fraction_subclonal_wgd['patient_id'] = fraction_subclonal_wgd['patient_id'].str.replace('SPECTRUM-', '')

patient_age = patient_info[['patient_id', 'patient_age_at_diagnosis']].copy()
patient_age['patient_id'] = patient_age['patient_id'].str.replace('SPECTRUM-', '')
patient_age = patient_age.set_index('patient_id')[['patient_age_at_diagnosis']]

hue_order = ['Rare WGD', 'Prevalent WGD']
site_hue_order = ['Right Adnexa', 'Left Adnexa', 'Omentum', 'Bowel', 'Peritoneum', 'Other']

order = cell_info.groupby(['patient_id'])[['is_wgd', 'ploidy']].mean().sort_values(['is_wgd', 'ploidy']).reset_index()['patient_id'].str.replace('SPECTRUM-', '').values

fig, axes = plt.subplots(
    nrows=8, ncols=1, height_ratios=[0.1, 0.1, 0.1, 0.1, 1, 0.2, 0.2, 0.2], figsize=(10, 6), dpi=300, sharex=True)

ax = axes[4]
sns.stripplot(ax=ax, x='patient_id', y='ploidy', hue='n_wgd', dodge=False, jitter=0.25, data=plot_data, order=order, s=2, palette=colors_dict['wgd_multiplicity'], rasterized=True)
legend = ax.legend(loc='lower left', bbox_to_anchor=(1.02, .0), title='#WGD', frameon=False, fontsize=6, markerscale=2.5)
legend.set_title('#WGD', prop={'size': 8})
ax.set_ylim((0, 8))
ax.set_ylabel('Ploidy', rotation=0, labelpad=4, ha='right', va='center')

ax = axes[5]
sns.barplot(
    ax=ax, x='patient_id', y='fraction_wgd', color='0.9', linewidth=0.5, edgecolor='k',
    dodge=False, data=wgd_plot_data, order=order)
ax.set_ylabel('Fraction\n#WGD≥1', rotation=0, labelpad=4, ha='right', va='center')
ax.set_xlabel('')

ax = axes[6]
num_cells_plot_data = plot_data.groupby('patient_id').size().rename('count').reset_index()
plot_min = num_cells_plot_data.query('count > 0')['count'].min() / 2
sns.barplot(
    ax=ax, x='patient_id', y='count', color='0.9', linewidth=0.5, edgecolor='k',
    dodge=False, data=num_cells_plot_data, order=order, bottom=plot_min)
ax.set_ylabel('Num. Cells', rotation=0, labelpad=4, ha='right', va='center')
ax.set_yscale('log')

ax = axes[7]
plot_min = fraction_subclonal_wgd.query('fraction_subclonal_wgd > 0')['fraction_subclonal_wgd'].min() / 2
sns.barplot(
    ax=ax, x='patient_id', y='fraction_subclonal_wgd', color='0.9', linewidth=0.5, edgecolor='k',
    data=fraction_subclonal_wgd, order=order, bottom=plot_min)
sns.despine(ax=ax)
ax.set_yscale('log')
ax.set_ylabel('Fraction\nSubclonal WGD', rotation=0, labelpad=4, ha='right', va='center')
ax.set_xlabel('')

ax = axes[0]
age_values = patient_age.loc[order]['patient_age_at_diagnosis'].values
im = ax.imshow([age_values], aspect='auto', interpolation='none', cmap='bone_r', vmin=30, vmax=100)
axins = ax.inset_axes([1.04, 0.1, 0.02, 1.5])
cbar = plt.colorbar(im, cax=axins)
cbar.outline.set_visible(False)
axins.set_title('Age', size=8)
axins.tick_params(axis='y', labelsize=6)
setup_categorical_axis(ax, 'Age', tick=0.)

ax = axes[1]
sns.barplot(
    ax=ax, x='patient_id', y='ones', hue='consensus_signature', palette=colors_dict['consensus_signature'],
    dodge=False, data=sigs_plot_data[['patient_id', 'consensus_signature']].assign(ones=1).reset_index(), width=1, order=order)
legend = ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), frameon=False, fontsize=6)
legend.set_title('Signature', prop={'size': 8})
setup_categorical_axis(ax, 'Signature')

ax = axes[2]
sns.barplot(
    ax=ax, x='patient_id', y='ones', hue='BRCA_gene_mutation_status', palette=colors_dict['BRCA_gene_mutation_status'],
    dodge=False, data=sigs_plot_data[['patient_id', 'BRCA_gene_mutation_status']].assign(ones=1), width=1, order=order)
legend = ax.legend(loc='upper left', bbox_to_anchor=(1.02, -1.3), frameon=False, fontsize=6)
legend.set_title('BRCA Status', prop={'size': 8})
setup_categorical_axis(ax, 'BRCA Status')

ax = axes[3]
sns.barplot(
    ax=ax, x='patient_id', y='ones', hue='wgd_class', palette=colors_dict['wgd_prevalence'], hue_order=hue_order,
    dodge=False, data=wgd_plot_data.assign(ones=1), width=1, order=order)
legend = ax.legend(loc='upper left', bbox_to_anchor=(1.02, -4.4), frameon=False, fontsize=6)
legend.set_title('WGD class', prop={'size': 8})
setup_categorical_axis(ax, 'WGD class')

ax = axes[7]
ax.set_xticklabels(
    order, 
    rotation=60, 
    ha='right',  
    rotation_mode='anchor')
ax.set_xlabel('Patient')

sns.despine(ax=axes[4])
sns.despine(ax=axes[5])
sns.despine(ax=axes[6])
sns.despine(ax=axes[7])
plt.subplots_adjust(hspace=0.2)

fig.savefig('../../../../figures/figure1/cohort_ploidy_wgd.svg', bbox_inches='tight', metadata={'Date': None})

```

```python

plot_data = fraction_wgd.copy()
plot_data = plot_data.merge(sigs[['patient_id', 'consensus_signature', 'BRCA_gene_mutation_status']])
plot_data['BRCA1_Germline'] = (plot_data['BRCA_gene_mutation_status'] == 'gBRCA1').astype(str).astype('category')
plot_data['is_wgd_patient'] = plot_data['wgd_class'] == 'Prevalent WGD'
plot_data = plot_data.groupby(['consensus_signature', 'wgd_class']).size().rename('num_wgd_patients').reset_index()

order = ['FBI', 'HRD-Del', 'HRD-Dup', 'TD']

plt.figure(figsize=(1.5, 2), dpi=150)
ax = plt.gca()
sns.barplot(
    ax=ax, x='consensus_signature', y='num_wgd_patients', hue='wgd_class', dodge=True,
    data=plot_data, palette=colors_dict['wgd_prevalence'], order=order)
ax.get_legend().remove()
# sns.move_legend(
#     plt.gca(), 'upper left',
#     bbox_to_anchor=(0.6, 1.1), ncol=1, title='WGD class', frameon=False)
ax.set_ylabel('# Patients')
ax.set_xlabel('')
ax.set_xticklabels(
    ax.get_xticklabels(), 
    rotation=60, 
    ha='right',  
    rotation_mode='anchor')
sns.despine()

```

```python

plot_data = fraction_wgd.copy()
plot_data = plot_data.merge(sigs[['patient_id', 'consensus_signature', 'BRCA_gene_mutation_status']])
plot_data['BRCA1_Germline'] = (plot_data['BRCA_gene_mutation_status'] == 'gBRCA1').astype(str).astype('category')
plot_data['is_wgd_patient'] = plot_data['wgd_class'] == 'Prevalent WGD'

n_patients = plot_data.groupby(['consensus_signature']).size()

plot_data = plot_data.groupby(['consensus_signature'])['is_wgd_patient'].mean().rename('Prevalent WGD').to_frame()
plot_data['Rare WGD'] = 1. - plot_data['Prevalent WGD']
plot_data

order = ['HRD-Dup', 'HRD-Del', 'FBI', 'TD']

fig = plt.figure(figsize=(1.33, 2), dpi=150)
ax = plt.gca()
plot_data.loc[order].plot.bar(ax=ax, stacked=True, color=colors_dict['wgd_prevalence'])

for i, c in enumerate(order):
    ax.text((i+.5)/len(order), 1., n_patients.loc[c], ha='center', transform=ax.transAxes)

ax.get_legend().remove()
ax.set_ylabel('Fraction patients (SPECTRUM)')
ax.set_xlabel('')
ax.set_xticklabels(
    ax.get_xticklabels(), 
    rotation=60, 
    ha='right',  
    rotation_mode='anchor')
sns.despine(ax=ax)
ax.spines.left.set_bounds((0, 1))

fig.savefig('../../../../figures/edfigure2/fraction_wgd.svg', bbox_inches='tight', metadata={'Date': None})

```

```python

plot_data = fraction_wgd.copy()
plot_data = plot_data.merge(sigs[['patient_id', 'consensus_signature', 'BRCA_gene_mutation_status']])
plot_data = plot_data.groupby(['BRCA_gene_mutation_status', 'wgd_class']).size().rename('num_wgd_patients').reset_index()

plt.figure(figsize=(1.75, 2))
ax = plt.gca()
sns.barplot(
    ax=ax, x='BRCA_gene_mutation_status', y='num_wgd_patients', hue='wgd_class', dodge=True,
    data=plot_data, palette=colors_dict['wgd_prevalence'])#, order=order)
ax.get_legend().remove()
# sns.move_legend(
#     plt.gca(), 'upper left',
#     bbox_to_anchor=(0.2, 1.1), ncol=1, title='BRCA status', frameon=False)
ax.set_ylabel('# Patients')
ax.set_ylim((0, 20))
ax.set_yticks([0, 4, 8, 12, 16, 20])
ax.set_xlabel('')
ax.set_xticklabels(
    ax.get_xticklabels(), 
    rotation=60, 
    ha='right',  
    rotation_mode='anchor')
sns.despine()

```

```python

plot_data = fraction_wgd.copy()
plot_data = plot_data.merge(patient_info[['patient_id', 'patient_age_at_diagnosis']])
plot_data = plot_data.merge(sigs[['patient_id', 'consensus_signature', 'BRCA_gene_mutation_status']])

stat, pvalue = scipy.stats.mannwhitneyu(
    plot_data.loc[plot_data['wgd_class'] == 'Prevalent WGD', 'patient_age_at_diagnosis'],
    plot_data.loc[plot_data['wgd_class'] == 'Rare WGD', 'patient_age_at_diagnosis'],
    alternative='greater',
)

order = ['Rare WGD', 'Prevalent WGD']

fig = plt.figure(figsize=(.5, 2), dpi=150)
ax = plt.gca()
sns.boxplot(
    ax=ax, x='wgd_class', y='patient_age_at_diagnosis', hue='wgd_class', dodge=False,
    data=plot_data, palette=colors_dict['wgd_prevalence'], order=order, fliersize=0, linewidth=1)
ax.set_ylabel('Age at diagnosis (SPECTRUM)')
ax.set_xlabel('')
ax.set_xticklabels(
    ax.get_xticklabels(), 
    rotation=60, 
    ha='right',  
    rotation_mode='anchor')
sns.despine()

import spectrumanalysis.stats
mwu_tests = spectrumanalysis.stats.mwu_tests(plot_data, ['wgd_class'], 'patient_age_at_diagnosis')
spectrumanalysis.stats.add_significance_line(ax, mwu_tests.iloc[0]['significance'], 0, 1, 1.05)

fig.savefig('../../../../figures/edfigure2/wgd_age.svg', bbox_inches='tight', metadata={'Date': None})

```

```python

plot_data = fraction_wgd.copy()
plot_data = plot_data.merge(patient_info[['patient_id', 'patient_age_at_diagnosis']])
plot_data = plot_data.merge(sigs[['patient_id', 'consensus_signature', 'BRCA_gene_mutation_status']])

plt.figure(figsize=(1, 2), dpi=150)
ax = plt.gca()
sns.boxplot(
    ax=ax, x='consensus_signature', y='patient_age_at_diagnosis', hue='consensus_signature', dodge=False,
    data=plot_data, palette=colors_dict['consensus_signature'], fliersize=0, linewidth=1)
ax.set_ylabel('Patient Age')
ax.set_xlabel('')
ax.set_xticklabels(
    ax.get_xticklabels(), 
    rotation=60, 
    ha='right',  
    rotation_mode='anchor')
sns.despine()

```

```python

site_data = cell_info[['patient_id', 'sample_id']].drop_duplicates()
site_data = site_data.merge(fraction_wgd)
site_data = site_data.merge(sample_info, left_on=['patient_id', 'sample_id'], right_on=['patient_id', 'sample_id'], how='left')
assert not site_data['tumor_megasite'].isnull().any()

table = site_data.groupby(['wgd_class', 'tumor_megasite']).size().unstack(fill_value=0)
table = (table.T / table.sum(axis=1)).T
table

```

```python

```
