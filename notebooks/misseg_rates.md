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
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import scipy.stats
from statannotations.Annotator import Annotator
import vetica.mpl

import spectrumanalysis.wgd
import spectrumanalysis.stats


project_dir = os.environ['SPECTRUM_PROJECT_DIR']

normalized = True

colors_dict = yaml.safe_load(open('../../../../config/colors.yaml', 'r'))

event_rates_filename = f'{project_dir}/postprocessing/sankoff_ar/sankoff_ar_rates.tsv'
all_event_rates = pd.read_csv(event_rates_filename, sep='\t')

sigs = pd.read_table('../../../../annotations/mutational_signatures.tsv')
all_event_rates = all_event_rates.merge(sigs)

wgd_class = pd.read_csv('../../../../annotations/fraction_wgd_class.csv')
all_event_rates = all_event_rates.merge(wgd_class)

all_event_rates.head()

```

```python

event_rates = all_event_rates.query(f'group_level == "patient" & normalized == {normalized}')

```

```python

patient_info = pd.read_csv('../../../../metadata/tables/patients.tsv', sep='\t')
patient_info = patient_info.merge(event_rates[['patient_id']].drop_duplicates())

```

```python

cell_info = pd.read_csv(f'{project_dir}/preprocessing/summary/filtered_cell_table.csv.gz')
cell_info = cell_info[(cell_info['include_cell'] == True)]
cell_info = spectrumanalysis.wgd.classify_subclonal_wgd(cell_info)

sample_info = pd.read_csv('../../../../metadata/tables/sequencing_scdna.tsv', sep='\t')

# Restrict to patients with >= 100 cells
cell_counts = cell_info.groupby('patient_id').size()
patient_ids = cell_counts[cell_counts >= 100].index

# Calculate multipolar rate
multipolar = cell_info[cell_info['multipolar'] == True].groupby('patient_id').size().rename('n_multipolar').to_frame()
multipolar['rate_multipolar'] = multipolar['n_multipolar'] / cell_info.groupby('patient_id').size().loc[patient_ids]
multipolar = multipolar.reset_index()

event_rates = event_rates[event_rates['patient_id'].isin(patient_ids)]
event_rates.head()

```

```python

def plot_rates(events, event_types, order=None, rate_y_lim=None):
    plot_data = events.copy()
    plot_data.index = plot_data.index.str.replace('SPECTRUM-', '')

    if order is not None:
        plot_data = plot_data.loc[order]
    else:
        plot_data = plot_data.sort_values(by='all')
        order = list(plot_data.index)
    plot_data['iindex'] = np.arange(len(plot_data))

    fig, axes = plt.subplots(
        nrows=3+len(event_types), ncols=1, height_ratios=[0.1, 0.1] + [0.4] * len(event_types) + [0.3],
        figsize=(8, 5), dpi=300, sharex=True)

    ax = axes[0]
    sns.barplot(
        ax=ax, x='patient_id', y='ones', hue='consensus_signature',
        dodge=False, data=plot_data[['consensus_signature']].assign(ones=1).reset_index(), width=1, order=order,
        palette=colors_dict['consensus_signature'])
    legend = ax.legend(loc='upper left', bbox_to_anchor=(1, 2), frameon=False, fontsize=4)
    legend.set_title('Signature', prop={'size': 6})
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

    ax = axes[1]
    sns.barplot(
        ax=ax, x='patient_id', y='ones', hue='wgd_class', palette=colors_dict['wgd_class'],
        dodge=False, data=plot_data[['wgd_class']].assign(ones=1).reset_index(), width=1, order=order)
    legend = ax.legend(loc='upper left', bbox_to_anchor=(1, -1), frameon=False, fontsize=4)
    legend.set_title('WGD Class', prop={'size': 6})
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

    ax_idx = 2
    for event_type in event_types:
        ax = axes[ax_idx]
        sns.barplot(data=plot_data.reset_index(), x='patient_id', y=event_type, facecolor=colors_dict['misseg_event'].get(event_type, '0.75'), ax=ax, order=order)
        ax.set_ylabel(event_type, fontsize=8, rotation=0, labelpad=4, ha='right', va='center')
        ax.get_yaxis().set_label_coords(-0.05,0.5)
        ax.tick_params(axis='y', which='major', labelsize=8)
        sns.despine(ax=ax)
        ax_idx += 1

    ax =  axes[ax_idx]
    sns.barplot(data=plot_data.reset_index(), x='patient_id', y='n_cells', color='0.75', dodge=False, ax=ax, order=order)
    ax.set_xlabel('')
    ax.set_xticklabels([a[-3:] for a in order], rotation=90)
    ax.set_ylabel('Num. cells', fontsize=8, rotation=0, labelpad=4, ha='right', va='center')
    ax.get_yaxis().set_label_coords(-0.05,0.5)
    ax.tick_params(axis='y', which='major', labelsize=8)
    ax.set_yscale('log')
    sns.despine(ax=ax)

    ax = axes[ax_idx]
    ax.set_xticklabels(
        order,
        size=8,
        rotation=60,
        ha='right',  
        rotation_mode='anchor')
    ax.set_xlabel('Patient')
    
    plt.subplots_adjust(hspace=0.3)

    return plot_data

```

```python

event_type_names = {
    'chromosome_gain': 'Chrom. gain',
    'arm_gain': 'Arm gain',
    'segment_gain': 'Segment gain',
    'chromosome_loss': 'Chrom. loss',
    'arm_loss': 'Arm loss',
    'segment_loss': 'Segment loss',
}

event_rates = event_rates.rename(columns=event_type_names)

if 'Segment gain' in event_rates:
    event_types = [
        'Chrom. gain',
        'Arm gain',
        'Segment gain',
        'Chrom. loss',
        'Arm loss',
        'Segment loss',
    ]
else:
    event_types = [
        'Chrom. gain',
        'Chrom. loss',
        'Arm gain',
        'Arm loss',
    ]

event_rates['all'] = event_rates[event_types].sum(axis=1)
event_rates = event_rates.merge(multipolar[['patient_id', 'rate_multipolar']].rename(columns={'rate_multipolar': 'Multipolar'}))

plot_data = plot_rates(event_rates.set_index('patient_id'), event_types + ['Multipolar'], rate_y_lim=None)

```

```python

plot_data2 = event_rates.melt(
    value_vars=event_types, var_name='kind', value_name='rate',
    id_vars=['patient_id',  'wgd_class', 'consensus_signature', 'n_cells'])

sns.lmplot(
    x='n_cells', y='rate', col='kind', data=plot_data2.query('n_cells > 100'),
    sharex=False, sharey=False, col_wrap=3,
    height=3, aspect=1)

```

```python

plot_data2[(plot_data2['kind'] == 'Chrom. gain')].sort_values('rate', ascending=False).head()

```

# Misseg fraction wgd correlations


```python

wgd_fraction = cell_info.groupby('patient_id')['subclonal_wgd'].mean().rename('fraction_wgd').reset_index()

plot_data2 = event_rates.melt(
    value_vars=event_types, var_name='kind', value_name='rate',
    id_vars=['patient_id',  'wgd_class', 'consensus_signature', 'n_cells'])

plot_data2 = plot_data2.merge(wgd_fraction)

plot_data2 = plot_data2.query('patient_id != "SPECTRUM-OV-081"')

g = sns.lmplot(
    x='rate', y='fraction_wgd', col='kind', hue='wgd_class', data=plot_data2.query('n_cells > 100'),
    sharex=False, sharey=False, col_wrap=3,
    height=3, aspect=1)

def annotate(data, **kws):
    offset = 0
    if data.iloc[0]['wgd_class'] == 'WGD-high':
        offset = 2.
    r, p = scipy.stats.spearmanr(data['rate'], data['fraction_wgd'])
    ax = plt.gca()
    ax.text(.5, .8 - offset * .1, 'r={:.2f}, p={:.2g}'.format(r, p),
            transform=ax.transAxes, color=kws['color'])

g.map_dataframe(annotate)

```

```python

sample_wgd_event_rates = all_event_rates.query(f'group_level == "sample_wgd" & normalized == {normalized}')
sample_wgd_event_rates = sample_wgd_event_rates.rename(columns=event_type_names)

sample_wgd_fraction = cell_info.groupby(['patient_id', 'sample_id'])['subclonal_wgd'].mean().rename('fraction_wgd').reset_index()

plot_data = sample_wgd_event_rates.query('wgd_class == "WGD-low"').query('subclonal_wgd == False').melt(
    value_vars=event_types, var_name='kind', value_name='rate',
    id_vars=['patient_id', 'sample_id', 'wgd_class', 'consensus_signature', 'n_cells'])

plot_data = plot_data.merge(sample_wgd_fraction)

plot_data = plot_data.query('patient_id != "SPECTRUM-OV-081"')

g = sns.lmplot(
    x='rate', y='fraction_wgd', col='kind', hue='wgd_class', data=plot_data.query('n_cells > 50'),
    sharex=False, sharey=False, col_wrap=3,
    height=3, aspect=1)

def annotate(data, **kws):
    offset = 0
    if data.iloc[0]['wgd_class'] == 'WGD-high':
        offset = 2.
    r, p = scipy.stats.spearmanr(data['rate'], data['fraction_wgd'])
    ax = plt.gca()
    ax.text(.5, .8 - offset * .1, 'r={:.2f}, p={:.2g}'.format(r, p),
            transform=ax.transAxes, color=kws['color'])

g.map_dataframe(annotate)

```

# Misseg ISG correlation


```python

isg_score = pd.read_csv(f'{project_dir}/analyses/scrna/isg_score.csv.gz')
isg_score = isg_score.groupby('patient_id')['score'].mean().rename('mean_isg_score').reset_index()

plot_data2 = event_rates.melt(
    value_vars=event_types, var_name='kind', value_name='rate',
    id_vars=['patient_id',  'wgd_class', 'consensus_signature', 'n_cells'])
plot_data2 = plot_data2.merge(isg_score)

g = sns.lmplot(
    x='rate', y='mean_isg_score', col='kind', hue='wgd_class', data=plot_data2.query('n_cells > 100'),
    sharex=False, sharey=False, col_wrap=3,
    height=3, aspect=1)

def annotate(data, **kws):
    offset = 0
    if data.iloc[0]['wgd_class'] == 'WGD-high':
        offset = 2.
    r, p = scipy.stats.spearmanr(data['rate'], data['mean_isg_score'])
    ax = plt.gca()
    ax.text(.5, .8 - offset * .1, 'r={:.2f}, p={:.2g}'.format(r, p),
            transform=ax.transAxes, color=kws['color'])

g.map_dataframe(annotate)

```

```python

scrna = pd.read_csv(f'{project_dir}/analyses//scrna/merged_all_cells.csv.gz')
scrna = scrna[scrna['wgd_class'] != 'Unknown']
scrna = scrna[scrna['cell_type'] == 'Ovarian.cancer.cell']
scrna['is_g1'] = (scrna['Phase'] == 0)

g1_fraction = scrna.groupby(['patient_id', 'wgd_class'])['is_g1'].mean().rename('g1_fraction').reset_index()

plot_data3 = isg_score.merge(g1_fraction)
plot_data3

g = sns.lmplot(
    x='g1_fraction', y='mean_isg_score', col='wgd_class', data=plot_data3,
    sharex=False, sharey=False,
    height=3, aspect=1)

def annotate(data, **kws):
    offset = 0
    r, p = scipy.stats.spearmanr(data['g1_fraction'], data['mean_isg_score'])
    ax = plt.gca()
    ax.text(.5, .8 - offset * .1, 'r={:.2f}, p={:.2g}'.format(r, p),
            transform=ax.transAxes, color=kws['color'])

g.map_dataframe(annotate)

```


# Patient specific rates


```python

plot_data = event_rates.melt(
    value_vars=event_types, var_name='kind', value_name='rate',
    id_vars=['patient_id',  'wgd_class', 'consensus_signature'])
plot_data

```

```python

import spectrumanalysis.statplots

spectrumanalysis.statplots.boxplot1(
    data=plot_data,
    col='kind', y='rate', x='wgd_class',
    order=['WGD-low', 'WGD-high'], col_order=event_types,
    pairs=[('WGD-high', 'WGD-low'),],
    ylabel='Event count per cell',
    palette=colors_dict['wgd_class'],
    yscale='log',
)

```

```python

# Within HRD-Dup and HRD-Del only

spectrumanalysis.statplots.boxplot1(
    data=plot_data[plot_data['consensus_signature'].isin(['HRD-Dup', 'HRD-Del'])],
    col='kind', y='rate', x='wgd_class',
    order=['WGD-low', 'WGD-high'], col_order=event_types,
    pairs=[('WGD-high', 'WGD-low'),],
    ylabel='Event count per cell',
    palette=colors_dict['wgd_class'],
    yscale='log',
)

```

```python

signature_labels = ['HRD-Dup', 'HRD-Del', 'FBI', 'TD']

pairs = []
for a in signature_labels[:3]:
    for b in signature_labels[1:3]:
        if a <= b:
            continue
        pairs.append((a, b))

spectrumanalysis.statplots.boxplot1(
    data=plot_data,
    col='kind', y='rate', x='consensus_signature',
    order=signature_labels, col_order=event_types,
    pairs=pairs,
    ylabel='Event count per cell',
    palette=colors_dict['consensus_signature'],
    yscale='log',
)

```

```python

spectrumanalysis.statplots.boxplot1(
    data=plot_data.query('wgd_class == "WGD-high"'),
    col='kind', y='rate', x='consensus_signature',
    order=signature_labels, col_order=event_types,
    pairs=pairs,
    ylabel='Event count per cell',
    palette=colors_dict['consensus_signature'],
    yscale='log',
)

```

```python

import pandas as pd
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
patient_info['patient_age_at_diagnosis_standardized'] = scaler.fit_transform(
    patient_info['patient_age_at_diagnosis'].values[:, np.newaxis])

ols_df = []

for event_type in event_types:
    df_encoded = pd.get_dummies(
        event_rates.set_index('patient_id')[['wgd_class', 'consensus_signature']],
        columns=['wgd_class', 'consensus_signature'], drop_first=False) * 1
    df_encoded = df_encoded[['wgd_class_WGD-high', 'consensus_signature_FBI']]
    df_encoded['Age'] = patient_info.set_index('patient_id')['patient_age_at_diagnosis_standardized']

    X = df_encoded
    X = sm.add_constant(X)

    y = event_rates[event_type].values

    model = sm.OLS(y, X)
    results = model.fit()
    
    err_series = results.params - results.conf_int()[0]

    coef_df = pd.DataFrame({'coef': results.params.values,
                            'err': err_series.values,
                            'varname': err_series.index.values,
                            'pvalue': results.pvalues,
                           })
    coef_df['varname'] = coef_df['varname'].str.replace('.*_', '', regex=True)
    ols_df.append(coef_df.assign(event_type=event_type))

ols_df = pd.concat(ols_df, ignore_index=True)
ols_df = ols_df[ols_df['varname'] != 'const']
ols_df.head()

```

```python

fig, axes = plt.subplots(nrows=1, ncols=len(ols_df['event_type'].unique()), figsize=(10, .75), dpi=150)

for idx, (event_type, coef_df) in enumerate(ols_df.groupby('event_type')):
    ax = axes[idx]
    coef_df[coef_df['event_type'] == event_type].plot(x='varname', y='coef', kind='barh', 
                 ax=ax, color='none',
                 xerr='err', legend=False)
    ax.set_ylabel('')
    ax.set_xlabel('')
    ax.scatter(y=np.arange(coef_df.shape[0]), 
               marker='s', s=10, 
               x=coef_df['coef'], color='black')
    ax.axvline(x=0, linestyle='--', color='black', linewidth=1)
    ax.set_xlabel('coefficient')
    ax.set_title(event_type)
    if idx != 0:
        ax.set_yticklabels([])
    sns.despine()

ols_df

```

# Sample specific rates


```python

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def summarize_features(X):
    scaler = StandardScaler()
    X_standardized = scaler.fit_transform(X)
    
    pca = PCA(n_components=1)
    X_pca = pca.fit_transform(X_standardized)

    correlation = np.array([np.corrcoef(x_col, X_pca[:, 0])[0, 1] for x_col in X_standardized.T])

    if np.mean(correlation) < 0:
        X_pca = -X_pca

    return X_pca[:, 0]

```

```python

sample_event_rates = all_event_rates.query(f'group_level == "sample" & normalized == {normalized}')

# Alternatively only focus on dominant wgd class
sample_event_rates = all_event_rates.query(f'group_level == "sample_wgd" & normalized == {normalized} & subclonal_wgd == False')

sample_event_rates = sample_event_rates.merge(wgd_class)
sample_event_rates = sample_event_rates.merge(sigs)
sample_event_rates = sample_event_rates.merge(sample_info[['spectrum_sample_id', 'tumor_megasite']].drop_duplicates(), left_on='sample_id', right_on='spectrum_sample_id', how='left')
sample_event_rates = sample_event_rates.rename(columns=event_type_names)
sample_event_rates = sample_event_rates.query('n_cells >= 100')

sample_event_rates['All'] = np.exp(summarize_features(np.log(1e-8 + sample_event_rates[event_types].values)))

plot_data = sample_event_rates.melt(
    value_vars=event_types, var_name='kind', value_name='rate',
    id_vars=['patient_id', 'sample_id', 'wgd_class', 'tumor_megasite'])

plt.figure(figsize=(4, 1), dpi=150)
g = sns.catplot(
    col='kind', y='rate', hue='tumor_megasite', x='wgd_class',
    data=plot_data, kind='box', sharey=False, dodge=True, fliersize=1,
    col_order=event_types, height=3, aspect=0.4, palette=colors_dict['tumor_megasite'])

for ax in g.axes.flatten():
    ax.set_yscale('log')
    ax.set_xlabel('')
    for label in ax.get_xticklabels():
        label.set_rotation(60)
        label.set_ha('right')
        label.set_rotation_mode('anchor')
g.set_titles('{col_name}')

plt.tight_layout()

```

```python

plt.figure(figsize=(4, 1), dpi=150)
g = sns.catplot(
    col='kind', y='rate', x='tumor_megasite', hue='tumor_megasite',
    data=plot_data, kind='box', sharey=False, dodge=False, fliersize=1,
    col_order=event_types, height=3, aspect=0.4, palette=colors_dict['tumor_megasite'])

g.map_dataframe(
    sns.stripplot, y='rate', x='tumor_megasite', hue='tumor_megasite',
    dodge=False, linewidth=1,
    palette=colors_dict['tumor_megasite'])

for ax in g.axes.flatten():
    ax.set_yscale('log')
    ax.set_xlabel('')
    for label in ax.get_xticklabels():
        label.set_rotation(60)
        label.set_ha('right')
        label.set_rotation_mode('anchor')
g.set_titles('{col_name}')

pairs = [('Adnexa', 'Non-Adnexa')]

for ax, event_type in zip(g.axes.flatten(), event_types):
    annotator = Annotator(
        ax, pairs,
        y='rate', x='tumor_megasite',
        data=plot_data[plot_data['kind'] == event_type], kind='box')
    annotator.configure(test='Mann-Whitney', text_format='star', loc='inside', line_width=1)
    annotator.apply_and_annotate()

plt.tight_layout()

```

```python

plot_data2 = plot_data.groupby(['patient_id', 'wgd_class', 'kind', 'tumor_megasite'])['rate'].mean().unstack().dropna().reset_index()

g = sns.lmplot(
    x='Adnexa', y='Non-Adnexa', col='kind', data=plot_data2,
    sharex=False, sharey=False, col_wrap=3,
    height=3, aspect=1)

```

```python

import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.genmod.cov_struct import Exchangeable, Autoregressive, Independence
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

ols_df = []

for event_type in event_types:
    df_encoded = pd.get_dummies(
        sample_event_rates.set_index('patient_id')[['wgd_class', 'consensus_signature', 'tumor_megasite']],
        columns=['wgd_class', 'consensus_signature', 'tumor_megasite'], drop_first=False) * 1
    df_encoded = df_encoded[['wgd_class_WGD-high', 'consensus_signature_FBI', 'tumor_megasite_Adnexa']]
    df_encoded['Age'] = patient_info.set_index('patient_id')['patient_age_at_diagnosis']

    for col in df_encoded.columns:
        df_encoded[col] = scaler.fit_transform(df_encoded[col].values[:, np.newaxis])

    # Interaction terms
    # df_encoded['WGD Age'] = df_encoded['Age'] * df_encoded['wgd_class_WGD-high']
    # df_encoded['WGD signature'] = df_encoded['Age'] * df_encoded['consensus_signature_FBI']

    X = df_encoded
    X = sm.add_constant(X)

    y = scaler.fit_transform(sample_event_rates[event_type].values[:, np.newaxis])

    # GEE model with patient as group
    model = sm.GEE(y, X, df_encoded.index.values, family=sm.families.Gaussian(), cov_struct=Exchangeable())
    results = model.fit()
    
    err_series = results.params - results.conf_int()[0]

    coef_df = pd.DataFrame({
        'coef': results.params.values,
        'pvalue': results.pvalues,
        'err': err_series.values,
        'varname': err_series.index.values,
    })
    coef_df['varname'] = coef_df['varname'].str.replace('.*_', '', regex=True)
    ols_df.append(coef_df.assign(event_type=event_type))

ols_df = pd.concat(ols_df, ignore_index=True)
ols_df = ols_df[ols_df['varname'] != 'const']
ols_df.head()

```

```python

import statsmodels.stats.multitest
import spectrumanalysis.stats

fdr = ols_df.query('varname == "WGD-high"')
fdr['is_significant_corrected'], fdr['pvalue_corrected'] = statsmodels.stats.multitest.fdrcorrection(
    fdr['pvalue'],
    alpha=0.05, method='poscorr', is_sorted=False)
fdr['significance_corrected'] = fdr['pvalue_corrected'].apply(spectrumanalysis.stats.get_significance_string)
fdr

```

```python

fig, axes = plt.subplots(nrows=1, ncols=len(event_types), figsize=(10, 1.5), dpi=300)

for idx, (event_type, coef_df) in enumerate(ols_df[ols_df['varname'] != 'Group Var'].groupby('event_type')):
    ax = axes[idx]
    coef_df[coef_df['event_type'] == event_type].plot(x='varname', y='coef', kind='barh', 
                 ax=ax, color='none',
                 xerr='err', legend=False)
    ax.set_ylabel('')
    ax.set_xlabel('')
    ax.scatter(y=np.arange(coef_df.shape[0]),
               marker='o', s=10,
               x=coef_df['coef'], color='black')
    ax.axvline(x=0, linestyle='--', color='black', linewidth=1)
    ax.set_xlabel('coefficient')
    ax.set_title(event_type, y=1.35)
    pvalue = fdr.set_index('event_type').loc[event_type, 'pvalue_corrected']
    ax.text(0.5, 1.2, f'p = {pvalue:.1e}', ha='center', va='center', transform=ax.transAxes)
    if idx != 0:
        ax.set_yticklabels([])
    sns.despine()

```


```python

plot_data = fdr.query('event_type != "All"')

plot_data = plot_data.set_index('event_type').loc[[
    'Segment gain', 'Segment loss',
    'Arm gain', 'Arm loss',
    'Chrom. gain', 'Chrom. loss',
]].reset_index()

fig, ax = plt.subplots(figsize=(2.25, 1.75), dpi=300)

plot_data.plot(x='event_type', y='coef', kind='barh', 
    ax=ax, color='none',
    xerr='err', legend=False)
ax.set_ylabel('')
ax.scatter(
    y=np.arange(plot_data.shape[0]),
    marker='o', s=10, 
    x=plot_data['coef'], color='black')
ax.axvline(x=0, linestyle=':', color='black', linewidth=1)
ax.set_xlabel('Regression coefficient (95% CI)')
for idx, (i, row) in enumerate(plot_data.iterrows()):
    text = row['significance_corrected']
    y_offset = 0.4
    if '*' in text:
        y_offset= 0.3
    ax.text(row['coef'], idx + y_offset, text, ha='center', va='center')
sns.despine()

fig.savefig(f'../../../../figures/edfigure4/misseg_rates_model.svg', bbox_inches='tight', metadata={'Date': None})

```

# WGD subpopulation specific rates


```python

wgd_event_rates = all_event_rates.query(f'group_level == "wgd" & normalized == {normalized}')

wgd_event_rates = wgd_event_rates.merge(sigs)
wgd_event_rates = wgd_event_rates.merge(wgd_class)
wgd_event_rates = wgd_event_rates[wgd_event_rates['patient_id'].isin(patient_ids)]

wgd_event_rates.head()

```

```python

event_types = [
    'Chrom. gain',
    'Chrom. loss',
    'Arm gain',
    'Arm loss',
    'Segment gain',
    'Segment loss',
]

wgd_event_rates = wgd_event_rates.rename(columns=event_type_names)

wgd_event_rates2 = wgd_event_rates.query('n_cells >= 5').melt(
    value_vars=event_types, var_name='kind', value_name='rate',
    id_vars=['patient_id',  'wgd_class', 'consensus_signature', 'subclonal_wgd', 'majority_n_wgd', 'n_wgd', 'n_cells'])
wgd_event_rates2['n_wgd'] = wgd_event_rates2['n_wgd'].astype(int)

mwu_stats = spectrumanalysis.stats.mwu_tests(wgd_event_rates2, ['wgd_class', 'n_wgd', 'kind'], 'rate')
mwu_stats = mwu_stats[mwu_stats['kind_1'] == mwu_stats['kind_2']]
mwu_stats = spectrumanalysis.stats.fdr_correction(mwu_stats)
mwu_stats['fold_change'] = mwu_stats[['y_mean_1', 'y_mean_2']].max(axis=1) / mwu_stats[['y_mean_1', 'y_mean_2']].min(axis=1)

```

```python

from matplotlib.transforms import blended_transform_factory

event_types_plot_name = 'chromosome'
# event_types_plot_name = 'arm'
# event_types_plot_name = 'segment'

event_types1 = {
    'chromosome': [
        'Chrom. gain',
        'Chrom. loss',
    ],
    'arm': [
        'Arm gain',
        'Arm loss',
    ],
    'segment': [
        'Segment gain',
        'Segment loss',
    ],
}[event_types_plot_name]

plot_data = wgd_event_rates2[wgd_event_rates2['kind'].isin(event_types1)]
plot_data['n_wgd'] = plot_data['n_wgd'].astype(int).astype('category')

g = sns.catplot(
    col='kind', y='rate', x='wgd_class', hue='n_wgd',
    data=plot_data, kind='box', sharey=True, dodge=True, fliersize=1,
    order=['WGD-low', 'WGD-high'], height=2.5, aspect=.6, palette=colors_dict['wgd_multiplicity'])
g.set_titles('{col_name}', y=1.4)
if normalized:
    g.set_ylabels('Event count per cell (normalized)')
else:
    g.set_ylabels('Event count per cell')
sns.move_legend(
    g, 'upper left', prop={'size': 8}, markerscale=0.5, bbox_to_anchor=(1., 0.7),
    # labelspacing=0.4, handletextpad=0.4, columnspacing=0.5,
    ncol=1, title='# WGD', title_fontsize=10, frameon=False)
g.fig.set_dpi(150)
for ax in g.axes.flatten():
    ax.set_xlabel('')
    for label in ax.get_xticklabels():
        label.set_rotation(60)
        label.set_ha('right')
        label.set_rotation_mode('anchor')

pairs = [
    (('WGD-low', 0), ('WGD-low', 1)),
    (('WGD-high', 0), ('WGD-high', 1)),
    (('WGD-high', 1), ('WGD-high', 2)),
    (('WGD-high', 1), ('WGD-low', 1)),
    (('WGD-high', 1), ('WGD-low', 0)),
]

def get_position(wgd_class, n_wgd):
    if wgd_class == 'WGD-high':
        x = 1
    else:
        x = 0
    if n_wgd == 0:
        x -= 0.25
    elif n_wgd == 2:
        x += 0.25
    return x

for ax, event_type in zip(g.axes.flatten(), event_types1):
    y_pos = 1.05
    for wgd_class_1, n_wgd_1, wgd_class_2, n_wgd_2 in mwu_stats[['wgd_class_1', 'n_wgd_1', 'wgd_class_2', 'n_wgd_2']].drop_duplicates().values:
        stat_row = mwu_stats.set_index([
            'wgd_class_1', 'n_wgd_1',
            'wgd_class_2', 'n_wgd_2',
            'kind_1', 'kind_2']).loc[wgd_class_1, n_wgd_1, wgd_class_2, n_wgd_2, event_type, event_type]
        if stat_row['significance_corrected'] == 'ns':
            continue
        x_pos_1 = get_position(wgd_class_1, n_wgd_1)
        x_pos_2 = get_position(wgd_class_2, n_wgd_2)
        spectrumanalysis.stats.add_significance_line(ax, stat_row['significance_corrected'], x_pos_1, x_pos_2, y_pos)
        y_pos += 0.08

plt.subplots_adjust(hspace=0., wspace=0.1)

if not normalized:
    g.fig.savefig(f'../../../../figures/figure3/misseg_rates_{event_types_plot_name}.svg', bbox_inches='tight', metadata={'Date': None})
else:
    g.fig.savefig(f'../../../../figures/edfigure4/misseg_rates_normalized_{event_types_plot_name}.svg', bbox_inches='tight', metadata={'Date': None})

```

```python

mwu_stats[
    (
        (mwu_stats['wgd_class_1'] == 'WGD-high') &
        (mwu_stats['n_wgd_1'] == 1) &
        (mwu_stats['wgd_class_2'] == 'WGD-low') &
        (mwu_stats['n_wgd_2'] == 0)
    ) |
    (
        (mwu_stats['wgd_class_2'] == 'WGD-high') &
        (mwu_stats['n_wgd_2'] == 1) &
        (mwu_stats['wgd_class_1'] == 'WGD-low') &
        (mwu_stats['n_wgd_1'] == 0)
    )
]

```

```python

mwu_stats = spectrumanalysis.stats.mwu_tests(wgd_event_rates2, ['wgd_class', 'n_wgd', 'kind'], 'rate')
mwu_stats = spectrumanalysis.stats.fdr_correction(mwu_stats)
mwu_stats['fold_change'] = mwu_stats[['y_mean_1', 'y_mean_2']].max(axis=1) / mwu_stats[['y_mean_1', 'y_mean_2']].min(axis=1)

mwu_stats[
    (mwu_stats['wgd_class_1'] == mwu_stats['wgd_class_2']) &
    (mwu_stats['n_wgd_1'] == mwu_stats['n_wgd_2']) &
    (
        (mwu_stats['kind_1'].str.startswith('Arm') & mwu_stats['kind_2'].str.startswith('Arm')) |
        (mwu_stats['kind_1'].str.startswith('Chrom') & mwu_stats['kind_2'].str.startswith('Chrom'))
    )
]

```

# Correlations across different metrics


```python

scrna_metadata = pd.read_csv('../../../../metadata/tables/sequencing_scrna.tsv', sep='\t')

scrna = pd.read_csv(f'{project_dir}/analyses/scrna/merged_all_cells.csv.gz')
scrna = scrna[scrna['wgd_class'] != 'Unknown']
scrna = scrna[scrna['cell_type'] == 'Ovarian.cancer.cell']
scrna = scrna[scrna['sort'] != 'CD45P']
scrna['is_g1'] = (scrna['Phase'] == 0)
scrna['spectrum_aliquot_id'] = scrna['sample_id']
scrna = scrna.merge(scrna_metadata[['spectrum_aliquot_id', 'spectrum_sample_id']].drop_duplicates())
scrna['sample_id'] = scrna['spectrum_sample_id']

```

```python

scrna_tcells = pd.read_csv(f'{project_dir}/analyses/scrna/merged_all_cells.csv.gz')
scrna_tcells['spectrum_aliquot_id'] = scrna_tcells['sample_id']
scrna_tcells = scrna_tcells[scrna_tcells['sort'] == 'CD45P']
scrna_tcells = scrna_tcells.merge(scrna_metadata[['spectrum_aliquot_id', 'spectrum_sample_id']].drop_duplicates(), on='spectrum_aliquot_id')
scrna_tcells['sample_id'] = scrna_tcells['spectrum_sample_id']

scrna_tcells.loc[scrna_tcells['cluster_label_sub'].isnull(), 'cluster_label_sub'] = 'non-t'
scrna_tcells = scrna_tcells.groupby(['patient_id', 'sample_id', 'cluster_label_sub']).size().unstack(fill_value=0)
scrna_tcells = (scrna_tcells.T / scrna_tcells.sum(axis=1)).T
scrna_tcells

```

```python

scrna2 = pd.read_csv(f'{project_dir}/analyses/scrna/nfkb_module_scores.txt.gz', sep='\t')
scrna = scrna.merge(scrna2[['cell_id', 'STING', 'ISG']])

```

```python

sample_wgd_event_rates = all_event_rates.query(f'group_level == "sample_wgd" & normalized == {normalized}').query('subclonal_wgd == False')
sample_wgd_event_rates = sample_wgd_event_rates.rename(columns=event_type_names)[['patient_id', 'sample_id', 'n_cells', 'wgd_class'] + event_types]

sample_wgd_fraction = cell_info.groupby(['patient_id', 'sample_id'])['subclonal_wgd'].mean().rename('fraction_subclonal_wgd').reset_index()

sample_multipolar_fraction = cell_info.groupby(['patient_id', 'sample_id'])['multipolar'].mean().rename('fraction_multipolar').reset_index()

g1_fraction = scrna.groupby(['patient_id', 'sample_id'])['is_g1'].mean().rename('g1_fraction').reset_index()

isg = scrna.groupby(['patient_id', 'sample_id'])['ISG'].mean().rename('ISG').reset_index()

sting = scrna.groupby(['patient_id', 'sample_id'])['STING'].mean().rename('STING').reset_index()

cd8cyto = scrna_tcells['CD8.T.cytotoxic'].reset_index()

plot_data = (
    sample_wgd_event_rates
        .merge(sample_wgd_fraction)
        .merge(sample_multipolar_fraction)
        .merge(g1_fraction)
        .merge(isg)
        .merge(sting)
        .merge(cd8cyto))

plot_data = plot_data.rename(columns={
    'fraction_subclonal_wgd': 'Fraction +1 WGD',
    'fraction_multipolar': 'Fraction multipolar',
    'g1_fraction': 'Fraction G1',
})

plot_data = plot_data[plot_data['n_cells'] >= 100]

plot_data2 = plot_data.copy()
plot_data2 = plot_data2[event_types + ['Fraction +1 WGD', 'Fraction multipolar', 'Fraction G1', 'ISG', 'STING', 'CD8.T.cytotoxic']].corr(method='spearman')

g = sns.clustermap(plot_data2, vmin=-1, vmax=1, cmap='RdBu_r')
g.fig.set_figwidth(3)
g.fig.set_figheight(3)
g.fig.set_dpi(150)


plot_data2 = plot_data[plot_data['wgd_class'] == 'WGD-low']
plot_data2 = plot_data2[event_types + ['Fraction +1 WGD', 'Fraction multipolar', 'Fraction G1', 'ISG', 'STING', 'CD8.T.cytotoxic']].corr(method='spearman')

g = sns.clustermap(plot_data2, vmin=-1, vmax=1, cmap='RdBu_r')
g.fig.set_figwidth(3)
g.fig.set_figheight(3)
g.fig.set_dpi(150)
g.fig.suptitle('WGD-low', y=1.1)

plot_data2 = plot_data[plot_data['wgd_class'] == 'WGD-high']
plot_data2 = plot_data2[event_types + ['Fraction +1 WGD', 'Fraction multipolar', 'Fraction G1', 'ISG', 'STING', 'CD8.T.cytotoxic']].corr(method='spearman')

g = sns.clustermap(plot_data2, vmin=-1, vmax=1, cmap='RdBu_r')
g.fig.set_figwidth(3)
g.fig.set_figheight(3)
g.fig.set_dpi(150)
g.fig.suptitle('WGD-high', y=1.1)

```

```python

sns.scatterplot(x='ISG', y='Segment gain', hue='wgd_class', data=plot_data)

```

```python

plot_data2 = plot_data.copy()
plot_data2 = plot_data2[['patient_id', 'sample_id'] + event_types + ['Fraction +1 WGD', 'Fraction multipolar', 'Fraction G1', 'ISG', 'STING']]
plot_data2 = plot_data2.set_index(['patient_id', 'sample_id'])

for col in event_types:
    plot_data2[col] = np.log(plot_data2[col])

col_colors = plot_data.copy()
col_colors['is_wgd'] = 'r'
col_colors.loc[(col_colors['wgd_class'] == 'WGD-high'), 'is_wgd'] = 'b'
col_colors = col_colors.set_index(['patient_id', 'sample_id'])

g = sns.clustermap(plot_data2.T, z_score=0, col_colors=col_colors[['is_wgd']], cmap='RdBu_r')

```

```python

plot_data2 = plot_data[plot_data['wgd_class'] == 'WGD-low']

sns.pairplot(plot_data2[event_types + ['Fraction +1 WGD', 'Fraction multipolar', 'Fraction G1', 'ISG', 'STING']])

```

```python

```

```python

```
