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
import pickle
import glob
import yaml
import pandas as pd
import anndata as ad
import tqdm
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import vetica.mpl
import statsmodels.api as sm
import statsmodels.formula.api as smf

import spectrumanalysis.stats


project_dir = os.environ['SPECTRUM_PROJECT_DIR']

colors_dict = yaml.safe_load(open('../../../../config/colors.yaml', 'r'))

```

```python

events_filename = f'{project_dir}/postprocessing/sankoff_ar/greedy_events/sankoff_ar_events.tsv'
events = pd.read_csv(events_filename, sep='\t')

events['arm'] = None
events.loc[events['region'] == 'p-arm', 'arm'] = 'p'
events.loc[events['region'] == 'q-arm', 'arm'] = 'q'
events['arm'] = events['chr'] + events['arm']

events.loc[events['region'] == 'p-arm', 'region'] = 'arm'
events.loc[events['region'] == 'q-arm', 'region'] = 'arm'

events['class'] = events['region'] + '_' + events['kind']
events['timing_wgd'] = events['timing_wgd'].fillna('not_wgd')

```

```python

cell_counts = []

import pickle
for patient_id in events['patient_id'].unique():
    tree_filename = f'{project_dir}/postprocessing/sankoff_ar/{patient_id}/sankoff_ar_tree_{patient_id}.pickle'

    with open(tree_filename, 'rb') as f:
        tree = pickle.load(f)

    patient_cell_count = tree.count_terminals()
    root_clade = tree.clade.name
    for clade in tree.find_clades():
        assert clade.name != '128762A-R17-C59'
        if clade.wgd:
            for timing_wgd in ('pre', 'post'):
                cell_counts.append({
                    'patient_id': patient_id,
                    'obs_id': clade.name,
                    'is_wgd': clade.wgd,
                    'n_wgd': clade.n_wgd,
                    'timing_wgd': timing_wgd,
                    'cell_count': clade.count_terminals(),
                    'patient_cell_count': patient_cell_count,
                    'root': clade.name == root_clade,
                    'internal': not clade.is_terminal()})
        else:
            cell_counts.append({
                'patient_id': patient_id,
                'obs_id': clade.name,
                'is_wgd': clade.wgd,
                'n_wgd': clade.n_wgd,
                'timing_wgd': 'not_wgd',
                'cell_count': clade.count_terminals(),
                'patient_cell_count': patient_cell_count,
                'root': clade.name == root_clade,
                'internal': not clade.is_terminal()})

cell_counts = pd.DataFrame(cell_counts)
assert not cell_counts.duplicated(subset=['patient_id', 'obs_id', 'timing_wgd']).any()

cell_counts['cell_fraction'] = cell_counts['cell_count'] / cell_counts['patient_cell_count']

wgd_class = pd.read_csv('../../../../annotations/fraction_wgd_class.csv')
cell_counts = cell_counts.merge(wgd_class)

cell_counts.head()

```


# Post-WGD changes in all clones


```python

medicc_wgd_branches = cell_counts.query('timing_wgd == "post"')[['patient_id', 'obs_id', 'cell_count', 'patient_cell_count', 'cell_fraction']].drop_duplicates()
medicc_wgd_branches = medicc_wgd_branches[(medicc_wgd_branches['cell_count'] >= 20) | (medicc_wgd_branches['cell_fraction'] > 0.5)]

# These patients do not have any WGD sub-clones in medicc trees that are
# supported by either sbmclone blocks or manually identified post-WGD changes.
medicc_wgd_branches = medicc_wgd_branches[~medicc_wgd_branches['patient_id'].isin([
    'SPECTRUM-OV-070',
    'SPECTRUM-OV-006',
    'SPECTRUM-OV-139',
])]

medicc_wgd_branches.sort_values('cell_fraction')

```

```python

wgd_event_cell_counts = cell_counts[['patient_id', 'obs_id', 'cell_count']].drop_duplicates()

wgd_clone_events = events.merge(medicc_wgd_branches[['patient_id', 'obs_id']]).query('timing_wgd == "post"')

for patient_id in ['SPECTRUM-OV-006', 'SPECTRUM-OV-031', 'SPECTRUM-OV-139']:
    wgd_subclone_events = pd.read_csv(f'../../../../results/tables/wgd_subclone_events/wgd_subclone_events_{patient_id}.csv')
    wgd_subclone_events = wgd_subclone_events[wgd_subclone_events['timing_wgd'] == 'post']

    wgd_subclone_events.loc[wgd_subclone_events['region'] == 'p-arm', 'region'] = 'arm'
    wgd_subclone_events.loc[wgd_subclone_events['region'] == 'q-arm', 'region'] = 'arm'
    
    wgd_subclone_events['class'] = wgd_subclone_events['region'] + '_' + wgd_subclone_events['kind']

    wgd_subclone_events['wgd_name'] = patient_id.replace('SPECTRUM-', '') + '/1'
    wgd_subclone_events['n_wgd'] = 1
    wgd_subclone_events['obs_id'] = '1'

    wgd_clone_events = pd.concat([wgd_clone_events, wgd_subclone_events])
    wgd_event_cell_counts = pd.concat([wgd_event_cell_counts, wgd_subclone_events[['patient_id', 'obs_id', 'cell_count']].drop_duplicates()])

# Count events
wgd_clone_events = wgd_clone_events.groupby(['patient_id', 'obs_id', 'n_wgd', 'class']).size().rename('count').unstack(fill_value=0).reset_index()

# Add cell fraction
wgd_clone_events = wgd_clone_events.merge(wgd_event_cell_counts[['patient_id', 'obs_id', 'cell_count']].drop_duplicates())
wgd_clone_events = wgd_clone_events.merge(cell_counts[['patient_id', 'patient_cell_count']].drop_duplicates())
wgd_clone_events['cell_fraction'] = wgd_clone_events['cell_count'] / wgd_clone_events['patient_cell_count']
wgd_clone_events['is_subclonal'] = wgd_clone_events['cell_fraction'] < 0.99

# Generate a unique name
wgd_clone_events['wgd_id'] = (wgd_clone_events.groupby('patient_id').cumcount() + 1).astype(str)
wgd_clone_events['wgd_name'] = wgd_clone_events['patient_id'].str.replace('SPECTRUM-', '') + '/' + wgd_clone_events['wgd_id']

wgd_clone_events.sort_values('cell_fraction')

```

```python

event_type_rename = {
    'chromosome_gain': 'Chrom. gain',
    'arm_gain': 'Arm gain',
    'segment_gain': 'Segment gain',
    'chromosome_loss': 'Chrom. loss',
    'arm_loss': 'Arm loss',
    'segment_loss': 'Segment loss',
}

def setup_categorical_axis(ax):
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

wgd_name_order = wgd_clone_events.sort_values('cell_fraction')['wgd_name'].values

plot_data = wgd_clone_events.rename(columns=event_type_rename)

event_types = [
    'Chrom. gain',
    'Arm gain',
    'Chrom. loss',
    'Arm loss',
]

fig, axes = plt.subplots(
    nrows=4, ncols=1, height_ratios=[0.1, 0.1, 1.2, 0.4], figsize=(9, 3.5), dpi=300, sharex=True, sharey=False)

ax = axes[0]
sns.barplot(
    ax=ax, x='wgd_name', y='ones', hue='n_wgd',
    dodge=False, data=plot_data.assign(ones=1), width=1, order=wgd_name_order, hue_order=[0, 1, 2],
    palette=colors_dict['wgd_multiplicity'])
legend = ax.legend(loc='upper left', bbox_to_anchor=(1, 3), frameon=False, fontsize=6)
legend.set_title('#WGD', prop={'size': 8})
setup_categorical_axis(ax)
ax.set_ylabel('Clone #WGD', ha='right', va='center', rotation=0)

plot_data['clonality'] = plot_data['is_subclonal'].map({True: 'Subclonal', False: 'Clonal'})

ax = axes[1]
sns.barplot(
    ax=ax, x='wgd_name', y='ones', hue='clonality',
    dodge=False, data=plot_data.assign(ones=1), width=1, order=wgd_name_order, hue_order=['Subclonal', 'Clonal'],
    palette={'Clonal': '#004369', 'Subclonal': '#C0DCEC'})
legend = ax.legend(loc='upper left', bbox_to_anchor=(1, 1), frameon=False, fontsize=6)
legend.set_title('Clonality', prop={'size': 8})
setup_categorical_axis(ax)
ax.set_ylabel('WGD clonality', ha='right', va='center', rotation=0)

plot_data = plot_data.set_index('wgd_name', drop=False).loc[wgd_name_order]
ax = axes[2]
bottom = pd.Series(0, index=plot_data.index)
for idx, event_type in enumerate(event_types):
    x = plot_data['wgd_name']
    y = plot_data[event_type]
    sns.barplot(
        ax=ax, x=x, bottom=bottom, y=y, color='0.9', linewidth=0, facecolor=colors_dict['misseg_event'][event_type],
        label=event_type)
    bottom += plot_data[event_type]
    ax.set_ylim((0, 30))
    ax.set_ylabel(event_type, ha='right', va='center', rotation=0)

# Average multipolar post-wgd losses (see analysis/notebooks/scdna/heterogeneity/multipolar_changes.md)
average_multipolar_count = 8.631579
ax.axhline(average_multipolar_count, ls=':', color='k')
ax.annotate(
    'Average divergent cell\narm+chrom. count', xy=(1, average_multipolar_count), xytext=(3, 30),
    textcoords='offset points', arrowprops=dict(
        facecolor='black', arrowstyle="-", connectionstyle="arc3"))

sns.despine(ax=ax)
sns.move_legend(ax, loc='lower left', bbox_to_anchor=(1, 0.25), title='Event type', ncols=1, markerscale=1, frameon=False, fontsize=6, title_fontsize=8)

ax = axes[3]
sns.barplot(
    ax=ax, x='wgd_name', y='cell_fraction', color='0.9', linewidth=0.5, edgecolor='k',
    data=plot_data, order=wgd_name_order, rasterized=True)
ax.set_yscale('log')
ax.set_ylabel('WGD clone\ncell fraction', ha='right', va='center', rotation=0)
sns.despine(ax=ax)

plt.subplots_adjust(hspace=0.3)

ax.set_xticklabels(
    plt.gca().get_xticklabels(), 
    rotation=60, 
    ha='right',  
    rotation_mode='anchor')

ax.set_xlabel('')

fig.savefig(f'../../../../figures/figure4/clone_post_wgd_overview.svg', bbox_inches='tight', metadata={'Date': None})

```

```python

import scipy

event_types = [
    'Chrom. gain',
    'Arm gain',
    'Chrom. loss',
    'Arm loss',
]

plot_data2 = plot_data.melt(id_vars=['patient_id', 'is_subclonal'], value_vars=event_types, value_name='count', var_name='class')

mwu_tests = spectrumanalysis.stats.mwu_tests(plot_data2, ['is_subclonal', 'class'], 'count')
mwu_tests = mwu_tests.query('class_1 == class_2')
mwu_tests = spectrumanalysis.stats.fdr_correction(mwu_tests)

fig, axes = plt.subplots(
    nrows=1, ncols=len(event_types), figsize=(3, 2), dpi=150, sharex=False, sharey=True)

for idx, event_type in enumerate(event_types):
    ax = axes[idx]
    sns.barplot(
        ax=ax, x='is_subclonal', y=event_type, color='0.9', linewidth=0,
        edgecolor='k', facecolor=colors_dict['misseg_event'][event_type],
        err_kws={'linewidth': 1, 'color': 'k'}, capsize=0.2,
        data=plot_data)
    ax.set_xlabel('')
    ax.set_title(event_type.replace(' ', '\n'), y=1.05, fontsize=10)
    ax.set_xticklabels(
        ['Clonal', 'Subclonal'],
        rotation=60, 
        ha='right',
        va='center',
        rotation_mode='anchor',
        fontsize=10)

    significance = mwu_tests.set_index('class_1').loc[event_type, 'significance_corrected']
    spectrumanalysis.stats.add_significance_line(ax, significance, 0, 1, 1.)
    sns.despine(ax=ax)

axes[0].set_ylabel('Post-WGD\nevent count', ha='center', va='bottom', rotation=90, labelpad=0)

plt.tight_layout()

fig.savefig(f'../../../../figures/figure4/clone_post_wgd_comparison.svg', bbox_inches='tight', metadata={'Date': None})

```

```python

snv_leaf_table = pd.read_csv('../../../../results/tables/snv_tree/snv_leaf_table.csv')
snv_leaf_table = snv_leaf_table[snv_leaf_table['is_wgd']]
age_since_wgd = snv_leaf_table.groupby(['patient_id'])['snv_count_age_per_gb_since_wgd'].mean().reset_index()
age_since_wgd.head()

```

```python

plot_data = wgd_clone_events.rename(columns=event_type_rename).merge(age_since_wgd[['patient_id', 'snv_count_age_per_gb_since_wgd']])
plot_data = plot_data[plot_data['n_wgd'] == 1]

fig, axes = plt.subplots(
    nrows=1, ncols=4, figsize=(6, 2), dpi=150, sharex=False, sharey=True)

for idx, event_type in enumerate(event_types):
    r, p = scipy.stats.spearmanr(
        plot_data[event_type], plot_data['snv_count_age_per_gb_since_wgd'])

    ax = axes.flatten()[idx]
    sns.regplot(
        ax=ax, x=event_type, y='snv_count_age_per_gb_since_wgd', data=plot_data,
        scatter_kws=dict(s=10, color='k'), line_kws=dict(color="k", linestyle=':', linewidth=1))

    if idx == 0:
        ax.set_ylabel('C>T CpG Mut / GB\nsince WGD')
    else:
        ax.set_ylabel('')
    ax.annotate(f'$r={r:.2f}$\n$p={p:.3f}$', (0.15, 1.05), xycoords='axes fraction', fontsize=8)
    sns.despine(ax=ax, trim=True, offset=10)

plt.tight_layout()

fig.savefig(f'../../../../figures/edfigure4/since_wgd_events.svg', bbox_inches='tight', metadata={'Date': None})

```


# Pre and post-wgd events


```python

selected_branches = pd.concat([
    cell_counts.query('root and wgd_class == "WGD-low"').query('n_wgd == 0')[['patient_id', 'obs_id']].drop_duplicates().assign(timing_wgd='not_wgd'),
    wgd_clone_events.query('~is_subclonal and patient_id != "SPECTRUM-OV-024"')[['patient_id', 'obs_id']].drop_duplicates().assign(timing_wgd='pre'),
    wgd_clone_events.query('~is_subclonal and patient_id != "SPECTRUM-OV-024"')[['patient_id', 'obs_id']].drop_duplicates().assign(timing_wgd='post'),
])

```

```python

event_type_rename = {
    'chromosome_gain': 'Chrom. gain',
    'arm_gain': 'Arm gain',
    'segment_gain': 'Segment gain',
    'chromosome_loss': 'Chrom. loss',
    'arm_loss': 'Arm loss',
    'segment_loss': 'Segment loss',
}

event_types = [
    'Chrom. gain',
    'Chrom. loss',
    'Arm gain',
    'Arm loss',
    # 'Segment gain',
    # 'Segment loss',
]

timing_wgd_order = [
    'not_wgd',
    'pre',
    'post',
]

timing_wgd_names = [
    'non-WGD',
    'pre-WGD',
    'post-WGD',
]

plot_data = events.merge(selected_branches)
plot_data = plot_data.groupby(['patient_id', 'timing_wgd', 'class'], observed=True).size().unstack(fill_value=0).stack().rename('count')
plot_data = plot_data.reset_index().merge(events[['kind', 'class']].drop_duplicates())
plot_data['class'] = plot_data['class'].map(event_type_rename)

def poisson_test(grp1, grp2):
    df1 = grp1[['count']].assign(group=0)
    df2 = grp2[['count']].assign(group=1)
    df = pd.concat([df1, df2], ignore_index=True)
    model = smf.glm(formula="count ~ group", data=df, family=sm.families.Poisson()).fit()
    info = {
        'coef': model.params['group'],
        'p': model.pvalues['group'],
        'n_1': len(df1),
        'n_2': len(df2),
    }
    return info

glm_tests = spectrumanalysis.stats.run_unpaired_tests(plot_data, ['timing_wgd', 'kind', 'class'], poisson_test)
glm_tests = glm_tests.query('class_1 == class_2').query('kind_1 == kind_2')
glm_tests = spectrumanalysis.stats.fdr_correction(glm_tests)

fig, axes = plt.subplots(ncols=len(event_types), sharex=True, sharey=True, figsize=(4, 2), dpi=150)

def add_glm_pvalue(ax, df, event_type):
    y = 1.
    x_map = {'not_wgd': 0, 'pre': 1, 'post': 2}
    for timing_1, timing_2 in (('post', 'pre'), ('not_wgd', 'post'), ('not_wgd', 'pre')):
        significance = glm_tests.loc[
            (glm_tests['class_1'] == event_type) &
            (glm_tests['timing_wgd_1'] == timing_1) &
            (glm_tests['timing_wgd_2'] == timing_2),
            'significance_corrected'
        ].values
        assert len(significance) == 1, f'invalid significance {significance} for {event_type}, {timing_1}, {timing_2}'
        significance = significance[0]
        if significance == 'ns':
            continue
        spectrumanalysis.stats.add_significance_line(ax, significance, x_map[timing_1], x_map[timing_2], y)
        y += 0.075

for ax, event_type in zip(axes, event_types):
    sns.barplot(
        ax=ax, data=plot_data[(plot_data['class'] == event_type)],
        x='timing_wgd', y='count', order=timing_wgd_order, color=colors_dict['misseg_event'][event_type],
        err_kws={'linewidth': 1, 'color': 'k'}, capsize=0.2,
    )
    sns.despine(ax=ax)
    ax.set_title(event_type.replace(' ', '\n'), y=1.25, fontsize=10)

    ax.set_xticklabels(
        timing_wgd_names,
        rotation=60,
        ha='right',
        va='center',
        rotation_mode='anchor')
    ax.set_xlabel('')
    add_glm_pvalue(ax, plot_data, event_type)

plt.subplots_adjust(hspace=0.3)

fig.savefig(f'../../../../figures/figure4/root_branch_changes.svg', bbox_inches='tight', metadata={'Date': None})

```

```python

glm_tests.query('class_1 == class_2').query('kind_1 == kind_2').query('class_1 == "Arm loss"')

```

```python

mwu_tests = spectrumanalysis.stats.mwu_tests(plot_data, ['timing_wgd', 'kind', 'class'], 'count')
mwu_tests['fold_change'] = mwu_tests[['y_mean_1', 'y_mean_2']].max(axis=1) / mwu_tests[['y_mean_1', 'y_mean_2']].min(axis=1)

mwu_tests[
    (mwu_tests['timing_wgd_1'] == mwu_tests['timing_wgd_2']) &
    (
        (mwu_tests['class_1'].str.startswith('Arm') & mwu_tests['class_2'].str.startswith('Arm')) |
        (mwu_tests['class_1'].str.startswith('Chrom') & mwu_tests['class_2'].str.startswith('Chrom'))
    )
]

```


# Compare loss / gain ratios


```python

event_rates_filename = f'{project_dir}/postprocessing/sankoff_ar/sankoff_ar_rates.tsv'
event_rates = pd.read_csv(event_rates_filename, sep='\t')
event_rates = event_rates.query(f'group_level == "patient" & normalized == False')
event_rates = event_rates.merge(wgd_class)

event_type_names = {
    'chromosome_gain': 'Chrom. gain',
    'arm_gain': 'Arm gain',
    'segment_gain': 'Segment gain',
    'chromosome_loss': 'Chrom. loss',
    'arm_loss': 'Arm loss',
    'segment_loss': 'Segment loss',
}

event_types = [
    'Chrom. gain',
    'Chrom. loss',
    'Arm gain',
    'Arm loss',
    'Segment gain',
    'Segment loss',
]

event_rates = event_rates.rename(columns=event_type_names)

event_rates2 = event_rates.query('n_cells >= 5').melt(
    value_vars=event_types, var_name='kind', value_name='rate',
    id_vars=['patient_id',  'wgd_class', 'n_cells'])

event_ratios = event_rates2.set_index(['patient_id', 'wgd_class', 'n_cells', 'kind'])['rate'].unstack().reset_index()
event_ratios['chromosome'] = (event_ratios['Chrom. loss'] + (1/event_ratios['n_cells'])) / (event_ratios['Chrom. gain'] + (1/event_ratios['n_cells']))
event_ratios['arm'] = (event_ratios['Arm loss'] + (1/event_ratios['n_cells'])) / (event_ratios['Arm gain'] + (1/event_ratios['n_cells']))

event_ratios = event_ratios.melt(id_vars=['patient_id', 'wgd_class'], value_vars=['chromosome', 'arm'], var_name='region', value_name='loss_ratio')
event_ratios['timing_wgd'] = event_ratios['wgd_class']
event_ratios

```

```python

plot_data = events.merge(selected_branches)
plot_data = plot_data.query('kind != "wgd"')
plot_data = plot_data.groupby(['patient_id', 'timing_wgd', 'class'], observed=False).size().unstack(fill_value=0).stack().rename('count')
plot_data = plot_data.reset_index().merge(events[['class', 'kind', 'region']].drop_duplicates())
plot_data = plot_data.set_index(['patient_id', 'timing_wgd', 'region', 'kind'])['count'].unstack().reset_index()
plot_data['loss_ratio'] = (plot_data['loss'] + 1) / (plot_data['gain'] + 1)

plot_data = pd.concat([plot_data.assign(branch='Ancestral'), event_ratios.assign(branch='Cell specific')])

mwu_tests = spectrumanalysis.stats.mwu_tests(plot_data, ['timing_wgd', 'region'], 'loss_ratio')
mwu_tests = mwu_tests.query('region_1 == region_2')
mwu_tests = spectrumanalysis.stats.fdr_correction(mwu_tests)

region_types = [
    'chromosome',
    'arm',
]

def add_mwu_pvalue(ax, df, region_type):
    y = 0.85
    x_map = {'WGD-low': 0, 'WGD-high': 1, 'not_wgd': 2, 'pre': 3, 'post': 4}
    for timing_1, timing_2 in itertools.product(['WGD-low', 'WGD-high'], ['not_wgd', 'pre', 'post']):
        significance = mwu_tests.loc[
            (mwu_tests['region_1'] == region_type) &
            (
                (
                    (mwu_tests['timing_wgd_1'] == timing_1) &
                    (mwu_tests['timing_wgd_2'] == timing_2)
                ) |
                (
                    (mwu_tests['timing_wgd_1'] == timing_1) &
                    (mwu_tests['timing_wgd_2'] == timing_2)
                )
            ),
            'significance_corrected'
        ].values
        assert len(significance) == 1, f'invalid significance {significance} for {region_type}, {timing_1}, {timing_2}'
        significance = significance[0]
        if significance == 'ns':
            continue
        spectrumanalysis.stats.add_significance_line(ax, significance, x_map[timing_1], x_map[timing_2], y)
        y += 0.0765

fig, axes = plt.subplots(ncols=len(region_types), sharex=True, sharey=True, figsize=(2.5, 1.5), dpi=150)

for ax, region_type in zip(axes, region_types):
    sns.barplot(
        ax=ax, data=plot_data[(plot_data['region'] == region_type)],
        x='timing_wgd', y='loss_ratio', order=['WGD-low', 'WGD-high'] + timing_wgd_order,
        hue='branch', palette={'Ancestral': '#059cad', 'Cell specific': '#344ea2'},
        err_kws={'linewidth': 1, 'color': 'k'}, capsize=0.2,
    )
    sns.despine(ax=ax)
    ax.set_title(region_type.replace(' ', '\n'), y=1.35, fontsize=10)

    ax.set_xticklabels(
        ['WGD-low', 'WGD-high'] + timing_wgd_names,
        rotation=60,
        ha='right',
        va='center',
        rotation_mode='anchor')
    ax.set_xlabel('')
    add_mwu_pvalue(ax, plot_data, region_type)

axes[0].set_ylabel('Ratio of losses / gains')
axes[0].get_legend().remove()

ax.get_ylim()

fig.savefig(f'../../../../figures/edfigure4/loss_gain_ratios.svg', bbox_inches='tight', metadata={'Date': None})

```

```python

```
