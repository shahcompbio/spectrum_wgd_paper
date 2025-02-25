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
import scipy.stats
import vetica.mpl

import spectrumanalysis.wgd
import spectrumanalysis.stats

project_dir = os.environ['SPECTRUM_PROJECT_DIR']

colors_dict = yaml.safe_load(open('../../../../config/colors.yaml', 'r'))

```

```python

cell_info = pd.read_csv(f'{project_dir}/preprocessing/summary/filtered_cell_table.csv.gz')
cell_info = cell_info[(cell_info['include_cell'] == True)]
cell_info = spectrumanalysis.wgd.classify_subclonal_wgd(cell_info)

sbmclone_cell_info = pd.read_csv(f'{project_dir}/sbmclone/sbmclone_cell_table.csv.gz')
cell_info = cell_info.merge(sbmclone_cell_info.drop(['brief_cell_id'], axis=1), on='cell_id')

cell_info.head()

```

```python

cluster_label = 'sbmclone_cluster_id'

# Per SNV cluster data
cluster_data = cell_info.groupby(['patient_id', 'majority_n_wgd', cluster_label], observed=True).agg(
    n_cells=('subclonal_wgd', 'size'),
    n_subclonal_wgd=('subclonal_wgd', 'sum'),
    fraction_wgd=('subclonal_wgd', 'mean')).reset_index()

cluster_data['n_subclonal_wgd_patient'] = cluster_data.groupby('patient_id')['n_subclonal_wgd'].transform('sum')
cluster_data['n_cells_patient'] = cluster_data.groupby('patient_id')['n_cells'].transform('sum')
cluster_data['fraction_subclonal_wgd_patient'] = cluster_data['n_subclonal_wgd_patient'] / cluster_data['n_cells_patient']

cluster_data['binom_pvalue'] = -1.
for idx, row in cluster_data.iterrows():
    cluster_data.loc[idx, 'binom_pvalue'] = scipy.stats.binomtest(
        row['n_subclonal_wgd'],
        row['n_cells'],
        # p=row['fraction_subclonal_wgd_patient'],  # Compare against patient specific mean
        p=cluster_data['fraction_subclonal_wgd_patient'].mean(),
        alternative='greater').pvalue
cluster_data['log_binom_pvalue'] = -np.log10(cluster_data['binom_pvalue'])
cluster_data['n_clones'] = cluster_data.groupby('patient_id').transform('size')

significance_threshold = 0.1

cluster_data['p'] = cluster_data['binom_pvalue']
cluster_data['significance'] = cluster_data['binom_pvalue'] < significance_threshold
cluster_data = spectrumanalysis.stats.fdr_correction(cluster_data)

cluster_data.query('patient_id == "SPECTRUM-OV-031"')

```

```python

import adjustText

pvalue_col = 'pvalue_corrected'

plot_data = cluster_data.query('fraction_wgd < 0.5').query('n_cells > 20')
plot_data[pvalue_col] = plot_data[pvalue_col].clip(lower=1e-6)

fig, ax = plt.subplots(figsize=(3, 2), dpi=150)
sns.scatterplot(ax=ax, x='fraction_wgd', y=pvalue_col, size='n_cells', data=plot_data, color='0.75')
ax.axhline(significance_threshold, ls=':', lw=1, color='0.75')
ax.set_xlabel('Fraction WGD')
ax.set_yscale('log')
ax.set_yticks(np.logspace(0, -6, 4))
ax.set_ylim((ax.get_ylim()[1], ax.get_ylim()[0]))
ax.set_ylabel('Binomial log(p) enriched\nsubclonal WGD per clone')
ax.spines[['top', 'right']].set_visible(False)
ax.spines['left'].set_bounds((ax.get_yticks()[-1], ax.get_yticks()[0]))
ax.spines['bottom'].set_bounds((0, 0.2))
sns.move_legend(ax, 'upper left', bbox_to_anchor=(1, 1), ncol=1, title='#Cells', frameon=False)

texts = []
for idx, row in plot_data.query(f'{pvalue_col} < 0.1').iterrows():
    print(row['patient_id'], row['fraction_wgd'], row[pvalue_col])
    text = plt.text(
        row['fraction_wgd'], row[pvalue_col],
        row['patient_id'].replace('SPECTRUM-', '') + '/' + str(row['sbmclone_cluster_id']))
    texts.append(text)
adjustText.adjust_text(texts, expand_text=(2, 2), # expand text bounding boxes by 1.2 fold in x direction and 2 fold in y direction
            arrowprops=dict(arrowstyle='-', color='black'))

fig.savefig(f'../../../../figures/figure2/binomial_wgd_enrich.svg', metadata={'Date': None})

```

```python

plot_data.query(f'pvalue_corrected < {significance_threshold}')

```

```python

cluster_data.query('fraction_wgd < 0.5').query('n_cells > 20').query('patient_id == "SPECTRUM-OV-139"')

```

```python

cluster_data.query('fraction_wgd > 0.09').query('fraction_wgd < 0.5').query('n_cells > 20')#.query('binom_pvalue < 0.15')

```

```python

cluster_data.query('fraction_wgd > .1')

```

```python

cluster_data.query('n_clones == 1')

```

```python

import matplotlib as mpl

import spectrumanalysis.dataload

patient_id = 'SPECTRUM-OV-115'
adata = spectrumanalysis.dataload.load_filtered_cna_adata(project_dir, patient_id)

# Filter for cells in cell_info
adata = adata[adata.obs.index.isin(cell_info['cell_id'].values)]

patient_cell_info = cell_info.set_index('cell_id').loc[adata.obs.index]

adata.obs[cluster_label] = patient_cell_info[cluster_label]

adata_wgd_plot = adata[(adata.obs['is_wgd'] == 1),  (adata.var['has_allele_cn'])].copy()
adata_wgd_plot = scgenome.tl.sort_cells(adata_wgd_plot, layer_name='state')

adata_nwgd_plot = adata[(adata.obs['is_wgd'] == 0), (adata.var['has_allele_cn'])]

# Subsample nwgd cells
adata_nwgd_plot = adata_nwgd_plot[adata_nwgd_plot.obs.groupby(cluster_label).sample(20).index]

cluster_ids = adata.obs[cluster_label].unique()

height_ratios = []
for cluster_id in cluster_ids:
    height_ratios.append(adata_nwgd_plot[adata_nwgd_plot.obs[cluster_label] == cluster_id].shape[0])
    height_ratios.append(adata_wgd_plot[adata_wgd_plot.obs[cluster_label] == cluster_id].shape[0])

with mpl.rc_context({'axes.linewidth': 0.5, 'axes.edgecolor': '0.5'}):

    fig, axes = plt.subplots(
        nrows=len(height_ratios), ncols=1, height_ratios=height_ratios, figsize=(7, 4), dpi=300, sharex=True)

    for idx, cluster_id in enumerate(cluster_ids):
        
        nwgd_cells = adata_nwgd_plot.obs[adata_nwgd_plot.obs[cluster_label] == cluster_id].index
        
        # Recalculate full count in case we downsampled
        num_nwgd_cells = adata.obs[(adata.obs['is_wgd'] == 0) & (adata.obs[cluster_label] == cluster_id)].shape[0]

        ax = axes[2 * idx]
        g = scgenome.pl.plot_cell_cn_matrix(
            adata_nwgd_plot[nwgd_cells],
            layer_name='state',
            ax=ax,
            cell_order_fields=['cell_order'],
        )
        ax.tick_params(axis='x', which='major', bottom=False)
        ax.set_ylabel(f'Clone {cluster_id}\nnWGD\nn={num_nwgd_cells}', size=4, rotation=0, ha='right', va='center')
        ax.set_xlabel('')
        
        wgd_cells = adata_wgd_plot.obs[adata_wgd_plot.obs[cluster_label] == cluster_id].index

        ax = axes[2 * idx + 1]
        g = scgenome.pl.plot_cell_cn_matrix(
            adata_wgd_plot[wgd_cells],
            layer_name='state',
            ax=ax,
            cell_order_fields=['cell_order'],
        )
        ax.set_ylabel(f'Clone {cluster_id}\nWGD\nn={wgd_cells.shape[0]}', size=4, rotation=0, ha='right', va='center')
        
        if idx < len(cluster_ids) - 1:
            ax.set_xlabel('')
            ax.tick_params(axis='x', which='major', bottom=False)

    plt.subplots_adjust(hspace=0.05)

```

```python

```

```python

```
