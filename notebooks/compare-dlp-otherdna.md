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
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

```python
import pandas as pd
import vcf
import seaborn as sns
import matplotlib.pyplot as plt
import os
import yaml
import numpy as np
import scipy
import vetica.mpl

os.environ['ISABL_API_URL'] = 'https://isabl.shahlab.mskcc.org/api/v1'
import isabl_cli as ii
import matplotlib
matplotlib.rcParams['svg.fonttype'] = 'none'
```

```python
ignacio_repo = 'repos/spectrum-genomic-instability'
pipeline_outputs = pipeline_dir # path to root directory of scWGS pipeline outputs
spectrumanalysis_repo =  '.'
```

# load DLP data and aggregate by patient

```python
cell_info = pd.read_csv(os.path.join(pipeline_outputs, 'preprocessing/summary/filtered_cell_table.csv.gz'))
dlp_means = cell_info[cell_info.include_cell][['patient_id', 'ploidy', 'fraction_loh']].groupby('patient_id').aggregate('mean')
```

```python
dlp_means.head()
```

# load facets data

```python
patient_mapping = pd.read_table(os.path.join(ignacio_repo, 'resources/db/genomic_instability/SPECTRUM/sequencing_msk_impact_custom.tsv'))
patient_mapping = patient_mapping[['impact_dmp_patient_id', 'patient_id']].drop_duplicates().set_index('impact_dmp_patient_id').patient_id
```

```python
patient_mapping
```

```python
facets = pd.read_table('users/vazquezi/projects/ccs/shared/resources/impact/cohort-level/50K/facets/2022_02_09/msk_impact_facets_annotated.cohort.txt.gz')
```

```python
facets['patient_id'] = facets.patient.map(patient_mapping)
```

## load facets seg files

```python
facets.iloc[0]
```

```python
facets.frac_loh
```

```python
facets.purity
```

```python
facets.ploidy
```

# load remixt data
* skip 068 (bad ploidy call)
* use more recent results for 115 (better ploidy call)

```python
remixt_table = pd.read_csv(os.path.join(spectrumanalysis_repo, 'analysis/notebooks/bulk-dna/genome_instability.csv'))
```

## replace results for 115 with more recent values

```python
remixt_analysis = ii.Analysis(22074)
```

```python
mixture
```

<!-- #raw -->
def get_purity_ploidy(pk):
    remixt_analysis = ii.Analysis(pk)
    mixture = yaml.safe_load(open(remixt_analysis.results['meta'], 'r').read())['mix']
    remixt = pd.read_table(remixt_analysis.results['remixt_cn'])
    purity = 1 - mixture[0]
    remixt['frac1'] = (remixt.major_1 + remixt.minor_1) * (mixture[1] / purity)
    remixt['frac2'] = (remixt.major_2 + remixt.minor_2) * (mixture[2] / purity)
    remixt['width'] = remixt.end - remixt.start
    ploidy = ((remixt.frac1 + remixt.frac2) * remixt.length).sum() / (remixt.length.sum())
    return {'ploidy':ploidy, 
            'tumour_proportion': purity, 
            'normal_proportion': mixture[0], 
            'clone_1_proportion': mixture[1], 
            'clone_2_proportion': mixture[2]}

<!-- #endraw -->

```python
def get_purity_ploidy(pk):
    remixt_analysis = ii.Analysis(pk)
    meta = yaml.safe_load(open(remixt_analysis.results['meta'], 'r').read())
    mixture = meta['mix']
    ploidy = meta['ploidy']
    return {'ploidy':ploidy, 
            'tumour_proportion': 1-mixture[0], 
            'normal_proportion': mixture[0], 
            'clone_1_proportion': mixture[1], 
            'clone_2_proportion': mixture[2],
           'proportion_divergent':meta['proportion_divergent']}

```

```python
values115 = get_purity_ploidy(22074)
values115
```

```python
remixt_table.loc[remixt_table.isabl_patient_id == 'SPECTRUM-OV-115', 'ploidy'] = values115['ploidy']
remixt_table.loc[remixt_table.isabl_patient_id == 'SPECTRUM-OV-115', 'tumour_proportion'] = values115['tumour_proportion']
remixt_table.loc[remixt_table.isabl_patient_id == 'SPECTRUM-OV-115', 'normal_proportion'] = values115['normal_proportion']
remixt_table.loc[remixt_table.isabl_patient_id == 'SPECTRUM-OV-115', 'clone_1_proportion'] = values115['clone_1_proportion']
remixt_table.loc[remixt_table.isabl_patient_id == 'SPECTRUM-OV-115', 'proportion_divergent'] = values115['proportion_divergent']
```

```python
remixt_table.loc[remixt_table.isabl_patient_id == 'SPECTRUM-OV-115']
```

```python
get_purity_ploidy(23771)
```

```python
remixt_table.loc[remixt_table.isabl_patient_id == 'SPECTRUM-OV-067']
```

```python

```

# combine into one table

```python
merged = dlp_means.reset_index().merge(facets[['patient_id', 'purity', 'ploidy', 'frac_loh']].dropna().rename(
    columns={'purity':'facets_purity', 'ploidy':'facets_ploidy', 'frac_loh':'facets_fraction_loh'}))

merged = merged.merge(remixt_table[['isabl_patient_id', 'ploidy', 'tumour_proportion']].rename(
    columns={'isabl_patient_id':'patient_id', 'ploidy':'remixt_ploidy', 'tumour_proportion':'remixt_purity'}), on='patient_id')
```

# make plots

```python
from adjustText import adjust_text
def plot_comparison(table, x_field, y_field, x_label, y_label, figsize=(3.5,3.5), dpi = 150):
    fig = plt.figure(figsize=figsize, dpi = dpi)
    sns.regplot(x=x_field, y=y_field, data=table, ax = plt.gca(), 
                scatter_kws=dict(s=8, facecolor='k', edgecolors='k'), 
                line_kws=dict(color="k", linestyle=':', linewidth=1))
    
    r, p = scipy.stats.spearmanr(table[x_field], table[y_field])
    ax = plt.gca()
    ax.text(.1, .95, 'r={:.2f}, p={:.2g}'.format(r, p),
            transform=ax.transAxes, color='k')

    x = table[x_field].values
    y = table[y_field].values
    labels = table.patient_id.str.split('-', expand=True)[2].values
    ts = []
    for i, txt in enumerate(labels):
        ts.append(ax.annotate(txt, (x[i], y[i]), fontsize='small', ha='center', va='bottom'))
    adjust_text(ts, x=x, y=y,  arrowprops=dict(arrowstyle='-', color='k'), min_arrow_len=9.5) #force_text=(0.2, 0.3), force_static=(0.1, 0.2))
    
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.xlim(1.45, 4.25)
    plt.ylim(1.45, 4.6)
    sns.despine()
```

```python
plot_comparison(merged, x_field='facets_ploidy', x_label='IMPACT ploidy', y_field= 'ploidy', y_label='Mean ploidy (DLP+ scWGS)')
plt.savefig(os.path.join(spectrumanalysis_repo, 'figures/edfigure2/dlp_vs_impact_ploidy.svg'), metadata={'Date': None})
```

```python
# look for points cut out of the plot
merged[(((merged.ploidy < 1.45) | (merged.ploidy > 4.6) | (merged.facets_ploidy < 1.45) | (merged.facets_ploidy > 4.25)))]
```

```python
merged[merged.remixt_ploidy >= 5]
```

```python
# look for points cut out of the plot
merged[(merged.remixt_ploidy < 5) & (((merged.ploidy < 1.45) | (merged.ploidy > 4.6) | (merged.remixt_ploidy < 1.45) | (merged.remixt_ploidy > 4.25)))]
```

```python
plot_comparison(merged[merged.remixt_ploidy < 5], x_field='remixt_ploidy', x_label='Bulk WGS ploidy', y_field= 'ploidy', y_label='Mean ploidy (DLP+ scWGS)')
plt.savefig(os.path.join(spectrumanalysis_repo, 'figures/edfigure2/dlp_vs_bulkwgs_ploidy.svg'), metadata={'Date': None})

```

```python

```

```python

```

```python

```
