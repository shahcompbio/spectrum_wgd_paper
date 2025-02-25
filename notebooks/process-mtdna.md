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
import numpy as np
import tqdm
import os
from scipy.stats import linregress, mannwhitneyu
import spectrumanalysis.stats

import matplotlib.colors as mcolors
wgd_colors = {0:mcolors.to_hex((197/255, 197/255, 197/255)),
              1:mcolors.to_hex((252/255, 130/255, 79/255)),
              2:mcolors.to_hex((170/255, 0, 0/255))}
```

```python
cohort = pd.read_csv('/data1/shahs3/users/myersm2/repos/spectrumanalysis/pipelines/scdna/inputs/hmmcopy_table.csv')
cell_info = pd.read_csv('/data1/shahs3/users/myersm2/repos/spectrum_wgd_data5/preprocessing/summary/filtered_cell_table.csv.gz')
```

# check agreement between Minsoo's counts and my counts on test library


## attempt just counting reads directly: does okay job

```python
from collections import Counter
```

```python
sample_id = 'SPECTRUM-OV-002_S1_RIGHT_OVARY-128762A'
aliquot_id = 'SPECTRUM-OV-002_S1_UNSORTED_RIGHT_OVARY_128762A_L1'
patient_id = aliquot_id.split('_')[0]

read_barcodes = f'/data1/shahs3/users/myersm2/repos/mtdna-smk/results/{aliquot_id}/{aliquot_id}_read_barcodes.txt'

df2 = pd.read_table(f'/data1/shahs3/users/kimm/project/scDNA/bam/SPECTRUM/{patient_id}/{sample_id}/all_cells/merged_files/{aliquot_id}_master.tsv')
orig_df = df2.T
orig_df.columns = orig_df.iloc[0]
orig_df = orig_df[['mtDNAreadcount', 'mt_depth', 'nuclear_depth', 'nuclear_ploidy', 'mt_copynumber']].iloc[24:].astype(float).reset_index()
orig_df.index = np.arange(len(orig_df.index))
orig_df = orig_df.rename(columns={'index':'cell_id'})

cntr = Counter([a.strip() for a in open(read_barcodes).readlines()])
df = pd.DataFrame(cntr.items(), columns=['brief_cell_id', 'n_mt_reads'])
df['mt_coverage'] = (df.n_mt_reads * 46) / 16569

```

```python
plt.figure(figsize=(8,4), dpi = 150)
plt.subplot(1,2,1)
orig_df['brief_cell_id'] = ['-'.join(a.split('-')[-3:]) for a in orig_df.cell_id]
joint = orig_df.merge(df, on='brief_cell_id')
joint.mt_depth = joint.mt_depth.astype(float)
plt.xlabel("my coverage")
plt.ylabel("minsoo coverage")

sns.scatterplot(joint, x='mt_depth', y='mt_coverage')
plt.axis('square')
plt.plot([0, 200], [0, 200], 'k--')

plt.subplot(1,2,2)
test = df.merge(cell_info, on='brief_cell_id')
test['mtdna_cn'] = (test.mt_coverage / test.coverage_depth) * test.ploidy
testmerge = test.merge(orig_df, on ='cell_id')

sns.scatterplot(testmerge, x='mtdna_cn', y='mt_copynumber')
plt.plot([0, 6000], [0, 6000], 'k--')
plt.tight_layout()

plt.xlabel("my copy number")
plt.ylabel("minsoo copy number")
```

# try another sample to see if multiplier holds

```python
sample_id = 'SPECTRUM-OV-025_S1_RIGHT_OVARY-128663A'
minsoo_aliquot_id = 'SPECTRUM-OV-025_S1_RIGHT_OVARY-128663A'
aliquot_id = '/SPECTRUM-OV-025_S1_CD45N_RIGHT_OVARY_128663A_L2'
patient_id = aliquot_id.split('_')[0]

read_barcodes = f'/data1/shahs3/users/myersm2/repos/mtdna-smk/results/{aliquot_id}/{aliquot_id}_read_barcodes.txt'
cntr = Counter([a.strip() for a in open(read_barcodes).readlines()])
df = pd.DataFrame(cntr.items(), columns=['brief_cell_id', 'n_mt_reads'])
df['mt_coverage'] = (df.n_mt_reads * 46) / 16569


df2 = pd.read_table(f'/data1/shahs3/users/kimm/project/scDNA/bam/SPECTRUM/{patient_id}/{sample_id}/all_cells/merged_files/{minsoo_aliquot_id}_master.tsv')
orig_df = df2.T
orig_df.columns = orig_df.iloc[0]
orig_df = orig_df[['mtDNAreadcount', 'mt_depth', 'nuclear_depth', 'nuclear_ploidy', 'mt_copynumber']].iloc[24:].astype(float).reset_index()
orig_df.index = np.arange(len(orig_df.index))
orig_df = orig_df.rename(columns={'index':'cell_id'})

```

```python
plt.figure(figsize=(8,4), dpi = 150)
plt.subplot(1,2,1)
orig_df['brief_cell_id'] = ['-'.join(a.split('-')[-3:]) for a in orig_df.cell_id]
joint = orig_df.merge(df, on='brief_cell_id')
joint.mt_depth = joint.mt_depth.astype(float)
plt.xlabel("my coverage")
plt.ylabel("minsoo coverage")

sns.scatterplot(joint, x='mt_depth', y='mt_coverage', s = 5)
plt.axis('square')
plt.plot([0, 200], [0, 200], 'k--')

plt.subplot(1,2,2)
test = df.merge(cell_info, on='brief_cell_id')
test['mtdna_cn'] = (test.mt_coverage / test.coverage_depth) * test.ploidy
testmerge = test.merge(orig_df, on ='cell_id')

sns.scatterplot(testmerge, x='mtdna_cn', y='mt_copynumber', s = 5)
plt.plot([0, 6000], [0, 6000], 'k--')
plt.tight_layout()

plt.xlabel("my copy number")
plt.ylabel("minsoo copy number")
```

```python

```

### can we improve this by just using samtools stats?

```python
def skip_ahead(lines, i, skip_fields):
    while  i < len(lines) and (lines[i].startswith('#') or lines[i].split('\t')[0] in skip_fields):
        i += 1
    return i
def parse_bamstat(fname):
    skip_fields = set(['#', 'CHK', 'GCT', 'GCC', 'GCT', 'FBC', 'FTC', 'LBC', 'LTC', 'BCC', 'CRC', 'OXC', 'RXC', 'GCD'])
    lines = open(fname).readlines()
    result = {}

    summary = {}
    i = skip_ahead(lines, 0, skip_fields)
    
    # summary fields
    while lines[i].startswith('SN'):
        tkns = lines[i].split('\t')
        summary[tkns[1]] = float(tkns[2].strip())
        i += 1
    result['summary'] = summary

    i = skip_ahead(lines, i, skip_fields)
    if summary['filtered sequences:'] == 0:
        return result
    
    ffq = []
    while lines[i].startswith('FFQ'):
        tkns = lines[i].strip().split('\t')
        ffq.append(np.array(tkns[2:]).astype(int))
        i += 1
    ffq = np.array(ffq)
    result['first_fragment_quality'] = ffq
    
    i = skip_ahead(lines, i, skip_fields)
    
    lfq = []
    while lines[i].startswith('LFQ'):
        tkns = lines[i].strip().split('\t')
        lfq.append(np.array(tkns[2:]).astype(int))
        i += 1
    lfq = np.array(lfq)
    result['last_fragment_quality'] = lfq
    
    i = skip_ahead(lines, i, skip_fields)
    
    gcf = []
    while lines[i].startswith('GCF'):
        tkns = lines[i].strip().split('\t')
        gcf.append(np.array(tkns[1:]))
        i += 1
    gcf = np.array(gcf)
    result['first_fragment_gc'] = gcf

    i = skip_ahead(lines, i, skip_fields)
    
    gcl = []
    while lines[i].startswith('GCL'):
        tkns = lines[i].strip().split('\t')
        gcl.append(np.array(tkns[1:]))
        i += 1
    gcl = np.array(gcl)
    result['last_fragment_gc'] = gcl
   
    # skip the next few fields
    i = skip_ahead(lines, i, skip_fields)
    
    insert_sizes = []
    while lines[i].startswith('IS'):
        tkns = lines[i].strip().split('\t')
        insert_sizes.append({'insert_size':int(tkns[1]),
                             'pairs_total':int(tkns[2]),
                             'inward_oriented_pairs':int(tkns[3]),
                             'outward_oriented_pairs':int(tkns[4]),
                             'other_pairs':int(tkns[5])
                            })
        i += 1
    insert_sizes = pd.DataFrame(insert_sizes)
    result['insert_sizes'] = insert_sizes
    
    i = skip_ahead(lines, i, skip_fields)
    
    rl = []
    while lines[i].startswith('RL'):
        tkns = lines[i].strip().split('\t')
        rl.append({'read_length':int(tkns[1]),
                             'count':int(tkns[2]),
                            })
        i += 1
    rl = pd.DataFrame(rl)
    result['read_lengths'] = rl

    i = skip_ahead(lines, i, skip_fields)
    
    frl = []
    while lines[i].startswith('FRL'):
        tkns = lines[i].strip().split('\t')
        frl.append({'fragment_length':int(tkns[1]),
                             'count':int(tkns[2]),
                            })
        i += 1
    frl = pd.DataFrame(frl)
    result['first_fragment_lengths'] = frl

    i = skip_ahead(lines, i, skip_fields)
    
    lrl = []
    while lines[i].startswith('LRL'):
        tkns = lines[i].strip().split('\t')
        lrl.append({'fragment_length':int(tkns[1]),
                             'count':int(tkns[2]),
                            })
        i += 1
    lrl = pd.DataFrame(lrl)
    result['last_fragment_lengths'] = lrl

    i = skip_ahead(lines, i, skip_fields)
    
    mapq = []
    while lines[i].startswith('MAPQ'):
        tkns = lines[i].strip().split('\t')
        mapq.append({'mapq':int(tkns[1]),
                             'count':int(tkns[2]),
                            })
        i += 1
    mapq = pd.DataFrame(mapq)
    result['mapq'] = mapq

    i = skip_ahead(lines, i, skip_fields)
    
    indels = []
    while lines[i].startswith('ID'):
        tkns = lines[i].strip().split('\t')
        indels.append({'length':int(tkns[1]),
                             'n_insertions':int(tkns[2]),
                             'n_deletions':int(tkns[3])
                            })
        i += 1
    indels = pd.DataFrame(indels)
    result['indels'] = indels

    i = skip_ahead(lines, i, skip_fields)
    
    indels_per_cycle = []
    while lines[i].startswith('IC'):
        tkns = lines[i].strip().split('\t')
        indels_per_cycle.append({'cycle':int(tkns[1]),
                             'n_insertions_fwd':int(tkns[2]),
                             'n_insertions_rev':int(tkns[3]),
                             'n_deletions_fwd':int(tkns[4]),
                             'n_deletions_rev':int(tkns[5]),
                            })
        i += 1
    indels_per_cycle = pd.DataFrame(indels_per_cycle)
    result['indels_per_cycle'] = indels_per_cycle

    
    i = skip_ahead(lines, i, skip_fields)
    
    cov = []
    while lines[i].startswith('COV'):
        tkns = lines[i].strip().split('\t')
        cov.append({'cov':int(tkns[2]),
                             'n_bases':int(tkns[3]),
                            })
        i += 1
    cov = pd.DataFrame(cov)
    result['coverage'] = cov

    return result
```

```python
bs = parse_bamstat('/data1/shahs3/users/myersm2/repos/mtdna-smk/results/SPECTRUM-OV-002_S1_UNSORTED_RIGHT_OVARY_128762A_L1/bamfile_stats.txt')
```

```python
bs['summary']['bases mapped:'] / bs['summary']['reads mapped:']
```

# try making figure anyway

```python
results_dir = '/data1/shahs3/users/myersm2/repos/mtdna-smk/results'
mt_genome_length = 16569
avg_bases_per_read = 50

dfs = []
for aliquot in tqdm.tqdm(os.listdir(results_dir)):
    read_barcodes = f'/data1/shahs3/users/myersm2/repos/mtdna-smk/results/{aliquot}/{aliquot}_read_barcodes.txt'
    cntr = Counter([a.strip() for a in open(read_barcodes).readlines()])
    df = pd.DataFrame(cntr.items(), columns=['brief_cell_id', 'n_mt_reads'])
    df['mt_coverage'] = (df.n_mt_reads * avg_bases_per_read) / mt_genome_length
    df = df.merge(cell_info, on='brief_cell_id')
    df['mtdna_cn'] = (df.mt_coverage / df.coverage_depth) * df.ploidy
    dfs.append(df)
```

```python
all_results = pd.concat(dfs).reset_index(drop=True)

```

```python
from matplotlib.transforms import blended_transform_factory
def add_significance_line(ax, sig, x_pos_1, x_pos_2, y_pos):
    """
    Add a significance line with text annotation to a plot.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes object to add the significance line to.
    sig : str
        The text annotation to display above the line.
    x_pos_1 : float
        The x-coordinate of the start of the line.
    x_pos_2 : float
        The x-coordinate of the end of the line.
    y_pos : float
        The y-coordinate of the line.
    """
    transform = blended_transform_factory(ax.transData, ax.transAxes)
    line = plt.Line2D([x_pos_1, x_pos_2], [y_pos, y_pos], transform=transform, color='k', linestyle='-', linewidth=1)
    lineL = plt.Line2D([x_pos_1, x_pos_1], [y_pos, y_pos-0.02], transform=transform, color='k', linestyle='-', linewidth=1)
    lineR = plt.Line2D([x_pos_2, x_pos_2], [y_pos, y_pos-0.02], transform=transform, color='k', linestyle='-', linewidth=1)
    ax.get_figure().add_artist(line)
    ax.get_figure().add_artist(lineL)
    ax.get_figure().add_artist(lineR)
    y_pos_offset = 0

    plt.text((x_pos_1+x_pos_2)/2, y_pos+y_pos_offset, sig, ha='center', va='bottom', fontsize=8, transform=transform)

def boxplot_field(table, field, ylabel=None, figsize=(1.5, 3), dpi=150):
    result01 = mannwhitneyu(table[~table[field].isna() & (table.n_wgd == 0)][field].values, table[~table[field].isna() & (table.n_wgd == 1)][field].values)
    p01 = result01.pvalue
    significance01 = spectrumanalysis.stats.get_significance_string(p01)
    
    result12 = mannwhitneyu(table[~table[field].isna() & (table.n_wgd == 2)][field].values, table[~table[field].isna() & (table.n_wgd == 1)][field].values)
    p12 = result12.pvalue
    significance12 = spectrumanalysis.stats.get_significance_string(p12)
    
    fig, ax = plt.subplots(figsize=figsize, dpi = dpi)
    
    sns.boxplot(data=table[~table[field].isna()], x = 'n_wgd', y = field, hue='n_wgd',
                palette={i:wgd_colors[i] for i in sorted(table.n_wgd.unique())}, dodge=False, fliersize=1)
    add_significance_line(ax, significance01, 0, 1, 1.05)
    add_significance_line(ax, significance12, 1, 2, 1.1)
    ax.get_legend().remove()
    sns.despine(ax=ax)

    if ylabel is not None:
        plt.ylabel(ylabel)
    
    plt.xlabel("#WGD")
    return fig
```

```python
all_results['log10_mtdna_copynumber'] = np.log10(all_results.mtdna_cn)
all_results = all_results[all_results.brief_cell_id.isin(cell_info.brief_cell_id)].copy()
all_results.to_csv('my_mtdna_table.csv.gz', index=False)
```

```python
boxplot_field(table=all_results[all_results.include_cell], field='log10_mtdna_copynumber', ylabel='log10(mtDNA Copy Number)')

```

```python

```
