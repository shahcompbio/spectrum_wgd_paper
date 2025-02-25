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
import anndata as ad
import numpy as np
import os
import itertools
import matplotlib.pyplot as plt
from scipy.stats import binom
import seaborn as sns
import scipy
import tqdm

import scgenome
import matplotlib as mpl
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.colors as mcolors
import vetica.mpl
```

```python
project_dir = '/data1/shahs3/users/myersm2/repos/spectrum_wgd_data5'

```

```python

snv_pair_utility = {
    '0/0': '0/0 (uninformative)',
    '0/1': '0/1 (uninformative)',
    '1/0': '1/0 (uninformative)',
    '2/2': '2/2 (uninformative)',
    '2/0': '2/0 (independent)',
    '0/2': '0/2 (independent)',
    '1/1': '1/1 (common)',
    '1/2': '1/2 (contradictory)',
    '2/1': '2/1 (contradictory)',
}

snv_pair_utility_colors = {
    '0/0 (uninformative)': '#DDDDDD',
    '0/1 (uninformative)': '#D7DF23',
    '1/0 (uninformative)': '#F7941D',
    '2/2 (uninformative)': '#2E3192',
    '2/0 (independent)': '#EF4136',
    '0/2 (independent)': '#006838',
    '1/1 (common)': '#27AAE1',
    '1/2 (contradictory)': '#8B5E3C',
    '2/1 (contradictory)': '#8B5E3C',
}


```

```python

def compute_ml_genotypes_vec(alt1, tot1, alt2, tot2, epsilon = 0.01, return_genotypes = False):
    # allowed in either 1 or 2 WGDs
    prob00 = binom.logpmf(k=alt1, n=tot1, p=epsilon) + binom.logpmf(k=alt2, n=tot2, p=epsilon)
    prob10 = binom.logpmf(k=alt1, n=tot1, p=0.5) + binom.logpmf(k=alt2, n=tot2, p=epsilon)
    prob01 = binom.logpmf(k=alt1, n=tot1, p=epsilon) + binom.logpmf(k=alt2, n=tot2, p=0.5)
    prob22 = binom.logpmf(k=alt1, n=tot1, p=1 - epsilon) + binom.logpmf(k=alt2, n=tot2, p=1 - epsilon)

    # not allowed in either
    #prob12 = binom.logpmf(k=alt1, n=tot1, p=0.5) * binom.logpmf(k=alt2, n=tot2, p=1 - epsilon)
    #prob21 = binom.logpmf(k=alt1, n=tot1, p=1-epsilon) * binom.logpmf(k=alt2, n=tot2, p=0.5)

    # exclusive to 1 WGD (shared 1-copy post-WGD)
    prob11 = binom.logpmf(k=alt1, n=tot1, p=0.5) + binom.logpmf(k=alt2, n=tot2, p=0.5)

    # exclusive to 2 WGDs (distinct 2-copy pre-WGD)
    prob02 = binom.logpmf(k=alt1, n=tot1, p=epsilon) + binom.logpmf(k=alt2, n=tot2, p=1 - epsilon)
    prob20 = binom.logpmf(k=alt1, n=tot1, p=1 - epsilon) + binom.logpmf(k=alt2, n=tot2, p=epsilon)
    
    probs1 = np.vstack([prob00, prob10, prob01, prob11, prob22])
    probs2 = np.vstack([prob00, prob10, prob01, prob02, prob20, prob22])

    if return_genotypes:
        best_geno1 = np.argmax(probs1, axis = 0)
        best_geno2 = np.argmax(probs2, axis = 0)
        
        genos1 = [[0,0], [1,0], [0,1], [1,1], [2,2]]
        genos2 = [[0,0], [1,0], [0,1], [0,2], [2,0], [2,2]]
        g1 = [genos1[i] for i in best_geno1]
        g2 = [genos2[i] for i in best_geno2]
        cidx = np.arange(probs1.shape[1])
        
        return g1, g2, probs1[best_geno1, cidx], probs2[best_geno2, cidx]
    else:
        return np.max(probs1, axis = 0), np.max(probs2, axis = 0)

```

```python

block1 = '1'
block2 = '2'
min_snvcov_reads = 10


def get_block_snvs(blocks_adata, block_name, epsilon=0.01):
    block_snvs = scgenome.tl.get_obs_data(
        blocks_adata, block_name,
        var_columns=['chromosome', 'position', 'ref', 'alt', 'is_cnloh'],
        layer_names=['ref_count', 'alt_count', 'total_count', 'vaf'])
    block_snvs = block_snvs[block_snvs['total_count'] > min_snvcov_reads]
    block_snvs = block_snvs[block_snvs['is_cnloh']]
    
    block_snvs['p_cn0'] = binom.logpmf(k=block_snvs['alt_count'], n=block_snvs['total_count'], p=epsilon)
    block_snvs['p_cn1'] = binom.logpmf(k=block_snvs['alt_count'], n=block_snvs['total_count'], p=0.5)
    block_snvs['p_cn2'] = binom.logpmf(k=block_snvs['alt_count'], n=block_snvs['total_count'], p=1-epsilon)
    block_snvs['cn'] = block_snvs[['p_cn0', 'p_cn1', 'p_cn2']].idxmax(axis=1).str.replace('p_cn', '').astype(int)

    return block_snvs


def get_block_snvs_pairs(blocks_adata, block1, block2):
    block1_snvs = get_block_snvs(blocks_adata, block1)
    block2_snvs = get_block_snvs(blocks_adata, block2)

    if block1_snvs.shape[0] == 0 or block2_snvs.shape[0] == 0:
        return None
    
    block_snvs_pairs = block1_snvs.merge(block2_snvs, on=['chromosome', 'position', 'ref', 'alt'], suffixes=['_1', '_2'])

    g1, g2, p1, p2 = compute_ml_genotypes_vec(
        block_snvs_pairs['alt_count_1'], block_snvs_pairs['total_count_1'],
        block_snvs_pairs['alt_count_2'], block_snvs_pairs['total_count_2'],
        return_genotypes=True)

    g1 = np.array(g1)
    g2 = np.array(g2)
    
    block_snvs_pairs['wgd1_1'] = g1[:, 0]
    block_snvs_pairs['wgd1_2'] = g1[:, 1]

    block_snvs_pairs['wgd2_1'] = g2[:, 0]
    block_snvs_pairs['wgd2_2'] = g2[:, 1]

    block_snvs_pairs['p_wgd1'] = p1
    block_snvs_pairs['p_wgd2'] = p2

    block_snvs_pairs['cn'] = block_snvs_pairs['cn_1'].astype(str) + '/' + block_snvs_pairs['cn_2'].astype(str)
    
    return block_snvs_pairs

```

```python
cell_info = pd.read_csv(os.path.join(project_dir, 'preprocessing/summary/filtered_cell_table.csv.gz'))
cell_info = cell_info[cell_info.include_cell]
prevalent_wgd_patients = set(cell_info.groupby('patient_id').is_wgd.mean().reset_index().query('is_wgd > 0.7').patient_id)
```

```python
patients = sorted([a.split('_')[0] for a in os.listdir(f'{project_dir}/tree_snv/outputs')])

all_adatas = {}
for p in patients:
    blocks_adata = ad.read_h5ad(f'{project_dir}/tree_snv/inputs/{p}_general_clone_adata.h5')
    if blocks_adata.shape[0] > 1 and p in prevalent_wgd_patients:
        all_adatas[p] = blocks_adata
```

```python
len(all_adatas)
```

```python
sorted(all_adatas.keys())
```

```python
def plot_patient(patient_id):
    print(patient_id)
    blocks_adata = all_adatas[patient_id]
    blocks_adata.var['is_cnloh'] = blocks_adata.var['snv_type'] == '2:0'
    
    blocks_adata = blocks_adata[blocks_adata.obs.index != 'nan']
    
    
    clones = blocks_adata.obs.index
    snv_classes = [
        '0/0 (uninformative)',
        '0/1 (uninformative)',
        '1/0 (uninformative)',
        '2/2 (uninformative)',
        '2/0 (independent)',
        '0/2 (independent)',
        '1/1 (common)',
        '1/2 (contradictory)',
        '2/1 (contradictory)',
    ]
    
    fig, axes = plt.subplots(nrows=len(clones)-1, ncols=len(clones)-1, figsize=(3 * (len(clones)-1), 3 * (len(clones)-1)), dpi=150, sharex=True, sharey=True)
    if len(clones) == 2:
        axes = {(0,0):axes}

    all_empty = True
    for block1_idx, block2_idx in itertools.combinations(range(len(clones)), 2):
        block1 = clones[block1_idx]
        block2 = clones[block2_idx]
    
        block_snvs_pairs = get_block_snvs_pairs(blocks_adata, block1, block2)
        if block_snvs_pairs is None or block_snvs_pairs.shape[0] < 100:
            # no common cnLOH SNVs with sufficient counts
            continue
        else:
            all_empty = False
        print(block_snvs_pairs.shape)

        block_snvs_pairs['SNV class'] = block_snvs_pairs['cn'].map(snv_pair_utility)
        # block_snvs_pairs = block_snvs_pairs[~(block_snvs_pairs['cn'].isin(['0/0', '2/2']))]
        
        # block1_idx, block2_idx = sorted([int(block1), int(block2)])
        ax = axes[block2_idx - 1, block1_idx]
        with mpl.rc_context({'lines.linewidth': 0.25}):
            sns.kdeplot(ax=ax, x='vaf_1', y='vaf_2', data=block_snvs_pairs, fill=True, bw_adjust=0.5, color='slategrey', zorder=-100, alpha=0.4, clip=((0., 1.), (0., 1.)))#, alpha=0.1, lw=1)#, clip=((-0.1, 1.1), (-0.1, 1.1)))#bw_method=.1)#
        sns.scatterplot(ax=ax, x='vaf_1', y='vaf_2', hue='SNV class', hue_order=snv_classes, data=block_snvs_pairs, s=10, alpha=1., linewidth=0.2, palette=snv_pair_utility_colors, edgecolor='k')
    
        if block1_idx == 0:
            ax.set_ylabel(f'clone {block2} VAF', labelpad=10)
            
        if block2_idx == len(clones) - 1:
            ax.set_xlabel(f'clone {block1} VAF', labelpad=10)
    
        ax.tick_params(axis='y', labelrotation=0)
      
        ax.set_xlim([-0.1, 1.1])
        ax.set_ylim([-0.1, 1.1])
     
        ax.set_xticks([0, 0.5, 1])
        ax.set_yticks([0, 0.5, 1])
        
        sns.despine(ax=ax, trim=True, offset=5)
        if block1_idx == 0 and block2_idx == 1:
            sns.move_legend(
                ax, 'upper left',
                bbox_to_anchor=(1.2, 1), ncol=1, frameon=False,
                title='Variant copy number (X/Y)',
                markerscale=2,
                prop={'size': 8}, title_fontsize=10,
                labelspacing=0.4, handletextpad=0, columnspacing=0.5,
            )
        else:
            if ax.get_legend():
                ax.get_legend().remove()
        
    for row in range(len(clones)-1):
        for col in range(len(clones)-1):
            if col > row:
                ax = axes[row, col]
                ax.axis('off')
    
    plt.suptitle(patient_id.replace('SPECTRUM-', ''), y=0.95)
    plt.subplots_adjust(wspace=0.2, hspace=0.2)#, left=0.15, right=0.85, bottom=0.15, top=0.85)
    
    return all_empty
```

<!-- #raw -->
with PdfPages('../../figures/final/pairwise-vafs.pdf') as pdf:
    for i, p in enumerate(sorted(all_adatas.keys())):
        all_empty = plot_patient(p)
        if all_empty:
            plt.close()
            continue
            #print(p, ' has all empty')
        else:
            pdf.savefig(bbox_inches='tight', pad_inches=0.3)
        if i > 6:
            #pass
            break
<!-- #endraw -->

# draft alternate forms of the figure

```python
import met_brewer
```

```python
colors = met_brewer.met_brew(name="Manet", n=7, brew_type="continuous")

snv_pair_utility2 = {
    '0/0': 'Non-phylogenetic',
    '0/1': '0/1 (uninformative)',
    '1/0': '1/0 (uninformative)',
    '2/2': '2/2 (uninformative)',
    '2/0': '2/0 (independent)',
    '0/2': '0/2 (independent)',
    '1/1': '1/1 (common)',
    '1/2': 'Non-phylogenetic',
    '2/1': 'Non-phylogenetic',
}


snv_pair_utility_colors2 = {
    '0/1 (uninformative)':colors[2],
    '0/2 (independent)':colors[1],
    
    '1/0 (uninformative)':colors[4],
    '2/0 (independent)':colors[5],
    
    '1/1 (common)':colors[3],
    '2/2 (uninformative)':colors[6],
    'Non-phylogenetic':colors[0],
}

snv_classes2 = snv_pair_utility_colors2.keys()


c2snv = {v:k for k,v in snv_pair_utility_colors2.items()}

plt.figure(figsize=(1, 4), dpi = 150)
plt.scatter(np.zeros(len(colors)), np.arange(len(colors)), c = colors, s = 250)
plt.xticks([], [])
plt.yticks(np.arange(len(colors)), [c2snv[colors[i]] for i in range(len(colors))])
plt.title("Colors")
```

```python
patient_id = 'SPECTRUM-OV-025'
blocks_adata = all_adatas[patient_id]
blocks_adata.var['is_cnloh'] = blocks_adata.var['snv_type'] == '2:0'

blocks_adata = blocks_adata[blocks_adata.obs.index != 'nan']

clones = blocks_adata.obs.index
```

```python
include_scatter = True

fig, axes = plt.subplots(nrows=len(clones)-1, ncols=len(clones)-1, figsize=(3.5 * (len(clones)-1), 3 * (len(clones)-1)), dpi=200, sharex=True, sharey=True)
if len(clones) == 2:
    axes = {(0,0):axes}

all_empty = True
for block1_idx, block2_idx in itertools.combinations(range(len(clones)), 2):
    block1 = clones[block1_idx]
    block2 = clones[block2_idx]

    block_snvs_pairs = get_block_snvs_pairs(blocks_adata, block1, block2)
    if block_snvs_pairs is None or block_snvs_pairs.shape[0] < 100:
        # no common cnLOH SNVs with sufficient counts
        continue
    else:
        all_empty = False
    print(block_snvs_pairs.shape)

    block_snvs_pairs['SNV class'] = block_snvs_pairs['cn'].map(snv_pair_utility2)
    # block_snvs_pairs = block_snvs_pairs[~(block_snvs_pairs['cn'].isin(['0/0', '2/2']))]
    
    ax = axes[block2_idx - 1, block1_idx]

    # original density plot
    
    with mpl.rc_context({'lines.linewidth': 0.25}):
        sns.kdeplot(ax=ax, x='vaf_1', y='vaf_2', data=block_snvs_pairs, fill=True, bw_adjust=0.5, color='slategrey', zorder=-100, alpha=0.4, 
                    clip=((0., 1.), (0., 1.)), levels=5)#, alpha=0.1, lw=1)#, clip=((-0.1, 1.1), (-0.1, 1.1)))#bw_method=.1)#

    #sns.histplot(ax=ax, x='vaf_1', y='vaf_2', data=block_snvs_pairs, fill=True, color='slategrey', zorder=-100, cbar=True, bins=20, alpha=0.7) 
    #hb = ax.hexbin(block_snvs_pairs.vaf_1, block_snvs_pairs.vaf_2, gridsize=8, cmap = 'Greys')
    #plt.colorbar(hb)

    if include_scatter:
        sns.scatterplot(ax=ax, x='vaf_1', y='vaf_2', hue='SNV class', hue_order=snv_classes2, data=block_snvs_pairs, s=25, alpha=0.7, linewidth=0, 
                        palette=snv_pair_utility_colors2)

    if block1_idx == 0:
        ax.set_ylabel(f'clone {block2} VAF', labelpad=10)
        
    if block2_idx == len(clones) - 1:
        ax.set_xlabel(f'clone {block1} VAF', labelpad=10)

    ax.tick_params(axis='y', labelrotation=0)
  
    ax.set_xlim([-0.1, 1.1])
    ax.set_ylim([-0.1, 1.1])
 
    ax.set_xticks([0, 0.5, 1])
    ax.set_yticks([0, 0.5, 1])
    
    sns.despine(ax=ax, trim=True, offset=5)

    if include_scatter:
        if block1_idx == 0 and block2_idx == 1:
            sns.move_legend(
                ax, 'upper left',
                bbox_to_anchor=(1.2, 1), ncol=1, frameon=False,
                title='Variant copy number (X/Y)',
                markerscale=2,
                prop={'size': 8}, title_fontsize=10,
                labelspacing=0.4, handletextpad=0, columnspacing=0.5)
        else:
            if ax.get_legend():
                ax.get_legend().remove()
    
for row in range(len(clones)-1):
    for col in range(len(clones)-1):
        if col > row:
            ax = axes[row, col]
            ax.axis('off')

plt.suptitle(patient_id.replace('SPECTRUM-', ''), y=0.95)
plt.subplots_adjust(wspace=0.2, hspace=0.2)#, left=0.15, right=0.85, bottom=0.15, top=0.85)

```

## try to color histogram bins by SNV colors

```python
include_scatter = False

fig, axes = plt.subplots(nrows=len(clones)-1, ncols=len(clones)-1, figsize=(3.5 * (len(clones)-1), 3 * (len(clones)-1)), dpi=200, sharex=True, sharey=True)
if len(clones) == 2:
    axes = {(0,0):axes}

all_empty = True
for block1_idx, block2_idx in itertools.combinations(range(len(clones)), 2):
    block1 = clones[block1_idx]
    block2 = clones[block2_idx]

    block_snvs_pairs = get_block_snvs_pairs(blocks_adata, block1, block2)
    if block_snvs_pairs is None or block_snvs_pairs.shape[0] < 100:
        # no common cnLOH SNVs with sufficient counts
        continue
    else:
        all_empty = False
    print(block_snvs_pairs.shape)

    block_snvs_pairs['SNV class'] = block_snvs_pairs['cn'].map(snv_pair_utility2)
    # block_snvs_pairs = block_snvs_pairs[~(block_snvs_pairs['cn'].isin(['0/0', '2/2']))]
    
    ax = axes[block2_idx - 1, block1_idx]

    # original density plot
    '''
    with mpl.rc_context({'lines.linewidth': 0.25}):
        sns.kdeplot(ax=ax, x='vaf_1', y='vaf_2', data=block_snvs_pairs, fill=True, bw_adjust=0.5, color='slategrey', zorder=-100, alpha=0.4, clip=((0., 1.), (0., 1.)))#, alpha=0.1, lw=1)#, clip=((-0.1, 1.1), (-0.1, 1.1)))#bw_method=.1)#
    '''        
    hp = sns.histplot(ax=ax, x='vaf_1', y='vaf_2', data=block_snvs_pairs, fill=True, zorder=-100, bins=20,
                      hue='SNV class', hue_order=snv_classes2, palette=snv_pair_utility_colors2) 
    #hb = ax.hexbin(block_snvs_pairs.vaf_1, block_snvs_pairs.vaf_2, gridsize=8, cmap = 'Greys')
    #plt.colorbar(hb)

    if include_scatter:
        sns.scatterplot(ax=ax, x='vaf_1', y='vaf_2', hue='SNV class', hue_order=snv_classes2, data=block_snvs_pairs, s=10, alpha=1., linewidth=0.2, 
                        palette=snv_pair_utility_colors2, edgecolor='k')

    if block1_idx == 0:
        ax.set_ylabel(f'clone {block2} VAF', labelpad=10)
        
    if block2_idx == len(clones) - 1:
        ax.set_xlabel(f'clone {block1} VAF', labelpad=10)

    ax.tick_params(axis='y', labelrotation=0)
  
    ax.set_xlim([-0.1, 1.1])
    ax.set_ylim([-0.1, 1.1])
 
    ax.set_xticks([0, 0.5, 1])
    ax.set_yticks([0, 0.5, 1])
    
    sns.despine(ax=ax, trim=True, offset=5)

    if block1_idx == 0 and block2_idx == 1:
        sns.move_legend(
            ax, 'upper left',
            bbox_to_anchor=(1.2, 1), ncol=1, frameon=False,
            title='Variant copy number (X/Y)',
            markerscale=2,
            prop={'size': 8}, title_fontsize=10,
            labelspacing=0.4, handletextpad=0, columnspacing=0.5)
    else:
        if ax.get_legend():
            ax.get_legend().remove()
    
for row in range(len(clones)-1):
    for col in range(len(clones)-1):
        if col > row:
            ax = axes[row, col]
            ax.axis('off')

plt.suptitle(patient_id.replace('SPECTRUM-', ''), y=0.95)
plt.subplots_adjust(wspace=0.2, hspace=0.2)#, left=0.15, right=0.85, bottom=0.15, top=0.85)

```

## remove non-phylogenetic SNVs

```python
colors3 = met_brewer.met_brew(name="Degas", n=6, brew_type="continuous")

snv_pair_utility3 = {
    '0/1': '0/1 (uninformative)',
    '1/0': '1/0 (uninformative)',
    '2/2': '2/2 (uninformative)',
    '2/0': '2/0 (independent)',
    '0/2': '0/2 (independent)',
    '1/1': '1/1 (common)',
}


snv_pair_utility_colors3 = {
    '0/1 (uninformative)':colors3[2],
    '0/2 (independent)':colors3[1],
    
    '1/0 (uninformative)':colors3[4],
    '2/0 (independent)':colors3[5],
    
    '1/1 (common)':colors3[3],
    '2/2 (uninformative)':colors3[0],
}

snv_classes3 = snv_pair_utility_colors3.keys()


c2snv3 = {v:k for k,v in snv_pair_utility_colors3.items()}

plt.figure(figsize=(1, 4), dpi = 150)
plt.scatter(np.zeros(len(colors3)), np.arange(len(colors3)), c = colors3, s = 250)
plt.xticks([], [])
plt.yticks(np.arange(len(colors3)), [c2snv3[colors3[i]] for i in range(len(colors3))])
plt.title("Colors")
```

```python

```

```python
include_scatter = True

fig, axes = plt.subplots(nrows=len(clones)-1, ncols=len(clones)-1, figsize=(3.5 * (len(clones)-1), 3 * (len(clones)-1)), dpi=200, sharex=True, sharey=True)
if len(clones) == 2:
    axes = {(0,0):axes}

all_empty = True
for block1_idx, block2_idx in itertools.combinations(range(len(clones)), 2):
    block1 = clones[block1_idx]
    block2 = clones[block2_idx]

    block_snvs_pairs = get_block_snvs_pairs(blocks_adata, block1, block2)
    if block_snvs_pairs is None or block_snvs_pairs.shape[0] < 100:
        # no common cnLOH SNVs with sufficient counts
        continue
    else:
        all_empty = False
    print(block_snvs_pairs.shape)

    block_snvs_pairs = block_snvs_pairs[~block_snvs_pairs['cn'].isin(['0/0', '1/2', '2/1'])]
    block_snvs_pairs['SNV class'] = block_snvs_pairs['cn'].map(snv_pair_utility3)
    block_snvs_pairs = block_snvs_pairs[block_snvs_pairs['SNV class'] != 'Non-phylogenetic']
    
    # block_snvs_pairs = block_snvs_pairs[~(block_snvs_pairs['cn'].isin(['0/0', '2/2']))]
    
    ax = axes[block2_idx - 1, block1_idx]

    # original density plot
    
    with mpl.rc_context({'lines.linewidth': 0.25}):
        sns.kdeplot(ax=ax, x='vaf_1', y='vaf_2', data=block_snvs_pairs, fill=True, bw_adjust=0.5, color='slategrey', zorder=-100, alpha=0.4, 
                    clip=((0., 1.), (0., 1.)), levels=5)#, alpha=0.1, lw=1)#, clip=((-0.1, 1.1), (-0.1, 1.1)))#bw_method=.1)#

    #sns.histplot(ax=ax, x='vaf_1', y='vaf_2', data=block_snvs_pairs, fill=True, color='slategrey', zorder=-100, cbar=True, bins=20, alpha=0.7) 
    #hb = ax.hexbin(block_snvs_pairs.vaf_1, block_snvs_pairs.vaf_2, gridsize=8, cmap = 'Greys')
    #plt.colorbar(hb)

    if include_scatter:
        sns.scatterplot(ax=ax, x='vaf_1', y='vaf_2', hue='SNV class', hue_order=snv_classes3, data=block_snvs_pairs, s=25, alpha=0.7, linewidth=0, 
                        palette=snv_pair_utility_colors3)

    if block1_idx == 0:
        ax.set_ylabel(f'clone {block2} VAF', labelpad=10)
        
    if block2_idx == len(clones) - 1:
        ax.set_xlabel(f'clone {block1} VAF', labelpad=10)

    ax.tick_params(axis='y', labelrotation=0)
  
    ax.set_xlim([-0.1, 1.1])
    ax.set_ylim([-0.1, 1.1])
 
    ax.set_xticks([0, 0.5, 1])
    ax.set_yticks([0, 0.5, 1])
    
    sns.despine(ax=ax, trim=True, offset=5)

    if include_scatter:
        if block1_idx == 0 and block2_idx == 1:
            sns.move_legend(
                ax, 'upper left',
                bbox_to_anchor=(1.2, 1), ncol=1, frameon=False,
                title='Variant copy number (X/Y)',
                markerscale=2,
                prop={'size': 8}, title_fontsize=10,
                labelspacing=0.4, handletextpad=0, columnspacing=0.5)
        else:
            if ax.get_legend():
                ax.get_legend().remove()
    
for row in range(len(clones)-1):
    for col in range(len(clones)-1):
        if col > row:
            ax = axes[row, col]
            ax.axis('off')

plt.suptitle(patient_id.replace('SPECTRUM-', ''), y=0.95)
plt.subplots_adjust(wspace=0.2, hspace=0.2)#, left=0.15, right=0.85, bottom=0.15, top=0.85)

```

```python
include_scatter = False

fig, axes = plt.subplots(nrows=len(clones)-1, ncols=len(clones)-1, figsize=(3 * (len(clones)-1), 3 * (len(clones)-1)), dpi=200, sharex=True, sharey=True)
if len(clones) == 2:
    axes = {(0,0):axes}

all_empty = True
for block1_idx, block2_idx in itertools.combinations(range(len(clones)), 2):
    block1 = clones[block1_idx]
    block2 = clones[block2_idx]

    block_snvs_pairs = get_block_snvs_pairs(blocks_adata, block1, block2)
    if block_snvs_pairs is None or block_snvs_pairs.shape[0] < 100:
        # no common cnLOH SNVs with sufficient counts
        continue
    else:
        all_empty = False
    print(block_snvs_pairs.shape)

    
    block_snvs_pairs = block_snvs_pairs[~block_snvs_pairs['cn'].isin(['0/0', '1/2', '2/1'])]
    block_snvs_pairs['SNV class'] = block_snvs_pairs['cn'].map(snv_pair_utility2)
    block_snvs_pairs = block_snvs_pairs[block_snvs_pairs['SNV class'] != 'Non-phylogenetic']
    # block_snvs_pairs = block_snvs_pairs[~(block_snvs_pairs['cn'].isin(['0/0', '2/2']))]
    
    ax = axes[block2_idx - 1, block1_idx]

    # original density plot
    '''
    with mpl.rc_context({'lines.linewidth': 0.25}):
        sns.kdeplot(ax=ax, x='vaf_1', y='vaf_2', data=block_snvs_pairs, fill=True, bw_adjust=0.5, color='slategrey', zorder=-100, alpha=0.4, clip=((0., 1.), (0., 1.)))#, alpha=0.1, lw=1)#, clip=((-0.1, 1.1), (-0.1, 1.1)))#bw_method=.1)#
    '''        
    hp = sns.histplot(ax=ax, x='vaf_1', y='vaf_2', data=block_snvs_pairs, fill=True, zorder=-100, bins=20,
                      hue='SNV class', hue_order=snv_classes3, palette=snv_pair_utility_colors3) 
    #hb = ax.hexbin(block_snvs_pairs.vaf_1, block_snvs_pairs.vaf_2, gridsize=8, cmap = 'Greys')
    #plt.colorbar(hb)

    if include_scatter:
        sns.scatterplot(ax=ax, x='vaf_1', y='vaf_2', hue='SNV class', hue_order=snv_classes3, data=block_snvs_pairs, s=10, alpha=1., linewidth=0.2, 
                        palette=snv_pair_utility_colors3, edgecolor='k')

    if block1_idx == 0:
        ax.set_ylabel(f'clone {block2} VAF', labelpad=10)
        
    if block2_idx == len(clones) - 1:
        ax.set_xlabel(f'clone {block1} VAF', labelpad=10)

    ax.tick_params(axis='y', labelrotation=0)
  
    ax.set_xlim([-0.1, 1.1])
    ax.set_ylim([-0.1, 1.1])
 
    ax.set_xticks([0, 0.5, 1])
    ax.set_yticks([0, 0.5, 1])
    
    sns.despine(ax=ax, trim=True, offset=5)

    if block1_idx == 0 and block2_idx == 1:
        sns.move_legend(
            ax, 'upper left',
            bbox_to_anchor=(1.2, 1), ncol=1, frameon=False,
            title='Variant copy number (X/Y)',
            markerscale=2,
            prop={'size': 8}, title_fontsize=10,
            labelspacing=0.4, handletextpad=0, columnspacing=0.5)
    else:
        if ax.get_legend():
            ax.get_legend().remove()
    
for row in range(len(clones)-1):
    for col in range(len(clones)-1):
        if col > row:
            ax = axes[row, col]
            ax.axis('off')

plt.suptitle(patient_id.replace('SPECTRUM-', ''), y=0.95)
plt.subplots_adjust(wspace=0.2, hspace=0.2)#, left=0.15, right=0.85, bottom=0.15, top=0.85)

```

# lower alpha

```python
include_scatter = True

fig, axes = plt.subplots(nrows=len(clones)-1, ncols=len(clones)-1, figsize=(3.5 * (len(clones)-1), 3 * (len(clones)-1)), dpi=200, sharex=True, sharey=True)
if len(clones) == 2:
    axes = {(0,0):axes}

all_empty = True
for block1_idx, block2_idx in itertools.combinations(range(len(clones)), 2):
    block1 = clones[block1_idx]
    block2 = clones[block2_idx]

    block_snvs_pairs = get_block_snvs_pairs(blocks_adata, block1, block2)
    if block_snvs_pairs is None or block_snvs_pairs.shape[0] < 100:
        # no common cnLOH SNVs with sufficient counts
        continue
    else:
        all_empty = False
    print(block_snvs_pairs.shape)

    block_snvs_pairs = block_snvs_pairs[~block_snvs_pairs['cn'].isin(['0/0', '1/2', '2/1'])]
    block_snvs_pairs['SNV class'] = block_snvs_pairs['cn'].map(snv_pair_utility3)
    block_snvs_pairs = block_snvs_pairs[block_snvs_pairs['SNV class'] != 'Non-phylogenetic']
    
    # block_snvs_pairs = block_snvs_pairs[~(block_snvs_pairs['cn'].isin(['0/0', '2/2']))]
    
    ax = axes[block2_idx - 1, block1_idx]

    # original density plot
    
    with mpl.rc_context({'lines.linewidth': 0.25}):
        sns.kdeplot(ax=ax, x='vaf_1', y='vaf_2', data=block_snvs_pairs, fill=True, bw_adjust=0.5, color='slategrey', zorder=-100, alpha=0.4, 
                    clip=((0., 1.), (0., 1.)), levels=5)#, alpha=0.1, lw=1)#, clip=((-0.1, 1.1), (-0.1, 1.1)))#bw_method=.1)#

    #sns.histplot(ax=ax, x='vaf_1', y='vaf_2', data=block_snvs_pairs, fill=True, color='slategrey', zorder=-100, cbar=True, bins=20, alpha=0.7) 
    #hb = ax.hexbin(block_snvs_pairs.vaf_1, block_snvs_pairs.vaf_2, gridsize=8, cmap = 'Greys')
    #plt.colorbar(hb)

    if include_scatter:
        sns.scatterplot(ax=ax, x='vaf_1', y='vaf_2', hue='SNV class', hue_order=snv_classes3, data=block_snvs_pairs, s=25, alpha=0.1, linewidth=0, 
                        palette=snv_pair_utility_colors3)

    if block1_idx == 0:
        ax.set_ylabel(f'clone {block2} VAF', labelpad=10)
        
    if block2_idx == len(clones) - 1:
        ax.set_xlabel(f'clone {block1} VAF', labelpad=10)

    ax.tick_params(axis='y', labelrotation=0)
  
    ax.set_xlim([-0.1, 1.1])
    ax.set_ylim([-0.1, 1.1])
 
    ax.set_xticks([0, 0.5, 1])
    ax.set_yticks([0, 0.5, 1])
    
    sns.despine(ax=ax, trim=True, offset=5)

    if include_scatter:
        if block1_idx == 0 and block2_idx == 1:
            sns.move_legend(
                ax, 'upper left',
                bbox_to_anchor=(1.2, 1), ncol=1, frameon=False,
                title='Variant copy number (X/Y)',
                markerscale=2,
                prop={'size': 8}, title_fontsize=10,
                labelspacing=0.4, handletextpad=0, columnspacing=0.5)
        else:
            if ax.get_legend():
                ax.get_legend().remove()
    
for row in range(len(clones)-1):
    for col in range(len(clones)-1):
        if col > row:
            ax = axes[row, col]
            ax.axis('off')

plt.suptitle(patient_id.replace('SPECTRUM-', ''), y=0.95)
plt.subplots_adjust(wspace=0.2, hspace=0.2)#, left=0.15, right=0.85, bottom=0.15, top=0.85)

```

# marginal histograms

```python
g = sns.jointplot(x='vaf_1', y='vaf_2', hue='SNV class', hue_order=snv_classes3, data=block_snvs_pairs, palette=snv_pair_utility_colors3,s=25, alpha=0.7, linewidth=0)
sns.kdeplot(ax=g.ax_joint, x='vaf_1', y='vaf_2', data=block_snvs_pairs, fill=True, bw_adjust=0.5, color='slategrey', zorder=-100, alpha=0.4, 
            clip=((0., 1.), (0., 1.)), levels=5)#, alpha=0.1, lw=1)#, clip=((-0.1, 1.1), (-0.1, 1.1)))#bw_method=.1)#
g.plot_marginals(sns.histplot)

handles, labels = g.ax_marg_x.get_legend_handles_labels()

sns.move_legend(
    g.ax_joint, 'upper left',
    bbox_to_anchor=(1.2, 1), ncol=1, frameon=False,
    title='Variant copy number (X/Y)',
    markerscale=2,
    prop={'size': 8}, title_fontsize=10,
    labelspacing=0.4, handletextpad=0, columnspacing=0.5)
```

```python
fig, axes = plt.subplots(nrows=len(clones)-1, ncols=len(clones)-1, figsize=(3.5 * (len(clones)-1), 3 * (len(clones)-1)), dpi=200, sharex=True, sharey=True)

for block1_idx, block2_idx in itertools.combinations(range(len(clones)), 2):
    block1 = clones[block1_idx]
    block2 = clones[block2_idx]

    block_snvs_pairs = get_block_snvs_pairs(blocks_adata, block1, block2)
    if block_snvs_pairs is None or block_snvs_pairs.shape[0] < 100:
        # no common cnLOH SNVs with sufficient counts
        continue
    else:
        all_empty = False
    print(block_snvs_pairs.shape)

    block_snvs_pairs = block_snvs_pairs[~block_snvs_pairs['cn'].isin(['0/0', '1/2', '2/1'])]
    block_snvs_pairs['SNV class'] = block_snvs_pairs['cn'].map(snv_pair_utility3)
    block_snvs_pairs = block_snvs_pairs[block_snvs_pairs['SNV class'] != 'Non-phylogenetic']
    
    # block_snvs_pairs = block_snvs_pairs[~(block_snvs_pairs['cn'].isin(['0/0', '2/2']))]
    
    ax = axes[block2_idx - 1, block1_idx]

    counts = block_snvs_pairs[['cn_1', 'cn_2']].value_counts().reset_index().pivot(index='cn_1', columns='cn_2').fillna(0).astype(int)[::-1]
    sns.heatmap(counts, annot=counts, fmt = 'd', ax=ax, cbar=False)

    if block1_idx == 0:
        ax.set_ylabel(f'clone {block2} variant CN', labelpad=10)
    else:
        ax.set_ylabel('')
        
    if block2_idx == len(clones) - 1:
        ax.set_xlabel(f'clone {block1} variant CN', labelpad=10)
        ax.set_xticklabels([0, 1, 2])
    else:
        ax.set_xlabel('')
    
    if include_scatter:
        if not( block1_idx == 0 and block2_idx == 1):
            if ax.get_legend():
                ax.get_legend().remove()

for row in range(len(clones)-1):
    for col in range(len(clones)-1):
        if col > row:
            ax = axes[row, col]
            ax.axis('off')
```

# subfigures with marginal histograms

```python
margin = 0.05
hist_ratio = 0.2

ylim = [-margin, 1 + margin]
xlim = [-margin, 1 + margin]
ticks = [0, 0.5, 1]

fig = plt.figure(figsize=(2 * (len(clones)-1), 2 * (len(clones)-1)), dpi=200)
subfigs = fig.subfigures(nrows=len(clones)-1, ncols=len(clones)-1)

if len(clones) == 2:
    axes = {(0,0):axes}

all_empty = True
for block1_idx, block2_idx in itertools.combinations(range(len(clones)), 2):
    block1 = clones[block1_idx]
    block2 = clones[block2_idx]

    block_snvs_pairs = get_block_snvs_pairs(blocks_adata, block1, block2)
    if block_snvs_pairs is None or block_snvs_pairs.shape[0] < 100:
        # no common cnLOH SNVs with sufficient counts
        continue
    else:
        all_empty = False
    print(block_snvs_pairs.shape)
    block_snvs_pairs = block_snvs_pairs[~block_snvs_pairs['cn'].isin(['0/0', '1/2', '2/1'])]
    block_snvs_pairs['SNV class'] = block_snvs_pairs['cn'].map(snv_pair_utility3)
    block_snvs_pairs = block_snvs_pairs[block_snvs_pairs['SNV class'] != 'Non-phylogenetic']

    # manually recreate scatterplot with marginal histograms to emulate JointPlot
    subfig = subfigs[block2_idx - 1, block1_idx]
    axes = subfig.subplots(2, 2, width_ratios = [0.9, hist_ratio], height_ratios = [hist_ratio, 0.9])

    # joint scatterplot
    sns.scatterplot(ax=axes[1][0], x='vaf_1', y='vaf_2', hue='SNV class', hue_order=snv_classes3, data=block_snvs_pairs, palette=snv_pair_utility_colors3,s=25, alpha=0.7, linewidth=0)
    sns.kdeplot(ax=axes[1][0], x='vaf_1', y='vaf_2', data=block_snvs_pairs, fill=True, bw_adjust=0.5, color='slategrey', zorder=-100, alpha=0.4, 
                clip=((0., 1.), (0., 1.)), levels=5)#, alpha=0.1, lw=1)#, clip=((-0.1, 1.1), (-0.1, 1.1)))#bw_method=.1)#
    axes[1][0].set_xlim(xlim)
    axes[1][0].set_ylim(ylim)
    axes[1][0].set_xticks(ticks)
    axes[1][0].set_yticks(ticks)

    # horizontal marginal histogram
    sns.histplot(ax=axes[0][0], x='vaf_1', hue='SNV class', hue_order=snv_classes3, data=block_snvs_pairs, bins=10,
                 palette=snv_pair_utility_colors3, legend=False)
    axes[0][0].set_xlabel('')
    axes[0][0].set_ylabel('')
    axes[0][0].set_xticklabels([])
    axes[0][0].set_xticks(axes[1][0].get_xticks())
    axes[0][0].set_xlim(xlim)
    axes[0][0].set_yticklabels([])
    sns.despine()

    # vertical marginal histogram
    sns.histplot(ax=axes[1][1], y='vaf_2', hue='SNV class', hue_order=snv_classes3, data=block_snvs_pairs, bins=10,
                 palette=snv_pair_utility_colors3, legend=False)
    axes[1][1].set_xlabel('')
    axes[1][1].set_ylabel('')
    axes[1][1].set_xticklabels([])
    axes[1][1].set_yticks(axes[1][0].get_yticks())
    axes[1][1].set_ylim(ylim)
    axes[1][1].set_yticklabels([])
    sns.despine()
    
    axes[0][1].remove()

    if block1_idx == 0 and block2_idx == 1:
        sns.move_legend(
            axes[1][0], 'upper left',
            bbox_to_anchor=(1.4, 1.05), ncol=1, frameon=False,
            title='Variant copy number (X/Y)',
            markerscale=2,
            prop={'size': 8}, title_fontsize=10,
            labelspacing=0.4, handletextpad=0, columnspacing=0.5)
    else:
        axes[1][0].get_legend().remove()

    if block1_idx == 0:
        axes[1][0].set_ylabel(f'clone {block2} VAF', labelpad=10)
    else:
        axes[1][0].set_ylabel('')
        
    if block2_idx == len(clones) - 1:
        axes[1][0].set_xlabel(f'clone {block1} VAF', labelpad=10)
    else:
        axes[1][0].set_xlabel('')

    
fig.suptitle(patient_id.replace('SPECTRUM-', ''), y=0.99)
fig.subplots_adjust(wspace=0.15, hspace=0.15)#, left=0.15, right=0.85, bottom=0.15, top=0.85)

```

```python

```

```python

```

# try to get colorbars in there

<!-- #raw -->
include_scatter = False

margin = 0.1
xlim = (-margin, 1 + margin)
ylim = (-margin, 1 + margin)

fig = plt.figure(figsize=(3.5 * (len(clones)-1), 3 * (len(clones)-1)), dpi=200)
spec = mpl.gridspec.GridSpec(ncols=len(clones)-1, nrows=len(clones)-1, figure=fig)
top_axis = fig.add_subplot(spec[0, :2])
axes = {}
for i in range(1, len(clones) - 1):
    for j in range(len(clones) - 1):
        axes[i, j] = fig.add_subplot(spec[i, j])

if len(clones) == 2:
    axes = {(0,0):axes}

all_empty = True
for block1_idx, block2_idx in itertools.combinations(range(len(clones)), 2):
    block1 = clones[block1_idx]
    block2 = clones[block2_idx]

    block_snvs_pairs = get_block_snvs_pairs(blocks_adata, block1, block2)
    if block_snvs_pairs is None or block_snvs_pairs.shape[0] < 100:
        # no common cnLOH SNVs with sufficient counts
        continue
    else:
        all_empty = False
    print(block_snvs_pairs.shape)

    block_snvs_pairs['SNV class'] = block_snvs_pairs['cn'].map(snv_pair_utility2)
    # block_snvs_pairs = block_snvs_pairs[~(block_snvs_pairs['cn'].isin(['0/0', '2/2']))]

    if block1_idx == 0 and block2_idx == 1:
        ax = top_axis
    else:
        ax = axes[block2_idx - 1, block1_idx]

    # original density plot
    '''
    with mpl.rc_context({'lines.linewidth': 0.25}):
        sns.kdeplot(ax=ax, x='vaf_1', y='vaf_2', data=block_snvs_pairs, fill=True, bw_adjust=0.5, color='slategrey', zorder=-100, alpha=0.4, clip=((0., 1.), (0., 1.)))#, alpha=0.1, lw=1)#, clip=((-0.1, 1.1), (-0.1, 1.1)))#bw_method=.1)#
    '''        
    hp = sns.histplot(ax=ax, x='vaf_1', y='vaf_2', data=block_snvs_pairs, fill=True, zorder=-100, bins=20,
                      hue='SNV class', hue_order=snv_classes2, palette=snv_pair_utility_colors2,
                     cbar = (block1_idx == 0 and block2_idx == 1))#, cbar_ax = axes[0][1])
    #hb = ax.hexbin(block_snvs_pairs.vaf_1, block_snvs_pairs.vaf_2, gridsize=8, cmap = 'Greys')
    #plt.colorbar(hb)

    if include_scatter:
        sns.scatterplot(ax=ax, x='vaf_1', y='vaf_2', hue='SNV class', hue_order=snv_classes2, data=block_snvs_pairs, s=10, alpha=1., linewidth=0.2, 
                        palette=snv_pair_utility_colors2, edgecolor='k')

    if block1_idx == 0:
        ax.set_ylabel(f'clone {block2} VAF', labelpad=10)
        
    if block2_idx == len(clones) - 1:
        ax.set_xlabel(f'clone {block1} VAF', labelpad=10)

    ax.tick_params(axis='y', labelrotation=0)
  
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
 
    ax.set_xticks([0, 0.5, 1])
    ax.set_yticks([0, 0.5, 1])
    
    sns.despine(ax=ax, trim=True, offset=5)

    if block1_idx == 0 and block2_idx == 1:
        sns.move_legend(
            ax, 'upper left',
            bbox_to_anchor=(2.2, 1), ncol=1, frameon=False,
            title='Variant copy number (X/Y)',
            markerscale=2,
            prop={'size': 8}, title_fontsize=10,
            labelspacing=0.4, handletextpad=0, columnspacing=0.5)
    else:
        if ax.get_legend():
            ax.get_legend().remove()
    ddd
for row in range(len(clones)-1):
    for col in range(len(clones)-1):
        if col > row and row != 0:
            ax = axes[row, col]
            ax.axis('off')

plt.suptitle(patient_id.replace('SPECTRUM-', ''), y=0.95)
plt.subplots_adjust(wspace=0.2, hspace=0.2)#, left=0.15, right=0.85, bottom=0.15, top=0.85)

<!-- #endraw -->

```python
hp
```

```python

```
