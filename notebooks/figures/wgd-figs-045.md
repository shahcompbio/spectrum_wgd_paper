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
import anndata as ad
import numpy as np
import os
import itertools
import matplotlib.pyplot as plt
from scipy.stats import binom
import seaborn as sns
import scipy
import tqdm
import pickle
import Bio

import scgenome
import vetica.mpl

patient_id = 'SPECTRUM-OV-045'

project_dir = os.environ['SPECTRUM_PROJECT_DIR']

blocks_adata = ad.read(f'{project_dir}/tree_snv/inputs/{patient_id}_general_clone_adata.h5')
blocks_adata.var['is_cnloh'] = blocks_adata.var['snv_type'] == '2:0'

tree_filename = f'{project_dir}/tree_snv/inputs/{patient_id}_clones_pruned.pickle'

plt.figure(figsize=(4, 1.5))
Bio.Phylo.draw(pickle.load(open(tree_filename, 'rb')), axes=plt.gca())

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

    block_snvs_pairs = block1_snvs.merge(block2_snvs, on=['chromosome', 'position', 'ref', 'alt'], suffixes=['_1', '_2'])

    g1, g2, p1, p2 = compute_ml_genotypes_vec(
        block_snvs_pairs['alt_count_1'], block_snvs_pairs['total_count_1'],
        block_snvs_pairs['alt_count_2'], block_snvs_pairs['total_count_2'],
        return_genotypes=True)

    block_snvs_pairs['wgd1_1'] = np.array(g1)[:, 0]
    block_snvs_pairs['wgd1_2'] = np.array(g1)[:, 1]

    block_snvs_pairs['wgd2_1'] = np.array(g2)[:, 0]
    block_snvs_pairs['wgd2_2'] = np.array(g2)[:, 1]

    block_snvs_pairs['p_wgd1'] = p1
    block_snvs_pairs['p_wgd2'] = p2

    block_snvs_pairs['cn'] = block_snvs_pairs['cn_1'].astype(str) + '/' + block_snvs_pairs['cn_2'].astype(str)
    
    return block_snvs_pairs

```

```python

blocks_adata = blocks_adata[blocks_adata.obs.index != 'nan']
blocks_adata = blocks_adata[blocks_adata.obs.index.astype(int) >= 0]

clones = blocks_adata.obs.index

fig, axes = plt.subplots(nrows=len(clones)-1, ncols=len(clones)-1, figsize=(5, 5), dpi=150)

for block1_idx, block2_idx in itertools.combinations(range(len(clones)), 2):
    block1 = clones[block1_idx]
    block2 = clones[block2_idx]

    block_snvs_pairs = get_block_snvs_pairs(blocks_adata, block1, block2)
    
    ax = axes[block1_idx, block2_idx-1]
    plot_data = block_snvs_pairs.groupby(['cn_1', 'cn_2']).size().unstack(fill_value=0).iloc[::-1, :]
    vmax = max(plot_data.loc[0, 2], plot_data.loc[1, 1], plot_data.loc[2, 0])
    # plot_data.loc[0, 0] = np.NaN
    # plot_data.loc[2, 2] = np.NaN

    sns.heatmap(ax=ax, data=plot_data, vmax=vmax, cbar=False, annot=True, fmt='d', annot_kws={"fontsize":8})

    if block1 == clones[0]:
        ax.set_xlabel(f'clone {block2}')
        ax.xaxis.set_label_position('top')
    else:
        ax.set_xlabel('')
    if block2 == clones[-1]:
        ax.set_ylabel(f'clone {block1}', labelpad=15)
        ax.yaxis.set_label_position('right')
        ax.yaxis.label.set(rotation=270)
    else:
        ax.set_ylabel('')
    ax.tick_params(axis='y', labelrotation=0)
        
    if int(block1) != int(block2)-1:
        ax.set_xticks([])
        ax.set_yticks([])
    
for row in range(blocks_adata.obs.shape[0]-1):
    for col in range(blocks_adata.obs.shape[0]-1):
        ax = axes[row, col]
        if row > col:
            ax.set_xticks([])
            ax.set_yticks([])
            ax.axis('off')

plt.subplots_adjust(wspace=0.1, hspace=0.1)

```

```python

block_snvs_pairs = get_block_snvs_pairs(blocks_adata, '3', '0')

plt.figure(figsize=(2, 2))
sns.histplot(x='vaf_1', data=block_snvs_pairs[(block_snvs_pairs['cn_2'] == 0) & (block_snvs_pairs['cn_1'] != 0)], bins=21)
sns.despine()

plt.figure(figsize=(2, 2))
sns.histplot(x='vaf_2', data=block_snvs_pairs[(block_snvs_pairs['cn_1'] == 0) & (block_snvs_pairs['cn_2'] != 0)], bins=21)
sns.despine()

```

```python

block_snvs_pairs['min_total_count'] = block_snvs_pairs[['total_count_1', 'total_count_2']].min(axis=1)
sns.scatterplot(x='vaf_1', y='vaf_2', data=block_snvs_pairs[(block_snvs_pairs['cn'] != '0/0') & (block_snvs_pairs['cn'] != '2/2')], size='min_total_count', hue='cn')
sns.move_legend(plt.gca(), 'upper left', bbox_to_anchor=(1, 1), ncol=1, title=None, frameon=False)
sns.despine(trim=True)

```

```python

plot_data = get_block_snvs_pairs(blocks_adata, '3', '0')

sns.kdeplot(x='vaf_1', y='vaf_2', data=plot_data, clip=((0, 1), (0, 1)), fill=True)#, hue='cn')
sns.despine(trim=True)

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
    '0/0 (uninformative)': '#8B5E3C',
    '0/1 (uninformative)': '#D7DF23',
    '1/0 (uninformative)': '#F7941D',
    '2/2 (uninformative)': '#2E3192',
    '2/0 (independent)': '#EF4136',
    '0/2 (independent)': '#006838',
    '1/1 (common)': '#27AAE1',
    '1/2 (contradictory)': '#8B5E3C',
    '2/1 (contradictory)': '#8B5E3C',
}

block_snvs_pairs['cn'].map(snv_pair_utility)

```

```python

import matplotlib as mpl

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

fig, axes = plt.subplots(nrows=len(clones)-1, ncols=len(clones)-1, figsize=(1.5 * (len(clones)-1), 1.5 * (len(clones)-1)), dpi=150, sharex=True, sharey=True)

for block1_idx, block2_idx in itertools.combinations(range(len(clones)), 2):
    block1 = clones[block1_idx]
    block2 = clones[block2_idx]

    block_snvs_pairs = get_block_snvs_pairs(blocks_adata, block1, block2)
    block_snvs_pairs['SNV class'] = block_snvs_pairs['cn'].map(snv_pair_utility)
    # block_snvs_pairs = block_snvs_pairs[~(block_snvs_pairs['cn'].isin(['0/0', '2/2']))]
    
    # block1_idx, block2_idx = sorted([int(block1), int(block2)])
    ax = axes[block2_idx - 1, block1_idx]
    with mpl.rc_context({'lines.linewidth': 0.25}):
        sns.kdeplot(ax=ax, x='vaf_1', y='vaf_2', data=block_snvs_pairs, fill=True, bw_adjust=0.5, color='slategrey', zorder=-100, alpha=0.4, clip=((0., 1.), (0., 1.)))#, alpha=0.1, lw=1)#, clip=((-0.1, 1.1), (-0.1, 1.1)))#bw_method=.1)#
    sns.scatterplot(ax=ax, x='vaf_1', y='vaf_2', hue='SNV class', hue_order=snv_classes, data=block_snvs_pairs, s=10, alpha=1., linewidth=0, palette=snv_pair_utility_colors)

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
            bbox_to_anchor=(2.2, 1.2), ncol=1, frameon=False,
            title='Copy Number (A/B)',
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

plt.subplots_adjust(wspace=0.2, hspace=0.2)
plt.suptitle(patient_id.replace('SPECTRUM-', ''), y=1.)

fig.savefig(f'../../../../figures/figure2/indep_wgd_{patient_id}.svg', metadata={'Date': None})

```


## Better contour plot if required


```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

block_snvs_pairs = get_block_snvs_pairs(blocks_adata, '1', '2')
block_snvs_pairs = block_snvs_pairs[~(block_snvs_pairs['cn'].isin(['0/0', '2/2']))]

data = block_snvs_pairs[['vaf_1', 'vaf_2']].T.values

covariance = np.array([
    [0.01,  0],
    [0, 0.01]])
bandwidth = 0.1

class gaussian_kde_set_covariance(gaussian_kde):
    def __init__(self, dataset, covariance):
        self.covariance = covariance
        scipy.stats.gaussian_kde.__init__(self, dataset)

    def _compute_covariance(self):
        self._norm_factor = np.sqrt(2 * np.pi * self.covariance) * self.n
        self._data_cho_cov = scipy.linalg.cholesky(self.covariance, lower=True)
        self.cho_cov = self._data_cho_cov.astype(np.float64)

    @property
    def inv_cov(self):
        return 1.0 / self.covariance
    
kde = gaussian_kde_set_covariance(data, covariance)

print(kde.covariance)

# Create a grid of points where we want to evaluate the KDE
xmin, xmax = -0.05, 1.05
ymin, ymax = -0.05, 1.05
X, Y = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
positions = np.vstack([X.ravel(), Y.ravel()])

# Evaluate the KDE on the grid
Z = np.reshape(kde(positions).T, X.shape)

# Make the contour plot
plt.figure(figsize=(8, 6))
plt.contourf(X, Y, Z, cmap='Greys')
plt.colorbar(label='Density')
plt.title('KDE Contour Plot with Custom Bandwidth')
plt.xlabel('X axis')
plt.ylabel('Y axis')
plt.show()

```

```python

```
