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
import os
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import pandas as pd
import numpy as np
from IPython.display import Image
import anndata as ad
import scgenome
from matplotlib.patches import Rectangle
import yaml
import anndata as ad
from Bio import Phylo
import json
from collections import Counter
from matplotlib import colors as mcolors
from scipy.spatial.distance import pdist, squareform
import Bio
from copy import deepcopy
import shutil
from scipy.sparse import load_npz
import matplotlib.colors as mcolors
import warnings
from time import perf_counter

from scipy.stats import binom, gaussian_kde, poisson
import scipy
from multiprocessing import Pool
from sklearn.metrics import confusion_matrix
from scipy.cluster.hierarchy import linkage, fcluster
import itertools
import math
import warnings
import tqdm
import vetica.mpl
```

```python
pipeline_outputs = '/data1/shahs3/users/myersm2/repos/spectrum_wgd_data5'
```

```python
snvs = pd.read_csv(os.path.join(pipeline_outputs, 'tree_snv/outputs/SPECTRUM-OV-025_general_snv_tree_assignment.csv'))
tree = pickle.load(open(os.path.join(pipeline_outputs, 'tree_snv/inputs/SPECTRUM-OV-025_clones_pruned.pickle'), 'rb'))
clustered_adata = ad.read_h5ad(os.path.join(pipeline_outputs, 'tree_snv/inputs/SPECTRUM-OV-025_cna_clustered.h5'))
```

```python
clustered_adata.obs
```

```python
Phylo.draw(tree)
```

```python
cntr_i1 = snvs[(snvs.leaf == 'clone_0') & (snvs.clade == 'internal_1') & snvs.is_cpg].wgd_timing.value_counts()
```

```python
cntr_i1
```

```python
cntr_c3 = snvs[(snvs.leaf == 'clone_0') & (snvs.clade == 'clone_3') & snvs.is_cpg].wgd_timing.value_counts()
```

```python
cntr_c3
```

## uncorrected timing values

```python
 2 * cntr_i1['prewgd'] / (2 * cntr_i1['prewgd'] + cntr_i1['postwgd'])
```

```python
 2 * cntr_c3['prewgd'] / (2 * cntr_c3['prewgd'] + cntr_c3['postwgd'])
```

## corrected timing vlaues

```python
def logistic_curve(x, k=0.62355594, x0=4.1070332):
    return 0.92 / (1 + np.exp(-1 * k * (x - x0)))

```

```python
cntr_i1 / logistic_curve(clustered_adata.obs.haploid_depth.loc['0'] + clustered_adata.obs.haploid_depth.loc['1'])
```

```python
cntr_c3 / logistic_curve(clustered_adata.obs.haploid_depth.loc['3'])
```

## bootstrap timing from root to WGD

```python
my_snvs = snvs[(snvs.leaf == 'clone_0') & (snvs.clade == 'internal_1') & snvs.is_cpg]
my_prop = (my_snvs.wgd_timing == 'prewgd').mean()
my_n = len(my_snvs)
my_corr_factor =  logistic_curve(clustered_adata.obs.haploid_depth.loc['0'] + clustered_adata.obs.haploid_depth.loc['1'])

sim_prewgd = binom.rvs(p=my_prop, n=my_n, size=10000)
sim_postwgd = my_n - sim_prewgd
sim_timing =  2 * (sim_prewgd / my_corr_factor) / (2 * (sim_prewgd / my_corr_factor) + (sim_postwgd / my_corr_factor))
internal1_ci = {'timing':(np.percentile(sim_timing, 5), np.percentile(sim_timing, 95)),
                 'prewgd':(np.percentile(sim_prewgd/my_corr_factor, 5), np.percentile(sim_prewgd/my_corr_factor, 95)),
                 'postwgd':(np.percentile(sim_postwgd/my_corr_factor, 5), np.percentile(sim_postwgd/my_corr_factor, 95))}
internal1_ci
```

```python
my_snvs = snvs[(snvs.leaf == 'clone_0') & (snvs.clade == 'clone_3') & snvs.is_cpg]
my_prop = (my_snvs.wgd_timing == 'prewgd').mean()
my_n = len(my_snvs)
my_corr_factor =  logistic_curve(clustered_adata.obs.haploid_depth.loc['3'])

sim_prewgd = binom.rvs(p=my_prop, n=my_n, size=10000)
sim_postwgd = my_n - sim_prewgd
sim_timing =  2 * (sim_prewgd / my_corr_factor) / (2 * (sim_prewgd / my_corr_factor) + (sim_postwgd / my_corr_factor))
clone3_ci = {'timing':(np.percentile(sim_timing, 5), np.percentile(sim_timing, 95)),
                 'prewgd':(np.percentile(sim_prewgd/my_corr_factor, 5), np.percentile(sim_prewgd/my_corr_factor, 95)),
                 'postwgd':(np.percentile(sim_postwgd/my_corr_factor, 5), np.percentile(sim_postwgd/my_corr_factor, 95))}
clone3_ci
```

```python
plt.figure(dpi = 300, figsize=(8, 2))
plt.plot(internal1_ci['prewgd'], [0, 0],  marker='|', c = 'k')

plt.plot(clone3_ci['prewgd'], [1, 1],  marker='|', c = 'k')
#plt.ylim(0, 80)
plt.yticks([0, 1], ['parent of clones 0/1', 'clone 3'])
plt.xlabel("Corrected CpG C>T SNVs from bifurcation to WGD")
plt.scatter([cntr_i1.prewgd / logistic_curve(clustered_adata.obs.haploid_depth.loc['0'] + clustered_adata.obs.haploid_depth.loc['1']),
             cntr_c3.prewgd / logistic_curve(clustered_adata.obs.haploid_depth.loc['3'])], [0, 1],
           c = 'k')
plt.xticks(np.arange(5, 35, 5), np.arange(5, 35, 5))
plt.ylim(-0.5, 1.5)
sns.despine()
```

# plot doubletime tree vertically

```python

```

```python

```

```python

```

<!-- #raw -->
# try including trunk
<!-- #endraw -->

<!-- #raw -->
n_truncal =  ((snvs.leaf == 'clone_0') & snvs.clade.isin(['internal_0']) & snvs.is_cpg).sum()
<!-- #endraw -->

<!-- #raw -->
my_snvs = snvs[(snvs.leaf == 'clone_0') & (snvs.clade == 'internal_1') & snvs.is_cpg]
my_corr_factor =  logistic_curve(clustered_adata.obs.haploid_depth.loc['0'] + clustered_adata.obs.haploid_depth.loc['1'])

my_pre_wgd = (my_snvs.wgd_timing == 'prewgd').sum() / my_corr_factor
my_post_wgd = (my_snvs.wgd_timing == 'postwgd').sum() / my_corr_factor

my_n = int(my_pre_wgd + n_truncal + my_post_wgd)
my_prop = (my_pre_wgd + n_truncal) / my_n

sim_total = binom.rvs(p=my_prop, n=my_n, size=10000)
internal1_mean = np.mean(sim_total)
internal1_ci = (np.percentile(sim_total, 5), np.percentile(sim_total, 95))
internal1_ci
<!-- #endraw -->

<!-- #raw -->
my_snvs = snvs[(snvs.leaf == 'clone_0') & (snvs.clade == 'clone_3') & snvs.is_cpg]
my_corr_factor =  logistic_curve(clustered_adata.obs.haploid_depth.loc['3'])

my_pre_wgd = (my_snvs.wgd_timing == 'prewgd').sum() / my_corr_factor
my_post_wgd = (my_snvs.wgd_timing == 'postwgd').sum() / my_corr_factor

my_n = int(my_pre_wgd + n_truncal + my_post_wgd)
my_prop = (my_pre_wgd + n_truncal) / my_n

sim_total = binom.rvs(p=my_prop, n=my_n, size=10000)
clone3_mean = np.mean(sim_total)
clone3_ci = (np.percentile(sim_total, 5), np.percentile(sim_total, 95))
clone3_ci
<!-- #endraw -->

<!-- #raw -->
plt.figure(dpi = 200, figsize=(3,4))
plt.plot([0, 0], internal1_ci, marker='_')

plt.plot([1, 1], clone3_ci, marker='_')
#plt.ylim(0, 80)
plt.xticks([0, 1], ['internal_1', 'clone_3'])
plt.ylabel("Corrected num. SNVs from root to WGD")
plt.scatter([0, 1], [internal1_mean, clone3_mean],
           c = [plt.get_cmap('tab10')(x) for x in [0, 1]])
plt.xlim(-0.5, 1.5)
<!-- #endraw -->

<!-- #raw -->
# try without correction
<!-- #endraw -->

<!-- #raw -->
my_snvs = snvs[(snvs.leaf == 'clone_0') & (snvs.clade == 'internal_1') & snvs.is_cpg]
my_corr_factor =  1

my_pre_wgd = (my_snvs.wgd_timing == 'prewgd').sum() / my_corr_factor
my_post_wgd = (my_snvs.wgd_timing == 'postwgd').sum() / my_corr_factor

my_n = int(my_pre_wgd + n_truncal + my_post_wgd)
my_prop = (my_pre_wgd + n_truncal) / my_n

sim_total = binom.rvs(p=my_prop, n=my_n, size=10000)
internal1_mean = np.mean(sim_total)
internal1_ci = (np.percentile(sim_total, 5), np.percentile(sim_total, 95))
internal1_ci
<!-- #endraw -->

<!-- #raw -->
my_snvs = snvs[(snvs.leaf == 'clone_0') & (snvs.clade == 'clone_3') & snvs.is_cpg]
my_corr_factor =  1

my_pre_wgd = (my_snvs.wgd_timing == 'prewgd').sum() / my_corr_factor
my_post_wgd = (my_snvs.wgd_timing == 'postwgd').sum() / my_corr_factor

my_n = int(my_pre_wgd + n_truncal + my_post_wgd)
my_prop = (my_pre_wgd + n_truncal) / my_n

sim_total = binom.rvs(p=my_prop, n=my_n, size=10000)
clone3_mean = np.mean(sim_total)
clone3_ci = (np.percentile(sim_total, 5), np.percentile(sim_total, 95))
clone3_ci
<!-- #endraw -->

<!-- #raw -->
plt.figure(dpi = 200, figsize=(3,4))
plt.plot([0, 0], internal1_ci, marker='_')

plt.plot([1, 1], clone3_ci, marker='_')
#plt.ylim(0, 80)
plt.xticks([0, 1], ['internal_1', 'clone_3'])
plt.ylabel("Num. CpG SNVs from root to WGD")
plt.scatter([0, 1], [internal1_mean, clone3_mean],
           c = [plt.get_cmap('tab10')(x) for x in [0, 1]])
plt.xlim(-0.5, 1.5)
<!-- #endraw -->
