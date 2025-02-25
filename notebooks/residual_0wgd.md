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
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import vetica.mpl

import spectrumanalysis.dataload
import spectrumanalysis.plots

project_dir = os.environ['SPECTRUM_PROJECT_DIR']

```

```python

cell_info = pd.read_csv(f'{project_dir}/preprocessing/summary/filtered_cell_table.csv.gz')
cell_info = cell_info[cell_info['include_cell']]

fraction_wgd = pd.read_csv('../../../../annotations/fraction_wgd_class.csv')
cell_info = cell_info.merge(fraction_wgd)
cell_info = spectrumanalysis.wgd.classify_subclonal_wgd(cell_info)

n_wgd = cell_info.query('wgd_class == "WGD-high"').groupby(['patient_id', 'n_wgd']).size().unstack(fill_value = 0)
n_wgd[(n_wgd[0] > 0) & (n_wgd[1] > 0)]

```

```python

n_wgd_fraction = (n_wgd.T / n_wgd.sum(axis=1)).T
n_wgd_fraction[(n_wgd[0] > 0) & (n_wgd[1] > 0)]

```

```python

cell_info2 = cell_info[~cell_info['multipolar']]

n_wgd = cell_info2.query('wgd_class == "WGD-high"').groupby(['patient_id', 'n_wgd']).size().unstack(fill_value = 0)
n_wgd[(n_wgd[0] > 0) & (n_wgd[1] > 0)]

```

```python

n_wgd_fraction = (n_wgd.T / n_wgd.sum(axis=1)).T
n_wgd_fraction[(n_wgd[0] > 0) & (n_wgd[1] > 0)]

```

```python

patient_ids = [
    'SPECTRUM-OV-045',
    'SPECTRUM-OV-051',
    'SPECTRUM-OV-075',
    'SPECTRUM-OV-087',
    'SPECTRUM-OV-107',
    'SPECTRUM-OV-110',
]

for patient_id in patient_ids:
    adata = spectrumanalysis.dataload.load_filtered_cna_adata(project_dir, patient_id)
    adata.var['state_pseudobulk'] = np.nanmedian(adata.layers['state'], axis=0)
    
    cell_ids = adata.obs.query('n_wgd == 0').index[:5]
    
    fig, axes = plt.subplots(
        nrows=len(cell_ids)+1, figsize=(6, 1.2 + 1.2 * len(cell_ids)), dpi=300, sharex=True, sharey=True)
    
    ax = axes[0]
    spectrumanalysis.plots.pretty_pseudobulk_tcn(adata[:, adata.var['has_allele_cn']], fig=fig, ax=ax, rasterized=True)
    ax.set_ylabel('CN')
    ax.set_title('Pseudobulk', fontsize=8)
    ax.get_legend().remove()
    
    for ax, cell_id in zip(axes[1:], cell_ids):
        spectrumanalysis.plots.pretty_cell_tcn(adata[:, adata.var['has_allele_cn']], cell_id, fig=fig, ax=ax, rasterized=True)
        ax.set_ylabel('CN')
        ax.set_title(f'Cell {cell_id}', fontsize=8)
        ax.get_legend().remove()
    
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.75)
    
    fig.savefig(f'../../../../figures/supplementary/residual_0xwgd_{patient_id}.png', bbox_inches='tight', metadata={'Date': None})

```

```python

```

```python

```
