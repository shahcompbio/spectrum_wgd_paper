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
import matplotlib.pyplot as plt
import numpy as np
import vetica.mpl

import spectrumanalysis.dataload
import spectrumanalysis.plots

project_dir = os.environ['SPECTRUM_PROJECT_DIR']

```

```python

patient_id = 'SPECTRUM-OV-004'

adata = spectrumanalysis.dataload.load_filtered_cna_adata(project_dir, patient_id)

```

```python

cell_id = 'SPECTRUM-OV-004_S1_LEFT_ADNEXA-A96121B-R44-C67'

# cell_id = adata.obs.query('multipolar == True').index[17]

```

```python
adata[cell_id].layers['state'].toarray()[0, :]
```

```python

adata.var['state_pseudobulk'] = np.nanmedian(adata.layers['state'], axis=0)

adata.var['state_diff'] = adata.var['state_pseudobulk'].values != adata[cell_id].layers['state'].toarray()[0, :]

```

```python

from scgenome.plotting.cn import genome_axis_plot, setup_genome_xaxis_ticks

def plot_shaded(adata, var_col, ax):
    plot_data = adata.var.copy()

    def shade_regions(data, ax=None):
        if ax is None:
            ax = plt.gca()
        for start, end in data.query(var_col)[['start', 'end']].values:
            ax.fill_between([start, end], -0.05, 100, color='0.95')

    genome_axis_plot(
        plot_data,
        shade_regions,
        ('start', 'end'),
        ax=ax)

```

```python

fig, axes = plt.subplots(
    nrows=2, figsize=(6, 3), dpi=300, sharex=True, sharey=True)

ax = axes[0]
plot_shaded(adata, 'state_diff', ax)
spectrumanalysis.plots.pretty_cell_tcn(adata[:, adata.var['has_allele_cn']], cell_id, fig=fig, ax=ax, rasterized=True)
ax.set_ylabel('Divergent\ncell CN')

ax = axes[1]
spectrumanalysis.plots.pretty_pseudobulk_tcn(adata[:, adata.var['has_allele_cn']], fig=fig, ax=ax, rasterized=True)
ax.set_ylabel('Pseudobulk\nCN')

fig.savefig('../../../../figures/figure3/example_multipolar.svg', bbox_inches='tight', metadata={'Date': None})

```

```python

```
