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
import numpy as np
import scgenome
import Bio
import pickle
import panel as pn
import json
import anndata as ad
import tqdm
import yaml
import seaborn as sns

pn.extension()

import anndata as ad
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import spectrumanalysis.dataload
import spectrumanalysis.phylocn
import spectrumanalysis.plots


import vetica.mpl

chromosomes = list(str(a) for a in range(1, 23)) + ['X']
scgenome.refgenome.set_genome_version('hg19', chromosomes=chromosomes, plot_chromosomes=chromosomes)

project_dir = os.environ['SPECTRUM_PROJECT_DIR']

colors_dict = yaml.safe_load(open('../../../../config/colors.yaml', 'r'))

fraction_wgd = pd.read_csv('../../../../annotations/fraction_wgd_class.csv')

```

```python


def plot_wgd_history(wgd_ar_info, wgd_clade, patient_id, title=False, ylabels=False):
    wgd_clades = wgd_ar_info['wgd_clades'].set_index('name')

    adata = wgd_ar_info['adata']
    adata.layers['state'] = adata.layers['cn_a_2'] + adata.layers['cn_b_2']

    wgd_changes = wgd_ar_info['wgd_changes']
    wgd_changes.layers['state'] = wgd_changes.layers['cn_a'] + wgd_changes.layers['cn_b']
    wgd_changes.layers['state_pre'] = wgd_changes.layers['cn_a_pre'] + wgd_changes.layers['cn_b_pre']
    wgd_changes.layers['state_post'] = 2 * wgd_changes.layers['state_pre']

    fig, axes = plt.subplots(
        nrows=4, ncols=1,
        height_ratios=[3, 3, 3, 3],
        figsize=(6, 3.5), dpi=150, sharex=True, sharey=False)

    ax = axes[0]
    spectrumanalysis.plots.plot_cn_rect(
        adata,
        obs_id=wgd_clades.loc[wgd_clade, 'parent'],
        ax=ax,
    )
    ax.set_xlabel('')
    ax.set_ylim((-0.5, 7.5))
    ax.set_yticks([0, 5])
    ax.set_yticks([0, 1, 2, 3, 4, 5, 6, 7], minor=True)
    ax.grid(ls=':', lw=0.5, zorder=0, which='both', axis='y')
    if ylabels:
        ax.set_ylabel('nWGD clone', rotation=0, ha='right', va='center')

    wgd_clade_id = wgd_clade + '_pre'
    ax = axes[1]
    spectrumanalysis.plots.plot_cn_rect(
        wgd_changes,
        obs_id=wgd_clade_id,
        ax=ax,
        y='state',
        cmap='coolwarm',
        vmin=-2,
        vmax=2,
    )
    ax.set_xlabel('')
    ax.spines['bottom'].set_visible(False)
    if ylabels:
        ax.set_ylabel('Pre-WGD changes', rotation=0, ha='right', va='center')
    ax.set_ylim((-2.5, 2.5))
    ax.set_yticks([-2, 0, 2])
    ax.set_yticklabels([-2, 0, 2])
    ax.set_yticks([-2, -1, 0, 1, 2], minor=True)
    ax.grid(ls=':', lw=0.5, zorder=0, which='both', axis='y')
    ax.tick_params(length=0, which='both', axis='x')

    wgd_clade_id = wgd_clade + '_post'
    ax = axes[2]
    spectrumanalysis.plots.plot_cn_rect(
        wgd_changes,
        obs_id=wgd_clade_id,
        ax=ax,
        y='state',
        cmap='coolwarm',
        vmin=-2,
        vmax=2,
    )
    ax.set_xlabel('')
    ax.spines['bottom'].set_visible(False)
    if ylabels:
        ax.set_ylabel('Post-WGD changes', rotation=0, ha='right', va='center')
    ax.set_ylim((-2.5, 2.5))
    ax.set_yticks([-2, 0, 2])
    ax.set_yticklabels([-2, 0, 2])
    ax.set_yticks([-2, -1, 0, 1, 2], minor=True)
    ax.grid(ls=':', lw=0.5, zorder=0, which='both', axis='y')
    ax.tick_params(length=0, which='both', axis='x')

    ax = axes[3]
    spectrumanalysis.plots.plot_cn_rect(
        adata,
        obs_id=wgd_clade,
        ax=ax,
        #rect_kws=dict(zorder=3),
    )
    ax.set_xlabel('Chromosome')
    ax.set_ylim((-0.5, 7.5))
    ax.set_yticks([0, 5])
    ax.set_yticks([0, 1, 2, 3, 4, 5, 6, 7], minor=True)
    ax.grid(ls=':', lw=0.5, zorder=0, which='both', axis='y')
    if ylabels:
        ax.set_ylabel('WGD clone', rotation=0, ha='right', va='center')

    plt.subplots_adjust(hspace=0.5)
    if title:
        plt.suptitle(patient_id, y=1.05)

    scgenome.plotting.cn.setup_genome_xaxis_ticks(
        ax, chromosome_names=dict(zip(
            [str(a) for a in range(1, 23)] + ['X'],
            ['1', '2', '3', '4', '5', '6', '7', '8', '9', '', '11', '', '13', '', '15', '', '', '18', '', '', '21', '', 'X'])))

    return fig


```

```python

patient_id = 'SPECTRUM-OV-045'

wgd_ar_info = spectrumanalysis.dataload.load_wgd_ar(
    project_dir,
    patient_id,
    use_sankoff_ar=True,
    diploid_parent=False,
    wgd_clade_cell_threshold=10)

wgd_clade = wgd_ar_info['wgd_clades']['name'].values[2]

fig = plot_wgd_history(wgd_ar_info, wgd_clade, patient_id)

fig.savefig(f'../../../../figures/figure4/patient_{patient_id}_wgd_history.svg', bbox_inches='tight', metadata={'Date': None})

```

```python

```
