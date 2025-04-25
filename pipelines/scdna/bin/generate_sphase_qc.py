#!/juno/work/shah/users/mcphera1/repos/spectrumanalysis/analysis/notebooks/scdna/venv/bin/python

# TODO
###!/usr/bin/env python

import click
import scipy
import pandas as pd
import anndata as ad
import numpy as np
import seaborn as sns

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import AutoMinorLocator
from PIL import Image
from tqdm import tqdm
from collections import OrderedDict

import spectrumanalysis.changepoints

import scgenome


def threshold_plot(adata, obs_column, ascending=True):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 10), dpi=150, sharey=True)
    
    cell_ids = adata.obs.sort_values(obs_column, ascending=ascending).head(200).index
    
    cell_ids = cell_ids[::-1]

    g = scgenome.pl.plot_cell_cn_matrix(
        adata[cell_ids],
        layer_name='state',
        raw=False,
        ax=ax1,
    )

    rt_plot_data = g['adata'].obs

    ax2.barh(y=rt_plot_data.index, width=rt_plot_data[obs_column], height=1, color='0.75')
    ax2.invert_yaxis()
    ax2.xaxis.set_minor_locator(AutoMinorLocator())
    ax2.grid(visible=True, which='major', axis='x')
    ax2.grid(visible=True, which='minor', axis='x', linestyle=':', linewidth=0.5)
    ax2.tick_params(which='both', width=2)
    ax2.tick_params(which='major', length=7)
    ax2.tick_params(which='minor', length=4)
    ax2.set_xlabel(obs_column)


def write_qc_pdf(adata, pdf_filename):
    pdf = PdfPages(pdf_filename)
    plt.gray()

    # Feature map to ascending/descending
    features = OrderedDict(
        median_changepoint_prevalence=True,
        state_rt_spearman=False,
        n_brk=False,
        n_uniq_brk=False,
    )
    
    # pairplot of each feature
    # 

    g = sns.PairGrid(adata.obs[list(features.keys())], diag_sharey=False)
    g.map_upper(sns.scatterplot, s=15)
    g.map_lower(sns.kdeplot)
    g.map_diag(sns.kdeplot, lw=2)
    pdf.savefig()
    plt.close()

    # threshold plots of each feature
    # 

    for feature, ascending in features.items():
        threshold_plot(adata, feature, ascending=ascending)
        pdf.savefig()
        plt.close()

    # plot cells
    #

    plotted_cell_ids = set()
    for feature, ascending in features.items():
        cell_ids = adata.obs.sort_values(feature, ascending=ascending).head(50).index

        for cell_id in cell_ids:
            if cell_id in plotted_cell_ids:
                continue

            feature_val = adata.obs.loc[cell_id, feature]

            plt.figure(figsize=(10, 2))
            scgenome.pl.plot_cn_profile(
                adata,
                cell_id,
                value_layer_name='copy',
                state_layer_name='state',
                squashy=True,
            )
            plt.title(f'{cell_id}\n{feature}={feature_val}')
            pdf.savefig(bbox_inches='tight')
            plt.close()

            plotted_cell_ids.add(cell_id)

    pdf.close()


@click.command()
@click.argument('adata_filename')
@click.argument('pdf_filename')
def generate_sphase_qc(
        adata_filename,
        pdf_filename,
    ):

    adata = ad.read(adata_filename)

    adata = adata[adata.obs['is_normal'] == False]
    adata = adata[adata.obs['is_aberrant_normal_cell'] == False]

    # Calculate Features
    #

    adata.obs['state_rt_spearman'] = np.NaN
    adata.obs['copy_rt_spearman'] = np.NaN

    for cell_id in adata.obs.index:
        cell_data = adata[cell_id, adata.var['gc'] > 0]
        adata.obs.loc[cell_id, 'state_rt_spearman'] = scipy.stats.spearmanr(
            np.array(cell_data.layers['state'][0]),
            cell_data.var['replication_timing'].values).statistic
        adata.obs.loc[cell_id, 'copy_rt_spearman'] = scipy.stats.spearmanr(
            np.array(cell_data.layers['copy'][0]),
            cell_data.var['replication_timing'].values).statistic

    changepoint_matrix = spectrumanalysis.changepoints.compute_cn_changepoint_matrix(adata)

    # Calculate total and unique breakpoints for each cell 
    changepoint_cell_count = changepoint_matrix.sum(axis=1)
    adata.obs['n_brk'] = changepoint_matrix.sum(axis=0)
    adata.obs['n_uniq_brk'] = changepoint_matrix.loc[changepoint_cell_count == 1, :].sum(axis=0)

    # Calculate median prevalence of detected changepoints per cell
    changepoint_prevalence = changepoint_matrix.mean(axis=1)
    adata.obs['median_changepoint_prevalence'] = np.NaN
    for cell_id in adata.obs.index:
        adata.obs.loc[cell_id, 'median_changepoint_prevalence'] = changepoint_prevalence[changepoint_matrix.loc[:, cell_id]].median()

    write_qc_pdf(adata, pdf_filename)


if __name__ == '__main__':
    generate_sphase_qc()
