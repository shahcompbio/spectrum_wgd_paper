#!/usr/bin/env python

import logging
import sys
import pandas as pd
import numpy as np
import anndata as ad
import click
import scipy.stats
import os
import glob

import scgenome
from scgenome.tools.ranges import dataframe_to_pyranges, pyranges_to_dataframe
import spectrumanalysis.changepoints
import spectrumanalysis.cnmetrics


def calculate_loh_ploidy(adata):
    """ Calculate ploidy and LOH """
    adata_filtered = adata[:, (adata.var['gc'] > 0) & (adata.var['has_allele_cn'])]

    lengths = (adata_filtered.var['end'] - adata_filtered.var['start'] + 1).values
    minor_cn = np.minimum(adata_filtered.layers['B'], adata_filtered.layers['A'])
    is_loh = (minor_cn == 0)
    total_cn = adata_filtered.layers['copy']

    adata.obs['fraction_loh'] = np.array(np.nansum(is_loh * lengths, axis=1) / np.nansum(lengths))
    adata.obs['ploidy'] = np.array(np.nansum(total_cn * lengths, axis=1) / np.nansum(lengths))

    return adata


def add_rt_from_bigwig(adata, bigwig_filename, column_name):
    bins = dataframe_to_pyranges(adata.var)
    rt = scgenome.tl.mean_from_bigwig(bins, bigwig_filename, column_name, chr_prefix='chr')
    rt = pyranges_to_dataframe(rt)

    def create_bin_name(data):
        bin_name = data['chr'].astype(str) + ':' + data['start'].astype(str) + '-' + data['end'].astype(str)
        return bin_name

    rt['bin'] = create_bin_name(rt)
    adata.var = adata.var.merge(rt.set_index('bin')[[column_name]], left_index=True, right_index=True, how='left')

    return adata


def classify_s_phase_cells(adata, rt_thresh, bk_thresh, ch_thresh):

    # Calculate replication timing correlations
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

    # Filter normals before calculating changepoint prevalence
    filtered_changepoint_matrix = changepoint_matrix.loc[:, adata.obs.query('is_normal == False & is_aberrant_normal_cell == False').index.values]
    assert filtered_changepoint_matrix.shape[1] == len(adata.obs.query('is_normal == False & is_aberrant_normal_cell == False').index)

    # Calculate median prevalence of detected changepoints per cell
    changepoint_prevalence = filtered_changepoint_matrix.mean(axis=1)
    adata.obs['median_changepoint_prevalence'] = np.NaN
    for cell_id in adata.obs.index:
        adata.obs.loc[cell_id, 'median_changepoint_prevalence'] = changepoint_prevalence[changepoint_matrix.loc[:, cell_id]].median()

    # Individual thresholds
    adata.obs['is_s_phase_rt'] = False
    adata.obs.loc[adata.obs['state_rt_spearman'] >= rt_thresh, 'is_s_phase_rt'] = True

    adata.obs['is_s_phase_bk'] = False
    adata.obs.loc[adata.obs['n_brk'] >= bk_thresh, 'is_s_phase_bk'] = True

    adata.obs['is_s_phase_ch'] = False
    adata.obs.loc[adata.obs['median_changepoint_prevalence'] <= ch_thresh, 'is_s_phase_ch'] = True

    # Combined thresholding, cells failing 2 or more thresholds are called as s phase
    adata.obs['is_s_phase_thresholds'] = (adata.obs[['is_s_phase_rt', 'is_s_phase_bk', 'is_s_phase_ch']].sum(axis=1) >= 2)

    return adata

def calculate_cn_stats(adata):
    adata_filtered = adata[:, (adata.var['gc'] > 0) & (adata.var['has_allele_cn'])]    
    lengths = (adata_filtered.var['end'] - adata_filtered.var['start'] + 1).values
    minor_cn = np.minimum(adata_filtered.layers['B'], adata_filtered.layers['A'])
    major_cn = np.maximum(adata_filtered.layers['B'], adata_filtered.layers['A'])
    mean_allele_diff = (major_cn - minor_cn)
    cn_gt2 = (major_cn >= 2)
    cn_gt3 = (major_cn >= 3)
    cn_gt4 = (major_cn >= 4)
    is_loh = (minor_cn == 0)
    total_cn = adata_filtered.layers['copy']    
    adata.obs['fraction_loh'] = np.array(np.nansum(is_loh * lengths, axis=1) / np.nansum(lengths))
    adata.obs['ploidy'] = np.array(np.nansum(total_cn * lengths, axis=1) / np.nansum(lengths))
    adata.obs['cn_gt2'] = np.array(np.nansum(cn_gt2 * lengths, axis=1) / np.nansum(lengths))
    adata.obs['cn_gt3'] = np.array(np.nansum(cn_gt3 * lengths, axis=1) / np.nansum(lengths))
    adata.obs['cn_gt4'] = np.array(np.nansum(cn_gt4 * lengths, axis=1) / np.nansum(lengths))
    adata.obs['mean_allele_diff'] = np.array(np.nansum(mean_allele_diff * lengths, axis=1) / np.nansum(lengths))
    
    return adata

def calculate_n_wgd(adata):
    adata = calculate_cn_stats(adata)    
    adata.obs['n_wgd'] = 0
    adata.obs.loc[adata.obs['cn_gt2'] > 0.5, 'n_wgd'] = 1
    adata.obs.loc[adata.obs['cn_gt3'] > 0.5, 'n_wgd'] = 2
    
    return adata 

@click.command()
@click.argument('patient_id')
@click.argument('hscn_filename')
@click.argument('hmmcopy_filename')
@click.argument('adata_filename')
@click.option('--manual_normal_cells_csv', required=True)
@click.option('--classifier_normal_cells_csv', required=True)
@click.option('--image_features_dir', required=True)
@click.option('--doublets_csv', required=True)
@click.option('--rt_bigwig', required=True)
@click.option('--s_phase_thresholds_csv', required=True)
def create_signals_anndata(
        patient_id,
        hscn_filename,
        hmmcopy_filename,
        adata_filename,
        manual_normal_cells_csv,
        classifier_normal_cells_csv,
        image_features_dir,
        doublets_csv,
        rt_bigwig,
        s_phase_thresholds_csv,
    ):

    adata = ad.read(hmmcopy_filename)

    hscn_dtype = {
        'cell_id': 'category',
        'chr': 'category',
        'state_AS_phased': 'category',
        'state_AS': 'category',
        'phase': 'category',
        'state_phase': 'category',
    }

    hscn = pd.read_csv(hscn_filename, dtype=hscn_dtype)

    # Add sample id to cell id
    hscn = hscn.rename(columns={'cell_id': 'brief_cell_id'})
    hscn = hscn.merge(adata.obs[['brief_cell_id']].reset_index().drop_duplicates())

    # KLUDGE, this will be dropped
    X_column = 'totalcounts'

    layers_columns = [
        'totalcounts',
        'alleleA',
        'alleleB',
        'BAF',
        'A',
        'B',
    ]

    bin_metrics_columns = [
        'chr',
        'start',
        'end',
    ]

    bin_metrics_data = hscn[bin_metrics_columns].drop_duplicates()
    cell_metrics_data = hscn[['cell_id']].drop_duplicates()

    # Create anndata from signals input
    signals = scgenome.pp.create_cn_anndata(
        hscn,
        X_column=X_column,
        layers_columns=layers_columns,
        cell_metrics_data=cell_metrics_data,
        bin_metrics_data=bin_metrics_data,
    )

    def check_subset(a, b):
        a, b = set(a), set(b)
        if not a.issubset(b):
            raise ValueError(f'not subset, a-b: {a-b}')
    check_subset(signals.obs.index, adata.obs.index)

    # Filter hmmcopy cells to those in signals
    adata = adata[signals.obs.index, :]

    # Add each signals data to hmmcopy anndata
    for layer in layers_columns:
        layer_data = signals.to_df(layer)
        layer_data = layer_data.reindex(columns=adata.var.index)
        adata.layers[layer] = layer_data

    # Patient id check
    assert (adata.obs['patient_id'] == patient_id).all()

    # Additional annotations from reference data
    #

    # Add replication timing data
    adata = add_rt_from_bigwig(adata, rt_bigwig, 'replication_timing')

    # Additional external data
    #

    # Add image data
    image_filenames = glob.glob(os.path.join(image_features_dir, '*.csv'))
    image_data = pd.concat([pd.read_csv(f) for f in image_filenames], ignore_index=True)
    image_data = image_data.rename(columns={'cell_id': 'brief_cell_id'})
    image_data = image_data.merge(adata.obs.reset_index()[['cell_id', 'brief_cell_id']])
    for column in ['Diameter', 'Elongation', 'Circularity', 'Intensity']:
        adata.obs[column] = np.NaN
        adata.obs.loc[image_data['cell_id'].values, column] = image_data[column].values

    # Add doublet data
    doublets = pd.read_csv(doublets_csv)
    doublets = doublets.merge(adata.obs.reset_index()[['cell_id', 'brief_cell_id']])
    adata.obs['is_doublet'] = 'No'
    adata.obs.loc[doublets['cell_id'].values, 'is_doublet'] = doublets['is_doublet'].values
    adata.obs['is_doublet'] = adata.obs['is_doublet'].astype('category')

    ## Additional manual annotation
    #

    # Calculate smoothness of BAF signal
    adata = spectrumanalysis.cnmetrics.calculate_local_baf_variance(adata)

    # Classify cells as WGD
    local_baf_variance_threshold = 0.01
    adata.var['has_allele_cn'] = (
        np.array(np.isfinite(adata.layers['B']).all(axis=0)) &
        (adata.var['local_baf_variance'] < local_baf_variance_threshold))
    adata = calculate_loh_ploidy(adata)
    def classify_wgd(p, l):
        return -(1.1/0.85) * l + 2.8 < p
    adata.obs['is_wgd'] = adata.obs.apply(lambda row: classify_wgd(row['ploidy'], row['fraction_loh']), axis=1) * 1
    adata.obs['is_wgd'] = adata.obs['is_wgd'].astype('category')

    # Normal cell classification by loh threshold, manual identification and a classifier
    adata.obs['is_aberrant_normal_cell'] = False
    adata.obs.loc[adata.obs['fraction_loh'] < 0.01, 'is_aberrant_normal_cell'] = True

    # Normal cells from marc's classifier
    # HACK: must be done before rename cell ids right now
    classifier_normal_cells = pd.read_csv(classifier_normal_cells_csv)
    classifier_normal_cells = classifier_normal_cells.query('is_tumor_cell != "Y"')
    adata.obs.loc[adata.obs['brief_cell_id'].isin(classifier_normal_cells['cell_id'].values), 'is_aberrant_normal_cell'] = True

    # Check fixed cell ids
    assert not adata.obs['brief_cell_id'].str.contains('Project_').any()

    # Manually identified normal cells
    aberrant_normal_cells = pd.read_csv(manual_normal_cells_csv)
    adata.obs.loc[adata.obs.index.isin(aberrant_normal_cells['cell_id'].values), 'is_aberrant_normal_cell'] = True

    # Classify cells as normal (remove chr 9 because it has centromere issues)
    autosomes = set([str(a) for a in range(1, 23)]).difference(['9'])
    mean = adata[:, adata.var['chr'].isin(autosomes)].to_df('state').mean(axis=1)
    std = adata[:, adata.var['chr'].isin(autosomes)].to_df('state').std(axis=1)
    adata.obs['is_normal'] = ((mean > 1.95) & (mean < 2.05) & (std < 0.5))
    adata.obs['is_normal'] = adata.obs['is_normal'].astype('category')

    # Classify cells as s phase based on changepoints
    s_phase_thresholds = pd.read_csv(s_phase_thresholds_csv).set_index('patient_id').loc[patient_id]
    adata = classify_s_phase_cells(
        adata,
        rt_thresh=s_phase_thresholds['rt_thresh'],
        bk_thresh=s_phase_thresholds['bk_thresh'],
        ch_thresh=s_phase_thresholds['ch_thresh'])

    # Basic analyses
    #

    # Detect outliers
    adata = scgenome.tl.detect_outliers(
        adata,
        layer_name=['copy', 'BAF'],
        method='local_outlier_factor'
    )

    # Basic clustering, exclude outliers
    inlier_cell_ids = adata.obs[adata.obs['is_outlier'] != 1].index.values
    try:
        adata = scgenome.tl.cluster_cells(
            adata,
            min_k=20,
            max_k=min(80, len(inlier_cell_ids) - 1),
            layer_name=['copy', 'BAF'],
            method='gmm_diag_bic',
            cell_ids=inlier_cell_ids,
        )
    except ValueError:
        # some patients are getting ValueError for fitting sklearn cholelesky precision
        adata = scgenome.tl.cluster_cells(
            adata,
            min_k=20,
            max_k=min(30, len(inlier_cell_ids) - 1),
            layer_name=['copy', 'BAF'],
            method='kmeans_bic',
            cell_ids=inlier_cell_ids,
        )

    # Add cell order based on hierchical clustering
    adata = scgenome.tl.sort_cells(adata, layer_name=['copy', 'BAF'])

    # Add # WGDs based on TracerX criteria: prop. genome w/ major allele >=2 for 1 WGD, >=3 for 2 WGD
    adata = calculate_n_wgd(adata)

    adata.obs.cell_order = adata.obs.cell_order.astype(str).astype('category')
    adata.write(adata_filename)


if __name__ == "__main__":
    logging.basicConfig(stream=sys.stderr, level=logging.INFO)
    create_signals_anndata()

