import logging
import sys
import pandas as pd
import numpy as np
import anndata as ad
import click

import spectrumanalysis.wgd


event_types = [
    'arm_gain', 'arm_loss',
    'chromosome_gain', 'chromosome_loss',
    'segment_gain', 'segment_loss',
]

chrom_event_types = [
    'chromosome_gain', 'chromosome_loss',
]

arm_event_types = [
    'arm_gain', 'arm_loss',
]

segment_event_types = [
    'segment_gain', 'segment_loss',
]

@click.command()
@click.argument('events_filename')
@click.argument('adata_filename')
@click.argument('cell_info_filename')
@click.argument('rates_filename')
def compute_misseg_rates(
        events_filename,
        adata_filename,
        cell_info_filename,
        rates_filename,
    ):

    adata = ad.read_h5ad(adata_filename)
    adata.layers['state'] = adata.layers['cn_a_2'] + adata.layers['cn_b_2']
    adata.obs.index.name = 'brief_cell_id'

    # Calculate n_cells
    cell_ids = adata.obs.query('is_cell == True').index.values

    events = pd.read_csv(events_filename, sep='\t')
    events['brief_cell_id'] = events['obs_id']
    events['length'] = events['end'] - events['start'] + 1
    events['class'] = events['region'].str.replace('[pq]-', '', regex=True) + '_' + events['kind']

    # Event filtering on 15 MB or more
    events = events.query('length >= 15000000')

    # Only cell specific events
    events = events[events['is_cell'] == True]
    events = events[events['kind'] != 'wgd']

    cell_info = pd.read_csv(cell_info_filename)
    cell_info = cell_info[cell_info['include_cell']]
    cell_info = spectrumanalysis.wgd.classify_subclonal_wgd(cell_info)

    # Calculate normalization factors for different event types
    #

    # Number of copies of chromosomes
    chrom_copies = (
        adata
            .to_df('state').T
            .set_index(adata.var['chr'], append=True)
            .groupby(level=[1], observed=True)
            .agg('median').stack().rename('median_state').reset_index())
    n_chromosomes = chrom_copies.groupby(['brief_cell_id'])['median_state'].sum().rename('n_chromosomes').reset_index()

    # Number of copies of arms
    arm_copies = (
        adata
            .to_df('state').T
            .set_index(adata.var['chr'], append=True)
            .set_index(adata.var['arm'], append=True)
            .groupby(level=[1, 2], observed=True)
            .agg('median').stack().rename('median_state').reset_index())
    n_arms = arm_copies.groupby(['brief_cell_id'])['median_state'].sum().rename('n_arms').reset_index()

    # Length in MB
    genome_copies = (
        adata
            .to_df('state').T
            .agg(['size', 'sum']).T
            .rename(columns={'size': 'n_bins', 'sum': 'sum_state'})
            .assign(mean_state=lambda df: df['sum_state'] / df['n_bins'])
            .reset_index())
    genome_copies['genome_length'] = genome_copies['sum_state'] * 500000 / 1e6

    # Counts of event per cell
    cell_event_counts = events.groupby(['brief_cell_id', 'class']).size().unstack(fill_value=0)
    cell_event_counts = cell_event_counts.reindex(index=cell_ids, columns=event_types, fill_value=0)
    cell_event_counts = cell_event_counts.reset_index()

    # Merge additional info
    cell_event_counts = cell_event_counts.merge(
        cell_info[['brief_cell_id', 'patient_id', 'sample_id', 'aliquot_id', 'n_wgd', 'majority_n_wgd', 'subclonal_wgd']], on='brief_cell_id', how='left')
    assert not cell_event_counts['patient_id'].isnull().any()
    assert not cell_event_counts['sample_id'].isnull().any()
    assert not cell_event_counts['aliquot_id'].isnull().any()
    assert not cell_event_counts['n_wgd'].isnull().any()
    assert not cell_event_counts['majority_n_wgd'].isnull().any()
    assert not cell_event_counts['subclonal_wgd'].isnull().any()

    # Rates per cell normalized by opportunity
    norm_cell_event_rates = cell_event_counts.merge(n_chromosomes)
    norm_cell_event_rates = norm_cell_event_rates.merge(n_arms)
    norm_cell_event_rates = norm_cell_event_rates.merge(genome_copies)

    norm_cell_event_rates[chrom_event_types] = norm_cell_event_rates[chrom_event_types] / norm_cell_event_rates['n_chromosomes'].values[:, np.newaxis]
    norm_cell_event_rates[arm_event_types] = norm_cell_event_rates[arm_event_types] / norm_cell_event_rates['n_arms'].values[:, np.newaxis]
    norm_cell_event_rates[segment_event_types] = norm_cell_event_rates[segment_event_types] / norm_cell_event_rates['genome_length'].values[:, np.newaxis]

    event_rate_groups = {
        'patient': ['patient_id',],
        'sample': ['patient_id', 'sample_id'],
        'aliquot': ['patient_id', 'sample_id', 'aliquot_id'],
        'wgd': ['patient_id', 'n_wgd', 'majority_n_wgd', 'subclonal_wgd'],
        'sample_wgd': ['patient_id', 'sample_id', 'n_wgd', 'majority_n_wgd', 'subclonal_wgd'],
    }

    all_rates = []
    for group_level, group_index in event_rate_groups.items():

        n_cells = cell_event_counts.groupby(group_index).size().rename('n_cells').reset_index()

        # Event rates normalized by cell count
        rates = cell_event_counts.groupby(group_index)[event_types].sum().reset_index()
        rates = rates.merge(n_cells)
        rates[event_types] = rates[event_types] / rates['n_cells'].values[:, np.newaxis]
        rates['group_level'] = group_level
        rates['normalized'] = False
        all_rates.append(rates)

        # Event rates normalized by opportunity and cell count
        norm_rates = norm_cell_event_rates.groupby(group_index)[event_types].sum().reset_index()
        norm_rates = norm_rates.merge(n_cells)
        norm_rates[event_types] = norm_rates[event_types] / norm_rates['n_cells'].values[:, np.newaxis]
        norm_rates['group_level'] = group_level
        norm_rates['normalized'] = True
        all_rates.append(norm_rates)

    all_rates = pd.concat(all_rates)

    all_rates.to_csv(rates_filename, index=False, sep='\t')


if __name__ == "__main__":
    logging.basicConfig(stream=sys.stderr, level=logging.INFO)
    compute_misseg_rates()

