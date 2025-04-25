#!/usr/bin/env python

import logging
import sys
import scgenome.utils
import pandas as pd
import pyranges as pr
import numpy as np
import click
import csverve


def dataframe_to_pyranges(data):
    data = pr.PyRanges(data.rename(columns={
        'chr': 'Chromosome',
        'start': 'Start',
        'end': 'End',
    }))

    return data


def pyranges_to_dataframe(data):
    data = data.as_df().rename(columns={
        'Chromosome': 'chr',
        'Start': 'start',
        'End': 'end',
    })

    return data


def resegment(cn_data, segments, cn_cols, allow_bins_dropped = True):
    # Consolodate segments
    segments = segments[['chr', 'start', 'end']].drop_duplicates().sort_values(['chr', 'start']).reset_index(drop = True)

    bins = cn_data[['chr', 'start', 'end']].drop_duplicates()

    segments['segment_idx'] = range(len(segments.index))
    bins['bin_idx'] = range(len(bins.index))

    pyr_bins = dataframe_to_pyranges(bins)
    pyr_segments = dataframe_to_pyranges(segments)

    intersect_1 = pyr_segments.intersect(pyr_bins)
    intersect_2 = pyr_bins.intersect(pyr_segments)

    intersect = pd.merge(
        pyranges_to_dataframe(intersect_1),
        pyranges_to_dataframe(intersect_2))

    cn_data = cn_data.merge(intersect, how='left')
    if not allow_bins_dropped:
        assert not cn_data['segment_idx'].isnull().any()

    cn_data = cn_data.dropna()
    segment_data = cn_data.groupby(['cell_id', 'segment_idx'], observed=True)[cn_cols].mean().round().astype(int).reset_index()
    
    segment_data = segment_data.merge(segments)
    return segment_data



@click.command()
@click.argument('output_filename')
@click.option('--hmmcopy_reads', multiple=True)
@click.option('--signals_results', multiple=True)
@click.option('--annotation_metrics', multiple=True)
@click.option('--segments_filename')
@click.option('--allele_specific', is_flag=True)
@click.option('--cell_list')
def create_medicc2_input(
        output_filename,
        hmmcopy_reads,
        signals_results,
        annotation_metrics,
        segments_filename,
        allele_specific,
        cell_list
    ):

    if len(hmmcopy_reads) > 0:
        assert len(signals_results) == 0
        assert len(annotation_metrics) > 0, 'medicc from raw hmmcopy will not work without filtering from annotation metrics'
        assert allele_specific is not True, 'can only do total copy number from hmmcopy input'

    elif len(signals_results) > 0:
        assert len(hmmcopy_reads) == 0

    else:
        raise ValueError('specify either signals or hmmcopy input')

    if allele_specific:
        cn_cols = ['cn_a', 'cn_b']
    
    else:
        cn_cols = ['cn']

    if len(hmmcopy_reads) > 0:
        cn_data = []
        for filename in hmmcopy_reads:
            cn_data.append(csverve.api.api.read_csv(filename))
        cn_data = scgenome.utils.concat_with_categories(cn_data, ignore_index=True)

        cn_data['cn'] = cn_data['state']

    else:
        signals_dtype = {
            'chr': 'category',
            'cell_id': 'category',
        }

        cn_data = []
        for filename in signals_results:
            cn_data.append(pd.read_csv(filename, dtype=signals_dtype))
        cn_data = scgenome.utils.concat_with_categories(cn_data, ignore_index=True)

        if 'Min' in cn_data.columns:
            signals_cn_cols = ['Maj', 'Min']
            use_minmaj = True
        else:
            signals_cn_cols = ['A', 'B']
            use_minmaj = False
        signals_cols = [
            'chr',
            'start',
            'end',
            'cell_id',
            'state'] + signals_cn_cols
        cn_data = cn_data[signals_cols]

        cn_data['cn'] = cn_data['state']
        cn_data['cn_a'] = cn_data['Maj' if use_minmaj else 'A']
        cn_data['cn_b'] = cn_data['Min' if use_minmaj else 'B']

        # Remove chromosomes for which all bins are null in all cells (eg Y chromosome)
        non_null_chroms = (
            cn_data.set_index(['chr', 'start', 'end', 'cell_id'])[cn_cols].unstack()
                .isna().all(axis=1).groupby('chr').all().rename('null_chrom').reset_index().query('null_chrom == False'))
        cn_data = cn_data.merge(non_null_chroms[['chr']])

    # Resegment according to input segments
    if segments_filename is not None:
        segments = pd.read_csv(segments_filename, dtype={'chr': str})

        # Require the same set of chromosomes between segments and cn_data
        segments = segments[segments['chr'].isin(cn_data['chr'].unique())]
        cn_data = cn_data[cn_data['chr'].isin(segments['chr'].unique())]

        cn_data = resegment(cn_data, segments, cn_cols)

    # Filter using annotation metrics if provided
    if len(annotation_metrics) > 0:
        metrics_data = []
        for filename in annotation_metrics:
            metrics_data.append(csverve.api.api.read_csv(filename))
        metrics_data = scgenome.utils.concat_with_categories(metrics_data, ignore_index=True)

        metrics_data = metrics_data.query('quality > 0.75 and not is_control')
        cn_data = cn_data.merge(
            metrics_data[['cell_id', 'library_id', 'sample_id']].rename(columns={
                'sample_id': 'original_sample_id',
                'library_id': 'original_library_id',
        }))

    # HACK: Parse cell id for sample and library
    else:
        cn_data['original_sample_id'] = cn_data['cell_id'].str.rsplit('-', expand=True, n=3)[0]
        cn_data['original_library_id'] = cn_data['cell_id'].str.rsplit('-', expand=True, n=3)[1]

    # Restrict to cells in cell_list if present
    if cell_list and cell_list != 'None':
        print('>>>>>>>>>>>>>>>>>>>>>>>', cell_list)
        list_cells = set(c.strip() for c in open(cell_list, 'r').readlines())
        cn_data = cn_data[cn_data['cell_id'].isin(list_cells)]

    # Fill NaN segments with 0s, assuming that these result from 0-read bins
    # (they may also be produced by gc=-1 sections, which should then be uniformly represented among all cells)

    metadata = cn_data[['cell_id', 'original_sample_id', 'original_library_id']].drop_duplicates()
    new_cn_data = cn_data.copy()
    new_cn_data = new_cn_data.set_index(['chr', 'start', 'end', 'segment_idx', 'cell_id'])[cn_cols].unstack().fillna(0).stack(future_stack=True).reset_index()
    new_cn_data.segment_idx = new_cn_data.segment_idx.astype(int)
    for col in cn_cols:
        new_cn_data[col] = new_cn_data[col].astype(int)
    new_cn_data = new_cn_data.merge(metadata)[cn_data.columns]
    cn_data = new_cn_data

    null_bins = (
        cn_data.set_index(['chr', 'start', 'end', 'cell_id'])[cn_cols].unstack()
            .isna().stack().any(axis=1).rename('null_bin').reset_index().query('null_bin == True'))

    if not null_bins.empty:
        raise ValueError(f'missing data for {null_bins.iloc[0].to_dict()}')

    medicc2_input = cn_data.rename(columns={'cell_id': 'sample_id'})

    autosomes = []
    for chr in cn_data['chr'].unique():
        try:
            autosomes.append(int(chr))
        except ValueError:
            if chr not in ['X', 'Y']:
                logging.warning(f'skipping chromosome {chr}')
            continue

    max_autosome = max(autosomes)
    chrom_X = max_autosome + 1
    chrom_Y = max_autosome + 2
    chromosomes_str = [str(a) for a in autosomes + ['X', 'Y']]

    medicc2_input = medicc2_input[medicc2_input['chr'].isin(chromosomes_str)]

    medicc2_input['chrom'] = medicc2_input['chr'].astype(str)
    medicc2_input.loc[medicc2_input['chr'] == 'X', 'chrom'] = chrom_X
    medicc2_input.loc[medicc2_input['chr'] == 'Y', 'chrom'] = chrom_Y
    medicc2_input['chrom'] = medicc2_input['chrom'].astype(int)

    cols = [
        'sample_id',
        'chrom',
        'start',
        'end',
        'chr',
        'original_sample_id',
        'original_library_id',
    ]

    medicc2_input[cols + cn_cols].to_csv(output_filename, sep='\t', index=False)


if __name__ == "__main__":
    logging.basicConfig(stream=sys.stderr, level=logging.INFO)
    create_medicc2_input()
