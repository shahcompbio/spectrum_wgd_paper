import numpy as np
import pandas as pd

import scgenome


def calculate_segments(data, col):
    """
    Calculate segments based on the provided data and column.

    Parameters
    ----------
    data : pandas.DataFrame
        The input data containing the necessary columns 'chr', 'start', and 'end'.
    col : str
        The name of the column to calculate segments on.

    Returns
    -------
    pandas.DataFrame
        A DataFrame containing the calculated segments with columns 'chr', 'start', 'end', and 'col'.

    """
    data = data.sort_values(['chr', 'start'])
    data['state_diff'] = data[col].diff().fillna(0).astype(int)
    data['chr_diff'] = (data['chr'].shift(1) != data['chr']) * 1
    data['transition'] = (data['state_diff'] != 0) | (data['chr_diff'] == 1)
    data['segment_idx'] = data['transition'].cumsum()
    data = data.groupby(['segment_idx']).agg(
        chr=('chr', 'first'),
        start=('start', 'min'),
        end=('end', 'max'),
        state=(col, 'first'),
    ).reset_index()
    data = data.rename(columns={'state': col})
    return data


def calculate_hlamp_segments(data, col, gap_threshold=2000000):
    """
    Calculate segments based on binary 'col' and remove gaps between segments.

    Parameters
    ----------
    data : DataFrame
        The input data containing the segments.
    col : str
        The column name used to calculate the segments.
    gap_threshold : int, optional
        The threshold for removing gaps between segments. Defaults to 2000000.

    Returns
    -------
    DataFrame
        The updated data with segments and removed gaps.
    """
    
    # Create segments based on binary 'col'
    data = calculate_segments(data, col)

    # Remove gaps between segments < gap_threshold
    data['length'] = data['end'] - data['start'] + 1
    data = data[(data[col]) | (data['length'] > gap_threshold)]

    # Reindex segments
    data = calculate_segments(data, col)
    data['length'] = data['end'] - data['start'] + 1
    
    return data


def compute_hlamp_events(adata, copy_norm_threshold=3., copy_context_threshold=2.):
    """
    Compute high lamp events from the given AnnData object.

    Parameters
    ----------
    adata : AnnData
        The input AnnData object containing copy number data.
    copy_norm_threshold : float, optional
        The threshold for copy number normalization. Defaults to 3.0.
    copy_context_threshold : float, optional
        The threshold for copy number normalization. Defaults to 2.0. None for no filtering.

    Returns
    -------
    cell_hlamps : DataFrame
        A DataFrame containing the high lamp events.

    """
    adata.layers['copy_norm'] = adata.layers['copy'] / np.nanmean(adata.layers['copy'], axis=1)[:, np.newaxis]
    adata.layers['is_hlamp'] = adata.layers['copy_norm'] > copy_norm_threshold
    adata.var['is_hlamp_any'] = adata.layers['is_hlamp'].any(axis=0)
    adata.var['copy_mean'] = np.nanmean(adata.layers['copy'], axis=0)

    # Compute segments
    hlamp_segments = calculate_hlamp_segments(adata.var, 'is_hlamp_any')
    hlamp_segments['hlamp_id'] = hlamp_segments['segment_idx'].astype(str)
    hlamp_segments = hlamp_segments.query('is_hlamp_any')

    # Compute segments before and after 5M
    before_segments = hlamp_segments.copy()
    before_segments['end'] = before_segments['start'] - 1
    before_segments['start'] -= 5000000

    after_segments = hlamp_segments.copy()
    after_segments['start'] = after_segments['end'] + 1
    after_segments['end'] += 5000000

    # Aggregate copy number
    agg_layers = {'copy': 'mean', 'copy_norm': 'mean', 'is_hlamp': 'mean'}
    adata_hlamps = scgenome.tl.rebin(adata, hlamp_segments, agg_layers=agg_layers)
    adata_hlamps.var = adata_hlamps.var.reset_index(drop=False).set_index('hlamp_id')
    
    adata_before = scgenome.tl.rebin(adata, before_segments, agg_layers=agg_layers)
    adata_before.var = adata_before.var.set_index('hlamp_id')
    
    adata_after = scgenome.tl.rebin(adata, after_segments, agg_layers=agg_layers)
    adata_after.var = adata_after.var.set_index('hlamp_id')

    # Add before and after copy number
    for layer_name in list(adata_hlamps.layers.keys()):
        adata_hlamps.layers[f'{layer_name}_before'] = (
            adata_before.to_df(layer_name).reindex(
                index=adata_hlamps.obs.index,
                columns=adata_hlamps.var.index))

        adata_hlamps.layers[f'{layer_name}_after'] = (
            adata_after.to_df(layer_name).reindex(
                index=adata_hlamps.obs.index,
                columns=adata_hlamps.var.index))

    # Generate event table
    cell_hlamps = pd.DataFrame({
        'is_hlamp': adata_hlamps.to_df('is_hlamp').stack(),
        'copy_norm': adata_hlamps.to_df('copy_norm').stack(),
        'copy_norm_before': adata_hlamps.to_df('copy_norm_before').stack(),
        'copy_norm_after': adata_hlamps.to_df('copy_norm_after').stack(),
        'copy': adata_hlamps.to_df('copy').stack()}).merge(adata_hlamps.var, left_index=True, right_index=True).reset_index()

    # Redo copy threshold
    cell_hlamps = cell_hlamps[cell_hlamps['copy_norm'] >= copy_norm_threshold]

    # Threshold on context for focal events
    if copy_context_threshold is not None:
        assert not (cell_hlamps['copy_norm_before'].isnull() & cell_hlamps['copy_norm_after'].isnull()).any()
        cell_hlamps['copy_norm_context'] = np.nanmean(cell_hlamps[['copy_norm_before', 'copy_norm_after']], axis=1)
        cell_hlamps['copy_norm_context_ratio'] = cell_hlamps['copy_norm'] / cell_hlamps['copy_norm_context']
        cell_hlamps = cell_hlamps[(cell_hlamps['copy_norm_context_ratio'] > copy_context_threshold)]

    return cell_hlamps
