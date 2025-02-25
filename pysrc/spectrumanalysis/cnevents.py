import numpy as np

import scgenome


def annotate_bins(adata, telomeric_length=5000000, centromeric_length=5000000):
    """ Annotate bins with cytoband, chromosome arm, telomere / centromere

    Parameters
    ----------
    adata : AnnData
        
    telomeric_length : int, optional
        threshold on length from telomere to be called telomeric, by default 5000000
    centromeric_length : int, optional
        threshold on length from centromere to be called centromeric, by default 5000000
    """

    # Add cytoband
    if 'cyto_band_name' not in adata.var.columns:
        adata.var = scgenome.tl.add_cyto_giemsa_stain(adata.var)

    # Add chromosome arm
    adata.var['arm'] = None
    adata.var.loc[adata.var['cyto_band_name'].str.startswith('p'), 'arm'] = 'p'
    adata.var.loc[adata.var['cyto_band_name'].str.startswith('q'), 'arm'] = 'q'
    assert adata.var['arm'].notnull().any()

    # Classify bins as telomere / centromere
    adata.var['p_telomere'] = False
    adata.var['q_telomere'] = False
    adata.var['p_centromere'] = False
    adata.var['q_centromere'] = False

    for chrom in adata.var['chr'].unique():
        chrom_start = adata.var.query(f'chr == "{chrom}"')['start'].min()
        chrom_end = adata.var.query(f'chr == "{chrom}"')['end'].max()
        p_centromere_start = adata.var.query(f'chr == "{chrom}" and arm == "p"')['end'].max() + 1
        q_centromere_end = adata.var.query(f'chr == "{chrom}" and arm == "q"')['start'].min()

        adata.var.loc[(adata.var['chr'] == chrom) & (adata.var['end'] - chrom_start <= telomeric_length), 'p_telomere'] = True
        adata.var.loc[(adata.var['chr'] == chrom) & (chrom_end - adata.var['start'] <= telomeric_length), 'q_telomere'] = True

        adata.var.loc[(adata.var['chr'] == chrom) & (adata.var['arm'] == 'p') & (p_centromere_start - adata.var['start'] <= centromeric_length), 'p_centromere'] = True
        adata.var.loc[(adata.var['chr'] == chrom) & (adata.var['arm'] == 'q') & (adata.var['end'] - q_centromere_end <= centromeric_length), 'q_centromere'] = True


def calc_end_above_threshold(is_change, threshold):
    """ Calculate the end of the first segment that is above a threshold percent changed

    Parameters
    ----------
    is_change : bool
        Boolean series representing cn changes
    threshold : float
        Percent changed threshold

    Returns
    -------
    end : int
        End of first segment above threshold
    """
    is_change_sum = is_change.cumsum()
    percent_changed = is_change_sum / np.arange(1, len(is_change_sum)+1)
    return is_change.loc[percent_changed > threshold].index.get_level_values('end').max()


def calc_start_above_threshold(is_change, threshold):
    """ Calculate the start of the first segment that is above a threshold percent changed

    Parameters
    ----------
    is_change : bool
        Boolean series representing cn changes
    threshold : float
        Percent changed threshold

    Returns
    -------
    start : int
        Start of first segment above threshold
    """
    is_change = is_change.iloc[::-1]
    is_change_sum = is_change.cumsum()
    percent_changed = is_change_sum / np.arange(1, len(is_change_sum)+1)
    return is_change.loc[percent_changed > threshold].index.get_level_values('start').min()


def calc_longest_segment(changes):
    """ Calculate the longest segment of consecutive changed bins

    Parameters
    ----------
    changes : DataFrame
        cn changes with columns start, end, is_change

    Returns
    -------
    start : int
        Start of longest segment
    end : int
        End of longest segment
    """
    if changes['is_change'].sum() == 0:
        return 0, 0

    # Add 0 to the start and end to handle edge cases
    is_change = np.concatenate(([0], changes['is_change'].values, [0]))
    
    # Find the change points in the array
    diffs = np.diff(is_change)
    
    # Start of consecutive 1s sequence has a difference of 1
    # End of consecutive 1s sequence has a difference of -1
    starts = np.where(diffs == 1)[0]
    ends = np.where(diffs == -1)[0]
    
    # Index into original start and end
    starts = changes.loc[changes.index[starts], 'start'].values
    ends = changes.loc[changes.index[ends - 1], 'end'].values
    
    # Calculate lengths of sequences
    lengths = ends - starts
    
    # Get the index of the longest sequence
    max_index = lengths.argmax()
    
    # Return start and end index of the longest sequence
    return starts[max_index], ends[max_index]



def is_telomeric_change(arm, bins, is_change, telomere_threshold):
    """ Check if there is a telomeric change above threshold

    Parameters
    ----------
    arm : str
        p or q arm
    bins : DataFrame
        DataFrame of bin information
    is_change : Series
        Boolean series representing cn changes
    telomere_threshold : float
        Percent changed threshold

    Returns
    -------
    is_telomeric_change : bool
        Whether there is a telomeric change above threshold
    telomere_end : int
        End of the telomeric segment above threshold
    """
    if arm == 'p':
        if is_change[bins['p_telomere']].mean() > telomere_threshold:
            p_telomere_end = calc_end_above_threshold(is_change[bins['arm'] == 'p'], telomere_threshold)
            return True, p_telomere_end
        else:
            return False, None
    elif arm == 'q':
        if is_change[bins['q_telomere']].mean() > telomere_threshold:
            p_telomere_start = calc_start_above_threshold(is_change[bins['arm'] == 'q'], telomere_threshold)
            return True, p_telomere_start
        else:
            return False, None


def is_centromeric_change(arm, bins, is_change, centromere_threshold):
    """ Check if there is a centromeric change above threshold

    Parameters
    ----------
    arm : str
        p or q arm
    bins : DataFrame
        DataFrame of bin information
    is_change : Series
        Boolean series representing cn changes
    centromere_threshold : float
        Percent changed threshold

    Returns
    -------
    is_centromeric_change : bool
        Whether there is a centromeric change above threshold
    centromere_end : int
        End of the centromeric segment above threshold
    """
    if arm == 'p':
        if is_change[bins['p_centromere']].mean() > centromere_threshold:
            p_centromere_start = calc_start_above_threshold(is_change[bins['arm'] == 'p'], centromere_threshold)
            return True, p_centromere_start
        else:
            return False, None
    elif arm == 'q':
        if is_change[bins['q_centromere']].mean() > centromere_threshold:
            p_centromere_end = calc_end_above_threshold(is_change[bins['arm'] == 'q'], centromere_threshold)
            return True, p_centromere_end
        else:
            return False, None



def classify_longest_segment(
        data,
        cn_change_col='cn_change',
        chrom_threshold=0.9,
        arm_threshold=0.9,
        telomere_threshold=0.9,
        centromere_threshold=0.9,
    ):
    """ Classify the longest CN change

    Parameters
    ----------
    data : DataFrame
        DataFrame of bin and CN change information
    cn_change_col : str, optional
        Column name for CN change, by default 'cn_change'
    chrom_threshold : float, optional
        Threshold for whole chromosome change, by default 0.9
    arm_threshold : float, optional
        Threshold for chromosome arm change, by default 0.9
    telomere_threshold : float, optional
        Threshold for telomeric change, by default 0.9
    centromere_threshold : float, optional
        Threshold for centromeric change, by default 0.9

    Returns
    -------
    change : dict
        Dictionary of change information
    """
    
    if (data[cn_change_col] == 0).all():
        return None

    chrom_start = data.index.get_level_values('start').min()
    chrom_end = data.index.get_level_values('end').max()
    p_arm_end = data.query('arm == "p"').index.get_level_values('end').max()
    q_arm_start = data.query('arm == "q"').index.get_level_values('start').min()

    # Whole chromosome change
    for direction in ['loss', 'gain']:
        if direction == 'gain':
            is_change = data[cn_change_col] > 0
        else:
            is_change = data[cn_change_col] < 0
        if is_change.mean() > chrom_threshold:
            event = {
                'start': chrom_start,
                'end': chrom_end,
                'kind': direction,
                'region': 'chromosome',
                'centromere': True,
                'telomere': True,
                'p_overlap': True,
                'q_overlap': True,
            }
            return event

    # Chromosome arm
    for direction in ['loss', 'gain']:
        if direction == 'gain':
            is_change = data[cn_change_col] > 0
        else:
            is_change = data[cn_change_col] < 0

        # P arm
        if is_change[data['arm'] == 'p'].mean() > arm_threshold:
            region = 'p-arm'
            telomere = True
            is_centromeric, q_centromere_end = is_centromeric_change('q', data, is_change, centromere_threshold)
            if is_centromeric:
                start = chrom_start
                end = q_centromere_end
                centromere = True
            else:
                start = chrom_start
                end = p_arm_end
                centromere = False

            event = {
                'start': start,
                'end': end,
                'kind': direction,
                'region': region,
                'centromere': centromere,
                'telomere': telomere,
                'p_overlap': True,
                'q_overlap': False,
            }

            return event

        # Q arm
        if is_change[data['arm'] == 'q'].mean() > arm_threshold:
            region = 'q-arm'
            telomere = True
            is_centromeric, p_centromere_start = is_centromeric_change('p', data, is_change, centromere_threshold)
            if is_centromeric:
                start = p_centromere_start
                end = chrom_end
                centromere = True
            else:
                start = q_arm_start
                end = chrom_end
                centromere = False

            event = {
                'start': start,
                'end': end,
                'kind': direction,
                'region': region,
                'centromere': centromere,
                'telomere': telomere,
                'p_overlap': False,
                'q_overlap': True,
            }

            return event

    # Segment change

    # Select gain or loss depending on largest change
    gain_start, gain_end = calc_longest_segment(data.assign(is_change=lambda df: df[cn_change_col] > 0).reset_index())
    loss_start, loss_end = calc_longest_segment(data.assign(is_change=lambda df: df[cn_change_col] < 0).reset_index())

    if gain_end - gain_start > loss_end - loss_start:
        direction = 'gain'
        is_change = (
            (data[cn_change_col] > 0) &
            (data.index.get_level_values('start') >= gain_start) &
            (data.index.get_level_values('end') <= gain_end))
        start = gain_start
        end = gain_end
    else:
        direction = 'loss'
        is_change = (
            (data[cn_change_col] < 0) &
            (data.index.get_level_values('start') >= loss_start) &
            (data.index.get_level_values('end') <= loss_end))
        start = loss_start
        end = loss_end

    p_centromere, _ = is_centromeric_change('p', data, is_change, centromere_threshold)
    q_centromere, _ = is_centromeric_change('q', data, is_change, centromere_threshold)
    p_telomere, _ = is_telomeric_change('p', data, is_change, telomere_threshold)
    q_telomere, _ = is_telomeric_change('q', data, is_change, telomere_threshold)
    region = 'segment'
    telomere = p_telomere or q_telomere
    centromere = p_centromere and q_centromere
    p_overlap = is_change[data['arm'] == 'p'].sum() > 0
    q_overlap = is_change[data['arm'] == 'q'].sum() > 0

    event = {
        'start': start,
        'end': end,
        'kind': direction,
        'region': region,
        'centromere': centromere,
        'telomere': telomere,
        'p_overlap': p_overlap,
        'q_overlap': q_overlap,
    }
    
    return event


acrocentric_chromosomes = ['13', '14', '15', '21', '22']

def classify_segments(
        data,
        cn_change_col='cn_change',
        chrom_threshold=0.9,
        arm_threshold=0.9,
        telomere_threshold=0.9,
        centromere_threshold=0.9,
    ):
    """ Classify CN segments

    Parameters
    ----------
    data : DataFrame
        DataFrame of bin and CN change information
    cn_change_col : str, optional
        Column name for CN change, by default 'cn_change'
    chrom_threshold : float, optional
        Threshold for whole chromosome change, by default 0.9
    arm_threshold : float, optional
        Threshold for chromosome arm change, by default 0.9
    telomere_threshold : float, optional
        Threshold for telomeric change, by default 0.9
    centromere_threshold : float, optional
        Threshold for centromeric change, by default 0.9

    Yields
    ------
    event : dict
        Dictionary of change event information
    """

    assert np.issubdtype(data[cn_change_col].dtype, np.integer)

    for chrom, chrom_data in data.groupby('chr', observed=True):
        assert chrom_data['start'].is_monotonic_increasing
        chrom_data = chrom_data.set_index(['start', 'end'])

        while True:
            event = classify_longest_segment(
                chrom_data,
                cn_change_col=cn_change_col,
                chrom_threshold=chrom_threshold,
                arm_threshold=arm_threshold,
                telomere_threshold=telomere_threshold,
                centromere_threshold=centromere_threshold,
            )

            if event is None:
                break

            # For the acrocentric chromosomes, all arm events are full chromosome
            if chrom in acrocentric_chromosomes and event['region'] == 'arm':
                event['region'] = 'chromosome'

            event['chr'] = chrom
            yield event

            if event['kind'] == 'loss':
                adjust = 1
            else:
                adjust = -1
            chrom_data.loc[(
                (chrom_data.index.get_level_values('start') >= event['start']) &
                (chrom_data.index.get_level_values('end') <= event['end'])), cn_change_col] += adjust

