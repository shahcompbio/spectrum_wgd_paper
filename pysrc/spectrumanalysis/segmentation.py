import matplotlib.pyplot as plt
import os

import logging
import sys
import click
import warnings
import json
import numpy as np
import pandas as pd
import anndata as ad
from multiprocessing import Pool
from datetime import datetime
import pyranges as pr


def weighted_segmented_piecewise(X, P=2, entry_weights = None, verbose = False):
    """
    Finds P segments (i.e., P-1 internal breakpoints) that divide the columns of X such that the squared error within each segment is minimized.
    Returns:
        -A: np.ndarray of shape (X.shape[0], P) where penalty A[i, p] is the minimum error possible for segmenting X[:i+1] with p segments (i.e., p-1 internal breakpoints)
        -bp: np.ndarray of shape (X.shape[0], P) where bp[i, p] contains the start index (inclusive) of the optimal segment ending at i (inclusive)
    """
    assert np.all(np.logical_or(entry_weights == 0, entry_weights == 1)), "Non-binary entry weights are poorly defined in this implementation"
    
    n, s = X.shape
    segcost_memo = {}
    
    if entry_weights is None:
        entry_weights = np.ones((n, s))

    X_weighted = entry_weights * X
    running_sum = np.cumsum(X_weighted, axis = 0)
    running_denom = np.cumsum(entry_weights, axis = 0)
    
    def segment_cost(i, j):
        """
        Computes the squared error for a closed segment [i, j]
        """
        assert j >= i
        if (i, j) in segcost_memo:
            return segcost_memo[i, j]
        else:
            # identify tracks with NaNs for all bins in this segment
            if i == 0:
                # no second term in "running" expressions
                track_totals = running_denom[j]
                valid_tracks = np.where(track_totals > 0)[0]
                my_mean = (running_sum[j])[valid_tracks] / track_totals[valid_tracks]
                
            else:
                # second term is sum of all preceding elements
                track_totals = running_denom[j] - running_denom[i - 1]
                valid_tracks = np.where(track_totals > 0)[0]
                my_mean = (running_sum[j] - running_sum[i - 1])[valid_tracks] / track_totals[valid_tracks]
                
            result = np.sum(
                np.square(X[i:j + 1, valid_tracks] - my_mean) * entry_weights[i:j+1, valid_tracks]
            )
            segcost_memo[i, j] = result
            return result
    
    # penalty A[i, p] is the minimum error possible for fitting X[:i+1] with p+1 pieces (i.e., p internal breakpoints)
    A = np.zeros((n, P))
    backpoint = np.zeros((n, P), dtype=int)

    A[:, 0] = [segment_cost(0, i) for i in range(n)]

    for p in range(1, P):
        
        # t is the right bound (closed) of the proposed new segment
        for t in range(n):
        
            # t' is the left bound (closed) of the proposed new segment
            # search over t' in [0,t]
            best_cost = np.inf
            best_tprime = None
            for tprime in range(t + 1):
                # compute cost for adding a breakpoint at tprime 
                # i.e., minimum cost for segments [..., [_, tprime - 1], [tprime, t]]
                if tprime == 0:
                    # no prev. segments, avoid underflow in indexing
                    prevcost = 0
                else:
                    prevcost = A[tprime - 1, p - 1]
                cost = prevcost + segment_cost(tprime, t)
                if verbose:
                    print(f"t={t}, p={p}, tprime={tprime}, prevcost={prevcost}, cost={cost}")
                if cost < best_cost:
                    best_cost = cost
                    best_tprime = tprime

            if verbose:
                print(f"At t={t} and p={p}, selected t`={best_tprime}")
            A[t, p] = best_cost
            backpoint[t, p] = best_tprime   # if this throws a TypeError for int(None), then X may have NaNs

    return A, backpoint


def backtrack(bp, verbose = False):
    """
    Backtracks through given backpointer array, starting with the bottom-right corner.
    Returns a list of segment start indices (including the length of the array at the end).
    """
    
    n, p = bp.shape

    starts = [bp[-1, -1]]
    if verbose:
        print(f'Appending value at bp[{n-1}, {p-1}]: {bp[-1, -1]}')
    
    for i in range(p - 2, -1, -1):
        if starts[-1] == 0 and p < n:
            warnings.warn('Found 0-cost prefix, stopping backtracking early. This should only occur on noise-free toy data.')
            break
        
        # Find the start of the best segment ending at prev-1
        if verbose:
            print(f'Appending value at bp[{starts[-1] - 1}, {i}]: {bp[starts[-1] - 1, i]}')
        starts.append(bp[starts[-1] - 1, i])

    if verbose:
        print("Reversing order and appending n")
    thresholds = starts[::-1] + [n]

    assert thresholds[0] == 0, "Error in backtracking: did not get to start of array!"
    return thresholds


def preprocess_data(adata, n_bins_trim_telomere = 2, centromere_buffer_size = 2.5e6,
                  skipY = True):
    """
    adata: AnnData with "copy" layer, "BAF" layer, and chromosome info in var
    n_bins_trum_telomere = int number of bins to ignore at the ends of each chr
    centromere_buffer_size = distance from centromere ends to ignore bins
    """
    
    chr2centro = json.load(open('/juno/work/shah/users/myersm2/ref-genomes/hg19.chr2centro.json', 'r'))

    pre_cent_idx = {}
    for i, (ch, df) in enumerate(adata.var.groupby('chr')):
        pre_idx = np.argmax(df.iloc[1:].start.to_numpy() - df.iloc[:-1].end.to_numpy())
        pre_cent_idx[ch] = pre_idx, df.iloc[pre_idx].end, df.iloc[pre_idx + 1].start, df.iloc[pre_idx + 1].start - df.iloc[pre_idx].end
    
    # Break down "copy" layer into per-chrom array
    chr_arr = adata.var.chr.to_numpy()   
    chr_breaks = np.where(chr_arr[:-1] != chr_arr[1:])[0]
    X_by_chr = {}
    my_thresholds = np.concatenate([[0], chr_breaks + 1, [len(chr_arr)]])
    for i in range(len(my_thresholds) - 1):
        start_idx = my_thresholds[i]
        end_idx = my_thresholds[i + 1]
        ch = chr_arr[start_idx + 1]

        if skipY and ch.endswith('Y'):
            continue
        
        X = adata.layers['copy'][:, start_idx:end_idx].copy()
        X = np.nan_to_num(X, 0)
        X_by_chr[ch] = X
    
    X_by_arm = {}
    valid_bins = {}
    
    # Separate into chromosome arms and trim centromere/telomere bins
    for ch, myX in X_by_chr.items():
        if ch == 'Y':
            continue
        ch_var = adata.var.query(f'chr == "{ch}"')
        assert len(ch_var) == myX.shape[1], ch

        # Identify bins within 2.5 MB of centromere or 1 MB of telomere
        cent_start, cent_end = chr2centro[f'chr{ch}']
        d = pre_cent_idx[ch]

        N = myX.shape[0]

        if ch in ['13', '14', '15', '21', '22']:
            # no bins before the centromere
            start_invalid_region = cent_start - centromere_buffer_size - 5
            end_invalid_region = cent_end + centromere_buffer_size + 5

            # get indices for invalid bins and trim X
            invalid_bin_idx = np.where(
                ((ch_var.start >= start_invalid_region) & (ch_var.start <= end_invalid_region)) |
                ((ch_var.end >= start_invalid_region) & (ch_var.end <= end_invalid_region)))[0]
            my_valid_bins = np.ones(myX.shape[1], dtype = bool)
            if len(invalid_bin_idx) > 0:
                # these assertions don't hold in the context of Andrew's arbitrarily located dropped bins
                #assert invalid_bin_idx[0] == 0, (ch, invalid_bin_idx) # first invalid bin should be at the start of acrocentric chromosome
                #assert invalid_bin_idx[-1] < myX.shape[1] - 1, (ch, invalid_bin_idx) # last invalid bin should be internal
                #assert np.all(np.diff(invalid_bin_idx) == 1), ch # invalid bins should be contiguous at start of acrocentric chromosome

                my_valid_bins[invalid_bin_idx] = False

            my_valid_bins[(-1 * n_bins_trim_telomere):] = False
            X_q = myX[:, my_valid_bins]            
            X_by_arm[ch + 'q'] = X_q
            valid_bins[ch + 'q'] = my_valid_bins

        else:
            start_invalid_region = min(d[1], cent_start) - centromere_buffer_size - 5
            end_invalid_region = max(d[2], cent_end) + centromere_buffer_size + 5

            # get indices for invalid bins (i.e., centromere regions) and split X
            invalid_bin_idx = np.where(
                ((ch_var.start >= start_invalid_region) & (ch_var.start <= end_invalid_region)) |
                ((ch_var.end >= start_invalid_region) & (ch_var.end <= end_invalid_region)))[0]
        
            # these assertions don't hold in the context of Andrew's arbitrarily located dropped bins
            #assert invalid_bin_idx[0] > 0, (ch, invalid_bin_idx) # first invalid bin should be internal
            #assert invalid_bin_idx[-1] < myX.shape[1] - 1, (ch, invalid_bin_idx) # last invalid bin should be internal
            #assert np.all(np.diff(invalid_bin_idx) == 1), ch # invalid bins should be contiguous in center of chromosome

            my_valid_bins_p = np.ones(myX.shape[1], dtype = bool)
            my_valid_bins_q = np.ones(myX.shape[1], dtype = bool)
            
            my_valid_bins_p[:n_bins_trim_telomere] = False
            my_valid_bins_p[invalid_bin_idx[0]:] = False
            X_p = myX[:, my_valid_bins_p]
            
            
            my_valid_bins_q[:invalid_bin_idx[-1] + 1] = False
            my_valid_bins_q[(-1 * n_bins_trim_telomere):] = False
            X_q = myX[:, my_valid_bins_q]

            X_by_arm[ch + 'p'] = X_p
            X_by_arm[ch + 'q'] = X_q
            valid_bins[ch + 'p'] = my_valid_bins_p
            valid_bins[ch + 'q'] = my_valid_bins_q
            
    outputs = {}
    outputs['X_by_chr'] = X_by_chr
    outputs['valid_bins'] = valid_bins
    outputs['X_by_arm'] = X_by_arm
    
    modal_cn = adata.obs.multiplier.to_numpy()
    outputs['modal_cn'] = modal_cn
    outputs['Xnorm_by_arm'] = {k:(v.T/modal_cn).T for k,v in X_by_arm.items()}
                
    # pull out BAF matrix
    baf_by_arm = {}

    for c, df in adata.var.groupby('chr'):
        df = df.sort_values(by = 'chr')
        myad = adata[:, df.index]

        if c + 'p' in valid_bins:
            myad_p = myad[:, valid_bins[c + 'p']]
            baf_by_arm[c + 'p'] = myad_p.layers['BAF']

        if c + 'q' in valid_bins:
            myad_q = myad[:, valid_bins[c + 'q']]
            baf_by_arm[c + 'q'] = myad_q.layers['BAF']
            
    outputs['baf_by_arm'] = baf_by_arm

    return outputs


def run_mspcf_arm_gamma(param, max_P = 40):
    xnorm, baf, gamma = param
    stacked = np.vstack([xnorm, baf])
    masknan_weights = np.ones(stacked.shape)
    masknan_weights[np.isnan(stacked)] = 0
    stacked = np.nan_to_num(stacked, nan = 0.5)
    
    A, bp = weighted_segmented_piecewise(stacked.T, P=min(max_P, stacked.shape[1]), entry_weights = masknan_weights.T)
    my_bpsets = {}
    for p in range(A.shape[1]):
        bps = backtrack(bp[:, :p + 1])
        my_bpsets[p] = bps
        
    n_cells = xnorm.shape[0]
    loss_curve = A[-1] + gamma * n_cells * (np.arange(A.shape[1]) + 1)
    winner = np.argmin(loss_curve)

    return A, bp, my_bpsets , winner


def bps2segments(inputs, results, adata):
    """
    Given breakpoints and input bins, map bins to segments and output table of segments.
    """
    startcol = []
    endcol = []
    chrcol = []
    for ch, vb in inputs['valid_bins'].items():
        if ch not in results:
            print("skipping arm ", ch, " with no results")
            continue
    
        # subset the adata to the bins given as input to the segmentation
        ch_adata = adata[:, adata.var.chr == ch[:-1]]
        assert ch_adata.shape[1] == len(vb)
        valid_adata = ch_adata[:, vb]

        # retrieve the set of identified breakpoints and do a sanity check
        _, _, bpsets, winner = results[ch]
        my_bps = np.array(bpsets[winner].copy())
        assert my_bps[0] == 0, ch
        assert my_bps[-1] == valid_adata.shape[1], ch
        # decrement the value of the last bp so that it can be used to get the end of the last segment

        # the start of each "breakpoint" bin is a segment start
        segment_starts = valid_adata.var.iloc[my_bps[:-1]].start
        segment_ends = valid_adata.var.iloc[my_bps[1:] - 1].end

        startcol.extend(segment_starts)
        endcol.extend(segment_ends)
        chrcol.extend([ch[:-1]] * len(segment_starts))
    segdf = pd.DataFrame({'chr':chrcol, 'start':startcol, 'end':endcol}).sort_values(by = ['chr', 'start']).reset_index(drop = True)
    segdf_allcells = pd.concat([segdf] * len(adata))
    segdf_allcells['cell_id'] = np.repeat(adata.obs.index, len(segdf))
    segdf_allcells = segdf_allcells.reset_index(drop = True)
    return segdf_allcells

def unstack_adata(adata, layers = None):
    my_cn_data = []
    
    if layers is None:
        layers = adata.layers.keys()
    
    for layer in layers:
        df = pd.DataFrame(adata.layers[layer])
        df.columns = adata.var.index
        df.index = adata.obs.index

        longdf = df.unstack().reset_index()

        bin_cols1 = longdf.bin.str.split(':', expand = True)
        bin_cols2 = bin_cols1.iloc[:, 1].str.split('-', expand = True)

        longdf['chr'] = bin_cols1.iloc[:, 0]
        longdf['start'] = bin_cols2.iloc[:, 0].astype(int)
        longdf['end'] = bin_cols2.iloc[:, 1].astype(int)
        longdf[layer] = longdf[0]
        longdf = longdf.drop(columns = [0])
        my_cn_data.append(longdf)

    unstacked = my_cn_data[0]
    for df in my_cn_data[1:]:
        assert unstacked.iloc[:, :4].equals(df.iloc[:, :4])
        unstacked[df.columns[-1]] = df.iloc[:, -1]
    return unstacked

def consecutive(data, stepsize=1):
    return np.split(data, np.where(np.diff(data) != stepsize)[0]+1)


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

def resegment(cn_data, segments, cn_cols, allow_bins_dropped = False):
    # Consolodate segments
    segments = segments[['chr', 'start', 'end']].drop_duplicates().sort_values(['chr', 'start']).reset_index(drop = True)
    #segments = segments.groupby(['chr'], observed=True).apply(create_segments).reset_index()[['chr', 'start', 'end']].sort_values(['chr', 'start'])

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

    segment_data = cn_data.groupby(['cell_id', 'segment_idx'], observed=True)[cn_cols].mean().round().astype(int).reset_index()
    
    segment_data = segment_data.merge(segments)
    return segment_data


def run_mspcf_arm_gamma(xnorm, baf, gamma, max_P = 40):
    stacked = np.vstack([xnorm, baf])
    masknan_weights = np.ones(stacked.shape)
    masknan_weights[np.isnan(stacked)] = 0
    stacked = np.nan_to_num(stacked, nan = 0.5)
    
    A, bp = weighted_segmented_piecewise(stacked.T, P=min(max_P, stacked.shape[1]), entry_weights = masknan_weights.T)
    my_bpsets = {}
    for p in range(A.shape[1]):
        bps = backtrack(bp[:, :p + 1])
        my_bpsets[p] = bps
        
    n_cells = xnorm.shape[0]
    loss_curve = A[-1] + gamma * n_cells * (np.arange(A.shape[1]) + 1)
    winner = np.argmin(loss_curve)

    return A, bp, my_bpsets, winner


