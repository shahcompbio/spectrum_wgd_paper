import logging
import sys
import click
import warnings
import numpy as np
import anndata as ad
import pandas as pd

def consecutive(data, stepsize=0):
    return np.split(data, np.where(np.diff(data) != stepsize)[0]+1)

@click.command()
@click.argument('signals_adata_path')
@click.argument('whitelist_bins_path')
@click.argument('outfile_path')
def run_mspcf(
        signals_adata_path,
        whitelist_bins_path,
        outfile_path,
    ):

    adata = ad.read_h5ad(signals_adata_path)

    # subset to whitelisted low-noise bins
    whitelist_bins = pd.read_csv(whitelist_bins_path)
    adata = adata[:, whitelist_bins.bin].copy()

    adata.layers['residual'] = adata.layers['copy'] - adata.layers['state']
    adata.layers['abs_residual'] = np.abs(adata.layers['residual'])
    adata.layers['scaled_residual'] = (adata.layers['residual'].T / adata.obs.multiplier.values).T
    adata.layers['scaled_abs_residual'] = (adata.layers['abs_residual'].T / adata.obs.multiplier.values).T
    adata.layers['is_odd'] = adata.layers['state'] % 2
    adata.layers['is_135'] = np.logical_or(adata.layers['state'] == 1, 
                                        np.logical_or(adata.layers['state'] == 3, adata.layers['state'] == 5))

    all_segments = []

    warnings.simplefilter('ignore', category=FutureWarning)
    warnings.simplefilter('ignore', category=ad.ImplicitModificationWarning)
    for cell_id in adata.obs.index:
        my_ad = adata[cell_id]
        for ch, cdf in my_ad.var.groupby('chr', observed = True):
            chr_ad = my_ad[:, cdf.sort_values(by = 'start').index]

            states = np.array(my_ad[:, cdf.index].layers['state'][0])
            segment_bounds = np.concatenate([[0], np.where(np.diff(states) != 0)[0] + 1, [len(states)]])
            for i in range(len(segment_bounds) - 1):
                start = segment_bounds[i] # inclusive
                end = segment_bounds[i + 1] # exclusive

                seg_ad = chr_ad[:, start:end]        

                result = {}
                result['cell_id'] = cell_id
                result['chr'] = ch
                result['start'] = seg_ad.var.iloc[0].start
                result['end'] = seg_ad.var.iloc[-1].end
                result['n_bins'] = seg_ad.shape[1]
                result['state'] = seg_ad.layers['state'][0][0]
                result['variance'] = np.nanvar(seg_ad.layers['copy'])
                result['mean_signed_residual'] = np.nanmean(seg_ad.layers['residual'])
                result['median_signed_residual'] = np.nanmedian(seg_ad.layers['residual'])
                result['mean_residual'] = np.nanmean(seg_ad.layers['abs_residual'])
                result['median_residual'] = np.nanmedian(seg_ad.layers['abs_residual'])
                result['median_scaled_residual'] = np.nanmedian(seg_ad.layers['scaled_abs_residual'])
                
                all_segments.append(result)


    segments = pd.DataFrame(all_segments)
    segments = segments[~segments.variance.isna()]
    segments['is_135'] = segments.state.isin([1, 3, 5])
    segments['is_odd'] = segments.state % 2 == 1
    segments['length'] = segments.n_bins
    n135 = segments[segments.is_135].groupby('cell_id').size().reset_index().rename(columns={0:'n_135_segments'})
    length135 = segments[segments.is_135].groupby('cell_id').length.max().reset_index().rename(columns={'length':'longest_135_segment'})
    prop135 = segments.groupby('cell_id').is_135.sum().reset_index().rename(columns={0:'prop_135', 'is_135':'prop_135'})
    prop_odd = segments.groupby('cell_id').is_odd.mean().reset_index().rename(columns={0:'prop_odd', 'is_odd':'prop_odd'})
    df = n135.merge(length135, how = 'outer').merge(prop135, how = 'outer').merge(prop_odd, how = 'outer').fillna(0)
    df.n_135_segments = df.n_135_segments.astype(int)
    df.longest_135_segment = df.longest_135_segment.astype(int)

    result = adata.obs.merge(df, right_on='cell_id', left_index=True).set_index('cell_id')
    result.to_csv(outfile_path, index = False, compression={'method':'gzip', 'mtime':0, 'compresslevel':9})

if __name__ == "__main__":
    logging.basicConfig(stream=sys.stderr, level=logging.INFO)
    run_mspcf()
