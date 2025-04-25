#!/usr/bin/env python

import logging
import sys
import seaborn as sns
import click
import matplotlib.pyplot as plt
import numpy as np
import pickle
import os
from sklearn.metrics import adjusted_rand_score
from collections import Counter
from datetime import datetime
from scipy import sparse

def refactor_labels(M, rowl, coll):
    """
        Reassign row and column block labels to be in descending order of density.
    """
    rowl = np.array(rowl)
    coll = np.array(coll)
    
    row_block_means = {l:M[rowl == l].mean() for l in np.unique(rowl)}
    col_block_means = {l:M[:, coll == l].mean() for l in np.unique(coll)}

    row_bymean = [x[0] for x in sorted(row_block_means.items(), key = lambda a:a[1], reverse = True)]
    col_bymean = [x[0] for x in sorted(col_block_means.items(), key = lambda a:a[1], reverse = True)]
    
    new_rowl = rowl.copy()
    for l in np.unique(rowl):
        new_rowl[rowl == l] = row_bymean.index(l)
    
    new_coll = coll.copy()
    for l in np.unique(coll):
        new_coll[coll == l] = col_bymean.index(l)    
    
    return new_rowl, new_coll

def subsample(M, target_rows = 1000, target_cols = 1500):
    # code stub for subsampling sparse matrices from https://stackoverflow.com/questions/42210594/plot-heatmap-of-sparse-matrix
    data = M.tocsc()
    N, M = data.shape

    # decimation factors for y and x directions
    s = N // target_rows if N > target_rows else 1
    t = M // target_rows if M > target_cols else 1

    T = sparse.csc_matrix((np.ones((M,)), np.arange(M), np.r_[np.arange(0, M, t), M]), (M, (M-1) // t + 1))
    S = sparse.csr_matrix((np.ones((N,)), np.arange(N), np.r_[np.arange(0, N, s), N]), ((N-1) // s + 1, N))
    result = S @ data @ T     # downsample by binning into s x t rectangles
    result = result.todense()
    return result

def plot_matrix(M, row_labels, col_labels, x_label = 'SNVs'):
    #new_row_labels, new_col_labels = refactor_labels(M, row_labels, col_labels)
    new_row_labels = row_labels.copy()
    new_col_labels = col_labels.copy()
    
    row_block_labels = np.array(sorted(new_row_labels))
    col_block_labels = np.array(sorted(new_col_labels))

    row_idx = np.argsort(new_row_labels)
    col_idx = np.argsort(new_col_labels)
    
    row_tick_labels = np.arange(len(Counter(row_block_labels)))
    row_tick_locs = [np.mean(np.where(row_block_labels == i)[0]) for i in range(len(row_tick_labels))]

    col_tick_labels = np.arange(len(Counter(col_block_labels)))
    col_tick_locs = [np.mean(np.where(col_block_labels == i)[0]) for i in range(len(col_tick_labels))]

    fig, axes = plt.subplots(nrows=2, ncols=2, width_ratios=[0.05, 1], height_ratios=[1, 0.05])
    axes[1, 0].set_axis_off()

    # column block colors bottom right
    axes[1, 1].imshow(plt.get_cmap('tab20')(col_block_labels).reshape(1, len(col_block_labels), 4), aspect = 'auto')
    axes[1, 1].set_xticks(col_tick_locs, col_tick_labels)
    axes[1, 1].set_yticks([])
    axes[1, 1].set_xlabel(x_label)

    # row block colors top left
    axes[0, 0].imshow(plt.get_cmap('tab20')(row_block_labels).reshape(len(row_block_labels), 1, 4), aspect = 'auto')
    axes[0, 0].set_yticks(row_tick_locs, row_tick_labels)
    axes[0, 0].set_xticks([])
    axes[0, 0].set_ylabel("Cells")

    # matrix itself middle right
    subsampled = subsample(M[row_idx][:, col_idx])
    sns.heatmap(subsampled, ax = axes[0, 1], cmap = 'Greys', cbar = False)
    axes[0, 1].set_xticks([])
    axes[0, 1].set_yticks([])

    plt.tight_layout()
    return fig

def plot_blocks(M, row_labels, col_labels):
    #new_row_labels, new_col_labels = refactor_labels(M, row_labels, col_labels)
    new_row_labels = row_labels.copy()
    new_col_labels = col_labels.copy()
    
    n_rowblocks = max(new_row_labels) + 1
    n_colblocks = max(new_col_labels) + 1
    
    block_means = np.zeros((n_rowblocks, n_colblocks))
    rowblock_ns = np.zeros(n_rowblocks, dtype = int)
    colblock_ns = np.zeros(n_colblocks, dtype = int)
    colmap = {}
    for i in range(n_rowblocks):
        my_rows = M[row_labels == i]
        rowblock_ns[i] = my_rows.shape[0]

        for j in range(np.max(new_col_labels) + 1):
            if j not in colmap:
                colmap[j] = new_col_labels == j
                colblock_ns[j] = np.sum(colmap[j])

            my_sum = np.sum(my_rows[:, colmap[j]])
            my_denom = my_rows.shape[0] * np.sum(colmap[j])
            block_means[i, j] = my_sum / my_denom
            
    if n_rowblocks > 1 and np.all(block_means[-1] == 0):
        target = block_means[:-1]
    else:
        target = block_means

    N, M = target.shape
    if N == 1:
        fig = plt.figure(figsize = (6,6))
        sns.heatmap(target * 100, annot = target * 100, fmt = '.2f')
        plt.title("Percent of 1-entries")
        plt.xlabel("SNV (column) block")
        plt.ylabel("Cell (row) block")   
        return fig
    else:
        
        cm = sns.clustermap(target * 100, col_cluster = False, metric = 'cosine',
                            figsize = (6,6), annot = target * 100, fmt = '.2f')
        cm.ax_heatmap.set_xticks(np.arange(M) + 0.5, [f'{i} (n={colblock_ns[i]})' for i in range(M)], # columns are not reordered
                                rotation = 270)

        cm.ax_heatmap.set_yticks(np.arange(N) + 0.5, [f'{i} (n={rowblock_ns[i]})' 
                                                      for i in cm.dendrogram_row.reordered_ind],
                                rotation = 0)

        cm.ax_heatmap.set_title("Percent of 1-entries")
        cm.ax_heatmap.set_xlabel("SNV (column) block")
        cm.ax_heatmap.set_ylabel("Cell (row) block")
        return cm.fig

def agreement_plots(all_results, figure_stem):
    n_restarts = len(all_results)
    rows_comparison = np.zeros((n_restarts, n_restarts))
    cols_comparison = np.zeros((n_restarts, n_restarts))

    for i in range(n_restarts):
        rl1 = all_results[i][0]
        cl1 = all_results[i][1]
        for j in range(n_restarts):
            rl2 = all_results[j][0]
            cl2 = all_results[j][1]

            rows_comparison[i, j] = adjusted_rand_score(rl1, rl2)
            cols_comparison[i, j] = adjusted_rand_score(cl1, cl2)
            
            rows_comparison[j, i] = adjusted_rand_score(rl1, rl2)
            cols_comparison[j, i] = adjusted_rand_score(cl1, cl2)

    # write row and col agreement heatmaps
    rowblock_heatmap = sns.clustermap(rows_comparison, annot = rows_comparison)
    rowblock_heatmap.ax_heatmap.set_xlabel("SBMClone run")
    rowblock_heatmap.ax_heatmap.set_ylabel("SBMClone run")
    rowblock_heatmap.ax_col_dendrogram.set_title("Row block ARI")    
    rowblock_heatmap.savefig(os.path.join(figure_stem, 'row_block_ari.png'), dpi = 300)
    plt.close(rowblock_heatmap.fig)

    colblock_heatmap = sns.clustermap(cols_comparison, annot = cols_comparison)
    colblock_heatmap.ax_heatmap.set_xlabel("SBMClone run")
    colblock_heatmap.ax_heatmap.set_ylabel("SBMClone run")
    colblock_heatmap.ax_col_dendrogram.set_title("Column block ARI")
    colblock_heatmap.savefig(os.path.join(figure_stem, 'col_block_ari.png'), dpi = 300)
    plt.close(colblock_heatmap.fig)

    # Identify high-confidence cell and SNV clusters
    my_rowlabels = [a[0] for a in all_results]
    my_collabels = [a[1] for a in all_results]

    
    n_rows = len(my_rowlabels[0])
    row_adj = np.zeros((n_rows, n_rows))
    for labeling in my_rowlabels:
        n_cl = np.max(labeling) + 1
        for c in range(n_cl):
            members = np.where(labeling == c)[0]
            row_adj[np.tile(members, len(members)), np.repeat(members, len(members))] += 1
    row_adj[np.arange(n_rows), np.arange(n_rows)] = 0
    row_adj = row_adj / n_restarts
 
    row_adj[row_adj < 1] = 0

    """
    n_cols = len(my_collabels[0])
    col_adj = np.zeros((n_cols, n_cols))
    for labeling in my_collabels:
        n_cl = np.max(labeling) + 1
        for c in range(n_cl):
            members = np.where(labeling == c)[0]
            col_adj[np.tile(members, len(members)), np.repeat(members, len(members))] += 1
    col_adj[np.arange(n_cols), np.arange(n_cols)] = 0
    col_adj = col_adj / n_restarts
    col_adj[col_adj < 1] = 0
    """

    # Output the block labels from the min entropy solution
    entropies = np.array([a[2].entropy() for a in all_results])
    best_run = np.argmin(entropies)

    plt.plot(entropies)    
    plt.scatter(best_run, entropies[best_run], c = 'r', zorder = 2)
    plt.xticks(np.arange(n_restarts), np.arange(n_restarts))
    plt.xlabel("SBMClone run")
    plt.ylabel("Model description length")
    plt.tight_layout()
    plt.savefig(os.path.join(figure_stem, 'objectives.png'), dpi = 150)
    plt.close()

    mean_row_ari = np.mean(rows_comparison[np.tril_indices(n_restarts)])
    mean_col_ari = np.mean(cols_comparison[np.tril_indices(n_restarts)])

    return best_run, mean_row_ari, mean_col_ari

def write_tables(input_metadata, best_rowlabels, best_collabels, mean_row_ari, mean_col_ari, figure_stem):
    cell_ids, snv_ids = pickle.load(open(input_metadata, 'rb'))

    cell_table = cell_ids.to_frame()
    cell_table['block_assignment'] = best_rowlabels
    cell_table['mean_cell_block_ari'] = mean_row_ari
    cell_table['mean_snv_block_ari'] = mean_col_ari    
    cell_table.to_csv(os.path.join(figure_stem, 'cells.csv'), 
                        index = False)

    snv_table = snv_ids.to_frame()
    snv_table['block_assignment'] = best_collabels
    snv_table['mean_cell_block_ari'] = mean_row_ari
    snv_table['mean_snv_block_ari'] = mean_col_ari
    snv_table.to_csv(os.path.join(figure_stem, 'snvs.csv'), 
                        index = False)

@click.command()
@click.argument('sbmclone_results', nargs=-1)
@click.argument('input_matrix')
@click.argument('input_metadata')
@click.argument('matrix_figure')
@click.option('--verbose', '-v', is_flag=True)
def sbmclone_plots(
        input_matrix,
        input_metadata,
        sbmclone_results,
        matrix_figure,
        verbose
    ):

        
    all_results = [pickle.load(open(r, 'rb')) for r in sbmclone_results]
    if verbose:
        print(datetime.now(), f"Loaded {len(all_results)} SBMClone results")

    figure_stem = os.sep.join(matrix_figure.split(os.sep)[:-1])
    best_run, mean_row_ari, mean_col_ari = agreement_plots(all_results, figure_stem)

    if verbose:
        print(datetime.now(), "Plotted agreement plots")

    M = sparse.load_npz(input_matrix)
    best_rowlabels, best_collabels = all_results[best_run][0], all_results[best_run][1]

    write_tables(input_metadata, best_rowlabels, best_collabels, mean_row_ari, mean_col_ari, figure_stem)

    if verbose:
        print(datetime.now(), "Wrote tables to file")

    density_fig = plot_blocks(M, best_rowlabels, best_collabels)
    plt.tight_layout()
    density_fig.savefig(os.path.join(figure_stem, 'density_fig.png'), dpi = 300)
    plt.close()

    if verbose:
        print(datetime.now(), "Plotted density figure")

    matrix_fig = plot_matrix(M, best_rowlabels, best_collabels)
    plt.tight_layout()
    matrix_fig.savefig(matrix_figure, dpi = 300)
    plt.close()

    if verbose:
        print(datetime.now(), "Plotted matrix figure")

if __name__ == "__main__":
    logging.basicConfig(stream=sys.stderr, level=logging.INFO)
    sbmclone_plots()
