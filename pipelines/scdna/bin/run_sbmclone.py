#!/usr/bin/env python

import logging
import sys
from time import perf_counter
from collections import Counter
import graph_tool
from graph_tool.inference import minimize as gt_min
from scipy.sparse import load_npz
import numpy as np
import click
import pickle

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

def SBMClone(M, min_blocks = 2, max_blocks = None, nested = True, verbose = True, seed = 0):
    """
    Apply SBMClone to the given biadjacency matrix M (represented as scipy sparse matrix).
    Returns row and column block assignments.
    """
    assert min_blocks is None or isinstance(min_blocks, int)
    assert max_blocks is None or isinstance(max_blocks, int)
    start = perf_counter()

    m, n = M.shape
    G, label2id, vtype = construct_graph_graphtool(M)
    
    graph_tool.seed_rng(seed)
    np.random.seed(seed)
    if max_blocks is None:
        max_blocks = 1000
    if nested:
        r = gt_min.minimize_nested_blockmodel_dl(G, multilevel_mcmc_args={'B_min':min_blocks, 'B_max':max_blocks}, state_args = {'clabel': vtype})
        b = r.get_bs()[0]
    else:
        r = gt_min.minimize_blockmodel_dl(G, multilevel_mcmc_args={'B_min':min_blocks, 'B_max':max_blocks},state_args = {'clabel': vtype})
        b = r.get_blocks()

    end = perf_counter()
    if verbose:
        print("Block inference elapsed time: {}".format(str(end - start)))

    labels = blockmodel_to_labels(b, label2id, n_cells = m, n_muts = n)
    relabels = refactor_labels(M, *labels)
        
    return relabels, r

def construct_graph_graphtool(M):
    """
    Given an (unweighted, undirected) biadjacency matrix M, construct a graph_tool graph object with the corresponding edges.
    Returns the graph G as well as a mapping from vertices of G to cells (rows) and mutations (columns) in M. 
    """
    M = M.tocoo()
    G = graph_tool.Graph(directed = False)
    
    vtype = G.new_vertex_property("int")

    label2id = {}
    for i,j,value in zip(M.row, M.col, M.data):
        assert value == 1
        cell_key = 'cell{}'.format(i)
        mut_key = 'mut{}'.format(j)
        if cell_key in label2id:
            v = label2id[cell_key]
        else:
            v = G.add_vertex()
            label2id[cell_key] = int(v)
            vtype[v] = 1
            
        if mut_key in label2id:
            w = label2id[mut_key]
        else:
            w = G.add_vertex()
            label2id[mut_key] = int(w)
            vtype[w] = 2

        G.add_edge(v, w)  
    return G, label2id, vtype


def blockmodel_to_labels(b, label2id, n_cells = None, n_muts = None, maxblocks = 100):
    """
    Converts the graph_tool blockmodel return objects to partition vectors for cells (rows) and mutations (columns).
    """
    if n_cells is None:
        n_cells = max([int(x[4:]) if x.startswith('cell') else 0 for x in label2id.keys()])
    if n_muts is None:
        n_muts = max([int(x[3:]) if x.startswith('mut') else 0 for x in label2id.keys()])
    assert n_cells > 0
    assert n_muts > 0
    
    
    cell_array = [(b[label2id['cell{}'.format(i)]] if 'cell{}'.format(i) in label2id else maxblocks + 1) for i in range(n_cells)]
    temp = sorted(list(Counter(cell_array).keys()))
    cell_idx_to_blocknum = {temp[i]:i + 1 for i in range(len(temp))}
    cell_array = [cell_idx_to_blocknum[a] for a in cell_array]
    
    mut_array = [(b[label2id['mut{}'.format(i)]] if 'mut{}'.format(i) in label2id else maxblocks + 1) for i in range(n_muts)]
    temp = sorted(list(Counter(mut_array).keys()))
    mut_idx_to_blocknum = {temp[i]:i + 1 for i in range(len(temp))}
    mut_array = [mut_idx_to_blocknum[a] for a in mut_array]
    return cell_array, mut_array

reverse_dict = lambda d:  {v:k for k,v in list(d.items())}


@click.command()
@click.argument('input_matrix')
@click.argument('output_filename')
@click.argument('random_seed', type = int)
@click.option('--max_blocks', is_flag = False, default = 10)
def run_sbmclone(
        input_matrix,
        output_filename,
        random_seed,
        max_blocks
    ):

    M = load_npz(input_matrix)
    (row_labels, col_labels), r = SBMClone(M, max_blocks = max_blocks, seed = random_seed)
    with open(output_filename, 'wb') as f:
        pickle.dump((row_labels, col_labels, r), f)

if __name__ == "__main__":
    logging.basicConfig(stream=sys.stderr, level=logging.INFO)
    run_sbmclone()
