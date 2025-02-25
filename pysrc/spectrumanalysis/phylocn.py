import numpy as np
import pandas as pd


def count_transitions(clade, adata, layer, transition_layer):
    """ Recursively count the number of cn transitions from the root for each node/bin

    Parameters
    ----------
    clade : Bio.Phylo.BaseTree.Clade
        clade to identify transitions
    adata : anndata.AnnData
        copy number data and where to store transitions
    layer : str
        layer of copy number data
    transition_layer : str
        layer for transitions
    """
    if clade.name is None:
        clade_name = 'diploid'
    else:
        clade_name = clade.name
    clade_idx = adata.obs.index.get_loc(clade_name)
    clade_cn = np.array(adata.layers[layer][clade_idx, :])

    for child in clade.clades:
        child_idx = adata.obs.index.get_loc(child.name)
        child_cn = np.array(adata.layers[layer][child_idx, :])
        
        is_different = np.array((clade_cn != child_cn) * 1)

        adata.layers[transition_layer][child_idx, :] = adata.layers[transition_layer][clade_idx, :] + is_different
        
        count_transitions(child, adata, layer, transition_layer)


def count_visited(clade, adata, layer, visit_counts_layer, n_visited=None):
    """ Recursively count number of states visited on the path from root for each node/bin

    Parameters
    ----------
    clade : Bio.Phylo.BaseTree.Clade
        clade to calculate visited counts
    adata : anndata.AnnData
        copy number data and where to store visited counts
    layer : str
        layer of copy number data
    visit_counts_layer : str
        layer for visited counts
    n_visited : numpy.array, optional
        matrix of visited states from parent, shape=(n_var,n_states), by default None
    """
    if clade.name is None:
        clade_name = 'diploid'
    else:
        clade_name = clade.name
    
    if n_visited is None:
        n_states = int(adata.layers[layer].max()) + 1
        n_visited = np.zeros((adata.shape[1], n_states))

    clade_idx = adata.obs.index.get_loc(clade_name)
    clade_cn = np.array(adata.layers[layer][clade_idx, :]).astype(int)

    n_visited[list(range(len(clade_cn))), clade_cn] = 1

    adata.layers[visit_counts_layer][clade_idx, :] = np.sum(n_visited, axis=1)

    for child in clade.clades:
        count_visited(child, adata, layer, visit_counts_layer, n_visited=np.array(n_visited))


def count_unique_visited(clade, adata, layer):
    """ Recursively count number of unique visits to states across the tree for each bin

    Parameters
    ----------
    clade : Bio.Phylo.BaseTree.Clade
        clade to calculate visited counts
    adata : anndata.AnnData
        copy number data and where to store visited counts
    layer : str
        layer of copy number data

    Returns
    -------
    numpy.array
        matrix of counts of transitions to each state, shape=(n_var,n_states)
    """
    if clade.name is None:
        clade_name = 'diploid'
    else:
        clade_name = clade.name

    n_states = int(adata.layers[layer].max()) + 1
    n_visited = np.zeros((adata.shape[1], n_states))

    clade_idx = adata.obs.index.get_loc(clade_name)
    clade_cn = np.array(adata.layers[layer][clade_idx, :]).astype(int)

    for child in clade.clades:
        child_idx = adata.obs.index.get_loc(child.name)
        child_cn = np.array(adata.layers[layer][child_idx, :])
        
        is_different = np.array((clade_cn != child_cn) * 1)

        # Add transitions for this child clade
        # add one to each child copy number state when there is a transition for that
        # bin to that copy number state
        n_visited[list(range(len(clade_cn))), clade_cn] += is_different

        # Recursively add transitions for this child clades children
        n_visited += count_unique_visited(child, adata, layer)

    return n_visited



def calculate_homoplasy_score(tree, adata, layer):
    """ Compute homplasy score per bin

    Definition: Given a bin and copy number state, if there was only one
    transition to that state for that bin throughout the entire tree then
    the homoplasy for that bin for that state is 0. If throughout the tree
    there were n instances of transitions to that state for that bin then
    the homoplasy is n-1. If there were 0 transitions to that state the
    homoplasy for that bin for that state is 0. The homoplasy score is the
    sum of homoplasy across all states for each bin.

    Parameters
    ----------
    clade : Bio.Phylo.BaseTree.Tree
        tree to calculate homoplasy score
    adata : anndata.AnnData
        copy number data and where to store visited counts
    layer : str
        layer of copy number data

    Returns
    -------
    numpy.array
        homoplasy score per bin, shape=(n_var,)
    """
    unique_visited = count_unique_visited(tree.clade, adata, layer)

    # Homoplasy is defined as the additional visits in excess
    # of 1 visit per state for each visited state.
    unique_visited[unique_visited > 0] -= 1

    return unique_visited.sum(axis=1)



def compute_cn_change(tree, adata, layer, cn_change_layer):
    """ Compute cn change between parent and child and add to anndata

    Parameters
    ----------
    tree : Bio.Phylo.BaseTree.Tree
        tree on which to compute cn changes
    adata : anndata.AnnData
        copy number data and where to store transitions
    layer : str
        layer of copy number data
    cn_change_layer : str
        layer for cn changes
    """

    child_parent = {tree.clade.name: 'diploid'}
    for clade in tree.find_clades():
        for child in clade.clades:
            child_parent[child.name] = clade.name

    for child_name, parent_name in child_parent.items():
        child_idx = adata.obs.index.get_loc(child_name)
        parent_idx = adata.obs.index.get_loc(parent_name)

        child_cn = np.array(adata.layers[layer][child_idx, :])
        parent_cn = np.array(adata.layers[layer][parent_idx, :])

        cn_change = np.array(child_cn - parent_cn)

        adata.layers[cn_change_layer][child_idx, :] = cn_change


def compute_wgd_cn_change(tree, adata, layer, wgd_timing, cn_change_layer):
    """ Compute cn changes between parent and child for a specific wgd timing

    Parameters
    ----------
    tree : Bio.Phylo.BaseTree.Tree
        tree on which to compute cn changes
    adata : anndata.AnnData
        copy number data and where to store transitions
    layer : str
        layer of copy number data
    wgd_timing : str
        timing relative to wgd, 'pre' or 'post'
    cn_change_layer : str
        layer for cn changes
    """

    # Pre post WGD changes that minimize total changes
    n_states = int(adata.layers[layer].max() + 1)
    pre_post_changes = calculate_pre_post_changes(n_states)

    child_parent = {tree.clade.name: 'diploid'}
    for clade in tree.find_clades():
        for child in clade.clades:
            if child.wgd:
                child_parent[child.name] = clade.name

    for child_name, parent_name in child_parent.items():
        child_idx = adata.obs.index.get_loc(child_name)
        parent_idx = adata.obs.index.get_loc(parent_name)

        child_cn = np.array(adata.layers[layer][child_idx, :])
        parent_cn = np.array(adata.layers[layer][parent_idx, :])

        cn_change = pre_post_changes.loc[pd.IndexSlice[zip(parent_cn, child_cn)], wgd_timing].values

        adata.layers[cn_change_layer][child_idx, :] = cn_change


def calculate_score_recursive(clade, transition, transition_wgd, n_states, n_bins):
    """ Tree recursion for maximum parsimony with WGD

    Parameters
    ----------
    clade : Bio.Phylo.BaseTree.Clade
        clade to score recursively
    transition : numpy.array
        state transition matrix
    transition_wgd : numpy.array
        state transition matrix for a wgd transition
    n_states : int
        number of copy number states
    n_bins : int
        number of bins
    """
    if not clade.is_terminal():

        clade.state_scores = np.zeros((n_bins, n_states))

        for child in clade.clades:
            calculate_score_recursive(child, transition, transition_wgd, n_states, n_bins)

            if child.wgd:
                child_state_scores = child.state_scores[:, np.newaxis, :] + transition_wgd[np.newaxis, :, :]
            else:
                child_state_scores = child.state_scores[:, np.newaxis, :] + transition[np.newaxis, :, :]

            child.state_backtrack = np.argmin(child_state_scores, axis=2)
            clade.state_scores += np.amin(child_state_scores, axis=2)


def calculate_score_recursive_tree(tree, transition, transition_wgd, n_states, n_bins):
    """ Tree recursion for maximum parsimony with WGD

    Parameters
    ----------
    clade : Bio.Phylo.BaseTree.Tree
        clade to score recursively
    transition : numpy.array
        state transition matrix
    transition_wgd : numpy.array
        state transition matrix for a wgd transition
    n_states : int
        number of copy number states
    n_bins : int
        number of bins
    """
    calculate_score_recursive(tree.clade, transition, transition_wgd, n_states, n_bins)

    if tree.clade.wgd:
        child_state_scores = tree.clade.state_scores[:, np.newaxis, :] + transition_wgd[np.newaxis, :, :]
    else:
        child_state_scores = tree.clade.state_scores[:, np.newaxis, :] + transition[np.newaxis, :, :]

    tree.clade.state_backtrack = np.argmin(child_state_scores, axis=2)
    tree.state_scores = np.amin(child_state_scores, axis=2)


def backtrack_state_recursive(clade, state):
    """ Tree backtracking for maximum parsimony state

    Parameters
    ----------
    clade : Bio.Phylo.BaseTree.Clade
        clade to backtrack recursively
    state : numpy.array
        state of bins for this clade

    Assumes state_scores and state_backtrack are valid for all nodes.  At return,
    each node has a state variable representing state for one maximum parsimony
    solution.
    """
    clade.state = state

    for child in clade.clades:
        backtrack_state_recursive(child, child.state_backtrack[range(clade.state.shape[0]), clade.state])


def backtrack_state_recursive_tree(tree, ancestral_state=None):
    """ Tree backtracking for maximum parsimony state

    Parameters
    ----------
    tree : Bio.Phylo.BaseTree.Tree
        clade to backtrack recursively
    ancestral_state : numpy.array, optional
        ancestral state of bins for the tree, by default None

    Assumes state_scores and state_backtrack are valid for all nodes.  At return,
    each node has a state variable representing state for one maximum parsimony
    solution.
    """
    if ancestral_state is None:
        tree.state = np.argmin(tree.state_scores, axis=1)
    else:
        tree.state = ancestral_state

    backtrack_state_recursive(tree.clade, state=tree.clade.state_backtrack[range(tree.state.shape[0]), tree.state])


def generate_uniform_transition(n_states):
    """ Generate uniform transition matrix

    Parameters
    ----------
    n_states : int
        number of copy number states

    Returns
    -------
    numpy.array
        transition matrix
    """
    uniform_transition = np.zeros(shape=(n_states, n_states))
    for state in range(n_states):
        for child_state in range(n_states):
            if (state == 0) and (child_state > 0):
                uniform_transition[state, child_state] = np.inf
            elif state == child_state:
                uniform_transition[state, child_state] = 0
            else:
                uniform_transition[state, child_state] = 1
    return uniform_transition


def generate_linear_transition(n_states):
    """ Generate linear transition matrix

    Parameters
    ----------
    n_states : int
        number of copy number states

    Returns
    -------
    numpy.array
        transition matrix
    """
    linear_transition = np.zeros(shape=(n_states, n_states))
    for state in range(n_states):
        for child_state in range(n_states):
            if (state == 0) and (child_state > 0):
                linear_transition[state, child_state] = np.inf
            else:
                linear_transition[state, child_state] = abs(state - child_state)
    return linear_transition


def generate_linear_wgd_transition(n_states):
    """ Generate linear wgd transition matrix

    cost = min_{pre, post} abs(pre) + abs(post)
    such that: 2 * (state + pre) + post = child_state

    Parameters
    ----------
    n_states : int
        number of copy number states
    
    Returns
    -------
    numpy.array
        transition matrix
    """
    linear_wgd_transition = np.full((n_states, n_states), np.inf)
    for state in range(n_states):
        for child_state in range(n_states):
            if (state == 0) and (child_state > 0):
                linear_wgd_transition[state, child_state] = np.inf
            else:
                for pre in range(-n_states, n_states):
                    if (state + pre) < 0:
                        continue
                    if (state + pre) == 0 and child_state != 0:
                        continue
                    post = child_state - 2 * (state + pre)
                    cost = abs(pre) + abs(post)
                    if cost < linear_wgd_transition[state, child_state]:
                        linear_wgd_transition[state, child_state] = cost

    return linear_wgd_transition


def add_states_to_tree(tree, adata, layers, n_bins, n_states):
    """ Add copy number states to leaves of tree

    Parameters
    ----------
    tree : Bio.Phylo.BaseTree.Clade
        tree to modify with leaf states
    adata : anndata.AnnData
        copy number states
    layers : list of str
        layers of copy number states
    n_bins : int
        total number of bins across all layers
    n_states : int
        number of distinct copy number states
    """
    for clade in tree.find_clades():
        if clade.is_terminal():
            states = np.concatenate([adata[clade.name].layers[layer][0] for layer in layers])
            assert len(states) == n_bins
            assert np.all(np.isfinite(states))
            states = states.astype(int)
            assert states.min() >= 0
            assert states.max() < n_states
            clade.state_scores = np.full((n_bins, n_states), np.inf)
            clade.state_scores[list(range(n_bins)), states] = 0


def calculate_pre_post_changes(n_states):
    """ Calculate pre/post WGD changes for given parent child state

    Parameters
    ----------
    n_states : int
        number of copy number states

    Returns
    -------
    pandas.DataFrame
        table of pre/post changes
    """
    pre_post_changes = []

    pre_post_changes.append({'parent': 0, 'child': 0, 'pre': 0, 'post': 0, 'cost': 0})

    for state in range(1, n_states):
        for child_state in range(n_states):
            for pre in range(-n_states, n_states):
                if (state + pre) < 0:
                    continue
                if (state + pre) == 0 and child_state != 0:
                    continue
                post = child_state - 2 * (state + pre)
                cost = abs(pre) + abs(post)
                pre_post_changes.append({'parent': state, 'child': child_state, 'pre': pre, 'post': post, 'cost': cost})

    pre_post_changes = pd.DataFrame(pre_post_changes)

    pre_post_changes['min_cost'] = pre_post_changes.groupby(['parent', 'child'])['cost'].transform('min')
    pre_post_changes = pre_post_changes.query('cost == min_cost')
    assert not pre_post_changes.duplicated(['parent', 'child']).any()
    pre_post_changes = pre_post_changes.set_index(['parent', 'child'])

    return pre_post_changes

