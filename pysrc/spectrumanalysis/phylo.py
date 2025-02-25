import pandas as pd


def get_parent(tree, child_clade):
    """ Get the parent of node in a tree

    Parameters
    ----------
    tree : Bio.Phylo.BaseTree.Tree
        BioPython tree
    child_clade : Bio.Phylo.BaseTree.Clade
        clade within the tree

    Returns
    -------
    Bio.Phylo.BaseTree.Clade
        parent clade

    Raises
    ------
    Exception
        no parent found in tree
    """
    node_path = tree.get_path(child_clade)
    if len(node_path) < 1:
        raise Exception(f'error calculating path for {child_clade.name}')
    elif len(node_path) == 1:
        return tree.clade
    return node_path[-2]


def get_cell_parents(tree):
    """ Get each cells parents

    Parameters
    ----------
    tree : Bio.Phylo.BaseTree
        BioPython Tree

    Returns
    -------
    pandas.DataFrame
        table of cell_id, parent_cell_id
    """
    parents = []
    cell_ids = []

    for a in tree.get_terminals():
        if a.name == 'diploid':
            continue
        parents.append({
            'cell_id': a.name,
            'parent_cell_id': get_parent(tree, a).name,
        })
        cell_ids.append(a.name)

    parents = pd.DataFrame(parents)

    return parents


def get_path(tree, cell_id):
    """ Get a path from the root to a cell

    Parameters
    ----------
    tree : Bio.Phylo.BaseTree.Tree
        BioPython Tree
    cell_id : str
        name of the terminal clade
    
    Returns
    -------
    list
        list of node names
    """

    clade = next(tree.find_clades(cell_id))

    return ['diploid'] + [tree.clade.name] + [c.name for c in tree.get_path(clade)]
