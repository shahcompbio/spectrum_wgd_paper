import anndata as ad

from .phylo import get_cell_parents


def calculate_cell_changes(tree, adata):
    parents = get_cell_parents(tree)

    X = adata[parents['cell_id']].X - adata[parents['parent_cell_id']].X

    layers = {}
    layers['is_gain'] = (X > 0)
    layers['is_loss'] = (X < 0)

    cell_changes = ad.AnnData(
        X,
        layers=layers,
        obs=adata[parents['cell_id']].obs,
        var=adata.var,
    )

    cell_changes
    
    return cell_changes


def calculate_cell_changes_as(tree, adata):
    parents = get_cell_parents(tree)

    X = adata[parents['cell_id']].X - adata[parents['parent_cell_id']].X

    layers = {}
    layers['cn_a'] = adata[parents['cell_id']].layers['cn_a'] - adata[parents['parent_cell_id']].layers['cn_a']
    layers['cn_b'] = adata[parents['cell_id']].layers['cn_b'] - adata[parents['parent_cell_id']].layers['cn_b']
    layers['is_gain'] = (layers['cn_a'] > 0) | (layers['cn_b'] > 0)
    layers['is_loss'] = (layers['cn_a'] < 0) | (layers['cn_b'] < 0)
    layers['is_mirror'] = ((layers['cn_a'] < 0) & (layers['cn_b'] > 0) | (layers['cn_a'] > 0) & (layers['cn_b'] < 0))

    cell_changes = ad.AnnData(
        X,
        layers=layers,
        obs=adata[parents['cell_id']].obs,
        var=adata.var,
    )

    cell_changes
    
    return cell_changes


