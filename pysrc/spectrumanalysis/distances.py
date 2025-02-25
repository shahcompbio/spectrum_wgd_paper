import numpy as np
import pandas as pd
import tqdm


def compute_mean_sq_diff(adata_1, adata_2, cell_id_1, cell_id_2, layer):
    cell_idx_1 = adata_1.obs.index.get_loc(cell_id_1)
    cell_idx_2 = adata_2.obs.index.get_loc(cell_id_2)
    distance = np.nanmean(np.square(
        adata_1.layers[layer][cell_idx_1, :] - adata_2.layers[layer][cell_idx_2, :]))
    return distance


def compute_mean_sq_diff_wgd(adata_1, adata_2, cell_id_1, cell_id_2, layer):
    cell_idx_1 = adata_1.obs.index.get_loc(cell_id_1)
    cell_idx_2 = adata_2.obs.index.get_loc(cell_id_2)
    distance0 = np.nanmean(np.square(
        adata_1.layers[layer][cell_idx_1, :] - adata_2.layers[layer][cell_idx_2, :]))
    distance1 = np.nanmean(np.square(
        adata_1.layers[layer][cell_idx_1, :] * 2 - adata_2.layers[layer][cell_idx_2, :]))
    distance2 = np.nanmean(np.square(
        adata_1.layers[layer][cell_idx_1, :] - adata_2.layers[layer][cell_idx_2, :] * 2))
    return min(distance0, distance1, distance2)


def compute_mean_is_diff(adata_1, adata_2, cell_id_1, cell_id_2, layer):
    cell_idx_1 = adata_1.obs.index.get_loc(cell_id_1)
    cell_idx_2 = adata_2.obs.index.get_loc(cell_id_2)
    distance = np.nanmean(
        adata_1.layers[layer][cell_idx_1, :] != adata_2.layers[layer][cell_idx_2, :])
    return distance


def compute_mean_is_diff_wgd(adata_1, adata_2, cell_id_1, cell_id_2, layer):
    cell_idx_1 = adata_1.obs.index.get_loc(cell_id_1)
    cell_idx_2 = adata_2.obs.index.get_loc(cell_id_2)
    distance0 = np.nanmean(
        adata_1.layers[layer][cell_idx_1, :] != adata_2.layers[layer][cell_idx_2, :])
    distance1 = np.nanmean(
        adata_1.layers[layer][cell_idx_1, :] * 2 != adata_2.layers[layer][cell_idx_2, :])
    distance2 = np.nanmean(
        adata_1.layers[layer][cell_idx_1, :] != adata_2.layers[layer][cell_idx_2, :] * 2)
    return min(distance0, distance1, distance2)


def compute_largest_segment_is_diff(adata_1, adata_2, cell_id_1, cell_id_2, layer):
    assert np.array_equal(adata_1.var.index, adata_2.var.index)
    cell_idx_1 = adata_1.obs.index.get_loc(cell_id_1)
    cell_idx_2 = adata_2.obs.index.get_loc(cell_id_2)
    is_diff_state = adata_1.layers[layer][cell_idx_1, :] != adata_2.layers[layer][cell_idx_2, :]
    if is_diff_state.sum() == 0:
        return 0
    # difference between adjacent bins, then cumsum to give each consecutive run a different integer id
    is_diff_state_group = np.zeros(is_diff_state.shape, dtype=int)
    is_diff_state_group[1:] = np.cumsum(np.diff(is_diff_state) | np.diff(adata_1.var['chr'].cat.codes.values))
    # count the number of each integer id
    is_diff_bin_count = np.unique(is_diff_state_group[is_diff_state], return_counts=True)[1]
    # return the max length of all segments
    max_is_diff_bin_count = is_diff_bin_count.max()
    return max_is_diff_bin_count


def compute_total_segment_is_diff_threshold(adata_1, adata_2, cell_id_1, cell_id_2, layer, threshold):
    assert np.array_equal(adata_1.var.index, adata_2.var.index)
    cell_idx_1 = adata_1.obs.index.get_loc(cell_id_1)
    cell_idx_2 = adata_2.obs.index.get_loc(cell_id_2)
    is_diff_state = adata_1.layers[layer][cell_idx_1, :] != adata_2.layers[layer][cell_idx_2, :]
    if is_diff_state.sum() == 0:
        return 0
    # difference between adjacent bins, then cumsum to give each consecutive run a different integer id
    is_diff_state_group = np.zeros(is_diff_state.shape, dtype=int)
    is_diff_state_group[1:] = np.cumsum(np.diff(is_diff_state) | np.diff(adata_1.var['chr'].cat.codes.values))
    # count the number of each integer id
    is_diff_bin_count = np.unique(is_diff_state_group[is_diff_state], return_counts=True)[1]
    # return the total number of bins different in segments that are above a given threshold in length
    return is_diff_bin_count[is_diff_bin_count >= threshold].sum()


def compute_distances(adata_1, adata_2=None, obs_ids_1=None):
    """ Compute distances between observations in anndatas

    Parameters
    ----------
    adata_1 : AnnData
        AnnData containing copy and state matrices
    adata_2 : AnnData, optional
        Second AnnData to compute distances against, by default None, in which case adata_1 is used
    obs_ids_1 : list, optional
        cells from adata_1 to restrict to, by default None, in which case all cells are used

    Returns
    -------
    pandas.DataFrame
        distances between observations in adata_1 and adata_2

    Raises
    ------
    ValueError
        if obs_id is not in adata
    """

    # Sort adata by segment chromosome, start position, required by distance computation
    adata_1 = adata_1[:, adata_1.var.sort_values(['chr', 'start']).index].copy()
    if adata_2 is not None:
        adata_2 = adata_2[:, adata_2.var.sort_values(['chr', 'start']).index].copy()

    if adata_2 is None:
        adata_2 = adata_1

    if obs_ids_1 is None:
        obs_ids_1 = adata_1.obs.index

    distances = []
    for obs_id in tqdm.tqdm(obs_ids_1):
        if obs_id not in adata_1.obs.index:
            raise ValueError(f'obs id {obs_id} not in adata')

        for obs_id2 in adata_2.obs.index:
            distances.append({
                'obs_id_1': obs_id,
                'obs_id_2': obs_id2,
                'copy_mean_sq_distance': compute_mean_sq_diff(adata_1, adata_2, obs_id, obs_id2, 'copy'),
                'copy_mean_sq_wgd_distance': compute_mean_sq_diff_wgd(adata_1, adata_2, obs_id, obs_id2, 'copy'),
                'state_mean_is_diff_distance': compute_mean_is_diff(adata_1, adata_2, obs_id, obs_id2, 'state'),
                'copy_mean_is_diff_wgd_distance': compute_mean_is_diff_wgd(adata_1, adata_2, obs_id, obs_id2, 'state'),
                'state_largest_segment_is_diff_distance': compute_largest_segment_is_diff(adata_1, adata_2, obs_id, obs_id2, 'state'),
                'state_total_segment_is_diff_threshold_4': compute_total_segment_is_diff_threshold(adata_1, adata_2, obs_id, obs_id2, 'state', 4),
                'state_total_segment_is_diff_threshold_10': compute_total_segment_is_diff_threshold(adata_1, adata_2, obs_id, obs_id2, 'state', 10),
                'state_total_segment_is_diff_threshold_20': compute_total_segment_is_diff_threshold(adata_1, adata_2, obs_id, obs_id2, 'state', 20),
            })
    distances = pd.DataFrame(distances)
    
    return distances
