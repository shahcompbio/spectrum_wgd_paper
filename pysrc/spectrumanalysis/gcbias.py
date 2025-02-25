from sklearn.preprocessing import PolynomialFeatures
import statsmodels.api as sm
import numpy as np

import scgenome


def recorrect_cell_gc(cell_data, raw_col='copy'):
    """
    Recorrects the cell GC bias in the given cell data.

    Parameters:
    - cell_data (pandas.DataFrame): The input cell data.
    - raw_col (str): The name of the column containing raw data. Default is 'copy'.

    Returns:
    - pandas.DataFrame: The cell data with GC bias recorrected.

    """

    training_data = cell_data.copy()
    training_data['copy_norm'] = training_data[raw_col] / training_data['state']
    training_data = training_data[np.isfinite(training_data['gc']) & np.isfinite(training_data['copy_norm'])]

    polynomial_features = PolynomialFeatures(degree=5)
    xp_train = polynomial_features.fit_transform(training_data['gc'].values[:, np.newaxis])

    model = sm.OLS(training_data['copy_norm'].values, xp_train).fit()

    xp_fit = polynomial_features.fit_transform(cell_data['gc'].values[:, np.newaxis])
    cell_data['copy_gc_recorrected'] = cell_data[raw_col] / model.predict(xp_fit)

    return cell_data


def recorrect_gc(adata, raw_col='copy'):
    """
    Recorrects the GC bias in the given AnnData object.

    Parameters:
        adata (AnnData): The AnnData object containing the data to be recorrected.
        raw_col (str): The name of the column in `adata` that contains the raw copy data. Default is 'copy'.

    Returns:
        AnnData: The updated AnnData object with the GC bias recorrected.
    """
    copy_gc_recorrected = np.zeros(adata.shape)
    for idx, cell_id in enumerate(adata.obs.index):
        cell_data = scgenome.tl.get_obs_data(adata, cell_id, layer_names=(raw_col, 'state'))
        cell_data = recorrect_cell_gc(cell_data, raw_col=raw_col)
        assert (cell_data.index == adata.var.index).all()
        copy_gc_recorrected[idx, :] = cell_data['copy_gc_recorrected'].values
    adata.layers['copy2'] = copy_gc_recorrected    

    return adata