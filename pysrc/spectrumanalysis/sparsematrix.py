from scipy.sparse import csr_matrix


def create_sparse_matrices(data, row, col, values):
    """
    Create sparse matrices from a long form pandas dataframe.

    Note this function will also eliminate zero entries from each returned matrix.

    Parameters
    ----------
    data : pandas.DataFrame
        The input dataframe containing the data.
    row : str
        The column name to be used as the row index.
    col : str
        The column name to be used as the column index.
    values : list of str
        The column names to be used as the values in the sparse matrices.

    Returns
    -------
    matrices : dict of scipy.sparse.csr_matrix
        A dictionary where keys are the value column names and values are the corresponding sparse matrices.
    index : pandas.Index
        The index categories derived from the `row` column.
    columns : pandas.Index
        The column categories derived from the `col` column.
    """
    data[row] = data[row].astype('category')
    data[col] = data[col].astype('category')

    index = data[row].cat.categories
    index.name = row

    columns = data[col].cat.categories
    columns.name = col

    matrices = {}
    for value in values:
        matrices[value] = csr_matrix(
            (data[value], (data[row].cat.codes, data[col].cat.codes)),
            shape=(index.size, columns.size))
        matrices[value].eliminate_zeros()

    return matrices, index, columns


