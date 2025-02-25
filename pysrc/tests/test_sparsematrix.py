import pytest
import pandas as pd
from scipy.sparse import csr_matrix
import numpy as np

from spectrumanalysis.sparsematrix import create_sparse_matrices


def test_basic_functionality():
    data = pd.DataFrame({
        'row': ['A', 'A', 'B', 'B', 'C'],
        'col': ['X', 'Y', 'X', 'Z', 'Z'],
        'value1': [1, 2, 3, 4, 5],
        'value2': [6, 7, 8, 9, 10]
    })
    matrices, index, columns = create_sparse_matrices(data, 'row', 'col', ['value1', 'value2'])

    assert index.name == 'row'
    assert columns.name == 'col'
    assert list(index) == ['A', 'B', 'C']
    assert list(columns) == ['X', 'Y', 'Z']
    assert (matrices['value1'].toarray() == np.array([[1, 2, 0], [3, 0, 4], [0, 0, 5]])).all()
    assert (matrices['value2'].toarray() == np.array([[6, 7, 0], [8, 0, 9], [0, 0, 10]])).all()


def test_non_string_indices():
    data = pd.DataFrame({
        'row': [1, 1, 2, 2, 3],
        'col': [101, 102, 101, 103, 103],
        'value': [1, 2, 3, 4, 5]
    })
    matrices, index, columns = create_sparse_matrices(data, 'row', 'col', ['value'])
    
    assert list(index) == [1, 2, 3]
    assert list(columns) == [101, 102, 103]
    assert (matrices['value'].toarray() == np.array([[1, 2, 0], [3, 0, 4], [0, 0, 5]])).all()


def test_empty_dataframe():
    data = pd.DataFrame({
        'row': [],
        'col': [],
        'value': []
    })
    matrices, index, columns = create_sparse_matrices(data, 'row', 'col', ['value'])
    
    assert matrices['value'].shape == (0, 0)
    assert len(index) == 0
    assert len(columns) == 0


def test_single_value_column():
    data = pd.DataFrame({
        'row': ['A', 'B', 'C'],
        'col': ['X', 'Y', 'Z'],
        'value': [1, 2, 3]
    })
    matrices, index, columns = create_sparse_matrices(data, 'row', 'col', ['value'])
    
    assert list(index) == ['A', 'B', 'C']
    assert list(columns) == ['X', 'Y', 'Z']
    assert (matrices['value'].toarray() == np.array([[1, 0, 0], [0, 2, 0], [0, 0, 3]])).all()


def test_multiple_value_columns():
    data = pd.DataFrame({
        'row': ['A', 'B', 'C'],
        'col': ['X', 'Y', 'Z'],
        'value1': [1, 2, 3],
        'value2': [4, 5, 6]
    })
    matrices, index, columns = create_sparse_matrices(data, 'row', 'col', ['value1', 'value2'])
    
    expected_value1 = np.array([[1, 0, 0], [0, 2, 0], [0, 0, 3]])
    expected_value2 = np.array([[4, 0, 0], [0, 5, 0], [0, 0, 6]])
    
    assert (matrices['value1'].toarray() == expected_value1).all()
    assert (matrices['value2'].toarray() == expected_value2).all()


def test_rep_zeros():
    data = pd.DataFrame({
        'row': ['A', 'B', 'C'],
        'col': ['X', 'Y', 'Z'],
        'value1': [1, 0, 3],
        'value2': [4, 5, 0]
    })
    matrices, index, columns = create_sparse_matrices(data, 'row', 'col', ['value1', 'value2'])
    
    assert (matrices['value1'].tocoo().data != 0).all()
    assert (matrices['value2'].tocoo().data != 0).all()


if __name__ == '__main__':
    pytest.main()

