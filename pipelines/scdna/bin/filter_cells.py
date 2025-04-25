
import logging
import sys
import pandas as pd
import click

@click.command()
@click.argument('cell_table_filename')
@click.argument('cell_nnd_filename')
@click.argument('segment_info_filename')
@click.argument('multipolar_nnd_threshold', type=float)
@click.argument('filtered_cell_table_filename')
def filter_cells(
        cell_table_filename,
        cell_nnd_filename,
        segment_info_filename,
        multipolar_nnd_threshold,
        filtered_cell_table_filename):

    cell_info = pd.read_csv(cell_table_filename)
    cell_nnd = pd.read_csv(cell_nnd_filename)
    noisy_cells = pd.read_csv(segment_info_filename)

    cell_info = cell_info.merge(cell_nnd, how='left')
    cell_info['multipolar'] = False
    cell_info.loc[cell_info['nnd'] > multipolar_nnd_threshold, 'multipolar'] = True

    cell_info = cell_info.merge(noisy_cells[['brief_cell_id', 'n_135_segments', 'longest_135_segment', 'prop_odd', 'prop_135']], how='left')
    assert not cell_info['n_135_segments'].isnull().any().any()

    cell_info['include_cell'] = (
        (cell_info['is_normal'] == False) &
        (cell_info['is_aberrant_normal_cell'] == False) &
        (cell_info['is_doublet'] == 'No') &
        (cell_info['is_s_phase_thresholds'] == False) &
        (cell_info['longest_135_segment'] >= 20)
    )

    cell_info.to_csv(filtered_cell_table_filename, index=False, compression={'method':'gzip', 'mtime':0, 'compresslevel':9})


if __name__ == "__main__":
    logging.basicConfig(stream=sys.stderr, level=logging.INFO)
    filter_cells()

