import logging
import sys
import pandas as pd
import click

@click.command()
@click.argument('cell_table_filename')
@click.argument('cell_nnd_filename')
@click.argument('segment_info_filename')
@click.argument('patient_id')
@click.argument('multipolar_nnd_threshold', type=float)
@click.argument('output_filename')
def create_sbmclone_cell_list(
        cell_table_filename,
        cell_nnd_filename,
        segment_info_filename,  
        patient_id,
        multipolar_nnd_threshold,
        output_filename):

    cell_info = pd.read_csv(cell_table_filename)
    cell_nnd = pd.read_csv(cell_nnd_filename)
    noisy_cells = pd.read_csv(segment_info_filename)

    cell_info = cell_info.merge(cell_nnd, how='left')
    cell_info['multipolar'] = False
    cell_info.loc[cell_info['nnd'] > multipolar_nnd_threshold, 'multipolar'] = True

    cell_info = cell_info.merge(noisy_cells[['brief_cell_id', 'n_135_segments', 'longest_135_segment']], how='left')
    assert not cell_info['n_135_segments'].isnull().any().any()

    cell_info['include_cell'] = (
        (cell_info['is_normal'] == False) &
        (cell_info['is_aberrant_normal_cell'] == False) &
        (cell_info['is_doublet'] == 'No') &
        (cell_info['is_s_phase_thresholds'] == False) &
        (cell_info['longest_135_segment'] >= 20)
    )

    # for SBMClone, include multipolar cells
    my_cells = cell_info[(cell_info.include_cell) & 
                        (cell_info.patient_id == patient_id)].brief_cell_id
    my_cells.to_csv(output_filename, index=False, header=False)

if __name__ == "__main__":
    logging.basicConfig(stream=sys.stderr, level=logging.INFO)
    create_sbmclone_cell_list()
