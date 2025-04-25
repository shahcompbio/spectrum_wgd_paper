#!/usr/bin/env python

import logging
import sys
import pandas as pd
import os
import click

@click.command()
@click.argument('cell_lists_dir')
@click.argument('segments_dir')
@click.argument('output_dir')
@click.argument('signals_table')
@click.argument('medicc2_suffix')
@click.argument('outfile')
def create_medicc2_cohort(cell_lists_dir,
                           segments_dir,
                           output_dir,
                           signals_table,
                           medicc2_suffix,
                           outfile
                          ):
    signals_table = pd.read_csv(signals_table).set_index('isabl_patient_id')
    
    patients = [p.split('.')[0] for p in os.listdir(segments_dir)]

    rows = []
    for p in patients:
        segments_file = os.path.join(segments_dir, p + '.csv.gz')
        cell_list_file = os.path.join(cell_lists_dir, p + '.txt')
        assert os.path.exists(cell_list_file), f"Missing cell list file for patient: {p}"
        
        id = p + medicc2_suffix
        
        rows.append([p,
                     id,
                     signals_table.loc[p, 'MONDRIAN_SIGNALS_hscn'],
                     segments_file,
                     '--wgd-x2',
                     '--allele_specific',
                     cell_list_file,
                     os.path.join(output_dir, id)])
    cohort = pd.DataFrame(rows, columns = ['patient', 'id', 'signals_filename', 'segments_filename', 'medicc_args',
       'allele_specific', 'cell_list', 'output_directory'])
    cohort.to_csv(outfile, index = False)  

if __name__ == "__main__":
    logging.basicConfig(stream=sys.stderr, level=logging.INFO)
    create_medicc2_cohort()

