import logging
import sys
import pandas as pd
import anndata as ad
import numpy as np
import pyranges as pr
import click

import scgenome


@click.command()
@click.argument('amp_events_filename')
@click.argument('gtf_filename')
@click.argument('amp_events_genes_filename')
def annotate_amplifications(
        amp_events_filename,
        gtf_filename,
        amp_events_genes_filename,
    ):

    amp_events = pd.read_csv(amp_events_filename)

    genes = scgenome.tl.read_ensemble_genes_gtf(gtf_filename)

    hlamp_segments = amp_events[[
        'patient_id',
        'hlamp_id',
        'target_bin',
        'chr',
        'start',
        'end',
    ]].drop_duplicates()

    hlamp_segments_pr = pr.from_dict({
        'Chromosome': hlamp_segments['chr'],
        'Start': hlamp_segments['start'],
        'End': hlamp_segments['end'],
        'patient_id': hlamp_segments['patient_id'],
        'hlamp_id': hlamp_segments['hlamp_id'],
        'target_bin': hlamp_segments['target_bin'],
    })

    intersect_1 = hlamp_segments_pr.intersect(genes)
    intersect_2 = genes.intersect(hlamp_segments_pr)

    overlapping_genes = pd.merge(
        scgenome.tools.ranges.pyranges_to_dataframe(intersect_1),
        scgenome.tools.ranges.pyranges_to_dataframe(intersect_2),
        on=['chr', 'start', 'end']).drop_duplicates()
    
    overlapping_genes.to_csv(amp_events_genes_filename, index=False)


if __name__ == "__main__":
    logging.basicConfig(stream=sys.stderr, level=logging.INFO)
    annotate_amplifications()

