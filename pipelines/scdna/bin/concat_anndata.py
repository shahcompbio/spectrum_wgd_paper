#!/usr/bin/env python

import logging
import sys
import pandas as pd
import anndata as ad
import click

import scgenome


@click.command()
@click.argument('output_filename')
@click.option('--input_filename', required=True, multiple=True)
def concatenate_anndata(
        output_filename,
        input_filename,
    ):

    adatas = []
    for filename in input_filename:
        adata = ad.read(filename)
        adatas.append(adata)
    adata = scgenome.tl.ad_concat_cells(adatas)

    # Resolve issue with duplicate controls
    adata.obs_names_make_unique()

    adata.write(output_filename)


if __name__ == "__main__":
    logging.basicConfig(stream=sys.stderr, level=logging.INFO)
    concatenate_anndata()

