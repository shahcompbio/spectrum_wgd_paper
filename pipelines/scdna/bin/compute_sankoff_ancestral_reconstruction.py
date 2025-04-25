import click
import pandas as pd
import numpy as np
import scgenome
import pickle
import tqdm
from Bio import Phylo

import anndata as ad

import spectrumanalysis.dataload
import spectrumanalysis.phylocn
import spectrumanalysis.cnevents

def compute_overlaps(adata, events, overlap_threshold = 0.9):
    all_bounds = get_bounds(adata)

    overlap_rows = []
    for chrom, edf in events.groupby('chrom'):
        ch = chrom[3:]
        if ch == '0':
            assert (edf.type == 'wgd').all()
            continue

        p_bounds, q_bounds, chr_bounds = all_bounds[ch]
        plen = p_bounds[1] - p_bounds[0]
        qlen = q_bounds[1] - q_bounds[0]
        ch = ch.replace('23', 'X')

        for _, r in edf.iterrows():
            p_overlap = min(p_bounds[1], r.end) - max(p_bounds[0], r.start)
            p_overlap_frac = p_overlap / max(plen, 1)
            q_overlap = min(q_bounds[1], r.end) - max(q_bounds[0], r.start)
            q_overlap_frac = q_overlap / qlen
            chr_overlap = p_overlap + q_overlap
            chr_overlap_frac = chr_overlap / (plen + qlen)

            event_sign = '+' if r.type == 'gain' else '-'

            if chr_overlap_frac >= overlap_threshold and chr_overlap_frac >= q_overlap_frac:
                # whole-chromosome event
                overlap_rows.append([r.sample_id, 'chr' + ch, chr_overlap_frac, r.type, ch + ' ' + event_sign, 
                                     r.allele, r.region, r.centromere, r.telomere, r.timing_wgd, r.cell_count])

            elif p_overlap_frac >= overlap_threshold:
                overlap_rows.append([r.sample_id, 'chr' + ch + 'p', p_overlap_frac, r.type, ch + 'p ' + event_sign, 
                                     r.allele, r.region, r.centromere, r.telomere, r.timing_wgd, r.cell_count])
            elif q_overlap_frac >= overlap_threshold:
                overlap_rows.append([r.sample_id, 'chr' + ch + 'q', q_overlap_frac, r.type, ch + 'q ' + event_sign, 
                                     r.allele, r.region, r.centromere, r.telomere, r.timing_wgd, r.cell_count])

    overlaps = pd.DataFrame(overlap_rows, columns = ['branch', 'name', 'FractionOverlaps', 'event', 'final_name', 'allele',
                                                     'region', 'centromere', 'telomere', 'timing_wgd', 'cell_count'])
    return overlaps


def get_bounds(adata):        
    all_bounds = {}
    for ch, df in adata.var.groupby('chr'):
        if ch == 'Y':
            continue
            
        chr_start = df.start.min()
        chr_end = df.end.max()
        
        if 'p' not in df.arm.values:
            assert ch in ['13', '14', '15', '21', '22'], ch
            cent_start = chr_start
        else:
            cent_start = df[df.arm == 'p'].end.max()
        cent_end = df[df.arm == 'q'].start.min()
                
        p_bounds = chr_start, cent_start
        q_bounds = cent_end, chr_end
        
        assert p_bounds[0] <= p_bounds[1], ch
        assert q_bounds[0] < q_bounds[1], ch
        assert p_bounds[1] <= q_bounds[0], ch
            
        all_bounds[ch] = p_bounds, q_bounds, (chr_start, chr_end)
    all_bounds['23'] = all_bounds['X']
    return all_bounds

@click.command()
@click.argument('medicc_cn_filename')
@click.argument('tree_filename')
@click.argument('events_filename')
@click.argument('cell_table_filename')
@click.argument('patient_id')
@click.argument('output_adata_filename')
@click.argument('output_tree_pickle_filename')
@click.argument('output_events_filename')
@click.argument('output_overlaps_filename')
def compute_medicc_ancestral_reconstruction(
        medicc_cn_filename,
        tree_filename,
        events_filename,
        cell_table_filename,
        patient_id,
        output_adata_filename,
        output_tree_pickle_filename,
        output_events_filename,
        output_overlaps_filename
    ):

    tree, adata, medicc_events = spectrumanalysis.dataload.load_medicc_as(
        medicc_cn_filename,
        tree_filename,
        events_filename)

    cell_info = pd.read_csv(cell_table_filename)

    # Get wgd clades from medicc events
    wgd_nodes = set(medicc_events.query('type == "wgd"')['sample_id'].values)

    # Patient specific fixes
    if patient_id == 'SPECTRUM-OV-002':
        wgd_nodes = wgd_nodes.union(['internal30'])
    elif patient_id == 'SPECTRUM-OV-003':
        wgd_nodes = wgd_nodes.union(['internal44'])
    elif patient_id == 'SPECTRUM-OV-014':
        wgd_nodes = wgd_nodes.union(['internal66'])
    elif patient_id == 'SPECTRUM-OV-024':
        wgd_nodes = wgd_nodes.union(['internal10'])
    elif patient_id == 'SPECTRUM-OV-025':
        wgd_nodes = wgd_nodes.union(['internal312', 'internal240']).difference(['internal242'])
    elif patient_id == 'SPECTRUM-OV-036':
        wgd_nodes = wgd_nodes.union(['internal482'])
    elif patient_id == 'SPECTRUM-OV-044':
        wgd_nodes = wgd_nodes.union(['internal657'])
    elif patient_id == 'SPECTRUM-OV-045':
        wgd_nodes = wgd_nodes.union(['internal545', 'internal593', 'internal881']).difference(['internal729'])
    elif patient_id == 'SPECTRUM-OV-051':
        wgd_nodes = wgd_nodes.union(['internal1'])
    elif patient_id == 'SPECTRUM-OV-052':
        wgd_nodes = wgd_nodes.union(['internal94'])
    elif patient_id == 'SPECTRUM-OV-071':
        wgd_nodes = wgd_nodes.union(['internal72'])
    elif patient_id == 'SPECTRUM-OV-083':
        wgd_nodes = wgd_nodes.union(['internal20'])

    # Add wgd status to branches
    for clade in tree.find_clades():
        if clade.name in wgd_nodes:
            clade.wgd = True
        else:
            clade.wgd = False

    # Cell filtering
    #
    cell_info = cell_info[cell_info['include_cell']]
    cell_info = cell_info[cell_info['multipolar'] == False]

    # Additional filtering of cell for which n_wgd doesnt match between medicc2 and basic calling
    for clade in tree.find_clades():
        clade.n_wgd = 0
    for clade in tree.find_clades():
        if clade.wgd:
            for subclade in clade.find_clades():
                subclade.n_wgd += 1
    medicc_n_wgd = []
    for clade in tree.get_terminals():
        medicc_n_wgd.append({'brief_cell_id': clade.name, 'n_wgd': clade.n_wgd})
    medicc_n_wgd = pd.DataFrame(medicc_n_wgd)
    cell_info = cell_info.merge(medicc_n_wgd, how='right', on='brief_cell_id', suffixes=('', '_medicc2'))
    print("n_cells:",  len(cell_info))
    cell_info = cell_info[cell_info['n_wgd_medicc2'] == cell_info['n_wgd']]
    print("n_cells with WGD agreement between MEDICC2 and heuristic:", len(cell_info))

    # Prune leaves from tree
    n_leaves = tree.count_terminals()
    tree = scgenome.tl.prune_leaves(tree, lambda a: (a.name not in cell_info['brief_cell_id'].values))
    n_removed = n_leaves - tree.count_terminals()
    print(f'removed {n_removed} of {n_leaves} leaves')
    assert (n_removed / n_leaves) <= 0.2, f'removed {100*(n_removed / n_leaves):.2f}% >20% of cells'

    # Merge branches
    def f_merge(parent, child):
        return {
            'name': child.name,
            'branch_length': parent.branch_length + child.branch_length,
            'wgd': parent.wgd or child.wgd,
            'n_wgd': max(parent.n_wgd, child.n_wgd),
        }
    tree = scgenome.tl.aggregate_tree_branches(tree, f_merge)

    # Redo is_wgd calls in the anndata (used for event calling), and add n_wgd
    adata.obs['n_wgd'] = -1
    adata.obs['cell_count'] = -1
    adata.obs.loc['diploid', 'n_wgd'] = 0
    adata.obs.loc['diploid', 'cell_count'] = 0
    for clade in tree.find_clades():
        adata.obs.loc[clade.name, 'is_wgd'] = clade.wgd
        adata.obs.loc[clade.name, 'n_wgd'] = clade.n_wgd
        adata.obs.loc[clade.name, 'cell_count'] = clade.count_terminals()

    # Also redo is_root
    adata.obs['is_root'] = False
    adata.obs.loc[tree.clade.name, 'is_root'] = True

    # Remove adata entries
    clade_names = [a.name for a in tree.find_clades()] + ['diploid']
    adata = adata[clade_names].copy()

    # Medicc events no longer valid
    del medicc_events

    adata = scgenome.tl.rebin_regular(
        adata,
        bin_size=500000,
        outer_join=False,
        agg_X=np.nanmedian,
        agg_layers={'cn_a': np.nanmedian, 'cn_b': np.nanmedian})

    # Save out tree before ancestral reconstruction to avoid including
    # additional extraneous state per clade
    with open(output_tree_pickle_filename, 'wb') as f:
        pickle.dump(tree, f)

    # Ancestral Reconstruction
    #

    n_states = int(max(adata.layers['cn_a'].max(), adata.layers['cn_b'].max()) + 1)
    n_bins = adata.layers['cn_a'].shape[1] + adata.layers['cn_b'].shape[1]

    spectrumanalysis.phylocn.add_states_to_tree(tree, adata, ['cn_a', 'cn_b'], n_bins, n_states)

    linear_transition = spectrumanalysis.phylocn.generate_linear_transition(n_states)
    linear_wgd_transition = spectrumanalysis.phylocn.generate_linear_wgd_transition(n_states)

    spectrumanalysis.phylocn.calculate_score_recursive_tree(tree, linear_transition, linear_wgd_transition, n_states, n_bins)

    diploid_state = np.repeat(1, (n_bins,))
    spectrumanalysis.phylocn.backtrack_state_recursive_tree(tree, ancestral_state=diploid_state)

    adata.layers['cn_a_2'] = np.zeros(adata.shape, dtype=int)
    adata.layers['cn_b_2'] = np.zeros(adata.shape, dtype=int)

    adata.layers['cn_a_2'][adata.obs.index.get_loc('diploid'), :] = 1
    adata.layers['cn_b_2'][adata.obs.index.get_loc('diploid'), :] = 1

    for clade in tree.find_clades():
        adata.layers['cn_a_2'][adata.obs.index.get_loc(clade.name)] = clade.state[:n_bins//2]
        adata.layers['cn_b_2'][adata.obs.index.get_loc(clade.name)] = clade.state[n_bins//2:]

    adata.layers['cn_a_2_change'] = np.zeros(adata.shape, dtype=int)
    adata.layers['cn_b_2_change'] = np.zeros(adata.shape, dtype=int)

    spectrumanalysis.phylocn.compute_cn_change(tree, adata, 'cn_a_2', 'cn_a_2_change')
    spectrumanalysis.phylocn.compute_cn_change(tree, adata, 'cn_b_2', 'cn_b_2_change')

    adata.layers['cn_a_2_change_pre'] = np.zeros(adata.shape, dtype=int)
    adata.layers['cn_b_2_change_pre'] = np.zeros(adata.shape, dtype=int)

    spectrumanalysis.phylocn.compute_wgd_cn_change(tree, adata, 'cn_a_2', 'pre', 'cn_a_2_change_pre')
    spectrumanalysis.phylocn.compute_wgd_cn_change(tree, adata, 'cn_b_2', 'pre', 'cn_b_2_change_pre')

    adata.layers['cn_a_2_change_post'] = np.zeros(adata.shape, dtype=int)
    adata.layers['cn_b_2_change_post'] = np.zeros(adata.shape, dtype=int)

    spectrumanalysis.phylocn.compute_wgd_cn_change(tree, adata, 'cn_a_2', 'post', 'cn_a_2_change_post')
    spectrumanalysis.phylocn.compute_wgd_cn_change(tree, adata, 'cn_b_2', 'post', 'cn_b_2_change_post')

    # Event calling
    #

    # Add arm, telomere and centromere annotations
    spectrumanalysis.cnevents.annotate_bins(adata)

    # Sorted order required for several downstream steps
    adata = adata[:, adata.var.sort_values(['chr', 'start']).index].copy()

    # Columns from adata.var
    var_columns = [
        'chr', 'start', 'end', 'arm',
        'p_telomere', 'q_telomere',
        'p_centromere', 'q_centromere',
    ]

    # Columns from adata.obs
    obs_columns = [
        'is_root', 'is_internal', 'is_cell',
        'is_wgd', 'n_wgd', 'cell_count',
    ]

    events = []
    for obs_idx, obs_id in tqdm.tqdm(list(enumerate(adata.obs.index))):
        if obs_id == 'diploid':
            continue

        if adata.obs.loc[obs_id, 'is_wgd']:

            # explicitly encode WGD event
            event = {
                'start':1500001, 'end':246500000, 'kind':'wgd',  'region':'genome', 'centromere':True, 'telomere':True,
                'chr': '0', 'allele': 'both', 'obs_id': obs_id, 'timing_wgd': None}
            for col in obs_columns:
                event[col] = adata.obs.loc[obs_id, col]
            events.append(event)

        for allele in ['a', 'b']:
            if adata.obs.loc[obs_id, 'is_wgd']:
                for wgd_timing in ['pre', 'post']:
                    cell_cn_change = adata.var[var_columns].copy()
                    cell_cn_change['cn_change'] = adata.layers[f'cn_{allele}_2_change_{wgd_timing}'][obs_idx, :]

                    for event in spectrumanalysis.cnevents.classify_segments(cell_cn_change):
                        event['allele'] = allele
                        event['obs_id'] = obs_id
                        event['timing_wgd'] = wgd_timing
                        for col in obs_columns:
                            event[col] = adata.obs.loc[obs_id, col]
                        events.append(event)

            else:
                cell_cn_change = adata.var[var_columns].copy()
                cell_cn_change['cn_change'] = adata.layers[f'cn_{allele}_2_change'][obs_idx, :]

                for event in spectrumanalysis.cnevents.classify_segments(cell_cn_change):
                    event['allele'] = allele
                    event['obs_id'] = obs_id
                    event['timing_wgd'] = None
                    for col in obs_columns:
                        event[col] = adata.obs.loc[obs_id, col]
                    events.append(event)

    events = pd.DataFrame(events)

    # Add patient id to event table
    events['patient_id'] = patient_id

    events.to_csv(output_events_filename, index=False, sep='\t')

    # translate to MEDICC2 events columns
    medicc2_events = events.copy()
    medicc2_events['sample_id'] = medicc2_events['obs_id']
    medicc2_events['chrom'] = 'chr' + medicc2_events['chr'].str.replace('X', '23')
    medicc2_events['type'] = medicc2_events['kind']

    # infer overlaps as in MEDICC2
    overlaps = compute_overlaps(adata, medicc2_events)

    # Add patient id
    overlaps['patient_id'] = patient_id

    overlaps.to_csv(output_overlaps_filename, index=False, sep='\t')

    adata.write(output_adata_filename)


if __name__ == "__main__":
    compute_medicc_ancestral_reconstruction()
