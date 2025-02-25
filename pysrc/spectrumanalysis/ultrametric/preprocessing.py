import Bio.Phylo
import itertools
import numpy as np
import pandas as pd
import xarray as xr

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


def split_wgd_branches(tree):
    """ Split branches that are marked as WGD events into two branches, one
    before the WGD and one after.

    Parameters
    ----------
    tree : Bio.Phylo.BaseTree.Tree
        Tree to split

    Returns
    -------
    Bio.Phylo.BaseTree.Tree
        Tree with WGD branches split
    """
    for clade in list(tree.find_clades()):
        if clade.is_wgd:
            post_wgd_clade = Bio.Phylo.BaseTree.Clade(
                branch_length=1.,
                name=clade.name + '_post_wgd',
                clades=clade.clades,
            )
            post_wgd_clade.is_wgd = False
            post_wgd_clade.cluster_id = clade.cluster_id
            clade.clades = [post_wgd_clade]
            clade.name = clade.name + '_pre_wgd'
            clade.cluster_id = None
    return tree


def annotate_wgd_timing(tree):
    """ Annotate each clade with the timing of the WGD event.

    Parameters
    ----------
    tree : Bio.Phylo.BaseTree.Tree
        Tree to annotate

    Returns
    -------
    Bio.Phylo.BaseTree.Tree
        Tree with WGD timing annotated
    """
    for clade in tree.find_clades():
        clade.wgd_timing = 'pre'
    for clade in tree.find_clades():
        if clade.is_wgd:
            for descendent in clade.find_clades():
                if descendent != clade:
                    descendent.wgd_timing = 'post'
    return tree


def preprocess_ultrametric_tree(tree):
    """ Preprocess an tree for the ultrametric model.

    Parameters
    ----------
    tree : Bio.Phylo.BaseTree.Tree
        Tree to preprocess

    Returns
    -------
    Bio.Phylo.BaseTree.Tree
        Tree with WGD timing annotated
    """

    # Set cluster id for leaves
    for clade in tree.find_clades():
        if clade.is_terminal():
            clade.cluster_id = clade.name.split('_')[1]
        else:
            clade.cluster_id = None

    tree = split_wgd_branches(tree)

    tree = annotate_wgd_timing(tree)

    return tree


def interleave_lists(lists):
    """ Generate all possible ways to interleave elements from a list of lists
    while preserving the order within each individual list.
    """
    
    if len(lists) == 1:
        return lists

    interleaved = []
    for idx in range(len(lists)):
        if len(lists[idx]) == 1:
            sublists = interleave_lists(lists[:idx] + lists[idx+1:])

        else:
            sublists = interleave_lists(lists[:idx] + [lists[idx][1:]] + lists[idx+1:])
            
        for l in sublists:
            interleaved.append([lists[idx][0]] + l)

    return interleaved


def tree_orderings(clade):
    """ Generate a list of all possible branch orderings.
    """

    clades = list(filter(lambda a: len(a.clades) > 0, clade.clades))
    if len(clades) == 0:
        return [[clade.name]]
    else:
        orderings = [tree_orderings(a) for a in clades]
        combined_orderings = []
        for x in itertools.product(*orderings):
            for l in interleave_lists(list(x)):
                combined_orderings.append([clade.name] + l)
        return combined_orderings


def generate_tree_orderings_table(tree):
    """ Generate a table of all possible orderings of branching events.

    Parameters
    ----------
    tree : Bio.Phylo.BaseTree.Tree
        Tree to generate orderings for

    Returns
    -------
    pandas.DataFrame
        Table of all possible orderings
    """
    internal_branches = [a.name for a in tree.get_nonterminals()]

    orderings = []
    for ordering in tree_orderings(tree.clade):
        ordering = [ordering.index(a) for a in internal_branches]
        orderings.append(ordering)
    orderings = np.array(orderings)

    orderings = pd.DataFrame(orderings, columns=internal_branches)
    orderings['leaves'] = len(internal_branches)
    
    return orderings


def generate_branchlength_matrix(tree, orderings):
    """ Generate a matrix mapping branches to branching events at the beginning, end, and during the branch.

    Parameters
    ----------
    tree : Bio.Phylo.BaseTree.Tree
        Tree to generate matrix for
    orderings : pandas.DataFrame
        Table of all possible orderings of branching events for the tree

    Returns
    -------
    xarray.DataArray
        Matrix mapping branches to branching events for each ordering
    """


    branches = [a.name for a in tree.find_clades()]
    internal_branches = [a.name for a in tree.get_nonterminals()]

    if len(branches) == 1:
        branchlength_matrix = xr.DataArray(
            1,
            coords=[orderings.index, branches, [0]],
            dims=['orderings', 'branches', 'order'])

    else:
        branchlength_matrix = xr.DataArray(
            0,
            coords=[orderings.index, branches, range(len(internal_branches)+1)],
            dims=['orderings', 'branches', 'order'])

        for idx, ordering in orderings.iterrows():
            child_order = ordering[tree.clade.name]
            branchlength_matrix.loc[idx, tree.clade.name, child_order] = 1

            for clade in tree.get_nonterminals():
                parent_order = ordering[clade.name]
                for child in clade.clades:
                    if child.is_terminal():
                        child_order = ordering['leaves']
                    else:
                        child_order = ordering[child.name]
                    assert parent_order < child_order
                    for i in range(parent_order+1, child_order+1):
                        branchlength_matrix.loc[idx, child.name, i] = 1

    return branchlength_matrix


def is_c_to_t_in_cpg_context(ref_base, alt_base, trinucleotide_context):
    """
    This function checks if a single nucleotide variant (SNV) is a C to T mutation
    in a CpG context or its reverse complement G to A in a CpG context.
    
    Parameters:
    ref_base (str): The reference nucleotide
    alt_base (str): The alternate nucleotide
    trinucleotide_context (str): The trinucleotide context of the SNV (string of 3 nucleotides)
    
    Returns:
    bool: True if the mutation is a C to T mutation in a CpG context or a G to A mutation
          in a CpG context on the reverse strand, False otherwise.
    """
    
    # Check if the mutation is C to T in a CpG context on the forward strand
    if ref_base == 'C' and alt_base == 'T':
        if len(trinucleotide_context) == 3 and trinucleotide_context[1] == 'C' and trinucleotide_context[2] == 'G':
            return True

    # Check if the mutation is G to A in a CpG context on the reverse strand
    if ref_base == 'G' and alt_base == 'A':
        if len(trinucleotide_context) == 3 and trinucleotide_context[0] == 'C' and trinucleotide_context[1] == 'G':
            return True
    
    return False


def generate_snv_counts(tree_assignments, tree):
    """ Generate a table of SNV counts for each branch segment.

    Parameters
    ----------
    tree_assignments : pandas.DataFrame
        Table of SNV assignments to branches
    tree : Bio.Phylo.BaseTree.Tree
        Tree to generate SNV counts for

    Returns
    -------
    pandas.Series
        Table of SNV counts for each branch segment
    """
    branches = [a.name for a in tree.find_clades()]

    age_mut_types = [
        'A[C>T]G',
        'C[C>T]G',
        'T[C>T]G',
        'G[C>T]G',
    ]

    # HACK
    if 'MutationType' in tree_assignments:
        tree_assignments['is_age_mut'] = tree_assignments['MutationType'].isin(age_mut_types)
    elif 'tri_nucleotide_context' in tree_assignments:
        tree_assignments['is_age_mut'] = np.logical_or(np.logical_and(tree_assignments.ref == 'C', np.logical_and(tree_assignments.alt == 'T', tree_assignments.tri_nucleotide_context.str.slice(2) == 'G')),
                               np.logical_and(tree_assignments.ref == 'G', np.logical_and(tree_assignments.alt == 'A', tree_assignments.tri_nucleotide_context.str.slice(0,1) == 'C')))
        tree_assignments['is_age_mut2'] = [is_c_to_t_in_cpg_context(r.ref, r.alt, r.tri_nucleotide_context) for _, r in tree_assignments.iterrows()]
        assert tree_assignments.is_age_mut2.equals(tree_assignments.is_age_mut)

    tree_assignments['branch_segment'] = tree_assignments['clade']
    tree_assignments.loc[(tree_assignments['wgd_timing'] == 'prewgd') & (tree_assignments['is_wgd'] == True), 'branch_segment'] += '_pre_wgd'
    tree_assignments.loc[(tree_assignments['wgd_timing'] == 'postwgd') & (tree_assignments['is_wgd'] == True), 'branch_segment'] += '_post_wgd'

    tree_assignments = tree_assignments.drop_duplicates(['snv_id', 'branch_segment'])

    snv_counts = tree_assignments.query('clade != "none"').groupby(['branch_segment']).size().reindex(branches, fill_value=0)
    snv_counts_age = tree_assignments.query('is_age_mut').query('clade != "none"').groupby(['branch_segment']).size().reindex(branches, fill_value=0)

    data = pd.DataFrame({
        'snv_count': snv_counts,
        'snv_count_age': snv_counts_age,
    })

    data = data.reindex(branches, fill_value=0)

    return data


def logistic_curve(x, k=0.62355594, x0=4.1070332):
    return 0.92 / (1 + np.exp(-1 * k * (x - x0)))


def calc_sensitivity(d):
    return logistic_curve(d)


def count_wgd(clade, n_wgd):
    if clade.is_wgd:
        clade.n_wgd = n_wgd + 1
    else:
        clade.n_wgd = n_wgd
    for child in clade.clades:
        count_wgd(child, clade.n_wgd)


def generate_snv_genotype_subsets_table(adata_clusters, tree):
    """ Generate a table of SNV information per allele and WGD timing.

    Parameters
    ----------
    adata_clusters : anndata.AnnData
        cluster copy number and per cluster annotation
    tree : Bio.Phylo.BaseTree.Tree
        Tree representing cluster phylogeny

    Returns
    -------
    pandas.DataFrame
        Table of SNV information per allele and WGD timing
    """

    # Subset to homogenous copy number in compatible states
    adata_clusters = adata_clusters[:, (adata_clusters.var['is_homogenous_cn']) & (adata_clusters.var['snv_type'] != 'incompatible')]

    # Count number of bins of each snv_type
    # TODO: fix snvtype cn type
    bin_count = adata_clusters.var.groupby('snv_type').size().rename('bin_count').reset_index()

    count_wgd(tree.clade, 0)

    snv_types = ['1:0', '2:0', '1:1', '2:1', '2:2']

    # # Copy number information of branches for each snv type, allele, and n_wgd state on the branch
    # #  - cn_branch: number of copies of the allele on a branch with the given n_wgd and snv type
    # snv_type_branch_cn = []
    # for snv_type in snv_types:
    #     maj, min = (int(a) for a in snv_type.split(':'))
    #     for n_wgd in (0, 1):
    #         for allele, cn in zip(('maj', 'min'), (maj, min)):
    #             if cn == 0:
    #                 cn_branch = 0
    #             elif n_wgd == 0:
    #                 cn_branch = 1
    #             elif n_wgd == 1:
    #                 cn_branch = cn
    #             snv_type_branch_cn.append({
    #                 'snv_type': snv_type,
    #                 'n_wgd_branch': n_wgd,
    #                 'allele': allele,
    #                 'cn_branch': cn_branch,
    #             })
    # snv_type_branch_cn = pd.DataFrame(snv_type_branch_cn)

    # # Leaf snv multiplicity information for each snv type, allele, wgd timing of the branch, and n_wgd state at the leaf
    # #  - multiplicity: number of copies of the snv if the snv is present on a leaf with the given n_wgd state for a given snv type
    # snv_type_leaf_multiplicity = []
    # for snv_type in snv_types:
    #     maj, min = (int(a) for a in snv_type.split(':'))
    #     for n_wgd in (0, 1):
    #         for wgd_timing in ('pre', 'post'):
    #             # Cannot have a leaf that is n_wgd=0 below a branch that is post-wgd
    #             if (n_wgd == 0) and (wgd_timing == 'post'):
    #                 continue
    #             for allele, cn in zip(('maj', 'min'), (maj, min)):
    #                 if cn == 0:
    #                     multiplicity = 0
    #                 elif (wgd_timing == 'pre') and (n_wgd == 1):
    #                     multiplicity = cn
    #                 else:
    #                     multiplicity = 1
    #                 snv_type_leaf_multiplicity.append({
    #                     'snv_type': snv_type,
    #                     'n_wgd_leaf': n_wgd,
    #                     'wgd_timing': wgd_timing,
    #                     'allele': allele,
    #                     'multiplicity': multiplicity,
    #                 })
    # snv_type_leaf_multiplicity = pd.DataFrame(snv_type_leaf_multiplicity)
    # snv_type_leaf_multiplicity

    # genome_length_info = []
    # for clade in tree.find_clades():
    #     for snv_type in snv_types:
    #         maj, min = (int(a) for a in snv_type.split(':'))
    #         for allele, cn in zip(('maj', 'min'), (maj, min)):

    # Variant depth and sensitivity per clade, snv type, and allele
    clade_info = []
    for clade in tree.find_clades():
        for snv_type in snv_types:
            maj, min = (int(a) for a in snv_type.split(':'))
            for allele, cn in zip(('maj', 'min'), (maj, min)):

                # Calculate the total variant depth for an snv of
                # the given type, allele and branch
                variant_depth = 0.
                for leaf in clade.get_terminals():
                    leaf_n_wgd = adata_clusters.obs.loc[leaf.cluster_id, 'n_wgd']
                    total_haploid_depth = adata_clusters.obs.loc[leaf.cluster_id, 'haploid_depth']
                    wgd_timing = clade.wgd_timing

                    # snvs occuring pre-wgd and observed in an n_wgd=1
                    # branch will have multiplicity same as copy number
                    if cn == 0:
                        multiplicity = 0
                    elif (wgd_timing == 'pre') and (leaf_n_wgd == 1):
                        multiplicity = cn
                    else:
                        multiplicity = 1

                    variant_depth += multiplicity * total_haploid_depth

                # Calculate the copy number on the branch historically
                clade_wgd_timing = clade.wgd_timing
                if cn == 0:
                    cn_branch = 0
                elif clade_wgd_timing == 'pre':
                    cn_branch = 1
                else:
                    assert clade_wgd_timing == 'post'
                    cn_branch = cn

                clade_info.append({
                    'branch_segment': clade.name,
                    'snv_type': snv_type,
                    'allele': allele,
                    'variant_depth': variant_depth,
                    'cn_branch': cn_branch,
                })

    clade_info = pd.DataFrame(clade_info)
    clade_info['sensitivity'] = calc_sensitivity(clade_info['variant_depth'])

    clade_info = clade_info.merge(bin_count)

    clade_info['genome_length'] = clade_info['cn_branch'] * clade_info['bin_count'] * 5e5 / 1e9
    clade_info['opportunity'] = clade_info['sensitivity'] * clade_info['genome_length']

    return clade_info


