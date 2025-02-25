import panel as pn
import matplotlib.pyplot as plt
import matplotlib.colors
import seaborn as sns
import pandas as pd
import numpy as np
import collections
import anndata
from collections.abc import Mapping
import matplotlib.patches as patches
import matplotlib.cm as cm
import matplotlib.collections as mc
from scgenome.plotting.cn import genome_axis_plot, setup_genome_xaxis_ticks, setup_genome_xaxis_lims
from scgenome.plotting.cn_colors import color_reference


import scgenome


default_heatmap_tab_plot_params = [
    (
        'state',
        {
            'layer_name': 'state',
        },
    ),
    (
        'copy',
        {
            'layer_name': 'copy',
            'raw': True,
            'vmin': 0.,
            'vmax': 10.,
        },
    ),
    (
        'state_abs_diff',
        {
            'layer_name': 'state_abs_diff',
            'raw': True,
            'vmin': -0.5,
            'vmax': 0.5,
        },
    ),
    (
        'copy_abs_diff',
        {
            'layer_name': 'copy_abs_diff',
            'raw': True,
            'vmin': -0.5,
            'vmax': 0.5,
        },
    ),
    (
        'copy_state_abs_diff',
        {
            'layer_name': 'copy_state_abs_diff',
            'raw': True,
            'vmin': -0.5,
            'vmax': 0.5,
        },
    ),
]


default_asheatmap_tab_plot_params = [
    (
        'state',
        {
            'layer_name': 'state',
        },
    ),
    (
        'Min',
        {
            'layer_name': 'Min',
        },
    ),
    (
        'Maj',
        {
            'layer_name': 'Maj',
        },
    ),
    (
        'copy',
        {
            'layer_name': 'copy',
            'raw': True,
            'vmin': 0.,
            'vmax': 10.,
        },
    ),
    (
        'BAF',
        {
            'layer_name': 'BAF',
            'raw': True,
            'vmin': 0.,
            'vmax': 1.,
        },
    ),
    (
        'state_abs_diff',
        {
            'layer_name': 'state_abs_diff',
            'raw': True,
            'vmin': -0.5,
            'vmax': 0.5,
        },
    ),
    (
        'copy_abs_diff',
        {
            'layer_name': 'copy_abs_diff',
            'raw': True,
            'vmin': -0.5,
            'vmax': 0.5,
        },
    ),
    (
        'copy_state_abs_diff',
        {
            'layer_name': 'copy_state_abs_diff',
            'raw': True,
            'vmin': -0.5,
            'vmax': 0.5,
        },
    ),
    (
        'BAF_abs_diff',
        {
            'layer_name': 'BAF_abs_diff',
            'raw': True,
            'vmin': -0.5,
            'vmax': 0.5,
        },
    ),
    (
        'BAF_ideal_abs_diff',
        {
            'layer_name': 'BAF_ideal_abs_diff',
            'raw': True,
            'vmin': -0.5,
            'vmax': 0.5,
        },
    ),
]


def plot_heatmap_tabs(
        adata,
        plot_params,
        cell_order_fields=('cluster_id', 'cell_order'),
        annotation_fields=('cluster_id', 'is_outlier', 'library_id'),
        var_annotation_fields=('cyto_band_giemsa_stain', 'gc'),
    ):
    """ Plot copy number for a chromosome in tabs with panel
    
    plot the current clustering showing raw and integer copy number
    also, plot the difference between the cluster copy and each cell copy, notice some structure
    """

    tabs = pn.Tabs()

    for name, params in plot_params:
        fig = plt.figure(figsize=(8, 8), dpi=144)
        scgenome.pl.plot_cell_cn_matrix_fig(
            adata,
            fig=fig,
            cell_order_fields=cell_order_fields,
            annotation_fields=annotation_fields,
            var_annotation_fields=var_annotation_fields,
            **params)
        pfig = pn.pane.Matplotlib(fig, dpi=144, tight=True)
        pfig.width = 800
        pfig.height = 800
        tabs.append((name, pfig))
        plt.close()
        
    return tabs


def get_feature_colors(features, palettes):
    """ Get colors for features

    Parameters
    ----------
    features : pandas.DataFrame
        categorical features to color
    palettes : dict
        dictionary of palettes to use for each feature

    Returns
    -------
    colors : pandas.DataFrame
        dataframe of colors for each feature
    attribute_to_color : dict
        dictionary of attribute to color mapping
    """
    colors = features[palettes.keys()].copy()
    attribute_to_color = dict()
    for col in palettes.keys():
        if colors[col].dtype.name == 'category':
            unique_attrs = colors[col].cat.categories
        else:
            unique_attrs = sorted(colors[col].astype(str).unique())
        if not isinstance(palettes[col], Mapping):
            cmap = sns.color_palette(palette=palettes[col], n_colors=len(unique_attrs))
            attribute_to_color[col] = dict(zip(unique_attrs, cmap))
        else:
            attribute_to_color[col] = {str(k): v for k, v in palettes[col].items()}
        colors[col] = colors[col].astype(str).map(attribute_to_color[col])
    return colors, attribute_to_color


def plot_feature_colors_legends(attribute_to_color):
    """ Plot feature colors legends

    Parameters
    ----------
    attribute_to_color : dict
        dictionary of attribute to color mapping, from `get_feature_colors`
    """
    for feature in attribute_to_color.keys():
        plt.figure(figsize=(4, 1))
        ax = plt.gca()
        for attribute, color in attribute_to_color[feature].items():
            ax.bar(0, 0, color=color, label=attribute, linewidth=0)
        ax.legend(loc='center', ncols=7, bbox_to_anchor=(0.5, 0.5), title=feature)
        ax.axis('off')


def remove_xticklabels(ax, labels):
    """ Remove a subset of the tick labels on the x axis, leaving ticks in place

    Parameters
    ----------
    ax : matplotlib.Axes
        axes to modify
    labels : list
        list of labels to remove
    """
    xticklabels = ax.get_xticklabels()
    for label in xticklabels:
        if label.get_text() in labels:
            label.set_text('')
    ax.set_xticklabels(xticklabels)


def generate_color_legend(color_dict, dpi=150, order=None, ax=None, **kwargs):
    """
    Generates a legend for a given dictionary of colors.

    Parameters
    ----------
    color_dict: dict
        A dictionary where keys are the labels and values are the corresponding colors.
    dpi : int
        Figure dpi
    

    Returns
    -------
    legend : matplotlib.legend.Legend
        A matplotlib legend object.
    """
    # Defaults
    kwargs.setdefault('loc', 'upper left')
    kwargs.setdefault('frameon', False)

    # Create a list to hold the legend elements
    legend_elements = []
    
    # Iterate over the dictionary to create a list of patches for the legend
    if order is None:
        order = color_dict.keys()
    for label in order:
        legend_elements.append(matplotlib.patches.Patch(color=color_dict[label], label=label))
    
    if ax is None:
        # Create the figure and axes objects to plot the legend
        fig, ax = plt.subplots(dpi=dpi)

        # Hide the axes
        ax.axis('off')
    
    else:
        fig = plt.gcf()

    # Add the legend to the plot
    legend = ax.legend(handles=legend_elements, **kwargs)
    ax.add_artist(legend)
    # legend._legend_box.align = "left"

    return {
        'fig': fig,
        'ax': ax,
        'legend': legend,
    }


def style_barplot(ax):
    ax.spines[['top', 'right']].set_visible(False)
    ax.tick_params(axis='x', which='major', rotation=0)
    ax.set_yticks(np.linspace(0, 1, 6))
    ax.spines['left'].set_bounds(ax.get_yticks().min(), ax.get_yticks().max())
    ax.yaxis.label.set_horizontalalignment('right')
    ax.yaxis.label.set_verticalalignment('center')
    ax.yaxis.label.set_rotation(0)


# `A-Hom` = "#56941E",
# `B-Hom` = "#471871",
# `A-Gained` = "#94C773",
# `B-Gained` = "#7B52AE",
# `Balanced` = "#d5d5d4",

allele_state_colors = collections.OrderedDict({
    'A-Hom': '#56941E',
    'A-Gained': '#94C773',
    'Balanced': '#d5d5d4',
    'B-Gained': '#7B52AE',
    'B-Hom': '#471871',
})

allele_state_cmap = matplotlib.colors.ListedColormap([
    '#56941E',
    '#94C773',
    '#d5d5d4',
    '#7B52AE',
    '#471871',
])


def add_allele_state_layer(adata):
    """ Add a layer representing allelic states for plotting.

    Parameters
    ----------
    adata : AnnData
        Annotated data object containing allelic state information.

    Returns
    -------
    AnnData
        Annotated data object with the 'allele_state' layer added.
    """
    allele_state = np.zeros(adata.shape)
    allele_state[adata.layers['B'] == 0] = 0
    allele_state[(adata.layers['A'] != 0) & (adata.layers['B'] != 0) & (adata.layers['A'] > adata.layers['B'])] = 1
    allele_state[(adata.layers['A'] != 0) & (adata.layers['B'] != 0) & (adata.layers['A'] == adata.layers['B'])] = 2
    allele_state[(adata.layers['A'] != 0) & (adata.layers['B'] != 0) & (adata.layers['B'] > adata.layers['A'])] = 3
    allele_state[adata.layers['A'] == 0] = 4

    adata.layers['allele_state'] = allele_state

    return adata


def plot_allele_cn_profile(adata, cell_id, ax=None, **kwargs):
    """ Plot BAF colored by allele specific copy number state

    Parameters
    ----------
    adata : anndata.AnnData
        copy number anndata
    cell_id : str
        cell from adata.obs.index to plot
    ax : matplotlib.Axes, optional
        axes on which to plot, by default None, use plt.gca()
    kwargs : dict
        additional arguments to plot_profile
    """

    if ax is None:
        ax = plt.gca()

    if 's' not in kwargs:
        kwargs['s'] = 2

    if 'alpha' not in kwargs:
        kwargs['alpha'] = 1.

    plot_data = scgenome.tl.get_obs_data(
        adata,
        cell_id,
        layer_names=['copy', 'BAF', 'state', 'A', 'B']
    )

    plot_data['ascn_state'] = 'Balanced'
    plot_data.loc[plot_data['A'] > plot_data['B'], 'ascn_state'] = 'A-Gained'
    plot_data.loc[plot_data['B'] > plot_data['A'], 'ascn_state'] = 'B-Gained'
    plot_data.loc[plot_data['B'] == 0, 'ascn_state'] = 'A-Hom'
    plot_data.loc[plot_data['A'] == 0, 'ascn_state'] = 'B-Hom'

    scgenome.pl.plot_profile(
        plot_data,
        y='BAF',
        hue='ascn_state',
        ax=ax,
        palette=allele_state_colors,
        hue_order=allele_state_colors.keys(),
    )

    plt.ylabel('BAF')
    sns.despine(trim=True)

    scgenome.plotting.cn.setup_genome_xaxis_ticks(
        ax, chromosome_names=dict(zip(
            [str(a) for a in range(1, 23)] + ['X'],
            ['1', '2', '3', '4', '5', '6', '7', '8', '9', '', '11', '', '13', '', '15', '', '', '18', '', '', '21', '', 'X'])))


def get_color_from_colormap(value, min_value, max_value, colormap_name):
    # Normalize the value to the range [0, 1]
    normalized_value = (value - min_value) / (max_value - min_value)
    
    # Get the colormap
    colormap = cm.get_cmap(colormap_name)
    
    # Get the color for the normalized value
    color = colormap(normalized_value)
    
    return color


def plot_cn_rect(
        data,
        obs_id=None,
        ax=None,
        y='state',
        hue='state',
        chromosome=None,
        cmap=None,
        vmin=None,
        vmax=None,
        color=None,
        offset=0,
        rect_kws=None,
        fill_gaps=True):
    """
    Plot copy number rectangles on a genome axis.

    Parameters
    ----------
    data : pandas.DataFrame or anndata.AnnData
        The data containing copy number information.
    obs_id : str
        The observation ID to extract data from an AnnData object.
    ax : matplotlib.axes.Axes, optional
        The axes to plot on. If not provided, the current axes will be used.
    y : str, optional
        The column name in the data to use as the y-coordinate of the rectangles. Default is 'state'.
    hue : str, optional
        The column name in the data to use for coloring the rectangles. Default is 'state'.
    chromosome : str, optional
        The chromosome to plot. If not provided, all chromosomes will be plotted.
    cmap : matplotlib.colors.Colormap, optional
        The colormap to use for coloring the rectangles. If not provided, a default colormap will be used based on the 'hue' column.
    vmin : float, optional
        The minimum value for the colormap. If not provided, the minimum value from the 'hue' column will be used.
    vmax : float, optional
        The maximum value for the colormap. If not provided, the maximum value from the 'hue' column will be used.
    color : color, optional
        Color value for all rectangles
    offset : float, optional, default 0
        y offset for rects
    rect_kws : dict, optional
        Additional keyword arguments for patches.Rectangle
    fill_gaps: bool, optional, default True
        fill gaps between segments

    Returns
    -------
    ax : matplotlib.axes.Axes
        The axes object containing the plotted copy number rectangles.
    """

    if ax is None:
        ax = plt.gca()

    if rect_kws is None:
        rect_kws = dict()

    rect_kws.setdefault('height', 0.7)
    rect_kws.setdefault('linewidth', 0.)

    # Check data is an adata
    if isinstance(data, anndata.AnnData):
        assert obs_id is not None

        layers = {y}
        if hue is not None:
            layers.add(hue)

        data = scgenome.tl.get_obs_data(
            data,
            obs_id,
            layer_names=list(layers),
        )

    if fill_gaps:
        data = data.sort_values(['chr', 'start'])

        for chrom_, df in data.groupby('chr'):
            # Bring end of each segment to start of next segment
            data.loc[df.index[:-1], 'end'] = data.loc[df.index[1:], 'start'].values

            # First segment starts at 0
            data.loc[df.index[0], 'start'] = 0

            # Last segment ends at chromosome length
            data.loc[df.index[-1], 'end'] = scgenome.refgenome.info.chromosome_info.set_index('chr').loc[chrom_, 'chromosome_length']

    if chromosome is not None:
        data = data[data['chr'] == chromosome]

    if hue is not None:
        if cmap is None:
            data['color'] = data[hue].map(color_reference)
    
        else:
            if vmin is None:
                vmin = data[hue].min()
            if vmax is None:
                vmax = data[hue].max()
            data['color'] = data[hue].apply(lambda a: get_color_from_colormap(a, vmin, vmax, cmap))

    elif color is not None:
        data['color'] = color

    def plot_rect(data, ax=None):
        rectangles = []
        for idx, row in data.iterrows():
            width = row['end'] - row['start']
            lower_left_x = row['start']
            lower_left_y = row[y] - (rect_kws['height'] / 2.) + offset

            # Create a rectangle patch
            rect = patches.Rectangle(
                (lower_left_x, lower_left_y),
                width,
                facecolor=row['color'],
                **rect_kws)
            rectangles.append(rect)

        # Create a patch collection with the rectangles and add it to the Axes
        pc = mc.PatchCollection(rectangles, match_original=True, zorder=2)
        ax.add_collection(pc)

    genome_axis_plot(
        data,
        plot_rect,
        ('start', 'end'),
        ax=ax,
    )

    ax.set_ylim((data[y].min() - 0.5, data[y].max() + 0.5))
    setup_genome_xaxis_ticks(ax, chromosome=chromosome)
    setup_genome_xaxis_lims(ax, chromosome=chromosome)

    ax.spines[['right', 'top']].set_visible(False)

    return ax


def pretty_cell_tcn(adata, cell_id, chromosome=None, fig=None, ax=None, rasterized=False):
    """
    Generate a pretty cell-specific total copy number profile plot.

    Parameters
    ----------
    adata : AnnData
        Annotated data object containing copy number information.
    cell_id : str
        Identifier of the cell for which to generate the plot.
    chromosome : str, optional
        Chromosome for which to generate the plot. If None, the plot will include all chromosomes.
    fig : matplotlib.figure.Figure, optional
        The figure to plot on. If not provided, a new figure will be created.
    ax : matplotlib.axes.Axes, optional
        The axes to plot on. If not provided, new axes will be created.
    rasterized : bool, optional
        Whether to rasterize the plot. Default is False.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The generated figure object.
    """

    if fig is None:
        fig = plt.figure(figsize=(6, 1.5), dpi=300)

    if ax is None:
        ax = plt.gca()

    g = scgenome.pl.plot_cn_profile(
        adata,
        cell_id,
        state_layer_name='state',
        value_layer_name='copy',
        chromosome=chromosome,
        ax=ax,
        squashy=True,
        linewidth=0,
        s=4,
        alpha=1.,
        hue_order=sorted(range(12)),
        rasterized=rasterized)
    sns.despine(trim=True)
    
    sns.move_legend(
        ax, 'upper left', prop={'size': 8}, markerscale=3, bbox_to_anchor=(1, 1),
        labelspacing=0.4, handletextpad=0, columnspacing=0.5,
        ncol=3, title='Total CN state', title_fontsize=10, frameon=False)

    ax.grid(ls=':', lw=0.5, zorder=-100, which='major', axis='y')
    ax.set_axisbelow(True)

    if chromosome is None:
        scgenome.plotting.cn.setup_genome_xaxis_ticks(
            ax, chromosome_names=dict(zip(
                [str(a) for a in range(1, 23)] + ['X'],
                ['1', '2', '3', '4', '5', '6', '7', '8', '9', '', '11', '', '13', '', '15', '', '', '18', '', '', '21', '', 'X'])))

    return fig


def pretty_cell_ascn(adata, cell_id, fig=None, axes=None, rasterized=False):
    """
    Plot a pretty cell ASCN (Allele-Specific Copy Number) profile.

    Parameters
    ----------
    adata : AnnData
        Annotated data object containing the copy number data.
    cell_id : str
        Identifier of the cell to plot.
    fig : matplotlib.figure.Figure, optional
        The figure to plot on. If not provided, a new figure will be created.
    axes : matplotlib.axes.Axes, optional
        The axes to plot on. If not provided, new axes will be created.
    rasterized : bool, optional
        Whether to rasterize the plot. Default is False.

    Returns
    -------
    matplotlib.figure.Figure
        The generated figure.

    """
    plot_data = scgenome.tl.get_obs_data(
        adata,
        cell_id,
        layer_names=['copy', 'BAF', 'state', 'A', 'B']
    )

    assert (fig is None) == (axes is None)
    if fig is None:
        fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(6, 3), dpi=300, sharex=True)

    ax = axes[0]
    g = scgenome.pl.plot_cn_profile(
        adata,
        cell_id,
        state_layer_name='state',
        value_layer_name='copy',
        ax=ax,
        squashy=True,
        linewidth=0,
        s=4,
        alpha=1.,
        hue_order=sorted(range(12)),
        rasterized=rasterized)
    sns.despine(trim=True)

    sns.move_legend(
        ax, 'upper left', prop={'size': 8}, markerscale=3, bbox_to_anchor=(1, 1),
        labelspacing=0.4, handletextpad=0, columnspacing=0.5,
        ncol=3, title='Total CN state', title_fontsize=10, frameon=False)

    ax.grid(ls=':', lw=0.5, zorder=-100, which='major', axis='y')
    ax.set_axisbelow(True)

    plot_data['ascn_state'] = 'Balanced'
    plot_data.loc[plot_data['A'] > plot_data['B'], 'ascn_state'] = 'A-Gained'
    plot_data.loc[plot_data['B'] > plot_data['A'], 'ascn_state'] = 'B-Gained'
    plot_data.loc[plot_data['B'] == 0, 'ascn_state'] = 'A-Hom'
    plot_data.loc[plot_data['A'] == 0, 'ascn_state'] = 'B-Hom'

    ax = axes[1]
    scgenome.pl.plot_profile(
        plot_data,
        y='BAF',
        hue='ascn_state',
        ax=ax,
        palette=allele_state_colors,
        s=4,
        alpha=1.,
        hue_order=allele_state_colors.keys(),
        rasterized=rasterized,
    )

    plt.ylabel('BAF')
    sns.despine(trim=True)

    scgenome.plotting.cn.setup_genome_xaxis_ticks(
        ax, chromosome_names=dict(zip(
            [str(a) for a in range(1, 23)] + ['X'],
            ['1', '2', '3', '4', '5', '6', '7', '8', '9', '', '11', '', '13', '', '15', '', '', '18', '', '', '21', '', 'X'])))

    sns.move_legend(
        ax, 'upper left', prop={'size': 8}, markerscale=3, bbox_to_anchor=(1, 1),
        labelspacing=0.4, handletextpad=0, columnspacing=0.5,
        ncol=1, title='AS CN state', title_fontsize=10, frameon=False)

    ax.grid(ls=':', lw=0.5, zorder=-100, which='major', axis='y')
    ax.set_axisbelow(True)

    return fig


def pretty_pseudobulk_tcn(adata, chromosome=None, fig=None, ax=None, rasterized=True):
    """
    Plot a pretty pseudobulk total copy number profile.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix containing the copy number information.
    chromosome : str, optional
        Chromosome for which to generate the plot. If None, the plot will include all chromosomes.
    fig : numpy.ndarray, optional
        The generated matplotlib figure object.
    ax : numpy.ndarray, optional
        The matplotlib axes object.
    rasterized : bool, optional
        Whether to rasterize the plot. Default is True.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The generated matplotlib figure object.

    Examples
    --------
    >>> fig = pretty_pseudobulk_tcn(adata)
    >>> plt.show()
    """

    agg_layers = {
        'copy': np.nanmean,
        'state': np.nanmedian,
    }

    adata.obs['pseudobulk'] = '1'
    adata_pseudobulk = scgenome.tl.aggregate_clusters(adata, agg_layers=agg_layers, cluster_col='pseudobulk')

    if fig is None:
        fig = plt.figure(figsize=(6, 1.5), dpi=300)

    if ax is None:
        ax = plt.gca()

    g = scgenome.pl.plot_cn_profile(
        adata_pseudobulk,
        '1',
        state_layer_name='state',
        value_layer_name='copy',
        chromosome=chromosome,
        ax=ax,
        squashy=True,
        linewidth=0,
        s=4,
        alpha=1.,
        hue_order=sorted(range(12)),
        rasterized=rasterized)
    # sns.despine(trim=True)

    sns.move_legend(
        ax, 'upper left', prop={'size': 8}, markerscale=3, bbox_to_anchor=(1, 1),
        labelspacing=0.4, handletextpad=0, columnspacing=0.5,
        ncol=3, title='Total CN state', title_fontsize=10, frameon=False)
    
    ax.grid(ls=':', lw=0.5, zorder=-100, which='major', axis='y')
    ax.set_axisbelow(True)

    if chromosome is None:
        scgenome.plotting.cn.setup_genome_xaxis_ticks(
            ax, chromosome_names=dict(zip(
                [str(a) for a in range(1, 23)] + ['X'],
                ['1', '2', '3', '4', '5', '6', '7', '8', '9', '', '11', '', '13', '', '15', '', '', '18', '', '', '21', '', 'X'])))

    return fig


def pretty_pseudobulk_ascn(adata, chromosome=None, fig=None):
    """
    Plot a pretty pseudobulk allele specific copy number profile.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix containing the copy number information.
    chromosome : str, optional
        Chromosome for which to generate the plot. If None, the plot will include all chromosomes.
    fig : numpy.ndarray, optional
        The generated matplotlib figure object.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The generated matplotlib figure object.

    Examples
    --------
    >>> fig = pretty_pseudobulk_tcn(adata)
    >>> plt.show()
    """

    agg_layers = {
        'copy': np.nanmean,
        'state': np.nanmedian,
        'alleleA': np.nansum,
        'alleleB': np.nansum,
        'totalcounts': np.nansum,
        'Min': np.nanmedian,
        'Maj': np.nanmedian,
        'A': np.nanmedian,
        'B': np.nanmedian,
    }

    adata.obs['pseudobulk'] = '1'
    adata_pseudobulk = scgenome.tl.aggregate_clusters(adata, agg_layers=agg_layers, cluster_col='pseudobulk')
    adata_pseudobulk.layers['Min'] = adata_pseudobulk.layers['Min'].round()
    adata_pseudobulk.layers['Maj'] = adata_pseudobulk.layers['Maj'].round()
    adata_pseudobulk.layers['BAF'] = adata_pseudobulk.layers['alleleB'] / adata_pseudobulk.layers['totalcounts']

    plot_data = scgenome.tl.get_obs_data(
        adata_pseudobulk,
        '1',
        layer_names=['copy', 'BAF', 'state', 'A', 'B', 'Min', 'Maj', 'BAF']
    )

    if fig is None:
        fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(6, 3), dpi=300, sharex=True)

    ax = axes[0]
    g = scgenome.pl.plot_cn_profile(
        adata_pseudobulk,
        '1',
        state_layer_name='state',
        value_layer_name='copy',
        chromosome=chromosome,
        ax=ax,
        squashy=True,
        linewidth=0,
        s=4,
        alpha=1.,
        hue_order=sorted(range(12)))
    # sns.despine(trim=True)

    ax.grid(ls=':', lw=0.5, zorder=-100, which='major', axis='y')
    ax.set_axisbelow(True)

    sns.move_legend(
        ax, 'upper left', prop={'size': 8}, markerscale=3, bbox_to_anchor=(1, 1),
        labelspacing=0.4, handletextpad=0, columnspacing=0.5,
        ncol=3, title='Total CN state', title_fontsize=10, frameon=False)

    plot_data['ascn_state'] = 'Balanced'
    plot_data.loc[plot_data['A'] > plot_data['B'], 'ascn_state'] = 'A-Gained'
    plot_data.loc[plot_data['B'] > plot_data['A'], 'ascn_state'] = 'B-Gained'
    plot_data.loc[plot_data['B'] == 0, 'ascn_state'] = 'A-Hom'
    plot_data.loc[plot_data['A'] == 0, 'ascn_state'] = 'B-Hom'

    ax = axes[1]
    scgenome.pl.plot_profile(
        plot_data,
        y='BAF',
        hue='ascn_state',
        ax=ax,
        palette=allele_state_colors,
        s=4,
        alpha=1.,
        hue_order=allele_state_colors.keys(),
    )

    sns.move_legend(
        ax, 'upper left', prop={'size': 8}, markerscale=3, bbox_to_anchor=(1, 1),
        labelspacing=0.4, handletextpad=0, columnspacing=0.5,
        ncol=1, title='Total CN state', title_fontsize=10, frameon=False)
    
    ax.grid(ls=':', lw=0.5, zorder=-100, which='major', axis='y')
    ax.set_axisbelow(True)

    if chromosome is None:
        scgenome.plotting.cn.setup_genome_xaxis_ticks(
            ax, chromosome_names=dict(zip(
                [str(a) for a in range(1, 23)] + ['X'],
                ['1', '2', '3', '4', '5', '6', '7', '8', '9', '', '11', '', '13', '', '15', '', '', '18', '', '', '21', '', 'X'])))

    return fig
