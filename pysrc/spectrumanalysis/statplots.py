import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from statannotations.Annotator import Annotator


def boxplot1(data, col, y, x, order, col_order, pairs, ylabel, palette, yscale='log'):
    """
    Create boxplots with annotations.

    Parameters
    ----------
    data : DataFrame
        DataFrame containing the data to plot.
    col : str
        Column name for the categorical variable to create subplots.
    y : str
        Column name for the dependent variable.
    x : str
        Column name for the independent variable.
    order : list
        Order of the categories in the x-axis.
    col_order : list
        List of categories to plot in the subplots.
    pairs : list
        List of pairs for statistical annotation.
    ylabel : str
        Label for the y-axis.
    palette : dict
        Color palette for the plot.
    yscale : str, optional
        Scale for the y-axis, by default 'log'.
    """
    g = sns.catplot(
        col=col, y=y, x=x, hue=x,
        data=data, kind='box', sharey=False, dodge=False, fliersize=1,
        order=order, col_order=col_order,
        height=3, aspect=0.4, palette=palette)

    g.map_dataframe(
        sns.stripplot, y=y, x=x, hue=x,
        order=order,
        dodge=False, linewidth=1,
        palette=palette)

    for ax in g.axes.flatten():
        if yscale:
            ax.set_yscale(yscale)
        ax.set_xlabel('')
        for label in ax.get_xticklabels():
            label.set_rotation(60)
            label.set_ha('right')
            label.set_rotation_mode('anchor')
    g.set_titles('{col_name}')
    g.set_ylabels(ylabel)
    g._legend.remove()
    g.fig.set_dpi(150)

    for ax, event_type in zip(g.axes.flatten(), col_order):
        annotator = Annotator(
            ax, pairs,
            y=y, x=x,
            data=data[data[col] == event_type], kind='box')
        annotator.configure(test='Mann-Whitney', text_format='star', loc='inside', line_width=1)
        annotator.apply_and_annotate()

    plt.tight_layout()

    return g
