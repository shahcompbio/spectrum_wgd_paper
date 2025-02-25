import itertools
import scipy
import pandas as pd
import statsmodels.stats.multitest
import matplotlib.pyplot as plt
from matplotlib.transforms import blended_transform_factory


def get_significance_string(p_value):
    """
    Compute significance level string based on the p-value.

    Parameters
    ----------
    p_value : float
        The p-value to evaluate.

    Returns
    -------
    str
        The significance level as a string.
    """
    if p_value <= 1.00e-04:
        return '****'
    elif p_value <= 1.00e-03:
        return '***'
    elif p_value <= 1.00e-02:
        return '**'
    elif p_value <= 5.00e-02:
        return '*'
    else:
        return 'ns'


def run_unpaired_tests(data, categories, test_func, **kwargs):
    """ Perform a series of statistical tests on the given data.

    Parameters
    ----------
    data : DataFrame
        Input data containing the variables of interest.
    categories : list of str
        Column names of the categorical variables in the data.
    test_func : function
        The function to use for statistical testing.
    **kwargs : dict
        Additional keyword arguments to pass to the test function.

    Returns
    -------
    DataFrame
        DataFrame containing the Mann-Whitney U test statistics and p-values for each combination of categories.

    Notes
    -----
    - The test is applied to each combination of categories to compare the distributions of the dependent variable.
    - Significance values
          ns: 5.00e-02 < p <= 1.00e+00
           *: 1.00e-02 < p <= 5.00e-02
          **: 1.00e-03 < p <= 1.00e-02
         ***: 1.00e-04 < p <= 1.00e-03
        ****: p <= 1.00e-04
    """

    stats = []
    for (idx1, grp1), (idx2, grp2) in itertools.combinations(data.groupby(categories), 2):
        info = test_func(grp1, grp2, **kwargs)
        info.update(dict(zip([a + '_1' for a in categories], idx1)))
        info.update(dict(zip([a + '_2' for a in categories], idx2)))
        stats.append(info)
    
    stats = pd.DataFrame(stats)

    stats['significance'] = stats['p'].apply(get_significance_string)

    return stats


def mwu_tests(data, categories, y):
    """
    Perform Mann-Whitney U tests one or more categories.

    Parameters
    ----------
    data : DataFrame
        Input data containing the variables of interest.
    categories : list of str
        Column names of the categorical variables in the data.
    y : str
        Column name of the dependent variable in the data.

    Returns
    -------
    DataFrame
        DataFrame containing the Mann-Whitney U test statistics and p-values for each combination of categories.

    Notes
    -----
    - The Mann-Whitney U test is a non-parametric test used to compare the distributions of two independent samples.
    - The test is applied to each combination of categories to compare the distributions of the dependent variable.
    - The alternative hypothesis can be either 'greater' or 'less', depending on the mean values of the two samples.
    - Significance values
          ns: 5.00e-02 < p <= 1.00e+00
           *: 1.00e-02 < p <= 5.00e-02
          **: 1.00e-03 < p <= 1.00e-02
         ***: 1.00e-04 < p <= 1.00e-03
        ****: p <= 1.00e-04
    """
    mwu_stats = []
    for (idx1, grp1), (idx2, grp2) in itertools.combinations(data.groupby(categories), 2):
        d1 = grp1[y]
        d2 = grp2[y]
        if d1.mean() > d2.mean():
            alternative = 'greater'
        else:
            alternative = 'less'
        s, p = scipy.stats.mannwhitneyu(d1, d2, alternative=alternative)
        info = {
            'alternative': alternative,
            's': s,
            'p': p,
            'y_mean_1': d1.mean(),
            'y_mean_2': d2.mean(),
            'n_1': len(d1),
            'n_2': len(d2),
        }
        info.update(dict(zip([a + '_1' for a in categories], idx1)))
        info.update(dict(zip([a + '_2' for a in categories], idx2)))
        mwu_stats.append(info)
    
    mwu_stats = pd.DataFrame(mwu_stats)

    mwu_stats['significance'] = mwu_stats['p'].apply(get_significance_string)

    return mwu_stats


def fdr_correction(mwu_stats, alpha=0.05, method='poscorr'):
    """
    Perform False Discovery Rate (FDR) correction on the given p-values in the DataFrame.

    Parameters
    ----------
    mwu_stats : DataFrame
        DataFrame containing the p-values to correct for multiple testing.
    alpha : float, optional
        The significance level to control the FDR, by default 0.05.
    method : str, optional
        The method to use for FDR correction. Options are 'indep', 'negcorr', by default 'indep'.

    Returns
    -------
    DataFrame
        DataFrame with additional columns for corrected p-values and significance levels.

    Notes
    -----
    - see https://www.statsmodels.org/dev/generated/statsmodels.stats.multitest.fdrcorrection.html for details on FDR correction
    - The function adds the following columns to the input DataFrame:
        - 'is_significant_corrected': Boolean values indicating which hypotheses were rejected after FDR correction.
        - 'pvalue_corrected': Corrected p-values after FDR correction.
        - 'significance_corrected': Significance levels after FDR correction.
        - 'is_significant_uncorrected': Boolean values indicating which hypotheses were rejected before FDR correction.
    """
    mwu_stats['is_significant_corrected'], mwu_stats['pvalue_corrected'] = statsmodels.stats.multitest.fdrcorrection(
        mwu_stats['p'],
        alpha=alpha,
        method=method,
        is_sorted=False)

    mwu_stats['significance_corrected'] = mwu_stats['pvalue_corrected'].apply(get_significance_string)

    mwu_stats['is_significant_uncorrected'] = mwu_stats['significance'] != 'ns'
    mwu_stats[mwu_stats['is_significant_corrected'] != mwu_stats['is_significant_uncorrected']]

    return mwu_stats


def add_significance_line(ax, sig, x_pos_1, x_pos_2, y_pos):
    """
    Add a significance line with text annotation to a plot.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes object to add the significance line to.
    sig : str
        The text annotation to display above the line.
    x_pos_1 : float
        The x-coordinate of the start of the line.
    x_pos_2 : float
        The x-coordinate of the end of the line.
    y_pos : float
        The y-coordinate of the line.
    """
    transform = blended_transform_factory(ax.transData, ax.transAxes)
    line = plt.Line2D([x_pos_1, x_pos_2], [y_pos, y_pos], transform=transform, color='k', linestyle='-', linewidth=1)
    ax.get_figure().add_artist(line)
    y_pos_offset = 0
    if '*' in sig:
        y_pos_offset -= 0.02
    plt.text((x_pos_1+x_pos_2)/2, y_pos+y_pos_offset, sig, ha='center', va='bottom', fontsize=8, transform=transform)

