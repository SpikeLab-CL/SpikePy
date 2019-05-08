import itertools
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import ks_2samp
from statsmodels.distributions.empirical_distribution import ECDF
import pandas as pd
from tqdm import tqdm_notebook as progress_bar
from scipy.stats import rankdata
from typing import List
from scipy.stats import norm
from scipy.stats import wasserstein_distance as em_distance



def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='',
                          ylabel='', xlabel='',
                          cmap=plt.cm.Blues, fmt=None):
    """
    Prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.

    Example:
    -------
    ```
    np.set_printoptions(precision=2)
    cnf_matrix = sklearn.metrics.confusion_matrix(df_pred['catReal'],
                 df_pred['catPredict'])
    plt.figure(figsize=(10, 8))
    plot_confusion_matrix(cnf_matrix, classes=names_siniestralidad,
                      normalize=True,
            title="Modelo Completo \n Matriz de confusión normalizada")
    ```

    Taken from http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html#sphx-glr-auto-examples-model-selection-plot-confusion-matrix-py
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    else:
        pass

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=16)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if fmt is None:
        fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel(ylabel, fontsize=14)
    plt.xlabel(xlabel, fontsize=14)
    

def plot_most_important_features(h2o_model, title=None, num_of_features=None):
    """
    Plots the feature importance for a trained H2o model in a better way compared with varimp_plot.

    Example:
    -------
    ```
    plot_most_important_features(best_model, title="a very cool H2o model", num_of_features=10)
    ```
    Returns matplotlib fig and ax
    """

    plt.rcdefaults()
    fig, ax = plt.subplots()
    
    variables = h2o_model._model_json['output']['variable_importances']['variable']
    if num_of_features is None:
        num_of_features = min(len(variables), 10)
    variables = variables[0:num_of_features]
    y_pos = np.arange(len(variables))
    scaled_importance = h2o_model._model_json['output']['variable_importances']['scaled_importance']
    ax.barh(y_pos, scaled_importance[0:num_of_features], align='center', color='#3498db', ecolor='black')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(variables)
    ax.invert_yaxis()
    ax.set_xlabel('scaled importance')
    ax.set_title(title)
    fig.set_size_inches(6, 8)
    return fig, ax


def compare_cont_dists(df_list: List, variables=None, labels=None,
                       divisor_step=2, nsample=5000, nbins=50, all=False,
                       sort_by='earth_mover', nvars=None, plot=True, plot_cdf=True,
                       figsize=(7, 4), normalize=True) -> tuple:
    """
    Compares two distributions on a list of continuous or numerical variables

    :param df_list: list of dataframe
    :param variables: list of variables to plot
    :param labels: labels of dataframes for plots
    :param divisor_step: fraction of minimal difference between distributions to reflect on plot
    :param nsample: maximum number of observations per distribution
    :param nbins: number of bins in histogram
    :param all: plots the mixture of all distributions (all dataframes in df_list)
    :param sort_by: sorts histograms order by some metric (example: earth mover distance)
    :param nvars: numbers of variables that will be plot
    :param plot_cdf: if True it plots the cdf
    :param normalize: transform variables to [0,1] to compute the sort metric
    :param: figsize: tuple of fig size

    :return: fig, axes, em_dist
    """
    if variables is None:
        variables = df_list[0].columns
    ndf = len(df_list)
    if labels is None:
        labels = [f'df{df_index}' for df_index in range(ndf)]


    if plot_cdf:
        ncolumns = 2
    else:
        ncolumns = 1

    if nvars is None:
        nvars = len(variables)

    df_sample = {}
    for df_index, df in enumerate(df_list):
        df_sample[df_index] = df.sample(min(len(df), nsample))

    values = {}
    ecdf = {}
    pvalor_ks = pd.DataFrame(columns=variables, index=['metric'])
    em_dist = pvalor_ks.copy()

    for var in progress_bar(list(variables)):
        nvalues = min([len(df_sample[i][var][df_sample[i][var].notna()].values.flatten()) for i in range(min(2, ndf))])
        if nvalues > 0:
            for df_index in range(min(2, ndf)):
                values[df_index] = df_sample[df_index][var][df_sample[df_index][var].notna()].values.flatten()
                if normalize is True:
                    values[df_index] = (values[df_index] - values[df_index].min())/(values[df_index].max()- values[df_index].min())

            if len(values.keys()) > 1:
                pvalor_ks.loc['metric', var] = ks_2samp(values[0], values[1]).pvalue
                em_dist.loc['metric', var] =  em_distance(values[0], values[1])
        else:
            em_dist.loc['metric', var] = np.NaN


    #plots
    if plot:
        fig, axes = plt.subplots(nvars, ncolumns, figsize=(figsize[0] * ncolumns, nvars * figsize[1]))
        axes = axes.reshape((nvars, ncolumns))
        if sort_by == 'earth_mover':
            em_dist.sort_values(by='metric', axis=1, ascending=False, inplace=True)
            vars_sort = em_dist.columns
            vars_sort = vars_sort[:nvars]
        else:
            vars_sort = variables

        for iv, var in progress_bar(list(enumerate(vars_sort))):

            for df_index in range(ndf):
                values[df_index] = df_sample[df_index][var][df_sample[df_index][var].notna()].values.flatten()
            allvalues = np.concatenate(list(values.values()))

            #histogramas
            bins = np.linspace(allvalues.min(), allvalues.max(), nbins)
            if all:
                _, _, _ = axes[iv, 0].hist(allvalues, bins=bins, density=True, label=['ambos'], alpha=0.5)

            for df_index in range(ndf):
                _, _, _ = axes[iv, 0].hist(values[df_index], bins=bins, alpha=0.5, density=True, label=labels[df_index])

            axes[iv, 0].legend()
            axes[iv, 0].set_title(var)
            axes[iv, 0].grid(False)

            if plot_cdf:
                # cdf
                diff = abs(np.diff(allvalues))
                steps = diff[diff > 0].min() / divisor_step
                start = allvalues.min()
                stop = allvalues.max()
                x = np.arange(start, stop, steps)

                for df_index in range(ndf):
                    ecdf[df_index] = ECDF(values[df_index])
                if all:
                    ecdf_all = ECDF(allvalues)
                    axes[iv, 1].plot(x, ecdf_all(x), label='ambos')

                for df_index in range(ndf):
                    axes[iv, 1].plot(x, ecdf[df_index](x), label=labels[df_index])
                axes[iv, 1].legend()
                axes[iv, 1].set_title(f'w_distance = {em_dist[var].values[0]:.3f} ')
                axes[iv, 1].grid(False)
        plt.tight_layout()
    else:
        fig, axes = None, None

    return fig, axes, em_dist


def compare_categorical_dists(df1: pd.DataFrame, df2: pd.DataFrame, variables: List, labels=['df1', 'df2'],
                              minimal_category_size=0.01, width=.97, nsample=5000):
    """
    Compares two distributions on a list of categorical variables

    :param df1:
    :param df2:
    :param variables:
    :param labels:
    :param minimal_category_size: minimal percentage of data that a category should have to be shown
    :param width: width of bar in barplot
    :param nsample: maximum number of observations per distribution
    :return: fig, axes
    """
    fig, axes = plt.subplots(len(variables), figsize=(15, 7 * len(variables)))
    df1_sample = df1.sample(min(len(df1), nsample))
    df2_sample = df2.sample(min(len(df2), nsample))
    if len(variables) == 1:
        axes = [axes]
    for iv, var in progress_bar(list(enumerate(variables))):
        prop_general = pd.concat([df1_sample[var], df2_sample[var]]).value_counts(normalize=True)
        categs_1 = set(df1_sample[var].unique())
        categs_2 = set(df2_sample[var].unique())
        categs = categs_1.intersection(categs_2)

        categ_grandes = list(set(prop_general.index[prop_general >= minimal_category_size]).intersection(categs))
        categ_chicas = list(set(prop_general.index[prop_general < minimal_category_size]).intersection(categs))

        props1_ = df1[var].value_counts(normalize=True)
        props2_ = df2[var].value_counts(normalize=True)

        if len(categ_chicas) > 0:
            props1 = props1_[categ_grandes]
            props2 = props2_[categ_grandes]

            props1['otras'] = props1_[categ_chicas].sum()
            props2['otras'] = props2_[categ_chicas].sum()
        else:
            props1 = props1_
            props2 = props2_

        props = 100 * pd.concat([props1, props2], axis=1, keys=[labels[0], labels[1]])
        props['sum'] = props.sum(axis=1)
        props.sort_values(by='sum', inplace=True, ascending=False)
        props.drop(['sum'], axis=1, inplace=True)
        props.plot(kind='bar', width=width, ax=axes[iv])

        axes[iv].set_title(var)
        for i, p in enumerate(np.array(axes[iv].patches).reshape(2, -1).T):
            diff = 100 * np.abs(props.iloc[i, 0] - props.iloc[i, 1]) / props.iloc[i].mean()
            altura_max = max(p[0].get_height(), p[1].get_height())

            x = p[0].get_x()
            axes[iv].annotate(f'{diff:.1f}%', (x, altura_max * 1.01))
        axes[iv].grid(False)
    plt.tight_layout()

    return fig, axes


def pdplot(df: pd.DataFrame, variables: List, target: str, numeric: bool,
           pos_class=1, nsample=1_000_000, nbins=100, size_subplot=(15, 7),
           confidence_q=[.95], sort_q=.3, ncategories=100) -> tuple:
    """
    Partial dependency plot for a categorical target variable.
    For now, all variables must be either all numerical or all categorical

    :param df:
    :param variables:
    :param target: name of target column
    :param numeric: whether features are numeric
    :param pos_class: label of positive class
    :param nsample: maximum number of observations per distribution
    :param nbins: number of bins to discretize numerical variables
    :param confidence_q: probability of confidence interval
    :param sort_q: probability of confidece interval for wich the plot will be order by
    :param ncategories: number of categories to show
    :return:
    """
    def npenetracion(df_, target_, pos_class_):
        return len(df_[df_[target_] == pos_class_])

    df_sample = df.sample(min(nsample, len(df)))
    fig, axes = plt.subplots(len(variables),
                             figsize=(size_subplot[0], size_subplot[1] * len(variables)))

    global_pen = 100 * (df_sample[target] == pos_class).mean()
    for iv, var in progress_bar(list(enumerate(variables))):
        axes[iv].axhline(y=global_pen, linestyle='--', color='k')
        if numeric:
            quantiles = (list(np.unique(np.quantile(
                              df_sample[var].values, q=(1 / nbins) * np.arange(1, nbins)))))
            min_df = df_sample[var].min()
            max_df = df_sample[var].max()
            rango = max_df - min_df
            eps = .001 * rango
            quantiles = np.array([min_df - eps] + quantiles + [max_df + eps])
            quant_interval = np.array([f'({quantiles[i]:.2f}, {quantiles[i + 1]:.2f}]'
                                       for i in range(len(quantiles) - 1)])
            cdf_values = ECDF(quantiles)(df_sample[var])
            index_quantile = rankdata(cdf_values, method='dense') - 1
            df_sample[var] = quant_interval[index_quantile]

        size = df_sample[var].value_counts()
        size_posclass = df_sample.groupby(var).apply(npenetracion, target, pos_class)
        pen_posclass = (size_posclass / size)
        variance = pen_posclass * (1 - pen_posclass)

        lower_conf = {}
        upper_conf = {}
        for q in confidence_q + [sort_q]:
            factor_confidence =  -norm.ppf((1-q)/2)
            lower_conf[q] = 100 * np.maximum(pen_posclass - factor_confidence * np.sqrt(variance / size), 0)
            upper_conf[q] = 100 * np.minimum(pen_posclass + factor_confidence * np.sqrt(variance / size), 1)

        if numeric:
            sort_categories = quant_interval
        else:
            sort_categories = list(lower_conf[sort_q].sort_values(ascending=False).index)[:ncategories]

        pen_posclass = 100 * pd.DataFrame(pen_posclass[sort_categories])
        pen_posclass.rename(columns={0: 'prob'}, inplace=True)
        pen_posclass.plot(ax=axes[iv], kind='bar')
        for q in confidence_q:
            axes[iv].fill_between(range(len(sort_categories)), lower_conf[q][sort_categories].values,
                                  upper_conf[q][sort_categories].values,
                                  alpha=0.5, label=f'{100 * q} %')
        axes[iv].set_ylabel('% ' + target + ' = ' + str(pos_class))
        axes[iv].set_title(var)
        axes[iv].grid(False)
        axes[iv].legend()
    plt.tight_layout()

    return fig, axes
