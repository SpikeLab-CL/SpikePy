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


def compare_cont_dists(df1: pd.DataFrame, df2: pd.DataFrame, variables: List, labels=['df1', 'df2'],
                       divisor_step=2, nsample=5000) -> tuple:
    """
    Compares two distributions on a list of continuous or numerical variables

    :param df1:
    :param df2:
    :param variables:
    :param labels: labels of dataframes for plots
    :param divisor_step: fraction of minimal difference between distributions to reflect on plot
    :param nsample: maximum number of observations per distribution
    :return: fig, axes
    """
    fig, axes = plt.subplots(len(variables), 2, figsize=(14, len(variables) * 4))
    axes = axes.reshape((len(variables), 2))
    df1_sample = df1.sample(min(len(df1), nsample))
    df2_sample = df2.sample(min(len(df2), nsample))
    for iv, var in progress_bar(list(enumerate(variables))):
        values1 = df1_sample[var][df1_sample[var].notna()].values.flatten()
        values2 = df2_sample[var][df2_sample[var].notna()].values.flatten()
        allvalues = np.concatenate((values1, values2))
        pvalor = ks_2samp(values1, values2).pvalue

        # histogramas
        _, bins, _ = axes[iv, 0].hist(values1, bins=50, range=[allvalues.min(), allvalues.max()], density=True, label=labels[0])
        _, _, _ = axes[iv, 0].hist(values2, bins=bins, alpha=0.4, density=True, label=labels[1])
        axes[iv, 0].legend()
        axes[iv, 0].set_title(var)

        # cdf
        diff = abs(np.diff(allvalues))
        steps = diff[diff > 0].min() / divisor_step
        start = allvalues.min()
        stop = allvalues.max()
        x = np.arange(start, stop, steps)

        ecdf1 = ECDF(values1)
        ecdf2 = ECDF(values2)
        axes[iv, 1].plot(x, ecdf1(x), label=labels[0])
        axes[iv, 1].plot(x, ecdf2(x), label=labels[1])
        axes[iv, 1].legend()
        axes[iv, 1].set_title(f'p-valor = {pvalor:.3f} (K-S)')
        axes[iv, 1].grid(False)
    plt.tight_layout()

    return fig, axes


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
