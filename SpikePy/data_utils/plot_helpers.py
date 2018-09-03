import itertools
import matplotlib.pyplot as plt
import numpy as np


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
