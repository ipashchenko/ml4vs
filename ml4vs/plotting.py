# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sea


def plot_corr(df, size=10):
    """Function plots a graphical correlation matrix for each pair of columns in
    the dataframe.

    Input:
        df: pandas DataFrame
        size: vertical and horizontal size of the plot
    """

    corr = df.corr()
    fig, ax = plt.subplots(figsize=(size, size))
    ax.matshow(corr)
    plt.xticks(range(len(corr.columns)), corr.columns)
    plt.yticks(range(len(corr.columns)), corr.columns)


def plot_corr_matrix(corr_matrix):
    """
    Plot correlation matrix.

    :param corr_matrix:
        Correaltion matrix.
    """
    sea.set(style="white")
    mask = np.zeros_like(corr_matrix, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True
    f, ax = plt.subplots(figsize=(11, 9))
    sea.heatmap(corr_matrix, mask=mask, square=True, xticklabels=5,
                yticklabels=5, linewidths=.5, cbar_kws={"shrink": .5}, ax=ax)


def deviance_plot(est, X_test, y_test, ax=None, label='', train_color='#2c7bb6',
                  test_color='#d7191c', alpha=1.0):
    """
    Deviance plot for ``est``, use ``X_test`` and ``y_test`` for test error.
    https://www.datarobot.com/blog/gradient-boosted-regression-trees/
    """
    test_dev = np.empty(est.n_estimators)

    for i, pred in enumerate(est.staged_predict(X_test)):
       test_dev[i] = est.loss_(y_test, pred)

    if ax is None:
        fig = plt.figure(figsize=(8, 5))
        ax = plt.gca()

    ax.plot(np.arange(est.n_estimators) + 1, test_dev, color=test_color,
            label='Test %s' % label, linewidth=2, alpha=alpha)
    ax.plot(np.arange(est.n_estimators) + 1, est.train_score_,
            color=train_color, label='Train %s' % label, linewidth=2,
            alpha=alpha)
    ax.set_ylabel('Error')
    ax.set_xlabel('n_estimators')
    ax.set_ylim((0, 2))
    return test_dev, ax


def plot_importance(est, names, outfile):
    # sort importances
    indices = np.argsort(est.feature_importances_)
    # plot as bar chart
    plt.barh(np.arange(len(names)), est.feature_importances_[indices])
    plt.yticks(np.arange(len(names)) + 0.25, np.array(names)[indices])
    _ = plt.xlabel('Relative importance')
    plt.savefig(outfile, bbox_inches='tight', dpi=500)
    plt.close()



