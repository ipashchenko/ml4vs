# -*- coding: utf-8 -*-
import copy

import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.model_selection import learning_curve


def plot_corr(df, size=10):
    """Function plots a graphical correlation matrix for each pair of columns in
    the dataframe.

    Input:
        df: pandas DataFrame
        size: vertical and horizontal size of the plot
    """

    corr = df.corr()
    fig, ax = plt.subplots(figsize=(size, size))
    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True
    m = ax.matshow(np.ma.array(corr, mask=mask))
    cbar = fig.colorbar(m, ticks=[-1, 1])
    cbar.ax.set_yticklabels([-1, 0, 1])
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=45)
    plt.yticks(range(len(corr.columns)), corr.columns, rotation=45)


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


def plot_importance(est, names):
    # sort importances
    indices = np.argsort(est.feature_importances_)
    # plot as bar chart
    fig, ax = plt.subplots()
    ax.barh(np.arange(len(names)), est.feature_importances_[indices])
    ax.set_yticks(np.arange(len(names)) + 0.25)
    ax.set_yticklabels(np.array(names)[indices])
    ax.set_xlabel('Relative importance')
    fig.show()
    return fig


def plot_importance_xgb(clf, feature_names):
    scores_dict = clf.booster().get_fscore()
    scores_dict_ = dict()
    for i, feature in enumerate(feature_names):
        scores_dict_[feature] = scores_dict['f{}'.format(i)]

    feat_imp = pd.Series(scores_dict_).sort_values(ascending=False)
    feat_imp.plot(kind='bar', title='Feature Importances')
    plt.ylabel('Feature Importance Score')
    plt.show()


def plot_features_hist(X, names, bins=100):
    for i, name in enumerate(names):
        fig = plt.figure()
        ax = plt.gca()
        ax.hist(X[:, i], bins=bins)
        ax.set_xlabel("X_{}".format(i))
        fig.savefig("X_{}_hist.png".format(i))
        fig.close()


def plot_reliability_curve(y_true, y_prob, fig_index=None, n_bins=5):
    from sklearn.calibration import calibration_curve
    b, a = calibration_curve(y_true, y_prob, n_bins=n_bins, normalize=True)

    fig = plt.figure(fig_index, figsize=(10, 10))
    ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
    ax2 = plt.subplot2grid((3, 1), (2, 0))
    ax1.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
    ax1.plot(a, b, '.k')
    ax1.plot(a, b)
    ax1.set_ylabel("Fraction of positives")
    ax1.set_ylim([-0.05, 1.05])
    prob_pos = (y_prob - y_prob.min()) / (y_prob.max() - y_prob.min())
    ax2.hist(prob_pos, n_bins, range=[0, 1], lw=2)
    ax2.set_ylabel("Count")
    ax2.set_xlabel("Mean predicted value")
    plt.tight_layout()
    return fig


def plot_cv_reliability(clf, X, y, n_cv=4, n_bins=4, seed=1, fit_params=None):
    from itertools import cycle
    from sklearn.model_selection import StratifiedKFold

    if fit_params is None:
        fit_params = {}

    colors = cycle(['cyan', 'indigo', 'seagreen', 'yellow', 'blue',
                    'darkorange'])

    fig, axes = plt.subplots(1, 1)
    axes.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")

    cv = StratifiedKFold(n_splits=n_cv, shuffle=True, random_state=seed)
    for (train, test), color in zip(cv.split(X, y), colors):
        clf.fit(X[train], y[train], **fit_params)
        probas_ = clf.predict_proba(X[test])
        # Scikit-learn
        try:
            probas_ = probas_[:, 1]
        # Keras
        except IndexError:
            probas_ = probas_[:, 0]
        # probas_ = clf.fit(X[train], y[train]).predict_proba(X[test])[:, 1]
        b, a = calibration_curve(y[test], probas_, n_bins=n_bins,
                                 normalize=True)
        axes.plot(a, b, color=color)
        axes.plot(a, b, '.k')
    axes.set_ylim([-0.05, 1.05])
    axes.set_ylabel("Fraction of positives")
    axes.set_xlabel("Mean predicted value")
    axes.legend(loc="upper left")
    plt.show()
    return fig


def plot_cv_roc(clf, X, y, n_cv=4, seed=1, fit_params=None):
    from itertools import cycle
    from sklearn.metrics import roc_curve, auc
    from sklearn.model_selection import StratifiedKFold
    from scipy import interp

    if isinstance(clf, keras.models.Sequential):
        clf.save('model.h5')

    if fit_params is None:
        fit_params = {}

    colors = cycle(['cyan', 'indigo', 'seagreen', 'yellow', 'blue',
                    'darkorange'])
    lw = 2

    i = 0
    mean_tpr = 0.0
    mean_fpr = np.linspace(0, 1, 100)
    fig = plt.figure()

    cv = StratifiedKFold(n_splits=n_cv, shuffle=True, random_state=seed)
    for (train, test), color in zip(cv.split(X, y), colors):

        if isinstance(clf, keras.models.Sequential):
            clf_ = keras.models.load_model('model.h5')
        else:
            clf_ = copy.deepcopy(clf)

        clf_.fit(X[train], y[train], **fit_params)
        probas_ = clf_.predict_proba(X[test])
        # Scikit-learn
        try:
            probas_ = probas_[:, 1]
        # Keras
        except IndexError:
            probas_ = probas_[:, 0]
        # probas_ = clf.fit(X[train], y[train]).predict_proba(X[test])
        # Compute ROC curve and area the curve
        fpr, tpr, thresholds = roc_curve(y[test], probas_)
        mean_tpr += interp(mean_fpr, fpr, tpr)
        mean_tpr[0] = 0.0
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=lw, color=color,
                 label='ROC fold %d (area = %0.2f)' % (i, roc_auc))

        i += 1
    plt.plot([0, 1], [0, 1], linestyle='--', lw=lw, color='k',
             label='Luck')

    mean_tpr /= cv.get_n_splits(X, y)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    plt.plot(mean_fpr, mean_tpr, color='g', linestyle='--',
             label='Mean ROC (area = %0.2f)' % mean_auc, lw=lw)

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC')
    plt.legend(loc="lower right")
    plt.show()

    return fig


def plot_cv_pr(clf, X, y, n_cv=4, seeds=None, fit_params=None, fig=None,
               label=None, plot_color=None):
    """
    This averages P for the same values of R.
    """
    from itertools import cycle
    from sklearn.metrics import precision_recall_curve, auc, average_precision_score
    from sklearn.model_selection import StratifiedKFold
    from scipy import interp

    if isinstance(clf, keras.models.Sequential):
        clf.save('model.h5')

    if fit_params is None:
        fit_params = {}

    colors = cycle(['cyan', 'indigo', 'seagreen', 'yellow', 'blue',
                    'darkorange'])
    lw = 1

    if not fig:
        fig = plt.figure()

    for seed in seeds:
        i = 0
        mean_pr = 0.0
        mean_rec = np.linspace(0, 1, 100)[::-1]

        cv = StratifiedKFold(n_splits=n_cv, shuffle=True, random_state=int(seed))
        for (train, test), color in zip(cv.split(X, y), colors):

            if isinstance(clf, keras.models.Sequential):
                clf_ = keras.models.load_model('model.h5')
            else:
                clf_ = copy.deepcopy(clf)

            clf_.fit(X[train], y[train], **fit_params)
            probas_ = clf_.predict_proba(X[test])
            # Scikit-learn
            try:
                probas_ = probas_[:, 1]
            # Keras
            except IndexError:
                probas_ = probas_[:, 0]
            # Compute PR curve and area the curve
            # fpr, tpr, thresholds = precision_recall_curve(y[test], probas_[:, 1])
            pr, rec, thresholds = precision_recall_curve(y[test], probas_)
            mean_pr += interp(mean_rec, rec[::-1], pr[::-1])[::-1]
            pr_auc = average_precision_score(y[test], probas_)
            # plt.plot(rec, pr, lw=lw, color=color,
            #          label='PR-curve fold %d (area = %0.2f)' % (i, pr_auc))
            i += 1

        mean_pr /= cv.get_n_splits(X, y)
        mean_pr[-1] = 0.0
        mean_pr[0] = 1.0
        mean_auc = auc(mean_rec, mean_pr, reorder=True)
        # plt.plot(mean_rec[::-1], mean_pr, color='g', linestyle='--',
        #          label='Mean PR-curve (area = %0.2f)' % mean_auc, lw=lw)
        if not plot_color:
            plot_color = 'g'
        plt.plot(mean_rec[::-1], mean_pr, color=plot_color,
                 label=label, lw=lw, alpha=0.25)

        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('PR-curve')
        plt.legend(loc="lower left")
        # plt.show()

    plt.show()

    return fig


def best_f1_thresholds(clf, X, y, n_cv=4, seed=1, fit_params=None):
    from sklearn.metrics import precision_recall_curve, f1_score,\
        confusion_matrix
    from sklearn.cross_validation import StratifiedKFold

    if isinstance(clf, keras.models.Sequential):
        clf.save('model.h5')

    if fit_params is None:
        fit_params = {}

    cv = StratifiedKFold(y, n_folds=n_cv, shuffle=True, random_state=seed)
    cv_threshs = list()
    cv_probas = list()
    CMs = list()
    for train_idx, test_idx in cv:

        if isinstance(clf, keras.models.Sequential):
            clf_ = keras.models.load_model('model.h5')
        else:
            clf_ = copy.deepcopy(clf)

        clf_.fit(X[train_idx], y[train_idx], **fit_params)
        probas = clf_.predict_proba(X[test_idx])
        # Scikit-learn
        try:
            probas = probas[:, 1]
        # Keras
        except IndexError:
            probas = probas[:, 0]
        # probas = clf.fit(X[train_idx], y[train_idx]).predict_proba(X[test_idx])
        cv_probas.append(probas)
        pr, rec, threshs = precision_recall_curve(y[test_idx], probas)
        f1 = [f1_score(y[test_idx], np.array(probas > thresh, dtype=int))
              for thresh in threshs]
        f1_max = max(f1)
        idx = f1.index(f1_max)
        thresh_max = threshs[idx]
        cv_threshs.append(thresh_max)
        print "Maximum F1={} for threshold={}".format(f1_max, thresh_max)
    best_thresh = np.mean(cv_threshs)
    for (train_idx, test_idx), cv_proba in zip(cv, cv_probas):
        preds = np.array(cv_proba > best_thresh, dtype=int)
        CMs.append(confusion_matrix(y[test_idx], preds))

    CM = np.sum(CMs, axis=0)
    FN = CM[1][0]
    TP = CM[1][1]
    FP = CM[0][1]
    print "TP = {}".format(TP)
    print "FP = {}".format(FP)
    print "FN = {}".format(FN)
    f1 = 2. * TP / (2. * TP + FP + FN)
    P = float(TP) / (TP + FP)
    R = float(TP) / (TP + FN)
    print "P={}".format(P)
    print "R={}".format(R)
    print "F1={}".format(f1)

    return best_thresh, f1


def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5),
                        scoring=None):
    """
    Generate a simple plot of the test and training learning curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
          - None, to use the default 3-fold cross-validation,
          - integer, to specify the number of folds.
          - An object to be used as a cross-validation generator.
          - An iterable yielding train/test splits.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : integer, optional
        Number of jobs to run in parallel (default 1).
    """
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores =\
        learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs,
                       train_sizes=train_sizes, scoring=scoring)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt
