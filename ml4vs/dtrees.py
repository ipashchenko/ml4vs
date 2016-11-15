# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeClassifier
from hyperopt import hp, fmin, tpe, STATUS_OK, Trials
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import StandardScaler

names = ['Magnitude', 'clipped_sigma', 'meaningless_1', 'meaningless_2',
         'star_ID', 'weighted_sigma', 'skew', 'kurt', 'I', 'J', 'K', 'L',
         'Npts', 'MAD', 'lag1', 'RoMS', 'rCh2', 'Isgn', 'Vp2p', 'Jclp', 'Lclp',
         'Jtim', 'Ltim', 'CSSD', 'Ex', 'inv_eta', 'E_A', 'S_B', 'NXS', 'IQR']

names_to_delete = ['Magnitude', 'meaningless_1', 'meaningless_2', 'star_ID']


def shift_log_transform(df, name, shift):
    df[name] = np.log(df[name] + shift)


def load_data(fnames, names, names_to_delete):
    """
    Function that loads data from series of files where first file contains
    class of zeros and other files - classes of ones.

    :param fnames:
        Iterable of file names.
    :param names:
        Names of columns in files.
    :param names_to_delete:
        Column names to delete.
    :return:
        X, y - ``sklearn`` arrays of features & responces.
    """
    # Load data
    dfs = list()
    for fn in fnames:
        dfs.append(pd.read_table(fn, names=names, engine='python',
                                 na_values='+inf', sep=r"\s*",
                                 usecols=range(30)))

    # Remove meaningless features
    delta = list()
    for df in dfs:
        delta.append(df['CSSD'].min())
    delta = np.min([d for d in delta if not np.isinf(d)])
    print "delta = {}".format(delta)

    for df in dfs:
        for name in names_to_delete:
            del df[name]
        try:
            shift_log_transform(df, 'CSSD', -delta + 0.1)
        except KeyError:
            pass

    # List of feature names
    features_names = list(dfs[0])
    # Count number of NaN for each feature
    for i, df in enumerate(dfs):
        print("File {}".format(i))
        for feature in features_names:
            print("Feature {} has {} NaNs".format(feature,
                                                  df[feature].isnull().sum()))
        print("=======================")

    # Convert to numpy arrays
    # Features
    X = list()
    for df in dfs:
        X.append(np.array(df[list(features_names)].values, dtype=float))
    X = np.vstack(X)
    # Responses
    y = np.zeros(len(X))
    y[len(dfs[0]):] = np.ones(len(X) - len(dfs[0]))

    df = pd.concat(dfs)
    df['variable'] = y

    return X, y, df, features_names, delta


def load_data_tgt(fname, names, names_to_delete, delta):
    """
    Function that loads target data for classification.

    :param fname:
        Target data file.
    :param names:
        Names of columns in files.
    :param names_to_delete:
        Column names to delete.
    :return:
        X, ``sklearn`` array of features, list of feature names
    """
    # Load data
    df = pd.read_table(fname, names=names, engine='python', na_values='+inf',
                       sep=r"\s*", usecols=range(30))

    for name in names_to_delete:
        del df[name]
    try:
        shift_log_transform(df, 'CSSD', -delta + 0.1)
    except KeyError:
        pass

    # List of feature names
    features_names = list(df)
    # Count number of NaN for each feature
    for feature in features_names:
        print("Feature {} has {} NaNs".format(feature,
                                              df[feature].isnull().sum()))
    print("=======================")

    # Convert to numpy arrays
    # Features
    X = np.array(df[list(features_names)].values, dtype=float)

    # Original data
    df_orig = pd.read_table(fname, names=names, engine='python', na_values='+inf',
                       sep=r"\s*", usecols=range(30))

    return X, features_names, df, df_orig


import os
# load data
# Load data
data_dir = '/home/ilya/code/ml4vs/data/dataset_OGLE/indexes_normalized'
file_1 = 'vast_lightcurve_statistics_normalized_variables_only.log'
file_0 = 'vast_lightcurve_statistics_normalized_constant_only.log'
file_0 = os.path.join(data_dir, file_0)
file_1 = os.path.join(data_dir, file_1)
names = ['Magnitude', 'clipped_sigma', 'meaningless_1', 'meaningless_2',
         'star_ID', 'weighted_sigma', 'skew', 'kurt', 'I', 'J', 'K', 'L',
         'Npts', 'MAD', 'lag1', 'RoMS', 'rCh2', 'Isgn', 'Vp2p', 'Jclp', 'Lclp',
         'Jtim', 'Ltim', 'CSSD', 'Ex', 'inv_eta', 'E_A', 'S_B', 'NXS', 'IQR']
names_to_delete = ['Magnitude', 'meaningless_1', 'meaningless_2', 'star_ID',
                   'Npts', 'CSSD']

X, y, df, features_names, delta = load_data([file_0, file_1], names, names_to_delete)
target = 'variable'
predictors = list(df)
predictors.remove(target)
dtrain = df

sss = StratifiedShuffleSplit(n_splits=1, test_size=0.25, random_state=123)
for train_indx, test_indx in sss.split(dtrain[predictors].index, dtrain['variable']):
    print train_indx, test_indx
    train = dtrain.iloc[train_indx]
    valid = dtrain.iloc[test_indx]


def objective(space):
    clf = DecisionTreeClassifier(max_depth=space['max_depth'],
                                 max_features=space['max_features'],
                                 criterion=space['criterion'],
                                 min_weight_fraction_leaf=space['mwfl'],
                                 class_weight='balanced')
    estimators = list()
    estimators.append(('imputer', Imputer(missing_values='NaN', strategy='median',
                                          axis=0, verbose=2)))
    estimators.append(('clf', clf))
    pipeline = Pipeline(estimators)

    pipeline.fit(train[predictors], train['variable'])

    pred = pipeline.predict_proba(valid[predictors])[:, 1]
    auc = roc_auc_score(valid['variable'], pred)
    print "SCORE:", auc

    return{'loss': 1-auc, 'status': STATUS_OK }


space ={
    'max_depth': hp.choice("max_depth", (1, 2, 3)),
    'max_features': hp.choice("max_features", range(1, 25, 1)),
    'criterion': hp.choice("criterion", ("gini", "entropy")),
    'mwfl': hp.uniform("mwfl", 0, 0.1)
}


trials = Trials()
best = fmin(fn=objective,
            space=space,
            algo=tpe.suggest,
            max_evals=1000,
            trials=trials)

import hyperopt
print hyperopt.space_eval(space, best)
best_pars = hyperopt.space_eval(space, best)

# Construct classifier with best parameters
clf = DecisionTreeClassifier(max_depth=best_pars['max_depth'],
                             max_features=best_pars['max_features'],
                             criterion=best_pars['criterion'],
                             min_weight_fraction_leaf=best_pars['mwfl'],
                             class_weight='balanced')
calibrated_clf = CalibratedClassifierCV(clf, method='sigmoid', cv=4)
estimators = list()
estimators.append(('imputer', Imputer(missing_values='NaN', strategy='median',
                                      axis=0, verbose=2)))
estimators.append(('clf', calibrated_clf))
pipeline = Pipeline(estimators)

# Fit classifier with best hyperparameters on whole data set
pipeline.fit(dtrain[predictors], dtrain['variable'])

# Fit using uncalibrated classifier to plot tree
clf_ = DecisionTreeClassifier(max_depth=best_pars['max_depth'],
                              max_features=best_pars['max_features'],
                              criterion=best_pars['criterion'],
                              min_weight_fraction_leaf=best_pars['mwfl'],
                              class_weight='balanced')
estimators_ = list()
estimators_.append(('imputer', Imputer(missing_values='NaN', strategy='median',
                                       axis=0, verbose=2)))
estimators_.append(('clf', clf_))
pipeline_ = Pipeline(estimators_)

# Fit classifier with best hyperparameters on whole data set
pipeline_.fit(dtrain[predictors], dtrain['variable'])

from sklearn import tree
import pydotplus
dot_data = tree.export_graphviz(clf_, out_file=None,
                                feature_names=predictors,
                                class_names=["0", "1"],
                                filled=True, rounded=True,
                                special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data)
graph.write_pdf("tree_31.pdf")

# Load blind test data
file_tgt = 'LMC_SC19_PSF_Pgood98__vast_lightcurve_statistics_normalized.log'
file_tgt = os.path.join(data_dir, file_tgt)
X_tgt, feature_names, df, df_orig = load_data_tgt(file_tgt, names, names_to_delete,
                                                  delta)

y_probs = pipeline.predict_proba(df[predictors])[:, 1]
idx = y_probs > 0.3
idx_ = y_probs < 0.3
dtree_no = list(df_orig['star_ID'][idx_])
print("Found {} variables".format(np.count_nonzero(idx)))

# with open('dtree_results.txt', 'w') as fo:
#     for line in list(df_orig['star_ID'][idx]):
#         fo.write(line + '\n')



