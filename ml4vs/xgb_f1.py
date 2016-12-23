# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
import xgboost as xgb
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


def load_to_df(fnames, names, names_to_delete, target='variable'):
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
        Pandas data frame.
    """
    # Load data
    dfs = list()
    for fn in fnames:
        dfs.append(pd.read_table(fn, names=names, engine='python',
                                 na_values='+inf', sep=r"\s*",
                                 usecols=range(30)))
    df = pd.concat(dfs)
    y = np.zeros(len(df))
    y[len(dfs[0]):] = np.ones(len(df) - len(dfs[0]))

    df[target] = y

    # Remove meaningless features
    delta = min(df['CSSD'][np.isfinite(df['CSSD'].values)])
    # print delta
    print delta

    for name in names_to_delete:
        del df[name]
    try:
        shift_log_transform(df, 'CSSD', -delta + 0.1)
    except KeyError:
        pass

    return df, delta


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
    df = pd.read_table(fname, names=names, engine='python', na_values='+inf',
                       sep=r"\s*", usecols=range(30))

    return X, features_names, df


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

sss = StratifiedShuffleSplit(n_splits=1, test_size=0.25, random_state=7)
for train_indx, test_indx in sss.split(dtrain[predictors].index, dtrain['variable']):
    print train_indx, test_indx
    train = dtrain.iloc[train_indx]
    valid = dtrain.iloc[test_indx]


def xg_f1(y,t):
    t = t.get_label()
    y_bin = [1. if y_cont > 0.5 else 0. for y_cont in y] # binaryzing your output
    return 'f1',f1_score(t,y_bin)


def objective(space):
    clf = xgb.XGBClassifier(n_estimators=10000, learning_rate=space['lr'],
                            max_depth=space['max_depth'],
                            min_child_weight=space['min_child_weight'],
                            subsample=space['subsample'],
                            colsample_bytree=space['colsample_bytree'],
                            colsample_bylevel=space['colsample_bylevel'])

    # Try using pipeline
    estimators = list()
    estimators.append(('imputer', Imputer(missing_values='NaN', strategy='median',
                                          axis=0, verbose=2)))
    # estimators.append(('scaler', StandardScaler()))
    estimators.append(('clf', clf))
    pipeline = Pipeline(estimators)

    valid_ = valid[predictors]
    train_ = train[predictors]
    for name, transform in pipeline.steps[:-1]:
        transform.fit(train_)
        valid_ = transform.transform(valid_)
        train_ = transform.transform(train_)

    eval_set = [(train_, train['variable']),
                (valid_, valid['variable'])]

    pipeline.fit(train[predictors], train['variable'],
                 clf__eval_set=eval_set, clf__eval_metric=xg_f1,
                 clf__early_stopping_rounds=50)
    # clf.fit(train[predictors], train['variable'],
    #         eval_set=eval_set, eval_metric="auc",
    #         early_stopping_rounds=30)

    y_pred = pipeline.predict(valid[predictors])
    # pred = clf.predict_proba(valid[predictors])[:, 1]
    f1 = f1_score(valid['variable'], y_pred)
    print "F1:", f1

    return{'loss': 1-f1, 'status': STATUS_OK }


space ={
    'max_depth': hp.choice("x_max_depth", (2,  5,  7,  9, 12, 15)),
    'min_child_weight': hp.quniform('x_min_child', 1, 10, 2),
    'subsample': hp.uniform('x_subsample', 0.5, 1),
    'colsample_bytree': hp.uniform('x_csbtree', 0.5, 1),
    'colsample_bylevel': hp.uniform('x_csblevel', 0.5, 1),
    'gamma': hp.uniform('x_gamma', 0.0, 1),
    'scale_pos_weight': hp.choice('x_spweight', (0, 50, 150)),
    'lr': hp.loguniform('lr', -7, -1)
}


trials = Trials()
best = fmin(fn=objective,
            space=space,
            algo=tpe.suggest,
            max_evals=100,
            trials=trials)

import hyperopt
print hyperopt.space_eval(space, best)

# Load blind test data
# file_tgt = 'LMC_SC19_PSF_Pgood98__vast_lightcurve_statistics_normalized.log'
# file_tgt = os.path.join(data_dir, file_tgt)
# X, feature_names, df = load_data_tgt(file_tgt, names, names_to_delete,
#                                      delta)
