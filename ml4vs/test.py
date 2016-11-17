# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
import xgboost as xgb
from hyperopt import hp, fmin, tpe, STATUS_OK, Trials
# from sklearn.model_selection import StratifiedKFold
from sklearn.cross_validation import StratifiedKFold
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
                   'Npts', 'CSSD', 'Lclp', 'J', 'Jclp', 'rCh2', 'NXS', 'Ex',
                   'inv_eta', 'weighted_sigma', 'Jtim',
                   'Vp2p', 'E_A', 'kurt', 'K', 'Isgn', 'IQR']

X, y, df, features_names, delta = load_data([file_0, file_1], names, names_to_delete)
target = 'variable'
predictors = list(df)
predictors.remove(target)
dtrain = df

# Was 7
# 5000 - was 77
# sss = StratifiedShuffleSplit(n_splits=1, test_size=0.25, random_state=777)
# for train_indx, test_indx in sss.split(dtrain[predictors].index, dtrain['variable']):
#     print train_indx, test_indx
#     train = dtrain.iloc[train_indx]
#     valid = dtrain.iloc[test_indx]

# kfold = StratifiedKFold(n_splits=4, shuffle=True, random_state=123)
kfold = StratifiedKFold(dtrain[target], n_folds=4, shuffle=True,
                        random_state=123)


def objective(space):
    clf = xgb.XGBClassifier(n_estimators=10000, learning_rate=0.025,
                            max_depth=space['max_depth'],
                            min_child_weight=space['min_child_weight'],
                            subsample=space['subsample'],
                            colsample_bytree=space['colsample_bytree'],
                            colsample_bylevel=space['colsample_bylevel'],
                            gamma=space['gamma'],
                            scale_pos_weight=200., seed=123)
    # scale_pos_weight=space['scale_pos_weight'])
    xgb_param = clf.get_xgb_params()
    xgtrain = xgb.DMatrix(dtrain[predictors].values, label=dtrain[target].values)
    cvresult = xgb.cv(xgb_param, xgtrain,
                      num_boost_round=clf.get_params()['n_estimators'],
                      folds=kfold, metrics='auc',
                      early_stopping_rounds=50, verbose_eval=True,
                      as_pandas=False, seed=123)

    print "SCORE:", cvresult['test-auc-mean'][-1]

    return{'loss': 1-cvresult['test-auc-mean'][-1], 'status': STATUS_OK ,
           'attachments': {'best_n': str(len(cvresult['test-auc-mean']))}}


space ={
    'max_depth': hp.choice("x_max_depth", np.arange(5, 12, 1, dtype=int)),
    'min_child_weight': hp.quniform('x_min_child', 1, 40, 2),
    'subsample': hp.quniform('x_subsample', 0.5, 1, 0.05),
    'colsample_bytree': hp.quniform('x_csbtree', 0.5, 1, 0.05),
    'colsample_bylevel': hp.quniform('x_csblevel', 0.5, 1, 0.05),
    'gamma': hp.quniform('x_gamma', 0.0, 5, 0.1)
    # 'scale_pos_weight': hp.quniform('x_spweight', 0, 300, 50),
    # 'lr': hp.quniform('lr', 0.001, 0.5, 0.025)
    # 'lr': hp.loguniform('lr', -7, -1)
}


trials = Trials()
best = fmin(fn=objective,
            space=space,
            algo=tpe.suggest,
            max_evals=75,
            trials=trials)

import hyperopt
print hyperopt.space_eval(space, best)

best_pars = hyperopt.space_eval(space, best)
best_n = trials.attachments['ATTACH::{}::best_n'.format(trials.best_trial['tid'])]
best_n = int(best_n)

clf = xgb.XGBClassifier(n_estimators=16000,
                        learning_rate=0.01,
                        max_depth=best_pars['max_depth'],
                        min_child_weight=best_pars['min_child_weight'],
                        subsample=best_pars['subsample'],
                        colsample_bytree=best_pars['colsample_bytree'],
                        colsample_bylevel=best_pars['colsample_bylevel'],
                        gamma=best_pars['gamma'],
                        scale_pos_weight=200., seed=123)

estimators = list()
estimators.append(('imputer', Imputer(missing_values='NaN', strategy='median',
                                      axis=0, verbose=2)))
estimators.append(('clf', clf))
pipeline = Pipeline(estimators)

# Fit classifier with best hyperparameters on whole data set
pipeline.fit(dtrain[predictors], dtrain['variable'])

# Load blind test data
file_tgt = 'LMC_SC19_PSF_Pgood98__vast_lightcurve_statistics_normalized.log'
file_tgt = os.path.join(data_dir, file_tgt)
X_tgt, feature_names, df, df_orig = load_data_tgt(file_tgt, names, names_to_delete,
                                                  delta)

y_probs = pipeline.predict_proba(df[predictors])[:, 1]
idx = y_probs > 0.5
idx_ = y_probs < 0.5
gb_no = list(df_orig['star_ID'][idx_])
print("Found {} variables".format(np.count_nonzero(idx)))

with open('gb_results.txt', 'w') as fo:
    for line in list(df_orig['star_ID'][idx]):
        fo.write(line + '\n')

# Check F1
with open('clean_list_of_new_variables.txt', 'r') as fo:
    news = fo.readlines()
news = [line.strip().split(' ')[1] for line in news]
news = set(news)

with open('gb_results.txt', 'r') as fo:
    gb = fo.readlines()
gb = [line.strip().split('_')[4].split('.')[0] for line in gb]
gb = set(gb)

print "Among new vars found {}".format(len(news.intersection(gb)))

with open('candidates_50perc_threshold.txt', 'r') as fo:
    c50 = fo.readlines()
c50 = [line.strip("\", ', \", \n, }, {") for line in c50]

with open('variables_not_in_catalogs.txt', 'r') as fo:
    not_in_cat = fo.readlines()
nic = [line.strip().split(' ')[1] for line in not_in_cat]

# Catalogue variables
cat_vars = set(c50).difference(set(nic))
# Non-catalogue variable
noncat_vars = set([line.strip().split(' ')[1] for line in not_in_cat if 'CST' not in line])

# All variables
all_vars = news.union(cat_vars).union(noncat_vars)
gb_no = set([line.strip().split('_')[4].split('.')[0] for line in gb_no])

found_bad = '181193' in gb
print "Found known variable : ", found_bad

FN = len(gb_no.intersection(all_vars))
TP = len(all_vars.intersection(gb))
TN = len(gb_no) - FN
FP = len(gb) - TP
recall = float(TP) / (TP + FN)
precision = float(TP) / (TP + FP)
F1 = 2 * precision * recall / (precision + recall)
print "precision: {}".format(precision)
print "recall: {}".format(recall)
print "F1: {}".format(F1)
print "TN={}, FP={}".format(TN, FP)
print "FN={}, TP={}".format(FN, TP)
