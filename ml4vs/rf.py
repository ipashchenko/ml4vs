# -*- coding: utf-8 -*-
import os
import numpy as np
import hyperopt
from sklearn import decomposition
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score, f1_score
from hyperopt import hp, fmin, tpe, STATUS_OK, Trials
from sklearn.cross_validation import StratifiedKFold
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from data_load import load_data, load_data_tgt
from sklearn.calibration import CalibratedClassifierCV


data_dir = '/home/ilya/code/ml4vs/data/LMC_SC20__corrected_list_of_variables/raw_index_values'
file_1 = 'vast_lightcurve_statistics_variables_only.log'
file_0 = 'vast_lightcurve_statistics_constant_only.log'
file_0 = os.path.join(data_dir, file_0)
file_1 = os.path.join(data_dir, file_1)
names = ['Magnitude', 'clipped_sigma', 'meaningless_1', 'meaningless_2',
         'star_ID', 'weighted_sigma', 'skew', 'kurt', 'I', 'J', 'K', 'L',
         'Npts', 'MAD', 'lag1', 'RoMS', 'rCh2', 'Isgn', 'Vp2p', 'Jclp', 'Lclp',
         'Jtim', 'Ltim', 'CSSD', 'Ex', 'inv_eta', 'E_A', 'S_B', 'NXS', 'IQR']
# names_to_delete = ['meaningless_1', 'meaningless_2', 'star_ID',
#                    'Npts', 'CSSD']
names_to_delete = ['meaningless_1', 'meaningless_2', 'star_ID',
                   'Npts', 'CSSD', 'clipped_sigma', 'lag1', 'L', 'Lclp', 'Jclp',
                   'MAD', 'Ltim']

X, y, df, features_names, delta = load_data([file_0, file_1], names,
                                            names_to_delete)

from sklearn.cluster import DBSCAN
db = DBSCAN(eps=0.5, min_samples=5).fit(X)
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
labels = db.labels_
X = np.hstack((X, labels[:, np.newaxis]))

target = 'variable'
predictors = list(df)
predictors.remove(target)
dtrain = df

kfold = StratifiedKFold(dtrain[target], n_folds=4, shuffle=True, random_state=1)


def objective(space):
    clf = RandomForestClassifier(n_estimators=space['n_estimators'],
                                 max_depth=space['max_depth'],
                                 max_features=space['max_features'],
                                 min_samples_split=space['mss'],
                                 min_samples_leaf=space['msl'],
                                 class_weight={0: 1, 1: space['cw']},
                                 verbose=1, random_state=1, n_jobs=4)
    estimators = list()
    estimators.append(('imputer', Imputer(missing_values='NaN', strategy='median',
                                          axis=0, verbose=2)))
    # estimators.append(('scaler', StandardScaler()))
    estimators.append(('clf', clf))
    pipeline = Pipeline(estimators)

    # auc = np.mean(cross_val_score(pipeline, X, y, cv=kfold, scoring='roc_auc',
    #                               verbose=1, n_jobs=4))

    y_preds = cross_val_predict(pipeline, X, y, cv=kfold, n_jobs=4)
    CMs = list()
    for train_idx, test_idx in kfold:
        CMs.append(confusion_matrix(y[test_idx], y_preds[test_idx]))
    CM = np.sum(CMs, axis=0)

    FN = CM[1][0]
    TP = CM[1][1]
    FP = CM[0][1]
    print "TP = {}".format(TP)
    print "FP = {}".format(FP)
    print "FN = {}".format(FN)

    f1 = 2. * TP / (2. * TP + FP + FN)

    return{'loss': 1-f1, 'status': STATUS_OK}


space = {'n_estimators': hp.choice('n_estimators', np.arange(1300, 1600, 100,
                                                             dtype=int)),
         'max_depth': hp.choice('max_depth', np.arange(14, 19, dtype=int)),
         'max_features': hp.choice('max_features', np.arange(4, 7, dtype=int)),
         'mss': hp.choice('mss', np.arange(14, 19, 1, dtype=int)),
         'cw': hp.uniform('cw', 20, 40),
         'msl': hp.choice('msl', np.arange(1, 6, dtype=int))}


trials = Trials()
best = fmin(fn=objective,
            space=space,
            algo=tpe.suggest,
            max_evals=100,
            trials=trials)

print hyperopt.space_eval(space, best)
best_pars = hyperopt.space_eval(space, best)

# # Load blind test data
# file_tgt = 'LMC_SC19_PSF_Pgood98__vast_lightcurve_statistics.log'
# file_tgt = os.path.join(data_dir, file_tgt)
# X_tgt, feature_names, df, df_orig = load_data_tgt(file_tgt, names, names_to_delete,
#                                                   delta)
#
# # Fit model on all training data
# clf = RandomForestClassifier(n_estimators=best_pars['n_estimators'],
#                              max_depth=best_pars['max_depth'],
#                              max_features=best_pars['max_features'],
#                              min_samples_split=best_pars['mss'],
#                              min_samples_leaf=best_pars['msl'],
#                              class_weight='balanced_subsample',
#                              verbose=1, random_state=1)
# cal_clf = CalibratedClassifierCV(clf, method='sigmoid')
# estimators = list()
# estimators.append(('imputer', Imputer(missing_values='NaN', strategy='median',
#                                       axis=0, verbose=2)))
# # estimators.append(('scaler', StandardScaler()))
# estimators.append(('clf', clf))
# pipeline = Pipeline(estimators)
# pipeline.fit(X, y)
#
# # Predict classes on new data
# y_probs = pipeline.predict_proba(X_tgt)[:, 1]
# idx = y_probs > thresh
# idx_ = y_probs < thresh
# rf_no = list(df_orig['star_ID'][idx_])
# print("Found {} variables".format(np.count_nonzero(idx)))
#
# with open('rf_results.txt', 'w') as fo:
#     for line in list(df_orig['star_ID'][idx]):
#         fo.write(line + '\n')
#
# # Check F1
# with open('clean_list_of_new_variables.txt', 'r') as fo:
#     news = fo.readlines()
# news = [line.strip().split(' ')[1] for line in news]
# news = set(news)
#
# with open('rf_results.txt', 'r') as fo:
#     rf = fo.readlines()
# rf = [line.strip().split('_')[4].split('.')[0] for line in rf]
# rf = set(rf)
#
# print "Among new vars found {}".format(len(news.intersection(rf)))
#
# with open('candidates_50perc_threshold.txt', 'r') as fo:
#     c50 = fo.readlines()
# c50 = [line.strip("\", ', \", \n, }, {") for line in c50]
#
# with open('variables_not_in_catalogs.txt', 'r') as fo:
#     not_in_cat = fo.readlines()
# nic = [line.strip().split(' ')[1] for line in not_in_cat]
#
# # Catalogue variables
# cat_vars = set(c50).difference(set(nic))
# # Non-catalogue variable
# noncat_vars = set([line.strip().split(' ')[1] for line in not_in_cat if 'CST'
#                    not in line])
#
# # All variables
# all_vars = news.union(cat_vars).union(noncat_vars)
# rf_no = set([line.strip().split('_')[4].split('.')[0] for line in rf_no])
#
# found_bad = '181193' in rf
# print "Found known variable : ", found_bad
#
# FN = len(rf_no.intersection(all_vars))
# TP = len(all_vars.intersection(rf))
# TN = len(rf_no) - FN
# FP = len(rf) - TP
# recall = float(TP) / (TP + FN)
# precision = float(TP) / (TP + FP)
# F1 = 2 * precision * recall / (precision + recall)
# print "precision: {}".format(precision)
# print "recall: {}".format(recall)
# print "F1: {}".format(F1)
# print "TN={}, FP={}".format(TN, FP)
# print "FN={}, TP={}".format(FN, TP)
