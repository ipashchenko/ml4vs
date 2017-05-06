# -*- coding: utf-8 -*-
import os
import numpy as np
import hyperopt
from sklearn import decomposition
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import confusion_matrix, precision_recall_curve, f1_score
from hyperopt import hp, fmin, tpe, STATUS_OK, Trials
from sklearn.cross_validation import StratifiedKFold, cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Imputer, StandardScaler, FunctionTransformer,\
    RobustScaler
from sklearn.linear_model import LogisticRegression
from data_load import load_data, load_data_tgt
from sklearn.cluster import FeatureAgglomeration


data_dir = '/home/ilya/code/ml4vs/data/LMC_SC20__corrected_list_of_variables/raw_index_values'
file_1 = 'vast_lightcurve_statistics_variables_only.log'
file_0 = 'vast_lightcurve_statistics_constant_only.log'
file_0 = os.path.join(data_dir, file_0)
file_1 = os.path.join(data_dir, file_1)
names = ['Magnitude', 'clipped_sigma', 'meaningless_1', 'meaningless_2',
         'star_ID', 'weighted_sigma', 'skew', 'kurt', 'I', 'J', 'K', 'L',
         'Npts', 'MAD', 'lag1', 'RoMS', 'rCh2', 'Isgn', 'Vp2p', 'Jclp', 'Lclp',
         'Jtim', 'Ltim', 'CSSD', 'Ex', 'inv_eta', 'E_A', 'S_B', 'NXS', 'IQR']
names_to_delete = ['meaningless_1', 'meaningless_2', 'star_ID',
                   'Npts', 'CSSD', 'clipped_sigma', 'lag1', 'L', 'Lclp', 'Jclp',
                   'MAD', 'Ltim']
# names_to_delete = ['meaningless_1', 'meaningless_2', 'star_ID',
#                    'Npts', 'CSSD']

X, y, df, features_names, delta = load_data([file_0, file_1], names,
                                            names_to_delete)
# from sklearn.cluster import DBSCAN
# db = DBSCAN(eps=0.5, min_samples=5).fit(X)
# core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
# core_samples_mask[db.core_sample_indices_] = True
# labels = db.labels_
# X = np.hstack((X, labels[:, np.newaxis]))
target = 'variable'
predictors = list(df)
predictors.remove(target)
dtrain = df

kfold = StratifiedKFold(y, n_folds=4, shuffle=True, random_state=1)


def log_axis(X_, names=None):
    X = X_.copy()
    tr_names = ['clipped_sigma', 'weighted_sigma', 'RoMS', 'rCh2', 'Vp2p',
                'Ex', 'inv_eta', 'S_B']
    for name in tr_names:
        try:
            # print "Log-Transforming {}".format(name)
            i = names.index(name)
            X[:, i] = np.log(X[:, i])
        except ValueError:
            print "No {} in predictors".format(name)
            pass
    return X


def objective(space):
    print space
    clf = LogisticRegression(C=space['C'], class_weight={0: 1, 1: space['cw']},
                             random_state=1, max_iter=300, n_jobs=1,
                             tol=10.**(-5), penalty='l2')
    # pca = decomposition.PCA(n_components=space['n_pca'], random_state=1)
    # cal_clf = CalibratedClassifierCV(clf, method='sigmoid')
    # ward = FeatureAgglomeration(n_clusters=space['n_clusters'])
    estimators = list()
    estimators.append(('imputer', Imputer(missing_values='NaN', strategy='median',
                                          axis=0, verbose=2)))
    estimators.append(('func', FunctionTransformer(log_axis, kw_args={'names':
                                                                      predictors})))
    estimators.append(('scaler', StandardScaler()))
    # estimators.append(('pca', pca))
    # estimators.append(('ward', ward))
    estimators.append(('clf', clf))
    pipeline = Pipeline(estimators)

    # auc = np.mean(cross_val_score(pipeline, X, y, cv=kfold, scoring='roc_auc',
    #                               verbose=1, n_jobs=2))
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

    # print "AUC: {}".format(auc)
    print "F1: {}".format(f1)

    return{'loss': 1-f1, 'status': STATUS_OK}


space = {'C': hp.loguniform('C', 2.3, 4.3),
         'cw': hp.uniform('cw', 1, 5)}
         # 'n_clusters': hp.choice('n_clusters', np.arange(3, 22, 1, dtype=int))}
         # 'n_pca': hp.choice('n_pca', np.arange(5, 18, 1, dtype=int))}


trials = Trials()
best = fmin(fn=objective,
            space=space,
            algo=tpe.suggest,
            max_evals=300,
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
# clf = LogisticRegression(C=best_pars['C'], class_weight={0: 1,
#                                                          1: best_pars['cw']},
#                          random_state=1, max_iter=300, n_jobs=1, tol=10.**(-5))
# # cal_clf = CalibratedClassifierCV(clf, method='sigmoid')
# pca = decomposition.PCA(n_components=best_pars['n_pca'], random_state=1)
# # ward = FeatureAgglomeration(n_clusters=best_pars['n_clusters'])
# estimators = list()
# estimators.append(('imputer', Imputer(missing_values='NaN', strategy='median',
#                                       axis=0, verbose=2)))
# # estimators.append(('func', FunctionTransformer(log_axis, kw_args={'names':
# #                                                                   predictors})))
# estimators.append(('scaler', StandardScaler()))
# # estimators.append(('ward', ward))
# estimators.append(('pca', pca))
# estimators.append(('clf', clf))
# pipeline = Pipeline(estimators)
# pipeline.fit(X, y)
# # y_probs = pipeline.predict(X_tgt)
# from scipy.optimize import fmin
# from sklearn.metrics import f1_score
# # thresh = fmin(lambda x: 1-f1_score(y, y_probs > x), 0.75, xtol=10**(-6),
# #               ftol=10**(-6), maxiter=10**5)
# # thresh = 1. - np.array(np.count_nonzero(y_probs), dtype=float) / np.count_nonzero(1. - y_probs)
# thresh = 0.5
#
# # Predict classes on new data
# y_probs = pipeline.predict_proba(X_tgt)[:, 1]
# idx = y_probs > thresh
# idx_ = y_probs < thresh
# gb_no = list(df_orig['star_ID'][idx_])
# print("Found {} variables".format(np.count_nonzero(idx)))
#
# with open('lr_results.txt', 'w') as fo:
#     for line in list(df_orig['star_ID'][idx]):
#         fo.write(line + '\n')
#
# # Check F1
# with open('clean_list_of_new_variables.txt', 'r') as fo:
#     news = fo.readlines()
# news = [line.strip().split(' ')[1] for line in news]
# news = set(news)
#
# with open('lr_results.txt', 'r') as fo:
#     gb = fo.readlines()
# gb = [line.strip().split('_')[4].split('.')[0] for line in gb]
# gb = set(gb)
#
# print "Among new vars found {}".format(len(news.intersection(gb)))
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
# gb_no = set([line.strip().split('_')[4].split('.')[0] for line in gb_no])
#
# found_bad = '181193' in gb
# print "Found known variable : ", found_bad
#
# FN = len(gb_no.intersection(all_vars))
# TP = len(all_vars.intersection(gb))
# TN = len(gb_no) - FN
# FP = len(gb) - TP
# recall = float(TP) / (TP + FN)
# precision = float(TP) / (TP + FP)
# F1 = 2 * precision * recall / (precision + recall)
# print "precision: {}".format(precision)
# print "recall: {}".format(recall)
# print "F1: {}".format(F1)
# print "TN={}, FP={}".format(TN, FP)
# print "FN={}, TP={}".format(FN, TP)
