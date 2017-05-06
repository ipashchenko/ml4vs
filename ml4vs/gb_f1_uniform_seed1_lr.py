# -*- coding: utf-8 -*-
import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pprint
from sklearn.metrics import average_precision_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score, f1_score
sys.path.append('/home/ilya/xgboost/xgboost/python-package/')
import xgboost as xgb
from hyperopt import hp, fmin, tpe, STATUS_OK, Trials
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Imputer
from data_load import load_data, load_data_tgt


# Load data
data_dir = '/home/ilya/code/ml4vs/data/LMC_SC20__corrected_list_of_variables/raw_index_values'
file_1 = 'vast_lightcurve_statistics_variables_only.log'
file_0 = 'vast_lightcurve_statistics_constant_only.log'
file_0 = os.path.join(data_dir, file_0)
file_1 = os.path.join(data_dir, file_1)
names = ['Magnitude', 'clipped_sigma', 'meaningless_1', 'meaningless_2',
         'star_ID', 'weighted_sigma', 'skew', 'kurt', 'I', 'J', 'K', 'L',
         'Npts', 'MAD', 'lag1', 'RoMS', 'rCh2', 'Isgn', 'Vp2p', 'Jclp', 'Lclp',
         'Jtim', 'Ltim', 'CSSD', 'Ex', 'inv_eta', 'E_A', 'S_B', 'NXS', 'IQR']
# names_to_delete = ['Magnitude', 'meaningless_1', 'meaningless_2', 'star_ID',
#                    'Npts', 'CSSD']
# names_to_delete = ['meaningless_1', 'meaningless_2', 'star_ID',
#                    'Npts', 'CSSD', 'clipped_sigma', 'lag1', 'L', 'Lclp', 'Jclp',
#                    'MAD', 'Ltim']
names_to_delete = ['meaningless_1', 'meaningless_2', 'star_ID',
                   'Npts', 'CSSD', 'clipped_sigma', 'lag1', 'L', 'Lclp', 'Jclp',
                   'MAD', 'Ltim', 'NXS', 'E_A', 'Ex', 'K', 'rCh2', 'weighted_sigma',
                   'S_B', 'RoMS', 'Vp2p', 'Isgn', 'skew', 'Magnitude', 'IQR',
                   'inv_eta', 'Jtim', 'I']

X, y, df, features_names, delta = load_data([file_0, file_1], names, names_to_delete)
target = 'variable'
predictors = list(df)
predictors.remove(target)
dtrain = df

kfold = StratifiedKFold(n_splits=4, shuffle=True, random_state=1)


def xg_f1(y, t):
    t = t.get_label()
    # Binaryzing your output
    y_bin = [1. if y_cont > 0.5 else 0. for y_cont in y]
    return 'f1', 1-f1_score(t, y_bin)

# For given number of features:
# First fit using CV and find mean F1

clf = xgb.XGBClassifier(n_estimators=10000, learning_rate=0.086287,
                        max_depth=6,
                        min_child_weight=3,
                        subsample=0.400675,
                        colsample_bytree=0.72917,
                        colsample_bylevel=0.86287,
                        gamma=3.387,
                        max_delta_step=10,
                        scale_pos_weight=3.10297,
                        seed=1)

# Try using pipeline
estimators = list()
estimators.append(('imputer', Imputer(missing_values='NaN', strategy='median',
                                      axis=0, verbose=2)))
estimators.append(('clf', clf))
pipeline = Pipeline(estimators)

best_n = list()
CMs = list()
# aprs = list()
for train_indx, test_indx in kfold.split(dtrain[predictors].index,
                                         dtrain['variable']):
    train = dtrain.iloc[train_indx]
    valid = dtrain.iloc[test_indx]

    # X_test
    valid_ = valid[predictors]
    # X_train
    train_ = train[predictors]
    for name, transform in pipeline.steps[:-1]:
        transform.fit(train_)
        # X_test
        valid_ = transform.transform(valid_)
        # X_train
        train_ = transform.transform(train_)

    eval_set = [(train_, train['variable']),
                (valid_, valid['variable'])]

    # TODO: Try ES on default eval. metric or AUC!!!
    pipeline.fit(train[predictors], train['variable'],
                 clf__eval_set=eval_set, clf__eval_metric=xg_f1,
                 clf__early_stopping_rounds=50)
    pred = pipeline.predict(valid[predictors])
    CMs.append(confusion_matrix(y[test_indx], pred))
    best_n.append(clf.best_ntree_limit)

CM = np.sum(CMs, axis=0)

FN = CM[1][0]
TP = CM[1][1]
FP = CM[0][1]

f1 = 2. * TP / (2. * TP + FP + FN)
print("=== F1 : {} ===".format(f1))
print("Ns = {}".format(best_n))
print("Best n = {}".format(np.mean(best_n)))


# Second, fit all and find features importance

clf = xgb.XGBClassifier(n_estimators=int(1.25*np.mean(best_n)), learning_rate=0.086287,
                        max_depth=6,
                        min_child_weight=3,
                        subsample=0.400675,
                        colsample_bytree=0.72917,
                        colsample_bylevel=0.86287,
                        gamma=3.387,
                        max_delta_step=10,
                        scale_pos_weight=3.10297,
                        seed=1)
estimators = list()
estimators.append(('imputer', Imputer(missing_values='NaN', strategy='median',
                                      axis=0, verbose=2)))
estimators.append(('clf', clf))
pipeline = Pipeline(estimators)
pipeline.fit(dtrain[predictors], dtrain['variable'])

scores_dict = clf.booster().get_fscore()
scores_dict_ = dict()
for i, feature in enumerate(features_names):
    scores_dict_[feature] = scores_dict['f{}'.format(i)]

feat_imp = pd.Series(scores_dict_).sort_values(ascending=False)
feat_imp.plot(kind='bar', title='Feature Importances')
plt.ylabel('Feature Importance Score')
plt.show()

# Load blind test data
file_tgt = 'LMC_SC19_PSF_Pgood98__vast_lightcurve_statistics.log'
file_tgt = os.path.join(data_dir, file_tgt)
names = ['Magnitude', 'clipped_sigma', 'meaningless_1', 'meaningless_2',
         'star_ID', 'weighted_sigma', 'skew', 'kurt', 'I', 'J', 'K', 'L',
         'Npts', 'MAD', 'lag1', 'RoMS', 'rCh2', 'Isgn', 'Vp2p', 'Jclp', 'Lclp',
         'Jtim', 'Ltim', 'CSSD', 'Ex', 'inv_eta', 'E_A', 'S_B', 'NXS', 'IQR']
X_tgt, feature_names, df, df_orig = load_data_tgt(file_tgt, names,
                                                  names_to_delete, delta)
y_pred = pipeline.predict(X_tgt)
y_probs = pipeline.predict_proba(X_tgt)[:, 1]

idx = y_probs > 0.5
idx_ = y_probs < 0.5
gb_no = list(df_orig['star_ID'][idx_])
print("Found {} variables".format(np.count_nonzero(idx)))

with open('gb_results_final.txt', 'w') as fo:
    for line in list(df_orig['star_ID'][idx]):
        fo.write(line + '\n')

# Check F1
with open('clean_list_of_new_variables.txt', 'r') as fo:
    news = fo.readlines()
news = [line.strip().split(' ')[1] for line in news]
news = set(news)

with open('gb_results_final.txt', 'r') as fo:
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
