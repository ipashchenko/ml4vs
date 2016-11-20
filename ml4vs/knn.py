# -*- coding: utf-8 -*-
import os
import hyperopt
import numpy as np
from sklearn.cross_validation import StratifiedKFold
from sklearn.model_selection import cross_val_predict, cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from hyperopt import hp, fmin, tpe, STATUS_OK, Trials
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import StandardScaler
from data_load import load_data, load_data_tgt


data_dir = '/home/ilya/code/ml4vs/data/LMC_SC20__corrected_list_of_variables/normalized'
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

X, y, df, features_names, delta = load_data([file_0, file_1], names,
                                            names_to_delete)
target = 'variable'
predictors = list(df)
predictors.remove(target)
dtrain = df

kfold = StratifiedKFold(dtrain[target], n_folds=4, shuffle=True, random_state=1)


def objective(space):
    print "Using n, w, ls : {}, {}, {}".format(space['n_neighbors'],
                                               space['weights'],
                                               space['leaf_size'])
    clf = KNeighborsClassifier(n_neighbors=space['n_neighbors'],
                               weights=space['weights'],
                               leaf_size=space['leaf_size'], n_jobs=4)
    estimators = list()
    estimators.append(('imputer', Imputer(missing_values='NaN', strategy='median',
                                          axis=0, verbose=2)))
    estimators.append(('scaler', StandardScaler()))
    estimators.append(('clf', clf))
    pipeline = Pipeline(estimators)

    # f1 = np.mean(cross_val_score(pipeline, X, y, cv=kfold, scoring='f1',
    #                              verbose=1, n_jobs=4))
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
    print "SCORE:", f1

    return{'loss': 1-f1, 'status': STATUS_OK}


space = {'n_neighbors': hp.choice("n_neighbors", (8, 9, 10, 11, 12, 13, 14, 15)),
         'leaf_size': hp.choice("leaf_size", (30,)),
         'weights': hp.choice('weights', ('distance',))}


trials = Trials()
best = fmin(fn=objective,
            space=space,
            algo=tpe.suggest,
            max_evals=20,
            trials=trials)

print hyperopt.space_eval(space, best)
best_pars = hyperopt.space_eval(space, best)

# Load blind test data
file_tgt = 'LMC_SC19_PSF_Pgood98__vast_lightcurve_statistics_normalized.log'
file_tgt = os.path.join(data_dir, file_tgt)
X_tgt, feature_names, df, df_orig = load_data_tgt(file_tgt, names, names_to_delete,
                                                  delta)

# Fit model on all training data
clf = KNeighborsClassifier(n_neighbors=best_pars['n_neighbors'],
                           weights=best_pars['weights'],
                           leaf_size=best_pars['leaf_size'], n_jobs=4)
estimators = list()
estimators.append(('imputer', Imputer(missing_values='NaN', strategy='median',
                                      axis=0, verbose=2)))
estimators.append(('scaler', StandardScaler()))
estimators.append(('clf', clf))
pipeline = Pipeline(estimators)
pipeline.fit(X, y)

# Predict clases on new data
y_probs = pipeline.predict_proba(X_tgt)[:, 1]
idx = y_probs > 0.5
idx_ = y_probs < 0.5
knn_no = list(df_orig['star_ID'][idx_])
print("Found {} variables".format(np.count_nonzero(idx)))

with open('knn_results.txt', 'w') as fo:
    for line in list(df_orig['star_ID'][idx]):
        fo.write(line + '\n')

# Check F1
with open('clean_list_of_new_variables.txt', 'r') as fo:
    news = fo.readlines()
news = [line.strip().split(' ')[1] for line in news]
news = set(news)

with open('knn_results.txt', 'r') as fo:
    knn = fo.readlines()
knn = [line.strip().split('_')[4].split('.')[0] for line in knn]
knn = set(knn)

print "Among new vars found {}".format(len(news.intersection(knn)))

with open('candidates_50perc_threshold.txt', 'r') as fo:
    c50 = fo.readlines()
c50 = [line.strip("\", ', \", \n, }, {") for line in c50]

with open('variables_not_in_catalogs.txt', 'r') as fo:
    not_in_cat = fo.readlines()
nic = [line.strip().split(' ')[1] for line in not_in_cat]

# Catalogue variables
cat_vars = set(c50).difference(set(nic))
# Non-catalogue variable
noncat_vars = set([line.strip().split(' ')[1] for line in not_in_cat if 'CST'
                   not in line])

# All variables
all_vars = news.union(cat_vars).union(noncat_vars)
knn_no = set([line.strip().split('_')[4].split('.')[0] for line in knn_no])

found_bad = '181193' in knn
print "Found known variable : ", found_bad

FN = len(knn_no.intersection(all_vars))
TP = len(all_vars.intersection(knn))
TN = len(knn_no) - FN
FP = len(knn) - TP
recall = float(TP) / (TP + FN)
precision = float(TP) / (TP + FP)
F1 = 2 * precision * recall / (precision + recall)
print "precision: {}".format(precision)
print "recall: {}".format(recall)
print "F1: {}".format(F1)
print "TN={}, FP={}".format(TN, FP)
print "FN={}, TP={}".format(FN, TP)
