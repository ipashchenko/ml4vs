# -*- coding: utf-8 -*-
import os
import pprint
import numpy as np
from sklearn.cross_validation import StratifiedKFold
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
import hyperopt
from hyperopt import hp, fmin, tpe, STATUS_OK, Trials
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import StandardScaler
from data_load import load_data


# Load data
data_dir = '/home/ilya/github/ml4vs/data/Kr/raw_index_values'
file_1 = 'vast_lightcurve_statistics_variables_only.log'
file_0 = 'vast_lightcurve_statistics_constant_only.log'
file_0 = os.path.join(data_dir, file_0)
file_1 = os.path.join(data_dir, file_1)
names = ['Magnitude', 'clipped_sigma', 'meaningless_1', 'meaningless_2',
         'star_ID', 'weighted_sigma', 'skew', 'kurt', 'I', 'J', 'K', 'L',
         'Npts', 'MAD', 'lag1', 'RoMS', 'rCh2', 'Isgn', 'Vp2p', 'Jclp', 'Lclp',
         'Jtim', 'Ltim', 'CSSD', 'Ex', 'inv_eta', 'E_A', 'S_B', 'NXS', 'IQR']
# clipped_sigma - weighted_sigma
# weighted_sigma - clipped_sigma, MAD, IQR
# MAD - weighted_sigma, IQR
# Jtim - J
# Ltim - L
# rCh2 - I
names_to_delete = ['meaningless_1', 'meaningless_2', 'star_ID', 'Npts', 'CSSD',
                   'weighted_sigma', 'MAD', 'Jtim', 'Ltim', 'rCh2']

X, y, df, features_names, delta = load_data([file_0, file_1], names,
                                            names_to_delete)
target = 'variable'
predictors = list(df)
predictors.remove(target)
dtrain = df

kfold = StratifiedKFold(dtrain[target], n_folds=4, shuffle=True, random_state=1)


def objective(space):
    pprint.pprint(space)
    clf = SVC(C=space['C'], class_weight={0: 1, 1: space['cw']},
              probability=False, gamma=space['gamma'], random_state=1)
    estimators = list()
    estimators.append(('imputer', Imputer(missing_values='NaN',
                                          strategy='median',
                                          axis=0, verbose=2)))
    estimators.append(('scaler', StandardScaler()))
    estimators.append(('clf', clf))
    pipeline = Pipeline(estimators)

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
    print "F1 : ", f1

    return{'loss': 1-f1, 'status': STATUS_OK}

space = {'C': hp.loguniform('C', -6, 4),
         'gamma': hp.loguniform('gamma', -6, 4),
         'cw': hp.qloguniform('cw', 0, 6, 1)}


trials = Trials()
best = fmin(fn=objective,
            space=space,
            algo=tpe.suggest,
            max_evals=400,
            trials=trials)

pprint.pprint(hyperopt.space_eval(space, best))
best_pars = hyperopt.space_eval(space, best)