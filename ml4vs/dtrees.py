# -*- coding: utf-8 -*-
import os
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.cross_validation import cross_val_predict
from sklearn.cross_validation import StratifiedKFold, cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeClassifier
from hyperopt import hp, fmin, tpe, STATUS_OK, Trials
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Imputer
from data_load import load_data, load_data_tgt


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

X, y, df, features_names, delta = load_data([file_0, file_1], names, names_to_delete)
target = 'variable'
predictors = list(df)
predictors.remove(target)
dtrain = df

kfold = StratifiedKFold(y, n_folds=4, shuffle=True, random_state=1)


def objective(space):
    print space
    clf = DecisionTreeClassifier(max_depth=space['max_depth'],
                                 max_features=space['max_features'],
                                 criterion=space['criterion'],
                                 min_weight_fraction_leaf=space['mwfl'],
                                 class_weight={0: 1, 1: space['cw']})
    estimators = list()
    estimators.append(('imputer', Imputer(missing_values='NaN', strategy='median',
                                          axis=0, verbose=2)))
    estimators.append(('clf', clf))
    pipeline = Pipeline(estimators)

    y_preds = cross_val_predict(pipeline, X, y, cv=kfold, n_jobs=1)
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


space ={
    'max_depth': hp.choice("max_depth", (1, 2)),
    'max_features': hp.choice("max_features", range(1, 19, 1)),
    'criterion': hp.choice("criterion", ("gini", "entropy")),
    'mwfl': hp.loguniform("mwfl", -10, -6.9),
    'cw': hp.uniform('cw', 1, 5)
}


trials = Trials()
best = fmin(fn=objective,
            space=space,
            algo=tpe.suggest,
            max_evals=500,
            trials=trials)

import hyperopt
print hyperopt.space_eval(space, best)
best_pars = hyperopt.space_eval(space, best)


clf = DecisionTreeClassifier(max_depth=2,
                             max_features=15,
                             criterion='gini',
                             min_weight_fraction_leaf=0.0001706,
                             class_weight={0: 1, 1: 2.4876})
estimators = list()
estimators.append(('imputer', Imputer(missing_values='NaN', strategy='median',
                                      axis=0, verbose=2)))
estimators.append(('clf', clf))
pipeline = Pipeline(estimators)
pipeline.fit(X, y)
