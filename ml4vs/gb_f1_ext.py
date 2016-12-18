# -*- coding: utf-8 -*-
import sys
import os
import numpy as np
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
names_to_delete = ['meaningless_1', 'meaningless_2', 'star_ID',
                   'Npts', 'CSSD', 'clipped_sigma', 'lag1', 'L', 'Lclp', 'Jclp',
                   'MAD', 'Ltim']

X, y, df, features_names, delta = load_data([file_0, file_1], names, names_to_delete)
target = 'variable'
predictors = list(df)
predictors.remove(target)
dtrain = df

kfold = StratifiedKFold(n_splits=4, shuffle=True, random_state=123)


def xg_f1(y, t):
    t = t.get_label()
    # Binaryzing your output
    y_bin = [1. if y_cont > 0.5 else 0. for y_cont in y]
    return 'f1', 1-f1_score(t, y_bin)


def objective(space):
    print "Using LR = {}".format(space['lr'])
    clf = xgb.XGBClassifier(n_estimators=10000, learning_rate=space['lr'],
                            max_depth=7,
                            min_child_weight=2,
                            subsample=0.775,
                            colsample_bytree=1.,
                            colsample_bylevel=0.625,
                            gamma=0.525,
                            scale_pos_weight=2)
                            # scale_pos_weight=space['scale_pos_weight'])

    # Try using pipeline
    estimators = list()
    estimators.append(('imputer', Imputer(missing_values='NaN', strategy='median',
                                          axis=0, verbose=2)))
    # estimators.append(('scaler', StandardScaler()))
    estimators.append(('clf', clf))
    pipeline = Pipeline(estimators)

    best_n = ""
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
                     # clf__eval_set=eval_set, clf__eval_metric="map",
                     clf__eval_set=eval_set, clf__eval_metric=xg_f1,
                     clf__early_stopping_rounds=50)

        pred = pipeline.predict(valid[predictors])
        # pred = pipeline.predict_proba(valid[predictors])[:, 1]
        # aps = average_precision_score(valid['variable'], pred)
        # aprs.append(aps)
        CMs.append(confusion_matrix(y[test_indx], pred))
        best_n = best_n + " " + str(clf.best_ntree_limit)

    CM = np.sum(CMs, axis=0)

    FN = CM[1][0]
    TP = CM[1][1]
    FP = CM[0][1]
    print "TP = {}".format(TP)
    print "FP = {}".format(FP)
    print "FN = {}".format(FN)

    f1 = 2. * TP / (2. * TP + FP + FN)
    # APR = np.mean(aprs)
    print "=== F1 : {} ===".format(f1)

    return{'loss': 1-f1, 'status': STATUS_OK ,
           'attachments': {'best_n': best_n}}


space ={
    'lr': hp.quniform('lr', 0.0001, 0.5, 0.0025)
}


trials = Trials()
best = fmin(fn=objective,
            space=space,
            algo=tpe.suggest,
            max_evals=100,
            trials=trials)

import hyperopt
print hyperopt.space_eval(space, best)

