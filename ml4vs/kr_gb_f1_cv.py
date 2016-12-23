# -*- coding: utf-8 -*-
import sys
sys.path.append('/home/ilya/github/xgboost/python-package/')
import xgboost as xgb
import pprint
import os
import numpy as np
from sklearn.metrics import f1_score
from hyperopt import hp, fmin, tpe, STATUS_OK, Trials
from sklearn.cross_validation import StratifiedKFold
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
# names_to_delete = ['Magnitude', 'meaningless_1', 'meaningless_2', 'star_ID',
#                    'Npts', 'CSSD']

X, y, df, features_names, delta = load_data([file_0, file_1], names, names_to_delete)
target = 'variable'
predictors = list(df)
predictors.remove(target)
dtrain = df

# from imblearn.over_sampling import SMOTE
# ratio = 0.05
# smote = SMOTE(ratio=ratio, kind='regular')
# X, y = smote.fit_sample(X, y)


kfold = StratifiedKFold(dtrain[target], n_folds=4, shuffle=True,
                        random_state=1)


def xg_f1(y, t):
    t = t.get_label()
    # Binaryzing your output
    y_bin = [1. if y_cont > 0.5 else 0. for y_cont in y]
    return 'f1', 1-f1_score(t, y_bin)


def objective(space):
    pprint.pprint(space)
    clf = xgb.XGBClassifier(n_estimators=10000, learning_rate=space['lr'],
                            max_depth=space['max_depth'],
                            min_child_weight=space['min_child_weight'],
                            subsample=space['subsample'],
                            colsample_bytree=space['colsample_bytree'],
                            colsample_bylevel=space['colsample_bylevel'],
                            gamma=space['gamma'],
                            scale_pos_weight=space['scale_pos_weight'],
                            max_delta_step=space['mds'],
                            seed=1)
    xgb_param = clf.get_xgb_params()
    xgtrain = xgb.DMatrix(dtrain[predictors].values,
                          label=dtrain[target].values)
    cvresult = xgb.cv(xgb_param, xgtrain,
                      num_boost_round=clf.get_params()['n_estimators'],
                      folds=kfold, feval=xg_f1,
                      early_stopping_rounds=20, verbose_eval=True,
                      as_pandas=False, seed=1)

    print "F1:", 1-cvresult['test-f1-mean'][-1]

    return{'loss': cvresult['test-f1-mean'][-1], 'status': STATUS_OK,
           'attachments': {'best_n': str(len(cvresult['test-f1-mean']))}}

space = {
    'max_depth': hp.choice("x_max_depth", np.arange(2, 21, 1, dtype=int)),
    'min_child_weight': hp.qloguniform('x_min_child', 0, 5, 1),
    'subsample': hp.quniform('x_subsample', 0.25, 1, 0.0125),
    'colsample_bytree': hp.quniform('x_csbtree', 0.25, 1, 0.0125),
    'colsample_bylevel': hp.quniform('x_csblevel', 0.25, 1, 0.0125),
    'gamma': hp.uniform('x_gamma', 0.0, 20),
    'scale_pos_weight': hp.qloguniform('x_spweight', 0, 6, 1),
    'mds': hp.choice('mds', np.arange(0, 20, dtype=int)),
    'lr': hp.loguniform('lr', -4.7, -1.0)
}


trials = Trials()
best = fmin(fn=objective,
            space=space,
            algo=tpe.suggest,
            max_evals=500,
            trials=trials)

import hyperopt
pprint.pprint(hyperopt.space_eval(space, best))

best_pars = hyperopt.space_eval(space, best)
best_n = trials.attachments['ATTACH::{}::best_n'.format(trials.best_trial['tid'])]
best_n = int(best_n)
print best_n
