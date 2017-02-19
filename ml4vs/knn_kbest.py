# -*- coding: utf-8 -*-
import os
import hyperopt
import pprint
import numpy as np
from sklearn import decomposition
from sklearn.base import TransformerMixin
from sklearn.cross_validation import StratifiedKFold
from sklearn.feature_selection import SelectPercentile
from sklearn.model_selection import cross_val_predict, cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from hyperopt import hp, fmin, tpe, STATUS_OK, Trials
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif, f_regression
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
# names_to_delete = ['meaningless_1', 'meaningless_2', 'star_ID',
#                    'Npts', 'CSSD']
names_to_delete = ['meaningless_1', 'meaningless_2', 'star_ID',
                   'Npts', 'CSSD', 'clipped_sigma', 'lag1', 'L', 'Lclp', 'Jclp',
                   'MAD', 'Ltim', 'NXS', 'Vp2p', 'skew', 'kurt']

X, y, df, features_names, delta = load_data([file_0, file_1], names,
                                            names_to_delete)
kfold = StratifiedKFold(y, n_folds=4, shuffle=True, random_state=1)


class LogTransform(TransformerMixin):
    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X, **transform_params):
        X_ = X.copy()
        tr_names = ['clipped_sigma', 'weighted_sigma', 'RoMS', 'rCh2', 'Vp2p',
                    'Ex', 'inv_eta', 'S_B']
        for name in tr_names:
            try:
                i = features_names.index(name)
                X_[:, i] = np.log(X_[:, i])
            except ValueError:
                pass
        return X_


class MySelectKBest(SelectKBest):
    def _check_params(self, X, y):
        if not (self.k == "all" or 0 <= self.k <= X.shape[1]):
            pass


class MySelectPercentile(SelectPercentile):
    def _check_params(self, X, y):
        if not 0 <= self.percentile <= 100:
            pass

def objective(space):
    pprint.pprint(space)
    combined_features = FeatureUnion([
        # ('raw', Pipeline([('imputer', Imputer(missing_values='NaN',
        #                                       strategy='median', axis=0,
        #                                       verbose=2)),
        #                   ('logtransform', LogTransform())])),
        ('poly', Pipeline([('logtransform', LogTransform()),
                           ('scaler', StandardScaler()),
                           ('polyf', PolynomialFeatures())])),
        ('pca', Pipeline([('imputer', Imputer(missing_values='NaN',
                                              strategy='median', axis=0,
                                              verbose=2)),
                          ('logtransform', LogTransform()),
                          ('scaler', StandardScaler()),
                          ('pca', decomposition.PCA(n_components=10,
                                                    random_state=1))]))
    ])

    pipeline = Pipeline([
        ('features', combined_features),
        ('anova', MySelectPercentile(f_classif, space['perc'])),
        ('estimator', Pipeline([('scaler', StandardScaler()),
                                ('clf', KNeighborsClassifier(n_neighbors=space['n_neighbors'],
                                                             weights=space['weights'], n_jobs=1,
                                                             metric=space['metric']))]))
    ])

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


space = {'n_neighbors': hp.qloguniform("n_neighbors", 0, 6.55, 1),
         'perc': hp.choice('k', np.arange(1, 101, 1, dtype=int)),
         'metric': hp.choice('metric', ("euclidean", "manhattan", "chebyshev")),
         }


trials = Trials()
best = fmin(fn=objective,
            space=space,
            algo=tpe.suggest,
            max_evals=200,
            trials=trials)

pprint.pprint(hyperopt.space_eval(space, best))
best_pars = hyperopt.space_eval(space, best)
