# -*- coding: utf-8 -*-
import os
import pprint
import numpy as np
import hyperopt
from sklearn import decomposition
from sklearn.base import TransformerMixin
from sklearn.feature_selection import SelectKBest, SelectPercentile
from sklearn.feature_selection import f_classif
from sklearn.metrics import confusion_matrix
from hyperopt import hp, fmin, tpe, STATUS_OK, Trials
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_predict
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import Imputer, StandardScaler, RobustScaler,\
    PolynomialFeatures
from sklearn.linear_model import LogisticRegression
from data_load import load_data


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
                   'MAD', 'Ltim', 'NXS', 'Vp2p', 'kurt']
# names_to_delete = ['meaningless_1', 'meaningless_2', 'star_ID',
#                    'Npts', 'CSSD']

X, y, df, features_names, delta = load_data([file_0, file_1], names,
                                            names_to_delete)
target = 'variable'
predictors = list(df)
predictors.remove(target)
dtrain = df

kfold = StratifiedKFold(n_splits=4, shuffle=True, random_state=1)


def log_axis(X_, names=None):
    X = X_.copy()
    tr_names = ['clipped_sigma', 'weighted_sigma', 'RoMS', 'rCh2', 'Vp2p',
                'Ex', 'inv_eta', 'S_B']
    for name in tr_names:
        try:
            i = names.index(name)
            X[:, i] = np.log(X[:, i])
        except ValueError:
            pass
    return X


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
    # clf = LogisticRegression(C=space['C'], class_weight={0: 1, 1: space['cw']},
    #                          random_state=1, penalty='l2')
    # estimators = list()
    # estimators.append(('imputer', Imputer(missing_values='NaN', strategy='median',
    #                                       axis=0, verbose=2)))
    # estimators.append(('func', FunctionTransformer(log_axis, kw_args={'names':
    #                                                                   predictors})))
    # estimators.append(('scaler', StandardScaler()))
    # estimators.append(('clf', clf))
    # pipeline = Pipeline(estimators)

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
                              ('pca', decomposition.PCA(n_components=space['n_pca'],
                                                        random_state=1))]))
        ])

    pipeline = Pipeline([
        ('features', combined_features),
        #('anova', MySelectPercentile(f_classif, space['perc'])),
        ('estimator', Pipeline([('scaler', StandardScaler()),
                                ('clf', LogisticRegression(C=space['C'],
                                                           class_weight={0: 1, 1: space['cw']},
                                                           random_state=1, penalty='l2'))]))
    ])

    y_preds = cross_val_predict(pipeline, X, y, cv=kfold, n_jobs=4)
    CMs = list()
    for train_idx, test_idx in kfold.split(X, y):
        CMs.append(confusion_matrix(y[test_idx], y_preds[test_idx]))
    CM = np.sum(CMs, axis=0)

    FN = CM[1][0]
    TP = CM[1][1]
    FP = CM[0][1]
    print("TP = {}".format(TP))
    print("FP = {}".format(FP))
    print("FN = {}".format(FN))
    f1 = 2. * TP / (2. * TP + FP + FN)
    print("F1: {}".format(f1))
    return{'loss': 1-f1, 'status': STATUS_OK}


space = {'C': hp.uniform('C', 4.00, 10.00),
         'cw': hp.uniform('cw', 1., 4.),
         # 'perc': hp.choice('perc', np.arange(10, 101, 1)),
         'n_pca': hp.choice('n_pca', np.arange(1, 16, dtype=int))}

trials = Trials()
best = fmin(fn=objective,
            space=space,
            algo=tpe.suggest,
            max_evals=200,
            trials=trials)

pprint.pprint(trials.best_trial)
print(hyperopt.space_eval(space, best))
best_pars = hyperopt.space_eval(space, best)
