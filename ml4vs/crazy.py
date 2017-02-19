# -*- coding: utf-8 -*-
import os
import pprint
import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn import decomposition
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import confusion_matrix, precision_recall_curve, f1_score
from hyperopt import hp, fmin, tpe, STATUS_OK, Trials
from sklearn.cross_validation import StratifiedKFold, cross_val_score
from sklearn.model_selection import cross_val_predict, StratifiedShuffleSplit
from sklearn.preprocessing import Imputer, StandardScaler, FunctionTransformer, \
    RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, confusion_matrix, classification_report
from sklearn.tree import DecisionTreeClassifier

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
                   'MAD', 'Ltim', 'NXS', 'E_A', 'Ex']

X, y, df, features_names, delta = load_data([file_0, file_1], names,
                                            names_to_delete)
target = 'variable'
predictors = list(df)
predictors.remove(target)
dtrain = df

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


class ModelTransformer(TransformerMixin):
    def __init__(self, model):
        self.model = model

    def fit(self, *args, **kwargs):
        self.model.fit(*args, **kwargs)
        return self

    def transform(self, X, **transform_params):
        return pd.DataFrame(self.model.predict(X))


pipeline = Pipeline([
    ('features', FeatureUnion([
        ('raw', Pipeline([('imputer', Imputer(missing_values='NaN',
                                              strategy='median', axis=0,
                                              verbose=2))])),
                          # ('logtransform', LogTransform()),
                          # ('scaler', StandardScaler())])),
        ('pca', Pipeline([('imputer', Imputer(missing_values='NaN',
                                              strategy='median', axis=0,
                                              verbose=2)),
                          ('logtransform', LogTransform()),
                          ('scaler', StandardScaler()),
                          ('pca', decomposition.PCA(n_components=5,
                                                    random_state=1))]))
        # ('logistic', Pipeline([('imputer', Imputer(missing_values='NaN',
        #                                       strategy='median', axis=0,
        #                                       verbose=2)),
        #                        ('logtransform', LogTransform()),
        #                        ('scaler', StandardScaler()),
        #                        ('lr', ModelTransformer(LogisticRegression(C=1.29,
        #                                                                   class_weight={0: 1,
        #                                                                                 1: 2},
        #                                                                   random_state=1,
        #                                                                   n_jobs=1)))]))
    ])),
    ('estimators', FeatureUnion([
        # ('lrc', ModelTransformer(LogisticRegression())),
        ('rfc', ModelTransformer(RandomForestClassifier())),
        ('gbc', ModelTransformer(GradientBoostingClassifier())),
        ('knn', ModelTransformer(KNeighborsClassifier())),
        ('dtc', ModelTransformer(DecisionTreeClassifier())),
        ('etc', ModelTransformer(ExtraTreesClassifier()))
    ])),
    ('estimator', LogisticRegression())
])


# sss = StratifiedShuffleSplit(n_splits=1, test_size=0.33)
# for train_index, test_index in sss.split(X, y):
#     X_train, X_test = X[train_index], X[test_index]
#     y_train, y_test = y[train_index], y[test_index]
# pipeline.fit(X_train, y_train)
# predicted = pipeline.predict(X_test)
# print(f1_score(y_test, predicted))
# print(classification_report(y_test, predicted))
# pprint.pprint(confusion_matrix(y_test, predicted))

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
