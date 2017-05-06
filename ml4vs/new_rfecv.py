# -*- coding: utf-8 -*-
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn import decomposition
from sklearn.base import TransformerMixin
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import Imputer, StandardScaler, RobustScaler, PolynomialFeatures
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
                   'MAD', 'Ltim']

X, y, df, features_names, delta = load_data([file_0, file_1], names,
                                            names_to_delete)
target = 'variable'
predictors = list(df)
predictors.remove(target)
dtrain = df

kfold = StratifiedKFold(n_splits=4, shuffle=True, random_state=1)


class myPipe(Pipeline):
    def fit(self, X,y):
        """Calls last elements .coef_ method.
        Based on the sourcecode for decision_function(X).
        Link: https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/pipeline.py
        ----------
        """
        super(myPipe, self).fit(X,y)
        self.coef_= self.steps[-1][-1].coef_
        return


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

clf = LogisticRegression(C=10.5, class_weight={0: 1, 1: 1.585}, random_state=1,
                         penalty='l2')
estimators = list()
estimators.append(('imputer', Imputer(missing_values='NaN', strategy='median',
                                      axis=0, verbose=2)))
estimators.append(('scaler', RobustScaler()))
estimators.append(('clf', clf))
pipeline = myPipe(estimators)


# raw = Pipeline([('imputer', Imputer(missing_values='NaN', strategy='median',
#                                     axis=0, verbose=2)),
#                 ('logtransform', LogTransform())])
# pca = Pipeline([('imputer', Imputer(missing_values='NaN', strategy='median',
#                                     axis=0, verbose=2)),
#                 ('logtransform', LogTransform()),
#                 ('scaler', StandardScaler()),
#                 ('pca', decomposition.PCA(n_components=2,
#                                           random_state=1))])
# poly = Pipeline([('logtransform', LogTransform()),
#                  ('scaler', RobustScaler()),
#                  ('polyf', PolynomialFeatures(interaction_only=True))])
# combined_features = FeatureUnion([("raw", raw), ("pca", pca), ("poly", poly)])
# X_features = combined_features.fit(X, y).transform(X)
#
#
# pipeline = myPipe([
#     ('features', combined_features),
#     ('anova', SelectKBest(f_regression, 30)),
#     ('estimator', Pipeline([('scaler', StandardScaler()),
#                             ('clf', LogisticRegression(C=6.3,
#                                                        class_weight={0: 1, 1: 2.1},
#                                                        random_state=1, penalty='l2'))]))
# ])

from sklearn.feature_selection import RFECV
rfecv = RFECV(pipeline, step=1, cv=kfold, scoring='f1')
rfecv.fit(X, y)

# Plot number of features VS. cross-validation scores
plt.figure()
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score (F1)")
plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
plt.show()
