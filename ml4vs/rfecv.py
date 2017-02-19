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

clf = LogisticRegression(C=2.34, class_weight={0: 1, 1: 1.76}, random_state=1)
estimators = list()
estimators.append(('imputer', Imputer(missing_values='NaN', strategy='median',
                                      axis=0, verbose=2)))
estimators.append(('scaler', RobustScaler()))
estimators.append(('clf', clf))
pipeline = Pipeline(estimators)

from sklearn.feature_selection import RFECV
rfecv = RFECV(clf, step=1, cv=kfold, scoring='f1')
rfecv.fit(X, y)

# Plot number of features VS. cross-validation scores
plt.figure()
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score (F1)")
plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
plt.show()
