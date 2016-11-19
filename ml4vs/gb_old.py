# -*- coding: utf-8 -*-
import glob
import os
from data_load import load_data
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.cross_validation import StratifiedShuffleSplit, cross_val_score, \
    StratifiedKFold
from sklearn.grid_search import GridSearchCV
from sklearn.preprocessing import StandardScaler, Imputer
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

n_cv_iter = 5
seed = 1

# load dataset
data_dir = '/home/ilya/code/ml4vs/data/dataset_OGLE/indexes_normalized'
file_1 = 'vast_lightcurve_statistics_normalized_variables_only.log'
file_0 = 'vast_lightcurve_statistics_normalized_constant_only.log'
file_0 = os.path.join(data_dir, file_0)
file_1 = os.path.join(data_dir, file_1)
names = ['Magnitude', 'clipped_sigma', 'meaningless_1', 'meaningless_2',
         'star_ID', 'weighted_sigma', 'skew', 'kurt', 'I', 'J', 'K', 'L',
         'Npts', 'MAD', 'lag1', 'RoMS', 'rCh2', 'Isgn', 'Vp2p', 'Jclp', 'Lclp',
         'Jtim', 'Ltim', 'CSSD', 'Ex', 'inv_eta', 'E_A', 'S_B', 'NXS', 'IQR']
names_to_delete = ['Magnitude', 'meaningless_1', 'meaningless_2', 'star_ID',
                   'Npts']
X, y, feature_names, delta = load_data([file_0, file_1], names, names_to_delete)


sss = StratifiedShuffleSplit(y, n_iter=n_cv_iter, test_size=1./n_cv_iter)
skf = StratifiedKFold(y, n_folds=5, shuffle=True, random_state=seed)
imp = Imputer(missing_values='NaN', strategy='median', axis=0, verbose=2)
steps = [('imputation', imp),
         ('scaling', StandardScaler()),
         ('classification',
          GradientBoostingClassifier(random_state=42, learning_rate=0.01,
                                     max_features=7,
                                     n_estimators=800,
                                     subsample=0.6,
                                     min_samples_split=300,
                                     max_depth=7))]

pipeline = Pipeline(steps)

param_grid = {'classification__learning_rate': [0.1, 0.05, 0.01],
              'classification__max_depth': [2, 3, 5, 7],
              'classification__min_samples_leaf': [2, 4, 8, 16],
              'classification__max_features': [1.0, 0.5, 0.2, 0.1]}

print("Searching best parameters...")
gs_cv = GridSearchCV(pipeline, param_grid, cv=skf, n_jobs=4).fit(X, y)
print("The best parameters are %s with a score of %0.2f"
      % (gs_cv.best_params_, gs_cv.best_score_))
print "Feature" \
      " importance : {}".format(gs_cv.best_estimator_.named_steps['classification'].feature_importances_ /
                                np.sum(gs_cv.best_estimator_.named_steps['classification'].feature_importances_))

# Plot importance first way
from plotting import plot_importance
plot_importance(gs_cv.best_estimator_.named_steps['classification'],
                feature_names)

# Plot importance second way
feat_imp = pd.Series(gs_cv.best_estimator_.named_steps['classification'].feature_importances_,
                     feature_names).sort_values(ascending=False)
feat_imp.plot(kind='bar', title="Feature Importance")
plt.ylabel("Feature Importance Score")
plt.show()

# Final fit with best parameters
best_pipe = gs_cv.best_estimator_
best_pipe.named_steps['classification'].set_params(n_estimators=3000)
scores = cross_val_score(best_pipe, X, y, cv=skf, scoring='f1_weighted')
print("CV scores: ", scores)
