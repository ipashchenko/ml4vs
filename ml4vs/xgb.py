import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn import metrics
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler, Imputer
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import StratifiedKFold, cross_val_score, \
    train_test_split, StratifiedShuffleSplit
from matplotlib import pyplot
from data_load import load_data, load_data_tgt
import xgboost as xgb


# load data
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


imp = Imputer(missing_values='NaN', strategy='median', axis=0, verbose=2)
steps = [('imputation', imp),
         ('classification', XGBClassifier(max_depth=5, min_child_weight=1,
                                          gamma=0, subsample=0.8,
                                          colsample_bytree=0.8,
                                          scale_pos_weight=150,
                                          learning_rate=0.03,
                                          n_estimators=450))]

pipeline = Pipeline(steps)

##########################################################
# First check default parameters and check learning curves
sss = StratifiedShuffleSplit(y, n_iter=1, test_size=0.25, random_state=7)
for train_index, test_index in sss:
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

# FIXME: Problem with evaluating pipeline
imp.fit(X_train)
X_trained_scaled = imp.transform(X_train)
# Use the same transformation & imputation for testing data!
X_test_scaled = imp.transform(X_test)
eval_set = [(X_trained_scaled, y_train), (X_test_scaled, y_test)]

model = pipeline.named_steps['classification']
model.fit(X_trained_scaled, y_train, eval_metric=["error", "logloss"],
          eval_set=eval_set, verbose=True)

# make predictions for test data - now using pipeline
y_pred = pipeline.predict(X_test)
predictions = [round(value) for value in y_pred]
# evaluate predictions
accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))
# retrieve performance metrics
results = pipeline.named_steps['classification'].evals_result()
epochs = len(results['validation_0']['error'])
x_axis = range(0, epochs)
# plot log loss
fig, ax = pyplot.subplots()
ax.plot(x_axis, results['validation_0']['logloss'], label='Train')
ax.plot(x_axis, results['validation_1']['logloss'], label='Test')
ax.legend()
pyplot.ylabel('Log Loss')
pyplot.title('XGBoost Log Loss')
pyplot.show()
# plot classification error
fig, ax = pyplot.subplots()
ax.plot(x_axis, results['validation_0']['error'], label='Train')
ax.plot(x_axis, results['validation_1']['error'], label='Test')
ax.legend()
pyplot.ylabel('Classification Error')
pyplot.title('XGBoost Classification Error')
pyplot.show()
###########################################################

file_tgt = 'LMC_SC19_PSF_Pgood98__vast_lightcurve_statistics_normalized.log'
file_tgt = os.path.join(data_dir, file_tgt)
X_tgt, feature_names, df = load_data_tgt(file_tgt, names, names_to_delete, delta)
y_pred = pipeline.predict(X_tgt)
proba_pred = pipeline.predict_proba(X_tgt)










kfold = StratifiedKFold(y, n_folds=4, shuffle=True, random_state=7)
# scores = cross_val_score(pipeline, X, y, cv=kfold, scoring='recall', n_jobs=-1)
# print("CV scores: ", scores)


# param_grid = {'classification__n_estimators': [5, 100, 150, 200],
#               'classification__max_depth': [2, 4, 6, 8]}
#
# print("Searching best parameters...")
# result = GridSearchCV(pipeline, param_grid, cv=kfold, n_jobs=-1, verbose=1,
#                      scoring='f1').fit(X, y)
# print("The best parameters are %s with a score of %0.2f"
#       % (result.best_params_, result.best_score_))
#
# # summarize results
# print("Best: %f using %s" % (result.best_score_, result.best_params_))
# means, stdevs = [], []
# for params, mean_score, scores in result.grid_scores_:
#     stdev = scores.std()
#     means.append(mean_score)
#     stdevs.append(stdev)
#     print("%f (%f) with: %r" % (mean_score, stdev, params))
# # plot results
# scores = [x[1] for x in result.grid_scores_]
# max_depth = param_grid['classification__max_depth']
# n_estimators = param_grid['classification__n_estimators']
# scores = np.array(scores).reshape(len(max_depth), len(n_estimators))
# for i, value in enumerate(max_depth):
#     pyplot.plot(n_estimators, scores[i], label='depth: ' + str(value))
# pyplot.legend()
# pyplot.xlabel('n_estimators')
# pyplot.ylabel('F1-score')
# pyplot.show()
