import os
import numpy as np
from sklearn.metrics import (accuracy_score, f1_score, precision_recall_curve,
                             precision_score, recall_score)

from xgboost import XGBClassifier
from sklearn.preprocessing import Imputer
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import StratifiedKFold, cross_val_score, \
    train_test_split, StratifiedShuffleSplit
from matplotlib import pyplot
from data_load import load_data, load_data_tgt
import xgboost as xgb
import pandas as pd


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
X, y, df, feature_names, delta = load_data([file_0, file_1], names, names_to_delete)
# names_to_delete = ['Magnitude', 'meaningless_1', 'meaningless_2', 'star_ID',
#                    'Npts']
# X_c, y, df, feature_names3, delta = load_data([file_0, file_1], names, names_to_delete)


imp = Imputer(missing_values='NaN', strategy='median', axis=0, verbose=2)
steps = [('imputation', imp),
         ('classification', XGBClassifier(max_depth=5, min_child_weight=1,
                                          gamma=0, subsample=0.8,
                                          colsample_bytree=0.8,
                                          scale_pos_weight=150,
                                          learning_rate=0.03,
                                          n_estimators=1000))]

pipeline = Pipeline(steps)
# steps2 = [('imputation', imp),
#           ('classification', XGBClassifier(max_depth=5, min_child_weight=1,
#                                            gamma=0, subsample=0.8,
#                                            colsample_bytree=0.8,
#                                            scale_pos_weight=1,
#                                            learning_rate=0.03,
#                                            n_estimators=250))]
#
# pipeline2 = Pipeline(steps2)
imp3 = Imputer(missing_values='NaN', strategy='median', axis=0, verbose=2)
steps3 = [('imputation', imp3),
         ('classification', XGBClassifier(max_depth=5, min_child_weight=10,
                                          gamma=0, subsample=0.8,
                                          colsample_bytree=0.8,
                                          colsample_bylevel=1.,
                                          scale_pos_weight=150,
                                          learning_rate=0.03,
                                          n_estimators=950))]
pipeline3 = Pipeline(steps3)

##########################################################
# First check default parameters and check learning curves
sss = StratifiedShuffleSplit(y, n_iter=1, test_size=0.25, random_state=7)
for train_index, test_index in sss:
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    # X_train_c, X_test_c = X_c[train_index], X_c[test_index]

# FIXME: Problem with evaluating pipeline
imp.fit(X_train)
X_trained_scaled = imp.transform(X_train)
# Use the same transformation & imputation for testing data!
X_test_scaled = imp.transform(X_test)
eval_set = [(X_trained_scaled, y_train), (X_test_scaled, y_test)]

# imp3.fit(X_train_c)
# X_trained_scaled_c = imp3.transform(X_train_c)
# # Use the same transformation & imputation for testing data!
# X_test_scaled_c = imp3.transform(X_test_c)
# eval_set3 = [(X_trained_scaled_c, y_train), (X_test_scaled_c, y_test)]

model = pipeline.named_steps['classification']
model.fit(X_trained_scaled, y_train, eval_metric=["error", "auc", "logloss"],
          eval_set=eval_set, verbose=True, early_stopping_rounds=50)
# model2 = pipeline2.named_steps['classification']
# model2.fit(X_trained_scaled, y_train, eval_metric=["error", "logloss", "auc"],
#            eval_set=eval_set, verbose=False)
# model3 = pipeline3.named_steps['classification']
# model3.fit(X_trained_scaled_c, y_train, eval_metric=["auc", "error", "logloss"],
#            eval_set=eval_set3, verbose=False, early_stopping_rounds=30)

# make predictions for test data - now using pipeline
y_pred = pipeline.predict(X_test)
y_probapred = pipeline.predict_proba(X_test)
predictions = [round(value) for value in y_pred]
# y_pred2 = pipeline2.predict(X_test)
# y_probapred2 = pipeline2.predict_proba(X_test)
# predictions2 = [round(value) for value in y_pred2]
# y_pred3 = pipeline3.predict(X_test_c)
# y_probapred3 = pipeline3.predict_proba(X_test_c)
# predictions3 = [round(value) for value in y_pred3]
# evaluate predictions
accuracy = accuracy_score(y_test, predictions)
print("Model baseline")
print("Accuracy: %.2f%%" % (accuracy * 100.0))
f1 = f1_score(y_test, predictions)
print("F1-score: %.2f%%" % (f1 * 100.0))
precision = precision_score(y_test, y_pred)
print("Precision: %.2f%%" % (precision * 100.0))
recall = recall_score(y_test, predictions)
print("Recall: %.2f%%" % (recall * 100.0))

# print("Model with modification")
# accuracy3 = accuracy_score(y_test, predictions3)
# print("Accuracy: %.2f%%" % (accuracy3 * 100.0))
# f13 = f1_score(y_test, predictions3)
# print("F1-score: %.2f%%" % (f13 * 100.0))
# precision3 = precision_score(y_test, y_pred3)
# print("Precision: %.2f%%" % (precision3 * 100.0))
# recall3 = recall_score(y_test, predictions3)
# print("Recall: %.2f%%" % (recall3 * 100.0))

# print("Model with scale_pos_weight=1")
# accuracy2 = accuracy_score(y_test, predictions2)
# print("Accuracy: %.2f%%" % (accuracy2 * 100.0))
# f12 = f1_score(y_test, predictions2)
# print("F1-score: %.2f%%" % (f12 * 100.0))
# precision2 = precision_score(y_test, y_pred2)
# print("Precision: %.2f%%" % (precision2 * 100.0))
# recall2 = recall_score(y_test, predictions2)
# print("Recall: %.2f%%" % (recall2 * 100.0))
precisions, recalls, _ = precision_recall_curve(y_test, y_probapred[:, 1])
# precisions2, recalls2, _ = precision_recall_curve(y_test, y_probapred2[:, 1])
# precisions3, recalls3, _ = precision_recall_curve(y_test, y_probapred3[:, 1])
# Plot Precision-Recall curve
fig, ax = pyplot.subplots()
ax.plot(recalls, precisions, label='Baseline')
# ax.plot(recalls3, precisions3, label='With modifications')
# ax.plot(recalls2, precisions2, label='scale_pos_weight=1')
ax.set_xlabel('Recall')
ax.set_ylabel('Precision')
ax.set_ylim([0.0, 1.05])
ax.set_xlim([0.0, 1.0])
ax.set_title('XGBoost P-R curve')
ax.legend(loc='lower left')
fig.show()

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
# plot auc
fig, ax = pyplot.subplots()
ax.plot(x_axis, results['validation_0']['auc'], label='Train')
ax.plot(x_axis, results['validation_1']['auc'], label='Test')
ax.legend(loc='lower left')
pyplot.ylabel('AUC')
pyplot.title('XGBoost AUC')
pyplot.show()

scores_dict = model.booster().get_fscore()
scores_dict_ = dict()
for i, feature in enumerate(feature_names):
    scores_dict_[feature] = scores_dict['f{}'.format(i)]

feat_imp = pd.Series(scores_dict_).sort_values(ascending=False)
feat_imp.plot(kind='bar', title='Feature Importances')
pyplot.ylabel('Feature Importance Score')
pyplot.show()
###########################################################
# kfold = StratifiedKFold(y, n_folds=4, shuffle=True, random_state=7)
# scores = cross_val_score(pipeline, X, y, cv=kfold, scoring='f1', n_jobs=-1)
# print("CV scores: ", scores)

model = XGBClassifier(max_depth=5, min_child_weight=1, gamma=0, subsample=0.8,
                      colsample_bytree=0.8, scale_pos_weight=1,
                      learning_rate=0.03, n_estimators=450)
imp = Imputer(missing_values='NaN', strategy='median', axis=0, verbose=2)
imp.fit(X)
X_scaled = imp.transform(X)
model.fit(X_scaled, y, verbose=True, eval_metric='auc')
file_tgt = 'LMC_SC19_PSF_Pgood98__vast_lightcurve_statistics_normalized.log'
file_tgt = os.path.join(data_dir, file_tgt)
X_tgt, feature_names, df = load_data_tgt(file_tgt, names, names_to_delete, delta)
X_tgt_scaled = imp.transform(X_tgt)
y_pred = model.predict(X_tgt_scaled)
proba_pred = model.predict_proba(X_tgt_scaled)
idx = proba_pred[:, 1] > 0.25
print("Found {} variables".format(np.count_nonzero(idx)))

with open('first_results.txt', 'w') as fo:
    for line in list(df['star_ID'][idx]):
        fo.write(line + '\n')


# kfold = StratifiedKFold(y, n_folds=4, shuffle=True, random_state=7)
# scores = cross_val_score(pipeline, X, y, cv=kfold, scoring='f1', n_jobs=-1)
# print("CV scores: ", scores)
#
#
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
