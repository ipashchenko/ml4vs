import os
import numpy as np
from sklearn.cross_validation import StratifiedShuffleSplit, cross_val_score, \
    StratifiedKFold
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Imputer
from xgboost import XGBClassifier
from data_load import load_data, load_data_tgt
from plotting import plot_importance
from matplotlib import pyplot


# Load data
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
X, y, df, feature_names, delta = load_data([file_0, file_1], names,
                                           names_to_delete)

imp = Imputer(missing_values='NaN', strategy='median', axis=0, verbose=2)
# This one is good
# model = XGBClassifier(max_depth=5, min_child_weight=1, gamma=0, subsample=0.8,
#                       colsample_bytree=0.8, scale_pos_weight=150,
#                       learning_rate=0.03, n_estimators=5000)
model = XGBClassifier(max_depth=5, min_child_weight=3, gamma=0.2, subsample=0.8,
                      colsample_bytree=0.8, scale_pos_weight=150,
                      learning_rate=0.01, n_estimators=5000,
                      reg_alpha=10.**(-5), reg_lambda=10.**(-5))

sss = StratifiedShuffleSplit(y, n_iter=1, test_size=0.25, random_state=7)
for train_index, test_index in sss:
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
imp.fit(X_train)
X_trained_scaled = imp.transform(X_train)
# Use the same transformation & imputation for testing data!
X_test_scaled = imp.transform(X_test)
eval_set = [(X_trained_scaled, y_train), (X_test_scaled, y_test)]

model.fit(X_trained_scaled, y_train, eval_metric=["error", "auc", "logloss"],
          eval_set=eval_set, verbose=True, early_stopping_rounds=50)
y_pred = model.predict(X_test_scaled)
y_predproba = model.predict_proba(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: %.2f%%" % (accuracy * 100.0))
f1 = f1_score(y_test, y_pred)
print("F1-score: %.2f%%" % (f1 * 100.0))
precision = precision_score(y_test, y_pred)
print("Precision: %.2f%%" % (precision * 100.0))
recall = recall_score(y_test, y_pred)
print("Recall: %.2f%%" % (recall * 100.0))

precisions, recalls, _ = precision_recall_curve(y_test, y_predproba[:, 1])

# retrieve performance metrics
results = model.evals_result()
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

plot_importance(model, feature_names)
# Plot Precision-Recall curve
fig, ax = pyplot.subplots()
ax.plot(recalls, precisions, label='Baseline')
ax.set_xlabel('Recall')
ax.set_ylabel('Precision')
ax.set_ylim([0.0, 1.05])
ax.set_xlim([0.0, 1.0])
ax.set_title('XGBoost P-R curve')
ax.legend(loc='lower left')
fig.show()


do_cv = False
if do_cv:
    imp = Imputer(missing_values='NaN', strategy='median', axis=0, verbose=2)
    steps = [('imputation', imp),
             ('classification', XGBClassifier(max_depth=5, min_child_weight=1,
                                              gamma=0.0, subsample=0.8,
                                              colsample_bytree=0.8,
                                              scale_pos_weight=150,
                                              learning_rate=0.1,
                                              n_estimators=model.best_ntree_limit))]

    pipeline = Pipeline(steps)

    kfold = StratifiedKFold(y, n_folds=4, shuffle=True, random_state=7)
    # results = cross_val_score(pipeline, X, y, scoring='f1', n_jobs=1)
    # print results


    # Using grdi search for tuning
    param_grid = {#'classification__learning_rate': [0.1, 0.6, 0.01],
                  #'classification__max_depth': [4, 5, 6],
                  #'classification__min_child_weight': [2, 3, 4]
                  #'classification__gamma': [i/10.0 for i in range(0, 5)]
                   'classification__reg_alpha': [1e-5, 1e-2, 0.1, 1, 100],
                   'classification__reg_lambda': [1e-5, 1e-2, 0.1, 1, 100]
                  # 'classification__subsample': [i/10.0 for i in range(6, 10)],
                  # 'classification__colsample_bytree': [i/10.0 for i in range(6, 10)]
                  # 'classification__scale_pos_weight': [0., 1., 10, 100, 300]
                  # 'classification__subsample': [0.5, 0.75, 1]
                  }


    print "Grid search CV..."
    gs_cv = GridSearchCV(pipeline, param_grid, scoring="f1", n_jobs=1,
                         cv=kfold, verbose=1).fit(X, y)

    print("The best parameters are %s with a score of %0.2f"
          % (gs_cv.best_params_, gs_cv.best_score_))

    plot_importance(gs_cv.best_estimator_.named_steps['classification'],
                    feature_names)

    # Best parameters
    best_xgb_params = gs_cv.best_estimator_.named_steps['classification'].get_params()
    best_xgb_params['n_estimators'] = 5000

    best_model = XGBClassifier(**best_xgb_params)

    sss = StratifiedShuffleSplit(y, n_iter=1, test_size=0.25, random_state=7)
    # Working with the same data as baseline model
    best_model.fit(X_trained_scaled, y_train, eval_metric=["logloss"],
                   eval_set=eval_set, verbose=True, early_stopping_rounds=50)

    y_predproba_ = best_model.predict_proba(X_test_scaled)
    y_pred_ = best_model.predict(X_test_scaled)
    precisions_, recalls_, _ = precision_recall_curve(y_test, y_predproba_[:, 1])
    accuracy = accuracy_score(y_test, y_pred_)
    print("Accuracy: %.2f%%" % (accuracy * 100.0))
    f1 = f1_score(y_test, y_pred_)
    print("F1-score: %.2f%%" % (f1 * 100.0))
    precision = precision_score(y_test, y_pred_)
    print("Precision: %.2f%%" % (precision * 100.0))
    recall = recall_score(y_test, y_pred_)
    print("Recall: %.2f%%" % (recall * 100.0))

    # Plot Precision-Recall curve
    fig, ax = pyplot.subplots()
    ax.plot(recalls, precisions, label='Baseline')
    ax.plot(recalls_, precisions_, label='Best CV')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_ylim([0.0, 1.05])
    ax.set_xlim([0.0, 1.0])
    ax.set_title('XGBoost P-R curve')
    ax.legend(loc='lower left')
    fig.show()

predict_target = True
if predict_target:
    imp = Imputer(missing_values='NaN', strategy='median', axis=0, verbose=2)
    imp.fit(X)
    X_scaled = imp.transform(X)
    # Use the same transformation & imputation for testing data!
    n = model.best_ntree_limit
    model = XGBClassifier(max_depth=5, min_child_weight=3, gamma=0.2,
                          subsample=0.8, colsample_bytree=0.8,
                          scale_pos_weight=150, learning_rate=0.01,
                          n_estimators=n, reg_alpha=10.**(-5),
                          reg_lambda=10.**(-5))
    model.fit(X_scaled, y)
    file_tgt = 'LMC_SC19_PSF_Pgood98__vast_lightcurve_statistics_normalized.log'
    file_tgt = os.path.join(data_dir, file_tgt)
    X_tgt, feature_names, df = load_data_tgt(file_tgt, names, names_to_delete, delta)
    X_tgt_scaled = imp.transform(X_tgt)
    proba_pred = model.predict_proba(X_tgt_scaled)
    idx = proba_pred[:, 1] > 0.25
    print("Found {} variables".format(np.count_nonzero(idx)))

    with open('target_variables.txt', 'w') as fo:
        for line in list(df['star_ID'][idx]):
            fo.write(line + '\n')
