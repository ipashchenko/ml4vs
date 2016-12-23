import pprint
import numpy as np
import sys

from hyperopt import hp, fmin, tpe, STATUS_OK, Trials
from keras import callbacks
from keras.constraints import maxnorm
from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras.optimizers import SGD
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn import decomposition
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


class StackingEstimator(object):
    def __init__(self, base_estimators, meta_estimator):
        self.base_estimators = base_estimators
        self.meta_estimator = meta_estimator

    def fit(self, X, y, seed=1):
        cv = StratifiedShuffleSplit(y, n_iter=1, test_size=0.50,
                                    random_state=seed)
        for train_index1, train_index2 in cv:
            X_train1, X_train2 = X[train_index1], X[train_index2]
            y_train1, y_train2 = y[train_index1], y[train_index2]

        # Fit base learners using (X_train1, y_train1)
        for base_estimator in self.base_estimators:
            base_estimator.fit(X_train1, y_train1)

        # Predict responses for X_train2 using learned estimators
        y_predicted = list()
        for base_estimator in self.base_estimators:
            y_pred = base_estimator.predict_proba(X_train2)
            y_predicted.append(y_pred)

        # Stack predictions and X_train2 (#samples, #features)
        X_stacked = X_train2.copy()
        # X_stacked = list()
        for y_pred in y_predicted:
            # X_stacked.append(y_pred[:, 1])
            X_stacked = np.hstack((X_stacked, y_pred[:, 1][..., np.newaxis]))
        # X_stacked = np.dstack(X_stacked)[0, ...]

        # Fit meta estimator on stacked data
        self.meta_estimator.fit(X_stacked, y_train2)

    def predict(self, X):
        # First find predictions of base learners
        y_predicted = list()
        for base_estimator in self.base_estimators:
            y_pred = base_estimator.predict_proba(X)
            y_predicted.append(y_pred)

        # Stack predictions with original data
        X_stacked = X.copy()
        # X_stacked = list()
        for y_pred in y_predicted:
            # X_stacked.append(y_pred[:, 1])
            X_stacked = np.hstack((X_stacked, y_pred[:, 1][..., np.newaxis]))
        # X_stacked = np.dstack(X_stacked)[0, ...]

        # Predict using meta learner
        return self.meta_estimator.predict(X_stacked)


# Function that transforms some features
def log_axis(X_, names=None):
    X = X_.copy()
    tr_names = ['clipped_sigma', 'weighted_sigma', 'RoMS', 'rCh2', 'Vp2p',
                'Ex', 'inv_eta', 'S_B']
    for name in tr_names:
        try:
            # print "Log-Transforming {}".format(name)
            i = names.index(name)
            X[:, i] = np.log(X[:, i])
        except ValueError:
            print "No {} in predictors".format(name)
            pass
    return X


if __name__ == '__main__':
    import os
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
    names_to_delete = ['meaningless_1', 'meaningless_2', 'star_ID',
                       'Npts', 'CSSD', 'clipped_sigma', 'lag1', 'L', 'Lclp', 'Jclp',
                       'MAD', 'Ltim']
    X, y, df, feature_names, delta = load_data([file_0, file_1], names,
                                               names_to_delete)
    target = 'variable'
    predictors = list(df)
    predictors.remove(target)

    # Split X on train/test
    sss = StratifiedShuffleSplit(y, n_iter=1, test_size=0.33, random_state=1)
    for train_index, test_index in sss:
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]


def objective(space):
    print "====================="
    import pprint
    pprint.pprint(space)

    # Create model for NN
    def create_baseline():
        model = Sequential()
        model.add(Dense(18, input_dim=18, init='normal', activation='relu',
                        W_constraint=maxnorm(9.388)))
        model.add(Dropout(0.04))
        model.add(Dense(13, init='normal', activation='relu',
                        W_constraint=maxnorm(2.72)))
        # model.add(Activation(space['Activation']))
        model.add(Dropout(0.09))
        model.add(Dense(1, init='normal', activation='sigmoid'))

        # Compile model
        learning_rate = 0.213
        decay_rate = 0.001
        momentum = 0.9
        sgd = SGD(lr=learning_rate, decay=decay_rate, momentum=momentum,
                  nesterov=False)
        model.compile(loss='binary_crossentropy', optimizer=sgd,
                      metrics=['accuracy'])
        return model

    estimators = list()
    estimators.append(('imputer', Imputer(missing_values='NaN', strategy='median',
                                          axis=0, verbose=0)))
    estimators.append(('scaler', StandardScaler()))
    estimators.append(('mlp', KerasClassifier(build_fn=create_baseline,
                                              nb_epoch=125,
                                              batch_size=1024,
                                              verbose=0)))
    pipeline_nn = Pipeline(estimators)

    # Create model for RF
    clf = RandomForestClassifier(n_estimators=1200,
                                 max_depth=17,
                                 max_features=3,
                                 min_samples_split=2,
                                 min_samples_leaf=3,
                                 class_weight='balanced_subsample',
                                 verbose=0, random_state=1, n_jobs=4)
    estimators = list()
    estimators.append(('imputer', Imputer(missing_values='NaN', strategy='median',
                                          axis=0, verbose=2)))
    estimators.append(('clf', clf))
    pipeline_rf = Pipeline(estimators)

    # Create model for LR
    clf = LogisticRegression(C=1.29, class_weight={0: 1, 1: 2},
                             random_state=1, max_iter=300, n_jobs=1,
                             tol=10.**(-5))
    pca = decomposition.PCA(n_components=16, random_state=1)
    estimators = list()
    estimators.append(('imputer', Imputer(missing_values='NaN', strategy='median',
                                          axis=0, verbose=2)))
    estimators.append(('func', FunctionTransformer(log_axis, kw_args={'names':
                                                                          predictors})))
    estimators.append(('scaler', StandardScaler()))
    estimators.append(('pca', pca))
    estimators.append(('clf', clf))
    pipeline_lr = Pipeline(estimators)

    # Model for kNN
    clf = KNeighborsClassifier(n_neighbors=6,
                               weights='distance', n_jobs=4)
    estimators = list()
    estimators.append(('imputer', Imputer(missing_values='NaN', strategy='median',
                                          axis=0, verbose=2)))
    estimators.append(('scaler', StandardScaler()))
    estimators.append(('clf', clf))
    pipeline_knn = Pipeline(estimators)

    # Create model for GB
    sys.path.append('/home/ilya/xgboost/xgboost/python-package/')
    import xgboost as xgb
    clf = xgb.XGBClassifier(n_estimators=87, learning_rate=0.111,
                            max_depth=6,
                            min_child_weight=2,
                            subsample=0.275,
                            colsample_bytree=0.85,
                            colsample_bylevel=0.55,
                            gamma=3.14,
                            scale_pos_weight=6)
    estimators = list()
    estimators.append(('imputer', Imputer(missing_values='NaN', strategy='median',
                                          axis=0, verbose=2)))
    estimators.append(('clf', clf))
    pipeline_xgb = Pipeline(estimators)

    # Create model for SVM
    clf = SVC(C=37.3, class_weight={0: 1, 1: 3}, probability=True,
              gamma=0.0126, random_state=1)
    estimators = list()
    estimators.append(('imputer', Imputer(missing_values='NaN', strategy='median',
                                          axis=0, verbose=2)))
    estimators.append(('scaler', StandardScaler()))
    estimators.append(('clf', clf))
    pipeline_svm = Pipeline(estimators)

    # Create meta estimator
    clf = RandomForestClassifier(n_estimators=space['n'],
                                 max_depth=space['max_depth'],
                                 max_features=space['max_features'],
                                 min_samples_split=space['mss'],
                                 min_samples_leaf=space['msl'],
                                 class_weight={0:1, 1: space['cw']},
                                 verbose=0, random_state=1, n_jobs=4)
    estimators = list()
    estimators.append(('imputer', Imputer(missing_values='NaN', strategy='median',
                                          axis=0, verbose=2)))
    estimators.append(('clf', clf))

    # clf = KNeighborsClassifier(n_neighbors=space['n'],
    #                            weights='distance', n_jobs=2)
    # estimators = list()
    # estimators.append(('imputer', Imputer(missing_values='NaN', strategy='median',
    #                                       axis=0, verbose=2)))
    # estimators.append(('scaler', StandardScaler()))
    # estimators.append(('clf', clf))

    # pca = decomposition.PCA(n_components=space['n_pca'], random_state=1)
    # clf = LogisticRegression(C=space['C'], class_weight={0: 1, 1: space['cw']},
    #                          random_state=1, max_iter=300, n_jobs=1,
    #                          tol=10.**(-5))
    # estimators = list()
    # estimators.append(('imputer', Imputer(missing_values='NaN', strategy='median',
    #                                       axis=0, verbose=2)))
    # estimators.append(('scaler', StandardScaler()))
    # estimators.append(('pca', pca))
    # estimators.append(('clf', clf))

    meta_pipeline = Pipeline(estimators)

    base_estimators = [pipeline_rf, pipeline_lr, pipeline_xgb, pipeline_knn,
                       pipeline_nn, pipeline_svm]
    stacking_ensemble = StackingEstimator(base_estimators, meta_pipeline)

    stacking_ensemble.fit(X_train, y_train, seed=1)
    y_pred = stacking_ensemble.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    import pprint
    pprint.pprint(cm)
    print "F1 = ", f1
    return{'loss': 1-f1, 'status': STATUS_OK}


# space = {'C': hp.loguniform('C', -4.6, 4.0),
#          'cw': hp.qloguniform('cw', 0, 6, 1),
#          'n_pca': hp.choice('n_pca', np.arange(1, 7, 1, dtype=int))}
space = {'n': hp.choice('n', (400, 600, 800, 1000, 1200)),
         'max_depth': hp.choice('max_depth', np.arange(5, 24, dtype=int)),
         'max_features': hp.choice('max_features', np.arange(3, 24, dtype=int)),
         'mss': hp.choice('mss', (2, 3, 4, 7, 15)),
         'msl': hp.choice('msl', (2, 3, 5, 10, 15, 30)),
         'cw': hp.qloguniform('cw', 0, 6, 1)}

trials = Trials()
best = fmin(fn=objective,
            space=space,
            algo=tpe.suggest,
            max_evals=50,
            trials=trials)

import hyperopt
pprint.pprint(hyperopt.space_eval(space, best))


# # Load blind test data
# file_tgt = 'LMC_SC19_PSF_Pgood98__vast_lightcurve_statistics.log'
# file_tgt = os.path.join(data_dir, file_tgt)
# names = ['Magnitude', 'clipped_sigma', 'meaningless_1', 'meaningless_2',
#          'star_ID', 'weighted_sigma', 'skew', 'kurt', 'I', 'J', 'K', 'L',
#          'Npts', 'MAD', 'lag1', 'RoMS', 'rCh2', 'Isgn', 'Vp2p', 'Jclp', 'Lclp',
#          'Jtim', 'Ltim', 'CSSD', 'Ex', 'inv_eta', 'E_A', 'S_B', 'NXS', 'IQR']
# names_to_delete = ['meaningless_1', 'meaningless_2', 'star_ID',
#                    'Npts', 'CSSD', 'clipped_sigma', 'lag1', 'L', 'Lclp', 'Jclp',
#                    'MAD', 'Ltim']
# X_tgt, feature_names, df, df_orig = load_data_tgt(file_tgt, names,
#                                                   names_to_delete, delta)
#
# pca = decomposition.PCA(n_components=1, random_state=1)
# clf = LogisticRegression(C=0.037, class_weight={0: 1, 1: 3},
#                          random_state=1, max_iter=300, n_jobs=1,
#                          tol=10.**(-5))
# estimators = list()
# estimators.append(('imputer', Imputer(missing_values='NaN', strategy='median',
#                                       axis=0, verbose=2)))
# estimators.append(('scaler', StandardScaler()))
# estimators.append(('pca', pca))
# estimators.append(('clf', clf))
#
# meta_pipeline = Pipeline(estimators)
#
# base_estimators = [pipeline_rf, pipeline_lr, pipeline_xgb, pipeline_knn,
#                    pipeline_nn, pipeline_svm]
# stacking_ensemble = StackingEstimator(base_estimators, meta_pipeline)
#
# stacking_ensemble.fit(X, y, seed=1)
# y_pred = stacking_ensemble.predict(X_tgt)
#
# idx = y_pred == 1
# idx_ = y_pred == 0
#
#
# # idx = y_probs > 0.250
# # idx_ = y_probs < 0.250
# ens_no = list(df_orig['star_ID'][idx_])
# print("Found {} variables".format(np.count_nonzero(idx)))
#
# with open('ens_results_final.txt', 'w') as fo:
#     for line in list(df_orig['star_ID'][idx]):
#         fo.write(line + '\n')
#
# # Check F1
# with open('clean_list_of_new_variables.txt', 'r') as fo:
#     news = fo.readlines()
# news = [line.strip().split(' ')[1] for line in news]
# news = set(news)
#
# with open('ens_results_final.txt', 'r') as fo:
#     ens = fo.readlines()
# ens = [line.strip().split('_')[4].split('.')[0] for line in ens]
# ens = set(ens)
#
# print "Among new vars found {}".format(len(news.intersection(ens)))
#
# with open('candidates_50perc_threshold.txt', 'r') as fo:
#     c50 = fo.readlines()
# c50 = [line.strip("\", ', \", \n, }, {") for line in c50]
#
# with open('variables_not_in_catalogs.txt', 'r') as fo:
#     not_in_cat = fo.readlines()
# nic = [line.strip().split(' ')[1] for line in not_in_cat]
#
# # Catalogue variables
# cat_vars = set(c50).difference(set(nic))
# # Non-catalogue variable
# noncat_vars = set([line.strip().split(' ')[1] for line in not_in_cat if 'CST' not in line])
#
# # All variables
# all_vars = news.union(cat_vars).union(noncat_vars)
# ens_no = set([line.strip().split('_')[4].split('.')[0] for line in ens_no])
#
# found_bad = '181193' in ens
# print "Found known variable : ", found_bad
#
# FN = len(ens_no.intersection(all_vars))
# TP = len(all_vars.intersection(ens))
# TN = len(ens_no) - FN
# FP = len(ens) - TP
# recall = float(TP) / (TP + FN)
# precision = float(TP) / (TP + FP)
# F1 = 2 * precision * recall / (precision + recall)
# print "precision: {}".format(precision)
# print "recall: {}".format(recall)
# print "F1: {}".format(F1)
# print "TN={}, FP={}".format(TN, FP)
# print "FN={}, TP={}".format(FN, TP)
