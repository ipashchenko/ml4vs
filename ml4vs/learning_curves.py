import os
import numpy as np

import sys
sys.path.append('/home/ilya/code/mlxtend')
from mlxtend.plotting import plot_learning_curves
from sklearn import decomposition
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from data_load import load_data
sys.path.append('/home/ilya/xgboost/xgboost/python-package/')
import xgboost as xgb
from keras.constraints import maxnorm
from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras.optimizers import SGD
from keras.wrappers.scikit_learn import KerasClassifier


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


clf_xgb = xgb.XGBClassifier(n_estimators=87, learning_rate=0.111,
                            max_depth=6,
                            min_child_weight=2,
                            subsample=0.275,
                            colsample_bytree=0.85,
                            colsample_bylevel=0.55,
                            gamma=3.14,
                            max_delta_step=7,
                            scale_pos_weight=6,
                            seed=1)

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
                                      axis=0, verbose=2)))
estimators.append(('scaler', StandardScaler()))
estimators.append(('mlp', KerasClassifier(build_fn=create_baseline,
                                          nb_epoch=175,
                                          batch_size=1024,
                                          verbose=2)))
clf_nn = Pipeline(estimators)


clf_rf = RandomForestClassifier(n_estimators=1400,
                                max_depth=16,
                                max_features=5,
                                min_samples_split=16,
                                min_samples_leaf=2,
                                class_weight={0: 1, 1: 28},
                                verbose=1, random_state=1, n_jobs=4)


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
clf_lr = Pipeline(estimators)


# Model for kNN
clf = KNeighborsClassifier(n_neighbors=6,
                           weights='distance', n_jobs=4)
estimators = list()
estimators.append(('imputer', Imputer(missing_values='NaN', strategy='median',
                                      axis=0, verbose=2)))
estimators.append(('scaler', StandardScaler()))
estimators.append(('clf', clf))
clf_knn = Pipeline(estimators)


# Model for SVM
clf = SVC(C=37.3, class_weight={0: 1, 1: 3}, probability=False,
          gamma=0.0126, random_state=1)
estimators = list()
estimators.append(('imputer', Imputer(missing_values='NaN', strategy='median',
                                      axis=0, verbose=2)))
estimators.append(('scaler', StandardScaler()))
estimators.append(('clf', clf))
clf_svm = Pipeline(estimators)


# sss = StratifiedShuffleSplit(n_splits=4, test_size=0.25, random_state=1)
# errors_full = list()
# for train_index, test_index in sss.split(X, y):
#     X_train, X_test = X[train_index], X[test_index]
#     y_train, y_test = y[train_index], y[test_index]
#     errors = plot_learning_curves(X_train, y_train, X_test, y_test, clf_lr,
#                                   scoring='f1', print_model=False)
#     errors_full.append(errors)


cv = StratifiedShuffleSplit(n_splits=20, test_size=0.25, random_state=0)
from plotting import plot_learning_curve
plot_learning_curve(clf_xgb, 'SGB', X, y, cv=cv, n_jobs=4, scoring='f1')
