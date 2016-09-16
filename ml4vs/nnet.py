# -*- coding: utf-8 -*-
import glob
import os

from sklearn_evaluation.plot import confusion_matrix as plot_cm

from data_load import load_data
import numpy
from keras import callbacks
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.constraints import maxnorm
from keras.optimizers import SGD
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import StratifiedShuffleSplit, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.preprocessing import Imputer
from sklearn.metrics import confusion_matrix

from utils import print_cm_summary
import matplotlib.pyplot as plt

remote = callbacks.RemoteMonitor(root='http://localhost:9000')

# fix random seed for reproducibility
seed = 1
numpy.random.seed(seed)

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

n_cv_iter = 5


def create_baseline():
    # create model
    model = Sequential()
    model.add(Dense(25, input_dim=25, init='normal', activation='relu',
                    W_constraint=maxnorm(3)))
    model.add(Dropout(0.1))
    model.add(Dense(25, init='normal', activation='relu',
                    W_constraint=maxnorm(3)))
    model.add(Dropout(0.1))
    model.add(Dense(13, init='normal', activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(1, init='normal', activation='sigmoid'))
    # Compile model
    learning_rate = 0.1
    decay_rate = learning_rate / epochs
    momentum = 0.90
    sgd = SGD(lr=learning_rate, decay=decay_rate, momentum=momentum,
              nesterov=False)
    model.compile(loss='binary_crossentropy', optimizer=sgd,
                  metrics=['accuracy'])
    return model


epochs = 50
# epochs = 125
batch_size = 12
estimators = list()
estimators.append(('imputer', Imputer(missing_values='NaN', strategy='median',
                                      axis=0, verbose=2)))
estimators.append(('scaler', StandardScaler()))
estimators.append(('mlp', KerasClassifier(build_fn=create_baseline,
                                          nb_epoch=epochs,
                                          batch_size=batch_size,
                                          verbose=0)))
skf = StratifiedKFold(y, n_folds=5, shuffle=True, random_state=seed)

pipeline = Pipeline(estimators)

results = cross_val_score(pipeline, X, y, cv=skf, scoring='f1', n_jobs=3)
print("\n")
print(results)
print("\n")
print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
print("\n")

# # Check overfitting
# sss = StratifiedShuffleSplit(y, n_iter=1, test_size=1. / n_cv_iter,
#                              random_state=seed)
# for train_index, test_index in sss:
#     X_train, X_test = X[train_index], X[test_index]
#     y_train, y_test = y[train_index], y[test_index]
#
# import keras
# history = keras.callbacks.History()
# print("Fitting...")
# X_test_ = X_test.copy()
# X_train_ = X_train.copy()
# for name, transform in pipeline.steps[:-1]:
#     print(name, transform)
#     transform.fit(X_train_)
#     X_test_ = transform.transform(X_test_)
#     X_train_ = transform.transform(X_train_)
# pipeline.fit(X_train, y_train, mlp__validation_data=(X_test_, y_test),
#              mlp__batch_size=batch_size, mlp__nb_epoch=epochs,
#              mlp__callbacks=[history])
# model = pipeline.named_steps['mlp']
#
# y_pred = model.predict(X_test_)
# y_pred[y_pred < 0.5] = 0.
# y_pred[y_pred >= 0.5] = 1.
# y_probs = model.predict_proba(X_test_)
# cm = confusion_matrix(y_test, y_pred)
# print_cm_summary(cm)
#
# plt.plot(history.history['acc'])
# plt.plot(history.history['val_acc'])
# plt.title('model accuracy')
# plt.ylabel('accuracy')
# plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='upper left')
# plt.show()
# # summarize history for loss
# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.title('model loss')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='upper left')
# plt.show()

# # Build several cm
# skf = StratifiedKFold(y, n_folds=4, shuffle=True, random_state=seed)
# for train_index, test_index in skf:
#     X_train, X_test = X[train_index], X[test_index]
#     y_train, y_test = y[train_index], y[test_index]
#
#     import keras
#     history = keras.callbacks.History()
#     print("Fitting...")
#     X_test_ = X_test.copy()
#     X_train_ = X_train.copy()
#     estimators = list()
#     estimators.append(('imputer', Imputer(missing_values='NaN', strategy='mean',
#                                           axis=0, verbose=2)))
#     estimators.append(('scaler', StandardScaler()))
#     estimators.append(('mlp', KerasClassifier(build_fn=create_baseline,
#                                               nb_epoch=epochs,
#                                               batch_size=batch_size,
#                                               verbose=0)))
#     pipeline = Pipeline(estimators)
#     for name, transform in pipeline.steps[:-1]:
#         print(name, transform)
#         transform.fit(X_train_)
#         X_test_ = transform.transform(X_test_)
#         X_train_ = transform.transform(X_train_)
#     pipeline.fit(X_train, y_train, mlp__validation_data=(X_test_, y_test),
#                  mlp__batch_size=batch_size, mlp__nb_epoch=epochs,
#                  mlp__callbacks=[history])
#     model = pipeline.named_steps['mlp']
#
#     y_pred = model.predict(X_test_)
#     y_pred[y_pred < 0.5] = 0.
#     y_pred[y_pred >= 0.5] = 1.
#     y_probs = model.predict_proba(X_test_)
#     cm = confusion_matrix(y_test, y_pred)
#     print_cm_summary(cm)
#
#     # summarize history for loss
#     plt.plot(history.history['acc'])
#     plt.plot(history.history['val_acc'])
#     plt.title('model accuracy')
#     plt.ylabel('accuracy')
#     plt.xlabel('epoch')
#     plt.legend(['train', 'test'], loc='upper left')
#     plt.show()
#     plt.plot(history.history['loss'])
#     plt.plot(history.history['val_loss'])
#     plt.title('model loss')
#     plt.ylabel('loss')
#     plt.xlabel('epoch')
#     plt.legend(['train', 'test'], loc='upper left')
#     plt.show()
