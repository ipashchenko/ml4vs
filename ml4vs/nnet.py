# -*- coding: utf-8 -*-
import glob
import os
import numpy as np
import pandas as pd

from sklearn_evaluation.plot import confusion_matrix as plot_cm

# from data_load import load_data
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


def shift_log_transform(df, name, shift):
    df[name] = np.log(df[name] + shift)


def load_data(fnames, names, names_to_delete):
    """
    Function that loads data from series of files where first file contains
    class of zeros and other files - classes of ones.

    :param fnames:
        Iterable of file names.
    :param names:
        Names of columns in files.
    :param names_to_delete:
        Column names to delete.
    :return:
        X, y - ``sklearn`` arrays of features & responces.
    """
    # Load data
    dfs = list()
    for fn in fnames:
        dfs.append(pd.read_table(fn, names=names, engine='python',
                                 na_values='+inf', sep=r"\s*",
                                 usecols=range(30)))

    # Remove meaningless features
    delta = list()
    for df in dfs:
        delta.append(df['CSSD'].min())
    delta = np.min([d for d in delta if not np.isinf(d)])
    print "delta = {}".format(delta)

    for df in dfs:
        for name in names_to_delete:
            del df[name]
        try:
            shift_log_transform(df, 'CSSD', -delta + 0.1)
        except KeyError:
            pass

    # List of feature names
    features_names = list(dfs[0])
    # Count number of NaN for each feature
    for i, df in enumerate(dfs):
        print("File {}".format(i))
        for feature in features_names:
            print("Feature {} has {} NaNs".format(feature,
                                                  df[feature].isnull().sum()))
        print("=======================")

    # Convert to numpy arrays
    # Features
    X = list()
    for df in dfs:
        X.append(np.array(df[list(features_names)].values, dtype=float))
    X = np.vstack(X)
    # Responses
    y = np.zeros(len(X))
    y[len(dfs[0]):] = np.ones(len(X) - len(dfs[0]))

    df = pd.concat(dfs)
    df['variable'] = y

    return X, y, df, features_names, delta


def load_data_tgt(fname, names, names_to_delete, delta):
    """
    Function that loads target data for classification.

    :param fname:
        Target data file.
    :param names:
        Names of columns in files.
    :param names_to_delete:
        Column names to delete.
    :return:
        X, ``sklearn`` array of features, list of feature names
    """
    # Load data
    df = pd.read_table(fname, names=names, engine='python', na_values='+inf',
                       sep=r"\s*", usecols=range(30))

    for name in names_to_delete:
        del df[name]
    try:
        shift_log_transform(df, 'CSSD', -delta + 0.1)
    except KeyError:
        pass

    # List of feature names
    features_names = list(df)
    # Count number of NaN for each feature
    for feature in features_names:
        print("Feature {} has {} NaNs".format(feature,
                                              df[feature].isnull().sum()))
    print("=======================")

    # Convert to numpy arrays
    # Features
    X = np.array(df[list(features_names)].values, dtype=float)

    # Original data
    df = pd.read_table(fname, names=names, engine='python', na_values='+inf',
                       sep=r"\s*", usecols=range(30))

    return X, features_names, df


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
X, y, df, feature_names, delta = load_data([file_0, file_1], names, names_to_delete)

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
skf = StratifiedKFold(y, n_folds=4, shuffle=True, random_state=seed)

pipeline = Pipeline(estimators)

results = cross_val_score(pipeline, X, y, cv=skf, scoring='f1', n_jobs=3)
print("\n")
print(results)
print("\n")
print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
print("\n")

results = cross_val_score(pipeline, X, y, cv=skf, scoring='roc_auc', n_jobs=3)
print("\n")
print(results)
print("\n")
print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
print("\n")


# Load blind test data
file_tgt = 'LMC_SC19_PSF_Pgood98__vast_lightcurve_statistics_normalized.log'
file_tgt = os.path.join(data_dir, file_tgt)
X_tgt, feature_names, df = load_data_tgt(file_tgt, names, names_to_delete,
                                         delta)
pipeline.fit(X, y, mlp__batch_size=batch_size, mlp__nb_epoch=epochs)
model = pipeline.named_steps['mlp']

y_pred = model.predict(X_tgt)
y_probs = model.predict_proba(X_tgt)

idx = y_probs[:, 1] > 0.5
idx_ = y_probs[:, 1] < 0.5
nns_no = list(df['star_ID'][idx_])
print("Found {} variables".format(np.count_nonzero(idx)))

with open('nn_results.txt', 'w') as fo:
    for line in list(df['star_ID'][idx]):
        fo.write(line + '\n')

# Found negatives
nns_no = set([line.strip().split('_')[4].split('.')[0] for line in nns_no])

with open('clean_list_of_new_variables.txt', 'r') as fo:
    news = fo.readlines()
news = [line.strip().split(' ')[1] for line in news]

with open('nn_results.txt', 'r') as fo:
    nns = fo.readlines()
nns = [line.strip().split('_')[4].split('.')[0] for line in nns]

nns = set(nns)
# New variables discovered by GBC
news = set(news)
# 11 new variables are found
len(news.intersection(nns))
# It was found
'181193' in nns

with open('candidates_50perc_threshold.txt', 'r') as fo:
    c50 = fo.readlines()
c50 = [line.strip("\", ', \", \n, }, {") for line in c50]

with open('variables_not_in_catalogs.txt', 'r') as fo:
    not_in_cat = fo.readlines()
nic = [line.strip().split(' ')[1] for line in not_in_cat]

# Catalogue variables
cat_vars = set(c50).difference(set(nic))
# Non-catalogue variable
noncat_vars = set([line.strip().split(' ')[1] for line in not_in_cat if 'CST' not in line])

# All variables
all_vars = news.union(cat_vars).union(noncat_vars)
# Number of true positives
# 145
len(all_vars.intersection(nns))
# Number of false negatives
# 43
len(nns_no.intersection(all_vars))


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
