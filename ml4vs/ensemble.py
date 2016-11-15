# -*- coding: utf-8 -*-
import sys
sys.setrecursionlimit(10000)
import pandas as pd
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
import xgboost as xgb
from sklearn.ensemble import VotingClassifier
from sklearn.svm import SVC
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.constraints import maxnorm
from keras.optimizers import SGD
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import StandardScaler


class FixedKerasClassifier(KerasClassifier):
    def predict_proba(self, X, **kwargs):
        kwargs = self.filter_sk_params(Sequential.predict_proba, kwargs)
        probs = self.model.predict_proba(X, **kwargs)
        if(probs.shape[1] == 1):
            probs = np.hstack([1-probs,probs])
        return probs

    def predict(self, X, **kwargs):
        kwargs = self.filter_sk_params(Sequential.predict, kwargs)
        y = self.model.predict(X, **kwargs)
        if(y.shape[1] == 1):
            y = y[:, 0]
        return y



names = ['Magnitude', 'clipped_sigma', 'meaningless_1', 'meaningless_2',
         'star_ID', 'weighted_sigma', 'skew', 'kurt', 'I', 'J', 'K', 'L',
         'Npts', 'MAD', 'lag1', 'RoMS', 'rCh2', 'Isgn', 'Vp2p', 'Jclp', 'Lclp',
         'Jtim', 'Ltim', 'CSSD', 'Ex', 'inv_eta', 'E_A', 'S_B', 'NXS', 'IQR']

names_to_delete = ['Magnitude', 'meaningless_1', 'meaningless_2', 'star_ID']


def shift_log_transform(df, name, shift):
    df[name] = np.log(df[name] + shift)


def load_to_df(fnames, names, names_to_delete, target='variable'):
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
        Pandas data frame.
    """
    # Load data
    dfs = list()
    for fn in fnames:
        dfs.append(pd.read_table(fn, names=names, engine='python',
                                 na_values='+inf', sep=r"\s*",
                                 usecols=range(30)))
    df = pd.concat(dfs)
    y = np.zeros(len(df))
    y[len(dfs[0]):] = np.ones(len(df) - len(dfs[0]))

    df[target] = y

    # Remove meaningless features
    delta = min(df['CSSD'][np.isfinite(df['CSSD'].values)])
    # print delta
    print delta

    for name in names_to_delete:
        del df[name]
    try:
        shift_log_transform(df, 'CSSD', -delta + 0.1)
    except KeyError:
        pass

    return df, delta


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
    df_orig = pd.read_table(fname, names=names, engine='python', na_values='+inf',
                            sep=r"\s*", usecols=range(30))

    return X, features_names, df, df_orig


import os
# load data
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
                   'Npts', 'CSSD']

X, y, df, features_names, delta = load_data([file_0, file_1], names, names_to_delete)
target = 'variable'
predictors = list(df)
predictors.remove(target)
dtrain = df

sss = StratifiedShuffleSplit(n_splits=1, test_size=0.25, random_state=7)
for train_indx, test_indx in sss.split(dtrain[predictors].index, dtrain['variable']):
    print train_indx, test_indx
    train = dtrain.iloc[train_indx]
    valid = dtrain.iloc[test_indx]


clf_gb = xgb.XGBClassifier(n_estimators=100, learning_rate=0.1,
                           max_depth=9, gamma=0.74, colsample_bylevel=0.72,
                           colsample_bytree=0.58,
                           min_child_weight=1,
                           subsample=0.8)
clf_knn = KNeighborsClassifier(n_neighbors=362, weights='distance', leaf_size=22,
                               n_jobs=3)
clf_svm = SVC(C=16.5036, class_weight='balanced', probability=True,
              gamma=0.09138)


def create_baseline():
    # create model
    model = Sequential()
    model.add(Dense(24, input_dim=24, init='normal', activation='relu',
                    W_constraint=maxnorm(3)))
    model.add(Dropout(0.1))
    model.add(Dense(24, init='normal', activation='relu',
                    W_constraint=maxnorm(3)))
    model.add(Dropout(0.1))
    model.add(Dense(12, init='normal', activation='relu'))
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
clf_mlp = FixedKerasClassifier(build_fn=create_baseline, nb_epoch=epochs,
                               batch_size=batch_size, verbose=0)

# calibrated_clf_gb = CalibratedClassifierCV(clf_gb, method='sigmoid', cv=3)
# calibrated_clf_knn = CalibratedClassifierCV(clf_knn, method='sigmoid', cv=3)
# calibrated_clf_mlp = CalibratedClassifierCV(clf_mlp, method='sigmoid', cv=3)
# calibrated_clf_svm = CalibratedClassifierCV(clf_svm, method='sigmoid', cv=3)


# eclf = VotingClassifier(estimators=[('gb', calibrated_clf_gb),
#                                     ('knn', calibrated_clf_knn),
#                                     ('nn', calibrated_clf_mlp),
#                                     ('svm', calibrated_clf_svm)],
#                         voting='soft', weights=[1, 1, 1, 1], n_jobs=-1)
eclf = VotingClassifier(estimators=[('gb', clf_gb),
                                    ('knn', clf_knn),
                                    ('nn', clf_mlp),
                                    ('svm', clf_svm)],
                        voting='soft', weights=[2, 1, 1, 1])

estimators = list()
estimators.append(('imputer', Imputer(missing_values='NaN', strategy='median',
                                      axis=0, verbose=2)))
estimators.append(('scaler', StandardScaler()))
estimators.append(('clf', eclf))
pipeline = Pipeline(estimators)

valid_ = valid[predictors]
train_ = train[predictors]
for name, transform in pipeline.steps[:-1]:
    transform.fit(train_)
    valid_ = transform.transform(valid_)
    train_ = transform.transform(train_)

eclf.fit(train_, train['variable'])

# pred = eclf.predict_proba(valid_)[:, 1]
y_pred = eclf.predict(valid_)
# auc = roc_auc_score(valid['variable'], pred)
recall = recall_score(valid['variable'], y_pred)
pre = precision_score(valid['variable'], y_pred)
print "Pr, Re: {} {}".format(pre, recall)
# print "SCORE:", auc


# Fit full training set
train_ = dtrain[predictors]
for name, transform in pipeline.steps[:-1]:
    transform.fit(train_)
    train_ = transform.transform(train_)
eclf.fit(train_, dtrain['variable'])

# Load blind test data
file_tgt = 'LMC_SC19_PSF_Pgood98__vast_lightcurve_statistics_normalized.log'
file_tgt = os.path.join(data_dir, file_tgt)
X, feature_names, df, df_orig = load_data_tgt(file_tgt, names, names_to_delete,
                                              delta)
# Use fitted transformation steps
for name, transform in pipeline.steps[:-1]:
    print name, transform
    X = transform.transform(X)


y_pred = eclf.predict(X)
# y_probs = eclf.predict_proba(X)

idx = y_pred == 1.
# idx = y_probs[:, 1] > 0.5
# idx_ = y_probs[:, 1] < 0.5
# ens_no = list(df_orig['star_ID'][idx_])
print("Found {} variables".format(np.count_nonzero(idx)))

# with open('ens_results.txt', 'w') as fo:
#     for line in list(df_orig['star_ID'][idx]):
#         fo.write(line + '\n')
