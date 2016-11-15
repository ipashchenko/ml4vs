# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
from hyperopt import Trials, STATUS_OK, tpe
import numpy
from keras import callbacks
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Activation
from keras.constraints import maxnorm
from keras.optimizers import SGD
from keras.metrics import fbeta_score
from keras.wrappers.scikit_learn import KerasClassifier
from hyperas import optim
from hyperas.distributions import choice, uniform, conditional, loguniform
from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import StratifiedShuffleSplit, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Imputer


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
    df_orig = pd.read_table(fname, names=names, engine='python', na_values='+inf',
                            sep=r"\s*", usecols=range(30))

    return X, features_names, df, df_orig


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


def data():
    """
    Data providing function:

    This function is separated from model() so that hyperopt
    won't reload data for each evaluation run.
    """
    sss = StratifiedShuffleSplit(y, n_iter=1, test_size=0.25, random_state=seed)
    for train_index, test_index in sss:
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

    transforms = list()
    transforms.append(('imputer', Imputer(missing_values='NaN', strategy='median',
                                          axis=0, verbose=2)))
    transforms.append(('scaler', StandardScaler()))
    pipeline = Pipeline(transforms)

    for name, transform in pipeline.steps:
        transform.fit(X_train)
        X_test = transform.transform(X_test)
        X_train = transform.transform(X_train)

    return X_train, y_train, X_test, y_test


def model(X_train, y_train, X_test, y_test):
    '''
    Model providing function:

    Create Keras model with double curly brackets dropped-in as needed.
    Return value has to be a valid python dictionary with two customary keys:
        - loss: Specify a numeric evaluation metric to be minimized
        - status: Just use STATUS_OK and see hyperopt documentation if not feasible
    The last one is optional, though recommended, namely:
        - model: specify the model just created so that we can later use it again.
    '''

    # create model
    model = Sequential()
    model.add(Dense(25, input_dim=25, init='normal', activation='relu',
                    W_constraint=maxnorm(3)))
    model.add(Dropout({{uniform(0, 1)}}))
    model.add(Dense({{choice([12, 25, 50])}}, init='normal',
                    W_constraint=maxnorm(3)))
    model.add(Activation({{choice(['relu', 'sigmoid'])}}))
    model.add(Dropout({{uniform(0, 1)}}))

    if conditional({{choice(['two', 'three'])}}) == 'three':
        model.add(Dense({{choice(12, 25)}}, init='normal'))
        model.add(Activation({{choice(['relu', 'sigmoid'])}}))
        model.add(Dropout({{uniform(0, 1)}}))
    model.add(Dense(1, init='normal', activation='sigmoid'))
    # Compile model
    learning_rate = {{loguniform(-7, -2)}}
    decay_rate = 0.00001
    momentum = {{uniform(0.50, 1)}}
    sgd = SGD(lr=learning_rate, decay=decay_rate, momentum=momentum,
              nesterov=False)
    model.compile(loss='binary_crossentropy', optimizer=sgd,
                  metrics=[fbeta_score])

    model.fit(X_train, y_train,
              batch_size={{choice([64, 128])}},
              nb_epoch=epochs,
              show_accuracy=True,
              verbose=2,
              validation_data=(X_test, y_test))
    score, f1 = model.evaluate(X_test, y_test, verbose=0)
    print('Test accuracy:', f1)
    return {'loss': 1-f1, 'status': STATUS_OK, 'model': model}


if __name__ == '__main__':
    best_run, best_model = optim.minimize(model=model,
                                          data=data,
                                          algo=tpe.suggest,
                                          max_evals=5,
                                          trials=Trials())
    X_train, Y_train, X_test, Y_test = data()
    print("Evalutation of best performing model:")
    print(best_model.evaluate(X_test, Y_test))