import hyperopt
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
import os
import numpy as np
from hyperopt import Trials, STATUS_OK, tpe
from keras import callbacks
from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.layers import Dropout
from keras.constraints import maxnorm
from keras.optimizers import SGD
from sklearn.cross_validation import (StratifiedShuffleSplit, StratifiedKFold,
                                      cross_val_score)
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Imputer
from sklearn.metrics import roc_auc_score, average_precision_score
from utils import print_cm_summary
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
# names_to_delete = ['Magnitude', 'meaningless_1', 'meaningless_2', 'star_ID',
#                    'Npts', 'CSSD']
names_to_delete = ['meaningless_1', 'meaningless_2', 'star_ID',
                   'Npts', 'CSSD', 'clipped_sigma', 'lag1', 'L', 'Lclp', 'Jclp',
                   'MAD', 'Ltim']
X, y, df, feature_names, delta = load_data([file_0, file_1], names, names_to_delete)
target = 'variable'
predictors = list(df)
predictors.remove(target)
dtrain = df

kfold = StratifiedKFold(dtrain[target], n_folds=4, shuffle=True, random_state=1)

sss = StratifiedShuffleSplit(y, n_iter=1, test_size=0.25, random_state=123)
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


def auc(y_true, y_pred):
    return roc_auc_score(y_true, y_pred)


def keras_fmin_fnct(space):
    print "Using hyperparameters ================================="
    print "==================================================="

    # Create and compile model
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
    learning_rate = space['lr']
    decay_rate = 0.001
    momentum = space['momentum']
    sgd = SGD(lr=learning_rate, decay=decay_rate, momentum=momentum,
              nesterov=False)
    model.compile(loss='binary_crossentropy', optimizer=sgd,
                  metrics=['accuracy'])

    # Save model to HDF5
    model.save('model.h5')
    del model

    # earlyStopping = callbacks.EarlyStopping(monitor='val_loss', patience=10,
    #                                         verbose=1, mode='auto')
    earlyStopping = callbacks.EarlyStopping(monitor='val_loss', patience=25,
                                            verbose=1, mode='auto')

    CMs = list()
    for train_index, test_index in kfold:
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        for name, transform in pipeline.steps:
            transform.fit(X_train)
            X_test = transform.transform(X_test)
            X_train = transform.transform(X_train)

        model = load_model('model.h5')
        model.fit(X_train, y_train,
                  batch_size=1024,
                  nb_epoch=1000,
                  verbose=2,
                  validation_data=(X_test, y_test),
                  callbacks=[earlyStopping],
                  class_weight={0: 1, 1: 2})
        # TODO: Use CV and cross_val_score
        # score, acc = model.evaluate(X_test, y_test, verbose=1)
        y_pred = model.predict(X_test, batch_size=1024)
        del model

        y_pred = [1. if y_ > 0.5 else 0. for y_ in y_pred]

        CMs.append(confusion_matrix(y_test, y_pred))

    CM = np.sum(CMs, axis=0)

    FN = CM[1][0]
    TP = CM[1][1]
    FP = CM[0][1]
    print "TP = {}".format(TP)
    print "FP = {}".format(FP)
    print "FN = {}".format(FN)

    f1 = 2. * TP / (2. * TP + FP + FN)
    print "F1: ", f1

    print "== F1 : {} ==".format(f1)

    return {'loss': 1-f1, 'status': STATUS_OK}

space = {
        'lr': hp.loguniform('lr', 1.1, 1.8),
        'momentum': hp.quniform('momentum', 0.5, 0.975, 0.025)
    }

trials = Trials()
best = fmin(fn=keras_fmin_fnct,
            space=space,
            algo=tpe.suggest,
            max_evals=100,
            trials=trials)

print hyperopt.space_eval(space, best)
best_pars = hyperopt.space_eval(space, best)

