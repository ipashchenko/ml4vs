import hyperopt
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
import os
import numpy as np
import pandas as pd
from hyperopt import Trials, STATUS_OK, tpe
from keras import callbacks
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Activation
from keras.constraints import maxnorm
from keras.optimizers import SGD
from hyperas.distributions import choice, uniform, conditional, loguniform
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Imputer
from sklearn.metrics import roc_auc_score
from utils import print_cm_summary
from data_load import load_data, load_data_tgt


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
# names_to_delete = ['Magnitude', 'meaningless_1', 'meaningless_2', 'star_ID',
#                    'Npts', 'CSSD', 'Lclp', 'J', 'Jclp', 'rCh2', 'NXS', 'Ex',
#                    'inv_eta', 'weighted_sigma', 'Jtim',
#                    'Vp2p', 'E_A', 'kurt', 'K', 'Isgn', 'IQR']
X, y, df, feature_names, delta = load_data([file_0, file_1], names, names_to_delete)
target = 'variable'
predictors = list(df)
predictors.remove(target)


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
    model = Sequential()
    model.add(Dense(24, input_dim=24, init='normal', activation='relu',
                    W_constraint=maxnorm(space['w1'])))
    model.add(Dropout(space['Dropout']))
    model.add(Dense(space['Dense'], init='normal', activation='relu',
                    W_constraint=maxnorm(space['w2'])))
    # model.add(Activation(space['Activation']))
    model.add(Dropout(space['Dropout_1']))

    if conditional(space['conditional']) == 'three':
        model.add(Dense(12, activation='relu',
                        W_constraint=maxnorm(space['w3']),
                        init='normal'))
        model.add(Dropout(space['Dropout_2']))
    model.add(Dense(1, init='normal', activation='sigmoid'))

    # Compile model
    learning_rate = space["lr"]
    decay_rate = learning_rate / 400.
    momentum = space['momentum']
    sgd = SGD(lr=learning_rate, decay=decay_rate, momentum=momentum,
              nesterov=False)
    model.compile(loss='binary_crossentropy', optimizer=sgd,
                  metrics=['accuracy'])

    earlyStopping = callbacks.EarlyStopping(monitor='val_loss', patience=10,
                                            verbose=1, mode='auto')

    model.fit(X_train, y_train,
              batch_size=space['batch_size'],
              nb_epoch=400,
              show_accuracy=True,
              verbose=2,
              validation_data=(X_test, y_test),
              callbacks=[earlyStopping],
              class_weight={0: 1, 1: space['cw']})
    # TODO: Use CV and cross_val_score
    # score, acc = model.evaluate(X_test, y_test, verbose=1)
    pred = model.predict(X_test, batch_size=space['batch_size'])
    auc = roc_auc_score(y_test, pred)
    print('Test auc:', auc)
    return {'loss': 1-auc, 'status': STATUS_OK, 'model': model}


space = {
        'Dropout': hp.choice('Dropout', [0.42]),
        'Dense': hp.choice('Dense', [24]),
        # 'Activation': hp.choice('Activation', ['relu', 'sigmoid']),
        'Dropout_1': hp.choice('Dropout_1', [0.3]),
        'conditional': hp.choice('conditional', ['three']),
        'Dropout_2': hp.choice('Dropout_2', [0.56]),
        # 'optimizer': hp.choice('optimizer', ['rmsprop', 'adam', 'sgd']),
        'lr': hp.choice('lr', [0.075]),
        'w1': hp.choice('w1', [5]),
        'w2': hp.choice('w2', [3]),
        'w3': hp.choice('w3', [3]),
        'cw': hp.choice('cw', [2,3,4,5,6,7,8]),
        'momentum': hp.choice('momentum', [0.825]),
        'batch_size': hp.choice('batch_size', [256]),
    }

trials = Trials()
best = fmin(fn=keras_fmin_fnct,
            space=space,
            algo=tpe.suggest,
            max_evals=30,
            trials=trials)

print hyperopt.space_eval(space, best)
best_pars = hyperopt.space_eval(space, best)


# Now show plots
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

history = callbacks.History()
earlyStopping = callbacks.EarlyStopping(monitor='val_loss', patience=30,
                                        verbose=1, mode='auto')

# Build model with best parameters
model = Sequential()
model.add(Dense(24, input_dim=24, init='normal', activation='relu',
                W_constraint=maxnorm(best_pars['w1'])))
model.add(Dropout(best_pars['Dropout']))
model.add(Dense(best_pars['Dense'], init='normal', activation='relu',
                W_constraint=maxnorm(best_pars['w2'])))
# model.add(Activation(space['Activation']))
model.add(Dropout(best_pars['Dropout_1']))
if conditional(best_pars['conditional']) == 'three':
    model.add(Dense(12, activation='relu', W_constraint=maxnorm(best_pars['w3']),
                    init='normal'))
    model.add(Dropout(best_pars['Dropout_2']))
model.add(Dense(1, init='normal', activation='sigmoid'))

# Compile model
learning_rate = best_pars["lr"]
decay_rate = learning_rate / 400.
momentum = best_pars['momentum']
sgd = SGD(lr=learning_rate, decay=decay_rate, momentum=momentum,
          nesterov=False)
model.compile(loss='binary_crossentropy', optimizer=sgd,
              metrics=['accuracy'])
model.fit(X_train, y_train,
          batch_size=best_pars['batch_size'],
          nb_epoch=400,
          show_accuracy=True,
          verbose=2,
          validation_data=(X_test, y_test),
          callbacks=[earlyStopping, history],
          class_weight={0: 1, 1: best_pars['cw']})

n_epoch = history.epoch[-1]

y_pred = model.predict(X_test)
y_pred[y_pred < 0.5] = 0.
y_pred[y_pred >= 0.5] = 1.
y_probs = model.predict_proba(X_test)
cm = confusion_matrix(y_test, y_pred)
print_cm_summary(cm)


import matplotlib.pyplot as plt
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
plt.savefig("nn_accuracy_weights.png")
plt.close()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
plt.savefig("nn_loss_weights.png")
plt.close()

################################################################################
# Now fit all train data
transforms = list()
transforms.append(('imputer', Imputer(missing_values='NaN', strategy='median',
                                      axis=0, verbose=2)))
transforms.append(('scaler', StandardScaler()))
pipeline = Pipeline(transforms)
for name, transform in pipeline.steps:
    transform.fit(X)
    X = transform.transform(X)

# Build model with best parameters
model = Sequential()
model.add(Dense(24, input_dim=24, init='normal', activation='relu',
                W_constraint=maxnorm(best_pars['w1'])))
model.add(Dropout(best_pars['Dropout']))
model.add(Dense(best_pars['Dense'], init='normal', activation='relu',
                W_constraint=maxnorm(best_pars['w2'])))
# model.add(Activation(space['Activation']))
model.add(Dropout(best_pars['Dropout_1']))
if conditional(best_pars['conditional']) == 'three':
    model.add(Dense(12, activation='relu', W_constraint=maxnorm(best_pars['w3']),
                    init='normal'))
    model.add(Dropout(best_pars['Dropout_2']))
model.add(Dense(1, init='normal', activation='sigmoid'))
# Compile model
learning_rate = best_pars["lr"]
decay_rate = learning_rate / 400.
momentum = best_pars['momentum']
sgd = SGD(lr=learning_rate, decay=decay_rate, momentum=momentum,
          nesterov=False)
model.compile(loss='binary_crossentropy', optimizer=sgd,
              metrics=['accuracy'])
model.fit(X, y, batch_size=best_pars['batch_size'], nb_epoch=int(1.25*n_epoch),
          show_accuracy=True, verbose=2, class_weight={0: 1, 1: 1})

# Load blind test data
file_tgt = 'LMC_SC19_PSF_Pgood98__vast_lightcurve_statistics_normalized.log'
file_tgt = os.path.join(data_dir, file_tgt)
X_tgt, feature_names, df, df_orig = load_data_tgt(file_tgt, names, names_to_delete,
                                                  delta)
# Use the same transform
for name, transform in pipeline.steps:
    X_tgt = transform.transform(X_tgt)

y_probs = model.predict(X_tgt)[:, 0]
idx = y_probs > 0.5
idx_ = y_probs < 0.5
nn_no = list(df_orig['star_ID'][idx_])
print("Found {} variables".format(np.count_nonzero(idx)))

with open('nn_results.txt', 'w') as fo:
    for line in list(df_orig['star_ID'][idx]):
        fo.write(line + '\n')

# Analyze results
with open('clean_list_of_new_variables.txt', 'r') as fo:
    news = fo.readlines()
news = [line.strip().split(' ')[1] for line in news]
news = set(news)

with open('nn_results.txt', 'r') as fo:
    nn = fo.readlines()
nn = [line.strip().split('_')[4].split('.')[0] for line in nn]
nn = set(nn)

print "Among new vars found {}".format(len(news.intersection(nn)))

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
nn_no = set([line.strip().split('_')[4].split('.')[0] for line in nn_no])

found_bad = '181193' in nn
print "Found known variable : ", found_bad

FN = len(nn_no.intersection(all_vars))
TP = len(all_vars.intersection(nn))
TN = len(nn_no) - FN
FP = len(nn) - TP
recall = float(TP) / (TP + FN)
precision = float(TP) / (TP + FP)
F1 = 2 * precision * recall / (precision + recall)
print "precision: {}".format(precision)
print "recall: {}".format(recall)
print "F1: {}".format(F1)
print "TN={}, FP={}".format(TN, FP)
print "FN={}, TP={}".format(FN, TP)