import os
import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.append('/home/ilya/code/mlxtend')
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
# names = ['Magnitude', 'clipped_sigma', 'meaningless_1', 'meaningless_2',
#          'star_ID', 'weighted_sigma', 'skew', 'kurt', 'I', 'J', 'K', 'L',
#          'Npts', 'MAD', 'lag1', 'RoMS', 'rCh2', 'Isgn', 'Vp2p', 'Jclp', 'Lclp',
#          'Jtim', 'Ltim', 'CSSD', 'Ex', 'inv_eta', 'E_A', 'S_B', 'NXS', 'IQR']
names = [r'${\rm Magnitude}$', r'$\sigma_{\rm clip}$', 'meaningless_1', 'meaningless_2',
         'star_ID', r'$\sigma$', r'${\rm skewness}$', r'${\rm kurtosis}$',
         r'$I_{\rm WS}$', r'$J$', r'$K$', r'$L$',
         r'$N_{points}$', r'${\rm MAD}$', r'$l_1$', r'${\rm RoMS}$', r'$\chi_{\rm red}^2$',
         r'$I_{\rm fi}$', r'$v$',
         r'$J_{\rm clip}$', r'$L_{\rm clip}$',
         r'$J_{\rm time}$', r'$L_{\rm time}$', r'${\rm CSSD}$', r'$E_x$', r'$1/\eta$',
         r'$\mathcal{E}_\mathcal{A}$', r'$S_B$', r'$\sigma_{\rm NXS}^2$', r'${\rm IQR}$']
# names_to_delete = ['meaningless_1', 'meaningless_2', 'star_ID',
#                    'Npts', 'CSSD', 'clipped_sigma', 'lag1', 'L', 'Lclp', 'Jclp',
#                    'MAD', 'Ltim']
names_to_delete = ['meaningless_1', 'meaningless_2', 'star_ID']
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


clf_xgb = xgb.XGBClassifier(n_estimators=94, learning_rate=0.085,
                            max_depth=6,
                            min_child_weight=2.36,
                            subsample=0.439,
                            colsample_bytree=0.348,
                            colsample_bylevel=0.758,
                            gamma=4.159,
                            max_delta_step=2.85,
                            scale_pos_weight=4.087,
                            seed=1)

# Create model for NN
def create_baseline():
    model = Sequential()
    model.add(Dense(18, input_dim=18, init='normal', activation='relu',
                    W_constraint=maxnorm(9.038)))
    model.add(Dense(13, init='normal', activation='relu',
                    W_constraint=maxnorm(5.62)))
    # model.add(Activation(space['Activation']))
    model.add(Dropout(0.17))
    model.add(Dense(1, init='normal', activation='sigmoid'))

    # Compile model
    learning_rate = 0.2
    decay_rate = 0.001
    momentum = 0.95
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
                                          nb_epoch=100,
                                          batch_size=1024,
                                          verbose=2,
                                          class_weight={0: 1, 1: 2.03})))
clf_nn = Pipeline(estimators)


clf_rf = RandomForestClassifier(n_estimators=1400,
                                max_depth=16,
                                max_features=5,
                                min_samples_split=16,
                                min_samples_leaf=2,
                                class_weight={0: 1, 1: 28},
                                verbose=1, random_state=1, n_jobs=4)


# Create model for LR
# clf = LogisticRegression(C=10.5, class_weight={0: 1, 1: 1.585},
#                          random_state=1, n_jobs=1)
clf = LogisticRegression()
estimators = list()
estimators.append(('imputer', Imputer(missing_values='NaN', strategy='median',
                                      axis=0, verbose=2)))
estimators.append(('func', FunctionTransformer(log_axis, kw_args={'names':
                                                                      predictors})))
estimators.append(('scaler', StandardScaler()))
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
clf = SVC(C=25.053, class_weight={0: 1, 1: 2.93}, probability=False,
          gamma=0.0173, random_state=1)
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


cv = StratifiedShuffleSplit(n_splits=40, test_size=0.25, random_state=1)
from plotting import plot_learning_curve
fig_dict = {}
result_dict = {}
for clf, name in zip((clf_knn, clf_lr, clf_rf, clf_svm, clf_xgb, clf_nn),
                     ('kNN', 'LR', 'RF', 'RBF-SVM', 'SGB', 'NN')):
    if name not in ['LR']:
        continue
    fig, result = plot_learning_curve(clf, name, X, y, cv=cv, n_jobs=4,
                                      scoring='f1', return_scores=True)
    fig.show()
    fig_dict.update({name: fig})
    result_dict.update({name: result})
    fig.savefig(os.path.join('/home/ilya/Dropbox/papers/mlvs/new_pics/',
                             name + '_new.pdf'), bbox_inches='tight',
                dpi=1200, format='pdf')
    fig.savefig(os.path.join('/home/ilya/Dropbox/papers/mlvs/new_pics/',
                             name + '_new.eps'), bbox_inches='tight',
                dpi=1200, format='eps')
    fig.savefig(os.path.join('/home/ilya/Dropbox/papers/mlvs/new_pics/',
                             name + '_new.svg'), bbox_inches='tight',
                dpi=1200, format='svg')
    plt.close()

# import json
# with open('/home/ilya/Dropbox/papers/mlvs/new_pics/lc_dict.json', 'w') as fo:
#     json.dump(result_dict, fo)
import pickle
with open('/home/ilya/Dropbox/papers/mlvs/new_pics/lc_dict.json', 'r') as fo:
    result_dict_old = pickle.load(fo)
result_dict_old.update(result_dict)
result_dict = result_dict_old

import matplotlib
fig, axes = matplotlib.pyplot.subplots(nrows=2, ncols=3, sharex=True,
                                       sharey=True)
coords = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]
for name, coord in zip(('LR', 'kNN', 'RBF-SVM', 'RF', 'SGB', 'NN'), coords):
    axes[coord].fill_between(result_dict[name][0],
                             result_dict[name][1] - result_dict[name][2],
                             result_dict[name][1] + result_dict[name][2],
                             alpha=0.3, color="gray")
    axes[coord].fill_between(result_dict[name][0],
                             result_dict[name][3] - result_dict[name][4],
                             result_dict[name][3] + result_dict[name][4],
                             alpha=0.3, color="gray")
    axes[coord].plot(result_dict[name][0], result_dict[name][1], 'o-',
                     color="black")
    axes[coord].plot(result_dict[name][0], result_dict[name][3], 'o-.',
                     color="black")
    axes[coord].text(0.95, 0.01, name, verticalalignment='bottom',
                     horizontalalignment='right',
                     transform=axes[coord].transAxes, color='black',
                     fontsize=16)
fig.text(0.5, 0.01, r"Training sample size", ha='center', fontsize=16,
         color='black')
fig.text(0.03, 0.5, r"$F_1$-score", va='center', rotation='vertical',
         fontsize=16, color='black')

# fig.savefig('/home/ilya/Dropbox/papers/mlvs/new_pics/LC_nn100.pdf',
#             bbox_inches='tight', dpi=1200, format='pdf')
# fig.savefig('/home/ilya/Dropbox/papers/mlvs/new_pics/LC_nn100.eps',
#             bbox_inches='tight', dpi=1200, format='eps')
# fig.savefig('/home/ilya/Dropbox/papers/mlvs/new_pics/LC_nn100.ps',
#             bbox_inches='tight', dpi=1200, format='ps')
# fig.savefig('/home/ilya/Dropbox/papers/mlvs/new_pics/LC_nn100.svg',
#             bbox_inches='tight', dpi=1200, format='svg')
plt.savefig('/home/ilya/Dropbox/papers/mlvs/new_pics/corrplot_new.pdf',
         bbox_inches='tight', dpi=1200, format='pdf')
plt.savefig('/home/ilya/Dropbox/papers/mlvs/new_pics/corrplot_new.eps',
         bbox_inches='tight', dpi=1200, format='eps')
plt.savefig('/home/ilya/Dropbox/papers/mlvs/new_pics/corrplot_new.ps',
         bbox_inches='tight', dpi=1200, format='ps')
plt.savefig('/home/ilya/Dropbox/papers/mlvs/new_pics/corrplot_new.svg',
            bbox_inches='tight', dpi=1200, format='svg')
