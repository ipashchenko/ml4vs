import os
import numpy as np
import sys
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
label_size = 16
import matplotlib
matplotlib.rcParams['xtick.labelsize'] = label_size
matplotlib.rcParams['ytick.labelsize'] = label_size
matplotlib.rcParams['axes.titlesize'] = label_size
matplotlib.rcParams['axes.labelsize'] = label_size
matplotlib.rcParams['font.size'] = 16
matplotlib.rcParams['legend.fontsize'] = 16

from sklearn import decomposition
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import StandardScaler

# NN
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.constraints import maxnorm
from keras.optimizers import SGD
from keras import callbacks
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.svm import SVC

from data_load import load_data
from plotting import plot_cv_pr


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


# Load data
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

earlyStopping = callbacks.EarlyStopping(monitor='val_loss', patience=50,
                                        verbose=1, mode='auto')

estimators = list()
estimators.append(('imputer', Imputer(missing_values='NaN', strategy='median',
                                      axis=0, verbose=2)))
estimators.append(('scaler', StandardScaler()))
estimators.append(('mlp', KerasClassifier(build_fn=create_baseline,
                                          nb_epoch=100,
                                          batch_size=1024,
                                          verbose=2,
                                          class_weight={0: 1, 1: 2.03})))
pipeline_nn = Pipeline(estimators)


# Create model for GB
sys.path.append('/home/ilya/xgboost/xgboost/python-package/')
import xgboost as xgb
clf = xgb.XGBClassifier(n_estimators=94, learning_rate=0.085,
                        max_depth=6,
                        min_child_weight=2.36,
                        subsample=0.439,
                        colsample_bytree=0.348,
                        colsample_bylevel=0.758,
                        gamma=4.159,
                        scale_pos_weight=4.087,
                        max_delta_step=2,
                        reg_lambda=0.088)
estimators = list()
estimators.append(('imputer', Imputer(missing_values='NaN', strategy='median',
                                      axis=0, verbose=2)))
estimators.append(('clf', clf))
pipeline_xgb = Pipeline(estimators)


# Create model for RF
clf = RandomForestClassifier(n_estimators=1400,
                             max_depth=16,
                             max_features=5,
                             min_samples_split=16,
                             min_samples_leaf=2,
                             class_weight={0:1, 1:28},
                             verbose=1, random_state=1, n_jobs=4)
estimators = list()
estimators.append(('imputer', Imputer(missing_values='NaN', strategy='median',
                                      axis=0, verbose=2)))
estimators.append(('clf', clf))
pipeline_rf = Pipeline(estimators)


# Create model for LR
clf = LogisticRegression(C=50.78, class_weight={0: 1, 1: 2.65},
                         random_state=1, n_jobs=1)
# pca = decomposition.PCA(n_components=16, random_state=1)
estimators = list()
estimators.append(('imputer', Imputer(missing_values='NaN', strategy='median',
                                      axis=0, verbose=2)))
# estimators.append(('func', FunctionTransformer(log_axis, kw_args={'names':
#                                                                   predictors})))
estimators.append(('scaler', StandardScaler()))
# estimators.append(('pca', pca))
estimators.append(('clf', clf))
pipeline_lr = Pipeline(estimators)


# Create model for SVM
clf = SVC(C=25.053, class_weight={0: 1, 1: 2.93}, probability=True,
          gamma=0.0173, random_state=1)
estimators = list()
estimators.append(('imputer', Imputer(missing_values='NaN', strategy='median',
                                      axis=0, verbose=2)))
estimators.append(('scaler', StandardScaler()))
estimators.append(('clf', clf))
pipeline_svm = Pipeline(estimators)


# Create model for KNN
clf = KNeighborsClassifier(n_neighbors=6,
                           weights='distance', n_jobs=2)
estimators = list()
estimators.append(('imputer', Imputer(missing_values='NaN', strategy='median',
                                      axis=0, verbose=2)))
estimators.append(('scaler', StandardScaler()))
estimators.append(('clf', clf))
pipeline_knn = Pipeline(estimators)


fig = None
colors_dict = {pipeline_lr: 'lime', pipeline_rf: 'blue',
               pipeline_xgb: 'black', pipeline_nn: 'red',
               pipeline_knn: 'orange', pipeline_svm: 'magenta'}
labels_dict = {pipeline_rf: 'RF', pipeline_nn: 'NN', pipeline_xgb: 'SGB',
            pipeline_lr: 'LR', pipeline_knn: 'kNN', pipeline_svm: 'SVM'}
pipelines = [pipeline_lr, pipeline_rf, pipeline_xgb, pipeline_nn, pipeline_knn,
             pipeline_svm]

for i, pipeline in enumerate(pipelines):

    fig = plot_cv_pr(pipeline, X, y, seeds=range(1, 97, 8),
                     plot_color=colors_dict[pipeline],
                     fig=fig, label=labels_dict[pipeline])

patches = list()
for pipeline in pipelines:
    patches.append(mpatches.Patch(color=colors_dict[pipeline],
                                  label=labels_dict[pipeline]))
plt.legend(handles=patches, loc="lower left")
plt.show()

import os
path = '/home/ilya/Dropbox/papers/mlvs/new_pics/'
fig.savefig(os.path.join(path, 'auprc_all_update2.svg'), format='svg', dpi=1200)
fig.savefig(os.path.join(path, 'auprc_all_update2.pdf'), format='pdf', dpi=1200)
fig.savefig(os.path.join(path, 'auprc_all_update2.eps'), format='eps', dpi=1200)
