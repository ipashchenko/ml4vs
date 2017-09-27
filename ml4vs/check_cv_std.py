import os
import numpy as np
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix

# NN
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.constraints import maxnorm
from keras.optimizers import SGD
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.svm import SVC

from mlxtend.classifier import StackingClassifier
from data_load import load_data

import hyperopt
from hyperopt import hp, fmin, tpe, STATUS_OK, Trials

import scipy.cluster.hierarchy as hierarchy
import scipy.spatial.distance as distance
import easydev

import string
from colormap import cmap_builder
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.patches import Ellipse, Circle, Rectangle, Wedge
from matplotlib.collections import PatchCollection
import pandas as pd
import scipy.cluster.hierarchy as hierarchy


data_dir = '/home/ilya/github/ml4vs/data/LMC_SC20__corrected_list_of_variables/raw_index_values'
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


vint = np.vectorize(int)


class FixedKerasClassifier(KerasClassifier):
    def predict(self, X, **kwargs):
        kwargs = self.filter_sk_params(Sequential.predict_classes, kwargs)
        classes = self.model.predict_classes(X, **kwargs)
        y = self.classes_[classes]
        if y.shape[1] == 1:
            y = y[:, 0]
        return vint(y)


# Create model for NN
def create_baseline():
    model = Sequential()
    model.add(Dense(18, input_dim=18, init='normal', activation='relu',
                    W_constraint=maxnorm(9.04)))
    model.add(Dense(13, init='normal', activation='relu',
                    W_constraint=maxnorm(5.62)))
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
                                               epochs=60, batch_size=1024,
                                               verbose=2,
                                               class_weight={0: 1, 1: 2.03})))


pipeline_nn = Pipeline(estimators)

# Create model for RF
clf = RandomForestClassifier(n_estimators=1400,
                             max_depth=16,
                             max_features=5,
                             min_samples_split=16,
                             min_samples_leaf=2,
                             class_weight={0: 1, 1: 28},
                             verbose=1, random_state=1, n_jobs=4)
estimators = list()
estimators.append(('imputer', Imputer(missing_values='NaN', strategy='median',
                                      axis=0, verbose=2)))
estimators.append(('clf', clf))
pipeline_rf = Pipeline(estimators)

# Model for SVM
clf = SVC(C=25.05, class_weight={0: 1, 1: 2.93}, probability=True,
          gamma=0.017, random_state=1)
estimators = list()
estimators.append(('imputer', Imputer(missing_values='NaN', strategy='median',
                                      axis=0, verbose=2)))
estimators.append(('scaler', StandardScaler()))
estimators.append(('clf', clf))
pipeline_svm = Pipeline(estimators)

# Create model for GB
# sys.path.append('/home/ilya/xgboost/xgboost/python-package/')
import xgboost as xgb
clf = xgb.XGBClassifier(n_estimators=94, learning_rate=0.085,
                        max_depth=6,
                        min_child_weight=2.36,
                        subsample=0.44,
                        colsample_bytree=0.35,
                        colsample_bylevel=0.76,
                        gamma=4.16,
                        scale_pos_weight=4.09,
                        max_delta_step=2,
                        reg_lambda=0.09)
estimators = list()
estimators.append(
    ('imputer', Imputer(missing_values='NaN', strategy='median',
                        axis=0, verbose=2)))
estimators.append(('clf', clf))
pipeline_xgb = Pipeline(estimators)


# Create model for LR
clf = LogisticRegression(C=50.78, class_weight={0: 1, 1: 2.65},
                         random_state=1)
estimators = list()
estimators.append(('imputer', Imputer(missing_values='NaN', strategy='median',
                                      axis=0, verbose=2)))
estimators.append(('scaler', StandardScaler()))
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


f_dict = {}
for algo in ('LR', 'kNN', 'RF', 'SGB', 'SVM', 'NN'):
    f_dict[algo] = list()

for random_state in np.arange(1, 301, 10):
    kfold = StratifiedKFold(n_splits=4, shuffle=True, random_state=random_state)
    y_preds_knn = cross_val_predict(pipeline_knn, X, y, cv=kfold, n_jobs=4)
    y_preds_lr = cross_val_predict(pipeline_lr, X, y, cv=kfold, n_jobs=4)
    y_preds_rf = cross_val_predict(pipeline_rf, X, y, cv=kfold, n_jobs=4)
    y_preds_xgb = cross_val_predict(pipeline_xgb, X, y, cv=kfold, n_jobs=4)
    y_preds_svm = cross_val_predict(pipeline_svm, X, y, cv=kfold, n_jobs=4)
    y_preds_nn = cross_val_predict(pipeline_nn, X, y, cv=kfold, n_jobs=4)
    for clf, y_pred in zip(('LR', 'kNN', 'RF', 'SGB', 'SVM', 'NN'),
                           (y_preds_lr, y_preds_knn, y_preds_rf, y_preds_xgb,
                            y_preds_svm, y_preds_nn)):
        print(clf)
        CMs = list()
        for train_idx, test_idx in kfold.split(X, y):
            CMs.append(confusion_matrix(y[test_idx], y_pred[test_idx]))
        CM = np.sum(CMs, axis=0)

        FN = CM[1][0]
        TP = CM[1][1]
        FP = CM[0][1]
        print("TP = {}".format(TP))
        print("FP = {}".format(FP))
        print("FN = {}".format(FN))

        f1 = 2. * TP / (2. * TP + FP + FN)
        f_dict[clf].append(f1)

        # print "AUC: {}".format(auc)
        print("{} F1={}".format(clf, f1))

for algo in ('LR', 'kNN', 'RF', 'SGB', 'SVM', 'NN'):
    mean = np.mean(f_dict[algo])
    std = np.std(f_dict[algo])
    print("For {} F_mean={}, F_std={}".format(algo, mean, std))

import pickle
with open("/home/ilya/github/ml4vs/ml4vs/cv_std.pkl", "w") as fo:
    pickle.dump(f_dict, fo)


with open("/home/ilya/github/ml4vs/ml4vs/cv_std.pkl", "r") as fo:
    f_dict = pickle.load(fo)
import matplotlib.pyplot as plt
boxes = list()
labels = ('LR', 'kNN', 'RF', 'SGB', 'SVM', 'NN')
colors = ('green', 'orange', 'blue', 'black', 'magenta', 'red')
double_colors = [color for color in colors for _ in (0, 1)]
for clf in labels:
    boxes.append(f_dict[clf])

# Create a figure instance
fig = plt.figure()

# Create an axes instance
ax = fig.add_subplot(111)

bp = ax.boxplot(boxes, notch=True, vert=True, bootstrap=1000)
for i, box in enumerate(bp['boxes']):
    # change outline color
    box.set(color=colors[i], linewidth=2)
    # change fill color
    # box.set(facecolor=colors[i], alpha=0.5)

## change color and linewidth of the whiskers
for i, whisker in enumerate(bp['whiskers']):
    whisker.set(linewidth=2, color=double_colors[i])

## change color and linewidth of the caps
for i, cap in enumerate(bp['caps']):
    cap.set(linewidth=2, color=double_colors[i])

## change color and linewidth of the medians
for i, median in enumerate(bp['medians']):
    median.set(linewidth=2, color=colors[i])

## change the style of fliers and their fill
for flier in bp['fliers']:
    flier.set(marker='o', color=colors[i])

## Custom x-axis labels
ax.set_xticklabels(labels, fontsize=16)
ax.set_yticks([0.7, 0.8])
ax.set_yticklabels([0.7, 0.8], fontsize=16)
## Remove top axes and right axes ticks
ax.get_xaxis().tick_bottom()
ax.get_yaxis().tick_left()

ax.set_ylabel("$F_1^{CV}$", fontsize=16)

plt.savefig('/home/ilya/Dropbox/papers/mlvs/submitted/cv_boxplot.pdf',
         bbox_inches='tight', dpi=1200, format='pdf')
plt.savefig('/home/ilya/Dropbox/papers/mlvs/submitted/cv_boxplot.eps',
         bbox_inches='tight', dpi=1200, format='eps')
plt.savefig('/home/ilya/Dropbox/papers/mlvs/submitted/cv_boxplot.ps',
         bbox_inches='tight', dpi=1200, format='ps')