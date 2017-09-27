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
estimators.append(('mlp', FixedKerasClassifier(build_fn=create_baseline,
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


vc = VotingClassifier(estimators=[("xgb", pipeline_xgb),
                                  ("nn", pipeline_nn),
                                  ("rf", pipeline_rf),
                                  ("svm", pipeline_svm),
                                  ("lr", pipeline_lr),
                                  ("knn", pipeline_knn)],
                      weights=[0.79, 0.81, 0.77, 0.80, 0.68, 0.68])


kfold = StratifiedKFold(n_splits=4, shuffle=True, random_state=1)
y_preds = cross_val_predict(vc, X, y, cv=kfold, n_jobs=4)

CMs = list()
for train_idx, test_idx in kfold.split(X, y):
    CMs.append(confusion_matrix(y[test_idx], y_preds[test_idx]))
CM = np.sum(CMs, axis=0)

FN = CM[1][0]
TP = CM[1][1]
FP = CM[0][1]
print("TP = {}".format(TP))
print("FP = {}".format(FP))
print("FN = {}".format(FN))

f1 = 2. * TP / (2. * TP + FP + FN)

# print "AUC: {}".format(auc)
print("F1={}".format(f1))
