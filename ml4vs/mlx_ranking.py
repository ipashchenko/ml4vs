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


def rank(y_probs):
    from scipy.stats import rankdata
    ranks = np.zeros(len(y_probs[0]), dtype=float)
    for y_prob in y_probs:
        rank = rankdata(y_prob, method='min') - 1
        ranks += rank
    return ranks / max(ranks)


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


kfold = StratifiedKFold(n_splits=4, shuffle=True, random_state=1)

y_prob_knn = cross_val_predict(pipeline_knn, X, y, cv=kfold, n_jobs=4,
                                method="predict_proba")[:, 1]
y_prob_lr = cross_val_predict(pipeline_lr, X, y, cv=kfold, n_jobs=4,
                               method="predict_proba")[:, 1]
# y_prob_rf = cross_val_predict(pipeline_rf, X, y, cv=kfold, n_jobs=4,
#                                method="predict_proba")[:, 1]
# y_prob_xgb = cross_val_predict(pipeline_xgb, X, y, cv=kfold, n_jobs=4,
#                                 method="predict_proba")[:, 1]
y_prob_svm = cross_val_predict(pipeline_svm, X, y, cv=kfold, n_jobs=4,
                                method="predict_proba")[:, 1]
y_prob_nn = cross_val_predict(pipeline_nn, X, y, cv=kfold, n_jobs=4,
                               method="predict_proba")[:, 1]

# new_prob = rank([y_prob_lr, y_prob_knn, y_prob_nn, y_prob_rf, y_prob_xgb,
#                  y_prob_svm])

print("nn+svm")
new_prob = rank([y_prob_nn,
                 y_prob_svm])
idx = np.array(new_prob >= 0.994449, dtype=int)

CMs = list()
for train_idx, test_idx in kfold.split(X, y):
    CMs.append(confusion_matrix(y[test_idx], idx[test_idx]))
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

