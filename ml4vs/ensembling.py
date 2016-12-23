import os
import numpy as np
import sys
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

from sklearn import decomposition, pipeline
from sklearn.cross_validation import StratifiedShuffleSplit
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
names_to_delete = ['meaningless_1', 'meaningless_2', 'star_ID',
                   'Npts', 'CSSD', 'clipped_sigma', 'lag1', 'L', 'Lclp', 'Jclp',
                   'MAD', 'Ltim']
X, y, df, feature_names, delta = load_data([file_0, file_1], names,
                                           names_to_delete)
target = 'variable'
predictors = list(df)
predictors.remove(target)

# Split on train/test
sss = StratifiedShuffleSplit(y, n_iter=1, test_size=0.25, random_state=123)
for train_index, test_index in sss:
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
# Split on train1/train2
sss_ = StratifiedShuffleSplit(y_train, n_iter=1, test_size=0.5,
                              random_state=123)
for train_index, test_index in sss_:
    X_train_, X_test_ = X_train[train_index], X_train[test_index]
    y_train_, y_test_ = y_train[train_index], y_train[test_index]

# Fit algos on train_


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


# Create model for NN
def create_baseline():
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
    learning_rate = 0.213
    decay_rate = 0.001
    momentum = 0.9
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
                                          nb_epoch=175,
                                          batch_size=1024,
                                          verbose=2)))
pipeline_nn = Pipeline(estimators)


# Create model for GB
sys.path.append('/home/ilya/xgboost/xgboost/python-package/')
import xgboost as xgb
clf = xgb.XGBClassifier(n_estimators=87, learning_rate=0.111,
                        max_depth=6,
                        min_child_weight=2,
                        subsample=0.275,
                        colsample_bytree=0.85,
                        colsample_bylevel=0.55,
                        gamma=3.14,
                        scale_pos_weight=6,
                        max_delta_step=6)
estimators = list()
estimators.append(('imputer', Imputer(missing_values='NaN', strategy='median',
                                      axis=0, verbose=2)))
estimators.append(('clf', clf))
pipeline_xgb = Pipeline(estimators)


# Create model for RF
clf = RandomForestClassifier(n_estimators=1200,
                             max_depth=17,
                             max_features=3,
                             min_samples_split=2,
                             min_samples_leaf=3,
                             class_weight='balanced_subsample',
                             verbose=1, random_state=1, n_jobs=4)
estimators = list()
estimators.append(('imputer', Imputer(missing_values='NaN', strategy='median',
                                      axis=0, verbose=2)))
estimators.append(('clf', clf))
pipeline_rf = Pipeline(estimators)

# Create model for LR
clf = LogisticRegression(C=1.29, class_weight={0: 1, 1: 2},
                         random_state=1, max_iter=300, n_jobs=1,
                         tol=10.**(-5))
pca = decomposition.PCA(n_components=16, random_state=1)
estimators = list()
estimators.append(('imputer', Imputer(missing_values='NaN', strategy='median',
                                      axis=0, verbose=2)))
estimators.append(('func', FunctionTransformer(log_axis, kw_args={'names':
                                                                      predictors})))
estimators.append(('scaler', StandardScaler()))
estimators.append(('pca', pca))
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

# Model for SVM
clf = SVC(C=37.3, class_weight={0: 1, 1: 3}, probability=True,
          gamma=0.0126, random_state=1)
estimators = list()
estimators.append(('imputer', Imputer(missing_values='NaN', strategy='median',
                                      axis=0, verbose=2)))
estimators.append(('scaler', StandardScaler()))
estimators.append(('clf', clf))
pipeline_svm = Pipeline(estimators)

# Fit on all training data
pipeline_lr.fit(X, y)
pipeline_knn.fit(X, y)
pipeline_rf.fit(X, y)
pipeline_xgb.fit(X, y)
pipeline_nn.fit(X, y)
pipeline_svm.fit(X, y)

# Load blind test data
file_tgt = 'LMC_SC19_PSF_Pgood98__vast_lightcurve_statistics.log'
file_tgt = os.path.join(data_dir, file_tgt)
names = ['Magnitude', 'clipped_sigma', 'meaningless_1', 'meaningless_2',
         'star_ID', 'weighted_sigma', 'skew', 'kurt', 'I', 'J', 'K', 'L',
         'Npts', 'MAD', 'lag1', 'RoMS', 'rCh2', 'Isgn', 'Vp2p', 'Jclp', 'Lclp',
         'Jtim', 'Ltim', 'CSSD', 'Ex', 'inv_eta', 'E_A', 'S_B', 'NXS', 'IQR']
names_to_delete = ['meaningless_1', 'meaningless_2', 'star_ID',
                   'Npts', 'CSSD', 'clipped_sigma', 'lag1', 'L', 'Lclp', 'Jclp',
                   'MAD', 'Ltim']
X_tgt, feature_names, df, df_orig = load_data_tgt(file_tgt, names,
                                                  names_to_delete, delta)
# Predict for different algos
y_pred_rf = pipeline_rf.predict(X_tgt)
y_pred_lr = pipeline_lr.predict(X_tgt)
y_pred_knn = pipeline_knn.predict(X_tgt)
y_pred_xgb = pipeline_xgb.predict(X_tgt)
y_pred_nn = pipeline_nn.predict(X_tgt)[:, 0]
y_pred_svm = pipeline_svm.predict(X_tgt)
# Probabilities
y_prob_rf = pipeline_rf.predict_proba(X_tgt)[:, 1]
y_prob_lr = pipeline_lr.predict_proba(X_tgt)[:, 1]
y_prob_knn = pipeline_knn.predict_proba(X_tgt)[:, 1]
y_prob_xgb = pipeline_xgb.predict_proba(X_tgt)[:, 1]
y_prob_nn = pipeline_nn.predict_proba(X_tgt)[:, 1]
y_prob_knn = pipeline_nn.predict_proba(X_tgt)[:, 1]
y_prob_svm = pipeline_nn.predict_proba(X_tgt)[:, 1]

y_preds = (0.67*y_pred_lr + 0.68*y_pred_knn + 0.81*y_pred_xgb +
           0.76*y_pred_rf + 0.81*y_pred_nn + 0.79*y_pred_svm) / 4.52
y_preds = np.asarray(y_preds, dtype=float)
# y_preds /= 6

idx = y_preds >= 0.5
idx_ = y_preds < 0.5

def rank(y_probs):
    from scipy.stats import rankdata
    ranks = np.zeros(len(y_probs[0]), dtype=float)
    for y_prob in y_probs:
        rank = rankdata(y_prob, method='min') - 1
        ranks += rank
    return ranks / max(ranks)

# new_prob = rank([y_prob_lr, y_prob_knn, y_prob_nn, y_prob_rf, y_prob_xgb,
#                  y_pred_svm])
# idx = new_prob >= 0.95
# idx_ = new_prob < 0.95

# idx = y_probs > 0.250
# idx_ = y_probs < 0.250
ens_no = list(df_orig['star_ID'][idx_])
print("Found {} variables".format(np.count_nonzero(idx)))

with open('ens_results_final.txt', 'w') as fo:
    for line in list(df_orig['star_ID'][idx]):
        fo.write(line + '\n')

# Check F1
with open('clean_list_of_new_variables.txt', 'r') as fo:
    news = fo.readlines()
news = [line.strip().split(' ')[1] for line in news]
news = set(news)

with open('ens_results_final.txt', 'r') as fo:
    ens = fo.readlines()
ens = [line.strip().split('_')[4].split('.')[0] for line in ens]
ens = set(ens)

print "Among new vars found {}".format(len(news.intersection(ens)))

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
ens_no = set([line.strip().split('_')[4].split('.')[0] for line in ens_no])

found_bad = '181193' in ens
print "Found known variable : ", found_bad

FN = len(ens_no.intersection(all_vars))
TP = len(all_vars.intersection(ens))
TN = len(ens_no) - FN
FP = len(ens) - TP
recall = float(TP) / (TP + FN)
precision = float(TP) / (TP + FP)
F1 = 2 * precision * recall / (precision + recall)
print "precision: {}".format(precision)
print "recall: {}".format(recall)
print "F1: {}".format(F1)
print "TN={}, FP={}".format(TN, FP)
print "FN={}, TP={}".format(FN, TP)
