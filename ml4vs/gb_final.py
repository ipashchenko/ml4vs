import os
import sys
import numpy as np
from sklearn.cross_validation import (StratifiedShuffleSplit, StratifiedKFold,
                                      cross_val_score)
sys.path.append('/home/ilya/xgboost/xgboost/python-package/')
import xgboost as xgb
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Imputer
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
dtrain = df

clf = xgb.XGBClassifier(n_estimators=85, learning_rate=0.087,
                        max_depth=6,
                        min_child_weight=3,
                        subsample=0.401,
                        colsample_bytree=0.729,
                        colsample_bylevel=0.863,
                        gamma=3.387,
                        scale_pos_weight=3.103,
                        max_delta_step=10, seed=1)

# Try using pipeline
estimators = list()
estimators.append(('imputer', Imputer(missing_values='NaN', strategy='median',
                                      axis=0, verbose=2)))
estimators.append(('clf', clf))
pipeline = Pipeline(estimators)

pipeline.fit(X, y)


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
y_pred = pipeline.predict(X_tgt)
y_probs = pipeline.predict_proba(X_tgt)[:, 1]

idx = y_probs > 0.5
idx_ = y_probs < 0.5
gb_no = list(df_orig['star_ID'][idx_])
print("Found {} variables".format(np.count_nonzero(idx)))

with open('gb_results_final.txt', 'w') as fo:
    for line in list(df_orig['star_ID'][idx]):
        fo.write(line + '\n')

# Check F1
with open('clean_list_of_new_variables.txt', 'r') as fo:
    news = fo.readlines()
news = [line.strip().split(' ')[1] for line in news]
news = set(news)

with open('gb_results_final.txt', 'r') as fo:
    gb = fo.readlines()
gb = [line.strip().split('_')[4].split('.')[0] for line in gb]
gb = set(gb)

print "Among new vars found {}".format(len(news.intersection(gb)))

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
gb_no = set([line.strip().split('_')[4].split('.')[0] for line in gb_no])

found_bad = '181193' in gb
print "Found known variable : ", found_bad

FN = len(gb_no.intersection(all_vars))
TP = len(all_vars.intersection(gb))
TN = len(gb_no) - FN
FP = len(gb) - TP
recall = float(TP) / (TP + FN)
precision = float(TP) / (TP + FP)
F1 = 2 * precision * recall / (precision + recall)
print "precision: {}".format(precision)
print "recall: {}".format(recall)
print "F1: {}".format(F1)
print "TN={}, FP={}".format(TN, FP)
print "FN={}, TP={}".format(FN, TP)
