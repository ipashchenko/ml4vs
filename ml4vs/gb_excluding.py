import os
import sys
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix

sys.path.append('/home/ilya/xgboost/xgboost/python-package/')
import xgboost as xgb
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Imputer
from sklearn.metrics import f1_score
from data_load import load_data, load_data_tgt
from plotting import plot_importance_xgb


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


def xg_f1(y, t):
    t = t.get_label()
    # Binaryzing your output
    y_bin = [1. if y_cont > 0.5 else 0. for y_cont in y]
    return 'f1', 1-f1_score(t, y_bin)

kfold = StratifiedKFold(n_splits=4, shuffle=True, random_state=1)

clf = xgb.XGBClassifier(n_estimators=1000, learning_rate=0.111,
                        max_depth=6,
                        min_child_weight=2,
                        subsample=0.275,
                        colsample_bytree=0.85,
                        colsample_bylevel=0.55,
                        gamma=3.14,
                        max_delta_step=7,
                        scale_pos_weight=6,
                        seed=1)

xgb_param = clf.get_xgb_params()
xgtrain = xgb.DMatrix(dtrain[predictors].values,
                      label=dtrain[target].values)
cvresult = xgb.cv(xgb_param, xgtrain,
                  num_boost_round=clf.get_params()['n_estimators'],
                  folds=kfold, feval=xg_f1,
                  early_stopping_rounds=50, verbose_eval=True,
                  as_pandas=False, seed=0)
print "F1:", 1-cvresult['test-f1-mean'][-1]


# pipeline.fit(X, y)
# plot_importance_xgb(clf, feature_names)
