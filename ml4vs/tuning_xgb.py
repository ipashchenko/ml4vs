import os
import pandas as pd
import numpy as np
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from sklearn import cross_validation, metrics   #Additional scklearn functions
from sklearn.grid_search import GridSearchCV   #Perforing grid search
from data_load import load_to_df
import matplotlib.pylab as plt
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 12, 4


# Load data
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
dtrain, delta = load_to_df([file_0, file_1], names, names_to_delete)
target = 'variable'
predictors = list(dtrain)
predictors.remove(target)


def modelfit(alg, dtrain, predictors, useTrainCV=True, cv_folds=5,
             early_stopping_rounds=50):

    if useTrainCV:
        xgb_param = alg.get_xgb_params()
        xgtrain = xgb.DMatrix(dtrain[predictors].values,
                              label=dtrain[target].values, missing=np.nan,
                              feature_names=predictors)
        cvresult = xgb.cv(xgb_param, xgtrain,
                          num_boost_round=alg.get_params()['n_estimators'],
                          nfold=cv_folds, metrics=['auc'],
                          early_stopping_rounds=early_stopping_rounds,
                          show_progress=False)
        alg.set_params(n_estimators=cvresult.shape[0])

    # Fit the algorithm on the data
    alg.fit(dtrain[predictors], dtrain[target], eval_metric='auc')

    # Predict training set:
    dtrain_predictions = alg.predict(dtrain[predictors])
    dtrain_predprob = alg.predict_proba(dtrain[predictors])[:, 1]

    # Print model report:
    print "\nModel Report"
    print "Accuracy : %.4g" % metrics.accuracy_score(dtrain[target],
                                                     dtrain_predictions)
    print "AUC Score (Train): %f" % metrics.roc_auc_score(dtrain[target],
                                                          dtrain_predprob)

    feat_imp = pd.Series(alg.booster().get_fscore()).sort_values(ascending=False)
    feat_imp.plot(kind='bar', title='Feature Importances')
    plt.ylabel('Feature Importance Score')
    plt.show()


xgb1 = XGBClassifier(learning_rate=0.1, n_estimators=1000, max_depth=5,
                     min_child_weight=1, gamma=0, subsample=0.8,
                     colsample_bytree=0.8, objective='binary:logistic',
                     nthread=4, scale_pos_weight=1, seed=27)
modelfit(xgb1, dtrain, predictors)


param_test1 = {
 'max_depth': range(3, 10, 2),
 'min_child_weight': range(1, 6, 2)
}
gsearch1 = GridSearchCV(estimator=XGBClassifier(learning_rate=0.1,
                                                n_estimators=69, max_depth=5,
 min_child_weight=1, gamma=0, subsample=0.8, colsample_bytree=0.8,
 objective='binary:logistic', nthread=4, scale_pos_weight=1, seed=27),
 param_grid=param_test1, scoring='roc_auc', n_jobs=4, iid=False, cv=4)
gsearch1.fit(dtrain[predictors], dtrain[target])
print gsearch1.grid_scores_, gsearch1.best_params_, gsearch1.best_score_