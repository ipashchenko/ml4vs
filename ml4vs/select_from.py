import os
import numpy as np
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.preprocessing import Imputer
from xgboost import XGBClassifier
from data_load import load_data


# load data
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
                   'Npts']
X, y, df, feature_names, delta = load_data([file_0, file_1], names,
                                           names_to_delete)

imp = Imputer(missing_values='NaN', strategy='median', axis=0, verbose=2)
model = XGBClassifier(max_depth=5, min_child_weight=1, gamma=0, subsample=0.8,
                      colsample_bytree=0.8, scale_pos_weight=150,
                      learning_rate=0.03, n_estimators=450)

sss = StratifiedShuffleSplit(y, n_iter=1, test_size=0.25, random_state=7)
for train_index, test_index in sss:
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
imp.fit(X_train)
X_trained_scaled = imp.transform(X_train)
# Use the same transformation & imputation for testing data!
X_test_scaled = imp.transform(X_test)
eval_set = [(X_trained_scaled, y_train), (X_test_scaled, y_test)]

model.fit(X_trained_scaled, y_train)
y_pred = model.predict(X_test_scaled)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: %.2f%%" % (accuracy * 100.0))
f1 = f1_score(y_test, y_pred)
print("F1-score: %.2f%%" % (f1 * 100.0))
precision = precision_score(y_test, y_pred)
print("Precision: %.2f%%" % (precision * 100.0))
recall = recall_score(y_test, y_pred)
print("Recall: %.2f%%" % (recall * 100.0))

thresholds = np.sort(model.feature_importances_)
for thresh in thresholds:
    # select features using threshold
    selection = SelectFromModel(model, threshold=thresh, prefit=True)
    select_X_train = selection.transform(X_trained_scaled)
    # train model
    selection_model = XGBClassifier()
    selection_model.fit(select_X_train, y_train)
    # eval model
    select_X_test = selection.transform(X_test_scaled)
    y_pred = selection_model.predict(select_X_test)
    predictions = [round(value) for value in y_pred]
    accuracy = accuracy_score(y_test, predictions)
    print("Thresh=%.3f, n=%d, Accuracy: %.2f%%" % (thresh, select_X_train.shape[1], accuracy*100.0))
    f1 = f1_score(y_test, predictions)
    print("F1-score: %.2f%%" % (f1 * 100.0))
