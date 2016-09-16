# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np


names = ['Magnitude', 'clipped_sigma', 'meaningless_1', 'meaningless_2',
         'star_ID', 'weighted_sigma', 'skew', 'kurt', 'I', 'J', 'K', 'L',
         'Npts', 'MAD', 'lag1', 'RoMS', 'rCh2', 'Isgn', 'Vp2p', 'Jclp', 'Lclp',
         'Jtim', 'Ltim', 'CSSD', 'Ex', 'inv_eta', 'E_A', 'S_B', 'NXS', 'IQR']

names_to_delete = ['Magnitude', 'meaningless_1', 'meaningless_2', 'star_ID']


def shift_log_transform(df, name, shift):
    df[name] = np.log(df[name] + shift)


def load_data(fnames, names, names_to_delete):
    """
    Function that loads data from series of files where first file contains
    class of zeros and other files - classes of ones.

    :param fnames:
        Iterable of file names.
    :param names:
        Names of columns in files.
    :param names_to_delete:
        Column names to delete.
    :return:
        X, y - ``sklearn`` arrays of features & responces.
    """
    # Load data
    dfs = list()
    for fn in fnames:
        dfs.append(pd.read_table(fn, names=names, engine='python',
                                 na_values='+inf', sep=r"\s*",
                                 usecols=range(30)))

    # Remove meaningless features
    delta = list()
    for df in dfs:
        delta.append(df['CSSD'].min())
    # print delta
    delta = np.min([d for d in delta if not np.isinf(d)])

    for df in dfs:
        for name in names_to_delete:
            del df[name]
        try:
            shift_log_transform(df, 'CSSD', -delta + 0.1)
        except KeyError:
            pass

    # List of feature names
    features_names = list(dfs[0])
    # Count number of NaN for each feature
    # for i, df in enumerate(dfs):
        # print("File {}".format(i))
        # for feature in features_names:
        #     print("Feature {} has {} NaNs".format(feature,
        #                                           df[feature].isnull().sum()))
        # print("=======================")

    # Convert to numpy arrays
    # Features
    X = list()
    for df in dfs:
        X.append(np.array(df[list(features_names)].values, dtype=float))
    X = np.vstack(X)
    # Responses
    y = np.zeros(len(X))
    y[len(dfs[0]):] = np.ones(len(X) - len(dfs[0]))

    return X, y, features_names


if __name__ == '__main__':
    import os
    import glob
    data_dir = '/home/ilya/code/mllc/data/dataset_OGLE/indexes_normalized'

    fnames = glob.glob(os.path.join(data_dir, '*.log'))[::-1]
    n_cv_iter = 5
    names = ['Magnitude', 'clipped_sigma', 'meaningless_1', 'meaningless_2',
             'star_ID', 'weighted_sigma', 'skew', 'kurt', 'I', 'J', 'K', 'L',
             'Npts', 'MAD', 'lag1', 'RoMS', 'rCh2', 'Isgn', 'Vp2p', 'Jclp',
             'Lclp',
             'Jtim', 'Ltim', 'CSSD', 'Ex', 'inv_eta', 'E_A', 'S_B', 'NXS',
             'IQR']

    names_to_delete = ['Magnitude', 'meaningless_1', 'meaningless_2', 'star_ID',
                       'Npts']

    X, y, feature_names = load_data(fnames, names, names_to_delete)
    import pickle
    with open('X_y_feat_names.pkl', 'wb') as fo:
        pickle.dump((X, y, feature_names), fo)