import os
import matplotlib.pyplot as plt
from sklearn import decomposition
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import StandardScaler
from data_load import load_data
import numpy as np
label_size = 14
import matplotlib
matplotlib.rcParams['xtick.labelsize'] = label_size
matplotlib.rcParams['ytick.labelsize'] = label_size
matplotlib.rcParams['axes.titlesize'] = label_size
matplotlib.rcParams['axes.labelsize'] = label_size
matplotlib.rcParams['font.size'] = 14
matplotlib.rcParams['legend.fontsize'] = 14



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

names_to_delete = ['meaningless_1', 'meaningless_2', 'star_ID']
X, y, df, feature_names, delta = load_data([file_0, file_1], names, names_to_delete)


def get_eigenvalues(X):
    """
    Used for choosing number of PCA-components using "Eigenvalue-one criterion"
    (see https://gugginotes.wordpress.com/2013/11/04/pca/) or "Kaiser criterion"
    (Kaiser, 1960) (see https://en.wikipedia.org/wiki/Factor_analysis).

    http://stackoverflow.com/a/31941631
    :param X:
        Sample
    :return:
        Prints eigenvalues calculated using two methods.
    """
    n_samples = X.shape[0]
    pca = PCA()
    X_transformed = pca.fit_transform(X)

    # We center the data and compute the sample covariance matrix.
    X_centered = X - np.mean(X, axis=0)
    cov_matrix = np.dot(X_centered.T, X_centered) / n_samples
    eigenvalues = pca.explained_variance_
    for eigenvalue, eigenvector in zip(eigenvalues, pca.components_):
        print(np.dot(eigenvector.T, np.dot(cov_matrix, eigenvector)))
        print(eigenvalue)


rpca = decomposition.PCA(svd_solver='full', n_components='mle')
imp = Imputer(missing_values='NaN', strategy='median', axis=0, verbose=2)
rpca_pipe = Pipeline(steps=[('imputation', imp),
                            ('scaling', StandardScaler()),
                            ('pca', rpca)])
rpca_pipe.fit(X)


pca = decomposition.PCA()
imp = Imputer(missing_values='NaN', strategy='median', axis=0, verbose=2)
pca_pipe = Pipeline(steps=[('imputation', imp),
                           ('scaling', StandardScaler()),
                           ('pca', pca)])
pca_pipe.fit(X)

plt.figure(1, figsize=(4, 3))
plt.clf()
plt.axes([.2, .2, .7, .7])
# plt.plot(np.arange(1, pca.n_components_+1), pca.explained_variance_ratio_,
#          linewidth=2, color="#4682b4")
# plt.plot(np.arange(1, pca.n_components_+1), pca.explained_variance_ratio_,
#          linewidth=2, color="black")
plt.plot(np.arange(1, pca.n_components_+1),
         np.cumsum(pca.explained_variance_ratio_), linewidth=2, color="black")
# plt.xticks(np.arange(1, pca.n_components_+1, 5))
plt.xticks([1, 5, 10, 15, 20, 25])
plt.axis('tight')
plt.xlim([1, pca.n_components_])
plt.ylim([0, None])
plt.legend()
plt.xlabel(u'Number of PCA-component')
# plt.ylabel(u'explained variance ratio')
plt.ylabel(u'Explained variance fraction')
plt.show()

# import os
# path = '/home/ilya/Dropbox/papers/mlvs/new_pics/'
# plt.savefig(os.path.join(path, 'pca_nobounds.svg'), format='svg', dpi=1200)
# plt.savefig(os.path.join(path, 'pca_nobounds.eps'), format='eps', dpi=1200)
# plt.savefig(os.path.join(path, 'pca_nobounds.pdf'), format='pdf', dpi=1200)
