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


class Linkage(object):
    """Linkage used in other tools such as Heatmap"""

    def __init__(self):
        """.. rubric:: constructor

        :param data: a dataframe or possibly a numpy matrix.

        """
        pass

    def linkage(self, df, method, metric):
        # self.check_metric(metric)
        # self.check_method(method)
        d = distance.pdist(df)
        D = distance.squareform(d)
        Y = hierarchy.linkage(D, method=method, metric=metric)
        return Y


class Corrplot(Linkage):
    """An implementation of correlation plotting tools (corrplot)

    Here is a simple example with a correlation matrix as an input (stored in
    a pandas dataframe):

    .. plot::
        :width: 50%
        :include-source:

        # create a correlation-like data set stored in a Pandas' dataframe.
        import string
        # letters = string.uppercase[0:10] # python2
        letters = string.ascii_uppercase[0:10]
        import pandas as pd
        df = pd.DataFrame(dict(( (k, np.random.random(10)+ord(k)-65) for k in letters)))

        # and use corrplot
        from biokit.viz import corrplot
        c = corrplot.Corrplot(df)
        c.plot()

    .. seealso::    All functionalities are covered in this
        `notebook <http://nbviewer.ipython.org/github/biokit/biokit/blob/master/notebooks/viz/corrplot.ipynb>`_

    """
    def __init__(self, data, na=0):
        """.. rubric:: Constructor

        Plots the content of square matrix that contains correlation values.

        :param data: input can be a dataframe (Pandas), or list of lists (python) or
            a numpy matrix. Note, however, that values must be between -1 and 1. If not,
            or if the matrix (or list of lists) is not squared, then correlation is
            computed. The data or computed correlation is stored in :attr:`df` attribute.
        :param bool compute_correlation: if the matrix is non-squared or values are not
            bounded in -1,+1, correlation is computed. If you do not want that behaviour,
            set this parameter to False. (True by default).
        :param na: replace NA values with this value (default 0)

        The :attr:`params` contains some tunable parameters for the colorbar in the
        :meth:`plot` method.

        ::

            # can be a list of lists, the correlation matrix is then a 2x2 matrix
            c = corrplot.Corrplot([[1,1], [2,4], [3,3], [4,4]])

        """
        super(Corrplot, self).__init__()
        #: The input data is stored in a dataframe and must therefore be
        #: compatible (list of lists, dictionary, matrices...)
        self.df = pd.DataFrame(data, copy=True)

        compute_correlation = False

        w, h = self.df.shape
        if self.df.max().max() > 1 or self.df.min().min()<-1:
            compute_correlation = True
        if w !=h:
            compute_correlation = True
        if list(self.df.index) != list(self.df.columns):
            compute_correlation = True

        if compute_correlation:
            print("Computing correlation")
            cor = self.df.corr()
            self.df = cor

        # replace NA with zero
        self.df.fillna(na, inplace=True)

        #: tunable parameters for the :meth:`plot` method.
        self.params = {
                'colorbar.N': 100,
                'colorbar.shrink': .8,
                'colorbar.orientation':'vertical'}

    def _set_default_cmap(self):
        # self.cm = cmap_builder('#AA0000','white','darkblue')
        self.cm = cmap_builder('darkblue', 'white', '#AA0000')

    def order(self, method='complete', metric='euclidean',inplace=False):
        """Rearrange the order of rows and columns after clustering

        :param method: any scipy method (e.g., single, average, centroid,
            median, ward). See scipy.cluster.hierarchy.linkage
        :param metric: any scipy distance (euclidean, hamming, jaccard)
            See scipy.spatial.distance or scipy.cluster.hieararchy
        :param bool inplace: if set to True, the dataframe is replaced

        You probably do not need to use that method. Use :meth:`plot` and
        the two parameters order_metric and order_method instead.
        """
        Y = self.linkage(self.df, method=method, metric=metric)
        ind1 = hierarchy.fcluster(Y, 0.7*max(Y[:,2]), 'distance')
        Z = hierarchy.dendrogram(Y, no_plot=True)
        idx1 = Z['leaves']
        cor2 = self.df.iloc[idx1,idx1]
        if inplace is True:
            self.df = cor2
        else:
            return cor2
        self.Y = Y
        self.Z = Z
        self.idx1 = idx1
        self.ind1 = ind1

        #treee$order == Z.leaves and c.idx1
        # hc = c.ind1

        #clustab <- table(hc)[unique(hc[tree$order])]
        #cu <- c(0, cumsum(clustab))
        #mat <- cbind(cu[-(k + 1)] + 0.5, n - cu[-(k + 1)] + 0.5,
        #cu[-1] + 0.5, n - cu[-1] + 0.5)
        #rect(mat[,1], mat[,2], mat[,3], mat[,4], border = col, lwd = lwd)

    def plot(self, fig=None, grid=True,
            rotation=30, lower=None, upper=None,
            shrink=0.9, facecolor='white', colorbar=True, label_color='black',
            fontsize='large', edgecolor='black', method='ellipse',
            order_method='complete', order_metric='euclidean', cmap=None,
            ax=None, binarise_color=False):
        """plot the correlation matrix from the content of :attr:`df`
        (dataframe)

        By default, the correlation is shown on the upper and lower triangle and is
        symmetric wrt to the diagonal. The symbols are ellipses. The symbols can
        be changed to e.g. rectangle. The symbols are shown on upper and lower sides but
        you could choose a symbol for the upper side and another for the lower side using
        the **lower** and **upper** parameters.

        :param fig: Create a new figure by default. If an instance of an existing
            figure is provided, the corrplot is overlayed on the figure provided.
            Can also be the number of the figure.
        :param grid: add grid (Defaults to grey color). You can set it to False or a color.
        :param rotation: rotate labels on y-axis
        :param lower: if set to a valid method, plots the data on the lower
            left triangle
        :param upper: if set to a valid method, plots the data on the upper
            left triangle
        :param float shrink: maximum space used (in percent) by a symbol.
            If negative values are provided, the absolute value is taken.
            If greater than 1, the symbols wiill overlap.
        :param facecolor: color of the background (defaults to white).
        :param colorbar: add the colorbar (defaults to True).
        :param str label_color: (defaults to black).
        :param fontsize: size of the fonts defaults to 'small'.
        :param method: shape to be used in 'ellipse', 'square', 'rectangle',
            'color', 'text', 'circle',  'number', 'pie'.

        :param order_method: see :meth:`order`.
        :param order_metric: see : meth:`order`.
        :param cmap: a valid cmap from matplotlib or colormap package (e.g.,
            'jet', or 'copper'). Default is red/white/blue colors.
        :param ax: a matplotlib axes.

        The colorbar can be tuned with the parameters stored in :attr:`params`.

        Here is an example. See notebook for other examples::

            c = corrplot.Corrplot(dataframe)
            c.plot(cmap=('Orange', 'white', 'green'))
            c.plot(method='circle')
            c.plot(colorbar=False, shrink=.8, upper='circle'  )

        """
        # default
        if cmap != None:
            try:
                if isinstance(cmap, str):
                    self.cm = cmap_builder(cmap)
                else:
                    self.cm = cmap_builder(*cmap)
            except:
                print("incorrect cmap. Use default one")
                self._set_default_cmap()
        else:
            self._set_default_cmap()

        self.shrink = abs(shrink)
        self.fontsize = fontsize
        self.edgecolor = edgecolor

        df = self.order(method=order_method, metric=order_metric)

        # figure can be a number or an instance; otherwise creates it
        if isinstance(fig, int):
            fig = plt.figure(num=fig, facecolor=facecolor)
        elif fig is not None:
            fig = plt.figure(num=fig.number, facecolor=facecolor)
        else:
            fig = plt.figure(num=None, facecolor=facecolor)

        # do we have an axes to plot the data in ?
        if ax is None:
            ax = plt.subplot(1, 1, 1, aspect='equal', facecolor=facecolor)
        else:
            # if so, clear the axes. Colorbar cannot be removed easily.
            plt.sca(ax)
            ax.clear()

        # subplot resets the bg color, let us set it again
        fig.set_facecolor(facecolor)

        width, height = df.shape
        labels = (df.columns)

        # add all patches to the figure
        # TODO check value of lower and upper

        if upper is None and lower is None:
            mode = 'method'
            diagonal = True
        elif upper and lower:
            mode = 'both'
            diagonal = False
        elif lower is not None:
            mode = 'lower'
            diagonal = True
        elif upper is not None:
            mode = 'upper'
            diagonal = True

        self.binarise_color = binarise_color
        if mode == 'upper':
            self._add_patches(df, upper, 'upper',  ax, diagonal=True)
        elif mode == 'lower':
            self._add_patches(df, lower, 'lower',  ax, diagonal=True)
        elif mode == 'method':
            self._add_patches(df, method, 'both',  ax, diagonal=True)
        elif mode == 'both':
            self._add_patches(df, upper, 'upper',  ax, diagonal=False)
            self._add_patches(df, lower, 'lower',  ax, diagonal=False)

        # shift the limits to englobe the patches correctly
        ax.set_xlim(-0.5, width-.5)
        ax.set_ylim(-0.5, height-.5)

        # set xticks/xlabels on top
        ax.xaxis.tick_top()
        xtickslocs = np.arange(len(labels))
        ax.set_xticks(xtickslocs)
        ax.set_xticklabels(labels, rotation=rotation, color=label_color,
                fontsize=fontsize, ha='left')

        ax.invert_yaxis()
        ytickslocs = np.arange(len(labels))
        ax.set_yticks(ytickslocs)
        ax.set_yticklabels(labels, fontsize=fontsize, color=label_color)
        plt.tight_layout()

        if grid is not False:
            if grid is True:
                grid = 'grey'
            for i in range(0, width):
                ratio1 = float(i)/width
                ratio2 = float(i+2)/width
                # TODO 1- set axis off
                # 2 - set xlabels along the diagonal
                # set colorbar either on left or bottom
                if mode == 'lower':
                    plt.axvline(i+.5, ymin=1-ratio1, ymax=0., color=grid)
                    plt.axhline(i+.5, xmin=0, xmax=ratio2, color=grid)
                if mode == 'upper':
                    plt.axvline(i+.5, ymin=1 - ratio2, ymax=1, color=grid)
                    plt.axhline(i+.5, xmin=ratio1, xmax=1, color=grid)
                if mode in ['method', 'both']:
                    plt.axvline(i+.5, color=grid)
                    plt.axhline(i+.5, color=grid)

            # can probably be simplified
            if mode == 'lower':
                plt.axvline(-.5, ymin=0, ymax=1, color='grey')
                plt.axvline(width-.5, ymin=0, ymax=1./width, color='grey', lw=2)
                plt.axhline(width-.5, xmin=0, xmax=1, color='grey',lw=2)
                plt.axhline(-.5, xmin=0, xmax=1./width, color='grey',lw=2)
                plt.xticks([])
                for i in range(0, width):
                    plt.text(i, i-.6 ,labels[i],fontsize=fontsize,
                            color=label_color,
                            rotation=rotation, verticalalignment='bottom')
                    plt.text(-.6, i ,labels[i],fontsize=fontsize,
                            color=label_color,
                            rotation=0, horizontalalignment='right')
                plt.axis('off')
            # can probably be simplified
            elif mode == 'upper':
                plt.axvline(width-.5, ymin=0, ymax=1, color='grey', lw=2)
                plt.axvline(-.5, ymin=1-1./width, ymax=1, color='grey', lw=2)
                plt.axhline(-.5, xmin=0, xmax=1, color='grey',lw=2)
                plt.axhline(width-.5, xmin=1-1./width, xmax=1, color='grey',lw=2)
                plt.yticks([])
                for i in range(0, width):
                    plt.text(-.6+i, i ,labels[i],fontsize=fontsize,
                            color=label_color, horizontalalignment='right',
                            rotation=0)
                    plt.text(i, -.5 ,labels[i],fontsize=fontsize,
                            color=label_color, rotation=rotation, verticalalignment='bottom')
                plt.axis('off')

        # set all ticks length to zero
        ax = plt.gca()
        ax.tick_params(axis='both',which='both', length=0)

        if colorbar:
            from mpl_toolkits.axes_grid1 import make_axes_locatable
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.00)
            N = self.params['colorbar.N'] + 1
            assert N >=2
            cb = plt.gcf().colorbar(self.collection, cax=cax,
                    orientation=self.params['colorbar.orientation'],
                    # shrink=self.params['colorbar.shrink'],
                    boundaries= np.linspace(0,1,N), ticks=[0,.25, 0.5, 0.75,1])
            cb.ax.set_yticklabels([-1,-.5,0,.5,1], fontsize=14)
            cb.set_clim(0,1) # make sure it goes from -1 to 1 even though actual values may not reach that range

        return fig

    def _add_patches(self, df, method, fill, ax, diagonal=True):
        width, height = df.shape
        labels = (df.columns)

        patches = []
        colors = []
        for x in range(width):
            for y in range(height):
                if fill == 'lower' and x > y:
                    continue
                elif fill == 'upper' and x < y:
                    continue
                if diagonal is False and x==y:
                    continue

                datum = (df.iloc[x, y] +1.)/2.
                d = df.iloc[x, y]
                d_abs = np.abs(d)
                #c = self.pvalues[x, y]
                rotate = -45 if d > 0 else +45
                #cmap = self.poscm if d >= 0 else self.negcm
                if method in ['ellipse', 'square', 'rectangle', 'color']:
                    if method == 'ellipse':
                        func = Ellipse
                        patch = func((x, y), width=1 * self.shrink,
                                     height=(self.shrink - d_abs*self.shrink),
                                     angle=rotate, edgecolor='black')
                    else:
                        func = Rectangle
                        w = h = d_abs * self.shrink
                        #FIXME shring must be <=1
                        offset = (1-w)/2.
                        if method == 'color':
                            w = 1
                            h = 1
                            offset = 0
                        patch = func((x + offset-.5, y + offset-.5), width=w,
                                  height=h, angle=0)
                    if self.edgecolor:
                        patch.set_edgecolor(self.edgecolor)
                    #patch.set_facecolor(cmap(d_abs))
                    colors.append(datum)
                    if d_abs > 0.05:
                        patch.set_linestyle('dotted')
                    #ax.add_artist(patch)
                    patches.append(patch)
                    #FIXME edgecolor is always printed
                elif method=='circle':
                    patch = Circle((x, y), radius=d_abs*self.shrink/2.)
                    if self.edgecolor:
                        patch.set_edgecolor(self.edgecolor)
                    #patch.set_facecolor(cmap(d_abs))
                    colors.append(datum)
                    if d_abs > 0.05:
                        patch.set_linestyle('dotted')
                    #ax.add_artist(patch)
                    patches.append(patch)
                elif method in ['number', 'text']:
                    if d<0:
                        edgecolor = self.cm(-1.0)
                    elif d>=0:
                        edgecolor = self.cm(1.0)
                    d_str = "{:.2f}".format(d).replace("0.", ".").replace(".00", "")
                    ax.text(x,y, d_str, color=edgecolor,
                            fontsize=self.fontsize, horizontalalignment='center',
                            weight='bold', alpha=max(0.5, d_abs),
                            withdash=False)
                elif method == 'pie':
                    S = 360 * d_abs
                    patch = [
                        Wedge((x,y), 1*self.shrink/2., -90, S-90),
                        Wedge((x,y), 1*self.shrink/2., S-90, 360-90),
                        ]
                    #patch[0].set_facecolor(cmap(d_abs))
                    #patch[1].set_facecolor('white')
                    colors.append(datum)
                    colors.append(0.5)
                    if self.edgecolor:
                        patch[0].set_edgecolor(self.edgecolor)
                        patch[1].set_edgecolor(self.edgecolor)

                    #ax.add_artist(patch[0])
                    #ax.add_artist(patch[1])
                    patches.append(patch[0])
                    patches.append(patch[1])
                else:
                    raise ValueError('Method for the symbols is not known. Use e.g, square, circle')

        if self.binarise_color:
            colors = [1 if color >0.5 else -1 for color in colors]

        if len(patches):
            col1 = PatchCollection(patches, array=np.array(colors), cmap=self.cm)
            ax.add_collection(col1)

            self.collection = col1
            # Somehow a release of matplotlib prevent the edge color
            # from working but the set_edgecolor on the collection itself does
            # work...
            if self.edgecolor:
                self.collection.set_edgecolor(self.edgecolor)




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
        kwargs = self.filter_sk_params(Sequential.predict, kwargs)
        y = self.model.predict(X, **kwargs)
        if(y.shape[1] == 1):
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


import pandas as pd
corr_coefs = list()
for random_state in np.arange(1, 301, 10):
    kfold = StratifiedKFold(n_splits=4, shuffle=True, random_state=random_state)

    y_preds_knn = cross_val_predict(pipeline_knn, X, y, cv=kfold, n_jobs=4)
    y_preds_lr = cross_val_predict(pipeline_lr, X, y, cv=kfold, n_jobs=4)
    y_preds_rf = cross_val_predict(pipeline_rf, X, y, cv=kfold, n_jobs=4)
    y_preds_xgb = cross_val_predict(pipeline_xgb, X, y, cv=kfold, n_jobs=4)
    y_preds_svm = cross_val_predict(pipeline_svm, X, y, cv=kfold, n_jobs=4)
    y_preds_nn = cross_val_predict(pipeline_nn, X, y, cv=kfold, n_jobs=4)

# for clf, y_pred in zip(('LR', 'kNN', 'RF', 'SGB', 'SVM', 'NN'),
#                        (y_preds_lr, y_preds_knn, y_preds_rf, y_preds_xgb,
#                         y_preds_svm, y_preds_nn)):
#     print(clf)
#     CMs = list()
#     for train_idx, test_idx in kfold.split(X, y):
#         CMs.append(confusion_matrix(y[test_idx], y_pred[test_idx]))
#     CM = np.sum(CMs, axis=0)
#
#     FN = CM[1][0]
#     TP = CM[1][1]
#     FP = CM[0][1]
#     print("TP = {}".format(TP))
#     print("FP = {}".format(FP))
#     print("FN = {}".format(FN))
#
#     f1 = 2. * TP / (2. * TP + FP + FN)
#
#     # print "AUC: {}".format(auc)
#     print("{} F1={}".format(clf, f1))
#
#
# # algos = {"knn": y_preds_knn, "lr": y_preds_lr, "rf": y_preds_rf,
# #          "xgb": y_preds_xgb, "svm": y_preds_svm, "nn": y_preds_nn}
# # algos_names = algos.keys()
# #
# # from itertools import combinations
# #
# # for algo_name_1, algo_name_2 in combinations(algos_names, 2):
# #     corr = np.corrcoef(algos[algo_name_1], algos[algo_name_2])[1, 0]
# #     print("Correlation {}-{} = {}".format(algo_name_1, algo_name_2, corr))
#
#


    df = pd.DataFrame(data=np.column_stack((y_preds_knn, y_preds_lr, y_preds_rf,
                                            y_preds_xgb, y_preds_svm, y_preds_nn)),
                      columns=['kNN', 'LR', 'RF', 'SGB', 'SVM', 'NN'])
    corr_coefs.append(df.corr())

import pickle
with open("/home/ilya/github/ml4vs/ml4vs/cv_corr.pkl", "w") as fo:
          pickle.dump(corr_coefs, fo)

with open("/home/ilya/github/ml4vs/ml4vs/cv_corr.pkl", "r") as fo:
          corr_coefs = pickle.load(fo)

p = pd.Panel({n: df for n, df in enumerate(corr_coefs)})
mean_corr = p.mean(axis=0)
std_corr = p.std(axis=0)



# cp = Corrplot(df.corr())
# fig = cp.plot()
# fig.savefig("/home/ilya/Dropbox/papers/mlvs/submitted/predictions_corr.pdf",
#             format='pdf', dpi=1200)
# fig.savefig("/home/ilya/Dropbox/papers/mlvs/submitted/predictions_corr.eps",
#             format='eps', dpi=1200)
# fig.savefig("/home/ilya/Dropbox/papers/mlvs/submitted/predictions_corr.svg",
#             format='svg', dpi=1200)
#
# fig = cp.plot(method='text')
# fig.savefig("/home/ilya/Dropbox/papers/mlvs/submitted/predictions_corr_text.pdf",
#             format='pdf', dpi=1200)
# fig.savefig("/home/ilya/Dropbox/papers/mlvs/submitted/predictions_corr_text.eps",
#             format='eps', dpi=1200)
# fig.savefig("/home/ilya/Dropbox/papers/mlvs/submitted/predictions_corr_text.svg",
#             format='svg', dpi=1200)
#
# fig = cp.plot(method='text', colorbar=False)
# fig.savefig("/home/ilya/Dropbox/papers/mlvs/submitted/predictions_corr_text_nocb.pdf",
#             format='pdf', dpi=1200)
# fig.savefig("/home/ilya/Dropbox/papers/mlvs/submitted/predictions_corr_text_nocb.eps",
#             format='eps', dpi=1200)
# fig.savefig("/home/ilya/Dropbox/papers/mlvs/submitted/predictions_corr_text_nocb.svg",
#             format='svg', dpi=1200)