# -*- coding: utf-8 -*-
from pandas import DataFrame
from sklearn.base import TransformerMixin


def precision(cm):
    """
    Return precision.

    :param cm:
        Confusion matrix.
    :return:
        Value of precision.
    """
    return float(cm[1, 1]) / (cm[1, 1] + cm[0, 1])


def recall(cm):
    """
    Return recall.

    :param cm:
        Confusion matrix.
    :return:
        Value of recall.
    """
    return float(cm[1, 1]) / (cm[1, 1] + cm[1, 0])


def f1(cm):
    """
    Return F1-score.

    :param cm:
        Confusion matrix.
    :return:
        Value of F1-score.
    """
    return 2 * (precision(cm) * recall(cm) / (precision(cm) + recall(cm)))


def print_cm_summary(cm):
    print("Confusion matrix:", cm)
    print("Precision: ", precision(cm))
    print("Recall :", recall(cm))
    print("F1-score :", f1(cm))


def cm_scores(cm):
    return precision(cm), recall(cm), f1(cm)


class ModelTransformer(TransformerMixin):

    def __init__(self, model):
        self.model = model

    def fit(self, *args, **kwargs):
        self.model.fit(*args, **kwargs)
        return self

    def transform(self, X, **transform_params):
        return DataFrame(self.model.predict(X))


