# -*- coding: utf-8 -*-

# Autor: Luise Köhler
# Datum: Tue Sep 14 18:01:02 2021
# Python 3.8.8
# Ubuntu 20.04.1

import logging
from typing import Any

import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.metrics import f1_score
from sklearn.svm import SVC

logger = logging.getLogger(__name__)


def compute_micro_f1(y_true: Any, y_pred: Any) -> float:
    """
    Computes micro F1 for labels. Returns average of F1 scores per label.
    F1 scores for each label will be logged.

    Parameters
    ----------
    y_true : Any
        Array of true class labels.
    y_pred : Any
        Array of predicted class labels.

    Returns
    -------
    float
        Average of micro F1 scores for all labels.

    """
    f1_scores_per_label = []
    for i in range(y_true.shape[1]):
        f1_scores_per_label.append(
            f1_score(y_true[:, i], y_pred[:, i], average="micro")
        )
    logger.info(f"f1 score per label of best classifier: {f1_scores_per_label}")
    return sum(f1_scores_per_label) / len(f1_scores_per_label)


def majority_class_baseline(
    X_train: pd.DataFrame, y_train: pd.Series
) -> DummyClassifier:
    """
    Trains baseline classifier for majority class baseline.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training data.
    y_train : pd.Series
        Labels of training data.

    Returns
    -------
    DummyClassifier
        Majority baseline classifier.

    """
    baseline_classifier = DummyClassifier(strategy="most_frequent")
    baseline_classifier.fit(X_train, y_train)
    return baseline_classifier


def get_best_classifier(scores: dict) -> SVC:
    """Finds best classifier from cross-validation"""
    return scores["estimator"][
        list(scores["test_score"]).index(max(scores["test_score"]))
    ]


def transform_categorical_data(data: Any, label_columns: list) -> pd.DataFrame:
    """
    Transforms data from categorical to one-hot encoded data in order
    to be processed by Decision tree classifier.

    Parameters
    ----------
    data : Any
        Data to transform, array or pd.DataFrame.
    label_columns : list
        List of original column names.

    Returns
    -------
    DataFrame with transformed data
    """
    data = pd.DataFrame(data, columns=label_columns)
    new_data = pd.get_dummies(data)

    new_label_columns = create_new_label_columns(label_columns)
    for column in new_label_columns:
        if column not in new_data.columns:
            new_data[column] = 0
    return new_data[new_label_columns]


def create_new_label_columns(label_columns: list) -> list:
    """
    Extracts names for new label columns for one-hot-encoded data

    """
    transformed_label_columns = []
    for column in label_columns:
        transformed_label_columns.append(column + "_" + "none")
        transformed_label_columns.append(column + "_" + "favor")
        transformed_label_columns.append(column + "_" + "against")
    return transformed_label_columns
