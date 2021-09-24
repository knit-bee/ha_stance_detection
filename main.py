# -*- coding: utf-8 -*-

# Autor: Luise Köhler
# Datum: Fri Sep 24 14:41:58 2021
# Python 3.8.8
# Ubuntu 20.04.1
import argparse
import logging
import os
from typing import List, Optional, Union

import numpy as np
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from classify import (
    compute_micro_f1,
    get_best_classifier,
    majority_class_baseline,
    transform_categorical_data,
)
from feature_extractor import FeatureExtractor
from preprocessing import prepare_data

logging.basicConfig(
    filename="stance.log",
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main(arguments: Optional[List[str]] = None) -> None:
    args = parse_arguments(arguments)
    data = prepare_data(args.data_path)
    label_columns = [
        column
        for column in data.columns
        if column not in {"id", "text", "id  text", "[debate stance:polarity]"}
    ]
    logger.info(f"labels: {label_columns}")
    train_set, test_set = train_test_split(data, test_size=0.1, random_state=0)

    extractor = FeatureExtractor()
    extractor.collect_features(list(train_set["text"]))

    X_train = train_set["text"].apply(extractor.get_features_for_instance)
    X_test = test_set["text"].apply(extractor.get_features_for_instance)

    # explicit stance label sets
    y_train = np.array(train_set[label_columns])
    y_test = np.array(test_set[label_columns])

    # debate stance label sets
    y_debate_train = train_set["[debate stance:polarity]"]
    y_debate_test = test_set["[debate stance:polarity]"]

    # Explicit stance classification
    # set up majority class baseline
    majority_baseline = majority_class_baseline(X_train, y_train)
    baseline_label = majority_baseline.predict(X_test)
    logger.info("baseline")
    print(
        f"F1 for majority class baseline of explicit stance: \n \
{round(compute_micro_f1(y_test, baseline_label), 2)}"
    )
    # perform 10-fold cross-validation on the training set with a svm
    svm = SVC(kernel="linear", C=1, random_state=11)
    classifier = MultiOutputClassifier(svm, n_jobs=-1)

    scores = cross_validate(
        classifier,
        X_train,
        y_train,
        cv=10,
        return_estimator=True,
    )

    logger.info(f"Cross-validation accuracy per classifier: \n {scores['test_score']}")

    # use best classifier to predict test set
    estimator = get_best_classifier(scores)
    y_pred = estimator.predict(X_test)
    print("Micro F1 score for classification of explicit stance targets:")
    print(round(compute_micro_f1(y_test, y_pred), 2))

    # Debate stance classification
    logger.info("Debate stance classification")
    debate_majority_baseline = majority_class_baseline(
        X_train,
        y_debate_train,
    ).predict(X_test)
    print(
        f"F1 score for baseline of debate stance: \n\
{round(f1_score(y_debate_test, debate_majority_baseline,average='micro'), 2)}"
    )
    debate_scores = cross_validate(
        svm,
        X_train,
        y_debate_train,
        cv=10,
        return_estimator=True,
    )
    logger.info(
        f"Cross-validation accuracy per classifier: \n {debate_scores['test_score']}"
    )
    debate_estimator = get_best_classifier(debate_scores)
    y_pred_debate = debate_estimator.predict(X_test)
    print(
        f"F1 score for classification of debate stance: \n \
{round(f1_score(y_debate_test, y_pred_debate, average='micro'), 2)}"
    )

    # Decision tree classification
    # prepare data: use one-hot-encoding to transform categorical data
    y_train_num = transform_categorical_data(y_train, label_columns)
    y_pred_num = transform_categorical_data(y_pred, label_columns)
    y_test_num = transform_categorical_data(y_test, label_columns)

    dt = DecisionTreeClassifier(criterion="entropy", max_depth=5)
    dt.fit(y_train_num, y_debate_train)
    dt_prediction = dt.predict(y_pred_num)
    print(
        f"F1 score of decision tree classification: \n \
{round(f1_score(np.array(y_debate_test), dt_prediction, average='micro'), 2)}"
    )

    # "orcale" decision tree
    oracle_prediction = dt.predict(y_test_num)
    print(
        f"F1 score of 'oracle' decision tree: \n \
{round(f1_score(y_debate_test, oracle_prediction, average='micro'), 2)}"
    )


def parse_arguments(arguments: Optional[List[str]] = None) -> argparse.Namespace:
    """Construct argument parser"""
    parser = argparse.ArgumentParser(add_help=False, description="")
    parser.add_argument(
        "--help",
        "-h",
        action="help",
        default=argparse.SUPPRESS,
        help="Show this help message an exit",
    )
    parser.add_argument(
        "data_path",
        help="File or folder to process",
        type=valid_path,
    )

    return parser.parse_args(arguments)


def valid_path(input_string: str) -> Union[bool, str]:
    """
    Checks if a file or directory exists.

    Parameters
    ----------
    input_string : str
        File or folder to check.

    Returns
    -------
    Union[bool, str]
        False if file or folder is not found, else the path of the file or
        directory is returned.

    """
    if os.path.isfile(input_string) or os.path.isdir(input_string):
        return input_string
    return False


if __name__ == "__main__":
    main()
