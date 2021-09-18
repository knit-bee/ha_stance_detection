# -*- coding: utf-8 -*-

# Autor: Luise Köhler
# Datum: Tue Sep 14 18:01:02 2021
# Python 3.8.8
# Ubuntu 20.04.1

import argparse
import logging
import os
from typing import Any, List, Optional, Union

import numpy as np
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.svm import SVC

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
    y_train = np.array(train_set[label_columns])
    X_test = test_set["text"].apply(extractor.get_features_for_instance)
    y_test = np.array(test_set[label_columns])

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
    estimator = scores["estimator"][
        list(scores["test_score"]).index(max(scores["test_score"]))
    ]
    y_pred = estimator.predict(X_test)
    print("Micro F1 score for classification of explicit stance targets:")
    print(round(compute_micro_f1(y_test, y_pred) * 100, 2))


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
    parser.add_argument("data_path", help="File or folder to process", type=valid_path)

    return parser.parse_args(arguments)


def compute_micro_f1(y_true: Any, y_pred: Any) -> float:
    f1_scores_per_label = []
    for i in range(y_true.shape[1]):
        f1_scores_per_label.append(
            f1_score(y_true[:, i], y_pred[:, i], average="micro")
        )
    logger.info(f"f1 score per label of best classifier: {f1_scores_per_label}")
    return sum(f1_scores_per_label) / len(f1_scores_per_label)


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
