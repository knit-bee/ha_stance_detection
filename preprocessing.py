# -*- coding: utf-8 -*-

# Autor: Luise Köhler
# Datum: Mon Sep 13 14:32:05 2021
# Python 3.8.8
# Ubuntu 20.04.1

import re

import pandas as pd


def split_text_id_columns(line: str, id_column: bool = False) -> str:
    """Split column containing ids and text"""
    pattern = re.compile(r"(\d{3,5}\s+)(.+)")
    match = re.match(pattern, line)
    assert match is not None
    if id_column:
        return match.group(1)
    return match.group(2)


def reduce_labels(line: str) -> str:
    """Simplify labels in data"""
    labels = ["none", "favor", "against"]
    label = line.split(":")[1]
    return label if label in labels else "none"


def prepare_data(datapath: str, delimiter: str = "\t") -> pd.DataFrame:
    """
    Splits text-id column in two separate columns and adds new columns
    to data frame. Transforms labels in other columns to numeric values,
    none: 0; favor: 1; against: 2.

    Parameters
    ----------
    datapath : str
        path to data file.
    delimiter : str, optional
        Separator character used in the data file. The default is '\t'.

    Returns
    -------
    data : pd.DataFrame
        DataFrame with new columns.

    """
    data = pd.read_csv(datapath, delimiter)

    # transform labels to consistent labels
    for column in data.columns:
        if column == "id  text":
            continue
        data[column] = data[column].apply(reduce_labels)

    # separate columns
    data["text"] = data["id  text"].apply(split_text_id_columns)
    data["id"] = data["id  text"].apply(split_text_id_columns, args=(True,))
    return data
