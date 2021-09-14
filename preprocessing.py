# -*- coding: utf-8 -*-

# Autor: Luise Köhler
# Datum: Mon Sep 13 14:32:05 2021
# Python 3.8.8
# Ubuntu 20.04.1

import re

import pandas as pd


def split_text_id_columns(line, id_column=False):
    """Split column containing ids and text"""
    pattern = re.compile(r"(\d{3,5}\s+)(.+)")
    match = re.match(pattern, line)
    if id_column:
        return match.group(1)
    return match.group(2)


def transform_to_num_labels(line):
    label_map = {"none": 0, "favor": 1, "against": 2}
    label = line.split(":")[1]
    if label in label_map:
        return label_map[label]
    return 0


def prepare_data(datapath, delimiter="\t"):
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
    # transform labels to numeric values
    for column in data.columns:
        if column == "id  text":
            continue
        data[column] = data[column].apply(transform_to_num_labels)

    # separate columns
    data["text"] = data["id  text"].apply(split_text_id_columns)
    data["id"] = data["id  text"].apply(split_text_id_columns, args=(True,))
    return data
