# -*- coding: utf-8 -*-

# Autor: Luise Köhler
# Datum: Tue Sep 14 18:00:32 2021
# Python 3.8.8
# Ubuntu 20.04.1

from typing import List, Tuple

import pandas as pd
from nltk.probability import FreqDist
from nltk.tokenize.casual import TweetTokenizer
from nltk.util import ngrams


class FeatureExtractor:
    """
    Collect features (n-grams for words and characters) over a data set
    and compute these features for single instances.
    """

    def __init__(
        self,
    ) -> None:
        self.feature_vector: List[Tuple] = []

    def collect_features(self, data: List[str]) -> None:
        """
        Collect features over a data set. Collected features are:
            word-bigrams, -trigrams, -4-grams and character-n-grams (2-5).

        Parameters
        ----------
        data : List[str]
            List of texts in training set.

        Returns
        -------
        None

        """
        tokenizer = TweetTokenizer()
        features = set()
        for sentence in data:
            tokens = tokenizer.tokenize(sentence.lower())
            features.update(set(self._extract_word_n_grams(tokens)))
            features.update(set(self._extract_character_n_grams(tokens)))
        self.feature_vector = list(features)

    @staticmethod
    def _extract_word_n_grams(tokens: List[str]) -> List[Tuple[str]]:
        features = []
        for i in range(1, 4):
            features += ngrams(tokens, i)
        return features

    @staticmethod
    def _extract_character_n_grams(tokens: List[str]) -> List[Tuple[str]]:
        char_features = []
        for token in tokens:
            for i in range(2, 6):
                char_features += ngrams(token, i)
        return char_features

    def get_features_for_instance(self, instance_text: str) -> List[int]:
        """
        Apply collected features to a single instance.

        Parameters
        ----------
        instance_text : str
            Text of instance to compute features for.

        Returns
        -------
        List[int]
            Feature vector for instance.

        """
        tokenizer = TweetTokenizer()
        tokens = tokenizer.tokenize(instance_text)
        instance_features = FreqDist(
            self._extract_word_n_grams(tokens) + self._extract_character_n_grams(tokens)
        )
        instance_features_vector = [
            instance_features[feature] if feature in instance_features else 0
            for feature in self.feature_vector
        ]
        return pd.Series(instance_features_vector)
