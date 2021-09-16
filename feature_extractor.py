# -*- coding: utf-8 -*-

# Autor: Luise Köhler
# Datum: Tue Sep 14 18:00:32 2021
# Python 3.8.8
# Ubuntu 20.04.1
from typing import List, Optional, Set, Tuple

import pandas as pd
from nltk.probability import FreqDist
from nltk.tokenize.casual import TweetTokenizer
from nltk.util import bigrams, ngrams, trigrams


class FeatureExtractor:
    def __init__(
        self,
    ) -> None:
        self.feature_vector: List[Tuple] = []

    def collect_features(self, data: List[str]) -> None:
        tokenizer = TweetTokenizer()
        features = set()
        for sentence in data:
            tokens = tokenizer.tokenize(sentence.lower())
            features.update(set(self._extract_word_n_grams(tokens)))
            features.update(set(self._extract_character_n_grams(tokens)))
        self.feature_vector = list(features)

    def _extract_word_n_grams(self, tokens: List[str]) -> List[Tuple[str]]:
        features = []
        for i in range(1, 4):
            features += ngrams(tokens, i)
        return features

    def _extract_character_n_grams(self, tokens: List[str]) -> List[Tuple[str]]:
        char_features = []
        for token in tokens:
            for i in range(2, 6):
                char_features += ngrams(token, i)
        return char_features

    def get_features_for_instance(self, instance_text: str) -> List[int]:
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
