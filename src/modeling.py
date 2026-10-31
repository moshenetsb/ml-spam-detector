import unicodedata

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, MaxAbsScaler

NUM_COLS = [
    "message_length", "uppercase_ratio", "url_count",
    "currency_count", "number_count",
]


class MessageFeatures(TransformerMixin, BaseEstimator):
    """Compute all message features from raw texts during training and inference."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):

        text = pd.Series(X, dtype="str")
        result = pd.DataFrame({"text": text})
        result["message_length"] = text.str.len()

        result["uppercase_ratio"] = (
            text.str.count(r"[A-Z]") / text.str.count(r"[A-Za-z]").replace(0, 1)
        )

        result["url_count"] = text.str.count(r"https?://|www\.")

        result["currency_count"] = text.apply(
            lambda message: sum(unicodedata.category(char) == "Sc" for char in message)
        )

        result["number_count"] = text.str.count(r"\d")

        return result


def build_preprocessor(include_numeric=False):

    transformers = [
        ("text", TfidfVectorizer(max_features=3000, lowercase=True, stop_words="english"), "text"),
    ]

    if include_numeric:
        numeric = Pipeline([
            ("log1p", FunctionTransformer(np.log1p)),
            ("scale", MaxAbsScaler()),
        ])

        transformers.append(("num", numeric, NUM_COLS))

    return ColumnTransformer(transformers)


def build_model_pipeline(classifier, include_numeric=False):
    
    return Pipeline([
        ("message_features", MessageFeatures()),
        ("features", build_preprocessor(include_numeric)),
        ("clf", classifier),
    ])
