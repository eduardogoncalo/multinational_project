import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from config import Config

class TextConcatenator(BaseEstimator, TransformerMixin):
    """
    Custom transformer to concatenate text columns within a sckit-learn pipeline.
    Ensures the transformation is applied independently during fit and transform.   
    """
    def __init__(self, col1: str = Config.TICKET_SUMMARY_TRANSLATED, col2: str = Config.INTERACTION_CONTENT_TRANSLATED):
        self.col1 = col1
        self.col2 = col2

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.Series:
            return X.astype(str).fillna('').apply(lambda row: ' '.join(row), axis=1)