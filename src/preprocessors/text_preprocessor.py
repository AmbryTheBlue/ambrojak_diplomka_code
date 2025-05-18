import unicodedata
import re
from sklearn.base import BaseEstimator, TransformerMixin


class TextPreprocessor(TransformerMixin, BaseEstimator):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self  # stateless

    def transform(self, X):
        X_copy = X.copy()
        for col in self.columns:
            X_copy[col] = X_copy[col].astype(str)
            X_copy[col] = X_copy[col].apply(self.preprocess_text)
        return X_copy

    def preprocess_text(self, text):
        # Normalize and replace accented chars with ASCII equivalents
        text = unicodedata.normalize('NFKD', text)
        text = text.encode('ascii', 'ignore').decode('ascii')
        text = re.sub(r'\s', ' ', text)
        text = text.upper()
        text = re.sub(r'[^A-Z0-9]', ' ', text)
        text = re.sub(r' +', ' ', text)
        return text.strip()


class PrinterNonTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        ...

    def fit(self, X, y=None):
        print(X)
        return self

    def transform(self, X):
        print(X)
        return X
