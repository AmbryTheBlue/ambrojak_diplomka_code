import mlflow.pyfunc
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class OODDContinuousModel(BaseEstimator, TransformerMixin, mlflow.pyfunc.PythonModel):
    """A class for detecting anomalies in continuous data using a lookup table with statistical properties.
    """

    def __init__(self, X_cols, target_col, threshold=1.0):
        self.group_keys = X_cols
        self.target_col = target_col
        self.threshold = threshold
        self.lookup = None

    def fit(self, df):
        df_copy = df.copy()
        df_copy[self.target_col] = pd.to_numeric(df_copy[self.target_col], errors='coerce')
        self.lookup = df_copy.groupby(self.group_keys)[self.target_col] \
                        .agg(['mean', 'std', 'count']) \
                        .reset_index()

    def _lookup_row(self, row):
        # Try exact match
        mask = pd.Series(True, index=self.lookup.index)
        for key in self.group_keys:
            mask &= self.lookup[key] == row[key]
        match = self.lookup[mask]
        return match if not match.empty else None

    def predict(self, df):
        df = df.copy()
        df[self.target_col] = pd.to_numeric(df[self.target_col], errors='coerce')
        results = []
        for _, row in df.iterrows():
            match = self._lookup_row(row)
            if match is None or match['std'].values[0] == 0 or pd.isna(match['std'].values[0]):
                results.append(False)
                continue
            mu = match['mean'].values[0]
            sigma = match['std'].values[0]
            val = row[self.target_col]
            is_outlier = abs(val - mu) > self.threshold * sigma
            results.append(is_outlier)
        return pd.Series(results, index=df.index)

    def predict_proba(self, df):
        """Return z-score distance from mean instead of boolean."""
        df = df.copy()
        df[self.target_col] = pd.to_numeric(df[self.target_col], errors='coerce')
        scores = []
        for _, row in df.iterrows():
            match = self._lookup_row(row)
            if match is None or match['std'].values[0] == 0 or pd.isna(match['std'].values[0]):
                scores.append(0.0)
                continue
            mu = match['mean'].values[0]
            sigma = match['std'].values[0]
            val = row[self.target_col]
            z = abs(val - mu) / sigma
            scores.append(z)
        return pd.Series(scores, index=df.index)
