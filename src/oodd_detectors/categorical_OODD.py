import mlflow.pyfunc
from sklearn.base import BaseEstimator, TransformerMixin


class OODDSmoothedCategoricalModel(BaseEstimator, TransformerMixin, mlflow.pyfunc.PythonModel):
    """A class for detecting anomalies in categorical data using a lookup table with smoothing.
    This model computes the probability of each category under each group and uses it to identify anomalies.
    """

    def __init__(self, X_cols, target_col, threshold=0.01, smooth_factor=0.01):
        self.group_keys = X_cols
        self.target_col = target_col
        self.min_prob = threshold
        self.smooth_factor = smooth_factor  # Smoothing coefficient
        self.lookup = None
        self.freq_table = None

    def fit(self, df):
        # Compute probability for each category under each group
        self.lookup = (
            df.groupby(self.group_keys)[self.target_col]
              .value_counts(normalize=True)
              .rename('prob')
              .reset_index()
        )

        # Frequency table for smoothing
        self.freq_table = (
            df.groupby(self.group_keys)[self.target_col]
              .value_counts()
              .rename('freq')
              .reset_index()
        )

    def _lookup_row(self, row):
        cond = True
        for key in self.group_keys:
            cond &= self.lookup[key] == row[key]
        cond &= self.lookup[self.target_col] == row[self.target_col]
        match = self.lookup[cond]
        return match if not match.empty else None

    def _lookup_frequency(self, row):
        cond = True
        for key in self.group_keys:
            cond &= self.freq_table[key] == row[key]
        match = self.freq_table[cond]
        return match if not match.empty else None

    def predict(self, df):
        #TODO WTF for TEMP_FROM this seems to work well with NOT but on easiest task: HEATING TYPE it performs abyssmaly
        return df.apply(lambda row: not self._predict_row(row), axis=1)

    def _predict_row(self, row):
        # Look for the category with the same context
        match = self._lookup_row(row)

        # If no match is found in training data for this context and category → anomaly
        if match is None:
            return False  # Unseen category/context → anomaly

        # Apply smoothing based on frequency
        freq_match = self._lookup_frequency(row)
        if freq_match is None:
            return True  # Fallback: treat as normal (no frequency data)

        category_freq = freq_match['freq'].values[0]
        smooth_prob = match['prob'].values[0] * \
            (1 - self.smooth_factor * category_freq)

        # Anomaly if probability is lower than threshold
        return smooth_prob <= self.min_prob

    def predict_proba(self, df):
        return df.apply(lambda row: self._predict_proba_row(row), axis=1)

    def _predict_proba_row(self, row):
        match = self._lookup_row(row)

        # If no match is found in training data for this context and category → anomaly
        if match is None:
            return 1.0  # Completely unseen → max anomaly (1.0)

        # Similar smoothing for anomaly score
        freq_match = self._lookup_frequency(row)
        if freq_match is None:
            return 0.0  # No data → normal (fallback)

        category_freq = freq_match['freq'].values[0]
        smooth_prob = match['prob'].values[0] * \
            (1 - self.smooth_factor * category_freq)

        return 1.0 - smooth_prob  # Inverted: Higher probability means more anomalous


class OODDCategoricalModel(BaseEstimator, TransformerMixin, mlflow.pyfunc.PythonModel):
    def __init__(self, X_cols, target_col, threshold=0.01):
        self.group_keys = X_cols
        self.target_col = target_col
        self.min_prob = threshold
        self.lookup = None

    def fit(self, df):
        self.lookup = (
            df.groupby(self.group_keys)[self.target_col]
              .value_counts(normalize=True)
              .rename('prob')
              .reset_index()
        )

    def _lookup_row(self, row):
        cond = True
        for key in self.group_keys:
            cond &= self.lookup[key] == row[key]
        cond &= self.lookup[self.target_col] == row[self.target_col]
        match = self.lookup[cond]
        return match if not match.empty else None

    def predict(self, df):
        return df.apply(lambda row: self._predict_row(row), axis=1)

    def _predict_row(self, row):
        match = self._lookup_row(row)
        if match is None:
            return True  # unknown combo, not enough data
        return match['prob'].values[0] <= self.min_prob

    def predict_proba(self, df):
        return df.apply(lambda row: self._predict_proba_row(row), axis=1)

    def _predict_proba_row(self, row):
        match = self._lookup_row(row)
        if match is None:
            return 0.0
        return match['prob'].values[0]
