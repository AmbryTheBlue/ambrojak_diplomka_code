from collections import defaultdict, Counter
from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np
import pandas as pd


class ContinuousCountBasedClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, X_cols, target_col, threshold=0.1, range_width=5):
        """
        X_cols: list of feature-column names to count on
        target_col: name of the target/class column
        threshold: min prob to output a class (else None)
        range_width: range for continuous data (+-range_width)
        """
        self.X_cols = X_cols
        self.target_col = target_col
        self.threshold = threshold
        self.range_width = range_width

    def fit(self, X, y=None):
        """
        X: pandas DataFrame (must contain both X_cols and target_col)
        y: ignored (we pull target from X[self.target_col])
        """
        # Ensure DataFrame
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=self.X_cols + [self.target_col])

        # Keep track of all seen classes
        self.classes_ = np.unique(X[self.target_col].values)

        # Nested dict: feature-tuple → Counter of classes
        self.counts_ = defaultdict(Counter)
        for _, row in X.iterrows():
            key = tuple(row[col] for col in self.X_cols)
            cls = row[self.target_col]
            self.counts_[key][cls] += 1

        # Also store total counts per key for fast lookup
        self.totals_ = {k: sum(cnt.values())
                        for k, cnt in self.counts_.items()}
        return self

    def _get_range_counts(self, key, cls):
        """
        Helper function to sum counts for a range of values.
        """
        total_count = 0
        for offset in range(-self.range_width, self.range_width + 1):
            shifted_key = tuple(
                k + offset if isinstance(k, (int, float)) else k for k in key)
            total_count += self.counts_.get(shifted_key, Counter()).get(cls, 0)
        return total_count

    def predict_proba(self, X):
        """
        Returns an array of shape (n_samples, n_classes)
        where each entry is:
          sum(count((x±range_width),c)) / (total_count((x±range_width)) + 1.0)
        The order of classes is given by self.classes_
        """
        # Convert to DataFrame if needed
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=self.X_cols)

        n = X.shape[0]
        m = len(self.classes_)
        proba = np.zeros((n, m), dtype=float)

        for i, (_, row) in enumerate(X.iterrows()):
            key = tuple(row[col] for col in self.X_cols)
            total = 0
            for offset in range(-self.range_width, self.range_width + 1):
                shifted_key = tuple(
                    k + offset if isinstance(k, (int, float)) else k for k in key)
                total += self.totals_.get(shifted_key, 0)

            for j, cls in enumerate(self.classes_):
                cnt = self._get_range_counts(key, cls)
                # Denominator smoothing +1.0
                proba[i, j] = cnt / (total + 1.0)

        return proba

    def predict(self, X):
        """
        Returns True = anomalous (P(true class) < threshold)
                False = confident (P >= threshold)
        Expects X with both features and target_col.
        """
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=self.X_cols + [self.target_col])

        proba = self.predict_proba(X[self.X_cols])
        flags = []
        for i, true_cls in enumerate(X[self.target_col]):
            if true_cls not in self.classes_:
                flags.append(True)
                continue
            idx = np.where(self.classes_ == true_cls)[0][0]
            flags.append(proba[i, idx] < self.threshold)
        return np.array(flags)


class FallbackContinuousCountClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, X_cols, target_col, threshold=0.1, range_width=5):
        """
        X_cols: list of feature-column names (e.g., ['f1', 'f2'])
        target_col: name of the class column
        threshold: P(true class|X) cutoff for 'anomaly' (True = anomaly)
        range_width: range for continuous data (+-range_width)
        """
        self.X_cols = X_cols
        self.target_col = target_col
        self.threshold = threshold
        self.range_width = range_width

        # Define fallback combinations
        self.fallback_combos = []
        # 1. Full combination of features
        self.fallback_combos.append(tuple(X_cols))
        # 2. First two features as a pair, if applicable
        if len(X_cols) > 2:
            self.fallback_combos.append(tuple(X_cols[:2]))
        # 3. Each of the first 5 features individually
        for col in X_cols[:5]:
            if (col,) not in self.fallback_combos:
                self.fallback_combos.append((col,))
        print(f"Fallback combinations: {self.fallback_combos}")

    def fit(self, X, y=None):
        """
        X: pandas DataFrame (must contain both X_cols and target_col)
        y: ignored (we pull target from X[self.target_col])
        """
        X = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X.copy()

        # Record classes
        self.classes_ = np.unique(X[self.target_col])

        # Build counts and totals for each fallback combination
        self.counts_ = {combo: defaultdict(Counter)
                        for combo in self.fallback_combos}
        self.totals_ = {combo: defaultdict(int)
                        for combo in self.fallback_combos}

        for _, row in X.iterrows():
            cls = row[self.target_col]
            for combo in self.fallback_combos:
                key = tuple(row[c] for c in combo)
                self.counts_[combo][key][cls] += 1
                self.totals_[combo][key] += 1

        return self

    def _get_range_counts(self, combo, key, cls):
        """
        Helper function to sum counts for a range of values.
        """
        total_count = 0
        for offset in range(-self.range_width, self.range_width + 1):
            shifted_key = tuple(
                k + offset if isinstance(k, (int, float)) else k for k in key)
            total_count += self.counts_[combo].get(
                shifted_key, Counter()).get(cls, 0)
        return total_count

    def predict_proba(self, X):
        """
        Returns an array of shape (n_samples, n_classes)
        where each entry is:
          sum(count((x±range_width),c)) / (total_count((x±range_width)) + 1.0)
        The order of classes is given by self.classes_
        """
        X = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X
        n, m = X.shape[0], len(self.classes_)
        proba = np.zeros((n, m), dtype=float)

        for i, (_, row) in enumerate(X.iterrows()):
            # Find the first fallback combination with data
            for combo in self.fallback_combos:
                key = tuple(row[c] for c in combo)
                total = 0
                for offset in range(-self.range_width, self.range_width + 1):
                    shifted_key = tuple(
                        k + offset if isinstance(k, (int, float)) else k for k in key)
                    total += self.totals_[combo].get(shifted_key, 0)

                if total > 0:
                    for j, cls in enumerate(self.classes_):
                        cnt = self._get_range_counts(combo, key, cls)
                        # Denominator smoothing +1.0
                        proba[i, j] = cnt / (total + 1.0)
                    break
            # else: proba[i, :] stays zeros

        return proba

    def predict(self, X):
        """
        Returns True = anomalous (P(true class) < threshold)
                False = confident (P >= threshold)
        Expects X with both features and target_col.
        """
        X = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X
        proba = self.predict_proba(X[self.X_cols])
        flags = []
        for i, true_cls in enumerate(X[self.target_col]):
            if true_cls not in self.classes_:
                flags.append(True)
                continue
            idx = np.where(self.classes_ == true_cls)[0][0]
            flags.append(proba[i, idx] < self.threshold)
        return np.array(flags)
