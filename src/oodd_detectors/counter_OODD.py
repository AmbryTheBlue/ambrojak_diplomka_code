from collections import defaultdict, Counter
from sklearn.base import BaseEstimator, ClassifierMixin
from itertools import combinations
import numpy as np
import pandas as pd

class CountBasedClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, X_cols, target_col, threshold=0.1):
        """
        X_cols: list of feature‐column names to count on
        target_col: name of the target/class column
        threshold: min prob to output a class (else None)
        """
        self.X_cols = X_cols
        self.target_col = target_col
        self.threshold = threshold

    def fit(self, X, y=None):
        """
        X: pandas DataFrame (must contain both X_cols and target_col)
        y: ignored (we pull target from X[self.target_col])
        """
        # ensure DataFrame
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=self.X_cols + [self.target_col])

        # keep track of all seen classes
        self.classes_ = np.unique(X[self.target_col].values)

        # nested dict: feature‐tuple → Counter of classes
        self.counts_ = defaultdict(Counter)
        for _, row in X.iterrows():
            key = tuple(row[col] for col in self.X_cols)
            cls = row[self.target_col]
            self.counts_[key][cls] += 1

        # also store total counts per key for fast lookup
        self.totals_ = {k: sum(cnt.values()) for k, cnt in self.counts_.items()}
        return self

    def predict_proba(self, X):
        """
        Returns an array of shape (n_samples, n_classes)
        where each entry is:
          count((x),c) / (total_count((x)) + 1.0)
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
            counter = self.counts_.get(key, Counter())
            total = self.totals_.get(key, 0)

            for j, cls in enumerate(self.classes_):
                cnt = counter.get(cls, 0)
                # denominator smoothing +1.0
                proba[i, j] = cnt / (total + 1.0)

        return proba

    def predict(self, X):
        """
        Returns True if P(true class | X) < threshold (i.e. anomalous),
        else False.
        Assumes X includes the target_col.
        """
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=self.X_cols + [self.target_col])

        proba = self.predict_proba(X[self.X_cols])
        result = []

        for i, true_class in enumerate(X[self.target_col]):
            if true_class not in self.classes_:
                result.append(True)  # unknown class = suspicious
                continue
            idx = np.where(self.classes_ == true_class)[0][0]
            prob = proba[i, idx]
            result.append(prob < self.threshold)

        return np.array(result)

class FallbackCountClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, X_cols, target_col, threshold=0.1):
        """
        X_cols: list of feature‐column names (e.g. ['f1','f2'])
        target_col: name of the class column
        threshold: P(true class|X) cutoff for 'anomaly' (True = anomaly)
        """
        self.X_cols = X_cols
        self.target_col = target_col
        self.threshold = threshold
        
        self.fallback_combos = []
        # 1. Full combo
        self.fallback_combos.append(tuple(X_cols))
        # 2. First two as pair, if applicable
        if len(X_cols) > 2:
            self.fallback_combos.append(tuple(X_cols[:2]))
        # 3. Each of the first 5 columns individually
        for col in X_cols[:10]:
            if (col,) not in self.fallback_combos:
                self.fallback_combos.append((col,))
        print(self.fallback_combos)

    def fit(self, X, y=None):
        X = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X.copy()
        # record classes
        self.classes_ = np.unique(X[self.target_col])

        # build counts & totals for each combo
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

    def predict_proba(self, X):
        X = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X
        n, m = X.shape[0], len(self.classes_)
        proba = np.zeros((n, m), float)

        for i, (_, row) in enumerate(X.iterrows()):
            # find first combo with data
            for combo in self.fallback_combos:
                key = tuple(row[c] for c in combo)
                total = self.totals_[combo].get(key, 0)
                if total > 0:
                    counter = self.counts_[combo][key]
                    for j, cls in enumerate(self.classes_):
                        cnt = counter.get(cls, 0)
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

class FBODBackOff(BaseEstimator, ClassifierMixin):
    def __init__(self, X_cols, target_col, threshold=0.1):
        """
        X_cols: list of feature‐column names (e.g. ['f1','f2'])
        target_col: name of the class column
        threshold: P(true class|X) cutoff for 'anomaly' (True = anomaly)
        """
        self.X_cols = X_cols
        self.target_col = target_col
        self.threshold = threshold
        
        self.fallback_combos = []
        n = len(X_cols)
        for r in range(n, 0, -1):
            for combo in combinations(X_cols, r):
                self.fallback_combos.append(combo)
        print(self.fallback_combos)

    def fit(self, X, y=None):
        X = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X.copy()
        # record classes
        self.classes_ = np.unique(X[self.target_col])

        # build counts & totals for each combo
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

    def predict_proba(self, X):
        X = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X
        n, m = X.shape[0], len(self.classes_)
        proba = np.zeros((n, m), float)

        for i, (_, row) in enumerate(X.iterrows()):
            # find first combo with data
            for combo in self.fallback_combos:
                key = tuple(row[c] for c in combo)
                total = self.totals_[combo].get(key, 0)
                if total > 0:
                    counter = self.counts_[combo][key]
                    for j, cls in enumerate(self.classes_):
                        cnt = counter.get(cls, 0)
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
