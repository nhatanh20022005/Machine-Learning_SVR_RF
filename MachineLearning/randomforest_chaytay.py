
from __future__ import annotations

import numpy as np
from typing import Optional, Tuple, Any, Dict


class TreeNode:
    __slots__ = ('feature', 'threshold', 'left', 'right', 'value')
    def __init__(self):
        self.feature = None
        self.threshold = None
        self.left = None
        self.right = None
        self.value = None


def _mse(y: np.ndarray) -> float:
    if y.size == 0:
        return 0.0
    return ((y - y.mean()) ** 2).mean()


class DecisionTreeRegressor:
    def __init__(self, max_depth: int = 5, min_samples_split: int = 4, max_features: Optional[int] = None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.root = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        X = np.asarray(X)
        y = np.asarray(y).ravel()
        self.n_features_ = X.shape[1]
        if self.max_features is None:
            self.max_features_ = self.n_features_
        else:
            self.max_features_ = min(self.max_features, self.n_features_)
        self.root = self._build_tree(X, y, depth=0)

    def _best_split(self, X: np.ndarray, y: np.ndarray) -> Tuple[Optional[int], Optional[float], float]:
        n_samples, n_features = X.shape
        if n_samples < self.min_samples_split:
            return None, None, float('inf')

        feat_idx = np.random.permutation(n_features)[:self.max_features_]
        best_feat, best_thr, best_loss = None, None, float('inf')

        for j in feat_idx:
            vals = X[:, j]
            uniq = np.unique(vals)
            if uniq.size > 50:
                thresholds = np.unique(np.percentile(vals, np.linspace(1, 99, 50)))
            else:
                thresholds = (uniq[:-1] + uniq[1:]) / 2.0

            for thr in thresholds:
                left_mask = vals <= thr
                right_mask = ~left_mask
                if left_mask.sum() < 2 or right_mask.sum() < 2:
                    continue
                loss = (left_mask.sum() * _mse(y[left_mask]) + right_mask.sum() * _mse(y[right_mask])) / n_samples
                if loss < best_loss:
                    best_loss = loss
                    best_feat = j
                    best_thr = thr

        return best_feat, best_thr, best_loss

    def _build_tree(self, X: np.ndarray, y: np.ndarray, depth: int) -> TreeNode:
        node = TreeNode()
        node.value = y.mean()

        if depth >= self.max_depth or X.shape[0] < self.min_samples_split or np.unique(y).size == 1:
            return node

        feat, thr, loss = self._best_split(X, y)
        if feat is None:
            return node

        node.feature = feat
        node.threshold = thr
        left_idx = X[:, feat] <= thr
        right_idx = ~left_idx
        node.left = self._build_tree(X[left_idx], y[left_idx], depth + 1)
        node.right = self._build_tree(X[right_idx], y[right_idx], depth + 1)
        return node

    def _predict_row(self, x: np.ndarray, node: TreeNode) -> float:
        if node.feature is None or node.left is None:
            return node.value
        if x[node.feature] <= node.threshold:
            return self._predict_row(x, node.left)
        else:
            return self._predict_row(x, node.right)

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X)
        return np.array([self._predict_row(x, self.root) for x in X])


class RandomForestRegressor:
    def __init__(self, n_estimators: int = 10, max_depth: int = 8, min_samples_split: int = 4, max_features: Any = 'sqrt', random_state: Optional[int] = None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.trees = []
        self.rng = np.random.RandomState(random_state)

    def _interpret_max_features(self, n_features: int) -> int:
        if self.max_features in (None, 'auto'):
            return n_features
        if self.max_features == 'sqrt':
            return max(1, int(np.sqrt(n_features)))
        if isinstance(self.max_features, float):
            return max(1, int(self.max_features * n_features))
        return int(self.max_features)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        X = np.asarray(X)
        y = np.asarray(y).ravel()
        n, d = X.shape
        self.trees = []
        mf = self._interpret_max_features(d)

        for i in range(self.n_estimators):
            idx = self.rng.randint(0, n, size=n)
            Xb = X[idx]
            yb = y[idx]
            tree = DecisionTreeRegressor(max_depth=self.max_depth, min_samples_split=self.min_samples_split, max_features=mf)
            tree.fit(Xb, yb)
            self.trees.append(tree)

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X)
        preds = np.vstack([t.predict(X) for t in self.trees])
        return preds.mean(axis=0)


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return 1.0 - ss_res / ss_tot if ss_tot != 0 else 0.0


if __name__ == '__main__':
    try:
        from sklearn.datasets import make_regression
    except Exception:
        make_regression = None

    if make_regression is None:
        print('skipping quick smoke test (sklearn not available)')
    else:
        X, y = make_regression(n_samples=400, n_features=6, noise=10.0, random_state=1)
        rf = RandomForestRegressor(n_estimators=8, max_depth=6, random_state=1)
        rf.fit(X, y)
        pred = rf.predict(X)
        print('R2 (train RF):', r2_score(y, pred))
