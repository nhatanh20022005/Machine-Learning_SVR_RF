
from __future__ import annotations

import numpy as np
from typing import Tuple, Dict


def train_svr(X: np.ndarray, y: np.ndarray,
              C: float = 1.0, eps: float = 0.1,
              lr: float = 0.001, n_epochs: int = 2000,
              verbose: bool = False) -> Dict:
    """Train linear SVR (primal) via subgradient descent.

    X: (n_samples, n_features)
    y: (n_samples,)
    Returns model dict.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    n, d = X.shape

    w = np.zeros(d, dtype=float)
    b = 0.0

    for epoch in range(1, n_epochs + 1):
        y_pred = X.dot(w) + b
        resid = y_pred - y 

        mask_pos = resid > eps   
        mask_neg = resid < -eps  

       
        grad_w = w.copy()  #
        if mask_pos.any():
            grad_w += C * X[mask_pos].sum(axis=0)
        if mask_neg.any():
            grad_w -= C * X[mask_neg].sum(axis=0)

        grad_b = 0.0
        if mask_pos.any():
            grad_b += C * mask_pos.sum()
        if mask_neg.any():
            grad_b -= C * mask_neg.sum()

        w -= lr * grad_w
        b -= lr * grad_b

        if verbose and (epoch % (n_epochs // 5 or 1) == 0):
            loss = 0.5 * np.dot(w, w) + C * np.maximum(0, np.abs(resid) - eps).sum()
            print(f"[SVR] epoch {epoch}/{n_epochs} loss={loss:.6f}")

    return {'w': w, 'b': b, 'eps': eps}


def predict_svr(model: Dict, X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    return X.dot(model['w']) + model['b']


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
        X, y = make_regression(n_samples=300, n_features=5, noise=8.0, random_state=0)
        model = train_svr(X, y, C=1.0, eps=0.5, lr=1e-4, n_epochs=2000, verbose=True)
        pred = predict_svr(model, X)
        print('R2 (train):', r2_score(y, pred))
