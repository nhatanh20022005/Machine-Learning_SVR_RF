from __future__ import annotations

import argparse
import sys
from typing import Tuple

import numpy as np
import pandas as pd

from svr_chaytay import train_svr, predict_svr
from randomforest_chaytay import RandomForestRegressor as RF_Chaytay

try:
    from sklearn.svm import SVR as SKSVR
    from sklearn.ensemble import RandomForestRegressor as SKRF
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False


class StandardScalerSimple:
    def fit(self, X: np.ndarray):
        self.mean_ = X.mean(axis=0)
        self.scale_ = X.std(axis=0)
        self.scale_[self.scale_ == 0] = 1.0

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean_) / self.scale_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        self.fit(X)
        return self.transform(X)


def prepare_dataframe(df: pd.DataFrame, target_col: str = 'Profit') -> Tuple[np.ndarray, np.ndarray]:
    df = df.copy()
    if target_col not in df.columns:
        raise ValueError('target column Profit not found')
    y = df[target_col].values.astype(float)
    Xdf = df.drop(columns=[target_col])
    for col in Xdf.columns:
        if Xdf[col].dtype == object:
            Xdf[col] = pd.factorize(Xdf[col].astype(str))[0]
    X = Xdf.fillna(0).values.astype(float)
    return X, y


def split(X: np.ndarray, y: np.ndarray, seed: int = 0):
    rng = np.random.RandomState(seed)
    idx = np.arange(X.shape[0])
    rng.shuffle(idx)
    n = X.shape[0]
    n_tr = int(0.7 * n)
    n_val = int(0.15 * n)
    tr = idx[:n_tr]
    val = idx[n_tr:n_tr + n_val]
    te = idx[n_tr + n_val:]
    return X[tr], y[tr], X[val], y[val], X[te], y[te]


def r2(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1.0 - ss_res / ss_tot if ss_tot != 0 else 0.0


def mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))


def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def run_compare(path: str):
    df = pd.read_csv(path, encoding='latin1')
    X, y = prepare_dataframe(df)
    X_tr, y_tr, X_val, y_val, X_te, y_te = split(X, y)

    scaler = StandardScalerSimple()
    X_tr_sc = scaler.fit_transform(X_tr)
    X_val_sc = scaler.transform(X_val)
    X_te_sc = scaler.transform(X_te)

    print('Training from-scratch SVR (linear primal)...')
    svr_model = train_svr(X_tr_sc, y_tr, C=1.0, eps=0.1, lr=1e-4, n_epochs=2000)
    svr_val = predict_svr(svr_model, X_val_sc)
    svr_test = predict_svr(svr_model, X_te_sc)

    print('Training from-scratch RandomForest...')
    rf_ch = RF_Chaytay(n_estimators=30, max_depth=8, random_state=1)
    rf_ch.fit(X_tr, y_tr)
    rf_ch_val = rf_ch.predict(X_val)
    rf_ch_test = rf_ch.predict(X_te)

    results = []
    results.append(('SVR_chaytay', r2(y_val, svr_val), mae(y_val, svr_val), rmse(y_val, svr_val), r2(y_te, svr_test), mae(y_te, svr_test), rmse(y_te, svr_test)))
    results.append(('RF_chaytay', r2(y_val, rf_ch_val), mae(y_val, rf_ch_val), rmse(y_val, rf_ch_val), r2(y_te, rf_ch_test), mae(y_te, rf_ch_test), rmse(y_te, rf_ch_test)))

    if SKLEARN_AVAILABLE:
        print('Training sklearn SVR (RBF) ...')
        sk_svr = SKSVR(kernel='rbf', C=10.0, epsilon=0.1)
        sk_svr.fit(X_tr_sc, y_tr)
        sk_svr_val = sk_svr.predict(X_val_sc)
        sk_svr_test = sk_svr.predict(X_te_sc)

        print('Training sklearn RandomForest ...')
        sk_rf = SKRF(n_estimators=200, max_depth=15, random_state=1)
        sk_rf.fit(X_tr, y_tr)
        sk_rf_val = sk_rf.predict(X_val)
        sk_rf_test = sk_rf.predict(X_te)

        results.append(('SVR_sklearn', r2(y_val, sk_svr_val), mae(y_val, sk_svr_val), rmse(y_val, sk_svr_val), r2(y_te, sk_svr_test), mae(y_te, sk_svr_test), rmse(y_te, sk_svr_test)))
        results.append(('RF_sklearn', r2(y_val, sk_rf_val), mae(y_val, sk_rf_val), rmse(y_val, sk_rf_val), r2(y_te, sk_rf_test), mae(y_te, sk_rf_test), rmse(y_te, sk_rf_test)))

        ens_val = 0.5 * (sk_svr_val + sk_rf_val)
        ens_test = 0.5 * (sk_svr_test + sk_rf_test)
        results.append(('Ensemble_sklearn', r2(y_val, ens_val), mae(y_val, ens_val), rmse(y_val, ens_val), r2(y_te, ens_test), mae(y_te, ens_test), rmse(y_te, ens_test)))

    else:
        print('scikit-learn not available; skipping sklearn baselines.')

    hdr = ('Model', 'Val R2', 'Val MAE', 'Val RMSE', 'Test R2', 'Test MAE', 'Test RMSE')
    print('\n' + ' | '.join(hdr))
    print('-' * 80)
    for row in results:
        print(f"{row[0]:15} | {row[1]:7.4f} | {row[2]:8.4f} | {row[3]:9.4f} | {row[4]:7.4f} | {row[5]:8.4f} | {row[6]:9.4f}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', required=True, help='CSV path with target Profit')
    args = p.parse_args()
    try:
        run_compare(args.data)
    except Exception as e:
        print('Error:', e)
        sys.exit(1)


if __name__ == '__main__':
    main()
