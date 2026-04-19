from __future__ import annotations

import argparse
import numpy as np
import pandas as pd
from typing import Tuple

from svr_chaytay import train_svr, predict_svr
from randomforest_chaytay import RandomForestRegressor


def train_val_test_split(X: np.ndarray, y: np.ndarray, train_frac=0.7, val_frac=0.15, seed=0):
    rng = np.random.RandomState(seed)
    n = X.shape[0]
    idx = np.arange(n)
    rng.shuffle(idx)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    train_idx = idx[:n_train]
    val_idx = idx[n_train:n_train + n_val]
    test_idx = idx[n_train + n_val:]
    return X[train_idx], y[train_idx], X[val_idx], y[val_idx], X[test_idx], y[test_idx]


def mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))


def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def r2(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1.0 - ss_res / ss_tot if ss_tot != 0 else 0.0


def prepare_dataframe(df: pd.DataFrame, target_col: str = 'Profit') -> Tuple[np.ndarray, np.ndarray, list]:
    df = df.copy()
    if target_col not in df.columns:
        raise ValueError('target column Profit not found')

    # simple preprocessing: numeric features + label-encode categoricals
    y = df[target_col].values.astype(float)
    Xdf = df.drop(columns=[target_col])
    for col in Xdf.columns:
        if Xdf[col].dtype == object:
            Xdf[col] = pd.factorize(Xdf[col].astype(str))[0]
    X = Xdf.fillna(0).values.astype(float)
    return X, y, Xdf.columns.tolist()


def evaluate_models(df: pd.DataFrame):
    X, y, feature_names = prepare_dataframe(df)
    X_tr, y_tr, X_val, y_val, X_te, y_te = train_val_test_split(X, y)

    svr_model = train_svr(X_tr, y_tr, C=1.0, eps=0.1, lr=1e-4, n_epochs=2000, verbose=False)
    svr_val_pred = predict_svr(svr_model, X_val)
    svr_test_pred = predict_svr(svr_model, X_te)

    rf = RandomForestRegressor(n_estimators=30, max_depth=8, random_state=1)
    rf.fit(X_tr, y_tr)
    rf_val_pred = rf.predict(X_val)
    rf_test_pred = rf.predict(X_te)

    ens_val_pred = 0.5 * (svr_val_pred + rf_val_pred)
    ens_test_pred = 0.5 * (svr_test_pred + rf_test_pred)

    results = {
        'SVR': {
            'val': (r2(y_val, svr_val_pred), mae(y_val, svr_val_pred), rmse(y_val, svr_val_pred)),
            'test': (r2(y_te, svr_test_pred), mae(y_te, svr_test_pred), rmse(y_te, svr_test_pred)),
        },
        'RF': {
            'val': (r2(y_val, rf_val_pred), mae(y_val, rf_val_pred), rmse(y_val, rf_val_pred)),
            'test': (r2(y_te, rf_test_pred), mae(y_te, rf_test_pred), rmse(y_te, rf_test_pred)),
        },
        'Ensemble': {
            'val': (r2(y_val, ens_val_pred), mae(y_val, ens_val_pred), rmse(y_val, ens_val_pred)),
            'test': (r2(y_te, ens_test_pred), mae(y_te, ens_test_pred), rmse(y_te, ens_test_pred)),
        }
    }

    return results


def print_results(results):
    print('\nModel comparison (R2, MAE, RMSE)')
    for model in ['SVR', 'RF', 'Ensemble']:
        v = results[model]['val']
        t = results[model]['test']
        print(f"\n{model}:\n  Val  -> R2={v[0]:.4f}, MAE={v[1]:.4f}, RMSE={v[2]:.4f}\n  Test -> R2={t[0]:.4f}, MAE={t[1]:.4f}, RMSE={t[2]:.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True, help='Path to CSV with target column Profit')
    args = parser.parse_args()

    df = pd.read_csv(args.data, encoding='latin1')
    results = evaluate_models(df)
    print_results(results)

    print('\nNotes:')
    print('- The from-scratch SVR implemented here is linear (primal SGD) and may underfit non-linear data.')
    print('- The Random Forest captures non-linear patterns via tree splits; it usually gives lower bias at cost of variance.')
    print('- Ensemble averages predictions to reduce variance; if RF and SVR make different errors this often improves test RMSE/R2.')
    print('\nPossible improvements: kernelize SVR (dual/QP), increase RF estimators, tune hyperparams, or use weighted ensemble based on validation performance.')


if __name__ == '__main__':
    main()
