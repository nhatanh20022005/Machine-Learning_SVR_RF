import numpy as np
import pandas as pd
import streamlit as st

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def init_state():
    """Khởi tạo tất cả các key trong st.session_state nếu chưa tồn tại."""
    defaults = dict(
        df=None, df_clean=None, X=None, y=None,
        X_train=None, X_val=None, X_test=None,
        y_train=None, y_val=None, y_test=None,
        X_train_scaled=None, X_val_scaled=None, X_test_scaled=None,
        y_train_scaled=None, y_val_scaled=None, y_test_scaled=None,
        scaler_X=None, scaler_y=None,
        le_dict={}, top_features=None,
        X_train_sel=None, X_val_sel=None, X_test_sel=None,
        rf_model=None, svr_model=None,
        rf_pred=None, rf_val_pred=None,
        svr_pred=None, svr_val_pred=None,
        rf_mae=None, rf_rmse=None, rf_r2=None,
        rf_val_mae=None, rf_val_rmse=None, rf_val_r2=None,
        svr_mae=None, svr_rmse=None, svr_r2=None,
        svr_val_mae=None, svr_val_rmse=None, svr_val_r2=None,
        rf_cv=None, svr_cv=None,
        imp_df=None,
        pipeline_done=False,
    )
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


# Tiền xử lý 

def preprocess(df_raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Làm sạch và feature engineering từ raw DataFrame.

    Returns:
        df_clean (pd.DataFrame): DataFrame đã xử lý (không còn cột Profit).
        le_dict  (dict):         Các LabelEncoder đã fit cho từng cột categorical.
    """
    drop_cols = [
        'Row ID', 'Order ID', 'Customer ID', 'Customer Name',
        'Product ID', 'Product Name', 'City', 'State', 'Postal Code',
    ]
    df_clean = df_raw.drop(
        columns=[c for c in drop_cols if c in df_raw.columns]
    ).copy()

    df_clean['Order Date'] = pd.to_datetime(
        df_clean['Order Date'], format='%m/%d/%Y', errors='coerce'
    )
    df_clean['Ship Date'] = pd.to_datetime(
        df_clean['Ship Date'], format='%m/%d/%Y', errors='coerce'
    )
    df_clean['Order_Year']      = df_clean['Order Date'].dt.year
    df_clean['Order_Month']     = df_clean['Order Date'].dt.month
    df_clean['Order_DayOfWeek'] = df_clean['Order Date'].dt.dayofweek
    df_clean['Ship_Days']       = (
        df_clean['Ship Date'] - df_clean['Order Date']
    ).dt.days
    df_clean.drop(columns=['Order Date', 'Ship Date'], inplace=True)

    # Label Encoding cho biến categorical
    le_dict = {}
    for col in df_clean.select_dtypes(include='object').columns:
        le = LabelEncoder()
        df_clean[col] = le.fit_transform(df_clean[col].astype(str))
        le_dict[col]  = le

    # IQR clipping cho Profit (loại bỏ outlier cực đoan)
    Q1 = df_clean['Profit'].quantile(0.05)
    Q3 = df_clean['Profit'].quantile(0.95)
    df_clean['Profit'] = df_clean['Profit'].clip(Q1, Q3)

    return df_clean, le_dict


def select_features(
    X_train_sc: np.ndarray,
    y_train_sc: np.ndarray,
    feature_names: list[str],
    k: int,
) -> tuple[list[str], pd.DataFrame]:
    """
    Chọn top-k features dựa trên Feature Importance của Random Forest (Embedded method).

    Args:
        X_train_sc:    Ma trận features đã scale (train).
        y_train_sc:    Vector target đã scale (train).
        feature_names: Danh sách tên features.
        k:             Số lượng features muốn chọn.

    Returns:
        top_features (list[str]):   Danh sách tên top-k features.
        imp_df       (pd.DataFrame): DataFrame importance đầy đủ, đã sắp xếp giảm dần.
    """
    rf_imp = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=1)
    rf_imp.fit(X_train_sc, y_train_sc)

    imp_df = pd.DataFrame({
        'Feature':    feature_names,
        'Importance': rf_imp.feature_importances_,
    }).sort_values('Importance', ascending=False)

    top_features = imp_df.head(k)['Feature'].tolist()
    return top_features, imp_df


def make_profit_bins(y: pd.Series, max_bins: int = 10) -> pd.Series | None:
    """Tạo bin từ target Profit để stratify cho bài toán regression."""
    y_series = pd.Series(y).astype(float)
    nunique = y_series.nunique(dropna=True)
    if nunique < 2:
        return None

    q = int(min(max_bins, nunique))
    if q < 2:
        return None

    try:
        bins = pd.qcut(y_series, q=q, labels=False, duplicates='drop')
    except ValueError:
        return None

    if pd.Series(bins).nunique(dropna=True) < 2:
        return None
    return bins


# train mô hình

def train_random_forest(
    X_train_sel: np.ndarray,
    y_train_sc: np.ndarray,
    n_estimators: int,
    max_depth: int,
    use_cv: bool = True,
) -> tuple[RandomForestRegressor, np.ndarray]:
    """
    Huấn luyện Random Forest Regressor.

    Returns:
        model  (RandomForestRegressor): Mô hình đã fit.
        cv     (np.ndarray):            Cross-validation R² scores (5-fold).
    """
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features='sqrt',
        random_state=42,
        n_jobs=1,
    )
    model.fit(X_train_sel, y_train_sc)
    if use_cv:
        cv = cross_val_score(
            model, X_train_sel, y_train_sc, cv=5, scoring='r2', n_jobs=1
        )
    else:
        cv = np.array([], dtype=float)
    return model, cv


def train_svr(
    X_train_sel: np.ndarray,
    y_train_sc: np.ndarray,
    C: float,
    epsilon: float,
    use_cv: bool = True,
) -> tuple[SVR, np.ndarray]:
    """
    Huấn luyện Support Vector Regression (RBF kernel).

    Returns:
        model (SVR):         Mô hình đã fit.
        cv    (np.ndarray):  Cross-validation R² scores (5-fold).
    """
    model = SVR(kernel='rbf', C=C, epsilon=epsilon, gamma='scale')
    model.fit(X_train_sel, y_train_sc)
    if use_cv:
        cv = cross_val_score(
            model, X_train_sel, y_train_sc, cv=5, scoring='r2', n_jobs=1
        )
    else:
        cv = np.array([], dtype=float)
    return model, cv


def run_pipeline(
    df_raw: pd.DataFrame,
    n_estimators: int,
    max_depth: int,
    svr_C: float,
    svr_eps: float,
    k_features: int,
    test_size: float,
) -> None:
    """
    Chạy toàn bộ pipeline và lưu kết quả vào st.session_state.

    Args:
        df_raw:       DataFrame thô từ file CSV upload.
        n_estimators: Số cây của Random Forest.
        max_depth:    Độ sâu tối đa của mỗi cây.
        svr_C:        Tham số C của SVR.
        svr_eps:      Tham số epsilon của SVR.
        k_features:   Số features cần chọn.
        test_size:    Tỉ lệ tập test (0 –1).
    """
    s = st.session_state

    # 1. Tiền
    df_clean, le_dict = preprocess(df_raw)

    # 2 — Tách X, y
    X = df_clean.drop(columns=['Profit'])
    y = df_clean['Profit']

    # 3 — Train/Val/Test split cố định 70/15/15
    _ = test_size
    idx = np.arange(len(X))
    profit_bins = make_profit_bins(y)
    strat_all = profit_bins if profit_bins is not None else None

    idx_train, idx_temp = train_test_split(
        idx,
        test_size=0.30,
        shuffle=True,
        random_state=42,
        stratify=strat_all,
    )

    strat_temp = None
    if profit_bins is not None:
        strat_temp = profit_bins.iloc[idx_temp]
        if pd.Series(strat_temp).nunique(dropna=True) < 2:
            strat_temp = None

    idx_val, idx_test = train_test_split(
        idx_temp,
        test_size=0.50,
        shuffle=True,
        random_state=42,
        stratify=strat_temp,
    )

    X_train, X_val, X_test = X.iloc[idx_train], X.iloc[idx_val], X.iloc[idx_test]
    y_train, y_val, y_test = y.iloc[idx_train], y.iloc[idx_val], y.iloc[idx_test]

    # 4 — Scaling
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X_train_sc = scaler_X.fit_transform(X_train)
    X_val_sc   = scaler_X.transform(X_val)
    X_test_sc  = scaler_X.transform(X_test)
    y_train_sc = scaler_y.fit_transform(
        y_train.values.reshape(-1, 1)
    ).ravel()
    y_val_sc   = scaler_y.transform(
        y_val.values.reshape(-1, 1)
    ).ravel()
    y_test_sc  = scaler_y.transform(
        y_test.values.reshape(-1, 1)
    ).ravel()

    # 5 — Feature selection
    top_features, imp_df = select_features(
        X_train_sc, y_train_sc, X.columns.tolist(), k_features
    )
    feat_idx     = [X.columns.tolist().index(f) for f in top_features]
    X_train_sel  = X_train_sc[:, feat_idx]
    X_val_sel    = X_val_sc[:,   feat_idx]
    X_test_sel   = X_test_sc[:,  feat_idx]

    # 6 — Huấn luyện Random Forest
    rf_model, rf_cv = train_random_forest(
        X_train_sel, y_train_sc, n_estimators, max_depth
    )
    rf_pred_sc = rf_model.predict(X_test_sel)
    rf_pred    = scaler_y.inverse_transform(
        rf_pred_sc.reshape(-1, 1)
    ).ravel()
    rf_val_pred_sc = rf_model.predict(X_val_sel)
    rf_val_pred    = scaler_y.inverse_transform(
        rf_val_pred_sc.reshape(-1, 1)
    ).ravel()

    # 7 — Huấn luyện SVR
    svr_model, svr_cv = train_svr(
        X_train_sel, y_train_sc, svr_C, svr_eps
    )
    svr_pred_sc = svr_model.predict(X_test_sel)
    svr_pred    = scaler_y.inverse_transform(
        svr_pred_sc.reshape(-1, 1)
    ).ravel()
    svr_val_pred_sc = svr_model.predict(X_val_sel)
    svr_val_pred    = scaler_y.inverse_transform(
        svr_val_pred_sc.reshape(-1, 1)
    ).ravel()

    # 7.5 — Ensemble (simple average of RF and SVR predictions)
    ensemble_pred = 0.5 * (rf_pred + svr_pred)
    ensemble_val_pred = 0.5 * (rf_val_pred + svr_val_pred)

    # 8 — Lưu tất cả vào session_state
    s.df            = df_raw
    s.df_clean      = df_clean
    s.X             = X
    s.y             = y
    s.X_train       = X_train
    s.X_val         = X_val
    s.X_test        = X_test
    s.y_train       = y_train
    s.y_val         = y_val
    s.y_test        = y_test
    s.X_train_scaled = X_train_sc
    s.X_val_scaled   = X_val_sc
    s.X_test_scaled  = X_test_sc
    s.y_train_scaled = y_train_sc
    s.y_val_scaled   = y_val_sc
    s.y_test_scaled  = y_test_sc
    s.scaler_X      = scaler_X
    s.scaler_y      = scaler_y
    s.le_dict       = le_dict
    s.top_features  = top_features
    s.imp_df        = imp_df
    s.X_train_sel   = X_train_sel
    s.X_val_sel     = X_val_sel
    s.X_test_sel    = X_test_sel
    s.rf_model      = rf_model
    s.svr_model     = svr_model
    s.rf_pred       = rf_pred
    s.rf_val_pred   = rf_val_pred
    s.svr_pred      = svr_pred
    s.svr_val_pred  = svr_val_pred
    s.rf_cv         = rf_cv
    s.svr_cv        = svr_cv
    s.rf_mae        = mean_absolute_error(y_test, rf_pred)
    s.rf_rmse       = np.sqrt(mean_squared_error(y_test, rf_pred))
    s.rf_r2         = r2_score(y_test, rf_pred)
    s.rf_val_mae    = mean_absolute_error(y_val, rf_val_pred)
    s.rf_val_rmse   = np.sqrt(mean_squared_error(y_val, rf_val_pred))
    s.rf_val_r2     = r2_score(y_val, rf_val_pred)
    s.svr_mae       = mean_absolute_error(y_test, svr_pred)
    s.svr_rmse      = np.sqrt(mean_squared_error(y_test, svr_pred))
    s.svr_r2        = r2_score(y_test, svr_pred)
    s.svr_val_mae   = mean_absolute_error(y_val, svr_val_pred)
    s.svr_val_rmse  = np.sqrt(mean_squared_error(y_val, svr_val_pred))
    s.svr_val_r2    = r2_score(y_val, svr_val_pred)
    # Ensemble metrics (average of rf + svr predictions)
    s.ensemble_pred      = ensemble_pred
    s.ensemble_val_pred  = ensemble_val_pred
    s.ensemble_mae       = mean_absolute_error(y_test, ensemble_pred)
    s.ensemble_rmse      = np.sqrt(mean_squared_error(y_test, ensemble_pred))
    s.ensemble_r2        = r2_score(y_test, ensemble_pred)
    s.ensemble_val_mae   = mean_absolute_error(y_val, ensemble_val_pred)
    s.ensemble_val_rmse  = np.sqrt(mean_squared_error(y_val, ensemble_val_pred))
    s.ensemble_val_r2    = r2_score(y_val, ensemble_val_pred)
    # store simple ensemble "cv" summary as means of the two model CV arrays
    s.ensemble_cv = np.array([
        rf_cv.mean() if rf_cv.size else np.nan,
        svr_cv.mean() if svr_cv.size else np.nan,
    ])
    s.pipeline_done = True


def find_best_hyperparameters(df_raw: pd.DataFrame) -> dict:
    """Tìm bộ hyperparameters tốt nhất dựa trên Validation R²."""
    df_clean, _ = preprocess(df_raw)
    X = df_clean.drop(columns=['Profit'])
    y = df_clean['Profit']

    idx = np.arange(len(X))
    profit_bins = make_profit_bins(y)
    strat_all = profit_bins if profit_bins is not None else None

    idx_train, idx_temp = train_test_split(
        idx,
        test_size=0.30,
        shuffle=True,
        random_state=42,
        stratify=strat_all,
    )

    strat_temp = None
    if profit_bins is not None:
        strat_temp = profit_bins.iloc[idx_temp]
        if pd.Series(strat_temp).nunique(dropna=True) < 2:
            strat_temp = None

    idx_val, _ = train_test_split(
        idx_temp,
        test_size=0.50,
        shuffle=True,
        random_state=42,
        stratify=strat_temp,
    )

    X_train, X_val = X.iloc[idx_train], X.iloc[idx_val]
    y_train, y_val = y.iloc[idx_train], y.iloc[idx_val]

    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X_train_sc = scaler_X.fit_transform(X_train)
    X_val_sc = scaler_X.transform(X_val)
    y_train_sc = scaler_y.fit_transform(y_train.values.reshape(-1, 1)).ravel()
    y_val_arr = y_val.values

    # Coarse grid for fast tuning: fewer trials, still large enough to show
    # visible differences on validation/test.
    n_estimators_grid = [100, 250]
    max_depth_grid = [10, 20]
    svr_c_grid = [5, 20]
    svr_eps_grid = [0.05, 0.20]

    max_k = min(12, X.shape[1])
    k_grid = [k for k in [6, 10] if k <= max_k]
    if not k_grid:
        k_grid = [max(1, min(4, X.shape[1]))]

    best_overall = None

    for k in k_grid:
        top_features, _ = select_features(
            X_train_sc, y_train_sc, X.columns.tolist(), k
        )
        feat_idx = [X.columns.tolist().index(f) for f in top_features]
        X_train_sel = X_train_sc[:, feat_idx]
        X_val_sel = X_val_sc[:, feat_idx]

        best_rf = None
        for n_estimators in n_estimators_grid:
            for max_depth in max_depth_grid:
                rf_model, _ = train_random_forest(
                    X_train_sel, y_train_sc, n_estimators, max_depth,
                    use_cv=False
                )
                rf_val_pred = scaler_y.inverse_transform(
                    rf_model.predict(X_val_sel).reshape(-1, 1)
                ).ravel()
                rf_val_r2 = r2_score(y_val_arr, rf_val_pred)

                if best_rf is None or rf_val_r2 > best_rf['rf_val_r2']:
                    best_rf = {
                        'n_estimators': n_estimators,
                        'max_depth': max_depth,
                        'rf_val_r2': float(rf_val_r2),
                    }

        best_svr = None
        for svr_c in svr_c_grid:
            for svr_eps in svr_eps_grid:
                svr_model, _ = train_svr(
                    X_train_sel, y_train_sc, float(svr_c), float(svr_eps),
                    use_cv=False
                )
                svr_val_pred = scaler_y.inverse_transform(
                    svr_model.predict(X_val_sel).reshape(-1, 1)
                ).ravel()
                svr_val_r2 = r2_score(y_val_arr, svr_val_pred)

                if best_svr is None or svr_val_r2 > best_svr['svr_val_r2']:
                    best_svr = {
                        'svr_C': float(svr_c),
                        'svr_eps': float(svr_eps),
                        'svr_val_r2': float(svr_val_r2),
                    }

        combo_val_r2 = (best_rf['rf_val_r2'] + best_svr['svr_val_r2']) / 2
        candidate = {
            'k_features': int(k),
            **best_rf,
            **best_svr,
            'combo_val_r2': float(combo_val_r2),
        }

        if best_overall is None or candidate['combo_val_r2'] > best_overall['combo_val_r2']:
            best_overall = candidate

    return best_overall