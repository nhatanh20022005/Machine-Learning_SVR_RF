import io
import warnings

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.feature_selection import SelectKBest, f_regression
import time
from pandas.errors import EmptyDataError

from train_model import find_best_hyperparameters, init_state, run_pipeline

matplotlib.use('Agg')
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Superstore Profit · Math4AI",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

def load_css(path: str) -> None:
    with open(path, 'r', encoding='utf-8') as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css("style.css")

sns.set_theme(style='darkgrid')
plt.rcParams.update({
    'figure.facecolor':  '#0f0f18',
    'axes.facecolor':    '#0a0a0f',
    'axes.edgecolor':    '#1e1e2e',
    'axes.labelcolor':   '#6b6a84',
    'axes.spines.top':   False,
    'axes.spines.right': False,
    'xtick.color':       '#6b6a84',
    'ytick.color':       '#6b6a84',
    'text.color':        '#e8e6f0',
    'grid.color':        '#1e1e2e',
    'grid.alpha':        0.6,
    'font.family':       'monospace',
    'font.size':         10,
    'axes.titlesize':    12,
    'axes.titleweight':  '500',
    'axes.labelsize':    10,
})

C_ACCENT  = '#7c6af7'
C_ACCENT2 = '#4fc4a4'
C_ACCENT3 = '#f0834e'
C_MUTED   = '#252535'
C_BORDER  = '#1e1e2e'

init_state()

def fig_to_st(fig, use_container_width=True) -> None:
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=130, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    buf.seek(0)
    st.image(buf, use_container_width=use_container_width)
    plt.close(fig)


def metric_card(label: str, value: str, sub: str = '') -> None:
    sub_html = f'<span class="metric-sub">{sub}</span>' if sub else ''
    st.markdown(f"""
    <div class="metric-card">
      <span class="metric-label">{label}</span>
      <span class="metric-value">{value}</span>
      {sub_html}
    </div>""", unsafe_allow_html=True)


def section(title: str) -> None:
    st.markdown(
        f"<h2 class='section-header'>{title}</h2>",
        unsafe_allow_html=True,
    )


def format_bytes(size: int) -> str:
    units = ["B", "KB", "MB", "GB"]
    val = float(size)
    for unit in units:
        if val < 1024 or unit == units[-1]:
            return f"{val:.1f} {unit}" if unit != "B" else f"{int(val)} {unit}"
        val /= 1024


with st.sidebar:
    st.markdown("""
    <div style='padding:0 0 20px;border-bottom:1px solid #1e1e2e;margin-bottom:20px'>
      <div style='font-family:"DM Mono",monospace;font-size:9px;letter-spacing:.14em;
                  text-transform:uppercase;color:#6b6a84;margin-bottom:6px'>
            Machine Learning : Superstore Profit Prediction
      </div>
      <div style='font-family:"Syne",sans-serif;font-size:15px;font-weight:700;
                  letter-spacing:-.02em;color:#e8e6f0;line-height:1.25'>
        Superstore<br><span style="color:#7c6af7">Profit</span> Pred.
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<p class='upload-title'>Dataset CSV</p>", unsafe_allow_html=True)
    uploaded = st.file_uploader(
        "dataset_csv", type=['csv'],
        help="Cần có cột: Profit, Sales, Quantity, Discount …",
        label_visibility='collapsed',
    )
    if uploaded is None:
        st.caption("CSV only · Profit is required as target column")
    else:
        st.markdown(
            f"""
            <div class="upload-meta">
              <span class="upload-meta-label">Dataset loaded</span>
              <span class="upload-meta-name">{uploaded.name}</span>
              <span class="upload-meta-size">{format_bytes(uploaded.size)}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.markdown("**Hyperparameters**")
    st.session_state.setdefault("n_estimators", 200)
    st.session_state.setdefault("max_depth", 15)
    st.session_state.setdefault("svr_C", 10)
    st.session_state.setdefault("svr_eps", 0.10)
    st.session_state.setdefault("k_features", 8)

    auto_tune_btn = st.button("Auto-select best params")
    if auto_tune_btn:
        if uploaded is None:
            st.warning("Upload a CSV file before auto-tuning.")
        else:
            uploaded.seek(0)
            df_tune = pd.read_csv(uploaded, encoding='latin1')
            with st.spinner("Searching best hyperparameters on validation set…"):
                t0 = time.time()
                best = find_best_hyperparameters(df_tune)

            st.session_state.n_estimators = int(best["n_estimators"])
            st.session_state.max_depth = int(best["max_depth"])
            st.session_state.svr_C = int(round(best["svr_C"]))
            st.session_state.svr_eps = float(best["svr_eps"])
            st.session_state.k_features = int(best["k_features"])

            st.success(f"Auto-tuning completed in {time.time() - t0:.1f}s")
            st.caption(
                f"Best validation R² — RF: {best['rf_val_r2']:.4f} | "
                f"SVR: {best['svr_val_r2']:.4f}"
            )

    n_estimators = st.slider("RF – n_estimators", 50, 300, 200, 50,
                             key="n_estimators")
    max_depth    = st.slider("RF – max_depth",      5,  30,  15,  1,
                             key="max_depth")
    svr_C        = st.slider("SVR – C",             1,  50,  10,  1,
                             key="svr_C")
    svr_eps      = st.slider("SVR – epsilon",    0.01, 0.5, 0.1, 0.01,
                             key="svr_eps")
    k_features   = st.slider("Top K features",      4,  12,   8,  1,
                             key="k_features")
    test_size = 0.15
    st.caption("Split policy: 70% train · 15% validation · 15% test")

    st.markdown("---")
    run_btn = st.button("Run pipeline")

    st.markdown("---")
    st.markdown("**Navigation**")
    nav = st.radio(
        "",
        ["EDA", "Feature Selection",
         "Random Forest", "SVR",
         "Comparison", "Predict"],
        label_visibility='collapsed',
    )

    if st.session_state.pipeline_done:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(
            '<div class="status-pill"><span class="status-dot"></span>'
            'Pipeline ready</div>',
            unsafe_allow_html=True,
        )

st.markdown("""
<div class="page-hero">
  <div class="page-eyebrow">Profit Prediction · Random Forest &amp; SVR</div>
  <div class="page-title-main">Superstore <em>Profit</em> Analysis</div>
  <div class="page-subtitle">Feature selection · EDA · Model evaluation · Inference</div>
</div>
""", unsafe_allow_html=True)

if uploaded is not None:
    try:
        uploaded.seek(0)
        df_raw = pd.read_csv(uploaded, encoding='latin1')
    except EmptyDataError:
        st.error("Uploaded file is empty or unreadable. Please select a valid CSV.")
        st.stop()
    except Exception as e:
        st.error(f"Cannot read CSV: {e}")
        st.stop()

    if run_btn:
        with st.spinner("Running pipeline…"):
            t0 = time.time()
            run_pipeline(df_raw, n_estimators, max_depth,
                         svr_C, svr_eps, k_features, test_size)
            st.success(f"Pipeline completed in {time.time() - t0:.1f}s")
elif run_btn:
    st.warning("Upload a CSV file first.")

if not st.session_state.pipeline_done:
    st.info("Upload your CSV and click **Run pipeline** to begin.")
    st.markdown("""
    <div class="about-card">
      <span class="about-card-title">About this project</span>

      <div class="about-row">
        <span class="about-tag">EDA</span>
        <span class="about-desc">Profit distribution, correlation matrix, group statistics across Category, Segment and Region.</span>
      </div>
      <div class="about-row">
        <span class="about-tag">Preprocessing</span>
        <span class="about-desc">Feature engineering from dates, Label Encoding for categoricals, StandardScaler, IQR clipping on Profit.</span>
      </div>
      <div class="about-row">
        <span class="about-tag">Feature Sel.</span>
        <span class="about-desc">Embedded method via RF importance (Gini decrease) and Filter method via F-score (SelectKBest).</span>
      </div>
      <div class="about-row">
        <span class="about-tag">RF</span>
        <span class="about-desc">Random Forest Regressor with 5-fold cross-validation, residual analysis and feature importance visualization.</span>
      </div>
      <div class="about-row">
        <span class="about-tag">SVR</span>
        <span class="about-desc">Support Vector Regression with RBF kernel — scaled inputs, cross-validation, residual distribution.</span>
      </div>
      <div class="about-row">
        <span class="about-tag">Comparison</span>
        <span class="about-desc">Side-by-side R², MAE, RMSE and CV scores across both models.</span>
      </div>
      <div class="about-row">
        <span class="about-tag">Predict</span>
        <span class="about-desc">Real-time inference — enter order details and receive profit prediction from RF, SVR or both.</span>
      </div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

s = st.session_state


if nav == "EDA":
    section("Exploratory Data Analysis")

    col1, col2, col3, col4 = st.columns(4)
    with col1: metric_card("Rows",            f"{s.df.shape[0]:,}")
    with col2: metric_card("Features",        s.df.shape[1])
    with col3: metric_card("Missing values",  int(s.df.isnull().sum().sum()))
    with col4: metric_card("Mean profit",     f"${s.df['Profit'].mean():.2f}")

    st.markdown("#### Sample data")
    st.dataframe(s.df.head(20), use_container_width=True)

    st.markdown("#### Descriptive statistics")
    st.dataframe(s.df.describe(), use_container_width=True)

    st.markdown("#### Profit distribution")
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    fig.patch.set_facecolor('#0f0f18')
    axes[0].hist(s.df['Profit'], bins=80, color=C_ACCENT,
                 edgecolor='#0a0a0f', alpha=0.9, linewidth=.4)
    axes[0].set_title('Full distribution')
    axes[0].set_xlabel('Profit')
    clipped = s.df['Profit'].clip(-500, 500)
    axes[1].hist(clipped, bins=80, color=C_ACCENT2,
                 edgecolor='#0a0a0f', alpha=0.9, linewidth=.4)
    axes[1].set_title('Clipped [-500, 500]')
    axes[1].set_xlabel('Profit')
    plt.tight_layout(pad=1.5)
    fig_to_st(fig)

    c1, c2 = st.columns(2)
    with c1: st.metric("Skewness", f"{s.df['Profit'].skew():.3f}")
    with c2: st.metric("Kurtosis", f"{s.df['Profit'].kurtosis():.3f}")

    # Group profit
    st.markdown("#### Profit by group")
    palette = [C_ACCENT, C_ACCENT2, C_ACCENT3, '#a78bf6']
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    fig.patch.set_facecolor('#0f0f18')
    for ax, col in zip(axes, ['Category', 'Segment', 'Region']):
        if col in s.df.columns:
            avg    = s.df.groupby(col)['Profit'].mean().sort_values()
            colors = [palette[i % len(palette)] for i in range(len(avg))]
            avg.plot(kind='bar', ax=ax, color=colors, edgecolor='#0a0a0f',
                     linewidth=.4)
            ax.set_title(f'Mean profit by {col}')
            ax.tick_params(axis='x', rotation=30)
    plt.tight_layout(pad=1.5)
    fig_to_st(fig)

    # Correlation heatmap
    st.markdown("#### Correlation matrix")
    # Build matrix from the exact training feature space to keep all features.
    feature_cols = s.X.columns.tolist()
    df_corr = s.X.copy()
    df_corr['Profit'] = s.y.values
    corr = df_corr.corr(numeric_only=True)
    corr = corr.reindex(index=feature_cols + ['Profit'],
                        columns=feature_cols + ['Profit'])
    corr_display = corr.fillna(0.0)

    st.caption(f"Showing {len(feature_cols)} features + Profit target")

    nan_cols = corr.columns[corr.isna().all()].tolist()
    if nan_cols:
        st.caption(
            f"Columns with undefined correlation shown as 0: {', '.join(nan_cols)}"
        )

    fig, ax = plt.subplots(figsize=(14, 11))
    fig.patch.set_facecolor('#0f0f18')
    sns.heatmap(corr_display, annot=True, fmt='.2f', cmap='coolwarm',
                square=True, linewidths=.4, ax=ax, linecolor='#0a0a0f',
                annot_kws={'size': 8},
                cbar_kws={'label': 'Pearson r', 'shrink': .8})
    ax.set_title('Full feature correlation matrix', pad=14)
    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.yticks(fontsize=8)
    plt.tight_layout()
    fig_to_st(fig)

    st.markdown("#### Feature correlation with Profit")
    if 'Profit' in corr.columns:
        profit_corr = corr['Profit'].drop('Profit').dropna().sort_values(ascending=False)
        if not profit_corr.empty:
            fig, ax = plt.subplots(figsize=(10, 7))
            fig.patch.set_facecolor('#0f0f18')
            colors = [C_ACCENT2 if x > 0 else '#f05e5e' for x in profit_corr.values]
            profit_corr.plot(kind='barh', ax=ax, color=colors,
                             edgecolor='#0a0a0f', linewidth=.4)
            ax.set_xlabel('Pearson correlation with Profit')
            ax.set_title('Feature correlation with Profit (green = positive, red = negative)')
            plt.tight_layout()
            fig_to_st(fig)

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Highest correlation", f"{profit_corr.iloc[0]:.4f}",
                          profit_corr.index[0])
            with col2:
                st.metric("Lowest correlation", f"{profit_corr.iloc[-1]:.4f}",
                          profit_corr.index[-1])
            with col3:
                st.metric("Mean |correlation|", f"{abs(profit_corr).mean():.4f}")

elif nav == "Feature Selection":
    section("Feature Selection")

    imp_df = s.imp_df

    st.markdown("#### Embedded method — RF feature importance")
    with st.expander("Formula: Gini impurity decrease", expanded=False):
        st.markdown(r"""
        For a node $t$ splitting on feature $j$, the impurity decrease is:
        $$\Delta I(t,j) = I(t) - \frac{N_L}{N_t}I(t_L) - \frac{N_R}{N_t}I(t_R)$$
        Averaged across all trees and normalized to sum to 1:
        $$\widehat{FI}_j = \frac{FI_j}{\sum_m FI_m}$$
        """)

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor('#0f0f18')
    colors = [C_ACCENT if f in s.top_features else C_MUTED
              for f in imp_df['Feature']]
    ax.barh(imp_df['Feature'], imp_df['Importance'],
            color=colors, edgecolor='#0a0a0f', linewidth=.4)
    ax.set_xlabel('Importance score')
    ax.set_title(f'RF feature importance — top {k_features} selected (purple)')
    plt.tight_layout()
    fig_to_st(fig)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Selected features**")
        for i, f in enumerate(s.top_features, 1):
            st.markdown(f"`{i:02d}` {f}")
    with col2:
        st.markdown("**Importance scores**")
        st.dataframe(imp_df.reset_index(drop=True), use_container_width=True)

    # F-Score
    st.markdown("#### Filter method — F-score (SelectKBest)")
    with st.expander("Formula: F-statistic via Pearson r", expanded=False):
        st.markdown(r"""
        For feature $x_j$ and target $y$, the F-statistic is:
        $$F_j = \frac{r_j^2}{1-r_j^2}(n-2)$$
        where $r_j = \mathrm{corr}(x_j, y)$ and $n$ is sample size.
        Higher $F_j$ = stronger linear relationship with the target.
        """)

    sel = SelectKBest(score_func=f_regression, k='all')
    sel.fit(s.X_train_scaled, s.y_train_scaled)
    fs_df = pd.DataFrame({
        'Feature': s.X.columns,
        'F_Score': sel.scores_,
        'P_Value': sel.pvalues_,
    }).sort_values('F_Score', ascending=False)

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor('#0f0f18')
    colors2 = [C_ACCENT2 if f in s.top_features else C_MUTED
               for f in fs_df['Feature']]
    ax.barh(fs_df['Feature'], fs_df['F_Score'],
            color=colors2, edgecolor='#0a0a0f', linewidth=.4)
    ax.set_xlabel('F-score')
    ax.set_title('F-score per feature (teal = selected)')
    plt.tight_layout()
    fig_to_st(fig)


# RANDOM FOREST
elif nav == "Random Forest":
    section("Random Forest Regressor")

    c1, c2, c3 = st.columns(3)
    with c1: metric_card("R²",   f"{s.rf_r2:.4f}",   "test set")
    with c2: metric_card("MAE",  f"{s.rf_mae:.3f}",  "test set")
    with c3: metric_card("RMSE", f"{s.rf_rmse:.3f}", "test set")

    st.markdown("#### 5-fold cross-validation")
    rf_cv_df = pd.DataFrame({
        'Fold':     np.arange(1, len(s.rf_cv) + 1),
        'R2_Score': s.rf_cv,
    })
    best_idx = rf_cv_df['R2_Score'].idxmax()

    cv1, cv2, cv3 = st.columns(3)
    with cv1: metric_card("CV mean R²", f"{rf_cv_df['R2_Score'].mean():.4f}")
    with cv2: metric_card("CV std",     f"{rf_cv_df['R2_Score'].std():.4f}")
    with cv3: metric_card("Best fold",  f"{rf_cv_df.loc[best_idx,'R2_Score']:.4f}",
                          f"fold {int(rf_cv_df.loc[best_idx,'Fold'])}")

    st.dataframe(
        rf_cv_df.style.format({'R2_Score': '{:.4f}'}),
        use_container_width=True, hide_index=True,
    )

    fig_cv, ax_cv = plt.subplots(figsize=(8.5, 3.5))
    fig_cv.patch.set_facecolor('#0f0f18')
    ax_cv.bar(rf_cv_df['Fold'].astype(str), rf_cv_df['R2_Score'],
              color=C_ACCENT, edgecolor='#0a0a0f', linewidth=.4)
    cv_mean = rf_cv_df['R2_Score'].mean()
    ax_cv.axhline(cv_mean, color=C_ACCENT3, linestyle='--', linewidth=1.5,
                  label=f'Mean = {cv_mean:.4f}')
    ax_cv.set_xlabel('Fold')
    ax_cv.set_ylabel('R²')
    ax_cv.set_title('Cross-validation R² by fold')
    ax_cv.legend(fontsize=9)
    plt.tight_layout()
    fig_to_st(fig_cv)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    fig.patch.set_facecolor('#0f0f18')
    axes[0].scatter(s.y_test, s.rf_pred, alpha=0.3, color=C_ACCENT, s=12)
    lims = [min(s.y_test.min(), s.rf_pred.min()),
            max(s.y_test.max(), s.rf_pred.max())]
    axes[0].plot(lims, lims, color=C_ACCENT3, lw=1.5,
                 linestyle='--', label='Ideal')
    axes[0].set_xlabel('Actual profit')
    axes[0].set_ylabel('Predicted profit')
    axes[0].set_title(f'Actual vs Predicted  (R² = {s.rf_r2:.3f})')
    axes[0].legend(fontsize=9)
    residuals = s.y_test.values - s.rf_pred
    axes[1].hist(residuals, bins=60, color=C_ACCENT,
                 edgecolor='#0a0a0f', alpha=0.9, linewidth=.4)
    axes[1].axvline(0, color=C_ACCENT3, linestyle='--', lw=1.5)
    axes[1].set_xlabel('Residual')
    axes[1].set_title('Residual distribution')
    plt.suptitle('Random Forest — Model evaluation', fontsize=12,
                 fontweight='500', y=1.02)
    plt.tight_layout()
    fig_to_st(fig)

# SVR
elif nav == "SVR":
    section("Support Vector Regression")

    c1, c2, c3 = st.columns(3)
    with c1: metric_card("R²",   f"{s.svr_r2:.4f}",   "test set")
    with c2: metric_card("MAE",  f"{s.svr_mae:.3f}",  "test set")
    with c3: metric_card("RMSE", f"{s.svr_rmse:.3f}", "test set")

    st.markdown("#### 5-fold cross-validation")
    svr_cv_df = pd.DataFrame({
        'Fold':     np.arange(1, len(s.svr_cv) + 1),
        'R2_Score': s.svr_cv,
    })
    best_idx = svr_cv_df['R2_Score'].idxmax()

    cv1, cv2, cv3 = st.columns(3)
    with cv1: metric_card("CV mean R²", f"{svr_cv_df['R2_Score'].mean():.4f}")
    with cv2: metric_card("CV std",     f"{svr_cv_df['R2_Score'].std():.4f}")
    with cv3: metric_card("Best fold",  f"{svr_cv_df.loc[best_idx,'R2_Score']:.4f}",
                          f"fold {int(svr_cv_df.loc[best_idx,'Fold'])}")

    st.dataframe(
        svr_cv_df.style.format({'R2_Score': '{:.4f}'}),
        use_container_width=True, hide_index=True,
    )

    fig_cv, ax_cv = plt.subplots(figsize=(8.5, 3.5))
    fig_cv.patch.set_facecolor('#0f0f18')
    ax_cv.bar(svr_cv_df['Fold'].astype(str), svr_cv_df['R2_Score'],
              color=C_ACCENT3, edgecolor='#0a0a0f', linewidth=.4)
    cv_mean = svr_cv_df['R2_Score'].mean()
    ax_cv.axhline(cv_mean, color=C_ACCENT, linestyle='--', linewidth=1.5,
                  label=f'Mean = {cv_mean:.4f}')
    ax_cv.set_xlabel('Fold')
    ax_cv.set_ylabel('R²')
    ax_cv.set_title('Cross-validation R² by fold')
    ax_cv.legend(fontsize=9)
    plt.tight_layout()
    fig_to_st(fig_cv)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    fig.patch.set_facecolor('#0f0f18')
    axes[0].scatter(s.y_test, s.svr_pred, alpha=0.3, color=C_ACCENT3, s=12)
    lims = [min(s.y_test.min(), s.svr_pred.min()),
            max(s.y_test.max(), s.svr_pred.max())]
    axes[0].plot(lims, lims, color=C_ACCENT, lw=1.5,
                 linestyle='--', label='Ideal')
    axes[0].set_xlabel('Actual profit')
    axes[0].set_ylabel('Predicted profit')
    axes[0].set_title(f'Actual vs Predicted  (R² = {s.svr_r2:.3f})')
    axes[0].legend(fontsize=9)
    residuals_svr = s.y_test.values - s.svr_pred
    axes[1].hist(residuals_svr, bins=60, color=C_ACCENT3,
                 edgecolor='#0a0a0f', alpha=0.9, linewidth=.4)
    axes[1].axvline(0, color=C_ACCENT, linestyle='--', lw=1.5)
    axes[1].set_xlabel('Residual')
    axes[1].set_title('Residual distribution')
    plt.suptitle('SVR — Model evaluation', fontsize=12, fontweight='500', y=1.02)
    plt.tight_layout()
    fig_to_st(fig)

# COMPARISON
elif nav == "Comparison":
    section("Model Comparison")

    compare_df = pd.DataFrame({
        'Model':      ['Random Forest', 'SVR', 'Ensemble'],
        'CV R² Mean': [round(s.rf_cv.mean(), 4),  round(s.svr_cv.mean(), 4),
                       round(np.nanmean(s.ensemble_cv), 4)],
        'CV R² Std':  [round(s.rf_cv.std(),  4),  round(s.svr_cv.std(),  4),
                       round(np.nanstd(s.ensemble_cv), 4)],
        'Test MAE':   [round(s.rf_mae,  4),         round(s.svr_mae,  4),        round(s.ensemble_mae, 4)],
        'Test RMSE':  [round(s.rf_rmse, 4),         round(s.svr_rmse, 4),        round(s.ensemble_rmse, 4)],
        'Test R²':    [round(s.rf_r2,   4),         round(s.svr_r2,   4),         round(s.ensemble_r2,   4)],
    }).set_index('Model')

    st.dataframe(
        compare_df.style
        .highlight_max(axis=0, subset=['CV R² Mean', 'Test R²'],
                       color='rgba(79,196,164,.15)')
        .highlight_min(axis=0, subset=['Test MAE', 'Test RMSE'],
                       color='rgba(79,196,164,.15)'),
        use_container_width=True,
    )

    # Bar comparison
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    fig.patch.set_facecolor('#0f0f18')
    models = ['Random Forest', 'SVR', 'Ensemble']
    colors = [C_ACCENT, C_ACCENT3, C_ACCENT2]

    for ax, vals, title in zip(
        axes,
        [[s.rf_r2, s.svr_r2, s.ensemble_r2], [s.rf_mae, s.svr_mae, s.ensemble_mae], [s.rf_rmse, s.svr_rmse, s.ensemble_rmse]],
        ['R²  (higher is better)', 'MAE  (lower is better)', 'RMSE  (lower is better)'],
    ):
        bars = ax.bar(models, vals, color=colors, edgecolor='#0a0a0f',
                      linewidth=.4, width=0.45)
        ax.set_title(title, fontsize=10)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() * 1.02,
                    f'{val:.3f}', ha='center', fontsize=10, color='#e8e6f0')

    plt.suptitle('Test set comparison — RF vs SVR', fontsize=12,
                 fontweight='500', y=1.02)
    plt.tight_layout()
    fig_to_st(fig)

    # Overlay scatter
    st.markdown("#### Actual vs Predicted — both models")
    fig, ax = plt.subplots(figsize=(9, 5.5))
    fig.patch.set_facecolor('#0f0f18')
    ax.scatter(s.y_test, s.rf_pred,  alpha=0.22, color=C_ACCENT,  s=11,
               label='Random Forest')
    ax.scatter(s.y_test, s.svr_pred, alpha=0.22, color=C_ACCENT3, s=11,
               label='SVR')
    # Ensemble scatter
    if hasattr(s, 'ensemble_pred'):
        ax.scatter(s.y_test, s.ensemble_pred, alpha=0.22, color=C_ACCENT2, s=11,
                   label='Ensemble')
    # bounds include ensemble predictions if present
    preds_min = min(s.y_test.min(), s.rf_pred.min(), s.svr_pred.min())
    preds_max = max(s.y_test.max(), s.rf_pred.max(), s.svr_pred.max())
    if hasattr(s, 'ensemble_pred'):
        preds_min = min(preds_min, s.ensemble_pred.min())
        preds_max = max(preds_max, s.ensemble_pred.max())
    lims = [preds_min, preds_max]
    ax.plot(lims, lims, color='#e8e6f0', lw=1, linestyle='--', alpha=.4,
            label='Ideal')
    ax.set_xlabel('Actual profit')
    ax.set_ylabel('Predicted profit')
    ax.set_title('Actual vs Predicted — RF & SVR overlay')
    ax.legend(fontsize=10)
    plt.tight_layout()
    fig_to_st(fig)

# PREDICT
elif nav == "Predict":
    section("Predict Profit")

    def get_options(col: str) -> list:
        if col in s.le_dict:
            return s.le_dict[col].classes_.tolist()
        return []

    with st.form("predict_form"):
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Order details**")
            sales     = st.number_input("Sales ($)",    0.0, 50000.0, 200.0, 10.0)
            quantity  = st.number_input("Quantity",       1,    100,     7,    1)
            discount  = st.slider("Discount",          0.0,    0.8,  0.10, 0.05)
            ship_days = st.number_input("Ship days",      0,     30,     2,    1)
            order_year  = st.selectbox(
                "Order year",
                [2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023],
            )
            order_month = st.selectbox("Order month", list(range(1, 13)))
            order_dow   = st.selectbox(
                "Day of week",
                [(0, 'Monday'), (1, 'Tuesday'), (2, 'Wednesday'),
                 (3, 'Thursday'), (4, 'Friday'), (5, 'Saturday'), (6, 'Sunday')],
                format_func=lambda x: x[1],
            )

        with col2:
            st.markdown("**Product & shipping**")
            segment_opts  = get_options('Segment')      or ['Consumer', 'Corporate', 'Home Office']
            category_opts = get_options('Category')     or ['Furniture', 'Office Supplies', 'Technology']
            subcat_opts   = get_options('Sub-Category') or ['Phones', 'Chairs', 'Binders']
            shipmode_opts = get_options('Ship Mode')    or ['First Class', 'Second Class',
                                                            'Standard Class', 'Same Day']
            region_opts   = get_options('Region')       or ['East', 'West', 'Central', 'South']

            segment   = st.selectbox("Segment",      segment_opts)
            category  = st.selectbox("Category",     category_opts)
            sub_cat   = st.selectbox("Sub-category", subcat_opts)
            ship_mode = st.selectbox("Ship mode",    shipmode_opts)
            region    = st.selectbox("Region",       region_opts)

        model_choice = st.radio(
            "Model",
            ["Random Forest", "SVR", "Both"],
            horizontal=True,
        )
        submitted = st.form_submit_button("Predict profit")

    if submitted:
        user_input = {
            'Sales': sales, 'Quantity': quantity,
            'Discount': discount, 'Ship_Days': ship_days,
            'Segment': segment, 'Category': category,
            'Sub-Category': sub_cat, 'Ship Mode': ship_mode,
            'Region': region,
            'Order_Year': order_year,
            'Order_Month': order_month,
            'Order_DayOfWeek': order_dow[0] if isinstance(order_dow, tuple) else order_dow,
        }

        all_features = s.X.columns.tolist()
        row = {}
        for feat in all_features:
            if feat in s.le_dict:
                val = user_input.get(feat, '')
                try:
                    row[feat] = s.le_dict[feat].transform([str(val)])[0]
                except ValueError:
                    row[feat] = s.le_dict[feat].transform(
                        [s.le_dict[feat].classes_[0]]
                    )[0]
            else:
                row[feat] = user_input.get(feat, 0)

        df_inp   = pd.DataFrame([row])[all_features]
        inp_sc   = s.scaler_X.transform(df_inp)
        feat_idx = [all_features.index(f) for f in s.top_features]
        inp_sel  = inp_sc[:, feat_idx]

        def predict(model) -> float:
            pred_sc = model.predict(inp_sel)
            return s.scaler_y.inverse_transform(
                pred_sc.reshape(-1, 1)
            ).ravel()[0]

        results = {}
        if model_choice in ["Random Forest", "Both"]:
            results["Random Forest"] = (predict(s.rf_model), s.rf_r2)
        if model_choice in ["SVR", "Both"]:
            results["SVR"] = (predict(s.svr_model), s.svr_r2)

        cols = st.columns(len(results))
        for col, (name, (profit, r2)) in zip(cols, results.items()):
            with col:
                display_profit = round(float(profit), 2)
                if display_profit == 0.0:
                    sign, status, cls = "", "Break-even",  "profit-breakeven"
                elif display_profit > 0:
                    sign, status, cls = "+", "Profitable", "profit-positive"
                else:
                    sign, status, cls = "",  "Loss",       "profit-negative"

                st.markdown(f"""
                <div class="result-box {cls}">
                  <div style='font-family:"DM Mono",monospace;font-size:9px;
                              letter-spacing:.12em;text-transform:uppercase;
                              opacity:.6;margin-bottom:10px'>{name}</div>
                  <div style='font-family:"Syne",sans-serif;font-size:2.2rem;
                              font-weight:700;letter-spacing:-.04em;
                              margin-bottom:8px'>{sign}${display_profit:,.2f}</div>
                  <div style='font-family:"Syne",sans-serif;font-size:13px;
                              font-weight:600;letter-spacing:.02em'>{status}</div>
                  <div style='font-family:"DM Mono",monospace;font-size:10px;
                              opacity:.5;margin-top:8px'>R² = {r2:.4f}</div>
                </div>""", unsafe_allow_html=True)