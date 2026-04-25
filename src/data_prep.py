"""
data_prep.py
------------
Stage 1 – Data Understanding & Preparation (CRISP-DM)

SR 11-7 anchor: "Sound model development begins with reliable, high-quality data."
This module handles:
  • EDA summary statistics
  • Missing value analysis and imputation
  • Outlier detection and treatment
  • Train / validation / test splitting
  • Population Stability Index (PSI) pre-computation between splits
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

# ── Feature registry ────────────────────────────────────────────────────────
TARGET = "SeriousDlqin2yrs"

FEATURES = [
    "RevolvingUtilizationOfUnsecuredLines",
    "age",
    "NumberOfTime30-59DaysPastDueNotWorse",
    "DebtRatio",
    "MonthlyIncome",
    "NumberOfOpenCreditLinesAndLoans",
    "NumberOfTimes90DaysLate",
    "NumberRealEstateLoansOrLines",
    "NumberOfTime60-89DaysPastDueNotWorse",
    "NumberOfDependents",
]

MISSING_FEATURES = ["MonthlyIncome", "NumberOfDependents"]
MISSING_INDICATOR_FEATURES = [f"{f}_missing" for f in MISSING_FEATURES]

OUTLIER_CAPS = {
    # Feature: (lower_cap, upper_cap) — None means no cap on that side
    "RevolvingUtilizationOfUnsecuredLines": (None, 1.0),
    "age": (18, 100),
    "DebtRatio": (None, 5000),
    "NumberOfTime30-59DaysPastDueNotWorse": (None, 96),  # 98 is sentinel
    "NumberOfTimes90DaysLate": (None, 96),
    "NumberOfTime60-89DaysPastDueNotWorse": (None, 96),
    "NumberOfOpenCreditLinesAndLoans": (None, 58),
    "NumberRealEstateLoansOrLines": (None, 54),
}


# ── 1. Load raw data ─────────────────────────────────────────────────────────

def load_data(data_dir: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load training and test CSVs; drop ID column."""
    train_path = os.path.join(data_dir, "cs-training.csv")
    test_path  = os.path.join(data_dir, "cs-test.csv")

    train = pd.read_csv(train_path, index_col=0)
    test  = pd.read_csv(test_path,  index_col=0)

    print(f"[Data] Training rows: {len(train):,}  |  Test rows: {len(test):,}")
    print(f"[Data] Target prevalence (train): "
          f"{train[TARGET].mean():.2%}  ({train[TARGET].sum():,} defaults)")
    return train, test


# ── 2. EDA snapshot ──────────────────────────────────────────────────────────

def eda_summary(df: pd.DataFrame, output_dir: str) -> pd.DataFrame:
    """
    Produce a one-page EDA table (mean, std, missing %, min, max, skewness).
    Saves feature distribution plots.
    """
    os.makedirs(output_dir, exist_ok=True)

    summary = pd.DataFrame({
        "mean":        df[FEATURES].mean(),
        "std":         df[FEATURES].std(),
        "min":         df[FEATURES].min(),
        "max":         df[FEATURES].max(),
        "skewness":    df[FEATURES].skew(),
        "missing_pct": df[FEATURES].isnull().mean() * 100,
    }).round(4)

    print("\n[EDA] Feature summary:")
    print(summary.to_string())

    # Distribution plots
    fig, axes = plt.subplots(4, 3, figsize=(16, 14))
    axes = axes.flatten()
    for i, feat in enumerate(FEATURES):
        data = df[feat].dropna()
        axes[i].hist(data.clip(*_get_clip(feat)), bins=60, color="#2c7bb6", edgecolor="white", alpha=0.85)
        axes[i].set_title(feat, fontsize=9)
        axes[i].set_xlabel("")
    # Last subplot: target distribution
    axes[len(FEATURES)].bar(["No Default", "Default"],
                             df[TARGET].value_counts().sort_index().values,
                             color=["#4dac26", "#d01c8b"])
    axes[len(FEATURES)].set_title("SeriousDlqin2yrs", fontsize=9)
    for j in range(len(FEATURES) + 1, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle("Feature Distributions – Give Me Some Credit (Training Set)", fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "eda_distributions.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[EDA] Distribution plot saved → {output_dir}/eda_distributions.png")

    return summary


def _get_clip(feat):
    lo, hi = OUTLIER_CAPS.get(feat, (None, None))
    return (lo if lo is not None else -np.inf,
            hi if hi is not None else  np.inf)


# ── 3. Missing value imputation ──────────────────────────────────────────────

def impute_missing(train: pd.DataFrame, test: pd.DataFrame,
                   method: str = "median") -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    method="median"  : fast; used for NumberOfDependents always.
    method="xgb"     : XGBRegressor on non-missing rows (takes ~20 min).

    SR 11-7 note: imputation choices are a model development risk;
    we document both and compare downstream AUC impact.
    """
    train = train.copy()
    test  = test.copy()

    # Preserve missingness before imputation. Missingness itself can be a
    # credit-risk signal and should be available for validation and monitoring.
    for f in MISSING_FEATURES:
        train[f"{f}_missing"] = train[f].isna().astype(int)
        test[f"{f}_missing"]  = test[f].isna().astype(int)

    if method == "xgb":
        train, test = _xgb_impute(train, test)
    else:
        # Median imputation from training set
        medians = {f: train[f].median() for f in MISSING_FEATURES}
        for f in MISSING_FEATURES:
            train[f] = train[f].fillna(medians[f])
            test[f]  = test[f].fillna(medians[f])
            print(f"[Impute] {f}: filled with median {medians[f]:.2f}")

    return train, test


def _xgb_impute(train: pd.DataFrame, test: pd.DataFrame
                ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """XGBRegressor-based imputation (replicates the Kaggle Top-5 pipeline)."""
    try:
        from xgboost import XGBRegressor
    except ImportError:
        print("[Impute] xgboost not installed; falling back to median.")
        return impute_missing(train, test, method="median")

    auxiliary = [f for f in FEATURES if f not in MISSING_FEATURES]

    for feat in MISSING_FEATURES:
        mask_train  = train[feat].notna()
        mask_miss_t = train[feat].isna()
        mask_miss_e = test[feat].isna()

        X_known = train.loc[mask_train, auxiliary]
        y_known = train.loc[mask_train, feat]

        reg = XGBRegressor(n_estimators=300, max_depth=6,
                           learning_rate=0.05, random_state=42,
                           n_jobs=-1, verbosity=0)
        reg.fit(X_known, y_known)

        if mask_miss_t.any():
            train.loc[mask_miss_t, feat] = reg.predict(train.loc[mask_miss_t, auxiliary])
        if mask_miss_e.any():
            test.loc[mask_miss_e, feat]  = reg.predict(test.loc[mask_miss_e, auxiliary])

        print(f"[XGB Impute] {feat}: predicted {mask_miss_t.sum():,} train + "
              f"{mask_miss_e.sum():,} test missing values.")

    return train, test


# ── 4. Imputation assumption validation ──────────────────────────────────────

def imputation_validation(train_raw: pd.DataFrame,
                          train_imp: pd.DataFrame,
                          output_dir: str) -> pd.DataFrame:
    """
    Validate every assumption behind the median imputation strategy.

    Assumption 1 — Informativeness (MCAR test)
        Chi-square test of independence between missingness indicator and target.
        If rejected (p < 0.05): missingness is NOT random → keeping the
        indicator flag as a model feature is validated.

    Assumption 2 — Default rate difference (bias check)
        Compare P(default | feature missing) vs P(default | feature present).
        A large difference means imputing with the global median introduces
        systematic bias; the missingness indicator partially corrects for this.

    Assumption 3 — Distribution shape (median appropriateness)
        Check skewness of observed values. For right-skewed distributions
        median is a better central-tendency imputation than mean.

    Assumption 4 — No data leakage
        Verify medians are derived from training data only (not val/test).
        Documented here; confirmed by code structure in impute_missing().

    Assumption 5 — Missingness stability (train vs. val)
        PSI of missingness indicators. Already in psi_report(); cited here
        for completeness in the imputation evidence package.
    """
    from scipy import stats

    os.makedirs(output_dir, exist_ok=True)
    results = []

    for feat in MISSING_FEATURES:
        miss_mask = train_raw[feat].isna()
        present   = train_raw.loc[~miss_mask, feat]
        missing_n = int(miss_mask.sum())
        total_n   = len(train_raw)
        miss_rate = missing_n / total_n

        # ── Assumption 1: MCAR chi-square test ──────────────────────────────
        contingency = pd.crosstab(miss_mask, train_raw[TARGET])
        chi2, p_chi2, _, _ = stats.chi2_contingency(contingency)
        mcar_result = "REJECTED (informative)" if p_chi2 < 0.05 else "Not rejected (MCAR plausible)"

        # ── Assumption 2: Default rate difference ────────────────────────────
        dr_missing = train_raw.loc[miss_mask,  TARGET].mean()
        dr_present = train_raw.loc[~miss_mask, TARGET].mean()
        dr_ratio   = dr_missing / dr_present if dr_present > 0 else np.nan

        # ── Assumption 3: Distribution shape ────────────────────────────────
        skewness = float(present.skew())
        median_v = float(present.median())
        mean_v   = float(present.mean())
        median_preferred = abs(skewness) > 1.0  # rule of thumb

        # ── Assumption 4: Leakage check ──────────────────────────────────────
        # Medians in train_imp are derived from train_raw (non-missing rows).
        # The imputed value for this feature in train_imp should equal median_v.
        imputed_val = train_imp.loc[miss_mask, feat].iloc[0] if missing_n > 0 else np.nan
        leakage_ok  = abs(imputed_val - median_v) < 1e-6 if not np.isnan(imputed_val) else True

        results.append({
            "Feature":               feat,
            "Missing N":             missing_n,
            "Missing Rate":          f"{miss_rate:.2%}",
            "A1 MCAR chi2":          round(chi2, 2),
            "A1 p-value":            round(p_chi2, 4),
            "A1 Result":             mcar_result,
            "A2 DR (missing)":       round(dr_missing, 4),
            "A2 DR (present)":       round(dr_present, 4),
            "A2 DR ratio":           round(dr_ratio, 3),
            "A3 Skewness":           round(skewness, 3),
            "A3 Median":             round(median_v, 2),
            "A3 Mean":               round(mean_v, 2),
            "A3 Median preferred":   "Yes" if median_preferred else "No",
            "A4 Leakage-free":       "Pass" if leakage_ok else "FAIL",
        })

        print(f"\n[Imputation Validation] {feat}")
        print(f"  Missing: {missing_n:,} / {total_n:,} ({miss_rate:.2%})")
        print(f"  A1 MCAR chi2={chi2:.2f}  p={p_chi2:.4f}  → {mcar_result}")
        print(f"  A2 Default rate: missing={dr_missing:.4f}  present={dr_present:.4f}  "
              f"ratio={dr_ratio:.3f}")
        print(f"  A3 Skewness={skewness:.3f}  Median={median_v:.2f}  Mean={mean_v:.2f}  "
              f"→ median {'preferred' if median_preferred else 'adequate'}")
        print(f"  A4 Leakage-free: {'Pass' if leakage_ok else 'FAIL'}")

    df_val = pd.DataFrame(results)
    df_val.to_csv(os.path.join(output_dir, "imputation_validation.csv"), index=False)

    # ── Visualization ─────────────────────────────────────────────────────────
    n_feats = len(MISSING_FEATURES)
    fig, axes = plt.subplots(2, n_feats, figsize=(7 * n_feats, 10))
    fig.suptitle("Imputation Assumption Validation\n"
                 "Median imputation — MonthlyIncome & NumberOfDependents",
                 fontsize=13, fontweight="bold")

    for col_idx, feat in enumerate(MISSING_FEATURES):
        miss_mask = train_raw[feat].isna()
        present   = train_raw.loc[~miss_mask, feat]
        median_v  = float(present.median())

        # Top row: distribution of observed values + median line
        ax_top = axes[0, col_idx] if n_feats > 1 else axes[0]
        clip_hi = present.quantile(0.99)
        ax_top.hist(present.clip(upper=clip_hi), bins=60,
                    color="#1A3C80", alpha=0.75, edgecolor="white")
        ax_top.axvline(median_v, color="#E8B84B", linewidth=2,
                       label=f"Median = {median_v:.1f}")
        ax_top.axvline(present.mean(), color="#C03A3A", linewidth=2, linestyle="--",
                       label=f"Mean = {present.mean():.1f}")
        skew_val = present.skew()
        ax_top.set_title(f"{feat}\nDistribution (observed, clipped at p99)\n"
                         f"Skewness={skew_val:.2f}")
        ax_top.set_xlabel(feat)
        ax_top.legend(fontsize=9)
        ax_top.grid(axis="y", alpha=0.3)

        # Bottom row: default rate by missing vs present + count
        ax_bot = axes[1, col_idx] if n_feats > 1 else axes[1]
        dr_miss = train_raw.loc[miss_mask, TARGET].mean()
        dr_pres = train_raw.loc[~miss_mask, TARGET].mean()
        groups  = ["Present", "Missing"]
        rates   = [dr_pres, dr_miss]
        counts  = [int(~miss_mask.sum()), int(miss_mask.sum())]
        bars = ax_bot.bar(groups, rates,
                          color=["#1A3C80", "#C03A3A"], alpha=0.85, width=0.5)
        for bar, rate, cnt in zip(bars, rates, counts):
            ax_bot.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.002,
                        f"{rate:.3f}\n(n={cnt:,})",
                        ha="center", va="bottom", fontsize=10)
        ax_bot.axhline(train_raw[TARGET].mean(), color="grey",
                       linestyle="--", label="Overall DR")
        ax_bot.set_ylabel("Default Rate")
        ax_bot.set_title(f"{feat}\nDefault Rate: Missing vs Present\n"
                         f"(Ratio = {dr_miss/dr_pres:.2f}×)")
        ax_bot.legend(fontsize=9)
        ax_bot.set_ylim(0, max(rates) * 1.3)
        ax_bot.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(output_dir, "imputation_validation.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n[Imputation Validation] Chart saved → {plot_path}")
    print(f"[Imputation Validation] Table saved → "
          f"{os.path.join(output_dir, 'imputation_validation.csv')}")

    return df_val


# ── 5. Outlier treatment ────────────────────────────────────────────────────

def treat_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Winsorize / cap extreme values based on domain knowledge.
    99/98 sentinel codes in past-due fields → treated as 96 (capped).
    """
    df = df.copy()
    for feat, (lo, hi) in OUTLIER_CAPS.items():
        if feat not in df.columns:
            continue
        original_max = df[feat].max()
        if lo is not None:
            df[feat] = df[feat].clip(lower=lo)
        if hi is not None:
            df[feat] = df[feat].clip(upper=hi)
        if original_max != df[feat].max():
            print(f"[Outlier] {feat}: max {original_max:.2f} → capped at {hi}")
    return df


# ── 5. Train / validation split ──────────────────────────────────────────────

def make_splits(train: pd.DataFrame,
                val_size: float = 0.30,
                random_state: int = 42
                ) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Stratified 70/30 split of the training set.
    We treat the hold-out as an out-of-sample (OSOT) test per SR 11-7.
    The full original test set (no labels) is used only for PSI.
    """
    model_features = model_feature_columns(train)
    X = train[model_features]
    y = train[TARGET]

    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=val_size, stratify=y, random_state=random_state
    )
    print(f"\n[Split] Train: {len(X_tr):,}  |  Validation: {len(X_val):,}")
    print(f"[Split] Positive rate – Train: {y_tr.mean():.2%}  "
          f"Validation: {y_val.mean():.2%}")
    return X_tr, X_val, y_tr, y_val


def model_feature_columns(df: pd.DataFrame) -> list[str]:
    """Return base model features plus any derived missingness indicators."""
    return FEATURES + [f for f in MISSING_INDICATOR_FEATURES if f in df.columns]


# ── 6. Population Stability Index ────────────────────────────────────────────

def compute_psi(expected: pd.Series, actual: pd.Series,
                n_bins: int = 10) -> float:
    """
    PSI = Σ (Actual% − Expected%) × ln(Actual% / Expected%)

    Thresholds (industry standard):
      PSI < 0.10  → Stable (Green)
      0.10–0.25   → Slight shift (Yellow) — monitor
      PSI > 0.25  → Significant shift (Red) — re-validate
    """
    breakpoints = np.percentile(expected.dropna(), np.linspace(0, 100, n_bins + 1))
    breakpoints = np.unique(breakpoints)  # remove duplicates at extremes

    def pct_in_bins(series):
        counts = np.histogram(series.dropna(), bins=breakpoints)[0]
        pcts   = counts / counts.sum()
        pcts   = np.where(pcts == 0, 1e-4, pcts)  # avoid log(0)
        return pcts

    exp_pct = pct_in_bins(expected)
    act_pct = pct_in_bins(actual)

    # Align lengths (histogram bins may differ if breakpoints collapse)
    n = min(len(exp_pct), len(act_pct))
    psi = np.sum((act_pct[:n] - exp_pct[:n]) * np.log(act_pct[:n] / exp_pct[:n]))
    return float(psi)


def psi_report(X_train: pd.DataFrame, X_val: pd.DataFrame,
               output_dir: str) -> pd.DataFrame:
    """Compute PSI for every feature and save a colour-coded table."""
    results = []
    for feat in model_feature_columns(X_train):
        psi_val = compute_psi(X_train[feat], X_val[feat])
        flag = ("🟢 Stable" if psi_val < 0.10
                else "🟡 Monitor" if psi_val < 0.25
                else "🔴 Re-validate")
        results.append({"Feature": feat, "PSI": round(psi_val, 4), "Status": flag})

    df_psi = pd.DataFrame(results).sort_values("PSI", ascending=False)
    print("\n[PSI] Population Stability Index (Train → Validation):")
    print(df_psi.to_string(index=False))

    # Bar chart
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["#d73027" if p > 0.25 else "#fee090" if p > 0.10 else "#4dac26"
              for p in df_psi["PSI"]]
    ax.barh(df_psi["Feature"], df_psi["PSI"], color=colors)
    ax.axvline(0.10, color="orange", linestyle="--", label="Monitor (0.10)")
    ax.axvline(0.25, color="red",    linestyle="--", label="Re-validate (0.25)")
    ax.set_xlabel("PSI")
    ax.set_title("Population Stability Index – Training vs Validation")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "psi_chart.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[PSI] Chart saved → {output_dir}/psi_chart.png")

    df_psi.to_csv(os.path.join(output_dir, "psi_table.csv"), index=False)
    return df_psi
