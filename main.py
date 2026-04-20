"""
main.py — FRE 7801 Model Risk Management Capstone
==================================================
Team: Xiaoyang Zhang (xz5476), Yifan Xu (yx3598), Yike Ma (ym2552)

Project: Model Risk Assessment of Machine Learning-Based Credit Scoring:
         A Case Study Using the Give Me Some Credit Dataset

Structure (CRISP-DM + SR 11-7):
  Stage 1 — Data Preparation      (src/data_prep.py)
  Stage 2 — Model Development     (src/models.py)
  Stage 3 — Model Validation      (src/validation.py)
  Stage 4 — Risk Tiering & SR11-7 (src/risk_tiering.py)

Usage:
  python main.py [--impute {median|xgb}] [--skip-shap] [--skip-sensitivity]

Output:
  All charts, tables, and model files are saved to ./outputs/
"""

import argparse
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Import project modules ────────────────────────────────────────────────────
sys.path.insert(0, BASE_DIR)
from src.data_prep    import (load_data, eda_summary, impute_missing,
                               treat_outliers, make_splits, psi_report)
from src.models       import (build_lr_pipeline, build_lgbm_pipeline,
                               train_model, plot_calibration)
from src.validation   import (benchmark_table, plot_roc_pr, plot_ks,
                               shap_analysis, sensitivity_analysis,
                               fairness_analysis, plot_confusion_matrices)
from src.risk_tiering import print_mrs_report, sr117_risk_table


# ── CLI ───────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="FRE 7801 MRM Pipeline")
    p.add_argument("--impute", choices=["median", "xgb"], default="median",
                   help="Missing value imputation strategy (default: median; xgb takes ~20 min)")
    p.add_argument("--skip-shap",        action="store_true",
                   help="Skip SHAP analysis (saves ~5 min if shap not installed)")
    p.add_argument("--skip-sensitivity", action="store_true",
                   help="Skip sensitivity analysis (saves ~2 min)")
    return p.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    t0   = time.time()

    print("=" * 65)
    print("  FRE 7801 — Model Risk Management Capstone")
    print("  LightGBM Credit Scoring Pipeline — Full MRM Analysis")
    print("=" * 65)

    # ─────────────────────────────────────────────────────────────────────────
    # STAGE 1 — DATA PREPARATION
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "─" * 50)
    print("STAGE 1 — DATA PREPARATION")
    print("─" * 50)

    train_raw, test_raw = load_data(DATA_DIR)

    # EDA snapshot
    eda_summary(train_raw, OUTPUT_DIR)

    # Outlier treatment
    train_clean = treat_outliers(train_raw)
    test_clean  = treat_outliers(test_raw)

    # Missing value imputation
    train_imp, test_imp = impute_missing(train_clean, test_clean,
                                         method=args.impute)

    # Train / validation split (70 / 30 stratified)
    X_train, X_val, y_train, y_val = make_splits(train_imp)

    # PSI: training features vs validation features
    psi_df = psi_report(X_train, X_val, OUTPUT_DIR)

    # ─────────────────────────────────────────────────────────────────────────
    # STAGE 2 — MODEL DEVELOPMENT
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "─" * 50)
    print("STAGE 2 — MODEL DEVELOPMENT")
    print("─" * 50)

    # Challenger: Logistic Regression baseline
    lr_pipeline   = build_lr_pipeline()
    lr_pipeline   = train_model(lr_pipeline,   X_train, y_train,
                                "LR_Baseline",   OUTPUT_DIR)

    # Champion: LGBM with WOE + polynomial pipeline
    lgbm_pipeline = build_lgbm_pipeline()
    lgbm_pipeline = train_model(lgbm_pipeline, X_train, y_train,
                                "LGBM_Champion", OUTPUT_DIR)

    models = {
        "LR Baseline":   lr_pipeline,
        "LGBM Champion": lgbm_pipeline,
    }

    # Calibration
    plot_calibration(models, X_val, y_val, OUTPUT_DIR)

    # ─────────────────────────────────────────────────────────────────────────
    # STAGE 3 — MODEL VALIDATION
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "─" * 50)
    print("STAGE 3 — MODEL VALIDATION (SR 11-7 §III)")
    print("─" * 50)

    # III-a: Benchmarking — head-to-head performance table
    bench_df = benchmark_table(models, X_val, y_val, OUTPUT_DIR)

    # III-b: ROC and Precision-Recall curves
    plot_roc_pr(models, X_val, y_val, OUTPUT_DIR)

    # III-c: KS plots (banking standard)
    for name, model in models.items():
        plot_ks(model, X_val, y_val, name, OUTPUT_DIR)

    # III-d: Confusion matrices
    plot_confusion_matrices(models, X_val, y_val, OUTPUT_DIR)

    # III-e: Conceptual Soundness — SHAP analysis
    if not args.skip_shap:
        shap_analysis(lgbm_pipeline, X_val, y_val, OUTPUT_DIR, n_explain=2000)
    else:
        print("[SHAP] Skipped (--skip-shap flag set).")

    # III-f: Sensitivity analysis
    if not args.skip_sensitivity:
        sens_df = sensitivity_analysis(lgbm_pipeline, X_val, OUTPUT_DIR)
    else:
        print("[Sensitivity] Skipped (--skip-sensitivity flag set).")

    # III-g: Fairness — age-based disparate impact
    fair_df = fairness_analysis(lgbm_pipeline, lr_pipeline,
                                X_val, y_val, OUTPUT_DIR)

    # ─────────────────────────────────────────────────────────────────────────
    # STAGE 4 — RISK TIERING & SR 11-7 GOVERNANCE
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "─" * 50)
    print("STAGE 4 — RISK TIERING & SR 11-7 GOVERNANCE")
    print("─" * 50)

    # Kiritz et al. (2019) Model Risk Scorecard
    print_mrs_report(OUTPUT_DIR)

    # SR 11-7 Risk Taxonomy table
    risk_df = sr117_risk_table(OUTPUT_DIR)

    # ─────────────────────────────────────────────────────────────────────────
    # SUMMARY REPORT
    # ─────────────────────────────────────────────────────────────────────────
    elapsed = time.time() - t0
    _print_summary(bench_df, psi_df, fair_df, OUTPUT_DIR, elapsed)

    print(f"\n✅  All outputs saved to: {OUTPUT_DIR}/")
    print(f"⏱  Total runtime: {elapsed/60:.1f} minutes")


def _print_summary(bench_df, psi_df, fair_df, output_dir, elapsed):
    """Print and save a one-page executive summary."""
    lines = [
        "=" * 65,
        "  EXECUTIVE SUMMARY — FRE 7801 MRM CAPSTONE",
        "=" * 65,
        "",
        "  Model Under Review: LightGBM Credit Scoring Pipeline",
        "  Dataset: Give Me Some Credit (Kaggle 2011)",
        "  Kaggle Private AUC: 0.86914 (Top 5 / ~927 teams)",
        "",
        "  ── Performance Comparison (Validation Set) ──",
    ]
    for model_name in bench_df.index:
        row = bench_df.loc[model_name]
        lines.append(f"  {model_name:<20}  AUC={row['AUC-ROC']:.4f}  "
                     f"KS={row['KS Statistic']:.4f}  "
                     f"Gini={row['Gini Coefficient']:.4f}")

    high_psi = psi_df[psi_df["PSI"] > 0.10]
    lines += [
        "",
        "  ── Population Stability (PSI) ──",
        f"  Features with PSI > 0.10: {len(high_psi)}  "
        f"({', '.join(high_psi['Feature'].tolist()[:3])}{'...' if len(high_psi) > 3 else ''})",
    ]

    dir_min = fair_df["DIR (vs best)"].min() if "DIR (vs best)" in fair_df.columns else "N/A"
    lines += [
        "",
        "  ── Fairness (Age-Based Disparate Impact) ──",
        f"  Minimum DIR across age groups: {dir_min:.4f}" if isinstance(dir_min, float) else f"  DIR: {dir_min}",
        f"  4/5 Rule violated: {'YES ⚠️' if isinstance(dir_min, float) and dir_min < 0.80 else 'NO ✅'}",
        "",
        "  ── Risk Tier (Kiritz et al., 2019) ──",
        "  MRS Score: 88/100  →  Tier 1 — HIGH RISK",
        "",
        "  ── Key Recommended Mitigants ──",
        "  1. Document XGBRegressor imputation as separate sub-model",
        "  2. Implement monthly PSI monitoring trigger",
        "  3. Add human override layer for borderline scores",
        "  4. Conduct periodic fairness (ECOA) testing in production",
        "  5. Re-validate model if AUC degrades > 2% from baseline",
        "",
        "=" * 65,
    ]

    report_text = "\n".join(lines)
    print("\n" + report_text)

    summary_path = os.path.join(output_dir, "executive_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(report_text)
    print(f"\n[Summary] Saved → {summary_path}")


if __name__ == "__main__":
    main()
