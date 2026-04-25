"""
validation.py
-------------
Stage 3 – Model Validation (SR 11-7 §III / Lecture 2 five-part framework)

Covers:
  I.   Complexity classification  (see risk_tiering.py)
  II.  Conceptual soundness       → SHAP global & local explanations
  III. Quantitative validation    → Benchmarking, back-test (IS/OSOT),
                                    sensitivity, statistical tests
  IV.  Ongoing governance signals → PSI (see data_prep.py)
  V.   Fairness / bias testing    → Age-based disparate impact

Primary performance metrics (aligned with Kaggle & course):
  • AUC-ROC        (discriminatory power)
  • KS Statistic   (maximum separation; banking standard)
  • Gini Coeff.    = 2 × AUC − 1
  • F-beta Score   (beta=2 weights recall over precision; credit risk focus)
  • Average Precision (PR-AUC)
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    roc_auc_score, roc_curve, precision_recall_curve,
    average_precision_score, fbeta_score, confusion_matrix,
    ConfusionMatrixDisplay, brier_score_loss,
)

warnings.filterwarnings("ignore")

AGE_COL = "age"
AGE_GROUPS = {
    "Young (<30)":     (0,  30),
    "Middle (30–60)":  (30, 60),
    "Senior (>60)":    (60, 200),
}


# ── 1. Core metrics ──────────────────────────────────────────────────────────

def compute_metrics(y_true: pd.Series, y_prob: np.ndarray,
                    threshold: float = 0.5) -> dict:
    """Return a dict of all key performance metrics."""
    y_pred = (y_prob >= threshold).astype(int)

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    ks           = float(np.max(tpr - fpr))
    auc_roc      = roc_auc_score(y_true, y_prob)
    gini         = 2 * auc_roc - 1
    ap           = average_precision_score(y_true, y_prob)
    fb           = fbeta_score(y_true, y_pred, beta=2, zero_division=0)
    brier        = brier_score_loss(y_true, y_prob)

    return {
        "AUC-ROC":           round(auc_roc, 4),
        "KS Statistic":      round(ks,      4),
        "Gini Coefficient":  round(gini,    4),
        "PR-AUC (Avg Prec)": round(ap,      4),
        "F-beta (β=2)":      round(fb,      4),
        "Brier Score":       round(brier,   4),
    }


def benchmark_table(models: dict, X_val: pd.DataFrame, y_val: pd.Series,
                    output_dir: str,
                    X_train: pd.DataFrame | None = None,
                    y_train: pd.Series | None = None) -> pd.DataFrame:
    """
    Head-to-head benchmarking table: LR Challenger vs LGBM Champion.
    SR 11-7 requires comparison against a simpler benchmark model and
    outcomes analysis/back-testing against data not used to fit the model.
    """
    rows = []
    for name, model in models.items():
        val_probs = model.predict_proba(X_val)[:, 1]
        m = compute_metrics(y_val, val_probs)

        if X_train is not None and y_train is not None:
            train_probs = model.predict_proba(X_train)[:, 1]
            train_auc = roc_auc_score(y_train, train_probs)
            val_auc = m["AUC-ROC"]
            m["Train AUC-ROC"] = round(train_auc, 4)
            m["AUC Gap (Train-Val)"] = round(train_auc - val_auc, 4)

        m["Model"] = name
        rows.append(m)

    df = pd.DataFrame(rows).set_index("Model")
    print("\n[Benchmark] Performance comparison (Validation Set):")
    print(df.to_string())

    df.to_csv(os.path.join(output_dir, "benchmark_table.csv"))
    return df


def validation_standard_checks(bench_df: pd.DataFrame, psi_df: pd.DataFrame,
                               fair_df: pd.DataFrame, output_dir: str
                               ) -> pd.DataFrame:
    """
    Map generated evidence to SR 11-7-style validation expectations.
    This is a control checklist, not an official regulatory score.
    """
    lgbm = bench_df.loc["LGBM Champion"]
    lr = bench_df.loc["LR Baseline"]
    auc_lift = lgbm["AUC-ROC"] - lr["AUC-ROC"]
    max_psi = psi_df["PSI"].max()
    min_dir = fair_df["DIR (vs best)"].min()
    auc_gap = lgbm.get("AUC Gap (Train-Val)", np.nan)

    checks = [
        {
            "SR 11-7 Area": "Purpose and Scope",
            "Standard": "Intended use and model output are clearly defined",
            "Evidence": "Credit default probability for SeriousDlqin2yrs; project plan and main pipeline document the use case",
            "Status": "Pass",
            "Action": "Keep model-use limitations in final report and model inventory",
        },
        {
            "SR 11-7 Area": "Data Quality",
            "Standard": "Inputs are assessed for completeness, relevance, and quality",
            "Evidence": "EDA, missingness indicators, outlier caps, and PSI are generated",
            "Status": "Pass",
            "Action": "Document aged Kaggle data as a high limitation; do not claim production representativeness",
        },
        {
            "SR 11-7 Area": "Benchmarking",
            "Standard": "Complex model is compared with a simpler alternative",
            "Evidence": f"LGBM AUC {lgbm['AUC-ROC']:.4f} vs LR AUC {lr['AUC-ROC']:.4f}; lift {auc_lift:.4f}",
            "Status": "Pass" if auc_lift > 0 else "Review",
            "Action": "Explain why the incremental lift justifies added complexity",
        },
        {
            "SR 11-7 Area": "Outcomes Analysis",
            "Standard": "Performance is tested out of sample and checked for overfit",
            "Evidence": f"Train-validation AUC gap: {auc_gap:.4f}" if not np.isnan(auc_gap) else "Train-validation gap not computed",
            "Status": "Pass" if not np.isnan(auc_gap) and abs(auc_gap) <= 0.03 else "Review",
            "Action": "Investigate if AUC gap exceeds 3 percentage points",
        },
        {
            "SR 11-7 Area": "Ongoing Monitoring",
            "Standard": "Stability monitoring and escalation thresholds are defined",
            "Evidence": f"Max validation PSI is {max_psi:.4f}; monitoring thresholds are in the execution plan",
            "Status": "Pass" if max_psi < 0.10 else "Monitor",
            "Action": "Add production cadence and accountable owner in final report",
        },
        {
            "SR 11-7 Area": "Fair Lending / Use Risk",
            "Standard": "Material compliance risks and limitations are identified",
            "Evidence": f"Minimum age-group DIR is {min_dir:.4f}",
            "Status": "Issue" if min_dir < 0.80 else "Pass",
            "Action": "Treat as a material limitation; consider age-blind challenger and policy override thresholds",
        },
        {
            "SR 11-7 Area": "Governance",
            "Standard": "Limitations, approvals, model tier, and controls are documented",
            "Evidence": "MRS scorecard and SR 11-7 risk taxonomy are generated",
            "Status": "Pass",
            "Action": "State that independent validation and audit are recommended controls, not performed by this student project",
        },
    ]

    df = pd.DataFrame(checks)
    df.to_csv(os.path.join(output_dir, "sr117_validation_checklist.csv"), index=False)
    print("\n[SR 11-7] Validation standard checklist:")
    print(df[["SR 11-7 Area", "Status", "Action"]].to_string(index=False))
    return df


# ── 2. ROC & PR curve overlay ────────────────────────────────────────────────

def plot_roc_pr(models: dict, X_val: pd.DataFrame, y_val: pd.Series,
                output_dir: str) -> None:
    """Plot ROC and Precision-Recall curves for all models on one figure."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    colors = ["#2c7bb6", "#d7191c", "#1a9641", "#fdae61"]

    for i, (name, model) in enumerate(models.items()):
        probs = model.predict_proba(X_val)[:, 1]

        # ROC
        fpr, tpr, _ = roc_curve(y_val, probs)
        auc = roc_auc_score(y_val, probs)
        ax1.plot(fpr, tpr, color=colors[i], lw=2, label=f"{name}  (AUC={auc:.4f})")

        # PR
        prec, rec, _ = precision_recall_curve(y_val, probs)
        ap = average_precision_score(y_val, probs)
        ax2.plot(rec, prec, color=colors[i], lw=2, label=f"{name}  (AP={ap:.4f})")

    ax1.plot([0,1],[0,1],"k--",lw=1)
    ax1.set_xlabel("False Positive Rate")
    ax1.set_ylabel("True Positive Rate")
    ax1.set_title("ROC Curves")
    ax1.legend(loc="lower right", fontsize=9)

    ax2.axhline(y_val.mean(), color="gray", linestyle="--",
                label=f"Random baseline ({y_val.mean():.2%})")
    ax2.set_xlabel("Recall")
    ax2.set_ylabel("Precision")
    ax2.set_title("Precision-Recall Curves")
    ax2.legend(loc="upper right", fontsize=9)

    plt.suptitle("Model Performance — Validation Set", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "roc_pr_curves.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Curves] ROC/PR plot saved → {output_dir}/roc_pr_curves.png")


# ── 3. KS plot (banking standard) ────────────────────────────────────────────

def plot_ks(model, X_val: pd.DataFrame, y_val: pd.Series,
            model_name: str, output_dir: str) -> float:
    """
    KS separation chart — shows maximum separation between
    cumulative default / non-default distributions.
    Standard in bank model validation.
    """
    probs = model.predict_proba(X_val)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_val, probs)
    ks_val  = float(np.max(tpr - fpr))
    ks_idx  = int(np.argmax(tpr - fpr))
    ks_thr  = thresholds[min(ks_idx, len(thresholds) - 1)]

    # sklearn may return len(thresholds) == len(tpr) or len(tpr)-1 depending on version
    # align by taking the last n elements of tpr/fpr to match thresholds length
    n = len(thresholds)
    tpr_plot = tpr[-n:]
    fpr_plot = fpr[-n:]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(thresholds, tpr_plot, label="TPR (Cum. Default)", color="#d7191c")
    ax.plot(thresholds, fpr_plot, label="FPR (Cum. Non-Default)", color="#2c7bb6")
    ax.axvline(ks_thr, color="gray", linestyle="--",
               label=f"KS = {ks_val:.4f} @ threshold {ks_thr:.3f}")
    ax.fill_between(thresholds, fpr_plot, tpr_plot, alpha=0.15, color="purple")
    ax.set_xlabel("Score Threshold")
    ax.set_ylabel("Cumulative Rate")
    ax.set_title(f"KS Plot — {model_name}")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"ks_plot_{model_name.replace(' ','_')}.png"),
                dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[KS] {model_name} — KS = {ks_val:.4f}")
    return ks_val


# ── 4. SHAP conceptual soundness ─────────────────────────────────────────────

def shap_analysis(lgbm_pipeline, X_val: pd.DataFrame, y_val: pd.Series,
                  output_dir: str, n_explain: int = 2000) -> None:
    """
    SR 11-7 Conceptual Soundness: 'The model should make sense given
    what is known about the underlying phenomenon.'

    Global SHAP → feature importance ranking
    Local SHAP  → individual borrower explanation (waterfall plot)
    """
    try:
        import shap
    except ImportError:
        print("[SHAP] shap not installed. Run: pip install shap --break-system-packages")
        return

    print(f"\n[SHAP] Computing SHAP values on {n_explain:,} validation samples...")

    # Extract the LGBM model from the pipeline
    lgbm_model = lgbm_pipeline.named_steps["lgbm"]

    # Transform X through pipeline (all steps except final estimator)
    steps_excl_last = list(lgbm_pipeline.named_steps.items())[:-1]
    X_transformed = X_val.copy()
    for _, step in steps_excl_last:
        X_transformed = step.transform(X_transformed)

    X_sample = X_transformed[:n_explain]

    explainer   = shap.TreeExplainer(lgbm_model)
    shap_values = explainer.shap_values(X_sample)

    # For binary classification, shap_values may be a list [class0, class1]
    if isinstance(shap_values, list):
        sv = shap_values[1]
    else:
        sv = shap_values

    # ── Global summary plot (beeswarm)
    fig1, ax1 = plt.subplots(figsize=(10, 7))
    shap.summary_plot(sv, X_sample, show=False, max_display=20,
                      plot_type="dot")
    plt.title("SHAP Feature Importance — LGBM Champion (Global)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "shap_summary_beeswarm.png"),
                dpi=150, bbox_inches="tight")
    plt.close()

    # ── Global bar plot (mean |SHAP|)
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    shap.summary_plot(sv, X_sample, show=False, max_display=20,
                      plot_type="bar")
    plt.title("SHAP Mean |Value| — LGBM Champion (Feature Importance)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "shap_summary_bar.png"),
                dpi=150, bbox_inches="tight")
    plt.close()

    # ── Local: waterfall for highest-risk borrower
    risk_idx = int(np.argmax(sv.sum(axis=1)))
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    shap.waterfall_plot(
        shap.Explanation(
            values       = sv[risk_idx],
            base_values  = explainer.expected_value if not isinstance(
                               explainer.expected_value, list)
                           else explainer.expected_value[1],
            data         = X_sample[risk_idx],
        ),
        show=False,
    )
    plt.title("SHAP Waterfall — Highest-Risk Borrower (Local Explanation)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "shap_waterfall_high_risk.png"),
                dpi=150, bbox_inches="tight")
    plt.close()

    print(f"[SHAP] Plots saved to {output_dir}/")


# ── 5. Cross-validation overfitting investigation ────────────────────────────

def cross_validate_lgbm(X_train: pd.DataFrame, y_train: pd.Series,
                        output_dir: str, n_folds: int = 5) -> pd.DataFrame:
    """
    SR 11-7 Outcomes Analysis: investigate whether the AUC gap of 0.0644 reflects
    true overfitting or single-split variance.

    Uses StratifiedKFold CV with a computationally feasible LGBM config
    (n_estimators=1000, learning_rate=0.05; same pipeline architecture).
    Reports per-fold AUC, mean ± std, and produces a comparison chart.
    """
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import roc_auc_score

    try:
        from lightgbm import LGBMClassifier
    except ImportError:
        print("[CV] lightgbm not installed — skipping cross-validation.")
        return pd.DataFrame()

    # Import pipeline builder locally to avoid circular imports
    import sys, os as _os
    sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
    from src.models import build_lgbm_pipeline
    from sklearn.preprocessing import PowerTransformer
    from sklearn.pipeline import Pipeline

    print(f"\n[CV] 5-fold stratified cross-validation (LGBM architecture, "
          f"n_estimators=1000, lr=0.05 for computational efficiency)...")

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    fold_results = []
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train), 1):
        X_f_train = X_train.iloc[train_idx]
        y_f_train = y_train.iloc[train_idx]
        X_f_val   = X_train.iloc[val_idx]
        y_f_val   = y_train.iloc[val_idx]

        pipeline = build_lgbm_pipeline(n_estimators=1000, learning_rate=0.05)
        pipeline.fit(X_f_train, y_f_train)

        train_probs = pipeline.predict_proba(X_f_train)[:, 1]
        val_probs   = pipeline.predict_proba(X_f_val)[:, 1]

        train_auc = roc_auc_score(y_f_train, train_probs)
        val_auc   = roc_auc_score(y_f_val,   val_probs)
        gap       = train_auc - val_auc

        fpr, tpr, _ = roc_curve(y_f_val, val_probs)
        ks = float(np.max(tpr - fpr))

        fold_results.append({
            "Fold":      fold_idx,
            "Train AUC": round(train_auc, 4),
            "Val AUC":   round(val_auc,   4),
            "AUC Gap":   round(gap,       4),
            "KS":        round(ks,        4),
        })
        print(f"  Fold {fold_idx}: Train AUC={train_auc:.4f}  "
              f"Val AUC={val_auc:.4f}  Gap={gap:.4f}  KS={ks:.4f}")

    cv_df = pd.DataFrame(fold_results)

    mean_row = {
        "Fold":      "Mean",
        "Train AUC": round(cv_df["Train AUC"].mean(), 4),
        "Val AUC":   round(cv_df["Val AUC"].mean(),   4),
        "AUC Gap":   round(cv_df["AUC Gap"].mean(),   4),
        "KS":        round(cv_df["KS"].mean(),        4),
    }
    std_row = {
        "Fold":      "Std",
        "Train AUC": round(cv_df["Train AUC"].std(),  4),
        "Val AUC":   round(cv_df["Val AUC"].std(),    4),
        "AUC Gap":   round(cv_df["AUC Gap"].std(),    4),
        "KS":        round(cv_df["KS"].std(),         4),
    }
    summary_df = pd.concat([cv_df, pd.DataFrame([mean_row, std_row])], ignore_index=True)

    csv_path = os.path.join(output_dir, "cv_results_lgbm.csv")
    summary_df.to_csv(csv_path, index=False)
    print(f"[CV] Results saved → {csv_path}")

    # ── Visualization ──────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("5-Fold Cross-Validation — LGBM Champion\n"
                 "SR 11-7 Overfitting Investigation", fontsize=13, fontweight="bold")

    folds      = cv_df["Fold"].astype(str)
    x          = np.arange(len(folds))
    bar_w      = 0.35
    mean_train = cv_df["Train AUC"].mean()
    mean_val   = cv_df["Val AUC"].mean()
    std_val    = cv_df["Val AUC"].std()

    # Left panel: Train vs Val AUC per fold
    ax = axes[0]
    ax.bar(x - bar_w/2, cv_df["Train AUC"], bar_w, label="Train AUC", color="#1A3C80", alpha=0.85)
    ax.bar(x + bar_w/2, cv_df["Val AUC"],   bar_w, label="Val AUC",   color="#E8B84B", alpha=0.85)
    ax.axhline(mean_train, color="#1A3C80", linestyle="--", linewidth=1.2,
               label=f"Mean Train {mean_train:.4f}")
    ax.axhline(mean_val,   color="#E8B84B", linestyle="--", linewidth=1.2,
               label=f"Mean Val   {mean_val:.4f} ± {std_val:.4f}")
    ax.set_xticks(x)
    ax.set_xticklabels([f"Fold {i}" for i in folds])
    ax.set_ylim(0.80, 0.97)
    ax.set_ylabel("AUC-ROC")
    ax.set_title("Train vs. Validation AUC per Fold")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    # Right panel: AUC Gap per fold vs single-split gap
    ax2 = axes[1]
    colors = ["#C03A3A" if g > 0.03 else "#278C4A" for g in cv_df["AUC Gap"]]
    ax2.bar(x, cv_df["AUC Gap"], color=colors, alpha=0.85)
    ax2.axhline(cv_df["AUC Gap"].mean(), color="#E8B84B", linestyle="--", linewidth=1.5,
                label=f"CV Mean Gap {cv_df['AUC Gap'].mean():.4f}")
    ax2.axhline(0.0644, color="#C03A3A", linestyle=":", linewidth=1.5,
                label="Single-split gap 0.0644")
    ax2.axhline(0.03, color="grey", linestyle=":", linewidth=1.0,
                label="Review threshold 0.03")
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"Fold {i}" for i in folds])
    ax2.set_ylabel("AUC Gap (Train − Val)")
    ax2.set_title("AUC Gap per Fold vs. Thresholds")
    ax2.legend(fontsize=9)
    ax2.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(output_dir, "cv_auc_folds.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[CV] Chart saved → {plot_path}")

    mean_gap = cv_df["AUC Gap"].mean()
    print(f"\n[CV] Summary: Mean Val AUC={mean_val:.4f} ± {std_val:.4f}  |  "
          f"Mean AUC Gap={mean_gap:.4f}  |  "
          f"{'OVERFITTING CONFIRMED' if mean_gap > 0.03 else 'Gap within threshold'}")

    return summary_df


# ── 6. Sensitivity analysis ──────────────────────────────────────────────────

def sensitivity_analysis(lgbm_pipeline, X_val: pd.DataFrame,
                          output_dir: str) -> pd.DataFrame:
    """
    Perturb each feature ±10% and measure resulting average score change.
    SR 11-7: 'Sensitivity analysis tests how model outputs respond
    to changes in assumptions and inputs.'
    """
    base_prob = lgbm_pipeline.predict_proba(X_val)[:, 1]

    results = []
    numeric_feats = [f for f in X_val.columns]

    for feat in numeric_feats:
        for delta_pct, label in [(+0.10, "+10%"), (-0.10, "−10%")]:
            X_perturbed = X_val.copy()
            X_perturbed[feat] = X_perturbed[feat] * (1 + delta_pct)
            try:
                new_prob = lgbm_pipeline.predict_proba(X_perturbed)[:, 1]
                delta_auc = float(np.mean(new_prob - base_prob))
                results.append({
                    "Feature":    feat,
                    "Perturbation": label,
                    "Avg Score Δ": round(delta_auc, 5),
                })
            except Exception:
                pass

    df_sens = pd.DataFrame(results)
    # Rank features by absolute sensitivity
    abs_sens = (df_sens.groupby("Feature")["Avg Score Δ"]
                .apply(lambda x: x.abs().mean())
                .reset_index()
                .rename(columns={"Avg Score Δ": "Avg |Score Δ|"})
                .sort_values("Avg |Score Δ|", ascending=False))

    print("\n[Sensitivity] Top-10 most sensitive features:")
    print(abs_sens.head(10).to_string(index=False))

    abs_sens.to_csv(os.path.join(output_dir, "sensitivity_analysis.csv"), index=False)
    return abs_sens


# ── 6. Age-based fairness testing ────────────────────────────────────────────

def fairness_analysis(lgbm_pipeline, lr_pipeline,
                      X_val: pd.DataFrame, y_val: pd.Series,
                      output_dir: str) -> pd.DataFrame:
    """
    Disparate Impact Analysis by age group (young / middle / senior).

    SR 11-7 and ECOA: Models must not produce discriminatory outcomes.
    Disparate Impact Ratio (DIR) = Approval rate of group / Approval rate of most-favoured group.
    DIR < 0.80 is the "4/5 Rule" (EEOC standard; used by bank examiners).
    """
    X_val = X_val.reset_index(drop=True)
    y_val = y_val.reset_index(drop=True)

    lgbm_prob = lgbm_pipeline.predict_proba(X_val)[:, 1]
    lr_prob   = lr_pipeline.predict_proba(X_val)[:, 1]

    # Use median probability as approval threshold
    lgbm_thr = np.median(lgbm_prob)
    lr_thr   = np.median(lr_prob)

    rows = []
    for group_name, (lo, hi) in AGE_GROUPS.items():
        mask = (X_val[AGE_COL] >= lo) & (X_val[AGE_COL] < hi)
        if mask.sum() < 30:
            continue

        for model_name, prob, thr in [
            ("LGBM Champion", lgbm_prob, lgbm_thr),
            ("LR Baseline",   lr_prob,   lr_thr),
        ]:
            grp_prob    = prob[mask]
            grp_y       = y_val[mask]
            approval_rt = float((grp_prob < thr).mean())  # low P(default) → approved
            auc         = roc_auc_score(grp_y, grp_prob) if grp_y.nunique() > 1 else np.nan
            rows.append({
                "Age Group":     group_name,
                "Model":         model_name,
                "N":             int(mask.sum()),
                "Default Rate":  round(float(grp_y.mean()), 4),
                "Approval Rate": round(approval_rt, 4),
                "AUC":           round(auc, 4) if not np.isnan(auc) else "N/A",
            })

    df_fair = pd.DataFrame(rows)

    # Compute Disparate Impact Ratio (vs most-favoured group per model)
    for model_name in df_fair["Model"].unique():
        mask_m   = df_fair["Model"] == model_name
        best_ar  = df_fair.loc[mask_m, "Approval Rate"].max()
        df_fair.loc[mask_m, "DIR (vs best)"] = (
            df_fair.loc[mask_m, "Approval Rate"] / best_ar
        ).round(4)

    print("\n[Fairness] Age-based disparate impact analysis:")
    print(df_fair.to_string(index=False))

    # Age is an ECOA-permitted factor in empirically derived credit scoring systems.
    # DIR gaps below 0.80 are expected given actual default rate differences across
    # age groups (12.54% young vs 2.89% senior). LDA test documented separately.
    flagged = df_fair[df_fair["DIR (vs best)"] < 0.80]
    if not flagged.empty:
        print("\n[Fairness] Groups below 4/5 screening threshold (DIR < 0.80):")
        print(flagged[["Age Group", "Model", "DIR (vs best)"]].to_string(index=False))
        print("[Fairness] Note: age use is ECOA-permitted; gaps reflect actual default "
              "rate differences. Monitor semiannually.")

    # Bar chart
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=False)
    for ax, metric in zip(axes, ["Approval Rate", "AUC"]):
        for model_name in df_fair["Model"].unique():
            sub = df_fair[df_fair["Model"] == model_name]
            ax.bar(
                [f"{g}\n({model_name[:4]})" for g in sub["Age Group"]],
                sub[metric].astype(float),
                alpha=0.75, label=model_name,
            )
        ax.axhline(0.80, color="red", linestyle="--", label="4/5 Rule (0.80)")
        ax.set_title(f"{metric} by Age Group")
        ax.set_ylabel(metric)
        ax.legend(fontsize=7)

    plt.suptitle("Fairness / Disparate Impact by Age Group", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "fairness_age_groups.png"),
                dpi=150, bbox_inches="tight")
    plt.close()

    df_fair.to_csv(os.path.join(output_dir, "fairness_table.csv"), index=False)
    print(f"[Fairness] Table & chart saved → {output_dir}/")
    return df_fair


# ── 7. Age-blind less-discriminatory-alternative (LDA) test ─────────────────

def age_blind_comparison(X_train: pd.DataFrame, y_train: pd.Series,
                         X_val: pd.DataFrame,   y_val: pd.Series,
                         lgbm_champion,         lr_pipeline,
                         output_dir: str) -> pd.DataFrame:
    """
    SR 11-7 / ECOA Less-Discriminatory Alternative (LDA) test.

    Retrains LGBM with 'age' dropped. Compares against the champion and LR
    baseline on AUC, KS, and age-group Disparate Impact Ratio.

    A model that achieves similar AUC with higher DIR is a viable LDA.
    """
    import joblib
    import sys as _sys, os as _os
    _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
    from src.models import build_lgbm_pipeline

    print("\n[LDA] Training age-blind LGBM (age feature dropped)...")

    X_train_blind = X_train.drop(columns=[AGE_COL])
    X_val_blind   = X_val.drop(columns=[AGE_COL])

    pipeline_blind = build_lgbm_pipeline(n_estimators=1000, learning_rate=0.05)
    pipeline_blind.fit(X_train_blind, y_train)

    save_path = os.path.join(output_dir, "LGBM_Age_Blind.pkl")
    joblib.dump(pipeline_blind, save_path)
    print(f"[LDA] Age-blind model saved → {save_path}")

    # ── Performance comparison ─────────────────────────────────────────────────
    X_val = X_val.reset_index(drop=True)
    y_val = y_val.reset_index(drop=True)

    champion_prob  = lgbm_champion.predict_proba(X_val)[:, 1]
    lr_prob        = lr_pipeline.predict_proba(X_val)[:, 1]
    blind_prob     = pipeline_blind.predict_proba(X_val_blind.reset_index(drop=True))[:, 1]

    def _auc_ks(y_true, probs):
        auc = roc_auc_score(y_true, probs)
        fpr, tpr, _ = roc_curve(y_true, probs)
        ks = float(np.max(tpr - fpr))
        return round(auc, 4), round(ks, 4)

    champ_auc,  champ_ks  = _auc_ks(y_val, champion_prob)
    lr_auc,     lr_ks     = _auc_ks(y_val, lr_prob)
    blind_auc,  blind_ks  = _auc_ks(y_val, blind_prob)

    print(f"\n[LDA] Overall performance:")
    print(f"  Champion (age-inclusive): AUC={champ_auc}  KS={champ_ks}")
    print(f"  Age-Blind LGBM:           AUC={blind_auc}  KS={blind_ks}")
    print(f"  LR Baseline:              AUC={lr_auc}     KS={lr_ks}")

    # ── DIR comparison ─────────────────────────────────────────────────────────
    models_info = [
        ("LGBM Champion",  champion_prob, np.median(champion_prob)),
        ("LGBM Age-Blind", blind_prob,    np.median(blind_prob)),
        ("LR Baseline",    lr_prob,       np.median(lr_prob)),
    ]

    rows = []
    for group_name, (lo, hi) in AGE_GROUPS.items():
        mask = (X_val[AGE_COL] >= lo) & (X_val[AGE_COL] < hi)
        if mask.sum() < 30:
            continue
        for model_name, prob, thr in models_info:
            grp_prob    = prob[mask]
            grp_y       = y_val[mask]
            approval_rt = float((grp_prob < thr).mean())
            auc_grp     = roc_auc_score(grp_y, grp_prob) if grp_y.nunique() > 1 else np.nan
            rows.append({
                "Age Group":     group_name,
                "Model":         model_name,
                "N":             int(mask.sum()),
                "Default Rate":  round(float(grp_y.mean()), 4),
                "Approval Rate": round(approval_rt, 4),
                "AUC":           round(auc_grp, 4) if not np.isnan(auc_grp) else np.nan,
            })

    df = pd.DataFrame(rows)

    for model_name in df["Model"].unique():
        m       = df["Model"] == model_name
        best_ar = df.loc[m, "Approval Rate"].max()
        df.loc[m, "DIR (vs best)"] = (df.loc[m, "Approval Rate"] / best_ar).round(4)

    print("\n[LDA] DIR comparison by age group:")
    print(df.to_string(index=False))

    df.to_csv(os.path.join(output_dir, "age_blind_table.csv"), index=False)

    # ── Overall summary row ────────────────────────────────────────────────────
    perf_rows = [
        {"Model": "LGBM Champion",  "Overall AUC": champ_auc, "Overall KS": champ_ks,
         "Min DIR": df[df["Model"]=="LGBM Champion"]["DIR (vs best)"].min()},
        {"Model": "LGBM Age-Blind", "Overall AUC": blind_auc, "Overall KS": blind_ks,
         "Min DIR": df[df["Model"]=="LGBM Age-Blind"]["DIR (vs best)"].min()},
        {"Model": "LR Baseline",    "Overall AUC": lr_auc,    "Overall KS": lr_ks,
         "Min DIR": df[df["Model"]=="LR Baseline"]["DIR (vs best)"].min()},
    ]
    perf_df = pd.DataFrame(perf_rows)
    perf_df.to_csv(os.path.join(output_dir, "age_blind_perf_summary.csv"), index=False)

    # ── Visualization ──────────────────────────────────────────────────────────
    groups     = list(AGE_GROUPS.keys())
    model_names = ["LGBM Champion", "LGBM Age-Blind", "LR Baseline"]
    colors      = ["#1A3C80", "#E8B84B", "#4A9E6B"]
    x           = np.arange(len(groups))
    bar_w       = 0.25

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Less-Discriminatory Alternative Analysis\n"
                 "LGBM Champion vs Age-Blind LGBM vs LR Baseline",
                 fontsize=13, fontweight="bold")

    # Panel 1: Approval Rate
    ax = axes[0]
    for i, (mname, col) in enumerate(zip(model_names, colors)):
        sub = df[df["Model"] == mname].set_index("Age Group").reindex(groups)
        ax.bar(x + (i - 1) * bar_w, sub["Approval Rate"], bar_w,
               label=mname, color=col, alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(groups, fontsize=9)
    ax.set_ylabel("Approval Rate"); ax.set_title("Approval Rate by Age Group")
    ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)

    # Panel 2: DIR
    ax2 = axes[1]
    for i, (mname, col) in enumerate(zip(model_names, colors)):
        sub = df[df["Model"] == mname].set_index("Age Group").reindex(groups)
        ax2.bar(x + (i - 1) * bar_w, sub["DIR (vs best)"], bar_w,
                label=mname, color=col, alpha=0.85)
    ax2.axhline(0.80, color="red", linestyle="--", linewidth=1.2, label="4/5 Rule (0.80)")
    ax2.set_xticks(x); ax2.set_xticklabels(groups, fontsize=9)
    ax2.set_ylabel("DIR (vs best group)"); ax2.set_title("Disparate Impact Ratio by Age Group")
    ax2.legend(fontsize=8); ax2.grid(axis="y", alpha=0.3)
    ax2.set_ylim(0, 1.15)

    # Panel 3: Overall AUC + KS summary
    ax3 = axes[2]
    model_labels = ["Champion", "Age-Blind", "LR Base"]
    auc_vals     = [champ_auc, blind_auc, lr_auc]
    ks_vals      = [champ_ks,  blind_ks,  lr_ks]
    xi           = np.arange(len(model_labels))
    ax3.bar(xi - 0.2, auc_vals, 0.35, label="AUC-ROC", color=colors, alpha=0.85)
    ax3.bar(xi + 0.2, ks_vals,  0.35, label="KS Stat",  color=colors, alpha=0.5,
            edgecolor="black", linewidth=0.5)
    for j, (a, k) in enumerate(zip(auc_vals, ks_vals)):
        ax3.text(j - 0.2, a + 0.003, f"{a:.4f}", ha="center", fontsize=8)
        ax3.text(j + 0.2, k + 0.003, f"{k:.4f}", ha="center", fontsize=8)
    ax3.set_xticks(xi); ax3.set_xticklabels(model_labels)
    ax3.set_ylabel("Score"); ax3.set_title("Overall AUC & KS (Validation Set)")
    ax3.set_ylim(0, 1.05)
    from matplotlib.patches import Patch
    ax3.legend(handles=[Patch(color="grey", alpha=0.85, label="AUC-ROC (solid)"),
                         Patch(color="grey", alpha=0.5,  label="KS Stat (faded)")],
               fontsize=8)
    ax3.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(output_dir, "age_blind_comparison.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[LDA] Chart saved → {plot_path}")

    # Summary verdict
    auc_delta = round(blind_auc - champ_auc, 4)
    dir_delta = round(
        df[df["Model"]=="LGBM Age-Blind"]["DIR (vs best)"].min() -
        df[df["Model"]=="LGBM Champion"]["DIR (vs best)"].min(), 4)
    print(f"\n[LDA] AUC delta (age-blind − champion): {auc_delta:+.4f}")
    print(f"[LDA] Min DIR delta (age-blind − champion): {dir_delta:+.4f}")
    verdict = ("VIABLE LDA: fairness improves without material AUC loss (< 0.02)."
               if dir_delta > 0 and abs(auc_delta) < 0.02
               else "TRADE-OFF: review AUC cost vs. fairness gain before deciding.")
    print(f"[LDA] Verdict: {verdict}")

    return df


# ── 9. Confusion matrix ──────────────────────────────────────────────────────

def plot_confusion_matrices(models: dict, X_val: pd.DataFrame, y_val: pd.Series,
                            output_dir: str, threshold: float = 0.5) -> None:
    fig, axes = plt.subplots(1, len(models), figsize=(7 * len(models), 6))
    if len(models) == 1:
        axes = [axes]

    for ax, (name, model) in zip(axes, models.items()):
        probs = model.predict_proba(X_val)[:, 1]
        preds = (probs >= threshold).astype(int)
        cm    = confusion_matrix(y_val, preds)
        disp  = ConfusionMatrixDisplay(cm, display_labels=["No Default", "Default"])
        disp.plot(ax=ax, colorbar=False, cmap="Blues")
        ax.set_title(f"Confusion Matrix — {name}\n(threshold={threshold})")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "confusion_matrices.png"),
                dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Confusion] Matrices saved → {output_dir}/confusion_matrices.png")
