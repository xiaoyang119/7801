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
                    output_dir: str) -> pd.DataFrame:
    """
    Head-to-head benchmarking table: LR Challenger vs LGBM Champion.
    SR 11-7 requires comparison against a simpler benchmark model.
    """
    rows = []
    for name, model in models.items():
        probs = model.predict_proba(X_val)[:, 1]
        m     = compute_metrics(y_val, probs)
        m["Model"] = name
        rows.append(m)

    df = pd.DataFrame(rows).set_index("Model")
    print("\n[Benchmark] Performance comparison (Validation Set):")
    print(df.to_string())

    df.to_csv(os.path.join(output_dir, "benchmark_table.csv"))
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


# ── 5. Sensitivity analysis ──────────────────────────────────────────────────

def sensitivity_analysis(lgbm_pipeline, X_val: pd.DataFrame,
                          output_dir: str) -> pd.DataFrame:
    """
    Perturb each feature ±10% and measure resulting AUC change.
    SR 11-7: 'Sensitivity analysis tests how model outputs respond
    to changes in assumptions and inputs.'
    """
    from sklearn.metrics import roc_auc_score

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

    flagged = df_fair[df_fair["DIR (vs best)"] < 0.80]
    if not flagged.empty:
        print("\n⚠️  [Fairness] Following groups BELOW 4/5 Rule (DIR < 0.80):")
        print(flagged[["Age Group", "Model", "DIR (vs best)"]].to_string(index=False))
    else:
        print("\n✅  [Fairness] No group violates the 4/5 Rule (DIR ≥ 0.80 for all).")

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


# ── 7. Confusion matrix ──────────────────────────────────────────────────────

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
