"""
risk_tiering.py
---------------
Stage 4 – Model Risk Tiering & SR 11-7 Governance Mapping

Implements the Kiritz, Ravitz & Levonian (2019) Model Risk Scorecard (MRS).
Also produces the SR 11-7 Risk Taxonomy summary expected in the capstone report.

MRS Scorecard (max 100 pts):
  • Materiality (30 pts)           — business/financial impact
  • Regulatory Exposure (30 pts)   — regulatory scrutiny & reporting
  • Operational Factor (10 pts)    — automated use, override controls
  • Model Complexity (30 pts)      — algorithmic complexity, interpretability

Tier Assignment:
  ≥ 80 pts  →  Tier 1 (High Risk)
  50–79 pts →  Tier 2 (Medium Risk)
  < 50 pts  →  Tier 3 (Low Risk)
"""

import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd


# ── 1. Scorecard definition ──────────────────────────────────────────────────

SCORECARD = {
    "Materiality": {
        "weight": 30,
        "factors": {
            "Financial Impact":
                "Model directly scores 150K borrowers; credit approval/denial decisions → High (10/10)",
            "Business Criticality":
                "Credit scoring is a core banking function → High (10/10)",
            "Strategic Importance":
                "Demonstrates ML superiority over LR scorecards → Moderate (7/10)",
        },
        "raw_score": 27,          # sum of factor scores above, scaled to weight
        "justification":
            "Model affects loan decisions for 150,000 borrowers. "
            "Credit scoring is core to financial institution solvency. Score: 27/30."
    },
    "Regulatory Exposure": {
        "weight": 30,
        "factors": {
            "SR 11-7 Applicability":
                "Directly subject to Federal Reserve SR 11-7 → High (10/10)",
            "ECOA / Fair Lending":
                "Age, income characteristics create disparate impact risk → Moderate (8/10)",
            "Audit Trail":
                "LightGBM is opaque; SHAP partially satisfies explainability → Moderate (7/10)",
        },
        "raw_score": 25,
        "justification":
            "SR 11-7 compliance is mandatory. Age-based features create ECOA exposure. "
            "LightGBM lacks native explainability, increasing examiner risk. Score: 25/30."
    },
    "Operational Factor": {
        "weight": 10,
        "factors": {
            "Automation Level":
                "Fully automated sklearn Pipeline; no human override layer → High (8/10)",
            "Override Controls":
                "No documented override policy → Elevated (7/10)",
        },
        "raw_score": 8,
        "justification":
            "Pipeline is fully automated with no manual override documented. "
            "Operational risk is elevated. Score: 8/10."
    },
    "Model Complexity": {
        "weight": 30,
        "factors": {
            "Algorithmic Complexity":
                "LGBM + WOE encoding + polynomial features + feature selection "
                "→ Multi-layer non-linear model → High (10/10)",
            "Interpretability":
                "LightGBM is a black box; SHAP provides post-hoc explanation only → High (9/10)",
            "Data Transformations":
                "PowerTransformer + KMeans clustering + WOE → 3 sequential transforms "
                "compound uncertainty → High (9/10)",
        },
        "raw_score": 28,
        "justification":
            "Pipeline has 5 sequential steps including polynomial feature expansion. "
            "LGBM is inherently non-interpretable. WOE encoding on cluster labels introduces "
            "double-layer non-linearity. Score: 28/30."
    },
}


def compute_mrs() -> tuple[int, str, pd.DataFrame]:
    """
    Compute the total Model Risk Score (MRS) and assign tier.
    Returns (total_score, tier, detail_df).
    """
    rows = []
    total = 0
    for dimension, data in SCORECARD.items():
        score = data["raw_score"]
        weight = data["weight"]
        total += score
        rows.append({
            "Dimension":     dimension,
            "Max Points":    weight,
            "Score":         score,
            "Score %":       f"{score / weight:.0%}",
            "Justification": data["justification"],
        })

    if total >= 80:
        tier = "Tier 1 — HIGH RISK"
        tier_color = "#d73027"
    elif total >= 50:
        tier = "Tier 2 — MEDIUM RISK"
        tier_color = "#fee090"
    else:
        tier = "Tier 3 — LOW RISK"
        tier_color = "#4dac26"

    df = pd.DataFrame(rows)
    return total, tier, tier_color, df


def print_mrs_report(output_dir: str) -> None:
    """Print MRS report to console and save scorecard CSV."""
    total, tier, tier_color, df = compute_mrs()

    print("\n" + "=" * 65)
    print("  MODEL RISK SCORECARD (Kiritz, Ravitz & Levonian 2019)")
    print("=" * 65)
    for _, row in df.iterrows():
        print(f"\n  {row['Dimension']}  [{row['Score']}/{row['Max Points']}]")
        print(f"    {row['Justification']}")
    print("\n" + "-" * 65)
    print(f"  TOTAL MRS SCORE:  {total} / 100")
    print(f"  RISK TIER:        {tier}")
    print("=" * 65)

    df.to_csv(os.path.join(output_dir, "mrs_scorecard.csv"), index=False)

    # ── Radar / bar chart
    _plot_scorecard(df, total, tier, tier_color, output_dir)


def _plot_scorecard(df: pd.DataFrame, total: int, tier: str,
                    tier_color: str, output_dir: str) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Bar chart
    colors = ["#4dac26" if s/m >= 0.80 else "#fee090" if s/m >= 0.60 else "#d73027"
              for s, m in zip(df["Score"], df["Max Points"])]
    bars = ax1.barh(df["Dimension"], df["Score"], color=colors)
    ax1.set_xlim(0, 35)
    for bar, (_, row) in zip(bars, df.iterrows()):
        ax1.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                 f'{row["Score"]}/{row["Max Points"]}',
                 va="center", fontsize=10, fontweight="bold")
    ax1.set_xlabel("Points Scored")
    ax1.set_title("MRS Scorecard by Dimension\n(Kiritz et al., 2019)")
    ax1.axvline(0, color="black", linewidth=0.5)

    # Risk tier gauge (simple)
    tiers = {"Tier 3\nLow Risk\n(<50)": 50,
             "Tier 2\nMedium Risk\n(50–79)": 30,
             "Tier 1\nHigh Risk\n(≥80)": 20}
    tier_colors = ["#4dac26", "#fee090", "#d73027"]
    ax2.barh(list(tiers.keys()), list(tiers.values()),
             color=tier_colors, edgecolor="white", height=0.5)
    ax2.axvline(total - 50, color="black", linewidth=3,
                label=f"This model: {total} pts")
    ax2.set_xlim(0, 100)
    ax2.set_xlabel("MRS Points")
    ax2.set_title(f"Risk Tier Assignment\nTotal MRS = {total}/100\n{tier}",
                  color=tier_color, fontweight="bold")
    ax2.legend()

    plt.suptitle("LightGBM Credit Scoring Pipeline — Model Risk Assessment",
                 fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "mrs_scorecard_chart.png"),
                dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[MRS] Scorecard chart saved → {output_dir}/mrs_scorecard_chart.png")


# ── 2. SR 11-7 Risk Taxonomy ─────────────────────────────────────────────────

SR117_RISKS = [
    {
        "SR 11-7 Category":    "Model Development Risk",
        "Specific Risk":       "Missing Income Imputation",
        "Evidence":            "~20% of MonthlyIncome missing; XGBRegressor imputation introduces a sub-model with its own error propagation",
        "Severity":            "High",
        "Mitigant":            "Sensitivity analysis comparing median vs. XGB imputation; document imputation model separately as a sub-model per SR 11-7",
    },
    {
        "SR 11-7 Category":    "Model Development Risk",
        "Specific Risk":       "Polynomial Feature Explosion",
        "Evidence":            "10 features → ~74 after degree-2 expansion; risk of spurious correlations and overfitting",
        "Severity":            "Medium",
        "Mitigant":            "Feature selection (SelectPercentile 95%) reduces dimensionality; monitor OSOT performance vs. IS",
    },
    {
        "SR 11-7 Category":    "Data Risk",
        "Specific Risk":       "Class Imbalance",
        "Evidence":            "6.7% positive rate (10,026/150,000); naive accuracy misleading",
        "Severity":            "Medium",
        "Mitigant":            "Use F-beta (β=2), PR-AUC, and KS statistic as primary metrics; LGBM class_weight='balanced'",
    },
    {
        "SR 11-7 Category":    "Data Risk",
        "Specific Risk":       "Vintage / Distribution Shift",
        "Evidence":            "Dataset is 15 years old (2008–2011 financial crisis era); PSI elevated for delinquency features",
        "Severity":            "High",
        "Mitigant":            "PSI monitoring framework; flag features with PSI > 0.20; recommend periodic re-training trigger",
    },
    {
        "SR 11-7 Category":    "Implementation Risk",
        "Specific Risk":       "Pipeline Opacity (WOE + KMeans step)",
        "Evidence":            "K-Means cluster assignments create non-monotonic WOE mapping that cannot be audited as a traditional scorecard",
        "Severity":            "High",
        "Mitigant":            "SHAP TreeExplainer provides post-hoc attribution; document WOE map in model inventory",
    },
    {
        "SR 11-7 Category":    "Implementation Risk",
        "Specific Risk":       "Lack of Override Controls",
        "Evidence":            "Fully automated sklearn pipeline; no human review trigger documented",
        "Severity":            "Medium",
        "Mitigant":            "Define score thresholds for manual review (e.g., borderline decile); document in Model Use Policy",
    },
    {
        "SR 11-7 Category":    "Ongoing Monitoring",
        "Specific Risk":       "No Monitoring Framework",
        "Evidence":            "Kaggle solution has no production monitoring; PSI and performance drift unchecked",
        "Severity":            "High",
        "Mitigant":            "Implement monthly PSI checks; trigger re-validation if PSI > 0.25 or AUC degrades > 2%",
    },
    {
        "SR 11-7 Category":    "Fair Lending / Compliance Risk",
        "Specific Risk":       "Age-Based Disparate Impact",
        "Evidence":            "Age is a direct model input; differential approval rates across age groups possible under ECOA",
        "Severity":            "High",
        "Mitigant":            "Disparate impact testing (4/5 Rule) documented in this report; consider age-blind re-training if DIR < 0.80",
    },
]


def sr117_risk_table(output_dir: str) -> pd.DataFrame:
    """Print and save the SR 11-7 risk taxonomy table."""
    df = pd.DataFrame(SR117_RISKS)

    print("\n[SR 11-7] Risk Taxonomy:")
    print(df[["SR 11-7 Category", "Specific Risk", "Severity"]].to_string(index=False))

    df.to_csv(os.path.join(output_dir, "sr117_risk_taxonomy.csv"), index=False)
    print(f"[SR 11-7] Full table saved → {output_dir}/sr117_risk_taxonomy.csv")

    # Heatmap-style severity chart
    severity_map = {"High": 3, "Medium": 2, "Low": 1}
    df["Severity_num"] = df["Severity"].map(severity_map)

    fig, ax = plt.subplots(figsize=(12, 6))
    colors = {"High": "#d73027", "Medium": "#fee090", "Low": "#4dac26"}
    bar_colors = [colors[s] for s in df["Severity"]]

    ax.barh(df["Specific Risk"], df["Severity_num"], color=bar_colors)
    ax.set_xlabel("Severity (1=Low, 2=Medium, 3=High)")
    ax.set_title("SR 11-7 Risk Taxonomy — Identified Model Risks by Severity")
    ax.set_xlim(0, 4)

    patches = [mpatches.Patch(color=c, label=l) for l, c in colors.items()]
    ax.legend(handles=patches, loc="lower right")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "sr117_risk_chart.png"),
                dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[SR 11-7] Risk chart saved → {output_dir}/sr117_risk_chart.png")

    return df
