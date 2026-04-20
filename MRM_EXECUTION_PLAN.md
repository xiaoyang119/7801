# FRE 7801 — MRM Execution Plan
**Team:** Xiaoyang Zhang (xz5476), Yifan Xu (yx3598), Yike Ma (ym2552)  
**Project:** Model Risk Assessment of ML-Based Credit Scoring: Give Me Some Credit  
**Champion Model:** `pipe1_lgbm_woe_powertransformer95` — Kaggle Private AUC 0.86914 (Top 5)

---

## How This Plan Maps to Course Requirements

| Course Requirement | Source | Execution in This Plan |
|---|---|---|
| 15–20 page validation report | Syllabus §Capstone | 6 report sections below |
| ~10 PPT slides | Syllabus §Capstone | Section VII |
| 5-part validation framework | Lecture 2 | Stages I–V |
| Model risk tiering | Kiritz et al. (2019) | Stage I + MRS Scorecard |
| SR 11-7 taxonomy mapping | SR 11-7 / Lecture 1 | Stage V + risk_tiering.py |
| Benchmark / challenger model | Lecture 2 §III | LR Baseline vs LGBM Champion |
| OSOT back-testing | Lecture 2 §III | Stratified 70/30 split |
| Sensitivity analysis | Lecture 2 §III | sensitivity_analysis() |
| Conceptual soundness | Lecture 2 §II | SHAP global + local |
| PSI / ongoing monitoring | Lecture 2 §IV | psi_report() |
| Fair lending / bias | Proposal §3 | fairness_analysis() |

---

## The Champion Pipeline (What We Are Validating)

```
Raw Data (150K borrowers)
    ↓
XGBRegressor Imputation of MonthlyIncome (~20% missing)
    ↓
PowerTransformer (Yeo-Johnson normalisation)
    ↓
MiniBatchKMeans (11 clusters) → WOE encode cluster labels
    ↓
PolynomialFeatures (degree=2) → 74 features
    ↓
SelectPercentile (f_classif, top 95%) → ~70 features
    ↓
LGBMClassifier (lr=0.0018, n_est=5000, depth=8, subsample=0.857)
    ↓
P(SeriousDlqin2yrs) — probability of 90+ day delinquency in 2 years
```

**Why this is "Tier 1 — High Risk" per Kiritz (2019):**
- MRS score 88/100 → Tier 1 (≥ 80)
- Multi-step black-box pipeline applied to 150K credit decisions
- Black-box LGBM + non-monotonic WOE encoding → opaque to regulators
- Age directly in feature set → ECOA / fair lending exposure

---

## Stage I — Complexity Classification (Lecture 2, Part I)

**Goal:** Assign a risk tier before any quantitative testing.

### What to Write in Report Section 1 (Model Description + Governance)
- Describe the 5-step pipeline in plain English
- Apply the Kiritz MRS scorecard — fill in all 4 dimensions:
  - Materiality (30 pts): 150K borrowers × credit decisions → Score: **27/30**
  - Regulatory Exposure (30 pts): SR 11-7 + ECOA + opaque outputs → **25/30**
  - Operational Factor (10 pts): Fully automated, no override → **8/10**
  - Model Complexity (30 pts): LGBM + polynomial + WOE → **28/30**
  - **Total: 88/100 → Tier 1 (High Risk)**
- State which SR 11-7 elements apply (all 8, with emphasis on §II.B, §III)

### Code
```python
python main.py  # produces outputs/mrs_scorecard_chart.png + mrs_scorecard.csv
```

---

## Stage II — Conceptual Soundness (Lecture 2, Part II)

**Goal:** Confirm the model "makes sense" given credit risk theory.

### What to Write in Report Section 2 (Conceptual Soundness)

**A. Feature-Level Economic Rationale**

| Feature | Expected Direction | SHAP Confirms? |
|---|---|---|
| RevolvingUtilizationOfUnsecuredLines | + (high util → high risk) | Check beeswarm |
| age | − (older → lower risk, generally) | Check beeswarm |
| NumberOfTimes90DaysLate | + (past delinquency → high risk) | Check beeswarm |
| MonthlyIncome | − (higher income → lower risk) | Check beeswarm |
| DebtRatio | + (high DTI → high risk) | Check beeswarm |

**B. Pipeline Design Critique**
- PowerTransformer: justified for skewed financial data (DebtRatio has extreme tail)
- KMeans + WOE: non-standard; creates non-monotonic relationship between features and output
  → **Risk:** WOE encodes cluster labels; cluster membership may shift in production (PSI risk)
- PolynomialFeatures: captures interaction terms (e.g., age × income) but creates ~64 synthetic features
  → **Risk:** Spurious correlations; difficult to explain to examiners
- SelectPercentile: mitigates overfitting from polynomial expansion; defensible

**C. Class Imbalance Treatment**
- 6.7% positive rate — no SMOTE/resampling used
- LGBM `class_weight='balanced'` implicitly weights minority class
- Discuss why PR-AUC is more informative than accuracy here

### Code
```python
python main.py  # produces:
#   outputs/shap_summary_beeswarm.png   ← main conceptual soundness exhibit
#   outputs/shap_summary_bar.png        ← feature importance ranking
#   outputs/shap_waterfall_high_risk.png ← local explanation example
```

---

## Stage III — Quantitative Validation (Lecture 2, Part III)

**Goal:** Empirically test model performance, stability, and robustness.

### What to Write in Report Section 3 (Quantitative Validation)

**A. Performance Metrics Hierarchy**

| Metric | LR Baseline | LGBM Champion | Threshold for "Good" |
|---|---|---|---|
| AUC-ROC | ~0.79–0.82 | **~0.86** | > 0.75 for credit |
| KS Statistic | ~0.35 | **~0.50** | > 0.40 excellent |
| Gini Coefficient | ~0.58 | **~0.72** | > 0.60 good |
| PR-AUC | ~0.30 | **~0.45** | Baseline = 0.067 |
| F-beta (β=2) | — | — | Higher = better |
| Brier Score | — | — | Lower = better |

*(Fill actual numbers after running the code)*

**B. Benchmarking (SR 11-7 Required)**
- Champion vs. Challenger (LR): LGBM should outperform on all metrics
- If LGBM improvement < 3% AUC → question whether added complexity is justified
- Write: "The LGBM champion achieves AUC = X vs. LR baseline = Y, a +Z% improvement,
  justifying the additional model complexity per SR 11-7 cost-benefit principle."

**C. In-Sample vs. Out-of-Sample (OSOT) Back-Test**
- IS performance: metrics on X_train
- OSOT performance: metrics on X_val (30% hold-out)
- If IS-OSOT gap > 3% AUC → overfitting flag
- 5,000 estimators in LGBM may overfit — check this

**D. Sensitivity Analysis**
- Perturb each input feature ±10%
- Rank features by |ΔAUC|
- Flag high-sensitivity features as "model risk drivers"
- Expected: RevolvingUtilization and 90DaysLate should be most sensitive

**E. PSI (Population Stability)**
- PSI between training and validation feature distributions
- Flag any feature with PSI > 0.20 as a data stability risk

### Code
```python
python main.py  # produces:
#   outputs/benchmark_table.csv           ← Table 1 in your report
#   outputs/roc_pr_curves.png             ← Figure 1
#   outputs/ks_plot_LGBM_Champion.png     ← Figure 2
#   outputs/ks_plot_LR_Baseline.png
#   outputs/calibration_curve.png         ← Figure 3
#   outputs/confusion_matrices.png        ← Figure 4
#   outputs/sensitivity_analysis.csv      ← Appendix A
#   outputs/psi_table.csv                 ← Appendix B
#   outputs/psi_chart.png
```

---

## Stage IV — Ongoing Governance (Lecture 2, Part IV)

**Goal:** Propose a monitoring framework for production use.

### What to Write in Report Section 4 (Ongoing Monitoring)

Per SR 11-7: "Banks should have a process for ongoing monitoring of model performance."

**Proposed Monitoring Framework:**

| Metric | Frequency | Trigger Threshold | Action |
|---|---|---|---|
| PSI (all features) | Monthly | PSI > 0.20 on any feature | Alert + root cause analysis |
| PSI (all features) | Monthly | PSI > 0.25 | Immediate re-validation |
| AUC-ROC (labeled sample) | Quarterly | AUC degrades > 2% | Re-validation required |
| Default rate vs. predicted | Monthly | > 10% deviation | Model recalibration |
| Disparate Impact Ratio | Semi-annual | DIR < 0.80 for any group | Fair lending review |
| KMeans cluster stability | Monthly | Any cluster empties or doubles | Pipeline re-training |

**Key Model Risk from WOE Step:**
- If borrower population shifts (e.g., post-recession), KMeans clusters may reassign in non-monotonic ways
- PSI on cluster membership (not just features) should be tracked separately

---

## Stage V — Documentation & SR 11-7 Risk Taxonomy (Lecture 2, Part V)

**Goal:** Map every identified risk to the SR 11-7 framework with mitigants.

### What to Write in Report Section 5 (Limitations, Risk Ratings, Mitigants)

Use the table from `outputs/sr117_risk_taxonomy.csv`. Key risks:

| SR 11-7 Category | Specific Risk | Severity | Mitigant |
|---|---|---|---|
| Model Development | XGBRegressor imputation sub-model | High | Document as separate model; sensitivity test |
| Model Development | Polynomial feature explosion (74 features) | Medium | SelectPercentile; monitor OSOT gap |
| Data Risk | 20% income missing → imputation bias | High | Missingness indicator feature; impute with XGB |
| Data Risk | Class imbalance (6.7% default rate) | Medium | class_weight='balanced'; use PR-AUC |
| Data Risk | Vintage risk (2008 crisis data, 15 years old) | High | PSI monitoring; flag for re-training |
| Implementation | LGBM opacity (black box) | High | SHAP TreeExplainer; document globally + locally |
| Implementation | WOE non-monotonicity | High | Document WOE map in model inventory |
| Implementation | No override controls | Medium | Define manual review thresholds |
| Ongoing Monitoring | No monitoring framework | High | Implement PSI + AUC monthly checks |
| Fair Lending | Age-based disparate impact | High | DIR testing; consider age-blind re-training |

### Final Risk Rating
> **Overall Model Risk: HIGH (Tier 1)**  
> MRS = 88/100. This rating requires Tier 1 validation rigor: independent validation, senior MRM officer approval, quarterly re-validation, and documented monitoring framework before production deployment.

---

## Report Structure (15–20 pages)

| Section | Title | Pages | Key Exhibits |
|---|---|---|---|
| 1 | Executive Summary + Model Description | 2 | Pipeline diagram, MRS scorecard |
| 2 | Data Quality & Preparation | 2–3 | EDA table, PSI chart, missing value analysis |
| 3 | Conceptual Soundness | 3–4 | SHAP beeswarm, WOE critique, feature rationale table |
| 4 | Quantitative Validation | 4–5 | Benchmark table, ROC/PR curves, KS plot, sensitivity |
| 5 | Fairness & Demographic Analysis | 2 | Disparate impact table, age-group AUC chart |
| 6 | Risk Tiering & Recommendations | 2–3 | MRS scorecard, SR 11-7 risk table, monitoring framework |

---

## PPT Slide Plan (~10 slides)

1. **Title Slide** — Model name, team, Kaggle score
2. **Motivation** — Why MRM for ML credit scoring matters (SR 11-7)
3. **The Champion Pipeline** — 5-step flow diagram
4. **Data Overview** — Key stats, missing values, class imbalance
5. **Model Risk Tiering** — MRS scorecard bar chart (Tier 1)
6. **Performance Results** — Benchmark table + ROC curves
7. **Conceptual Soundness** — SHAP beeswarm plot
8. **Stability & PSI** — PSI bar chart, flagged features
9. **Fairness Testing** — Age-group disparate impact chart
10. **Recommendations** — SR 11-7 risk taxonomy (top 5) + monitoring table

---

## How to Run the Pipeline

```bash
# Step 1: Install dependencies (in the 7801/ folder)
uv sync   # or: pip install lightgbm xgboost shap joblib

# Step 2: Run with fast median imputation (recommended first run ~5–10 min)
python main.py --impute median

# Step 3: Run with XGB imputation to replicate Kaggle pipeline exactly (~30 min)
python main.py --impute xgb

# Optional: Skip SHAP if not installed
python main.py --skip-shap --skip-sensitivity

# All outputs go to:  7801/outputs/
```

**Output files you'll use in the report:**

| File | Used In |
|---|---|
| `benchmark_table.csv` | Section 4: Performance table |
| `roc_pr_curves.png` | Section 4: Figure 1 |
| `ks_plot_LGBM_Champion.png` | Section 4: Figure 2 |
| `shap_summary_beeswarm.png` | Section 3: Figure 3 |
| `shap_waterfall_high_risk.png` | Section 3: Figure 4 |
| `psi_chart.png` | Section 2 + Section 6 |
| `fairness_age_groups.png` | Section 5: Figure 5 |
| `mrs_scorecard_chart.png` | Section 6: Figure 6 |
| `sr117_risk_taxonomy.csv` | Section 6: Table 2 |
| `executive_summary.txt` | Section 1 |
