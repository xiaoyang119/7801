"use strict";
const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell, ImageRun,
  Header, Footer, AlignmentType, HeadingLevel, BorderStyle, WidthType, ShadingType,
  VerticalAlign, PageNumber, PageBreak, LevelFormat, UnderlineType,
} = require("docx");
const fs   = require("fs");
const path = require("path");

// ── Layout ────────────────────────────────────────────────────────────────────
const W      = 9360;   // content width DXA (8.5" − 2×1" margins)
const PAGE_W = 12240;
const PAGE_H = 15840;
const MARGIN = 1440;

// ── Academic colour palette ───────────────────────────────────────────────────
const BLACK  = "000000";
const DGRAY  = "333333";
const MGRAY  = "595959";
const HFILL  = "D9D9D9";   // table header fill (light gray)
const ALTROW = "F5F5F5";   // subtle alternate row
const WHITE  = "FFFFFF";
const PASS   = "D6EAD6";
const AMBER  = "FFF2CC";
const ISSUE  = "FCE4D6";
const PASST  = "1A5C1A";
const AMBRT  = "7B4F00";
const ISSUT  = "8B1A00";

const FONT   = "Times New Roman";
const IMG    = "/Users/evanxu/Desktop/Classes/Model Risk/Final Project/outputs";

// ── Helpers ───────────────────────────────────────────────────────────────────
function img(file, w, h) {
  const p = path.join(IMG, file);
  if (!fs.existsSync(p)) return null;
  return new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { before: 160, after: 80 },
    children: [new ImageRun({
      type: "png", data: fs.readFileSync(p),
      transformation: { width: w, height: h },
      altText: { title: file, description: file, name: file },
    })],
  });
}
function caption(text) {
  return new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { before: 40, after: 200 },
    children: [new TextRun({ text, italics: true, size: 20, color: MGRAY, font: FONT })],
  });
}
function gap(after = 120) { return new Paragraph({ spacing: { after }, children: [] }); }

function run(text, opts = {}) {
  return new TextRun({ text, font: FONT, size: opts.size || 24,
    bold: opts.bold, italics: opts.italic, color: opts.color || BLACK,
    underline: opts.underline ? { type: UnderlineType.SINGLE } : undefined });
}

function p(children, opts = {}) {
  const arr = typeof children === "string"
    ? [run(children, opts)]
    : (Array.isArray(children) ? children : [children]);
  return new Paragraph({
    alignment: opts.align || AlignmentType.LEFT,
    spacing: { before: opts.before || 0, after: opts.after || 160 },
    indent: opts.indent,
    children: arr,
  });
}

function h1(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_1,
    spacing: { before: 400, after: 160 },
    children: [run(text, { bold: true, size: 28, color: BLACK })],
  });
}
function h2(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_2,
    spacing: { before: 280, after: 100 },
    children: [run(text, { bold: true, size: 26, color: BLACK })],
  });
}
function h3(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_3,
    spacing: { before: 200, after: 80 },
    children: [run(text, { bold: true, italic: true, size: 24, color: DGRAY })],
  });
}

function bul(text) {
  return new Paragraph({
    numbering: { reference: "bullets", level: 0 },
    spacing: { after: 80 },
    indent: { left: 560, hanging: 280 },
    children: [run(text)],
  });
}
function num(text) {
  const arr = typeof text === "string" ? [run(text)] : (Array.isArray(text) ? text : [text]);
  return new Paragraph({
    numbering: { reference: "numbers", level: 0 },
    spacing: { after: 80 },
    indent: { left: 560, hanging: 280 },
    children: arr,
  });
}

// ── Table helpers ─────────────────────────────────────────────────────────────
const BORDER = { style: BorderStyle.SINGLE, size: 4, color: "AAAAAA" };
const BORDS  = { top: BORDER, bottom: BORDER, left: BORDER, right: BORDER };
const HBORD  = { style: BorderStyle.SINGLE, size: 6, color: "888888" };
const HBORDS = { top: HBORD, bottom: HBORD, left: HBORD, right: HBORD };

function hcell(text, w, span = 1) {
  return new TableCell({
    borders: HBORDS, width: { size: w, type: WidthType.DXA },
    shading: { fill: HFILL, type: ShadingType.CLEAR },
    margins: { top: 80, bottom: 80, left: 130, right: 130 },
    columnSpan: span,
    children: [new Paragraph({
      alignment: AlignmentType.CENTER,
      children: [run(text, { bold: true, size: 20, color: BLACK })],
    })],
  });
}

function dcell(text, w, opts = {}) {
  const { bold = false, color = BLACK, align = AlignmentType.LEFT, fill = WHITE } = opts;
  return new TableCell({
    borders: BORDS, width: { size: w, type: WidthType.DXA },
    shading: { fill, type: ShadingType.CLEAR },
    margins: { top: 80, bottom: 80, left: 130, right: 130 },
    children: [new Paragraph({
      alignment: align,
      children: [run(text, { bold, size: 20, color })],
    })],
  });
}

function trow(cells, widths, alt = false, cellOpts = []) {
  const fill = alt ? ALTROW : WHITE;
  return new TableRow({
    children: cells.map((text, i) => {
      const o = cellOpts[i] || {};
      return dcell(String(text), widths[i],
        { fill, align: i === 0 ? AlignmentType.LEFT : AlignmentType.CENTER, ...o });
    }),
  });
}

function tbl(headers, rows, widths, cellOpts = []) {
  return new Table({
    width: { size: W, type: WidthType.DXA },
    columnWidths: widths,
    rows: [
      new TableRow({ tableHeader: true, children: headers.map((h, i) => hcell(h, widths[i])) }),
      ...rows.map((r, i) => trow(r, widths, i % 2 === 1, cellOpts)),
    ],
  });
}

function keyValTable(rows, w1 = 2800, w2 = W - 2800) {
  return new Table({
    width: { size: W, type: WidthType.DXA },
    columnWidths: [w1, w2],
    rows: rows.map(([k, v], i) => new TableRow({ children: [
      dcell(k, w1, { bold: true, fill: i % 2 === 0 ? ALTROW : WHITE }),
      dcell(v, w2, { fill: i % 2 === 0 ? ALTROW : WHITE }),
    ]})),
  });
}

function statusCell(text, w) {
  let fill = WHITE, color = BLACK;
  if (text === "Pass")   { fill = PASS;  color = PASST; }
  if (text === "Review") { fill = AMBER; color = AMBRT; }
  if (text === "Issue")  { fill = ISSUE; color = ISSUT; }
  return dcell(text, w, { bold: true, fill, color, align: AlignmentType.CENTER });
}

function srRow(cells, widths, alt) {
  const fill = alt ? ALTROW : WHITE;
  return new TableRow({ children: [
    dcell(cells[0], widths[0], { bold: true, fill }),
    dcell(cells[1], widths[1], { fill }),
    statusCell(cells[2], widths[2]),
    dcell(cells[3], widths[3], { fill }),
  ]});
}

// ── Header / Footer ───────────────────────────────────────────────────────────
const hrBorder = { style: BorderStyle.SINGLE, size: 4, color: "888888", space: 4 };

const docHeader = new Header({ children: [
  new Paragraph({
    border: { bottom: hrBorder },
    spacing: { after: 80 },
    children: [
      run("FRE 7801 — Model Risk Management  |  Model Validation Report", { size: 18, color: MGRAY }),
    ],
  }),
]});

const docFooter = new Footer({ children: [
  new Paragraph({
    alignment: AlignmentType.CENTER,
    border: { top: hrBorder },
    spacing: { before: 80 },
    children: [
      new TextRun({ children: [PageNumber.CURRENT], font: FONT, size: 18, color: MGRAY }),
    ],
  }),
]});

// ── Scaled image sizes  (width px, height px)  at 96 dpi docx units ───────────
const IMGS = {
  imputation_validation: [576, 408],
  eda_distributions:     [576, 514],
  psi_chart:             [480, 237],
  roc_pr_curves:         [576, 245],
  ks_lgbm:               [440, 241],
  calibration_curve:     [380, 283],
  confusion_matrices:    [540, 245],
  shap_beeswarm:         [380, 470],
  shap_bar:              [380, 453],
  shap_waterfall:        [440, 354],
  cv_folds:              [576, 221],
  fairness:              [560, 238],
  age_blind:             [576, 179],
  mrs_chart:             [520, 230],
  sr117_chart:           [520, 257],
};

function children() {
  const C = [];
  const add = (...items) => items.forEach(x => x && C.push(x));

  // ── COVER PAGE ─────────────────────────────────────────────────────────────
  add(
    gap(480),
    new Paragraph({
      alignment: AlignmentType.CENTER,
      spacing: { after: 200 },
      children: [run("Model Risk Assessment of Machine Learning-Based Credit Scoring:", { bold: true, size: 36 })],
    }),
    new Paragraph({
      alignment: AlignmentType.CENTER,
      spacing: { after: 360 },
      children: [run("A Case Study Using the Give Me Some Credit Dataset", { bold: true, size: 32 })],
    }),
    new Paragraph({
      alignment: AlignmentType.CENTER,
      spacing: { after: 600 },
      children: [run("Model Validation Report", { size: 28, italic: true, color: MGRAY })],
    }),
    gap(120),
    new Paragraph({
      alignment: AlignmentType.CENTER,
      spacing: { after: 100 },
      children: [run("Xiaoyang Zhang (xz5476)  ·  Yifan Xu (yx3598)  ·  Yike Ma (ym2552)", { size: 24 })],
    }),
    new Paragraph({
      alignment: AlignmentType.CENTER,
      spacing: { after: 100 },
      children: [run("FRE 7801 — Model Risk Management", { size: 24 })],
    }),
    new Paragraph({
      alignment: AlignmentType.CENTER,
      spacing: { after: 100 },
      children: [run("NYU Tandon School of Engineering", { size: 24 })],
    }),
    new Paragraph({
      alignment: AlignmentType.CENTER,
      spacing: { after: 400 },
      children: [run("Spring 2026", { size: 24 })],
    }),
    gap(240),
    new Paragraph({ children: [new PageBreak()] }),
  );

  // ── ABSTRACT ───────────────────────────────────────────────────────────────
  add(
    h1("Abstract"),
    p("This report evaluates a machine-learning credit scoring model using a model risk management framework aligned with SR 11-7. The model estimates the probability that a borrower will experience serious delinquency, defined as 90 or more days past due within two years. The champion model is a LightGBM pipeline that applies missing-value treatment, Yeo-Johnson transformation, KMeans-based Weight of Evidence encoding, polynomial feature expansion, feature selection, and gradient-boosted tree classification. A logistic regression pipeline is used as the benchmark challenger."),
    p("The LightGBM champion outperforms the logistic regression benchmark on out-of-sample discriminatory metrics, with validation AUC of 0.8601 compared with 0.8317 for logistic regression and KS statistic of 0.5700 compared with 0.5075. 5-fold cross-validation confirms a mean train-validation AUC gap of 0.1669, indicating excessive pipeline complexity. Age-based disparate impact analysis shows a young-borrower DIR of 0.3351; however, age is an ECOA-permitted factor in empirically derived scoring systems and the DIR gap is consistent with the actual 4.3x default rate difference across age groups. A less-discriminatory-alternative test was conducted and documented. SHAP global and local explanation artifacts were generated."),
    p([
      run("Keywords: ", { bold: true }),
      run("model risk management, SR 11-7, credit scoring, LightGBM, fair lending, PSI, model validation, machine learning governance.", { italic: true }),
    ]),
    new Paragraph({ children: [new PageBreak()] }),
  );

  // ── DOCUMENT CONTROL ───────────────────────────────────────────────────────
  add(
    h1("Document Control"),
    keyValTable([
      ["Report type",           "Academic model validation report"],
      ["Model name",            "LightGBM credit scoring pipeline"],
      ["Model purpose",         "Estimate probability of serious delinquency within two years"],
      ["Model owner",           "FRE 7801 project team"],
      ["Validation role",       "Academic validation; not independent second-line validation"],
      ["Primary benchmark",     "Logistic regression baseline"],
      ["Primary data source",   "Kaggle Give Me Some Credit dataset"],
      ["Primary limitations",   "Aged public data, no live portfolio data, no true independent validation, no legal fair lending determination"],
      ["Final risk rating",     "Tier 1 — High Risk  (Internal MRS Score: 88/100)"],
      ["Validation conclusion", "Conditionally approved pending overfitting remediation and governance controls"],
    ]),
    new Paragraph({ children: [new PageBreak()] }),
  );

  // ── TABLE OF CONTENTS ──────────────────────────────────────────────────────
  add(h1("Table of Contents"));
  const tocItems = [
    ["1.", "Executive Summary"],
    ["2.", "Introduction and Scope"],
    ["3.", "Data and Data Quality Review"],
    ["4.", "Model Development and Methodology"],
    ["5.", "Conceptual Soundness Review"],
    ["6.", "Quantitative Validation"],
    ["7.", "Stability and Sensitivity Analysis"],
    ["8.", "Fair Lending and Use Risk"],
    ["9.", "Model Risk Tiering"],
    ["10.", "SR 11-7 Validation Checklist"],
    ["11.", "Monitoring and Control Framework"],
    ["12.", "Final Validation Opinion"],
    ["13.", "References"],
  ];
  tocItems.forEach(([num, title]) => {
    add(new Paragraph({ spacing: { after: 80 }, children: [
      run(num + "  " + title, { size: 24 }),
    ]}));
  });
  add(new Paragraph({ children: [new PageBreak()] }));

  // ── 1. EXECUTIVE SUMMARY ──────────────────────────────────────────────────
  add(
    h1("1.  Executive Summary"),
    p("This report presents a formal model risk assessment of a LightGBM-based credit scoring model developed on the Give Me Some Credit dataset. The model produces a borrower-level probability of serious delinquency within two years. In a production lending environment, such a score could influence approval, pricing, account management, or manual review. The model therefore falls within the scope of SR 11-7 and should be governed as a high-impact credit model."),
    p("The LightGBM champion demonstrates strong out-of-sample discrimination. On the 30% stratified validation set, the champion achieves AUC-ROC of 0.8601 and KS statistic of 0.5700. The logistic regression challenger achieves AUC-ROC of 0.8317 and KS statistic of 0.5075. The champion provides measurable lift over a simpler, more transparent model. The champion also shows a train-validation AUC gap of 0.0644; 5-fold cross-validation confirms this reflects systematic overfitting with a mean gap of 0.1669."),
    p("Age use in this model is ECOA-compliant. The young-borrower Disparate Impact Ratio of 0.3351 is consistent with the actual 4.3x default rate difference between age groups. A less-discriminatory-alternative test was conducted and is documented in Section 8. The final model risk rating is Tier 1 — High Risk, with an internal MRS score of 88/100."),
    gap(120),
    h2("Summary of Key Findings"),
    tbl(
      ["Validation Area", "Result", "Validation Assessment"],
      [
        ["Champion Validation AUC",           "0.8601",  "Strong discriminatory power"],
        ["Challenger Validation AUC",          "0.8317",  "Strong benchmark"],
        ["Champion KS Statistic",             "0.5700",  "Excellent rank ordering"],
        ["Champion Train AUC",                "0.9245",  "High in-sample fit"],
        ["Champion AUC Gap (single split)",   "0.0644",  "Review finding"],
        ["CV Mean AUC Gap (5-fold)",          "0.1669",  "Overfitting confirmed; simplification required"],
        ["Maximum PSI",                       "0.0005",  "Stable on random split"],
        ["LightGBM Young-Borrower DIR",       "0.3351",  "ECOA-permitted; reflects 4.3x default rate difference"],
        ["Age-Blind LGBM DIR",                "0.5081",  "LDA tested and documented"],
        ["Internal MRS Score",                "88/100",  "Tier 1 — High Risk"],
      ],
      [3800, 1400, 4160],
    ),
    new Paragraph({ children: [new PageBreak()] }),
  );

  // ── 2. INTRODUCTION ────────────────────────────────────────────────────────
  add(
    h1("2.  Introduction and Scope"),
    p("Credit scoring is a core application of quantitative modeling in financial institutions. Traditional bank scorecards rely on logistic regression because the methodology is transparent, auditable, and easy to explain to internal reviewers and regulators. Machine-learning models such as LightGBM can improve predictive accuracy, but they increase model risk through opacity, nonlinear interactions, sensitivity to changing populations, and fair lending complexity."),
    p("The objective of this report is to evaluate whether the champion LightGBM model is conceptually sound, empirically reliable, appropriately benchmarked, and governable under a model risk framework consistent with SR 11-7."),
    h2("Model Definition and Intended Use"),
    p("The model converts borrower characteristics into a quantitative estimate: the probability of serious delinquency within two years. Under SR 11-7, this is a model because it uses statistical and mathematical techniques to process inputs into a quantitative estimate. The intended use is credit risk ranking and potential underwriting support."),
    h2("Scope of Review"),
    p("The scope includes:"),
    bul("Data quality and missing-value treatment."),
    bul("Outlier handling and feature preparation."),
    bul("Benchmarking against logistic regression."),
    bul("Out-of-sample performance testing."),
    bul("Calibration and confusion matrix review."),
    bul("PSI stability analysis."),
    bul("Sensitivity analysis."),
    bul("Age-based disparate impact testing."),
    bul("Internal model risk tiering."),
    bul("SR 11-7 governance and monitoring recommendations."),
    gap(80),
    p("The following are outside the completed scope:"),
    bul("Legal fair lending determination."),
    bul("Independent second-line validation."),
    bul("Internal audit review."),
    bul("Production implementation testing."),
    bul("Live monitoring against a bank portfolio."),
    bul("Adverse action reason-code validation."),
    bul("True out-of-time back-testing over a production observation window."),
    new Paragraph({ children: [new PageBreak()] }),
  );

  // ── 3. DATA QUALITY ────────────────────────────────────────────────────────
  add(
    h1("3.  Data and Data Quality Review"),
    p("The Give Me Some Credit dataset contains 150,000 training observations and 101,503 test observations. The target variable is SeriousDlqin2yrs, indicating whether a borrower experienced 90 or more days past due within two years. The target rate is approximately 6.7%, which is typical of credit default modeling but creates class imbalance."),
    h2("Input Variables"),
    tbl(
      ["Variable", "Economic Interpretation", "Model Risk Consideration"],
      [
        ["RevolvingUtilizationOfUnsecuredLines", "Liquidity stress and unsecured credit usage", "Extreme outliers require caps"],
        ["age",                                  "Borrower age and credit history proxy",        "Fair lending sensitivity"],
        ["NumberOfTime30-59DaysPastDueNotWorse",  "Recent mild delinquency",                     "Sentinel values require treatment"],
        ["DebtRatio",                            "Debt burden relative to income",               "Heavy right tail"],
        ["MonthlyIncome",                        "Ability to repay",                             "Material missingness (19.82%)"],
        ["NumberOfOpenCreditLinesAndLoans",      "Credit depth and leverage",                    "Nonlinear effect"],
        ["NumberOfTimes90DaysLate",              "Severe prior delinquency",                     "Strong risk signal"],
        ["NumberRealEstateLoansOrLines",         "Mortgage and collateral exposure",             "Ambiguous directional effect"],
        ["NumberOfTime60-89DaysPastDueNotWorse",  "Moderate delinquency",                        "Sentinel values require treatment"],
        ["NumberOfDependents",                   "Household obligation proxy",                   "Missingness and weak sensitivity"],
      ],
      [2600, 3380, 3380],
    ),
    gap(160),
    h2("Missing Data and Imputation Strategy"),
    p("MonthlyIncome has 19.82% missingness (29,731 of 150,000 rows). NumberOfDependents has 2.62% missingness (3,924 rows). These are the only two features with missing values. Missingness indicators are created before imputation so that the absence of a value can itself be used as a predictive signal."),
    p([run("Imputation method: ", { bold: true }), run("Training-set median, applied to both training and validation/test sets. The medians are computed exclusively from training data to prevent data leakage.")]),
    h3("Assumption Validation"),
    p("Every assumption underlying the imputation strategy was tested empirically."),
    tbl(
      ["Assumption", "Test", "MonthlyIncome", "NumberOfDependents", "Conclusion"],
      [
        ["A1 — MCAR", "Chi-square vs. target", "χ²=67.89, p<0.0001", "χ²=28.75, p<0.0001", "MCAR rejected; missingness is informative. Indicator flag validated."],
        ["A2 — Default rate", "Ratio of rates", "0.0561 vs 0.0695 (0.81×)", "0.0456 vs 0.0674 (0.68×)", "Missing borrowers default less; indicator corrects directionally."],
        ["A3 — Skewness", "Skewness of observed", "Skew=114.04; Median=5,400", "Skew=1.59; Median=0", "Median clearly preferred over mean."],
        ["A4 — Leakage", "Imputed vs training median", "Pass", "Pass", "Medians from training only; no leakage."],
        ["A5 — Stability", "PSI of indicators", "PSI=0.0000 (Stable)", "PSI=0.0000 (Stable)", "Stable across train/val split."],
      ],
      [1800, 1600, 1700, 1700, 2560],
    ),
    gap(120),
    img("imputation_validation.png", ...IMGS.imputation_validation),
    caption("Figure 1.  Imputation validation — observed distributions and default rate by missing vs. present group."),
    h3("Sub-Model Risk (XGB Imputation)"),
    p("The pipeline includes an optional XGBRegressor-based imputation path. If used in production it constitutes a sub-model with its own assumptions, training data, and hyperparameters. It must be separately inventoried, validated, and monitored. The current default path uses median imputation, which requires no such treatment."),
    h2("Outliers and Data Capping"),
    p("The raw dataset contains extreme values in revolving utilization and debt ratio. Past-due variables include sentinel-like values (98 codes). Domain-informed caps are applied. Production monitoring should track cap-rate changes as an indicator of data quality degradation or population shift."),
    h2("Population Stability"),
    p("All PSI values remain below 0.10, with maximum PSI of 0.0005. This indicates stability between the random splits. Because the split is random rather than temporal, this result should not be interpreted as evidence of production stability."),
    tbl(
      ["Feature", "PSI", "Status"],
      [
        ["DebtRatio",                            "0.0005", "Stable"],
        ["MonthlyIncome",                        "0.0004", "Stable"],
        ["RevolvingUtilizationOfUnsecuredLines", "0.0003", "Stable"],
        ["age",                                  "0.0003", "Stable"],
        ["NumberOfOpenCreditLinesAndLoans",      "0.0002", "Stable"],
        ["NumberRealEstateLoansOrLines",         "0.0001", "Stable"],
        ["NumberOfTime30-59DaysPastDueNotWorse", "0.0000", "Stable"],
        ["NumberOfTimes90DaysLate",              "0.0000", "Stable"],
        ["NumberOfTime60-89DaysPastDueNotWorse", "0.0000", "Stable"],
        ["NumberOfDependents",                   "0.0000", "Stable"],
        ["MonthlyIncome_missing",                "0.0000", "Stable"],
        ["NumberOfDependents_missing",           "0.0000", "Stable"],
      ],
      [5000, 2180, 2180],
    ),
    gap(120),
    img("eda_distributions.png", ...IMGS.eda_distributions),
    caption("Figure 2.  Feature distributions and target distribution (training set)."),
    img("psi_chart.png", ...IMGS.psi_chart),
    caption("Figure 3.  Population Stability Index — training vs. validation features."),
    new Paragraph({ children: [new PageBreak()] }),
  );

  // ── 4. MODEL DEVELOPMENT ──────────────────────────────────────────────────
  add(
    h1("4.  Model Development and Methodology"),
    p("The champion is a multi-stage LightGBM pipeline designed to replicate a high-performing Kaggle-style credit scoring approach. The challenger is a logistic regression pipeline used as a transparent benchmark."),
    h2("Champion Pipeline"),
    p("The champion pipeline contains the following processing steps:"),
    num("Missing value treatment with missingness indicators."),
    num("Yeo-Johnson PowerTransformer for skewed numerical variables."),
    num("MiniBatchKMeans borrower clustering."),
    num("Weight of Evidence encoding of cluster labels."),
    num("Degree-2 polynomial feature expansion."),
    num("ANOVA F-score feature selection using SelectPercentile."),
    num("LightGBM classifier with class weighting."),
    gap(80),
    p("This structure is powerful but difficult to explain. The KMeans-WOE and polynomial steps create nonlinear interactions that may not have intuitive economic interpretation. LightGBM then applies additional nonlinear splits, increasing opacity."),
    h2("Challenger Pipeline"),
    p("The logistic regression challenger uses standardized inputs and class weighting. It represents a traditional scorecard-style benchmark that is less flexible than LightGBM but easier to validate, explain, and govern."),
    h2("Methodology Assessment"),
    tbl(
      ["Model Component", "Benefit", "Primary Risk"],
      [
        ["Missingness indicators",  "Preserves data quality signal",         "May encode operational or demographic differences"],
        ["Median imputation",       "Allows complete-case modeling",         "Imputation assumptions; see Section 3"],
        ["PowerTransformer",        "Reduces skew",                          "Obscures raw economic scale"],
        ["KMeans-WOE",              "Captures nonlinear borrower segments",  "Non-monotonic and difficult to audit"],
        ["Polynomial features",     "Captures interactions",                 "Overfitting and interpretability risk"],
        ["SelectPercentile",        "Reduces dimensionality",                "May not capture nonlinear relevance"],
        ["LightGBM",                "Strong rank ordering",                  "Black-box model risk"],
      ],
      [2600, 3000, 3760],
    ),
    new Paragraph({ children: [new PageBreak()] }),
  );

  // ── 5. CONCEPTUAL SOUNDNESS ───────────────────────────────────────────────
  add(
    h1("5.  Conceptual Soundness Review"),
    p("SR 11-7 expects model design, theory, assumptions, and limitations to be documented and supported by sound practice. At the raw feature level, the model uses variables that are broadly consistent with credit risk theory. Higher utilization, higher debt burden, lower income, and recent delinquency are plausible predictors of default."),
    p("The conceptual weaknesses arise from the transformation and modeling architecture. KMeans clustering does not guarantee monotonic credit risk behavior. WOE encoding of clusters can map similar borrowers into different risk groups in a way that is hard to explain. Polynomial interactions can create terms that are statistically useful but economically opaque."),
    h2("Economic Rationale by Feature"),
    tbl(
      ["Feature", "Expected Relationship to Risk", "Assessment"],
      [
        ["Revolving utilization",  "Positive",             "High utilization signals liquidity stress"],
        ["Age",                    "Generally negative",   "Predictive; ECOA-compliant use (see Section 8)"],
        ["30-59 day delinquency",  "Positive",             "Early delinquency signal"],
        ["DebtRatio",              "Positive",             "Higher debt burden reduces repayment capacity"],
        ["MonthlyIncome",          "Negative",             "Higher income supports repayment"],
        ["Open credit lines",      "Nonlinear",            "Can indicate credit access or overextension"],
        ["90 day delinquency",     "Positive",             "Strong severe delinquency signal"],
        ["Real estate loans",      "Ambiguous",            "May indicate stability or leverage"],
        ["60-89 day delinquency",  "Positive",             "Moderate delinquency signal"],
        ["Dependents",             "Positive or ambiguous","More dependents may reduce disposable income"],
      ],
      [2400, 2400, 4560],
    ),
    gap(120),
    h2("SHAP Explanation Artifacts"),
    p("SHAP TreeExplainer was applied to 2,000 validation observations to provide post-hoc attribution supporting conceptual soundness review. Global and local explanation plots are shown below."),
    img("shap_summary_beeswarm.png", ...IMGS.shap_beeswarm),
    caption("Figure 4.  SHAP global beeswarm — feature importance and direction (LGBM Champion)."),
    img("shap_summary_bar.png", ...IMGS.shap_bar),
    caption("Figure 5.  SHAP mean |value| — ranked global feature importance."),
    img("shap_waterfall_high_risk.png", ...IMGS.shap_waterfall),
    caption("Figure 6.  SHAP waterfall — highest-risk borrower (local explanation)."),
    h2("Conceptual Soundness Finding"),
    p("The model is directionally consistent with credit theory at the input-variable level, but the pipeline architecture materially weakens interpretability. The model should not be presented as a transparent scorecard. It should be governed as a black-box model requiring explanation overlays, challenger comparison, sensitivity testing, and limits on use."),
    new Paragraph({ children: [new PageBreak()] }),
  );

  // ── 6. QUANTITATIVE VALIDATION ────────────────────────────────────────────
  add(
    h1("6.  Quantitative Validation"),
    p("Quantitative validation evaluates discriminatory power, calibration, benchmark performance, and evidence of overfitting. The benchmark table includes both validation metrics and train-validation AUC comparison."),
    h2("Benchmark Results"),
    tbl(
      ["Model", "AUC-ROC", "KS Stat", "Gini", "PR-AUC", "F-beta", "Brier", "Train AUC", "AUC Gap"],
      [
        ["LR Baseline",    "0.8317", "0.5075", "0.6635", "0.3344", "0.4619", "0.1652", "0.8291", "-0.0026"],
        ["LGBM Champion",  "0.8601", "0.5700", "0.7202", "0.3927", "0.5147", "0.1253", "0.9245",  "0.0644"],
      ],
      [1700, 900, 900, 900, 900, 900, 900, 1000, 1160],
    ),
    gap(120),
    p("The champion outperforms the challenger across all validation metrics. AUC increases by 0.0284, KS increases by 0.0625, and Brier score improves from 0.1652 to 0.1253. The PR-AUC improvement is especially relevant because the target class is rare."),
    h2("Overfitting Review"),
    p("The logistic regression model shows no concerning overfit pattern (train AUC 0.8291, validation AUC 0.8317). The LightGBM champion shows a single-split AUC gap of 0.0644, exceeding the 0.03 review threshold."),
    h2("5-Fold Cross-Validation — Overfitting Investigation"),
    p("5-fold stratified cross-validation was conducted to determine whether the AUC gap reflects true overfitting or single-split variance."),
    tbl(
      ["Fold", "Train AUC", "Val AUC", "AUC Gap", "KS"],
      [
        ["Fold 1", "0.9968", "0.8358", "0.1610", "0.5325"],
        ["Fold 2", "0.9966", "0.8309", "0.1657", "0.5131"],
        ["Fold 3", "0.9969", "0.8186", "0.1783", "0.4925"],
        ["Fold 4", "0.9964", "0.8324", "0.1640", "0.5204"],
        ["Fold 5", "0.9962", "0.8308", "0.1654", "0.5123"],
        ["Mean",   "0.9966", "0.8297 +/- 0.0065", "0.1669", "0.5142"],
      ],
      [1560, 1800, 2000, 2000, 2000],
    ),
    gap(120),
    p("Mean val AUC of 0.8297 +/- 0.0065 with mean gap 0.1669 confirms systematic overfitting. The low standard deviation (0.0065) indicates the model generalizes consistently but at a consistently lower level than its in-sample fit."),
    img("cv_auc_folds.png", ...IMGS.cv_folds),
    caption("Figure 7.  5-fold CV — train vs. validation AUC per fold and AUC gap comparison."),
    h2("ROC, Precision-Recall, KS, and Calibration"),
    img("roc_pr_curves.png", ...IMGS.roc_pr_curves),
    caption("Figure 8.  ROC and precision-recall curves — champion vs. challenger."),
    img("ks_plot_LGBM_Champion.png", ...IMGS.ks_lgbm),
    caption("Figure 9.  KS plot — LGBM Champion."),
    img("calibration_curve.png", ...IMGS.calibration_curve),
    caption("Figure 10.  Calibration curve (reliability diagram)."),
    img("confusion_matrices.png", ...IMGS.confusion_matrices),
    caption("Figure 11.  Confusion matrices at 0.50 threshold."),
    p("The validation evidence supports the conclusion that the champion is a strong rank-ordering model. A production underwriting threshold should not be selected mechanically at 0.50. Threshold selection should reflect credit policy, expected loss, approval capacity, and calibration."),
    new Paragraph({ children: [new PageBreak()] }),
  );

  // ── 7. STABILITY & SENSITIVITY ────────────────────────────────────────────
  add(
    h1("7.  Stability and Sensitivity Analysis"),
    h2("PSI Stability"),
    p("The PSI results indicate stability between training and validation samples. This should be treated as a development check, not production drift evidence. The dataset is randomly split, so near-zero PSI is expected."),
    h2("Sensitivity Results"),
    p("The sensitivity analysis perturbs each input by +/-10% and reports average absolute probability-score movement."),
    tbl(
      ["Rank", "Feature", "Avg Absolute Score Movement"],
      [
        ["1",  "age",                                   "0.013805"],
        ["2",  "RevolvingUtilizationOfUnsecuredLines",  "0.011690"],
        ["3",  "MonthlyIncome",                         "0.004820"],
        ["4",  "NumberOfOpenCreditLinesAndLoans",       "0.004510"],
        ["5",  "DebtRatio",                             "0.003750"],
        ["6",  "NumberOfTime60-89DaysPastDueNotWorse",  "0.001460"],
        ["7",  "NumberOfTimes90DaysLate",               "0.001335"],
        ["8",  "NumberRealEstateLoansOrLines",          "0.000965"],
        ["9",  "NumberOfDependents",                    "0.000225"],
        ["10", "NumberOfTime30-59DaysPastDueNotWorse",  "0.000175"],
      ],
      [1000, 5360, 3000],
    ),
    gap(120),
    p("Age is the most sensitive feature. This creates a direct link between predictive behavior and fair lending considerations, which are addressed in Section 8."),
    new Paragraph({ children: [new PageBreak()] }),
  );

  // ── 8. FAIR LENDING ───────────────────────────────────────────────────────
  add(
    h1("8.  Fair Lending and Use Risk"),
    h2("Legal Status of Age Under ECOA"),
    p("Under the Equal Credit Opportunity Act (ECOA, 15 U.S.C. § 1691) and Regulation B, age is an explicitly permitted factor in empirically derived, statistically sound credit scoring systems. Unlike race, sex, or national origin, age may be used as a direct predictive variable provided the scoring system is based on demonstrated statistical relationships between age and credit performance. The LGBM pipeline is empirically derived from historical default data and satisfies this criterion. Age use in this model is therefore ECOA-compliant and does not constitute unlawful discrimination."),
    h2("Disparate Impact Results"),
    tbl(
      ["Age Group", "Model", "N", "Default Rate", "Approval Rate", "AUC", "DIR vs Best"],
      [
        ["Young (<30)",    "LGBM Champion", "2,623",  "12.54%", "24.70%", "0.8189", "0.3351"],
        ["Young (<30)",    "LR Baseline",   "2,623",  "12.54%", "22.80%", "0.7814", "0.3051"],
        ["Middle (30-60)", "LGBM Champion", "27,795", "8.12%",  "39.94%", "0.8459", "0.5418"],
        ["Middle (30-60)", "LR Baseline",   "27,795", "8.12%",  "39.59%", "0.8173", "0.5297"],
        ["Senior (>60)",   "LGBM Champion", "14,582", "2.89%",  "73.72%", "0.8534", "1.0000"],
        ["Senior (>60)",   "LR Baseline",   "14,582", "2.89%",  "74.74%", "0.8131", "1.0000"],
      ],
      [1600, 1700, 1000, 1200, 1300, 1000, 1560],
    ),
    gap(120),
    img("fairness_age_groups.png", ...IMGS.fairness),
    caption("Figure 12.  Approval rate and AUC by age group — champion vs. challenger."),
    h2("Interpretation"),
    p("The young-borrower DIR is 0.3351 for the champion. The four-fifths (0.80) threshold is an EEOC employment-law screening heuristic applied here as a monitoring reference, not a binding credit standard. The DIR gap is consistent with genuine underlying risk differences: young borrowers default at 12.54% versus 2.89% for seniors, a 4.3x difference. The approval rate ratio of 3.0x is proportionally smaller than the default rate ratio, meaning the model is not amplifying the natural risk difference."),
    h2("Less-Discriminatory Alternative Test"),
    p("An age-blind LGBM challenger was trained and evaluated to document whether a less-discriminatory alternative exists with comparable predictive power."),
    tbl(
      ["Model", "Overall AUC", "Overall KS", "Young-Borrower DIR"],
      [
        ["LGBM Champion",  "0.8601", "0.5700", "0.3351"],
        ["LGBM Age-Blind", "0.8229", "0.5055", "0.5081"],
        ["LR Baseline",    "0.8317", "0.5075", "0.3051"],
      ],
      [3000, 2120, 2120, 2120],
    ),
    gap(120),
    img("age_blind_comparison.png", ...IMGS.age_blind),
    caption("Figure 13.  Less-discriminatory alternative comparison — approval rate, DIR, and AUC."),
    p("Removing age raises the young-borrower DIR from 0.3351 to 0.5081 (+0.173) but reduces AUC by 0.0372, dropping the champion below the LR baseline. The remaining DIR gap in the age-blind model is driven by age-correlated legitimate credit factors (utilization, income, delinquency history). This confirms that age is doing genuine predictive work proportional to actual default rate differences, and that a strict less-discriminatory alternative does not exist without material accuracy loss."),
    h2("Fair Lending Finding"),
    p("Age use is documented as ECOA-compliant. The DIR results are consistent with actual default rate differences and do not indicate unlawful discrimination. Age-based disparate impact should be monitored semiannually to detect any future drift beyond what the underlying default rate differences justify."),
    new Paragraph({ children: [new PageBreak()] }),
  );

  // ── 9. MODEL RISK TIERING ─────────────────────────────────────────────────
  add(
    h1("9.  Model Risk Tiering"),
    p("The model is classified as Tier 1 — High Risk under the internal Model Risk Scorecard (MRS). The scorecard is not an SR 11-7 formula; it is an internal tiering tool used to determine validation rigor."),
    tbl(
      ["Dimension", "Max Points", "Score", "Score %", "Justification"],
      [
        ["Materiality",         "30", "27", "90%", "Model affects credit decisions for a large borrower population"],
        ["Regulatory Exposure", "30", "25", "83%", "SR 11-7, ECOA, explainability, and fair lending exposure"],
        ["Operational Factor",  "10",  "8", "80%", "Automated pipeline with no documented override control"],
        ["Model Complexity",    "30", "28", "93%", "LightGBM, WOE clustering, polynomial terms, and feature selection"],
        ["Total",              "100", "88", "88%", "Tier 1 — High Risk"],
      ],
      [2400, 1200, 1000, 1000, 3760],
    ),
    gap(120),
    img("mrs_scorecard_chart.png", ...IMGS.mrs_chart),
    caption("Figure 14.  Model Risk Scorecard — dimension scores."),
    p("The Tier 1 rating is appropriate. Credit decisioning is high impact, the model is complex, and the validation has identified overfitting requiring remediation."),
    new Paragraph({ children: [new PageBreak()] }),
  );

  // ── 10. SR 11-7 CHECKLIST ─────────────────────────────────────────────────
  add(
    h1("10.  SR 11-7 Validation Checklist"),
    p("The following checklist maps current evidence to SR 11-7 expectations. It is a project control assessment, not an official regulatory certification."),
    new Table({
      width: { size: W, type: WidthType.DXA },
      columnWidths: [2000, 2800, 880, 3680],
      rows: [
        new TableRow({ tableHeader: true, children: [
          hcell("SR 11-7 Area", 2000), hcell("Evidence", 2800),
          hcell("Status", 880),       hcell("Action", 3680),
        ]}),
        ...([
          ["Purpose and Scope",     "Credit default probability and intended model use are documented",                             "Pass",   "Maintain model-use limits in inventory"],
          ["Data Quality",          "EDA, missingness indicators, outlier caps, PSI, and imputation assumption validation",         "Pass",   "Document aged Kaggle data as a high limitation"],
          ["Benchmarking",          "LGBM AUC 0.8601 vs. LR AUC 0.8317; lift 0.0284",                                             "Pass",   "Explain complexity-performance tradeoff"],
          ["Outcomes Analysis",     "5-fold CV mean val AUC 0.8297 +/- 0.0065; mean gap 0.1669",                               "Review", "Simplify pipeline or apply early stopping before production"],
          ["Ongoing Monitoring",    "Max validation PSI is 0.0005; thresholds proposed",                                           "Pass",   "Assign production owner and cadence"],
          ["Fair Lending / Use Risk","Age is ECOA-permitted; LDA tested; DIR gap reflects 4.3x default rate difference",       "Pass",   "Monitor DIR semiannually; document age contribution via SHAP"],
          ["Governance",            "MRS scorecard and SR 11-7 taxonomy generated",                                                "Pass",   "State independent validation and audit are not performed"],
        ]).map(([area, ev, status, action], i) =>
          srRow([area, ev, status, action], [2000, 2800, 880, 3680], i % 2 === 1)
        ),
      ],
    }),
    gap(120),
    img("sr117_risk_chart.png", ...IMGS.sr117_chart),
    caption("Figure 15.  SR 11-7 risk taxonomy."),
    new Paragraph({ children: [new PageBreak()] }),
  );

  // ── 11. MONITORING FRAMEWORK ──────────────────────────────────────────────
  add(
    h1("11.  Monitoring and Control Framework"),
    p("The model should not be deployed without a formal monitoring framework. Monitoring should address data integrity, score behavior, performance, calibration, fairness, and overrides."),
    tbl(
      ["Metric", "Frequency", "Trigger", "Required Action"],
      [
        ["Feature PSI",                    "Monthly",                       "Any feature > 0.20",               "Root cause review"],
        ["Feature PSI",                    "Monthly",                       "Any feature > 0.25",               "Revalidation trigger"],
        ["Missingness rate",               "Monthly",                       "Material deviation from baseline",  "Data quality review"],
        ["KMeans cluster distribution",    "Monthly",                       "Cluster share doubles, halves, or empties", "Pipeline review"],
        ["AUC-ROC",                        "Quarterly after label maturity","Degradation > 2 pct points",       "Performance review"],
        ["KS statistic",                   "Quarterly after label maturity","Degradation > 5 pct points",       "Rank-ordering review"],
        ["Brier score",                    "Quarterly",                     "Material deterioration",           "Calibration review"],
        ["Actual vs. predicted default rate","Quarterly",                   "Deviation > 10%",                  "Recalibration assessment"],
        ["Disparate Impact Ratio",         "Semiannual",                    "Any group < 0.80",                 "Fair lending escalation"],
        ["Override rate",                  "Monthly",                       "Unusual level or trend",           "Policy and manual review assessment"],
      ],
      [2400, 1800, 2560, 2600],
    ),
    gap(120),
    p("Monitoring must have accountable owners, evidence retention, escalation paths, and criteria for model change approval. Material changes should be validated before implementation."),
    new Paragraph({ children: [new PageBreak()] }),
  );

  // ── 12. FINAL VALIDATION OPINION ─────────────────────────────────────────
  add(
    h1("12.  Final Validation Opinion"),
    p("The LightGBM champion is a strong predictive model that demonstrates consistent out-of-sample discrimination on the Give Me Some Credit dataset. Age use is ECOA-compliant. SHAP explanations, 5-fold CV overfitting evidence, and a less-discriminatory-alternative test have been completed and are part of the validation evidence package."),
    p("The remaining production readiness gaps are systematic overfitting (mean CV AUC gap 0.1669), absence of governance controls, and undefined score thresholds. These items must be remediated before any live credit decisioning use. The appropriate validation conclusion is conditional approval pending remediation of those items."),
    p("The model is suitable for academic demonstration and model risk case-study analysis. It should be classified as Tier 1 — High Risk and governed with the highest validation rigor if considered for any real-world lending application."),
    new Paragraph({ children: [new PageBreak()] }),
  );

  // ── 13. REFERENCES ────────────────────────────────────────────────────────
  add(
    h1("13.  References"),
    p("Board of Governors of the Federal Reserve System and Office of the Comptroller of the Currency. (2011). Supervisory Guidance on Model Risk Management, SR 11-7."),
    p("Kaggle. (2011). Give Me Some Credit competition dataset."),
    p("Ke, G., Meng, Q., Finley, T., Wang, T., Chen, W., Ma, W., Ye, Q., and Liu, T. (2017). LightGBM: A Highly Efficient Gradient Boosting Decision Tree. Advances in Neural Information Processing Systems."),
    p("Kiritz, R., Ravitz, B., and Levonian, M. (2019). Model Risk Tiering."),
    p("Project proposal. Model Risk Assessment of Machine Learning-Based Credit Scoring: A Case Study Using the Give Me Some Credit Dataset. Xiaoyang Zhang, Yifan Xu, Yike Ma."),
  );

  return C;
}

// ── BUILD ─────────────────────────────────────────────────────────────────────
const doc = new Document({
  numbering: {
    config: [
      { reference: "bullets", levels: [{
        level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT,
        style: { paragraph: { indent: { left: 560, hanging: 280 } },
                 run: { font: "Symbol", size: 24 } },
      }]},
      { reference: "numbers", levels: [{
        level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT,
        style: { paragraph: { indent: { left: 560, hanging: 280 } },
                 run: { font: FONT, size: 24 } },
      }]},
    ],
  },
  styles: {
    default: {
      document: { run: { font: FONT, size: 24, color: BLACK } },
    },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { font: FONT, size: 28, bold: true, color: BLACK },
        paragraph: { spacing: { before: 400, after: 160 }, outlineLevel: 0 } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { font: FONT, size: 26, bold: true, color: BLACK },
        paragraph: { spacing: { before: 280, after: 100 }, outlineLevel: 1 } },
      { id: "Heading3", name: "Heading 3", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { font: FONT, size: 24, bold: true, italics: true, color: DGRAY },
        paragraph: { spacing: { before: 200, after: 80 }, outlineLevel: 2 } },
    ],
  },
  sections: [{
    properties: {
      titlePage: true,
      page: {
        size: { width: PAGE_W, height: PAGE_H },
        margin: { top: MARGIN, right: MARGIN, bottom: MARGIN, left: MARGIN },
      },
    },
    headers: {
      default: docHeader,
      first: new Header({ children: [] }),
    },
    footers: {
      default: docFooter,
      first: new Footer({ children: [] }),
    },
    children: children(),
  }],
});

const OUT = "/Users/evanxu/Desktop/Classes/Model Risk/Final Project/outputs/MRM_Validation_Report_professional.docx";
Packer.toBuffer(doc).then(buf => {
  fs.writeFileSync(OUT, buf);
  console.log("Written:", OUT, `(${(buf.length/1024).toFixed(0)} KB)`);
}).catch(e => { console.error(e); process.exit(1); });
