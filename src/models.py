"""
models.py
---------
Stage 2 – Model Development (CRISP-DM / SR 11-7 §II)

Builds TWO models for head-to-head comparison:

  1. Challenger (Baseline):  Logistic Regression
     • Interpretable, scorecard-equivalent
     • Serves as SR 11-7 "benchmark model"

  2. Champion (Complex):     LGBM pipeline (Kaggle Top-5 replica)
     • PowerTransformer → KMeans clusters → WOE encoding →
       PolynomialFeatures → SelectPercentile → LightGBM
     • Private Kaggle AUC ≈ 0.8691

Both models are returned as scikit-learn Pipeline objects so downstream
validation code can call .predict_proba() uniformly.
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.pipeline         import Pipeline
from sklearn.preprocessing    import StandardScaler, PowerTransformer
from sklearn.linear_model     import LogisticRegression
from sklearn.cluster          import MiniBatchKMeans
from sklearn.preprocessing    import PolynomialFeatures
from sklearn.feature_selection import SelectPercentile, f_classif
from sklearn.base             import BaseEstimator, TransformerMixin

warnings.filterwarnings("ignore")


# ── 1. Logistic Regression Baseline ─────────────────────────────────────────

def build_lr_pipeline() -> Pipeline:
    """
    Logistic Regression baseline.
    Comparable to a traditional bank scorecard.
    Provides SR 11-7 required benchmark for the complex champion.
    """
    return Pipeline([
        ("scaler", StandardScaler()),
        ("lr",     LogisticRegression(
            max_iter=1000,
            C=0.1,               # L2 regularisation
            class_weight="balanced",  # handles 6.7% imbalance
            solver="lbfgs",
            random_state=42,
        )),
    ])


# ── 2. WOE Encoder (manual) ──────────────────────────────────────────────────

class WOEKMeansEncoder(BaseEstimator, TransformerMixin):
    """
    Replicates the Kaggle Top-5 cluster + WOE step:
      1. MiniBatchKMeans → assign each row to one of n_clusters clusters
      2. Compute WOE per cluster using training labels
      3. Replace cluster label with WOE value

    WOE_k = ln(P(Event | cluster=k) / P(Non-Event | cluster=k))
    """
    def __init__(self, n_clusters: int = 11, random_state: int = 42):
        self.n_clusters   = n_clusters
        self.random_state = random_state
        self.kmeans_      = None
        self.woe_map_     = {}

    def fit(self, X, y=None):
        X = np.array(X)
        self.kmeans_ = MiniBatchKMeans(
            n_clusters=self.n_clusters,
            random_state=self.random_state,
            n_init=10,
        ).fit(X)

        labels = self.kmeans_.predict(X)
        y      = np.array(y)

        total_events     = max(y.sum(), 1)
        total_non_events = max((1 - y).sum(), 1)

        self.woe_map_ = {}
        for k in range(self.n_clusters):
            mask   = labels == k
            events     = y[mask].sum()
            non_events = mask.sum() - events
            # Laplace smoothing to avoid log(0)
            pct_e  = (events + 0.5)     / total_events
            pct_ne = (non_events + 0.5) / total_non_events
            self.woe_map_[k] = float(np.log(pct_e / pct_ne))

        return self

    def transform(self, X, y=None):
        X      = np.array(X)
        labels = self.kmeans_.predict(X)
        woe_col = np.array([self.woe_map_.get(k, 0.0) for k in labels])
        return np.column_stack([X, woe_col])


# ── 3. LGBM Champion Pipeline ────────────────────────────────────────────────

def build_lgbm_pipeline(n_estimators: int = 5000,
                        learning_rate: float = 0.0018) -> Pipeline:
    """
    Replicates pipe1_lgbm_woe_powertransformer95 from the Kaggle Top-5 solution.

    Steps:
      1. PowerTransformer   → Yeo-Johnson normalisation
      2. WOEKMeansEncoder   → 11 KMeans clusters + WOE feature appended
      3. PolynomialFeatures → degree-2 interaction terms (~74 features)
      4. SelectPercentile   → keep top 95% by ANOVA F-score
      5. LGBMClassifier     → tuned hyperparameters from Kaggle notebook
    """
    try:
        from lightgbm import LGBMClassifier
    except ImportError:
        raise ImportError(
            "LightGBM not installed. Run:  pip install lightgbm --break-system-packages"
        )

    lgbm_params = dict(
        learning_rate  = learning_rate,
        n_estimators   = n_estimators,
        max_depth      = 8,
        num_leaves     = 63,
        subsample      = 0.857,
        colsample_bytree = 0.857,
        min_child_samples = 20,
        reg_alpha      = 0.1,
        reg_lambda     = 0.1,
        class_weight   = "balanced",
        n_jobs         = -1,
        random_state   = 42,
        verbose        = -1,
    )

    return Pipeline([
        ("power",    PowerTransformer(method="yeo-johnson", standardize=True)),
        ("woe_kmeans", WOEKMeansEncoder(n_clusters=11, random_state=42)),
        ("poly",     PolynomialFeatures(degree=2, include_bias=False)),
        ("select",   SelectPercentile(score_func=f_classif, percentile=95)),
        ("lgbm",     LGBMClassifier(**lgbm_params)),
    ])


# ── 4. Training wrapper ──────────────────────────────────────────────────────

def train_model(pipeline: Pipeline, X_train: pd.DataFrame, y_train: pd.Series,
                model_name: str, output_dir: str) -> Pipeline:
    """Fit pipeline and persist to disk."""
    print(f"\n[Train] Fitting {model_name}...")
    pipeline.fit(X_train, y_train)
    print(f"[Train] {model_name} fitted successfully.")

    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f"{model_name}.pkl")
    joblib.dump(pipeline, save_path)
    print(f"[Train] Model saved → {save_path}")
    return pipeline


# ── 5. Calibration check ─────────────────────────────────────────────────────

def plot_calibration(models: dict, X_val: pd.DataFrame, y_val: pd.Series,
                     output_dir: str) -> None:
    """
    Reliability diagram: compare predicted probability vs actual default rate.
    SR 11-7: model outputs must be calibrated before deployment.
    """
    from sklearn.calibration import calibration_curve

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration")

    for name, model in models.items():
        probs = model.predict_proba(X_val)[:, 1]
        frac_pos, mean_pred = calibration_curve(y_val, probs, n_bins=10)
        ax.plot(mean_pred, frac_pos, marker="o", label=name)

    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Fraction of Positives")
    ax.set_title("Calibration Curve (Reliability Diagram)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "calibration_curve.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Calibration] Plot saved → {output_dir}/calibration_curve.png")
