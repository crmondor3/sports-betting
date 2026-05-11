"""
model_trainer.py — train XGBoost, LightGBM, and Logistic Regression models.

Uses TimeSeriesSplit to prevent look-ahead leakage, Optuna for hyperparameter
tuning, SHAP for interpretability, and sklearn's CalibratedClassifierCV for
Platt-scaling probability calibration.

Three target types are supported:
  • h2h     — home team wins (moneyline)
  • spreads — home team covers the DraftKings spread
  • totals  — Over hits

Trained bundles are saved under models/<sport>_<market>_ml.pkl
"""
from __future__ import annotations

import logging
import pickle
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=UserWarning)

_MODEL_DIR = Path(__file__).parent / "models"
_N_SPLITS = 5
_MIN_ROWS = 150   # minimum training rows per target
_OPTUNA_TRIALS = 40


# ── Model bundle path ─────────────────────────────────────────────────────────

def _bundle_path(sport: str, market: str) -> Path:
    return _MODEL_DIR / f"{sport.lower()}_{market}_ml.pkl"


# ── ModelTrainer ──────────────────────────────────────────────────────────────

class ModelTrainer:
    """
    Train and evaluate three models (XGBoost, LightGBM, LogReg) on a
    pre-built feature DataFrame from FeatureEngineer.

    Usage:
        trainer = ModelTrainer(sport="NFL")
        metrics = trainer.fit(df, market="h2h")
        bundle  = trainer.load("h2h")
        probs   = trainer.predict(features_dict, market="h2h")
    """

    def __init__(self, sport: str = "NFL") -> None:
        self.sport = sport.upper()
        _MODEL_DIR.mkdir(exist_ok=True)

    # ── Training ──────────────────────────────────────────────────────────────

    def fit(
        self,
        df: pd.DataFrame,
        market: str = "h2h",
        use_optuna: bool = True,
        n_trials: int = _OPTUNA_TRIALS,
    ) -> dict[str, Any]:
        """
        Train all three models on *df* for the given *market*.

        Args:
            df:         Feature DataFrame from FeatureEngineer.build()
            market:     "h2h" | "spreads" | "totals"
            use_optuna: Whether to run Optuna hyperparameter search
            n_trials:   Number of Optuna trials per model

        Returns:
            Metrics dict with keys per model + averaged ensemble metrics.
        """
        from feature_engineer import FEATURE_NAMES

        target_col = self._target_col(market)
        sub = df[df[target_col].notna()].copy()
        if len(sub) < _MIN_ROWS:
            logger.warning(
                "%s %s: only %d rows (need %d) — skipping",
                self.sport, market, len(sub), _MIN_ROWS,
            )
            return {"sport": self.sport, "market": market, "error": "insufficient data"}

        feat_cols = [c for c in FEATURE_NAMES if c in sub.columns]
        X = sub[feat_cols].fillna(0).values.astype(float)
        y = sub[target_col].values.astype(int)

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        logger.info(
            "%s %s: training on %d rows, %d features",
            self.sport, market, len(X), len(feat_cols),
        )

        # ── TimeSeriesSplit CV ─────────────────────────────────────────────
        tscv = TimeSeriesSplit(n_splits=_N_SPLITS)

        xgb_params  = self._tune_xgboost(X, y, tscv, n_trials) if use_optuna else {}
        lgb_params  = self._tune_lightgbm(X, y, tscv, n_trials) if use_optuna else {}

        xgb_model  = self._train_xgboost(X_scaled,  y, xgb_params)
        lgb_model  = self._train_lightgbm(X, y, lgb_params)   # LGB uses unscaled
        lr_model   = self._train_logreg(X_scaled, y)

        # ── SHAP feature importance ────────────────────────────────────────
        shap_importance = self._shap_importance(xgb_model, X_scaled, feat_cols)

        # ── Evaluate on final TimeSeriesSplit fold ────────────────────────
        metrics = self._evaluate_all(
            {"xgboost": xgb_model, "lightgbm": lgb_model, "logreg": lr_model},
            X, X_scaled, y, tscv,
        )
        metrics["shap_top_features"] = shap_importance[:8]

        # ── Persist ───────────────────────────────────────────────────────
        bundle = {
            "sport":        self.sport,
            "market":       market,
            "feature_cols": feat_cols,
            "scaler":       scaler,
            "xgboost":      xgb_model,
            "lightgbm":     lgb_model,
            "logreg":       lr_model,
            "metrics":      metrics,
            "trained_at":   pd.Timestamp.utcnow().isoformat(),
            "n_train":      len(X),
        }

        path = _bundle_path(self.sport, market)
        with path.open("wb") as fh:
            pickle.dump(bundle, fh)

        logger.info(
            "%s %s saved → %s  (AUC xgb=%.3f lgb=%.3f lr=%.3f)",
            self.sport, market, path.name,
            metrics.get("xgboost_auc", 0),
            metrics.get("lightgbm_auc", 0),
            metrics.get("logreg_auc", 0),
        )
        return metrics

    # ── Prediction ────────────────────────────────────────────────────────────

    def predict(
        self,
        features: dict[str, float],
        market: str = "h2h",
    ) -> dict[str, Any]:
        """
        Predict probability for a single game.

        Returns dict with keys:
            xgboost_prob, lightgbm_prob, logreg_prob, ensemble_prob, confidence
        """
        bundle = self.load(market)
        if bundle is None:
            raise RuntimeError(f"No trained model for {self.sport}/{market}. Run fit() first.")

        feat_cols = bundle["feature_cols"]
        x = np.array([[features.get(c, 0.0) for c in feat_cols]], dtype=float)
        x_scaled = bundle["scaler"].transform(x)

        xgb_p = float(bundle["xgboost"].predict_proba(x_scaled)[0, 1])
        lgb_p = float(bundle["lightgbm"].predict_proba(x)[0, 1])
        lr_p  = float(bundle["logreg"].predict_proba(x_scaled)[0, 1])
        ens   = (xgb_p + lgb_p + lr_p) / 3.0

        # Confidence = how far the ensemble is from 0.5
        confidence = int(abs(ens - 0.5) * 200)  # 0..100 scale

        return {
            "xgboost_prob":  round(xgb_p, 4),
            "lightgbm_prob": round(lgb_p, 4),
            "logreg_prob":   round(lr_p,  4),
            "ensemble_prob": round(ens,   4),
            "confidence":    confidence,
        }

    # ── Load / check ──────────────────────────────────────────────────────────

    def load(self, market: str = "h2h") -> dict | None:
        """Load a saved model bundle, or None if not found."""
        path = _bundle_path(self.sport, market)
        if not path.exists():
            return None
        try:
            with path.open("rb") as fh:
                return pickle.load(fh)
        except Exception as exc:
            logger.warning("Failed to load %s %s bundle: %s", self.sport, market, exc)
            return None

    def is_trained(self, market: str = "h2h") -> bool:
        return _bundle_path(self.sport, market).exists()

    # ── XGBoost ──────────────────────────────────────────────────────────────

    def _tune_xgboost(
        self, X: np.ndarray, y: np.ndarray, tscv: TimeSeriesSplit, n_trials: int
    ) -> dict:
        try:
            import optuna
            import xgboost as xgb
            optuna.logging.set_verbosity(optuna.logging.WARNING)

            def objective(trial: optuna.Trial) -> float:
                params = {
                    "max_depth":       trial.suggest_int("max_depth", 2, 6),
                    "learning_rate":   trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                    "n_estimators":    trial.suggest_int("n_estimators", 100, 500),
                    "subsample":       trial.suggest_float("subsample", 0.6, 1.0),
                    "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                    "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
                    "reg_alpha":       trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
                    "eval_metric": "logloss",
                    "random_state": 42,
                }
                scores = []
                for tr_idx, val_idx in tscv.split(X):
                    m = xgb.XGBClassifier(**params)
                    m.fit(X[tr_idx], y[tr_idx], eval_set=[(X[val_idx], y[val_idx])], verbose=False)
                    p = m.predict_proba(X[val_idx])[:, 1]
                    scores.append(log_loss(y[val_idx], p))
                return float(np.mean(scores))

            study = optuna.create_study(direction="minimize")
            study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
            return study.best_params
        except Exception as exc:
            logger.warning("XGBoost Optuna failed (%s); using defaults.", exc)
            return {}

    def _train_xgboost(self, X: np.ndarray, y: np.ndarray, params: dict) -> Any:
        import xgboost as xgb
        defaults = {
            "max_depth": 4, "learning_rate": 0.05, "n_estimators": 300,
            "subsample": 0.8, "colsample_bytree": 0.8,
            "eval_metric": "logloss", "random_state": 42,
        }
        defaults.update(params)
        model = xgb.XGBClassifier(**defaults)
        # Platt scaling calibration via CalibratedClassifierCV
        cal = CalibratedClassifierCV(model, method="sigmoid", cv=3)
        cal.fit(X, y)
        return cal

    # ── LightGBM ─────────────────────────────────────────────────────────────

    def _tune_lightgbm(
        self, X: np.ndarray, y: np.ndarray, tscv: TimeSeriesSplit, n_trials: int
    ) -> dict:
        try:
            import optuna
            import lightgbm as lgb
            optuna.logging.set_verbosity(optuna.logging.WARNING)

            def objective(trial: optuna.Trial) -> float:
                params = {
                    "num_leaves":      trial.suggest_int("num_leaves", 15, 63),
                    "learning_rate":   trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                    "n_estimators":    trial.suggest_int("n_estimators", 100, 500),
                    "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
                    "subsample":       trial.suggest_float("subsample", 0.6, 1.0),
                    "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                    "reg_alpha":       trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
                    "verbose": -1,
                    "random_state": 42,
                }
                scores = []
                for tr_idx, val_idx in tscv.split(X):
                    m = lgb.LGBMClassifier(**params)
                    m.fit(X[tr_idx], y[tr_idx], eval_set=[(X[val_idx], y[val_idx])],
                          callbacks=[lgb.early_stopping(50, verbose=False),
                                     lgb.log_evaluation(-1)])
                    p = m.predict_proba(X[val_idx])[:, 1]
                    scores.append(log_loss(y[val_idx], p))
                return float(np.mean(scores))

            study = optuna.create_study(direction="minimize")
            study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
            return study.best_params
        except Exception as exc:
            logger.warning("LightGBM Optuna failed (%s); using defaults.", exc)
            return {}

    def _train_lightgbm(self, X: np.ndarray, y: np.ndarray, params: dict) -> Any:
        import lightgbm as lgb
        defaults = {
            "num_leaves": 31, "learning_rate": 0.05, "n_estimators": 300,
            "subsample": 0.8, "colsample_bytree": 0.8,
            "verbose": -1, "random_state": 42,
        }
        defaults.update(params)
        model = lgb.LGBMClassifier(**defaults)
        cal = CalibratedClassifierCV(model, method="sigmoid", cv=3)
        cal.fit(X, y)
        return cal

    # ── Logistic Regression ───────────────────────────────────────────────────

    def _train_logreg(self, X: np.ndarray, y: np.ndarray) -> Any:
        lr = LogisticRegression(C=1.0, max_iter=1000, random_state=42, solver="lbfgs")
        cal = CalibratedClassifierCV(lr, method="sigmoid", cv=3)
        cal.fit(X, y)
        return cal

    # ── SHAP ─────────────────────────────────────────────────────────────────

    def _shap_importance(
        self, model: Any, X: np.ndarray, feat_cols: list[str]
    ) -> list[tuple[str, float]]:
        try:
            import shap
            # Unwrap CalibratedClassifierCV to get base estimator
            base = model.calibrated_classifiers_[0].estimator
            explainer = shap.TreeExplainer(base)
            shap_vals = explainer.shap_values(X[:min(500, len(X))])
            if isinstance(shap_vals, list):
                shap_vals = shap_vals[1]
            importance = np.abs(shap_vals).mean(axis=0)
            pairs = sorted(zip(feat_cols, importance.tolist()), key=lambda x: -x[1])
            return pairs
        except Exception as exc:
            logger.warning("SHAP analysis failed: %s", exc)
            return []

    # ── Evaluation ────────────────────────────────────────────────────────────

    def _evaluate_all(
        self,
        models: dict[str, Any],
        X_raw: np.ndarray,
        X_scaled: np.ndarray,
        y: np.ndarray,
        tscv: TimeSeriesSplit,
    ) -> dict[str, float]:
        metrics: dict[str, Any] = {}
        # Use final fold for evaluation
        splits = list(tscv.split(X_raw))
        if not splits:
            return metrics
        tr_idx, val_idx = splits[-1]

        for name, model in models.items():
            X_tr = X_raw[tr_idx] if name == "lightgbm" else X_scaled[tr_idx]
            X_val = X_raw[val_idx] if name == "lightgbm" else X_scaled[val_idx]
            try:
                proba = model.predict_proba(X_val)[:, 1]
                metrics[f"{name}_auc"]    = round(roc_auc_score(y[val_idx], proba), 4)
                metrics[f"{name}_brier"]  = round(brier_score_loss(y[val_idx], proba), 5)
                metrics[f"{name}_logloss"] = round(log_loss(y[val_idx], proba), 5)
                metrics[f"{name}_acc"]    = round(float(((proba >= 0.5) == y[val_idx]).mean()), 4)
            except Exception as exc:
                logger.warning("Eval failed for %s: %s", name, exc)

        return metrics

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _target_col(market: str) -> str:
        return {"h2h": "home_win", "spreads": "spread_win", "totals": "over_hit"}.get(market, "home_win")

    def train_all_markets(
        self,
        df: pd.DataFrame,
        use_optuna: bool = True,
    ) -> list[dict]:
        """Train h2h, spreads, and totals models (spread/totals require odds data)."""
        results = []
        for market in ("h2h", "spreads", "totals"):
            target_col = self._target_col(market)
            if target_col not in df.columns:
                logger.info("Skipping %s — target column '%s' not in DataFrame.", market, target_col)
                continue
            results.append(self.fit(df, market=market, use_optuna=use_optuna))
        return results
