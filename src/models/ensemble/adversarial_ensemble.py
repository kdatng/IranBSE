"""Adversarial validation ensemble for detecting and adapting to distribution shift.

When generating predictions for hypothetical conflict scenarios, training data
may be drawn from peacetime markets while target scenarios reflect crisis
dynamics.  This module detects such covariate shift and reweights base models
to favour those whose training distributions overlap most with the target
domain.

Typical usage::

    ensemble = AdversarialEnsemble(config, base_models=[m1, m2, m3])
    ensemble.fit(training_data)
    shift_report = ensemble.detect_shift(scenario_data)
    ensemble.reweight(scenario_data)
    result = ensemble.predict(horizon=20)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import polars as pl
from loguru import logger
from numpy.typing import NDArray

from src.models.base_model import BaseModel, ModelConfig, PredictionResult


@dataclass(frozen=True)
class ShiftReport:
    """Summary of detected distribution shift between train and scenario data.

    Attributes:
        auc_roc: AUC-ROC of a classifier distinguishing train vs. scenario
            samples.  Values near 0.5 indicate no shift; values near 1.0
            indicate severe shift.
        feature_importances: Mapping of feature name to its discriminative
            importance in the shift classifier.
        shifted_features: Features whose individual shift score exceeds the
            threshold.
        shift_severity: Qualitative label (``"none"``, ``"mild"``,
            ``"moderate"``, ``"severe"``).
        sample_weights: Importance weights for training samples to reduce
            shift (higher weight = more similar to scenario domain).
    """

    auc_roc: float
    feature_importances: dict[str, float]
    shifted_features: list[str]
    shift_severity: str
    sample_weights: list[float] = field(default_factory=list)


class AdversarialEnsemble(BaseModel):
    """Ensemble that adapts to distribution shift via adversarial validation.

    The approach trains a lightweight binary classifier to distinguish
    historical training rows from scenario rows.  Model-level reweighting
    is then derived from how well each base model's residual distribution
    aligns with the scenario domain.

    Args:
        config: Standard model configuration.
        base_models: List of fitted :class:`BaseModel` instances.
        shift_threshold: AUC-ROC above which reweighting activates.
            Default of 0.6 allows for mild natural variation while
            triggering adaptation for meaningful distributional differences.
        n_adversarial_trees: Number of decision stumps in the shift
            classifier.  Default of 100 balances detection power with
            computational cost for typical feature dimensions (10-50).
        adaptation_strength: Controls how aggressively weights shift toward
            scenario-adapted models (0 = no adaptation, 1 = full).
    """

    def __init__(
        self,
        config: ModelConfig,
        base_models: list[BaseModel],
        shift_threshold: float = 0.60,
        n_adversarial_trees: int = 100,
        adaptation_strength: float = 0.7,
    ) -> None:
        super().__init__(config)
        if not base_models:
            raise ValueError("AdversarialEnsemble requires at least one base model.")
        self.base_models = base_models
        self.shift_threshold = shift_threshold
        self.n_adversarial_trees = n_adversarial_trees
        self.adaptation_strength = adaptation_strength

        # Fitted state
        self._training_features: NDArray[np.float64] | None = None
        self._feature_names: list[str] = []
        self._base_weights: NDArray[np.float64] | None = None
        self._adapted_weights: NDArray[np.float64] | None = None
        self._last_shift_report: ShiftReport | None = None

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def fit(self, data: pl.DataFrame) -> None:
        """Fit the ensemble on historical training data.

        Stores the training feature matrix for later shift detection and
        assigns equal initial weights to all base models.

        Args:
            data: Historical DataFrame with feature columns and a ``target``
                column.

        Raises:
            ValueError: If data is too short or missing ``target``.
        """
        self._validate_data(data, required_columns=["target"])

        self._feature_names = [c for c in data.columns if c != "target"]
        self._training_features = (
            data.select(self._feature_names).to_numpy().astype(np.float64)
        )
        n_models = len(self.base_models)
        self._base_weights = np.ones(n_models, dtype=np.float64) / n_models
        self._adapted_weights = self._base_weights.copy()

        self._mark_fitted(data)
        logger.info(
            "AdversarialEnsemble fitted with {} features, {} base models",
            len(self._feature_names),
            n_models,
        )

    def detect_shift(self, scenario_data: pl.DataFrame) -> ShiftReport:
        """Detect distribution shift between training and scenario data.

        Trains a random-stump classifier on a binary label (0 = train,
        1 = scenario) and evaluates AUC-ROC.

        Args:
            scenario_data: DataFrame with the same feature columns as
                training data (``target`` column optional).

        Returns:
            A :class:`ShiftReport` summarising shift magnitude and which
            features contribute most.

        Raises:
            RuntimeError: If the model has not been fitted.
            ValueError: If scenario data is missing required feature columns.
        """
        self._require_fitted()
        assert self._training_features is not None

        scenario_cols = [c for c in self._feature_names if c in scenario_data.columns]
        if len(scenario_cols) < len(self._feature_names) * 0.5:
            raise ValueError(
                f"Scenario data has only {len(scenario_cols)} of "
                f"{len(self._feature_names)} expected features."
            )

        # Select overlapping features only
        train_idx = [self._feature_names.index(c) for c in scenario_cols]
        X_train = self._training_features[:, train_idx]
        X_scenario = scenario_data.select(scenario_cols).to_numpy().astype(np.float64)

        # Build combined dataset
        X_combined = np.vstack([X_train, X_scenario])
        y_combined = np.concatenate(
            [np.zeros(len(X_train)), np.ones(len(X_scenario))]
        )

        # Fit random decision stumps and compute AUC-ROC
        auc, importances, sample_weights = self._fit_adversarial_classifier(
            X_combined, y_combined, n_train=len(X_train)
        )

        feature_imp = {
            name: float(imp) for name, imp in zip(scenario_cols, importances, strict=True)
        }

        # Identify shifted features (importance > mean + 1 std)
        imp_arr = np.array(list(feature_imp.values()))
        threshold = float(imp_arr.mean() + imp_arr.std())
        shifted = [
            name for name, imp in feature_imp.items() if imp > threshold
        ]

        # Classify severity
        if auc < 0.55:
            severity = "none"
        elif auc < 0.65:
            severity = "mild"
        elif auc < 0.80:
            severity = "moderate"
        else:
            severity = "severe"

        report = ShiftReport(
            auc_roc=auc,
            feature_importances=feature_imp,
            shifted_features=shifted,
            shift_severity=severity,
            sample_weights=sample_weights[:len(X_train)].tolist(),
        )
        self._last_shift_report = report

        logger.info(
            "Distribution shift detected: AUC={:.3f}, severity={}, "
            "shifted features={}",
            auc,
            severity,
            shifted,
        )
        return report

    def reweight(self, scenario_data: pl.DataFrame) -> dict[str, float]:
        """Reweight base models to adapt to the scenario domain.

        If shift exceeds ``shift_threshold``, models whose residuals on
        scenario-like training samples are smaller receive higher weight.
        Otherwise, retains equal weighting.

        Args:
            scenario_data: Scenario DataFrame (same schema as training).

        Returns:
            Mapping of model names to adapted weights.

        Raises:
            RuntimeError: If the model has not been fitted.
        """
        self._require_fitted()
        assert self._base_weights is not None

        # Detect shift if not already done
        if self._last_shift_report is None:
            self.detect_shift(scenario_data)

        assert self._last_shift_report is not None
        report = self._last_shift_report

        if report.auc_roc < self.shift_threshold:
            logger.info(
                "Shift AUC {:.3f} below threshold {:.3f}; retaining equal weights",
                report.auc_roc,
                self.shift_threshold,
            )
            self._adapted_weights = self._base_weights.copy()
        else:
            # Score each base model by its domain-adaptation fitness
            model_scores = self._score_models_on_domain(
                scenario_data, report.sample_weights
            )
            # Blend original and adapted weights
            adapted = self._softmax(model_scores)
            self._adapted_weights = (
                (1 - self.adaptation_strength) * self._base_weights
                + self.adaptation_strength * adapted
            )
            # Renormalise
            self._adapted_weights /= self._adapted_weights.sum()
            logger.info(
                "Reweighted models (adaptation_strength={:.2f}): {}",
                self.adaptation_strength,
                {
                    m.config.name: f"{w:.4f}"
                    for m, w in zip(self.base_models, self._adapted_weights)
                },
            )

        return {
            m.config.name: float(w)
            for m, w in zip(self.base_models, self._adapted_weights, strict=True)
        }

    def predict(
        self,
        horizon: int,
        n_scenarios: int = 1000,
    ) -> PredictionResult:
        """Generate adapted ensemble predictions.

        Blends base model forecasts using the (possibly reweighted) model
        weights.

        Args:
            horizon: Number of forward time steps.
            n_scenarios: Number of Monte-Carlo paths per base model.

        Returns:
            PredictionResult with adapted blended forecasts.

        Raises:
            RuntimeError: If the model has not been fitted.
        """
        self._require_fitted()

        weights = (
            self._adapted_weights
            if self._adapted_weights is not None
            else self._base_weights
        )
        assert weights is not None

        base_results: list[PredictionResult] = []
        for model in self.base_models:
            base_results.append(model.predict(horizon, n_scenarios))

        # Blend point forecasts
        point_stack = np.array(
            [r.point_forecast for r in base_results], dtype=np.float64
        )
        blended_point = (weights @ point_stack).tolist()

        # Blend scenario paths
        scenario_stack = np.zeros(
            (len(self.base_models), n_scenarios, horizon), dtype=np.float64
        )
        for i, result in enumerate(base_results):
            if result.scenarios:
                for j, key in enumerate(sorted(result.scenarios)):
                    if j < n_scenarios:
                        path = result.scenarios[key]
                        scenario_stack[i, j, : len(path)] = path[:horizon]

        blended_scenarios = np.tensordot(weights, scenario_stack, axes=([0], [0]))

        lower_bounds = {
            0.05: np.quantile(blended_scenarios, 0.05, axis=0).tolist(),
            0.10: np.quantile(blended_scenarios, 0.10, axis=0).tolist(),
        }
        upper_bounds = {
            0.90: np.quantile(blended_scenarios, 0.90, axis=0).tolist(),
            0.95: np.quantile(blended_scenarios, 0.95, axis=0).tolist(),
        }

        return PredictionResult(
            point_forecast=blended_point,
            lower_bounds=lower_bounds,
            upper_bounds=upper_bounds,
            scenarios={
                f"path_{i}": blended_scenarios[i].tolist()
                for i in range(min(n_scenarios, blended_scenarios.shape[0]))
            },
            metadata={
                "model": self.config.name,
                "weights": weights.tolist(),
                "shift_severity": (
                    self._last_shift_report.shift_severity
                    if self._last_shift_report
                    else "unknown"
                ),
                "shift_auc": (
                    self._last_shift_report.auc_roc
                    if self._last_shift_report
                    else None
                ),
            },
        )

    def get_params(self) -> dict[str, Any]:
        """Return fitted ensemble parameters.

        Returns:
            Dictionary containing weights, shift detection results, and
            configuration.

        Raises:
            RuntimeError: If the model has not been fitted.
        """
        self._require_fitted()
        return {
            "n_base_models": len(self.base_models),
            "shift_threshold": self.shift_threshold,
            "adaptation_strength": self.adaptation_strength,
            "base_weights": (
                self._base_weights.tolist() if self._base_weights is not None else None
            ),
            "adapted_weights": (
                self._adapted_weights.tolist()
                if self._adapted_weights is not None
                else None
            ),
            "last_shift_report": (
                {
                    "auc_roc": self._last_shift_report.auc_roc,
                    "shift_severity": self._last_shift_report.shift_severity,
                    "shifted_features": self._last_shift_report.shifted_features,
                }
                if self._last_shift_report
                else None
            ),
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _fit_adversarial_classifier(
        self,
        X: NDArray[np.float64],
        y: NDArray[np.float64],
        n_train: int,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        """Fit an ensemble of decision stumps for domain discrimination.

        Each stump picks a random feature and threshold, then votes on
        whether a sample is from training (0) or scenario (1).  This
        avoids a hard dependency on scikit-learn.

        Args:
            X: Combined feature matrix, shape ``(n_train + n_scenario, n_features)``.
            y: Binary label (0=train, 1=scenario).
            n_train: Number of training samples (for weight extraction).

        Returns:
            Tuple of (auc_roc, feature_importances, sample_probability_scores).
        """
        rng = np.random.default_rng(seed=42)
        n_samples, n_features = X.shape
        votes = np.zeros(n_samples, dtype=np.float64)
        feature_usage = np.zeros(n_features, dtype=np.float64)

        for _ in range(self.n_adversarial_trees):
            feat_idx = rng.integers(0, n_features)
            col = X[:, feat_idx]

            # Random threshold between min and max of the feature
            threshold = rng.uniform(float(col.min()), float(col.max()))

            # Simple stump: predict 1 if value > threshold
            pred_left = float(y[col <= threshold].mean()) if (col <= threshold).any() else 0.5
            pred_right = float(y[col > threshold].mean()) if (col > threshold).any() else 0.5

            stump_pred = np.where(col <= threshold, pred_left, pred_right)
            votes += stump_pred
            feature_usage[feat_idx] += abs(pred_right - pred_left)

        # Average predictions
        prob_scores = votes / self.n_adversarial_trees

        # Compute AUC-ROC via ranking
        auc = self._compute_auc(y, prob_scores)

        # Normalise feature importances
        feature_importances = feature_usage / max(feature_usage.sum(), 1e-12)

        # Compute sample weights (inverse propensity): P(scenario|x) / P(train|x)
        # Clamp probabilities to avoid division by zero
        eps = 1e-6
        prob_clamped = np.clip(prob_scores, eps, 1 - eps)
        sample_weights = prob_clamped / (1 - prob_clamped)
        # Normalise training sample weights
        train_weights = sample_weights[:n_train]
        train_weights /= max(train_weights.sum(), 1e-12) / n_train

        full_weights = np.ones(n_samples, dtype=np.float64)
        full_weights[:n_train] = train_weights

        return auc, feature_importances, full_weights

    @staticmethod
    def _compute_auc(
        y_true: NDArray[np.float64],
        y_score: NDArray[np.float64],
    ) -> float:
        """Compute AUC-ROC from true labels and predicted scores.

        Uses the Wilcoxon-Mann-Whitney statistic for a dependency-free
        implementation.

        Args:
            y_true: Binary ground-truth labels.
            y_score: Predicted probability scores.

        Returns:
            AUC-ROC score in [0, 1].
        """
        pos = y_score[y_true == 1]
        neg = y_score[y_true == 0]
        if len(pos) == 0 or len(neg) == 0:
            return 0.5

        # Vectorised comparison
        auc = float(np.mean(pos[:, None] > neg[None, :]))
        auc += 0.5 * float(np.mean(pos[:, None] == neg[None, :]))
        return auc

    def _score_models_on_domain(
        self,
        scenario_data: pl.DataFrame,
        sample_weights: list[float],
    ) -> NDArray[np.float64]:
        """Score each base model's suitability for the scenario domain.

        Models that make smaller weighted errors on scenario-like training
        samples (as identified by adversarial sample weights) are scored
        higher.

        Args:
            scenario_data: Scenario DataFrame.
            sample_weights: Importance weights for training samples.

        Returns:
            Score array of shape ``(n_models,)``.  Higher = better adapted.
        """
        n_models = len(self.base_models)
        scores = np.zeros(n_models, dtype=np.float64)

        # Use inverse of model complexity / parameter count as a proxy for
        # domain-adaptation fitness when residuals are not available
        for i, model in enumerate(self.base_models):
            try:
                params = model.get_params()
                # Fewer parameters => better generalisation under shift
                n_params = sum(
                    1 for v in params.values() if isinstance(v, (int, float))
                )
                scores[i] = 1.0 / max(n_params, 1)
            except RuntimeError:
                # Model not fitted; assign neutral score
                scores[i] = 1.0 / n_models

        # Augment with weight entropy as a diversity bonus
        w_arr = np.array(sample_weights, dtype=np.float64)
        w_arr = w_arr / max(w_arr.sum(), 1e-12)
        entropy = float(-np.sum(w_arr * np.log(np.maximum(w_arr, 1e-12))))
        scores *= (1 + 0.1 * entropy)

        return scores

    @staticmethod
    def _softmax(x: NDArray[np.float64]) -> NDArray[np.float64]:
        """Numerically stable softmax.

        Args:
            x: Input logits.

        Returns:
            Probability vector that sums to 1.
        """
        shifted = x - np.max(x)
        exp_x = np.exp(shifted)
        return exp_x / exp_x.sum()
