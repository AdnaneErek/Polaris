# src/learning.py
"""
Learning from Outcomes - Self-improving AI system.

Records decisions, actual outcomes, and prediction errors to improve:
- Forecast accuracy (adjust forecast model weights)
- Portfolio scoring (adjust scoring weights based on what worked)
- Recommendation confidence (learn which bundles tend to be accepted)
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


@dataclass
class DecisionRecord:
    """Record of a single decision made by the system."""
    timestamp: str
    as_of: str
    recommended_bundle_id: str
    recommended_bundle_name: str
    recommended_score_stress_cvar10: float
    recommended_score_stress_mean: float
    confidence: float
    guardrails_passed: bool

    # Planned measurement point (for closing the loop)
    planned_evaluation_date: Optional[str] = None  # ISO date (YYYY-MM-DD) when outcomes should be measured
    selected_bundle_id: Optional[str] = None  # What SteerCo actually chose
    selected_bundle_name: Optional[str] = None
    selection_timestamp: Optional[str] = None
    
    # Outcome (filled later)
    accepted: Optional[bool] = None  # Was recommendation accepted?
    actual_outcomes: Optional[Dict[str, float]] = None  # kpi_id -> actual value at evaluation date
    predicted_outcomes: Optional[Dict[str, float]] = None  # kpi_id -> predicted value
    evaluation_date: Optional[str] = None  # When outcomes were measured
    
    # Learning metrics (computed)
    prediction_errors: Optional[Dict[str, float]] = None  # kpi_id -> absolute error
    mean_absolute_error: Optional[float] = None
    score_prediction_error: Optional[float] = None  # How far off was the score prediction?


@dataclass
class LearningMetrics:
    """Aggregated learning metrics."""
    total_decisions: int
    accepted_decisions: int
    mean_forecast_accuracy: float  # 0..1, higher = better
    mean_score_prediction_error: float  # Lower = better
    improvement_trend: str  # "improving", "stable", "degrading"
    last_updated: str


class DecisionLearner:
    """
    Manages learning from historical decisions.
    
    Stores decision records locally (JSON file) and uses them to:
    1. Improve forecast model weights
    2. Adjust portfolio scoring weights
    3. Improve recommendation confidence
    """
    
    def __init__(self, storage_dir: str = "artifacts/learning"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.decisions_file = self.storage_dir / "decisions.jsonl"
        self.metrics_file = self.storage_dir / "metrics.json"
    
    def record_decision(
        self,
        as_of: str,
        recommended_bundle_id: str,
        recommended_bundle_name: str,
        recommended_score_stress_cvar10: float,
        recommended_score_stress_mean: float,
        confidence: float,
        guardrails_passed: bool,
        planned_evaluation_date: Optional[str] = None,
        predicted_outcomes: Optional[Dict[str, float]] = None,
    ) -> DecisionRecord:
        """
        Record a decision made by the system.
        
        Returns a DecisionRecord that should be updated later with outcomes.
        """
        record = DecisionRecord(
            timestamp=datetime.now().isoformat(),
            as_of=as_of,
            recommended_bundle_id=recommended_bundle_id,
            recommended_bundle_name=recommended_bundle_name,
            recommended_score_stress_cvar10=recommended_score_stress_cvar10,
            recommended_score_stress_mean=recommended_score_stress_mean,
            confidence=confidence,
            guardrails_passed=guardrails_passed,
            planned_evaluation_date=planned_evaluation_date,
            predicted_outcomes=predicted_outcomes,
        )
        
        # Append to JSONL file
        with open(self.decisions_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")
        
        return record
    
    def update_decision_outcome(
        self,
        decision_timestamp: str,
        accepted: bool,
        actual_outcomes: Dict[str, float],
        predicted_outcomes: Dict[str, float],
        evaluation_date: str,
    ) -> None:
        """
        Update a decision record with actual outcomes.
        
        This should be called after outcomes are measured (e.g., 3-6 months later).
        """
        # Read all decisions
        decisions = self._load_all_decisions()
        
        # Find the decision to update
        updated = False
        for i, decision in enumerate(decisions):
            if decision["timestamp"] == decision_timestamp:
                decision["accepted"] = accepted
                decision["actual_outcomes"] = actual_outcomes
                decision["predicted_outcomes"] = predicted_outcomes
                decision["evaluation_date"] = evaluation_date
                
                # Compute prediction errors
                errors = {}
                for kpi_id in actual_outcomes.keys():
                    if kpi_id in predicted_outcomes:
                        errors[kpi_id] = abs(actual_outcomes[kpi_id] - predicted_outcomes[kpi_id])
                
                decision["prediction_errors"] = errors
                decision["mean_absolute_error"] = float(np.mean(list(errors.values()))) if errors else None
                
                # Proxy score prediction error from KPI prediction quality.
                # This is not a full portfolio re-score yet, but gives a useful drift signal.
                if errors:
                    decision["score_prediction_error"] = float(np.mean(list(errors.values())))
                
                updated = True
                break
        
        if not updated:
            raise ValueError(f"Decision with timestamp {decision_timestamp} not found")
        
        # Rewrite all decisions
        with open(self.decisions_file, "w", encoding="utf-8") as f:
            for decision in decisions:
                f.write(json.dumps(decision, ensure_ascii=False) + "\n")

    def record_option_selection(
        self,
        decision_timestamp: str,
        selected_bundle_id: str,
        selected_bundle_name: str,
        accepted: bool = True,
    ) -> None:
        """
        Record what option was actually selected by SteerCo (before outcomes are known).
        """
        decisions = self._load_all_decisions()
        updated = False
        now_iso = datetime.now().isoformat()

        for decision in decisions:
            if decision.get("timestamp") == decision_timestamp:
                decision["selected_bundle_id"] = str(selected_bundle_id)
                decision["selected_bundle_name"] = str(selected_bundle_name)
                decision["selection_timestamp"] = now_iso
                decision["accepted"] = bool(accepted)
                updated = True
                break

        if not updated:
            raise ValueError(f"Decision with timestamp {decision_timestamp} not found")

        with open(self.decisions_file, "w", encoding="utf-8") as f:
            for decision in decisions:
                f.write(json.dumps(decision, ensure_ascii=False) + "\n")
    
    def _load_all_decisions(self) -> List[Dict[str, Any]]:
        """Load all decision records from storage."""
        if not self.decisions_file.exists():
            return []
        
        decisions = []
        with open(self.decisions_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    decisions.append(json.loads(line))
        return decisions
    
    def get_learning_metrics(self) -> LearningMetrics:
        """Compute aggregated learning metrics from historical decisions."""
        decisions = self._load_all_decisions()
        
        if not decisions:
            return LearningMetrics(
                total_decisions=0,
                accepted_decisions=0,
                mean_forecast_accuracy=0.0,
                mean_score_prediction_error=0.0,
                improvement_trend="no_data",
                last_updated=datetime.now().isoformat(),
            )
        
        # Filter decisions with outcomes
        decisions_with_outcomes = [d for d in decisions if d.get("evaluation_date") is not None]
        
        if not decisions_with_outcomes:
            return LearningMetrics(
                total_decisions=len(decisions),
                accepted_decisions=sum(1 for d in decisions if d.get("accepted") is True),
                mean_forecast_accuracy=0.0,
                mean_score_prediction_error=0.0,
                improvement_trend="no_outcomes",
                last_updated=datetime.now().isoformat(),
            )
        
        # Compute metrics
        accepted_count = sum(1 for d in decisions_with_outcomes if d.get("accepted") is True)
        
        # Forecast accuracy (inverse of mean absolute error, normalized)
        maes = [d.get("mean_absolute_error", 0.0) for d in decisions_with_outcomes if d.get("mean_absolute_error") is not None]
        if maes:
            # Normalize: assume max error of 10.0 = 0 accuracy, 0 error = 1.0 accuracy
            max_error = 10.0
            normalized_accuracies = [max(0.0, 1.0 - (mae / max_error)) for mae in maes]
            mean_forecast_accuracy = float(np.mean(normalized_accuracies))
        else:
            mean_forecast_accuracy = 0.0
        
        # Score prediction error
        score_errors = [abs(d.get("score_prediction_error", 0.0)) for d in decisions_with_outcomes if d.get("score_prediction_error") is not None]
        mean_score_error = float(np.mean(score_errors)) if score_errors else 0.0
        
        # Improvement trend (compare recent vs older decisions)
        if len(decisions_with_outcomes) >= 4:
            recent = decisions_with_outcomes[-4:]
            older = decisions_with_outcomes[:-4] if len(decisions_with_outcomes) > 4 else []
            
            if older:
                recent_maes = [d.get("mean_absolute_error", 0.0) for d in recent if d.get("mean_absolute_error") is not None]
                older_maes = [d.get("mean_absolute_error", 0.0) for d in older if d.get("mean_absolute_error") is not None]
                
                if recent_maes and older_maes:
                    recent_avg = np.mean(recent_maes)
                    older_avg = np.mean(older_maes)
                    
                    if recent_avg < older_avg * 0.9:  # 10% improvement
                        trend = "improving"
                    elif recent_avg > older_avg * 1.1:  # 10% degradation
                        trend = "degrading"
                    else:
                        trend = "stable"
                else:
                    trend = "stable"
            else:
                trend = "stable"
        else:
            trend = "insufficient_data"
        
        return LearningMetrics(
            total_decisions=len(decisions),
            accepted_decisions=accepted_count,
            mean_forecast_accuracy=mean_forecast_accuracy,
            mean_score_prediction_error=mean_score_error,
            improvement_trend=trend,
            last_updated=datetime.now().isoformat(),
        )
    
    def get_improved_forecast_weights(self) -> Dict[str, float]:
        """
        Return adjusted forecast weights based on prediction errors.
        
        KPIs with lower prediction errors get higher weights.
        """
        decisions = self._load_all_decisions()
        decisions_with_outcomes = [d for d in decisions if d.get("evaluation_date") is not None]
        
        if not decisions_with_outcomes:
            return {}  # No learning yet, use defaults
        
        # Aggregate errors per KPI
        kpi_errors: Dict[str, List[float]] = {}
        for decision in decisions_with_outcomes:
            errors = decision.get("prediction_errors", {})
            for kpi_id, error in errors.items():
                if kpi_id not in kpi_errors:
                    kpi_errors[kpi_id] = []
                kpi_errors[kpi_id].append(error)
        
        # Compute weights: inverse of mean error (normalized)
        weights = {}
        if kpi_errors:
            mean_errors = {kpi_id: np.mean(errors) for kpi_id, errors in kpi_errors.items()}
            total_inverse = sum(1.0 / (err + 0.1) for err in mean_errors.values())  # +0.1 to avoid division by zero
            
            for kpi_id, mean_err in mean_errors.items():
                # Weight = normalized inverse error
                weights[kpi_id] = (1.0 / (mean_err + 0.1)) / total_inverse
        
        return weights

    def get_forecast_recalibration_params(self) -> Dict[str, Dict[str, float]]:
        """
        Estimate simple per-KPI recalibration parameters from historical outcomes.

        Returns:
          kpi_id -> {"bias": additive_shift, "sigma_mult": uncertainty_scaler, "n": n_obs}
        """
        decisions = self._load_all_decisions()
        decisions_with_outcomes = [d for d in decisions if d.get("evaluation_date") is not None]
        if not decisions_with_outcomes:
            return {}

        residuals: Dict[str, List[float]] = {}
        pred_abs: Dict[str, List[float]] = {}
        for d in decisions_with_outcomes:
            actual = d.get("actual_outcomes") or {}
            pred = d.get("predicted_outcomes") or {}
            for kpi_id, a in actual.items():
                if kpi_id not in pred:
                    continue
                try:
                    a_f = float(a)
                    p_f = float(pred[kpi_id])
                except Exception:
                    continue
                residuals.setdefault(str(kpi_id), []).append(a_f - p_f)
                pred_abs.setdefault(str(kpi_id), []).append(abs(p_f))

        out: Dict[str, Dict[str, float]] = {}
        for kpi_id, rs in residuals.items():
            if not rs:
                continue
            arr = np.asarray(rs, dtype=float)
            bias = float(np.mean(arr))
            mad = float(np.mean(np.abs(arr)))
            scale = float(np.mean(pred_abs.get(kpi_id, [1.0])))
            sigma_mult = float(np.clip(1.0 + (mad / max(1e-6, scale)), 0.8, 3.0))
            out[kpi_id] = {"bias": bias, "sigma_mult": sigma_mult, "n": float(len(arr))}
        return out

    def get_confidence_calibration_multiplier(self) -> float:
        """
        Calibrate confidence against realized forecast accuracy.
        Returns multiplicative factor to apply to raw confidence.
        """
        decisions = self._load_all_decisions()
        decisions_with_outcomes = [d for d in decisions if d.get("evaluation_date") is not None]
        if len(decisions_with_outcomes) < 2:
            return 1.0

        confs: List[float] = []
        realized_accs: List[float] = []
        for d in decisions_with_outcomes:
            conf = d.get("confidence")
            mae = d.get("mean_absolute_error")
            actual = d.get("actual_outcomes") or {}
            if conf is None or mae is None:
                continue
            try:
                conf_f = float(conf)
                mae_f = float(mae)
            except Exception:
                continue
            actual_scale_vals = []
            for v in actual.values():
                try:
                    actual_scale_vals.append(abs(float(v)))
                except Exception:
                    continue
            scale = float(np.mean(actual_scale_vals)) if actual_scale_vals else 10.0
            acc = max(0.0, 1.0 - (mae_f / max(1e-6, scale)))
            confs.append(conf_f)
            realized_accs.append(float(acc))

        if not confs or not realized_accs:
            return 1.0
        ratio = float(np.mean(realized_accs) / max(1e-6, np.mean(confs)))
        return float(np.clip(ratio, 0.6, 1.4))
    
    def get_improved_scoring_weights(self) -> Dict[str, float]:
        """
        Return adjusted portfolio scoring weights based on what worked.
        
        Bundles that were accepted and had good outcomes get higher weights.
        """
        decisions = self._load_all_decisions()
        # Prefer accepted decisions; fallback to all evaluated ones if too few.
        decisions_with_outcomes = [d for d in decisions if d.get("evaluation_date") is not None and d.get("accepted") is True]
        if len(decisions_with_outcomes) < 2:
            decisions_with_outcomes = [d for d in decisions if d.get("evaluation_date") is not None]
        if not decisions_with_outcomes:
            return {}

        kpi_errors: Dict[str, List[float]] = {}
        for d in decisions_with_outcomes:
            errs = d.get("prediction_errors") or {}
            for kpi_id, err in errs.items():
                try:
                    e = float(err)
                except Exception:
                    continue
                kpi_errors.setdefault(str(kpi_id), []).append(abs(e))

        if not kpi_errors:
            return {}

        inv: Dict[str, float] = {}
        for kpi_id, vals in kpi_errors.items():
            m = float(np.mean(vals))
            inv[kpi_id] = 1.0 / max(1e-6, m)

        inv_mean = float(np.mean(list(inv.values()))) if inv else 1.0
        if inv_mean <= 0:
            return {}

        # Multipliers around 1.0; lower error => higher weight.
        return {k: float(np.clip(v / inv_mean, 0.5, 1.5)) for k, v in inv.items()}
    
    def format_learning_report(self) -> str:
        """Format a human-readable learning report."""
        metrics = self.get_learning_metrics()
        
        lines = []
        lines.append("LEARNING FROM OUTCOMES")
        lines.append("=" * 80)
        lines.append(f"Total decisions recorded: {metrics.total_decisions}")
        lines.append(f"Decisions with outcomes: {sum(1 for d in self._load_all_decisions() if d.get('evaluation_date') is not None)}")
        lines.append(f"Accepted decisions: {metrics.accepted_decisions}")
        
        if metrics.total_decisions > 0:
            acceptance_rate = metrics.accepted_decisions / metrics.total_decisions
            lines.append(f"Acceptance rate: {acceptance_rate:.1%}")
        
        if metrics.mean_forecast_accuracy > 0:
            lines.append(f"\nForecast Accuracy: {metrics.mean_forecast_accuracy:.1%}")
            lines.append(f"Mean prediction error: {metrics.mean_score_prediction_error:.3f}")
            lines.append(f"Improvement trend: {metrics.improvement_trend}")
        
        if metrics.total_decisions == 0:
            lines.append("\nNo decisions recorded yet. Learning will begin after first decision.")
        elif metrics.improvement_trend == "no_outcomes":
            lines.append("\nDecisions recorded but no outcomes yet. Update decisions with actual outcomes to enable learning.")
        
        lines.append(f"\nLast updated: {metrics.last_updated}")
        
        return "\n".join(lines)


# Global learner instance
_learner: Optional[DecisionLearner] = None
_learner_storage_dir: Optional[str] = None


def get_learner(storage_dir: str = "artifacts/learning") -> DecisionLearner:
    """Get or create the global learner instance."""
    global _learner, _learner_storage_dir
    if _learner is None or _learner_storage_dir != str(storage_dir):
        _learner = DecisionLearner(storage_dir=storage_dir)
        _learner_storage_dir = str(storage_dir)
    return _learner
