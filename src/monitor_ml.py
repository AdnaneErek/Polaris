# src/monitor_ml.py
"""
ML-based anomaly detection for KPI monitoring.

Uses Isolation Forest (unsupervised ML) to detect unusual patterns in KPI trajectories.
Free, no API costs - uses scikit-learn.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Any

import numpy as np
import pandas as pd

try:
    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


@dataclass
class MLAnomalyResult:
    """Result of ML anomaly detection."""
    kpi_id: str
    is_anomaly: bool
    anomaly_score: float  # Lower = more anomalous (Isolation Forest decision function)
    confidence: float  # 0..1, higher = more confident
    explanation: str
    method: str = "Isolation Forest (ML)"
    features_used: Optional[List[str]] = None


def _extract_time_series_features(
    values: np.ndarray,
    window_size: int = 5,
) -> np.ndarray:
    """
    Extract features from time series for anomaly detection.
    
    Features:
    - Mean of window (level)
    - Std of window (volatility)
    - Recent change (first derivative)
    - Trend (mean of differences)
    - Acceleration (second derivative, if window >= 3)
    """
    if len(values) < window_size + 1:
        # Not enough data - return empty features
        return np.array([]).reshape(0, 4)
    
    features = []
    for i in range(window_size, len(values)):
        window = values[i - window_size:i]
        
        mean = float(np.mean(window))
        std = float(np.std(window)) if len(window) > 1 else 0.0
        recent_change = float(values[i] - values[i - 1]) if i > 0 else 0.0
        trend = float(np.mean(np.diff(window))) if len(window) > 1 else 0.0
        
        features.append([mean, std, recent_change, trend])
    
    return np.array(features, dtype=float)


def detect_kpi_anomalies_ml(
    kpi_history: pd.DataFrame,
    kpi_id: str,
    as_of: str,
    lookback_months: int = 12,
    window_size: int = 5,
    contamination: float = 0.1,
    min_data_points: int = 10,
) -> MLAnomalyResult:
    """
    Use ML (Isolation Forest) to detect anomalies in KPI trajectories.
    
    Args:
        kpi_history: DataFrame with columns: date, kpi_id, value
        kpi_id: KPI to analyze
        as_of: Current date (ISO format)
        lookback_months: How many months of history to use
        window_size: Size of rolling window for feature extraction
        contamination: Expected proportion of anomalies (0.1 = 10%)
        min_data_points: Minimum data points required
    
    Returns:
        MLAnomalyResult with anomaly detection results
    """
    if not SKLEARN_AVAILABLE:
        return MLAnomalyResult(
            kpi_id=kpi_id,
            is_anomaly=False,
            anomaly_score=0.0,
            confidence=0.0,
            explanation="ML anomaly detection unavailable: scikit-learn not installed",
            method="Not available",
        )
    
    # Filter to relevant KPI and time window
    kpi_data = kpi_history[kpi_history["kpi_id"] == kpi_id].copy()
    if kpi_data.empty:
        return MLAnomalyResult(
            kpi_id=kpi_id,
            is_anomaly=False,
            anomaly_score=0.0,
            confidence=0.0,
            explanation="No data available for this KPI",
            method="Isolation Forest (ML)",
        )
    
    # Filter by date
    kpi_data = kpi_data[kpi_data["date"] <= as_of].sort_values("date")
    
    if len(kpi_data) < min_data_points:
        return MLAnomalyResult(
            kpi_id=kpi_id,
            is_anomaly=False,
            anomaly_score=0.0,
            confidence=0.0,
            explanation=f"Insufficient data: {len(kpi_data)} points (need {min_data_points})",
            method="Isolation Forest (ML)",
        )
    
    # Extract values
    values = kpi_data["value"].astype(float).values
    
    # Remove NaN/Inf
    finite_mask = np.isfinite(values)
    if not np.all(finite_mask):
        values = values[finite_mask]
        if len(values) < min_data_points:
            return MLAnomalyResult(
                kpi_id=kpi_id,
                is_anomaly=False,
                anomaly_score=0.0,
                confidence=0.0,
                explanation="Too many missing/invalid values",
                method="Isolation Forest (ML)",
            )
    
    # Extract features
    features = _extract_time_series_features(values, window_size=window_size)
    
    if len(features) < 3:
        return MLAnomalyResult(
            kpi_id=kpi_id,
            is_anomaly=False,
            anomaly_score=0.0,
            confidence=0.0,
            explanation="Insufficient data for feature extraction",
            method="Isolation Forest (ML)",
        )
    
    # Split into train (historical "normal") and test (recent data)
    train_size = max(3, int(len(features) * 0.8))  # Use 80% for training
    X_train = features[:train_size]
    X_test = features[train_size:]
    
    if len(X_test) == 0:
        # All data used for training - use last few points as test
        X_test = features[-3:]
        X_train = features[:-3]
    
    if len(X_train) < 3:
        return MLAnomalyResult(
            kpi_id=kpi_id,
            is_anomaly=False,
            anomaly_score=0.0,
            confidence=0.0,
            explanation="Insufficient training data",
            method="Isolation Forest (ML)",
        )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train Isolation Forest
    model = IsolationForest(
        contamination=contamination,
        random_state=42,
        n_estimators=100,
    )
    model.fit(X_train_scaled)
    
    # Predict on recent data
    predictions = model.predict(X_test_scaled)  # -1 = anomaly, 1 = normal
    anomaly_scores = model.decision_function(X_test_scaled)  # Lower = more anomalous
    
    # Determine if anomaly detected
    is_anomaly = any(predictions == -1)
    avg_score = float(np.mean(anomaly_scores))
    min_score = float(np.min(anomaly_scores))
    
    # Confidence: based on how anomalous the worst point is
    # Isolation Forest scores: negative = anomaly, positive = normal
    # More negative = more anomalous
    confidence = abs(min_score) if min_score < 0 else 0.0
    confidence = min(1.0, confidence)  # Cap at 1.0
    
    # Build explanation
    if is_anomaly:
        explanation = (
            f"ML detected unusual pattern in {kpi_id} trajectory. "
            f"Isolation Forest flagged {sum(predictions == -1)}/{len(predictions)} recent points as anomalous "
            f"(anomaly score: {min_score:.2f}). "
            f"Pattern suggests deviation from historical normal behavior."
        )
    else:
        explanation = (
            f"ML analysis: {kpi_id} trajectory appears normal. "
            f"Recent patterns consistent with historical behavior "
            f"(anomaly score: {avg_score:.2f})."
        )
    
    return MLAnomalyResult(
        kpi_id=kpi_id,
        is_anomaly=is_anomaly,
        anomaly_score=min_score,
        confidence=confidence,
        explanation=explanation,
        method="Isolation Forest (ML)",
        features_used=["mean", "volatility", "recent_change", "trend"],
    )


def detect_all_kpi_anomalies_ml(
    plan,
    kpi_history: pd.DataFrame,
    as_of: str,
    lookback_months: int = 12,
) -> Dict[str, MLAnomalyResult]:
    """
    Detect ML anomalies for all KPIs in the plan.
    
    Returns:
        Dict mapping kpi_id -> MLAnomalyResult
    """
    results = {}
    
    for kpi in plan.kpis:
        result = detect_kpi_anomalies_ml(
            kpi_history=kpi_history,
            kpi_id=kpi.id,
            as_of=as_of,
            lookback_months=lookback_months,
        )
        results[kpi.id] = result
    
    return results
