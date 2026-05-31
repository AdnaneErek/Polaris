# src/monitor.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime

import pandas as pd

from .schemas import Plan, Direction


def _to_dt(s: str) -> datetime:
    return datetime.fromisoformat(s)


def _latest_kpi_values(df: pd.DataFrame, as_of: str) -> Dict[str, float]:
    """
    df format expected (long):
      date, kpi_id, value
    Returns latest value for each kpi_id at or before as_of.
    """
    as_of_dt = _to_dt(as_of)
    dfx = df.copy()
    dfx["date_dt"] = pd.to_datetime(dfx["date"])
    dfx = dfx[dfx["date_dt"] <= as_of_dt]
    if dfx.empty:
        return {}
    # latest per kpi
    idx = dfx.sort_values("date_dt").groupby("kpi_id")["date_dt"].idxmax()
    latest = dfx.loc[idx, ["kpi_id", "value"]]
    return dict(zip(latest["kpi_id"], latest["value"]))


def _expected_from_checkpoints(checkpoints: List[Tuple[str, float]], as_of: str) -> Optional[float]:
    """
    Piecewise linear interpolation using checkpoints.
    checkpoints: list of (date_iso, expected_value)
    """
    if not checkpoints:
        return None

    pts = sorted([( _to_dt(d), v) for d, v in checkpoints], key=lambda x: x[0])
    t = _to_dt(as_of)

    # before first checkpoint
    if t <= pts[0][0]:
        return float(pts[0][1])
    # after last checkpoint
    if t >= pts[-1][0]:
        return float(pts[-1][1])

    # between
    for (t0, v0), (t1, v1) in zip(pts[:-1], pts[1:]):
        if t0 <= t <= t1:
            # linear interpolate by time fraction
            span = (t1 - t0).days
            if span <= 0:
                return float(v1)
            frac = (t - t0).days / span
            return float(v0 + frac * (v1 - v0))
    return None


def _drift_score(direction: Direction, actual: float, expected: float) -> float:
    """
    Returns normalized drift: positive means "worse than expected".
    For direction 'up': drift = expected - actual
    For direction 'down': drift = actual - expected
    """
    if direction == "up":
        return expected - actual
    return actual - expected


def _confidence_from_drift(drift: float, scale: float) -> float:
    """
    Converts drift magnitude into a confidence in "there is drift".
    Simple mapping: drift/scale -> sigmoid-ish clamp.
    """
    if scale <= 1e-9:
        scale = 1.0
    x = drift / scale
    # Smooth clamp
    # x <=0 => low confidence drift; large positive => high confidence drift
    conf = 0.5 + 0.5 * (x / (1.0 + abs(x)))
    return float(max(0.0, min(1.0, conf)))


@dataclass
class KPIStatus:
    kpi_id: str
    actual: Optional[float]
    expected: Optional[float]
    drift: Optional[float]
    drift_confidence: Optional[float]


@dataclass
class ObjectiveStatus:
    objective_id: str
    weighted_drift: float
    drift_confidence: float
    kpi_statuses: List[KPIStatus]
    on_track: bool


@dataclass
class MonitorSnapshot:
    as_of: str
    objective_statuses: List[ObjectiveStatus]
    triggered_for_steerco: List[str]  # objective IDs requiring attention
    ml_anomalies: Optional[Dict[str, Any]] = None  # kpi_id -> MLAnomalyResult (if ML detection enabled)


def monitor(plan: Plan, kpi_df: pd.DataFrame, as_of: str) -> MonitorSnapshot:
    """
    Main monitoring function.

    kpi_df must contain columns: date (ISO), kpi_id, value.
    """
    latest = _latest_kpi_values(kpi_df, as_of)

    # thresholds
    drift_thr = plan.reporting.steering_committee.decision_thresholds.drift_confidence

    objective_statuses: List[ObjectiveStatus] = []
    triggered: List[str] = []

    # For scaling drift -> confidence we use a heuristic: 10% of target range
    kpi_targets = {k.id: k.target.value for k in plan.kpis}
    kpi_baselines = {k.id: k.baseline.value for k in plan.kpis}

    for obj in plan.objectives:
        kpi_statuses: List[KPIStatus] = []
        drift_list: List[float] = []
        conf_list: List[float] = []

        for okr in obj.okrs:
            actual = latest.get(okr.kpi_id)

            checkpoints = [(cp.date, cp.expected) for cp in okr.trajectory.checkpoints]
            expected = _expected_from_checkpoints(checkpoints, as_of)

            if actual is None or expected is None:
                kpi_statuses.append(KPIStatus(okr.kpi_id, actual, expected, None, None))
                continue

            drift = _drift_score(okr.direction, actual, expected)

            # scale: 10% of baseline-target span (or fallback)
            base = kpi_baselines.get(okr.kpi_id, okr.baseline)
            tgt = kpi_targets.get(okr.kpi_id, okr.target)
            span = abs(tgt - base)
            scale = max(0.05 * max(1.0, abs(tgt)), 0.10 * span)  # robust scaling
            conf = _confidence_from_drift(drift, scale)

            kpi_statuses.append(KPIStatus(okr.kpi_id, actual, expected, drift, conf))
            drift_list.append(drift)
            conf_list.append(conf)

        # objective aggregation
        # weighted drift: average drift over OKRs (if none, 0)
        if drift_list:
            avg_drift = sum(drift_list) / len(drift_list)
            avg_conf = sum(conf_list) / len(conf_list)
        else:
            avg_drift = 0.0
            avg_conf = 0.0

        weighted_drift = obj.weight * avg_drift
        on_track = avg_conf < drift_thr  # if confidence of drift below threshold => on track

        objective_statuses.append(
            ObjectiveStatus(
                objective_id=obj.id,
                weighted_drift=float(weighted_drift),
                drift_confidence=float(avg_conf),
                kpi_statuses=kpi_statuses,
                on_track=on_track,
            )
        )

        if avg_conf >= drift_thr:
            triggered.append(obj.id)
    
    # ML-based anomaly detection (if available)
    ml_anomalies = None
    try:
        from src.monitor_ml import detect_all_kpi_anomalies_ml
        ml_anomalies_dict = detect_all_kpi_anomalies_ml(plan, kpi_df, as_of)
        # Convert to dict for JSON serialization
        ml_anomalies = {
            kpi_id: {
                "is_anomaly": result.is_anomaly,
                "anomaly_score": result.anomaly_score,
                "confidence": result.confidence,
                "explanation": result.explanation,
                "method": result.method,
            }
            for kpi_id, result in ml_anomalies_dict.items()
        }
    except (ImportError, Exception):
        # ML detection not available or failed - continue without it
        ml_anomalies = None

    return MonitorSnapshot(as_of=as_of, objective_statuses=objective_statuses, triggered_for_steerco=triggered)


def snapshot_to_text(plan: Plan, snap: MonitorSnapshot) -> str:
    """
    Human-readable brief (for your 'weekly/monthly AI transformation brief').
    """
    lines = []
    lines.append(f"MONITOR SNAPSHOT — as of {snap.as_of}")
    lines.append("-" * 72)

    for os in snap.objective_statuses:
        obj_name = next((o.name for o in plan.objectives if o.id == os.objective_id), os.objective_id)
        status = "ON TRACK" if os.on_track else "DRIFT"
        lines.append(f"{os.objective_id} | {obj_name} | {status} | drift_conf={os.drift_confidence:.2f}")

        for ks in os.kpi_statuses:
            k = next((kk for kk in plan.kpis if kk.id == ks.kpi_id), None)
            kname = k.short_name if k else ks.kpi_id
            if ks.actual is None or ks.expected is None:
                lines.append(f"  - {kname}: insufficient data")
            else:
                lines.append(f"  - {kname}: actual={ks.actual:.2f} expected={ks.expected:.2f} drift={ks.drift:.2f} conf={ks.drift_confidence:.2f}")

        lines.append("")

    if snap.triggered_for_steerco:
        lines.append("⚠️  Steering Committee Attention Required:")
        for oid in snap.triggered_for_steerco:
            lines.append(f"  - {oid}")
    else:
        lines.append("✅ No objectives exceed drift threshold for Steering Committee.")
    return "\n".join(lines)
