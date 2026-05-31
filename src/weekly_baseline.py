from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


def _to_dt(s: str) -> datetime:
    return datetime.fromisoformat(str(s))


def _interp_checkpoints(checkpoints: List[Tuple[str, float]], date_iso: str) -> Optional[float]:
    if not checkpoints:
        return None
    pts = sorted([(_to_dt(d), float(v)) for d, v in checkpoints], key=lambda x: x[0])
    t = _to_dt(date_iso)
    if t <= pts[0][0]:
        return float(pts[0][1])
    if t >= pts[-1][0]:
        return float(pts[-1][1])
    for (t0, v0), (t1, v1) in zip(pts[:-1], pts[1:]):
        if t0 <= t <= t1:
            span = max(1, (t1 - t0).days)
            frac = (t - t0).days / span
            return float(v0 + frac * (v1 - v0))
    return float(pts[-1][1])


def _get_kpi_direction_map(plan: Any) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for obj in plan.objectives:
        for okr in obj.okrs:
            if okr.kpi_id not in out:
                out[okr.kpi_id] = str(okr.direction)
    return out


def build_weekly_objectives(
    plan: Any,
    start_date: str = "2026-01-01",
    end_date: str = "2028-12-31",
) -> pd.DataFrame:
    """
    Create weekly KPI objective baseline from plan trajectories.
    Output columns: date, kpi_id, baseline_expected, direction
    """
    dates = pd.date_range(start=start_date, end=end_date, freq="W-MON")
    if len(dates) == 0 or dates[0].date().isoformat() != start_date:
        dates = pd.DatetimeIndex([pd.Timestamp(start_date)]).append(dates)
    if dates[-1].date().isoformat() != end_date:
        dates = dates.append(pd.DatetimeIndex([pd.Timestamp(end_date)]))
    dates = pd.DatetimeIndex(sorted(set(dates)))

    # collect weighted trajectories per KPI (same approach as simulator)
    okrs_by_kpi: Dict[str, List[Tuple[float, List[Tuple[str, float]]]]] = {}
    for obj in plan.objectives:
        for okr in obj.okrs:
            cps = [(cp.date, float(cp.expected)) for cp in okr.trajectory.checkpoints]
            okrs_by_kpi.setdefault(okr.kpi_id, []).append((float(obj.weight), cps))

    direction_map = _get_kpi_direction_map(plan)
    rows: List[Dict[str, Any]] = []
    for d in dates:
        ds = d.date().isoformat()
        for kpi_id, trajs in okrs_by_kpi.items():
            wsum = sum(w for w, _ in trajs) or 1.0
            val = 0.0
            for w, cps in trajs:
                ev = _interp_checkpoints(cps, ds)
                if ev is None:
                    continue
                val += (w / wsum) * float(ev)
            rows.append(
                {
                    "date": ds,
                    "kpi_id": kpi_id,
                    "baseline_expected": float(val),
                    "direction": direction_map.get(kpi_id, "up"),
                }
            )
    return pd.DataFrame(rows).sort_values(["kpi_id", "date"]).reset_index(drop=True)


def _interp_from_weekly(weekly_df: pd.DataFrame, kpi_id: str, target_date: str) -> Optional[float]:
    g = weekly_df[weekly_df["kpi_id"] == kpi_id].copy()
    if g.empty:
        return None
    g["ts"] = pd.to_datetime(g["date"]).astype("int64")
    target_ts = pd.Timestamp(target_date).value
    xs = g["ts"].to_numpy(dtype=np.int64)
    ys = g["baseline_expected"].to_numpy(dtype=float)
    order = np.argsort(xs)
    xs = xs[order]
    ys = ys[order]
    if target_ts <= xs[0]:
        return float(ys[0])
    if target_ts >= xs[-1]:
        return float(ys[-1])
    return float(np.interp(target_ts, xs, ys))


def compute_forecast_deviation_alerts(
    plan: Any,
    history_df: pd.DataFrame,
    forecast_df: pd.DataFrame,
    weekly_baseline_df: pd.DataFrame,
    as_of: str,
    lookback_months: int = 24,
    warn_z: float = 1.0,
    alert_z: float = 1.5,
    critical_z: float = 2.0,
) -> Dict[str, Dict[str, Any]]:
    """
    Compares forecast against weekly baseline and emits direction-aware deviation risk alerts.
    """
    if forecast_df.empty:
        return {}

    direction_map = _get_kpi_direction_map(plan)
    hist = history_df.copy()
    hist["date"] = pd.to_datetime(hist["date"])
    as_of_ts = pd.Timestamp(as_of)
    lb = as_of_ts - pd.DateOffset(months=lookback_months)
    hist = hist[(hist["date"] <= as_of_ts) & (hist["date"] >= lb)].copy()

    # compute per-kpi residual sigma using actual vs baseline (history-only, no leakage)
    sigma_map: Dict[str, float] = {}
    for kpi_id, g in hist.groupby("kpi_id"):
        vals: List[float] = []
        for _, r in g.iterrows():
            b = _interp_from_weekly(weekly_baseline_df, str(kpi_id), r["date"].date().isoformat())
            if b is None:
                continue
            vals.append(float(r["value"]) - float(b))
        if vals:
            sigma = float(np.std(vals))
            sigma_map[str(kpi_id)] = max(1e-6, sigma)
        else:
            sigma_map[str(kpi_id)] = 1.0

    out: Dict[str, Dict[str, Any]] = {}
    for kpi_id, g in forecast_df.groupby("kpi_id"):
        direction = direction_map.get(str(kpi_id), "up")
        sigma = sigma_map.get(str(kpi_id), 1.0)
        items: List[Dict[str, Any]] = []
        max_z = 0.0
        max_item: Optional[Dict[str, Any]] = None
        for _, r in g.iterrows():
            d = str(r["date"])
            fc = float(r["forecast"])
            b = _interp_from_weekly(weekly_baseline_df, str(kpi_id), d)
            if b is None:
                continue
            # direction-aware "worse than baseline"
            worse_gap = (b - fc) if direction == "up" else (fc - b)
            z = float(worse_gap / sigma)
            severity = "none"
            if z >= critical_z:
                severity = "critical"
            elif z >= alert_z:
                severity = "alert"
            elif z >= warn_z:
                severity = "warning"
            row = {
                "date": d,
                "forecast": fc,
                "baseline": float(b),
                "worse_gap": float(worse_gap),
                "z_score": z,
                "severity": severity,
                "direction": direction,
            }
            items.append(row)
            if z > max_z:
                max_z = z
                max_item = row

        has_alert = any(x["severity"] in ("warning", "alert", "critical") for x in items)
        if has_alert and max_item is not None:
            out[str(kpi_id)] = {
                "is_deviation_alert": True,
                "max_z_score": float(max_z),
                "max_severity": str(max_item["severity"]),
                "explanation": (
                    f"Forecast deviates from weekly objective baseline (max {max_item['severity']} at {max_item['date']}: "
                    f"forecast={max_item['forecast']:.2f}, baseline={max_item['baseline']:.2f}, z={max_item['z_score']:.2f})."
                ),
                "details": items,
            }
        else:
            out[str(kpi_id)] = {
                "is_deviation_alert": False,
                "max_z_score": float(max_z),
                "max_severity": "none",
                "explanation": "Forecast is within weekly objective baseline tolerance.",
                "details": items,
            }

    return out

