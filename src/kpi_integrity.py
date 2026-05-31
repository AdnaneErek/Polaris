# src/kpi_integrity.py
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta

from src.schemas import Plan


def _dt(s: str) -> datetime:
    return datetime.fromisoformat(str(s))


def _month_floor(d: datetime) -> datetime:
    return d.replace(day=1, hour=0, minute=0, second=0, microsecond=0)


def _months_between(a: str, b: str) -> int:
    da = _dt(a)
    db = _dt(b)
    return (db.year - da.year) * 12 + (db.month - da.month)


def _normalize_direction(x) -> str:
    """
    Robust to:
      - enums (Direction.down)
      - strings ("down", "Direction.down")
      - None (defaults to "up")
    """
    s = str(x).strip().lower()
    if "down" in s:
        return "down"
    if "up" in s:
        return "up"
    return "up"


def _get_kpi_direction_from_plan(plan: Plan) -> Dict[str, str]:
    """
    Canonical source: KPI catalog (plan.kpis[*].target.direction)
    Fallback: OKRs (plan.objectives[*].okrs[*].direction)
    """
    out: Dict[str, str] = {}

    # 1) KPI catalog direction (preferred)
    for k in plan.kpis:
        try:
            out[k.id] = _normalize_direction(k.target.direction)
        except Exception:
            # keep going
            pass

    # 2) Fallback from OKRs if missing
    for obj in plan.objectives:
        for okr in obj.okrs:
            if okr.kpi_id not in out:
                out[okr.kpi_id] = _normalize_direction(okr.direction)

    return out


def _get_kpi_target_deadline(plan: Plan) -> Dict[str, Tuple[float, str]]:
    out: Dict[str, Tuple[float, str]] = {}
    for k in plan.kpis:
        out[k.id] = (float(k.target.value), str(k.target.deadline))
    return out


def _expected_from_plan(plan: Plan, kpi_id: str, date_iso: str) -> Optional[float]:
    pts: List[Tuple[datetime, float]] = []
    for obj in plan.objectives:
        for okr in obj.okrs:
            if okr.kpi_id != kpi_id:
                continue
            for cp in okr.trajectory.checkpoints:
                pts.append((_dt(cp.date), float(cp.expected)))

    if not pts:
        return None

    pts.sort(key=lambda x: x[0])
    t = _dt(date_iso)

    if t <= pts[0][0]:
        return pts[0][1]
    if t >= pts[-1][0]:
        return pts[-1][1]

    for (t0, v0), (t1, v1) in zip(pts[:-1], pts[1:]):
        if t0 <= t <= t1:
            span = max(1, (t1 - t0).days)
            frac = (t - t0).days / span
            return v0 + frac * (v1 - v0)
    return pts[-1][1]


def _safe_float(x) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


@dataclass(frozen=True)
class IntegrityResult:
    kpi_id: str
    score: float          # 0..1 (higher = more suspicious)
    flags: List[str]
    diagnostics: Dict[str, float]


def compute_kpi_integrity(
    plan: Plan,
    kpis_df: pd.DataFrame,
    as_of: str,
    lookback_months: int = 12,
    stale_after_months: int = 2,
) -> Dict[str, IntegrityResult]:
    df = kpis_df.copy()
    if not {"date", "kpi_id", "value"}.issubset(df.columns):
        raise ValueError("kpis_df must contain columns: date, kpi_id, value")

    df["date"] = df["date"].astype(str)
    df["kpi_id"] = df["kpi_id"].astype(str)
    df["value"] = df["value"].apply(_safe_float)

    start_dt = _month_floor(_dt(as_of) - relativedelta(months=lookback_months)).date().isoformat()
    df = df[(df["date"] <= as_of) & (df["date"] >= start_dt)].sort_values(["kpi_id", "date"])

    direction = _get_kpi_direction_from_plan(plan)
    target_deadline = _get_kpi_target_deadline(plan)

    results: Dict[str, IntegrityResult] = {}

    for kpi_id, g in df.groupby("kpi_id"):
        g = g.sort_values("date")
        vals = g["value"].astype(float).values
        dates = g["date"].astype(str).values

        finite_mask = np.isfinite(vals)
        v = vals[finite_mask]
        dts = dates[finite_mask]

        if len(v) == 0:
            results[kpi_id] = IntegrityResult(
                kpi_id=kpi_id, score=1.0, flags=["missing_all_values"], diagnostics={"n": 0}
            )
            continue

        latest = float(v[-1])
        latest_date = str(dts[-1])

        dirn = direction.get(kpi_id, "up")
        tgt, deadline = target_deadline.get(kpi_id, (np.nan, "2099-12-31"))
        months_to_deadline = _months_between(latest_date, deadline)

        # --- staleness ---
        months_stale = _months_between(latest_date, as_of)
        is_stale = months_stale > stale_after_months

        expected = _expected_from_plan(plan, kpi_id, latest_date)
        if expected is None:
            expected = latest

        std = float(np.std(v)) if len(v) >= 2 else 0.0
        diffs = np.diff(v) if len(v) >= 2 else np.array([])
        max_jump = float(np.max(np.abs(diffs))) if diffs.size else 0.0

        flags: List[str] = []
        score_parts: List[float] = []
        diag: Dict[str, float] = {
            "n": float(len(v)),
            "latest": float(latest),
            "latest_date": float(_months_between("2000-01-01", latest_date)),
            "as_of_staleness_months": float(months_stale),
            "expected": float(expected),
            "std": float(std),
            "max_jump": float(max_jump),
            "months_to_deadline": float(months_to_deadline),
            "direction_is_up": 1.0 if dirn == "up" else 0.0,
            "target": float(tgt) if np.isfinite(tgt) else float("nan"),
        }

        if is_stale:
            flags.append(f"stale_data_last_obs_{months_stale}_months_before_as_of")
            # Stale evidence => do not let integrity go to 1.0 just from trajectory comparison
            score_parts.append(0.25)

        # Goodness gap relative to trajectory (based on LAST OBSERVED DATE)
        if dirn == "up":
            gap = latest - float(expected)
        else:
            gap = float(expected) - latest
        diag["goodness_gap_vs_expected"] = float(gap)

        # S1) Too-good-too-early (but dampened if stale)
        if months_to_deadline >= 6:
            thr = 3.0 if kpi_id == "KPI_STP" else (0.2 if kpi_id == "KPI_E2E" else 2.0)
            if gap > thr:
                flags.append("too_good_too_early_vs_trajectory")
                raw = min(1.0, gap / (thr * 2.0))
                if is_stale:
                    raw *= 0.4  # dampen because we lack recent evidence
                score_parts.append(raw)

        # S2) Target reached very early (direction-aware)
        if np.isfinite(tgt) and months_to_deadline >= 6:
            if dirn == "up" and latest >= float(tgt):
                flags.append("target_reached_very_early")
                score_parts.append(0.7)
            if dirn == "down" and latest <= float(tgt):
                flags.append("target_reached_very_early")
                score_parts.append(0.7)

        # S3) Clamping / saturation for % KPI
        if kpi_id == "KPI_STP":
            near_100 = float(np.mean(v >= 99.5))
            if near_100 >= 0.6 and std < 0.5:
                flags.append("possible_clamping_near_100")
                score_parts.append(0.8)

        # S4) Sudden shift
        jump_thr = 5.0 if kpi_id == "KPI_STP" else (0.6 if kpi_id == "KPI_E2E" else 10.0)
        if max_jump > jump_thr:
            flags.append("sudden_level_shift_or_pipeline_change")
            score_parts.append(min(1.0, max_jump / (jump_thr * 2.0)))

        # S5) High variability heuristics
        if kpi_id == "KPI_STP" and std > 3.0:
            flags.append("high_variability_for_percentage_kpi")
            score_parts.append(0.5)
        if kpi_id == "KPI_E2E" and std > 0.6:
            flags.append("high_variability_for_cycle_time_kpi")
            score_parts.append(0.4)
        if kpi_id == "KPI_INC" and std > 8.0:
            flags.append("high_variability_for_incident_kpi")
            score_parts.append(0.4)

        # Aggregate score: soft OR
        if score_parts:
            s = 1.0
            for p in score_parts:
                s *= (1.0 - float(p))
            score = 1.0 - s
        else:
            score = 0.0

        results[kpi_id] = IntegrityResult(
            kpi_id=kpi_id,
            score=float(max(0.0, min(1.0, score))),
            flags=flags,
            diagnostics=diag,
        )

    # Ensure plan KPIs always appear
    for k in plan.kpis:
        if k.id not in results:
            results[k.id] = IntegrityResult(kpi_id=k.id, score=0.5, flags=["no_recent_data"], diagnostics={"n": 0})

    return results


def format_integrity_report(results: Dict[str, IntegrityResult], threshold: float = 0.6) -> str:
    lines = []
    lines.append("KPI INTEGRITY — signals (higher score = more suspicious)")
    lines.append("-" * 72)
    for kpi_id in sorted(results.keys()):
        r = results[kpi_id]
        badge = "⚠️" if (r.score >= threshold or r.flags) else "OK"
        flags = "; ".join(r.flags) if r.flags else "none"
        lines.append(f"{badge} {kpi_id}: score={r.score:.2f} | flags: {flags}")
    return "\n".join(lines)


def integrity_flags_for_agent(results: Dict[str, IntegrityResult], threshold: float = 0.6) -> List[str]:
    flags: List[str] = []
    for kpi_id, r in results.items():
        if r.score >= threshold or r.flags:
            flags.append(f"{kpi_id}: score={r.score:.2f} flags={','.join(r.flags) if r.flags else 'none'}")
    return flags
