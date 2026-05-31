# src/whatif.py
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Tuple, Optional

import math
import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta

from .schemas import Plan, Direction
from .actions import SteeringAction, apply_actions_to_initiatives


# ----------------------------
# KPI id augmentation (robust join between plan + csv ids)
# ----------------------------
def _augment_kpi_ids(kpis_df: pd.DataFrame) -> pd.DataFrame:
    if kpis_df is None or kpis_df.empty:
        return kpis_df
    df = kpis_df.copy()
    df["kpi_id"] = df["kpi_id"].astype(str)

    pairs = [
        ("KPI_STP", "STP_rate"),
        ("KPI_E2E", "processing_time"),
        ("KPI_INC", "incident_count"),
    ]

    extra = []
    for a, b in pairs:
        mask_a = df["kpi_id"] == a
        if mask_a.any():
            tmp = df.loc[mask_a].copy()
            tmp["kpi_id"] = b
            extra.append(tmp)

        mask_b = df["kpi_id"] == b
        if mask_b.any():
            tmp = df.loc[mask_b].copy()
            tmp["kpi_id"] = a
            extra.append(tmp)

    if extra:
        df = pd.concat([df] + extra, ignore_index=True)
    return df


# ----------------------------
# Time helpers
# ----------------------------
def _dt(s: str) -> datetime:
    return datetime.fromisoformat(s)


def _month_start(s: str) -> str:
    return _dt(s).date().replace(day=1).isoformat()


def month_range(start: str, end: str) -> List[str]:
    cur = _dt(start).date().replace(day=1)
    end_dt = _dt(end).date().replace(day=1)
    out: List[str] = []
    while cur <= end_dt:
        out.append(cur.isoformat())
        cur = (datetime.fromisoformat(cur.isoformat()) + relativedelta(months=1)).date().replace(day=1)
    return out


def expected_from_checkpoints(checkpoints: List[Tuple[str, float]], as_of: str) -> Optional[float]:
    if not checkpoints:
        return None
    pts = sorted([(datetime.fromisoformat(_month_start(d)), float(v)) for d, v in checkpoints], key=lambda t: t[0])
    t = datetime.fromisoformat(_month_start(as_of))

    if t <= pts[0][0]:
        return pts[0][1]
    if t >= pts[-1][0]:
        return pts[-1][1]

    for (t0, v0), (t1, v1) in zip(pts[:-1], pts[1:]):
        if t0 <= t <= t1:
            span = (t1 - t0).days
            if span <= 0:
                return v1
            frac = (t - t0).days / span
            return v0 + frac * (v1 - v0)
    return pts[-1][1]


def drift(direction: Direction, actual: float, expected: float) -> float:
    # positive drift means "worse than expected"
    if direction == "up":
        return expected - actual
    return actual - expected


# ----------------------------
# Soft clipping (neutral inside bounds)
# ----------------------------
def soft_cap(x: float, cap: float, softness: float = 1.0) -> float:
    if x <= cap:
        return x
    t = (x - cap) / max(1e-9, softness)
    return cap - max(1e-12, softness) * (1.0 - math.exp(-t))


def soft_floor(x: float, floor: float, softness: float = 1.0) -> float:
    if x >= floor:
        return x
    t = (floor - x) / max(1e-9, softness)
    return floor + max(1e-12, softness) * (1.0 - math.exp(-t))


# ----------------------------
# Stress events (applied DURING projection)
# ----------------------------
@dataclass(frozen=True)
class StressEvent:
    """
    type:
      - "initiative_progress_drop": subtract magnitude from initiative progress during [start,end]
      - "kpi_shock": add magnitude to KPI value during [start,end]
    """
    id: str
    type: str
    target_id: str
    start: str
    end: str
    magnitude: float
    description: str = ""


def _stress_active(ev: StressEvent, date_iso: str) -> bool:
    s = _month_start(ev.start)
    e = _month_start(ev.end)
    d = _month_start(date_iso)
    return s <= d <= e


# ----------------------------
# What-if config (deterministic)
# ----------------------------
@dataclass
class WhatIfConfig:
    hard_dependency_block: float = 0.50
    dep_block_threshold: float = 0.60
    inertia: float = 0.85
    base_speed: float = 0.10
    trace_attribution: bool = False  # If True, collect attribution drivers


# ----------------------------
# Stochastic config (AI-heavy Path 3)
# ----------------------------
@dataclass(frozen=True)
class WhatIfStochasticConfig:
    n_samples: int = 600
    seed: int = 7

    # KPI noise per month (measurement + residual model error)
    kpi_noise_frac: float = 0.015

    # Initiative progress uncertainty
    progress_noise_sigma: float = 0.03
    stress_progress_sigma_mult: float = 1.5

    # Parameter uncertainty (model risk)
    inertia_sigma: float = 0.03
    hard_dep_block_sigma: float = 0.10

    # If integrity score says “suspicious”, inflate KPI noise
    integrity_noise_mult: float = 1.8

    clamp_progress_0_1: bool = True


# ----------------------------
# Data extraction
# ----------------------------
def _latest_kpi_values(kpi_df: pd.DataFrame, as_of: str) -> Dict[str, float]:
    df = kpi_df.copy()
    df["date"] = df["date"].astype(str).map(_month_start)
    df = df[df["date"] <= _month_start(as_of)]
    if df.empty:
        return {}
    df = df.sort_values("date")
    latest = df.groupby("kpi_id").tail(1)
    return dict(zip(latest["kpi_id"], latest["value"]))


def _latest_initiative_progress(init_df: pd.DataFrame, as_of: str) -> Dict[str, float]:
    df = init_df.copy()
    df["date"] = df["date"].astype(str).map(_month_start)
    df = df[df["date"] <= _month_start(as_of)]
    if df.empty:
        return {}
    df = df.sort_values("date")
    latest = df.groupby("initiative_id").tail(1)
    return dict(zip(latest["initiative_id"], latest["progress"]))


def _initiative_dependencies(plan: Plan) -> Dict[str, List[Tuple[str, str]]]:
    out: Dict[str, List[Tuple[str, str]]] = {}
    for it in plan.initiatives:
        out[it.id] = [(dep.type, dep.initiative_id) for dep in it.dependencies]
    return out


def _initiative_impacts(plan: Plan) -> List[Dict]:
    impacts: List[Dict] = []
    for it in plan.initiatives:
        for imp in it.kpi_impacts:
            impacts.append(
                {
                    "initiative_id": it.id,
                    "kpi_id": imp.kpi_id,
                    "delta_end": float(imp.expected_delta_by_end),
                    "confidence": float(imp.confidence),
                    "lag_months": int(imp.lag_months),
                }
            )
    return impacts



def _okr_catalog(plan: Plan) -> Dict[str, Dict]:
    """
    Build OKR lookup keyed by:
      - canonical KPI id (e.g., KPI_STP)
      - KPI short_name (e.g., STP_rate) when present

    Values:
      { "direction": "up"|"down", "checkpoints": [(date_iso, expected), ...] }
    """
    out: Dict[str, Dict] = {}

    # Build alias map from plan.kpis: short_name -> id
    short_to_id: Dict[str, str] = {}
    for k in getattr(plan, "kpis", []) or []:
        kid = str(getattr(k, "id", ""))
        sn = str(getattr(k, "short_name", "") or "")
        if kid and sn:
            short_to_id[sn] = kid

    for obj in getattr(plan, "objectives", []) or []:
        for okr in getattr(obj, "okrs", []) or []:
            kpi_id = str(getattr(okr, "kpi_id", ""))

            cps = []
            traj = getattr(okr, "trajectory", None)
            if traj is not None:
                for cp in getattr(traj, "checkpoints", []) or []:
                    cps.append((str(getattr(cp, "date", "")), float(getattr(cp, "expected", 0.0))))

            entry = {
                # normalize: ALWAYS a string
                "direction": str(getattr(okr, "direction", "")),
                "checkpoints": cps,
            }

            if kpi_id:
                out[kpi_id] = entry

            # Add alias key via plan.kpis.short_name if it exists
            # (this avoids maintaining a hardcoded alias list forever)
            for sn, kid in short_to_id.items():
                if kid == kpi_id:
                    out[sn] = entry

    return out

    out: Dict[str, Dict] = {}
    for obj in plan.objectives:
        for okr in obj.okrs:
            cps = [(cp.date, float(cp.expected)) for cp in okr.trajectory.checkpoints]
            out[str(okr.kpi_id)] = {"direction": okr.direction, "checkpoints": cps}

    # --- PATCH: add KPI id aliases (robust direction/checkpoints lookup) ---
    aliases = [
        ("KPI_STP", "STP_rate"),
        ("KPI_E2E", "processing_time"),
        ("KPI_INC", "incident_count"),
    ]
    for a, b in aliases:
        if a in out and b not in out:
            out[b] = out[a]
        if b in out and a not in out:
            out[a] = out[b]

    return out



# ----------------------------
# Actions (dependency blocking reduction only here)
# ----------------------------
@dataclass
class ActionEffects:
    dep_block_reduction: Dict[str, float]


def compile_action_effects(actions: List[SteeringAction]) -> ActionEffects:
    dep_block_reduction: Dict[str, float] = {}
    for a in actions:
        if a.type == "split_scope":
            reduction = a.get("reduction", 0.5)
            dep_block_reduction[a.target_initiative] = max(dep_block_reduction.get(a.target_initiative, 0.0), reduction)
    return ActionEffects(dep_block_reduction=dep_block_reduction)


# ----------------------------
# Projection engine (deterministic)
# ----------------------------
def project_kpis(
    plan: Plan,
    kpi_history: pd.DataFrame,
    initiative_history: pd.DataFrame,
    as_of: str,
    horizon_end: str,
    actions: Optional[List[SteeringAction]] = None,
    stress_events: Optional[List[StressEvent]] = None,
    cfg: Optional[WhatIfConfig] = None,
) -> pd.DataFrame:
    cfg = cfg or WhatIfConfig()
    actions = actions or []
    stress_events = stress_events or []

    as_of_m = _month_start(as_of)
    horizon_end_m = _month_start(horizon_end)

    # robust KPI ids
    kpi_history = _augment_kpi_ids(kpi_history)

    # actions on initiatives
    initiative_history = initiative_history.copy()
    initiative_history["date"] = initiative_history["date"].astype(str).map(_month_start)

    initiative_history_adj, impact_scalers = apply_actions_to_initiatives(
        initiatives_df=initiative_history,
        as_of=as_of_m,
        actions=actions,
    )

    effects = compile_action_effects(actions)

    dates = month_range(as_of_m, horizon_end_m)
    okr = _okr_catalog(plan)

    current = _latest_kpi_values(kpi_history, as_of_m)
    if not current:
        current = {k.id: float(k.baseline.value) for k in plan.kpis}

    prog0 = _latest_initiative_progress(initiative_history_adj, as_of_m)
    deps = _initiative_dependencies(plan)
    impacts = _initiative_impacts(plan)

    progress_by_date: Dict[Tuple[str, str], float] = {}
    if initiative_history_adj is not None and not initiative_history_adj.empty:
        tmp = initiative_history_adj.copy()
        tmp = tmp[tmp["date"].isin(dates)].sort_values(["initiative_id", "date"])
        for _, r in tmp.iterrows():
            progress_by_date[(str(r["initiative_id"]), str(r["date"]))] = float(r["progress"])

    proj_progress: Dict[str, float] = {it.id: float(prog0.get(it.id, 0.0)) for it in plan.initiatives}
    progress_history: Dict[str, List[float]] = {it.id: [] for it in plan.initiatives}
    rows: List[Dict] = []

    for d in dates:
        # update initiative progress
        for it in plan.initiatives:
            key = (it.id, d)
            if key in progress_by_date:
                proj_progress[it.id] = float(progress_by_date[key])

            for ev in stress_events:
                if ev.type == "initiative_progress_drop" and ev.target_id == it.id and _stress_active(ev, d):
                    proj_progress[it.id] = max(0.0, min(1.0, proj_progress[it.id] - float(ev.magnitude)))

            progress_history[it.id].append(proj_progress[it.id])

        # project KPI
        for k in plan.kpis:
            kpi_id = k.id
            cps = okr.get(kpi_id, {}).get("checkpoints", [])
            expected = expected_from_checkpoints(cps, d)
            if expected is None:
                expected = float(k.baseline.value)

            prev = float(current.get(kpi_id, float(k.baseline.value)))
            val = cfg.inertia * prev + (1.0 - cfg.inertia) * float(expected)

            total_push = 0.0
            for imp in impacts:
                if imp["kpi_id"] != kpi_id:
                    continue

                it_id = imp["initiative_id"]
                lag = imp["lag_months"]

                hist = progress_history[it_id]
                lag_idx = len(hist) - 1 - lag
                eff_prog = hist[lag_idx] if lag_idx >= 0 else 0.0

                dep_factor = 1.0
                blocked = False
                for dep_type, dep_id in deps.get(it_id, []):
                    if dep_type == "hard":
                        dep_prog = proj_progress.get(dep_id, 0.0)
                        if dep_prog < cfg.dep_block_threshold:
                            blocked = True
                            dep_factor *= (1.0 - cfg.hard_dependency_block)

                if blocked and it_id in effects.dep_block_reduction:
                    reduction = effects.dep_block_reduction[it_id]
                    effective_block = cfg.hard_dependency_block * (1.0 - reduction)
                    dep_factor = 1.0
                    for dep_type, dep_id in deps.get(it_id, []):
                        if dep_type == "hard":
                            dep_prog = proj_progress.get(dep_id, 0.0)
                            if dep_prog < cfg.dep_block_threshold:
                                dep_factor *= (1.0 - effective_block)

                scale = float(impact_scalers.get(it_id, 1.0))

                dirn = okr.get(kpi_id, {}).get("direction", None)
                delta_end = float(imp["delta_end"])
                if dirn == "down" and delta_end > 0:
                    delta_end = -delta_end
                elif dirn == "up" and delta_end < 0:
                    delta_end = -delta_end

                total_push += scale * delta_end * eff_prog * dep_factor * imp["confidence"]

            val += total_push

            # KPI shock stress
            for ev in stress_events:
                if ev.type == "kpi_shock" and ev.target_id == kpi_id and _stress_active(ev, d):
                    val += float(ev.magnitude)

            # plausibility bounds
            if kpi_id == "KPI_STP":
                val = soft_floor(val, 0.0, softness=1.0)
                val = soft_cap(val, 100.0, softness=0.8)
            elif kpi_id == "KPI_E2E":
                val = soft_floor(val, 0.2, softness=0.3)
                val = soft_cap(val, 12.0, softness=1.2)
            elif kpi_id == "KPI_INC":
                val = soft_floor(val, 0.0, softness=3.0)
                val = soft_cap(val, 300.0, softness=25.0)

            current[kpi_id] = float(val)
            rows.append({"date": d, "kpi_id": kpi_id, "projected_value": float(val), "expected_value": float(expected)})

        # advance initiatives if missing explicit history
        for it in plan.initiatives:
            key = (it.id, d)
            if key not in progress_by_date:
                p = proj_progress[it.id]
                p = p + cfg.base_speed * (1.0 - p)
                proj_progress[it.id] = max(0.0, min(1.0, p))

    return pd.DataFrame(rows)


# ----------------------------
# Monte Carlo projection (strict per-sample)
# ----------------------------
def project_kpis_mc(
    plan: Plan,
    kpi_history: pd.DataFrame,
    initiative_history: pd.DataFrame,
    as_of: str,
    horizon_end: str,
    actions: Optional[List[SteeringAction]] = None,
    stress_events: Optional[List[StressEvent]] = None,
    cfg: Optional[WhatIfConfig] = None,
    mc: Optional[WhatIfStochasticConfig] = None,
    integrity_scores: Optional[Dict[str, float]] = None,  # 0..1
) -> pd.DataFrame:
    """
    Returns long DF:
      sample, date, kpi_id, projected_value, expected_value
    """
    cfg = cfg or WhatIfConfig()
    mc = mc or WhatIfStochasticConfig()
    actions = actions or []
    stress_events = stress_events or []
    integrity_scores = integrity_scores or {}

    as_of_m = _month_start(as_of)
    horizon_end_m = _month_start(horizon_end)

    rng = np.random.default_rng(int(mc.seed))

    # robust KPI ids
    kpi_history = _augment_kpi_ids(kpi_history)

    # actions on initiatives
    initiative_history = initiative_history.copy()
    initiative_history["date"] = initiative_history["date"].astype(str).map(_month_start)
    initiative_history_adj, impact_scalers = apply_actions_to_initiatives(
        initiatives_df=initiative_history,
        as_of=as_of_m,
        actions=actions,
    )
    effects = compile_action_effects(actions)

    dates = month_range(as_of_m, horizon_end_m)
    okr = _okr_catalog(plan)
    deps = _initiative_dependencies(plan)
    impacts = _initiative_impacts(plan)

    # initial levels
    current0 = _latest_kpi_values(kpi_history, as_of_m)
    if not current0:
        current0 = {k.id: float(k.baseline.value) for k in plan.kpis}

    prog0 = _latest_initiative_progress(initiative_history_adj, as_of_m)
    prog0 = {it.id: float(prog0.get(it.id, 0.0)) for it in plan.initiatives}

    # explicit progress means-path if provided
    progress_by_date: Dict[Tuple[str, str], float] = {}
    if initiative_history_adj is not None and not initiative_history_adj.empty:
        tmp = initiative_history_adj.copy()
        tmp = tmp[tmp["date"].isin(dates)].sort_values(["initiative_id", "date"])
        for _, r in tmp.iterrows():
            progress_by_date[(str(r["initiative_id"]), str(r["date"]))] = float(r["progress"])

    # KPI scale for noise
    kpi_scale: Dict[str, float] = {}
    for k in plan.kpis:
        scale = max(1.0, abs(float(k.target.value) - float(k.baseline.value)))
        kpi_scale[k.id] = float(scale)

    rows: List[Dict] = []

    for s in range(int(mc.n_samples)):
        inertia_s = float(np.clip(rng.normal(cfg.inertia, mc.inertia_sigma), 0.50, 0.98))
        hard_block_s = float(np.clip(rng.normal(cfg.hard_dependency_block, mc.hard_dep_block_sigma), 0.10, 0.90))

        current = dict(current0)
        proj_progress = dict(prog0)
        progress_history: Dict[str, List[float]] = {it.id: [] for it in plan.initiatives}

        for d in dates:
            stress_now = any(_stress_active(ev, d) for ev in stress_events)
            prog_sigma = mc.progress_noise_sigma * (mc.stress_progress_sigma_mult if stress_now else 1.0)

            # progress evolution
            for it in plan.initiatives:
                key = (it.id, d)
                if key in progress_by_date:
                    mean_p = float(progress_by_date[key])
                else:
                    mean_p = float(proj_progress[it.id] + cfg.base_speed * (1.0 - proj_progress[it.id]))

                noisy = float(mean_p + rng.normal(0.0, prog_sigma))
                if mc.clamp_progress_0_1:
                    noisy = float(np.clip(noisy, 0.0, 1.0))
                proj_progress[it.id] = noisy

                for ev in stress_events:
                    if ev.type == "initiative_progress_drop" and ev.target_id == it.id and _stress_active(ev, d):
                        proj_progress[it.id] = float(np.clip(proj_progress[it.id] - float(ev.magnitude), 0.0, 1.0))

                progress_history[it.id].append(float(proj_progress[it.id]))

            # KPI evolution
            for k in plan.kpis:
                kpi_id = k.id
                cps = okr.get(kpi_id, {}).get("checkpoints", [])
                expected = expected_from_checkpoints(cps, d)
                if expected is None:
                    expected = float(k.baseline.value)

                prev = float(current.get(kpi_id, float(k.baseline.value)))
                val = inertia_s * prev + (1.0 - inertia_s) * float(expected)

                total_push = 0.0
                for imp in impacts:
                    if imp["kpi_id"] != kpi_id:
                        continue
                    it_id = imp["initiative_id"]
                    lag = int(imp["lag_months"])

                    hist = progress_history[it_id]
                    lag_idx = len(hist) - 1 - lag
                    eff_prog = float(hist[lag_idx]) if lag_idx >= 0 else 0.0

                    dep_factor = 1.0
                    blocked = False
                    for dep_type, dep_id in deps.get(it_id, []):
                        if dep_type == "hard":
                            dep_prog = float(proj_progress.get(dep_id, 0.0))
                            if dep_prog < cfg.dep_block_threshold:
                                blocked = True
                                dep_factor *= (1.0 - hard_block_s)

                    if blocked and it_id in effects.dep_block_reduction:
                        reduction = float(effects.dep_block_reduction[it_id])
                        effective_block = hard_block_s * (1.0 - reduction)
                        dep_factor = 1.0
                        for dep_type, dep_id in deps.get(it_id, []):
                            if dep_type == "hard":
                                dep_prog = float(proj_progress.get(dep_id, 0.0))
                                if dep_prog < cfg.dep_block_threshold:
                                    dep_factor *= (1.0 - effective_block)

                    scale = float(impact_scalers.get(it_id, 1.0))

                    dirn = okr.get(kpi_id, {}).get("direction", None)
                    delta_end = float(imp["delta_end"])
                    if dirn == "down" and delta_end > 0:
                        delta_end = -delta_end
                    elif dirn == "up" and delta_end < 0:
                        delta_end = -delta_end

                    total_push += scale * delta_end * eff_prog * dep_factor * float(imp["confidence"])

                val += float(total_push)

                # shock stress with uncertainty
                for ev in stress_events:
                    if ev.type == "kpi_shock" and ev.target_id == kpi_id and _stress_active(ev, d):
                        mag = float(ev.magnitude) * float(rng.normal(1.0, 0.20))
                        val += mag

                # measurement/residual noise (inflated if integrity suspicious)
                integ = float(integrity_scores.get(kpi_id, 0.0))
                mult = mc.integrity_noise_mult if integ >= 0.6 else 1.0
                sigma = float(mc.kpi_noise_frac * kpi_scale.get(kpi_id, 1.0) * mult)
                val += float(rng.normal(0.0, sigma))

                # plausibility bounds
                if kpi_id == "KPI_STP":
                    val = soft_floor(val, 0.0, softness=1.0)
                    val = soft_cap(val, 100.0, softness=0.8)
                elif kpi_id == "KPI_E2E":
                    val = soft_floor(val, 0.2, softness=0.3)
                    val = soft_cap(val, 12.0, softness=1.2)
                elif kpi_id == "KPI_INC":
                    val = soft_floor(val, 0.0, softness=3.0)
                    val = soft_cap(val, 300.0, softness=25.0)

                current[kpi_id] = float(val)

                rows.append(
                    {
                        "sample": int(s),
                        "date": str(d),
                        "kpi_id": str(kpi_id),
                        "projected_value": float(val),
                        "expected_value": float(expected),
                    }
                )

    return pd.DataFrame(rows)


# ----------------------------
# Compare helpers (deterministic)
# ----------------------------
def compare_actions(
    plan: Plan,
    kpi_history: pd.DataFrame,
    initiative_history: pd.DataFrame,
    as_of: str,
    horizon_end: str,
    actions: List[SteeringAction],
    stress_events: Optional[List[StressEvent]] = None,
) -> pd.DataFrame:
    stress_events = stress_events or []
    horizon_end = _month_start(horizon_end)
    as_of = _month_start(as_of)

    base = project_kpis(plan, kpi_history, initiative_history, as_of, horizon_end, actions=[], stress_events=stress_events)
    sim = project_kpis(plan, kpi_history, initiative_history, as_of, horizon_end, actions=actions, stress_events=stress_events)

    base_last = base[base["date"] == horizon_end][["kpi_id", "projected_value"]].rename(columns={"projected_value": "baseline"})
    sim_last = sim[sim["date"] == horizon_end][["kpi_id", "projected_value"]].rename(columns={"projected_value": "with_action"})
    out = base_last.merge(sim_last, on="kpi_id", how="inner")
    out["delta"] = out["with_action"] - out["baseline"]
    return out.sort_values("kpi_id")


def compare_actions_at_dates(
    plan: Plan,
    kpi_history: pd.DataFrame,
    initiative_history: pd.DataFrame,
    as_of: str,
    dates_to_check: List[str],
    actions: List[SteeringAction],
    stress_events: Optional[List[StressEvent]] = None,
) -> pd.DataFrame:
    if not dates_to_check:
        raise ValueError("dates_to_check must not be empty")
    stress_events = stress_events or []

    as_of = _month_start(as_of)
    dates_to_check = sorted({_month_start(d) for d in dates_to_check})
    horizon_end = max(dates_to_check)

    base = project_kpis(plan, kpi_history, initiative_history, as_of, horizon_end, actions=[], stress_events=stress_events)
    sim = project_kpis(plan, kpi_history, initiative_history, as_of, horizon_end, actions=actions, stress_events=stress_events)

    base = base[base["date"].isin(dates_to_check)].rename(columns={"projected_value": "baseline"})
    sim = sim[sim["date"].isin(dates_to_check)].rename(columns={"projected_value": "with_action"})

    out = base[["date", "kpi_id", "baseline"]].merge(sim[["date", "kpi_id", "with_action"]], on=["date", "kpi_id"], how="inner")
    out["delta"] = out["with_action"] - out["baseline"]
    return out.sort_values(["date", "kpi_id"])


# ----------------------------
# Compare at dates (Monte Carlo SUMMARY)
# ----------------------------
def compare_actions_at_dates_mc(
    plan: Plan,
    kpi_history: pd.DataFrame,
    initiative_history: pd.DataFrame,
    as_of: str,
    dates_to_check: List[str],
    actions: List[SteeringAction],
    stress_events: Optional[List[StressEvent]] = None,
    cfg: Optional[WhatIfConfig] = None,
    mc: Optional[WhatIfStochasticConfig] = None,
    integrity_scores: Optional[Dict[str, float]] = None,
) -> pd.DataFrame:
    """
    Returns summary DF:
      date, kpi_id,
      delta_mean, delta_p10, delta_p50, delta_p90,
      p_improve, n_samples
    """
    raw = compare_actions_at_dates_mc_raw(
        plan=plan,
        kpi_history=kpi_history,
        initiative_history=initiative_history,
        as_of=as_of,
        dates_to_check=dates_to_check,
        actions=actions,
        stress_events=stress_events,
        cfg=cfg,
        mc=mc,
        integrity_scores=integrity_scores,
    )

    okr = _okr_catalog(plan)

    def _is_improve(kpi_id: str, delta: float) -> bool:
        dirn = str(okr.get(kpi_id, {}).get("direction", "")).lower()
        if dirn == "down":
            return float(delta) < 0.0
        if dirn == "up":
            return float(delta) > 0.0
        # unknown direction -> treat as no-improvement unless delta is strictly positive
        return float(delta) > 0.0

    rows = []
    for (d, kpi_id), g in raw.groupby(["date", "kpi_id"]):
        xs = g["delta"].astype(float).values
        q10, q50, q90 = np.quantile(xs, [0.10, 0.50, 0.90]).tolist()
        p_imp = float(np.mean([_is_improve(kpi_id, x) for x in xs]))
        rows.append(
            {
                "date": str(d),
                "kpi_id": str(kpi_id),
                "delta_mean": float(np.mean(xs)),
                "delta_p10": float(q10),
                "delta_p50": float(q50),
                "delta_p90": float(q90),
                "p_improve": float(p_imp),
                "n_samples": int(len(xs)),
            }
        )

    return pd.DataFrame(rows).sort_values(["date", "kpi_id"]).reset_index(drop=True)


# ----------------------------
# NEW: Compare at dates (Monte Carlo RAW deltas)  ✅ strict correlation-preserving
# ----------------------------
def compare_actions_at_dates_mc_raw(
    plan: Plan,
    kpi_history: pd.DataFrame,
    initiative_history: pd.DataFrame,
    as_of: str,
    dates_to_check: List[str],
    actions: List[SteeringAction],
    stress_events: Optional[List[StressEvent]] = None,
    cfg: Optional[WhatIfConfig] = None,
    mc: Optional[WhatIfStochasticConfig] = None,
    integrity_scores: Optional[Dict[str, float]] = None,
) -> pd.DataFrame:
    """
    Returns RAW per-sample deltas:
      sample, date, kpi_id, delta, baseline, with_action

    This is the function you need to compute TRUE portfolio score distributions (CVaR)
    without assuming KPI independence.
    """
    if not dates_to_check:
        raise ValueError("dates_to_check must not be empty")

    stress_events = stress_events or []
    cfg = cfg or WhatIfConfig()
    mc = mc or WhatIfStochasticConfig()
    integrity_scores = integrity_scores or {}

    as_of_m = _month_start(as_of)
    dates_to_check_m = sorted({_month_start(d) for d in dates_to_check})
    horizon_end = max(dates_to_check_m)

    # IMPORTANT: correlation preservation requires SAME RNG stream structure
    # between base and action runs. We do this by fixing mc.seed and using it
    # in BOTH calls (project_kpis_mc uses mc.seed).
    base_mc = project_kpis_mc(
        plan=plan,
        kpi_history=kpi_history,
        initiative_history=initiative_history,
        as_of=as_of_m,
        horizon_end=horizon_end,
        actions=[],
        stress_events=stress_events,
        cfg=cfg,
        mc=mc,
        integrity_scores=integrity_scores,
    )
    act_mc = project_kpis_mc(
        plan=plan,
        kpi_history=kpi_history,
        initiative_history=initiative_history,
        as_of=as_of_m,
        horizon_end=horizon_end,
        actions=actions,
        stress_events=stress_events,
        cfg=cfg,
        mc=mc,
        integrity_scores=integrity_scores,
    )

    # keep only dates of interest
    base_mc = base_mc[base_mc["date"].isin(dates_to_check_m)].copy()
    act_mc = act_mc[act_mc["date"].isin(dates_to_check_m)].copy()

    merged = base_mc.merge(
        act_mc,
        on=["sample", "date", "kpi_id"],
        how="inner",
        suffixes=("_base", "_act"),
    )

    merged["baseline"] = merged["projected_value_base"].astype(float)
    merged["with_action"] = merged["projected_value_act"].astype(float)
    merged["delta"] = merged["with_action"] - merged["baseline"]

    return merged[["sample", "date", "kpi_id", "delta", "baseline", "with_action"]].sort_values(
        ["date", "kpi_id", "sample"]
    ).reset_index(drop=True)
