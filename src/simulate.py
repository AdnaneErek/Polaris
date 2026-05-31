# src/simulate.py
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta

from .load_plan import load_plan
from .schemas import Plan, Initiative, KPI, Direction


# -------------------------
# Time helpers
# -------------------------
def _dt(s: str) -> datetime:
    return datetime.fromisoformat(s)


def month_range(start: str, end: str) -> List[str]:
    cur = _dt(start).date().replace(day=1)
    end_dt = _dt(end).date().replace(day=1)
    out = []
    while cur <= end_dt:
        out.append(cur.isoformat())
        cur = (datetime.fromisoformat(cur.isoformat()) + relativedelta(months=1)).date().replace(day=1)
    return out


def clamp(x: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, x)))


def interpolate_piecewise(checkpoints: List[Tuple[str, float]], as_of: str) -> float:
    pts = sorted([(datetime.fromisoformat(d), float(v)) for d, v in checkpoints], key=lambda t: t[0])
    t = datetime.fromisoformat(as_of)
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


# -------------------------
# Simulation primitives
# -------------------------
@dataclass
class SimConfig:
    seed: int = 7
    kpi_noise_frac: float = 0.03
    risk_event_prob: float = 0.08
    risk_delay_months_min: int = 1
    risk_delay_months_max: int = 3
    hard_dependency_block: float = 0.50

    # realism guardrails
    # max change per month (absolute) to avoid “teleporting” KPIs
    max_monthly_change: Dict[str, float] = None  # filled in __post_init__
    # allow KPI to drift away from expected but not explode
    max_abs_deviation_vs_expected: Dict[str, float] = None  # filled in __post_init__
    # simulate a longer pre-plan history to avoid short training windows
    history_start: str = "2020-01-01"
    history_end: Optional[str] = None  # default: plan.meta.horizon.end
    # regime realism
    regime_p_switch_to_stress: float = 0.06
    regime_p_recover: float = 0.30
    regime_p_back_to_normal: float = 0.35
    shock_prob_per_month: float = 0.04
    shock_decay: float = 0.65
    # AR(1)-style residual memory per KPI
    ar1_phi: Dict[str, float] = None  # filled in __post_init__
    seasonality_amp: Dict[str, float] = None  # filled in __post_init__
    seasonality_phase: Dict[str, float] = None  # filled in __post_init__

    def __post_init__(self):
        if self.max_monthly_change is None:
            self.max_monthly_change = {
                "KPI_STP": 2.0,   # percentage points per month
                "KPI_E2E": 0.20,  # hours per month
                "KPI_INC": 4.0,   # incidents per month
            }
        if self.max_abs_deviation_vs_expected is None:
            self.max_abs_deviation_vs_expected = {
                "KPI_STP": 10.0,  # pp
                "KPI_E2E": 1.0,   # hours
                "KPI_INC": 15.0,  # incidents
            }
        if self.ar1_phi is None:
            self.ar1_phi = {
                "KPI_STP": 0.55,
                "KPI_E2E": 0.50,
                "KPI_INC": 0.65,
            }
        if self.seasonality_amp is None:
            self.seasonality_amp = {
                "KPI_STP": 0.45,
                "KPI_E2E": 0.06,
                "KPI_INC": 1.25,
            }
        if self.seasonality_phase is None:
            self.seasonality_phase = {
                "KPI_STP": 0.5,
                "KPI_E2E": 1.2,
                "KPI_INC": -0.7,
            }


def _initiative_window_mask(dates: List[str], start: str, end: str) -> List[bool]:
    s = datetime.fromisoformat(start)
    e = datetime.fromisoformat(end)
    out = []
    for ds in dates:
        t = datetime.fromisoformat(ds)
        out.append(s <= t <= e)
    return out


def _progress_series(
    dates: List[str],
    start: str,
    end: str,
    rng: np.random.Generator,
    noise: float = 0.02,
) -> List[float]:
    """
    Progress rises from 0→1 approximately across the active window length.
    This prevents initiatives from finishing unrealistically early.
    """
    mask = _initiative_window_mask(dates, start, end)
    active_idxs = [i for i, a in enumerate(mask) if a]
    if not active_idxs:
        return [0.0 for _ in dates]

    # number of active months in window
    n_active = len(active_idxs)
    # base step so that cumulative reaches ~1 by end of window
    base_speed = 1.0 / max(1, n_active)

    prog = 0.0
    out = []
    for i, active in enumerate(mask):
        if active:
            # small random variation but keep monotone-ish
            prog += base_speed + rng.normal(0, noise * base_speed)
            prog = clamp(prog, 0.0, 1.0)
        out.append(prog)
    return out


def _shift_date_by_months(date_iso: str, months: int) -> str:
    d = datetime.fromisoformat(date_iso)
    return (d + relativedelta(months=months)).date().isoformat()


def _expected_with_preplan(
    checkpoints: List[Tuple[str, float]],
    as_of: str,
    history_start: str,
    baseline: float,
) -> float:
    """
    Piecewise expected path with a realistic pre-plan ramp:
    - before first checkpoint: linear ramp from baseline@history_start to first checkpoint
    - between checkpoints: linear interpolation
    - after last checkpoint: hold last expected
    """
    pts = sorted(checkpoints, key=lambda x: x[0])
    if not pts:
        return float(baseline)

    t = datetime.fromisoformat(as_of)
    first_t = datetime.fromisoformat(pts[0][0])
    first_v = float(pts[0][1])
    h0 = datetime.fromisoformat(history_start)

    if t <= first_t:
        if t <= h0:
            return float(baseline)
        span = max(1, (first_t - h0).days)
        frac = (t - h0).days / span
        return float(baseline + frac * (first_v - baseline))

    return float(interpolate_piecewise(checkpoints, as_of))


# -------------------------
# Main simulation
# -------------------------
def simulate(plan: Plan, cfg: SimConfig) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(cfg.seed)
    sim_end = cfg.history_end or plan.meta.horizon.end
    dates = month_range(cfg.history_start, sim_end)

    kpi_by_id: Dict[str, KPI] = {k.id: k for k in plan.kpis}
    init_by_id: Dict[str, Initiative] = {i.id: i for i in plan.initiatives}

    # ---- Expected KPI trajectory from OKR checkpoints
    okrs_by_kpi: Dict[str, List[Tuple[float, Direction, List[Tuple[str, float]]]]] = {}
    for obj in plan.objectives:
        for okr in obj.okrs:
            cps = [(cp.date, float(cp.expected)) for cp in okr.trajectory.checkpoints]
            okrs_by_kpi.setdefault(okr.kpi_id, []).append((obj.weight, okr.direction, cps))

    expected_kpi: Dict[Tuple[str, str], float] = {}
    for d in dates:
        for kpi_id, okr_list in okrs_by_kpi.items():
            wsum = sum(w for w, _, _ in okr_list) or 1.0
            val = 0.0
            for w, _, cps in okr_list:
                baseline = float(kpi_by_id[kpi_id].baseline.value) if kpi_id in kpi_by_id else float(cps[0][1])
                val += (w / wsum) * _expected_with_preplan(cps, d, cfg.history_start, baseline)
            expected_kpi[(d, kpi_id)] = float(val)

    # ---- Initiative progress + risk delays
    initiative_progress: Dict[str, List[float]] = {}
    initiative_delay_months: Dict[str, int] = {i.id: 0 for i in plan.initiatives}
    events: List[dict] = []

    for it in plan.initiatives:
        initiative_progress[it.id] = _progress_series(dates, it.timeline.start, it.timeline.end, rng, noise=0.15)

    # random delay events (optional)
    for it in plan.initiatives:
        if not it.risks:
            continue
        if rng.random() < 0.35:
            delay = int(rng.integers(cfg.risk_delay_months_min, cfg.risk_delay_months_max + 1))
            initiative_delay_months[it.id] = delay

            active_months = [d for d in dates if it.timeline.start <= d <= it.timeline.end]
            if active_months:
                ev_month = rng.choice(active_months)
                events.append({
                    "date": ev_month,
                    "type": "risk_delay",
                    "initiative_id": it.id,
                    "delay_months": delay,
                    "description": f"Synthetic delay event on {it.id} (+{delay} months)."
                })

            # shift progress right
            p = initiative_progress[it.id]
            shifted = [0.0] * len(p)
            for idx, val in enumerate(p):
                j = idx + delay
                if j < len(p):
                    shifted[j] = max(shifted[j], val)
            # fill forward
            m = 0.0
            out = []
            for v in shifted:
                m = max(m, v)
                out.append(m)
            initiative_progress[it.id] = out

    # Dependency info for KPI impacts
    impacts = []
    for it in plan.initiatives:
        for imp in it.kpi_impacts:
            impacts.append({
                "initiative_id": it.id,
                "kpi_id": imp.kpi_id,
                "delta_end": float(imp.expected_delta_by_end),
                "confidence": float(imp.confidence),
                "lag_months": int(imp.lag_months),
                "dependencies": [(dep.type, dep.initiative_id) for dep in it.dependencies],
            })

    # ---- Initiative dataframe
    init_rows = []
    for idx, d in enumerate(dates):
        for it in plan.initiatives:
            delay = initiative_delay_months[it.id]
            init_rows.append({
                "date": d,
                "initiative_id": it.id,
                "initiative_name": it.name,
                "owner": it.owner,
                "progress": float(initiative_progress[it.id][idx]),
                "delay_months": delay,
                "budget_k": float(it.budget.amount_k),
            })
    df_inits = pd.DataFrame(init_rows)

    # ---- KPI simulation (realistic + marginal progress impacts)
    current: Dict[str, float] = {k.id: float(k.baseline.value) for k in plan.kpis}
    inertia = 0.85
    residual_state: Dict[str, float] = {k.id: 0.0 for k in plan.kpis}
    active_shocks: Dict[str, Tuple[float, int]] = {k.id: (0.0, 0) for k in plan.kpis}
    regime = "normal"  # normal -> stress -> recovery -> normal

    kpi_rows = []

    for t_idx, d in enumerate(dates):
        # simple Markov-like regime transitions to create realistic episodes
        if regime == "normal" and rng.random() < cfg.regime_p_switch_to_stress:
            regime = "stress"
            events.append({
                "date": d,
                "type": "regime_start",
                "initiative_id": "",
                "description": "Synthetic stress regime starts (higher volatility and adverse shocks).",
            })
        elif regime == "stress" and rng.random() < cfg.regime_p_recover:
            regime = "recovery"
            events.append({
                "date": d,
                "type": "regime_recovery",
                "initiative_id": "",
                "description": "Synthetic recovery regime starts (partial normalization).",
            })
        elif regime == "recovery" and rng.random() < cfg.regime_p_back_to_normal:
            regime = "normal"
            events.append({
                "date": d,
                "type": "regime_normal",
                "initiative_id": "",
                "description": "Synthetic KPI dynamics return to normal regime.",
            })

        for k in plan.kpis:
            kpi_id = k.id
            exp = expected_kpi.get((d, kpi_id), float(k.baseline.value))

            prev_val = float(current[kpi_id])

            # snap slowly toward expected (keeps realism)
            base_val = inertia * prev_val + (1 - inertia) * exp

            # Apply initiative impacts as *marginal* realized impact this month
            total_push = 0.0
            for imp in impacts:
                if imp["kpi_id"] != kpi_id:
                    continue

                it_id = imp["initiative_id"]
                lag = imp["lag_months"]

                lag_idx_now = t_idx - lag
                lag_idx_prev = t_idx - lag - 1

                prog_now = float(initiative_progress[it_id][lag_idx_now]) if lag_idx_now >= 0 else 0.0
                prog_prev = float(initiative_progress[it_id][lag_idx_prev]) if lag_idx_prev >= 0 else 0.0

                # marginal progress only
                dprog = max(0.0, prog_now - prog_prev)

                # dependency blocking factor
                dep_factor = 1.0
                for dep_type, dep_id in imp["dependencies"]:
                    if dep_type == "hard":
                        dep_prog = float(initiative_progress.get(dep_id, [0.0] * len(dates))[t_idx])
                        if dep_prog < 0.6:
                            dep_factor *= (1.0 - cfg.hard_dependency_block)

                # explicit A1<-DQ1 bottleneck in 2026 (optional)
                if it_id == "INIT_A1" and "INIT_DQ1" in initiative_progress and d <= "2026-12-01":
                    dq_prog = float(initiative_progress["INIT_DQ1"][t_idx])
                    if dq_prog < 0.6:
                        dep_factor *= 0.6

                # end-delta distributed across timeline via marginal progress
                push = imp["delta_end"] * dprog * dep_factor * imp["confidence"]
                total_push += push

            val = base_val + total_push

            # seasonality (monthly)
            month_idx = datetime.fromisoformat(d).month - 1
            amp = float(cfg.seasonality_amp.get(kpi_id, 0.0))
            phase = float(cfg.seasonality_phase.get(kpi_id, 0.0))
            seasonal = amp * np.sin((2.0 * np.pi * month_idx / 12.0) + phase)
            val += float(seasonal)

            # noise scaled to KPI scale
            scale = max(1.0, abs(k.target.value - k.baseline.value))
            regime_vol_mult = 1.0
            regime_bias = 0.0
            if regime == "stress":
                regime_vol_mult = 1.8
                # "up" KPI tends to worsen (negative), "down" KPI tends to worsen (positive)
                regime_bias = -0.12 * scale if kpi_id == "KPI_STP" else (0.10 * scale if kpi_id in ("KPI_E2E", "KPI_INC") else 0.0)
            elif regime == "recovery":
                regime_vol_mult = 1.25
                regime_bias = +0.05 * scale if kpi_id == "KPI_STP" else (-0.04 * scale if kpi_id in ("KPI_E2E", "KPI_INC") else 0.0)
            val += float(regime_bias)

            # occasional shock with multi-month decay
            sh_mag, sh_ttl = active_shocks.get(kpi_id, (0.0, 0))
            if sh_ttl > 0:
                val += float(sh_mag)
                active_shocks[kpi_id] = (float(sh_mag * cfg.shock_decay), int(sh_ttl - 1))
            elif rng.random() < cfg.shock_prob_per_month:
                # direction-aware shock
                direction = "up"
                for obj in plan.objectives:
                    for okr in obj.okrs:
                        if okr.kpi_id == kpi_id:
                            direction = str(okr.direction)
                            break
                mag = float(rng.normal(0.0, 0.25 * scale))
                if direction == "up" and mag > 0:
                    mag = -mag
                if direction == "down" and mag < 0:
                    mag = -mag
                ttl = int(rng.integers(2, 5))
                active_shocks[kpi_id] = (mag, ttl)
                events.append({
                    "date": d,
                    "type": "kpi_shock",
                    "initiative_id": "",
                    "description": f"Synthetic {kpi_id} shock ({mag:+.2f}) lasting ~{ttl} months.",
                })
                val += mag

            # AR(1) residual memory
            phi = float(cfg.ar1_phi.get(kpi_id, 0.5))
            eps = float(rng.normal(0, cfg.kpi_noise_frac * scale * regime_vol_mult))
            residual_state[kpi_id] = float(phi * residual_state.get(kpi_id, 0.0) + eps)
            val += float(residual_state[kpi_id])

            # monthly change guardrail
            max_step = float(cfg.max_monthly_change.get(kpi_id, 999.0))
            val = clamp(val, prev_val - max_step, prev_val + max_step)

            # keep near expected (avoid huge divergence from OKR trajectory)
            max_dev = float(cfg.max_abs_deviation_vs_expected.get(kpi_id, 999.0))
            val = clamp(val, exp - max_dev, exp + max_dev)

            # plausible bounds
            if kpi_id == "KPI_STP":
                val = clamp(val, 50.0, 99.9)
            elif kpi_id == "KPI_E2E":
                val = clamp(val, 0.5, 10.0)
            elif kpi_id == "KPI_INC":
                val = clamp(val, 0.0, 200.0)

            current[kpi_id] = float(val)

            kpi_rows.append({
                "date": d,
                "kpi_id": kpi_id,
                "kpi_name": k.short_name,
                "value": float(val),
                "expected": float(exp),
                "regime": regime,
            })

    df_kpis = pd.DataFrame(kpi_rows)
    df_events = pd.DataFrame(events) if events else pd.DataFrame(columns=["date","type","initiative_id","description"])
    return df_kpis, df_inits, df_events


def main():
    plan = load_plan("data/plan.yaml")
    cfg = SimConfig(seed=7)
    df_kpis, df_inits, df_events = simulate(plan, cfg)

    df_kpis.to_csv("data/simulated_kpis.csv", index=False)
    df_inits.to_csv("data/simulated_initiatives.csv", index=False)
    df_events.to_csv("data/simulated_events.csv", index=False)

    print("[OK] Wrote data/simulated_kpis.csv rows:", len(df_kpis))
    print("[OK] Wrote data/simulated_initiatives.csv rows:", len(df_inits))
    print("[OK] Wrote data/simulated_events.csv rows:", len(df_events))


if __name__ == "__main__":
    main()
