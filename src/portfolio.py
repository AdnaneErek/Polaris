# src/portfolio.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Optional

import numpy as np
import pandas as pd

from .schemas import Plan, KPI
from .actions import SteeringAction
from .whatif import (
    compare_actions_at_dates,
    compare_actions_at_dates_mc,      # still useful (summary-only)
    compare_actions_at_dates_mc_raw,  # ✅ strict MC raw deltas
    StressEvent,
    WhatIfConfig,
    WhatIfStochasticConfig,
)

# -----------------------------
# Data structures
# -----------------------------
@dataclass
class ActionBundle:
    id: str
    name: str
    description: str
    actions: List[SteeringAction]
    requires_approval: bool = True


@dataclass
class OptionResult:
    bundle_id: str
    bundle_name: str
    base_deltas: Dict[str, float]          # kpi_id -> delta at main checkpoint (deterministic)
    stress_deltas: Dict[str, float]        # deterministic
    score_base: float
    score_stress: float
    robustness_gap: float                 # score_base - score_stress (smaller is better)
    notes: List[str]


# NEW: distributional option result for AI-heavy path
@dataclass
class OptionResultMC:
    bundle_id: str
    bundle_name: str
    main_date: str

    # KPI deltas (distribution summary at main_date, under stress)
    stress_delta_summary: Dict[str, Dict[str, float]]  # kpi_id -> {mean,p10,p50,p90,p_improve}
    base_delta_summary: Dict[str, Dict[str, float]]    # optional, same structure

    # Portfolio score distributions
    score_base_mean: float
    score_base_cvar10: float
    score_stress_mean: float
    score_stress_cvar10: float
    robustness_gap_mean: float  # base_mean - stress_mean

    notes: List[str]


# -----------------------------
# KPI utility / scoring
# -----------------------------
def _objective_weights(plan: Plan, weight_overrides: Optional[Dict[str, float]] = None) -> Dict[str, float]:
    """
    Returns mapping: kpi_id -> objective weight (for OKRs).
    Assumes 1 KPI per OKR here, which matches your plan.
    """
    weights: Dict[str, float] = {}
    for obj in plan.objectives:
        w = float(obj.weight)
        for okr in obj.okrs:
            weights[str(okr.kpi_id)] = w
    if not weight_overrides:
        return weights

    adjusted: Dict[str, float] = {}
    for kpi_id, w in weights.items():
        mult = float(weight_overrides.get(kpi_id, 1.0))
        if not np.isfinite(mult) or mult <= 0:
            mult = 1.0
        adjusted[kpi_id] = float(w) * mult

    s = float(sum(adjusted.values()))
    if s <= 0:
        return weights
    # Re-normalize so total objective mass remains stable.
    return {k: float(v / s) for k, v in adjusted.items()}


def _find_kpi(plan: Plan, kpi_id: str) -> Optional[KPI]:
    """
    Find KPI by ID or short_name (alias).
    Returns None if not found.
    """
    for k in plan.kpis:
        if k.id == kpi_id or k.short_name == kpi_id:
            return k
    return None


def _get_normalization_factor(kpi: KPI) -> float:
    """
    Get normalization factor for KPI based on its unit and scale.
    Returns a factor that makes deltas roughly comparable across KPIs.
    """
    unit = kpi.unit.lower()
    baseline = float(kpi.baseline.value)
    target = float(kpi.target.value)
    span = abs(target - baseline)
    
    # Use span as base normalization, but apply unit-specific adjustments
    if unit == "%" or "percent" in unit:
        # Percentage points: 1.0 utility per 1 pp
        return 1.0
    elif unit == "hours" or unit == "h":
        # Hours: 1.0 utility per 0.1 hour (6 minutes)
        return 0.1
    elif unit == "count" or unit == "":
        # Counts: 1.0 utility per 1 unit
        return 1.0
    else:
        # Default: normalize by span if available, else 1.0
        return max(1.0, span * 0.1) if span > 1e-9 else 1.0


def _normalize_delta(plan: Plan, kpi_id: str, delta: float) -> float:
    """
    Convert KPI delta into "utility units", roughly comparable across KPIs.
    Uses plan-defined direction and normalization factors.
    
    For "up" KPIs: positive delta is improvement -> positive utility
    For "down" KPIs: negative delta is improvement -> positive utility
    """
    kpi = _find_kpi(plan, kpi_id)
    if not kpi:
        # Unknown KPI: return raw delta (fallback)
        return delta
    
    direction = kpi.target.direction
    factor = _get_normalization_factor(kpi)
    
    if direction == "down":
        # For "down" KPIs, negative delta is improvement
        return (-delta) / factor
    else:
        # For "up" KPIs, positive delta is improvement
        return delta / factor


def score_option(plan: Plan, deltas: Dict[str, float], weight_overrides: Optional[Dict[str, float]] = None) -> float:
    """
    Weighted sum of normalized KPI utilities.
    """
    weights = _objective_weights(plan, weight_overrides=weight_overrides)
    total = 0.0
    for kpi_id, w in weights.items():
        dv = float(deltas.get(kpi_id, 0.0))
        total += float(w) * _normalize_delta(plan, kpi_id, dv)
    return float(total)


# -----------------------------
# What-if helpers
# -----------------------------
def _deltas_at_date(comp: pd.DataFrame, date: str) -> Dict[str, float]:
    sub = comp[comp["date"] == date]
    out: Dict[str, float] = {}
    for _, r in sub.iterrows():
        out[str(r["kpi_id"])] = float(r["delta"])
    return out


def _pick_main_checkpoint(dates_to_check: List[str], as_of: Optional[str] = None) -> str:
    """
    Pick a scoring checkpoint.
    - If as_of is provided, prefer the first checkpoint strictly after as_of.
    - Otherwise, fall back to the earliest checkpoint.
    """
    ordered = sorted(dates_to_check)
    if not ordered:
        raise ValueError("dates_to_check must contain at least one date")
    if as_of is None:
        return ordered[0]

    as_of_ts = pd.to_datetime(as_of)
    future = [d for d in ordered if pd.to_datetime(d) > as_of_ts]
    return future[0] if future else ordered[-1]


def _material_note(deltas: Dict[str, float]) -> List[str]:
    notes: List[str] = []
    stp = deltas.get("KPI_STP")
    e2e = deltas.get("KPI_E2E")
    inc = deltas.get("KPI_INC")

    if e2e is not None and stp is not None:
        if e2e < 0 and stp < 0:
            notes.append("Trade-off: E2E improves but STP slightly degrades (scope/coverage effect).")
        if e2e < 0 and stp > 0:
            notes.append("Win-win: both E2E and STP improve.")
    if inc is not None and inc < 0:
        notes.append("Resilience improves (INC decreases).")
    return notes


# -----------------------------
# Deterministic evaluator (unchanged)
# -----------------------------
def evaluate_bundles(
    plan: Plan,
    kpis_df: pd.DataFrame,                 # date,kpi_id,value
    initiatives_df: pd.DataFrame,          # simulated initiatives
    as_of: str,
    dates_to_check: List[str],
    bundles: List[ActionBundle],
    stress_events: Optional[List[StressEvent]] = None,
) -> List[OptionResult]:
    stress_events = stress_events or []
    main_date = _pick_main_checkpoint(dates_to_check, as_of=as_of)

    results: List[OptionResult] = []

    for b in bundles:
        comp_base = compare_actions_at_dates(
            plan,
            kpis_df,
            initiatives_df,
            as_of=as_of,
            dates_to_check=dates_to_check,
            actions=b.actions,
            stress_events=[],  # base
        )
        base_deltas = _deltas_at_date(comp_base, main_date) if comp_base is not None and not comp_base.empty else {}

        comp_stress = compare_actions_at_dates(
            plan,
            kpis_df,
            initiatives_df,
            as_of=as_of,
            dates_to_check=dates_to_check,
            actions=b.actions,
            stress_events=stress_events,
        )
        stress_deltas = _deltas_at_date(comp_stress, main_date) if comp_stress is not None and not comp_stress.empty else {}

        score_b = score_option(plan, base_deltas)
        score_s = score_option(plan, stress_deltas)
        gap = score_b - score_s

        notes: List[str] = []
        notes += _material_note(stress_deltas if stress_deltas else base_deltas)
        if stress_events:
            notes.append("Includes stress robustness check.")

        results.append(
            OptionResult(
                bundle_id=b.id,
                bundle_name=b.name,
                base_deltas=base_deltas,
                stress_deltas=stress_deltas,
                score_base=score_b,
                score_stress=score_s,
                robustness_gap=gap,
                notes=notes,
            )
        )

    results.sort(key=lambda r: (r.score_stress, r.score_base, -r.robustness_gap), reverse=True)
    return results


# -----------------------------
# Monte Carlo evaluator (STRICT)
# -----------------------------
def _cvar_lower(xs: np.ndarray, alpha: float = 0.10) -> float:
    """
    CVaR of the lower tail (risk-averse): mean of worst alpha fraction.
    Assumes higher score is better.
    """
    xs = np.asarray(xs, dtype=float)
    if xs.size == 0:
        return float("nan")
    q = float(np.quantile(xs, alpha))
    tail = xs[xs <= q]
    return float(np.mean(tail)) if tail.size else q


def _summarize_kpi_deltas_from_raw(raw: pd.DataFrame, plan: Plan) -> Dict[str, Dict[str, float]]:
    """
    raw columns: sample,date,kpi_id,delta,baseline,with_action (single date expected)
    returns: kpi_id -> {mean,p10,p50,p90,p_improve}
    """
    # direction-aware improvement
    okr_dir: Dict[str, str] = {}
    for obj in plan.objectives:
        for okr in obj.okrs:
            okr_dir[str(okr.kpi_id)] = str(okr.direction)

    out: Dict[str, Dict[str, float]] = {}
    for kpi_id, g in raw.groupby("kpi_id"):
        xs = g["delta"].astype(float).values
        p10, p50, p90 = np.quantile(xs, [0.10, 0.50, 0.90]).tolist()
        mu = float(np.mean(xs))

        dirn = okr_dir.get(str(kpi_id))
        if dirn == "down":
            p_imp = float(np.mean(xs < 0.0))
        else:
            p_imp = float(np.mean(xs > 0.0))

        out[str(kpi_id)] = {
            "mean": mu,
            "p10": float(p10),
            "p50": float(p50),
            "p90": float(p90),
            "p_improve": p_imp,
        }
    return out


def _score_samples_from_raw(raw: pd.DataFrame, plan: Plan, weight_overrides: Optional[Dict[str, float]] = None) -> np.ndarray:
    """
    raw columns include: sample,kpi_id,delta
    Returns score per sample (using full correlated delta vector).
    """
    weights = _objective_weights(plan, weight_overrides=weight_overrides)
    # pivot into sample x kpi_id
    piv = raw.pivot_table(index="sample", columns="kpi_id", values="delta", aggfunc="mean").fillna(0.0)

    scores = np.zeros(len(piv), dtype=float)
    for kpi_id, w in weights.items():
        if kpi_id in piv.columns:
            dv = piv[kpi_id].astype(float).values
        else:
            dv = np.zeros(len(piv), dtype=float)
        
        # Normalize deltas for this KPI
        normalized = np.array([_normalize_delta(plan, kpi_id, float(x)) for x in dv])
        scores += float(w) * normalized
    return scores


def evaluate_bundles_mc(
    plan: Plan,
    kpis_df: pd.DataFrame,
    initiatives_df: pd.DataFrame,
    as_of: str,
    dates_to_check: List[str],
    bundles: List[ActionBundle],
    stress_events: Optional[List[StressEvent]] = None,
    whatif_cfg: Optional[WhatIfConfig] = None,
    mc_cfg: Optional[WhatIfStochasticConfig] = None,
    integrity_scores: Optional[Dict[str, float]] = None,  # 0..1
    cvar_alpha: float = 0.10,
    scoring_weight_overrides: Optional[Dict[str, float]] = None,
) -> List[OptionResultMC]:
    """
    Strict ranking (robust): by stress CVaR(alpha) first, then stress mean, then base mean.
    Uses RAW Monte Carlo deltas (correlation-preserving).
    """
    stress_events = stress_events or []
    whatif_cfg = whatif_cfg or WhatIfConfig()
    mc_cfg = mc_cfg or WhatIfStochasticConfig()
    integrity_scores = integrity_scores or {}

    main_date = _pick_main_checkpoint(dates_to_check, as_of=as_of)

    results: List[OptionResultMC] = []

    for b in bundles:
        # ----- BASE (raw deltas at main_date) -----
        raw_base = compare_actions_at_dates_mc_raw(
            plan=plan,
            kpi_history=kpis_df,
            initiative_history=initiatives_df,
            as_of=as_of,
            dates_to_check=[main_date],
            actions=b.actions,
            stress_events=[],  # base
            cfg=whatif_cfg,
            mc=mc_cfg,
            integrity_scores=integrity_scores,
        )
        # safety: ensure date is the one we expect
        raw_base = raw_base[raw_base["date"] == main_date].copy()

        # ----- STRESS (raw deltas at main_date) -----
        raw_stress = compare_actions_at_dates_mc_raw(
            plan=plan,
            kpi_history=kpis_df,
            initiative_history=initiatives_df,
            as_of=as_of,
            dates_to_check=[main_date],
            actions=b.actions,
            stress_events=stress_events,
            cfg=whatif_cfg,
            mc=mc_cfg,
            integrity_scores=integrity_scores,
        )
        raw_stress = raw_stress[raw_stress["date"] == main_date].copy()

        # KPI summaries
        base_delta_summary = _summarize_kpi_deltas_from_raw(raw_base, plan=plan)
        stress_delta_summary = _summarize_kpi_deltas_from_raw(raw_stress, plan=plan)

        # Score distributions (STRICT: per-sample correlated KPI vector)
        base_scores = _score_samples_from_raw(raw_base, plan=plan, weight_overrides=scoring_weight_overrides)
        stress_scores = _score_samples_from_raw(raw_stress, plan=plan, weight_overrides=scoring_weight_overrides)

        base_mean = float(np.mean(base_scores))
        stress_mean = float(np.mean(stress_scores))
        base_cvar = _cvar_lower(base_scores, alpha=cvar_alpha)
        stress_cvar = _cvar_lower(stress_scores, alpha=cvar_alpha)
        gap_mean = base_mean - stress_mean

        # Notes (use stress mean deltas for narrative)
        mean_deltas_for_notes = {k: v["mean"] for k, v in (stress_delta_summary or base_delta_summary).items()}
        notes: List[str] = []
        notes += _material_note(mean_deltas_for_notes)
        if stress_events:
            notes.append(f"Monte Carlo robustness ranked by CVaR{int(cvar_alpha*100)} (lower tail of score).")

        results.append(
            OptionResultMC(
                bundle_id=b.id,
                bundle_name=b.name,
                main_date=main_date,
                stress_delta_summary=stress_delta_summary,
                base_delta_summary=base_delta_summary,
                score_base_mean=base_mean,
                score_base_cvar10=base_cvar,
                score_stress_mean=stress_mean,
                score_stress_cvar10=stress_cvar,
                robustness_gap_mean=gap_mean,
                notes=notes,
            )
        )

    # Sort with deterministic tie-break rules
    # Primary: stress_CVaR, Secondary: stress_mean, Tertiary: base_mean
    # Then: fewer actions (lower change risk), Then: low-risk bundle flag
    def _tie_break_key(r: OptionResultMC) -> tuple:
        bundle = next((b for b in bundles if b.id == r.bundle_id), None)
        n_actions = len(bundle.actions) if bundle else 999
        # Low-risk: <= 1 action and no split_scope
        is_low_risk = False
        if bundle and len(bundle.actions) <= 1:
            has_split = any(a.type == "split_scope" for a in bundle.actions)
            is_low_risk = not has_split
        return (
            r.score_stress_cvar10,
            r.score_stress_mean,
            r.score_base_mean,
            -n_actions,  # fewer actions = better (negative for reverse=True)
            -1 if is_low_risk else 0,  # low-risk = better
        )
    
    results.sort(key=_tie_break_key, reverse=True)
    return results


# -----------------------------
# Formatting helpers (deterministic)
# -----------------------------
def format_deltas(deltas: Dict[str, float]) -> str:
    def fmt(k: str, v: float) -> str:
        if k == "KPI_STP":
            return f"{v:+.2f}pp"
        if k == "KPI_E2E":
            return f"{v:+.2f}h"
        if k == "KPI_INC":
            return f"{v:+.2f}"
        return f"{v:+.2f}"

    parts = []
    for k in ["KPI_STP", "KPI_E2E", "KPI_INC"]:
        if k in deltas:
            parts.append(f"{k}:{fmt(k, deltas[k])}")
    return " | ".join(parts) if parts else "(no deltas)"


def format_option_table(results: List[OptionResult], main_date: str) -> str:
    lines: List[str] = []
    lines.append(f"PORTFOLIO OPTIONS — scored at {main_date}")
    lines.append("-" * 88)
    for i, r in enumerate(results, 1):
        lines.append(
            f"{i}) {r.bundle_name} "
            f"(score_stress={r.score_stress:.2f}, score_base={r.score_base:.2f}, gap={r.robustness_gap:.2f})"
        )
        lines.append(f"   Stress deltas: {format_deltas(r.stress_deltas)}")
        if r.base_deltas:
            lines.append(f"   Base deltas:   {format_deltas(r.base_deltas)}")
        if r.notes:
            for n in r.notes[:2]:
                lines.append(f"   Note: {n}")
        lines.append("")
    return "\n".join(lines)


# -----------------------------
# Formatter for MC results
# -----------------------------
def format_option_table_mc(results: List[OptionResultMC]) -> str:
    if not results:
        return "PORTFOLIO OPTIONS (MONTE CARLO) — (no results)"

    main_date = results[0].main_date
    lines: List[str] = []
    lines.append(f"PORTFOLIO OPTIONS (MONTE CARLO, STRICT) — scored at {main_date}")
    lines.append("-" * 110)

    def fmt_kpi(kpi_id: str, s: Dict[str, float]) -> str:
        m, p10, p50, p90, p = s["mean"], s["p10"], s["p50"], s["p90"], s["p_improve"]
        if kpi_id == "KPI_STP":
            unit = "pp"
        elif kpi_id == "KPI_E2E":
            unit = "h"
        else:
            unit = ""
        return (
            f"{kpi_id}: mean={m:+.2f}{unit} "
            f"p10={p10:+.2f}{unit} p50={p50:+.2f}{unit} p90={p90:+.2f}{unit} "
            f"P(improve)={p:.2f}"
        )

    for i, r in enumerate(results, 1):
        lines.append(
            f"{i}) {r.bundle_name} | stress_mean={r.score_stress_mean:.2f} "
            f"| stress_CVaR10={r.score_stress_cvar10:.2f} | base_mean={r.score_base_mean:.2f} "
            f"| base_CVaR10={r.score_base_cvar10:.2f} | gap_mean={r.robustness_gap_mean:.2f}"
        )

        for k in ["KPI_STP", "KPI_E2E", "KPI_INC"]:
            if k in r.stress_delta_summary:
                lines.append("   " + fmt_kpi(k, r.stress_delta_summary[k]))

        if r.notes:
            for n in r.notes[:2]:
                lines.append(f"   Note: {n}")
        lines.append("")

    return "\n".join(lines)
