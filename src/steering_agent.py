# src/steering_agent.py
from __future__ import annotations

from dataclasses import dataclass, asdict, is_dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
import hashlib
import json
import os

import pandas as pd

from src.schemas import Plan

# Bank-style tie tolerance:
# Portfolio scores/deltas are approximate; if values are equal to 2 decimals in reporting,
# they should be treated as ties to avoid misleading "lower score" statements.
EPS = 1e-3


# ----------------------------
# Decision objects
# ----------------------------
@dataclass(frozen=True)
class DecisionBrief:
    as_of: str
    headline: str
    recommended_option_id: str
    recommended_option_name: str
    decision_rationale: List[str]
    evidence: List[str]
    guardrails_triggered: List[str]
    approvals_required: List[str]
    next_actions: List[str]
    rejected_options: List[Dict[str, Any]]
    metrics: Dict[str, Any]


# ----------------------------
# Utilities
# ----------------------------
def _iso_now() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def _sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _df_fingerprint(df: pd.DataFrame, max_rows: int = 2000) -> str:
    if df is None:
        return "none"
    tmp = df.copy().head(max_rows)
    tmp = tmp.reindex(sorted(tmp.columns), axis=1)
    payload = tmp.to_csv(index=False)
    return _sha256_text(payload)


def _to_mapping(opt: Any) -> Dict[str, Any]:
    """
    Convert OptionResult (dataclass/pydantic/obj) OR dict into a plain dict.
    Supports:
      - dict
      - dataclass instances
      - Pydantic v2 model_dump()
      - generic objects with __dict__
    """
    if isinstance(opt, dict):
        return opt
    if is_dataclass(opt):
        return asdict(opt)
    if hasattr(opt, "model_dump"):  # pydantic v2
        return opt.model_dump()
    if hasattr(opt, "__dict__"):
        return dict(opt.__dict__)
    raise TypeError(f"Unsupported option type: {type(opt).__name__}")


def _format_deltas(deltas: Dict[str, float]) -> str:
    def fmt(k: str, v: float) -> str:
        if k == "KPI_STP":
            return f"{k}:{v:+.2f}pp"
        if k == "KPI_E2E":
            return f"{k}:{v:+.2f}h"
        return f"{k}:{v:+.2f}"

    parts = []
    for k in ["KPI_STP", "KPI_E2E", "KPI_INC"]:
        if k in deltas:
            parts.append(fmt(k, float(deltas[k])))
    for k in sorted(set(deltas.keys()) - {"KPI_STP", "KPI_E2E", "KPI_INC"}):
        parts.append(fmt(k, float(deltas[k])))
    return " | ".join(parts) if parts else "(no deltas)"


# ----------------------------
# Agent policy (Style A)
# ----------------------------
@dataclass(frozen=True)
class BankRealisticPolicy:
    """
    Style A: robust, risk-first, governance-aware.

    Weighting logic:
      - Primary: maximize score_stress
      - Secondary: minimize abs(gap)
      - Tertiary: maximize score_base
      - Tie-breaker: prefer fewer actions (less operational change)
    """
    min_score_stress: float = -999.0
    max_gap_abs: float = 5.0
    prefer_low_changes: bool = True

    # Governance guardrails
    block_if_kpi_integrity_flag: bool = True


def choose_option_style_a(
    plan: Plan,
    as_of: str,
    options: List[Any],  # OptionResult objects or dicts
    integrity_flags: Optional[List[str]] = None,
    policy: Optional[BankRealisticPolicy] = None,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], List[str]]:
    integrity_flags = integrity_flags or []
    policy = policy or BankRealisticPolicy()
    guardrails_triggered: List[str] = []

    if policy.block_if_kpi_integrity_flag and integrity_flags:
        guardrails_triggered.append(
            "KPI integrity flags present — recommendations must be conservative and require approval."
        )

    if not options:
        raise ValueError("No portfolio options provided to the agent.")

    # Normalize option objects to dicts with required keys
    normalized: List[Dict[str, Any]] = []
    for opt_obj in options:
        opt = _to_mapping(opt_obj)

        name = opt.get("name") or opt.get("option_name") or opt.get("bundle_name") or opt.get("id") or "Option"
        opt_id = opt.get("id") or opt.get("option_id") or opt.get("bundle_id") or name

        score_stress = float(opt.get("score_stress", opt.get("stress_score", 0.0)))
        score_base = float(opt.get("score_base", opt.get("base_score", 0.0)))
        gap = float(opt.get("gap", opt.get("robustness_gap", score_base - score_stress)))

        deltas_stress = opt.get("deltas_stress") or opt.get("stress_deltas") or {}
        deltas_base = opt.get("deltas_base") or opt.get("base_deltas") or {}

        actions = opt.get("actions") or opt.get("bundle_actions") or []

        normalized.append(
            {
                **opt,
                "id": str(opt_id),
                "name": str(name),
                "score_stress": score_stress,
                "score_base": score_base,
                "gap": gap,
                "deltas_stress": dict(deltas_stress),
                "deltas_base": dict(deltas_base),
                "actions": list(actions),
            }
        )

    # Filter by minimum stress score
    viable = [o for o in normalized if o["score_stress"] >= policy.min_score_stress]
    if not viable:
        viable = normalized
        guardrails_triggered.append("All options below min_score_stress; selecting best available under stress.")

    # Sort (Style A)
    def key_fn(o: Dict[str, Any]):
        n_actions = len(o.get("actions", []))
        return (-o["score_stress"], abs(o["gap"]), -o["score_base"], n_actions if policy.prefer_low_changes else 0)

    viable_sorted = sorted(viable, key=key_fn)
    chosen = viable_sorted[0]

    # ----------------------------
    # Bank tie-breaker:
    # If top options are tied on (stress score, abs gap, base score), and KPI integrity flags exist,
    # prefer the option that includes INIT_DQ1 acceleration.
    # ----------------------------
    if len(viable_sorted) >= 2:
        a = viable_sorted[0]
        b = viable_sorted[1]
        tied = (
            abs(a["score_stress"] - b["score_stress"]) <= EPS
            and abs(abs(a["gap"]) - abs(b["gap"])) <= EPS
            and abs(a["score_base"] - b["score_base"]) <= EPS
        )

        if tied and integrity_flags:
            def has_dq(opt: Dict[str, Any]) -> bool:
                for act in opt.get("actions", []):
                    if isinstance(act, dict):
                        tgt = act.get("target_initiative")
                    else:
                        tgt = getattr(act, "target_initiative", None)
                    if tgt == "INIT_DQ1":
                        return True
                return False

            if has_dq(b) and not has_dq(a):
                chosen = b

    # Guardrail on huge robustness gap
    if abs(chosen.get("gap", 0.0)) > policy.max_gap_abs:
        guardrails_triggered.append(f"Large robustness gap detected (gap={chosen['gap']:+.2f}). Require SteerCo review.")

    # Prepare “why not” for rejected (tie-aware)
    rejected: List[Dict[str, Any]] = []
    for r in viable_sorted[1:]:
        reason: List[str] = []

        # Stress score
        if r["score_stress"] < chosen["score_stress"] - EPS:
            reason.append("lower stress robustness score")
        elif abs(r["score_stress"] - chosen["score_stress"]) <= EPS:
            reason.append("tied on stress score")

        # Robustness gap
        if abs(r["gap"]) > abs(chosen["gap"]) + EPS:
            reason.append("less robust (larger stress gap)")
        elif abs(abs(r["gap"]) - abs(chosen["gap"])) <= EPS:
            reason.append("tied on robustness gap")

        # Base score
        if r["score_base"] < chosen["score_base"] - EPS:
            reason.append("lower base-case score")
        elif abs(r["score_base"] - chosen["score_base"]) <= EPS:
            reason.append("tied on base-case score")

        # Change magnitude
        if policy.prefer_low_changes and len(r.get("actions", [])) > len(chosen.get("actions", [])):
            reason.append("more operational change / coordination")

        rejected.append(
            {
                "id": r["id"],
                "name": r["name"],
                "score_stress": r["score_stress"],
                "score_base": r["score_base"],
                "gap": r["gap"],
                "stress_deltas": r["deltas_stress"],
                "why_not": ", ".join(reason) if reason else "ranked lower by policy",
            }
        )

    return chosen, rejected, guardrails_triggered


def build_decision_brief(
    plan: Plan,
    as_of: str,
    monitor_text: str,
    portfolio_text: str,
    chosen: Dict[str, Any],
    rejected: List[Dict[str, Any]],
    guardrails_triggered: List[str],
    integrity_flags: Optional[List[str]] = None,
) -> DecisionBrief:
    integrity_flags = integrity_flags or []

    try:
        approvals = list(plan.meta.governance.decision_rights.approve)  # type: ignore[attr-defined]
    except Exception:
        approvals = ["Transformation Lead", "IT Ops Head", "Risk/Compliance"]

    stress_deltas = chosen.get("deltas_stress", {}) or {}
    base_deltas = chosen.get("deltas_base", {}) or {}

    rationale = [
        "Selected the most robust option under stress (bank-realistic policy).",
        f"Highest stress score among viable options (score_stress={chosen.get('score_stress', 0.0):.2f}).",
        f"Robustness gap is acceptable (gap={chosen.get('gap', 0.0):+.2f}).",
    ]
    if integrity_flags:
        rationale.append("KPI governance flags detected — prioritizing low-regret actions while validation runs.")

    headline = "Steering recommendation: prioritize robustness and risk reduction"

    next_actions = [
        "Proceed with the selected option as a controlled steering decision (documented, auditable).",
        "Track KPI confidence and data quality signals; re-baseline if KPI definition/lineage changes.",
        "Re-run portfolio scoring next cycle with updated evidence (KPIs + initiative progress).",
    ]

    evidence = [
        "Monitor snapshot (evidence):",
        monitor_text.strip(),
        "Portfolio options (evidence):",
        portfolio_text.strip(),
        f"Chosen option stress deltas: {_format_deltas({k: float(v) for k, v in stress_deltas.items()})}",
        f"Chosen option base deltas:   {_format_deltas({k: float(v) for k, v in base_deltas.items()})}",
    ]

    metrics = {
        "chosen": {
            "id": chosen.get("id"),
            "name": chosen.get("name"),
            "score_stress": chosen.get("score_stress"),
            "score_base": chosen.get("score_base"),
            "gap": chosen.get("gap"),
            "stress_deltas": stress_deltas,
            "base_deltas": base_deltas,
        },
        "policy": "Style A — bank-realistic (robustness-first)",
    }

    return DecisionBrief(
        as_of=as_of,
        headline=headline,
        recommended_option_id=str(chosen.get("id")),
        recommended_option_name=str(chosen.get("name")),
        decision_rationale=rationale,
        evidence=evidence,
        guardrails_triggered=guardrails_triggered + integrity_flags,
        approvals_required=approvals,
        next_actions=next_actions,
        rejected_options=rejected,
        metrics=metrics,
    )


def write_audit_record(
    out_path: str,
    plan: Plan,
    as_of: str,
    kpis_df: pd.DataFrame,
    initiatives_df: pd.DataFrame,
    monitor_text: str,
    portfolio_text: str,
    decision: DecisionBrief,
    extra: Optional[Dict[str, Any]] = None,
) -> str:
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    if hasattr(plan, "model_dump"):
        plan_hash = _sha256_text(json.dumps(plan.model_dump(), sort_keys=True))
    else:
        plan_hash = _sha256_text(repr(plan))

    record = {
        "ts": _iso_now(),
        "as_of": as_of,
        "meta": {
            "org": getattr(plan.meta, "org", None),
            "version": getattr(plan.meta, "version", None),
            "plan_hash": plan_hash,
        },
        "inputs": {
            "kpis_fingerprint": _df_fingerprint(kpis_df),
            "initiatives_fingerprint": _df_fingerprint(initiatives_df),
            "monitor_hash": _sha256_text(monitor_text),
            "portfolio_hash": _sha256_text(portfolio_text),
        },
        "decision": asdict(decision),
        "extra": extra or {},
    }

    with open(out_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

    return out_path
