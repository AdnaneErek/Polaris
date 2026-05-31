# src/steerco_pack.py
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.actions import SteeringAction
from src.attribution import AttributionDriver, KPIAttribution, OptionAttribution, format_attribution_text
from src.attribution_trace import extract_attribution_for_bundle
from src.forecast import ForecastConfig, forecast_kpis
from src.guardrails import check_portfolio_guardrails, GuardrailResult, GuardrailViolation
from src.kpi_integrity import compute_kpi_integrity, format_integrity_report
from src.load_plan import load_plan
from src.portfolio import (
    ActionBundle,
    OptionResultMC,
    evaluate_bundles_mc,
    format_option_table_mc,
)
from src.whatif import StressEvent, WhatIfStochasticConfig, WhatIfConfig
from src.audit_log import (
    AuditPayload,
    df_sha256,
    file_sha256,
    json_sha256,
    write_audit_bundle,
    get_code_version_hash,
    get_python_version,
    get_dependency_snapshot,
)
from src.weekly_baseline import build_weekly_objectives, compute_forecast_deviation_alerts


# -----------------------------
# Pack structures
# -----------------------------
@dataclass(frozen=True)
class Recommendation:
    chosen_bundle_id: str
    chosen_bundle_name: str
    confidence: float  # 0..1
    why: List[str]
    alternatives: List[Dict[str, str]]  # [{bundle_id, bundle_name, reason}]
    guardrails_passed: bool = True
    guardrails_violations: List[Dict[str, Any]] = field(default_factory=list)  # Serialized GuardrailViolation


@dataclass(frozen=True)
class SteerCoPack:
    as_of: str
    horizon_end: str
    integrity_report: str
    integrity_scores: Dict[str, float]  # kpi_id -> 0..1
    forecast_head: List[Dict[str, Any]]  # small sample
    options_table_mc: str
    recommendation: Recommendation
    guardrails_report: str  # Formatted guardrails section
    options_explain: Dict[str, Dict[str, Any]]  # bundle_id -> attribution data
    option_narratives: Dict[str, str]  # bundle_id -> narrative text

    # raw-ish option objects for JSON (light)
    options_mc: List[Dict[str, Any]]
    ml_anomalies: Optional[Dict[str, Dict[str, Any]]] = field(default=None)  # kpi_id -> ML anomaly result
    weekly_objectives: Optional[List[Dict[str, Any]]] = field(default=None)  # weekly baseline 2026-2028
    forecast_deviation_alerts: Optional[Dict[str, Dict[str, Any]]] = field(default=None)  # forecast vs weekly baseline
    learning_metrics: Optional[Dict[str, Any]] = field(default=None)  # Learning from outcomes metrics
    situation_summaries: Optional[Dict[str, str]] = field(default=None)  # LLM-generated situation summaries
    bundle_generation: Optional[Dict[str, Any]] = field(default=None)  # provenance: llm/manual/fallback reason


def _apply_forecast_recalibration(
    fc: pd.DataFrame,
    recal_params: Dict[str, Dict[str, float]],
    z: float,
) -> pd.DataFrame:
    """
    Apply lightweight post-forecast recalibration learned from realized outcomes.
    - additive bias correction on forecast
    - interval width scaling via sigma_mult
    """
    if fc is None or fc.empty or not recal_params:
        return fc

    out = fc.copy()
    for kpi_id, p in recal_params.items():
        mask = out["kpi_id"].astype(str) == str(kpi_id)
        if not mask.any():
            continue

        bias = float(p.get("bias", 0.0))
        sigma_mult = float(p.get("sigma_mult", 1.0))
        sigma_mult = float(np.clip(sigma_mult, 0.5, 3.0))

        out.loc[mask, "forecast"] = out.loc[mask, "forecast"].astype(float) + bias
        if "lo" in out.columns and "hi" in out.columns:
            sigma = (out.loc[mask, "hi"].astype(float) - out.loc[mask, "lo"].astype(float)) / max(1e-6, 2.0 * float(z))
            sigma = sigma.clip(lower=0.0) * sigma_mult
            out.loc[mask, "lo"] = out.loc[mask, "forecast"].astype(float) - float(z) * sigma
            out.loc[mask, "hi"] = out.loc[mask, "forecast"].astype(float) + float(z) * sigma
    return out


# -----------------------------
# Guardrails formatting
# -----------------------------
def _format_guardrails_report(
    plan,
    guardrail_results: Dict[str, GuardrailResult],
    bundles: List[ActionBundle],
    results: List[OptionResultMC],
) -> str:
    """
    Format guardrails evaluation report for SteerCo pack.
    """
    lines = []
    lines.append("GUARDRAILS EVALUATION")
    lines.append("-" * 80)
    
    if not guardrail_results:
        lines.append("No guardrails evaluated.")
        return "\n".join(lines)
    
    # Group by bundle
    for result in results:
        bundle_id = result.bundle_id
        bundle = next((b for b in bundles if b.id == bundle_id), None)
        bundle_name = bundle.name if bundle else bundle_id
        
        gr = guardrail_results.get(bundle_id)
        if not gr:
            continue
        
        status = "PASS" if gr.passed else "FAIL"
        lines.append(f"\n{bundle_name} ({bundle_id}): {status}")
        
        if not gr.violations:
            lines.append("  No violations detected.")
        else:
            # Sort violations: ERRORs first, then WARNINGs, then others
            sorted_violations = sorted(
                gr.violations,
                key=lambda v: (0 if v.severity == "error" else 1 if v.severity == "warning" else 2)
            )
            
            for v in sorted_violations:
                # For PASS options, change "NO" control coverage messages from WARNING to INFO
                if gr.passed and v.rule_type == "control" and "NO" in v.message and v.severity == "warning":
                    severity_marker = "INFO"
                else:
                    severity_marker = "ERROR" if v.severity == "error" else "WARNING" if v.severity == "warning" else "INFO"
                lines.append(f"  [{severity_marker}] {v.message}")
                if v.actual_value is not None and v.limit_value is not None:
                    lines.append(f"      Actual: {v.actual_value:.2f}, Limit: {v.limit_value:.2f}")
                
                # Budget-specific: show budget by year
                if v.rule_type == "budget" and v.budget_by_year:
                    lines.append(f"      Budget by year:")
                    for year in sorted(v.budget_by_year.keys()):
                        spend = v.budget_by_year[year]
                        limit = next((c.year_limits.get(year, 0) for c in plan.portfolio.constraints if c.type == "budget"), 0)
                        lines.append(f"        {year}: {spend:.2f} EUR_k (limit: {limit:.0f} EUR_k)")
                
                # Approval path
                if v.approval_path:
                    lines.append(f"      Approval path:")
                    # Handle different approval_path structures
                    who_can_approve = v.approval_path.get('owner') or v.approval_path.get('who_can_approve', [])
                    if isinstance(who_can_approve, list):
                        lines.append(f"        Who can approve: {', '.join(who_can_approve)}")
                    elif isinstance(who_can_approve, str):
                        lines.append(f"        Who can approve: {who_can_approve}")
                    else:
                        lines.append(f"        Who can approve: {str(who_can_approve)}")
                    
                    evidence_required = v.approval_path.get('required_evidence') or v.approval_path.get('evidence_required', [])
                    if evidence_required:
                        lines.append(f"        Evidence required:")
                        if isinstance(evidence_required, list):
                            for evidence in evidence_required:
                                lines.append(f"          - {evidence}")
                        else:
                            lines.append(f"          - {evidence_required}")
                    else:
                        lines.append(f"        Evidence required: Standard documentation")
                
                if v.requires_approval:
                    lines.append(f"      Requires approval from: {', '.join(v.requires_approval)}")
        
        if gr.requires_approval and gr.approval_required_from:
            lines.append(f"  Approval required from: {', '.join(gr.approval_required_from)}")
    
    return "\n".join(lines)


# -----------------------------
# Simple policy logic
# -----------------------------
def _is_low_risk_bundle(bundle: ActionBundle) -> bool:
    """
    Heuristic "low risk":
      - no split_scope
      - no actions touching INIT_A1 (automation) acceleration
    Tunable; for banks, low-risk typically means governance/data/observability.
    """
    for a in bundle.actions:
        if str(a.type) == "split_scope":
            return False
        if str(a.type) == "accelerate_initiative" and str(a.target_initiative) == "INIT_A1":
            return False
    return True


def _confidence_from_top2(results: List[OptionResultMC]) -> float:
    """
    Convert separation between #1 and #2 into a 0..1 confidence.
    Based on stress CVaR10 margin primarily (risk-averse).
    """
    if not results:
        return 0.0
    if len(results) == 1:
        return 0.7
    r1, r2 = results[0], results[1]
    margin = float(r1.score_stress_cvar10 - r2.score_stress_cvar10)
    # squash into [0,1] with a conservative scale
    return float(np.clip(0.5 + 2.5 * margin, 0.0, 1.0))


def _detect_ties(results: List[OptionResultMC], epsilon: float = 0.01) -> List[List[str]]:
    """
    Detect tied options based on ranking criteria.
    Returns list of tied groups: [[bundle_id1, bundle_id2], [bundle_id3, bundle_id4]]
    """
    if len(results) < 2:
        return []
    
    tied_groups = []
    current_group = [results[0].bundle_id]
    
    for i in range(1, len(results)):
        prev = results[i-1]
        curr = results[i]
        
        # Check if tied on all primary criteria
        cvar_diff = abs(prev.score_stress_cvar10 - curr.score_stress_cvar10)
        mean_diff = abs(prev.score_stress_mean - curr.score_stress_mean)
        base_diff = abs(prev.score_base_mean - curr.score_base_mean)
        
        if cvar_diff < epsilon and mean_diff < epsilon and base_diff < epsilon:
            # Tied - add to current group
            if curr.bundle_id not in current_group:
                current_group.append(curr.bundle_id)
        else:
            # Not tied - finalize current group if it has multiple items
            if len(current_group) > 1:
                tied_groups.append(current_group)
            current_group = [curr.bundle_id]
    
    # Finalize last group
    if len(current_group) > 1:
        tied_groups.append(current_group)
    
    return tied_groups


def _build_recommendation(
    plan,
    results: List[OptionResultMC],
    bundles: List[ActionBundle],
    as_of: str,
    guardrail_results: Optional[Dict[str, GuardrailResult]] = None,
) -> Recommendation:
    """
    Build recommendation with guardrails evaluation and tie detection.
    
    If guardrails fail for top option:
    - Set recommendation to "NONE" or "SteerCo decision required"
    - Ensure at least 1 low-risk alternative is included
    """
    if not results:
        return Recommendation(
            chosen_bundle_id="NONE",
            chosen_bundle_name="No option evaluated",
            confidence=0.0,
            why=["No Monte Carlo results produced."],
            alternatives=[],
            guardrails_passed=True,
            guardrails_violations=[],
        )

    top = results[0]
    conf = _confidence_from_top2(results)
    guardrail_results = guardrail_results or {}
    
    # Detect ties
    tied_groups = _detect_ties(results, epsilon=0.01)

    # Find bundle for top result
    top_bundle = None
    for b in bundles:
        if b.id == top.bundle_id:
            top_bundle = b
            break

    # Check guardrails for top option
    guardrail_result = guardrail_results.get(top.bundle_id)
    if guardrail_result is None and top_bundle:
        # Evaluate guardrails if not already done
        guardrail_result = check_portfolio_guardrails(
            plan=plan,
            bundle=top_bundle,
            option_result=top,
            as_of=as_of,
        )
        guardrail_results[top.bundle_id] = guardrail_result

    def _serialize_violations(gr: Optional[GuardrailResult]) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        if not gr:
            return out
        for v in gr.violations:
            out.append({
                "rule_id": v.rule_id,
                "rule_type": v.rule_type,
                "severity": v.severity,
                "message": v.message,
                "actual_value": v.actual_value,
                "limit_value": v.limit_value,
                "requires_approval": v.requires_approval or [],
            })
        return out

    # Serialize top-option violations
    violations_dict = _serialize_violations(guardrail_result)

    why = [
        f"Ranked #1 by stress CVaR10 (risk-averse) at {top.main_date}.",
        f"stress_CVaR10={top.score_stress_cvar10:.2f}, stress_mean={top.score_stress_mean:.2f}.",
    ]

    # Add tie information if detected (decision-useful format)
    for tied_group in tied_groups:
        if top.bundle_id in tied_group:
            tied_options = []
            for bid in tied_group:
                result = next((r for r in results if r.bundle_id == bid), None)
                bundle = next((b for b in bundles if b.id == bid), None)
                if result and bundle:
                    n_actions = len(bundle.actions)
                    action_summary = []
                    for action in bundle.actions:
                        if action.type == "accelerate_initiative":
                            action_summary.append(f"accelerate {action.target_initiative}")
                        elif action.type == "split_scope":
                            action_summary.append(f"split {action.target_initiative}")
                        elif action.type == "add_capacity":
                            action_summary.append(f"add capacity to {action.target_initiative}")
                    
                    # Get initiative names for context
                    initiative_names = []
                    for action in bundle.actions:
                        for it in plan.initiatives:
                            if it.id == action.target_initiative:
                                initiative_names.append(it.name)
                    
                    tied_options.append({
                        "name": result.bundle_name,
                        "id": bid,
                        "n_actions": n_actions,
                        "actions": action_summary,
                        "initiatives": initiative_names,
                    })
            break
            
            if len(tied_options) == 2:
                opt1, opt2 = tied_options[0], tied_options[1]
                # Build decision-useful message
                tie_msg = f"(Tie) {opt1['name']} and {opt2['name']} are statistically tied at CVaR10 (difference < 0.01). "
                tie_msg += f"Tie-breaker rubric:\n"
                tie_msg += f"  • If optimizing for 'minimum change' → choose {opt1['name'] if opt1['n_actions'] < opt2['n_actions'] else opt2['name']} ({min(opt1['n_actions'], opt2['n_actions'])} action{'s' if min(opt1['n_actions'], opt2['n_actions']) != 1 else ''}, lowest change surface)\n"
                
                # Add what each option offers
                if opt1['n_actions'] != opt2['n_actions']:
                    more_actions = opt1 if opt1['n_actions'] > opt2['n_actions'] else opt2
                    fewer_actions = opt2 if opt1['n_actions'] > opt2['n_actions'] else opt1
                    tie_msg += f"  • If accepting added scope for potential upside → choose {more_actions['name']} ({more_actions['n_actions']} actions"
                    if more_actions['initiatives']:
                        tie_msg += f", includes {', '.join(more_actions['initiatives'][:2])}"
                    tie_msg += ")\n"
                
                why.append(tie_msg)
            else:
                tied_str = ", ".join([opt['name'] for opt in tied_options])
                why.append(f"(Tie) Multiple options ({tied_str}) are statistically tied at CVaR10; "
                          f"decision depends on preference for change vs optional upside.")

    # If guardrails fail for top option, auto-fallback to highest-ranked passing option.
    # If none pass, escalate to explicit SteerCo decision required.
    if guardrail_result and not guardrail_result.passed:
        why.append("⚠️ Guardrails FAILED: Hard constraint violations detected.")

        fallback_result: Optional[OptionResultMC] = None
        fallback_guardrail: Optional[GuardrailResult] = None
        for cand in results[1:]:
            gr = guardrail_results.get(cand.bundle_id)
            if gr is None:
                cand_bundle = next((b for b in bundles if b.id == cand.bundle_id), None)
                if cand_bundle is None:
                    continue
                gr = check_portfolio_guardrails(
                    plan=plan,
                    bundle=cand_bundle,
                    option_result=cand,
                    as_of=as_of,
                )
                guardrail_results[cand.bundle_id] = gr
            if gr and gr.passed:
                fallback_result = cand
                fallback_guardrail = gr
                break

        if fallback_result and fallback_guardrail:
            why.append(
                f"Top-ranked option {top.bundle_name} failed guardrails; "
                f"auto-selected highest-ranked passing option {fallback_result.bundle_name}."
            )
            if fallback_guardrail.requires_approval and fallback_guardrail.approval_required_from:
                why.append(
                    f"⚠️ Guardrails: Requires approval from: {', '.join(fallback_guardrail.approval_required_from)}"
                )

            alternatives = _build_alternatives_with_low_risk(results, bundles, guardrail_results, plan, as_of)
            alternatives = [a for a in alternatives if a.get("bundle_id") != fallback_result.bundle_id][:2]

            return Recommendation(
                chosen_bundle_id=fallback_result.bundle_id,
                chosen_bundle_name=fallback_result.bundle_name,
                confidence=float(np.clip(conf * 0.85, 0.0, 1.0)),
                why=why,
                alternatives=alternatives,
                guardrails_passed=True,
                guardrails_violations=_serialize_violations(fallback_guardrail),
            )

        why.append("Recommendation: SteerCo decision required (see Guardrails section).")
        return Recommendation(
            chosen_bundle_id="NONE",
            chosen_bundle_name="SteerCo decision required (guardrails failed)",
            confidence=0.0,
            why=why,
            alternatives=_build_alternatives_with_low_risk(results, bundles, guardrail_results, plan, as_of),
            guardrails_passed=False,
            guardrails_violations=violations_dict,
        )

    # Pick 2 alternatives; ensure at least one low-risk alternative
    alternatives = _build_alternatives_with_low_risk(results, bundles, guardrail_results, plan, as_of)

    # Escalation (plan policy): confidence below threshold → request SteerCo decision
    thresh = float(plan.agent_policy.escalation.if_confidence_below)
    if conf < thresh:
        # Add explicit "why so low" explanation
        if len(results) >= 2:
            top2_cvar_diff = abs(results[0].score_stress_cvar10 - results[1].score_stress_cvar10)
            if top2_cvar_diff < 0.01:
                why.append(f"Low confidence ({conf:.2f}) because top-2 CVaR10 separation < 0.01 ⇒ low discriminative power.")
        why.append(f"Confidence {conf:.2f} < escalation threshold {thresh:.2f}: escalate to SteerCo for decision.")

    # Add guardrail warnings if any
    if guardrail_result and guardrail_result.requires_approval:
        why.append(f"⚠️ Guardrails: Requires approval from: {', '.join(guardrail_result.approval_required_from)}")

    return Recommendation(
        chosen_bundle_id=top.bundle_id,
        chosen_bundle_name=top.bundle_name,
        confidence=float(conf),
        why=why,
        alternatives=alternatives,
        guardrails_passed=guardrail_result.passed if guardrail_result else True,
        guardrails_violations=violations_dict,
    )


def _build_alternatives_with_low_risk(
    results: List[OptionResultMC],
    bundles: List[ActionBundle],
    guardrail_results: Dict[str, GuardrailResult],
    plan,
    as_of: str,
) -> List[Dict[str, str]]:
    """
    Build alternatives list, ensuring at least 1 low-risk alternative that passes guardrails.
    """
    alternatives: List[Dict[str, str]] = []
    
    # Find low-risk alternatives that pass guardrails
    low_risk_passed = []
    for r in results[1:]:  # Skip top result
        bundle = next((b for b in bundles if b.id == r.bundle_id), None)
        if not bundle:
            continue
        
        if _is_low_risk_bundle(bundle):
            # Check guardrails
            gr = guardrail_results.get(r.bundle_id)
            if gr is None:
                gr = check_portfolio_guardrails(plan, bundle, r, as_of)
                guardrail_results[r.bundle_id] = gr
            
            if gr.passed:
                low_risk_passed.append((r, bundle))
    
    # Add next-best alternative
    if len(results) > 1:
        next_best = results[1]
        alternatives.append({
            "bundle_id": next_best.bundle_id,
            "bundle_name": next_best.bundle_name,
            "reason": "Next-best robustness score under stress (backup plan).",
        })
    
    # Ensure at least 1 low-risk alternative that passes guardrails
    if low_risk_passed:
        low_risk_result, low_risk_bundle = low_risk_passed[0]
        # Only add if not already in alternatives
        if not any(a["bundle_id"] == low_risk_result.bundle_id for a in alternatives):
            alternatives.append({
                "bundle_id": low_risk_result.bundle_id,
                "bundle_name": low_risk_result.bundle_name,
                "reason": "Low-risk alternative emphasizing governance/resilience (guardrails passed).",
            })
    else:
        # Fallback: find any low-risk alternative (even if guardrails have warnings)
        for r in results[1:]:
            bundle = next((b for b in bundles if b.id == r.bundle_id), None)
            if bundle and _is_low_risk_bundle(bundle):
                if not any(a["bundle_id"] == r.bundle_id for a in alternatives):
                    alternatives.append({
                        "bundle_id": r.bundle_id,
                        "bundle_name": r.bundle_name,
                        "reason": "Low-risk alternative emphasizing governance/resilience.",
                    })
                    break
    
    # Cap to 2
    return alternatives[:2]


# -----------------------------
# Public builder
# -----------------------------
def build_steerco_pack(
    plan_path: str,
    kpis_csv: str,
    initiatives_csv: str,
    as_of: str,
    horizon_end: str,
    bundles: List[ActionBundle],
    stress_events: Optional[List[StressEvent]] = None,
    mc_cfg: Optional[WhatIfStochasticConfig] = None,
    whatif_cfg: Optional[WhatIfConfig] = None,
    forecast_cfg: Optional[ForecastConfig] = None,
    out_dir: Optional[str | Path] = None,
    situation_summaries: Optional[Dict[str, str]] = None,
    bundle_generation: Optional[Dict[str, Any]] = None,
) -> SteerCoPack:
    plan = load_plan(plan_path)
    kpis = pd.read_csv(kpis_csv)
    inits = pd.read_csv(initiatives_csv)

    stress_events = stress_events or []
    mc_cfg = mc_cfg or WhatIfStochasticConfig(n_samples=600, seed=7)
    whatif_cfg = whatif_cfg or WhatIfConfig()
    forecast_cfg = forecast_cfg or ForecastConfig()

    # ---------- Learning adapter (pre-score/pre-forecast adaptation inputs) ----------
    learner = None
    learning_storage_dir = str(Path(out_dir) / "learning") if out_dir else "artifacts/learning"
    forecast_recalibration: Dict[str, Dict[str, float]] = {}
    scoring_weight_overrides: Dict[str, float] = {}
    confidence_multiplier: float = 1.0
    try:
        from src.learning import get_learner
        learner = get_learner(storage_dir=learning_storage_dir)
        forecast_recalibration = learner.get_forecast_recalibration_params()
        scoring_weight_overrides = learner.get_improved_scoring_weights()
        confidence_multiplier = learner.get_confidence_calibration_multiplier()
    except (ImportError, Exception):
        learner = None

    # ---------- Integrity ----------
    integrity = compute_kpi_integrity(plan, kpis[["date", "kpi_id", "value"]], as_of=as_of)
    integrity_scores = {k: float(v.score) for k, v in integrity.items()}
    integrity_report = format_integrity_report(integrity, threshold=0.6)
    
    # ---------- ML Anomaly Detection ----------
    ml_anomalies = None
    try:
        from src.monitor_ml import detect_all_kpi_anomalies_ml
        ml_anomalies_dict = detect_all_kpi_anomalies_ml(plan, kpis[["date", "kpi_id", "value"]], as_of)
        # Convert to serializable format
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
    except (ImportError, Exception) as e:
        # ML detection not available - continue without it
        ml_anomalies = None

    # ---------- Forecast ----------
    fc = forecast_kpis(
        plan=plan,
        kpis_df=kpis[["date", "kpi_id", "value"]],
        as_of=as_of,
        horizon_end=horizon_end,
        initiatives_df=inits,          # uses your exogenous features (optional)
        cfg=forecast_cfg,
    )
    fc = _apply_forecast_recalibration(fc, forecast_recalibration, z=float(forecast_cfg.z))
    forecast_head = fc.head(12).to_dict(orient="records")
    # Keep full forecast for visuals (Phase 8.2)
    forecast_full = fc.copy()
    
    # ---------- Weekly objectives baseline + forecast deviation alerts ----------
    weekly_obj_df = build_weekly_objectives(plan, start_date="2026-01-01", end_date="2028-12-31")
    weekly_objectives = weekly_obj_df.to_dict(orient="records")
    forecast_deviation_alerts = compute_forecast_deviation_alerts(
        plan=plan,
        history_df=kpis[["date", "kpi_id", "value"]],
        forecast_df=fc[["date", "kpi_id", "forecast"]],
        weekly_baseline_df=weekly_obj_df,
        as_of=as_of,
    )

    # ---------- Portfolio (MC) ----------
    results_mc = evaluate_bundles_mc(
        plan=plan,
        kpis_df=kpis[["date", "kpi_id", "value"]],
        initiatives_df=inits,
        as_of=as_of,
        dates_to_check=["2027-06-01", "2027-12-01", "2028-12-01"],
        bundles=bundles,
        stress_events=stress_events,
        whatif_cfg=whatif_cfg,
        mc_cfg=mc_cfg,
        integrity_scores=integrity_scores,
        cvar_alpha=0.10,
        scoring_weight_overrides=scoring_weight_overrides if scoring_weight_overrides else None,
    )
    options_table_mc = format_option_table_mc(results_mc)

    # Evaluate guardrails for all options
    guardrail_results: Dict[str, GuardrailResult] = {}
    for result in results_mc:
        bundle = next((b for b in bundles if b.id == result.bundle_id), None)
        if bundle:
            guardrail_results[result.bundle_id] = check_portfolio_guardrails(
                plan=plan,
                bundle=bundle,
                option_result=result,
                as_of=as_of,
            )

    recommendation = _build_recommendation(plan, results_mc, bundles, as_of, guardrail_results)
    # Recalibrate confidence using realized historical performance.
    if confidence_multiplier != 1.0:
        recal_conf = float(np.clip(float(recommendation.confidence) * float(confidence_multiplier), 0.0, 1.0))
        if abs(recal_conf - float(recommendation.confidence)) > 1e-6:
            why = list(recommendation.why)
            why.append(
                f"Confidence recalibrated by learning multiplier {confidence_multiplier:.2f} "
                f"(from {recommendation.confidence:.2f} to {recal_conf:.2f})."
            )
            recommendation = Recommendation(
                chosen_bundle_id=recommendation.chosen_bundle_id,
                chosen_bundle_name=recommendation.chosen_bundle_name,
                confidence=recal_conf,
                why=why,
                alternatives=recommendation.alternatives,
                guardrails_passed=recommendation.guardrails_passed,
                guardrails_violations=recommendation.guardrails_violations,
            )
    
    # ---------- Learning from Outcomes ----------
    learning_metrics = None
    try:
        if learner is None:
            from src.learning import get_learner
            learner = get_learner(storage_dir=learning_storage_dir)
        
        # Record this decision
        top_result = results_mc[0] if results_mc else None
        if top_result:
            # Planned evaluation date for closing the loop (use MC main_date if present, else horizon_end)
            planned_eval_date = getattr(top_result, "main_date", None) or horizon_end

            # Baseline predicted KPI values at planned evaluation date (from deterministic forecast)
            predicted_outcomes = None
            try:
                plan_kpi_ids = [k.id for k in plan.kpis]
                fc_at_eval = fc[(fc["date"].astype(str) == str(planned_eval_date)) & (fc["kpi_id"].astype(str).isin(plan_kpi_ids))].copy()
                if not fc_at_eval.empty and "forecast" in fc_at_eval.columns:
                    predicted_outcomes = {str(r["kpi_id"]): float(r["forecast"]) for _, r in fc_at_eval.iterrows()}
            except Exception:
                predicted_outcomes = None

            learner.record_decision(
                as_of=as_of,
                recommended_bundle_id=recommendation.chosen_bundle_id,
                recommended_bundle_name=recommendation.chosen_bundle_name,
                recommended_score_stress_cvar10=top_result.score_stress_cvar10,
                recommended_score_stress_mean=top_result.score_stress_mean,
                confidence=recommendation.confidence,
                guardrails_passed=recommendation.guardrails_passed,
                planned_evaluation_date=str(planned_eval_date) if planned_eval_date else None,
                predicted_outcomes=predicted_outcomes,
            )
        
        # Get learning metrics
        metrics = learner.get_learning_metrics()
        learning_metrics = {
            "total_decisions": metrics.total_decisions,
            "accepted_decisions": metrics.accepted_decisions,
            "mean_forecast_accuracy": metrics.mean_forecast_accuracy,
            "mean_score_prediction_error": metrics.mean_score_prediction_error,
            "improvement_trend": metrics.improvement_trend,
            "last_updated": metrics.last_updated,
            "confidence_multiplier": float(confidence_multiplier),
            "scoring_weight_overrides": scoring_weight_overrides,
            "forecast_recalibration_kpis": sorted(list(forecast_recalibration.keys())),
        }
    except (ImportError, Exception) as e:
        # Learning not available - continue without it
        learning_metrics = None

    # Format guardrails report
    guardrails_report = _format_guardrails_report(plan, guardrail_results, bundles, results_mc)

    # ---------- Attribution & Narratives ----------
    from src.attribution_mc import compute_attribution_from_raw_mc
    from src.attribution_trace import extract_attribution_for_bundle
    from src.whatif import compare_actions_at_dates_mc_raw
    
    # Choose narrative generator (Gemini if available, else template-based)
    use_gemini = os.getenv("GEMINI_API_KEY") is not None
    if use_gemini:
        try:
            from src.narrative_gemini import generate_bundle_narrative_gemini, GEMINI_AVAILABLE
            if GEMINI_AVAILABLE:
                generate_narrative = lambda *args, **kwargs: generate_bundle_narrative_gemini(*args, **kwargs)
            else:
                raise ImportError("GEMINI_AVAILABLE is False")
        except (ImportError, NameError):
            # Fallback to template if google-generativeai not installed
            from src.narrative_template_ai import generate_bundle_narrative_template
            generate_narrative = lambda *args, **kwargs: generate_bundle_narrative_template(*args, **kwargs)
            use_gemini = False
    else:
        from src.narrative_template_ai import generate_bundle_narrative_template
        generate_narrative = lambda *args, **kwargs: generate_bundle_narrative_template(*args, **kwargs)
    
    options_explain: Dict[str, Dict[str, Any]] = {}
    option_narratives: Dict[str, str] = {}
    
    main_date = "2027-12-01"  # Use main evaluation date
    
    for result in results_mc:
        bundle = next((b for b in bundles if b.id == result.bundle_id), None)
        if not bundle:
            continue
        
        # Extract attribution from raw MC deltas (Phase 5.1)
        # Get raw deltas for this bundle
        raw_stress = compare_actions_at_dates_mc_raw(
            plan=plan,
            kpi_history=kpis[["date", "kpi_id", "value"]],
            initiative_history=inits,
            as_of=as_of,
            dates_to_check=[main_date],
            actions=bundle.actions,
            stress_events=stress_events,
            cfg=whatif_cfg,
            mc=mc_cfg,
            integrity_scores=integrity_scores,
        )
        
        # Compute attribution from raw MC
        attribution = compute_attribution_from_raw_mc(
            plan=plan,
            raw_deltas=raw_stress,
            bundle_actions=bundle.actions,
            main_date=main_date,
        )
        
        # Store attribution (bundle_id is set in to_dict output)
        attr_dict = attribution.to_dict()
        attr_dict["bundle_id"] = result.bundle_id  # Ensure bundle_id is set
        options_explain[result.bundle_id] = attr_dict
        
        # Generate narrative (Gemini if API key available, else template-based)
        gr_result = guardrail_results.get(result.bundle_id)
        try:
            narrative = generate_narrative(
                plan=plan,
                bundle_id=result.bundle_id,
                bundle_name=result.bundle_name,
                attribution=attribution,
                option_result=result,
                guardrail_result=gr_result,
            )
        except ImportError:
            # Fallback to template if Gemini fails at runtime
            from src.narrative_template_ai import generate_bundle_narrative_template
            narrative = generate_bundle_narrative_template(
                plan=plan,
                bundle_id=result.bundle_id,
                bundle_name=result.bundle_name,
                attribution=attribution,
                option_result=result,
                guardrail_result=gr_result,
            )
        option_narratives[result.bundle_id] = narrative

    # light JSON-able options
    options_mc = []
    for r in results_mc:
        options_mc.append(
            {
                "bundle_id": r.bundle_id,
                "bundle_name": r.bundle_name,
                "main_date": r.main_date,
                "score_stress_mean": r.score_stress_mean,
                "score_stress_cvar10": r.score_stress_cvar10,
                "score_base_mean": r.score_base_mean,
                "score_base_cvar10": r.score_base_cvar10,
                "robustness_gap_mean": r.robustness_gap_mean,
                "stress_delta_summary": r.stress_delta_summary,
                "base_delta_summary": r.base_delta_summary,
                "notes": r.notes,
            }
        )

    pack = SteerCoPack(
        as_of=as_of,
        horizon_end=horizon_end,
        integrity_report=integrity_report,
        integrity_scores=integrity_scores,
        forecast_head=forecast_head,
        options_table_mc=options_table_mc,
        recommendation=recommendation,
        guardrails_report=guardrails_report,
        options_explain=options_explain,
        option_narratives=option_narratives,
        options_mc=options_mc,
        ml_anomalies=ml_anomalies,
        weekly_objectives=weekly_objectives,
        forecast_deviation_alerts=forecast_deviation_alerts,
        learning_metrics=learning_metrics,
        situation_summaries=situation_summaries,
        bundle_generation=bundle_generation,
    )

    # ---------- Output (optional) ----------
    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        # JSON pack
        (out_dir / "steerco_pack.json").write_text(
            json.dumps(
                {
                    "as_of": pack.as_of,
                    "horizon_end": pack.horizon_end,
                    "sources": {
                        "plan_path": str(plan_path),
                        "kpis_csv": str(kpis_csv),
                        "initiatives_csv": str(initiatives_csv),
                    },
                    "bundles": [
                        {
                            "id": b.id,
                            "name": b.name,
                            "description": b.description,
                            "actions": [
                                {
                                    "id": a.id,
                                    "type": a.type,
                                    "target_initiative": a.target_initiative,
                                    "parameters": dict(a.parameters),
                                    "description": a.description,
                                }
                                for a in b.actions
                            ],
                        }
                        for b in bundles
                    ],
                    "integrity_report": pack.integrity_report,
                    "integrity_scores": pack.integrity_scores,
                    "forecast_head": pack.forecast_head,
                    "options_table_mc": pack.options_table_mc,
                    "options_mc": pack.options_mc,
                    "recommendation": asdict(pack.recommendation),
                    "guardrails_report": pack.guardrails_report,
                    "options_explain": pack.options_explain,
                    "option_narratives": pack.option_narratives,
                    "ml_anomalies": pack.ml_anomalies,
                    "weekly_objectives": pack.weekly_objectives,
                    "forecast_deviation_alerts": pack.forecast_deviation_alerts,
                    "learning_metrics": pack.learning_metrics,
                    "situation_summaries": pack.situation_summaries,
                    "bundle_generation": pack.bundle_generation,
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        # Markdown narrative
        md = []
        md.append(f"# SteerCo Pack — {as_of}\n")
        md.append("## Integrity\n")
        md.append("```text\n" + pack.integrity_report + "\n```\n")
        
        # ML Anomaly Detection section
        if pack.ml_anomalies:
            md.append("## ML Anomaly Detection\n")
            md.append("```text\n")
            anomaly_lines = []
            for kpi_id, anomaly_data in pack.ml_anomalies.items():
                kpi = next((k for k in plan.kpis if k.id == kpi_id), None)
                kpi_name = kpi.short_name if kpi else kpi_id
                
                if anomaly_data["is_anomaly"]:
                    anomaly_lines.append(
                        f"[ANOMALY] {kpi_name} ({kpi_id}): {anomaly_data['explanation']}\n"
                        f"  Confidence: {anomaly_data['confidence']:.2f}, "
                        f"Anomaly Score: {anomaly_data['anomaly_score']:.2f}, "
                        f"Method: {anomaly_data['method']}"
                    )
                else:
                    anomaly_lines.append(
                        f"[NORMAL] {kpi_name} ({kpi_id}): {anomaly_data['explanation']}"
                    )
            
            if anomaly_lines:
                md.append("\n".join(anomaly_lines))
            else:
                md.append("No ML anomaly detection results available.")
            md.append("\n```\n")

        # Forecast deviation vs weekly objective baseline
        if pack.forecast_deviation_alerts:
            md.append("## Forecast Deviation Alerts (vs Weekly Objective Baseline)\n")
            md.append("```text\n")
            for kpi_id, data in pack.forecast_deviation_alerts.items():
                if data.get("is_deviation_alert", False):
                    md.append(f"[ALERT] {kpi_id}: {data.get('explanation', '')}")
                else:
                    md.append(f"[OK] {kpi_id}: {data.get('explanation', '')}")
            md.append("\n```\n")
        
        # Learning from Outcomes section
        if pack.learning_metrics:
            md.append("## Learning from Outcomes\n")
            md.append("```text\n")
            try:
                from src.learning import get_learner
                learner = get_learner()
                learning_report = learner.format_learning_report()
                md.append(learning_report)
            except Exception:
                md.append("Learning metrics available but report generation failed.")
            md.append("\n```\n")
        
        md.append("## Forecast (head)\n")
        md.append("```text\n" + pd.DataFrame(pack.forecast_head).head(12).to_string(index=False) + "\n```\n")
        md.append("## Options (Monte Carlo)\n")
        md.append("```text\n" + pack.options_table_mc + "\n```\n")
        md.append("## Guardrails\n")
        md.append("```text\n" + pack.guardrails_report + "\n```\n")
        md.append("## Option Narratives\n")
        for bundle_id, narrative in pack.option_narratives.items():
            md.append(narrative)
            md.append("")
        md.append("## Attribution\n")
        for bundle_id, explain in pack.options_explain.items():
            bundle_name = next((b.name for b in bundles if b.id == bundle_id), bundle_id)
            md.append(f"### {bundle_name} ({bundle_id})\n")
            md.append("```text\n")
            from src.attribution import format_attribution_text
            attribution_obj = OptionAttribution(
                bundle_id=bundle_id,
                main_date=explain["main_date"],
                kpi_attributions={
                    kpi_id: KPIAttribution(
                        kpi_id=attr["kpi_id"],
                        total_delta=attr["total_delta"],
                        drivers=[
                            AttributionDriver(
                                driver_type=d["driver_type"],
                                driver_id=d["driver_id"],
                                contribution=d["contribution"],
                                description=d["description"],
                            )
                            for d in attr["drivers"]
                        ],
                    )
                    for kpi_id, attr in explain["kpi_attributions"].items()
                },
            )
            md.append(format_attribution_text(plan, attribution_obj))
            md.append("\n```\n")
            md.append("")
        md.append("## Recommendation\n")
        md.append(f"- **Chosen:** {pack.recommendation.chosen_bundle_name} ({pack.recommendation.chosen_bundle_id})\n")
        md.append(f"- **Confidence:** {pack.recommendation.confidence:.2f}\n")
        md.append(f"- **Guardrails:** {'PASSED' if pack.recommendation.guardrails_passed else 'FAILED'}\n")
        for w in pack.recommendation.why:
            md.append(f"- {w}\n")
        if pack.recommendation.alternatives:
            md.append("\n### Alternatives\n")
            for a in pack.recommendation.alternatives:
                md.append(f"- {a['bundle_name']} ({a['bundle_id']}): {a['reason']}\n")

        (out_dir / "steerco_pack.md").write_text("".join(md), encoding="utf-8")

        # Phase 8.2: Generate visuals
        try:
            from src.visuals import generate_steerco_visuals
            visuals = generate_steerco_visuals(
                plan=plan,
                forecast_df=forecast_full,
                results=results_mc,
                output_dir=out_dir / "visuals",
            )
        except Exception as e:
            # Don't fail if visuals can't be generated
            import warnings
            warnings.warn(f"Could not generate visuals: {e}")

        # Audit bundle
        input_hashes = {
            "plan_sha256": file_sha256(plan_path),
            "kpis_sha256": file_sha256(kpis_csv),
            "initiatives_sha256": file_sha256(initiatives_csv),
            "kpis_df_sha256": df_sha256(kpis[["date", "kpi_id", "value"]]),
            "inits_df_sha256": df_sha256(inits),
        }
        config = {
            "as_of": as_of,
            "horizon_end": horizon_end,
            "whatif_cfg": asdict(whatif_cfg),
            "mc_cfg": asdict(mc_cfg),
            "forecast_cfg": asdict(forecast_cfg),
            "stress_events": [asdict(e) for e in stress_events],
            "bundle_generation": bundle_generation,
            "bundles": [
                {
                    "id": b.id,
                    "name": b.name,
                    "actions": [dict(id=a.id, type=a.type, target=a.target_initiative, parameters=a.parameters) for a in b.actions],
                }
                for b in bundles
            ],
        }
        outputs_hashes = {
            "steerco_pack_json_sha256": file_sha256(out_dir / "steerco_pack.json"),
            "steerco_pack_md_sha256": file_sha256(out_dir / "steerco_pack.md"),
            "pack_payload_sha256": json_sha256(
                {
                    "integrity_scores": pack.integrity_scores,
                    "forecast_head": pack.forecast_head,
                    "options_mc": pack.options_mc,
                    "recommendation": asdict(pack.recommendation),
                    "guardrails_report": pack.guardrails_report,
                    "bundle_generation": pack.bundle_generation,
                }
            ),
        }

        # Phase 7.2: Get audit completeness fields
        code_version_hash = get_code_version_hash()
        python_version = get_python_version()
        dependency_snapshot = get_dependency_snapshot()
        random_seed = mc_cfg.seed if mc_cfg else None

        write_audit_bundle(
            out_dir / "logs",
            payload=AuditPayload(
                as_of=as_of,
                plan_path=str(plan_path),
                kpis_path=str(kpis_csv),
                initiatives_path=str(initiatives_csv),
                input_hashes=input_hashes,
                config=config,
                outputs_hashes=outputs_hashes,
                code_version_hash=code_version_hash,
                python_version=python_version,
                dependency_snapshot=dependency_snapshot,
                random_seed=random_seed,
            ),
            artifacts={
                "pack": {
                    "as_of": pack.as_of,
                    "horizon_end": pack.horizon_end,
                    "integrity_report": pack.integrity_report,
                    "forecast_head": pack.forecast_head,
                    "options_table_mc": pack.options_table_mc,
                    "recommendation": asdict(pack.recommendation),
                    "guardrails_report": pack.guardrails_report,
                    "bundle_generation": pack.bundle_generation,
                }
            },
        )

    return pack
