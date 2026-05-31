# src/narrative_template_ai.py
"""
Template-based Natural Language Generation for SteerCo narratives.

Uses intelligent templates with variation to generate human-like narratives.
Zero cost, no API calls required.
"""
from __future__ import annotations

import random
from typing import Dict, List, Optional, Any

from .attribution import OptionAttribution, KPIAttribution
from .schemas import Plan, KPI
from .portfolio import OptionResultMC
from .guardrails import GuardrailResult


# Template library with variations
TEMPLATES = {
    "opening": [
        "This strategic option presents a {risk_level} approach to achieving our {objective} objectives.",
        "Our analysis indicates this bundle offers {impact_level} potential for {objective} outcomes.",
        "This option balances {strength} with {caution} considerations.",
        "Evaluated against our strategic priorities, this option demonstrates {characteristic} potential.",
        "This bundle represents a {strategy_type} strategy designed to address {focus_area}.",
    ],
    "impact_intro": [
        "Expected impact:",
        "Projected outcomes:",
        "Key performance improvements:",
        "Anticipated results:",
    ],
    "impact_kpi": [
        "{kpi_name} is projected to {direction} by {magnitude} by {date}.",
        "Primary benefit: {kpi_name} improvement of {magnitude}, driven primarily by {driver}.",
        "{kpi_name} shows a {direction} trajectory with an expected change of {magnitude}.",
        "The {kpi_name} metric is expected to {direction} by approximately {magnitude}.",
    ],
    "driver": [
        "The main driver of this option is {driver_name}, contributing {contribution_pct}% of the total impact.",
        "{driver_name} represents the primary lever, accounting for {contribution_pct}% of projected gains.",
        "Impact is primarily driven by {driver_name}, which contributes {contribution_pct}% to the overall outcome.",
        "{driver_name} serves as the key enabler, responsible for {contribution_pct}% of the expected improvement.",
    ],
    "risk_intro": [
        "Downside risks (p10 scenario):",
        "Under adverse conditions:",
        "Risk assessment (worst-case 10th percentile):",
        "Downside scenario analysis:",
    ],
    "risk_kpi": [
        "{kpi_name} may only {direction} by {p10_value}, {gap} below mean expectations.",
        "In the p10 scenario, {kpi_name} improvement could be limited to {p10_value}, representing a {gap} shortfall.",
        "Downside risk: {kpi_name} could underperform by {gap}, reaching only {p10_value} in adverse conditions.",
        "Under stress, {kpi_name} may achieve only {p10_value}, falling {gap} short of baseline projections.",
    ],
    "risk_general": [
        "Key risk: {risk_factor} could reduce expected gains by up to {reduction}%.",
        "Primary concern: {risk_factor} may impact outcomes, potentially reducing benefits by {reduction}%.",
        "Risk factor: {risk_factor} presents a {severity} challenge that could diminish returns by {reduction}%.",
    ],
    "robustness": [
        "Under stress conditions, this option maintains {robustness}% of baseline performance.",
        "Robustness analysis indicates {robustness}% resilience compared to baseline scenarios.",
        "Stress testing reveals {robustness}% performance retention under adverse conditions.",
    ],
    "approval_intro": [
        "Key approvals required:",
        "Governance requirements:",
        "Approval process:",
    ],
    "approval_item": [
        "{approver} due to {reason}.",
        "{approver} is required given {reason}.",
        "Approval from {approver} necessary because of {reason}.",
    ],
    "approval_evidence": [
        "Required evidence: {evidence}.",
        "Supporting documentation needed: {evidence}.",
        "Evidence requirements: {evidence}.",
    ],
    "closing": [
        "Overall, this option represents a {summary} strategy for {objective}.",
        "Recommendation: {recommendation} given {key_factor}.",
        "In summary, this bundle offers a {summary} path forward, balancing {balance}.",
        "Conclusion: This option provides a {summary} approach, with {strength} as the primary advantage.",
    ],
}


def _get_kpi_info(plan: Plan, kpi_id: str) -> tuple[str, str, str]:
    """Get KPI name, unit, and direction."""
    for kpi in plan.kpis:
        if kpi.id == kpi_id or kpi.short_name == kpi_id:
            # Get direction from OKRs
            direction = "up"  # default
            for obj in plan.objectives:
                for okr in obj.okrs:
                    if okr.kpi_id == kpi_id:
                        direction = okr.direction
                        break
            return kpi.short_name, kpi.unit, direction
    return kpi_id, "", "up"


def _format_delta(delta: float, unit: str) -> str:
    """Format delta value with appropriate unit."""
    if unit == "%":
        return f"{delta:+.2f}pp"
    elif unit in ("hours", "h"):
        return f"{delta:+.2f}h"
    else:
        return f"{delta:+.2f}{unit}"


def _get_risk_level(score_cvar: float) -> str:
    """Determine risk level from CVaR score."""
    if score_cvar > 2.0:
        return "aggressive"
    elif score_cvar > 0.5:
        return "balanced"
    elif score_cvar > -0.5:
        return "moderate"
    else:
        return "conservative"


def _get_impact_level(score_mean: float) -> str:
    """Determine impact level from mean score."""
    if score_mean > 3.0:
        return "high"
    elif score_mean > 1.0:
        return "moderate"
    elif score_mean > 0:
        return "modest"
    else:
        return "limited"


def _get_top_drivers(attribution: OptionAttribution, plan: Plan, top_n: int = 3) -> List[Dict[str, Any]]:
    """Extract top drivers from attribution."""
    driver_contributions: Dict[str, float] = {}
    driver_names: Dict[str, str] = {}
    
    for kpi_id, kpi_attr in attribution.kpi_attributions.items():
        for driver in kpi_attr.drivers:
            if driver.driver_type == "initiative":
                # Find initiative name
                driver_name = driver.driver_id
                for it in plan.initiatives:
                    if it.id == driver.driver_id:
                        driver_name = it.name
                        break
                driver_names[driver.driver_id] = driver_name
            else:
                driver_name = driver.driver_id
                driver_names[driver.driver_id] = driver_name
            
            key = f"{driver.driver_type}:{driver.driver_id}"
            driver_contributions[key] = driver_contributions.get(key, 0.0) + abs(driver.contribution)
    
    # Sort and get top N
    sorted_drivers = sorted(driver_contributions.items(), key=lambda x: x[1], reverse=True)
    
    result = []
    total_contrib = sum(driver_contributions.values())
    
    for key, contrib in sorted_drivers[:top_n]:
        driver_type, driver_id = key.split(":", 1)
        pct = (contrib / total_contrib * 100) if total_contrib > 0 else 0.0
        result.append({
            "name": driver_names.get(driver_id, driver_id),
            "type": driver_type,
            "contribution": contrib,
            "percentage": pct,
        })
    
    return result


def generate_bundle_narrative_template(
    plan: Plan,
    bundle_id: str,
    bundle_name: str,
    attribution: OptionAttribution,
    option_result: OptionResultMC,
    guardrail_result: Optional[GuardrailResult] = None,
) -> str:
    """
    Generate a SteerCo-grade narrative using template-based NLG.
    
    This creates human-like narratives without requiring LLM APIs.
    """
    lines = []
    
    # Header
    lines.append(f"## {bundle_name} ({bundle_id})")
    lines.append("")
    
    # Opening paragraph
    risk_level = _get_risk_level(option_result.score_stress_cvar10)
    impact_level = _get_impact_level(option_result.score_stress_mean)
    
    opening = random.choice(TEMPLATES["opening"]).format(
        risk_level=risk_level,
        objective="strategic",
        impact_level=impact_level,
        strength="robustness",
        caution="risk management",
        characteristic="promising",
        strategy_type="pragmatic",
        focus_area="operational excellence",
    )
    lines.append(opening)
    lines.append("")
    
    # Expected Impact section
    lines.append("### Expected Impact")
    lines.append("")
    lines.append(random.choice(TEMPLATES["impact_intro"]))
    lines.append("")
    
    # KPI impacts
    main_kpis = []
    for kpi_id, kpi_attr in attribution.kpi_attributions.items():
        kpi_name, unit, kpi_direction = _get_kpi_info(plan, kpi_id)
        mean_delta = option_result.stress_delta_summary.get(kpi_id, {}).get("mean", 0.0)
        
        if abs(mean_delta) < 0.01:
            continue
        
        # Determine if this is improvement based on KPI direction
        # For "up" KPIs: positive delta = improve, negative = decline
        # For "down" KPIs: negative delta = improve, positive = decline
        if kpi_direction == "down":
            is_improvement = mean_delta < 0
        else:  # "up" is default
            is_improvement = mean_delta > 0
        
        direction = "improve" if is_improvement else "decline"
        magnitude = _format_delta(abs(mean_delta), unit)
        
        # Find top driver for this KPI
        top_driver = None
        for driver in kpi_attr.drivers:
            if driver.driver_type == "initiative" and abs(driver.contribution) > 0.1:
                for it in plan.initiatives:
                    if it.id == driver.driver_id:
                        top_driver = it.name
                        break
                if top_driver:
                    break
        
        driver_name = top_driver or "initiative acceleration"
        
        impact_text = random.choice(TEMPLATES["impact_kpi"]).format(
            kpi_name=kpi_name,
            direction=direction,
            magnitude=magnitude,
            date=attribution.main_date,
            driver=driver_name,
        )
        main_kpis.append(impact_text)
    
    if main_kpis:
        lines.extend(main_kpis)
    else:
        lines.append("No significant KPI impacts projected.")
    
    lines.append("")
    
    # Main drivers
    top_drivers = _get_top_drivers(attribution, plan, top_n=3)
    if top_drivers:
        lines.append("### Main Drivers")
        lines.append("")
        for driver in top_drivers:
            driver_text = random.choice(TEMPLATES["driver"]).format(
                driver_name=driver["name"],
                contribution_pct=int(driver["percentage"]),
            )
            lines.append(driver_text)
        lines.append("")
    
    # Downside Risks section
    lines.append("### Downside Risks (p10)")
    lines.append("")
    lines.append(random.choice(TEMPLATES["risk_intro"]))
    lines.append("")
    
    p10_risks = []
    for kpi_id, kpi_attr in attribution.kpi_attributions.items():
        kpi_name, unit, kpi_direction = _get_kpi_info(plan, kpi_id)
        p10_delta = option_result.stress_delta_summary.get(kpi_id, {}).get("p10", 0.0)
        mean_delta = option_result.stress_delta_summary.get(kpi_id, {}).get("mean", 0.0)
        
        # For "down" KPIs, improvement is negative delta, so p10 < mean is worse
        # For "up" KPIs, improvement is positive delta, so p10 < mean is worse
        if kpi_direction == "down":
            is_worse = p10_delta > mean_delta  # More positive = worse for down KPIs
        else:
            is_worse = p10_delta < mean_delta  # Less positive = worse for up KPIs
        
        if not is_worse:
            continue  # No downside risk
        
        gap = abs(mean_delta - p10_delta)
        direction = "improve" if (p10_delta < 0 and kpi_direction == "down") or (p10_delta > 0 and kpi_direction == "up") else "decline"
        p10_value = _format_delta(abs(p10_delta), unit)
        gap_str = _format_delta(gap, unit)
        
        risk_text = random.choice(TEMPLATES["risk_kpi"]).format(
            kpi_name=kpi_name,
            direction=direction,
            p10_value=p10_value,
            gap=gap_str,
        )
        p10_risks.append(risk_text)
    
    if p10_risks:
        lines.extend(p10_risks)
    else:
        lines.append("No significant downside risks identified at p10.")
    
    lines.append("")
    
    # Robustness (if available)
    if option_result.score_base_mean != 0:
        # Calculate robustness percentage, clamped to reasonable range
        ratio = option_result.score_stress_mean / option_result.score_base_mean
        robustness_pct = int(max(0, min(100, ratio * 100)))  # Clamp to 0-100%
        if robustness_pct > 0:  # Only show if meaningful
            robustness_text = random.choice(TEMPLATES["robustness"]).format(
                robustness=robustness_pct
            )
            lines.append(robustness_text)
            lines.append("")
    
    # Key Approvals Required
    lines.append("### Key Approvals Required")
    lines.append("")
    
    approvals = []
    if guardrail_result:
        if guardrail_result.requires_approval and guardrail_result.approval_required_from:
            for approver in guardrail_result.approval_required_from:
                approvals.append({
                    "approver": approver,
                    "reason": "guardrail requirements",
                })
        
        for violation in guardrail_result.violations:
            if violation.requires_approval:
                for approver in violation.requires_approval:
                    reason = violation.message[:50] + "..." if len(violation.message) > 50 else violation.message
                    approvals.append({
                        "approver": approver,
                        "reason": reason,
                    })
            
            if violation.approval_path:
                approver = violation.approval_path.get("who_can_approve", "Relevant stakeholders")
                evidence = violation.approval_path.get("evidence_required", "Standard documentation")
                approvals.append({
                    "approver": approver,
                    "reason": violation.rule_type,
                    "evidence": evidence,
                })
    
    if approvals:
        lines.append(random.choice(TEMPLATES["approval_intro"]))
        lines.append("")
        # Deduplicate approvals
        seen = set()
        for approval in approvals:
            key = (approval["approver"], approval["reason"])
            if key not in seen:
                seen.add(key)
                approval_text = random.choice(TEMPLATES["approval_item"]).format(
                    approver=approval["approver"],
                    reason=approval["reason"],
                )
                lines.append(f"- {approval_text}")
                
                if "evidence" in approval:
                    evidence_text = random.choice(TEMPLATES["approval_evidence"]).format(
                        evidence=approval["evidence"],
                    )
                    lines.append(f"  - {evidence_text}")
    else:
        lines.append("No approvals required.")
    
    lines.append("")
    
    # Closing paragraph
    summary = "pragmatic" if risk_level == "balanced" else risk_level
    recommendation = "proceed with caution" if risk_level in ("aggressive", "moderate") else "recommended"
    key_factor = "moderate risk profile" if risk_level == "balanced" else f"{risk_level} risk profile"
    
    closing = random.choice(TEMPLATES["closing"]).format(
        summary=summary,
        objective="strategic execution",
        recommendation=recommendation,
        key_factor=key_factor,
        balance="risk and reward",
        strength="robustness" if option_result.score_stress_cvar10 > 0 else "stability",
    )
    lines.append(closing)
    lines.append("")
    
    return "\n".join(lines)
