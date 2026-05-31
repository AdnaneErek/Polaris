# src/narrative.py
"""
Generate SteerCo-grade narratives for action bundles.
"""
from __future__ import annotations

from typing import Dict, List, Optional

from .attribution import OptionAttribution, KPIAttribution
from .schemas import Plan
from .portfolio import OptionResultMC
from .guardrails import GuardrailResult


def generate_bundle_narrative(
    plan: Plan,
    bundle_id: str,
    bundle_name: str,
    attribution: OptionAttribution,
    option_result: OptionResultMC,
    guardrail_result: Optional[GuardrailResult] = None,
) -> str:
    """
    Generate a SteerCo-grade narrative for a bundle.
    
    Format:
    - Expected impact
    - Downside risks (p10)
    - Main drivers
    - Key approvals required
    """
    lines = []
    lines.append(f"## {bundle_name} ({bundle_id})")
    lines.append("")
    
    # Expected impact
    lines.append("### Expected Impact")
    lines.append("")
    
    # Get main KPI deltas
    main_kpis = []
    for kpi_id, kpi_attr in attribution.kpi_attributions.items():
        kpi = None
        for k in plan.kpis:
            if k.id == kpi_id or k.short_name == kpi_id:
                kpi = k
                break
        
        kpi_name = kpi.short_name if kpi else kpi_id
        unit = kpi.unit if kpi else ""
        
        # Get mean delta from option_result
        mean_delta = option_result.stress_delta_summary.get(kpi_id, {}).get("mean", 0.0)
        
        if unit == "%":
            delta_str = f"{mean_delta:+.2f}pp"
        elif unit in ("hours", "h"):
            delta_str = f"{mean_delta:+.2f}h"
        else:
            delta_str = f"{mean_delta:+.2f}{unit}"
        
        main_kpis.append(f"- **{kpi_name}**: {delta_str} at {attribution.main_date}")
    
    if main_kpis:
        lines.extend(main_kpis)
    else:
        lines.append("No significant KPI impacts projected.")
    
    lines.append("")
    
    # Downside risks (p10)
    lines.append("### Downside Risks (p10)")
    lines.append("")
    
    p10_risks = []
    for kpi_id, kpi_attr in attribution.kpi_attributions.items():
        kpi = None
        for k in plan.kpis:
            if k.id == kpi_id or k.short_name == kpi_id:
                kpi = k
                break
        
        kpi_name = kpi.short_name if kpi else kpi_id
        unit = kpi.unit if kpi else ""
        
        # Get p10 delta from option_result
        p10_delta = option_result.stress_delta_summary.get(kpi_id, {}).get("p10", 0.0)
        mean_delta = option_result.stress_delta_summary.get(kpi_id, {}).get("mean", 0.0)
        
        # Risk is if p10 is worse than mean
        if p10_delta < mean_delta:
            if unit == "%":
                risk_str = f"{p10_delta:+.2f}pp (vs {mean_delta:+.2f}pp mean)"
            elif unit in ("hours", "h"):
                risk_str = f"{p10_delta:+.2f}h (vs {mean_delta:+.2f}h mean)"
            else:
                risk_str = f"{p10_delta:+.2f}{unit} (vs {mean_delta:+.2f}{unit} mean)"
            
            p10_risks.append(f"- **{kpi_name}**: {risk_str}")
    
    if p10_risks:
        lines.extend(p10_risks)
    else:
        lines.append("No significant downside risks identified at p10.")
    
    lines.append("")
    
    # Main drivers
    lines.append("### Main Drivers")
    lines.append("")
    
    # Group drivers by type and sum contributions
    driver_summaries: Dict[str, float] = {}
    for kpi_id, kpi_attr in attribution.kpi_attributions.items():
        for driver in kpi_attr.drivers:
            key = f"{driver.driver_type}:{driver.driver_id}"
            driver_summaries[key] = driver_summaries.get(key, 0.0) + abs(driver.contribution)
    
    # Sort by total contribution
    sorted_drivers = sorted(driver_summaries.items(), key=lambda x: x[1], reverse=True)
    
    if sorted_drivers:
        # Show top 5 drivers
        for key, total_contrib in sorted_drivers[:5]:
            driver_type, driver_id = key.split(":", 1)
            
            # Find driver name
            driver_name = driver_id
            if driver_type == "initiative":
                for it in plan.initiatives:
                    if it.id == driver_id:
                        driver_name = it.name
                        break
            elif driver_type == "stress_shock":
                driver_name = f"Stress event: {driver_id}"
            elif driver_type == "dependency_penalty":
                driver_name = f"Dependency penalty: {driver_id}"
            elif driver_type == "action_effect":
                driver_name = f"Action effect: {driver_id}"
            
            lines.append(f"- **{driver_name}** ({driver_type}): {total_contrib:.2f} total impact")
    else:
        lines.append("No drivers identified.")
    
    lines.append("")
    
    # Key approvals required
    lines.append("### Key Approvals Required")
    lines.append("")
    
    approvals = []
    if guardrail_result:
        if guardrail_result.requires_approval and guardrail_result.approval_required_from:
            approvals.extend(guardrail_result.approval_required_from)
        
        for violation in guardrail_result.violations:
            if violation.requires_approval:
                approvals.extend(violation.requires_approval)
    
    if approvals:
        unique_approvals = list(set(approvals))
        for approval in unique_approvals:
            lines.append(f"- {approval}")
    else:
        lines.append("No approvals required.")
    
    lines.append("")
    
    return "\n".join(lines)
