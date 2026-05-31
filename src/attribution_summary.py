# src/attribution_summary.py
"""
Generate concise attribution summaries for console output.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Any

from .attribution import OptionAttribution, KPIAttribution
from .schemas import Plan
from .portfolio import OptionResultMC


def format_attribution_summary_console_simple(
    plan: Plan,
    bundle_id: str,
    bundle_name: str,
    attribution_data: Dict[str, Any],
    option_summary: Dict[str, Any],
    max_drivers: int = 3,
) -> str:
    """
    Generate a concise attribution summary for console output using data from pack.
    
    Format:
    Option A — No-regret:
      Main drivers (mean): RES1 → INC -2.2 (80% of impact), DQ1 → STP +0.0
      Downside (p10): INC -2.4 (vs -2.2 mean), STP -0.1pp
    """
    lines = []
    lines.append(f"{bundle_name} ({bundle_id}):")
    
    # Get stress delta summary from option_summary
    stress_delta_summary = option_summary.get("stress_delta_summary", {})
    
    # Main drivers (mean)
    driver_summaries: Dict[str, Dict[str, float]] = {}  # kpi_id -> {driver_key: abs_contrib}
    
    kpi_attributions = attribution_data.get("kpi_attributions", {})
    for kpi_id, kpi_attr in kpi_attributions.items():
        kpi = None
        for k in plan.kpis:
            if k.id == kpi_id or k.short_name == kpi_id:
                kpi = k
                break
        
        drivers = kpi_attr.get("drivers", [])
        for driver in drivers:
            # Skip stress events and statistical summaries for main drivers
            if driver.get("driver_type") in ("stress_shock", "statistical"):
                continue
            
            key = f"{driver.get('driver_type')}:{driver.get('driver_id')}"
            if kpi_id not in driver_summaries:
                driver_summaries[kpi_id] = {}
            contrib = abs(driver.get("contribution", 0.0))
            driver_summaries[kpi_id][key] = driver_summaries[kpi_id].get(key, 0.0) + contrib
    
    # Format main drivers
    main_drivers = []
    for kpi_id, drivers in driver_summaries.items():
        kpi = None
        for k in plan.kpis:
            if k.id == kpi_id or k.short_name == kpi_id:
                kpi = k
                break
        
        kpi_name = kpi.short_name if kpi else kpi_id
        unit = kpi.unit if kpi else ""
        
        # Get mean delta from stress_delta_summary
        mean_delta = stress_delta_summary.get(kpi_id, {}).get("mean", 0.0)
        
        # Sort drivers by contribution
        sorted_drivers = sorted(drivers.items(), key=lambda x: x[1], reverse=True)
        
        if sorted_drivers:
            # Get top driver
            top_key, top_contrib = sorted_drivers[0]
            driver_type, driver_id = top_key.split(":", 1)
            
            # Find driver name
            driver_name = driver_id
            if driver_type == "initiative":
                for it in plan.initiatives:
                    if it.id == driver_id:
                        driver_name = it.name[:30]  # Truncate long names
                        break
            elif driver_type == "dependency_penalty":
                driver_name = f"Dep penalty ({driver_id[:20]})"
            elif driver_type == "action_effect":
                driver_name = f"Action: {driver_id[:20]}"
            
            # Calculate percentage of KPI impact (per-KPI normalization)
            total_abs_impact = sum(drivers.values())
            pct = (top_contrib / total_abs_impact * 100) if total_abs_impact > 0 else 0.0
            
            if unit == "%":
                delta_str = f"{mean_delta:+.2f}pp"
            elif unit in ("hours", "h"):
                delta_str = f"{mean_delta:+.2f}h"
            else:
                delta_str = f"{mean_delta:+.2f}{unit}"
            
            main_drivers.append(f"{driver_name} → {kpi_name} {delta_str} ({pct:.0f}% of {kpi_name} contribution)")
    
    if main_drivers:
        lines.append(f"  Main drivers (mean): {', '.join(main_drivers[:max_drivers])}")
    else:
        lines.append("  Main drivers (mean): No significant drivers identified")
    
    # Downside risks (p10)
    p10_risks = []
    for kpi_id, kpi_attr in kpi_attributions.items():
        kpi = None
        for k in plan.kpis:
            if k.id == kpi_id or k.short_name == kpi_id:
                kpi = k
                break
        
        kpi_name = kpi.short_name if kpi else kpi_id
        unit = kpi.unit if kpi else ""
        
        p10_delta = stress_delta_summary.get(kpi_id, {}).get("p10", 0.0)
        mean_delta = stress_delta_summary.get(kpi_id, {}).get("mean", 0.0)
        
        # Risk is if p10 is worse than mean
        if p10_delta < mean_delta:
            if unit == "%":
                risk_str = f"{kpi_name} {p10_delta:+.2f}pp (vs {mean_delta:+.2f}pp mean)"
            elif unit in ("hours", "h"):
                risk_str = f"{kpi_name} {p10_delta:+.2f}h (vs {mean_delta:+.2f}h mean)"
            else:
                risk_str = f"{kpi_name} {p10_delta:+.2f}{unit} (vs {mean_delta:+.2f}{unit} mean)"
            
            p10_risks.append(risk_str)
    
    if p10_risks:
        lines.append(f"  Downside (p10): {', '.join(p10_risks)}")
    else:
        lines.append("  Downside (p10): No significant downside risks")
    
    return "\n".join(lines)


def format_attribution_summary_console(
    plan: Plan,
    bundle_id: str,
    bundle_name: str,
    attribution: OptionAttribution,
    option_result: OptionResultMC,
    max_drivers: int = 3,
) -> str:
    """
    Generate a concise attribution summary for console output.
    
    Format:
    Option A — No-regret:
      Main drivers (mean): RES1 → INC -2.2 (80% of impact), DQ1 → STP +0.0
      Downside (p10): INC -2.4 (vs -2.2 mean), STP -0.1pp
    """
    lines = []
    lines.append(f"{bundle_name} ({bundle_id}):")
    
    # Main drivers (mean)
    driver_summaries: Dict[str, Dict[str, float]] = {}  # kpi_id -> {driver_key: abs_contrib}
    
    for kpi_id, kpi_attr in attribution.kpi_attributions.items():
        kpi = None
        for k in plan.kpis:
            if k.id == kpi_id or k.short_name == kpi_id:
                kpi = k
                break
        
        kpi_name = kpi.short_name if kpi else kpi_id
        
        for driver in kpi_attr.drivers:
            # Skip stress events and statistical summaries for main drivers
            if driver.driver_type in ("stress_shock", "statistical"):
                continue
            
            key = f"{driver.driver_type}:{driver.driver_id}"
            if kpi_id not in driver_summaries:
                driver_summaries[kpi_id] = {}
            driver_summaries[kpi_id][key] = driver_summaries[kpi_id].get(key, 0.0) + abs(driver.contribution)
    
    # Format main drivers
    main_drivers = []
    for kpi_id, drivers in driver_summaries.items():
        kpi = None
        for k in plan.kpis:
            if k.id == kpi_id or k.short_name == kpi_id:
                kpi = k
                break
        
        kpi_name = kpi.short_name if kpi else kpi_id
        unit = kpi.unit if kpi else ""
        
        # Get mean delta
        mean_delta = option_result.stress_delta_summary.get(kpi_id, {}).get("mean", 0.0)
        
        # Sort drivers by contribution
        sorted_drivers = sorted(drivers.items(), key=lambda x: x[1], reverse=True)
        
        if sorted_drivers:
            # Get top driver
            top_key, top_contrib = sorted_drivers[0]
            driver_type, driver_id = top_key.split(":", 1)
            
            # Find driver name
            driver_name = driver_id
            if driver_type == "initiative":
                for it in plan.initiatives:
                    if it.id == driver_id:
                        driver_name = it.name[:30]  # Truncate long names
                        break
            elif driver_type == "dependency_penalty":
                driver_name = f"Dep penalty ({driver_id[:20]})"
            elif driver_type == "action_effect":
                driver_name = f"Action: {driver_id[:20]}"
            
            # Calculate percentage of KPI impact (per-KPI normalization)
            total_abs_impact = sum(drivers.values())
            pct = (top_contrib / total_abs_impact * 100) if total_abs_impact > 0 else 0.0
            
            if unit == "%":
                delta_str = f"{mean_delta:+.2f}pp"
            elif unit in ("hours", "h"):
                delta_str = f"{mean_delta:+.2f}h"
            else:
                delta_str = f"{mean_delta:+.2f}{unit}"
            
            main_drivers.append(f"{driver_name} → {kpi_name} {delta_str} ({pct:.0f}% of {kpi_name} contribution)")
    
    if main_drivers:
        lines.append(f"  Main drivers (mean): {', '.join(main_drivers[:max_drivers])}")
    else:
        lines.append("  Main drivers (mean): No significant drivers identified")
    
    # Downside risks (p10)
    p10_risks = []
    for kpi_id, kpi_attr in attribution.kpi_attributions.items():
        kpi = None
        for k in plan.kpis:
            if k.id == kpi_id or k.short_name == kpi_id:
                kpi = k
                break
        
        kpi_name = kpi.short_name if kpi else kpi_id
        unit = kpi.unit if kpi else ""
        
        p10_delta = option_result.stress_delta_summary.get(kpi_id, {}).get("p10", 0.0)
        mean_delta = option_result.stress_delta_summary.get(kpi_id, {}).get("mean", 0.0)
        
        # Risk is if p10 is worse than mean
        if p10_delta < mean_delta:
            if unit == "%":
                risk_str = f"{kpi_name} {p10_delta:+.2f}pp (vs {mean_delta:+.2f}pp mean)"
            elif unit in ("hours", "h"):
                risk_str = f"{kpi_name} {p10_delta:+.2f}h (vs {mean_delta:+.2f}h mean)"
            else:
                risk_str = f"{kpi_name} {p10_delta:+.2f}{unit} (vs {mean_delta:+.2f}{unit} mean)"
            
            p10_risks.append(risk_str)
    
    if p10_risks:
        lines.append(f"  Downside (p10): {', '.join(p10_risks)}")
    else:
        lines.append("  Downside (p10): No significant downside risks")
    
    return "\n".join(lines)
