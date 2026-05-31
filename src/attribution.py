# src/attribution.py
"""
Attribution tracking for KPI deltas.

Tracks which initiatives, actions, and stress events contribute to KPI changes.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .schemas import Plan


@dataclass
class AttributionDriver:
    """A single driver of KPI change."""
    driver_type: str  # "initiative", "dependency_penalty", "stress_shock", "action_effect"
    driver_id: str  # initiative_id, stress_event_id, or action_id
    contribution: float  # Amount contributed to KPI delta
    description: str  # Human-readable description


@dataclass
class KPIAttribution:
    """Attribution breakdown for a single KPI."""
    kpi_id: str
    total_delta: float
    drivers: List[AttributionDriver]
    
    def get_driver_summary(self) -> Dict[str, float]:
        """Group contributions by driver type."""
        summary: Dict[str, float] = {}
        for driver in self.drivers:
            key = f"{driver.driver_type}:{driver.driver_id}"
            summary[key] = summary.get(key, 0.0) + driver.contribution
        return summary


@dataclass
class OptionAttribution:
    """Attribution for a complete option/bundle."""
    bundle_id: str
    main_date: str
    kpi_attributions: Dict[str, KPIAttribution]  # kpi_id -> attribution
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "bundle_id": self.bundle_id,
            "main_date": self.main_date,
            "kpi_attributions": {
                kpi_id: {
                    "kpi_id": attr.kpi_id,
                    "total_delta": attr.total_delta,
                    "drivers": [
                        {
                            "driver_type": d.driver_type,
                            "driver_id": d.driver_id,
                            "contribution": d.contribution,
                            "description": d.description,
                        }
                        for d in attr.drivers
                    ],
                }
                for kpi_id, attr in self.kpi_attributions.items()
            },
        }


def format_attribution_text(plan: Plan, attribution: OptionAttribution) -> str:
    """
    Format attribution as human-readable text.
    
    Example:
    KPI_STP delta = +2.5pp explained by:
      - INIT_A1 (+3.2pp from automation progress)
      - INIT_DQ1 (+1.1pp from data quality improvements)
      - Dependency penalty (-1.8pp from INIT_DQ1 blocking)
    """
    lines = []
    lines.append(f"Attribution for {attribution.bundle_id} at {attribution.main_date}:")
    lines.append("-" * 80)
    
    for kpi_id, kpi_attr in attribution.kpi_attributions.items():
        # Find KPI name
        kpi = None
        for k in plan.kpis:
            if k.id == kpi_id or k.short_name == kpi_id:
                kpi = k
                break
        
        kpi_name = kpi.short_name if kpi else kpi_id
        unit = kpi.unit if kpi else ""
        
        # Format delta
        if unit == "%":
            delta_str = f"{kpi_attr.total_delta:+.2f}pp"
        elif unit in ("hours", "h"):
            delta_str = f"{kpi_attr.total_delta:+.2f}h"
        else:
            delta_str = f"{kpi_attr.total_delta:+.2f}{unit}"
        
        lines.append(f"\n{kpi_name} (delta = {delta_str}):")
        
        if not kpi_attr.drivers:
            lines.append("  No drivers identified.")
        else:
            # Group by driver type
            by_type: Dict[str, List[AttributionDriver]] = {}
            for driver in kpi_attr.drivers:
                by_type.setdefault(driver.driver_type, []).append(driver)
            
            # Initiative contributions
            if "initiative" in by_type:
                for driver in by_type["initiative"]:
                    # Find initiative name
                    init_name = driver.driver_id
                    for it in plan.initiatives:
                        if it.id == driver.driver_id:
                            init_name = it.name
                            break
                    
                    contrib_str = f"{driver.contribution:+.2f}{unit}" if unit else f"{driver.contribution:+.2f}"
                    lines.append(f"  - {init_name} ({contrib_str}): {driver.description}")
            
            # Dependency penalties
            if "dependency_penalty" in by_type:
                for driver in by_type["dependency_penalty"]:
                    contrib_str = f"{driver.contribution:+.2f}{unit}" if unit else f"{driver.contribution:+.2f}"
                    lines.append(f"  - Dependency penalty ({contrib_str}): {driver.description}")
            
            # Stress shocks
            if "stress_shock" in by_type:
                for driver in by_type["stress_shock"]:
                    contrib_str = f"{driver.contribution:+.2f}{unit}" if unit else f"{driver.contribution:+.2f}"
                    lines.append(f"  - Stress event ({contrib_str}): {driver.description}")
            
            # Action effects
            if "action_effect" in by_type:
                for driver in by_type["action_effect"]:
                    contrib_str = f"{driver.contribution:+.2f}{unit}" if unit else f"{driver.contribution:+.2f}"
                    lines.append(f"  - Action effect ({contrib_str}): {driver.description}")
    
    return "\n".join(lines)
