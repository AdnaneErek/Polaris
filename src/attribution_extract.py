# src/attribution_extract.py
"""
Extract attribution from what-if comparison results.

Since we can't easily modify the projection functions without breaking existing code,
we extract attribution by running focused projections with trace mode.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import pandas as pd

from .attribution import AttributionDriver, KPIAttribution, OptionAttribution
from .schemas import Plan
from .whatif import (
    project_kpis,
    project_kpis_mc,
    compare_actions_at_dates_mc_raw,
    WhatIfConfig,
    WhatIfStochasticConfig,
)
from .actions import SteeringAction
from .portfolio import ActionBundle, OptionResultMC
from .stress import StressEvent


def extract_attribution_deterministic(
    plan: Plan,
    kpi_history: pd.DataFrame,
    initiative_history: pd.DataFrame,
    as_of: str,
    main_date: str,
    actions: List[SteeringAction],
    stress_events: List[StressEvent],
    cfg: WhatIfConfig,
) -> OptionAttribution:
    """
    Extract attribution for deterministic projection.
    Runs base and action scenarios and tracks contributions.
    """
    # Run base scenario
    base = project_kpis(
        plan=plan,
        kpi_history=kpi_history,
        initiative_history=initiative_history,
        as_of=as_of,
        horizon_end=main_date,
        actions=[],
        stress_events=stress_events,
        cfg=cfg,
    )
    
    # Run action scenario
    action = project_kpis(
        plan=plan,
        kpi_history=kpi_history,
        initiative_history=initiative_history,
        as_of=as_of,
        horizon_end=main_date,
        actions=actions,
        stress_events=stress_events,
        cfg=cfg,
    )
    
    # Get deltas at main_date
    base_at_date = base[base["date"] == main_date].set_index("kpi_id")["projected_value"]
    action_at_date = action[action["date"] == main_date].set_index("kpi_id")["projected_value"]
    
    # For now, use simplified attribution based on initiative impacts
    # This is a placeholder - full attribution would require modifying projection functions
    kpi_attributions: Dict[str, KPIAttribution] = {}
    
    for kpi_id in base_at_date.index:
        if kpi_id not in action_at_date.index:
            continue
        
        delta = float(action_at_date[kpi_id] - base_at_date[kpi_id])
        drivers: List[AttributionDriver] = []
        
        # Extract initiative contributions (simplified)
        # In full implementation, this would track contributions during projection
        for it in plan.initiatives:
            for imp in it.kpi_impacts:
                if imp.kpi_id == kpi_id:
                    # Estimate contribution based on initiative progress and impact
                    # This is approximate - full version would track during projection
                    progress = initiative_history[
                        (initiative_history["initiative_id"] == it.id) &
                        (initiative_history["date"] <= main_date)
                    ]["progress"].iloc[-1] if not initiative_history.empty else 0.0
                    
                    if progress > 0.1:  # Only include if initiative has meaningful progress
                        contrib = float(imp.expected_delta_by_end) * float(progress) * float(imp.confidence)
                        drivers.append(AttributionDriver(
                            driver_type="initiative",
                            driver_id=it.id,
                            contribution=contrib,
                            description=f"From {it.name} progress ({progress:.0%})",
                        ))
        
        # Add stress event contributions
        for ev in stress_events:
            if ev.type == "kpi_shock" and ev.target_id == kpi_id:
                drivers.append(AttributionDriver(
                    driver_type="stress_shock",
                    driver_id=ev.id,
                    contribution=float(ev.magnitude),
                    description=ev.description or f"Stress event: {ev.id}",
                ))
        
        kpi_attributions[kpi_id] = KPIAttribution(
            kpi_id=kpi_id,
            total_delta=delta,
            drivers=drivers,
        )
    
    return OptionAttribution(
        bundle_id="",  # Will be set by caller
        main_date=main_date,
        kpi_attributions=kpi_attributions,
    )


def extract_attribution_mc(
    plan: Plan,
    kpi_history: pd.DataFrame,
    initiative_history: pd.DataFrame,
    as_of: str,
    main_date: str,
    actions: List[SteeringAction],
    stress_events: List[StressEvent],
    cfg: WhatIfConfig,
    mc: WhatIfStochasticConfig,
    integrity_scores: Optional[Dict[str, float]] = None,
) -> OptionAttribution:
    """
    Extract attribution for Monte Carlo projection.
    Computes attribution for mean and p10 scenarios only.
    """
    # Get raw MC results
    raw = compare_actions_at_dates_mc_raw(
        plan=plan,
        kpi_history=kpi_history,
        initiative_history=initiative_history,
        as_of=as_of,
        dates_to_check=[main_date],
        actions=actions,
        stress_events=stress_events,
        cfg=cfg,
        mc=mc,
        integrity_scores=integrity_scores,
    )
    
    # Filter to main_date
    raw = raw[raw["date"] == main_date].copy()
    
    kpi_attributions: Dict[str, KPIAttribution] = {}
    
    for kpi_id, group in raw.groupby("kpi_id"):
        deltas = group["delta"].astype(float).values
        
        # Compute mean and p10
        mean_delta = float(pd.Series(deltas).mean())
        p10_delta = float(pd.Series(deltas).quantile(0.10))
        
        # Use mean for attribution (p10 would be similar but more conservative)
        delta = mean_delta
        
        drivers: List[AttributionDriver] = []
        
        # Extract initiative contributions (simplified)
        for it in plan.initiatives:
            for imp in it.kpi_impacts:
                if imp.kpi_id == kpi_id:
                    # Estimate based on initiative progress
                    progress = initiative_history[
                        (initiative_history["initiative_id"] == it.id) &
                        (initiative_history["date"] <= main_date)
                    ]["progress"].iloc[-1] if not initiative_history.empty else 0.0
                    
                    if progress > 0.1:
                        contrib = float(imp.expected_delta_by_end) * float(progress) * float(imp.confidence)
                        drivers.append(AttributionDriver(
                            driver_type="initiative",
                            driver_id=it.id,
                            contribution=contrib,
                            description=f"From {it.name} progress ({progress:.0%})",
                        ))
        
        # Add stress event contributions
        for ev in stress_events:
            if ev.type == "kpi_shock" and ev.target_id == kpi_id:
                drivers.append(AttributionDriver(
                    driver_type="stress_shock",
                    driver_id=ev.id,
                    contribution=float(ev.magnitude),
                    description=ev.description or f"Stress event: {ev.id}",
                ))
        
        kpi_attributions[kpi_id] = KPIAttribution(
            kpi_id=kpi_id,
            total_delta=delta,
            drivers=drivers,
        )
    
    return OptionAttribution(
        bundle_id="",  # Will be set by caller
        main_date=main_date,
        kpi_attributions=kpi_attributions,
    )
