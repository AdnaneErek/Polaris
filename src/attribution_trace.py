# src/attribution_trace.py
"""
Attribution tracing by running focused projections.

To get attribution, we run projections with different scenarios and compare:
- Base vs Action (to get action contributions)
- With/without each initiative (to isolate initiative contributions)
- With/without stress events (to get stress contributions)
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
from .stress import StressEvent


def extract_attribution_for_bundle(
    plan: Plan,
    kpi_history: pd.DataFrame,
    initiative_history: pd.DataFrame,
    as_of: str,
    main_date: str,
    bundle_actions: List[SteeringAction],
    stress_events: List[StressEvent],
    cfg: WhatIfConfig,
    mc: Optional[WhatIfStochasticConfig] = None,
    integrity_scores: Optional[Dict[str, float]] = None,
) -> OptionAttribution:
    """
    Extract attribution for a bundle by comparing scenarios.
    
    Strategy:
    1. Run base scenario (no actions)
    2. Run full scenario (with actions)
    3. For each initiative affecting KPIs, run scenario without that initiative
    4. Compare to isolate contributions
    """
    kpi_attributions: Dict[str, KPIAttribution] = {}
    
    # Get base scenario at main_date
    if mc:
        # Use MC for mean attribution
        raw = compare_actions_at_dates_mc_raw(
            plan=plan,
            kpi_history=kpi_history,
            initiative_history=initiative_history,
            as_of=as_of,
            dates_to_check=[main_date],
            actions=[],
            stress_events=stress_events,
            cfg=cfg,
            mc=mc,
            integrity_scores=integrity_scores,
        )
        base_values = raw[raw["date"] == main_date].groupby("kpi_id")["baseline"].mean()
        
        raw_action = compare_actions_at_dates_mc_raw(
            plan=plan,
            kpi_history=kpi_history,
            initiative_history=initiative_history,
            as_of=as_of,
            dates_to_check=[main_date],
            actions=bundle_actions,
            stress_events=stress_events,
            cfg=cfg,
            mc=mc,
            integrity_scores=integrity_scores,
        )
        action_values = raw_action[raw_action["date"] == main_date].groupby("kpi_id")["with_action"].mean()
    else:
        # Deterministic
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
        base_values = base[base["date"] == main_date].set_index("kpi_id")["projected_value"]
        
        action = project_kpis(
            plan=plan,
            kpi_history=kpi_history,
            initiative_history=initiative_history,
            as_of=as_of,
            horizon_end=main_date,
            actions=bundle_actions,
            stress_events=stress_events,
            cfg=cfg,
        )
        action_values = action[action["date"] == main_date].set_index("kpi_id")["projected_value"]
    
    # Compute deltas
    for kpi_id in base_values.index:
        if kpi_id not in action_values.index:
            continue
        
        delta = float(action_values[kpi_id] - base_values[kpi_id])
        drivers: List[AttributionDriver] = []
        
        # Extract initiative contributions
        # For each initiative that affects this KPI, estimate contribution
        for it in plan.initiatives:
            for imp in it.kpi_impacts:
                if imp.kpi_id != kpi_id:
                    continue
                
                # Check if this initiative is affected by actions
                initiative_affected = any(
                    hasattr(a, 'target_initiative') and a.target_initiative == it.id for a in bundle_actions
                )
                
                if not initiative_affected:
                    continue
                
                # Estimate contribution based on impact parameters
                # This is approximate - full version would track during projection
                progress = initiative_history[
                    (initiative_history["initiative_id"] == it.id) &
                    (initiative_history["date"] <= main_date)
                ]["progress"].iloc[-1] if not initiative_history.empty else 0.0
                
                if progress > 0.05:  # Only include if meaningful
                    # Estimate: delta_end * progress * confidence
                    contrib = float(imp.expected_delta_by_end) * float(progress) * float(imp.confidence)
                    
                    # Check for dependency blocking
                    dependency_penalty = 0.0
                    for dep in it.dependencies:
                        if dep.type == "hard":
                            dep_progress = initiative_history[
                                (initiative_history["initiative_id"] == dep.initiative_id) &
                                (initiative_history["date"] <= main_date)
                            ]["progress"].iloc[-1] if not initiative_history.empty else 0.0
                            
                            if dep_progress < 0.6:  # Dependency blocking
                                penalty_frac = 0.5  # 50% reduction
                                dependency_penalty = contrib * penalty_frac
                                contrib *= (1.0 - penalty_frac)
                                
                                drivers.append(AttributionDriver(
                                    driver_type="dependency_penalty",
                                    driver_id=f"{it.id}_blocked_by_{dep.initiative_id}",
                                    contribution=-dependency_penalty,
                                    description=f"Dependency penalty: {it.name} blocked by {dep.initiative_id}",
                                ))
                    
                    if abs(contrib) > 0.01:  # Only include if material
                        drivers.append(AttributionDriver(
                            driver_type="initiative",
                            driver_id=it.id,
                            contribution=contrib,
                            description=f"From {it.name} (progress {progress:.0%}, impact {imp.expected_delta_by_end:+.2f})",
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
        
        # Add action effects (scope changes, capacity additions)
        for action in bundle_actions:
            if action.type == "split_scope":
                # Scope reduction affects all KPIs impacted by that initiative
                for it in plan.initiatives:
                    if it.id == action.target_initiative:
                        for imp in it.kpi_impacts:
                            if imp.kpi_id == kpi_id:
                                reduction = action.get("reduction", 0.0)
                                # Estimate negative impact from scope reduction
                                contrib = -float(imp.expected_delta_by_end) * float(reduction) * 0.5
                                drivers.append(AttributionDriver(
                                    driver_type="action_effect",
                                    driver_id=action.id,
                                    contribution=contrib,
                                    description=f"Scope reduction from {action.description}",
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
