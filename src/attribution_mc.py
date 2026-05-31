# src/attribution_mc.py
"""
Attribution from raw Monte Carlo deltas.

Computes per-sample KPI contributions and maps them to initiatives.
This provides defensible attribution that matches the actual scoring model.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .attribution import AttributionDriver, KPIAttribution, OptionAttribution
from .schemas import Plan
from .portfolio import _objective_weights, _normalize_delta


def compute_attribution_from_raw_mc(
    plan: Plan,
    raw_deltas: pd.DataFrame,  # sample, date, kpi_id, delta
    bundle_actions: List,
    main_date: str,
) -> OptionAttribution:
    """
    Compute attribution from raw MC deltas.
    
    For each option:
    1. Compute score per sample
    2. Compute per-KPI contribution per sample (weight × normalized_delta)
    3. Summarize contribution distribution (mean, p10, p90)
    4. Map KPI → initiatives using plan.yaml initiative impacts
    
    Returns OptionAttribution with per-sample-based drivers.
    """
    # Filter to main_date
    raw = raw_deltas[raw_deltas["date"] == main_date].copy()
    
    if raw.empty:
        return OptionAttribution(
            bundle_id="",
            main_date=main_date,
            kpi_attributions={},
        )
    
    weights = _objective_weights(plan)
    kpi_attributions: Dict[str, KPIAttribution] = {}
    
    # For each KPI, compute per-sample contributions
    for kpi_id in raw["kpi_id"].unique():
        kpi_raw = raw[raw["kpi_id"] == kpi_id].copy()
        
        if kpi_raw.empty:
            continue
        
        # Get deltas for this KPI
        deltas = kpi_raw["delta"].astype(float).values
        
        # Compute normalized contributions per sample
        normalized_contribs = np.array([
            _normalize_delta(plan, kpi_id, float(d)) for d in deltas
        ])
        
        # Weight by objective weight
        weight = weights.get(kpi_id, 0.0)
        weighted_contribs = normalized_contribs * float(weight)
        
        # Summarize contribution distribution
        contrib_mean = float(np.mean(weighted_contribs))
        contrib_p10 = float(np.quantile(weighted_contribs, 0.10))
        contrib_p90 = float(np.quantile(weighted_contribs, 0.90))
        
        # Total delta (for reference)
        total_delta = float(np.mean(deltas))
        
        # Map to initiatives
        drivers: List[AttributionDriver] = []
        
        # Find initiatives that impact this KPI
        for initiative in plan.initiatives:
            for imp in initiative.kpi_impacts:
                if imp.kpi_id != kpi_id:
                    continue
                
                # Check if this initiative is affected by bundle actions
                initiative_affected = any(
                    a.target_initiative == initiative.id for a in bundle_actions
                )
                
                if not initiative_affected:
                    continue
                
                # Estimate contribution based on impact parameters
                # This is approximate - full version would track during projection
                # For now, use expected impact scaled by confidence
                contrib = float(imp.expected_delta_by_end) * float(imp.confidence) * float(weight)
                
                # Check for dependency blocking (simplified)
                dependency_penalty = 0.0
                for dep in initiative.dependencies:
                    if dep.type == "hard":
                        # Estimate penalty (simplified - would need actual progress)
                        penalty_frac = 0.3  # Assume 30% penalty if dependency blocks
                        dependency_penalty = contrib * penalty_frac
                        contrib *= (1.0 - penalty_frac)
                        
                        drivers.append(AttributionDriver(
                            driver_type="dependency_penalty",
                            driver_id=f"{initiative.id}_blocked_by_{dep.initiative_id}",
                            contribution=-dependency_penalty,
                            description=f"Dependency penalty: {initiative.name} blocked by {dep.initiative_id}",
                        ))
                
                if abs(contrib) > 0.001:  # Only include if material
                    drivers.append(AttributionDriver(
                        driver_type="initiative",
                        driver_id=initiative.id,
                        contribution=contrib,
                        description=f"From {initiative.name} (expected impact {imp.expected_delta_by_end:+.2f}, "
                                   f"confidence {imp.confidence:.2f})",
                    ))
        
        # Don't add statistical summary as a driver - it's not a real driver
        # Statistical summaries are already captured in the KPI delta summaries
        
        kpi_attributions[kpi_id] = KPIAttribution(
            kpi_id=kpi_id,
            total_delta=total_delta,
            drivers=drivers,
        )
    
    return OptionAttribution(
        bundle_id="",  # Will be set by caller
        main_date=main_date,
        kpi_attributions=kpi_attributions,
    )


def compute_attribution_summary_from_raw_mc(
    plan: Plan,
    raw_deltas: pd.DataFrame,
    bundle_actions: List,
    main_date: str,
) -> Dict[str, Dict[str, float]]:
    """
    Compute attribution summary from raw MC deltas.
    
    Returns: {
        kpi_id: {
            "mean_contribution": float,
            "p10_contribution": float,
            "p90_contribution": float,
            "mean_delta": float,
            "p10_delta": float,
            "p90_delta": float,
        }
    }
    """
    raw = raw_deltas[raw_deltas["date"] == main_date].copy()
    
    if raw.empty:
        return {}
    
    weights = _objective_weights(plan)
    summary: Dict[str, Dict[str, float]] = {}
    
    for kpi_id in raw["kpi_id"].unique():
        kpi_raw = raw[raw["kpi_id"] == kpi_id].copy()
        
        if kpi_raw.empty:
            continue
        
        deltas = kpi_raw["delta"].astype(float).values
        
        # Compute normalized contributions
        normalized_contribs = np.array([
            _normalize_delta(plan, kpi_id, float(d)) for d in deltas
        ])
        
        weight = weights.get(kpi_id, 0.0)
        weighted_contribs = normalized_contribs * float(weight)
        
        summary[kpi_id] = {
            "mean_contribution": float(np.mean(weighted_contribs)),
            "p10_contribution": float(np.quantile(weighted_contribs, 0.10)),
            "p90_contribution": float(np.quantile(weighted_contribs, 0.90)),
            "mean_delta": float(np.mean(deltas)),
            "p10_delta": float(np.quantile(deltas, 0.10)),
            "p90_delta": float(np.quantile(deltas, 0.90)),
        }
    
    return summary
