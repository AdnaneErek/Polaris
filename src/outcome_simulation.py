# src/outcome_simulation.py
"""
Simulate actual KPI outcomes given a selected strategic option.

This module provides functions to simulate what the actual KPI values would be
at an evaluation date, given that a specific strategic option was executed.
This is used for closed-loop learning validation when real outcome data is not available.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from src.schemas import Plan
from src.portfolio import ActionBundle
from src.actions import SteeringAction
from src.whatif import project_kpis, WhatIfConfig


def simulate_actual_outcomes(
    plan: Plan,
    kpis_df: pd.DataFrame,
    initiatives_df: pd.DataFrame,
    selected_bundle: ActionBundle,
    original_as_of: str,
    evaluation_date: str,
    noise_level: float = 0.05,
    random_seed: Optional[int] = None,
) -> Dict[str, float]:
    """
    Simulate what the actual KPI values would be at evaluation_date,
    given that we executed the selected option.
    
    This re-runs the what-if projection with the option's actions applied,
    then adds noise to simulate forecast error / reality.
    
    Args:
        plan: Strategic plan
        kpis_df: Historical KPI data (should only include data up to original_as_of to avoid leakage)
        initiatives_df: Historical initiative progress data (can include data up to evaluation_date)
        selected_bundle: The strategic option that was actually selected
        original_as_of: When the decision was made (YYYY-MM-DD)
        evaluation_date: When we're measuring outcomes (YYYY-MM-DD)
        noise_level: Standard deviation of noise as fraction of value (default: 0.05 = 5%)
        random_seed: Optional seed for reproducibility
        
    Returns:
        Dict mapping kpi_id -> simulated actual value at evaluation_date
    """
    if random_seed is not None:
        rng = np.random.default_rng(random_seed)
    else:
        rng = np.random.default_rng()
    
    # Filter KPI history to only include data up to original_as_of (no leakage!)
    kpi_history = kpis_df[kpis_df["date"] <= original_as_of].copy()
    
    # Project forward with the selected option's actions applied
    projected = project_kpis(
        plan=plan,
        kpi_history=kpi_history,
        initiative_history=initiatives_df,
        as_of=original_as_of,
        horizon_end=evaluation_date,
        actions=selected_bundle.actions,
        cfg=WhatIfConfig(),
    )
    
    # Extract KPI values at evaluation_date
    eval_date_proj = projected[projected["date"] == evaluation_date].copy()
    
    if eval_date_proj.empty:
        # Fallback: use the last available date in the projection
        if not projected.empty:
            last_date = projected["date"].max()
            eval_date_proj = projected[projected["date"] == last_date].copy()
        else:
            raise ValueError(
                f"No projected values found for evaluation_date={evaluation_date}. "
                f"Projection may have failed or evaluation_date is out of range."
            )
    
    outcomes: Dict[str, float] = {}
    for _, row in eval_date_proj.iterrows():
        kpi_id = str(row["kpi_id"])
        value = float(row["projected_value"])
        
        # Add noise to simulate forecast error / reality
        # Noise is proportional to the value magnitude
        noise_std = abs(value) * noise_level
        noise = rng.normal(0, noise_std)
        outcomes[kpi_id] = value + noise
    
    return outcomes


def extract_predicted_outcomes(
    forecast_df: pd.DataFrame,
    evaluation_date: str,
    kpi_ids: Optional[List[str]] = None,
) -> Dict[str, float]:
    """
    Extract predicted KPI values from a forecast dataframe at a specific date.
    
    Args:
        forecast_df: Forecast dataframe with columns: date, kpi_id, forecast
        evaluation_date: Date to extract predictions for (YYYY-MM-DD)
        kpi_ids: Optional list of KPI IDs to extract (default: all in forecast_df)
        
    Returns:
        Dict mapping kpi_id -> predicted value
    """
    eval_data = forecast_df[forecast_df["date"] == evaluation_date].copy()
    
    if eval_data.empty:
        # Fallback: use the closest available date
        forecast_df["date"] = pd.to_datetime(forecast_df["date"])
        eval_date_ts = pd.to_datetime(evaluation_date)
        forecast_df["date_diff"] = (forecast_df["date"] - eval_date_ts).abs()
        closest_idx = forecast_df["date_diff"].idxmin()
        eval_data = forecast_df.loc[[closest_idx]].copy()
    
    predicted: Dict[str, float] = {}
    for _, row in eval_data.iterrows():
        kpi_id = str(row["kpi_id"])
        if kpi_ids is None or kpi_id in kpi_ids:
            value = row.get("forecast") or row.get("value")
            if pd.notna(value):
                predicted[kpi_id] = float(value)
    
    return predicted
