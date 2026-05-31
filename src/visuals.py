# src/visuals.py
"""
Visualization generation for SteerCo packs.

Creates:
1. KPI forecast vs expected (with uncertainty band)
2. Score distribution per option (base vs stress) showing CVaR
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .schemas import Plan
from .portfolio import OptionResultMC


def plot_kpi_forecast(
    plan: Plan,
    forecast_df: pd.DataFrame,
    kpi_id: str,
    output_path: Path,
    uncertainty_band: bool = True,
) -> None:
    """
    Plot KPI forecast vs expected trajectory with uncertainty band.
    
    Args:
        plan: Plan object
        forecast_df: DataFrame with columns: date, kpi_id, forecasted_value, expected_value (optional)
        kpi_id: KPI to plot
        output_path: Where to save the plot
        uncertainty_band: Whether to show uncertainty band (if available)
    """
    kpi_data = forecast_df[forecast_df["kpi_id"] == kpi_id].copy()
    if kpi_data.empty:
        return
    
    # Find KPI name
    kpi = None
    for k in plan.kpis:
        if k.id == kpi_id or k.short_name == kpi_id:
            kpi = k
            break
    
    kpi_name = kpi.short_name if kpi else kpi_id
    unit = kpi.unit if kpi else ""
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    dates = pd.to_datetime(kpi_data["date"])
    
    # Plot forecasted values
    if "forecasted_value" in kpi_data.columns:
        ax.plot(dates, kpi_data["forecasted_value"], label="Forecast", linewidth=2, color="blue")
    
    # Plot expected trajectory
    if "expected_value" in kpi_data.columns:
        ax.plot(dates, kpi_data["expected_value"], label="Expected", linewidth=2, linestyle="--", color="orange")
    
    # Add baseline and target
    if kpi:
        baseline = float(kpi.baseline.value)
        target = float(kpi.target.value)
        ax.axhline(y=baseline, color="gray", linestyle=":", label=f"Baseline ({baseline:.1f}{unit})", alpha=0.7)
        ax.axhline(y=target, color="green", linestyle=":", label=f"Target ({target:.1f}{unit})", alpha=0.7)
    
    # Uncertainty band (if available)
    if uncertainty_band and "forecast_lower" in kpi_data.columns and "forecast_upper" in kpi_data.columns:
        ax.fill_between(
            dates,
            kpi_data["forecast_lower"],
            kpi_data["forecast_upper"],
            alpha=0.2,
            color="blue",
            label="Uncertainty band",
        )
    
    ax.set_xlabel("Date")
    ax.set_ylabel(f"{kpi_name} ({unit})")
    ax.set_title(f"KPI Forecast: {kpi_name}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_score_distributions(
    results: List[OptionResultMC],
    output_path: Path,
    show_cvar: bool = True,
) -> None:
    """
    Plot score distribution per option (base vs stress) showing CVaR.
    
    Args:
        results: List of OptionResultMC
        output_path: Where to save the plot
        show_cvar: Whether to mark CVaR10 on the plot
    """
    if not results:
        return
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # For each option, create a simulated distribution
    # (We don't have raw scores, so we'll approximate from summary stats)
    x_pos = np.arange(len(results))
    width = 0.35
    
    base_means = [r.score_base_mean for r in results]
    stress_means = [r.score_stress_mean for r in results]
    base_cvars = [r.score_base_cvar10 for r in results]
    stress_cvars = [r.score_stress_cvar10 for r in results]
    
    bundle_names = [r.bundle_name[:30] for r in results]  # Truncate long names
    
    # Plot means
    bars1 = ax.bar(x_pos - width/2, base_means, width, label="Base (mean)", color="lightblue", alpha=0.7)
    bars2 = ax.bar(x_pos + width/2, stress_means, width, label="Stress (mean)", color="lightcoral", alpha=0.7)
    
    # Mark CVaR10
    if show_cvar:
        for i, (base_cvar, stress_cvar) in enumerate(zip(base_cvars, stress_cvars)):
            ax.plot(i - width/2, base_cvar, marker="v", color="darkblue", markersize=8, label="Base CVaR10" if i == 0 else "")
            ax.plot(i + width/2, stress_cvar, marker="v", color="darkred", markersize=8, label="Stress CVaR10" if i == 0 else "")
    
    ax.set_xlabel("Option")
    ax.set_ylabel("Portfolio Score")
    ax.set_title("Score Distribution per Option (Base vs Stress)")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(bundle_names, rotation=45, ha="right")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def generate_steerco_visuals(
    plan: Plan,
    forecast_df: pd.DataFrame,
    results: List[OptionResultMC],
    output_dir: Path,
) -> List[Path]:
    """
    Generate all SteerCo visuals.
    
    Returns list of generated plot file paths.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    generated = []
    
    # 1. KPI forecast plots (one per KPI)
    for kpi in plan.kpis:
        plot_path = output_dir / f"forecast_{kpi.id}.png"
        plot_kpi_forecast(plan, forecast_df, kpi.id, plot_path)
        generated.append(plot_path)
    
    # 2. Score distribution plot
    score_plot_path = output_dir / "score_distributions.png"
    plot_score_distributions(results, score_plot_path)
    generated.append(score_plot_path)
    
    return generated
