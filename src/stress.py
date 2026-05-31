# src/stress.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Literal, Optional
import pandas as pd

StressType = Literal["initiative_progress_drop", "kpi_shock"]

@dataclass(frozen=True)
class StressEvent:
    id: str
    type: StressType
    target_id: str          # initiative_id or kpi_id
    start: str              # YYYY-MM-01
    end: str                # YYYY-MM-01
    magnitude: float        # meaning depends on type
    description: str


def apply_stress_to_initiatives(init_df: pd.DataFrame, events: list[StressEvent]) -> pd.DataFrame:
    df = init_df.copy()
    for ev in events:
        if ev.type != "initiative_progress_drop":
            continue
        mask = (
            (df["initiative_id"] == ev.target_id)
            & (df["date"] >= ev.start)
            & (df["date"] <= ev.end)
        )
        # drop progress by magnitude (e.g., 0.20), keep within [0,1]
        df.loc[mask, "progress"] = (df.loc[mask, "progress"] - ev.magnitude).clip(0.0, 1.0)
    return df


def apply_stress_to_kpis(kpi_df: pd.DataFrame, events: list[StressEvent]) -> pd.DataFrame:
    df = kpi_df.copy()
    for ev in events:
        if ev.type != "kpi_shock":
            continue
        mask = (
            (df["kpi_id"] == ev.target_id)
            & (df["date"] >= ev.start)
            & (df["date"] <= ev.end)
        )
        # add shock (negative or positive depending on KPI)
        df.loc[mask, "value"] = df.loc[mask, "value"] + ev.magnitude
    return df
