# src/simulate_initiatives.py
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta

from src.load_plan import load_plan


def _dt(s: str) -> datetime:
    return datetime.fromisoformat(s)

def _month_start_iso(s: str) -> str:
    d = _dt(s)
    return d.replace(day=1).date().isoformat()

def month_range(start: str, end: str) -> List[str]:
    cur = _dt(_month_start_iso(start))
    end_dt = _dt(_month_start_iso(end))
    out = []
    while cur <= end_dt:
        out.append(cur.date().isoformat())
        cur = cur + relativedelta(months=1)
    return out

def clamp01(x: float) -> float:
    return float(max(0.0, min(1.0, x)))

def logistic_progress(t: float, k: float = 8.0, mid: float = 0.55) -> float:
    # smooth S-curve 0..1
    return 1.0 / (1.0 + np.exp(-k * (t - mid)))

def piecewise_progress(t: float, shape: str) -> float:
    """
    t in [0,1].
    - "frontloaded": faster early, then slows
    - "backloaded": slow early, then accelerates
    - "balanced": S-curve
    """
    t = float(max(0.0, min(1.0, t)))
    if shape == "frontloaded":
        return clamp01(t ** 0.65)
    if shape == "backloaded":
        return clamp01(t ** 1.6)
    return clamp01(logistic_progress(t))


@dataclass(frozen=True)
class Scenario:
    name: str
    # add constant delay to all initiatives (months)
    global_delay_months: int = 0
    # initiative-specific additional delays (months)
    per_initiative_delay: Optional[Dict[str, int]] = None
    # temporary slippage window: (start, end, drop_fraction_of_progress)
    # drop_fraction means: reduce effective progress by that fraction during window (not permanent)
    slippage: Optional[Tuple[str, str, float]] = None
    # noise strength (small random wobble)
    noise: float = 0.01


def simulate_initiatives(
    plan_path: str,
    out_csv: str,
    scenario: Scenario,
    seed: int = 7,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    plan = load_plan(plan_path)

    per_delay = scenario.per_initiative_delay or {}

    # shapes chosen to be realistic by initiative type
    # - DQ: often backloaded (governance, sign-offs)
    # - A1: balanced (delivery waves)
    # - RES1: frontloaded (instrumentation ramps early then steady)
    shape_by_id = {
        "INIT_DQ1": "backloaded",
        "INIT_A1": "balanced",
        "INIT_RES1": "frontloaded",
    }

    # Build monthly date index across plan horizon
    dates = month_range(plan.meta.horizon.start, plan.meta.horizon.end)

    rows = []
    for it in plan.initiatives:
        it_id = it.id
        it_name = it.name
        owner = it.owner
        budget_k = float(it.budget.amount_k)

        start = _month_start_iso(it.timeline.start)
        end = _month_start_iso(it.timeline.end)

        # apply scenario delays
        delay_m = int(scenario.global_delay_months) + int(per_delay.get(it_id, 0))
        if delay_m != 0:
            start_dt = _dt(start) + relativedelta(months=delay_m)
            end_dt = _dt(end) + relativedelta(months=delay_m)
            start = start_dt.date().isoformat()
            end = end_dt.date().isoformat()

        # progress model: 0 before start, 1 after end, smooth in between
        # map each month to t in [0,1] within [start,end]
        start_dt = _dt(start)
        end_dt = _dt(end)
        total_months = max(1, (end_dt.year - start_dt.year) * 12 + (end_dt.month - start_dt.month))

        shape = shape_by_id.get(it_id, "balanced")

        for d in dates:
            d_dt = _dt(d)
            if d_dt < start_dt:
                p = 0.0
            elif d_dt >= end_dt:
                p = 1.0
            else:
                m = (d_dt.year - start_dt.year) * 12 + (d_dt.month - start_dt.month)
                t = m / float(total_months)
                p = piecewise_progress(t, shape=shape)

            # small noise (keeps it realistic but not chaotic)
            p = clamp01(p + float(rng.normal(0.0, scenario.noise)))

            # temporary slippage window (effective progress dip)
            if scenario.slippage is not None:
                s0, s1, drop = scenario.slippage
                s0 = _month_start_iso(s0)
                s1 = _month_start_iso(s1)
                if s0 <= d <= s1:
                    p = clamp01(p * (1.0 - float(drop)))

            rows.append(
                {
                    "date": d,
                    "initiative_id": it_id,
                    "initiative_name": it_name,
                    "owner": owner,
                    "progress": round(float(p), 6),
                    "delay_months": int(delay_m),
                    "budget_k": round(budget_k, 2),
                }
            )

    df = pd.DataFrame(rows).sort_values(["date", "initiative_id"]).reset_index(drop=True)
    df.to_csv(out_csv, index=False)
    return df


def main():
    # --- choose one scenario ---
    base = Scenario(name="base", global_delay_months=0, noise=0.008)

    delayed = Scenario(
        name="delayed",
        global_delay_months=2,
        per_initiative_delay={"INIT_A1": 1},  # automation extra month
        noise=0.010,
    )

    stressed = Scenario(
        name="stressed",
        global_delay_months=0,
        slippage=("2027-01-01", "2027-06-01", 0.20),  # 20% effective dip
        noise=0.010,
    )

    scenario = base  # <-- switch to delayed / stressed if you want

    out = "data/simulated_initiatives.csv"
    df = simulate_initiatives("data/plan.yaml", out, scenario=scenario, seed=7)
    print(f"✅ wrote {out} ({len(df)} rows) scenario={scenario.name}")
    print(df.head(12).to_string(index=False))


if __name__ == "__main__":
    main()
