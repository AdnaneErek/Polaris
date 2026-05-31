# src/steering.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import pandas as pd

from .schemas import Plan, Initiative


@dataclass
class InitiativeHealth:
    initiative_id: str
    progress: float
    delay_months: int
    blocked_by: List[str]  # dependency ids
    health_flag: str       # "good" | "watch" | "bad"


def get_initiative_health(plan: Plan, initiatives_df: pd.DataFrame, as_of: str) -> Dict[str, InitiativeHealth]:
    """
    Uses simulated initiative progress + delays and plan dependencies
    to produce a per-initiative health snapshot.
    """
    init_by_id: Dict[str, Initiative] = {i.id: i for i in plan.initiatives}

    # latest status per initiative at or before as_of
    df = initiatives_df.copy()
    df = df[df["date"] <= as_of].sort_values("date")
    latest = df.groupby("initiative_id").tail(1)

    prog = dict(zip(latest["initiative_id"], latest["progress"]))
    delays = dict(zip(latest["initiative_id"], latest["delay_months"]))

    health: Dict[str, InitiativeHealth] = {}

    for it in plan.initiatives:
        p = float(prog.get(it.id, 0.0))
        d = int(delays.get(it.id, 0))

        blocked_by = []
        for dep in it.dependencies:
            if dep.type == "hard":
                dep_prog = float(prog.get(dep.initiative_id, 0.0))
                # heuristic: hard dependency considered blocking if not at least 0.6 progressed
                if dep_prog < 0.60:
                    blocked_by.append(dep.initiative_id)

        # health flag
        flag = "good"
        if d >= 2 or blocked_by:
            flag = "watch"
        if d >= 3 or (blocked_by and p < 0.50):
            flag = "bad"

        health[it.id] = InitiativeHealth(
            initiative_id=it.id,
            progress=p,
            delay_months=d,
            blocked_by=blocked_by,
            health_flag=flag
        )

    return health


def summarize_dependency_bottlenecks(health: Dict[str, InitiativeHealth]) -> List[str]:
    out = []
    for it_id, h in health.items():
        if h.blocked_by:
            out.append(f"{it_id} is blocked by hard dependencies: {', '.join(h.blocked_by)}")
    return out
