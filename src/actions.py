# src/actions.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Literal, List, Tuple

import pandas as pd
from dateutil.relativedelta import relativedelta
from datetime import datetime


ActionType = Literal[
    "accelerate_initiative",   # pull future progress earlier (time-shift) + optional capacity gain
    "add_capacity",            # modeled as capacity gain (and optional time-shift)
    "split_scope",             # partial delivery earlier, but reduced total upside
]


@dataclass(frozen=True)
class SteeringAction:
    id: str
    type: ActionType
    target_initiative: str
    parameters: Dict[str, float]
    description: str

    def get(self, key: str, default: float = 0.0) -> float:
        return float(self.parameters.get(key, default))


# -----------------------------
# Helpers
# -----------------------------
def _to_dt(s: str) -> datetime:
    return datetime.fromisoformat(s)


def _to_iso(d: datetime) -> str:
    return d.date().isoformat()


def _month_start(s: str) -> str:
    d = _to_dt(s)
    return _to_iso(d.replace(day=1))


def _add_months(date_iso: str, months: int) -> str:
    d = _to_dt(date_iso).replace(day=1)
    d2 = d + relativedelta(months=months)
    return _to_iso(d2)


def _interp_progress(df_one: pd.DataFrame, query_dates: List[str]) -> Dict[str, float]:
    """
    Linear interpolate progress for a single initiative over monthly grid.
    Expects df_one columns: date, progress (date ISO), sorted.
    Returns dict date->progress.
    """
    if df_one.empty:
        return {d: 0.0 for d in query_dates}

    tmp = df_one.copy()
    tmp["date"] = pd.to_datetime(tmp["date"])
    tmp = tmp.sort_values("date")

    # ensure bounds
    tmp["progress"] = tmp["progress"].astype(float).clip(lower=0.0, upper=1.0)

    q = pd.DataFrame({"date": pd.to_datetime(query_dates)})
    all_dates = pd.concat([tmp[["date"]], q], ignore_index=True).drop_duplicates().sort_values("date")

    merged = all_dates.merge(tmp, on="date", how="left")
    merged["progress"] = merged["progress"].interpolate(method="linear").ffill().bfill()
    merged["date_iso"] = merged["date"].dt.date.astype(str)

    return dict(zip(merged["date_iso"], merged["progress"]))


def _capacity_boost(p: float, gain: float) -> float:
    """
    Smooth saturating boost:
      p <- p + gain * (1 - p)
    - gain in [0, 1]
    - preserves 0..1 bounds
    - monotone increasing for gain >= 0
    """
    p = float(p)
    g = float(max(0.0, min(1.0, gain)))
    return float(min(1.0, max(0.0, p + g * (1.0 - p))))


# -----------------------------
# Core application logic
# -----------------------------
def apply_actions_to_initiatives(
    initiatives_df: pd.DataFrame,
    as_of: str,
    actions: List[SteeringAction],
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Applies steering actions to initiatives_df and returns:
      - modified initiatives_df (same schema, progress adjusted)
      - impact_scalers: initiative_id -> multiplier applied to KPI impacts (for scope cuts)

    Expected initiatives_df columns:
      - date (YYYY-MM-DD)
      - initiative_id
      - progress (0..1)

    Action semantics:
      - accelerate_initiative:
          * time-shift: progress(d) := progress(d + months) for d >= as_of
          * optional capacity gain after as_of (if parameters includes capacity_gain or if enable_capacity_gain=1)
      - add_capacity:
          * capacity gain after as_of (default), optional months time-shift
      - split_scope:
          * reduce total upside (impact_scaler = 1 - reduction)
          * deliver earlier (progress scaled up after as_of)
    """
    if initiatives_df is None or initiatives_df.empty or not actions:
        return initiatives_df, {}

    df = initiatives_df.copy()

    # Normalize
    df["date"] = df["date"].astype(str).map(_month_start)
    df["initiative_id"] = df["initiative_id"].astype(str)
    if "progress" not in df.columns:
        raise ValueError("initiatives_df must contain a 'progress' column")
    df["progress"] = df["progress"].astype(float).clip(0.0, 1.0)

    as_of_m = _month_start(as_of)

    # per-initiative groups
    by_init = {iid: g.sort_values("date") for iid, g in df.groupby("initiative_id")}

    # Output scalers for KPI impact (mainly split_scope)
    impact_scalers: Dict[str, float] = {}

    for act in actions:
        iid = act.target_initiative
        if iid not in by_init:
            continue

        g = by_init[iid].copy()
        dates = list(g["date"].astype(str).tolist())

        if act.type in ("accelerate_initiative", "add_capacity"):
            # --- 1) optional time-shift (pull progress earlier) ---
            months = int(round(act.get("months", 0.0)))
            if months != 0:
                prog_map = _interp_progress(g[["date", "progress"]], dates)
                shifted = []
                for d in dates:
                    if d < as_of_m:
                        shifted.append(float(prog_map[d]))
                    else:
                        d_future = _add_months(d, months)
                        dv = float(prog_map.get(d_future, prog_map[dates[-1]]))
                        shifted.append(dv)
                g["progress"] = pd.Series(shifted, index=g.index).astype(float).clip(0.0, 1.0)

            # --- 2) capacity gain (decision-sensitive) ---
            #
            # For add_capacity: ON by default.
            # For accelerate_initiative: OFF by default unless explicitly enabled.
            #
            # Parameters:
            #   capacity_gain: float in [0,1]  (recommended 0.03..0.10)
            #   enable_capacity_gain: 0/1 (useful if you want accelerate to also add throughput)
            #
            # If not provided:
            #   add_capacity -> default capacity_gain = 0.06
            #   accelerate_initiative -> default 0.00 (unless enable_capacity_gain=1)
            #
            enable = int(round(act.get("enable_capacity_gain", 0.0)))
            if act.type == "add_capacity":
                default_gain = 0.06
                gain = float(act.get("capacity_gain", default_gain))
                do_gain = True
            else:
                # accelerate_initiative
                gain = float(act.get("capacity_gain", 0.0))
                do_gain = (enable == 1) and (gain > 0.0)

            if do_gain and gain > 0.0:
                # apply month-by-month after as_of (monotone, saturating)
                boosted = []
                last = None
                for d, p in zip(dates, g["progress"].astype(float).tolist()):
                    if d < as_of_m:
                        boosted.append(float(p))
                        last = float(p)
                        continue

                    # enforce monotonicity vs previous month (delivery doesn't go backwards)
                    base_p = float(p)
                    if last is not None:
                        base_p = max(base_p, float(last))

                    new_p = _capacity_boost(base_p, gain)
                    boosted.append(float(new_p))
                    last = float(new_p)

                g["progress"] = pd.Series(boosted, index=g.index).astype(float).clip(0.0, 1.0)

        elif act.type == "split_scope":
            """
            Model:
              - reduce total scope -> total KPI upside reduces
              - deliver something earlier -> effective progress rises earlier

            parameter:
              reduction in [0,1], e.g. 0.6 means "cut 60% of scope", keep 40%.
              effective_scope = 1 - reduction.

            Transform:
              progress' = min(1, progress / effective_scope)  (after as_of)
              impact_scaler = effective_scope
            """
            reduction = float(act.get("reduction", 0.0))
            reduction = max(0.0, min(0.95, reduction))  # prevent divide by ~0
            effective_scope = 1.0 - reduction

            impact_scalers[iid] = min(impact_scalers.get(iid, 1.0), effective_scope)

            boosted = []
            last = None
            for d, p in zip(dates, g["progress"].astype(float).tolist()):
                if d < as_of_m:
                    boosted.append(float(p))
                    last = float(p)
                    continue

                base_p = float(p)
                if last is not None:
                    base_p = max(base_p, float(last))

                new_p = min(1.0, base_p / effective_scope)
                boosted.append(float(new_p))
                last = float(new_p)

            g["progress"] = pd.Series(boosted, index=g.index).astype(float).clip(0.0, 1.0)

        # Write back
        by_init[iid] = g
        df.loc[df["initiative_id"] == iid, "progress"] = g["progress"].values

    df["progress"] = df["progress"].astype(float).clip(0.0, 1.0)
    return df, impact_scalers
