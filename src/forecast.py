# src/forecast.py
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta

from src.schemas import Plan


# ----------------------------
# Time helpers
# ----------------------------
def _dt(s: str) -> datetime:
    return datetime.fromisoformat(str(s))


def _month_floor(d: datetime) -> datetime:
    return d.replace(day=1, hour=0, minute=0, second=0, microsecond=0)


def month_range(start: str, end: str) -> List[str]:
    cur = _month_floor(_dt(start)).date().replace(day=1)
    end_dt = _month_floor(_dt(end)).date().replace(day=1)
    out: List[str] = []
    while cur <= end_dt:
        out.append(cur.isoformat())
        cur = (datetime.fromisoformat(cur.isoformat()) + relativedelta(months=1)).date().replace(day=1)
    return out


def _safe_float(x) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


# ----------------------------
# Plan expected trajectory (for fallback + expected column)
# ----------------------------
def _expected_from_plan(plan: Plan, kpi_id: str, date_iso: str) -> Optional[float]:
    pts: List[Tuple[datetime, float]] = []
    for obj in plan.objectives:
        for okr in obj.okrs:
            if okr.kpi_id != kpi_id:
                continue
            for cp in okr.trajectory.checkpoints:
                pts.append((_dt(cp.date), float(cp.expected)))

    if not pts:
        return None

    pts.sort(key=lambda x: x[0])
    t = _dt(date_iso)

    if t <= pts[0][0]:
        return pts[0][1]
    if t >= pts[-1][0]:
        return pts[-1][1]

    for (t0, v0), (t1, v1) in zip(pts[:-1], pts[1:]):
        if t0 <= t <= t1:
            span = max(1, (t1 - t0).days)
            frac = (t - t0).days / span
            return v0 + frac * (v1 - v0)
    return pts[-1][1]


# ----------------------------
# Tiny Ridge (no sklearn dependency)
# ----------------------------
def _ridge_fit(X: np.ndarray, y: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    XtX = X.T @ X
    I = np.eye(XtX.shape[0], dtype=float)
    return np.linalg.solve(XtX + alpha * I, X.T @ y)


def _ridge_predict(X: np.ndarray, w: np.ndarray) -> np.ndarray:
    return X @ w


# ----------------------------
# Forecast config
# ----------------------------
@dataclass(frozen=True)
class ForecastConfig:
    lookback_months: int = 24
    min_points: int = 8
    lags: Tuple[int, ...] = (1, 2, 3)
    ridge_alpha: float = 3.0
    z: float = 1.64  # ~90% interval if Gaussian-ish
    sigma_floor_frac_of_scale: float = 0.01

    # Exogenous initiative features
    use_initiatives: bool = True
    max_initiatives_features: int = 12  # cap feature explosion (counts initiatives, not per-feature columns)
    initiative_future_hold: str = "last"  # "last" or "linear_to_1"

    # Feature mode
    # - "raw": use progress/delay lags as-is
    # - "impact": use plan-weighted impact features (recommended)
    initiative_feature_mode: str = "impact"  # "raw" or "impact"

    # Dependency gating: if hard dependency behind, dampen downstream impact
    hard_dep_progress_threshold: float = 0.60
    hard_dep_dampen_factor: float = 0.50  # multiply effect by this when dependency < threshold


# ----------------------------
# Initiative feature plumbing (plan-driven)
# ----------------------------
def _initiative_impacts_index(plan: Plan) -> Dict[str, List[Tuple[str, int]]]:
    """
    Returns: kpi_id -> list of (initiative_id, lag_months) only for initiatives that declare impacts on that KPI.
    """
    out: Dict[str, List[Tuple[str, int]]] = {}
    for it in plan.initiatives:
        for imp in it.kpi_impacts:
            out.setdefault(imp.kpi_id, []).append((it.id, int(imp.lag_months)))
    return out


def _impact_params_by_kpi_init(plan: Plan) -> Dict[Tuple[str, str], Tuple[float, float, int]]:
    """
    (kpi_id, initiative_id) -> (expected_delta_by_end, confidence, lag_months)
    """
    out: Dict[Tuple[str, str], Tuple[float, float, int]] = {}
    for it in plan.initiatives:
        for imp in it.kpi_impacts:
            out[(imp.kpi_id, it.id)] = (float(imp.expected_delta_by_end), float(imp.confidence), int(imp.lag_months))
    return out


def _hard_dependencies(plan: Plan) -> Dict[str, List[str]]:
    """
    initiative_id -> list of hard dependency initiative_ids
    """
    out: Dict[str, List[str]] = {}
    for it in plan.initiatives:
        deps = [d.initiative_id for d in it.dependencies if str(d.type).lower() == "hard"]
        out[it.id] = deps
    return out


def _build_initiative_panel(initiatives_df: pd.DataFrame) -> pd.DataFrame:
    """
    Input columns expected: date, initiative_id, progress, delay_months (optional)
    Output: wide panel indexed by date with columns:
      init__<ID>__progress
      init__<ID>__delay
    """
    df = initiatives_df.copy()
    if not {"date", "initiative_id", "progress"}.issubset(df.columns):
        raise ValueError("initiatives_df must contain columns: date, initiative_id, progress")

    df["date"] = df["date"].astype(str)
    df["initiative_id"] = df["initiative_id"].astype(str)
    df["progress"] = df["progress"].apply(_safe_float)

    if "delay_months" in df.columns:
        df["delay_months"] = df["delay_months"].apply(_safe_float)
    else:
        df["delay_months"] = 0.0

    p = df.pivot_table(index="date", columns="initiative_id", values="progress", aggfunc="last")
    d = df.pivot_table(index="date", columns="initiative_id", values="delay_months", aggfunc="last")

    p.columns = [f"init__{c}__progress" for c in p.columns]
    d.columns = [f"init__{c}__delay" for c in d.columns]

    out = pd.concat([p, d], axis=1).sort_index().reset_index()
    return out


def _extend_initiatives_future(
    panel: pd.DataFrame,
    as_of: str,
    horizon_end: str,
    cfg: ForecastConfig,
) -> pd.DataFrame:
    """
    Ensures the initiative panel has rows for all forecast months.
    For future months, either hold last value or drift progress toward 1.
    """
    dates = month_range(as_of, horizon_end)
    base = panel.copy().sort_values("date")
    base = base[base["date"] <= as_of].copy()

    if base.empty:
        base = pd.DataFrame({"date": [as_of]})

    if as_of not in set(base["date"]):
        last = base.iloc[-1:].copy()
        last["date"] = as_of
        base = pd.concat([base, last], ignore_index=True).sort_values("date")

    cols = [c for c in base.columns if c != "date"]
    if not cols:
        return pd.DataFrame({"date": dates})

    last_row = base[base["date"] == as_of].iloc[-1]
    last_vals = {c: float(last_row[c]) for c in cols}

    future_rows = []
    prev = dict(last_vals)
    for d in dates:
        if d <= as_of:
            continue
        row = {"date": d}
        for c in cols:
            v = float(prev[c])
            if cfg.initiative_future_hold == "linear_to_1" and c.endswith("__progress"):
                v = min(1.0, v + 0.05 * (1.0 - v))
            row[c] = v
        prev = {c: float(row[c]) for c in cols}
        future_rows.append(row)

    fut = pd.DataFrame(future_rows) if future_rows else pd.DataFrame(columns=["date"] + cols)
    out = pd.concat([base[["date"] + cols], fut], ignore_index=True).sort_values("date")
    out = out.drop_duplicates(subset=["date"], keep="last").reset_index(drop=True)
    return out


# ----------------------------
# Seasonality features
# ----------------------------
def _month_season_feats(dt_obj: datetime) -> Tuple[float, float]:
    m = dt_obj.month
    ang = 2.0 * np.pi * (m - 1) / 12.0
    return float(np.sin(ang)), float(np.cos(ang))


# ----------------------------
# Feature building (supervised)
# ----------------------------
def _build_supervised(
    series: pd.DataFrame,
    cfg: ForecastConfig,
    exog: Optional[pd.DataFrame] = None,
    exog_cols: Optional[List[str]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    series: columns [date, value] sorted by date.
    exog: optional dataframe with [date] + exog feature columns.
    """
    vals = series["value"].astype(float).values
    dates = series["date"].astype(str).values

    exog_map: Dict[str, np.ndarray] = {}
    if exog is not None and exog_cols:
        e = exog.copy()
        e["date"] = e["date"].astype(str)
        e = e.set_index("date")
        for c in exog_cols:
            if c in e.columns:
                exog_map[c] = e[c].astype(float).reindex(dates).values

    max_lag = max(cfg.lags)
    rows_X: List[List[float]] = []
    rows_y: List[float] = []

    for i in range(max_lag, len(vals)):
        y_i = float(vals[i])
        if not np.isfinite(y_i):
            continue

        feats: List[float] = [1.0]
        ok = True
        for lag in cfg.lags:
            v_l = float(vals[i - lag])
            if not np.isfinite(v_l):
                ok = False
                break
            feats.append(v_l)
        if not ok:
            continue

        s, c = _month_season_feats(_dt(dates[i]))
        feats.extend([s, c])

        if exog_map:
            for c_name in exog_cols or []:
                arr = exog_map.get(c_name)
                v = float(arr[i]) if arr is not None else 0.0
                if not np.isfinite(v):
                    v = 0.0
                feats.append(v)

        rows_X.append(feats)
        rows_y.append(y_i)

    if not rows_X:
        return np.zeros((0, 0)), np.zeros((0,))

    return np.array(rows_X, dtype=float), np.array(rows_y, dtype=float)


def _feature_for_next(
    history_vals: List[float],
    next_date: str,
    cfg: ForecastConfig,
    exog_row: Optional[Dict[str, float]] = None,
    exog_cols: Optional[List[str]] = None,
) -> np.ndarray:
    feats: List[float] = [1.0]
    for lag in cfg.lags:
        feats.append(float(history_vals[-lag]))
    s, c = _month_season_feats(_dt(next_date))
    feats.extend([s, c])

    if exog_row is not None and exog_cols:
        for c in exog_cols:
            feats.append(float(exog_row.get(c, 0.0)))

    return np.array(feats, dtype=float)


# ----------------------------
# Exogenous feature builder (plan-weighted impacts)
# ----------------------------
def _lagged_date(date_iso: str, months: int) -> str:
    return (datetime.fromisoformat(date_iso) - relativedelta(months=months)).date().isoformat()


def _get_panel_value(base: pd.DataFrame, date_iso: str, col: str) -> float:
    if date_iso not in base.index or col not in base.columns:
        return 0.0
    v = base.loc[date_iso, col]
    if isinstance(v, (pd.Series, pd.DataFrame)):
        v = float(np.array(v).reshape(-1)[-1])
    try:
        v = float(v)
    except Exception:
        v = 0.0
    if not np.isfinite(v):
        v = 0.0
    return float(v)


def _build_exog_rows_for_kpi(
    plan: Plan,
    cfg: ForecastConfig,
    kpi_id: str,
    future_dates: List[str],
    init_panel: pd.DataFrame,
    impact_index: Dict[str, List[Tuple[str, int]]],
    impact_params: Dict[Tuple[str, str], Tuple[float, float, int]],
    hard_deps: Dict[str, List[str]],
) -> Tuple[List[str], Dict[str, Dict[str, float]]]:
    """
    Returns:
      exog_cols, exog_by_date (date -> {col: value})
    """
    allowed = impact_index.get(kpi_id, [])
    if not allowed:
        return [], {}

    # cap initiatives (not columns)
    allowed = allowed[: cfg.max_initiatives_features]

    base = init_panel.copy()
    base["date"] = base["date"].astype(str)
    base = base.set_index("date")

    exog_by_date: Dict[str, Dict[str, float]] = {}

    for d in future_dates:
        row: Dict[str, float] = {}

        for init_id, lag_m in allowed:
            # plan params
            delta_end, conf, _ = impact_params.get((kpi_id, init_id), (0.0, 0.0, lag_m))

            # lagged progress (by impact lag)
            d_lag = _lagged_date(d, lag_m)
            prog = _get_panel_value(base, d_lag, f"init__{init_id}__progress")
            delay = _get_panel_value(base, d_lag, f"init__{init_id}__delay")

            # dependency gating on hard deps: if any hard dep progress below threshold, dampen
            gate = 1.0
            for dep_id in hard_deps.get(init_id, []):
                dep_prog = _get_panel_value(base, d, f"init__{dep_id}__progress")
                if dep_prog < cfg.hard_dep_progress_threshold:
                    gate *= cfg.hard_dep_dampen_factor

            if cfg.initiative_feature_mode == "raw":
                # raw features (still lagged by impact lag)
                row[f"init__{init_id}__progress_lag{lag_m}"] = float(prog)
                row[f"init__{init_id}__delay_lag{lag_m}"] = float(delay)
            else:
                # impact-weighted feature (recommended)
                # progress * confidence * expected_delta_by_end * dependency_gate
                row[f"impact__{init_id}_lag{lag_m}"] = float(prog) * float(conf) * float(delta_end) * float(gate)
                # keep delay as a small auxiliary regressor (delays can depress impact effectiveness)
                row[f"delay__{init_id}_lag{lag_m}"] = float(delay)

        exog_by_date[d] = row

    exog_cols = sorted(next(iter(exog_by_date.values())).keys()) if exog_by_date else []
    return exog_cols, exog_by_date


# ----------------------------
# Public API
# ----------------------------
def forecast_kpis(
    plan: Plan,
    kpis_df: pd.DataFrame,
    as_of: str,
    horizon_end: str,
    initiatives_df: Optional[pd.DataFrame] = None,
    cfg: Optional[ForecastConfig] = None,
) -> pd.DataFrame:
    """
    Output columns:
      date, kpi_id, forecast, expected, lo, hi, sigma

    If initiatives_df is provided, uses initiative exogenous regressors
    but ONLY for initiatives that declare impacts on that KPI in plan.yaml (and lagged).
    """
    cfg = cfg or ForecastConfig()

    df = kpis_df.copy()
    if not {"date", "kpi_id", "value"}.issubset(df.columns):
        raise ValueError("kpis_df must contain columns: date, kpi_id, value")

    df["date"] = df["date"].astype(str)
    df["kpi_id"] = df["kpi_id"].astype(str)
    df["value"] = df["value"].apply(_safe_float)

    start_lb = (_month_floor(_dt(as_of)) - relativedelta(months=cfg.lookback_months)).date().isoformat()
    df = df[(df["date"] <= as_of) & (df["date"] >= start_lb)].sort_values(["kpi_id", "date"])

    future_dates = month_range(as_of, horizon_end)

    # plan indices
    impact_index = _initiative_impacts_index(plan)
    impact_params = _impact_params_by_kpi_init(plan)
    hard_deps = _hard_dependencies(plan)

    # initiative panel (extended into the future)
    init_panel: Optional[pd.DataFrame] = None
    if cfg.use_initiatives and initiatives_df is not None and not initiatives_df.empty:
        panel = _build_initiative_panel(initiatives_df)
        panel = _extend_initiatives_future(panel, as_of=as_of, horizon_end=horizon_end, cfg=cfg)
        init_panel = panel

    rows: List[Dict[str, float | str]] = []

    plan_kpi_ids = [k.id for k in plan.kpis]
    all_kpi_ids = sorted(set(plan_kpi_ids) | set(df["kpi_id"].unique().tolist()))

    for kpi_id in all_kpi_ids:
        g = df[df["kpi_id"] == kpi_id].sort_values("date")[["date", "value"]].dropna()
        g = g[g["value"].apply(np.isfinite)].copy()

        expected_map = {d: _expected_from_plan(plan, kpi_id, d) for d in future_dates}

        # exog per KPI
        exog_cols: List[str] = []
        exog_by_date: Dict[str, Dict[str, float]] = {}

        if init_panel is not None:
            exog_cols, exog_by_date = _build_exog_rows_for_kpi(
                plan=plan,
                cfg=cfg,
                kpi_id=kpi_id,
                future_dates=future_dates,
                init_panel=init_panel,
                impact_index=impact_index,
                impact_params=impact_params,
                hard_deps=hard_deps,
            )

        # baseline fallback if no KPI history
        if g.empty:
            base_val = None
            for k in plan.kpis:
                if k.id == kpi_id:
                    base_val = float(k.baseline.value)
                    break
            base_val = float(base_val) if base_val is not None else 0.0
            sigma = max(1e-6, abs(base_val) * 0.05)

            history_vals = [base_val, base_val, base_val, base_val]
            for d in future_dates:
                exp = expected_map.get(d)
                exp = float(exp) if exp is not None else base_val
                fc = exp
                lo = fc - cfg.z * sigma
                hi = fc + cfg.z * sigma
                rows.append({"date": d, "kpi_id": kpi_id, "forecast": fc, "expected": exp, "lo": lo, "hi": hi, "sigma": sigma})
            continue

        g = g[g["date"] <= as_of].sort_values("date").copy()

        base0 = float(g["value"].iloc[0])
        last_obs = float(g["value"].iloc[-1])
        exp_asof = expected_map.get(as_of)
        exp_asof = float(exp_asof) if exp_asof is not None else last_obs
        scale = max(1.0, abs(exp_asof - base0), abs(last_obs - base0))

        # training exog aligned to observed dates
        exog_df: Optional[pd.DataFrame] = None
        if exog_cols and exog_by_date:
            obs_dates = g["date"].astype(str).tolist()
            ex_rows = []
            for d in obs_dates:
                row = exog_by_date.get(d)
                if row is None:
                    row = {c: 0.0 for c in exog_cols}
                ex_rows.append({"date": d, **row})
            exog_df = pd.DataFrame(ex_rows)

        X, y = _build_supervised(g, cfg, exog=exog_df, exog_cols=exog_cols if exog_df is not None else None)
        use_model = (len(y) >= cfg.min_points) and (X.shape[1] > 0)

        if use_model:
            w = _ridge_fit(X, y, alpha=cfg.ridge_alpha)
            yhat = _ridge_predict(X, w)
            resid = y - yhat
            sigma = float(np.std(resid)) if len(resid) >= 2 else 0.0
        else:
            w = None
            sigma = 0.0

        sigma_floor = cfg.sigma_floor_frac_of_scale * scale
        sigma = float(max(sigma, sigma_floor, 1e-6))

        max_lag = max(cfg.lags)
        history_vals = g["value"].astype(float).tolist()
        if len(history_vals) < max_lag + 1:
            while len(history_vals) < max_lag + 1:
                history_vals.insert(0, float(history_vals[0]))

        # recursive forecast
        for d in future_dates:
            exp = expected_map.get(d)
            exp = float(exp) if exp is not None else float(history_vals[-1])

            ex_row = exog_by_date.get(d, {}) if exog_cols else None

            if use_model and w is not None:
                x_next = _feature_for_next(history_vals, d, cfg, exog_row=ex_row, exog_cols=exog_cols)
                fc = float(_ridge_predict(x_next.reshape(1, -1), w)[0])
            else:
                prev = float(history_vals[-1])
                fc = 0.85 * prev + 0.15 * exp

            history_vals.append(fc)

            lo = fc - cfg.z * sigma
            hi = fc + cfg.z * sigma

            rows.append({"date": d, "kpi_id": kpi_id, "forecast": fc, "expected": exp, "lo": lo, "hi": hi, "sigma": sigma})

    return pd.DataFrame(rows).sort_values(["date", "kpi_id"]).reset_index(drop=True)
