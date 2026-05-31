from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta


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


def add_months(date_iso: str, months: int) -> str:
    d = _month_floor(_dt(date_iso))
    return (d + relativedelta(months=months)).date().isoformat()


def _ridge_fit(X: np.ndarray, y: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    XtX = X.T @ X
    I = np.eye(XtX.shape[0], dtype=float)
    return np.linalg.solve(XtX + alpha * I, X.T @ y)


def _ridge_predict(X: np.ndarray, w: np.ndarray) -> np.ndarray:
    return X @ w


def _month_season_feats(dt_obj: datetime) -> Tuple[float, float]:
    m = dt_obj.month
    ang = 2.0 * np.pi * (m - 1) / 12.0
    return float(np.sin(ang)), float(np.cos(ang))


@dataclass(frozen=True)
class SlidingForecastConfig:
    window_months: int = 36
    min_points: int = 18
    lags: Tuple[int, ...] = (1, 2, 3, 6, 12)
    ridge_alpha: float = 2.0
    horizon_months: int = 6


def _build_supervised(series: pd.DataFrame, lags: Tuple[int, ...]) -> Tuple[np.ndarray, np.ndarray]:
    vals = series["value"].astype(float).values
    dates = series["date"].astype(str).values
    max_lag = max(lags)

    X_rows: List[List[float]] = []
    y_rows: List[float] = []

    for i in range(max_lag, len(vals)):
        y_i = float(vals[i])
        if not np.isfinite(y_i):
            continue
        feats: List[float] = [1.0]
        ok = True
        for lag in lags:
            v_l = float(vals[i - lag])
            if not np.isfinite(v_l):
                ok = False
                break
            feats.append(v_l)
        if not ok:
            continue
        s, c = _month_season_feats(_dt(dates[i]))
        feats.extend([s, c])
        X_rows.append(feats)
        y_rows.append(y_i)

    if not X_rows:
        return np.zeros((0, 0)), np.zeros((0,))
    return np.array(X_rows, dtype=float), np.array(y_rows, dtype=float)


def _feature_next(history_vals: List[float], next_date: str, lags: Tuple[int, ...]) -> np.ndarray:
    feats: List[float] = [1.0]
    for lag in lags:
        feats.append(float(history_vals[-lag]))
    s, c = _month_season_feats(_dt(next_date))
    feats.extend([s, c])
    return np.array(feats, dtype=float)


def forecast_kpis_sliding(
    kpis_df: pd.DataFrame,
    as_of: str,
    cfg: Optional[SlidingForecastConfig] = None,
    horizon_months: Optional[int] = None,
) -> pd.DataFrame:
    """
    Strict no-leakage forecast:
    - train using only points with date <= as_of
    - use trailing window_months (sliding window)
    - forecast next horizon_months
    """
    cfg = cfg or SlidingForecastConfig()
    h = int(horizon_months if horizon_months is not None else cfg.horizon_months)

    df = kpis_df.copy()
    if not {"date", "kpi_id", "value"}.issubset(df.columns):
        raise ValueError("kpis_df must contain columns: date, kpi_id, value")
    df["date"] = df["date"].astype(str)
    df["kpi_id"] = df["kpi_id"].astype(str)
    df["value"] = pd.to_numeric(df["value"], errors="coerce")

    as_of_m = _month_floor(_dt(as_of)).date().isoformat()
    start_lb = (_month_floor(_dt(as_of_m)) - relativedelta(months=cfg.window_months)).date().isoformat()
    df = df[(df["date"] <= as_of_m) & (df["date"] >= start_lb)].sort_values(["kpi_id", "date"])

    future_dates = [add_months(as_of_m, i) for i in range(1, h + 1)]
    rows: List[Dict[str, object]] = []

    for kpi_id in sorted(df["kpi_id"].unique().tolist()):
        g = df[df["kpi_id"] == kpi_id].sort_values("date")[["date", "value"]].dropna()
        g = g[g["value"].apply(np.isfinite)].copy()
        if g.empty:
            continue

        X, y = _build_supervised(g, lags=cfg.lags)
        use_model = (len(y) >= cfg.min_points) and (X.shape[1] > 0)

        if use_model:
            w = _ridge_fit(X, y, alpha=cfg.ridge_alpha)
        else:
            w = None

        max_lag = max(cfg.lags)
        hist = g["value"].astype(float).tolist()
        while len(hist) < max_lag + 1:
            hist.insert(0, float(hist[0]))

        last_obs = float(hist[-1])
        for d in future_dates:
            if use_model and w is not None:
                x_next = _feature_next(hist, d, cfg.lags)
                fc = float(_ridge_predict(x_next.reshape(1, -1), w)[0])
            else:
                # naive fallback if insufficient history
                fc = float(last_obs)
            hist.append(fc)
            rows.append(
                {
                    "as_of": as_of_m,
                    "date": d,
                    "kpi_id": kpi_id,
                    "forecast": float(fc),
                    "model": "ridge_lag" if use_model else "naive_last_value",
                }
            )

    return pd.DataFrame(rows).sort_values(["kpi_id", "date"]).reset_index(drop=True)


def walk_forward_backtest(
    kpis_df: pd.DataFrame,
    start_as_of: str,
    end_as_of: str,
    cfg: Optional[SlidingForecastConfig] = None,
    horizon_months: Optional[int] = None,
) -> pd.DataFrame:
    """
    Rolling-origin evaluation:
    for each as_of in [start_as_of..end_as_of], train on history<=as_of, predict horizon, then score where actual exists.
    """
    cfg = cfg or SlidingForecastConfig()
    h = int(horizon_months if horizon_months is not None else cfg.horizon_months)

    all_as_of = month_range(start_as_of, end_as_of)
    kdf = kpis_df.copy()
    kdf["date"] = kdf["date"].astype(str)
    kdf["kpi_id"] = kdf["kpi_id"].astype(str)
    kdf["value"] = pd.to_numeric(kdf["value"], errors="coerce")

    rows: List[pd.DataFrame] = []
    for as_of in all_as_of:
        pred = forecast_kpis_sliding(kdf, as_of=as_of, cfg=cfg, horizon_months=h)
        if pred.empty:
            continue
        act = kdf[["date", "kpi_id", "value"]].rename(columns={"value": "actual"})
        merged = pred.merge(act, on=["date", "kpi_id"], how="left")
        merged["abs_error"] = (merged["forecast"] - merged["actual"]).abs()
        merged["sq_error"] = (merged["forecast"] - merged["actual"]) ** 2
        rows.append(merged)

    if not rows:
        return pd.DataFrame(columns=["as_of", "date", "kpi_id", "forecast", "model", "actual", "abs_error", "sq_error"])
    return pd.concat(rows, ignore_index=True)

