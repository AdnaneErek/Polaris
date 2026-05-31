# src/forecast_proba.py
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta

from src.schemas import Plan
from src.forecast import ForecastConfig, _expected_from_plan  # reuse your utilities


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
# Tiny Ridge (no sklearn)
# ----------------------------
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


def _build_supervised_autoreg(series: pd.DataFrame, lags: Tuple[int, ...]) -> Tuple[np.ndarray, np.ndarray]:
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


@dataclass(frozen=True)
class ProbaForecastConfig:
    # Model
    lookback_months: int = 24
    min_points: int = 10
    lags: Tuple[int, ...] = (1, 2, 3)
    ridge_alpha: float = 3.0

    # Uncertainty
    n_samples: int = 500
    # If integrity score is high, inflate sigma (measurement risk)
    integrity_sigma_mult: float = 2.0
    # Always keep some uncertainty
    sigma_floor_frac_of_scale: float = 0.01

    # Forecast interval output
    qs: Tuple[float, float, float] = (0.10, 0.50, 0.90)


def forecast_kpis_proba(
    plan: Plan,
    kpis_df: pd.DataFrame,
    as_of: str,
    horizon_end: str,
    integrity_scores: Optional[Dict[str, float]] = None,  # 0..1 suspiciousness
    cfg: Optional[ProbaForecastConfig] = None,
) -> pd.DataFrame:
    """
    Produces probabilistic forecasts using:
      - ridge autoregression + seasonality
      - residual bootstrap for uncertainty

    Output columns:
      date, kpi_id, p10, p50, p90, sigma, n_samples
    """
    cfg = cfg or ProbaForecastConfig()
    integrity_scores = integrity_scores or {}

    df = kpis_df.copy()
    if not {"date", "kpi_id", "value"}.issubset(df.columns):
        raise ValueError("kpis_df must contain columns: date, kpi_id, value")

    df["date"] = df["date"].astype(str)
    df["kpi_id"] = df["kpi_id"].astype(str)
    df["value"] = df["value"].apply(_safe_float)

    start_lb = (_month_floor(_dt(as_of)) - relativedelta(months=cfg.lookback_months)).date().isoformat()
    df = df[(df["date"] <= as_of) & (df["date"] >= start_lb)].sort_values(["kpi_id", "date"])

    future_dates = month_range(as_of, horizon_end)
    plan_kpis = [k.id for k in plan.kpis]
    all_kpi_ids = sorted(set(plan_kpis) | set(df["kpi_id"].unique().tolist()))

    rng = np.random.default_rng(7)
    rows: List[Dict[str, float | str | int]] = []

    for kpi_id in all_kpi_ids:
        g = df[df["kpi_id"] == kpi_id].sort_values("date")[["date", "value"]].dropna()
        g = g[g["value"].apply(np.isfinite)].copy()

        # If missing history: fallback to plan expected with wide uncertainty
        if g.empty:
            base_val = None
            for k in plan.kpis:
                if k.id == kpi_id:
                    base_val = float(k.baseline.value)
                    break
            base_val = float(base_val) if base_val is not None else 0.0
            scale = max(1.0, abs(base_val))
            sigma = max(1e-6, 0.10 * scale)

            for d in future_dates:
                exp = _expected_from_plan(plan, kpi_id, d)
                mu = float(exp) if exp is not None else base_val
                samples = rng.normal(mu, sigma, size=cfg.n_samples)
                q10, q50, q90 = np.quantile(samples, cfg.qs).tolist()
                rows.append(
                    {"date": d, "kpi_id": kpi_id, "p10": float(q10), "p50": float(q50), "p90": float(q90),
                     "sigma": float(sigma), "n_samples": int(cfg.n_samples)}
                )
            continue

        # Build model
        g = g[g["date"] <= as_of].sort_values("date").copy()
        base0 = float(g["value"].iloc[0])
        last_obs = float(g["value"].iloc[-1])
        exp_asof = _expected_from_plan(plan, kpi_id, as_of)
        exp_asof = float(exp_asof) if exp_asof is not None else last_obs
        scale = max(1.0, abs(exp_asof - base0), abs(last_obs - base0))

        X, y = _build_supervised_autoreg(g, lags=cfg.lags)
        use_model = (len(y) >= cfg.min_points) and (X.shape[1] > 0)

        if use_model:
            w = _ridge_fit(X, y, alpha=cfg.ridge_alpha)
            yhat = _ridge_predict(X, w)
            resid = (y - yhat)
            sigma = float(np.std(resid)) if len(resid) >= 2 else 0.0
            resid_pool = resid if resid.size else np.array([0.0])
        else:
            w = None
            sigma = 0.0
            resid_pool = np.array([0.0])

        sigma_floor = cfg.sigma_floor_frac_of_scale * scale
        sigma = float(max(sigma, sigma_floor, 1e-6))

        # Inflate sigma if integrity is suspicious
        integ = float(integrity_scores.get(kpi_id, 0.0))
        if integ >= 0.6:
            sigma *= cfg.integrity_sigma_mult

        # Recursive bootstrap sampling
        # We sample paths, not just points: uncertainty compounds over horizon
        hist = g["value"].astype(float).tolist()
        max_lag = max(cfg.lags)
        while len(hist) < max_lag + 1:
            hist.insert(0, float(hist[0]))

        # For each future month, collect samples across paths
        path_histories = [hist[:] for _ in range(cfg.n_samples)]

        for d in future_dates:
            samples_t = []
            for s_idx in range(cfg.n_samples):
                h = path_histories[s_idx]
                if use_model and w is not None:
                    x = _feature_next(h, d, lags=cfg.lags)
                    mu = float(_ridge_predict(x.reshape(1, -1), w)[0])
                else:
                    # conservative drift toward plan expected
                    exp = _expected_from_plan(plan, kpi_id, d)
                    exp = float(exp) if exp is not None else float(h[-1])
                    mu = 0.85 * float(h[-1]) + 0.15 * exp

                eps = float(rng.choice(resid_pool))
                y_next = mu + eps
                h.append(float(y_next))
                samples_t.append(float(y_next))

            q10, q50, q90 = np.quantile(np.array(samples_t), cfg.qs).tolist()
            rows.append(
                {"date": d, "kpi_id": kpi_id, "p10": float(q10), "p50": float(q50), "p90": float(q90),
                 "sigma": float(sigma), "n_samples": int(cfg.n_samples)}
            )

    return pd.DataFrame(rows).sort_values(["date", "kpi_id"]).reset_index(drop=True)
