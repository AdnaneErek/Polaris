from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.forecast_sliding import SlidingForecastConfig, forecast_kpis_sliding, walk_forward_backtest


def main() -> None:
    ap = argparse.ArgumentParser(description="Strict no-leakage sliding-window KPI forecast")
    ap.add_argument("--kpis", default="data/simulated_kpis.csv")
    ap.add_argument("--as_of", default="2027-06-01")
    ap.add_argument("--horizon_months", type=int, default=6)
    ap.add_argument("--window_months", type=int, default=36)
    ap.add_argument("--out_dir", default="artifacts/forecast")
    ap.add_argument("--backtest_start", default=None, help="Optional walk-forward start as_of")
    ap.add_argument("--backtest_end", default=None, help="Optional walk-forward end as_of")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    kpis = pd.read_csv(args.kpis)
    cfg = SlidingForecastConfig(window_months=args.window_months, horizon_months=args.horizon_months)

    pred = forecast_kpis_sliding(kpis, as_of=args.as_of, cfg=cfg, horizon_months=args.horizon_months)
    pred_path = out_dir / f"forecast_{args.as_of}.csv"
    pred.to_csv(pred_path, index=False)
    print(f"[OK] Wrote forecast: {pred_path} rows={len(pred)}")

    if args.backtest_start and args.backtest_end:
        bt = walk_forward_backtest(
            kpis_df=kpis,
            start_as_of=args.backtest_start,
            end_as_of=args.backtest_end,
            cfg=cfg,
            horizon_months=args.horizon_months,
        )
        bt_path = out_dir / f"backtest_{args.backtest_start}_to_{args.backtest_end}.csv"
        bt.to_csv(bt_path, index=False)
        print(f"[OK] Wrote backtest: {bt_path} rows={len(bt)}")

        if not bt.empty:
            summary = (
                bt.dropna(subset=["actual"])
                .groupby("kpi_id", as_index=False)
                .agg(mae=("abs_error", "mean"), rmse=("sq_error", lambda s: float((s.mean()) ** 0.5)))
            )
            summary_path = out_dir / f"backtest_summary_{args.backtest_start}_to_{args.backtest_end}.csv"
            summary.to_csv(summary_path, index=False)
            print(f"[OK] Wrote backtest summary: {summary_path}")
            print(summary.to_string(index=False))


if __name__ == "__main__":
    main()

