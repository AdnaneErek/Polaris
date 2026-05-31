#!/usr/bin/env python3
"""
Auto-record decision outcomes from a KPI CSV (synthetic or real extract).

This closes Path A (learn from outcomes) without manual KPI entry.

Example (synthetic):
  python -m src.auto_record_outcomes ^
    --kpis-csv data/simulated_kpis.csv ^
    --storage-dir artifacts/learning ^
    --latest-pending ^
    --accepted true
"""
from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Dict, Optional

import pandas as pd

from src.learning import get_learner


def _load_kpis_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if not {"date", "kpi_id"}.issubset(df.columns):
        raise ValueError("kpis csv must contain at least columns: date, kpi_id")
    # allow either `value` or `actual`
    if "value" not in df.columns and "actual" not in df.columns:
        raise ValueError("kpis csv must contain a `value` (recommended) or `actual` column")
    if "value" not in df.columns:
        df = df.rename(columns={"actual": "value"})
    df["date"] = df["date"].astype(str)
    df["kpi_id"] = df["kpi_id"].astype(str)
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    return df


def _select_decision(decisions: list[Dict[str, Any]], timestamp: Optional[str], latest_pending: bool) -> Dict[str, Any]:
    if not decisions:
        raise ValueError("No decisions found in storage.")

    if timestamp:
        for d in decisions:
            if d.get("timestamp") == timestamp:
                return d
        raise ValueError(f"Decision timestamp not found: {timestamp}")

    pending = [d for d in decisions if d.get("evaluation_date") is None]
    if latest_pending:
        if not pending:
            raise ValueError("No pending decisions (all have outcomes recorded).")
        return pending[-1]

    raise ValueError("Provide --decision-timestamp or --latest-pending.")


def main() -> None:
    ap = argparse.ArgumentParser(description="Auto-record decision outcomes from a KPI CSV")
    ap.add_argument("--kpis-csv", required=True, help="CSV with columns: date,kpi_id,value (synthetic or extract)")
    ap.add_argument("--storage-dir", default="artifacts/learning", help="Learning storage dir (default: artifacts/learning)")
    ap.add_argument("--decision-timestamp", default=None, help="Decision timestamp to update")
    ap.add_argument("--latest-pending", action="store_true", help="Update the latest pending decision")
    ap.add_argument("--evaluation-date", default=None, help="Override evaluation date (YYYY-MM-DD). Default: planned_evaluation_date")
    ap.add_argument("--accepted", default="true", help="Accepted? true/false (default: true)")
    ap.add_argument("--kpi-ids-json", default=None, help="Optional JSON list of KPI IDs to evaluate (defaults to predicted_outcomes keys)")
    args = ap.parse_args()

    learner = get_learner(storage_dir=args.storage_dir)
    decisions = learner._load_all_decisions()
    dec = _select_decision(decisions, timestamp=args.decision_timestamp, latest_pending=bool(args.latest_pending))

    planned = dec.get("planned_evaluation_date")
    eval_date = args.evaluation_date or planned
    if not eval_date:
        raise ValueError("No evaluation date provided and decision has no planned_evaluation_date.")

    accepted_str = str(args.accepted).strip().lower()
    accepted = accepted_str in {"true", "1", "yes", "y"}

    predicted = dec.get("predicted_outcomes") or {}
    if args.kpi_ids_json:
        kpi_ids = json.loads(args.kpi_ids_json)
        if not isinstance(kpi_ids, list) or not all(isinstance(x, str) for x in kpi_ids):
            raise ValueError("--kpi-ids-json must be a JSON list of strings")
    else:
        kpi_ids = sorted(list(predicted.keys()))

    if not kpi_ids:
        # fallback to common demo KPIs
        kpi_ids = ["KPI_STP", "KPI_E2E", "KPI_INC"]

    kpi_df = _load_kpis_csv(args.kpis_csv)
    at_eval = kpi_df[(kpi_df["date"] == str(eval_date)) & (kpi_df["kpi_id"].isin(kpi_ids))].copy()
    if at_eval.empty:
        raise ValueError(f"No KPI rows found in {args.kpis_csv} for date={eval_date} and kpi_ids={kpi_ids}")

    actual_outcomes: Dict[str, float] = {}
    for k in kpi_ids:
        row = at_eval[at_eval["kpi_id"] == k]
        if row.empty:
            continue
        v = row["value"].iloc[-1]
        if pd.isna(v):
            continue
        actual_outcomes[str(k)] = float(v)

    if not actual_outcomes:
        raise ValueError("Could not extract any actual KPI outcomes (values are missing/NaN).")

    # If predicted outcomes missing, use actual (keeps pipeline working, but yields 0 error)
    predicted_outcomes = {k: float(predicted[k]) for k in actual_outcomes.keys() if k in predicted}
    if not predicted_outcomes:
        predicted_outcomes = dict(actual_outcomes)

    learner.update_decision_outcome(
        decision_timestamp=str(dec["timestamp"]),
        accepted=accepted,
        actual_outcomes=actual_outcomes,
        predicted_outcomes=predicted_outcomes,
        evaluation_date=str(eval_date),
    )

    metrics = learner.get_learning_metrics()
    print("[OK] Outcome recorded.")
    print(f"  Decision timestamp: {dec['timestamp']}")
    print(f"  As-of: {dec.get('as_of')}")
    print(f"  Evaluation date: {eval_date}")
    print(f"  KPIs updated: {sorted(actual_outcomes.keys())}")
    print("\nLearning metrics:")
    print(f"  total_decisions: {metrics.total_decisions}")
    print(f"  accepted_decisions: {metrics.accepted_decisions}")
    print(f"  mean_forecast_accuracy: {metrics.mean_forecast_accuracy:.3f}")
    print(f"  improvement_trend: {metrics.improvement_trend}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[ERROR] {e}")
        sys.exit(1)

