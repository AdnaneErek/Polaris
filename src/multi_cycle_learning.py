#!/usr/bin/env python3
"""
Multi-Cycle Learning Loop Validation Script.

This script validates the continuous learning loop by:
1. Generating multiple SteerCo packs over time (e.g., monthly)
2. Recording decisions and option selections
3. Simulating actual outcomes using the what-if engine
4. Running the learning loop to recalibrate forecasts/confidence/scoring
5. Validating that predictions improve over cycles

Example:
    python -m src.multi_cycle_learning \
        --start-date 2026-02-01 \
        --end-date 2026-09-01 \
        --cycle-months 3 \
        --evaluation-months 3 \
        --out-dir artifacts/learning_validation
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from src.load_plan import load_plan
from src.steerco_pack import build_steerco_pack, SteerCoPack
from src.learning import get_learner, DecisionLearner
from src.outcome_simulation import simulate_actual_outcomes, extract_predicted_outcomes
from src.portfolio import ActionBundle
from src.actions import SteeringAction
from src.whatif import StressEvent, WhatIfStochasticConfig, WhatIfConfig
from src.forecast import ForecastConfig, forecast_kpis
from dateutil.relativedelta import relativedelta
from datetime import datetime


def _shift_date_by_months(date_str: str, months: int) -> str:
    """Shift a date string by N months."""
    dt = datetime.fromisoformat(date_str)
    new_dt = dt + relativedelta(months=months)
    return new_dt.date().isoformat()


def _find_bundle_by_id(bundles: List[ActionBundle], bundle_id: str) -> Optional[ActionBundle]:
    """Find a bundle by its ID."""
    for bundle in bundles:
        if bundle.id == bundle_id:
            return bundle
    return None


def _extract_bundles_from_pack(pack: SteerCoPack) -> List[ActionBundle]:
    """Extract ActionBundle objects from a SteerCoPack."""
    bundles = []
    for bundle_dict in pack.bundles:
        actions = []
        for action_dict in bundle_dict.get("actions", []):
            actions.append(SteeringAction(
                id=action_dict.get("id", ""),
                type=action_dict.get("type", ""),
                target_initiative=action_dict.get("target_initiative", ""),
                parameters=action_dict.get("parameters", {}),
                description=action_dict.get("description", ""),
            ))
        bundles.append(ActionBundle(
            id=bundle_dict.get("id", ""),
            name=bundle_dict.get("name", ""),
            description=bundle_dict.get("description", ""),
            actions=actions,
            requires_approval=bundle_dict.get("requires_approval", True),
        ))
    return bundles


def run_cycle(
    plan_path: str,
    kpis_csv: str,
    initiatives_csv: str,
    kpis_df: pd.DataFrame,
    initiatives_df: pd.DataFrame,
    as_of: str,
    horizon_end: str,
    evaluation_date: str,
    learner: DecisionLearner,
    out_dir: Path,
    cycle_num: int,
    use_llm_bundles: bool = False,
) -> Dict[str, Any]:
    """
    Run a single cycle: generate pack, record decision, simulate outcomes, update learning.
    
    Returns:
        Dict with cycle results including prediction errors, learning metrics, etc.
    """
    print(f"\n{'='*80}")
    print(f"CYCLE {cycle_num}: as_of={as_of}, evaluation_date={evaluation_date}")
    print(f"{'='*80}")
    
    # Get learning parameters (from previous cycles)
    forecast_recalibration = learner.get_forecast_recalibration_params()
    scoring_weight_overrides = learner.get_improved_scoring_weights()
    confidence_multiplier = learner.get_confidence_calibration_multiplier()
    
    print(f"Learning state:")
    print(f"  Forecast recalibration: {len(forecast_recalibration)} KPIs")
    print(f"  Scoring weight overrides: {len(scoring_weight_overrides)} KPIs")
    print(f"  Confidence multiplier: {confidence_multiplier:.3f}")
    
    # Generate bundles (use LLM if requested, otherwise manual fallback)
    bundles: List[ActionBundle] = []
    bundle_generation: Optional[Dict[str, Any]] = None
    
    if use_llm_bundles:
        try:
            from src.bundle_generator import generate_action_bundles_with_llm
            bundles = generate_action_bundles_with_llm(
                plan_path=plan_path,
                kpis_csv=kpis_csv,
                initiatives_csv=initiatives_csv,
                as_of=as_of,
                n_bundles=4,
            )
            bundle_generation = {"mode": "llm", "count": len(bundles)}
        except Exception as e:
            print(f"[WARNING] LLM bundle generation failed: {e}. Using manual fallback.")
            use_llm_bundles = False
    
    if not bundles:
        # Manual fallback bundles
        from src.steerco_run import _default_bundles
        bundles = _default_bundles()
        bundle_generation = {"mode": "manual", "count": len(bundles)}
    
    # Generate SteerCo pack
    print(f"\n[1/5] Generating SteerCo pack...")
    # Note: build_steerco_pack reads CSVs directly, but it filters by as_of internally
    pack = build_steerco_pack(
        plan_path=plan_path,
        kpis_csv=kpis_csv,  # Pass CSV path - build_steerco_pack will filter by as_of
        initiatives_csv=initiatives_csv,
        as_of=as_of,
        horizon_end=horizon_end,
        bundles=bundles,
        stress_events=[],
        mc_cfg=WhatIfStochasticConfig(n_samples=200, seed=42 + cycle_num),  # Vary seed per cycle
        whatif_cfg=WhatIfConfig(),
        forecast_cfg=ForecastConfig(),
        out_dir=None,  # Don't write files yet
        bundle_generation=bundle_generation,
    )
    
    # Save pack to cycle directory
    cycle_dir = out_dir / f"cycle_{cycle_num:02d}_{as_of}"
    cycle_dir.mkdir(parents=True, exist_ok=True)
    
    # Serialize pack to JSON (matching structure from build_steerco_pack)
    from dataclasses import asdict
    pack_dict = {
        "as_of": pack.as_of,
        "horizon_end": pack.horizon_end,
        "sources": {
            "plan_path": plan_path,
            "kpis_csv": kpis_csv,
            "initiatives_csv": initiatives_csv,
        },
        "bundles": [
            {
                "id": b.id,
                "name": b.name,
                "description": b.description,
                "actions": [
                    {
                        "id": a.id,
                        "type": a.type,
                        "target_initiative": a.target_initiative,
                        "parameters": dict(a.parameters),
                        "description": a.description,
                    }
                    for a in b.actions
                ],
            }
            for b in bundles
        ],
        "integrity_report": pack.integrity_report,
        "integrity_scores": pack.integrity_scores,
        "forecast_head": pack.forecast_head,
        "options_table_mc": pack.options_table_mc,
        "options_mc": pack.options_mc,
        "recommendation": asdict(pack.recommendation),
        "guardrails_report": pack.guardrails_report,
        "options_explain": pack.options_explain,
        "option_narratives": pack.option_narratives,
        "ml_anomalies": pack.ml_anomalies,
        "weekly_objectives": pack.weekly_objectives,
        "forecast_deviation_alerts": pack.forecast_deviation_alerts,
        "learning_metrics": pack.learning_metrics,
        "situation_summaries": pack.situation_summaries,
        "bundle_generation": pack.bundle_generation,
        "forecast_full": pack.forecast_full.to_dict(orient="records") if hasattr(pack, 'forecast_full') and pack.forecast_full is not None else None,
    }
    
    pack_json_path = cycle_dir / "steerco_pack.json"
    with open(pack_json_path, "w", encoding="utf-8") as f:
        json.dump(pack_dict, f, indent=2, default=str, ensure_ascii=False)
    print(f"  Pack saved to {pack_json_path}")
    
    # Extract recommendation
    rec = pack.recommendation
    selected_bundle_id = rec.chosen_bundle_id
    selected_bundle_name = rec.chosen_bundle_name
    
    # Find the top option result to get scores
    top_result = None
    for opt in pack.options_mc:
        if opt.get("bundle_id") == selected_bundle_id:
            top_result = opt
            break
    
    score_stress_cvar10 = float(top_result.get("score_stress_cvar10", 0.0)) if top_result else 0.0
    score_stress_mean = float(top_result.get("score_stress_mean", 0.0)) if top_result else 0.0
    
    print(f"\n[2/5] Recording decision...")
    print(f"  Recommended: {selected_bundle_name} ({selected_bundle_id})")
    print(f"  Confidence: {rec.confidence:.3f}")
    print(f"  Guardrails passed: {rec.guardrails_passed}")
    print(f"  Score (stress CVaR10): {score_stress_cvar10:.3f}")
    print(f"  Score (stress mean): {score_stress_mean:.3f}")
    
    # Extract predicted outcomes from a fresh forecast (no reliance on internal pack attributes)
    # We intentionally recompute the forecast here using the same plan and KPI history
    # that the SteerCo pack uses, to avoid depending on non-existent pack.forecast_full.
    plan = load_plan(plan_path)
    # Restrict KPI and initiative history to avoid leakage and reuse the main forecast engine
    kpi_history = kpis_df[kpis_df["date"] <= as_of].copy()
    init_history = initiatives_df.copy()
    forecast_cfg = ForecastConfig()
    forecast_df = forecast_kpis(
        plan=plan,
        kpis_df=kpi_history,
        as_of=as_of,
        horizon_end=evaluation_date,
        initiatives_df=init_history,
        cfg=forecast_cfg,
    )
    predicted_outcomes = extract_predicted_outcomes(
        forecast_df=forecast_df,
        evaluation_date=evaluation_date,
    )
    print(f"  Predicted outcomes: {predicted_outcomes}")
    
    # Record decision
    decision = learner.record_decision(
        as_of=as_of,
        recommended_bundle_id=selected_bundle_id,
        recommended_bundle_name=selected_bundle_name,
        recommended_score_stress_cvar10=score_stress_cvar10,
        recommended_score_stress_mean=score_stress_mean,
        confidence=rec.confidence,
        guardrails_passed=rec.guardrails_passed,
        planned_evaluation_date=evaluation_date,
        predicted_outcomes=predicted_outcomes,
    )
    print(f"  Decision recorded: {decision.timestamp}")
    
    # Record option selection (what SteerCo actually chose)
    print(f"\n[3/5] Recording option selection...")
    learner.record_option_selection(
        decision_timestamp=decision.timestamp,
        selected_bundle_id=selected_bundle_id,
        selected_bundle_name=selected_bundle_name,
        accepted=True,
    )
    print(f"  Selected option recorded: {selected_bundle_id}")
    
    # Simulate actual outcomes
    print(f"\n[4/5] Simulating actual outcomes...")
    selected_bundle = _find_bundle_by_id(bundles, selected_bundle_id)
    if not selected_bundle:
        raise ValueError(f"Selected bundle {selected_bundle_id} not found in bundles list")
    
    actual_outcomes = simulate_actual_outcomes(
        plan=plan,
        kpis_df=kpis_df,
        initiatives_df=initiatives_df,
        selected_bundle=selected_bundle,
        original_as_of=as_of,
        evaluation_date=evaluation_date,
        noise_level=0.05,  # 5% noise
        random_seed=42 + cycle_num,
    )
    print(f"  Actual outcomes: {actual_outcomes}")
    
    # Compute prediction errors
    prediction_errors = {
        kpi_id: abs(actual_outcomes.get(kpi_id, 0) - predicted_outcomes.get(kpi_id, 0))
        for kpi_id in set(list(actual_outcomes.keys()) + list(predicted_outcomes.keys()))
    }
    mean_absolute_error = sum(prediction_errors.values()) / len(prediction_errors) if prediction_errors else 0.0
    print(f"  Prediction errors: {prediction_errors}")
    print(f"  Mean absolute error: {mean_absolute_error:.3f}")
    
    # Update decision with outcomes
    print(f"\n[5/5] Updating decision with outcomes...")
    learner.update_decision_outcome(
        decision_timestamp=decision.timestamp,
        accepted=True,
        actual_outcomes=actual_outcomes,
        predicted_outcomes=predicted_outcomes,
        evaluation_date=evaluation_date,
    )
    print(f"  Decision updated with outcomes")
    
    # Get updated learning metrics
    metrics = learner.get_learning_metrics()
    
    cycle_results = {
        "cycle_num": cycle_num,
        "as_of": as_of,
        "evaluation_date": evaluation_date,
        "selected_bundle_id": selected_bundle_id,
        "selected_bundle_name": selected_bundle_name,
        "predicted_outcomes": predicted_outcomes,
        "actual_outcomes": actual_outcomes,
        "prediction_errors": prediction_errors,
        "mean_absolute_error": mean_absolute_error,
        "learning_metrics": {
            "total_decisions": metrics.total_decisions,
            "accepted_decisions": metrics.accepted_decisions,
            "mean_forecast_accuracy": metrics.mean_forecast_accuracy,
            "mean_score_prediction_error": metrics.mean_score_prediction_error,
            "improvement_trend": metrics.improvement_trend,
        },
    }
    
    return cycle_results


def main() -> None:
    ap = argparse.ArgumentParser(description="Multi-cycle learning loop validation")
    ap.add_argument("--plan-path", default="data/plan.yaml", help="Path to plan.yaml")
    ap.add_argument("--kpis-csv", default="data/simulated_kpis.csv", help="Path to KPI CSV")
    ap.add_argument("--initiatives-csv", default="data/simulated_initiatives.csv", help="Path to initiatives CSV")
    ap.add_argument("--start-date", default="2026-02-01", help="Start date for cycles (YYYY-MM-DD)")
    ap.add_argument("--end-date", default="2026-09-01", help="End date for cycles (YYYY-MM-DD)")
    ap.add_argument("--cycle-months", type=int, default=1, help="Months between cycles")
    ap.add_argument("--evaluation-months", type=int, default=3, help="Months from decision to evaluation")
    ap.add_argument("--out-dir", default="artifacts/learning_validation", help="Output directory")
    ap.add_argument("--use-llm-bundles", action="store_true", help="Use LLM to generate bundles (requires GEMINI_API_KEY)")
    args = ap.parse_args()
    
    # Load data
    print("Loading data...")
    plan = load_plan(args.plan_path)
    kpis_df = pd.read_csv(args.kpis_csv)
    initiatives_df = pd.read_csv(args.initiatives_csv)
    
    # Normalize dates
    kpis_df["date"] = kpis_df["date"].astype(str)
    initiatives_df["date"] = initiatives_df["date"].astype(str)
    
    # Setup output directory
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup learner
    learning_dir = out_dir / "learning"
    learner = get_learner(storage_dir=str(learning_dir))
    
    # Generate cycle dates
    cycles: List[Dict[str, str]] = []
    current_date = args.start_date
    cycle_num = 1
    
    while current_date < args.end_date:
        horizon_end = _shift_date_by_months(current_date, 12)  # 12-month horizon
        evaluation_date = _shift_date_by_months(current_date, args.evaluation_months)
        
        cycles.append({
            "as_of": current_date,
            "horizon_end": horizon_end,
            "evaluation_date": evaluation_date,
            "cycle_num": cycle_num,
        })
        
        current_date = _shift_date_by_months(current_date, args.cycle_months)
        cycle_num += 1
    
    print(f"\nWill run {len(cycles)} cycles:")
    for c in cycles:
        print(f"  Cycle {c['cycle_num']}: as_of={c['as_of']}, eval={c['evaluation_date']}")
    
    # Run cycles
    all_results: List[Dict[str, Any]] = []
    
    for cycle_info in cycles:
        try:
            results = run_cycle(
                plan_path=args.plan_path,
                kpis_csv=args.kpis_csv,
                initiatives_csv=args.initiatives_csv,
                kpis_df=kpis_df,
                initiatives_df=initiatives_df,
                as_of=cycle_info["as_of"],
                horizon_end=cycle_info["horizon_end"],
                evaluation_date=cycle_info["evaluation_date"],
                learner=learner,
                out_dir=out_dir,
                cycle_num=cycle_info["cycle_num"],
                use_llm_bundles=args.use_llm_bundles,
            )
            all_results.append(results)
        except Exception as e:
            print(f"\n[ERROR] Cycle {cycle_info['cycle_num']} failed: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Summary report
    print(f"\n{'='*80}")
    print("LEARNING VALIDATION SUMMARY")
    print(f"{'='*80}")
    
    if not all_results:
        print("No cycles completed successfully.")
        return
    
    # Aggregate metrics
    total_cycles = len(all_results)
    mae_by_cycle = [r["mean_absolute_error"] for r in all_results]
    
    print(f"\nCycles completed: {total_cycles}")
    print(f"\nPrediction accuracy by cycle:")
    for r in all_results:
        print(f"  Cycle {r['cycle_num']:2d} (as_of={r['as_of']}): MAE={r['mean_absolute_error']:.3f}")
    
    if len(mae_by_cycle) >= 2:
        first_half = mae_by_cycle[:len(mae_by_cycle)//2]
        second_half = mae_by_cycle[len(mae_by_cycle)//2:]
        first_avg = sum(first_half) / len(first_half)
        second_avg = sum(second_half) / len(second_half)
        improvement = ((first_avg - second_avg) / first_avg * 100) if first_avg > 0 else 0
        
        print(f"\nLearning trend:")
        print(f"  First half average MAE: {first_avg:.3f}")
        print(f"  Second half average MAE: {second_avg:.3f}")
        print(f"  Improvement: {improvement:+.1f}%")
        
        if improvement > 5:
            print("  Learning loop is improving predictions!")
        elif improvement > -5:
            print("  Learning loop is stable (no significant improvement)")
        else:
            print("  Learning loop may be degrading predictions")
    
    # Final learning metrics
    final_metrics = learner.get_learning_metrics()
    print(f"\nFinal learning metrics:")
    print(f"  Total decisions: {final_metrics.total_decisions}")
    print(f"  Accepted decisions: {final_metrics.accepted_decisions}")
    print(f"  Mean forecast accuracy: {final_metrics.mean_forecast_accuracy:.3f}")
    print(f"  Mean score prediction error: {final_metrics.mean_score_prediction_error:.3f}")
    print(f"  Improvement trend: {final_metrics.improvement_trend}")
    
    # Save summary
    summary_path = out_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({
            "total_cycles": total_cycles,
            "results": all_results,
            "final_metrics": {
                "total_decisions": final_metrics.total_decisions,
                "accepted_decisions": final_metrics.accepted_decisions,
                "mean_forecast_accuracy": final_metrics.mean_forecast_accuracy,
                "mean_score_prediction_error": final_metrics.mean_score_prediction_error,
                "improvement_trend": final_metrics.improvement_trend,
            },
        }, f, indent=2, default=str)
    print(f"\nSummary saved to {summary_path}")


if __name__ == "__main__":
    main()
