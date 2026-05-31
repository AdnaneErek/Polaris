#!/usr/bin/env python3
"""
CLI tool to record decision outcomes for the learning system.

Usage:
    # List pending decisions
    python -m src.record_outcome --list
    
    # Record outcome for a specific decision
    python -m src.record_outcome --decision-timestamp "2026-01-23T22:16:28.298914" \
        --accepted true \
        --kpi KPI_STP 88.5 \
        --kpi KPI_E2E 2.1 \
        --kpi KPI_INC 25.0 \
        --evaluation-date "2027-06-01"
    
    # Interactive mode
    python -m src.record_outcome --interactive
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

from src.learning import DecisionLearner, get_learner
from src.load_plan import load_plan


def list_pending_decisions(learner: DecisionLearner) -> None:
    """List all decisions that don't have outcomes yet."""
    decisions = learner._load_all_decisions()
    pending = [d for d in decisions if d.get("evaluation_date") is None]
    
    if not pending:
        print("[OK] No pending decisions. All decisions have outcomes recorded.")
        return
    
    print(f"\nPending Decisions ({len(pending)}):")
    print("=" * 80)
    
    for i, decision in enumerate(pending, 1):
        print(f"\n{i}. Decision from {decision['as_of']}")
        print(f"   Timestamp: {decision['timestamp']}")
        print(f"   Recommended: {decision['recommended_bundle_name']} ({decision['recommended_bundle_id']})")
        if decision.get("selected_bundle_id"):
            print(f"   Selected: {decision.get('selected_bundle_name', decision['selected_bundle_id'])} ({decision['selected_bundle_id']})")
        print(f"   Confidence: {decision['confidence']:.2f}")
        print(f"   Guardrails: {'PASSED' if decision['guardrails_passed'] else 'FAILED'}")
        print(f"   Score (stress CVaR10): {decision['recommended_score_stress_cvar10']:.2f}")
        if decision.get("planned_evaluation_date"):
            print(f"   Planned evaluation date: {decision['planned_evaluation_date']}")


def record_outcome_interactive(learner: DecisionLearner, plan_path: str = "data/plan.yaml") -> None:
    """Interactive mode to record outcomes."""
    decisions = learner._load_all_decisions()
    pending = [d for d in decisions if d.get("evaluation_date") is None]
    
    if not pending:
        print("[OK] No pending decisions to record outcomes for.")
        return
    
    print(f"\nRecording Outcome (Interactive Mode)")
    print("=" * 80)
    
    # Show pending decisions
    print("\nPending decisions:")
    for i, decision in enumerate(pending, 1):
        print(f"  {i}. {decision['as_of']}: {decision['recommended_bundle_name']} (timestamp: {decision['timestamp']})")
    
    # Select decision
    try:
        choice = input(f"\nSelect decision (1-{len(pending)}): ").strip()
        idx = int(choice) - 1
        if idx < 0 or idx >= len(pending):
            print("[ERROR] Invalid selection.")
            return
        decision = pending[idx]
    except (ValueError, KeyboardInterrupt):
        print("\n[CANCELLED]")
        return
    
    print(f"\nSelected: {decision['recommended_bundle_name']} from {decision['as_of']}")
    print(f"Timestamp: {decision['timestamp']}")
    
    # Get acceptance
    accepted_input = input("\nWas this recommendation accepted? (yes/no): ").strip().lower()
    accepted = accepted_input in ['yes', 'y', 'true', '1']
    
    # Get evaluation date
    planned = decision.get("planned_evaluation_date")
    if planned:
        eval_date = input(f"Evaluation date (YYYY-MM-DD) [default: planned {planned}]: ").strip()
        if not eval_date:
            eval_date = str(planned)
    else:
        eval_date = input("Evaluation date (YYYY-MM-DD) [default: today]: ").strip()
    if not eval_date:
        from datetime import datetime
        eval_date = datetime.now().date().isoformat()
    
    # Load plan to get KPI info
    try:
        plan = load_plan(plan_path)
        kpi_map = {k.id: k for k in plan.kpis}
    except Exception:
        kpi_map = {}
        print("[WARNING] Could not load plan.yaml, proceeding without KPI names")
    
    # Get actual outcomes
    print("\nEnter actual KPI values at evaluation date:")
    actual_outcomes = {}
    for kpi_id in ["KPI_STP", "KPI_E2E", "KPI_INC"]:
        kpi_name = kpi_map.get(kpi_id, kpi_id)
        try:
            value = input(f"  {kpi_name} ({kpi_id}): ").strip()
            if value:
                actual_outcomes[kpi_id] = float(value)
        except (ValueError, KeyboardInterrupt):
            print(f"  [SKIP] Skipping {kpi_id}")
    
    if not actual_outcomes:
        print("[ERROR] No actual outcomes provided. Cancelled.")
        return
    
    # Get predicted outcomes (optional; if present in decision record, you can accept defaults)
    default_pred = decision.get("predicted_outcomes") or {}
    if default_pred:
        print("\nPredicted KPI values found in decision record. Press Enter to keep defaults, or type a number to override:")
    else:
        print("\nEnter predicted KPI values (or press Enter to skip):")
    predicted_outcomes = {}
    for kpi_id in actual_outcomes.keys():
        kpi_name = kpi_map.get(kpi_id, kpi_id)
        try:
            default_txt = f" [default: {default_pred.get(kpi_id):.4g}]" if kpi_id in default_pred else ""
            value = input(f"  {kpi_name} ({kpi_id}){default_txt}: ").strip()
            if value:
                predicted_outcomes[kpi_id] = float(value)
            elif kpi_id in default_pred:
                predicted_outcomes[kpi_id] = float(default_pred[kpi_id])
        except (ValueError, KeyboardInterrupt):
            pass
    
    # If no predictions provided, we can't compute errors, but still record
    if not predicted_outcomes:
        print("[WARNING] No predicted outcomes provided. Prediction errors will not be computed.")
        # Use placeholder predictions (same as actual) so we can still record
        predicted_outcomes = actual_outcomes.copy()
    
    # Confirm
    print("\nSummary:")
    print(f"  Decision: {decision['recommended_bundle_name']}")
    print(f"  Accepted: {accepted}")
    print(f"  Evaluation date: {eval_date}")
    print(f"  Actual outcomes: {actual_outcomes}")
    print(f"  Predicted outcomes: {predicted_outcomes}")
    
    confirm = input("\nConfirm? (yes/no): ").strip().lower()
    if confirm not in ['yes', 'y']:
        print("[CANCELLED]")
        return
    
    # Record
    try:
        learner.update_decision_outcome(
            decision_timestamp=decision['timestamp'],
            accepted=accepted,
            actual_outcomes=actual_outcomes,
            predicted_outcomes=predicted_outcomes,
            evaluation_date=eval_date,
        )
        print(f"\n[SUCCESS] Outcome recorded successfully!")
        
        # Show updated metrics
        metrics = learner.get_learning_metrics()
        print(f"\nUpdated Learning Metrics:")
        print(f"  Total decisions: {metrics.total_decisions}")
        print(f"  Decisions with outcomes: {sum(1 for d in learner._load_all_decisions() if d.get('evaluation_date'))}")
        if metrics.mean_forecast_accuracy > 0:
            print(f"  Forecast accuracy: {metrics.mean_forecast_accuracy:.1%}")
            print(f"  Improvement trend: {metrics.improvement_trend}")
    except Exception as e:
        print(f"\n[ERROR] Error recording outcome: {e}")
        sys.exit(1)


def record_outcome_cli(
    learner: DecisionLearner,
    decision_timestamp: str,
    accepted: bool,
    kpi_values: List[tuple],  # List of (kpi_id, value) tuples
    evaluation_date: str,
    predicted_kpi_values: Optional[List[tuple]] = None,
) -> None:
    """Record outcome from command-line arguments."""
    # Convert kpi_values to dict
    actual_outcomes = {kpi_id: float(value) for kpi_id, value in kpi_values}
    
    # Convert predicted_kpi_values to dict if provided
    if predicted_kpi_values:
        predicted_outcomes = {kpi_id: float(value) for kpi_id, value in predicted_kpi_values}
    else:
        # If not provided, use actual values (will result in 0 error, but allows recording)
        predicted_outcomes = actual_outcomes.copy()
        print("[WARNING] No predicted outcomes provided. Using actual values as predictions.")
    
    try:
        learner.update_decision_outcome(
            decision_timestamp=decision_timestamp,
            accepted=accepted,
            actual_outcomes=actual_outcomes,
            predicted_outcomes=predicted_outcomes,
            evaluation_date=evaluation_date,
        )
        print(f"[SUCCESS] Outcome recorded successfully for decision {decision_timestamp}")
        
        # Show updated metrics
        metrics = learner.get_learning_metrics()
        print(f"\nLearning Metrics:")
        print(f"  Total decisions: {metrics.total_decisions}")
        print(f"  Decisions with outcomes: {sum(1 for d in learner._load_all_decisions() if d.get('evaluation_date'))}")
        if metrics.mean_forecast_accuracy > 0:
            print(f"  Forecast accuracy: {metrics.mean_forecast_accuracy:.1%}")
            print(f"  Improvement trend: {metrics.improvement_trend}")
    except Exception as e:
        print(f"[ERROR] Error recording outcome: {e}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Record decision outcomes for the learning system",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List pending decisions
  python -m src.record_outcome --list
  
  # Interactive mode
  python -m src.record_outcome --interactive
  
  # Record outcome via CLI
  python -m src.record_outcome \\
      --decision-timestamp "2026-01-23T22:16:28.298914" \\
      --accepted true \\
      --kpi KPI_STP 88.5 \\
      --kpi KPI_E2E 2.1 \\
      --kpi KPI_INC 25.0 \\
      --evaluation-date "2027-06-01"
        """
    )
    
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all pending decisions (without outcomes)",
    )
    
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Interactive mode to record outcomes",
    )
    
    parser.add_argument(
        "--decision-timestamp",
        type=str,
        help="Timestamp of the decision to update (from --list output)",
    )
    
    parser.add_argument(
        "--accepted",
        type=str,
        help="Was recommendation accepted? (true/false or yes/no)",
    )
    
    parser.add_argument(
        "--kpi",
        nargs=2,
        metavar=("KPI_ID", "VALUE"),
        action="append",
        help="Actual KPI value (can be repeated for multiple KPIs)",
    )
    
    parser.add_argument(
        "--predicted-kpi",
        nargs=2,
        metavar=("KPI_ID", "VALUE"),
        action="append",
        help="Predicted KPI value (can be repeated, optional)",
    )
    
    parser.add_argument(
        "--evaluation-date",
        type=str,
        help="Date when outcomes were measured (YYYY-MM-DD)",
    )
    
    parser.add_argument(
        "--storage-dir",
        type=str,
        default="artifacts/learning",
        help="Directory where learning data is stored",
    )
    
    parser.add_argument(
        "--plan",
        type=str,
        default="data/plan.yaml",
        help="Path to plan.yaml (for KPI names in interactive mode)",
    )
    
    args = parser.parse_args()
    
    # Get learner
    learner = get_learner(storage_dir=args.storage_dir)
    
    # Handle different modes
    if args.list:
        list_pending_decisions(learner)
    elif args.interactive:
        record_outcome_interactive(learner, plan_path=args.plan)
    elif args.decision_timestamp:
        # CLI mode
        if not args.accepted:
            print("[ERROR] --accepted is required")
            sys.exit(1)
        if not args.kpi:
            print("[ERROR] At least one --kpi is required")
            sys.exit(1)
        if not args.evaluation_date:
            print("[ERROR] --evaluation-date is required")
            sys.exit(1)
        
        # Parse accepted
        accepted_str = args.accepted.lower()
        accepted = accepted_str in ['true', 'yes', 'y', '1']
        
        # Record
        record_outcome_cli(
            learner=learner,
            decision_timestamp=args.decision_timestamp,
            accepted=accepted,
            kpi_values=args.kpi,
            evaluation_date=args.evaluation_date,
            predicted_kpi_values=args.predicted_kpi if args.predicted_kpi else None,
        )
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
