# src/steerco_run.py
from __future__ import annotations

import argparse
from pathlib import Path

from src.actions import SteeringAction
from src.portfolio import ActionBundle
from src.whatif import StressEvent, WhatIfStochasticConfig, WhatIfConfig
from src.forecast import ForecastConfig
from src.steerco_pack import build_steerco_pack


def _default_bundles() -> list[ActionBundle]:
    # --- Define action bundles (same as your portfolio_run, MC version) ---
    accel_dq1 = SteeringAction(
        id="A_ACCEL_DQ1_2M",
        type="accelerate_initiative",
        target_initiative="INIT_DQ1",
        parameters={"months": 2},
        description="Accelerate INIT_DQ1 by ~2 months.",
    )
    accel_a1 = SteeringAction(
        id="A_ACCEL_A1_2M",
        type="accelerate_initiative",
        target_initiative="INIT_A1",
        parameters={"months": 2},
        description="Accelerate INIT_A1 by ~2 months.",
    )
    accel_res1 = SteeringAction(
        id="A_ACCEL_RES1_2M",
        type="accelerate_initiative",
        target_initiative="INIT_RES1",
        parameters={"months": 2},
        description="Accelerate INIT_RES1 by ~2 months.",
    )
    split_a1 = SteeringAction(
        id="A_SPLIT_A1",
        type="split_scope",
        target_initiative="INIT_A1",
        parameters={"reduction": 0.6},
        description="Split INIT_A1 scope (reduce blocking, deliver stable flows first).",
    )

    return [
        ActionBundle(
            id="OPT_A",
            name="Option A — No-regret (DQ + Resilience)",
            description="Stabilize data & reduce incident tail-risk while KPI governance runs.",
            actions=[accel_dq1, accel_res1],
        ),
        ActionBundle(
            id="OPT_B",
            name="Option B — Growth (Accelerate Automation)",
            description="Maximize STP/E2E impact earlier (higher delivery/control risk).",
            actions=[accel_a1, accel_dq1],
        ),
        ActionBundle(
            id="OPT_C",
            name="Option C — Risk-first (Resilience only)",
            description="Focus on stability; minimal change to automation delivery.",
            actions=[accel_res1],
        ),
        ActionBundle(
            id="OPT_D",
            name="Option D — De-risk dependencies (Split + Resilience)",
            description="Reduce dependency bottleneck + stabilize production.",
            actions=[split_a1, accel_res1],
        ),
    ]


def _default_stress() -> list[StressEvent]:
    return [
        StressEvent(
            id="S_INC_SPIKE",
            type="kpi_shock",
            target_id="KPI_INC",
            start="2027-03-01",
            end="2027-05-01",
            magnitude=+8.0,
            description="Synthetic stress: incident spike due to platform instability.",
        ),
        StressEvent(
            id="S_STP_CORRECTION",
            type="kpi_shock",
            target_id="KPI_STP",
            start="2027-02-01",
            end="2027-04-01",
            magnitude=-4.0,
            description="Synthetic stress: STP correction after KPI lineage/definition fix.",
        ),
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", default="data/plan.yaml")
    ap.add_argument("--kpis", default="data/simulated_kpis.csv")
    ap.add_argument("--inits", default="data/simulated_initiatives.csv")
    ap.add_argument("--as_of", default="2026-12-01")
    ap.add_argument("--horizon_end", default="2027-12-01")
    ap.add_argument("--out_dir", default=None, help="e.g. artifacts/steerco/2026-12-01")
    ap.add_argument("--n_samples", type=int, default=600)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--use-manual-bundles", action="store_true", 
                    help="Use manual action bundles instead of LLM-generated ones (default: LLM)")
    args = ap.parse_args()

    out_dir = args.out_dir
    if out_dir is None:
        out_dir = str(Path("artifacts") / "steerco" / str(args.as_of))

    # Load plan and data
    from src.load_plan import load_plan
    import pandas as pd
    
    plan = load_plan(args.plan)
    kpis = pd.read_csv(args.kpis)
    inits = pd.read_csv(args.inits)
    
    # Generate bundles (LLM by default, manual if flag set)
    bundles = None
    bundle_generation = {
        "mode": "manual" if args.use_manual_bundles else "llm_attempt",
        "fallback_reason": None,
    }
    if not args.use_manual_bundles:
        print("[INFO] Generating action bundles using LLM...")
        try:
            from src.monitor import monitor
            from src.steering import get_initiative_health
            from src.kpi_integrity import compute_kpi_integrity
            from src.bundle_generator import generate_action_bundles_with_llm
            
            # Build analysis context
            monitor_snapshot = monitor(plan, kpis[["date", "kpi_id", "value"]], args.as_of)
            initiative_health = get_initiative_health(plan, inits, args.as_of)
            integrity = compute_kpi_integrity(plan, kpis[["date", "kpi_id", "value"]], args.as_of)
            integrity_scores = {k: float(v.score) for k, v in integrity.items()}
            
            # Get ML anomalies if available
            ml_anomalies = None
            try:
                from src.monitor_ml import detect_all_kpi_anomalies_ml
                ml_anomalies_dict = detect_all_kpi_anomalies_ml(plan, kpis[["date", "kpi_id", "value"]], args.as_of)
                ml_anomalies = {
                    kpi_id: {
                        "is_anomaly": result.is_anomaly,
                        "anomaly_score": result.anomaly_score,
                        "confidence": result.confidence,
                        "explanation": result.explanation,
                        "method": result.method,
                    }
                    for kpi_id, result in ml_anomalies_dict.items()
                }
            except (ImportError, Exception):
                ml_anomalies = None
            
            # Generate bundles with LLM
            bundles = generate_action_bundles_with_llm(
                plan=plan,
                monitor_snapshot=monitor_snapshot,
                initiative_health=initiative_health,
                ml_anomalies=ml_anomalies,
                integrity_scores=integrity_scores,
                kpi_history=kpis,
                initiative_history=inits,
                as_of=args.as_of,
            )
            
            print(f"[OK] Generated {len(bundles)} action bundles using LLM")
            for bundle in bundles:
                print(f"  - {bundle.id}: {bundle.name} ({len(bundle.actions)} actions)")
            bundle_generation = {
                "mode": "llm",
                "fallback_reason": None,
                "n_bundles": len(bundles),
            }
            
            # Generate situation summaries
            try:
                from src.situation_summary import generate_situation_summaries
                situation_summaries = generate_situation_summaries(
                    plan=plan,
                    monitor_snapshot=monitor_snapshot,
                    initiative_health=initiative_health,
                    ml_anomalies=ml_anomalies,
                    integrity_scores=integrity_scores,
                )
                print(f"[OK] Generated {len(situation_summaries)} situation summaries")
            except Exception as e:
                print(f"[WARNING] Situation summary generation failed: {e}")
                situation_summaries = None
        
        except Exception as e:
            print(f"[WARNING] LLM bundle generation failed: {e}")
            print("[INFO] Falling back to manual bundles...")
            bundles = None
            situation_summaries = None
            bundle_generation = {
                "mode": "manual_fallback",
                "fallback_reason": str(e),
            }
    
    # Fallback to manual bundles (if LLM failed or --use-manual-bundles flag)
    if bundles is None:
        bundles = _default_bundles()
        print(f"[INFO] Using {len(bundles)} manual action bundles")
        situation_summaries = None
        if args.use_manual_bundles:
            bundle_generation = {
                "mode": "manual",
                "fallback_reason": "manual flag enabled (--use-manual-bundles)",
                "n_bundles": len(bundles),
            }
        else:
            bundle_generation["n_bundles"] = len(bundles)
    
    stress_events = _default_stress()

    pack = build_steerco_pack(
        plan_path=args.plan,
        kpis_csv=args.kpis,
        initiatives_csv=args.inits,
        as_of=args.as_of,
        horizon_end=args.horizon_end,
        bundles=bundles,
        stress_events=stress_events,
        mc_cfg=WhatIfStochasticConfig(n_samples=args.n_samples, seed=args.seed),
        whatif_cfg=WhatIfConfig(),
        forecast_cfg=ForecastConfig(),
        out_dir=out_dir,
        situation_summaries=situation_summaries if 'situation_summaries' in locals() else None,
        bundle_generation=bundle_generation,
    )

    print(f"[OK] Wrote SteerCo pack to: {out_dir}")
    print("")
    print(pack.options_table_mc)
    
    # Offer chat interface
    pack_json_path = Path(out_dir) / "steerco_pack.json"
    if pack_json_path.exists():
        print("\n" + "=" * 80)
        print("Chat Interface Available")
        print("=" * 80)
        print(f"Ask questions about this pack using:")
        print(f"  python -m src.chat_cli --pack {pack_json_path}")
        print("\nExample questions:")
        print("  - 'Why was Option C recommended?'")
        print("  - 'What are the main risks for the recommended option?'")
        print("  - 'How accurate have our forecasts been?'")
        print("  - 'What anomalies were detected?'")
        print("=" * 80)
    print("")
    # ML Anomaly Detection summary
    if pack.ml_anomalies:
        print("\n=== ML Anomaly Detection ===")
        from src.load_plan import load_plan
        plan = load_plan(args.plan)
        
        anomalies_found = [kpi_id for kpi_id, data in pack.ml_anomalies.items() if data["is_anomaly"]]
        if anomalies_found:
            for kpi_id in anomalies_found:
                kpi = next((k for k in plan.kpis if k.id == kpi_id), None)
                kpi_name = kpi.short_name if kpi else kpi_id
                anomaly_data = pack.ml_anomalies[kpi_id]
                print(f"[ANOMALY] {kpi_name}: {anomaly_data['explanation']}")
                print(f"   Confidence: {anomaly_data['confidence']:.2f}, Method: {anomaly_data['method']}")
        else:
            print("[OK] No ML anomalies detected - all KPIs show normal patterns")
    
    # Learning from Outcomes summary
    if pack.learning_metrics:
        print("\n=== Learning from Outcomes ===")
        metrics = pack.learning_metrics
        print(f"Total decisions: {metrics.get('total_decisions', 0)}")
        if metrics.get('total_decisions', 0) > 0:
            print(f"Accepted: {metrics.get('accepted_decisions', 0)}")
            if metrics.get('mean_forecast_accuracy', 0) > 0:
                print(f"Forecast accuracy: {metrics.get('mean_forecast_accuracy', 0):.1%}")
                print(f"Improvement trend: {metrics.get('improvement_trend', 'N/A')}")
        else:
            print("No decisions recorded yet. Learning will begin after first decision.")
    
    print("\n=== Attribution Summary ===")
    from src.attribution_summary import format_attribution_summary_console_simple
    from src.load_plan import load_plan
    
    plan = load_plan(args.plan)
    
    for bundle_id, explain in pack.options_explain.items():
        bundle = next((b for b in bundles if b.id == bundle_id), None)
        bundle_name = bundle.name if bundle else bundle_id
        
        # Get option result summary from pack
        option_summary = next((o for o in pack.options_mc if o["bundle_id"] == bundle_id), None)
        if not option_summary:
            continue
        
        summary = format_attribution_summary_console_simple(
            plan=plan,
            bundle_id=bundle_id,
            bundle_name=bundle_name,
            attribution_data=explain,
            option_summary=option_summary,
        )
        try:
            print(summary)
        except UnicodeEncodeError:
            print(summary.encode("ascii", "ignore").decode("ascii"))
        print("")
    
    print("Guardrails:")
    print(pack.guardrails_report)
    print("")
    print("Recommendation:")
    print(f"- {pack.recommendation.chosen_bundle_name} ({pack.recommendation.chosen_bundle_id})")
    print(f"- confidence={pack.recommendation.confidence:.2f}")
    print(f"- guardrails: {'PASSED' if pack.recommendation.guardrails_passed else 'FAILED'}")
    for w in pack.recommendation.why:
        line = f"- {w}"
        try:
            print(line)
        except UnicodeEncodeError:
            print(line.encode("ascii", "ignore").decode("ascii"))


if __name__ == "__main__":
    main()
