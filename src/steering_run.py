# src/steering_run.py
from __future__ import annotations

from src.kpi_integrity import compute_kpi_integrity, integrity_flags_for_agent, format_integrity_report
from src.forecast import forecast_kpis


import pandas as pd

from src.load_plan import load_plan
from src.whatif import StressEvent
from src.portfolio import ActionBundle, evaluate_bundles, format_option_table
from src.actions import SteeringAction

from src.steering_agent import (
    choose_option_style_a,
    build_decision_brief,
    write_audit_record,
)


def _try_monitor(plan, kpis_df, as_of: str) -> str:
    """
    Uses src.monitor if available, otherwise prints a lightweight fallback summary.
    """
    try:
        from src.monitor import monitor, snapshot_to_text

        snap = monitor(plan, kpis_df[["date", "kpi_id", "value"]], as_of=as_of)
        return snapshot_to_text(plan, snap)
    except Exception as e:
        # Fallback
        df = kpis_df.copy()
        df = df[df["date"] <= as_of].sort_values("date")
        last = df.groupby("kpi_id").tail(1)
        lines = [f"MONITOR (fallback) — as of {as_of}", "-" * 72]
        for _, r in last.iterrows():
            lines.append(f"{r['kpi_id']}: {float(r['value']):.4f}")
        lines.append(f"(monitor module unavailable or failed: {type(e).__name__}: {e})")
        return "\n".join(lines)


def _integrity_flags_from_monitor_text(monitor_text: str) -> list[str]:
    """
    Cheap but effective: treat “anomaly”, “definition”, “lineage”, “denominator”
    as integrity flags requiring conservative steering.
    """
    t = monitor_text.lower()
    flags = []
    if any(w in t for w in ["anomal", "definition", "lineage", "denominator", "measurement", "clamp", "bias"]):
        flags.append("KPI integrity / measurement risk mentioned in monitoring narrative.")
    return flags


def main():
    plan = load_plan("data/plan.yaml")
    as_of = "2026-12-01"

    kpis = pd.read_csv("data/simulated_kpis.csv")
    inits = pd.read_csv("data/simulated_initiatives.csv")

    # Portfolio settings
    dates_to_check = ["2027-06-01", "2027-12-01", "2028-12-01"]
    main_date = "2027-06-01"

    # Stress scenario (same idea as your portfolio_run)
    stress_events = [
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

    # Action primitives
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

    bundles = [
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

    # ---- 1) Monitoring evidence
    monitor_text = _try_monitor(plan, kpis, as_of=as_of)

    integrity = compute_kpi_integrity(plan, kpis[["date", "kpi_id", "value"]], as_of=as_of)
    integrity_flags = integrity_flags_for_agent(integrity, threshold=0.6)
    integrity_text = format_integrity_report(integrity, threshold=0.6)

# Optional: forecast evidence (sample)
    fc = forecast_kpis(plan, kpis[["date", "kpi_id", "value"]], as_of=as_of, horizon_end="2027-12-01")
    forecast_text = "FORECAST (sample)\n" + fc.head(12).to_string(index=False)

    print("\n" + integrity_text.strip())
    print("\n" + forecast_text.strip())



    # ---- 2) Portfolio evaluation evidence
    results = evaluate_bundles(
        plan=plan,
        kpis_df=kpis,
        initiatives_df=inits,
        as_of=as_of,
        dates_to_check=dates_to_check,
        bundles=bundles,
        stress_events=stress_events,
    )

    portfolio_text = format_option_table(results, main_date)

    # ---- 3) Agentic decision (Style A)
    chosen, rejected, guardrails = choose_option_style_a(
        plan=plan,
        as_of=as_of,
        options=results,
        integrity_flags=integrity_flags,
    )

    decision = build_decision_brief(
        plan=plan,
        as_of=as_of,
        monitor_text=monitor_text,
        portfolio_text=portfolio_text,
        chosen=chosen,
        rejected=rejected,
        guardrails_triggered=guardrails,
        integrity_flags=integrity_flags,
    )

    # ---- 4) Print SteerCo brief
    print("\n" + "=" * 88)
    print(f"STEERING AGENT — Decision Brief (Style A) — as of {decision.as_of}")
    print("=" * 88)
    print(f"Headline: {decision.headline}")
    print(f"Recommendation: {decision.recommended_option_id} — {decision.recommended_option_name}")
    print("\nRationale:")
    for r in decision.decision_rationale:
        print(f" - {r}")

    if decision.guardrails_triggered:
        print("\nGuardrails / flags:")
        for g in decision.guardrails_triggered:
            print(f" - {g}")

    print("\nApprovals required:")
    for a in decision.approvals_required:
        print(f" - {a}")

    print("\nNext actions:")
    for n in decision.next_actions:
        print(f" - {n}")

    print("\nTop rejected options (why not):")
    for r in decision.rejected_options[:3]:
        print(f" - {r['id']} {r['name']}: {r['why_not']} (stress={r['score_stress']:.2f}, gap={r['gap']:+.2f})")

    print("\n--- Evidence (monitor + portfolio) ---")
    print(monitor_text.strip())
    print("\n" + portfolio_text.strip())

    # ---- 5) Audit log (JSONL)
    audit_path = "out/audit_steering.jsonl"
    written = write_audit_record(
        out_path=audit_path,
        plan=plan,
        as_of=as_of,
        kpis_df=kpis,
        initiatives_df=inits,
        monitor_text=monitor_text,
        portfolio_text=portfolio_text,
        decision=decision,
        extra={"stress_events": [e.__dict__ for e in stress_events],
               "integrity": {k: {"score": v.score, "flags": v.flags, "diagnostics": v.diagnostics} for k, v in integrity.items()},},
    )
    print(f"\n✅ Audit appended to {written}")


if __name__ == "__main__":
    main()
