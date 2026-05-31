# src/portfolio_run.py
import pandas as pd

from src.load_plan import load_plan
from src.portfolio import (
    ActionBundle,
    evaluate_bundles_mc,
    format_option_table_mc,
)
from src.actions import SteeringAction
from src.agent import _augment_kpi_ids  # reuse helpers
from src.whatif import StressEvent, WhatIfStochasticConfig, WhatIfConfig
from src.kpi_integrity import compute_kpi_integrity  # optional but recommended


def main():
    plan = load_plan("data/plan.yaml")
    as_of = "2026-12-01"

    kpis = pd.read_csv("data/simulated_kpis.csv")
    inits = pd.read_csv("data/simulated_initiatives.csv")

    # Ensure KPI ids are robust (plan uses KPI_* ids; some csvs may use short ids)
    kpis = _augment_kpi_ids(kpis[["date", "kpi_id", "value"]])

    # --- Dates to score (main_date will be the earliest: 2027-06-01) ---
    dates_to_check = ["2027-06-01", "2027-12-01", "2028-12-01"]

    # --- Define stress scenario (example) ---
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

    # --- Define action bundles ---
    accel_dq1 = SteeringAction(
    id="A_CAP_DQ1",
    type="add_capacity",
    target_initiative="INIT_DQ1",
    parameters={"capacity_gain": 0.07},  # try 0.05–0.10
    description="Add capacity to INIT_DQ1 (more throughput after as_of).",
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

    # --- OPTIONAL but strongly recommended: use integrity to inflate noise for suspicious KPIs ---
    # compute_kpi_integrity returns a mapping kpi_id -> object with .score in [0,1] (your integrity_run printed score=0.00 etc.)
    integrity = compute_kpi_integrity(plan, kpis[["date", "kpi_id", "value"]], as_of=as_of)
    integrity_scores = {str(k): float(v.score) for k, v in integrity.items()}  # 0..1

    # Monte Carlo config (tune n_samples for speed vs stability)
    mc_cfg = WhatIfStochasticConfig(n_samples=600, seed=7)

    # Deterministic what-if parameters (optional; defaults are fine)
    whatif_cfg = WhatIfConfig()

    results = evaluate_bundles_mc(
        plan=plan,
        kpis_df=kpis,
        initiatives_df=inits,
        as_of=as_of,
        dates_to_check=dates_to_check,
        bundles=bundles,
        stress_events=stress_events,
        whatif_cfg=whatif_cfg,       # ✅ correct kwarg name
        mc_cfg=mc_cfg,
        integrity_scores=integrity_scores,
        cvar_alpha=0.10,
    )

    print(format_option_table_mc(results))


if __name__ == "__main__":
    main()
