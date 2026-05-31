# src/whatif_run.py
import pandas as pd

from src.load_plan import load_plan
from src.actions import SteeringAction
from src.whatif import (
    StressEvent,
    compare_actions,
    compare_actions_at_dates,
    project_kpis,
)


def main():
    plan = load_plan("data/plan.yaml")

    kpis = pd.read_csv("data/simulated_kpis.csv")
    inits = pd.read_csv("data/simulated_initiatives.csv")

    as_of = "2026-12-01"
    horizon_end = "2027-12-01"

    # Stress applied DURING projection (exogenous shocks)
    stress_events = [
        StressEvent(
            id="S1",
            type="initiative_progress_drop",
            target_id="INIT_DQ1",
            start="2027-01-01",
            end="2027-06-01",
            magnitude=0.20,
            description="Synthetic: DQ slippage reduces effective progress for 6 months.",
        ),
        StressEvent(
            id="S2",
            type="kpi_shock",
            target_id="KPI_STP",
            start="2027-02-01",
            end="2027-04-01",
            magnitude=-4.0,
            description="Synthetic: STP KPI correction after definition/lineage fix.",
        ),
        StressEvent(
            id="S3",
            type="kpi_shock",
            target_id="KPI_INC",
            start="2027-03-01",
            end="2027-05-01",
            magnitude=+8.0,
            description="Synthetic: incident spike (ops instability).",
        ),
    ]

    actions = [
        SteeringAction(
            id="A1",
            type="accelerate_initiative",
            target_initiative="INIT_DQ1",
            parameters={"months": 2},
            description="Accelerate Data Quality initiative by ~2 months (extra capacity / focus).",
        ),
        SteeringAction(
            id="A2",
            type="split_scope",
            target_initiative="INIT_A1",
            parameters={"reduction": 0.6},
            description="Split Automation scope to deliver parallelizable parts despite DQ dependency.",
        ),
    ]

    print("=== Action impact at horizon (STRESSED) ===")
    comp = compare_actions(plan, kpis, inits, as_of, horizon_end, actions, stress_events=stress_events)
    print(comp.to_string(index=False))

    print("\n=== Action impact at checkpoint dates (STRESSED) ===")
    dates_to_check = ["2027-06-01", "2027-12-01", "2028-12-01"]
    comp2 = compare_actions_at_dates(plan, kpis, inits, as_of, dates_to_check, actions, stress_events=stress_events)
    print(comp2.to_string(index=False))

    print("\n=== Trajectory sample (stressed, with actions) ===")
    traj = project_kpis(plan, kpis, inits, as_of, horizon_end, actions=actions, stress_events=stress_events)
    print(traj.head(12).to_string(index=False))


if __name__ == "__main__":
    main()
