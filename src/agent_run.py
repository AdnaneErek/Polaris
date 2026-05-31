# src/agent_run.py
import pandas as pd

from src.load_plan import load_plan
from src.monitor import monitor, snapshot_to_text
from src.agent import generate_recommendations, brief_to_text
from src.whatif import StressEvent


def main():
    plan = load_plan("data/plan.yaml")

    kpis = pd.read_csv("data/simulated_kpis.csv")                 # must contain: date,kpi_id,value
    inits = pd.read_csv("data/simulated_initiatives.csv")         # must contain: date,initiative_id,progress,...
    events = pd.read_csv("data/simulated_events.csv")             # optional

    as_of = "2026-12-01"

    # Optional: keep empty if you want baseline what-if only
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

    # MONITOR
    snap = monitor(plan, kpis[["date", "kpi_id", "value"]], as_of=as_of)
    print(snapshot_to_text(plan, snap))
    print("\n" + "=" * 72 + "\n")

    # AGENT (IMPORTANT: pass kpis_df)
    brief = generate_recommendations(
        plan,
        snap,
        kpis_df=kpis[["date", "kpi_id", "value"]],
        initiatives_df=inits,
        events_df=events,
        stress_events=stress_events,   # set [] if you don’t want stress
    )
    print(brief_to_text(plan, brief))


if __name__ == "__main__":
    main()
