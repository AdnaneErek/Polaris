import pandas as pd

from src.load_plan import load_plan
from src.monitor import monitor, snapshot_to_text


def main():
    plan = load_plan("data/plan.yaml")
    df = pd.read_csv("data/simulated_kpis.csv")

    snap = monitor(
        plan,
        df[["date", "kpi_id", "value"]],
        as_of="2026-12-01"
    )

    print(snapshot_to_text(plan, snap))


if __name__ == "__main__":
    main()
