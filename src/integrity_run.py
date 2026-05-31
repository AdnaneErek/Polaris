# src/integrity_run.py
import pandas as pd
from src.load_plan import load_plan
from src.kpi_integrity import compute_kpi_integrity, format_integrity_report
from src.forecast import forecast_kpis

def main():
    plan = load_plan("data/plan.yaml")
    kpis = pd.read_csv("data/simulated_kpis.csv")
    inits = pd.read_csv("data/simulated_initiatives.csv")  # <-- add

    as_of = "2026-12-01"
    horizon_end = "2027-12-01"

    integrity = compute_kpi_integrity(plan, kpis[["date","kpi_id","value"]], as_of=as_of)
    print(format_integrity_report(integrity, threshold=0.6))

    print("\nFORECAST (sample)")
    fc = forecast_kpis(
        plan,
        kpis[["date","kpi_id","value"]],
        as_of=as_of,
        horizon_end=horizon_end,
        initiatives_df=inits[["date","initiative_id","progress","delay_months"]],
    )
    print(fc.head(12).to_string(index=False))

if __name__ == "__main__":
    main()
