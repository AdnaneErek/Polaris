from src.load_plan import load_plan

if __name__ == "__main__":
    plan = load_plan("data/plan.yaml")
    print("✅ Loaded:", plan.meta.org)
    print("KPIs:", len(plan.kpis))
    print("Objectives:", len(plan.objectives))
    print("Initiatives:", len(plan.initiatives))
    print("Capabilities:", len(plan.capabilities))
    print("Portfolio constraints:", len(plan.portfolio.constraints))
    print("Agent autonomy:", plan.agent_policy.autonomy_level)
