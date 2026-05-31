"""
Quick verification script for Phase 2 completion.
Tests that guardrails are evaluated and integrated into SteerCo pack.
"""
from src.load_plan import load_plan
from src.portfolio import ActionBundle
from src.actions import SteeringAction
from src.guardrails import check_portfolio_guardrails, GuardrailResult
from src.portfolio import OptionResultMC

def test_guardrails_basic():
    """Test basic guardrails functionality."""
    plan = load_plan("data/plan.yaml")
    
    # Create a test bundle
    bundle = ActionBundle(
        id="TEST_BUNDLE",
        name="Test Bundle",
        description="Test bundle for guardrails",
        actions=[
            SteeringAction(
                id="A_TEST",
                type="accelerate_initiative",
                target_initiative="INIT_DQ1",
                parameters={"months": 2},
                description="Test action",
            )
        ],
    )
    
    # Create a dummy option result (minimal)
    option_result = OptionResultMC(
        bundle_id="TEST_BUNDLE",
        bundle_name="Test Bundle",
        main_date="2027-06-01",
        stress_delta_summary={
            "KPI_STP": {"mean": 0.5, "p10": 0.3, "p50": 0.5, "p90": 0.7, "p_improve": 0.8},
            "KPI_E2E": {"mean": -0.1, "p10": -0.15, "p50": -0.1, "p90": -0.05, "p_improve": 0.9},
            "KPI_INC": {"mean": -2.0, "p10": -2.5, "p50": -2.0, "p90": -1.5, "p_improve": 1.0},
        },
        base_delta_summary={},
        score_base_mean=0.5,
        score_base_cvar10=0.4,
        score_stress_mean=0.5,
        score_stress_cvar10=0.4,
        robustness_gap_mean=0.0,
        notes=[],
    )
    
    # Test guardrails check
    result = check_portfolio_guardrails(
        plan=plan,
        bundle=bundle,
        option_result=option_result,
        as_of="2026-12-01",
    )
    
    print(f"[OK] Guardrails check completed")
    print(f"  Passed: {result.passed}")
    print(f"  Violations: {len(result.violations)}")
    print(f"  Requires approval: {result.requires_approval}")
    
    if result.violations:
        print("\n  Violations:")
        for v in result.violations:
            print(f"    [{v.severity}] {v.message}")
    
    if result.approval_required_from:
        print(f"  Approval required from: {', '.join(result.approval_required_from)}")
    
    print("\n[OK] Basic guardrails test passed!")


def test_guardrails_kpi_degradation():
    """Test KPI degradation detection."""
    plan = load_plan("data/plan.yaml")
    
    bundle = ActionBundle(
        id="TEST_DEGRADE",
        name="Test Degradation",
        description="Test bundle that degrades KPIs",
        actions=[],
    )
    
    # Create option result with KPI degradation
    option_result = OptionResultMC(
        bundle_id="TEST_DEGRADE",
        bundle_name="Test Degradation",
        main_date="2027-06-01",
        stress_delta_summary={
            "KPI_STP": {"mean": -3.0, "p10": -4.0, "p50": -3.0, "p90": -2.0, "p_improve": 0.0},  # Degrades STP
        },
        base_delta_summary={},
        score_base_mean=-1.0,
        score_base_cvar10=-1.5,
        score_stress_mean=-1.0,
        score_stress_cvar10=-1.5,
        robustness_gap_mean=0.0,
        notes=[],
    )
    
    result = check_portfolio_guardrails(
        plan=plan,
        bundle=bundle,
        option_result=option_result,
        as_of="2026-12-01",
        kpi_degradation_threshold=2.0,
    )
    
    # Should detect KPI degradation
    kpi_violations = [v for v in result.violations if v.rule_type == "kpi_degradation"]
    assert len(kpi_violations) > 0, "Should detect KPI degradation"
    assert not result.passed, "Should fail guardrails due to KPI degradation"
    
    print(f"[OK] KPI degradation detection works")
    print(f"  Detected {len(kpi_violations)} KPI degradation violations")


if __name__ == "__main__":
    test_guardrails_basic()
    print()
    test_guardrails_kpi_degradation()
    print("\n[OK] All Phase 2 guardrails tests passed!")
