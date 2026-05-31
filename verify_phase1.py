"""
Quick verification script for Phase 1 completion.
Tests that the plan-driven normalization works correctly.
"""
from src.load_plan import load_plan
from src.portfolio import _normalize_delta, _find_kpi

def test_plan_driven_normalization():
    """Verify that _normalize_delta uses plan-defined direction."""
    plan = load_plan("data/plan.yaml")
    
    # Test KPI_STP (direction="up")
    # Positive delta should give positive utility
    stp_delta = 5.0  # +5 percentage points
    stp_utility = _normalize_delta(plan, "KPI_STP", stp_delta)
    assert stp_utility > 0, f"STP positive delta should give positive utility, got {stp_utility}"
    print(f"[OK] KPI_STP: delta={stp_delta}, utility={stp_utility:.2f} (direction=up)")
    
    # Test KPI_E2E (direction="down")
    # Negative delta should give positive utility
    e2e_delta = -0.2  # -0.2 hours (improvement)
    e2e_utility = _normalize_delta(plan, "KPI_E2E", e2e_delta)
    assert e2e_utility > 0, f"E2E negative delta should give positive utility, got {e2e_utility}"
    print(f"[OK] KPI_E2E: delta={e2e_delta}, utility={e2e_utility:.2f} (direction=down)")
    
    # Test KPI_INC (direction="down")
    # Negative delta should give positive utility
    inc_delta = -3.0  # -3 incidents (improvement)
    inc_utility = _normalize_delta(plan, "KPI_INC", inc_delta)
    assert inc_utility > 0, f"INC negative delta should give positive utility, got {inc_utility}"
    print(f"[OK] KPI_INC: delta={inc_delta}, utility={inc_utility:.2f} (direction=down)")
    
    # Test alias handling (short_name)
    stp_utility_alias = _normalize_delta(plan, "STP_rate", stp_delta)
    assert abs(stp_utility - stp_utility_alias) < 1e-6, (
        f"Alias should give same result: {stp_utility} != {stp_utility_alias}"
    )
    print(f"[OK] Alias handling: STP_rate gives same result as KPI_STP")
    
    # Test unknown KPI (fallback)
    unknown_utility = _normalize_delta(plan, "UNKNOWN_KPI", 10.0)
    assert unknown_utility == 10.0, f"Unknown KPI should return raw delta, got {unknown_utility}"
    print(f"[OK] Unknown KPI fallback: returns raw delta")
    
    print("\n[OK] All plan-driven normalization tests passed!")

if __name__ == "__main__":
    test_plan_driven_normalization()
