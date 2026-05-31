"""
Test reproducibility of Monte Carlo simulations.

Phase 1 requirement: Ensure that rerunning with the same seed produces identical results.
"""
import pytest
import pandas as pd
import numpy as np

from src.load_plan import load_plan
from src.portfolio import (
    ActionBundle,
    evaluate_bundles_mc,
)
from src.actions import SteeringAction
from src.whatif import (
    StressEvent,
    WhatIfStochasticConfig,
    WhatIfConfig,
)
from src.kpi_integrity import compute_kpi_integrity


def test_mc_reproducibility():
    """
    Test that MC results are identical when rerun with the same seed.
    
    This ensures:
    1. One seed controls everything
    2. Base/action runs share the same random stream
    3. Results are deterministic
    """
    plan = load_plan("data/plan.yaml")
    kpis = pd.read_csv("data/simulated_kpis.csv")
    inits = pd.read_csv("data/simulated_initiatives.csv")
    
    as_of = "2026-12-01"
    dates_to_check = ["2027-06-01"]
    
    # Define test bundles
    bundles = [
        ActionBundle(
            id="TEST_BUNDLE_1",
            name="Test Bundle 1",
            description="Test bundle for reproducibility",
            actions=[
                SteeringAction(
                    id="A_TEST_1",
                    type="accelerate_initiative",
                    target_initiative="INIT_DQ1",
                    parameters={"months": 2},
                    description="Test action",
                )
            ],
        )
    ]
    
    stress_events = [
        StressEvent(
            id="S_TEST",
            type="kpi_shock",
            target_id="KPI_INC",
            start="2027-03-01",
            end="2027-05-01",
            magnitude=+5.0,
            description="Test stress event",
        )
    ]
    
    # Compute integrity scores
    integrity = compute_kpi_integrity(plan, kpis[["date", "kpi_id", "value"]], as_of=as_of)
    integrity_scores = {str(k): float(v.score) for k, v in integrity.items()}
    
    # Fixed seed for reproducibility
    seed = 42
    n_samples = 200  # Smaller for faster test
    
    mc_cfg = WhatIfStochasticConfig(n_samples=n_samples, seed=seed)
    whatif_cfg = WhatIfConfig()
    
    # Run 1
    results1 = evaluate_bundles_mc(
        plan=plan,
        kpis_df=kpis,
        initiatives_df=inits,
        as_of=as_of,
        dates_to_check=dates_to_check,
        bundles=bundles,
        stress_events=stress_events,
        whatif_cfg=whatif_cfg,
        mc_cfg=mc_cfg,
        integrity_scores=integrity_scores,
        cvar_alpha=0.10,
    )
    
    # Run 2 (should be identical)
    results2 = evaluate_bundles_mc(
        plan=plan,
        kpis_df=kpis,
        initiatives_df=inits,
        as_of=as_of,
        dates_to_check=dates_to_check,
        bundles=bundles,
        stress_events=stress_events,
        whatif_cfg=whatif_cfg,
        mc_cfg=mc_cfg,
        integrity_scores=integrity_scores,
        cvar_alpha=0.10,
    )
    
    # Assert we got results
    assert len(results1) > 0, "First run produced no results"
    assert len(results2) > 0, "Second run produced no results"
    assert len(results1) == len(results2), "Different number of results between runs"
    
    # Compare all results
    for r1, r2 in zip(results1, results2):
        # Bundle IDs should match
        assert r1.bundle_id == r2.bundle_id, f"Bundle ID mismatch: {r1.bundle_id} != {r2.bundle_id}"
        
        # Scores should be identical (exact match for reproducibility)
        assert r1.score_stress_mean == r2.score_stress_mean, (
            f"Stress mean mismatch for {r1.bundle_id}: "
            f"{r1.score_stress_mean} != {r2.score_stress_mean}"
        )
        assert r1.score_stress_cvar10 == r2.score_stress_cvar10, (
            f"Stress CVaR10 mismatch for {r1.bundle_id}: "
            f"{r1.score_stress_cvar10} != {r2.score_stress_cvar10}"
        )
        assert r1.score_base_mean == r2.score_base_mean, (
            f"Base mean mismatch for {r1.bundle_id}: "
            f"{r1.score_base_mean} != {r2.score_base_mean}"
        )
        assert r1.score_base_cvar10 == r2.score_base_cvar10, (
            f"Base CVaR10 mismatch for {r1.bundle_id}: "
            f"{r1.score_base_cvar10} != {r2.score_base_cvar10}"
        )
        assert r1.robustness_gap_mean == r2.robustness_gap_mean, (
            f"Robustness gap mismatch for {r1.bundle_id}: "
            f"{r1.robustness_gap_mean} != {r2.robustness_gap_mean}"
        )
        
        # KPI delta summaries should match
        for kpi_id in r1.stress_delta_summary.keys():
            assert kpi_id in r2.stress_delta_summary, f"Missing KPI {kpi_id} in second run"
            s1 = r1.stress_delta_summary[kpi_id]
            s2 = r2.stress_delta_summary[kpi_id]
            
            assert s1["mean"] == s2["mean"], (
                f"Mean delta mismatch for {kpi_id}: {s1['mean']} != {s2['mean']}"
            )
            assert s1["p10"] == s2["p10"], (
                f"P10 delta mismatch for {kpi_id}: {s1['p10']} != {s2['p10']}"
            )
            assert s1["p50"] == s2["p50"], (
                f"P50 delta mismatch for {kpi_id}: {s1['p50']} != {s2['p50']}"
            )
            assert s1["p90"] == s2["p90"], (
                f"P90 delta mismatch for {kpi_id}: {s1['p90']} != {s2['p90']}"
            )
            assert s1["p_improve"] == s2["p_improve"], (
                f"P(improve) mismatch for {kpi_id}: {s1['p_improve']} != {s2['p_improve']}"
            )


def test_different_seeds_produce_different_results():
    """
    Test that different seeds produce different results (sanity check).
    This ensures randomness is actually working.
    """
    plan = load_plan("data/plan.yaml")
    kpis = pd.read_csv("data/simulated_kpis.csv")
    inits = pd.read_csv("data/simulated_initiatives.csv")
    
    as_of = "2026-12-01"
    dates_to_check = ["2027-06-01"]
    
    bundles = [
        ActionBundle(
            id="TEST_BUNDLE_1",
            name="Test Bundle 1",
            description="Test bundle",
            actions=[
                SteeringAction(
                    id="A_TEST_1",
                    type="accelerate_initiative",
                    target_initiative="INIT_DQ1",
                    parameters={"months": 2},
                    description="Test action",
                )
            ],
        )
    ]
    
    integrity = compute_kpi_integrity(plan, kpis[["date", "kpi_id", "value"]], as_of=as_of)
    integrity_scores = {str(k): float(v.score) for k, v in integrity.items()}
    
    # Run with seed 1
    results1 = evaluate_bundles_mc(
        plan=plan,
        kpis_df=kpis,
        initiatives_df=inits,
        as_of=as_of,
        dates_to_check=dates_to_check,
        bundles=bundles,
        stress_events=[],
        whatif_cfg=WhatIfConfig(),
        mc_cfg=WhatIfStochasticConfig(n_samples=100, seed=1),
        integrity_scores=integrity_scores,
    )
    
    # Run with seed 2
    results2 = evaluate_bundles_mc(
        plan=plan,
        kpis_df=kpis,
        initiatives_df=inits,
        as_of=as_of,
        dates_to_check=dates_to_check,
        bundles=bundles,
        stress_events=[],
        whatif_cfg=WhatIfConfig(),
        mc_cfg=WhatIfStochasticConfig(n_samples=100, seed=2),
        integrity_scores=integrity_scores,
    )
    
    # Results should be different (with high probability)
    # Note: There's a tiny chance they could be the same, but extremely unlikely
    assert results1[0].score_stress_mean != results2[0].score_stress_mean, (
        "Different seeds produced identical results (unlikely but possible)"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
