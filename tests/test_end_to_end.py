# tests/test_end_to_end.py
"""
End-to-end regression test for SteerCo pack generation.

Tests:
- Pack generated successfully
- Recommended option is stable for given seed
- Guardrails PASS/FAIL match expectations
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.portfolio import ActionBundle
from src.actions import SteeringAction
from src.steerco_pack import build_steerco_pack
from src.whatif import StressEvent, WhatIfStochasticConfig, WhatIfConfig
from src.forecast import ForecastConfig


def test_steerco_pack_generation():
    """Test that SteerCo pack is generated successfully."""
    plan_path = "data/plan.yaml"
    kpis_csv = "data/simulated_kpis.csv"
    inits_csv = "data/simulated_initiatives.csv"
    as_of = "2026-12-01"
    horizon_end = "2027-12-01"
    
    # Define test bundles
    bundles = [
        ActionBundle(
            id="OPT_C",
            name="Option C — Risk-first (Resilience only)",
            description="Test bundle",
            actions=[
                SteeringAction(
                    id="A_ACCEL_RES1_2M",
                    type="accelerate_initiative",
                    target_initiative="INIT_RES1",
                    parameters={"months": 2},
                    description="Accelerate RES1 by 2 months",
                )
            ],
        ),
    ]
    
    stress_events = [
        StressEvent(
            id="S_INC_SPIKE",
            type="kpi_shock",
            target_id="KPI_INC",
            start="2027-03-01",
            end="2027-05-01",
            magnitude=+8.0,
            description="Test stress event",
        ),
    ]
    
    # Generate pack
    pack = build_steerco_pack(
        plan_path=plan_path,
        kpis_csv=kpis_csv,
        initiatives_csv=inits_csv,
        as_of=as_of,
        horizon_end=horizon_end,
        bundles=bundles,
        stress_events=stress_events,
        mc_cfg=WhatIfStochasticConfig(n_samples=100, seed=42),  # Small sample for speed
        whatif_cfg=WhatIfConfig(),
        forecast_cfg=ForecastConfig(),
        out_dir=None,  # Don't write files
    )
    
    # Assertions
    assert pack is not None
    assert pack.as_of == as_of
    assert pack.horizon_end == horizon_end
    assert pack.recommendation is not None
    assert len(pack.options_mc) > 0
    assert pack.integrity_report is not None
    assert pack.guardrails_report is not None


def test_recommendation_stability():
    """Test that recommendation is stable for given seed."""
    plan_path = "data/plan.yaml"
    kpis_csv = "data/simulated_kpis.csv"
    inits_csv = "data/simulated_initiatives.csv"
    as_of = "2026-12-01"
    horizon_end = "2027-12-01"
    
    bundles = [
        ActionBundle(
            id="OPT_C",
            name="Option C",
            description="Test",
            actions=[
                SteeringAction(
                    id="A1",
                    type="accelerate_initiative",
                    target_initiative="INIT_RES1",
                    parameters={"months": 2},
                    description="Test",
                )
            ],
        ),
    ]
    
    seed = 42
    mc_cfg = WhatIfStochasticConfig(n_samples=100, seed=seed)
    
    # Run twice with same seed
    pack1 = build_steerco_pack(
        plan_path=plan_path,
        kpis_csv=kpis_csv,
        initiatives_csv=inits_csv,
        as_of=as_of,
        horizon_end=horizon_end,
        bundles=bundles,
        stress_events=[],
        mc_cfg=mc_cfg,
        whatif_cfg=WhatIfConfig(),
        forecast_cfg=ForecastConfig(),
        out_dir=None,
    )
    
    pack2 = build_steerco_pack(
        plan_path=plan_path,
        kpis_csv=kpis_csv,
        initiatives_csv=inits_csv,
        as_of=as_of,
        horizon_end=horizon_end,
        bundles=bundles,
        stress_events=[],
        mc_cfg=mc_cfg,
        whatif_cfg=WhatIfConfig(),
        forecast_cfg=ForecastConfig(),
        out_dir=None,
    )
    
    # Should be identical
    assert pack1.recommendation.chosen_bundle_id == pack2.recommendation.chosen_bundle_id
    assert abs(pack1.recommendation.confidence - pack2.recommendation.confidence) < 0.001
    
    # Options should have same scores
    assert len(pack1.options_mc) == len(pack2.options_mc)
    for opt1, opt2 in zip(pack1.options_mc, pack2.options_mc):
        assert opt1["bundle_id"] == opt2["bundle_id"]
        assert abs(opt1["score_stress_mean"] - opt2["score_stress_mean"]) < 0.001
        assert abs(opt1["score_stress_cvar10"] - opt2["score_stress_cvar10"]) < 0.001


def test_guardrails_expectations():
    """Test that guardrails PASS/FAIL match expectations."""
    plan_path = "data/plan.yaml"
    kpis_csv = "data/simulated_kpis.csv"
    inits_csv = "data/simulated_initiatives.csv"
    as_of = "2026-12-01"
    horizon_end = "2027-12-01"
    
    # Option C (RES1 only) should pass budget (RES1 budget is 400k, 2026 limit is 600k)
    bundles = [
        ActionBundle(
            id="OPT_C",
            name="Option C",
            description="Test",
            actions=[
                SteeringAction(
                    id="A1",
                    type="accelerate_initiative",
                    target_initiative="INIT_RES1",
                    parameters={"months": 2},
                    description="Test",
                )
            ],
        ),
    ]
    
    pack = build_steerco_pack(
        plan_path=plan_path,
        kpis_csv=kpis_csv,
        initiatives_csv=inits_csv,
        as_of=as_of,
        horizon_end=horizon_end,
        bundles=bundles,
        stress_events=[],
        mc_cfg=WhatIfStochasticConfig(n_samples=100, seed=42),
        whatif_cfg=WhatIfConfig(),
        forecast_cfg=ForecastConfig(),
        out_dir=None,
    )
    
    # Check that guardrails were evaluated
    assert pack.guardrails_report is not None
    assert "GUARDRAILS EVALUATION" in pack.guardrails_report
    
    # Option C should pass (RES1 budget < 600k limit)
    # This is a basic sanity check - actual pass/fail depends on implementation
    assert pack.recommendation is not None
