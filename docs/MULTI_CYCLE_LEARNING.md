# Multi-Cycle Learning Loop Validation

## Overview

This document describes the implementation of the multi-cycle learning loop validation system, which validates that the continuous learning mechanism actually improves predictions over time.

## Problem Statement

The original `simulated_kpis.csv` data is a baseline trajectory that doesn't reflect the impact of selected strategic options. If we:
1. Generate a pack recommending "Option A: Accelerate INIT_DQ1"
2. Record that we selected Option A
3. Read "actual outcomes" from `simulated_kpis.csv` at the evaluation date

We're reading baseline values, not values that reflect Option A's impact. The learning loop would learn from the wrong outcomes.

## Solution: Closed-Loop Simulation

Instead of reading from `simulated_kpis.csv`, we use the what-if projection engine to simulate what the KPIs would be at the evaluation date, given the selected option was executed.

### How It Works

1. **At pack generation time** (e.g., `2026-02-01`):
   - Generate forecast (baseline, no actions)
   - Recommend Option A
   - Record the prediction: "We predict KPI_STP will be 88.5 at 2026-05-01 if Option A is executed"

2. **At evaluation time** (e.g., `2026-05-01`):
   - Use `project_kpis()` to simulate what KPIs would be at 2026-05-01, given that Option A was actually executed
   - This becomes the "synthetic ground truth" — not real data, but consistent with the model's own physics
   - Compare: predicted (88.5) vs. projected (88.7) → error = 0.2

3. **Learning loop**:
   - Computes prediction errors
   - Recalibrates forecast bias/sigma
   - Adjusts confidence and scoring weights

### Why This Works

- ✅ Validates the learning mechanism end-to-end
- ✅ Uses the model's own projections as ground truth (closed-loop)
- ✅ Tests whether recalibration actually improves future predictions
- ✅ Surfaces bugs in the learning logic

## Implementation

### New Modules

#### `src/outcome_simulation.py`

Provides `simulate_actual_outcomes()` function that:
- Re-runs the what-if projection with the selected option's actions applied
- Uses only historical data up to the original `as_of` date (no leakage!)
- Adds noise to simulate forecast error / reality
- Returns a dict of `kpi_id -> simulated actual value`

#### `src/multi_cycle_learning.py`

Main script that:
- Generates multiple SteerCo packs over time (e.g., monthly)
- Records decisions and option selections
- Simulates actual outcomes using the what-if engine
- Runs the learning loop to recalibrate forecasts/confidence/scoring
- Validates that predictions improve over cycles

### Usage

```bash
# Run 6 monthly cycles (Feb 2026 to Aug 2026)
python -m src.multi_cycle_learning \
    --start-date 2026-02-01 \
    --end-date 2026-09-01 \
    --cycle-months 1 \
    --evaluation-months 3 \
    --out-dir artifacts/learning_validation

# Use LLM-generated bundles (requires GEMINI_API_KEY)
python -m src.multi_cycle_learning \
    --start-date 2026-02-01 \
    --end-date 2026-09-01 \
    --cycle-months 1 \
    --evaluation-months 3 \
    --use-llm-bundles
```

### Output

The script generates:
- `artifacts/learning_validation/cycle_XX_YYYY-MM-DD/steerco_pack.json` — Pack for each cycle
- `artifacts/learning_validation/learning/decisions.jsonl` — Decision records with outcomes
- `artifacts/learning_validation/summary.json` — Aggregated results and learning metrics

### Validation Metrics

The script reports:
- **Prediction accuracy by cycle** — Mean absolute error (MAE) for each cycle
- **Learning trend** — Comparison of first half vs. second half average MAE
- **Final learning metrics** — Total decisions, acceptance rate, forecast accuracy, improvement trend

## Example Output

```
================================================================================
LEARNING VALIDATION SUMMARY
================================================================================

Cycles completed: 6

Prediction accuracy by cycle:
  Cycle  1 (as_of=2026-02-01): MAE=0.234
  Cycle  2 (as_of=2026-03-01): MAE=0.198
  Cycle  3 (as_of=2026-04-01): MAE=0.187
  Cycle  4 (as_of=2026-05-01): MAE=0.165
  Cycle  5 (as_of=2026-06-01): MAE=0.152
  Cycle  6 (as_of=2026-07-01): MAE=0.141

Learning trend:
  First half average MAE: 0.206
  Second half average MAE: 0.153
  Improvement: +25.7%
  ✅ Learning loop is improving predictions!

Final learning metrics:
  Total decisions: 6
  Accepted decisions: 6
  Mean forecast accuracy: 0.847
  Mean score prediction error: 0.163
  Improvement trend: improving
```

## Limitations

1. **Synthetic Ground Truth**: The "actual outcomes" are still synthetic (projected by the model), not real-world measurements. This validates the mechanism, not real-world accuracy.

2. **Noise Level**: The default 5% noise may not reflect real forecast error. Adjust `noise_level` parameter in `simulate_actual_outcomes()` if needed.

3. **Model Consistency**: The learning loop learns from the model's own projections. If the model is systematically biased, the learning may not help.

## Next Steps

1. **Wire `forecast_sliding.py` into `steerco_pack.py`** as the primary forecast engine (replacing basic Ridge model)
2. **Add unit tests** for learning loop components
3. **Run longer validation** (12+ cycles) to see if improvement plateaus
4. **Compare with real data** (when available) to validate the synthetic approach
