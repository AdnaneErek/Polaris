# Learning from Outcomes - Usage Guide

## Overview

The learning system records what was recommended and learns from actual outcomes to improve future recommendations.

## How It Works

1. **Automatic Recording**: Each time you run `steerco_run`, the system automatically records:
   - What was recommended (bundle, confidence, scores)
   - When it was recommended
   - Guardrails status

2. **Manual Outcome Recording**: After 3-6 months, when actual outcomes are known, you record:
   - Was the recommendation accepted?
   - What were the actual KPI values?
   - What were the predicted KPI values?
   - When were outcomes measured?

2.5 **SteerCo Option Selection (new)**:
   - In Dashboard → **Options Analysis**, use **Record SteerCo Selected Option** to save which option was actually chosen.
   - This marks the decision as accepted/rejected before KPI outcomes are available.

3. **Automatic Learning**: The system then:
   - Computes prediction errors
   - Adjusts forecast weights
   - Tracks improvement trends
   - Shows learning metrics in future packs

## Usage

### 1. List Pending Decisions

See which decisions need outcomes recorded:

```bash
python -m src.record_outcome --list
```

Output:
```
📋 Pending Decisions (2):
================================================================================

1. Decision from 2026-12-01
   Timestamp: 2026-01-23T22:16:28.298914
   Recommended: Option C — Risk-first (Resilience only) (OPT_C)
   Confidence: 0.50
   Guardrails: PASSED
   Score (stress CVaR10): 0.50
```

### 2. Record Outcome (Interactive Mode)

Easiest way to record outcomes:

```bash
python -m src.record_outcome --interactive
```

This will:
1. Show all pending decisions
2. Let you select which one to update
3. Ask for acceptance status
4. Ask for actual KPI values
5. Ask for predicted KPI values (optional)
6. Confirm and record

### 3. Record Outcome (CLI Mode)

For automation or scripts:

```bash
python -m src.record_outcome \
    --decision-timestamp "2026-01-23T22:16:28.298914" \
    --accepted true \
    --kpi KPI_STP 88.5 \
    --kpi KPI_E2E 2.1 \
    --kpi KPI_INC 25.0 \
    --evaluation-date "2027-06-01"
```

With predicted values:

```bash
python -m src.record_outcome \
    --decision-timestamp "2026-01-23T22:16:28.298914" \
    --accepted true \
    --kpi KPI_STP 88.5 \
    --kpi KPI_E2E 2.1 \
    --kpi KPI_INC 25.0 \
    --predicted-kpi KPI_STP 87.2 \
    --predicted-kpi KPI_E2E 2.15 \
    --predicted-kpi KPI_INC 26.5 \
    --evaluation-date "2027-06-01"
```

### 4. Auto-record outcomes from a KPI CSV (recommended for synthetic runs)

If you have KPI actuals in a CSV (like `data/simulated_kpis.csv`), you can auto-fill the outcome for the latest pending decision:

```bash
python -m src.auto_record_outcomes --kpis-csv data/simulated_kpis.csv --latest-pending --accepted true
```

If your decisions are stored per-pack (e.g., `artifacts/steerco/<as_of>/learning`), point the tool at that folder:

```bash
python -m src.auto_record_outcomes --kpis-csv data/simulated_kpis.csv --storage-dir artifacts/steerco/2026-02-06/learning --latest-pending
```

## Example Workflow

### Month 1: Decision Made
```bash
python -m src.steerco_run --as_of 2026-12-01
# System automatically records: "Recommended Option C"
```

### Month 6: Outcomes Known
```bash
# Check what needs updating
python -m src.record_outcome --list

# Record what actually happened
python -m src.record_outcome --interactive
# Enter:
#   - Accepted: yes
#   - Actual KPI_STP: 88.5
#   - Actual KPI_E2E: 2.1
#   - Actual KPI_INC: 25.0
#   - Evaluation date: 2027-06-01
```

### Month 7: Next Decision
```bash
python -m src.steerco_run --as_of 2027-07-01
# System now shows improved learning metrics:
#   "Forecast accuracy: 85%"
#   "Improvement trend: improving"
```

## Learning Metrics

After recording outcomes, the system tracks:

- **Total decisions**: How many recommendations made
- **Accepted decisions**: How many were accepted
- **Forecast accuracy**: How close predictions were to actuals
- **Improvement trend**: Is accuracy getting better over time?

These metrics appear in:
- Console output (when running `steerco_run`)
- Markdown pack (`steerco_pack.md`)
- JSON pack (`steerco_pack.json`)

## Storage

Decision records are stored in:
- `artifacts/steerco/{as_of}/learning/decisions.jsonl` (per-run storage)
- Or `artifacts/learning/decisions.jsonl` (global storage)

Each line is a JSON record of one decision.

## Tips

1. **Record outcomes regularly**: Don't wait too long, or you'll forget details
2. **Be accurate**: Actual values should be measured at the evaluation date
3. **Include predictions**: If you have the predicted values from the original pack, include them for better learning
4. **Check trends**: After a few decisions, check if accuracy is improving

## Future Enhancements

Potential improvements:
- Integration with KPI tracking systems (automatic outcome recording)
- Integration with project management tools (automatic acceptance tracking)
- Web UI for easier outcome recording
- Email reminders for pending decisions
