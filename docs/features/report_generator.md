# PDF Report Generator Feature

## Overview

Generates professional PDF reports summarizing strategic progress with KPI trends, options analysis, and recommendations.

## Features

### 1. **Period Selection**
Choose monthly, quarterly, or yearly reporting periods.

### 2. **Objective Selection**
Generate reports for all objectives or selected objective subsets.

### 3. **Report Contents**

#### Executive Summary
- High-level progress snapshot
- Decision framing and key asks

#### Progress Narrative
- KPI and initiative trend interpretation
- Issues and implications

#### Objective Sections
- Objective-by-objective status
- KPI-level progress and commentary

#### Charts & Visualizations
- KPI trend charts
- Comparative option visuals
- Forecast and baseline views

### 4. **Professional PDF Format**
- Structured sections and typography
- Visual hierarchy and page numbering
- Export-ready for steering committees

## Usage

### Via Dashboard
Use the report/export section to generate and download a PDF.

### Via Python API

```python
from src.report_generator import generate_progress_report
from src.load_plan import load_plan
import pandas as pd

# Load plan and data
plan = load_plan("data/plan.yaml")
kpi_df = pd.read_csv("data/simulated_kpis.csv")
init_df = pd.read_csv("data/simulated_initiatives.csv")

# Generate report
pdf_path = generate_progress_report(
    plan=plan,
    kpi_history=kpi_df,
    initiative_history=init_df,
    as_of="2026-12-01",
    period_type="quarterly",  # or "monthly", "yearly"
    objective_ids=None,  # None = all, or ["OBJ_1", "OBJ_2"]
)

print(f"Report saved to: {pdf_path}")
```

## Report Structure
1. Metadata and context
2. Executive decision summary
3. Critical issues and risk signals
4. Strategic options analysis
5. KPI progress and plots
6. Recommendations and ownership

## Technical Details

### Dependencies
- `reportlab`
- `matplotlib`
- `pandas`

### File Output
Reports are written to `artifacts/reports/` by default.

### Data Requirements
- KPI history with `date`, `kpi_id`, `value`
- Initiative history with `date`, `initiative_id`, `progress`

## Example Use Cases
- Monthly progress report
- Quarterly objective review
- Annual strategic review
- Objective-specific deep dive

## Future Enhancements
- Additional chart templates
- Comparison of multiple periods
- Presentation-friendly summary page
