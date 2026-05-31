## POLARIS
## Portfolio Optimization & Learning AI for Risk-Adjusted Strategy

*POLARIS: A guiding star for strategic decision-making*

---

## v1 to v2 Recap (What Changed)

This v2 document keeps the original README structure and updates it with the latest implementation improvements.

### Reporting & Executive Communication

- Added executive-first report flow with stronger decision framing
- Added highlighted decision box and clearer steering ask
- Added adverse-gap semantics to avoid KPI sign misinterpretation
- Added risk register with mitigation ownership in the report
- Improved options section order: ranking table first, detailed options after
- Strengthened conclusion with forward-looking confidence stance
- Added page numbers and stronger visual hierarchy in generated reports

### Charts & Visual Design

- Harmonized chart semantics across dashboard and report
- Improved readability: cleaner backgrounds, lighter grids, reduced legend clutter
- Increased emphasis on recommended-option forecast divergence
- Standardized brand-oriented typography and semantic color usage

### Forecasting, Baselines, and Option Impact

- Unified plotting logic around:
  - `Actual (observed)` up to `as_of`
  - `Plan baseline (weekly objectives)` across horizon
  - `Forecast (model)` after `as_of`
- Added recommended-option impact forecast alongside baseline forecast
- Added forecast spread note (recommended vs baseline at horizon)

### Options, Governance, and Budget

- Improved option generation quality by using richer initiative/workstream details
- Added bundle provenance visibility (LLM vs manual fallback context)
- Integrated plan budget data into report narratives:
  - budget envelope (baseline vs adjusted)
  - annual cap check references where available

### Learning Loop (Path A)

- Added decision/outcome capture and learning metrics surfacing
- Added forecast recalibration, confidence calibration, and scoring-weight adaptation hooks
- Exposed learning metrics in pack/dashboard/report for transparency

---

## Executive Summary

**POLARIS** (Portfolio Optimization & Learning AI for Risk-Adjusted Strategy) is an enterprise-grade AI-powered platform designed to transform strategic planning and decision-making. Named after the North Star, POLARIS serves as a reliable beacon for strategic decisions in complex, uncertain environments.

While initially developed from a banking operations use case, the system architecture is generalizable across industries that need risk-aware, auditable, and explainable strategic steering.

The system addresses core strategic planning challenges:
- **Complexity**: Multiple objectives, KPIs, initiatives, dependencies, and constraints
- **Uncertainty**: Probabilistic forecasting and stress-aware option evaluation
- **Compliance**: Built-in guardrails and governance checks
- **Learning**: Closed feedback loop from recommendation to realized outcomes
- **Transparency**: Traceable rationale, attribution, and decision evidence

---

## What It Does

### Core Functionality

POLARIS operates as an intelligent strategic advisor that:

1. **Monitors Performance**: Tracks KPIs, initiatives, and objective trajectory status  
2. **Detects Anomalies**: Uses ML anomaly detection and rule-based integrity checks  
3. **Forecasts Outcomes**: Generates baseline and scenario-based KPI forecasts  
4. **Evaluates Options**: Compares bundles using Monte Carlo and risk-adjusted metrics  
5. **Recommends Actions**: Selects options with rationale and confidence  
6. **Enforces Guardrails**: Validates budget, capacity, controls, and compliance constraints  
7. **Learns from Outcomes**: Recalibrates confidence, forecasts, and scoring weights  
8. **Generates Reports**: Produces executive dashboard views and steering-ready PDF reports  

### Key Differentiators

- **Human-in-the-Loop**: Recommendation-only autonomy with formal approvals  
- **Audit Trail**: Decision, rationale, and artifact traceability  
- **Risk-Aware Optimization**: CVaR-based stress-aware option ranking  
- **Attribution Transparency**: Initiative/workstream contribution visibility  
- **Continuous Learning**: Outcome-driven recalibration across cycles  

---

## System Architecture

### High-Level Components

```
┌─────────────────────────────────────────────────────────────┐
│                    Strategic Plan (YAML)                    │
│  - KPIs, Objectives, Initiatives, Constraints, Governance   │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                    Monitoring Layer                          │
│  - KPI drift, ML anomalies, integrity checks                │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                    Forecasting Engine                        │
│  - Baseline forecast, what-if projection, weekly baseline   │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                    Portfolio Evaluation                      │
│  - MC scoring, CVaR10, attribution, option comparison       │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                    Recommendation Engine                     │
│  - Guardrails, confidence, rationale                         │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                    Learning System                           │
│  - Outcome capture, recalibration, scoring adaptation       │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                    Presentation Layer                        │
│  - SteerCo pack, dashboard, PDF report, chat interface      │
└─────────────────────────────────────────────────────────────┘
```

### Data Flow

1. **Input**: `plan.yaml`, KPI history, initiative progress  
2. **Processing**: monitor → forecast → evaluate → recommend  
3. **Output**: SteerCo pack JSON/markdown, dashboard views, executive PDF  
4. **Feedback**: record outcomes → learn → recalibrate future recommendations  

---

## Core Features

### 1. Strategic Plan Management

**Purpose**: Centralized strategy definition (objectives, KPIs, initiatives, constraints)

**Components**:
- KPI catalog with definitions and exclusions
- Objectives/OKRs with trajectory checkpoints
- Initiatives with timelines, dependencies, risks, budgets, and workstreams
- Portfolio constraints (budget, capacity, control thresholds)
- Governance and decision policy

**Upgrades implemented**:
- Initiative/workstream detail is now used directly in option generation and reporting
- Plan budget fields are used in report budget envelope and annual cap checks

---

### 2. Monitoring & Anomaly Detection

**Purpose**: Detect material deviations early

**Components**:
- Direction-aware KPI drift monitoring  
- ML anomalies (Isolation Forest)  
- KPI integrity checks (staleness, volatility, suspicious patterns)  

**Upgrades implemented**:
- Dashboard/report harmonized views: `Actual (observed)`, `Plan baseline (weekly objectives)`, `Forecast (model)`  
- As-of marker and continuity fixes between actual and forecast lines  

---

### 3. Forecasting & What-If Analysis

**Purpose**: Project trajectories under baseline and option scenarios

**Components**:
- Baseline forecasting model  
- Recommended option impact forecast  
- Weekly objective baseline + deviation alerts  

**Upgrades implemented**:
- Leakage-safe behavior retained (as-of bounded usage)  
- Report now shows baseline forecast + recommended option forecast with spread note  

---

### 4. Portfolio Evaluation & Optimization

**Purpose**: Compare strategic options with risk-aware scoring

**Components**:
- Action bundles (manual or LLM-generated)  
- Monte Carlo evaluation  
- CVaR10 / expected return / best-case (p90) comparison  

**Upgrades implemented**:
- Scoring date and zero-score issues fixed  
- Option impact visibility improved in dashboard/report  
- Scoring table and narrative now positioned for executive consumption

---

### 5. Guardrails & Compliance

**Purpose**: Enforce constraints before recommendation execution

**Components**:
- Budget/capacity/control/compliance checks  
- Violation explanation and escalation signals  

**Upgrades implemented**:
- “SteerCo decision required (guardrails failed)” shown consistently without renaming top option  
- Budget envelope and annual cap usage are now surfaced in reporting text

---

### 6. Learning from Outcomes

**Purpose**: Improve recommendation quality over time

**Components**:
- Decision/outcome recording  
- Forecast recalibration parameters  
- Confidence calibration multiplier  
- Scoring weight adaptation  

**Upgrades implemented**:
- Learning metrics surfaced in pack/dashboard/report header sections

---

### 7. SteerCo Pack Generation

**Purpose**: Create decision-ready steering artifacts

**Components**:
- JSON pack + markdown memo  
- Recommendation, option analysis, guardrails, and learning metadata  

**Upgrades implemented**:
- Full forecast payload, weekly baseline, deviation alerts, bundle provenance, learning metrics added

---

### 8. Interactive Dashboard

**Purpose**: Interactive strategic steering UI

**Components**:
- Strategic brief, monitoring, options, learning, reports  
- Option scenario comparisons and selection history  

**Upgrades implemented**:
- Option impact charts and delta view  
- Selection history + learning panels  
- Bundle provenance + recalibration indicators

---

### 9. Conversational Interface

**Purpose**: Natural-language Q&A over strategic artifacts

**Components**:
- Context extraction from pack  
- Recommendation and risk explanation responses  

---

### 10. PDF Report Generation

**Purpose**: Produce steering-committee-ready PDF reports

**Current report capabilities (upgraded)**:
- Executive decision box (highlighted ask)
- KPI health dashboard with semantic status badges
- Critical issues with adverse-gap convention and risk interpretation
- CVaR10 explanation in executive language
- Option ranking table before detailed option narratives
- Budget/resource implication lines derived from plan data
- Risk register with mitigation ownership
- Improved chart styling, reduced legend clutter, page numbers
- Stronger conclusion with forward-looking confidence stance
- Technical KPI formula/details moved to appendix

---

### 11. Natural Language Objective Creation

**Purpose**: Convert free text strategic intent into structured objective data

**Components**:
- LLM-assisted parsing
- Missing information prompts
- Schema-compatible output integration

---

## Technical Requirements

### Software Dependencies

- `pydantic`, `pandas`, `numpy`, `pyyaml`
- `scikit-learn`
- `streamlit`, `plotly`, `matplotlib`, `reportlab`
- `google-generativeai` (optional, for LLM features)

### System Requirements

- Python 3.10+
- Windows/Linux/macOS

### Environment Variables

- `GEMINI_API_KEY` (required for LLM-based generation/chat features)

---

## Business Value

### 1. Better Decision Quality
- Quantified risk/return trade-offs with stress awareness

### 2. Time Savings
- Automated scoring/reporting and faster steering prep

### 3. Compliance & Risk Management
- Guardrail-first recommendations with explicit violations/escalation

### 4. Continuous Improvement
- Outcome-driven recalibration loop

### 5. Strategic Alignment
- KPI-to-initiative attribution and objective-focused intervention

### 6. Executive Communication
- Steering-ready report/dashboard with decision framing and accountability

---

## Use Cases

### 1. Monthly SteerCo
- Generate pack, review recommendation/risk, approve option, log selection/outcomes

### 2. Quarterly Strategic Review
- Use report synthesis, risk register, and forecast divergence views

### 3. Initiative Planning & Re-prioritization
- Compare options under capacity/budget constraints

### 4. Risk Assessment
- Stress-test options and assess downside via CVaR10

### 5. Ongoing Monitoring
- Detect anomalies and forecast deviations vs weekly objectives

### 6. Learning Cycle
- Compare predicted vs realized outcomes and recalibrate

---

## Key Differentiators

### 1. Human-in-the-Loop by Design
- Recommendation-only autonomy with clear escalation paths

### 2. Governance-Ready Outputs
- Decision framing, risk ownership, and auditability

### 3. Risk-Adjusted Optimization
- CVaR-aware ranking and scenario analysis

### 4. Attribution & Explainability
- Driver-level transparency in recommendation rationale

### 5. Adaptive Intelligence
- Learning metrics and calibration loop embedded in runtime

---

## Implementation Status

### ✅ Completed
- Strategic plan parsing and validation
- Monitoring/anomaly/integrity pipeline
- Forecasting + what-if option impact
- Monte Carlo option evaluation + guardrails
- Learning adapter + metrics
- Dashboard with scenario and learning views
- Executive PDF reporting (v2 upgraded)

### 🔄 In Progress
- Deeper attribution granularity
- Production data source connectors

### 📋 Future Enhancements
- Real-time ingestion and alert orchestration
- Automated outcome ingestion connectors
- Collaboration workflows (comments/approvals)

---

## Getting Started

### Quick Start (PowerShell)

```powershell
cd C:\Users\hp\Desktop\agentic-strategy-poc
pip install -r requirements.txt
```

Set optional LLM key:

```powershell
setx GEMINI_API_KEY "your-api-key"
```

Generate SteerCo pack:

```powershell
python -m src.steerco_run --as_of 2027-06-01 --horizon_end 2028-12-01
```

Launch dashboard:

```powershell
streamlit run src/dashboard.py
```

Generate report from pack:

```powershell
python -c "import json,pandas as pd; from src.load_plan import load_plan; from src.report_generator import generate_progress_report; p=json.load(open('artifacts/steerco/2027-06-01/steerco_pack.json',encoding='utf-8')); plan=load_plan('data/plan.yaml'); k=pd.read_csv('data/simulated_kpis.csv'); i=pd.read_csv('data/simulated_initiatives.csv'); print(generate_progress_report(plan,k,i,as_of='2027-06-01',period_type='monthly',pack=p))"
```

Record outcomes:

```powershell
python -m src.record_outcome --interactive
```

---

## Conclusion

POLARIS has evolved from a planning analytics prototype into a governance-ready strategic steering system. The current implementation combines risk-aware optimization, explicit guardrails, and a learning loop, while producing executive-grade outputs suitable for recurring steering cycles.

The latest upgrades materially improve decision usability: clearer asks, stronger option comparability, budget/capacity visibility, risk ownership, and more effective narrative synthesis.

---

*Document Version: 2.0*  
*Last Updated: February 2026*  
*Project: POLARIS (Portfolio Optimization & Learning AI for Risk-Adjusted Strategy)*
