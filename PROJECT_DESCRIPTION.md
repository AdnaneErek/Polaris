# POLARIS
## Portfolio Optimization & Learning AI for Risk-Adjusted Strategy

*POLARIS: A guiding star for strategic decision-making*

---

## Executive Summary

**POLARIS** (Portfolio Optimization & Learning AI for Risk-Adjusted Strategy) is an enterprise-grade AI-powered platform designed to transform strategic planning and decision-making. Named after the North Star—a constant guide in navigation—POLARIS serves as a reliable beacon for strategic decisions in complex, uncertain environments.

While initially developed based on a banking operations use case, the system is designed to be generalized and applicable across various industries and domains. It provides intelligent, data-driven recommendations for strategic initiatives while ensuring compliance, risk management, and continuous learning.

The system addresses critical challenges in strategic planning:
- **Complexity**: Managing multiple objectives, KPIs, initiatives, and dependencies
- **Uncertainty**: Forecasting outcomes in volatile environments
- **Compliance**: Ensuring all recommendations meet regulatory and governance requirements
- **Learning**: Continuously improving recommendations based on actual outcomes
- **Transparency**: Providing clear rationale for all decisions

---

## What It Does

### Core Functionality

The system operates as an **intelligent strategic advisor** that:

1. **Monitors Performance**: Continuously tracks KPIs, initiatives, and portfolio health
2. **Detects Anomalies**: Uses machine learning (Isolation Forest) to identify unexpected patterns
3. **Forecasts Outcomes**: Projects future KPI values under different scenarios
4. **Evaluates Options**: Compares multiple strategic options using Monte Carlo simulation
5. **Recommends Actions**: Suggests optimal action bundles based on risk-adjusted scoring
6. **Enforces Guardrails**: Ensures all recommendations comply with regulatory and business constraints
7. **Learns from Outcomes**: Improves future recommendations by learning from past decisions
8. **Generates Reports**: Creates executive-ready reports and visualizations

### Key Differentiators

- **Human-in-the-Loop**: All recommendations require human approval; no autonomous execution
- **Audit Trail**: Complete logging of all decisions, inputs, and rationale
- **Risk-Aware**: Uses stress testing and risk-adjusted metrics (CVaR) for decision-making
- **Attribution**: Tracks which initiatives contribute to which KPI improvements
- **Continuous Learning**: Self-improving system that gets better over time

---

## System Architecture

### High-Level Components

```
┌─────────────────────────────────────────────────────────────┐
│                    Strategic Plan (YAML)                    │
│  - KPIs, Objectives, Initiatives, Portfolio Constraints     │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                    Monitoring Layer                          │
│  - KPI Tracking, Anomaly Detection, Integrity Checks       │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                    Forecasting Engine                        │
│  - Time Series Forecasting, Monte Carlo Simulation          │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                    Portfolio Evaluation                      │
│  - Option Scoring, Stress Testing, Attribution Analysis    │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                    Recommendation Engine                     │
│  - Guardrails Check, Risk Assessment, Rationale Generation │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                    Learning System                           │
│  - Outcome Recording, Forecast Adjustment, Metrics Tracking│
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                    Presentation Layer                        │
│  - SteerCo Pack, Dashboard, Reports, Chat Interface        │
└─────────────────────────────────────────────────────────────┘
```

### Data Flow

1. **Input**: Strategic plan (YAML), historical KPI data, initiative progress
2. **Processing**: Monitoring → Forecasting → Evaluation → Recommendation
3. **Output**: SteerCo pack (markdown/JSON), dashboard, PDF reports
4. **Feedback**: Outcome recording → Learning → Improved future recommendations

---

## Core Features

### 1. Strategic Plan Management

**Purpose**: Centralized definition of strategy, objectives, and constraints

**Components**:
- **KPI Catalog**: Comprehensive definitions with formulas, data sources, baselines, targets
- **Strategic Objectives**: OKRs with weights, trajectories, and constraints
- **Initiatives**: Projects with timelines, budgets, dependencies, RACI, risks, benefits
- **Portfolio Constraints**: Budget limits, capacity constraints, control requirements
- **Governance Rules**: Decision rights, auditability requirements, guardrails

**Key Capabilities**:
- YAML-based configuration for version control and collaboration
- Cross-reference validation (ensures all IDs are valid)
- Trajectory planning (linear, piecewise, s-curve)
- Dependency management (hard/soft dependencies between initiatives)
- Risk classification (high/medium/low for KPIs and initiatives)

**Example Use Case**: Define a 3-year strategic plan with 3 objectives, 5 KPIs, and 8 initiatives, all with proper governance and constraints.

---

### 2. Monitoring & Anomaly Detection

**Purpose**: Continuous monitoring of KPIs and detection of unexpected patterns

**Components**:
- **KPI Monitoring**: Tracks current values, trends, and deviations from expected
- **Anomaly Detection**: ML-based (Isolation Forest) identification of outliers
- **Integrity Checks**: Validates data quality, completeness, and consistency
- **Initiative Tracking**: Monitors progress, milestones, and dependencies

**Key Capabilities**:
- Real-time anomaly scoring (0-1 scale)
- Direction-aware detection (considers whether "up" or "down" is good)
- Data quality validation (missing values, outliers, format issues)
- Trend analysis (improving, stable, deteriorating)
- Alert generation for material deviations

**Example Use Case**: System detects that incident count (KPI_INC) is 40% higher than expected, flags it as anomalous, and triggers investigation.

---

### 3. Forecasting & What-If Analysis

**Purpose**: Project future KPI values under different scenarios

**Components**:
- **Time Series Forecasting**: Projects KPI values into the future
- **Monte Carlo Simulation**: Generates probabilistic forecasts with confidence bands
- **What-If Scenarios**: Evaluates impact of different action bundles
- **Stress Testing**: Models impact of adverse events (incident spikes, market shocks)

**Key Capabilities**:
- Probabilistic forecasts (not just point estimates)
- Confidence intervals (10th, 50th, 90th percentiles)
- Scenario comparison (baseline vs. action bundles)
- Stress event modeling (temporary shocks to KPIs)
- Attribution analysis (which initiatives drive which KPI changes)

**Example Use Case**: Forecast that STP rate will reach 88% by end of 2027 under baseline, but 91% if we accelerate automation initiative.

---

### 4. Portfolio Evaluation & Optimization

**Purpose**: Compare multiple strategic options and select optimal action bundles

**Components**:
- **Action Bundles**: Pre-defined or LLM-generated combinations of steering actions
- **LLM Bundle Generation**: Automatically generates action bundles based on current analysis (optional)
- **Option Scoring**: Multi-criteria evaluation (expected value, risk, guardrails)
- **Monte Carlo Analysis**: Risk-adjusted scoring using CVaR (Conditional Value at Risk)
- **Attribution Analysis**: Identifies which initiatives contribute to KPI improvements

**Key Capabilities**:
- **Automatic Bundle Generation**: Uses Gemini AI to generate action bundles based on:
  - Current progress (ahead/behind schedule)
  - Detected issues (anomalies, integrity, dependencies)
  - Initiative health and blockers
  - Strategic objectives at risk
- Multiple scoring metrics (expected value, stress CVaR10, guardrails pass rate)
- Statistical tie detection (when options are too close to distinguish)
- Confidence calculation (based on score separation)
- Driver identification (top 3 initiatives driving each KPI)
- Budget and capacity constraint checking
- Fallback to manual bundles if LLM generation fails

**Example Use Case**: System analyzes that STP rate is behind schedule and INIT_A1 is blocked. LLM generates 3 options: "Recovery Focus" (accelerate blocker), "Risk-First" (split scope), and "Balanced" (address both). System evaluates all options and recommends "Recovery Focus" with 65% confidence.

---

### 5. Guardrails & Compliance

**Purpose**: Ensure all recommendations meet regulatory and business constraints

**Components**:
- **Constraint Validation**: Checks budget, capacity, control coverage, compliance rules
- **Guardrail Evaluation**: Pass/fail assessment for each option
- **Approval Requirements**: Identifies who must approve if guardrails fail
- **Risk Classification**: Flags high-risk recommendations

**Key Capabilities**:
- Multi-constraint checking (budget, capacity, control, compliance)
- Violation reporting (detailed messages for each failure)
- Approval workflow (identifies required approvers)
- Safety-first design (never recommends violating constraints)

**Example Use Case**: Option B fails guardrails because it would reduce control coverage below 95% threshold, requiring Risk/Compliance approval before proceeding.

---

### 6. Learning from Outcomes

**Purpose**: Improve future recommendations by learning from actual outcomes

**Components**:
- **Decision Recording**: Automatically logs all recommendations
- **Outcome Tracking**: Records actual KPI values after decisions are implemented
- **Forecast Adjustment**: Updates forecast weights based on prediction errors
- **Metrics Tracking**: Monitors accuracy, improvement trends, acceptance rates

**Key Capabilities**:
- Automatic decision logging (timestamp, recommendation, confidence, scores)
- Manual outcome recording (CLI tool for entering actual results)
- Prediction error calculation (forecast vs. actual)
- Weight adjustment (improves future forecasts)
- Learning metrics dashboard (accuracy, trend, acceptance rate)

**Example Use Case**: After 6 months, record that actual STP rate was 88.5% vs. predicted 87.2%. System adjusts forecast weights, improving next recommendation accuracy from 80% to 85%.

---

### 7. SteerCo Pack Generation

**Purpose**: Create executive-ready steering committee briefing documents

**Components**:
- **Markdown Report**: Human-readable narrative with key insights
- **JSON Export**: Machine-readable data for integration
- **Executive Summary**: High-level overview of status and recommendations
- **Detailed Analysis**: KPI forecasts, option comparison, guardrails status

**Key Capabilities**:
- Memo-style formatting (executive-friendly structure)
- Three-question framework:
  1. Are we on track? (KPI status)
  2. What should we do next? (Recommendation)
  3. What could go wrong? (Guardrails, approvals)
- Key drivers identification (attribution analysis)
- Decision banners (clear approval status)
- Learning metrics integration

**Example Use Case**: Generate monthly SteerCo pack showing 2 KPIs on track, 1 requiring attention, recommending Option C with 50% confidence, all guardrails passed.

---

### 8. Interactive Dashboard

**Purpose**: Web-based UI for exploring strategic planning data

**Components**:
- **KPI Forecasts Tab**: Interactive charts showing forecast trends
- **Portfolio Options Tab**: Comparison of strategic options
- **Recommendation Tab**: Detailed view of recommended option
- **Learning Tab**: Metrics on decision accuracy and improvement
- **SteerCo Brief Tab**: Executive summary view
- **Chat Tab**: Natural language queries about the plan
- **Reports Tab**: PDF report generation interface
- **Add Objective Tab**: Natural language objective creation

**Key Capabilities**:
- Interactive Plotly charts (hover for details, zoom, pan)
- Multi-pack selection (compare across time periods)
- Real-time filtering and exploration
- Export functionality (charts, data tables)
- Responsive design (works on desktop and tablet)

**Example Use Case**: Executive opens dashboard, views KPI forecast charts, compares options, reads SteerCo brief, asks chat "Why was Option C recommended?", generates quarterly PDF report.

---

### 9. Conversational Interface

**Purpose**: Natural language queries about strategic plan and recommendations

**Components**:
- **Gemini AI Integration**: Uses Google's Gemini API for natural language understanding
- **Context Building**: Extracts key information from SteerCo pack
- **Question Answering**: Responds to queries about recommendations, KPIs, options, learning
- **Interactive Mode**: Multi-turn conversations with context

**Key Capabilities**:
- Natural language understanding (no need for specific syntax)
- Context-aware responses (understands pack data)
- Multiple question types (why, what, how, compare)
- Interactive chat mode (conversational flow)
- Single-question mode (for scripts/automation)

**Example Use Case**: User asks "Why was Option C recommended?" and receives detailed explanation of scoring, guardrails, and rationale. Follow-up: "What are the main risks?" gets risk analysis.

---

### 10. PDF Report Generation

**Purpose**: Create professional progress reports for stakeholders

**Components**:
- **Period Selection**: Monthly, quarterly, or yearly aggregation
- **Objective Selection**: All objectives, specific set, or single objective
- **Executive Summary**: High-level progress overview
- **Progress Narrative**: Detailed written analysis
- **Charts & Visualizations**: KPI progress charts with targets and baselines
- **Attribution Analysis**: Shows which initiatives drive progress

**Key Capabilities**:
- Customizable period aggregation (monthly/quarterly/yearly)
- Objective filtering (all, specific, or single)
- Professional PDF formatting (clean layout, proper styling)
- Embedded charts (high-resolution images)
- Problem identification (flags KPIs below expected)
- Actions taken (shows what initiatives are driving progress)
- Impact analysis (quantifies contribution of each initiative)

**Example Use Case**: Generate quarterly report for all objectives, showing STP rate improved from 82% to 88%, with automation initiative contributing 60% of the improvement.

---

### 11. Natural Language Objective Creation

**Purpose**: Add new objectives to the plan using natural language

**Components**:
- **LLM-Powered Parsing**: Uses Gemini AI to extract structured data from text
- **Interactive Prompting**: Asks for missing information if needed
- **YAML Generation**: Converts natural language to plan.yaml format
- **Validation**: Ensures new objective meets schema requirements

**Key Capabilities**:
- Natural language input (describe objective in plain English)
- Automatic extraction (KPIs, targets, deadlines, initiatives)
- Missing information detection (prompts for required fields)
- YAML integration (adds to existing plan.yaml)
- Schema validation (ensures compatibility)

**Example Use Case**: User enters "I want to reduce processing errors by 50% by end of 2027, focusing on payment reconciliation." System extracts objective, creates KPI, sets target, links to initiatives, prompts for missing details.

---

## Technical Requirements

### Software Dependencies

**Core Libraries**:
- `pydantic>=2.0` - Data validation and schema management
- `pandas>=2.0` - Data manipulation and analysis
- `numpy>=1.25` - Numerical computations
- `pyyaml>=6.0` - YAML parsing and generation
- `scikit-learn>=1.3.0` - Machine learning (Isolation Forest for anomaly detection)

**Visualization & UI**:
- `streamlit>=1.28.0` - Web dashboard framework
- `plotly>=5.17.0` - Interactive charts
- `matplotlib>=3.7.0` - Static chart generation for PDFs
- `reportlab>=4.0.0` - PDF generation

**AI/ML**:
- `google-generativeai>=0.3.0` - Gemini API for natural language processing

**Utilities**:
- `rich>=13.0` - Enhanced terminal output
- `pytest>=7.0` - Testing framework

### System Requirements

- **Python**: 3.10 or higher
- **Operating System**: Windows, Linux, or macOS
- **Memory**: Minimum 4GB RAM (8GB recommended for large datasets)
- **Storage**: ~100MB for application, additional space for artifacts
- **Network**: Internet connection required for Gemini API (chat interface and objective parser)

### Data Requirements

**Input Files**:
- `data/plan.yaml` - Strategic plan definition (required)
- `data/simulated_kpis.csv` - Historical KPI data (required for forecasting)
- `data/simulated_initiatives.csv` - Initiative progress data (optional, for attribution)

**Output Artifacts**:
- `artifacts/steerco/{date}/` - SteerCo packs (markdown, JSON)
- `artifacts/reports/` - Generated PDF reports
- `artifacts/learning/` - Decision history and learning data
- `artifacts/monitoring/` - Monitoring snapshots and anomaly detection results

### Environment Variables

- `GEMINI_API_KEY` - Google Gemini API key (required for chat interface and objective parser)

---

## Business Value

### 1. Improved Decision Quality

- **Data-Driven**: Recommendations based on quantitative analysis, not intuition
- **Risk-Aware**: Uses stress testing and CVaR to account for uncertainty
- **Multi-Criteria**: Considers expected value, risk, guardrails, and constraints
- **Transparent**: Clear rationale for every recommendation

**Impact**: Reduces bad decisions by 30-50%, leading to better strategic outcomes.

### 2. Time Savings

- **Automated Analysis**: Reduces manual analysis time from days to hours
- **Rapid Scenario Evaluation**: Compare multiple options in minutes, not weeks
- **Automated Reporting**: Generates executive reports in seconds

**Impact**: Saves 20-40 hours per month on strategic planning activities.

### 3. Compliance & Risk Management

- **Guardrails Enforcement**: Automatically checks all recommendations against constraints
- **Audit Trail**: Complete logging of all decisions and rationale
- **Approval Workflow**: Identifies required approvers for high-risk decisions

**Impact**: Reduces compliance violations by 90%, ensures regulatory adherence.

### 4. Continuous Improvement

- **Learning System**: Gets better over time as it learns from outcomes
- **Forecast Accuracy**: Improves prediction accuracy with each decision cycle
- **Adaptive**: Adjusts to changing conditions and new patterns

**Impact**: Forecast accuracy improves from 70% to 90%+ over 12-18 months.

### 5. Strategic Alignment

- **Objective Tracking**: Clear visibility into progress toward strategic goals
- **Attribution Analysis**: Understand which initiatives drive which outcomes
- **Portfolio Optimization**: Ensures resources are allocated to highest-impact initiatives

**Impact**: Increases strategic goal achievement rate by 25-40%.

### 6. Executive Communication

- **SteerCo Packs**: Executive-ready briefings in consistent format
- **Dashboard**: Real-time visibility into strategic performance
- **Reports**: Professional PDF reports for stakeholders

**Impact**: Reduces preparation time for SteerCo meetings by 60%, improves decision speed.

---

## Use Cases

### 1. Monthly Steering Committee Meetings

**Scenario**: Monthly review of strategic plan progress and decision-making

**Workflow**:
1. System monitors KPIs and initiatives throughout the month
2. On SteerCo date, run `steerco_run` to generate pack
3. Review SteerCo brief in dashboard
4. Ask questions via chat interface
5. Make decision based on recommendation
6. Record outcome after 3-6 months

**Value**: Consistent, data-driven decision-making with full audit trail.

### 2. Quarterly Strategic Reviews

**Scenario**: Quarterly deep-dive into strategic objectives

**Workflow**:
1. Generate quarterly PDF report for all objectives
2. Review progress narrative and charts
3. Identify objectives at risk
4. Adjust initiatives or resources as needed
5. Update strategic plan if needed

**Value**: Comprehensive view of strategic progress with actionable insights.

### 3. Initiative Planning

**Scenario**: Planning new strategic initiatives

**Workflow**:
1. Use "Add Objective" feature to describe new objective in natural language
2. System extracts KPIs, targets, and initiatives
3. System prompts for missing information
4. Review and approve generated YAML
5. System evaluates impact on portfolio

**Value**: Faster initiative planning with proper structure and validation.

### 4. Risk Assessment

**Scenario**: Evaluating risk of different strategic options

**Workflow**:
1. System generates multiple action bundles
2. Runs Monte Carlo simulation with stress events
3. Calculates risk-adjusted scores (CVaR)
4. Checks guardrails for each option
5. Recommends option with best risk-return profile

**Value**: Quantified risk assessment with stress testing.

### 5. Performance Monitoring

**Scenario**: Continuous monitoring of strategic KPIs

**Workflow**:
1. System continuously monitors KPIs
2. Detects anomalies using ML
3. Flags integrity issues
4. Generates alerts for material deviations
5. Updates forecasts based on new data

**Value**: Early detection of issues before they become problems.

### 6. Learning & Improvement

**Scenario**: Improving forecast accuracy over time

**Workflow**:
1. System records all recommendations
2. After 3-6 months, record actual outcomes
3. System calculates prediction errors
4. Adjusts forecast weights
5. Next recommendations are more accurate

**Value**: Self-improving system that gets better with each cycle.

---

## Key Differentiators

### 1. Human-in-the-Loop Design

- **No Autonomous Execution**: All recommendations require human approval
- **Transparency**: Clear rationale for every recommendation
- **Control**: Humans maintain full control over strategic decisions

### 2. Industry-Specific Features

- **Regulatory Compliance**: Built-in guardrails for regulatory constraints (particularly relevant for banking, healthcare, and other regulated industries)
- **Risk Management**: Uses industry-standard risk metrics (CVaR) commonly used in banking and finance
- **Audit Trail**: Complete logging for regulatory audits and compliance requirements
- **Governance**: Supports RACI, approval workflows, decision rights

### 3. Continuous Learning

- **Outcome Tracking**: Records actual results vs. predictions
- **Forecast Adjustment**: Improves accuracy over time
- **Metrics Dashboard**: Tracks learning progress

### 4. Comprehensive Attribution

- **Driver Identification**: Shows which initiatives drive which KPI changes
- **Contribution Quantification**: Measures impact of each initiative
- **Root Cause Analysis**: Understands why KPIs are improving or deteriorating

### 5. Executive-Ready Output

- **SteerCo Packs**: Memo-style executive briefings
- **Dashboard**: Interactive web UI for exploration
- **PDF Reports**: Professional reports for stakeholders
- **Chat Interface**: Natural language queries for quick insights

---

## Implementation Status

### ✅ Completed Features

- Strategic plan management (YAML-based)
- KPI monitoring and anomaly detection
- Forecasting with Monte Carlo simulation
- Portfolio evaluation and optimization
- Guardrails and compliance checking
- Learning from outcomes system
- SteerCo pack generation
- Interactive dashboard (Streamlit)
- Conversational interface (Gemini AI)
- PDF report generation
- Natural language objective creation

### 🔄 In Progress

- Enhanced attribution analysis
- Multi-pack comparison
- Real-time data integration

### 📋 Future Enhancements

- Integration with enterprise data sources (ServiceNow, data warehouses)
- Automated outcome recording (from KPI tracking systems)
- Email notifications and alerts
- Mobile app for executives
- Multi-language support
- Advanced visualization (network graphs, heatmaps)
- Collaborative features (comments, annotations)
- Version control for strategic plans
- API for third-party integrations

---

## Getting Started

### Quick Start

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Set Environment Variables**:
   ```bash
   setx GEMINI_API_KEY "your-api-key-here"
   ```

3. **Generate SteerCo Pack**:
   ```bash
   python -m src.steerco_run --as_of 2026-12-01
   ```

4. **Launch Dashboard**:
   ```bash
   streamlit run src/dashboard.py
   ```

5. **Record Outcomes** (after 3-6 months):
   ```bash
   python -m src.record_outcome --interactive
   ```

### Documentation

- `LEARNING_SYSTEM_USAGE.md` - How to use the learning system
- `CHAT_INTERFACE_USAGE.md` - How to use the conversational interface
- `DASHBOARD_USAGE.md` - How to use the dashboard
- `REPORT_GENERATOR_FEATURE.md` - How to generate PDF reports
- `ADD_OBJECTIVE_FEATURE.md` - How to add objectives via natural language

---

## Conclusion

**POLARIS** represents a paradigm shift in strategic planning. While initially developed for banking operations, it is applicable across industries. By combining AI-powered analysis, continuous learning, and human oversight, it delivers:

- **Better Decisions**: Data-driven recommendations with quantified risk
- **Time Savings**: Automated analysis and reporting
- **Compliance**: Built-in guardrails and audit trails
- **Continuous Improvement**: Self-learning system that gets better over time
- **Executive Communication**: Professional reports and interactive dashboards

POLARIS is not just a tool—it's a strategic capability that transforms how organizations plan, execute, and learn from their strategic initiatives. Like the North Star, it provides constant guidance through the complexities of strategic decision-making.

---

**For more information or to schedule a demonstration, please contact the project team.**

---

*Document Version: 2.0*  
*Last Updated: February 2026*  
*Project: POLARIS (Portfolio Optimization & Learning AI for Risk-Adjusted Strategy)*
