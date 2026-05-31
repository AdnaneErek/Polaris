# Add Objective Feature

## Overview

A dashboard tab that allows users to add strategic objectives using natural language. The system uses Gemini AI to parse text and produce schema-compatible YAML for `plan.yaml`.

## Features

1. **Natural Language Input**
2. **AI-Powered Parsing**
3. **Interactive Clarification**
4. **Validation** against plan schema
5. **YAML Generation**
6. **Editable Output** before commit
7. **Conversation History**

## How to Use

1. Open dashboard: `python -m streamlit run src/dashboard.py`
2. Go to **Add Objective** tab
3. Describe objective in natural language
4. Review AI-extracted fields
5. Answer clarifying questions if prompted
6. Review/edit generated YAML
7. Add to `data/plan.yaml`

## Example Input

```
I want to improve customer satisfaction by reducing response time from 4 hours 
to under 2 hours by 2028. The current response time is 4 hours. This should 
be measured using the KPI_E2E metric. The weight should be 0.30.
```

## Technical Details

### Files
- `src/objective_parser.py`
- `src/dashboard.py` (Add Objective tab)

### Required Environment Variable
- `GEMINI_API_KEY`

## Validation Rules
- Required fields present
- Weight in `[0.0, 1.0]`
- Valid KPI references
- Direction compatibility
- Objective/OKR ID format checks

## Future Enhancements
- Direct `plan.yaml` update workflow
- Initiative and KPI creation assistants
- Bulk objective parsing
