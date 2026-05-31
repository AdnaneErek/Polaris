# AI Enhancement: Gemini API Narrative Generation

## Overview

Gemini API-based narrative generation as an AI-powered alternative to template-based narratives.

## Features

### 1. Automatic Detection
- If `GEMINI_API_KEY` is set: uses Gemini API
- If not set or API fails: falls back to template-based generator

### 2. Smart Context Building
Uses:
- KPI impacts (mean, p10, p90 deltas)
- Top drivers and contribution percentages
- Downside risks (p10 vs mean)
- Guardrail violations and approvals
- Portfolio scores

### 3. Professional Output
- Executive-style narrative
- 3–4 concise paragraphs
- Markdown format

## Setup

### Install dependency
```bash
pip install google-generativeai
```

### Set API key
```bash
# PowerShell
$env:GEMINI_API_KEY = "your-api-key-here"
```

## Usage

### Automatic mode
```bash
python -m src.steerco_run
```

### Manual mode
```python
from src.narrative_gemini import generate_bundle_narrative_gemini
narrative = generate_bundle_narrative_gemini(...)
```

## Model
Default: `gemini-1.5-flash`

## Fallback Behavior
If Gemini is unavailable/fails, template-based narratives are generated automatically.

## Modified Files
- `src/narrative_gemini.py`
- `src/steerco_pack.py`
- `requirements.txt`

## Benefits
- Natural language quality
- Context-aware summaries
- Graceful fallback safety
