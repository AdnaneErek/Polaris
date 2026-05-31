# Conversational Interface - Usage Guide

## Overview

The conversational interface allows you to ask natural language questions about your strategic plan using Gemini AI. It understands:
- Recommendations and rationale
- KPI forecasts and trends
- Portfolio options and guardrails
- Learning metrics
- Anomalies and integrity issues

## Prerequisites

1. **Gemini API Key**: Set the `GEMINI_API_KEY` environment variable, or pass it via `--api-key`
2. **SteerCo Pack**: You need a generated `steerco_pack.json` file (from running `steerco_run`)

## Usage

### Interactive Chat Mode

Start an interactive chat session:

```bash
python -m src.chat_cli --pack artifacts/steerco/2026-12-01/steerco_pack.json
```

This will start a chat session where you can ask multiple questions. Type `quit` or `exit` to end.

**Example session:**
```
You: Why was Option C recommended?

Assistant: Option C (Risk-first - Resilience only) was recommended because it ranked #1 
by stress CVaR10 (risk-averse metric) with a score of 0.50. It has a confidence of 0.50, 
which indicates the top options were statistically tied. The recommendation prioritizes 
resilience improvements, specifically reducing incident count (KPI_INC) by -2.21, which 
aligns with risk-first strategy. Guardrails passed, making it a safe choice.

You: What are the main risks?

Assistant: Based on the guardrails evaluation, Option C passed all checks, indicating 
low immediate risks. However, the low confidence (0.50) suggests the decision is not 
highly discriminative - meaning Option A was also very close. The main consideration 
is that this option focuses only on resilience, potentially missing opportunities for 
growth or governance improvements that other options might provide.

You: How accurate have our forecasts been?

Assistant: According to the learning metrics, there are currently 2 total decisions 
recorded, but no outcomes have been recorded yet. This means the system hasn't learned 
from actual results yet. Once you record outcomes using the record_outcome tool, the 
system will track forecast accuracy and show improvement trends.
```

### Single Question Mode

Ask a single question without entering interactive mode:

```bash
python -m src.chat_cli \
    --pack artifacts/steerco/2026-12-01/steerco_pack.json \
    --question "Why was Option C recommended?"
```

### With Custom API Key

If you don't have `GEMINI_API_KEY` set as an environment variable:

```bash
python -m src.chat_cli \
    --pack artifacts/steerco/2026-12-01/steerco_pack.json \
    --api-key YOUR_API_KEY_HERE
```

## Example Questions

### About Recommendations
- "Why was Option C recommended?"
- "What are the alternatives to the recommended option?"
- "What is the confidence level and why is it low?"

### About KPIs
- "What are the KPI forecasts for the next 3 months?"
- "Which KPIs are showing anomalies?"
- "What is the expected impact on STP rate?"

### About Options
- "Compare Option A and Option C"
- "What are the guardrail violations for Option B?"
- "Which option has the best risk-adjusted score?"

### About Learning
- "How accurate have our forecasts been?"
- "What is the improvement trend?"
- "How many decisions have been recorded?"

### About Anomalies
- "What anomalies were detected?"
- "Why was KPI_E2E flagged as anomalous?"
- "What integrity issues exist?"

## Integration with steerco_run

After running `steerco_run`, the system automatically suggests using the chat interface:

```bash
python -m src.steerco_run --as_of 2026-12-01

# Output includes:
# 💬 Chat Interface Available
# Ask questions about this pack using:
#   python -m src.chat_cli --pack artifacts/steerco/2026-12-01/steerco_pack.json
```

## How It Works

1. **Context Building**: The system extracts key information from the SteerCo pack:
   - Recommendation details
   - KPI forecasts
   - Portfolio options
   - Guardrails status
   - Learning metrics
   - Anomalies

2. **Prompt Construction**: Builds a detailed prompt with context and your question

3. **Gemini API**: Sends to Gemini AI for natural language understanding and response

4. **Response**: Returns a human-readable answer based on the pack data

## Tips

1. **Be specific**: More specific questions get better answers
   - Good: "Why was Option C recommended over Option A?"
   - Less clear: "What's the best option?"

2. **Ask follow-ups**: The interactive mode remembers context within a session

3. **Check the pack first**: For detailed data, check the markdown/JSON files. Chat is best for quick insights.

4. **Use for explanations**: Great for understanding "why" behind recommendations

## Limitations

- **Context window**: Only includes summary information, not full detailed data
- **API dependency**: Requires Gemini API key and internet connection
- **Cost**: Uses Gemini API (free tier available, but check usage limits)
- **Accuracy**: Responses are based on pack data - verify critical decisions independently

## Future Enhancements

Potential improvements:
- Multi-turn conversation with memory
- Integration with full pack data (not just summaries)
- Support for multiple packs (compare across time periods)
- Export conversation history
- Web UI for easier access
