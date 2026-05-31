# src/objective_parser.py
"""
LLM-powered parser for converting natural language objective descriptions
into structured YAML format for plan.yaml.

Uses Gemini API to:
- Extract structured fields (id, name, description, weight, OKRs, constraints)
- Validate against schema
- Ask clarifying questions if information is missing
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Tuple

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False


def parse_objective_from_text(
    text: str,
    plan: Any,
    api_key: Optional[str] = None,
    conversation_history: Optional[List[Dict[str, str]]] = None
) -> Tuple[Dict[str, Any], Optional[str], List[str]]:
    """
    Parse natural language objective description into structured YAML format.
    
    Args:
        text: Natural language description of the objective
        plan: The Plan object (to get available KPIs, existing objectives, etc.)
        api_key: Gemini API key (if None, uses GEMINI_API_KEY env var)
        conversation_history: Previous messages in the conversation (for follow-up questions)
    
    Returns:
        Tuple of:
        - parsed_objective: Dict with structured objective data (or None if incomplete)
        - clarifying_question: Optional question to ask the user if info is missing
        - validation_errors: List of validation error messages
    """
    if not GEMINI_AVAILABLE:
        return None, "Gemini API not available. Install with: pip install google-generativeai", []
    
    api_key = api_key or os.getenv("GEMINI_API_KEY")
    if not api_key:
        return None, "Gemini API key not found. Set GEMINI_API_KEY environment variable.", []
    
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-pro')
    except Exception as e:
        return None, f"Error configuring Gemini: {str(e)}", []
    
    # Build context about the plan
    context = _build_plan_context(plan)
    
    # Build prompt
    prompt = _build_parsing_prompt(text, context, conversation_history)
    
    try:
        response = model.generate_content(prompt)
        response_text = response.text.strip()
        
        # Parse the response
        return _parse_llm_response(response_text, plan)
    except Exception as e:
        return None, f"Error calling Gemini API: {str(e)}", []


def _build_plan_context(plan: Any) -> str:
    """Build context string about the plan for the LLM."""
    context_parts = []
    
    # Available KPIs
    context_parts.append("## AVAILABLE KPIs")
    for kpi in plan.kpis:
        context_parts.append(f"- {kpi.id}: {kpi.name} ({kpi.short_name}) - {kpi.unit}")
        context_parts.append(f"  Baseline: {kpi.baseline.value} (as of {kpi.baseline.date})")
        context_parts.append(f"  Target: {kpi.target.value} (by {kpi.target.deadline})")
        context_parts.append(f"  Direction: {kpi.target.direction}")
    context_parts.append("")
    
    # Existing objectives (for ID generation)
    context_parts.append("## EXISTING OBJECTIVES")
    for obj in plan.objectives:
        context_parts.append(f"- {obj.id}: {obj.name} (weight: {obj.weight})")
        for okr in obj.okrs:
            context_parts.append(f"  OKR {okr.id}: {okr.kpi_id} from {okr.baseline} to {okr.target} by {okr.deadline}")
    context_parts.append("")
    
    # Objective schema requirements
    context_parts.append("## REQUIRED FIELDS FOR OBJECTIVE")
    context_parts.append("""
An objective must have:
1. id: Unique identifier (format: OBJ_N where N is next number, e.g., OBJ_4)
2. name: Short descriptive name
3. description: Detailed description
4. weight: Float between 0.0 and 1.0 (should sum with other objectives to ~1.0)
5. okrs: List of OKRs, each with:
   - id: Unique OKR ID (format: OKR_X_Y where X is objective number, Y is OKR number)
   - kpi_id: Must reference an existing KPI from the list above
   - baseline: Current value (float)
   - target: Target value (float)
   - deadline: Date string (YYYY-MM-DD)
   - direction: "up" or "down" (must match KPI direction)
   - trajectory: Object with:
     - type: "linear", "piecewise", or "s-curve"
     - checkpoints: List of {date: "YYYY-MM-DD", expected: float}
6. constraints: (Optional) List of constraint objects with:
   - type: "compliance", "control_coverage", "availability", "regulatory", etc.
   - description: String describing the constraint
""")
    
    return "\n".join(context_parts)


def _build_parsing_prompt(
    text: str,
    context: str,
    conversation_history: Optional[List[Dict[str, str]]] = None
) -> str:
    """Build the prompt for the LLM."""
    prompt_parts = []
    
    prompt_parts.append("""You are a strategic planning assistant. Your task is to parse a natural language description of a strategic objective and convert it into structured YAML format.

You must extract all required fields. If any required information is missing, you should ask a clarifying question instead of guessing.

Respond in JSON format with this structure:
{
  "status": "complete" | "needs_clarification",
  "objective": { ... structured objective data ... } | null,
  "clarifying_question": "question text" | null,
  "validation_errors": ["error1", "error2", ...]
}

If status is "complete", provide the full objective structure.
If status is "needs_clarification", provide a clear, specific question asking for the missing information.
""")
    
    prompt_parts.append("\n## PLAN CONTEXT\n")
    prompt_parts.append(context)
    
    if conversation_history:
        prompt_parts.append("\n## CONVERSATION HISTORY\n")
        for msg in conversation_history[-5:]:  # Last 5 messages
            prompt_parts.append(f"{msg['role']}: {msg['content']}")
    
    prompt_parts.append("\n## USER INPUT\n")
    prompt_parts.append(text)
    
    prompt_parts.append("""
\n## INSTRUCTIONS
1. Extract all required fields from the user input
2. Generate appropriate IDs (OBJ_N for objectives, OKR_X_Y for OKRs)
3. Ensure KPI references match existing KPIs exactly
4. Ensure direction matches the KPI's direction
5. If information is missing, ask ONE specific clarifying question
6. Return valid JSON only, no markdown formatting
""")
    
    return "\n".join(prompt_parts)


def _parse_llm_response(response_text: str, plan: Any) -> Tuple[Dict[str, Any], Optional[str], List[str]]:
    """Parse the LLM response and validate it."""
    # Try to extract JSON from the response
    json_text = response_text
    
    # Remove markdown code blocks if present
    if "```json" in json_text:
        json_text = json_text.split("```json")[1].split("```")[0].strip()
    elif "```" in json_text:
        json_text = json_text.split("```")[1].split("```")[0].strip()
    
    try:
        result = json.loads(json_text)
    except json.JSONDecodeError:
        # Try to find JSON object in the text
        start_idx = json_text.find("{")
        end_idx = json_text.rfind("}")
        if start_idx >= 0 and end_idx > start_idx:
            try:
                result = json.loads(json_text[start_idx:end_idx+1])
            except:
                return None, "Could not parse LLM response as JSON. Please try rephrasing your objective.", []
        else:
            return None, "LLM response was not in valid JSON format. Please try again.", []
    
    status = result.get("status", "unknown")
    objective_data = result.get("objective")
    clarifying_question = result.get("clarifying_question")
    validation_errors = result.get("validation_errors", [])
    
    if status == "needs_clarification":
        return None, clarifying_question, validation_errors
    
    if status == "complete" and objective_data:
        # Validate the objective data
        validation_errors = _validate_objective_data(objective_data, plan)
        if validation_errors:
            return None, "The parsed objective has validation errors. Please review and correct.", validation_errors
        
        return objective_data, None, []
    
    return None, "Unexpected response format from LLM. Please try again.", []


def _validate_objective_data(objective_data: Dict[str, Any], plan: Any) -> List[str]:
    """Validate the parsed objective data against the schema."""
    errors = []
    
    # Check required fields
    required_fields = ["id", "name", "description", "weight", "okrs"]
    for field in required_fields:
        if field not in objective_data:
            errors.append(f"Missing required field: {field}")
    
    # Validate weight
    if "weight" in objective_data:
        weight = objective_data["weight"]
        if not isinstance(weight, (int, float)) or weight < 0 or weight > 1:
            errors.append(f"Invalid weight: {weight} (must be between 0.0 and 1.0)")
    
    # Validate OKRs
    if "okrs" in objective_data:
        okrs = objective_data["okrs"]
        if not isinstance(okrs, list) or len(okrs) == 0:
            errors.append("Must have at least one OKR")
        else:
            kpi_ids = {kpi.id for kpi in plan.kpis}
            for i, okr in enumerate(okrs):
                if "kpi_id" not in okr:
                    errors.append(f"OKR {i+1}: Missing kpi_id")
                elif okr["kpi_id"] not in kpi_ids:
                    errors.append(f"OKR {i+1}: Invalid kpi_id '{okr['kpi_id']}' (not in available KPIs)")
                else:
                    # Check direction matches KPI
                    kpi = next((k for k in plan.kpis if k.id == okr["kpi_id"]), None)
                    if kpi and "direction" in okr:
                        if okr["direction"] != kpi.target.direction:
                            errors.append(f"OKR {i+1}: Direction '{okr['direction']}' doesn't match KPI direction '{kpi.target.direction}'")
    
    # Validate ID format
    if "id" in objective_data:
        obj_id = objective_data["id"]
        if not obj_id.startswith("OBJ_"):
            errors.append(f"Objective ID should start with 'OBJ_' (got: {obj_id})")
    
    return errors


def format_objective_as_yaml(objective_data: Dict[str, Any]) -> str:
    """Format the objective data as YAML for plan.yaml."""
    # Convert to YAML format
    yaml_lines = ["  - id: " + objective_data["id"]]
    yaml_lines.append(f'    name: "{objective_data["name"]}"')
    yaml_lines.append(f'    description: "{objective_data["description"]}"')
    yaml_lines.append(f'    weight: {objective_data["weight"]}')
    
    # OKRs
    yaml_lines.append("    okrs:")
    for okr in objective_data.get("okrs", []):
        yaml_lines.append(f'      - id: {okr["id"]}')
        yaml_lines.append(f'        kpi_id: {okr["kpi_id"]}')
        yaml_lines.append(f'        baseline: {okr["baseline"]}')
        yaml_lines.append(f'        target: {okr["target"]}')
        yaml_lines.append(f'        deadline: "{okr["deadline"]}"')
        yaml_lines.append(f'        direction: "{okr["direction"]}"')
        
        # Trajectory
        if "trajectory" in okr:
            traj = okr["trajectory"]
            yaml_lines.append("        trajectory:")
            yaml_lines.append(f'          type: "{traj.get("type", "linear")}"')
            if "checkpoints" in traj:
                yaml_lines.append("          checkpoints:")
                for cp in traj["checkpoints"]:
                    yaml_lines.append(f'            - date: "{cp["date"]}"')
                    yaml_lines.append(f'              expected: {cp["expected"]}')
    
    # Constraints
    if "constraints" in objective_data and objective_data["constraints"]:
        yaml_lines.append("    constraints:")
        for constraint in objective_data["constraints"]:
            yaml_lines.append(f'      - type: "{constraint["type"]}"')
            yaml_lines.append(f'        description: "{constraint["description"]}"')
    
    return "\n".join(yaml_lines)
