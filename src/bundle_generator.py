# src/bundle_generator.py
"""
LLM-powered action bundle generator.

Uses Gemini API to intelligently generate action bundles based on:
- Current progress (ahead/behind schedule)
- Detected issues (anomalies, integrity, dependencies)
- Strategic objectives at risk
- Available initiatives and their health
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

from .actions import SteeringAction, ActionType
from .portfolio import ActionBundle
from .schemas import Plan
from .monitor import MonitorSnapshot
from .steering import InitiativeHealth


def _get_available_model(preferred_model: str = "gemini-pro") -> Optional[str]:
    """
    Get an available Gemini model that supports generateContent.
    Tries preferred model first, then falls back to other available models.
    """
    try:
        # List available models
        models = genai.list_models()
        available_models = []
        
        # Collect all models that support generateContent
        for m in models:
            try:
                model_name = m.name if hasattr(m, 'name') else str(m)
                # Check if it supports generateContent
                if hasattr(m, 'supported_generation_methods'):
                    if 'generateContent' in m.supported_generation_methods:
                        # Extract just the model name part
                        if '/' in model_name:
                            clean_name = model_name.split('/')[-1]
                        else:
                            clean_name = model_name
                        available_models.append(clean_name)
            except Exception:
                continue
        
        # Try preferred model first
        if preferred_model in available_models:
            return preferred_model
        
        # Try other common names in order
        for name in ["gemini-1.5-pro", "gemini-1.5-flash", "gemini-pro"]:
            if name in available_models:
                return name
        
        # Return first available if any found
        if available_models:
            return available_models[0]
        
        return None
    except Exception as e:
        # If listing models fails, try common model names directly
        fallback_models = ["gemini-1.5-pro", "gemini-pro", "gemini-1.5-flash"]
        for model_name in fallback_models:
            try:
                # Test if model works by creating instance (don't actually use it)
                test_model = genai.GenerativeModel(model_name)
                # If no exception, model exists
                return model_name
            except Exception:
                continue
        return None


def generate_action_bundles_with_llm(
    plan: Plan,
    monitor_snapshot: MonitorSnapshot,
    initiative_health: Dict[str, InitiativeHealth],
    ml_anomalies: Optional[Dict[str, Any]] = None,
    integrity_scores: Optional[Dict[str, float]] = None,
    kpi_history: Optional[Any] = None,
    initiative_history: Optional[Any] = None,
    as_of: str = "",
    api_key: Optional[str] = None,
    model: str = "gemini-pro",
) -> List[ActionBundle]:
    """
    Use LLM to intelligently generate action bundles based on current strategic situation.
    
    Args:
        plan: Strategic plan
        monitor_snapshot: Current monitoring snapshot (progress, drift, etc.)
        initiative_health: Health status of each initiative
        ml_anomalies: ML-detected anomalies (optional)
        integrity_scores: KPI integrity scores (optional)
        kpi_history: Historical KPI data (optional, for context)
        initiative_history: Historical initiative data (optional, for context)
        as_of: Current date
        api_key: Gemini API key (if None, uses GEMINI_API_KEY env var)
        model: Gemini model to use
    
    Returns:
        List of ActionBundle objects
    """
    if not GEMINI_AVAILABLE:
        raise ImportError("google-generativeai not installed. Install with: pip install google-generativeai")
    
    api_key = api_key or os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("Gemini API key not found. Set GEMINI_API_KEY environment variable.")
    
    try:
        genai.configure(api_key=api_key)
        
        # Get available models and find one that supports generateContent
        # Must be called after genai.configure()
        available_model = _get_available_model(model)
        if not available_model:
            # Last resort: try common model names directly
            for fallback in ["gemini-1.5-pro", "gemini-pro", "gemini-1.5-flash"]:
                try:
                    test = genai.GenerativeModel(fallback)
                    available_model = fallback
                    break
                except Exception:
                    continue
            
            if not available_model:
                # Try to list available models for better error message
                try:
                    models = genai.list_models()
                    available_list = []
                    for m in models:
                        try:
                            if hasattr(m, 'supported_generation_methods'):
                                if 'generateContent' in m.supported_generation_methods:
                                    name = m.name if hasattr(m, 'name') else str(m)
                                    available_list.append(name)
                        except Exception:
                            continue
                    if available_list:
                        raise RuntimeError(
                            f"No compatible Gemini model found. Available models: {', '.join(available_list[:5])}. "
                            "Check your API key and model access."
                        )
                except Exception:
                    pass
                raise RuntimeError(
                    "No available Gemini model found. Check your API key and model access. "
                    "Make sure your API key is valid and has access to Gemini models."
                )
        
        # Build comprehensive context
        context = _build_analysis_context(
            plan=plan,
            monitor_snapshot=monitor_snapshot,
            initiative_health=initiative_health,
            ml_anomalies=ml_anomalies,
            integrity_scores=integrity_scores,
            as_of=as_of
        )
        
        # Build prompt
        prompt = _build_bundle_generation_prompt(context, plan)
        
        # Call Gemini API
        model_instance = genai.GenerativeModel(available_model)
        response = model_instance.generate_content(prompt)
        response_text = response.text.strip()
        
        # Parse and validate response
        bundles = _parse_llm_response(response_text, plan)
        
        return bundles
    
    except Exception as e:
        raise RuntimeError(f"Error generating bundles with LLM: {str(e)}") from e


def _build_analysis_context(
    plan: Plan,
    monitor_snapshot: MonitorSnapshot,
    initiative_health: Dict[str, InitiativeHealth],
    ml_anomalies: Optional[Dict[str, Any]],
    integrity_scores: Optional[Dict[str, float]],
    as_of: str,
) -> str:
    """Build comprehensive context string for LLM analysis."""
    context_parts = []
    
    # Strategic objectives and KPIs
    context_parts.append("## STRATEGIC OBJECTIVES")
    for obj_status in monitor_snapshot.objective_statuses:
        obj = next((o for o in plan.objectives if o.id == obj_status.objective_id), None)
        if not obj:
            continue
        
        on_track = "✓ ON TRACK" if obj_status.on_track else "⚠ BEHIND SCHEDULE"
        context_parts.append(f"\n{obj.id}: {obj.name} (weight: {obj.weight}) - {on_track}")
        context_parts.append(f"  Drift confidence: {obj_status.drift_confidence:.2f}")
        
        for kpi_status in obj_status.kpi_statuses:
            kpi = next((k for k in plan.kpis if k.id == kpi_status.kpi_id), None)
            if not kpi:
                continue
            
            if kpi_status.actual is not None and kpi_status.expected is not None:
                drift = kpi_status.drift or 0.0
                direction = "↑" if drift < 0 else "↓" if drift > 0 else "→"
                context_parts.append(
                    f"  - {kpi.short_name}: actual={kpi_status.actual:.2f}, "
                    f"expected={kpi_status.expected:.2f} {direction} (drift: {drift:.2f})"
                )
    
    # Initiatives and health
    context_parts.append("\n## INITIATIVES STATUS")
    for init in plan.initiatives:
        health = initiative_health.get(init.id)
        if not health:
            continue
        
        status_icon = {"good": "✓", "watch": "⚠", "bad": "✗"}.get(health.health_flag, "?")
        context_parts.append(
            f"\n{init.id}: {init.name} {status_icon}"
        )
        context_parts.append(f"  Progress: {health.progress:.1%}")
        context_parts.append(f"  Delay: {health.delay_months} months")
        if hasattr(health, 'blocked_by') and health.blocked_by:
            context_parts.append(f"  BLOCKED BY: {', '.join(health.blocked_by)}")
        
        # Show which KPIs this initiative impacts
        kpi_impacts = []
        for impact in init.kpi_impacts:
            kpi = next((k for k in plan.kpis if k.id == impact.kpi_id), None)
            if kpi:
                # Determine impact direction from expected_delta
                direction = "up" if impact.expected_delta_by_end > 0 else "down"
                kpi_impacts.append(f"{kpi.short_name} ({direction})")
        if kpi_impacts:
            context_parts.append(f"  Impacts: {', '.join(kpi_impacts)}")

        # Show concrete initiative decomposition (workstreams + key deliverables)
        if hasattr(init, "workstreams") and init.workstreams:
            context_parts.append("  Workstreams:")
            for ws in init.workstreams:
                context_parts.append(
                    f"    - {ws.id}: {ws.name} | owner={ws.owner} | window={ws.start}..{ws.end} | effort={ws.effort_fte_months:.1f} FTE-months"
                )
                if ws.deliverables:
                    top_delivs = ws.deliverables[:2]
                    deliv_txt = "; ".join([f"{d.id}({d.due}) {d.name}" for d in top_delivs])
                    context_parts.append(f"      deliverables: {deliv_txt}")
    
    # Anomalies
    if ml_anomalies:
        anomalies_found = [kpi_id for kpi_id, data in ml_anomalies.items() if data.get('is_anomaly', False)]
        if anomalies_found:
            context_parts.append("\n## ML ANOMALIES DETECTED")
            for kpi_id in anomalies_found:
                kpi = next((k for k in plan.kpis if k.id == kpi_id), None)
                kpi_name = kpi.short_name if kpi else kpi_id
                anomaly_data = ml_anomalies[kpi_id]
                context_parts.append(
                    f"- {kpi_name} ({kpi_id}): {anomaly_data.get('explanation', 'Anomaly detected')}"
                )
    
    # Integrity issues
    if integrity_scores:
        low_integrity = [kpi_id for kpi_id, score in integrity_scores.items() if score < 0.6]
        if low_integrity:
            context_parts.append("\n## KPI INTEGRITY ISSUES")
            for kpi_id in low_integrity:
                kpi = next((k for k in plan.kpis if k.id == kpi_id), None)
                kpi_name = kpi.short_name if kpi else kpi_id
                context_parts.append(f"- {kpi_name} ({kpi_id}): integrity score = {integrity_scores[kpi_id]:.2f}")
    
    # Objectives requiring attention
    triggered = monitor_snapshot.triggered_for_steerco
    if triggered:
        context_parts.append("\n## OBJECTIVES REQUIRING ATTENTION")
        for obj_id in triggered:
            obj = next((o for o in plan.objectives if o.id == obj_id), None)
            if obj:
                context_parts.append(f"- {obj_id}: {obj.name}")
    
    return "\n".join(context_parts)


def _build_bundle_generation_prompt(context: str, plan: Plan) -> str:
    """Build the prompt for LLM to generate action bundles."""
    
    # List available initiatives
    initiatives_list = []
    for init in plan.initiatives:
        initiatives_list.append(f"- {init.id}: {init.name}")
        if hasattr(init, "workstreams") and init.workstreams:
            for ws in init.workstreams:
                initiatives_list.append(
                    f"    * {ws.id}: {ws.name} ({ws.start}..{ws.end}, owner={ws.owner})"
                )
    
    prompt = f"""You are a strategic planning AI assistant. Analyze the current strategic situation and generate 3-5 action bundles (strategic options) to address issues and improve outcomes.

STRATEGIC PLAN CONTEXT:
{context}

AVAILABLE INITIATIVES:
{chr(10).join(initiatives_list)}

AVAILABLE ACTION TYPES:
1. accelerate_initiative: Speed up an initiative by N months
   - Parameters: {{"months": 2}} (typically 1-3 months)
   - Use when: Initiative is on track but could be faster, or drives critical KPIs that are behind schedule
   - Example: accelerate_initiative on INIT_DQ1 with {{"months": 2}} to unblock dependent initiatives

2. split_scope: Reduce scope of an initiative to unblock dependencies or reduce risk
   - Parameters: {{"reduction": 0.6}} (0.0-1.0, fraction to cut - 0.6 means cut 60%, keep 40%)
   - Use when: Initiative is blocked by dependencies, too large/risky, or needs to deliver value earlier
   - Example: split_scope on INIT_A1 with {{"reduction": 0.6}} to deliver stable flows first

3. add_capacity: Add resources to an initiative
   - Parameters: {{"capacity_gain": 0.06}} (0.0-1.0, typically 0.03-0.10)
   - Use when: Initiative is behind schedule and needs more capacity
   - Example: add_capacity on INIT_RES1 with {{"capacity_gain": 0.08}} to accelerate resilience work

CURRENT SITUATION ANALYSIS:
- Objectives behind schedule need recovery actions
- Blocked initiatives need dependency resolution (accelerate dependencies or split scope)
- Anomalies need investigation or corrective actions
- Low integrity KPIs need data quality fixes

REQUIREMENTS:
- Generate 3-5 distinct action bundles (strategic options)
- Each bundle should have a clear strategic theme (e.g., "Risk-first", "Growth", "Recovery", "Stability")
- Each bundle should contain 1-3 actions that work together coherently
- Consider dependencies: don't accelerate initiatives that are blocked by dependencies (accelerate the blocker first, or split scope)
- Consider guardrails: ensure bundles respect budget, capacity, and compliance constraints
- Prioritize actions that address the most critical issues (behind-schedule KPIs, anomalies, blockers)
- Each action must reference a valid initiative ID from the available initiatives list
- Prefer targeting concrete workstreams when possible. If using workstream targeting, include `target_workstream`.

OUTPUT FORMAT (JSON only, no markdown):
{{
  "bundles": [
    {{
      "id": "OPT_A",
      "name": "Option A — [Theme]",
      "description": "Clear description of the strategic approach and why it helps",
      "actions": [
        {{
          "id": "A_ACCEL_DQ1_2M",
          "type": "accelerate_initiative",
          "target_initiative": "INIT_DQ1",
          "target_workstream": "WS_DQ1_INGEST", 
          "parameters": {{"months": 2}},
          "description": "Why this action helps address the current situation"
        }}
      ]
    }}
  ]
}}

Generate the JSON now:"""
    
    return prompt


def _parse_llm_response(response_text: str, plan: Plan) -> List[ActionBundle]:
    """Parse LLM response and convert to ActionBundle objects."""
    
    # Try to extract JSON from response (might be wrapped in markdown code blocks)
    json_text = response_text
    
    # Remove markdown code blocks if present
    if "```json" in json_text:
        json_text = json_text.split("```json")[1].split("```")[0].strip()
    elif "```" in json_text:
        json_text = json_text.split("```")[1].split("```")[0].strip()
    
    # Parse JSON
    try:
        data = json.loads(json_text)
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse LLM response as JSON: {e}\nResponse: {response_text[:500]}")
    
    if "bundles" not in data:
        raise ValueError(f"LLM response missing 'bundles' key. Response: {json.dumps(data, indent=2)}")
    
    bundles = []
    valid_initiative_ids = {init.id for init in plan.initiatives}
    valid_workstream_ids_by_init = {
        init.id: {ws.id for ws in getattr(init, "workstreams", [])}
        for init in plan.initiatives
    }
    valid_action_types = {"accelerate_initiative", "split_scope", "add_capacity"}
    
    for bundle_data in data["bundles"]:
        # Validate bundle structure
        if "id" not in bundle_data or "name" not in bundle_data or "actions" not in bundle_data:
            continue
        
        bundle_id = bundle_data["id"]
        bundle_name = bundle_data["name"]
        bundle_description = bundle_data.get("description", "")
        actions_data = bundle_data["actions"]
        
        # Parse actions
        actions = []
        for action_data in actions_data:
            if "type" not in action_data or "target_initiative" not in action_data:
                continue
            
            action_type = action_data["type"]
            target_initiative = action_data["target_initiative"]
            
            # Validate action type
            if action_type not in valid_action_types:
                continue
            
            # Validate initiative exists
            if target_initiative not in valid_initiative_ids:
                continue
            
            # Get parameters
            parameters = action_data.get("parameters", {})
            if not isinstance(parameters, dict):
                parameters = {}

            # Optional concrete targeting at workstream level
            target_workstream = action_data.get("target_workstream")
            if target_workstream:
                ws_ids = valid_workstream_ids_by_init.get(target_initiative, set())
                if str(target_workstream) in ws_ids:
                    parameters["workstream_id"] = str(target_workstream)
            
            # Validate parameters based on action type
            if action_type == "accelerate_initiative":
                if "months" not in parameters:
                    parameters["months"] = 2  # Default
                parameters["months"] = max(1, min(6, int(parameters["months"])))  # Clamp 1-6 months
            elif action_type == "split_scope":
                if "reduction" not in parameters:
                    parameters["reduction"] = 0.6  # Default
                parameters["reduction"] = max(0.1, min(0.9, float(parameters["reduction"])))  # Clamp 0.1-0.9
            elif action_type == "add_capacity":
                if "capacity_gain" not in parameters:
                    parameters["capacity_gain"] = 0.06  # Default
                parameters["capacity_gain"] = max(0.01, min(0.2, float(parameters["capacity_gain"])))  # Clamp 0.01-0.2
            
            action_id = action_data.get("id", f"{bundle_id}_{action_type}_{target_initiative}")
            action_description = action_data.get("description", f"{action_type} on {target_initiative}")
            
            try:
                action = SteeringAction(
                    id=action_id,
                    type=action_type,  # type: ignore
                    target_initiative=target_initiative,
                    parameters=parameters,
                    description=action_description
                )
                actions.append(action)
            except Exception as e:
                # Skip invalid actions
                continue
        
        # Only create bundle if it has at least one valid action
        if actions:
            bundle = ActionBundle(
                id=bundle_id,
                name=bundle_name,
                description=bundle_description,
                actions=actions
            )
            bundles.append(bundle)
    
    if not bundles:
        raise ValueError("LLM generated no valid bundles. Check that initiatives exist and actions are valid.")
    
    return bundles
