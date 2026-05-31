# src/narrative_gemini.py
"""
Gemini API-based Natural Language Generation for SteerCo narratives.

Uses Google's Gemini API to generate human-like, AI-powered narratives.
Requires GEMINI_API_KEY environment variable.
"""
from __future__ import annotations

import os
import json
from typing import Dict, List, Optional, Any

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

from .attribution import OptionAttribution, KPIAttribution
from .schemas import Plan, KPI
from .portfolio import OptionResultMC
from .guardrails import GuardrailResult


def _get_available_gemini_model() -> Optional[str]:
    """Get an available Gemini model that supports generateContent."""
    try:
        models = genai.list_models()
        available_models = []
        
        # Collect all models that support generateContent
        for m in models:
            try:
                model_name = m.name if hasattr(m, 'name') else str(m)
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
        
        # Try common names in order of preference
        for name in ["gemini-1.5-pro", "gemini-pro", "gemini-1.5-flash"]:
            if name in available_models:
                return name
        
        # Return first available if any found
        if available_models:
            return available_models[0]
        
        return None
    except Exception:
        # Last resort: try common names by testing
        for name in ["gemini-1.5-pro", "gemini-pro", "gemini-1.5-flash"]:
            try:
                genai.GenerativeModel(name)
                return name
            except Exception:
                continue
        return None


def _get_kpi_info(plan: Plan, kpi_id: str) -> tuple[str, str, str]:
    """Get KPI name, unit, and direction."""
    for kpi in plan.kpis:
        if kpi.id == kpi_id or kpi.short_name == kpi_id:
            # Get direction from OKRs
            direction = "up"  # default
            for obj in plan.objectives:
                for okr in obj.okrs:
                    if okr.kpi_id == kpi_id:
                        direction = okr.direction
                        break
            return kpi.short_name, kpi.unit, direction
    return kpi_id, "", "up"


def _format_delta(delta: float, unit: str) -> str:
    """Format delta value with appropriate unit."""
    if unit == "%":
        return f"{delta:+.2f}pp"
    elif unit in ("hours", "h"):
        return f"{delta:+.2f}h"
    else:
        return f"{delta:+.2f}{unit}"


def _get_top_drivers(attribution: OptionAttribution, plan: Plan, top_n: int = 3) -> List[Dict[str, Any]]:
    """Extract top drivers from attribution."""
    driver_contributions: Dict[str, float] = {}
    driver_names: Dict[str, str] = {}
    
    for kpi_id, kpi_attr in attribution.kpi_attributions.items():
        for driver in kpi_attr.drivers:
            if driver.driver_type == "initiative":
                # Find initiative name
                driver_name = driver.driver_id
                for it in plan.initiatives:
                    if it.id == driver.driver_id:
                        driver_name = it.name
                        break
                driver_names[driver.driver_id] = driver_name
            else:
                driver_name = driver.driver_id
                driver_names[driver.driver_id] = driver_name
            
            key = f"{driver.driver_type}:{driver.driver_id}"
            driver_contributions[key] = driver_contributions.get(key, 0.0) + abs(driver.contribution)
    
    # Sort and get top N
    sorted_drivers = sorted(driver_contributions.items(), key=lambda x: x[1], reverse=True)
    
    result = []
    total_contrib = sum(driver_contributions.values())
    
    for key, contrib in sorted_drivers[:top_n]:
        driver_type, driver_id = key.split(":", 1)
        pct = (contrib / total_contrib * 100) if total_contrib > 0 else 0.0
        result.append({
            "name": driver_names.get(driver_id, driver_id),
            "type": driver_type,
            "contribution": contrib,
            "percentage": pct,
        })
    
    return result


def _build_context_for_gemini(
    plan: Plan,
    bundle_id: str,
    bundle_name: str,
    attribution: OptionAttribution,
    option_result: OptionResultMC,
    guardrail_result: Optional[GuardrailResult] = None,
) -> str:
    """Build structured context for Gemini prompt."""
    context = {
        "bundle": {
            "id": bundle_id,
            "name": bundle_name,
        },
        "scores": {
            "stress_mean": option_result.score_stress_mean,
            "stress_cvar10": option_result.score_stress_cvar10,
            "base_mean": option_result.score_base_mean,
            "base_cvar10": option_result.score_base_cvar10,
        },
        "kpi_impacts": {},
        "drivers": [],
        "risks": {},
        "guardrails": {
            "passed": guardrail_result.passed if guardrail_result else True,
            "requires_approval": guardrail_result.requires_approval if guardrail_result else False,
            "violations": [],
        },
    }
    
    # KPI impacts
    for kpi_id, kpi_attr in attribution.kpi_attributions.items():
        kpi_name, unit, kpi_direction = _get_kpi_info(plan, kpi_id)
        mean_delta = option_result.stress_delta_summary.get(kpi_id, {}).get("mean", 0.0)
        p10_delta = option_result.stress_delta_summary.get(kpi_id, {}).get("p10", 0.0)
        p90_delta = option_result.stress_delta_summary.get(kpi_id, {}).get("p90", 0.0)
        
        # Determine if improvement
        if kpi_direction == "down":
            is_improvement = mean_delta < 0
        else:
            is_improvement = mean_delta > 0
        
        context["kpi_impacts"][kpi_name] = {
            "delta": mean_delta,
            "formatted": _format_delta(abs(mean_delta), unit),
            "unit": unit,
            "direction": kpi_direction,
            "is_improvement": is_improvement,
            "p10": p10_delta,
            "p90": p90_delta,
        }
    
    # Drivers
    top_drivers = _get_top_drivers(attribution, plan, top_n=5)
    context["drivers"] = [
        {
            "name": d["name"],
            "contribution_pct": int(d["percentage"]),
        }
        for d in top_drivers
    ]
    
    # Risks (p10 vs mean)
    for kpi_id, kpi_attr in attribution.kpi_attributions.items():
        kpi_name, unit, kpi_direction = _get_kpi_info(plan, kpi_id)
        p10_delta = option_result.stress_delta_summary.get(kpi_id, {}).get("p10", 0.0)
        mean_delta = option_result.stress_delta_summary.get(kpi_id, {}).get("mean", 0.0)
        
        if kpi_direction == "down":
            is_worse = p10_delta > mean_delta
        else:
            is_worse = p10_delta < mean_delta
        
        if is_worse:
            gap = abs(mean_delta - p10_delta)
            context["risks"][kpi_name] = {
                "p10_value": _format_delta(abs(p10_delta), unit),
                "mean_value": _format_delta(abs(mean_delta), unit),
                "gap": _format_delta(gap, unit),
            }
    
    # Guardrail violations
    if guardrail_result:
        for violation in guardrail_result.violations:
            context["guardrails"]["violations"].append({
                "rule_type": violation.rule_type,
                "message": violation.message,
                "severity": violation.severity,
            })
    
    return json.dumps(context, indent=2)


def generate_bundle_narrative_gemini(
    plan: Plan,
    bundle_id: str,
    bundle_name: str,
    attribution: OptionAttribution,
    option_result: OptionResultMC,
    guardrail_result: Optional[GuardrailResult] = None,
    api_key: Optional[str] = None,
    model_name: Optional[str] = None,  # Auto-detect if None
) -> str:
    """
    Generate a SteerCo-grade narrative using Google's Gemini API.
    
    Args:
        plan: Plan object
        bundle_id: Bundle identifier
        bundle_name: Bundle name
        attribution: Attribution data
        option_result: Option result with scores
        guardrail_result: Guardrail evaluation result
        api_key: Gemini API key (if None, uses GEMINI_API_KEY env var)
        model_name: Gemini model to use (default: gemini-1.5-flash for speed/cost)
    
    Returns:
        Generated narrative text
    """
    if not GEMINI_AVAILABLE:
        raise ImportError(
            "google-generativeai package not installed. "
            "Install with: pip install google-generativeai"
        )
    
    # Get API key
    api_key = api_key or os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError(
            "Gemini API key not found. Set GEMINI_API_KEY environment variable "
            "or pass api_key parameter."
        )
    
    # Configure Gemini
    genai.configure(api_key=api_key)
    
    # Auto-detect model if not provided
    if model_name is None:
        model_name = _get_available_gemini_model()
        if not model_name:
            raise RuntimeError("No available Gemini model found. Check your API key and model access.")
    
    model = genai.GenerativeModel(model_name)
    
    # Build context
    context = _build_context_for_gemini(
        plan, bundle_id, bundle_name, attribution, option_result, guardrail_result
    )
    
    # Create prompt
    prompt = f"""You are a strategic planning consultant for a bank, writing an executive summary for a steering committee.

Generate a professional, concise narrative (3-4 paragraphs) for this strategic option based on the following structured data:

{context}

Requirements:
1. Write in executive style - clear, professional, and actionable
2. Structure: Opening paragraph (strategic context), Expected Impact (key KPI changes), Main Drivers (top 2-3), Downside Risks (p10 scenario), Key Approvals (if any)
3. Use specific numbers and metrics from the data
4. Highlight trade-offs and risks clearly
5. Keep total length to 4-6 paragraphs maximum
6. Use markdown formatting with ## for section headers

Generate the narrative now:"""
    
    try:
        response = model.generate_content(prompt)
        narrative = response.text
        
        # Add header
        header = f"## {bundle_name} ({bundle_id})\n\n"
        return header + narrative
        
    except Exception as e:
        # Fallback to template-based if Gemini fails
        raise RuntimeError(
            f"Gemini API call failed: {e}. "
            "Consider using template-based narrative generator as fallback."
        ) from e
