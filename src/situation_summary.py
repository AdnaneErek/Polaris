# src/situation_summary.py
"""
LLM-powered situation summary generator.

Generates human-readable summaries for situations requiring attention:
- Objectives behind schedule
- Anomalies detected
- Initiative health issues
- Integrity problems
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

from .schemas import Plan
from .monitor import MonitorSnapshot
from .steering import InitiativeHealth


def generate_situation_summaries(
    plan: Plan,
    monitor_snapshot: MonitorSnapshot,
    initiative_health: Dict[str, InitiativeHealth],
    ml_anomalies: Optional[Dict[str, Any]] = None,
    integrity_scores: Optional[Dict[str, float]] = None,
    api_key: Optional[str] = None,
    model: str = "gemini-pro",
) -> Dict[str, str]:
    """
    Generate LLM-powered summaries for situations requiring attention.
    
    Returns a dictionary mapping situation IDs to human-readable summaries.
    """
    if not GEMINI_AVAILABLE:
        return _generate_fallback_summaries(plan, monitor_snapshot, initiative_health, ml_anomalies)
    
    api_key = api_key or os.getenv("GEMINI_API_KEY")
    if not api_key:
        return _generate_fallback_summaries(plan, monitor_snapshot, initiative_health, ml_anomalies)
    
    try:
        genai.configure(api_key=api_key)
        
        # Get available model (must be called after configure)
        available_model = _get_available_model() or model
        
        summaries = {}
        
        # Generate summary for each objective behind schedule
        for obj_status in monitor_snapshot.objective_statuses:
            if not obj_status.on_track:
                obj = next((o for o in plan.objectives if o.id == obj_status.objective_id), None)
                if obj:
                    summary = _generate_objective_summary(obj, obj_status, plan, available_model)
                    summaries[f"objective_{obj.id}"] = summary
        
        # Generate summary for each initiative with issues
        for init_id, health in initiative_health.items():
            if health.health_flag in ("watch", "bad"):
                init = next((i for i in plan.initiatives if i.id == init_id), None)
                if init:
                    summary = _generate_initiative_summary(init, health, plan, available_model)
                    summaries[f"initiative_{init_id}"] = summary
        
        # Generate summary for anomalies
        if ml_anomalies:
            for kpi_id, anomaly_data in ml_anomalies.items():
                if anomaly_data.get('is_anomaly', False):
                    kpi = next((k for k in plan.kpis if k.id == kpi_id), None)
                    if kpi:
                        summary = _generate_anomaly_summary(kpi, anomaly_data, plan, available_model)
                        summaries[f"anomaly_{kpi_id}"] = summary
        
        # Generate summary for integrity issues
        if integrity_scores:
            for kpi_id, score in integrity_scores.items():
                if score < 0.6:  # Low integrity
                    kpi = next((k for k in plan.kpis if k.id == kpi_id), None)
                    if kpi:
                        summary = _generate_integrity_summary(kpi, score, plan, available_model)
                        summaries[f"integrity_{kpi_id}"] = summary
        
        return summaries
    
    except Exception:
        return _generate_fallback_summaries(plan, monitor_snapshot, initiative_health, ml_anomalies)


def _get_available_model() -> Optional[str]:
    """Get an available Gemini model."""
    try:
        models = genai.list_models()
        for m in models:
            if hasattr(m, 'supported_generation_methods'):
                if 'generateContent' in m.supported_generation_methods:
                    model_name = m.name if hasattr(m, 'name') else str(m)
                    if '/' in model_name:
                        return model_name.split('/')[-1]
                    return model_name
    except Exception:
        pass
    
    # Fallback to common names
    for name in ["gemini-1.5-pro", "gemini-1.5-flash", "gemini-pro"]:
        try:
            genai.GenerativeModel(name)
            return name
        except Exception:
            continue
    return None


def _generate_objective_summary(
    obj: Any,
    obj_status: Any,
    plan: Plan,
    model: str
) -> str:
    """Generate summary for an objective behind schedule."""
    kpi_details = []
    for kpi_status in obj_status.kpi_statuses:
        kpi = next((k for k in plan.kpis if k.id == kpi_status.kpi_id), None)
        if kpi and kpi_status.drift and kpi_status.drift > 0:
            kpi_details.append(
                f"{kpi.short_name}: {kpi_status.actual:.2f} vs expected {kpi_status.expected:.2f} "
                f"(drift: {kpi_status.drift:.2f})"
            )
    
    prompt = f"""Generate a concise, executive-friendly summary (1-2 sentences) for an objective that is behind schedule.

Objective: {obj.name}
Objective ID: {obj.id}
Weight: {obj.weight}
Drift Confidence: {obj_status.drift_confidence:.2f}

KPI Issues:
{chr(10).join(f"- {detail}" for detail in kpi_details)}

Write a clear, actionable summary that explains the situation and why attention is needed. Be specific about the KPIs and the gap.

Summary:"""
    
    try:
        available_model = _get_available_model() or model
        genai_model = genai.GenerativeModel(available_model)
        response = genai_model.generate_content(prompt)
        return response.text.strip()
    except Exception:
        return f"{obj.name} is behind schedule with drift confidence {obj_status.drift_confidence:.2f}. KPIs are not meeting expected trajectory."


def _generate_initiative_summary(
    init: Any,
    health: InitiativeHealth,
    plan: Plan,
    model: str
) -> str:
    """Generate summary for an initiative with health issues."""
    prompt = f"""Generate a concise, executive-friendly summary (1-2 sentences) for an initiative that needs attention.

Initiative: {init.name}
Initiative ID: {init.id}
Progress: {health.progress:.1%}
Delay: {health.delay_months} months
Health Status: {health.health_flag}
Blocked By: {', '.join(health.blocked_by) if health.blocked_by else 'None'}

Write a clear summary explaining the issue and its impact.

Summary:"""
    
    try:
        available_model = _get_available_model() or model
        genai_model = genai.GenerativeModel(available_model)
        response = genai_model.generate_content(prompt)
        return response.text.strip()
    except Exception:
        if health.blocked_by:
            return f"{init.name} is blocked by {', '.join(health.blocked_by)} and {health.delay_months} months behind schedule."
        return f"{init.name} is {health.delay_months} months behind schedule with {health.progress:.1%} progress."


def _generate_anomaly_summary(
    kpi: Any,
    anomaly_data: Dict[str, Any],
    plan: Plan,
    model: str
) -> str:
    """Generate summary for a KPI anomaly."""
    prompt = f"""Generate a concise, executive-friendly summary (1-2 sentences) for a KPI showing anomalous behavior.

KPI: {kpi.name} ({kpi.short_name})
Anomaly Explanation: {anomaly_data.get('explanation', 'Anomaly detected')}
Confidence: {anomaly_data.get('confidence', 0):.2f}

Write a clear summary explaining what the anomaly means and why it matters.

Summary:"""
    
    try:
        available_model = _get_available_model() or model
        genai_model = genai.GenerativeModel(available_model)
        response = genai_model.generate_content(prompt)
        return response.text.strip()
    except Exception:
        return f"{kpi.short_name} shows anomalous behavior: {anomaly_data.get('explanation', 'unusual pattern detected')}."


def _generate_integrity_summary(
    kpi: Any,
    score: float,
    plan: Plan,
    model: str
) -> str:
    """Generate summary for a KPI integrity issue."""
    prompt = f"""Generate a concise, executive-friendly summary (1-2 sentences) for a KPI with data integrity concerns.

KPI: {kpi.name} ({kpi.short_name})
Integrity Score: {score:.2f} (lower is worse, 0.6 threshold)

Write a clear summary explaining the data quality issue and its implications.

Summary:"""
    
    try:
        available_model = _get_available_model() or model
        genai_model = genai.GenerativeModel(available_model)
        response = genai_model.generate_content(prompt)
        return response.text.strip()
    except Exception:
        return f"{kpi.short_name} has data integrity concerns (score: {score:.2f}). Data quality may affect decision-making."


def _generate_fallback_summaries(
    plan: Plan,
    monitor_snapshot: MonitorSnapshot,
    initiative_health: Dict[str, InitiativeHealth],
    ml_anomalies: Optional[Dict[str, Any]],
) -> Dict[str, str]:
    """Generate fallback summaries without LLM."""
    summaries = {}
    
    for obj_status in monitor_snapshot.objective_statuses:
        if not obj_status.on_track:
            obj = next((o for o in plan.objectives if o.id == obj_status.objective_id), None)
            if obj:
                summaries[f"objective_{obj.id}"] = (
                    f"{obj.name} is behind schedule (drift confidence: {obj_status.drift_confidence:.2f}). "
                    f"KPIs are not meeting expected trajectory."
                )
    
    for init_id, health in initiative_health.items():
        if health.health_flag in ("watch", "bad"):
            init = next((i for i in plan.initiatives if i.id == init_id), None)
            if init:
                if health.blocked_by:
                    summaries[f"initiative_{init_id}"] = (
                        f"{init.name} is blocked by {', '.join(health.blocked_by)} "
                        f"and {health.delay_months} months behind schedule."
                    )
                else:
                    summaries[f"initiative_{init_id}"] = (
                        f"{init.name} is {health.delay_months} months behind schedule "
                        f"with {health.progress:.1%} progress."
                    )
    
    if ml_anomalies:
        for kpi_id, anomaly_data in ml_anomalies.items():
            if anomaly_data.get('is_anomaly', False):
                kpi = next((k for k in plan.kpis if k.id == kpi_id), None)
                if kpi:
                    summaries[f"anomaly_{kpi_id}"] = (
                        f"{kpi.short_name} shows anomalous behavior: "
                        f"{anomaly_data.get('explanation', 'unusual pattern detected')}."
                    )
    
    return summaries
