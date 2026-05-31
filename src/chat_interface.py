# src/chat_interface.py
"""
Conversational interface for querying the strategic plan.

Uses Gemini API to answer natural language questions about:
- Recommendations and rationale
- KPI forecasts and trends
- Portfolio options and guardrails
- Learning metrics and outcomes
- Anomalies and integrity issues
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False


def build_context_from_pack(pack: Any, plan: Any) -> str:
    """
    Build a context string from the SteerCo pack for the LLM.
    
    Includes:
    - Recommendation summary
    - KPI forecasts
    - Portfolio options
    - Guardrails status
    - Learning metrics
    - Anomalies
    """
    context_parts = []
    
    # Recommendation
    rec = None
    if hasattr(pack, 'recommendation'):
        rec = pack.recommendation
    elif isinstance(pack, dict) and 'recommendation' in pack:
        rec = pack['recommendation']
    
    if rec:
        context_parts.append("## RECOMMENDATION")
        # Handle both dict and object access
        if isinstance(rec, dict):
            chosen_name = rec.get('chosen_bundle_name', 'N/A')
            chosen_id = rec.get('chosen_bundle_id', 'N/A')
            confidence = rec.get('confidence', 0)
            guardrails_passed = rec.get('guardrails_passed', True)
            why = rec.get('why', [])
            alternatives = rec.get('alternatives', [])
        else:
            chosen_name = getattr(rec, 'chosen_bundle_name', 'N/A')
            chosen_id = getattr(rec, 'chosen_bundle_id', 'N/A')
            confidence = getattr(rec, 'confidence', 0)
            guardrails_passed = getattr(rec, 'guardrails_passed', True)
            why = getattr(rec, 'why', [])
            alternatives = getattr(rec, 'alternatives', [])
        
        context_parts.append(f"Chosen option: {chosen_name} ({chosen_id})")
        context_parts.append(f"Confidence: {confidence:.2f}")
        context_parts.append(f"Guardrails: {'PASSED' if guardrails_passed else 'FAILED'}")
        
        # Add approval requirements from guardrails
        if isinstance(rec, dict):
            violations = rec.get('guardrails_violations', [])
            if violations:
                approval_needed = []
                for v in violations:
                    if isinstance(v, dict) and v.get('requires_approval'):
                        req_approval = v['requires_approval']
                        if isinstance(req_approval, list):
                            approval_needed.extend(req_approval)
                        else:
                            approval_needed.append(req_approval)
                if approval_needed:
                    context_parts.append(f"Approval required from: {', '.join(set(approval_needed))}")
        
        if why:
            why_str = '; '.join(why) if isinstance(why, list) else str(why)
            context_parts.append(f"Rationale: {why_str}")
        if alternatives:
            context_parts.append(f"Alternatives: {len(alternatives)} options considered")
        context_parts.append("")
    
    # Helper function to safely get attribute or dict value
    def _get_attr_or_dict(obj, key, default=None):
        if isinstance(obj, dict):
            return obj.get(key, default)
        return getattr(obj, key, default)
    
    # Forecast summary
    forecast_head = _get_attr_or_dict(pack, 'forecast_head', [])
    if forecast_head:
        context_parts.append("## KPI FORECASTS (next 4 months)")
        for fc in forecast_head[:12]:  # First 12 rows (4 months * 3 KPIs)
            kpi_id = fc.get('kpi_id', '') if isinstance(fc, dict) else getattr(fc, 'kpi_id', '')
            kpi = next((k for k in plan.kpis if k.id == kpi_id), None)
            kpi_name = kpi.short_name if kpi else kpi_id
            forecast_val = fc.get('forecast', 0) if isinstance(fc, dict) else getattr(fc, 'forecast', 0)
            expected_val = fc.get('expected', 0) if isinstance(fc, dict) else getattr(fc, 'expected', 0)
            date = fc.get('date', '') if isinstance(fc, dict) else getattr(fc, 'date', '')
            context_parts.append(f"{date} {kpi_name}: forecast={forecast_val:.2f}, expected={expected_val:.2f}")
        context_parts.append("")
    
    # Portfolio options summary with action details
    options_mc = _get_attr_or_dict(pack, 'options_mc', [])
    option_narratives = _get_attr_or_dict(pack, 'option_narratives', {})
    bundles_config = _get_attr_or_dict(pack, 'config', {})
    bundles_list = bundles_config.get('bundles', []) if isinstance(bundles_config, dict) else []
    
    if options_mc:
        context_parts.append("## PORTFOLIO OPTIONS")
        for i, opt in enumerate(options_mc[:4], 1):  # Top 4 options
            if isinstance(opt, dict):
                bundle_id = opt.get('bundle_id', '')
                name = opt.get('bundle_name', bundle_id)
                cvar = opt.get('score_stress_cvar10', 0)
                mean = opt.get('score_stress_mean', 0)
            else:
                bundle_id = getattr(opt, 'bundle_id', '')
                name = getattr(opt, 'bundle_name', bundle_id)
                cvar = getattr(opt, 'score_stress_cvar10', 0)
                mean = getattr(opt, 'score_stress_mean', 0)
            
            context_parts.append(f"{i}. {name} ({bundle_id}): CVaR10={cvar:.2f}, mean={mean:.2f}")
            
            # Add narrative if available (contains detailed description of what the option does)
            if isinstance(option_narratives, dict) and bundle_id in option_narratives:
                narrative = option_narratives[bundle_id]
                
                # Include the full narrative text so LLM can answer "what does this option consist of"
                # The narrative contains: Expected Impact, Main Drivers (which initiatives), and Approvals
                context_parts.append(f"   Full Description:")
                # Clean up the narrative for better readability
                narrative_clean = narrative.replace("## ", "").replace("### ", "").strip()
                # Split into lines and take key sections
                lines = narrative_clean.split('\n')
                for line in lines:
                    line = line.strip()
                    if line and not line.startswith('Conclusion:'):
                        # Limit line length to avoid token bloat
                        if len(line) > 300:
                            line = line[:300] + "..."
                        context_parts.append(f"     {line}")
                
                # Also extract approvals separately for quick reference
                if "Key Approvals Required" in narrative:
                    approvals_section = narrative.split("Key Approvals Required")[1]
                    if "No approvals" not in approvals_section:
                        # Extract approval names
                        approval_lines = [l.strip() for l in approvals_section.split('\n') if l.strip() and l.startswith('-')]
                        if approval_lines:
                            approval_names = []
                            for line in approval_lines[:8]:  # Get more approvals
                                # Extract name before "is required", "due to", or "necessary"
                                for keyword in ["is required", "due to", "necessary because"]:
                                    if keyword in line:
                                        name = line.split(keyword)[0].replace("-", "").replace("Approval from", "").strip()
                                        if name and len(name) > 2:
                                            approval_names.append(name)
                                            break
                            if approval_names:
                                context_parts.append(f"   Quick Reference - Approvals: {', '.join(set(approval_names))}")
            
            # Add action details if available from bundles config
            if bundles_list:
                bundle_info = next((b for b in bundles_list if b.get('id') == bundle_id), None)
                if bundle_info and bundle_info.get('actions'):
                    actions_desc = []
                    for action in bundle_info['actions']:
                        action_type = action.get('type', '')
                        target = action.get('target', '')
                        params = action.get('parameters', {})
                        
                        # Get initiative name from plan if available
                        target_name = target
                        if hasattr(plan, 'initiatives'):
                            init = next((i for i in plan.initiatives if i.id == target), None)
                            if init:
                                target_name = init.name
                        
                        if action_type == 'accelerate_initiative':
                            months = params.get('months', '')
                            actions_desc.append(f"Accelerate {target_name} ({target}) by {months} months")
                        elif action_type == 'split_scope':
                            reduction = params.get('reduction', '')
                            reduction_pct = f"{float(reduction)*100:.0f}%" if reduction else ""
                            actions_desc.append(f"Split scope of {target_name} ({target}) - reduce by {reduction_pct}")
                        elif action_type == 'add_capacity':
                            gain = params.get('capacity_gain', '')
                            gain_pct = f"{float(gain)*100:.0f}%" if gain else ""
                            actions_desc.append(f"Add {gain_pct} capacity to {target_name} ({target})")
                        else:
                            actions_desc.append(f"{action_type} on {target_name} ({target})")
                    
                    if actions_desc:
                        context_parts.append(f"   Actions: {'; '.join(actions_desc)}")
            
            context_parts.append("")
        context_parts.append("")
    
    # Guardrails summary
    guardrails_report = _get_attr_or_dict(pack, 'guardrails_report', '')
    if guardrails_report and "FAIL" in str(guardrails_report):
        context_parts.append("## GUARDRAILS")
        context_parts.append("Some options have guardrail violations (see details in full report)")
        context_parts.append("")
    
    # Learning metrics
    learning_metrics = _get_attr_or_dict(pack, 'learning_metrics', {})
    if learning_metrics:
        if isinstance(learning_metrics, dict):
            lm = learning_metrics
        else:
            lm = {k: getattr(learning_metrics, k, None) for k in ['total_decisions', 'mean_forecast_accuracy', 'improvement_trend']}
        context_parts.append("## LEARNING FROM OUTCOMES")
        context_parts.append(f"Total decisions: {lm.get('total_decisions', 0)}")
        if lm.get('mean_forecast_accuracy', 0) > 0:
            context_parts.append(f"Forecast accuracy: {lm.get('mean_forecast_accuracy', 0):.1%}")
            context_parts.append(f"Trend: {lm.get('improvement_trend', 'N/A')}")
        context_parts.append("")
    
    # ML Anomalies
    ml_anomalies = _get_attr_or_dict(pack, 'ml_anomalies', {})
    if ml_anomalies:
        if isinstance(ml_anomalies, dict):
            anomalies = [kpi_id for kpi_id, data in ml_anomalies.items() if isinstance(data, dict) and data.get('is_anomaly', False)]
        else:
            anomalies = []
        if anomalies:
            context_parts.append("## ML ANOMALIES DETECTED")
            for kpi_id in anomalies:
                kpi = next((k for k in plan.kpis if k.id == kpi_id), None)
                kpi_name = kpi.short_name if kpi else kpi_id
                anomaly_data = ml_anomalies[kpi_id] if isinstance(ml_anomalies, dict) else {}
                explanation = anomaly_data.get('explanation', '') if isinstance(anomaly_data, dict) else ''
                context_parts.append(f"- {kpi_name} ({kpi_id}): {explanation}")
            context_parts.append("")
    
    # Integrity summary
    integrity_scores = _get_attr_or_dict(pack, 'integrity_scores', {})
    if integrity_scores:
        if isinstance(integrity_scores, dict):
            low_integrity = [kpi_id for kpi_id, score in integrity_scores.items() if score > 0.6]
        else:
            low_integrity = []
        if low_integrity:
            context_parts.append("## KPI INTEGRITY ISSUES")
            for kpi_id in low_integrity:
                kpi = next((k for k in plan.kpis if k.id == kpi_id), None)
                kpi_name = kpi.short_name if kpi else kpi_id
                score = integrity_scores[kpi_id] if isinstance(integrity_scores, dict) else 0
                context_parts.append(f"- {kpi_name} ({kpi_id}): integrity score = {score:.2f}")
            context_parts.append("")
    
    return "\n".join(context_parts)


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


def query_strategic_plan(
    question: str,
    pack: Any,
    plan: Any,
    api_key: Optional[str] = None,
    model: str = "gemini-pro",
) -> str:
    """
    Answer a question about the strategic plan using Gemini.
    
    Args:
        question: Natural language question
        pack: SteerCoPack object
        plan: Plan object
        api_key: Gemini API key (if None, uses GEMINI_API_KEY env var)
        model: Gemini model to use (will auto-detect if not available)
    
    Returns:
        Answer as a string
    """
    if not GEMINI_AVAILABLE:
        return "Error: google-generativeai not installed. Install with: pip install google-generativeai"
    
    api_key = api_key or os.getenv("GEMINI_API_KEY")
    if not api_key:
        return "Error: Gemini API key not found. Set GEMINI_API_KEY environment variable."
    
    try:
        genai.configure(api_key=api_key)
        
        # Auto-detect available model
        available_model = _get_available_model(model)
        if not available_model:
            return "Error: No available Gemini model found that supports generateContent."
        
        # Build context
        context = build_context_from_pack(pack, plan)
        
        # Build prompt
        prompt = f"""You are an AI assistant helping with strategic planning for a bank (CACIB - Transaction Banking Services).

You have access to the following strategic plan information:

{context}

Answer the following question based on this information. Be concise, accurate, and focus on actionable insights.

Question: {question}

Answer:"""
        
        # Generate response
        model_instance = genai.GenerativeModel(available_model)
        response = model_instance.generate_content(prompt)
        
        return response.text.strip()
    
    except Exception as e:
        return f"Error querying Gemini: {str(e)}"


def chat_interactive(
    pack_path: str,
    plan_path: str = "data/plan.yaml",
    api_key: Optional[str] = None,
) -> None:
    """
    Interactive chat mode.
    
    Args:
        pack_path: Path to steerco_pack.json
        plan_path: Path to plan.yaml
        api_key: Gemini API key (if None, uses GEMINI_API_KEY env var)
    """
    # Load pack and plan
    from src.load_plan import load_plan
    
    try:
        with open(pack_path, 'r', encoding='utf-8') as f:
            pack_data = json.load(f)
        
        plan = load_plan(plan_path)
        
        # Create a simple pack-like object
        class PackProxy:
            def __init__(self, data):
                for key, value in data.items():
                    setattr(self, key, value)
        
        pack = PackProxy(pack_data)
        
    except Exception as e:
        print(f"[ERROR] Failed to load pack or plan: {e}")
        return
    
    print("\n" + "=" * 80)
    print("Strategic Plan Chat Interface")
    print("=" * 80)
    print("\nAsk questions about:")
    print("  - Recommendations and rationale")
    print("  - KPI forecasts and trends")
    print("  - Portfolio options and guardrails")
    print("  - Learning metrics")
    print("  - Anomalies and integrity issues")
    print("\nType 'quit' or 'exit' to end the session.")
    print("=" * 80)
    
    while True:
        try:
            question = input("\nYou: ").strip()
            
            if question.lower() in ['quit', 'exit', 'q']:
                print("\n[GOODBYE] Chat session ended.")
                break
            
            if not question:
                continue
            
            print("\n[THINKING]...")
            answer = query_strategic_plan(question, pack, plan, api_key=api_key)
            print(f"\nAssistant: {answer}")
        
        except KeyboardInterrupt:
            print("\n\n[GOODBYE] Chat session ended.")
            break
        except Exception as e:
            print(f"\n[ERROR] {e}")


def query_single(
    question: str,
    pack_path: str,
    plan_path: str = "data/plan.yaml",
    api_key: Optional[str] = None,
) -> str:
    """
    Answer a single question (non-interactive).
    
    Args:
        question: The question to ask
        pack_path: Path to steerco_pack.json
        plan_path: Path to plan.yaml
        api_key: Gemini API key (if None, uses GEMINI_API_KEY env var)
    
    Returns:
        Answer as a string
    """
    from src.load_plan import load_plan
    
    try:
        with open(pack_path, 'r', encoding='utf-8') as f:
            pack_data = json.load(f)
        
        plan = load_plan(plan_path)
        
        # Create a simple pack-like object
        class PackProxy:
            def __init__(self, data):
                for key, value in data.items():
                    setattr(self, key, value)
        
        pack = PackProxy(pack_data)
        
        return query_strategic_plan(question, pack, plan, api_key=api_key)
    
    except Exception as e:
        return f"Error: {str(e)}"
