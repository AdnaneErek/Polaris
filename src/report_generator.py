# src/report_generator.py
"""
PDF Report Generator for Strategic Planning Progress Reports.

Generates professional PDF reports with:
- Period-based progress (monthly, quarterly, etc.)
- Objective-specific or comprehensive reports
- Charts and visualizations
- Narrative progress descriptions
"""
from __future__ import annotations

import io
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from xml.sax.saxutils import escape

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, PageBreak, Image, Table, TableStyle
    )
    from reportlab.pdfgen import canvas
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False


REPORT_SPACING: Dict[str, float] = {
    "title_to_meta": 0.18,
    "section_gap": 0.22,
    "heading_to_content": 0.10,
    "block_gap": 0.08,
    "micro_gap": 0.05,
    "chart_pre_gap": 0.08,
    "chart_post_gap": 0.12,
}

REPORT_BRAND: Dict[str, str] = {
    "primary": "#1E3A8A",
    "secondary": "#334155",
    "text": "#0F172A",
    "muted": "#64748B",
    "accent": "#7C3AED",
    "success": "#166534",
    "warning": "#B45309",
    "danger": "#B91C1C",
    "surface": "#F8FAFC",
}

TYPO_SCALE: Dict[str, float] = {
    "display": 21,
    "h2": 15,
    "h3": 12.5,
    "body": 10.2,
    "meta": 9.2,
}


def _space(kind: str) -> Spacer:
    """Standardized vertical spacing for PDF layout consistency."""
    return Spacer(1, REPORT_SPACING.get(kind, REPORT_SPACING["block_gap"]) * inch)


def generate_progress_report(
    plan: Any,
    kpi_history: pd.DataFrame,
    initiative_history: pd.DataFrame,
    as_of: str,
    period_type: str = "monthly",  # "monthly", "quarterly", "yearly"
    objective_ids: Optional[List[str]] = None,  # None = all, [] = specific list
    output_path: Optional[str] = None,
    pack: Optional[Dict[str, Any]] = None,  # SteerCo pack for enhanced content
) -> str:
    """
    Generate a PDF progress report.
    
    Args:
        plan: The Plan object
        kpi_history: DataFrame with columns: date, kpi_id, value
        initiative_history: DataFrame with columns: date, initiative_id, progress
        as_of: Report date (YYYY-MM-DD)
        period_type: "monthly", "quarterly", or "yearly"
        objective_ids: List of objective IDs to include (None = all)
        output_path: Path to save PDF (if None, generates filename)
    
    Returns:
        Path to generated PDF file
    """
    if not REPORTLAB_AVAILABLE:
        raise ImportError("reportlab not installed. Install with: pip install reportlab")
    
    # Determine objectives to report on
    if objective_ids is None:
        objectives = plan.objectives
    else:
        objectives = [obj for obj in plan.objectives if obj.id in objective_ids]
    
    if not objectives:
        raise ValueError("No objectives selected for report")
    
    # Generate output path
    if output_path is None:
        period_str = period_type
        obj_str = "all" if objective_ids is None else f"{len(objective_ids)}_objs"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"artifacts/reports/progress_report_{period_str}_{obj_str}_{timestamp}.pdf"
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Prepare data
    kpi_df = kpi_history.copy()
    kpi_df['date'] = pd.to_datetime(kpi_df['date'])
    
    init_df = initiative_history.copy() if initiative_history is not None and not initiative_history.empty else pd.DataFrame()
    if not init_df.empty:
        init_df['date'] = pd.to_datetime(init_df['date'])
    
    # Aggregate by period
    period_data = _aggregate_by_period(kpi_df, init_df, as_of, period_type, plan)
    
    # Generate charts
    chart_paths = _generate_charts(plan, objectives, period_data, kpi_df, init_df, as_of, period_type, pack)
    
    # Generate enhanced narrative with problems, actions, impact
    narrative = _generate_enhanced_narrative(plan, objectives, period_data, as_of, period_type, pack)
    
    # Create PDF
    doc = SimpleDocTemplate(str(output_path), pagesize=A4)
    story = []
    styles = getSampleStyleSheet()
    
    # Brand typography baseline
    styles['Normal'].fontName = 'Helvetica'
    styles['Normal'].fontSize = TYPO_SCALE["body"]
    styles['Normal'].leading = 14
    styles['Normal'].textColor = colors.HexColor(REPORT_BRAND["text"])
    styles['Heading2'].fontName = 'Helvetica-Bold'
    styles['Heading2'].fontSize = TYPO_SCALE["h2"]
    styles['Heading2'].textColor = colors.HexColor(REPORT_BRAND["primary"])
    styles['Heading3'].fontName = 'Helvetica-Bold'
    styles['Heading3'].fontSize = TYPO_SCALE["h3"]
    styles['Heading3'].textColor = colors.HexColor(REPORT_BRAND["secondary"])

    # Title
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=TYPO_SCALE["display"],
        textColor=colors.HexColor(REPORT_BRAND["primary"]),
        spaceAfter=30,
        alignment=1,  # Center
    )
    story.append(Paragraph("Strategic Planning Progress Report", title_style))
    story.append(_space("title_to_meta"))
    
    # Report metadata
    meta_style = ParagraphStyle(
        'Meta',
        parent=styles['Normal'],
        fontSize=TYPO_SCALE["meta"],
        textColor=colors.HexColor(REPORT_BRAND["muted"]),
    )
    period_start, period_end = _get_period_range(kpi_df)
    story.append(Paragraph(f"<b>Report Date:</b> {as_of}", meta_style))
    story.append(Paragraph(f"<b>Period Type:</b> {period_type.title()}", meta_style))
    story.append(Paragraph(f"<b>Period Range:</b> {period_start} to {period_end}", meta_style))
    story.append(Paragraph(f"<b>Objectives:</b> {len(objectives)}", meta_style))
    if pack:
        gen = pack.get("bundle_generation", {}) or {}
        if gen:
            mode = str(gen.get("mode", "unknown")).replace("_", " ").title()
            n_bundles = gen.get("n_bundles", "N/A")
            fallback_reason = gen.get("fallback_reason")
            story.append(Paragraph(f"<b>Bundle Generation:</b> {mode} (bundles={n_bundles})", meta_style))
            if fallback_reason:
                story.append(Paragraph(f"<b>Generation Detail:</b> {fallback_reason}", meta_style))
        lm = pack.get("learning_metrics", {}) or {}
        if lm:
            conf_mult = float(lm.get("confidence_multiplier", 1.0))
            story.append(Paragraph(f"<b>Learning Confidence Multiplier:</b> {conf_mult:.2f}x", meta_style))
            recal_kpis = lm.get("forecast_recalibration_kpis", []) or []
            if recal_kpis:
                story.append(Paragraph(f"<b>Forecast Recalibrated KPIs:</b> {', '.join([str(k) for k in recal_kpis])}", meta_style))
            weight_overrides = lm.get("scoring_weight_overrides", {}) or {}
            if weight_overrides:
                small = ", ".join([f"{k}:{float(v):.2f}" for k, v in sorted(weight_overrides.items())])
                story.append(Paragraph(f"<b>Scoring Weight Overrides:</b> {small}", meta_style))
    story.append(_space("section_gap"))
    
    # Executive Decision Summary (page-1 governance view)
    story.append(Paragraph("Steering Decision Summary", styles['Heading2']))
    story.append(_space("heading_to_content"))
    on_track, at_risk = _get_overall_status_counts(plan, objectives, period_data)
    rec_name, rec_conf = _get_recommendation_summary(pack)
    guardrails_failed = bool(pack and not pack.get('recommendation', {}).get('guardrails_passed', True))
    forecast_alerts = pack.get("forecast_deviation_alerts", {}) if pack else {}
    alert_count = sum(1 for v in forecast_alerts.values() if bool(v.get("is_deviation_alert", False)))
    urgency = _estimate_decision_urgency(at_risk, guardrails_failed, alert_count)

    decision_text = "Approve the recommended option for immediate execution."
    if guardrails_failed:
        decision_text = "Review and arbitrate guardrail exceptions before execution approval."
    decision_box = Table(
        [[Paragraph(f"<b>DECISION REQUIRED THIS CYCLE</b><br/>{_safe_text(decision_text)}", styles['Normal'])]],
        colWidths=[6.1 * inch],
    )
    decision_box.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, 0), colors.HexColor('#FFF7ED')),
        ('TEXTCOLOR', (0, 0), (0, 0), colors.HexColor('#7C2D12')),
        ('LINEBEFORE', (0, 0), (0, 0), 4, colors.HexColor('#EA580C')),
        ('BOX', (0, 0), (0, 0), 0.5, colors.HexColor('#FDBA74')),
        ('LEFTPADDING', (0, 0), (0, 0), 8),
        ('RIGHTPADDING', (0, 0), (0, 0), 8),
        ('TOPPADDING', (0, 0), (0, 0), 8),
        ('BOTTOMPADDING', (0, 0), (0, 0), 8),
    ]))
    story.append(decision_box)
    story.append(_space("block_gap"))
    if rec_name:
        if rec_conf is not None:
            rec_line = f"<b>Recommended option:</b> {rec_name} (confidence: {_format_confidence_label(rec_conf)})"
        else:
            rec_line = f"<b>Recommended option:</b> {rec_name}"
        story.append(Paragraph(rec_line, styles['Normal']))
    story.append(Paragraph(f"<b>Urgency:</b> {urgency}", styles['Normal']))
    story.append(Paragraph(
        "<b>Section Navigation:</b> 1) Decision summary  2) Critical issues  3) Options and risk  4) KPI trends  5) Recommendations and ownership.",
        styles['Normal']
    ))

    if pack:
        options_mc = pack.get("options_mc", []) or []
        if options_mc:
            ranked = sorted(options_mc, key=lambda x: x.get("score_stress_cvar10", 0.0), reverse=True)
            top = ranked[0]
            next_best = ranked[1] if len(ranked) > 1 else None
            cvar_top = float(top.get("score_stress_cvar10", 0.0))
            mean_top = float(top.get("score_stress_mean", 0.0))
            if next_best is not None:
                cvar_delta = cvar_top - float(next_best.get("score_stress_cvar10", 0.0))
                story.append(Paragraph(
                    f"<b>Business impact signal:</b> Risk-adjusted downside score (CVaR10) improvement vs next best option: {cvar_delta:+.3f}; expected return: {mean_top:.3f}.",
                    styles['Normal']
                ))
            else:
                story.append(Paragraph(
                    f"<b>Business impact signal:</b> Risk-adjusted downside score (CVaR10): {cvar_top:.3f}; expected return: {mean_top:.3f}.",
                    styles['Normal']
                ))
        rec_payload = (pack.get("recommendation", {}) or {})
        chosen_id = str(rec_payload.get("chosen_bundle_id", "") or "")
        chosen_bundle = next((b for b in (pack.get("bundles", []) or []) if str(b.get("id", "")) == chosen_id), None)
        if chosen_bundle:
            actions = chosen_bundle.get("actions", []) or []
            touched_inits = len({str(a.get("target_initiative", "")) for a in actions if a.get("target_initiative")})
            story.append(Paragraph(
                f"<b>Implementation scope:</b> {len(actions)} action(s) across {touched_inits} initiative(s); target materialization window: next 2 reporting cycles.",
                styles['Normal']
            ))
            if actions:
                first_action = actions[0]
                a_type = str(first_action.get("type", ""))
                a_params = (first_action.get("parameters", {}) or {})
                story.append(Paragraph(
                    f"<b>{_estimate_action_resource_cost(a_type, a_params)}</b>",
                    styles['Normal']
                ))
                base_k, adj_k, ccy = _estimate_bundle_budget_envelope(plan, actions)
                if base_k > 0:
                    delta_k = adj_k - base_k
                    story.append(Paragraph(
                        f"<b>Budget envelope (plan-based):</b> baseline {base_k:.0f} {ccy}, adjusted {adj_k:.0f} {ccy} ({delta_k:+.0f}).",
                        styles['Normal']
                    ))
                    year_limits, budget_ccy = _budget_constraint_info(plan)
                    year_key = str(pd.to_datetime(as_of).year)
                    if year_limits and year_key in year_limits:
                        cap = float(year_limits.get(year_key, 0.0) or 0.0)
                        if cap > 0:
                            util = (adj_k / cap) * 100.0
                            story.append(Paragraph(
                                f"<b>Annual cap check ({year_key}):</b> {adj_k:.0f} / {cap:.0f} {budget_ccy or ccy} ({util:.0f}% of cap).",
                                styles['Normal']
                            ))
                story.append(Paragraph(
                    "<b>Delivery risk:</b> Execution delay risk remains medium if cross-team staffing is not secured within the current cycle.",
                    styles['Normal']
                ))
    story.append(_space("section_gap"))

    status_table = Table(
        [
            ["On Track", "Requiring Attention", "Forecast Alerts"],
            [str(on_track), str(at_risk), str(alert_count)],
        ],
        colWidths=[1.8 * inch, 1.8 * inch, 1.4 * inch],
    )
    status_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#EEF2FF')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.HexColor('#1E3A8A')),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 10),
        ('FONTSIZE', (0, 1), (-1, 1), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 6),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#CBD5F5')),
    ]))
    story.append(status_table)
    story.append(_space("section_gap"))

    # Executive KPI Health Dashboard (single-page skim)
    story.append(Paragraph("Current Strategic Health Dashboard", styles['Heading2']))
    story.append(_space("heading_to_content"))
    kpi_health_table = _generate_kpi_health_dashboard_table(plan, objectives, period_data)
    if kpi_health_table:
        story.append(kpi_health_table)
    else:
        story.append(Paragraph("KPI health dashboard unavailable for the selected period.", styles['Normal']))
    story.append(PageBreak())

    # Executive Summary
    story.append(Paragraph("Executive Summary", styles['Heading2']))
    story.append(_space("heading_to_content"))

    if rec_name:
        table_cell_style = ParagraphStyle(
            'RecTableCell',
            parent=styles['Normal'],
            fontSize=9,
            leading=11,
            wordWrap='CJK',
        )
        rec_table = Table(
            [
                [Paragraph("Key Recommendation", table_cell_style), Paragraph("Confidence Level", table_cell_style)],
                [Paragraph(_safe_text(rec_name), table_cell_style), Paragraph(_safe_text(_format_confidence_label(rec_conf)), table_cell_style)],
            ],
            colWidths=[2.9 * inch, 1.7 * inch],
        )
        rec_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#F1F5F9')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.HexColor('#0F172A')),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('FONTSIZE', (0, 1), (-1, 1), 9),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 6),
            ('TOPPADDING', (0, 1), (-1, 1), 5),
            ('BOTTOMPADDING', (0, 1), (-1, 1), 5),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#CBD5F5')),
            ('WORDWRAP', (0, 0), (-1, -1), 'CJK'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ]))
        story.append(rec_table)
        story.append(_space("block_gap"))

        if pack and not pack.get('recommendation', {}).get('guardrails_passed', True):
            story.append(Paragraph(
                "SteerCo decision required (guardrails failed)",
                ParagraphStyle('GuardrailsWarn', parent=styles['Normal'], textColor=colors.HexColor('#B45309'))
            ))
            story.append(_space("heading_to_content"))

    summary = _generate_executive_summary(plan, objectives, period_data, as_of, pack)
    story.append(Paragraph(summary, styles['Normal']))
    story.append(_space("section_gap"))

    # Common inline styles for markdown-like rendering
    heading2_style = ParagraphStyle(
        'Heading2Compact',
        parent=styles['Heading2'],
        fontSize=TYPO_SCALE["h2"],
        spaceBefore=8,
        spaceAfter=4,
    )
    heading3_style = ParagraphStyle(
        'Heading3Compact',
        parent=styles['Heading3'],
        fontSize=TYPO_SCALE["h3"],
        textColor=colors.HexColor(REPORT_BRAND["secondary"]),
        spaceBefore=6,
        spaceAfter=4,
    )
    heading4_style = ParagraphStyle(
        'Heading4Compact',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=TYPO_SCALE["body"],
        textColor=colors.HexColor(REPORT_BRAND["secondary"]),
        spaceBefore=4,
        spaceAfter=2,
    )
    body_style = ParagraphStyle(
        'Body',
        parent=styles['Normal'],
        spaceAfter=3,
    )
    bullet_style = ParagraphStyle(
        'Bullet',
        parent=styles['Normal'],
        leftIndent=14,
        bulletIndent=6,
        spaceAfter=2.5,
    )
    numbered_style = ParagraphStyle(
        'Numbered',
        parent=styles['Normal'],
        leftIndent=14,
        spaceAfter=2.5,
    )

    # Critical Issues Identified
    forecast_df = _resolve_forecast_df(plan, kpi_df, init_df, as_of, pack)
    critical_table = _generate_critical_issues_table(plan, objectives, forecast_df)
    story.append(Paragraph("Critical Issues Identified", styles['Heading2']))
    story.append(_space("heading_to_content"))
    if critical_table:
        story.append(Paragraph("The following performance gaps require strategic intervention:", styles['Normal']))
        story.append(Paragraph(
            "<i>*Adverse Gap convention: positive = performance risk (worse than plan), negative = better than plan.</i>",
            styles['Normal']
        ))
        story.append(_space("block_gap"))
        story.append(critical_table)
        risk_lines = _generate_critical_risk_statements(plan, objectives, forecast_df)
        if risk_lines:
            story.append(_space("block_gap"))
            story.append(Paragraph("<b>Executive risk interpretation:</b>", styles['Normal']))
            for risk_line in risk_lines:
                if risk_line.startswith("RED "):
                    txt = f"<font color='#B91C1C'><b>{risk_line[:3]}</b></font>{risk_line[3:]}"
                elif risk_line.startswith("AMBER "):
                    txt = f"<font color='#B45309'><b>{risk_line[:5]}</b></font>{risk_line[5:]}"
                else:
                    txt = risk_line
                story.append(Paragraph(f"• {txt}", styles['Normal']))
    else:
        story.append(Paragraph("No critical issues identified for the selected period.", styles['Normal']))

    risk_register = _generate_risk_register_table(plan, pack)
    if risk_register:
        story.append(_space("block_gap"))
        story.append(Paragraph("Risk Register and Mitigation Ownership", styles['Heading3']))
        story.append(_space("micro_gap"))
        story.append(risk_register)
    story.append(_space("section_gap"))

    # Strategic Options Analysis (if pack available)
    if pack:
        story.append(Paragraph("Strategic Options Analysis", styles['Heading2']))
        story.append(_space("heading_to_content"))
        story.append(Paragraph(
            "Strategic options were evaluated using risk-adjusted scoring. "
            "The recommended option balances downside protection with expected value creation.",
            styles['Normal']
        ))
        story.append(Paragraph(
            "CVaR10 reflects the expected downside risk in the worst 10% of scenarios, "
            "providing a conservative measure of execution risk.",
            styles['Normal']
        ))
        story.append(_space("block_gap"))

        options_section = _generate_options_section(plan, pack, as_of=as_of)
        if options_section:
            _append_formatted_lines(
                story,
                options_section,
                heading2_style=heading2_style,
                heading3_style=heading3_style,
                heading4_style=heading4_style,
                body_style=body_style,
                bullet_style=bullet_style,
                numbered_style=numbered_style,
            )
            story.append(_space("block_gap"))
        story.append(_space("block_gap"))

        # Options Scoring Table
        scoring_table = _generate_options_scoring_table(plan, pack)
        if scoring_table:
            story.append(scoring_table)
            story.append(_space("section_gap"))
    
    # KPI Progress
    story.append(Paragraph("Key Performance Indicator Progress", styles['Heading2']))
    story.append(_space("heading_to_content"))
    story.extend(_generate_kpi_progress_sections(
        plan,
        objectives,
        period_data,
        chart_paths,
        pack,
        heading_style=heading3_style,
        body_style=body_style,
    ))

    # Overall KPI trends
    story.append(Paragraph("Overall KPI Trends", styles['Heading2']))
    story.append(_space("heading_to_content"))
    overall_chart = chart_paths.get('_overall')
    if overall_chart and Path(overall_chart).exists():
        img = Image(str(overall_chart), width=6*inch, height=4*inch)
        story.append(img)
    story.append(PageBreak())

    # Recommendations & Next Steps
    story.append(Paragraph("Recommendations and Next Steps", styles['Heading2']))
    story.append(_space("heading_to_content"))
    for item in _generate_next_steps(plan, objectives, period_data, pack):
        story.append(Paragraph(f"• {item}", styles['Normal']))
    story.append(_space("section_gap"))

    # Conclusion
    conclusion = _generate_conclusion(plan, objectives, period_data, pack)
    story.append(Paragraph("Conclusion", styles['Heading2']))
    story.append(_space("heading_to_content"))
    story.append(Paragraph(conclusion, styles['Normal']))

    # Technical appendix for KPI governance definitions/formulas
    story.append(PageBreak())
    story.append(Paragraph("Appendix — KPI Governance Definitions", styles['Heading2']))
    story.append(_space("heading_to_content"))
    story.append(Paragraph(
        "This appendix provides KPI technical definitions and formulas for reference. "
        "The executive body intentionally focuses on decision, risk, and business impact.",
        styles['Normal'],
    ))
    story.append(_space("block_gap"))
    story.extend(_generate_kpi_definitions_appendix(plan, objectives))
    
    # Build PDF
    def _on_page(c: canvas.Canvas, d: SimpleDocTemplate) -> None:
        c.saveState()
        c.setFont("Helvetica", 8)
        c.setFillColor(colors.HexColor("#64748B"))
        c.drawRightString(A4[0] - 20, 18, f"Page {c.getPageNumber()}")
        c.restoreState()

    doc.build(story, onFirstPage=_on_page, onLaterPages=_on_page)
    
    # Cleanup chart files
    for chart_path in chart_paths.values():
        if chart_path and Path(chart_path).exists():
            try:
                Path(chart_path).unlink()
            except:
                pass
    
    return str(output_path)


def _aggregate_by_period(
    kpi_df: pd.DataFrame,
    init_df: pd.DataFrame,
    as_of: str,
    period_type: str,
    plan: Any,
) -> Dict[str, Any]:
    """Aggregate data by period type."""
    as_of_dt = pd.to_datetime(as_of)
    
    # Determine period boundaries
    if period_type == "monthly":
        periods = pd.period_range(start=kpi_df['date'].min(), end=as_of_dt, freq='M')
    elif period_type == "quarterly":
        periods = pd.period_range(start=kpi_df['date'].min(), end=as_of_dt, freq='Q')
    elif period_type == "yearly":
        periods = pd.period_range(start=kpi_df['date'].min(), end=as_of_dt, freq='Y')
    else:
        periods = pd.period_range(start=kpi_df['date'].min(), end=as_of_dt, freq='M')
    
    # Aggregate KPIs by period
    kpi_df['period'] = pd.to_datetime(kpi_df['date']).dt.to_period(period_type[0].upper())
    kpi_agg = kpi_df.groupby(['period', 'kpi_id'])['value'].mean().reset_index()
    
    # Aggregate initiatives if available
    init_agg = pd.DataFrame()
    if not init_df.empty and 'progress' in init_df.columns:
        init_df['period'] = pd.to_datetime(init_df['date']).dt.to_period(period_type[0].upper())
        init_agg = init_df.groupby(['period', 'initiative_id'])['progress'].mean().reset_index()
    
    return {
        'kpi_data': kpi_agg,
        'initiative_data': init_agg,
        'periods': periods,
        'period_type': period_type,
    }


def _append_formatted_lines(
    story: List[Any],
    text: str,
    heading2_style: ParagraphStyle,
    heading3_style: ParagraphStyle,
    heading4_style: ParagraphStyle,
    body_style: ParagraphStyle,
    bullet_style: ParagraphStyle,
    numbered_style: ParagraphStyle,
) -> None:
    """Render markdown-like lines into reportlab elements."""
    for raw_line in text.split('\n'):
        line = raw_line.strip()
        if not line:
            story.append(_space("micro_gap"))
            continue

        if line.startswith("#### "):
            story.append(Paragraph(line[5:].strip(), heading4_style))
            continue
        if line.startswith("### "):
            story.append(Paragraph(line[4:].strip(), heading4_style))
            continue
        if line.startswith("## "):
            story.append(Paragraph(line[3:].strip(), heading3_style))
            continue
        if line.startswith("# "):
            story.append(Paragraph(line[2:].strip(), heading2_style))
            continue

        if line.startswith(("• ", "- ")):
            story.append(Paragraph(f"• {line[2:].strip()}", bullet_style))
            continue

        if line[:2].isdigit() and line[2:4] == ". ":
            story.append(Paragraph(line, numbered_style))
            continue
        if line[:1].isdigit() and line[1:3] == ". ":
            story.append(Paragraph(line, numbered_style))
            continue

        story.append(Paragraph(line, body_style))


def _get_period_range(kpi_df: pd.DataFrame) -> Tuple[str, str]:
    if kpi_df is None or kpi_df.empty or "date" not in kpi_df.columns:
        return "N/A", "N/A"
    dates = pd.to_datetime(kpi_df["date"], errors="coerce").dropna()
    if dates.empty:
        return "N/A", "N/A"
    return dates.min().date().isoformat(), dates.max().date().isoformat()


def _get_best_case_score(opt: Dict[str, Any], plan: Any) -> str:
    """Calculate best case (p90) portfolio score from p90 deltas."""
    from src.portfolio import score_option

    stress_summary = opt.get('stress_delta_summary', {})
    if not stress_summary:
        return f"{opt.get('score_base_mean', opt.get('score_stress_mean', 0)):.3f}"

    p90_deltas = {}
    for kpi_id, summary in stress_summary.items():
        p90_deltas[kpi_id] = float(summary.get('p90', 0.0))

    try:
        p90_utility = score_option(plan, p90_deltas)
        mean_deltas = {kpi_id: summary.get('mean', 0.0) for kpi_id, summary in stress_summary.items()}
        mean_utility = score_option(plan, mean_deltas)
        stress_mean = opt.get('score_stress_mean', 0)
        best_case_score = stress_mean + (p90_utility - mean_utility)
        return f"{best_case_score:.3f}"
    except Exception:
        return f"{opt.get('score_base_mean', opt.get('score_stress_mean', 0)):.3f}"

def _get_overall_status_counts(plan: Any, objectives: List[Any], period_data: Dict[str, Any]) -> Tuple[int, int]:
    """Return counts of on-track vs requiring attention KPIs."""
    kpi_agg = period_data['kpi_data']
    on_track = 0
    at_risk = 0
    shown_actual = False
    shown_baseline = False
    shown_fc = False
    shown_rec = False
    for obj in objectives:
        for okr in obj.okrs:
            kpi_data = kpi_agg[kpi_agg['kpi_id'] == okr.kpi_id]
            if kpi_data.empty:
                continue
            latest_value = kpi_data['value'].iloc[-1]
            target = float(okr.target) if okr.target else None
            if target:
                baseline = float(okr.baseline) if okr.baseline else latest_value
                if okr.direction == "up":
                    progress_pct = ((latest_value - baseline) / (target - baseline) * 100) if (target - baseline) > 0 else 0
                    is_on_track = latest_value >= target or (progress_pct >= 75 and latest_value > baseline)
                else:
                    progress_pct = ((baseline - latest_value) / (baseline - target) * 100) if (baseline - target) > 0 else 0
                    is_on_track = latest_value <= target or (progress_pct >= 75 and latest_value < baseline)
                if is_on_track:
                    on_track += 1
                else:
                    at_risk += 1
    return on_track, at_risk


def _get_recommendation_summary(pack: Optional[Dict[str, Any]]) -> Tuple[Optional[str], Optional[float]]:
    if not pack:
        return None, None
    rec = pack.get('recommendation', {})
    name = rec.get('chosen_bundle_name')
    conf = rec.get('confidence')
    try:
        conf = float(conf) if conf is not None else None
    except Exception:
        conf = None
    return name, conf


def _safe_text(value: Any) -> str:
    """Escape text for reportlab Paragraph HTML parser."""
    return escape(str(value or ""))


def _kpi_semantic_context(kpi: Any, max_exclusions: int = 3) -> Dict[str, str]:
    """Extract KPI business semantics from plan metadata for clearer report wording."""
    definition = getattr(kpi, "definition", None)
    desc = _safe_text(getattr(definition, "description", ""))
    formula = _safe_text(getattr(definition, "formula", ""))
    exclusions = getattr(definition, "exclusions", []) or []
    exclusions_text = ", ".join([_safe_text(x) for x in exclusions[:max_exclusions]])
    return {
        "name": _safe_text(getattr(kpi, "name", "")),
        "short_name": _safe_text(getattr(kpi, "short_name", "")),
        "description": desc,
        "formula": formula,
        "exclusions": exclusions_text,
    }


def _classify_kpi_status(
    latest_value: float,
    baseline: float,
    target: Optional[float],
    direction: str,
) -> str:
    """Classify KPI status for executive dashboard usage."""
    if target is None:
        return "AMBER"
    if direction == "up":
        progress_pct = ((latest_value - baseline) / (target - baseline) * 100) if (target - baseline) > 0 else 0
        if latest_value >= target or progress_pct >= 80:
            return "GREEN"
        if progress_pct >= 60:
            return "AMBER"
        return "RED"
    progress_pct = ((baseline - latest_value) / (baseline - target) * 100) if (baseline - target) > 0 else 0
    if latest_value <= target or progress_pct >= 80:
        return "GREEN"
    if progress_pct >= 60:
        return "AMBER"
    return "RED"


def _estimate_decision_urgency(
    at_risk_count: int,
    guardrails_failed: bool,
    alert_count: int,
) -> str:
    if guardrails_failed:
        return "HIGH (guardrail review required)"
    if at_risk_count >= 2 or alert_count >= 2:
        return "HIGH"
    if at_risk_count == 1 or alert_count == 1:
        return "MEDIUM"
    return "LOW"


def _format_confidence_label(conf: Optional[float]) -> str:
    """Executive-safe confidence wording (avoid absolute certainty phrasing)."""
    if conf is None:
        return "N/A"
    c = float(conf)
    if c >= 0.95:
        return "Very high (model confidence >95%)"
    if c >= 0.80:
        return "High"
    if c >= 0.60:
        return "Moderate"
    return "Low"


def _estimate_action_resource_cost(action_type: str, params: Dict[str, Any]) -> str:
    """Approximate resource/cost implication for governance discussion."""
    if action_type == "add_capacity":
        gain = float(params.get("capacity_gain", 0.0) or 0.0)
        est_budget_k = max(0.0, gain * 50.0)  # aligned with guardrails approximation
        est_fte = max(0.0, gain * 12.0)       # annualized effective capacity proxy
        return f"Resource implication: +{est_fte:.1f} FTE equivalent; budget impact ~+{est_budget_k:.0f} EUR_k (annualized)."
    if action_type == "accelerate_initiative":
        months = float(params.get("months", 0.0) or 0.0)
        return f"Resource implication: timeline compression of {months:.0f} month(s), typically requiring temporary surge staffing over 1-2 quarters."
    if action_type == "split_scope":
        reduction_pct = float(params.get("reduction", 0.0) or 0.0) * 100.0
        return f"Resource implication: delivery load reduced by ~{reduction_pct:.0f}% in-cycle, with limited incremental budget."
    return "Resource implication: confirm staffing and budget impact during SteerCo approval."


def _budget_constraint_info(plan: Any) -> Tuple[Optional[Dict[str, float]], Optional[str]]:
    """Return budget year limits and currency from portfolio constraints if available."""
    for c in getattr(getattr(plan, "portfolio", None), "constraints", []) or []:
        if str(getattr(c, "type", "")) == "budget":
            return getattr(c, "year_limits", None), getattr(c, "currency", None)
    return None, None


def _estimate_bundle_budget_envelope(
    plan: Any,
    actions: List[Dict[str, Any]],
) -> Tuple[float, float, str]:
    """
    Estimate baseline and adjusted budget envelope for targeted initiatives.
    Uses initiative `budget.amount_k` and action-level adjustments aligned with guardrail assumptions.
    """
    init_map = {init.id: init for init in getattr(plan, "initiatives", []) or []}
    targeted_ids: List[str] = []
    for a in actions or []:
        tid = str(a.get("target_initiative", "") or "")
        if tid and tid in init_map and tid not in targeted_ids:
            targeted_ids.append(tid)

    if not targeted_ids:
        return 0.0, 0.0, "EUR_k"

    adjusted_by_init: Dict[str, float] = {}
    currency = None
    for tid in targeted_ids:
        init = init_map[tid]
        base = float(getattr(getattr(init, "budget", None), "amount_k", 0.0) or 0.0)
        adjusted_by_init[tid] = base
        if currency is None:
            currency = str(getattr(getattr(init, "budget", None), "currency", "") or "")

    for a in actions or []:
        tid = str(a.get("target_initiative", "") or "")
        if tid not in adjusted_by_init:
            continue
        a_type = str(a.get("type", "") or "")
        params = a.get("parameters", {}) or {}
        if a_type == "split_scope":
            reduction = float(params.get("reduction", 0.0) or 0.0)
            reduction = max(0.0, min(0.95, reduction))
            adjusted_by_init[tid] = adjusted_by_init[tid] * (1.0 - reduction)
        elif a_type == "add_capacity":
            gain = float(params.get("capacity_gain", 0.0) or 0.0)
            adjusted_by_init[tid] = adjusted_by_init[tid] + max(0.0, gain * 50.0)

    baseline_total = 0.0
    for tid in targeted_ids:
        init = init_map[tid]
        baseline_total += float(getattr(getattr(init, "budget", None), "amount_k", 0.0) or 0.0)
    adjusted_total = sum(float(v) for v in adjusted_by_init.values())
    return baseline_total, adjusted_total, (currency or "EUR_k")


def _generate_critical_risk_statements(
    plan: Any,
    objectives: List[Any],
    forecast_df: Optional[pd.DataFrame],
) -> List[str]:
    """Translate forecast gaps into executive risk statements."""
    if forecast_df is None or forecast_df.empty:
        return []
    df = forecast_df.copy()
    if "date" not in df.columns or "forecast" not in df.columns or "expected" not in df.columns:
        return []

    kpi_map = {k.id: k for k in plan.kpis}
    okr_map: Dict[str, Any] = {}
    for obj in objectives:
        for okr in obj.okrs:
            okr_map.setdefault(okr.kpi_id, okr)

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date")
    statements: List[str] = []
    for kpi_id, grp in df.groupby("kpi_id"):
        kpi = kpi_map.get(kpi_id)
        okr = okr_map.get(kpi_id)
        if not kpi or not okr:
            continue
        last = grp.iloc[-1]
        expected = float(last.get("expected", 0.0))
        forecast = float(last.get("forecast", 0.0))
        if expected == 0:
            continue

        if okr.direction == "up":
            gap_ratio = (expected - forecast) / abs(expected)
            is_risk = gap_ratio > 0.05
        else:
            gap_ratio = (forecast - expected) / abs(expected)
            is_risk = gap_ratio > 0.05
        if not is_risk:
            continue

        sev = "RED" if gap_ratio > 0.12 else "AMBER"
        risk_msg = (
            f"{sev} {kpi.short_name}: forecast trajectory is {gap_ratio * 100:.0f}% "
            f"off plan; target miss risk is increasing within the next reporting cycles."
        )
        statements.append(risk_msg)
    return statements[:5]


def _generate_critical_issues_table(
    plan: Any,
    objectives: List[Any],
    forecast_df: Optional[pd.DataFrame],
) -> Optional[Table]:
    """Build a table of forecast vs expected gaps for near and far horizon."""
    if forecast_df is None or forecast_df.empty:
        return None
    df = forecast_df.copy()
    if "date" not in df.columns or "forecast" not in df.columns or "expected" not in df.columns:
        return None

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")
    kpi_map = {k.id: k for k in plan.kpis}
    okr_map = {}
    for obj in objectives:
        for okr in obj.okrs:
            okr_map.setdefault(okr.kpi_id, okr)

    rows = []
    adverse_flags: List[bool] = []
    for kpi_id, grp in df.groupby("kpi_id"):
        if kpi_id not in kpi_map or kpi_id not in okr_map:
            continue
        kpi = kpi_map[kpi_id]
        okr = okr_map[kpi_id]
        grp = grp.sort_values("date")
        near = grp.iloc[0]
        far = grp.iloc[-1]

        def _adverse_gap_pct(forecast_val, expected_val):
            if expected_val == 0:
                return 0.0
            if okr.direction == "down":
                return ((forecast_val - expected_val) / abs(expected_val)) * 100
            return ((expected_val - forecast_val) / abs(expected_val)) * 100

        near_gap = _adverse_gap_pct(near['forecast'], near['expected'])
        far_gap = _adverse_gap_pct(far['forecast'], far['expected'])

        rows.append([
            f"{kpi.short_name} (Near Term)",
            f"{near['forecast']:.2f} {kpi.unit}",
            f"{near['expected']:.2f} {kpi.unit}",
            f"{near_gap:+.0f}% adverse",
        ])
        adverse_flags.append(near_gap > 0.0)
        rows.append([
            f"{kpi.short_name} (Mid Term)",
            f"{far['forecast']:.2f} {kpi.unit}",
            f"{far['expected']:.2f} {kpi.unit}",
            f"{far_gap:+.0f}% adverse",
        ])
        adverse_flags.append(far_gap > 0.0)

    if not rows:
        return None

    table_data = [["Metric", "Forecast", "Expected", "Adverse Gap*"]] + rows
    t = Table(table_data, colWidths=[2.5 * inch, 1.4 * inch, 1.4 * inch, 0.8 * inch])
    style_cmds = [
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#EEF2FF')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.HexColor('#1E3A8A')),
        ('ALIGN', (1, 0), (-1, -1), 'CENTER'),
        ('ALIGN', (0, 0), (0, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 9),
        ('FONTSIZE', (0, 1), (-1, -1), 9),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#CBD5F5')),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#F8FAFC')]),
    ]
    for idx, is_adverse in enumerate(adverse_flags, start=1):
        if is_adverse:
            style_cmds.append(('TEXTCOLOR', (3, idx), (3, idx), colors.HexColor('#B91C1C')))
            style_cmds.append(('FONTNAME', (3, idx), (3, idx), 'Helvetica-Bold'))
        else:
            style_cmds.append(('TEXTCOLOR', (3, idx), (3, idx), colors.HexColor('#166534')))
    t.setStyle(TableStyle(style_cmds))
    return t


def _generate_kpi_progress_sections(
    plan: Any,
    objectives: List[Any],
    period_data: Dict[str, Any],
    chart_paths: Dict[str, str],
    pack: Optional[Dict[str, Any]] = None,
    heading_style: Optional[ParagraphStyle] = None,
    body_style: Optional[ParagraphStyle] = None,
) -> List[Any]:
    """Build KPI progress section flowables."""
    out: List[Any] = []
    styles = getSampleStyleSheet()
    heading_style = heading_style or styles['Heading3']
    body_style = body_style or styles['Normal']
    kpi_agg = period_data['kpi_data']
    kpi_map = {k.id: k for k in plan.kpis}
    options_explain = pack.get('options_explain', {}) if pack else {}

    for idx, obj in enumerate(objectives, 1):
        out.append(Paragraph(f"{idx}. {obj.name}", heading_style))
        out.append(_space("micro_gap"))
        out.append(Paragraph(f"Objective: {obj.description}", body_style))
        out.append(_space("block_gap"))

        for okr in obj.okrs:
            kpi = kpi_map.get(okr.kpi_id)
            if not kpi:
                continue
            kpi_data = kpi_agg[kpi_agg['kpi_id'] == okr.kpi_id].copy()
            if kpi_data.empty:
                continue

            kpi_data = kpi_data.sort_values('period')
            latest_value = kpi_data['value'].iloc[-1]
            first_value = kpi_data['value'].iloc[0] if len(kpi_data) > 1 else latest_value
            baseline = float(okr.baseline) if okr.baseline else first_value
            target = float(okr.target) if okr.target else latest_value

            if okr.direction == "up":
                progress_pct = ((latest_value - baseline) / (target - baseline) * 100) if (target - baseline) > 0 else 0
                status = "In Progress" if latest_value > baseline else "Requires Attention"
            else:
                progress_pct = ((baseline - latest_value) / (baseline - target) * 100) if (baseline - target) > 0 else 0
                status = "In Progress" if latest_value < baseline else "Requires Attention"

            change = latest_value - first_value
            change_pct = (change / first_value * 100) if first_value != 0 else 0

            driver_text = ""
            if options_explain and pack:
                options_mc = pack.get('options_mc', [])
                if options_mc:
                    top_option = sorted(options_mc, key=lambda x: x.get('score_stress_cvar10', 0), reverse=True)[0]
                    top_bundle_id = top_option.get('bundle_id')
                    if top_bundle_id and top_bundle_id in options_explain:
                        attr_data = options_explain[top_bundle_id]
                        kpi_attrs = attr_data.get('kpi_attributions', {})
                        if okr.kpi_id in kpi_attrs:
                            drivers = kpi_attrs[okr.kpi_id].get('drivers', [])
                            if drivers:
                                top_driver = sorted(drivers, key=lambda d: abs(d.get('contribution', 0)), reverse=True)[0]
                                driver_text = top_driver.get('description', '')

            data = [
                ["Baseline", "Current", "Target", "Progress"],
                [f"{baseline:.2f} {kpi.unit}", f"{latest_value:.2f} {kpi.unit}", f"{target:.2f} {kpi.unit}", f"{progress_pct:.1f}%"],
            ]
            t = Table(data, colWidths=[1.5 * inch, 1.5 * inch, 1.5 * inch, 1.0 * inch])
            t.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#EEF2FF')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.HexColor('#1E3A8A')),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 9),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#CBD5F5')),
            ]))
            sem = _kpi_semantic_context(kpi)
            out.append(Paragraph(f"<b>{sem['short_name']}</b> — Status: {status}", body_style))
            if sem["name"]:
                out.append(Paragraph(f"<b>KPI:</b> {sem['name']}", body_style))
            if sem["description"]:
                out.append(Paragraph(f"<b>What it measures:</b> {sem['description']}", body_style))
            if sem["exclusions"]:
                out.append(Paragraph(f"<b>Exclusions:</b> {sem['exclusions']}", body_style))
            out.append(_space("micro_gap"))
            out.append(t)
            out.append(_space("block_gap"))
            out.append(Paragraph(
                f"Period Change: {change:+.2f} {kpi.unit} ({change_pct:+.1f}%).",
                body_style
            ))
            if driver_text:
                out.append(Paragraph(f"Key Driver: {driver_text}", body_style))
            chart_path = chart_paths.get(okr.kpi_id)
            if chart_path and Path(chart_path).exists():
                out.append(_space("chart_pre_gap"))
                out.append(Image(str(chart_path), width=5.8*inch, height=3.6*inch))
            out.append(_space("chart_post_gap"))

    return out


def _generate_next_steps(
    plan: Any,
    objectives: List[Any],
    period_data: Dict[str, Any],
    pack: Optional[Dict[str, Any]] = None,
) -> List[str]:
    steps: List[str] = []
    rec_name, _ = _get_recommendation_summary(pack)
    if rec_name:
        steps.append(f"Implement {rec_name} to address the most critical forecast gaps.")

    # Identify lowest-progress KPI
    kpi_agg = period_data['kpi_data']
    kpi_map = {k.id: k for k in plan.kpis}
    lowest = None
    for obj in objectives:
        for okr in obj.okrs:
            kpi_data = kpi_agg[kpi_agg['kpi_id'] == okr.kpi_id]
            if kpi_data.empty:
                continue
            latest_value = kpi_data['value'].iloc[-1]
            baseline = float(okr.baseline) if okr.baseline else latest_value
            target = float(okr.target) if okr.target else latest_value
            if okr.direction == "up":
                progress_pct = ((latest_value - baseline) / (target - baseline) * 100) if (target - baseline) > 0 else 0
            else:
                progress_pct = ((baseline - latest_value) / (baseline - target) * 100) if (baseline - target) > 0 else 0
            if lowest is None or progress_pct < lowest[0]:
                kpi = kpi_map.get(okr.kpi_id)
                if kpi:
                    lowest = (progress_pct, kpi.short_name)
    if lowest:
        lowest_kpi_id = None
        for obj in objectives:
            for okr in obj.okrs:
                kpi = kpi_map.get(okr.kpi_id)
                if kpi and kpi.short_name == lowest[1]:
                    lowest_kpi_id = okr.kpi_id
                    break
            if lowest_kpi_id:
                break
        if lowest_kpi_id and lowest_kpi_id in kpi_map:
            low_kpi = kpi_map[lowest_kpi_id]
            sem = _kpi_semantic_context(low_kpi)
            steps.append(
                f"Prioritize interventions for {sem['name']} ({sem['short_name']}) given the lowest progress to target."
            )
            if sem["formula"]:
                steps.append(f"Use the KPI formula as primary tracking reference: {sem['formula']}.")
            if sem["exclusions"]:
                steps.append(f"Validate exclusion handling to keep KPI interpretation consistent: {sem['exclusions']}.")
        else:
            steps.append(f"Prioritize interventions for {lowest[1]} given the lowest progress to target.")

    steps.append("Review forecast vs expected gaps monthly and adjust initiatives as needed.")
    steps.append("Maintain momentum on KPIs with positive trend improvements to avoid regression.")
    return steps


def _generate_conclusion(
    plan: Any,
    objectives: List[Any],
    period_data: Dict[str, Any],
    pack: Optional[Dict[str, Any]] = None,
) -> str:
    on_track, at_risk = _get_overall_status_counts(plan, objectives, period_data)
    rec_name, _ = _get_recommendation_summary(pack)
    kpi_agg = period_data['kpi_data']
    kpi_map = {k.id: k for k in plan.kpis}
    at_risk_semantics: List[str] = []
    for obj in objectives:
        for okr in obj.okrs:
            kpi_data = kpi_agg[kpi_agg['kpi_id'] == okr.kpi_id]
            if kpi_data.empty:
                continue
            latest_value = kpi_data['value'].iloc[-1]
            baseline = float(okr.baseline) if okr.baseline else latest_value
            target = float(okr.target) if okr.target else None
            if target is None:
                continue
            if okr.direction == "up":
                progress_pct = ((latest_value - baseline) / (target - baseline) * 100) if (target - baseline) > 0 else 0
                is_on_track = latest_value >= target or (progress_pct >= 75 and latest_value > baseline)
            else:
                progress_pct = ((baseline - latest_value) / (baseline - target) * 100) if (baseline - target) > 0 else 0
                is_on_track = latest_value <= target or (progress_pct >= 75 and latest_value < baseline)
            if not is_on_track:
                kpi = kpi_map.get(okr.kpi_id)
                if not kpi:
                    continue
                sem = _kpi_semantic_context(kpi, max_exclusions=2)
                at_risk_semantics.append(sem["name"] or sem["short_name"])

    parts: List[str] = []
    parts.append(
        f"Steering synthesis: {on_track} KPI(s) are on track and {at_risk} KPI(s) require intervention."
    )
    if rec_name:
        parts.append(
            f"The recommended route is {rec_name}, prioritizing downside protection while preserving expected value."
        )
    if at_risk_semantics:
        focus_list = "; ".join(at_risk_semantics[:3])
        parts.append(f"Primary execution pressure remains on: {focus_list}.")

    if pack:
        rec = (pack.get("recommendation", {}) or {})
        chosen_id = str(rec.get("chosen_bundle_id", "") or "")
        options_mc = pack.get("options_mc", []) or []
        chosen_opt = next((o for o in options_mc if str(o.get("bundle_id", "")) == chosen_id), None)
        if chosen_opt is not None:
            cvar10 = float(chosen_opt.get("score_stress_cvar10", 0.0))
            if cvar10 >= 0.20:
                parts.append("Forward-looking position: confidence is strong that period-end targets remain attainable under the recommended option.")
            elif cvar10 >= 0.05:
                parts.append("Forward-looking position: target attainment remains plausible, but execution quality must remain high over the next two cycles.")
            else:
                parts.append("Forward-looking position: confidence in period-end target attainment is limited unless mitigation intensity increases immediately.")
        bundle = next((b for b in (pack.get("bundles", []) or []) if str(b.get("id", "")) == chosen_id), None)
        if bundle:
            init_map = {i.id: i for i in getattr(plan, "initiatives", []) or []}
            owners: List[str] = []
            for a in bundle.get("actions", []) or []:
                tid = str(a.get("target_initiative", "") or "")
                if tid in init_map:
                    owners.append(str(getattr(init_map[tid], "owner", "Program Lead")))
            owners = sorted(set([o for o in owners if o]))
            base_k, adj_k, ccy = _estimate_bundle_budget_envelope(plan, bundle.get("actions", []) or [])
            if base_k > 0:
                parts.append(
                    f"Financial posture: execution envelope moves from {base_k:.0f} to {adj_k:.0f} {ccy} ({adj_k - base_k:+.0f})."
                )
            if owners:
                parts.append(f"Execution ownership is assigned to {', '.join(owners[:3])}.")
            else:
                parts.append("Execution ownership should be formally assigned in the next steering cycle.")

    next_date = (
        (pd.to_datetime(period_data["periods"][-1].to_timestamp()) + pd.Timedelta(days=60)).date().isoformat()
        if period_data.get("periods") is not None and len(period_data.get("periods")) > 0
        else "next cycle"
    )
    parts.append(
        f"Next checkpoint: confirm mitigation progress and forecast recalibration by {next_date}, with explicit owner-level accountability."
    )
    return " ".join(parts)


def _resolve_forecast_df(
    plan: Any,
    kpi_df: pd.DataFrame,
    init_df: pd.DataFrame,
    as_of: str,
    pack: Optional[Dict[str, Any]] = None,
) -> Optional[pd.DataFrame]:
    """Resolve a full forecast DataFrame for charts, recomputing if needed."""
    if pack:
        forecast_full = pack.get('forecast_full', None)
        if forecast_full:
            df = pd.DataFrame(forecast_full)
            if not df.empty:
                return df

        forecast_head = pack.get('forecast_head', None)
        if forecast_head:
            df = pd.DataFrame(forecast_head)
            if not df.empty:
                return df

    try:
        from src.forecast import ForecastConfig, forecast_kpis
        horizon_end = pack.get('horizon_end') if pack else None
        if not horizon_end:
            horizon_end = as_of
        initiatives_df = init_df if init_df is not None and not init_df.empty else None
        return forecast_kpis(
            plan=plan,
            kpis_df=kpi_df[["date", "kpi_id", "value"]],
            as_of=as_of,
            horizon_end=horizon_end,
            initiatives_df=initiatives_df,
            cfg=ForecastConfig(),
        )
    except Exception:
        return None


def _period_to_timestamp(period_series: pd.Series) -> pd.Series:
    """Convert period-like series to timestamps for stable plotting on x-axis."""
    try:
        if hasattr(period_series, "dt"):
            return period_series.dt.to_timestamp()
    except Exception:
        pass
    return pd.to_datetime(period_series.astype(str), errors="coerce")


def _resolve_recommended_option_forecast_df(
    plan: Any,
    kpi_df: pd.DataFrame,
    init_df: pd.DataFrame,
    as_of: str,
    forecast_df: Optional[pd.DataFrame],
    pack: Optional[Dict[str, Any]],
) -> pd.DataFrame:
    """
    Build recommended-option forecast aligned to model forecast horizon.
    Uses the same what-if projected `with_action` trajectory logic as dashboard.
    """
    if pack is None or forecast_df is None or forecast_df.empty:
        return pd.DataFrame()

    rec = (pack.get("recommendation", {}) or {})
    chosen_id = str(rec.get("chosen_bundle_id", "") or "")
    if not chosen_id or chosen_id == "NONE":
        return pd.DataFrame()

    bundles = pack.get("bundles", []) or []
    bundle = next((b for b in bundles if str(b.get("id", "")) == chosen_id), None)
    if not bundle:
        return pd.DataFrame()

    try:
        from src.actions import SteeringAction
        from src.whatif import compare_actions_at_dates
    except Exception:
        return pd.DataFrame()

    actions = []
    for a in bundle.get("actions", []) or []:
        try:
            raw_params = (a.get("parameters", {}) or {})
            parsed_params: Dict[str, Any] = {}
            for k, v in raw_params.items():
                if isinstance(v, (int, float)):
                    parsed_params[k] = float(v)
                else:
                    try:
                        parsed_params[k] = float(v)
                    except (TypeError, ValueError):
                        parsed_params[k] = v

            actions.append(
                SteeringAction(
                    id=str(a.get("id", "")),
                    type=str(a.get("type", "accelerate_initiative")),
                    target_initiative=str(a.get("target_initiative", "")),
                    parameters=parsed_params,
                    description=str(a.get("description", "")),
                )
            )
        except Exception:
            continue

    if not actions:
        return pd.DataFrame()

    fc_tmp = forecast_df.copy()
    fc_tmp["date"] = pd.to_datetime(fc_tmp["date"], errors="coerce")
    fc_tmp = fc_tmp.dropna(subset=["date"])
    if fc_tmp.empty:
        return pd.DataFrame()
    dates_to_check = sorted(fc_tmp["date"].dt.strftime("%Y-%m-%d").unique().tolist())
    if not dates_to_check:
        return pd.DataFrame()

    try:
        sim_df = compare_actions_at_dates(
            plan=plan,
            kpi_history=kpi_df[["date", "kpi_id", "value"]],
            initiative_history=init_df,
            as_of=str(as_of),
            dates_to_check=dates_to_check,
            actions=actions,
            stress_events=[],
        )
    except Exception:
        return pd.DataFrame()

    if sim_df is None or sim_df.empty:
        return pd.DataFrame()

    sim_df = sim_df.copy()
    sim_df["date"] = pd.to_datetime(sim_df["date"], errors="coerce")
    sim_df = sim_df.dropna(subset=["date"])
    if sim_df.empty or "delta" not in sim_df.columns:
        return pd.DataFrame()

    merged = fc_tmp[["date", "kpi_id", "forecast"]].merge(
        sim_df[["date", "kpi_id", "with_action"]],
        on=["date", "kpi_id"],
        how="left",
    )
    merged["recommended_forecast"] = pd.to_numeric(merged["with_action"], errors="coerce")
    # If simulation is missing at a point, fall back to model forecast to keep series continuous.
    merged["recommended_forecast"] = merged["recommended_forecast"].fillna(pd.to_numeric(merged["forecast"], errors="coerce"))
    return merged[["date", "kpi_id", "recommended_forecast"]]


def _generate_charts(
    plan: Any,
    objectives: List[Any],
    period_data: Dict[str, Any],
    kpi_df: pd.DataFrame,
    init_df: pd.DataFrame,
    as_of: str,
    period_type: str,
    pack: Optional[Dict[str, Any]] = None,
) -> Dict[str, str]:
    """Generate charts for the report."""
    chart_paths = {}
    temp_dir = Path("artifacts/reports/temp_charts")
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    kpi_map = {k.id: k for k in plan.kpis}
    kpi_agg = period_data['kpi_data']
    forecast_df = _resolve_forecast_df(plan, kpi_df, init_df, as_of, pack)
    rec_option_fc_df = _resolve_recommended_option_forecast_df(plan, kpi_df, init_df, as_of, forecast_df, pack)
    as_of_ts = pd.to_datetime(as_of)
    freq = period_type[0].upper()
    as_of_period = as_of_ts.to_period(freq)
    as_of_x = as_of_period.to_timestamp()
    weekly_obj_df = pd.DataFrame(pack.get("weekly_objectives", []) if pack else [])
    if not weekly_obj_df.empty and {"date", "kpi_id", "baseline_expected"}.issubset(weekly_obj_df.columns):
        weekly_obj_df["date"] = pd.to_datetime(weekly_obj_df["date"])
        weekly_obj_df["period"] = weekly_obj_df["date"].dt.to_period(freq)
    
    # Chart for each KPI in objectives
    for obj in objectives:
        for okr in obj.okrs:
            kpi_id = okr.kpi_id
            kpi = kpi_map.get(kpi_id)
            if not kpi:
                continue
            
            # Filter data for this KPI
            kpi_data = kpi_agg[kpi_agg['kpi_id'] == kpi_id].copy().sort_values('period')
            if kpi_data.empty:
                continue
            
            # Create chart
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Actual: only up to as_of
            actual_agg = kpi_data[kpi_data['period'] <= as_of_period].copy().sort_values('period')
            periods = _period_to_timestamp(actual_agg['period'])
            values = actual_agg['value']
            if not actual_agg.empty:
                ax.plot(periods, values, marker='o', linewidth=2, markersize=8, label='Actual (observed)', color='#3b82f6')

            # Weekly plan baseline aggregated to report period
            if not weekly_obj_df.empty:
                wb_kpi = weekly_obj_df[weekly_obj_df["kpi_id"] == kpi_id].copy()
                if not wb_kpi.empty:
                    wb_agg = wb_kpi.groupby("period")["baseline_expected"].mean().reset_index().sort_values("period")
                    wb_periods = _period_to_timestamp(wb_agg["period"]).tolist()
                    wb_values = wb_agg["baseline_expected"].tolist()
                    if wb_periods:
                        ax.plot(
                            wb_periods,
                            wb_values,
                            linewidth=1.8,
                            label='Plan baseline (weekly objectives)',
                            color='#f59e0b',
                            linestyle=':',
                            alpha=0.9,
                        )
            
            # Add forecast data if available
            forecast_agg = pd.DataFrame()
            if forecast_df is not None and not forecast_df.empty:
                forecast_kpi = forecast_df[forecast_df['kpi_id'] == kpi_id].copy()
                if not forecast_kpi.empty:
                    forecast_kpi['date'] = pd.to_datetime(forecast_kpi['date'])
                    forecast_kpi['period'] = forecast_kpi['date'].dt.to_period(period_type[0].upper())
                    forecast_agg = forecast_kpi.groupby('period')['forecast'].mean().reset_index().sort_values('period')
                    forecast_agg = forecast_agg[forecast_agg['period'] > as_of_period]
                    if not actual_agg.empty and not forecast_agg.empty:
                        bridge_row = pd.DataFrame([{
                            'period': as_of_period,
                            'forecast': float(actual_agg['value'].iloc[-1]),
                        }])
                        forecast_agg = pd.concat([bridge_row, forecast_agg], ignore_index=True).sort_values('period')
                    forecast_agg['period_str'] = forecast_agg['period'].astype(str)
                    forecast_periods = _period_to_timestamp(forecast_agg["period"]).tolist()
                    forecast_values = forecast_agg['forecast'].tolist()
                    if forecast_periods:
                        ax.plot(forecast_periods, forecast_values, marker='s', linewidth=2,
                               markersize=6, label='Forecast (model)', color='#10b981', linestyle='--', alpha=0.8)

            # Add recommended option impacted forecast
            rec_agg = pd.DataFrame()
            if rec_option_fc_df is not None and not rec_option_fc_df.empty:
                rec_kpi = rec_option_fc_df[rec_option_fc_df["kpi_id"] == kpi_id].copy()
                if not rec_kpi.empty:
                    rec_kpi["period"] = rec_kpi["date"].dt.to_period(period_type[0].upper())
                    rec_agg = rec_kpi.groupby("period")["recommended_forecast"].mean().reset_index().sort_values("period")
                    rec_agg = rec_agg[rec_agg["period"] > as_of_period]
                    if not actual_agg.empty and not rec_agg.empty:
                        bridge_row = pd.DataFrame([{
                            "period": as_of_period,
                            "recommended_forecast": float(actual_agg["value"].iloc[-1]),
                        }])
                        rec_agg = pd.concat([bridge_row, rec_agg], ignore_index=True).sort_values("period")
                    rec_periods = _period_to_timestamp(rec_agg["period"]).tolist()
                    rec_values = rec_agg["recommended_forecast"].tolist()
                    if rec_periods:
                        ax.plot(
                            rec_periods,
                            rec_values,
                            marker='^',
                            linewidth=3.5,
                            markersize=5,
                            label='Forecast (recommended option impact)',
                            color='#a855f7',
                            linestyle='-.',
                            alpha=1.0,
                        )

            # Forecast spread note (recommended - baseline at horizon)
            spread_note = None
            try:
                if not forecast_agg.empty and not rec_agg.empty:
                    base_last = float(forecast_agg["forecast"].iloc[-1])
                    rec_last = float(rec_agg["recommended_forecast"].iloc[-1])
                    spread = rec_last - base_last
                    spread_note = f"Forecast spread @ horizon (recommended - baseline): {spread:+.3f}"
            except Exception:
                spread_note = None
            
            # Add target line if available
            if okr.target:
                target_val = float(okr.target)
                ax.axhline(y=target_val, color='r', linestyle='--', linewidth=2, label=f'Target: {target_val}')
            
            # Add baseline
            if okr.baseline:
                baseline_val = float(okr.baseline)
                ax.axhline(y=baseline_val, color='g', linestyle='--', linewidth=1, alpha=0.5, label=f'Baseline: {baseline_val}')
            
            ax.set_xlabel(f'Period ({period_type})', fontsize=12)
            ax.set_ylabel(f'{kpi.short_name} ({kpi.unit})', fontsize=12)
            ax.set_title(f'{kpi.short_name} Progress Over Time', fontsize=14, fontweight='bold')
            ax.axvline(x=as_of_x, color='#94A3B8', linestyle='--', linewidth=1, alpha=0.8)
            ax.set_facecolor('#FAFAFA')
            fig.patch.set_facecolor('white')
            ax.grid(True, alpha=0.15, linewidth=0.4, color='#CBD5E1')
            ax.set_axisbelow(True)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.legend(
                loc='upper left',
                fontsize=8,
                frameon=True,
                framealpha=0.9,
                edgecolor='#E2E8F0',
                fancybox=False,
            )
            plt.xticks(rotation=45, ha='right')
            if spread_note:
                ax.text(
                    0.01,
                    0.02,
                    spread_note,
                    transform=ax.transAxes,
                    fontsize=9,
                    color="#475569",
                    bbox=dict(boxstyle="round,pad=0.25", facecolor="#f8fafc", alpha=0.8, edgecolor="#cbd5e1"),
                )
            plt.tight_layout()
            
            chart_path = temp_dir / f"chart_{kpi_id}.png"
            fig.savefig(chart_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            
            chart_paths[kpi_id] = str(chart_path)
    
    # Overall KPI trend chart
    fig, ax = plt.subplots(figsize=(10, 6))
    shown_actual = False
    shown_baseline = False
    shown_fc = False
    shown_rec = False
    
    for obj in objectives:
        for okr in obj.okrs:
            kpi_id = okr.kpi_id
            kpi = kpi_map.get(kpi_id)
            if not kpi:
                continue
            
            kpi_data = kpi_agg[kpi_agg['kpi_id'] == kpi_id].copy().sort_values('period')
            if kpi_data.empty:
                continue
            
            actual_agg = kpi_data[kpi_data['period'] <= as_of_period].copy().sort_values('period')
            periods = _period_to_timestamp(actual_agg['period'])
            values = actual_agg['value']
            
            # Plot actual values
            if not actual_agg.empty:
                ax.plot(
                    periods,
                    values,
                    marker='o',
                    linewidth=1.2,
                    markersize=3,
                    color='#2563EB',
                    alpha=0.55,
                    label='Actual (observed)' if not shown_actual else '_nolegend_',
                )
                shown_actual = True

            # Plot weekly baseline (aggregated)
            if not weekly_obj_df.empty:
                wb_kpi = weekly_obj_df[weekly_obj_df["kpi_id"] == kpi_id].copy()
                if not wb_kpi.empty:
                    wb_agg = wb_kpi.groupby("period")["baseline_expected"].mean().reset_index().sort_values("period")
                    ax.plot(
                        _period_to_timestamp(wb_agg["period"]),
                        wb_agg["baseline_expected"],
                        linewidth=1.0,
                        linestyle=':',
                        alpha=0.55,
                        color='#B45309',
                        label='Plan baseline (weekly objectives)' if not shown_baseline else '_nolegend_'
                    )
                    shown_baseline = True
            
            # Add forecast if available
            if forecast_df is not None and not forecast_df.empty:
                forecast_kpi = forecast_df[forecast_df['kpi_id'] == kpi_id].copy()
                if not forecast_kpi.empty:
                    forecast_kpi['date'] = pd.to_datetime(forecast_kpi['date'])
                    forecast_kpi['period'] = forecast_kpi['date'].dt.to_period(period_type[0].upper())
                    forecast_agg = forecast_kpi.groupby('period')['forecast'].mean().reset_index().sort_values('period')
                    forecast_agg = forecast_agg[forecast_agg['period'] > as_of_period]
                    if not actual_agg.empty and not forecast_agg.empty:
                        bridge_row = pd.DataFrame([{
                            'period': as_of_period,
                            'forecast': float(actual_agg['value'].iloc[-1]),
                        }])
                        forecast_agg = pd.concat([bridge_row, forecast_agg], ignore_index=True).sort_values('period')
                    forecast_agg['period_str'] = forecast_agg['period'].astype(str)
                    forecast_periods = _period_to_timestamp(forecast_agg["period"]).tolist()
                    forecast_values = forecast_agg['forecast'].tolist()
                    if forecast_periods:
                        ax.plot(
                            forecast_periods,
                            forecast_values,
                            marker='s',
                            linewidth=1.8,
                            markersize=3,
                            color='#0F766E',
                            linestyle='--',
                            alpha=0.7,
                            label='Forecast (model)' if not shown_fc else '_nolegend_',
                        )
                        shown_fc = True

            # Add recommended option impacted forecast (overall)
            if rec_option_fc_df is not None and not rec_option_fc_df.empty:
                rec_kpi = rec_option_fc_df[rec_option_fc_df["kpi_id"] == kpi_id].copy()
                if not rec_kpi.empty:
                    rec_kpi["period"] = rec_kpi["date"].dt.to_period(period_type[0].upper())
                    rec_agg = rec_kpi.groupby("period")["recommended_forecast"].mean().reset_index().sort_values("period")
                    rec_agg = rec_agg[rec_agg["period"] > as_of_period]
                    if not actual_agg.empty and not rec_agg.empty:
                        bridge_row = pd.DataFrame([{
                            "period": as_of_period,
                            "recommended_forecast": float(actual_agg["value"].iloc[-1]),
                        }])
                        rec_agg = pd.concat([bridge_row, rec_agg], ignore_index=True).sort_values("period")
                    rec_periods = _period_to_timestamp(rec_agg["period"]).tolist()
                    rec_values = rec_agg["recommended_forecast"].tolist()
                    if rec_periods:
                        ax.plot(
                            rec_periods,
                            rec_values,
                            marker='^',
                            linewidth=3.5,
                            markersize=4,
                            label='Forecast (recommended option impact)' if not shown_rec else '_nolegend_',
                            color='#7C3AED',
                            linestyle='-.',
                            alpha=1.0,
                        )
                        shown_rec = True
    
    ax.set_xlabel(f'Period ({period_type})', fontsize=12)
    ax.set_ylabel('Value', fontsize=12)
    ax.set_title('Overall KPI Trends', fontsize=14, fontweight='bold')
    ax.axvline(x=as_of_x, color='#94A3B8', linestyle='--', linewidth=1, alpha=0.8)
    ax.set_facecolor('#FAFAFA')
    fig.patch.set_facecolor('white')
    ax.grid(True, alpha=0.15, linewidth=0.4, color='#CBD5E1')
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2, fontsize=8, frameon=False)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout(rect=[0, 0.1, 1, 1])
    
    overall_chart_path = temp_dir / "chart_overall.png"
    fig.savefig(overall_chart_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    chart_paths['_overall'] = str(overall_chart_path)
    
    return chart_paths




def _generate_enhanced_narrative(
    plan: Any,
    objectives: List[Any],
    period_data: Dict[str, Any],
    as_of: str,
    period_type: str,
    pack: Optional[Dict[str, Any]] = None,
) -> str:
    """Generate enhanced narrative with problems, actions, impact, and progress."""
    narrative_parts = []
    
    narrative_parts.append(f"This report covers progress from {period_data['periods'][0]} to {period_data['periods'][-1]} ({period_type} aggregation).")
    narrative_parts.append("")
    
    # Problem Identification Section
    if pack:
        narrative_parts.append("<b>Problem Identification</b>")
        problems = []
        
        # Check for anomalies
        ml_anomalies = pack.get('ml_anomalies', {})
        for kpi_id, anomaly_data in ml_anomalies.items():
            if anomaly_data.get('is_anomaly', False):
                kpi = next((k for k in plan.kpis if k.id == kpi_id), None)
                if kpi:
                    problems.append(f"Anomaly detected in {kpi.short_name}: {anomaly_data.get('explanation', 'Unusual pattern detected')}")
        
        # Check for integrity issues
        integrity_scores = pack.get('integrity_scores', {})
        for kpi_id, score in integrity_scores.items():
            if score > 0.6:  # High integrity score = suspicious
                kpi = next((k for k in plan.kpis if k.id == kpi_id), None)
                if kpi:
                    problems.append(f"Data quality concerns in {kpi.short_name} (integrity score: {score:.2f})")
        
        # Check for guardrail violations
        rec = pack.get('recommendation', {})
        if rec and not rec.get('guardrails_passed', True):
            violations = rec.get('guardrails_violations', [])
            for v in violations:
                problems.append(f"Guardrail violation: {v.get('message', 'Constraint not met')}")
        
        if problems:
            for problem in problems:
                narrative_parts.append(f"• {problem}")
        else:
            narrative_parts.append("No significant problems identified during this period.")
        narrative_parts.append("")
        
        # Actions Taken Section
        narrative_parts.append("<b>Actions Taken</b>")
        narrative_parts.append("")
        if rec:
            chosen_bundle = rec.get('chosen_bundle_name', 'N/A')
            why = rec.get('why', [])
            narrative_parts.append(f"<b>Recommended action:</b> {chosen_bundle}")
            narrative_parts.append("")
            if why:
                narrative_parts.append("<b>Rationale:</b>")
                narrative_parts.append("")
                for reason in why:
                    # Clean up the reason text - split by sentences if needed
                    reason_clean = reason.strip()
                    # If reason contains multiple sentences or bullet points, split them
                    if "•" in reason_clean or ". " in reason_clean:
                        # Split by bullet points first
                        if "•" in reason_clean:
                            parts = reason_clean.split("•")
                            for part in parts:
                                part = part.strip()
                                if part:
                                    narrative_parts.append(f"• {part}")
                        else:
                            # Split by sentences
                            sentences = reason_clean.split(". ")
                            for i, sent in enumerate(sentences):
                                sent = sent.strip()
                                if sent:
                                    if not sent.endswith(".") and i < len(sentences) - 1:
                                        sent += "."
                                    narrative_parts.append(f"• {sent}")
                    else:
                        narrative_parts.append(f"• {reason_clean}")
        else:
            narrative_parts.append("No specific actions recommended during this period.")
        narrative_parts.append("")
    
    # Progress Analysis Section
    narrative_parts.append("<b>Progress Analysis</b>")
    kpi_map = {k.id: k for k in plan.kpis}
    kpi_agg = period_data['kpi_data']
    
    for obj in objectives:
        narrative_parts.append(f"<b>{obj.name}</b>")
        narrative_parts.append(f"{obj.description}")
        narrative_parts.append("")
        
        for okr in obj.okrs:
            kpi_id = okr.kpi_id
            kpi = kpi_map.get(kpi_id)
            if not kpi:
                continue
            
            kpi_data = kpi_agg[kpi_agg['kpi_id'] == kpi_id].copy()
            if kpi_data.empty:
                continue
            
            # Calculate progress
            latest_value = kpi_data['value'].iloc[-1]
            first_value = kpi_data['value'].iloc[0] if len(kpi_data) > 1 else latest_value
            baseline = float(okr.baseline) if okr.baseline else first_value
            target = float(okr.target) if okr.target else latest_value
            
            # Calculate change over period
            change = latest_value - first_value
            change_pct = (change / first_value * 100) if first_value != 0 else 0
            
            if okr.direction == "up":
                progress_pct = ((latest_value - baseline) / (target - baseline) * 100) if (target - baseline) > 0 else 0
                status = "on track" if latest_value >= target else "in progress" if latest_value > baseline else "at risk"
                trend = "improving" if change > 0 else "declining" if change < 0 else "stable"
            else:
                progress_pct = ((baseline - latest_value) / (baseline - target) * 100) if (baseline - target) > 0 else 0
                status = "on track" if latest_value <= target else "in progress" if latest_value < baseline else "at risk"
                trend = "improving" if change < 0 else "declining" if change > 0 else "stable"
            
            # Impact analysis with attribution
            impact_text = ""
            drivers_text = ""
            if pack and abs(change) > 0.01:  # Significant change
                if trend == "improving":
                    impact_text = f" Positive trend observed: {kpi.short_name} {trend} by {abs(change):.2f} {kpi.unit} ({abs(change_pct):.1f}%) over the reporting period."
                elif trend == "declining":
                    impact_text = f" Negative trend observed: {kpi.short_name} {trend} by {abs(change):.2f} {kpi.unit} ({abs(change_pct):.1f}%) over the reporting period."
                
                # Add attribution/drivers if available
                options_explain = pack.get('options_explain', {})
                if options_explain:
                    # Get the top-ranked option's attribution
                    options_mc = pack.get('options_mc', [])
                    if options_mc:
                        top_option = sorted(options_mc, key=lambda x: x.get('score_stress_cvar10', 0), reverse=True)[0]
                        top_bundle_id = top_option.get('bundle_id')
                        if top_bundle_id and top_bundle_id in options_explain:
                            attr_data = options_explain[top_bundle_id]
                            kpi_attrs = attr_data.get('kpi_attributions', {})
                            if kpi_id in kpi_attrs:
                                drivers = kpi_attrs[kpi_id].get('drivers', [])
                                if drivers:
                                    # Get top 3 drivers by contribution
                                    top_drivers = sorted(drivers, key=lambda d: abs(d.get('contribution', 0)), reverse=True)[:3]
                                    driver_names = []
                                    # Build initiative map for name resolution
                                    init_map = {init.id: init.name for init in plan.initiatives}
                                    
                                    for driver in top_drivers:
                                        # AttributionDriver has: driver_type, driver_id, contribution, description
                                        # Try to extract name from description first (most reliable)
                                        description = driver.get('description', '')
                                        driver_id = driver.get('driver_id', '')
                                        driver_type = driver.get('driver_type', '')
                                        
                                        # Extract initiative name from description or lookup by ID
                                        name = ''
                                        if description:
                                            # Description format: "From {initiative.name} ..." or "Dependency penalty: {initiative.name} ..."
                                            if 'From ' in description:
                                                name = description.split('From ')[1].split(' (')[0].strip()
                                            elif 'Dependency penalty:' in description:
                                                name = description.split('Dependency penalty: ')[1].split(' blocked')[0].strip()
                                            elif driver_type == 'dependency_penalty':
                                                # Parse dependency penalty format
                                                parts = driver_id.split('_blocked_by_')
                                                if len(parts) == 2:
                                                    init_id = parts[0]
                                                    name = init_map.get(init_id, init_id) + ' (dependency)'
                                        
                                        # Fallback: lookup by driver_id if it's an initiative ID
                                        if not name and driver_id:
                                            if driver_id.startswith('INIT_'):
                                                name = init_map.get(driver_id, driver_id)
                                            elif driver_type == 'initiative':
                                                name = init_map.get(driver_id, driver_id)
                                        
                                        # Final fallback: use driver type
                                        if not name:
                                            if driver_type:
                                                name = driver_type.replace('_', ' ').title()
                                            else:
                                                name = 'Various factors'
                                        
                                        contrib = driver.get('contribution', 0)
                                        driver_names.append(f"{name} ({contrib:+.2f} {kpi.unit})")
                                    if driver_names:
                                        drivers_text = f" Likely drivers: {', '.join(driver_names)}."
            
            narrative_parts.append(
                f"<b>{kpi.short_name}</b>: Current value is {latest_value:.2f} {kpi.unit}, "
                f"compared to baseline of {baseline:.2f} {kpi.unit} and target of {target:.2f} {kpi.unit}. "
                f"Progress is {progress_pct:.1f}% complete. Status: {status}.{impact_text}{drivers_text}"
            )
            narrative_parts.append("")
    
    # Overall Assessment
    narrative_parts.append("<b>Overall Assessment</b>")
    on_track_count = 0
    at_risk_count = 0
    
    for obj in objectives:
        for okr in obj.okrs:
            kpi_id = okr.kpi_id
            kpi_data = kpi_agg[kpi_agg['kpi_id'] == kpi_id]
            if kpi_data.empty:
                continue
            
            latest_value = kpi_data['value'].iloc[-1]
            baseline = float(okr.baseline) if okr.baseline else latest_value
            target = float(okr.target) if okr.target else None
            
            if target:
                # Calculate progress percentage
                if okr.direction == "up":
                    progress_pct = ((latest_value - baseline) / (target - baseline) * 100) if (target - baseline) > 0 else 0
                    # Consider "on track" if: at target OR >75% progress with positive trend
                    is_on_track = latest_value >= target or (progress_pct >= 75 and latest_value > baseline)
                else:  # direction == "down"
                    progress_pct = ((baseline - latest_value) / (baseline - target) * 100) if (baseline - target) > 0 else 0
                    # Consider "on track" if: at target OR >75% progress with positive trend
                    is_on_track = latest_value <= target or (progress_pct >= 75 and latest_value < baseline)
                
                if is_on_track:
                    on_track_count += 1
                else:
                    at_risk_count += 1
    
    # Add explanation of "requires attention" criteria
    narrative_parts.append(
        f"Overall progress: {on_track_count} KPI(s) on track, {at_risk_count} KPI(s) requiring attention. "
        f"The organization is making {'good' if on_track_count > at_risk_count else 'moderate' if on_track_count == at_risk_count else 'limited'} progress towards strategic objectives."
    )
    narrative_parts.append("")
    narrative_parts.append(
        "<b>Assessment Criteria:</b> A KPI is marked as 'requiring attention' if it has not reached its target "
        "AND has less than 75% progress toward the target. KPIs with 75% or more progress toward target "
        "(with positive trend) are considered 'on track' even if the target has not yet been achieved."
    )
    
    return "\n".join(narrative_parts)

def _generate_executive_summary(
    plan: Any,
    objectives: List[Any],
    period_data: Dict[str, Any],
    as_of: str,
    pack: Optional[Dict[str, Any]] = None,
) -> str:
    """Generate executive summary."""
    summary_parts = []
    
    n_obj = int(len(objectives))
    n_words = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five"}
    n_obj_text = n_words.get(n_obj, str(n_obj))
    summary_parts.append(
        f"This report reviews progress across {n_obj_text} strategic objectives as of {as_of}. "
        "It provides an integrated view of KPI trajectory, risks, and required strategic decisions."
    )
    
    # Count on-track vs at-risk
    kpi_map = {k.id: k for k in plan.kpis}
    kpi_agg = period_data['kpi_data']
    
    on_track = 0
    at_risk = 0
    
    for obj in objectives:
        for okr in obj.okrs:
            kpi_id = okr.kpi_id
            kpi_data = kpi_agg[kpi_agg['kpi_id'] == kpi_id]
            if kpi_data.empty:
                continue
            
            latest_value = kpi_data['value'].iloc[-1]
            target = float(okr.target) if okr.target else None
            
            if target:
                baseline = float(okr.baseline) if okr.baseline else latest_value
                # Calculate progress percentage
                if okr.direction == "up":
                    progress_pct = ((latest_value - baseline) / (target - baseline) * 100) if (target - baseline) > 0 else 0
                    # Consider "on track" if: at target OR >80% progress with positive trend
                    is_on_track = latest_value >= target or (progress_pct >= 80 and latest_value > baseline)
                else:  # direction == "down"
                    progress_pct = ((baseline - latest_value) / (baseline - target) * 100) if (baseline - target) > 0 else 0
                    # Consider "on track" if: at target OR >80% progress with positive trend
                    is_on_track = latest_value <= target or (progress_pct >= 80 and latest_value < baseline)
                
                if is_on_track:
                    on_track += 1
                else:
                    at_risk += 1
    
    summary_parts.append(
        f"Overall status: {on_track} KPI(s) on track, {at_risk} KPI(s) requiring attention."
    )
    
    # Add context from pack if available
    if pack:
        rec = pack.get('recommendation', {})
        if rec:
            conf_label = _format_confidence_label(rec.get('confidence'))
            summary_parts.append(
                f"Key recommendation: {rec.get('chosen_bundle_name', 'N/A')} "
                f"(confidence: {conf_label})."
            )
    
    return " ".join(summary_parts)


def _get_objective_progress(obj: Any, period_data: Dict[str, Any], as_of: str) -> str:
    """Get progress description for an objective."""
    kpi_agg = period_data['kpi_data']
    
    progress_items = []
    for okr in obj.okrs:
        kpi_data = kpi_agg[kpi_agg['kpi_id'] == okr.kpi_id]
        if not kpi_data.empty:
            latest = kpi_data['value'].iloc[-1]
            target = float(okr.target) if okr.target else None
            if target:
                if okr.direction == "up":
                    pct = (latest / target * 100) if target > 0 else 0
                else:
                    pct = (target / latest * 100) if latest > 0 else 0
                progress_items.append(f"{okr.kpi_id}: {pct:.1f}%")
    
    return "; ".join(progress_items) if progress_items else "No data available"


def _generate_problem_description(plan: Any, pack: Dict[str, Any], as_of: str) -> str:
    """Generate problem description explaining what triggered the search for options."""
    problem_parts = []
    
    # Collect all problems
    triggers = []
    
    # Check for anomalies
    ml_anomalies = pack.get('ml_anomalies', {})
    for kpi_id, anomaly_data in ml_anomalies.items():
        if anomaly_data.get('is_anomaly', False):
            kpi = next((k for k in plan.kpis if k.id == kpi_id), None)
            if kpi:
                triggers.append(f"ML-based anomaly detection flagged unusual patterns in {kpi.short_name}: {anomaly_data.get('explanation', 'Unusual pattern detected')}")
    
    # Check for integrity issues
    integrity_scores = pack.get('integrity_scores', {})
    for kpi_id, score in integrity_scores.items():
        if score > 0.6:  # High integrity score = suspicious
            kpi = next((k for k in plan.kpis if k.id == kpi_id), None)
            if kpi:
                triggers.append(f"Data integrity check revealed quality issues in {kpi.short_name} (integrity score: {score:.2f})")
    
    # Check for KPI drift (from forecast vs expected) - DIRECTION-AWARE
    forecast_head = pack.get('forecast_head', [])
    if forecast_head:
        forecast_df = pd.DataFrame(forecast_head)
        kpi_map = {k.id: k for k in plan.kpis}
        # Build OKR map to get direction
        okr_map = {}
        for obj in plan.objectives:
            for okr in obj.okrs:
                if okr.kpi_id not in okr_map:
                    okr_map[okr.kpi_id] = okr
        
        for _, row in forecast_df.iterrows():
            kpi_id = row.get('kpi_id')
            forecast_val = row.get('forecast')
            expected_val = row.get('expected')
            if kpi_id and forecast_val is not None and expected_val is not None:
                kpi = kpi_map.get(kpi_id)
                okr = okr_map.get(kpi_id)
                if kpi and okr:
                    # Get direction from OKR
                    direction = okr.direction if hasattr(okr, 'direction') and okr.direction else "up"
                    
                    # Direction-aware problem detection
                    is_problem = False
                    if direction == "up":
                        # For "up" KPIs: problem when forecast < expected (below is bad)
                        if forecast_val < expected_val:
                            is_problem = True
                            drift_pct = ((expected_val - forecast_val) / abs(expected_val)) * 100 if expected_val != 0 else 0
                    else:  # direction == "down"
                        # For "down" KPIs: problem when forecast > expected (above is bad)
                        if forecast_val > expected_val:
                            is_problem = True
                            drift_pct = ((forecast_val - expected_val) / abs(expected_val)) * 100 if expected_val != 0 else 0
                    
                    # Only flag if drift is significant (more than 5% of expected)
                    if is_problem:
                        drift = abs(forecast_val - expected_val)
                        if expected_val != 0 and (drift / abs(expected_val)) > 0.05:
                            direction_text = "below" if direction == "up" else "above"
                            triggers.append(f"{kpi.short_name} is {direction_text} expected trajectory (forecast: {forecast_val:.2f} vs expected: {expected_val:.2f})")

    # Check for forecast deviation alerts vs weekly objective baseline
    forecast_dev = pack.get('forecast_deviation_alerts', {})
    for kpi_id, dev_data in forecast_dev.items():
        if dev_data.get('is_deviation_alert', False):
            kpi = next((k for k in plan.kpis if k.id == kpi_id), None)
            if kpi:
                sev = dev_data.get('max_severity', 'warning')
                z = float(dev_data.get('max_z_score', 0.0))
                triggers.append(
                    f"Forecast deviation alert ({sev}) for {kpi.short_name} vs weekly objective baseline (z={z:.2f})"
                )
    
    # Check for guardrail violations
    rec = pack.get('recommendation', {})
    if rec and not rec.get('guardrails_passed', True):
        violations = rec.get('guardrails_violations', [])
        for v in violations:
            triggers.append(f"Guardrail constraint not met: {v.get('message', 'Constraint violation')}")
    
    # Build problem description
    if triggers:
        problem_parts.append("<b>What Triggered the Search for Options:</b>")
        problem_parts.append("")
        problem_parts.append("The following issues were identified that require strategic intervention:")
        problem_parts.append("")
        for i, trigger in enumerate(triggers, 1):
            problem_parts.append(f"{i}. {trigger}")
    else:
        problem_parts.append("<b>Context:</b>")
        problem_parts.append("")
        problem_parts.append("Regular strategic review identified opportunities to optimize progress toward objectives. No critical problems detected, but proactive options analysis was conducted to ensure optimal resource allocation.")
    
    return "\n".join(problem_parts)


def _build_bundle_lookup(pack: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Return bundle metadata keyed by bundle id using the pack payload."""
    out: Dict[str, Dict[str, Any]] = {}
    for b in pack.get("bundles", []) or []:
        bid = str(b.get("id", "")).strip()
        if bid:
            out[bid] = b
    return out


def _get_bundle_description(bundle_id: str, bundle_lookup: Dict[str, Dict[str, Any]]) -> str:
    """Get description from pack bundle payload."""
    b = bundle_lookup.get(bundle_id, {})
    return str(b.get("description", "") or "")


def _generate_options_section(plan: Any, pack: Dict[str, Any], as_of: Optional[str] = None) -> str:
    """Generate detailed description of each option with concrete actions."""
    options_parts = []
    
    options_mc = pack.get('options_mc', [])
    if not options_mc:
        return ""
    
    # Get initiative and workstream maps for better descriptions
    init_map = {init.id: init for init in plan.initiatives}
    ws_map: Dict[tuple, Any] = {}
    for init in plan.initiatives:
        for ws in getattr(init, "workstreams", []) or []:
            ws_map[(init.id, ws.id)] = ws
    bundle_lookup = _build_bundle_lookup(pack)
    
    # Sort options by rank (stress CVaR10)
    sorted_options = sorted(
        options_mc,
        key=lambda x: x.get('score_stress_cvar10', 0),
        reverse=True
    )
    
    options_parts.append("<b>Available Strategic Options (Ranked by Risk-Adjusted Score):</b>")
    options_parts.append("")
    
    for rank, option in enumerate(sorted_options, 1):
        bundle_id = option.get('bundle_id', 'N/A')
        bundle_name = option.get('bundle_name', 'N/A')
        
        # Get description and concrete actions from pack
        description = _get_bundle_description(bundle_id, bundle_lookup)
        actions = (bundle_lookup.get(bundle_id, {}) or {}).get("actions", []) or []
        
        options_parts.append(f"<b>Rank #{rank}: {bundle_name} ({bundle_id})</b>")
        if description:
            options_parts.append(f"Strategy: {description}")
        options_parts.append("")
        
        # Concrete actions
        if actions:
            options_parts.append("<b>Concrete Actions:</b>")
            for action in actions:
                action_type = action.get('type', 'unknown')
                target = action.get('target_initiative', action.get('target', 'N/A'))
                params = action.get('parameters', {})
                workstream_id = params.get("workstream_id")
                
                # Get initiative name
                init = init_map.get(target)
                init_name = init.name if init else target
                
                # Format action description
                if action_type == "accelerate_initiative":
                    months = params.get('months', 0)
                    options_parts.append(f"• Accelerate {init_name} by {months} months")
                    if workstream_id:
                        ws = ws_map.get((target, workstream_id))
                        ws_name = ws.name if ws else str(workstream_id)
                        options_parts.append(f"  Scope: target workstream {ws_name} ({workstream_id})")
                    options_parts.append("  Impact: Brings outcomes forward and protects near-term delivery milestones.")
                    options_parts.append(f"  {_estimate_action_resource_cost(action_type, params)}")
                elif action_type == "split_scope":
                    reduction = params.get('reduction', 0)
                    reduction_pct = reduction * 100
                    options_parts.append(f"• Split scope of {init_name} (reduce by {reduction_pct:.0f}%)")
                    if workstream_id:
                        ws = ws_map.get((target, workstream_id))
                        ws_name = ws.name if ws else str(workstream_id)
                        options_parts.append(f"  Scope: target workstream {ws_name} ({workstream_id})")
                    options_parts.append("  Impact: Reduces dependency blocking and prioritizes stable flows first.")
                    options_parts.append(f"  {_estimate_action_resource_cost(action_type, params)}")
                elif action_type == "add_capacity":
                    cap_gain = params.get('capacity_gain', 0)
                    throughput_pct = float(cap_gain) * 100.0
                    options_parts.append(f"• Add capacity to {init_name} (+{throughput_pct:.0f}% delivery throughput)")
                    if workstream_id:
                        ws = ws_map.get((target, workstream_id))
                        ws_name = ws.name if ws else str(workstream_id)
                        options_parts.append(f"  Scope: target workstream {ws_name} ({workstream_id})")
                    options_parts.append("  Impact: Increases delivery velocity and shortens completion lead time.")
                    options_parts.append(f"  {_estimate_action_resource_cost(action_type, params)}")
                else:
                    options_parts.append(f"• {action_type}: {init_name}")
            base_k, adj_k, ccy = _estimate_bundle_budget_envelope(plan, actions)
            if base_k > 0:
                delta_k = adj_k - base_k
                options_parts.append(
                    f"Budget envelope (plan-based): baseline {base_k:.0f} {ccy}, adjusted {adj_k:.0f} {ccy} ({delta_k:+.0f})."
                )
                if as_of:
                    year_limits, budget_ccy = _budget_constraint_info(plan)
                    year_key = str(pd.to_datetime(as_of).year)
                    if year_limits and year_key in year_limits:
                        cap = float(year_limits.get(year_key, 0.0) or 0.0)
                        if cap > 0:
                            util = (adj_k / cap) * 100.0
                            options_parts.append(
                                f"Annual cap check ({year_key}): {adj_k:.0f} / {cap:.0f} {budget_ccy or ccy} ({util:.0f}% of cap)."
                            )
        else:
            options_parts.append("No specific actions detailed.")
        
        options_parts.append("")
    
    return "\n".join(options_parts)


def _generate_options_scoring_table(plan: Any, pack: Dict[str, Any]) -> Optional[Table]:
    """Generate a scoring table for all options."""
    options_mc = pack.get('options_mc', [])
    if not options_mc:
        return None
    
    # Sort by score
    sorted_options = sorted(
        options_mc,
        key=lambda x: x.get('score_stress_cvar10', 0),
        reverse=True
    )
    
    cell_style = ParagraphStyle(
        'OptionsTableCell',
        parent=getSampleStyleSheet()['Normal'],
        fontSize=8.5,
        leading=10,
        wordWrap='CJK',
    )

    # Build table data with wrapped headers
    table_data = [[
        Paragraph("Rank", cell_style),
        Paragraph("Strategic Option", cell_style),
        Paragraph("Risk Score<br/>(CVaR10)", cell_style),
        Paragraph("Expected<br/>Return", cell_style),
        Paragraph("Best Case<br/>(p90)", cell_style),
    ]]
    
    for rank, option in enumerate(sorted_options, 1):
        bundle_name = option.get('bundle_name', 'N/A')
        cvar10 = option.get('score_stress_cvar10', 0)
        expected_return = option.get('score_stress_mean', 0)
        best_case = _get_best_case_score(option, plan)
        
        table_data.append([
            Paragraph(str(rank), cell_style),
            Paragraph(_safe_text(bundle_name[:55]), cell_style),
            Paragraph(f"{cvar10:.3f}", cell_style),
            Paragraph(f"{expected_return:.3f}", cell_style),
            Paragraph(best_case, cell_style),
        ])
    
    # Create featured full-width table
    t = Table(table_data, colWidths=[0.55*inch, 3.25*inch, 1.0*inch, 1.0*inch, 1.0*inch])
    style_cmds = [
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 9.5),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 7),
        ('GRID', (0, 0), (-1, -1), 0.8, colors.HexColor('#94A3B8')),
        ('FONTSIZE', (0, 1), (-1, -1), 8.8),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#F8FAFC')]),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('WORDWRAP', (0, 0), (-1, -1), 'CJK'),
        ('LEFTPADDING', (0, 0), (-1, -1), 4),
        ('RIGHTPADDING', (0, 0), (-1, -1), 4),
        ('TOPPADDING', (0, 1), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 1), (-1, -1), 4),
    ]
    # Highlight best-ranked row
    if len(table_data) > 1:
        style_cmds.append(('BACKGROUND', (0, 1), (-1, 1), colors.HexColor('#D1FAE5')))
    # Shade negative-score rows for quick visual interpretation
    for row_i in range(1, len(table_data)):
        try:
            row_score = float(table_data[row_i][2].getPlainText())
        except Exception:
            row_score = 0.0
        if row_score < 0:
            style_cmds.append(('BACKGROUND', (0, row_i), (-1, row_i), colors.HexColor('#FEF2F2')))
    # Semantic coloring for numeric columns
    for row_i in range(1, len(table_data)):
        for col_i in (2, 3, 4):
            txt = table_data[row_i][col_i].getPlainText() if hasattr(table_data[row_i][col_i], "getPlainText") else str(table_data[row_i][col_i])
            try:
                v = float(txt)
            except Exception:
                continue
            style_cmds.append(('TEXTCOLOR', (col_i, row_i), (col_i, row_i), colors.HexColor('#166534') if v >= 0 else colors.HexColor('#B91C1C')))
    t.setStyle(TableStyle(style_cmds))
    
    return t


def _generate_kpi_health_dashboard_table(
    plan: Any,
    objectives: List[Any],
    period_data: Dict[str, Any],
) -> Optional[Table]:
    """Generate one-page executive KPI health dashboard (RAG-style)."""
    kpi_agg = period_data.get("kpi_data", pd.DataFrame())
    if kpi_agg.empty:
        return None
    kpi_map = {k.id: k for k in plan.kpis}

    cell_style = ParagraphStyle(
        'KpiHealthCell',
        parent=getSampleStyleSheet()['Normal'],
        fontSize=8.5,
        leading=10,
        wordWrap='CJK',
    )
    rows = [[
        Paragraph("KPI", cell_style),
        Paragraph("Status", cell_style),
        Paragraph("Current vs Target", cell_style),
        Paragraph("Executive Signal", cell_style),
    ]]
    row_statuses: List[str] = []

    for obj in objectives:
        for okr in obj.okrs:
            kpi = kpi_map.get(okr.kpi_id)
            if not kpi:
                continue
            kpi_data = kpi_agg[kpi_agg["kpi_id"] == okr.kpi_id]
            if kpi_data.empty:
                continue
            latest = float(kpi_data["value"].iloc[-1])
            baseline = float(okr.baseline) if okr.baseline else latest
            target = float(okr.target) if okr.target else None
            status = _classify_kpi_status(latest, baseline, target, okr.direction)

            if target is None:
                delta_text = f"{latest:.2f} {kpi.unit} vs N/A"
            else:
                delta = latest - target
                delta_text = f"{latest:.2f} {kpi.unit} vs {target:.2f} {kpi.unit} ({delta:+.2f})"
            if status == "GREEN":
                signal = "Trajectory acceptable; continue current execution."
            elif status == "AMBER":
                signal = "Trajectory vulnerable; targeted corrective action advised."
            else:
                signal = "Target miss risk elevated; steering intervention required."

            rows.append([
                Paragraph(_safe_text(kpi.short_name), cell_style),
                Paragraph(
                    _safe_text("✓ On Track" if status == "GREEN" else "⚠ Attention" if status == "AMBER" else "✗ Critical"),
                    cell_style,
                ),
                Paragraph(_safe_text(delta_text), cell_style),
                Paragraph(_safe_text(signal), cell_style),
            ])
            row_statuses.append(status)

    if len(rows) <= 1:
        return None

    table = Table(rows, colWidths=[1.2 * inch, 0.8 * inch, 1.8 * inch, 2.8 * inch])
    style_cmds = [
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#E2E8F0')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.HexColor('#0F172A')),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 9),
        ('FONTSIZE', (0, 1), (-1, -1), 8.5),
        ('ALIGN', (1, 1), (1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#CBD5E1')),
        ('LEFTPADDING', (0, 0), (-1, -1), 5),
        ('RIGHTPADDING', (0, 0), (-1, -1), 5),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('WORDWRAP', (0, 0), (-1, -1), 'CJK'),
    ]
    for i, st in enumerate(row_statuses, start=1):
        if st == "GREEN":
            style_cmds.append(('BACKGROUND', (1, i), (1, i), colors.HexColor('#DCFCE7')))
            style_cmds.append(('TEXTCOLOR', (1, i), (1, i), colors.HexColor('#166534')))
        elif st == "AMBER":
            style_cmds.append(('BACKGROUND', (1, i), (1, i), colors.HexColor('#FEF3C7')))
            style_cmds.append(('TEXTCOLOR', (1, i), (1, i), colors.HexColor('#92400E')))
        else:
            style_cmds.append(('BACKGROUND', (1, i), (1, i), colors.HexColor('#FEE2E2')))
            style_cmds.append(('TEXTCOLOR', (1, i), (1, i), colors.HexColor('#991B1B')))
    table.setStyle(TableStyle(style_cmds))
    return table


def _generate_kpi_definitions_appendix(
    plan: Any,
    objectives: List[Any],
) -> List[Any]:
    """Appendix with KPI definitions and formulas (kept out of executive body)."""
    out: List[Any] = []
    styles = getSampleStyleSheet()
    kpi_map = {k.id: k for k in plan.kpis}
    seen: set = set()
    for obj in objectives:
        for okr in obj.okrs:
            kpi = kpi_map.get(okr.kpi_id)
            if not kpi or kpi.id in seen:
                continue
            seen.add(kpi.id)
            sem = _kpi_semantic_context(kpi)
            out.append(Paragraph(f"<b>{sem['short_name']}</b> — {sem['name']}", styles['Normal']))
            if sem["description"]:
                out.append(Paragraph(f"<b>Definition:</b> {sem['description']}", styles['Normal']))
            if sem["formula"]:
                out.append(Paragraph(f"<b>Formula:</b> {sem['formula']}", styles['Normal']))
            if sem["exclusions"]:
                out.append(Paragraph(f"<b>Exclusions:</b> {sem['exclusions']}", styles['Normal']))
            out.append(_space("block_gap"))
    return out


def _generate_risk_register_table(plan: Any, pack: Optional[Dict[str, Any]]) -> Optional[Table]:
    """Build a compact risk register with ownership and mitigations for steering review."""
    if not pack:
        return None
    rec = (pack.get("recommendation", {}) or {})
    chosen_id = str(rec.get("chosen_bundle_id", "") or "")
    if not chosen_id:
        return None
    bundle = next((b for b in (pack.get("bundles", []) or []) if str(b.get("id", "")) == chosen_id), None)
    if not bundle:
        return None

    init_map = {i.id: i for i in getattr(plan, "initiatives", []) or []}
    rows = [[
        Paragraph("Risk", getSampleStyleSheet()['Normal']),
        Paragraph("Owner", getSampleStyleSheet()['Normal']),
        Paragraph("Mitigation", getSampleStyleSheet()['Normal']),
    ]]
    seen: set = set()
    for a in bundle.get("actions", []) or []:
        tid = str(a.get("target_initiative", "") or "")
        init = init_map.get(tid)
        if not init:
            continue
        owner = str(getattr(init, "owner", "Program Lead"))
        for r in (getattr(init, "risks", []) or [])[:2]:
            rid = str(getattr(r, "id", ""))
            key = f"{tid}:{rid}"
            if key in seen:
                continue
            seen.add(key)
            desc = _safe_text(getattr(r, "description", "Execution risk"))
            mit = ", ".join([_safe_text(m) for m in (getattr(r, "mitigations", []) or [])[:2]]) or "Mitigation plan to confirm."
            rows.append([
                Paragraph(desc, getSampleStyleSheet()['Normal']),
                Paragraph(_safe_text(owner), getSampleStyleSheet()['Normal']),
                Paragraph(mit, getSampleStyleSheet()['Normal']),
            ])
    if len(rows) <= 1:
        return None
    t = Table(rows, colWidths=[2.5 * inch, 1.1 * inch, 2.5 * inch])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#F1F5F9')),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.HexColor('#0F172A')),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#CBD5E1')),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#F8FAFC')]),
        ('FONTSIZE', (0, 0), (-1, -1), 8.6),
    ]))
    return t
