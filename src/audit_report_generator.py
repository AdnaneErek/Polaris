"""
AI Decision Governance & Justification Report (Strategy Intelligence Audit Pack)

This report is a governance-focused companion artifact to the SteerCo Progress Report.
It is designed for transparency, traceability, and auditability of AI-assisted decisions.

Primary input: SteerCo pack directory containing:
  - steerco_pack.json
  - logs/audit.json (recommended)
  - visuals/*.png (optional appendix)
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.platypus import (
        SimpleDocTemplate,
        Paragraph,
        Spacer,
        Table,
        TableStyle,
        PageBreak,
        Image,
    )
    from reportlab.pdfgen import canvas

    REPORTLAB_AVAILABLE = True
except Exception:
    REPORTLAB_AVAILABLE = False


# --------------------------------------------------------------------------------------
# Shared look & feel (lightweight; we keep this report intentionally compact and readable)
# --------------------------------------------------------------------------------------
AUDIT_BRAND: Dict[str, str] = {
    "primary": "#1E3A8A",
    "secondary": "#334155",
    "muted": "#64748B",
    "surface": "#F8FAFC",
    "border": "#CBD5E1",
    "success": "#166534",
    "warning": "#B45309",
    "danger": "#B91C1C",
}


def _space(inches: float) -> Spacer:
    return Spacer(1, inches * inch)


def _safe(text: Any) -> str:
    if text is None:
        return ""
    return str(text)


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _fmt_conf_label(conf: Optional[float]) -> str:
    if conf is None:
        return "N/A"
    c = float(conf)
    if c >= 0.95:
        return "Very high"
    if c >= 0.80:
        return "High"
    if c >= 0.60:
        return "Moderate"
    return "Low"


def _pill(text: str, tone: str, styles) -> Paragraph:
    """Badge-like paragraph inside a table cell."""
    tone = tone.lower()
    if tone == "success":
        fg = AUDIT_BRAND["success"]
        bg = "#D1FAE5"
    elif tone == "warning":
        fg = AUDIT_BRAND["warning"]
        bg = "#FFEDD5"
    else:
        fg = AUDIT_BRAND["danger"]
        bg = "#FEF2F2"
    st = ParagraphStyle(
        name=f"Pill_{tone}",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=9,
        textColor=colors.HexColor(fg),
        backColor=colors.HexColor(bg),
        leading=11,
        leftIndent=4,
        rightIndent=4,
        spaceBefore=0,
        spaceAfter=0,
    )
    return Paragraph(_safe(text), st)


def _table(data: List[List[Any]], col_widths: Optional[List[float]] = None) -> Table:
    t = Table(data, colWidths=col_widths)
    t.setStyle(
        TableStyle(
            [
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor(AUDIT_BRAND["border"])),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    return t


def _h(text: str, styles, size: int = 14) -> Paragraph:
    st = ParagraphStyle(
        name=f"H_{size}",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=size,
        textColor=colors.HexColor(AUDIT_BRAND["primary"]),
        spaceBefore=0,
        spaceAfter=6,
    )
    return Paragraph(_safe(text), st)


def _small(text: str, styles) -> Paragraph:
    st = ParagraphStyle(
        name="Small",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9,
        textColor=colors.HexColor(AUDIT_BRAND["secondary"]),
        leading=11,
    )
    return Paragraph(_safe(text), st)


def _mono_pre(text: str, styles) -> Paragraph:
    st = ParagraphStyle(
        name="Mono",
        parent=styles["Normal"],
        fontName="Courier",
        fontSize=8.2,
        textColor=colors.HexColor("#0F172A"),
        leading=9.5,
        backColor=colors.HexColor("#F8FAFC"),
        borderPadding=6,
    )
    # escape < and > for reportlab paragraph parsing
    safe = _safe(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    return Paragraph(f"<pre>{safe}</pre>", st)


def _parse_guardrails_summary(guardrails_report: str) -> Dict[str, Dict[str, str]]:
    """
    Parse the formatted guardrails report into a compact per-option PASS/FAIL map.
    Best-effort only (source report is plain text).
    Returns: {bundle_id: {"name": str, "status": "PASS|FAIL"}}
    """
    out: Dict[str, Dict[str, str]] = {}
    if not guardrails_report:
        return out
    for raw in guardrails_report.splitlines():
        line = raw.strip()
        # Example:
        # "Option 3 — Robust Operational Resilience Push (OPT_3): PASS"
        if "(" in line and "):" in line and (line.endswith("PASS") or line.endswith("FAIL")):
            try:
                name_part, status = line.rsplit(":", 1)
                status = status.strip().upper()
                bid = name_part.split("(")[-1].split(")")[0].strip()
                name = name_part.split("(")[0].strip()
                out[bid] = {"name": name, "status": status}
            except Exception:
                continue
    return out


def generate_audit_report(
    pack_dir: str | Path,
    output_path: Optional[str | Path] = None,
    include_visual_appendix: bool = True,
) -> str:
    """
    Generate the Strategy Intelligence Audit Pack PDF.

    Args:
        pack_dir: directory containing steerco_pack.json (and optionally logs/audit.json, visuals/)
        output_path: optional explicit output pdf path
        include_visual_appendix: append images from visuals/ as evidence
    """
    if not REPORTLAB_AVAILABLE:
        raise ImportError("reportlab not installed. Install with: pip install reportlab")

    pack_dir = Path(pack_dir)
    pack_path = pack_dir / "steerco_pack.json"
    if not pack_path.exists():
        raise FileNotFoundError(f"steerco_pack.json not found in: {pack_dir}")

    pack = _load_json(pack_path)

    audit_path = pack_dir / "logs" / "audit.json"
    audit = _load_json(audit_path) if audit_path.exists() else {}

    as_of = _safe(pack.get("as_of") or audit.get("as_of") or "")
    horizon_end = _safe(pack.get("horizon_end") or audit.get("config", {}).get("horizon_end") or "")

    rec = pack.get("recommendation", {}) or {}
    chosen_id = _safe(rec.get("chosen_bundle_id"))
    chosen_name = _safe(rec.get("chosen_bundle_name"))
    conf = rec.get("confidence", None)
    conf_label = _fmt_conf_label(float(conf)) if conf is not None else "N/A"

    bundle_gen = pack.get("bundle_generation") or audit.get("config", {}).get("bundle_generation") or {}
    bundle_gen_mode = _safe(bundle_gen.get("mode", "unknown")).replace("_", " ").title()

    if output_path is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = Path("artifacts/reports") / f"audit_report_{as_of}_{ts}.pdf"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    styles = getSampleStyleSheet()
    styles["Normal"].fontName = "Helvetica"
    styles["Normal"].fontSize = 10
    styles["Heading2"].fontName = "Helvetica-Bold"
    styles["Heading2"].fontSize = 14
    styles["Heading2"].textColor = colors.HexColor(AUDIT_BRAND["primary"])

    story: List[Any] = []
    story.append(_h("AI Decision Governance & Justification Report", styles, size=18))
    story.append(_small("(Strategy Intelligence Audit Pack)", styles))
    story.append(_small(f"As of: <b>{as_of}</b> | Horizon end: <b>{horizon_end}</b> | Bundle generation: <b>{bundle_gen_mode}</b>", styles))
    story.append(_space(0.14))

    # -----------------------------
    # 1) Executive Governance Summary (target: 1 page)
    # -----------------------------
    story.append(_h("1. Executive Governance Summary", styles, size=14))

    snapshot = [
        ["Decision Date", as_of or "N/A"],
        ["Recommended Option", f"{chosen_name} ({chosen_id})" if chosen_id else (chosen_name or "N/A")],
        ["Decision Confidence", f"{float(conf):.2f} ({conf_label})" if conf is not None else "N/A"],
        ["Primary Objective", "Risk-adjusted execution of the 2026–2028 strategic plan"],
        ["Risk Profile", "Conservative / risk-averse (ranked by CVaR10 under stress)"],
        ["Approval Required", "Steering Committee"],
    ]
    story.append(_table([["Dimension", "Value"]] + snapshot, col_widths=[2.0 * inch, 4.2 * inch]))
    story.append(_space(0.10))

    why = rec.get("why", []) or []
    story.append(_small("<b>Strategic rationale (human language)</b>", styles))
    if why:
        for w in why[:8]:
            story.append(_small(f"- {_safe(w)}", styles))
    else:
        story.append(_small("- Rationale not available in pack.", styles))
    story.append(_space(0.10))

    # Governance verdict derived from recommendation guardrails
    guardrails_passed = bool(rec.get("guardrails_passed", True))
    violations = rec.get("guardrails_violations", []) or []
    has_error = any(_safe(v.get("severity")).lower() == "error" for v in violations)
    verdict = "Approved for execution" if guardrails_passed and not has_error else "SteerCo decision required (guardrails failed)"
    verdict_pill = _pill("✅ Approved for execution" if verdict.startswith("Approved") else "⚠ SteerCo decision required", "success" if verdict.startswith("Approved") else "warning", styles)

    checks = [
        ["KPI degradation", _pill("PASS", "success", styles) if guardrails_passed else _pill("REVIEW", "warning", styles)],
        ["Budget compliance", _pill("PASS", "success", styles) if guardrails_passed else _pill("REVIEW", "warning", styles)],
        ["Control coverage", _pill("PASS", "success", styles) if guardrails_passed else _pill("REVIEW", "warning", styles)],
        ["Compliance guardrails", _pill("PASS", "success", styles) if guardrails_passed else _pill("REVIEW", "warning", styles)],
        ["Final governance verdict", verdict_pill],
    ]
    story.append(_small("<b>Governance verdict</b>", styles))
    story.append(_table([["Check", "Result"]] + checks, col_widths=[2.3 * inch, 3.9 * inch]))

    story.append(PageBreak())

    # -----------------------------
    # 2) Data Integrity & Reliability Controls
    # -----------------------------
    story.append(_h("2. Data Integrity & Reliability Controls", styles, size=14))
    integrity_scores = pack.get("integrity_scores", {}) or {}
    integ_rows: List[List[Any]] = [["KPI", "Integrity Score", "Flags", "Status"]]
    for kpi_id, score in sorted(integrity_scores.items()):
        sc = float(score)
        status = _pill("PASS", "success", styles) if sc <= 0.6 else _pill("REVIEW", "warning", styles)
        integ_rows.append([_safe(kpi_id), f"{sc:.2f}", "None (see raw integrity report)", status])
    story.append(_table(integ_rows, col_widths=[1.3 * inch, 1.2 * inch, 2.6 * inch, 1.1 * inch]))
    story.append(_space(0.10))
    story.append(_small("<b>KPI integrity assessment (raw evidence)</b>", styles))
    story.append(_mono_pre(pack.get("integrity_report", ""), styles))
    story.append(_space(0.10))

    ml_anomalies = pack.get("ml_anomalies", {}) or {}
    story.append(_small("<b>ML anomaly detection</b>", styles))
    if ml_anomalies:
        rows: List[List[Any]] = [["KPI", "Anomaly", "Confidence", "Explanation"]]
        for kpi_id, d in sorted(ml_anomalies.items()):
            is_anom = bool(d.get("is_anomaly", False))
            pill = _pill("ANOMALY" if is_anom else "NORMAL", "warning" if is_anom else "success", styles)
            rows.append([_safe(kpi_id), pill, f"{float(d.get('confidence', 0.0)):.2f}", _safe(d.get("explanation", ""))])
        story.append(_table(rows, col_widths=[1.1 * inch, 1.0 * inch, 1.0 * inch, 2.9 * inch]))
    else:
        story.append(_small("No ML anomaly results available in pack.", styles))

    story.append(PageBreak())

    # -----------------------------
    # 3) Forecast Diagnostics & Risk Signals
    # -----------------------------
    story.append(_h("3. Forecast Diagnostics & Risk Signals", styles, size=14))
    fc_head = pd.DataFrame(pack.get("forecast_head", []) or [])
    fc_rows: List[List[Any]] = [["KPI", "Forecast", "Expected", "Risk level (heuristic)"]]
    if not fc_head.empty:
        try:
            fc_head["date"] = pd.to_datetime(fc_head["date"], errors="coerce")
            snap = fc_head[fc_head["date"] == pd.to_datetime(as_of)].copy()
        except Exception:
            snap = fc_head.copy()

        for _, r in snap.iterrows():
            kpi_id = _safe(r.get("kpi_id"))
            forecast = float(r.get("forecast", 0.0))
            expected = float(r.get("expected", 0.0))
            # Heuristic risk: relative gap magnitude
            gap = 0.0 if expected == 0 else abs((forecast - expected) / expected) * 100.0
            if gap >= 10:
                risk = _pill("HIGH", "warning", styles)
            elif gap >= 4:
                risk = _pill("MEDIUM", "warning", styles)
            else:
                risk = _pill("LOW", "success", styles)
            fc_rows.append([kpi_id, f"{forecast:.2f}", f"{expected:.2f}", risk])
    else:
        fc_rows.append(["N/A", "N/A", "N/A", _pill("N/A", "warning", styles)])
    story.append(_table(fc_rows, col_widths=[1.2 * inch, 1.3 * inch, 1.3 * inch, 2.3 * inch]))
    story.append(_space(0.12))

    dev = pack.get("forecast_deviation_alerts", {}) or {}
    story.append(_small("<b>Deviation alert system (forecast vs plan baseline weekly objectives)</b>", styles))
    dev_rows: List[List[Any]] = [["KPI", "Peak risk date", "Deviation", "Z-score", "Severity"]]
    if dev:
        for kpi_id, d in sorted(dev.items()):
            if not isinstance(d, dict):
                continue
            peak_date = _safe(d.get("max_critical_date") or d.get("max_date") or d.get("peak_date") or "")
            deviation = _safe(d.get("max_critical_deviation") or d.get("max_deviation") or d.get("deviation") or "")
            z = d.get("max_critical_z") or d.get("max_z_score") or d.get("z_score") or 0.0
            sev = _safe(d.get("max_critical_severity") or d.get("max_severity") or d.get("severity") or "").upper()
            if "CRIT" in sev:
                sev_p = _pill(sev or "CRITICAL", "danger", styles)
            elif "HIGH" in sev or "WARN" in sev:
                sev_p = _pill(sev or "HIGH", "warning", styles)
            else:
                sev_p = _pill(sev or "OK", "success", styles)
            dev_rows.append([_safe(kpi_id), peak_date or "N/A", deviation or "N/A", f"{float(z):.2f}", sev_p])
    else:
        dev_rows.append(["N/A", "N/A", "N/A", "0.00", _pill("N/A", "warning", styles)])
    story.append(_table(dev_rows, col_widths=[1.1 * inch, 1.4 * inch, 1.4 * inch, 0.9 * inch, 1.4 * inch]))

    story.append(PageBreak())

    # -----------------------------
    # 4) Monte Carlo Strategy Evaluation Framework
    # -----------------------------
    story.append(_h("4. Monte Carlo Strategy Evaluation Framework", styles, size=14))
    mc_cfg = (audit.get("config", {}) or {}).get("mc_cfg", {}) or {}
    n_samples = mc_cfg.get("n_samples", None)
    seed = audit.get("random_seed", mc_cfg.get("seed", None))
    story.append(
        _small(
            f"Method: Monte Carlo simulation under base + stress conditions (scenarios: <b>{_safe(n_samples) or 'N/A'}</b>, seed: <b>{_safe(seed) or 'N/A'}</b>). "
            "Options are ranked by <b>CVaR10</b> (expected outcome in the worst 10% of scenarios) to prioritize robustness.",
            styles,
        )
    )
    story.append(_space(0.10))

    options_mc = pack.get("options_mc", []) or []
    opt_rows: List[List[Any]] = [["Rank", "Option", "Stress CVaR10", "Expected Return", "Verdict"]]
    if options_mc:
        ranked = sorted(options_mc, key=lambda x: float((x or {}).get("score_stress_cvar10", 0.0)), reverse=True)
        for i, o in enumerate(ranked, 1):
            name = _safe(o.get("bundle_name", o.get("bundle_id", "")))
            cvar10 = float(o.get("score_stress_cvar10", 0.0))
            mean = float(o.get("score_stress_mean", 0.0))
            verdict = "🏆 Best" if i == 1 else ("Backup" if i == 2 else "Reject" if cvar10 < 0 else "Alternative")
            opt_rows.append([str(i), name, f"{cvar10:.2f}", f"{mean:.2f}", verdict])
    else:
        opt_rows.append(["-", "N/A", "N/A", "N/A", "N/A"])
    story.append(_table(opt_rows, col_widths=[0.6 * inch, 3.0 * inch, 1.0 * inch, 1.0 * inch, 1.2 * inch]))

    story.append(PageBreak())

    # -----------------------------
    # 5) Explainable AI — Attribution & Causality Layer
    # -----------------------------
    story.append(_h("5. Explainable AI — Attribution & Causality Layer", styles, size=14))
    options_explain = pack.get("options_explain", {}) or {}
    chosen_explain = options_explain.get(chosen_id, {}) if chosen_id else {}
    if chosen_explain:
        story.append(_small(f"<b>Impact attribution for chosen option ({chosen_id}) at {chosen_explain.get('main_date','')}</b>", styles))
        kpi_attr = chosen_explain.get("kpi_attributions", {}) or {}
        rows: List[List[Any]] = [["KPI", "Total delta", "Primary driver(s)"]]
        for kpi_id, a in sorted(kpi_attr.items()):
            total = a.get("total_delta", 0.0)
            drivers = a.get("drivers", []) or []
            if drivers:
                top = drivers[0]
                driver_txt = f"{_safe(top.get('driver_id'))}: {_safe(top.get('description'))}"
            else:
                driver_txt = "No drivers identified."
            rows.append([_safe(kpi_id), f"{float(total):+.2f}", driver_txt])
        story.append(_table(rows, col_widths=[1.1 * inch, 1.0 * inch, 4.1 * inch]))
        story.append(_space(0.10))
        story.append(_small("<b>Causal chain (illustrative)</b>", styles))
        story.append(
            _small(
                "Example: Enhanced observability → faster detection → reduced incident resolution time → fewer production incidents.",
                styles,
            )
        )
    else:
        story.append(_small("Attribution not available for chosen option in pack.", styles))

    story.append(PageBreak())

    # -----------------------------
    # 6) Guardrails & Compliance Enforcement
    # -----------------------------
    story.append(_h("6. Guardrails & Compliance Enforcement", styles, size=14))
    story.append(_small("<b>Guardrail rules</b>", styles))
    rules = [
        ["Category", "Threshold (policy / config)"],
        ["Processing time degradation", "≤ +0.20h (p10 worst case)"],
        ["Budget annual cap", "See plan portfolio constraints (enforced)"],
        ["Control coverage", "≥ 95% if automation is affected"],
        ["Incident risk increase", "Forbidden (direction-aware KPI degradation)"],
    ]
    story.append(_table(rules, col_widths=[2.2 * inch, 4.0 * inch]))
    story.append(_space(0.10))

    # Compact per-option verdict derived from guardrails_report text
    gr_map = _parse_guardrails_summary(_safe(pack.get("guardrails_report", "")))
    if gr_map:
        rows: List[List[Any]] = [["Option", "Verdict"]]
        for bid, d in sorted(gr_map.items()):
            status = _safe(d.get("status", ""))
            pill = _pill(status, "success" if status == "PASS" else "danger", styles)
            rows.append([f"{d.get('name','').strip()} ({bid})", pill])
        story.append(_small("<b>Guardrail results (all options)</b>", styles))
        story.append(_table(rows, col_widths=[4.7 * inch, 1.5 * inch]))
        story.append(_space(0.10))

    story.append(_small("<b>Guardrail evidence (raw)</b>", styles))
    story.append(_mono_pre(pack.get("guardrails_report", ""), styles))

    story.append(PageBreak())

    # -----------------------------
    # 7) Learning & Continuous Optimization System
    # -----------------------------
    story.append(_h("7. Learning & Continuous Optimization System", styles, size=14))
    lm = pack.get("learning_metrics", {}) or {}
    if lm:
        rows: List[List[Any]] = [["Metric", "Value"]]
        keys = [
            "total_decisions",
            "accepted_decisions",
            "mean_forecast_accuracy",
            "mean_score_prediction_error",
            "improvement_trend",
            "confidence_multiplier",
        ]
        for k in keys:
            if k in lm:
                rows.append([k, _safe(lm.get(k))])
        story.append(_table(rows, col_widths=[2.5 * inch, 3.7 * inch]))
        recal_kpis = lm.get("forecast_recalibration_kpis", []) or []
        if recal_kpis:
            story.append(_space(0.08))
            story.append(_small("Forecast recalibration applied for: " + ", ".join([_safe(x) for x in recal_kpis]), styles))
    else:
        story.append(_small("No learning metrics available yet (no recorded outcomes).", styles))

    story.append(PageBreak())

    # -----------------------------
    # 8) Audit Trail & Reproducibility Layer
    # -----------------------------
    story.append(_h("8. Audit Trail & Reproducibility Layer", styles, size=14))
    if audit:
        meta = [
            ["Element", "Value"],
            ["Timestamp (as_of)", _safe(audit.get("as_of", ""))],
            ["Code version (git)", _safe(audit.get("code_version_hash", "N/A"))],
            ["Python version", _safe(audit.get("python_version", "N/A"))],
            ["Dataset: plan", _safe(audit.get("plan_path", ""))],
            ["Dataset: kpis", _safe(audit.get("kpis_path", ""))],
            ["Dataset: initiatives", _safe(audit.get("initiatives_path", ""))],
            ["Simulation seed", _safe(audit.get("random_seed", "N/A"))],
            ["Scenarios (n_samples)", _safe((audit.get("config", {}) or {}).get("mc_cfg", {}).get("n_samples", "N/A"))],
        ]
        story.append(_table(meta, col_widths=[2.2 * inch, 4.0 * inch]))
        story.append(_space(0.10))
        story.append(_small("<b>Input hashes</b>", styles))
        story.append(_mono_pre(json.dumps(audit.get("input_hashes", {}) or {}, indent=2, ensure_ascii=False), styles))
        story.append(_space(0.10))
        story.append(_small("<b>Output hashes</b>", styles))
        story.append(_mono_pre(json.dumps(audit.get("outputs_hashes", {}) or {}, indent=2, ensure_ascii=False), styles))
    else:
        story.append(_small("audit.json not found. Reproducibility metadata is unavailable.", styles))

    # Optional: Visual appendix
    visuals_dir = pack_dir / "visuals"
    if include_visual_appendix and visuals_dir.exists():
        pngs = sorted(list(visuals_dir.glob("*.png")))
        if pngs:
            story.append(PageBreak())
            story.append(_h("Appendix — Visual Evidence", styles, size=14))
            story.append(_small("Key visuals generated during pack build (if available).", styles))
            story.append(_space(0.08))
            for p in pngs[:8]:
                story.append(_small(f"<b>{p.name}</b>", styles))
                try:
                    story.append(Image(str(p), width=6.6 * inch, height=3.6 * inch))
                except Exception:
                    story.append(_small("Image could not be rendered in PDF.", styles))
                story.append(_space(0.10))

    # Page numbers
    def _on_page(c: canvas.Canvas, d: SimpleDocTemplate) -> None:
        c.saveState()
        c.setFont("Helvetica", 8)
        c.setFillColor(colors.HexColor(AUDIT_BRAND["muted"]))
        c.drawRightString(A4[0] - 20, 18, f"Page {c.getPageNumber()}")
        c.restoreState()

    doc = SimpleDocTemplate(str(output_path), pagesize=A4, topMargin=40, bottomMargin=40, leftMargin=42, rightMargin=42)
    doc.build(story, onFirstPage=_on_page, onLaterPages=_on_page)

    return str(output_path)

