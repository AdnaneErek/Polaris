# src/dashboard.py
"""
POLARIS: Portfolio Optimization & Learning AI for Risk-Adjusted Strategy
Enterprise-grade strategic planning dashboard with stellar navigation theme.
"""
from __future__ import annotations

import json
import os
import base64
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional
from datetime import datetime

# Ensure project root is in path for imports
_project_root = Path(__file__).parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import pandas as pd
import streamlit as st

try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


# POLARIS Brand Colors
POLARIS_NAVY = "#0A1929"
POLARIS_BLUE = "#1E3A8A"
POLARIS_STELLAR = "#3B82F6"
POLARIS_COSMIC = "#8B5CF6"
POLARIS_GOLD = "#F59E0B"
POLARIS_SUCCESS = "#10B981"
POLARIS_WARNING = "#F59E0B"
POLARIS_DANGER = "#EF4444"
POLARIS_LIGHT = "#F8FAFC"
POLARIS_GRAY = "#64748B"

# Semantic chart/system colors (kept consistent across views)
SERIES_ACTUAL = "#3B82F6"
SERIES_BASELINE = "#F59E0B"
SERIES_FORECAST = "#10B981"
SERIES_RISK = "#EF4444"


# -----------------------------------------------------------------------------
# Professional SVG Icons (Heroicons-style, inline SVG)
# -----------------------------------------------------------------------------
_ICON_SVGS: Dict[str, str] = {
    "shield": """
    <svg class="icon {cls}" viewBox="0 0 24 24" aria-hidden="true">
      <path d="M12 2.5l7 4v6.2c0 5.1-3.3 9.5-7 10.8-3.7-1.3-7-5.7-7-10.8V6.5l7-4z" />
    </svg>
    """,
    "target": """
    <svg class="icon {cls}" viewBox="0 0 24 24" aria-hidden="true">
      <path d="M12 2.5a9.5 9.5 0 1 0 0 19 9.5 9.5 0 0 0 0-19z" />
      <path d="M12 7a5 5 0 1 0 0 10 5 5 0 0 0 0-10z" />
      <path d="M12 10a2 2 0 1 0 0 4 2 2 0 0 0 0-4z" />
    </svg>
    """,
    "switch": """
    <svg class="icon {cls}" viewBox="0 0 24 24" aria-hidden="true">
      <path d="M7 7h10l-2-2" />
      <path d="M17 17H7l2 2" />
      <path d="M7 7v10" />
      <path d="M17 17V7" />
    </svg>
    """,
    "calendar": """
    <svg class="icon {cls}" viewBox="0 0 24 24" aria-hidden="true">
      <path d="M8 3.5v3" />
      <path d="M16 3.5v3" />
      <path d="M4.5 8h15" />
      <path d="M6 5.5h12A1.5 1.5 0 0 1 19.5 7v13A1.5 1.5 0 0 1 18 21.5H6A1.5 1.5 0 0 1 4.5 20V7A1.5 1.5 0 0 1 6 5.5z" />
    </svg>
    """,
    "chart": """
    <svg class="icon {cls}" viewBox="0 0 24 24" aria-hidden="true">
      <path d="M4.5 19.5V5.5" />
      <path d="M4.5 19.5H19.5" />
      <path d="M7.5 16.5l4-5 3 3 5-7" />
    </svg>
    """,
    "spark": """
    <svg class="icon {cls}" viewBox="0 0 24 24" aria-hidden="true">
      <path d="M12 2.5l1.2 5.2L18.5 9l-5.3 1.3L12 15.5l-1.2-5.2L5.5 9l5.3-1.3L12 2.5z" />
      <path d="M18 12l.7 3 3 .7-3 .7-.7 3-.7-3-3-.7 3-.7.7-3z" />
    </svg>
    """,
    "warning": """
    <svg class="icon {cls}" viewBox="0 0 24 24" aria-hidden="true">
      <path d="M12 3.5l9 16H3l9-16z" />
      <path d="M12 9v5" />
      <path d="M12 16.8h.01" />
    </svg>
    """,
    "download": """
    <svg class="icon {cls}" viewBox="0 0 24 24" aria-hidden="true">
      <path d="M12 3.5v10" />
      <path d="M8.5 10.5L12 13.9l3.5-3.4" />
      <path d="M5 20.5h14" />
    </svg>
    """,
    "doc": """
    <svg class="icon {cls}" viewBox="0 0 24 24" aria-hidden="true">
      <path d="M7 3.5h7l3 3v14A1.5 1.5 0 0 1 15.5 22H7A1.5 1.5 0 0 1 5.5 20.5V5A1.5 1.5 0 0 1 7 3.5z" />
      <path d="M14 3.5v4h4" />
      <path d="M8 12h8" />
      <path d="M8 15h8" />
      <path d="M8 18h6" />
    </svg>
    """,
    "settings": """
    <svg class="icon {cls}" viewBox="0 0 24 24" aria-hidden="true">
      <path d="M12 15.5a3.5 3.5 0 1 0 0-7 3.5 3.5 0 0 0 0 7z" />
      <path d="M19.4 12a7.7 7.7 0 0 0-.1-1l2-1.6-2-3.5-2.4 1a8 8 0 0 0-1.7-1l-.4-2.5H10.2l-.4 2.5a8 8 0 0 0-1.7 1l-2.4-1-2 3.5 2 1.6a7.7 7.7 0 0 0 0 2l-2 1.6 2 3.5 2.4-1c.5.4 1.1.7 1.7 1l.4 2.5h4.6l.4-2.5c.6-.3 1.2-.6 1.7-1l2.4 1 2-3.5-2-1.6c.1-.3.1-.7.1-1z" />
    </svg>
    """,
    "robot": """
    <svg class="icon {cls}" viewBox="0 0 24 24" aria-hidden="true">
      <path d="M10 3.5h4" />
      <path d="M12 3.5v3" />
      <path d="M7 9.5h10A2.5 2.5 0 0 1 19.5 12v6A3.5 3.5 0 0 1 16 21.5H8A3.5 3.5 0 0 1 4.5 18v-6A2.5 2.5 0 0 1 7 9.5z" />
      <path d="M9 14h.01" />
      <path d="M15 14h.01" />
      <path d="M9 17h6" />
    </svg>
    """,
}


def _icon(name: str, cls: str = "") -> str:
    """Return inline SVG for a given icon name."""
    svg = _ICON_SVGS.get(name, "")
    return svg.replace("{cls}", cls).strip()


def _header_background_override_css() -> str:
    """
    Build CSS override for the POLARIS header background image.
    Uses inline base64 to avoid static-path issues in Streamlit deployments.
    """
    bg_path = _project_root / "360_F_486384164_j1OsWJc0RhTyomAXWxn5Xz1sv2kTbVRH.jpg"
    if not bg_path.exists():
        return ""
    try:
        encoded = base64.b64encode(bg_path.read_bytes()).decode("ascii")
    except Exception:
        return ""
    return f"""
    <style>
    .polaris-header {{
        background:
            linear-gradient(135deg, rgba(10, 25, 41, 0.72) 0%, rgba(30, 58, 138, 0.78) 100%),
            url("data:image/jpeg;base64,{encoded}");
        background-size: cover;
        background-position: center center;
        background-repeat: no-repeat;
    }}
    </style>
    """


def _header_logo_img_html() -> str:
    """Return an inline <img> tag for the Polaris logo, or empty string if unavailable."""
    logo_path = _project_root / "Polaris logo.png"
    if not logo_path.exists():
        return ""
    try:
        encoded = base64.b64encode(logo_path.read_bytes()).decode("ascii")
    except Exception:
        return ""
    return (
        f'<img class="polaris-logo-icon" '
        f'src="data:image/png;base64,{encoded}" '
        f'alt="Polaris logo" />'
    )


def _header_badge(text: str, tone: str = "info") -> str:
    tone = str(tone).lower()
    if tone == "success":
        bg, fg = "rgba(16,185,129,0.16)", "#6EE7B7"
    elif tone == "warning":
        bg, fg = "rgba(245,158,11,0.16)", "#FCD34D"
    elif tone == "danger":
        bg, fg = "rgba(239,68,68,0.16)", "#FCA5A5"
    else:
        bg, fg = "rgba(59,130,246,0.16)", "#93C5FD"
    return (
        f'<span style="display:inline-block;padding:0.2rem 0.55rem;border-radius:999px;'
        f'background:{bg};color:{fg};font-size:0.75rem;font-weight:600;">{text}</span>'
    )


def _confidence_band(conf: float) -> str:
    c = float(conf)
    if c >= 0.95:
        return "Very high"
    if c >= 0.80:
        return "High"
    if c >= 0.60:
        return "Moderate"
    return "Low"


def _get_kpi_direction_map(plan: Any) -> Dict[str, str]:
    """
    Canonical KPI direction map for dashboard logic.
    Priority:
      1) KPI catalog target.direction (if present)
      2) Objective OKR direction
    """
    out: Dict[str, str] = {}
    try:
        for k in getattr(plan, "kpis", []) or []:
            kid = str(getattr(k, "id", "") or "")
            if not kid:
                continue
            d = str(getattr(getattr(k, "target", None), "direction", "") or "").lower()
            if d in ("up", "down"):
                out[kid] = d
    except Exception:
        pass
    try:
        for obj in getattr(plan, "objectives", []) or []:
            for okr in getattr(obj, "okrs", []) or []:
                kid = str(getattr(okr, "kpi_id", "") or "")
                if not kid:
                    continue
                d = str(getattr(okr, "direction", "") or "").lower()
                if d in ("up", "down"):
                    # OKR direction takes precedence because scoring/eval is objective-driven
                    out[kid] = d
    except Exception:
        pass
    return out


def _render_section_header(title: str, icon_name: str = "chart") -> None:
    st.markdown(
        f'<div class="polaris-card-header"><span class="polaris-card-icon">{_icon(icon_name,"icon-primary")}</span>{title}</div>',
        unsafe_allow_html=True,
    )


def _render_empty_state(message: str, hint: Optional[str] = None) -> None:
    hint_html = f"<div style='margin-top:0.25rem;color:#94A3B8;font-size:0.82rem;'>{hint}</div>" if hint else ""
    st.markdown(
        f"""
        <div style="background: rgba(15,23,42,0.45); border:1px dashed rgba(148,163,184,0.35);
                    border-radius:12px; padding:0.9rem 1rem; margin:0.6rem 0;">
            <div style="color:#CBD5E1; font-weight:600;">{message}</div>
            {hint_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_sticky_decision_bar(pack: Dict[str, Any]) -> None:
    rec = pack.get("recommendation", {}) or {}
    chosen = str(rec.get("chosen_bundle_name", "N/A"))
    conf = float(rec.get("confidence", 0.0) or 0.0)
    band = _confidence_band(conf)
    guardrails_ok = bool(rec.get("guardrails_passed", True))
    approvals = []
    for v in rec.get("guardrails_violations", []) or []:
        approvals.extend(v.get("requires_approval", []) or [])
    approvals = sorted(set([str(a) for a in approvals if a]))
    approvals_txt = ", ".join(approvals) if approvals else "Standard approvals only"

    badge_guard = _header_badge("Guardrails PASS", "success") if guardrails_ok else _header_badge("Guardrails REVIEW", "warning")
    badge_conf = _header_badge(f"Confidence: {conf:.0%} ({band})", "info")
    badge_asof = _header_badge(f"Pack: {pack.get('as_of', 'N/A')}", "info")

    st.markdown(
        f"""
        <div class="polaris-decision-bar">
            <div style="display:flex; gap:0.5rem; flex-wrap:wrap; align-items:center;">
                {badge_asof}
                {badge_conf}
                {badge_guard}
            </div>
            <div style="margin-top:0.45rem; color:#E2E8F0;">
                <span style="font-weight:700;">Decision:</span> {chosen}
                <span style="margin:0 0.5rem; color:#64748B;">|</span>
                <span style="font-weight:700;">Approvals:</span> {approvals_txt}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_kpi_health_strip(pack: Dict[str, Any], plan: Any) -> None:
    forecast_head = pack.get("forecast_head", []) or []
    if not forecast_head:
        _render_empty_state("KPI status unavailable.", "No forecast data in current pack.")
        return
    df = pd.DataFrame(forecast_head)
    if df.empty:
        _render_empty_state("KPI status unavailable.", "Forecast dataframe is empty.")
        return
    try:
        as_of_ts = pd.to_datetime(str(pack.get("as_of")))
        df["date"] = pd.to_datetime(df["date"])
        snap = df[df["date"] == as_of_ts].copy()
        if snap.empty:
            snap = df.sort_values("date").groupby("kpi_id").tail(1)
    except Exception:
        snap = df.groupby("kpi_id").tail(1)

    kpi_map = {k.id: k for k in plan.kpis}
    direction_map = _get_kpi_direction_map(plan)
    target_map: Dict[str, float] = {}
    for obj in getattr(plan, "objectives", []) or []:
        for okr in getattr(obj, "okrs", []) or []:
            kid = str(getattr(okr, "kpi_id", "") or "")
            if not kid:
                continue
            try:
                target_map[kid] = float(getattr(okr, "target", None))
            except Exception:
                continue
    chips = []
    for _, r in snap.iterrows():
        kpi_id = str(r.get("kpi_id"))
        kpi = kpi_map.get(kpi_id)
        short = kpi.short_name if kpi else kpi_id
        f = float(r.get("forecast", 0.0))
        e = float(r.get("expected", 0.0))
        direction = direction_map.get(kpi_id, "up")
        t = target_map.get(kpi_id, None)

        reached_target = False
        if t is not None:
            reached_target = (f >= t) if direction == "up" else (f <= t)

        # Direction-aware "worse than expected" gap:
        # up KPI -> worse if forecast below expected
        # down KPI -> worse if forecast above expected
        worse_gap = (e - f) if direction == "up" else (f - e)
        rel_worse = worse_gap / max(1e-6, abs(e))

        if reached_target:
            tone = "success"
            label = "Target reached"
            arrow = "✓"
        elif worse_gap <= 0:
            tone = "success"
            label = "Ahead of trajectory"
            arrow = "↑" if direction == "up" else "↓"
        elif rel_worse >= 0.08:
            tone = "danger"
            label = "Critical"
            arrow = "↓" if direction == "up" else "↑"
        else:
            tone = "warning"
            label = "Attention"
            arrow = "↔"
        chip = _header_badge(f"{short} {arrow} {label}", tone=tone)
        chips.append(chip)

    st.markdown(
        f"""
        <div style="display:flex; gap:0.45rem; flex-wrap:wrap; margin:0.25rem 0 0.4rem 0;">
            {' '.join(chips)}
        </div>
        """,
        unsafe_allow_html=True,
    )


def apply_polaris_theme():
    """Apply POLARIS stellar navigation theme."""
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Exo+2:wght@600;700;800&display=swap');

    /* Global Styles */
    * { font-family: 'Inter', sans-serif; }

    .main {
        background: linear-gradient(135deg, #0A1929 0%, #1E3A8A 100%);
        padding: 0;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1400px;
    }

    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Professional SVG icons */
    .icon {
        width: 20px;
        height: 20px;
        stroke: #94A3B8;
        stroke-width: 1.9;
        fill: #94A3B8;
        vertical-align: middle;
        margin-right: 0.5rem;
        flex: 0 0 auto;
        display: inline-block;
    }
    .icon-primary { stroke: #3B82F6; fill: #3B82F6; }
    .icon-success { stroke: #10B981; fill: #10B981; }
    .icon-warning { stroke: #F59E0B; fill: #F59E0B; }
    .icon-danger  { stroke: #EF4444; fill: #EF4444; }

    /* POLARIS Header */
    .polaris-header {
        background: linear-gradient(135deg, rgba(10, 25, 41, 0.95) 0%, rgba(30, 58, 138, 0.95) 100%);
        backdrop-filter: blur(10px);
        padding: 1.5rem 2rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        border: 1px solid rgba(59, 130, 246, 0.2);
    }

    .polaris-logo {
        font-family: 'Exo 2', 'Inter', sans-serif;
        font-size: 2.2rem;
        font-weight: 700;
        color: #FFFFFF;
        text-shadow:
            0 0 6px rgba(255, 255, 255, 0.85),
            0 0 14px rgba(147, 197, 253, 0.55),
            0 0 28px rgba(59, 130, 246, 0.35);
        margin: 0;
        transform: translateY(4px);
        letter-spacing: 2px;
    }

    .polaris-logo-wrap {
        display: flex;
        align-items: center;
        gap: 0.45rem;
        margin-left: -0.75rem;
    }

    .polaris-logo-icon {
        width: 180px;
        height: 180px;
        object-fit: contain;
        display: block;
        box-shadow: none;
        border: none;
        background: transparent;
        padding: 0;
        margin-top: -30px;
        margin-bottom: -30px;
    }

    .polaris-tagline {
        font-family: 'Exo 2', 'Inter', sans-serif;
        color: #FFFFFF;
        font-size: 0.875rem;
        font-weight: 600;
        margin-top: 0.25rem;
        letter-spacing: 0.5px;
        text-shadow:
            0 0 2px rgba(255, 255, 255, 0.40),
            0 0 6px rgba(147, 197, 253, 0.20);
    }

    /* Navigation Pills */
    .stRadio > div {
        background: rgba(15, 23, 42, 0.6);
        padding: 0.5rem;
        border-radius: 12px;
        gap: 0.5rem;
        border: 1px solid rgba(59, 130, 246, 0.2);
    }

    .stRadio > div > label {
        background: transparent;
        color: #94A3B8;
        padding: 0.75rem 1.5rem;
        border-radius: 8px;
        transition: all 0.3s ease;
        font-weight: 500;
        border: 1px solid transparent;
    }

    .stRadio > div > label:hover {
        background: rgba(59, 130, 246, 0.1);
        color: #3B82F6;
        border-color: rgba(59, 130, 246, 0.3);
    }

    .stRadio > div > label[data-selected="true"] {
        background: linear-gradient(135deg, #3B82F6 0%, #8B5CF6 100%);
        color: white;
        border-color: transparent;
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.4);
    }

    /* Cards */
    .polaris-card {
        background: rgba(15, 23, 42, 0.6);
        backdrop-filter: blur(10px);
        border-radius: 16px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        border: 1px solid rgba(59, 130, 246, 0.2);
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.2);
        transition: all 0.3s ease;
    }

    .polaris-card:hover {
        border-color: rgba(59, 130, 246, 0.4);
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.3);
        transform: translateY(-2px);
    }

    .polaris-card-header {
        font-size: 1.25rem;
        font-weight: 600;
        color: #F8FAFC;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    .polaris-decision-bar {
        position: sticky;
        top: 0.35rem;
        z-index: 20;
        margin: -0.2rem 0 1rem 0;
        background: rgba(15, 23, 42, 0.90);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(59, 130, 246, 0.28);
        border-left: 4px solid rgba(59, 130, 246, 0.70);
        border-radius: 12px;
        padding: 0.7rem 0.9rem;
        box-shadow: 0 6px 18px rgba(0, 0, 0, 0.25);
    }

    .polaris-card-icon {
        display: inline-flex;
        align-items: center;
        justify-content: center;
    }

    /* Hero Recommendation Card */
    .polaris-hero {
        background: linear-gradient(135deg, rgba(59, 130, 246, 0.15) 0%, rgba(139, 92, 246, 0.15) 100%);
        border: 2px solid rgba(59, 130, 246, 0.4);
        border-radius: 20px;
        padding: 2rem;
        margin: 2rem 0;
        box-shadow: 0 12px 40px rgba(59, 130, 246, 0.3);
        position: relative;
        overflow: hidden;
    }

    .polaris-hero::before {
        content: '';
        position: absolute;
        top: -50%;
        right: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(59, 130, 246, 0.1) 0%, transparent 70%);
        animation: pulse 4s ease-in-out infinite;
    }

    @keyframes pulse {
        0%, 100% { transform: scale(1); opacity: 0.5; }
        50% { transform: scale(1.1); opacity: 0.8; }
    }

    .polaris-hero-label {
        color: #94A3B8;
        font-size: 0.875rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 0.5rem;
    }

    .polaris-hero-title {
        font-size: 2rem;
        font-weight: 700;
        background: linear-gradient(135deg, #3B82F6 0%, #8B5CF6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }

    .polaris-hero-subtitle {
        color: #94A3B8;
        font-size: 1rem;
        font-weight: 400;
    }

    /* Metrics */
    .polaris-metric {
        background: rgba(15, 23, 42, 0.8);
        border-radius: 12px;
        padding: 1.25rem;
        text-align: center;
        border: 1px solid rgba(59, 130, 246, 0.2);
        transition: all 0.3s ease;
    }

    .polaris-metric:hover {
        border-color: rgba(59, 130, 246, 0.4);
        transform: translateY(-2px);
    }

    .polaris-metric-label {
        color: #94A3B8;
        font-size: 0.75rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 0.5rem;
        display: inline-flex;
        align-items: center;
        justify-content: center;
    }

    .polaris-metric-value {
        color: #F8FAFC;
        font-size: 2rem;
        font-weight: 700;
        line-height: 1;
    }

    .polaris-metric-trend {
        color: #10B981;
        font-size: 0.875rem;
        margin-top: 0.5rem;
    }

    /* Status Badges */
    .polaris-badge {
        display: inline-flex;
        align-items: center;
        padding: 0.5rem 1rem;
        border-radius: 8px;
        font-size: 0.875rem;
        font-weight: 600;
        gap: 0.5rem;
    }

    .polaris-badge-success {
        background: rgba(16, 185, 129, 0.2);
        color: #10B981;
        border: 1px solid rgba(16, 185, 129, 0.3);
    }

    .polaris-badge-warning {
        background: rgba(245, 158, 11, 0.2);
        color: #F59E0B;
        border: 1px solid rgba(245, 158, 11, 0.3);
    }

    .polaris-badge-danger {
        background: rgba(239, 68, 68, 0.2);
        color: #EF4444;
        border: 1px solid rgba(239, 68, 68, 0.3);
    }

    .polaris-badge-info {
        background: rgba(59, 130, 246, 0.2);
        color: #3B82F6;
        border: 1px solid rgba(59, 130, 246, 0.3);
    }

    /* Progress Bars */
    .polaris-progress {
        width: 100%;
        height: 8px;
        background: rgba(15, 23, 42, 0.6);
        border-radius: 4px;
        overflow: hidden;
        margin: 0.5rem 0;
    }

    .polaris-progress-bar {
        height: 100%;
        background: linear-gradient(90deg, #3B82F6 0%, #8B5CF6 100%);
        border-radius: 4px;
        transition: width 0.3s ease;
    }

    .polaris-progress-bar-success { background: linear-gradient(90deg, #10B981 0%, #059669 100%); }
    .polaris-progress-bar-warning { background: linear-gradient(90deg, #F59E0B 0%, #D97706 100%); }
    .polaris-progress-bar-danger  { background: linear-gradient(90deg, #EF4444 0%, #DC2626 100%); }

    /* Decision Banner */
    .polaris-decision {
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1.5rem 0;
        border-left: 4px solid;
        font-weight: 500;
    }

    .polaris-decision-approved {
        background: rgba(16, 185, 129, 0.1);
        border-color: #10B981;
        color: #10B981;
    }

    .polaris-decision-required {
        background: rgba(245, 158, 11, 0.1);
        border-color: #F59E0B;
        color: #F59E0B;
    }

    /* KPI Items */
    .polaris-kpi-item {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 1rem;
        background: rgba(15, 23, 42, 0.4);
        border-radius: 8px;
        margin-bottom: 0.75rem;
        border: 1px solid rgba(59, 130, 246, 0.1);
        transition: all 0.2s ease;
    }

    .polaris-kpi-item:hover {
        background: rgba(15, 23, 42, 0.6);
        border-color: rgba(59, 130, 246, 0.3);
    }

    .polaris-kpi-name {
        color: #F8FAFC;
        font-weight: 600;
        flex: 2;
    }

    .polaris-kpi-value {
        color: #94A3B8;
        flex: 1;
        text-align: center;
    }

    .polaris-kpi-status {
        flex: 1;
        text-align: right;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0A1929 0%, #1E3A8A 100%);
        border-right: 1px solid rgba(59, 130, 246, 0.2);
    }

    section[data-testid="stSidebar"] > div { background: transparent; }

    /* Streamlit overrides */
    .stMarkdown { color: #E2E8F0; }
    h1, h2, h3, h4, h5, h6 { color: #F8FAFC !important; }

    /* Tables */
    .dataframe {
        background: rgba(15, 23, 42, 0.6);
        color: #E2E8F0;
        border: 1px solid rgba(59, 130, 246, 0.2);
        border-radius: 8px;
    }

    .dataframe thead tr th {
        background: rgba(59, 130, 246, 0.2);
        color: #F8FAFC;
        font-weight: 600;
    }

    /* Info boxes */
    .stAlert {
        background: rgba(59, 130, 246, 0.1);
        border: 1px solid rgba(59, 130, 246, 0.3);
        color: #E2E8F0;
        border-radius: 8px;
    }

    /* Confidence Gauge */
    .polaris-gauge {
        position: relative;
        width: 120px;
        height: 120px;
        margin: 0 auto;
    }

    .polaris-gauge-bg {
        fill: none;
        stroke: rgba(59, 130, 246, 0.2);
        stroke-width: 10;
    }

    .polaris-gauge-fill {
        fill: none;
        stroke: url(#gaugeGradient);
        stroke-width: 10;
        stroke-linecap: round;
        transition: stroke-dashoffset 0.3s ease;
    }

    .polaris-gauge-text {
        font-size: 2rem;
        font-weight: 700;
        fill: #F8FAFC;
    }

    /* Side sliding navigation bar */
    .nav-sidebar {
        position: fixed;
        left: -250px;
        top: 0;
        height: 100vh;
        width: 250px;
        background: linear-gradient(180deg, rgba(10, 25, 41, 0.98) 0%, rgba(30, 58, 138, 0.98) 100%);
        backdrop-filter: blur(10px);
        z-index: 1000;
        transition: left 0.3s ease;
        padding: 2rem 1rem;
        box-shadow: 4px 0 20px rgba(0, 0, 0, 0.3);
        border-right: 1px solid rgba(59, 130, 246, 0.3);
    }

    .nav-sidebar:hover { left: 0; }

    .nav-sidebar::before {
        content: "≡";
        position: absolute;
        right: -40px;
        top: 20px;
        width: 40px;
        height: 40px;
        background: linear-gradient(135deg, rgba(10, 25, 41, 0.95) 0%, rgba(30, 58, 138, 0.95) 100%);
        display: flex;
        align-items: center;
        justify-content: center;
        border-radius: 0 8px 8px 0;
        font-size: 1.5rem;
        color: #3B82F6;
        cursor: pointer;
        box-shadow: 2px 0 10px rgba(0, 0, 0, 0.2);
    }

    .nav-item {
        display: block;
        text-decoration: none !important;
        padding: 0.75rem 1rem;
        margin: 0.5rem 0;
        border-radius: 8px;
        color: #E2E8F0;
        cursor: pointer;
        transition: all 0.2s ease;
        border-left: 3px solid transparent;
    }

    .nav-item:link,
    .nav-item:visited,
    .nav-item:hover,
    .nav-item:active {
        text-decoration: none !important;
        color: #E2E8F0;
    }

    .nav-item:hover {
        background: rgba(59, 130, 246, 0.2);
        border-left-color: #3B82F6;
        transform: translateX(5px);
    }

    .nav-item.active {
        background: rgba(59, 130, 246, 0.3);
        border-left-color: #3B82F6;
        font-weight: 600;
    }

    /* Stellar Background Effect */
    .polaris-stars {
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        pointer-events: none;
        z-index: -1;
    }

    .polaris-star {
        position: absolute;
        width: 2px;
        height: 2px;
        background: white;
        border-radius: 50%;
        animation: twinkle 3s ease-in-out infinite;
    }

    @keyframes twinkle {
        0%, 100% { opacity: 0.3; }
        50% { opacity: 1; }
    }
    </style>

    <!-- Stellar Background -->
    <div class="polaris-stars" id="stars"></div>
    <script>
    // Generate stars
    const starsContainer = document.getElementById('stars');
    if (starsContainer) {
        for (let i = 0; i < 100; i++) {
            const star = document.createElement('div');
            star.className = 'polaris-star';
            star.style.left = Math.random() * 100 + '%';
            star.style.top = Math.random() * 100 + '%';
            star.style.animationDelay = Math.random() * 3 + 's';
            starsContainer.appendChild(star);
        }
    }
    </script>
    """, unsafe_allow_html=True)

    # Optional header background image override (if JPG is present at project root)
    header_bg_css = _header_background_override_css()
    if header_bg_css:
        st.markdown(header_bg_css, unsafe_allow_html=True)


def render_polaris_header():
    """Render POLARIS header."""
    logo_img = _header_logo_img_html()
    st.markdown("""
    <div class="polaris-header">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div class="polaris-logo-wrap">
                {}
                <div>
                    <div class="polaris-logo">POLARIS</div>
                    <div class="polaris-tagline">Portfolio Optimization & Learning AI for Risk-Adjusted Strategy</div>
                </div>
            </div>
            <div style="text-align: right; color: #94A3B8; font-size: 0.875rem;">
                <div style="font-weight: 600; color: #F8FAFC;">Strategic Intelligence Platform</div>
                <div>{}</div>
            </div>
        </div>
    </div>
    """.format(logo_img, datetime.now().strftime("%B %d, %Y • %H:%M")), unsafe_allow_html=True)


def render_metric_card(label: str, value: str, trend: Optional[str] = None, icon_html: str = ""):
    """Render a metric card."""
    trend_html = f'<div class="polaris-metric-trend">↑ {trend}</div>' if trend else ''

    st.markdown(f"""
    <div class="polaris-metric">
        <div class="polaris-metric-label">{icon_html}{label}</div>
        <div class="polaris-metric-value">{value}</div>
        {trend_html}
    </div>
    """, unsafe_allow_html=True)


def render_confidence_gauge(confidence: float):
    """Render confidence gauge visualization."""
    percentage = confidence * 100
    circumference = 2 * 3.14159 * 45
    offset = circumference - (percentage / 100 * circumference)

    color = "#10B981" if confidence >= 0.7 else "#F59E0B" if confidence >= 0.5 else "#EF4444"

    st.markdown(f"""
    <div style="text-align: center; margin: 1rem 0;">
        <svg class="polaris-gauge" viewBox="0 0 100 100">
            <defs>
                <linearGradient id="gaugeGradient" x1="0%" y1="0%" x2="100%" y2="100%">
                    <stop offset="0%" style="stop-color:{color};stop-opacity:1" />
                    <stop offset="100%" style="stop-color:#8B5CF6;stop-opacity:1" />
                </linearGradient>
            </defs>
            <circle class="polaris-gauge-bg" cx="50" cy="50" r="45"/>
            <circle class="polaris-gauge-fill" cx="50" cy="50" r="45"
                    stroke-dasharray="{circumference}"
                    stroke-dashoffset="{offset}"
                    transform="rotate(-90 50 50)"/>
            <text class="polaris-gauge-text" x="50" y="55" text-anchor="middle">{percentage:.0f}%</text>
        </svg>
        <div style="color: #94A3B8; font-size: 0.875rem; margin-top: 0.5rem;">Confidence Level</div>
    </div>
    """, unsafe_allow_html=True)


def load_pack(pack_path: str) -> Dict[str, Any]:
    """Load SteerCo pack from JSON."""
    with open(pack_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_plan(plan_path: str = "data/plan.yaml") -> Any:
    """Load plan from YAML."""
    from src.load_plan import load_plan
    return load_plan(plan_path)


def _align_as_of_to_series(as_of_ts: pd.Timestamp, x_dates: Optional[pd.Series] = None) -> pd.Timestamp:
    """
    Align as_of marker to nearest available plotted date to avoid mid-period markers
    when data is monthly/quarterly and as_of is a day within period.
    Preference: latest point <= as_of, else earliest point > as_of.
    """
    if x_dates is None:
        return pd.to_datetime(as_of_ts)
    try:
        xs = pd.to_datetime(x_dates, errors="coerce").dropna().sort_values().unique()
        if len(xs) == 0:
            return pd.to_datetime(as_of_ts)
        as_of = pd.to_datetime(as_of_ts)
        le = xs[xs <= as_of]
        if len(le) > 0:
            return pd.to_datetime(le[-1])
        gt = xs[xs > as_of]
        if len(gt) > 0:
            return pd.to_datetime(gt[0])
    except Exception:
        return pd.to_datetime(as_of_ts)
    return pd.to_datetime(as_of_ts)


def _add_as_of_marker(fig: Any, as_of_ts: pd.Timestamp, x_dates: Optional[pd.Series] = None) -> None:
    """
    Plotly-safe vertical marker for datetime x-axes.
    Uses add_shape/add_annotation instead of add_vline to avoid Timestamp arithmetic
    issues in some Plotly/Pandas version combinations.
    """
    aligned = _align_as_of_to_series(as_of_ts, x_dates=x_dates)
    x_val = pd.to_datetime(aligned).to_pydatetime()
    fig.add_shape(
        type="line",
        x0=x_val,
        x1=x_val,
        y0=0,
        y1=1,
        xref="x",
        yref="paper",
        line=dict(color="#94A3B8", width=1.5, dash="dash"),
    )
    fig.add_annotation(
        x=x_val,
        y=1.0,
        xref="x",
        yref="paper",
        text="as_of",
        showarrow=False,
        xanchor="left",
        yanchor="bottom",
        font=dict(color="#94A3B8", size=10),
    )


def _bridge_forecast_with_last_actual(
    forecast_df: pd.DataFrame,
    actual_df: pd.DataFrame,
    forecast_col: str,
    actual_col: str = "value",
) -> pd.DataFrame:
    """
    Prepend the last observed actual point to forecast series so the handoff has no
    visual gap between Actual and Forecast.
    """
    if forecast_df is None or forecast_df.empty or actual_df is None or actual_df.empty:
        return forecast_df

    f = forecast_df.copy()
    a = actual_df.copy()
    a = a.sort_values("date")
    last_actual = a.iloc[-1]

    bridge_row: Dict[str, Any] = {"date": pd.to_datetime(last_actual["date"]), forecast_col: float(last_actual[actual_col])}
    for ci_col in ("lo", "hi"):
        if ci_col in f.columns:
            bridge_row[ci_col] = float(last_actual[actual_col])

    f = pd.concat([pd.DataFrame([bridge_row]), f], ignore_index=True)
    f = f.sort_values("date").drop_duplicates(subset=["date"], keep="last")
    return f


def _load_learning_decisions(pack: Dict[str, Any], pack_path: Optional[str]) -> List[Dict[str, Any]]:
    """
    Load decision history from learning storage, preferring the current pack storage.
    """
    candidates: List[Path] = []
    pack_dir = pack.get("_pack_path")
    if pack_dir:
        candidates.append(Path(pack_dir) / "learning" / "decisions.jsonl")
    if pack_path:
        candidates.append(Path(pack_path).parent / "learning" / "decisions.jsonl")
    candidates.append(Path("artifacts") / "learning" / "decisions.jsonl")

    for fp in candidates:
        try:
            if not fp.exists():
                continue
            rows: List[Dict[str, Any]] = []
            for line in fp.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
            if rows:
                return rows
        except Exception:
            continue
    return []


def _render_floating_chatbot(pack: Dict[str, Any], plan: Any):
    """Render floating chatbot widget in bottom right corner."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return  # Don't show chatbot if API key not available

    # Initialize chat state
    if 'chatbot_open' not in st.session_state:
        st.session_state.chatbot_open = False
    if 'chatbot_history' not in st.session_state:
        st.session_state.chatbot_history = []

    # Floating chatbot button (always visible) - positioned via CSS
    chatbot_button_html = """
    <div class="chatbot-container">
        <div style="position: fixed; bottom: 20px; right: 20px; z-index: 1000;">
    """
    st.markdown(chatbot_button_html, unsafe_allow_html=True)

    # Streamlit buttons can't render HTML icons; keep it clean & professional.
    if st.button("Assistant", key="chatbot_toggle_btn", help="Open POLARIS Assistant", use_container_width=False):
        st.session_state.chatbot_open = not st.session_state.chatbot_open
        st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)

    # Chatbot window (shown when open) - use Streamlit's native components
    if st.session_state.chatbot_open:
        st.markdown("""
        <div style="position: fixed; bottom: 90px; right: 20px; width: 400px; max-height: 600px;
                    background: rgba(10, 25, 41, 0.98); backdrop-filter: blur(10px);
                    border-radius: 16px; box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4);
                    border: 1px solid rgba(59, 130, 246, 0.3); z-index: 999;
                    display: flex; flex-direction: column; overflow: hidden;">
        """, unsafe_allow_html=True)

        col1, col2 = st.columns([4, 1])
        with col1:
            st.markdown("### POLARIS Assistant")
        with col2:
            if st.button("Close", key="chatbot_close_btn", help="Close", use_container_width=True):
                st.session_state.chatbot_open = False
                st.rerun()

        st.markdown('<div style="flex: 1; overflow-y: auto; padding: 1rem;">', unsafe_allow_html=True)

        from src.chat_interface import query_strategic_plan

        for msg in st.session_state.chatbot_history:
            role = "user" if msg['role'] == 'user' else "assistant"
            with st.chat_message(role):
                st.write(msg['content'])

        st.markdown('</div>', unsafe_allow_html=True)

        user_input = st.chat_input("Ask about the strategic plan...", key="chatbot_input")
        if user_input:
            st.session_state.chatbot_history.append({'role': 'user', 'content': user_input})

            with st.chat_message("user"):
                st.write(user_input)

            class PackProxy:
                def __init__(self, data):
                    for key, value in data.items():
                        setattr(self, key, value)

            pack_obj = PackProxy(pack)

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        response = query_strategic_plan(user_input, pack_obj, plan, api_key=api_key)
                        st.write(response)
                        st.session_state.chatbot_history.append({'role': 'assistant', 'content': response})
                    except Exception as e:
                        error_msg = f"Error: {str(e)}"
                        st.write(error_msg)
                        st.session_state.chatbot_history.append({'role': 'assistant', 'content': error_msg})

            st.rerun()

        st.markdown("</div>", unsafe_allow_html=True)


def _generate_pack_automatically(as_of_str: str):
    """Helper function to generate pack automatically."""
    from src.steerco_pack import build_steerco_pack
    from src.monitor import monitor
    from src.steering import get_initiative_health
    from src.kpi_integrity import compute_kpi_integrity
    from src.bundle_generator import generate_action_bundles_with_llm
    from src.situation_summary import generate_situation_summaries
    from src.whatif import StressEvent, WhatIfStochasticConfig, WhatIfConfig
    from src.forecast import ForecastConfig
    import pandas as pd

    plan = load_plan()
    kpis = pd.read_csv("data/simulated_kpis.csv")
    inits = pd.read_csv("data/simulated_initiatives.csv")

    monitor_snapshot = monitor(plan, kpis[["date", "kpi_id", "value"]], as_of_str)
    initiative_health = get_initiative_health(plan, inits, as_of_str)
    integrity = compute_kpi_integrity(plan, kpis[["date", "kpi_id", "value"]], as_of_str)
    integrity_scores = {k: float(v.score) for k, v in integrity.items()}

    ml_anomalies = None
    try:
        from src.monitor_ml import detect_all_kpi_anomalies_ml
        ml_anomalies_dict = detect_all_kpi_anomalies_ml(plan, kpis[["date", "kpi_id", "value"]], as_of_str)
        ml_anomalies = {
            kpi_id: {
                "is_anomaly": result.is_anomaly,
                "anomaly_score": result.anomaly_score,
                "confidence": result.confidence,
                "explanation": result.explanation,
                "method": result.method,
            }
            for kpi_id, result in ml_anomalies_dict.items()
        }
    except Exception:
        pass

    bundles = generate_action_bundles_with_llm(
        plan=plan,
        monitor_snapshot=monitor_snapshot,
        initiative_health=initiative_health,
        ml_anomalies=ml_anomalies,
        integrity_scores=integrity_scores,
        kpi_history=kpis,
        initiative_history=inits,
        as_of=as_of_str,
    )

    situation_summaries = generate_situation_summaries(
        plan=plan,
        monitor_snapshot=monitor_snapshot,
        initiative_health=initiative_health,
        ml_anomalies=ml_anomalies,
        integrity_scores=integrity_scores,
    )

    out_dir = str(Path("artifacts") / "steerco" / as_of_str)
    pack = build_steerco_pack(
        plan_path="data/plan.yaml",
        kpis_csv="data/simulated_kpis.csv",
        initiatives_csv="data/simulated_initiatives.csv",
        as_of=as_of_str,
        horizon_end="2027-12-01",
        bundles=bundles,
        stress_events=[],
        mc_cfg=WhatIfStochasticConfig(n_samples=600, seed=7),
        whatif_cfg=WhatIfConfig(),
        forecast_cfg=ForecastConfig(),
        out_dir=out_dir,
        situation_summaries=situation_summaries,
    )


# ============================================================================
# PAGE 1: STRATEGIC BRIEF
# ============================================================================

def render_strategic_brief(pack: Dict[str, Any], plan: Any):
    """Render strategic executive brief."""
    rec = pack.get('recommendation', {})
    options_mc = pack.get('options_mc', [])

    _render_section_header("KPI Health Snapshot", "target")
    _render_kpi_health_strip(pack, plan)
    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        confidence = rec.get('confidence', 0)
        render_metric_card("Confidence", f"{confidence:.0%}", icon_html=_icon("target", "icon-primary"))

    with col2:
        guardrails_passed = rec.get('guardrails_passed', True)
        status = "PASS" if guardrails_passed else "REVIEW"
        render_metric_card("Guardrails", status, icon_html=_icon("shield", "icon-primary"))

    with col3:
        render_metric_card("Options", str(len(options_mc)), icon_html=_icon("switch", "icon-primary"))

    with col4:
        horizon = pack.get('horizon_end', 'N/A')
        render_metric_card("Horizon", horizon, icon_html=_icon("calendar", "icon-primary"))

    st.markdown("<br>", unsafe_allow_html=True)

    bundle_gen = pack.get("bundle_generation") or {}
    if bundle_gen:
        mode = str(bundle_gen.get("mode", "unknown")).replace("_", " ").title()
        fallback_reason = bundle_gen.get("fallback_reason")
        n_bundles = bundle_gen.get("n_bundles", "N/A")
        note = f"Action generation mode: {mode} | Bundles: {n_bundles}"
        if fallback_reason:
            note += f" | Detail: {fallback_reason}"
        st.markdown(
            f"""
            <div class="polaris-card" style="margin-bottom: 1rem;">
                <div style="color:#E2E8F0; font-size:0.95rem;">
                    <strong>Bundle Provenance:</strong> {note}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    forecast_head = pack.get('forecast_head', [])
    if forecast_head:
        _render_section_header("Strategic Performance", "chart")

        forecast_df = pd.DataFrame(forecast_head)
        kpi_map = {k.id: k for k in plan.kpis}
        okr_map = {}
        for obj in plan.objectives:
            for okr in obj.okrs:
                if okr.kpi_id not in okr_map:
                    okr_map[okr.kpi_id] = okr

        on_track = []
        needs_attention = []

        for kpi_id in forecast_df['kpi_id'].unique():
            kpi_data = forecast_df[forecast_df['kpi_id'] == kpi_id]
            latest = kpi_data.iloc[-1]
            kpi = kpi_map.get(kpi_id)
            okr = okr_map.get(kpi_id)

            if kpi and okr:
                forecast_val = latest.get('forecast', 0)
                target = float(okr.target) if okr.target else None
                baseline = float(okr.baseline) if okr.baseline else forecast_val

                if target:
                    reached_target = False
                    expected_val = float(latest.get('expected', forecast_val))
                    status_class = "danger"
                    status_label = "Requiring Attention"
                    if okr.direction == "up":
                        progress_pct = ((forecast_val - baseline) / (target - baseline) * 100) if (target - baseline) > 0 else 0
                        reached_target = forecast_val >= target
                        ahead_of_trajectory = forecast_val >= expected_val
                        worse_gap = expected_val - forecast_val
                    else:
                        progress_pct = ((baseline - forecast_val) / (baseline - target) * 100) if (baseline - target) > 0 else 0
                        reached_target = forecast_val <= target
                        ahead_of_trajectory = forecast_val <= expected_val
                        worse_gap = forecast_val - expected_val

                    progress_pct = max(0, min(100, progress_pct))
                    rel_worse = float(worse_gap) / max(1e-6, abs(expected_val))
                    if reached_target:
                        status_label = "Target reached"
                        status_class = "success"
                        is_on_track = True
                    elif ahead_of_trajectory:
                        status_label = "On trajectory"
                        status_class = "success"
                        is_on_track = True
                    else:
                        status_label = "Slightly behind trajectory" if rel_worse < 0.05 else "Requiring Attention"
                        status_class = "warning" if rel_worse < 0.05 else "danger"
                        is_on_track = False

                    if is_on_track:
                        on_track.append((kpi.short_name, forecast_val, target, kpi.unit, progress_pct, status_label, status_class))
                    else:
                        needs_attention.append((kpi.short_name, forecast_val, target, kpi.unit, progress_pct, status_label, status_class))

        # Create organized table with columns
        if on_track or needs_attention:
            # Combine both lists for display
            all_kpis = []
            if on_track:
                for name, current, target, unit, pct, status_label, status_class in on_track:
                    all_kpis.append({
                        'KPI': name,
                        'Current': f"{current:.2f} {unit}",
                        'Target': f"{target:.2f} {unit}",
                        'Progress': f"{pct:.1f}%",
                        'Status': status_label,
                        'status_class': status_class,
                        'pct': pct
                    })
            if needs_attention:
                for name, current, target, unit, pct, status_label, status_class in needs_attention:
                    all_kpis.append({
                        'KPI': name,
                        'Current': f"{current:.2f} {unit}",
                        'Target': f"{target:.2f} {unit}",
                        'Progress': f"{pct:.1f}%",
                        'Status': status_label,
                        'status_class': status_class,
                        'pct': pct
                    })
            
            # Display as table with progress bars
            for kpi_data in all_kpis:
                status_val = str(kpi_data['Status'])
                is_good = status_val in ("On trajectory", "Target reached")
                status_color = "#10B981" if is_good else "#EF4444"
                status_bg = "rgba(16, 185, 129, 0.1)" if is_good else "rgba(239, 68, 68, 0.1)"
                
                st.markdown(f"""
                <div style="background: {status_bg}; border-left: 3px solid {status_color}; padding: 1rem; border-radius: 8px; margin: 0.75rem 0;">
                    <div style="display: grid; grid-template-columns: 2fr 1.5fr 1.5fr 1fr 1.5fr; gap: 1rem; align-items: center;">
                        <div style="color: #F8FAFC; font-weight: 600;">{kpi_data['KPI']}</div>
                        <div style="color: #E2E8F0;">{kpi_data['Current']}</div>
                        <div style="color: #E2E8F0;">{kpi_data['Target']}</div>
                        <div style="color: {status_color}; font-weight: 600;">{kpi_data['Progress']}</div>
                        <div>
                            <div class="polaris-progress" style="margin: 0;">
                                <div class="polaris-progress-bar polaris-progress-bar-{kpi_data['status_class']}" style="width: {kpi_data['pct']}%"></div>
                            </div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("Generate New Recommendations", type="primary", use_container_width=True):
                as_of_str = datetime.now().date().isoformat()
                with st.spinner("Generating new strategic recommendations..."):
                    try:
                        _generate_pack_automatically(as_of_str)
                        st.success("New recommendations generated.")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error generating pack: {str(e)}")

        # Executive interpretation line
        try:
            attention = len(needs_attention)
            ok = len(on_track)
            msg = (
                f"Executive readout: {ok} KPI(s) on track, {attention} KPI(s) requiring attention."
                if (ok + attention) > 0
                else "Executive readout: KPI trend health is not conclusive for this pack."
            )
            st.markdown(f"<div style='color:#CBD5E1; margin-top:0.4rem;'><b>So what:</b> {msg}</div>", unsafe_allow_html=True)
        except Exception:
            pass
        st.markdown("<br>", unsafe_allow_html=True)

    situation_summaries = pack.get('situation_summaries', {})
    if situation_summaries:
        _render_section_header("Situations Requiring Attention", "spark")

        for situation_id, summary in situation_summaries.items():
            st.markdown(f"""
            <div style="background: rgba(59, 130, 246, 0.1); border-left: 3px solid #3B82F6; padding: 1rem; margin: 0.75rem 0; border-radius: 6px;">
                <div style="color: #E2E8F0; line-height: 1.6;">{summary}</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
    else:
        _render_section_header("Situations Requiring Attention", "spark")
        _render_empty_state("No elevated situations currently flagged.", "LLM summaries are not present in this pack.")
        st.markdown("<br>", unsafe_allow_html=True)

    if rec:
        chosen_bundle = rec.get('chosen_bundle_name', 'N/A')
        confidence = rec.get('confidence', 0)
        guardrails_passed = rec.get('guardrails_passed', True)

        _render_section_header("Strategic Recommendation", "spark")

        col1, col2 = st.columns([2, 1])

        with col1:
            st.markdown(f"""
            <div class="polaris-hero">
                <div class="polaris-hero-label">Recommended Strategy</div>
                <div class="polaris-hero-title">{chosen_bundle}</div>
                <div class="polaris-hero-subtitle">AI-optimized for risk-adjusted returns</div>
            </div>
            """, unsafe_allow_html=True)

            if not guardrails_passed:
                st.markdown("""
                <div class="polaris-badge polaris-badge-warning" style="margin-top: 0.75rem;">
                    SteerCo decision required (guardrails failed)
                </div>
                """, unsafe_allow_html=True)

            why = rec.get('why', [])
            if why:
                st.markdown("**Rationale:**")
                for i, reason in enumerate(why[:3], 1):
                    reason_clean = reason.replace('⚠️', '(Tie)').replace('■■', '').strip()
                    st.markdown(f"""
                    <div style="color: #E2E8F0; padding: 0.5rem 0; border-left: 2px solid rgba(59, 130, 246, 0.3); padding-left: 1rem; margin: 0.5rem 0;">
                        {i}. {reason_clean}
                    </div>
                    """, unsafe_allow_html=True)

        with col2:
            # Show confidence with context
            if confidence >= 0.95:
                confidence_note = "Very High (top option clearly superior)"
            elif confidence >= 0.75:
                confidence_note = "High (top option preferred)"
            elif confidence >= 0.5:
                confidence_note = "Moderate (options are close)"
            else:
                confidence_note = "Low (options are very close)"
            
            render_confidence_gauge(confidence)
            st.markdown(f"""
            <div style="text-align: center; color: #94A3B8; font-size: 0.75rem; margin-top: 0.5rem;">
                {confidence_note}
            </div>
            """, unsafe_allow_html=True)

            if any('(Tie)' in r or 'tied' in r.lower() for r in why):
                st.markdown("""
                <div class="polaris-badge polaris-badge-info" style="margin-top: 1rem; width: 100%;">
                    Statistical Tie
                </div>
                """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

    if rec:
        _render_section_header("Compliance & Risk Assessment", "shield")

        guardrails_passed = rec.get('guardrails_passed', True)
        violations = rec.get('guardrails_violations', [])

        col1, col2 = st.columns([1, 3])

        with col1:
            if guardrails_passed:
                st.markdown('<div class="polaris-badge polaris-badge-success">PASS</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="polaris-badge polaris-badge-danger">FAIL</div>', unsafe_allow_html=True)

        with col2:
            status_text = "All compliance constraints satisfied" if guardrails_passed else "Compliance violations detected"
            st.markdown(f'<div style="color: #94A3B8;">{status_text}</div>', unsafe_allow_html=True)

        if not guardrails_passed and violations:
            st.markdown("**Violations:**")
            for v in violations[:3]:
                st.markdown(f"""
                <div style="background: rgba(239, 68, 68, 0.1); padding: 0.75rem; border-radius: 8px; border-left: 3px solid #EF4444; margin: 0.5rem 0; color: #FCA5A5;">
                    • {v.get('message', 'Constraint violation')}
                </div>
                """, unsafe_allow_html=True)

        confidence = rec.get('confidence', 0)
        if not guardrails_passed or confidence < 0.5:
            st.markdown("""
            <div class="polaris-decision polaris-decision-required">
                <strong>Decision Required:</strong> Executive review needed before proceeding
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="polaris-decision polaris-decision-approved">
                <strong>Cleared for Execution:</strong> Ready to proceed with standard approvals
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

    options_explain = pack.get('options_explain', {})
    if options_explain and options_mc:
        top_option = sorted(options_mc, key=lambda x: x.get('score_stress_cvar10', 0), reverse=True)[0]
        top_bundle_id = top_option.get('bundle_id')

        if top_bundle_id and top_bundle_id in options_explain:
            attr_data = options_explain[top_bundle_id]
            kpi_attrs = attr_data.get('kpi_attributions', {})

            all_drivers = []
            for kpi_id, attr in kpi_attrs.items():
                drivers = attr.get('drivers', [])
                for driver in drivers:
                    contrib = abs(driver.get('contribution', 0))
                    if contrib > 0.01:
                        all_drivers.append((contrib, driver, kpi_id))

            if all_drivers:
                _render_section_header("Key Performance Drivers", "chart")

                all_drivers.sort(reverse=True, key=lambda x: x[0])
                for i, (contrib, driver, kpi_id) in enumerate(all_drivers[:3], 1):
                    desc = driver.get('description', '')
                    if 'From ' in desc:
                        name = desc.split('From ')[1].split(' (')[0]
                    else:
                        name = driver.get('driver_id', 'Unknown')

                    kpi = next((k for k in plan.kpis if k.id == kpi_id), None)
                    kpi_name = kpi.short_name if kpi else kpi_id

                    st.markdown(f"""
                    <div style="background: rgba(59, 130, 246, 0.1); padding: 1rem; border-radius: 8px; margin: 0.5rem 0; border-left: 3px solid #3B82F6;">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <div>
                                <div style="color: #F8FAFC; font-weight: 600;">{i}. {name}</div>
                                <div style="color: #94A3B8; font-size: 0.875rem;">→ {kpi_name}</div>
                            </div>
                            <div style="color: #3B82F6; font-size: 1.25rem; font-weight: 700;">{contrib:+.2f}</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)
            else:
                _render_section_header("Key Performance Drivers", "chart")
                _render_empty_state("No strong drivers identified.", "Attribution contributions are below display threshold.")
                st.markdown("<br>", unsafe_allow_html=True)


# ============================================================================
# PAGE 2: OPTIONS ANALYSIS
# ============================================================================

def _build_bundle_actions_map(pack: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Read serialized bundle/action definitions from pack JSON."""
    try:
        from src.actions import SteeringAction
    except Exception:
        return {}

    bundle_map: Dict[str, Dict[str, Any]] = {}
    for b in pack.get("bundles", []) or []:
        bid = str(b.get("id", ""))
        if not bid:
            continue
        actions = []
        for a in b.get("actions", []) or []:
            try:
                raw_params = (a.get("parameters", {}) or {})
                parsed_params: Dict[str, Any] = {}
                for k, v in raw_params.items():
                    # Preserve non-numeric params (e.g. workstream_id) while coercing numeric values.
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
        bundle_map[bid] = {
            "name": str(b.get("name", bid)),
            "actions": actions,
        }
    return bundle_map


def _record_selected_option(pack: Dict[str, Any], selected_bundle_id: str, selected_bundle_name: str, accepted: bool = True) -> str:
    """Persist SteerCo's selected option in learning storage."""
    from src.learning import get_learner

    storage_dir = "artifacts/learning"
    pack_dir = pack.get("_pack_path")
    if pack_dir:
        storage_dir = str(Path(pack_dir) / "learning")

    learner = get_learner(storage_dir=storage_dir)
    decisions = learner._load_all_decisions()
    as_of = pack.get("as_of")

    pending = [
        d for d in decisions
        if d.get("evaluation_date") is None and (as_of is None or d.get("as_of") == as_of)
    ]
    if not pending:
        pending = [d for d in decisions if d.get("evaluation_date") is None]
    if not pending:
        raise ValueError("No pending decision found to update. Generate a pack first.")

    target = pending[-1]
    learner.record_option_selection(
        decision_timestamp=str(target.get("timestamp")),
        selected_bundle_id=selected_bundle_id,
        selected_bundle_name=selected_bundle_name,
        accepted=accepted,
    )
    return str(target.get("timestamp"))


def _load_history_inputs_from_pack(pack: Dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load KPI and initiative history from pack-declared sources (fallback to synthetic files)."""
    src = pack.get("sources", {}) or {}
    kpis_csv = str(src.get("kpis_csv", "data/simulated_kpis.csv"))
    inits_csv = str(src.get("initiatives_csv", "data/simulated_initiatives.csv"))

    kpi_df = pd.read_csv(kpis_csv)
    init_df = pd.read_csv(inits_csv)
    return kpi_df, init_df


def _render_option_scenario_forecast(pack: Dict[str, Any], plan: Any, options_mc: List[Dict[str, Any]]) -> None:
    """Visualize baseline forecast vs option scenarios for a selected KPI."""
    if not PLOTLY_AVAILABLE:
        return

    forecast_head = pack.get("forecast_head", []) or []
    if not forecast_head:
        st.info("No forecast series available for option scenario comparison.")
        return

    bundle_map = _build_bundle_actions_map(pack)
    if not bundle_map:
        st.info("Option actions are not available in this pack. Regenerate pack to enable scenario forecast comparison.")
        return

    try:
        kpi_df, init_df = _load_history_inputs_from_pack(pack)
    except Exception as e:
        st.warning(f"Could not load source inputs for scenario simulation: {e}")
        return

    forecast_df = pd.DataFrame(forecast_head).copy()
    if forecast_df.empty or "kpi_id" not in forecast_df.columns:
        st.info("Forecast data is not available for scenario chart.")
        return

    st.markdown('<div class="polaris-card">', unsafe_allow_html=True)
    st.markdown(
        f'<div class="polaris-card-header"><span class="polaris-card-icon">{_icon("chart","icon-primary")}</span> Option Scenario Forecast Comparison</div>',
        unsafe_allow_html=True
    )

    kpi_options = {k.id: f"{k.short_name} ({k.unit})" for k in plan.kpis}
    selected_kpi_id = st.selectbox(
        "KPI for scenario comparison",
        list(kpi_options.keys()),
        format_func=lambda x: kpi_options.get(x, x),
        key="scenario_compare_kpi",
    )

    ranked_options = sorted(options_mc, key=lambda x: x.get("score_stress_cvar10", 0), reverse=True)
    option_labels = {str(o.get("bundle_id", "")): str(o.get("bundle_name", o.get("bundle_id", ""))) for o in ranked_options}
    option_ids = [oid for oid in option_labels.keys() if oid in bundle_map]
    # Default to options with strongest expected effect on the selected KPI (not just top overall rank).
    impact_rank: List[tuple[float, str]] = []
    for o in ranked_options:
        oid = str(o.get("bundle_id", ""))
        if oid not in option_ids:
            continue
        sds = (o.get("stress_delta_summary", {}) or {})
        kpi_summary = sds.get(selected_kpi_id, {}) if isinstance(sds, dict) else {}
        mean_delta = float(kpi_summary.get("mean", 0.0) or 0.0)
        impact_rank.append((abs(mean_delta), oid))
    impact_rank.sort(reverse=True, key=lambda x: x[0])
    default_options = [oid for _, oid in impact_rank[:2] if oid] or (option_ids[:2] if len(option_ids) >= 2 else option_ids[:1])

    selected_option_ids = st.multiselect(
        "Compare options",
        option_ids,
        default=default_options,
        format_func=lambda x: option_labels.get(x, x),
        key=f"scenario_compare_options_{selected_kpi_id}",
    )

    as_of_ts = pd.to_datetime(str(pack.get("as_of")))
    kpi_fc = forecast_df[forecast_df["kpi_id"] == selected_kpi_id].copy()
    if kpi_fc.empty:
        st.info("No baseline forecast available for selected KPI.")
        st.markdown('</div>', unsafe_allow_html=True)
        return

    kpi_fc["date"] = pd.to_datetime(kpi_fc["date"])
    kpi_fc = kpi_fc.sort_values("date")
    kpi_fc_plot = kpi_fc[kpi_fc["date"] > as_of_ts].copy()
    dates_to_check = [d.strftime("%Y-%m-%d") for d in kpi_fc["date"].tolist()]

    # Actuals up to as_of
    kpi_hist = kpi_df[kpi_df["kpi_id"] == selected_kpi_id].copy()
    kpi_hist["date"] = pd.to_datetime(kpi_hist["date"])
    kpi_hist = kpi_hist[(kpi_hist["date"] <= as_of_ts) & (kpi_hist["date"] >= pd.Timestamp("2026-01-01"))].sort_values("date")
    kpi_fc_plot = _bridge_forecast_with_last_actual(kpi_fc_plot, kpi_hist, forecast_col="forecast")

    # Weekly plan baseline
    wb = pd.DataFrame(pack.get("weekly_objectives", []) or [])
    wb_kpi = pd.DataFrame()
    if not wb.empty and {"date", "kpi_id", "baseline_expected"}.issubset(wb.columns):
        wb_kpi = wb[wb["kpi_id"] == selected_kpi_id].copy()
        wb_kpi["date"] = pd.to_datetime(wb_kpi["date"])
        wb_kpi = wb_kpi.sort_values("date")

    fig = go.Figure()
    if not kpi_hist.empty:
        fig.add_trace(go.Scatter(
            x=kpi_hist["date"],
            y=kpi_hist["value"],
            mode="lines+markers",
            name="Actual (observed)",
            line=dict(color="#3B82F6", width=3),
        ))
    if not wb_kpi.empty:
        fig.add_trace(go.Scatter(
            x=wb_kpi["date"],
            y=wb_kpi["baseline_expected"],
            mode="lines",
            name="Plan baseline (weekly objectives)",
            line=dict(color="#F59E0B", width=2, dash="dot"),
        ))
    if not kpi_fc_plot.empty:
        fig.add_trace(go.Scatter(
            x=kpi_fc_plot["date"],
            y=kpi_fc_plot["forecast"],
            mode="lines+markers",
            name="Forecast (model)",
            line=dict(color="#10B981", width=3),
        ))

    _add_as_of_marker(fig, as_of_ts, x_dates=kpi_fc["date"])

    summary_rows: List[Dict[str, str]] = []
    option_delta_series: List[Dict[str, Any]] = []
    direction_map = _get_kpi_direction_map(plan)
    selected_kpi_direction = str(direction_map.get(str(selected_kpi_id), "up")).lower()
    try:
        from src.whatif import compare_actions_at_dates

        for oid in selected_option_ids:
            actions = bundle_map.get(oid, {}).get("actions", [])
            if not actions:
                continue
            sim_df = compare_actions_at_dates(
                plan=plan,
                kpi_history=kpi_df[["date", "kpi_id", "value"]],
                initiative_history=init_df,
                as_of=str(pack.get("as_of")),
                dates_to_check=dates_to_check,
                actions=actions,
                stress_events=[],
            )
            sim_kpi_raw = sim_df[sim_df["kpi_id"] == selected_kpi_id].copy()
            if sim_kpi_raw.empty:
                continue

            sim_kpi_raw["date"] = pd.to_datetime(sim_kpi_raw["date"])
            sim_kpi_raw = sim_kpi_raw.sort_values("date")

            # Rebase option trajectories on the SAME model forecast baseline shown in this chart.
            # This keeps chart gap and table delta in one consistent reference frame.
            model_tail = kpi_fc[kpi_fc["date"] > as_of_ts][["date", "forecast"]].copy()
            delta_tail = sim_kpi_raw[sim_kpi_raw["date"] > as_of_ts][["date", "delta"]].copy()
            aligned = pd.merge(model_tail, delta_tail, on="date", how="inner").sort_values("date")
            if aligned.empty:
                # Fallback if date alignment is unexpectedly empty
                aligned = delta_tail.copy()
                if "forecast" not in aligned.columns:
                    aligned["forecast"] = 0.0
            aligned["with_action_model_ref"] = aligned["forecast"].astype(float) + aligned["delta"].astype(float)

            # Keep deltas for explicit impact chart (still valid, but now aligned to model timeline).
            delta_df = aligned[["date", "delta"]].copy()
            if not delta_df.empty:
                option_delta_series.append(
                    {
                        "option_id": oid,
                        "option_name": option_labels.get(oid, oid),
                        "delta_df": delta_df,
                    }
                )

            # Absolute trajectory (with bridge), now using model-rebased option forecast.
            sim_kpi = aligned[["date", "with_action_model_ref"]].rename(columns={"with_action_model_ref": "with_action"}).copy()
            sim_kpi = _bridge_forecast_with_last_actual(sim_kpi, kpi_hist, forecast_col="with_action")
            fig.add_trace(go.Scatter(
                x=sim_kpi["date"],
                y=sim_kpi["with_action"],
                mode="lines",
                name=option_labels.get(oid, oid),
                line=dict(width=2.5, dash="dash"),
            ))

            try:
                if aligned.empty:
                    continue
                base_last = float(aligned["forecast"].iloc[-1])
                opt_last = float(aligned["with_action_model_ref"].iloc[-1])
                delta_last = float(aligned["delta"].iloc[-1])
                improves = (delta_last < 0) if selected_kpi_direction == "down" else (delta_last > 0)
                impact_label = "Improves" if improves else ("No material change" if abs(delta_last) < 1e-6 else "Degrades")
                summary_rows.append({
                    "Option": option_labels.get(oid, oid),
                    "End baseline (model)": f"{base_last:.3f}",
                    "End with option": f"{opt_last:.3f}",
                    "Delta": f"{delta_last:+.3f}",
                    "Impact on selected KPI": impact_label,
                })
            except Exception:
                pass
    except Exception as e:
        st.warning(f"Could not compute scenario projections: {e}")

    fig.update_layout(
        title="",
        xaxis_title="Date",
        yaxis_title=kpi_options.get(selected_kpi_id, selected_kpi_id),
        height=520,
        plot_bgcolor='rgba(15, 23, 42, 0.6)',
        paper_bgcolor='rgba(0, 0, 0, 0)',
        font=dict(color='#E2E8F0'),
        xaxis=dict(gridcolor='rgba(59, 130, 246, 0.1)'),
        yaxis=dict(gridcolor='rgba(59, 130, 246, 0.1)')
    )
    st.plotly_chart(fig, use_container_width=True, key=f"scenario_compare_chart_{selected_kpi_id}")
    st.markdown(
        "<div style='color:#CBD5E1; margin-top:0.2rem;'><b>So what:</b> The gap between baseline and option lines shows expected strategic leverage on the selected KPI.</div>",
        unsafe_allow_html=True,
    )

    # Explicit impact chart (delta vs what-if baseline) so small effects are visible.
    if option_delta_series:
        fig_delta = go.Figure()
        max_abs_delta = 0.0
        for item in option_delta_series:
            ddf = item["delta_df"]
            if ddf.empty:
                continue
            max_abs_delta = max(max_abs_delta, float(ddf["delta"].abs().max()))
            fig_delta.add_trace(go.Scatter(
                x=ddf["date"],
                y=ddf["delta"],
                mode="lines+markers",
                name=item["option_name"],
                line=dict(width=2.5),
            ))

        fig_delta.add_hline(y=0.0, line_dash="dot", line_color="#94A3B8")
        fig_delta.update_layout(
            title="Option impact vs baseline (delta)",
            xaxis_title="Date",
            yaxis_title=f"Delta ({kpi_options.get(selected_kpi_id, selected_kpi_id)})",
            height=380,
            plot_bgcolor='rgba(15, 23, 42, 0.6)',
            paper_bgcolor='rgba(0, 0, 0, 0)',
            font=dict(color='#E2E8F0'),
            xaxis=dict(gridcolor='rgba(59, 130, 246, 0.1)'),
            yaxis=dict(gridcolor='rgba(59, 130, 246, 0.1)')
        )
        st.plotly_chart(fig_delta, use_container_width=True, key=f"scenario_compare_delta_{selected_kpi_id}")
        st.markdown(
            "<div style='color:#CBD5E1; margin-top:0.2rem;'><b>So what:</b> Positive/negative delta magnitude quantifies option impact versus baseline over time.</div>",
            unsafe_allow_html=True,
        )
        if max_abs_delta < 1e-3:
            st.info("Impacts are near zero for this KPI/time window in the what-if model.")

    if summary_rows:
        st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)

    st.markdown('</div>', unsafe_allow_html=True)


def _get_best_case_score(opt: Dict[str, Any], plan: Any) -> str:
    """Calculate best case (p90) portfolio score from p90 deltas."""
    from src.portfolio import score_option
    
    stress_summary = opt.get('stress_delta_summary', {})
    if not stress_summary:
        # Fallback: use base mean as best case proxy
        return f"{opt.get('score_base_mean', opt.get('score_stress_mean', 0)):.3f}"
    
    # Extract p90 deltas for each KPI (p90 is the 90th percentile, which is better for "up" KPIs and worse for "down" KPIs)
    # For best case, we want the optimistic scenario
    p90_deltas = {}
    for kpi_id, summary in stress_summary.items():
        p90 = summary.get('p90', 0.0)
        p90_deltas[kpi_id] = float(p90)
    
    # Calculate portfolio score from p90 deltas
    # Note: score_option returns a utility score, not the absolute portfolio score
    # We need to add this to a baseline or calculate relative to mean
    try:
        # Calculate utility from p90 deltas
        p90_utility = score_option(plan, p90_deltas)
        # Get mean utility for comparison
        mean_deltas = {kpi_id: summary.get('mean', 0.0) for kpi_id, summary in stress_summary.items()}
        mean_utility = score_option(plan, mean_deltas)
        
        # Best case score = stress_mean + (p90_utility - mean_utility)
        # This approximates what the portfolio score would be at p90
        stress_mean = opt.get('score_stress_mean', 0)
        utility_diff = p90_utility - mean_utility
        best_case_score = stress_mean + utility_diff
        
        return f"{best_case_score:.3f}"
    except Exception as e:
        # Fallback: use base mean as best case (it's typically higher than stress mean)
        return f"{opt.get('score_base_mean', opt.get('score_stress_mean', 0)):.3f}"


def render_options_analysis(pack: Dict[str, Any], plan: Any):
    """Render portfolio options analysis."""
    options_mc = pack.get('options_mc', [])
    if not options_mc:
        st.warning("No options data available")
        return

    st.markdown('<div class="polaris-card">', unsafe_allow_html=True)
    _render_section_header("Strategic Options Comparison", "switch")

    # Build options dataframe with proper handling of missing/zero values
    options_data = []
    for opt in sorted(options_mc, key=lambda x: x.get('score_stress_cvar10', 0), reverse=True):
        robustness_gap = opt.get('robustness_gap_mean')
        if robustness_gap is None:
            # Calculate it if missing: base_mean - stress_mean
            base_mean = opt.get('score_base_mean', 0)
            stress_mean = opt.get('score_stress_mean', 0)
            robustness_gap = base_mean - stress_mean
        
        options_data.append({
            'Option': opt.get('bundle_name', 'N/A'),
            'Risk Score (CVaR10)': f"{opt.get('score_stress_cvar10', 0):.3f}",
            'Expected Return': f"{opt.get('score_stress_mean', 0):.3f}",
            'Base Case': f"{opt.get('score_base_mean', 0):.3f}",
            'Best Case (p90)': _get_best_case_score(opt, plan),
            'Robustness Gap': f"{float(robustness_gap):.4f}",
        })
    
    options_df = pd.DataFrame(options_data)
    st.dataframe(options_df, use_container_width=True, hide_index=True)
    st.markdown(
        "<div style='color:#CBD5E1; margin-top:0.25rem;'><b>So what:</b> Prioritize options with higher CVaR10 and expected return; negative CVaR10 options are riskier under stress.</div>",
        unsafe_allow_html=True,
    )

    # SteerCo decision capture: explicitly record which option was chosen
    option_lookup = {
        str(opt.get("bundle_id", "")): str(opt.get("bundle_name", opt.get("bundle_id", "")))
        for opt in sorted(options_mc, key=lambda x: x.get('score_stress_cvar10', 0), reverse=True)
    }
    option_ids = [oid for oid in option_lookup.keys() if oid]
    if option_ids:
        st.markdown("#### Record SteerCo Selected Option")
        c1, c2 = st.columns([2, 1])
        with c1:
            selected_bundle_id = st.selectbox(
                "Selected option",
                option_ids,
                format_func=lambda x: option_lookup.get(x, x),
                key="selected_option_recording",
            )
        with c2:
            accepted = st.checkbox("Accepted / executed", value=True, key="selected_option_accepted")

        if st.button("Save selected option", use_container_width=False, key="save_selected_option_btn"):
            try:
                ts = _record_selected_option(
                    pack=pack,
                    selected_bundle_id=selected_bundle_id,
                    selected_bundle_name=option_lookup.get(selected_bundle_id, selected_bundle_id),
                    accepted=accepted,
                )
                st.success(f"Saved selection for decision timestamp {ts}: {option_lookup.get(selected_bundle_id, selected_bundle_id)}")
            except Exception as e:
                st.error(f"Could not save selected option: {e}")

    st.markdown('</div>', unsafe_allow_html=True)

    # Approval workflow summary for failing options
    failing = []
    guardrails_report = str(pack.get("guardrails_report", "") or "")
    for opt in options_mc:
        name = str(opt.get("bundle_name", opt.get("bundle_id", "")))
        bid = str(opt.get("bundle_id", ""))
        if f"{bid}): FAIL" in guardrails_report:
            failing.append((name, bid))
    if failing:
        st.markdown('<div class="polaris-card">', unsafe_allow_html=True)
        _render_section_header("Approval Workflow Engine", "warning")
        st.markdown(
            "<div style='color:#E2E8F0; margin-bottom:0.4rem;'>The following options fail automated guardrails and require explicit governance approvals.</div>",
            unsafe_allow_html=True,
        )
        for name, bid in failing:
            st.markdown(
                f"""
                <div style="background: rgba(245, 158, 11, 0.1); padding: 0.9rem; border-radius: 8px; border-left: 3px solid #F59E0B; margin: 0.45rem 0;">
                    <div style="color: #F8FAFC; font-weight: 600;">{name} ({bid})</div>
                    <div style="color: #FCD34D; font-size: 0.86rem; margin-top: 0.2rem;">
                        Evidence required: control matrix, reconciliation procedures, audit trail checks, and minimum 95% control coverage proof.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        st.markdown('</div>', unsafe_allow_html=True)

    if PLOTLY_AVAILABLE and len(options_mc) > 1:
        st.markdown('<div class="polaris-card">', unsafe_allow_html=True)
        _render_section_header("Risk-Return Profile", "chart")

        fig = go.Figure()

        for opt in options_mc:
            fig.add_trace(go.Scatter(
                x=[opt.get('score_stress_cvar10', 0)],
                y=[opt.get('score_stress_mean', 0)],
                mode='markers+text',
                text=[opt.get('bundle_name', 'N/A').split('—')[0].strip()],
                textposition="top center",
                name=opt.get('bundle_name', 'N/A'),
                marker=dict(size=15, line=dict(width=2, color='white'))
            ))

        fig.update_layout(
            title="",
            xaxis_title="Risk (Stress CVaR10)",
            yaxis_title="Expected Return (Stress Mean)",
            height=500,
            plot_bgcolor='rgba(15, 23, 42, 0.6)',
            paper_bgcolor='rgba(0, 0, 0, 0)',
            font=dict(color='#E2E8F0'),
            xaxis=dict(gridcolor='rgba(59, 130, 246, 0.1)'),
            yaxis=dict(gridcolor='rgba(59, 130, 246, 0.1)')
        )
        st.plotly_chart(fig, use_container_width=True, key="risk_return_chart")
        st.markdown(
            "<div style='color:#CBD5E1; margin-top:0.2rem;'><b>So what:</b> Upper-right placement indicates stronger risk-adjusted attractiveness; lower-left indicates weaker trade-off.</div>",
            unsafe_allow_html=True,
        )

        st.markdown('</div>', unsafe_allow_html=True)

    _render_option_scenario_forecast(pack, plan, options_mc)


# ============================================================================
# PAGE 3: MONITORING
# ============================================================================

def render_monitoring(pack: Dict[str, Any], plan: Any):
    """Render KPI monitoring dashboard."""
    forecast_head = pack.get('forecast_head', [])
    if not forecast_head:
        st.warning("No forecast data available")
        return

    forecast_df = pd.DataFrame(forecast_head)
    kpi_map = {k.id: k for k in plan.kpis}

    st.markdown('<div class="polaris-card">', unsafe_allow_html=True)
    _render_section_header("Performance Monitoring", "chart")

    kpi_options = {k.id: f"{k.short_name} ({k.unit})" for k in plan.kpis}
    selected_kpi_id = st.selectbox("Select KPI", list(kpi_options.keys()), format_func=lambda x: kpi_options[x])

    if selected_kpi_id:
        kpi_data = forecast_df[forecast_df['kpi_id'] == selected_kpi_id].copy()
        kpi = kpi_map.get(selected_kpi_id)

        if not kpi_data.empty and kpi and PLOTLY_AVAILABLE:
            as_of_ts = pd.to_datetime(str(pack.get("as_of")))
            kpi_data['date'] = pd.to_datetime(kpi_data['date'])
            kpi_data = kpi_data.sort_values('date')
            kpi_fc = kpi_data[kpi_data['date'] > as_of_ts].copy()

            # Actuals up to as_of
            kpi_hist = pd.DataFrame()
            try:
                kpi_df, _ = _load_history_inputs_from_pack(pack)
                kpi_hist = kpi_df[kpi_df["kpi_id"] == selected_kpi_id].copy()
                kpi_hist["date"] = pd.to_datetime(kpi_hist["date"])
                kpi_hist = kpi_hist[(kpi_hist["date"] <= as_of_ts) & (kpi_hist["date"] >= pd.Timestamp("2026-01-01"))].sort_values("date")
            except Exception:
                kpi_hist = pd.DataFrame()
            kpi_fc = _bridge_forecast_with_last_actual(kpi_fc, kpi_hist, forecast_col="forecast")

            # Weekly baseline
            wb = pd.DataFrame(pack.get("weekly_objectives", []) or [])
            wb_kpi = pd.DataFrame()
            if not wb.empty and {"date", "kpi_id", "baseline_expected"}.issubset(wb.columns):
                wb_kpi = wb[wb["kpi_id"] == selected_kpi_id].copy()
                wb_kpi["date"] = pd.to_datetime(wb_kpi["date"])
                wb_kpi = wb_kpi.sort_values("date")

            fig = go.Figure()

            if not kpi_hist.empty:
                fig.add_trace(go.Scatter(
                    x=kpi_hist['date'],
                    y=kpi_hist['value'],
                    mode='lines+markers',
                    name='Actual (observed)',
                    line=dict(color=SERIES_ACTUAL, width=3)
                ))

            if not wb_kpi.empty:
                fig.add_trace(go.Scatter(
                    x=wb_kpi['date'],
                    y=wb_kpi['baseline_expected'],
                    mode='lines',
                    name='Plan baseline (weekly objectives)',
                    line=dict(color=SERIES_BASELINE, dash='dot', width=2)
                ))

            if not kpi_fc.empty:
                fig.add_trace(go.Scatter(
                    x=kpi_fc['date'],
                    y=kpi_fc['forecast'],
                    mode='lines+markers',
                    name='Forecast (model)',
                    line=dict(color=SERIES_FORECAST, width=3)
                ))

            if not kpi_fc.empty and 'lo' in kpi_fc.columns and 'hi' in kpi_fc.columns:
                fig.add_trace(go.Scatter(
                    x=kpi_fc['date'],
                    y=kpi_fc['hi'],
                    mode='lines',
                    line=dict(width=0),
                    showlegend=False
                ))
                fig.add_trace(go.Scatter(
                    x=kpi_fc['date'],
                    y=kpi_fc['lo'],
                    mode='lines',
                    fill='tonexty',
                    fillcolor='rgba(59, 130, 246, 0.2)',
                    line=dict(width=0),
                    name='Confidence Interval'
                ))

            _add_as_of_marker(fig, as_of_ts, x_dates=kpi_data["date"])

            fig.update_layout(
                title=f"{kpi.short_name} Trajectory",
                xaxis_title="Date",
                yaxis_title=f"{kpi.short_name} ({kpi.unit})",
                height=500,
                plot_bgcolor='rgba(15, 23, 42, 0.6)',
                paper_bgcolor='rgba(0, 0, 0, 0)',
                font=dict(color='#E2E8F0'),
                xaxis=dict(gridcolor='rgba(59, 130, 246, 0.1)'),
                yaxis=dict(gridcolor='rgba(59, 130, 246, 0.1)')
            )
            st.plotly_chart(fig, use_container_width=True, key=f"kpi_chart_{selected_kpi_id}")
            try:
                last_fc = float(kpi_fc['forecast'].iloc[-1]) if not kpi_fc.empty else None
                last_exp = float(kpi_data['expected'].iloc[-1]) if not kpi_data.empty else None
                if last_fc is not None and last_exp is not None:
                    gap = last_fc - last_exp
                    st.markdown(
                        f"<div style='color:#CBD5E1; margin-top:0.2rem;'><b>So what:</b> At horizon, model forecast is <b>{gap:+.2f}</b> vs expected trajectory for this KPI.</div>",
                        unsafe_allow_html=True,
                    )
            except Exception:
                pass

    st.markdown('</div>', unsafe_allow_html=True)

    # Forecast deviation alerts vs weekly objective baseline
    forecast_deviation_alerts = pack.get('forecast_deviation_alerts', {})
    if forecast_deviation_alerts:
        st.markdown('<div class="polaris-card">', unsafe_allow_html=True)
        _render_section_header("Forecast Deviation Alerts (Weekly Baseline)", "warning")

        alert_count = sum(1 for a in forecast_deviation_alerts.values() if a.get('is_deviation_alert', False))
        if alert_count > 0:
            st.markdown(
                f'<div class="polaris-badge polaris-badge-warning">{alert_count} KPI(s) forecasted off weekly objective baseline</div>',
                unsafe_allow_html=True
            )
            for kpi_id, data in forecast_deviation_alerts.items():
                if not data.get('is_deviation_alert', False):
                    continue
                kpi = kpi_map.get(kpi_id)
                sev = str(data.get('max_severity', 'warning')).upper()
                z = float(data.get('max_z_score', 0.0))
                expl = data.get('explanation', 'Deviation alert')
                st.markdown(f"""
                <div style="background: rgba(245, 158, 11, 0.1); padding: 1rem; border-radius: 8px; border-left: 3px solid #F59E0B; margin: 0.5rem 0;">
                    <div style="color: #F8FAFC; font-weight: 600;">{kpi.short_name if kpi else kpi_id} — {sev} (z={z:.2f})</div>
                    <div style="color: #FCD34D; font-size: 0.875rem; margin-top: 0.25rem;">{expl}</div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown(
                '<div class="polaris-badge polaris-badge-success">Forecast is aligned with weekly objective baseline</div>',
                unsafe_allow_html=True
            )
        st.markdown('</div>', unsafe_allow_html=True)

    ml_anomalies = pack.get('ml_anomalies', {})
    if ml_anomalies:
        st.markdown('<div class="polaris-card">', unsafe_allow_html=True)
        _render_section_header("Anomaly Detection", "warning")

        anomaly_count = sum(1 for a in ml_anomalies.values() if a.get('is_anomaly', False))

        if anomaly_count > 0:
            st.markdown(f"""
            <div class="polaris-badge polaris-badge-warning">
                {anomaly_count} anomalies detected
            </div>
            """, unsafe_allow_html=True)

            for kpi_id, anomaly_data in ml_anomalies.items():
                if anomaly_data.get('is_anomaly', False):
                    kpi = kpi_map.get(kpi_id)
                    st.markdown(f"""
                    <div style="background: rgba(245, 158, 11, 0.1); padding: 1rem; border-radius: 8px; border-left: 3px solid #F59E0B; margin: 0.5rem 0;">
                        <div style="color: #F8FAFC; font-weight: 600;">{kpi.short_name if kpi else kpi_id}</div>
                        <div style="color: #FCD34D; font-size: 0.875rem; margin-top: 0.25rem;">{anomaly_data.get('explanation', 'Anomaly detected')}</div>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="polaris-badge polaris-badge-success">
                All systems normal
            </div>
            """, unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)


# ============================================================================
# PAGE 4: EXPORTS
# ============================================================================

def render_exports(pack: Dict[str, Any], pack_path: str, plan: Any):
    """Render export and download section."""
    pack_dir = Path(pack_path).parent

    st.markdown('<div class="polaris-card">', unsafe_allow_html=True)
    _render_section_header("Export Data", "download")

    col1, col2 = st.columns(2)

    with col1:
        pack_json = pack_dir / "steerco_pack.json"
        if pack_json.exists():
            with open(pack_json, 'rb') as f:
                st.download_button(
                    label="Download JSON",
                    data=f.read(),
                    file_name="polaris_pack.json",
                    mime="application/json",
                    use_container_width=True
                )

    with col2:
        pack_md = pack_dir / "steerco_pack.md"
        if pack_md.exists():
            with open(pack_md, 'rb') as f:
                st.download_button(
                    label="Download Markdown",
                    data=f.read(),
                    file_name="polaris_pack.md",
                    mime="text/markdown",
                    use_container_width=True
                )

    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="polaris-card">', unsafe_allow_html=True)
    _render_section_header("Generate Report", "doc")

    from src.report_generator import generate_progress_report
    from src.audit_report_generator import generate_audit_report

    kpis_csv = "data/simulated_kpis.csv"
    inits_csv = "data/simulated_initiatives.csv"
    kpi_preview = pd.DataFrame()
    if Path(kpis_csv).exists():
        try:
            kpi_preview = pd.read_csv(kpis_csv)
            if "date" in kpi_preview.columns:
                kpi_preview["date"] = pd.to_datetime(kpi_preview["date"])
        except Exception:
            kpi_preview = pd.DataFrame()

    report_type = st.selectbox(
        "Report Type",
        ["SteerCo Report", "Audit Report"],
        key="export_report_type",
        help="SteerCo Report = progress & decision pack. Audit Report = governance + traceability audit pack.",
    )

    period_type = st.selectbox("Period Type", ["monthly", "quarterly", "yearly"], key="report_period_type")

    with st.form("report_form"):
        objective_selection = st.selectbox("Scope", ["All Objectives", "Specific Objectives", "Single Objective"])

        start_year = None
        end_year = None
        start_month = None
        end_month = None
        start_quarter = None
        end_quarter = None
        if not kpi_preview.empty and "date" in kpi_preview.columns:
            years_available = sorted(kpi_preview["date"].dt.year.unique().tolist())
            if years_available:
                default_year = None
                try:
                    default_year = int(str(pack.get("as_of", ""))[:4])
                except Exception:
                    default_year = None

                if period_type == "monthly":
                    start_year = st.selectbox(
                        "Start Year",
                        years_available,
                        index=years_available.index(default_year) if default_year in years_available else 0,
                        key="report_monthly_start_year",
                    )
                    end_year = st.selectbox(
                        "End Year",
                        years_available,
                        index=years_available.index(default_year) if default_year in years_available else len(years_available) - 1,
                        key="report_monthly_end_year",
                    )
                    months_available = sorted(kpi_preview["date"].dt.month.unique().tolist())
                    month_names = {
                        1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr",
                        5: "May", 6: "Jun", 7: "Jul", 8: "Aug",
                        9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec",
                    }
                    start_month = st.selectbox(
                        "Start Month",
                        options=months_available,
                        format_func=lambda m: month_names.get(m, str(m)),
                        key="report_monthly_start_month",
                    )
                    end_month = st.selectbox(
                        "End Month",
                        options=months_available,
                        format_func=lambda m: month_names.get(m, str(m)),
                        index=len(months_available) - 1,
                        key="report_monthly_end_month",
                    )
                elif period_type == "quarterly":
                    start_year = st.selectbox(
                        "Start Year",
                        years_available,
                        index=years_available.index(default_year) if default_year in years_available else 0,
                        key="report_quarterly_start_year",
                    )
                    end_year = st.selectbox(
                        "End Year",
                        years_available,
                        index=years_available.index(default_year) if default_year in years_available else len(years_available) - 1,
                        key="report_quarterly_end_year",
                    )
                    quarters_available = [1, 2, 3, 4]
                    start_quarter = st.selectbox(
                        "Start Quarter",
                        options=quarters_available,
                        format_func=lambda q: f"Q{q}",
                        key="report_quarterly_start_quarter",
                    )
                    end_quarter = st.selectbox(
                        "End Quarter",
                        options=quarters_available,
                        format_func=lambda q: f"Q{q}",
                        index=len(quarters_available) - 1,
                        key="report_quarterly_end_quarter",
                    )
                else:
                    start_year = st.selectbox(
                        "Start Year",
                        years_available,
                        index=years_available.index(default_year) if default_year in years_available else 0,
                        key="report_yearly_start_year",
                    )
                    end_year = st.selectbox(
                        "End Year",
                        years_available,
                        index=years_available.index(default_year) if default_year in years_available else len(years_available) - 1,
                        key="report_yearly_end_year",
                    )

        selected_objectives = None
        if objective_selection == "Specific Objectives":
            objective_options = {obj.id: f"{obj.id}: {obj.name}" for obj in plan.objectives}
            selected_objectives = st.multiselect(
                "Choose Objectives",
                options=list(objective_options.keys()),
                format_func=lambda x: objective_options[x]
            )
        elif objective_selection == "Single Objective":
            objective_options = {obj.id: f"{obj.id}: {obj.name}" for obj in plan.objectives}
            selected_obj_id = st.selectbox(
                "Choose Objective",
                options=list(objective_options.keys()),
                format_func=lambda x: objective_options[x]
            )
            if selected_obj_id:
                selected_objectives = [selected_obj_id]

        btn_label = "Generate SteerCo PDF" if report_type == "SteerCo Report" else "Generate Audit PDF"
        generate_button = st.form_submit_button(btn_label, use_container_width=True)

    if 'report_generated' in st.session_state and st.session_state.report_generated:
        pdf_path = st.session_state.report_path

        with open(pdf_path, 'rb') as f:
            st.download_button(
                label="Download PDF Report",
                data=f.read(),
                file_name=Path(pdf_path).name,
                mime="application/pdf",
                use_container_width=True
            )
        st.success("Report generated successfully.")

    elif generate_button:
        as_of = pack.get('as_of', '2026-12-01')

        try:
            with st.spinner("Generating report..."):
                if report_type == "Audit Report":
                    # Audit report is generated directly from the pack directory (+ audit logs + visuals)
                    pdf_path = generate_audit_report(
                        pack_dir=pack_dir,
                        output_path=None,
                        include_visual_appendix=True,
                    )
                else:
                    # SteerCo progress report uses KPI/initiative history as inputs
                    kpi_df = pd.read_csv(kpis_csv)
                    init_df = pd.read_csv(inits_csv) if Path(inits_csv).exists() else pd.DataFrame()
                    if "date" in kpi_df.columns:
                        kpi_df["date"] = pd.to_datetime(kpi_df["date"])
                    if not init_df.empty and "date" in init_df.columns:
                        init_df["date"] = pd.to_datetime(init_df["date"])

                    if period_type == "monthly":
                        if start_year is None or end_year is None or start_month is None or end_month is None:
                            st.error("Please select a start and end month.")
                            st.stop()
                        start_key = (start_year, start_month)
                        end_key = (end_year, end_month)
                        if start_key > end_key:
                            st.error("Start month must be before end month.")
                            st.stop()
                        start_date = pd.Timestamp(year=start_year, month=start_month, day=1)
                        end_date = pd.Timestamp(year=end_year, month=end_month, day=1) + pd.offsets.MonthEnd(0)
                        kpi_df = kpi_df[(kpi_df["date"] >= start_date) & (kpi_df["date"] <= end_date)]
                        if not init_df.empty:
                            init_df = init_df[(init_df["date"] >= start_date) & (init_df["date"] <= end_date)]
                    elif period_type == "quarterly":
                        if start_year is None or end_year is None or start_quarter is None or end_quarter is None:
                            st.error("Please select a start and end quarter.")
                            st.stop()
                        start_key = (start_year, start_quarter)
                        end_key = (end_year, end_quarter)
                        if start_key > end_key:
                            st.error("Start quarter must be before end quarter.")
                            st.stop()
                        start_month_calc = (start_quarter - 1) * 3 + 1
                        end_month_calc = end_quarter * 3
                        start_date = pd.Timestamp(year=start_year, month=start_month_calc, day=1)
                        end_date = pd.Timestamp(year=end_year, month=end_month_calc, day=1) + pd.offsets.MonthEnd(0)
                        kpi_df = kpi_df[(kpi_df["date"] >= start_date) & (kpi_df["date"] <= end_date)]
                        if not init_df.empty:
                            init_df = init_df[(init_df["date"] >= start_date) & (init_df["date"] <= end_date)]
                    elif period_type == "yearly":
                        if start_year is None or end_year is None:
                            st.error("Please select a start and end year.")
                            st.stop()
                        if start_year > end_year:
                            st.error("Start year must be before end year.")
                            st.stop()
                        start_date = pd.Timestamp(year=start_year, month=1, day=1)
                        end_date = pd.Timestamp(year=end_year, month=12, day=31)
                        kpi_df = kpi_df[(kpi_df["date"] >= start_date) & (kpi_df["date"] <= end_date)]
                        if not init_df.empty:
                            init_df = init_df[(init_df["date"] >= start_date) & (init_df["date"] <= end_date)]

                    if not kpi_df.empty:
                        as_of = kpi_df["date"].max().strftime("%Y-%m-%d")
                        kpi_df["date"] = kpi_df["date"].dt.strftime("%Y-%m-%d")
                    if not init_df.empty:
                        init_df["date"] = init_df["date"].dt.strftime("%Y-%m-%d")

                    pdf_path = generate_progress_report(
                        plan=plan,
                        kpi_history=kpi_df,
                        initiative_history=init_df,
                        as_of=str(as_of),
                        period_type=period_type,
                        objective_ids=selected_objectives,
                        pack=pack,
                    )

                st.session_state.report_generated = True
                st.session_state.report_path = pdf_path
                st.rerun()
        except Exception as e:
            st.error(f"Error generating report: {str(e)}")

    st.markdown('</div>', unsafe_allow_html=True)


# ============================================================================
# PAGE 5: ADVANCED
# ============================================================================

def render_advanced(pack: Dict[str, Any], plan: Any, pack_path: str):
    """Render advanced features."""
    tab1, tab2 = st.tabs(["Learning", "Add Objective"])

    with tab1:
        st.markdown('<div class="polaris-card">', unsafe_allow_html=True)
        _render_section_header("Learning System", "spark")

        learning_metrics = pack.get('learning_metrics', {})
        if learning_metrics:
            c1, c2, c3 = st.columns(3)
            c1.metric("Forecast accuracy", f"{float(learning_metrics.get('mean_forecast_accuracy', 0.0)):.1%}")
            c2.metric("Confidence multiplier", f"{float(learning_metrics.get('confidence_multiplier', 1.0)):.2f}x")
            c3.metric("Score error", f"{float(learning_metrics.get('mean_score_prediction_error', 0.0)):.3f}")

            recal_kpis = learning_metrics.get("forecast_recalibration_kpis", []) or []
            if recal_kpis:
                st.markdown("**Forecast recalibration applied for KPIs:** " + ", ".join([str(k) for k in recal_kpis]))
            else:
                st.markdown("**Forecast recalibration applied for KPIs:** none")

            weight_overrides = learning_metrics.get("scoring_weight_overrides", {}) or {}
            if weight_overrides:
                rows = [
                    {"KPI": str(k), "Weight Multiplier": f"{float(v):.3f}"}
                    for k, v in sorted(weight_overrides.items())
                ]
                st.markdown("**Portfolio scoring weight overrides (learned):**")
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
            else:
                st.markdown("**Portfolio scoring weight overrides (learned):** none")

            with st.expander("Raw learning metrics"):
                st.json(learning_metrics)
        else:
            st.info("No learning metrics available yet. Record decision outcomes to enable learning.")

        st.markdown("### Selection History")
        decisions = _load_learning_decisions(pack, pack_path)
        if decisions:
            # Focus on current pack by default; allow viewing all
            current_as_of = str(pack.get("as_of", ""))
            show_all = st.checkbox("Show all as_of periods", value=False, key="selection_history_show_all")
            data = decisions if show_all else [d for d in decisions if str(d.get("as_of", "")) == current_as_of]

            if not data:
                st.info(f"No decisions found for as_of={current_as_of}.")
            else:
                rows = []
                selected_count = 0
                override_count = 0
                pending_count = 0

                for d in data:
                    recommended_id = str(d.get("recommended_bundle_id", "N/A"))
                    recommended_name = str(d.get("recommended_bundle_name", recommended_id))
                    selected_id = str(d.get("selected_bundle_id", "")) if d.get("selected_bundle_id") else ""
                    selected_name = str(d.get("selected_bundle_name", selected_id)) if selected_id else ""
                    if selected_id:
                        selected_count += 1
                    if selected_id and selected_id != recommended_id:
                        override_count += 1

                    eval_date = d.get("evaluation_date")
                    if not eval_date:
                        pending_count += 1
                    status = "Outcome recorded" if eval_date else "Pending outcome"

                    accepted = d.get("accepted")
                    if accepted is True:
                        accepted_txt = "Yes"
                    elif accepted is False:
                        accepted_txt = "No"
                    else:
                        accepted_txt = "N/A"

                    rows.append({
                        "As Of": str(d.get("as_of", "")),
                        "Decision Timestamp": str(d.get("timestamp", "")),
                        "Recommended": f"{recommended_name} ({recommended_id})",
                        "Selected": f"{selected_name} ({selected_id})" if selected_id else "Not recorded",
                        "Accepted": accepted_txt,
                        "Planned Eval": str(d.get("planned_evaluation_date", "")),
                        "Evaluation Date": str(eval_date) if eval_date else "",
                        "Status": status,
                    })

                s1, s2, s3 = st.columns(3)
                s1.metric("Selections recorded", f"{selected_count}/{len(data)}")
                s2.metric("Overrides (selected != recommended)", str(override_count))
                s3.metric("Pending outcomes", str(pending_count))

                hist_df = pd.DataFrame(rows)
                if "Decision Timestamp" in hist_df.columns:
                    hist_df = hist_df.sort_values("Decision Timestamp", ascending=False)
                st.dataframe(hist_df, use_container_width=True, hide_index=True)
        else:
            st.info("No selection history found yet. Record a selected option in Options Analysis.")

        st.markdown('</div>', unsafe_allow_html=True)

    with tab2:
        st.markdown('<div class="polaris-card">', unsafe_allow_html=True)
        _render_section_header("Add Strategic Objective", "settings")

        objective_text = st.text_area(
            "Describe your objective in natural language:",
            height=150,
            placeholder="e.g., Reduce payment processing errors by 50% by end of 2027..."
        )

        if st.button("Parse Objective", use_container_width=True):
            if objective_text:
                from src.objective_parser import parse_objective_from_text
                api_key = os.getenv("GEMINI_API_KEY")
                if not api_key:
                    st.error("GEMINI_API_KEY not found in environment")
                else:
                    try:
                        with st.spinner("Analyzing..."):
                            result = parse_objective_from_text(objective_text, api_key, plan)
                            if result.get('success'):
                                st.success("Objective parsed successfully.")
                                st.json(result.get('objective', {}))
                            else:
                                st.warning(result.get('message', 'Parsing failed'))
                                if result.get('clarifications'):
                                    st.markdown("**Clarifications needed:**")
                                    for q in result['clarifications']:
                                        st.markdown(f"- {q}")
                    except Exception as e:
                        st.error(f"Error: {str(e)}")

        st.markdown('</div>', unsafe_allow_html=True)


# ============================================================================
# MAIN APP
# ============================================================================

def main():
    """Main POLARIS application."""
    st.set_page_config(
        page_title="POLARIS | Strategic Intelligence",
        page_icon="⭐",  # Streamlit page_icon cannot be SVG; keep minimal.
        layout="wide",
        initial_sidebar_state="expanded"
    )

    apply_polaris_theme()
    render_polaris_header()

    pack = None
    plan = None
    selected_pack_path = None
    
    try:
        with st.sidebar:
            st.markdown(f"### <span style='display: inline-flex; align-items: center;'>{_icon('settings','icon-primary')}Configuration</span>", unsafe_allow_html=True)

            artifacts_dir = Path("artifacts/steerco")
            available_packs = []
            if artifacts_dir.exists():
                for pack_dir in sorted(artifacts_dir.iterdir(), reverse=True):
                    pack_json = pack_dir / "steerco_pack.json"
                    if pack_json.exists():
                        available_packs.append((pack_dir.name, str(pack_json)))

            if not available_packs:
                st.info("No packs available. Generate a new pack from the main dashboard when needed.")
                st.stop()

            default_index = 0
            today_str = datetime.now().date().isoformat()
            for i, (name, _) in enumerate(available_packs):
                if name == today_str:
                    default_index = i
                    break

            selected_pack_name = st.selectbox(
                "Select Pack",
                [name for name, _ in available_packs],
                index=default_index
            )

            selected_pack_path = next(path for name, path in available_packs if name == selected_pack_name)

            try:
                pack = load_pack(selected_pack_path)
                pack['_pack_path'] = Path(selected_pack_path).parent
                plan = load_plan()
            except Exception as e:
                st.error(f"Error loading pack: {str(e)}")
                import traceback
                st.exception(e)
                st.stop()
                
    except Exception as e:
        st.error(f"Error in sidebar: {str(e)}")
        import traceback
        st.exception(e)
        st.stop()

        st.markdown("---")
        st.markdown("**As of:** " + pack.get('as_of', 'N/A'))
        st.markdown("**Horizon:** " + pack.get('horizon_end', 'N/A'))

        rec = pack.get('recommendation', {})

        st.markdown("---")
        render_metric_card("Confidence", f"{rec.get('confidence', 0):.0%}", icon_html=_icon("target", "icon-primary"))

        st.markdown("---")
        st.markdown("""
        <div style="text-align: center; color: #64748B; font-size: 0.75rem; margin-top: 2rem;">
            <div>POLARIS v1.0</div>
            <div style="margin-top: 0.25rem;">Strategic Intelligence Platform</div>
        </div>
        """, unsafe_allow_html=True)

    pages = ["Strategic Brief", "Options Analysis", "Monitoring", "Exports", "Advanced"]

    def get_page_from_query(pages: list[str]) -> str:
        """Get current page from query parameters."""
        try:
            qp = st.query_params
            page = qp.get("page", None)
            if isinstance(page, list):
                page = page[0] if page else None
        except Exception:
            try:
                qp = st.experimental_get_query_params()
                page = qp.get("page", [None])[0]
            except Exception:
                page = None

        return page if page in pages else pages[0]

    selected = get_page_from_query(pages)
    st.session_state["current_page"] = selected

    nav_items_html = ""
    for page_name in pages:
        active_class = "active" if page_name == selected else ""
        page_url = page_name.replace(' ', '%20')
        nav_items_html += f'<a class="nav-item {active_class}" href="?page={page_url}">{page_name}</a>'

    sidebar_html = f"""
    <div class="nav-sidebar" id="navSidebar">
        <div style="color: #3B82F6; font-weight: 600; font-size: 1.1rem; margin-bottom: 2rem; padding-bottom: 1rem; border-bottom: 1px solid rgba(59, 130, 246, 0.3);">
            POLARIS
        </div>
        {nav_items_html}
    </div>
    """
    st.markdown(sidebar_html, unsafe_allow_html=True)

    selected = get_page_from_query(pages)
    st.session_state["current_page"] = selected

    nav_items_html = ""
    for page_name in pages:
        active_class = "active" if page_name == selected else ""
        page_url = page_name.replace(' ', '%20')
        nav_items_html += f'<a class="nav-item {active_class}" href="?page={page_url}">{page_name}</a>'

    sidebar_html = f"""
    <div class="nav-sidebar" id="navSidebar">
        <div style="color: #3B82F6; font-weight: 600; font-size: 1.1rem; margin-bottom: 2rem; padding-bottom: 1rem; border-bottom: 1px solid rgba(59, 130, 246, 0.3);">
            POLARIS
        </div>
        {nav_items_html}
    </div>
    """
    st.markdown(sidebar_html, unsafe_allow_html=True)

    page = st.session_state["current_page"]

    # Sticky decision context across pages
    if pack:
        _render_sticky_decision_bar(pack)

    if page == "Strategic Brief":
        render_strategic_brief(pack, plan)
    elif page == "Options Analysis":
        render_options_analysis(pack, plan)
    elif page == "Monitoring":
        render_monitoring(pack, plan)
    elif page == "Exports":
        render_exports(pack, selected_pack_path, plan)
    elif page == "Advanced":
        render_advanced(pack, plan, selected_pack_path)

    _render_floating_chatbot(pack, plan)


if __name__ == "__main__":
    main()
