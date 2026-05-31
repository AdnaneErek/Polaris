# src/agent.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple

from datetime import datetime
from dateutil.relativedelta import relativedelta
import pandas as pd

from .schemas import Plan, KPI
from .monitor import MonitorSnapshot
from .actions import SteeringAction
from .whatif import compare_actions_at_dates, StressEvent


# -----------------------------
# Agent outputs
# -----------------------------
@dataclass
class Recommendation:
    title: str
    rationale: str
    actions: List[str]
    expected_impact: List[str]
    risks: List[str]
    confidence: float
    requires_approval: List[str]


@dataclass
class AgentBrief:
    as_of: str
    headline: str
    anomalies: List[str]
    recommendations: List[Recommendation]
    needs_human_decision: bool


# -----------------------------
# Helpers
# -----------------------------
def _kpi_by_id(plan: Plan) -> Dict[str, KPI]:
    return {k.id: k for k in plan.kpis}


def _months_between(a: str, b: str) -> int:
    ya, ma = int(a[:4]), int(a[5:7])
    yb, mb = int(b[:4]), int(b[5:7])
    return (yb - ya) * 12 + (mb - ma)


def _plan_mid(plan: Plan) -> str:
    s = plan.meta.horizon.start
    e = plan.meta.horizon.end
    total = _months_between(s, e)
    mid_m = total // 2
    y, m = int(s[:4]), int(s[5:7])
    y2 = y + (m - 1 + mid_m) // 12
    m2 = (m - 1 + mid_m) % 12 + 1
    return f"{y2:04d}-{m2:02d}-01"


def _safe_approvers(plan: Plan) -> List[str]:
    try:
        return list(plan.meta.governance.decision_rights.approve)
    except Exception:
        return ["Transformation Lead", "IT Ops Head", "Risk/Compliance"]


def _safe_drift_threshold(plan: Plan) -> float:
    try:
        return float(plan.reporting.steering_committee.decision_thresholds.drift_confidence)
    except Exception:
        return 0.60


def _format_delta(kpi_id: str, delta: float) -> str:
    if kpi_id in ("KPI_STP", "STP_rate"):
        return f"{delta:+.2f} pp"
    if kpi_id in ("KPI_E2E", "processing_time"):
        return f"{delta:+.3f} h"
    if kpi_id in ("KPI_INC", "incident_count"):
        return f"{delta:+.2f}"
    return f"{delta:+.3f}"


def _rank(recs: List[Recommendation]) -> List[Recommendation]:
    return sorted(recs, key=lambda r: r.confidence, reverse=True)


def _iso_month_start(d: datetime) -> str:
    return d.replace(day=1).date().isoformat()


def _month_start_str(date_iso: str) -> str:
    dt = datetime.fromisoformat(date_iso)
    return _iso_month_start(dt)


def _fallback_checkpoints(as_of: str) -> List[str]:
    t = datetime.fromisoformat(as_of)
    return [
        _iso_month_start(t + relativedelta(months=+6)),
        _iso_month_start(t + relativedelta(months=+12)),
        _iso_month_start(t + relativedelta(months=+24)),
    ]


def _default_checkpoints(plan: Plan, as_of: str, max_n: int = 3) -> List[str]:
    as_of_month = _month_start_str(as_of)
    dates = set()
    try:
        for obj in plan.objectives:
            for okr in obj.okrs:
                for cp in okr.trajectory.checkpoints:
                    ms = _month_start_str(cp.date)
                    if ms > as_of_month:
                        dates.add(ms)
    except Exception:
        dates = set()

    picked = sorted(dates)[:max_n]
    if picked:
        return picked
    return _fallback_checkpoints(as_of)[:max_n]


# -----------------------------
# KPI ID augmentation (robust)
# -----------------------------
def _augment_kpi_ids(kpis_df: pd.DataFrame) -> pd.DataFrame:
    if kpis_df is None or kpis_df.empty:
        return kpis_df

    df = kpis_df.copy()
    df["kpi_id"] = df["kpi_id"].astype(str)

    pairs = [
        ("KPI_STP", "STP_rate"),
        ("KPI_E2E", "processing_time"),
        ("KPI_INC", "incident_count"),
    ]

    extra = []
    for a, b in pairs:
        mask_a = df["kpi_id"] == a
        if mask_a.any():
            tmp = df.loc[mask_a].copy()
            tmp["kpi_id"] = b
            extra.append(tmp)

        mask_b = df["kpi_id"] == b
        if mask_b.any():
            tmp = df.loc[mask_b].copy()
            tmp["kpi_id"] = a
            extra.append(tmp)

    if extra:
        df = pd.concat([df] + extra, ignore_index=True)
    return df


# -----------------------------
# Anomaly checks
# -----------------------------
def _kpi_anomaly_checks(plan: Plan, snap: MonitorSnapshot) -> List[str]:
    anomalies: List[str] = []
    kmap = _kpi_by_id(plan)
    mid = _plan_mid(plan)

    # Rule-based anomaly checks (existing)
    for os in snap.objective_statuses:
        for ks in os.kpi_statuses:
            if ks.actual is None:
                continue
            k = kmap.get(ks.kpi_id)
            if not k:
                continue

            if k.target.direction == "up":
                if ks.actual >= 99.0 and snap.as_of < mid:
                    anomalies.append(
                        f"KPI {k.short_name} is {ks.actual:.2f}{k.unit} very early (before {mid}). "
                        f"Verify KPI definition, denominator, and lineage."
                    )
                if ks.actual > float(k.target.value) + 3.0 and snap.as_of < mid:
                    anomalies.append(
                        f"KPI {k.short_name} exceeds target unusually early (actual {ks.actual:.2f} vs target {k.target.value:.2f}). "
                        f"Check for measurement bias or clamping."
                    )

            if k.target.direction == "down":
                if ks.actual < 0.75 * float(k.target.value) and snap.as_of < mid:
                    anomalies.append(
                        f"KPI {k.short_name} is far below target early (actual {ks.actual:.2f} vs target {k.target.value:.2f}). "
                        f"Check for reporting bias/scope changes."
                    )
    
    # ML-based anomaly detection (if available)
    if snap.ml_anomalies:
        for kpi_id, ml_result in snap.ml_anomalies.items():
            if ml_result.is_anomaly:
                k = kmap.get(kpi_id)
                kpi_name = k.short_name if k else kpi_id
                anomalies.append(
                    f"ML detected: {ml_result.explanation} "
                    f"(confidence: {ml_result.confidence:.2f}, method: {ml_result.method})"
                )

    return anomalies


# -----------------------------
# Dependency watch
# -----------------------------
def _dependency_watch(plan: Plan, initiatives_df: pd.DataFrame, as_of: str) -> List[str]:
    if initiatives_df is None or initiatives_df.empty:
        return []

    df = initiatives_df.copy()
    df = df[df["date"] <= as_of].sort_values("date")
    if df.empty:
        return []

    latest = df.groupby("initiative_id").tail(1)
    prog = dict(zip(latest["initiative_id"], latest["progress"]))

    notes: List[str] = []
    for it in plan.initiatives:
        blockers = []
        for dep in it.dependencies:
            if dep.type == "hard":
                dep_prog = float(prog.get(dep.initiative_id, 0.0))
                if dep_prog < 0.60:
                    blockers.append(dep.initiative_id)
        if blockers:
            notes.append(f"{it.id} blocked by hard dependencies: {', '.join(blockers)}")
    return notes


# -----------------------------
# What-if formatting helpers
# -----------------------------
def _extract_deltas_for_date(comp: pd.DataFrame, date: str) -> List[Tuple[str, float]]:
    sub = comp[comp["date"] == date]
    out: List[Tuple[str, float]] = []
    for _, row in sub.iterrows():
        out.append((str(row["kpi_id"]), float(row["delta"])))
    return out


def _tradeoff_note(deltas: List[Tuple[str, float]]) -> Optional[str]:
    d = dict(deltas)
    stp = d.get("KPI_STP", d.get("STP_rate"))
    e2e = d.get("KPI_E2E", d.get("processing_time"))
    if stp is None or e2e is None:
        return None
    if e2e < -1e-6 and stp < -1e-6:
        return "Trade-off: faster processing time improvements, but a slight reduction in STP benefit (likely due to scope-splitting/coverage effects)."
    return None


def _filter_material_deltas(deltas: List[Tuple[str, float]]) -> List[Tuple[str, float]]:
    """
    Hide cosmetic 0.000 outputs.
    Thresholds are KPI-specific to keep reporting realistic.
    """
    out: List[Tuple[str, float]] = []
    for kid, dv in deltas:
        # thresholds per KPI
        if kid in ("KPI_STP", "STP_rate"):
            thr = 0.05     # 0.05 percentage points
        elif kid in ("KPI_E2E", "processing_time"):
            thr = 0.01     # 0.01 hours (~36s)
        elif kid in ("KPI_INC", "incident_count"):
            thr = 0.05     # 0.05 incidents (synthetic smooth models may produce fractions)
        else:
            thr = 1e-6

        if abs(dv) >= thr:
            out.append((kid, dv))
    return out


# -----------------------------
# Main agent logic
# -----------------------------
def generate_recommendations(
    plan: Plan,
    snap: MonitorSnapshot,
    kpis_df: Optional[pd.DataFrame] = None,
    initiatives_df: Optional[pd.DataFrame] = None,
    events_df: Optional[pd.DataFrame] = None,
    stress_events: Optional[List[StressEvent]] = None,
) -> AgentBrief:
    drift_thr = _safe_drift_threshold(plan)
    approvers = _safe_approvers(plan)
    stress_events = stress_events or []

    anomalies = _kpi_anomaly_checks(plan, snap)

    dep_notes = []
    if initiatives_df is not None and not initiatives_df.empty:
        dep_notes = _dependency_watch(plan, initiatives_df, snap.as_of)
        anomalies.extend([f"Dependency watch: {x}" for x in dep_notes])

    # Augment KPI ids so both naming conventions exist
    kpis_df_norm = kpis_df
    if kpis_df is not None and not kpis_df.empty:
        kpis_df_norm = _augment_kpi_ids(kpis_df)

    # Candidate steering action sets (PoC defaults)
    ACTION_ACCEL_DQ = [
        SteeringAction(
            id="A_ACCEL_DQ1_2M",
            type="accelerate_initiative",
            target_initiative="INIT_DQ1",
            parameters={"months": 2},
            description="Accelerate INIT_DQ1 by ~2 months (extra capacity/focus).",
        )
    ]
    ACTION_SPLIT_A1 = [
        SteeringAction(
            id="A_SPLIT_A1",
            type="split_scope",
            target_initiative="INIT_A1",
            parameters={"reduction": 0.6},
            description="Split INIT_A1 scope to reduce dependency blocking.",
        )
    ]
    ACTION_ACCEL_RES = [
        SteeringAction(
            id="A_ACCEL_RES1_2M",
            type="accelerate_initiative",
            target_initiative="INIT_RES1",
            parameters={"months": 2},
            description="Accelerate INIT_RES1 by ~2 months (SRE capacity/tooling focus).",
        )
    ]

    # Default "portfolio steering" bundle
    ACTION_PORTFOLIO = ACTION_ACCEL_DQ + ACTION_SPLIT_A1 + ACTION_ACCEL_RES

    # Helper: attach quantified what-if deltas
    def attach_whatif(rec: Recommendation, actions: List[SteeringAction]) -> Recommendation:
        if kpis_df_norm is None or initiatives_df is None:
            return rec
        if kpis_df_norm.empty or initiatives_df.empty:
            return rec

        checkpoint_dates = _default_checkpoints(plan, snap.as_of, max_n=3)

        comp = compare_actions_at_dates(
            plan,
            kpis_df_norm,
            initiatives_df,
            as_of=snap.as_of,
            dates_to_check=checkpoint_dates,
            actions=actions,
            stress_events=stress_events,
        )

        if comp is None or comp.empty:
            rec.expected_impact = rec.expected_impact + [
                "What-if quantification unavailable (empty comparison). Check simulator date grid and KPI ids."
            ]
            return rec

        for d in checkpoint_dates:
            raw_deltas = _extract_deltas_for_date(comp, d)
            mat = _filter_material_deltas(raw_deltas)

            if mat:
                parts = [f"{kid}: {_format_delta(kid, dv)}" for kid, dv in mat]
                rec.expected_impact.append(f"Projected delta at {d}: " + ", ".join(parts))

                note = _tradeoff_note(mat)
                if note:
                    rec.expected_impact.append(note)
            else:
                # If everything is tiny, avoid printing zeros; add a narrative line once.
                rec.expected_impact.append(
                    f"Projected delta at {d}: no material KPI change expected (effects are small vs reporting thresholds)."
                )

        return rec

    recs: List[Recommendation] = []

    # 1) KPI governance (no what-if)
    if anomalies:
        recs.append(
            Recommendation(
                title="KPI governance: verify integrity and measurement scope (green-but-suspicious)",
                rationale="Some KPIs look unusually good early; this often indicates KPI definition drift, clamping, scope changes, or data quality issues. Fixing this protects steering decisions.",
                actions=[
                    "Run KPI lineage review (sources, joins, eligibility filters, denominator logic)",
                    "Add automated DQ tests (denominator consistency, missingness, outliers)",
                    "Cross-check with an independent proxy KPI (sanity metric)",
                    "Document scope changes and re-baseline if required",
                    "Add KPI confidence scoring to reporting (DQ + stability signals)",
                ],
                expected_impact=[
                    "Increase trustworthiness of reporting and decision-making",
                    "Prevent false confidence and mis-steering due to KPI definition/data issues",
                    "Create an auditable evidence trail (useful for Risk/Compliance)",
                ],
                risks=[
                    "May reveal measurement issues requiring remediation work",
                    "Short-term additional workload for data/ops teams",
                ],
                confidence=0.80,
                requires_approval=approvers,
            )
        )

        # 2) Steering option (quantified)
        rec = Recommendation(
            title="Steering option: protect 2027 checkpoints despite KPI uncertainty",
            rationale="While KPI definitions are being validated, proceed with no-regret steering actions that improve delivery flow and reduce operational risk.",
            actions=[
                "Accelerate INIT_DQ1 to stabilize reference data and controls",
                "Accelerate INIT_RES1 to reduce incident volatility",
                "Delay irreversible automation scale-out decisions until KPI lineage is confirmed",
            ],
            expected_impact=[
                "De-risk delivery regardless of KPI definition changes",
                "Improve operational stability and reduce tail-risk for 2027",
            ],
            risks=[
                "May consume capacity before KPI baselines are re-confirmed",
            ],
            confidence=0.70,
            requires_approval=approvers,
        )
        recs.append(attach_whatif(rec, actions=ACTION_ACCEL_DQ + ACTION_ACCEL_RES))

    # Drift-triggered steering (kept for completeness)
    if getattr(snap, "triggered_for_steerco", None):
        for oid in snap.triggered_for_steerco:
            rec = Recommendation(
                title=f"Steer objective {oid}: recover trajectory",
                rationale="Objective drift confidence exceeds steering threshold. Recommend corrective actions and quantify recovery at the next checkpoints.",
                actions=[
                    "Accelerate critical-path initiatives (INIT_DQ1, INIT_A1) to protect STP/time targets",
                    "Split scope of automation wave to deliver high-value flows earlier",
                    "Accelerate observability initiative (INIT_RES1) to reduce incident tail-risk",
                ],
                expected_impact=[
                    "Increase probability of meeting the next checkpoints",
                    "Reduce cascading delivery risk across dependent workstreams",
                ],
                risks=[
                    "Reallocation may slow other workstreams",
                    "Scope changes require stakeholder alignment and control validation",
                ],
                confidence=max(drift_thr, 0.80),
                requires_approval=approvers,
            )
            recs.append(attach_whatif(rec, actions=ACTION_PORTFOLIO))

    # Keep top-3
    recs = _rank(recs)[:3]

    needs_decision = bool(getattr(snap, "triggered_for_steerco", [])) or any(r.confidence >= drift_thr for r in recs)

    headline = "On track" if not getattr(snap, "triggered_for_steerco", []) else "Strategic drift detected"
    if anomalies and not getattr(snap, "triggered_for_steerco", []):
        headline = "On track, but KPI anomalies detected"

    return AgentBrief(
        as_of=snap.as_of,
        headline=headline,
        anomalies=anomalies,
        recommendations=recs,
        needs_human_decision=needs_decision,
    )


def brief_to_text(plan: Plan, brief: AgentBrief) -> str:
    lines: List[str] = []
    lines.append(f"AGENT BRIEF — as of {brief.as_of}")
    lines.append(f"Headline: {brief.headline}")
    lines.append("-" * 72)

    if brief.anomalies:
        lines.append("Anomalies / Watch-outs:")
        for a in brief.anomalies:
            lines.append(f"  - {a}")
        lines.append("")

    lines.append("Recommendations (ranked):")
    for i, r in enumerate(brief.recommendations, 1):
        lines.append(f"{i}) {r.title} (confidence={r.confidence:.2f})")
        lines.append(f"   Rationale: {r.rationale}")
        lines.append("   Actions:")
        for a in r.actions:
            lines.append(f"     - {a}")
        lines.append("   Expected impact:")
        for e in r.expected_impact:
            lines.append(f"     - {e}")
        lines.append("   Risks:")
        for rr in r.risks:
            lines.append(f"     - {rr}")
        if r.requires_approval:
            lines.append(f"   Requires approval: {', '.join(r.requires_approval)}")
        lines.append("")

    lines.append("Human decision required: " + ("YES" if brief.needs_human_decision else "NO"))
    return "\n".join(lines)
