# src/schemas.py
from __future__ import annotations

from typing import Dict, List, Literal, Optional, Any
from pydantic import BaseModel, Field, ConfigDict, model_validator

# -----------------------------
# Enums / Literals
# -----------------------------
Direction = Literal["up", "down"]
Frequency = Literal["daily", "weekly", "monthly", "quarterly"]
Severity = Literal["low", "medium", "high"]
ConstraintType = Literal[
    "budget",
    "capacity",
    "control",
    "control_coverage",
    "compliance",
    "availability",
    "regulatory",
]

DependencyType = Literal["hard", "soft"]
TrajectoryType = Literal["linear", "piecewise", "s-curve"]
AutonomyLevel = Literal["recommend_only", "approve_execute"]


# -----------------------------
# Meta / Governance
# -----------------------------
class Perimeter(BaseModel):
    business_line: str
    domain: str
    scope: str


class Horizon(BaseModel):
    start: str  # ISO date
    end: str    # ISO date
    cadence: Frequency


class DecisionRights(BaseModel):
    propose: List[str] = Field(default_factory=list)
    approve: List[str] = Field(default_factory=list)
    execute: List[str] = Field(default_factory=list)


class Auditability(BaseModel):
    log_all_recommendations: bool = True
    store_inputs_hashes: bool = True
    rationale_required: bool = True


class Governance(BaseModel):
    decision_rights: DecisionRights
    auditability: Auditability
    guardrails: List[str] = Field(default_factory=list)


class PlanMeta(BaseModel):
    org: str
    perimeter: Perimeter
    horizon: Horizon
    version: str
    authors: List[str] = Field(default_factory=list)
    last_updated: str
    assumptions: List[str] = Field(default_factory=list)
    governance: Governance


# -----------------------------
# KPI catalog
# -----------------------------
class KPIBaseline(BaseModel):
    value: float
    date: str


class KPITarget(BaseModel):
    value: float
    deadline: str
    direction: Direction


class KPIDefinition(BaseModel):
    description: str
    formula: str
    exclusions: List[str] = Field(default_factory=list)
    scope: Optional[str] = None


class DataSource(BaseModel):
    system: str
    table_or_stream: str


class QualityRule(BaseModel):
    rule: str
    severity: Severity


class DataLineage(BaseModel):
    sources: List[DataSource] = Field(default_factory=list)
    quality_rules: List[QualityRule] = Field(default_factory=list)


class KPIOwner(BaseModel):
    business: str
    data: str


class KPI(BaseModel):
    id: str
    name: str
    short_name: str
    unit: str
    frequency: Frequency
    baseline: KPIBaseline
    target: KPITarget
    definition: KPIDefinition
    data_lineage: DataLineage
    owner: KPIOwner
    risk_classification: Severity


# -----------------------------
# Objectives / OKRs
# -----------------------------
class TrajectoryCheckpoint(BaseModel):
    date: str
    expected: float


class Trajectory(BaseModel):
    type: TrajectoryType
    checkpoints: List[TrajectoryCheckpoint] = Field(default_factory=list)


class OKR(BaseModel):
    id: str
    kpi_id: str
    baseline: float
    target: float
    deadline: str
    direction: Direction
    trajectory: Trajectory


class ObjectiveConstraint(BaseModel):
    type: ConstraintType
    description: str


class Objective(BaseModel):
    id: str
    name: str
    description: str
    weight: float = Field(ge=0.0, le=1.0)
    okrs: List[OKR] = Field(default_factory=list)
    constraints: List[ObjectiveConstraint] = Field(default_factory=list)


# -----------------------------
# Capability map
# -----------------------------
class Capability(BaseModel):
    id: str
    name: str
    description: str


# -----------------------------
# Initiatives
# -----------------------------
class RACI(BaseModel):
    responsible: List[str] = Field(default_factory=list)
    accountable: List[str] = Field(default_factory=list)
    consulted: List[str] = Field(default_factory=list)
    informed: List[str] = Field(default_factory=list)


class InitiativeTimeline(BaseModel):
    start: str
    end: str
    phase: str


class CapexOpexSplit(BaseModel):
    capex: float = Field(ge=0.0, le=1.0)
    opex: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _sum_to_one(self) -> "CapexOpexSplit":
        s = (self.capex or 0.0) + (self.opex or 0.0)
        if abs(s - 1.0) > 1e-6:
            raise ValueError(f"capex_opex_split must sum to 1.0, got {s}")
        return self


class InitiativeBudget(BaseModel):
    currency: str
    amount_k: float = Field(ge=0.0)
    capex_opex_split: CapexOpexSplit


class Dependency(BaseModel):
    type: DependencyType
    initiative_id: str
    rationale: str


class Milestone(BaseModel):
    id: str
    name: str
    due: str
    deliverables: List[str] = Field(default_factory=list)


class WorkstreamDeliverable(BaseModel):
    id: str
    name: str
    due: str
    owner: str
    definition_of_done: List[str] = Field(default_factory=list)
    operational_kpis: List[str] = Field(default_factory=list)


class Workstream(BaseModel):
    id: str
    name: str
    description: str
    owner: str
    start: str
    end: str
    effort_fte_months: float = Field(ge=0.0)
    deliverables: List[WorkstreamDeliverable] = Field(default_factory=list)
    control_impacts: List[str] = Field(default_factory=list)
    depends_on_workstreams: List[str] = Field(default_factory=list)


class KPIImpact(BaseModel):
    kpi_id: str
    expected_delta_by_end: float
    confidence: float = Field(ge=0.0, le=1.0)
    lag_months: int = Field(ge=0)


class BenefitEstimate(BaseModel):
    unit: str
    value: float
    confidence: float = Field(ge=0.0, le=1.0)


class Benefit(BaseModel):
    type: str
    description: str
    estimate: BenefitEstimate


class Risk(BaseModel):
    id: str
    category: str
    description: str
    likelihood: float = Field(ge=0.0, le=1.0)
    impact: float = Field(ge=0.0, le=1.0)
    mitigations: List[str] = Field(default_factory=list)


class Initiative(BaseModel):
    id: str
    name: str
    description: str

    objective_ids: List[str] = Field(default_factory=list)
    capability_ids: List[str] = Field(default_factory=list)

    owner: str
    sponsor: str
    raci: RACI

    timeline: InitiativeTimeline
    budget: InitiativeBudget

    dependencies: List[Dependency] = Field(default_factory=list)
    workstreams: List[Workstream] = Field(default_factory=list)
    milestones: List[Milestone] = Field(default_factory=list)

    kpi_impacts: List[KPIImpact] = Field(default_factory=list)
    benefits: List[Benefit] = Field(default_factory=list)

    risks: List[Risk] = Field(default_factory=list)
    compliance_controls: List[str] = Field(default_factory=list)


# -----------------------------
# Portfolio constraints / policies
# -----------------------------
class YearLimits(BaseModel):
    # Keep as dict for YAML friendliness, validate keys loosely (e.g., "2026")
    model_config = ConfigDict(extra="allow")


class PortfolioConstraint(BaseModel):
    id: str
    type: ConstraintType
    description: str

    # Budget/capacity style
    year_limits: Optional[Dict[str, float]] = None
    currency: Optional[str] = None

    # Control threshold style
    min_value: Optional[float] = Field(default=None, ge=0.0, le=1.0)


class Portfolio(BaseModel):
    constraints: List[PortfolioConstraint] = Field(default_factory=list)
    prioritization_rules: List[str] = Field(default_factory=list)


# -----------------------------
# Reporting
# -----------------------------
class ExecutiveSummaryTemplate(BaseModel):
    required_sections: List[str] = Field(default_factory=list)


class DecisionThresholds(BaseModel):
    drift_confidence: float = Field(ge=0.0, le=1.0)
    budget_overrun_pct: float = Field(ge=0.0)
    milestone_delay_days: int = Field(ge=0)


class SteeringCommittee(BaseModel):
    cadence: Frequency
    decision_thresholds: DecisionThresholds


class Reporting(BaseModel):
    executive_summary: ExecutiveSummaryTemplate
    steering_committee: SteeringCommittee


# -----------------------------
# Agent policy
# -----------------------------
class EscalationPolicy(BaseModel):
    if_confidence_below: float = Field(ge=0.0, le=1.0)
    if_compliance_impact: Severity


class RecommendationLimits(BaseModel):
    # Keep these as free-form strings (human policy), but structured enough to display.
    rules: List[str] = Field(default_factory=list)


class ExplanationRequirements(BaseModel):
    include: List[str] = Field(default_factory=list)


class AgentPolicy(BaseModel):
    autonomy_level: AutonomyLevel
    escalation: EscalationPolicy
    recommendation_limits: List[str] = Field(default_factory=list)
    explanation_requirements: ExplanationRequirements


# -----------------------------
# Full Plan
# -----------------------------
class Plan(BaseModel):
    meta: PlanMeta

    kpis: List[KPI] = Field(default_factory=list)
    objectives: List[Objective] = Field(default_factory=list)
    capabilities: List[Capability] = Field(default_factory=list)
    initiatives: List[Initiative] = Field(default_factory=list)

    portfolio: Portfolio
    reporting: Reporting
    agent_policy: AgentPolicy

    model_config = ConfigDict(extra="forbid")

    # --------- cross-reference validation ---------
    @model_validator(mode="after")
    def _validate_references_and_uniqueness(self) -> "Plan":
        # Unique IDs
        def ensure_unique(items: List[Any], label: str) -> None:
            ids = [x.id for x in items]
            dup = {i for i in ids if ids.count(i) > 1}
            if dup:
                raise ValueError(f"Duplicate {label} ids: {sorted(dup)}")

        ensure_unique(self.kpis, "kpi")
        ensure_unique(self.objectives, "objective")
        ensure_unique(self.capabilities, "capability")
        ensure_unique(self.initiatives, "initiative")

        kpi_ids = {k.id for k in self.kpis}
        obj_ids = {o.id for o in self.objectives}
        cap_ids = {c.id for c in self.capabilities}
        init_ids = {i.id for i in self.initiatives}

        # Objectives: OKRs refer to KPI ids
        for o in self.objectives:
            for okr in o.okrs:
                if okr.kpi_id not in kpi_ids:
                    raise ValueError(f"Objective {o.id} OKR {okr.id} references unknown kpi_id={okr.kpi_id}")

        # Initiatives: objective_ids/capability_ids/dependencies/kpi_impacts refer to known ids
        for it in self.initiatives:
            for oid in it.objective_ids:
                if oid not in obj_ids:
                    raise ValueError(f"Initiative {it.id} references unknown objective_id={oid}")
            for cid in it.capability_ids:
                if cid not in cap_ids:
                    raise ValueError(f"Initiative {it.id} references unknown capability_id={cid}")
            for dep in it.dependencies:
                if dep.initiative_id not in init_ids:
                    raise ValueError(f"Initiative {it.id} dependency references unknown initiative_id={dep.initiative_id}")
            for imp in it.kpi_impacts:
                if imp.kpi_id not in kpi_ids:
                    raise ValueError(f"Initiative {it.id} KPI impact references unknown kpi_id={imp.kpi_id}")

        # Portfolio constraints sanity
        for c in self.portfolio.constraints:
            if c.type in ("budget", "capacity"):
                if not c.year_limits:
                    raise ValueError(f"Portfolio constraint {c.id} type={c.type} requires year_limits")
            if c.type == "control":
                if c.min_value is None:
                    raise ValueError(f"Portfolio constraint {c.id} type=control requires min_value")

        # Optional: objective weights ~ sum to 1 (tolerate small drift)
        wsum = sum(o.weight for o in self.objectives)
        if self.objectives and abs(wsum - 1.0) > 0.15:
            # We keep tolerance relatively loose for early PoC.
            # Tighten later if you want.
            raise ValueError(f"Objective weights should sum ~ 1.0 (±0.15). Got {wsum:.3f}")

        return self
