# src/guardrails.py
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from dateutil.relativedelta import relativedelta

from .schemas import Plan, KPI
from .actions import SteeringAction
from .portfolio import OptionResultMC, ActionBundle


@dataclass
class GuardrailViolation:
    """Represents a single guardrail violation."""
    rule_id: str
    rule_type: str  # "control", "kpi_degradation", "budget", "capacity", "action_limit"
    severity: str  # "error" (hard fail) or "warning" (requires approval)
    message: str
    actual_value: Optional[float] = None
    limit_value: Optional[float] = None
    requires_approval: Optional[List[str]] = None  # List of roles/teams that must approve
    budget_by_year: Optional[Dict[str, float]] = None  # For budget violations: year -> spend
    approval_path: Optional[Dict[str, Any]] = None  # Override/approval path details


@dataclass
class GuardrailResult:
    """Result of guardrail evaluation."""
    passed: bool  # True if no hard violations (errors)
    violations: List[GuardrailViolation]
    requires_approval: bool  # True if any violation requires approval
    approval_required_from: List[str]  # Aggregated list of approvers


def _extract_year(date_iso: str) -> str:
    """Extract year from ISO date string."""
    return date_iso[:4]


def _months_between(start: str, end: str) -> int:
    """Calculate number of months between two dates (YYYY-MM-DD)."""
    start_dt = datetime.fromisoformat(start)
    end_dt = datetime.fromisoformat(end)
    delta = relativedelta(end_dt, start_dt)
    return delta.years * 12 + delta.months


def _calculate_baseline_portfolio_spend_per_year(
    plan: Plan,
    as_of: str,
) -> Dict[str, float]:
    """
    Calculate baseline portfolio spend per year (without any actions).
    Assumes uniform spend distribution over initiative timeline.
    Returns: {year: total_spend_in_EUR_k}
    """
    year_spend: Dict[str, float] = {}
    
    for initiative in plan.initiatives:
        budget_k = float(initiative.budget.amount_k)
        start = initiative.timeline.start
        end = initiative.timeline.end
        
        # Calculate total months
        total_months = _months_between(start, end)
        if total_months <= 0:
            continue
        
        # Monthly spend (uniform distribution)
        monthly_spend = budget_k / total_months
        
        # Distribute across months
        start_dt = datetime.fromisoformat(start)
        end_dt = datetime.fromisoformat(end)
        
        current_dt = start_dt.replace(day=1)  # Start of month
        end_dt_month = end_dt.replace(day=1)
        
        while current_dt <= end_dt_month:
            year = str(current_dt.year)
            year_spend[year] = year_spend.get(year, 0.0) + monthly_spend
            
            # Move to next month
            current_dt += relativedelta(months=1)
    
    return year_spend


def _calculate_action_budget_impact(
    plan: Plan,
    bundle: ActionBundle,
    as_of: str,
    debug: bool = False,
) -> float:
    """
    Calculate budget impact of action bundle for the year containing as_of.
    
    Returns total spend in EUR_k for that year for initiatives in the bundle.
    
    Logic (Model A - yearly allocation):
    1. Identify initiatives in the bundle (from bundle actions)
    2. For each initiative in bundle:
       - Apply actions: acceleration shifts timeline earlier, split_scope reduces budget
       - Distribute budget uniformly across timeline months (start to end)
    3. Sum all months belonging to target year
    4. Compare vs annual limit (done in _check_budget_constraints)
    
    Note: 
    - Acceleration shifts spend earlier but doesn't add cost (unless crash costs modeled)
    - Only initiatives in the bundle are counted (not entire portfolio)
    - Budget is distributed uniformly over timeline months
    """
    target_year = _extract_year(as_of)
    
    # Calculate bundle spend for target year
    bundle_year_spend = 0.0
    
    # Track which initiatives are affected by bundle actions
    affected_initiatives = set()
    for action in bundle.actions:
        affected_initiatives.add(action.target_initiative)
    
    if debug:
        print(f"\n=== Budget Calculation Debug (target year: {target_year}) ===")
    
    # Calculate spend for initiatives in the bundle only
    # The budget constraint applies to the bundle's initiatives, not the entire portfolio
    bundle_initiative_ids = set()
    for action in bundle.actions:
        bundle_initiative_ids.add(action.target_initiative)
    
    # Process only initiatives that are in the bundle
    for initiative in plan.initiatives:
        if initiative.id not in bundle_initiative_ids:
            # Skip initiatives not in the bundle
            continue
        budget_k = float(initiative.budget.amount_k)
        start = initiative.timeline.start
        end = initiative.timeline.end
        
        # Check if this initiative is affected by bundle actions
        is_affected = initiative.id in affected_initiatives
        
        # Apply actions if affected, otherwise use baseline
        adjusted_budget = budget_k
        adjusted_start = start
        adjusted_end = end
        
        if is_affected:
            # Find actions affecting this initiative
            actions_for_init = [a for a in bundle.actions if a.target_initiative == initiative.id]
            
            # Apply actions
            for action in actions_for_init:
                if action.type == "accelerate_initiative":
                    months = int(round(action.get("months", 0.0)))
                    # Shift timeline earlier (don't add cost, just move)
                    start_dt = datetime.fromisoformat(adjusted_start)
                    end_dt = datetime.fromisoformat(adjusted_end)
                    
                    adjusted_start_dt = start_dt - relativedelta(months=months)
                    adjusted_end_dt = end_dt - relativedelta(months=months)
                    
                    adjusted_start = adjusted_start_dt.date().isoformat()
                    adjusted_end = adjusted_end_dt.date().isoformat()
                
                elif action.type == "split_scope":
                    reduction = action.get("reduction", 0.0)
                    # Reduce total budget
                    adjusted_budget = adjusted_budget * (1.0 - reduction)
                
                elif action.type == "add_capacity":
                    # Capacity addition has its own cost model
                    capacity_gain = action.get("capacity_gain", 0.0)
                    # For now, model as 50 EUR_k per FTE per year
                    # This is additional cost, not shifted
                    adjusted_budget += capacity_gain * 50.0
        
        # Calculate spend for this initiative in target year
        total_months = _months_between(adjusted_start, adjusted_end)
        if total_months <= 0:
            if debug:
                print(f"  {initiative.id}: SKIP (invalid timeline: {adjusted_start} to {adjusted_end})")
            continue
        
        monthly_spend = adjusted_budget / total_months
        
        # Count months in target year
        start_dt = datetime.fromisoformat(adjusted_start)
        end_dt = datetime.fromisoformat(adjusted_end)
        
        current_dt = start_dt.replace(day=1)
        end_dt_month = end_dt.replace(day=1)
        
        months_in_target_year = 0
        initiative_year_spend = 0.0
        
        while current_dt <= end_dt_month:
            year = str(current_dt.year)
            if year == target_year:
                months_in_target_year += 1
                initiative_year_spend += monthly_spend
                bundle_year_spend += monthly_spend
            
            current_dt += relativedelta(months=1)
        
        if debug:
            action_str = " (AFFECTED)" if is_affected else ""
            print(f"  {initiative.id}: budget={adjusted_budget:.2f}k, timeline={adjusted_start} to {adjusted_end}, "
                  f"total_months={total_months}, monthly={monthly_spend:.2f}k, "
                  f"{target_year}_months={months_in_target_year}, {target_year}_spend={initiative_year_spend:.2f}k{action_str}")
    
    if debug:
        print(f"\nTotal {target_year} spend: {bundle_year_spend:.2f} EUR_k")
    
    return bundle_year_spend


def _calculate_action_capacity_impact(
    plan: Plan,
    bundle: ActionBundle,
    as_of: str,
) -> float:
    """
    Calculate capacity impact of action bundle.
    Returns total capacity impact in FTE-months.
    """
    total_capacity = 0.0
    
    for action in bundle.actions:
        if action.type == "accelerate_initiative":
            months = int(round(action.get("months", 0.0)))
            # Accelerating requires extra capacity
            total_capacity += months * 0.5  # 0.5 FTE per month
        
        elif action.type == "add_capacity":
            capacity_gain = action.get("capacity_gain", 0.0)
            # Adding capacity increases available capacity (negative impact on constraint)
            total_capacity -= capacity_gain * 12  # Annualized
    
    return total_capacity


def _check_control_coverage(
    plan: Plan,
    bundle: ActionBundle,
    option_result: Optional[OptionResultMC],
) -> List[GuardrailViolation]:
    """
    Check control coverage constraint.
    For now, this is a placeholder that flags for human review.
    """
    violations = []
    
    # Find control coverage constraint
    control_constraint = None
    for constraint in plan.portfolio.constraints:
        if constraint.type == "control":
            control_constraint = constraint
            break
    
    if not control_constraint:
        return violations
    
    min_coverage = float(control_constraint.min_value)
    
    # Check if bundle affects automation (which requires control coverage)
    affects_automation = any(
        action.target_initiative == "INIT_A1" or 
        (action.type == "split_scope" and action.target_initiative == "INIT_A1")
        for action in bundle.actions
    )
    
    if affects_automation:
        # Structured approval ticket
        approval_path = {
            "owner": ["Ops Control Owner", "Risk/Compliance"],
            "required_evidence": [
                "Control matrix (coverage per automated flow)",
                "Reconciliation procedures documentation",
                "Audit trail checks and monitoring setup",
                f"Evidence of minimum {min_coverage*100:.0f}% control coverage",
            ],
            "review_type": "Control Coverage Review",
            "status": "REQUIRED",
        }
        
        violations.append(GuardrailViolation(
            rule_id=control_constraint.id,
            rule_type="control",
            severity="warning",
            message=f"Control Coverage Review Required: YES. "
                   f"Option affects automation (INIT_A1); requires Ops Control owner sign-off "
                   f"(minimum {min_coverage*100:.0f}% coverage).",
            limit_value=min_coverage,
            requires_approval=["Ops Control Owner", "Risk/Compliance"],
            approval_path=approval_path,
        ))
    else:
        # No automation affected - no review needed
        violations.append(GuardrailViolation(
            rule_id=f"{control_constraint.id}_check",
            rule_type="control",
            severity="info",
            message="Control Coverage Review Required: NO. Option does not affect automation.",
            requires_approval=[],
        ))
    
    return violations


def _check_kpi_degradation(
    plan: Plan,
    bundle: ActionBundle,
    option_result: OptionResultMC,
    kpi_degradation_threshold: float = 2.0,  # Default: 2 pp or 2 units
) -> List[GuardrailViolation]:
    """
    Check that no KPI degrades beyond threshold.
    Uses stress delta summary (worst case).
    """
    violations = []
    
    if not option_result:
        return violations
    
    for kpi_id, delta_summary in option_result.stress_delta_summary.items():
        # Find KPI in plan
        kpi = None
        for k in plan.kpis:
            if k.id == kpi_id or k.short_name == kpi_id:
                kpi = k
                break
        
        if not kpi:
            continue
        
        direction = kpi.target.direction
        mean_delta = delta_summary.get("mean", 0.0)
        p10_delta = delta_summary.get("p10", 0.0)  # Worst case (10th percentile)
        
        # Determine degradation threshold based on KPI unit
        if kpi.unit == "%":
            threshold = kpi_degradation_threshold  # percentage points
        elif kpi.unit in ("hours", "h"):
            threshold = kpi_degradation_threshold * 0.1  # 0.2 hours
        else:
            threshold = kpi_degradation_threshold  # counts or other
        
        # Check degradation (direction-aware)
        degraded = False
        degradation_amount = 0.0
        
        if direction == "up":
            # For "up" KPIs, negative delta is degradation
            if p10_delta < -threshold:
                degraded = True
                degradation_amount = abs(p10_delta)
        else:  # "down"
            # For "down" KPIs, positive delta is degradation
            if p10_delta > threshold:
                degraded = True
                degradation_amount = abs(p10_delta)
        
        if degraded:
            violations.append(GuardrailViolation(
                rule_id=f"KPI_DEG_{kpi_id}",
                rule_type="kpi_degradation",
                severity="error",
                message=f"KPI {kpi.short_name} degrades by {degradation_amount:.2f}{kpi.unit} "
                       f"(p10 worst case: {p10_delta:+.2f}{kpi.unit}, threshold: {threshold:.2f}{kpi.unit}).",
                actual_value=degradation_amount,
                limit_value=threshold,
                requires_approval=["Transformation Lead", "IT Ops Head"],
            ))
    
    return violations


def _calculate_budget_by_year(
    plan: Plan,
    bundle: ActionBundle,
    years: List[str],
) -> Dict[str, float]:
    """
    Calculate budget impact by year for all specified years.
    Returns: {year: spend_in_EUR_k}
    """
    year_spend: Dict[str, float] = {}
    
    # Track which initiatives are affected by bundle actions
    bundle_initiative_ids = set()
    for action in bundle.actions:
        bundle_initiative_ids.add(action.target_initiative)
    
    # Process only initiatives that are in the bundle
    for initiative in plan.initiatives:
        if initiative.id not in bundle_initiative_ids:
            continue
        
        budget_k = float(initiative.budget.amount_k)
        start = initiative.timeline.start
        end = initiative.timeline.end
        
        # Find actions affecting this initiative
        actions_for_init = [a for a in bundle.actions if a.target_initiative == initiative.id]
        
        # Apply actions
        adjusted_budget = budget_k
        adjusted_start = start
        adjusted_end = end
        
        for action in actions_for_init:
            if action.type == "accelerate_initiative":
                months = int(round(action.get("months", 0.0)))
                start_dt = datetime.fromisoformat(adjusted_start)
                end_dt = datetime.fromisoformat(adjusted_end)
                
                adjusted_start_dt = start_dt - relativedelta(months=months)
                adjusted_end_dt = end_dt - relativedelta(months=months)
                
                adjusted_start = adjusted_start_dt.date().isoformat()
                adjusted_end = adjusted_end_dt.date().isoformat()
            
            elif action.type == "split_scope":
                reduction = action.get("reduction", 0.0)
                adjusted_budget = adjusted_budget * (1.0 - reduction)
            
            elif action.type == "add_capacity":
                capacity_gain = action.get("capacity_gain", 0.0)
                adjusted_budget += capacity_gain * 50.0
        
        # Calculate spend for this initiative
        total_months = _months_between(adjusted_start, adjusted_end)
        if total_months <= 0:
            continue
        
        monthly_spend = adjusted_budget / total_months
        
        start_dt = datetime.fromisoformat(adjusted_start)
        end_dt = datetime.fromisoformat(adjusted_end)
        
        current_dt = start_dt.replace(day=1)
        end_dt_month = end_dt.replace(day=1)
        
        while current_dt <= end_dt_month:
            year = str(current_dt.year)
            if year in years:
                year_spend[year] = year_spend.get(year, 0.0) + monthly_spend
            
            current_dt += relativedelta(months=1)
    
    return year_spend


def _check_budget_constraints(
    plan: Plan,
    bundle: ActionBundle,
    as_of: str,
    debug: bool = False,
) -> List[GuardrailViolation]:
    """Check budget constraints from plan.yaml with explainable finance details."""
    violations = []
    
    # Find budget constraint
    budget_constraint = None
    for constraint in plan.portfolio.constraints:
        if constraint.type == "budget":
            budget_constraint = constraint
            break
    
    if not budget_constraint or not budget_constraint.year_limits:
        return violations
    
    # Calculate budget by year for all years in constraint
    years = list(budget_constraint.year_limits.keys())
    budget_by_year = _calculate_budget_by_year(plan, bundle, years)
    
    # Check each year
    for year, budget_limit in budget_constraint.year_limits.items():
        budget_impact = budget_by_year.get(year, 0.0)
        
        if budget_impact > budget_limit:
            # Build detailed message with assumptions (deduplicated)
            assumptions = []
            for action in bundle.actions:
                if action.type == "accelerate_initiative":
                    assumptions.append("accelerate_initiative: shifts spend earlier (no cost inflation)")
                elif action.type == "add_capacity":
                    capacity_gain = action.get("capacity_gain", 0.0)
                    assumptions.append(f"add_capacity: +{capacity_gain * 50.0:.0f} EUR_k per FTE")
                elif action.type == "split_scope":
                    reduction = action.get("reduction", 0.0)
                    assumptions.append(f"split_scope: reduces cost by {reduction*100:.0f}%")
            # Deduplicate assumptions
            assumptions_unique = sorted(list(set(assumptions)))
            assumptions_str = "; ".join(assumptions_unique) if assumptions_unique else "Baseline uniform distribution"
            
            violations.append(GuardrailViolation(
                rule_id=f"{budget_constraint.id}_{year}",
                rule_type="budget",
                severity="error",
                message=f"Budget impact {budget_impact:.0f} {budget_constraint.currency} "
                       f"exceeds annual limit {budget_limit:.0f} {budget_constraint.currency} for {year}. "
                       f"Assumptions: {assumptions_str}.",
                actual_value=budget_impact,
                limit_value=budget_limit,
                requires_approval=["Transformation Lead", "Finance"],
            ))
    
    return violations


def _check_capacity_constraints(
    plan: Plan,
    bundle: ActionBundle,
    as_of: str,
) -> List[GuardrailViolation]:
    """Check capacity constraints from plan.yaml."""
    violations = []
    
    # Find capacity constraint
    capacity_constraint = None
    for constraint in plan.portfolio.constraints:
        if constraint.type == "capacity":
            capacity_constraint = constraint
            break
    
    if not capacity_constraint or not capacity_constraint.year_limits:
        return violations
    
    year = _extract_year(as_of)
    capacity_limit = capacity_constraint.year_limits.get(year)
    
    if capacity_limit is None:
        return violations
    
    capacity_impact = _calculate_action_capacity_impact(plan, bundle, as_of)
    
    # Capacity impact is positive when consuming capacity
    if capacity_impact > capacity_limit:
        violations.append(GuardrailViolation(
            rule_id=capacity_constraint.id,
            rule_type="capacity",
            severity="error",
            message=f"Capacity impact {capacity_impact:.1f} FTE exceeds annual limit "
                   f"{capacity_limit:.0f} FTE for {year}.",
            actual_value=capacity_impact,
            limit_value=capacity_limit,
            requires_approval=["Transformation Lead", "IT Ops Head"],
        ))
    
    return violations


def _check_action_limits(
    plan: Plan,
    bundle: ActionBundle,
) -> List[GuardrailViolation]:
    """Check action count limits per objective from agent_policy."""
    violations = []
    
    # Parse recommendation limits from agent_policy
    # Example: "No more than 3 actions per objective per cycle"
    action_limit_per_objective = None
    for limit_rule in plan.agent_policy.recommendation_limits:
        if "actions per objective" in limit_rule.lower():
            # Try to extract number
            import re
            match = re.search(r'(\d+)', limit_rule)
            if match:
                action_limit_per_objective = int(match.group(1))
                break
    
    if action_limit_per_objective is None:
        # Default: 3 actions per objective
        action_limit_per_objective = 3
    
    # Count actions per objective
    actions_by_objective: Dict[str, int] = {}
    for action in bundle.actions:
        # Find initiative and its objectives
        for initiative in plan.initiatives:
            if initiative.id == action.target_initiative:
                for obj_id in initiative.objective_ids:
                    actions_by_objective[obj_id] = actions_by_objective.get(obj_id, 0) + 1
                break
    
    # Check limits
    for obj_id, count in actions_by_objective.items():
        if count > action_limit_per_objective:
            obj_name = next((o.name for o in plan.objectives if o.id == obj_id), obj_id)
            violations.append(GuardrailViolation(
                rule_id="ACTION_LIMIT",
                rule_type="action_limit",
                severity="warning",
                message=f"Objective {obj_id} ({obj_name}) has {count} actions, "
                       f"exceeds limit of {action_limit_per_objective} actions per objective.",
                actual_value=float(count),
                limit_value=float(action_limit_per_objective),
                requires_approval=["Transformation Lead"],
            ))
    
    return violations


def check_portfolio_guardrails(
    plan: Plan,
    bundle: ActionBundle,
    option_result: Optional[OptionResultMC],
    as_of: str,
    kpi_degradation_threshold: Optional[float] = None,
) -> GuardrailResult:
    """
    Comprehensive guardrail checker for portfolio options.
    
    Checks:
    1. Control coverage (placeholder - requires human review)
    2. KPI degradation thresholds
    3. Budget constraints
    4. Capacity constraints
    5. Action count limits per objective
    
    Returns:
        GuardrailResult with pass/fail status, violations, and approval requirements.
    """
    violations: List[GuardrailViolation] = []
    
    # 1. Control coverage
    violations.extend(_check_control_coverage(plan, bundle, option_result))
    
    # 2. KPI degradation
    if option_result:
        threshold = kpi_degradation_threshold or 2.0
        violations.extend(_check_kpi_degradation(plan, bundle, option_result, threshold))
    
    # 3. Budget constraints
    violations.extend(_check_budget_constraints(plan, bundle, as_of))
    
    # 4. Capacity constraints
    violations.extend(_check_capacity_constraints(plan, bundle, as_of))
    
    # 5. Action limits
    violations.extend(_check_action_limits(plan, bundle))
    
    # Determine pass/fail
    # Pass if no "error" severity violations
    errors = [v for v in violations if v.severity == "error"]
    passed = len(errors) == 0
    
    # Aggregate approval requirements
    approval_required_from = set()
    for v in violations:
        if v.requires_approval:
            approval_required_from.update(v.requires_approval)
    
    return GuardrailResult(
        passed=passed,
        violations=violations,
        requires_approval=len(approval_required_from) > 0,
        approval_required_from=sorted(list(approval_required_from)),
    )


# Backward compatibility
def check_guardrails(plan: Plan, actions: List[SteeringAction]) -> Tuple[bool, List[str]]:
    """
    Legacy function for backward compatibility.
    Use check_portfolio_guardrails() for new code.
    """
    # Create a dummy bundle for compatibility
    bundle = ActionBundle(
        id="LEGACY",
        name="Legacy check",
        description="Backward compatibility",
        actions=actions,
    )
    
    result = check_portfolio_guardrails(plan, bundle, None, "2026-01-01")
    
    notes = [v.message for v in result.violations]
    if not notes:
        notes.append("No guardrail violations detected (legacy check).")
    
    return result.passed, notes
