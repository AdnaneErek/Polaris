# src/decision.py
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, List, Optional
from datetime import datetime
import json
import hashlib


def sha256_json(obj) -> str:
    raw = json.dumps(obj, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


@dataclass(frozen=True)
class DecisionRecord:
    decision_id: str
    created_at: str

    as_of: str
    main_date: str

    chosen_bundle_id: str
    chosen_bundle_name: str
    confidence: float  # 0..1

    approvals_required: List[str]
    actions: List[Dict]  # serialized SteeringAction(s)

    # Evidence (short)
    stress_score_mean: float
    stress_score_cvar10: float
    base_score_mean: float
    base_score_cvar10: float

    # KPI delta summaries at main_date (stress)
    kpi_delta_summary: Dict[str, Dict[str, float]]

    # Guardrails / notes
    guardrails_passed: bool
    guardrails_notes: List[str]
    rationale: str

    # Audit hashes (optional but recommended)
    inputs_hash: str


def make_decision_id(org: str, as_of: str) -> str:
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    return f"{org}__{as_of}__{ts}"


def decision_to_json(dec: DecisionRecord) -> str:
    return json.dumps(asdict(dec), indent=2, ensure_ascii=False)
