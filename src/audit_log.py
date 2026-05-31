# src/audit_log.py
from __future__ import annotations

import hashlib
import json
import sys
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd


def _sha256_bytes(b: bytes) -> str:
    h = hashlib.sha256()
    h.update(b)
    return h.hexdigest()


def file_sha256(path: str | Path) -> str:
    p = Path(path)
    return _sha256_bytes(p.read_bytes())


def json_sha256(obj: Any) -> str:
    # stable hash for dict/list primitives
    b = json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return _sha256_bytes(b)


def df_sha256(df: pd.DataFrame) -> str:
    # stable hash for dataframe content
    # NOTE: we hash CSV bytes with stable column order
    df2 = df.copy()
    cols = list(df2.columns)
    df2 = df2[cols]
    b = df2.to_csv(index=False).encode("utf-8")
    return _sha256_bytes(b)


def get_code_version_hash() -> Optional[str]:
    """Get git commit hash if available, otherwise None."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (FileNotFoundError, subprocess.SubprocessError):
        pass
    return None


def get_python_version() -> str:
    """Get Python version string."""
    return f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"


def get_dependency_snapshot() -> Dict[str, str]:
    """Get dependency versions snapshot."""
    try:
        import pkg_resources
        deps = {}
        for dist in pkg_resources.working_set:
            deps[dist.project_name] = dist.version
        return deps
    except Exception:
        return {}


@dataclass(frozen=True)
class AuditPayload:
    as_of: str
    plan_path: str
    kpis_path: Optional[str]
    initiatives_path: Optional[str]

    input_hashes: Dict[str, str]
    config: Dict[str, Any]
    outputs_hashes: Dict[str, str]
    
    # Phase 7.2: Audit log completeness
    code_version_hash: Optional[str] = None  # Git commit hash
    python_version: Optional[str] = None  # Python version
    dependency_snapshot: Optional[Dict[str, str]] = None  # Dependency versions
    random_seed: Optional[int] = None  # Random seed used


def write_audit_bundle(
    out_dir: str | Path,
    payload: AuditPayload,
    artifacts: Optional[Dict[str, Any]] = None,
) -> Path:
    """
    Writes:
      - audit.json (hashes + config)
      - artifacts.json (optional full pack payload)
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    audit_path = out_dir / "audit.json"
    audit_data = {
        "as_of": payload.as_of,
        "plan_path": payload.plan_path,
        "kpis_path": payload.kpis_path,
        "initiatives_path": payload.initiatives_path,
        "input_hashes": payload.input_hashes,
        "config": payload.config,
        "outputs_hashes": payload.outputs_hashes,
    }
    
    # Add Phase 7.2 fields if available
    if payload.code_version_hash:
        audit_data["code_version_hash"] = payload.code_version_hash
    if payload.python_version:
        audit_data["python_version"] = payload.python_version
    if payload.dependency_snapshot:
        audit_data["dependency_snapshot"] = payload.dependency_snapshot
    if payload.random_seed is not None:
        audit_data["random_seed"] = payload.random_seed
    
    audit_path.write_text(
        json.dumps(audit_data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    if artifacts is not None:
        (out_dir / "artifacts.json").write_text(
            json.dumps(artifacts, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    return audit_path
