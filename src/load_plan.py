import yaml
from pathlib import Path
from .schemas import Plan

def load_plan(path: str | Path) -> Plan:
    p = Path(path)
    raw = yaml.safe_load(p.read_text(encoding="utf-8"))
    return Plan.model_validate(raw)
