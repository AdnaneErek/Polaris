#!/usr/bin/env python3
"""Compare rule-based vs ML anomaly detection."""
import sys
sys.path.insert(0, '.')

from src.load_plan import load_plan
from src.monitor import monitor
import pandas as pd

plan = load_plan('data/plan.yaml')
kpis = pd.read_csv('data/simulated_kpis.csv')
snap = monitor(plan, kpis[['date', 'kpi_id', 'value']], '2026-12-01')

from src.agent import _kpi_anomaly_checks

# Get rule-based anomalies (without ML)
rule_anomalies_raw = []
kmap = {k.id: k for k in plan.kpis}
mid = "2027-06-30"  # Approximate mid-point

for os in snap.objective_statuses:
    for ks in os.kpi_statuses:
        if ks.actual is None:
            continue
        k = kmap.get(ks.kpi_id)
        if not k:
            continue

        if k.target.direction == "up":
            if ks.actual >= 99.0 and snap.as_of < mid:
                rule_anomalies_raw.append(
                    f"KPI {k.short_name} is {ks.actual:.2f}{k.unit} very early (before {mid}). "
                    f"Verify KPI definition, denominator, and lineage."
                )
            if ks.actual > float(k.target.value) + 3.0 and snap.as_of < mid:
                rule_anomalies_raw.append(
                    f"KPI {k.short_name} exceeds target unusually early (actual {ks.actual:.2f} vs target {k.target.value:.2f}). "
                    f"Check for measurement bias or clamping."
                )

        if k.target.direction == "down":
            if ks.actual < 0.75 * float(k.target.value) and snap.as_of < mid:
                rule_anomalies_raw.append(
                    f"KPI {k.short_name} is far below target early (actual {ks.actual:.2f} vs target {k.target.value:.2f}). "
                    f"Check for reporting bias/scope changes."
                )

# Get all anomalies (rule-based + ML)
all_anomalies = _kpi_anomaly_checks(plan, snap)

print("=" * 80)
print("ANOMALY DETECTION COMPARISON")
print("=" * 80)
print(f"\nAs of: {snap.as_of}")
print(f"\nLatest KPI values:")
for os in snap.objective_statuses:
    for ks in os.kpi_statuses:
        if ks.actual is not None:
            k = kmap.get(ks.kpi_id)
            if k:
                print(f"  {k.short_name} ({ks.kpi_id}): {ks.actual:.2f}{k.unit} (target: {k.target.value:.2f}{k.unit}, direction: {k.target.direction})")

print("\n" + "=" * 80)
print("RULE-BASED ANOMALIES (threshold-based checks)")
print("=" * 80)
if rule_anomalies_raw:
    for i, a in enumerate(rule_anomalies_raw, 1):
        print(f"{i}. {a}")
else:
    print("  None detected")
    print("\n  Rule-based checks look for:")
    print("    - 'Up' KPIs: actual >= 99.0 OR actual > target + 3.0 (before mid-point)")
    print("    - 'Down' KPIs: actual < 0.75 * target (before mid-point)")

print("\n" + "=" * 80)
print("ML ANOMALIES (pattern-based detection)")
print("=" * 80)
if snap.ml_anomalies:
    ml_anomalies_found = []
    for kpi_id, ml_data in snap.ml_anomalies.items():
        if isinstance(ml_data, dict):
            is_anomaly = ml_data.get('is_anomaly', False)
        else:
            is_anomaly = getattr(ml_data, 'is_anomaly', False)
        
        if is_anomaly:
            k = kmap.get(kpi_id)
            kpi_name = k.short_name if k else kpi_id
            if isinstance(ml_data, dict):
                explanation = ml_data.get('explanation', '')
                confidence = ml_data.get('confidence', 0.0)
                score = ml_data.get('anomaly_score', 0.0)
            else:
                explanation = getattr(ml_data, 'explanation', '')
                confidence = getattr(ml_data, 'confidence', 0.0)
                score = getattr(ml_data, 'anomaly_score', 0.0)
            
            ml_anomalies_found.append({
                'kpi_id': kpi_id,
                'kpi_name': kpi_name,
                'explanation': explanation,
                'confidence': confidence,
                'score': score
            })
    
    if ml_anomalies_found:
        for i, a in enumerate(ml_anomalies_found, 1):
            print(f"{i}. {a['kpi_name']} ({a['kpi_id']}):")
            print(f"   {a['explanation']}")
            print(f"   Confidence: {a['confidence']:.2f}, Score: {a['score']:.2f}")
    else:
        print("  None detected")
else:
    print("  ML detection not available or no anomalies found")
    print("\n  ML checks look for:")
    print("    - Unusual patterns in time series (volatility, trend, recent changes)")
    print("    - Deviations from historical normal behavior")

print("\n" + "=" * 80)
print("COMPARISON SUMMARY")
print("=" * 80)
print(f"Rule-based anomalies: {len(rule_anomalies_raw)}")
if snap.ml_anomalies:
    ml_count = sum(1 for d in snap.ml_anomalies.values() 
                   if (isinstance(d, dict) and d.get('is_anomaly', False)) or 
                      (hasattr(d, 'is_anomaly') and d.is_anomaly))
    print(f"ML anomalies: {ml_count}")
else:
    print(f"ML anomalies: 0 (not available)")

print(f"\nTotal anomalies (combined): {len(all_anomalies)}")

# Check for overlap
rule_kpis = set()
for a in rule_anomalies_raw:
    for kpi_id in ['KPI_STP', 'KPI_E2E', 'KPI_INC']:
        if kpi_id in a:
            rule_kpis.add(kpi_id)

ml_kpis = set()
if snap.ml_anomalies:
    for kpi_id, ml_data in snap.ml_anomalies.items():
        if isinstance(ml_data, dict):
            if ml_data.get('is_anomaly', False):
                ml_kpis.add(kpi_id)
        elif hasattr(ml_data, 'is_anomaly') and ml_data.is_anomaly:
            ml_kpis.add(kpi_id)

print(f"\nRule-based flagged KPIs: {rule_kpis if rule_kpis else 'None'}")
print(f"ML flagged KPIs: {ml_kpis if ml_kpis else 'None'}")
print(f"Overlap: {rule_kpis & ml_kpis if rule_kpis and ml_kpis else 'None'}")
print(f"Rule-only: {rule_kpis - ml_kpis if rule_kpis else 'None'}")
print(f"ML-only: {ml_kpis - rule_kpis if ml_kpis else 'None'}")
