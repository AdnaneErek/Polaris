# Rule-Based vs ML Anomaly Detection Comparison

## Rule-Based Anomaly Detection (Existing System)

**What it detects:**
1. **For "up" KPIs** (like STP_rate):
   - If actual >= 99.0 before mid-point → "very early achievement" (suspicious)
   - If actual > target + 3.0 before mid-point → "exceeds target unusually early"

2. **For "down" KPIs** (like processing_time, incident_count):
   - If actual < 0.75 * target before mid-point → "far below target early"

**Characteristics:**
- ✅ Threshold-based (absolute values)
- ✅ Compares against targets and deadlines
- ✅ Detects "too good to be true" scenarios
- ❌ Doesn't detect gradual drift or pattern changes
- ❌ Doesn't catch volatility spikes
- ❌ Only triggers if values are extreme relative to targets

**Example scenarios it would catch:**
- STP_rate jumps to 99% in month 2 (should take 3 years)
- Processing_time drops to 0.5 hours when target is 1.5 (suspiciously fast)
- Incident_count drops to 5 when target is 20 (too good too early)

## ML Anomaly Detection (New System)

**What it detects:**
1. **Pattern deviations** in time series:
   - Unusual volatility (sudden spikes/drops)
   - Trend breaks (trajectory changes direction unexpectedly)
   - Recent changes that deviate from historical patterns
   - Acceleration/deceleration anomalies

**Characteristics:**
- ✅ Pattern-based (relative to historical behavior)
- ✅ Detects gradual drift
- ✅ Catches volatility spikes
- ✅ Works even if values are within "normal" ranges
- ❌ Doesn't know about targets/deadlines
- ❌ Might flag normal business changes as anomalies

**Example scenarios it would catch:**
- Processing_time shows sudden volatility (even if still above target)
- STP_rate trajectory changes direction unexpectedly
- Incident_count has unusual spike pattern (even if still below target)
- Any KPI showing deviation from its own historical pattern

## Key Differences

| Aspect | Rule-Based | ML-Based |
|--------|-----------|----------|
| **Focus** | Target achievement | Historical patterns |
| **Trigger** | Extreme values vs targets | Pattern deviations |
| **Time awareness** | Yes (mid-point, deadlines) | No (just patterns) |
| **Business context** | Yes (knows targets) | No (statistical only) |
| **Gradual drift** | No | Yes |
| **Volatility** | No | Yes |
| **"Too good" detection** | Yes | No |

## Complementarity

**They complement each other:**

1. **Rule-based catches:**
   - "Too good to be true" scenarios
   - Measurement errors (KPI definition issues)
   - Early achievement anomalies

2. **ML catches:**
   - Pattern breaks that rule-based misses
   - Volatility spikes
   - Gradual drift before hitting thresholds

## Answer to Your Question

**"Are there anomalies detected by rule-based but not by ML?"**

**Yes, potentially:**

1. **Early achievement anomalies:**
   - If STP_rate reaches 99% in month 2 → Rule-based flags it, ML might not (if pattern is consistent)
   - If processing_time drops to 0.5h early → Rule-based flags it, ML might not

2. **"Too good to be true" scenarios:**
   - Rule-based checks if values exceed targets by large margins early
   - ML only looks at patterns, not target relationships

**However, in your current data:**
- Current values are within normal ranges
- No rule-based anomalies triggered
- ML detected a pattern anomaly in KPI_E2E (volatility/pattern deviation)

**So in practice:**
- Rule-based: 0 anomalies (values are reasonable)
- ML: 1 anomaly (KPI_E2E pattern deviation)

This shows they're complementary: ML caught something rule-based didn't!
