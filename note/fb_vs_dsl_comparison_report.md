# FB Pipeline vs DSL Pipeline Comparison Report

**Date**: 2026-06-16  
**Instance**: ta71 (101 orders, single run)  
**FB Pipeline Version**: Refactored (fixed schema, 6-step architecture)  
**DSL Pipeline**: Current version (deepseek-v3 batch API)

---

## Executive Summary

The refactored FB pipeline significantly outperforms the DSL pipeline in extraction quality (2.3x BLEU) and constraint accuracy (+16.7% IoU), while achieving identical scheduling performance on end-to-end runs. The fixed-schema approach eliminates SRD complexity and produces zero bad cases.

---

## Evaluation Results

### CSE-2: End-to-End Constraint Accuracy (NL → Constraints)

| Metric | DSL Pipeline | FB Pipeline | Delta |
|--------|-------------|-------------|-------|
| Constraint IoU | 0.325 | **0.379** | +16.7% |
| Runtime Error Rate | 0.0 | 0.0 | = |
| Makespan Ratio (vs GT) | 9.25 | 9.25 | = |

### CSE-1: Constraint Accuracy from Ground Truth Routes

| Metric | DSL Pipeline | FB Pipeline | Delta |
|--------|-------------|-------------|-------|
| Constraint IoU | 0.675 | **1.000** | +48.1% |
| Runtime Error Rate | 0.0 | 0.0 | = |
| Makespan Ratio (vs GT) | 9.18 | **2.59** | -71.8% (better) |

> CSE-1 uses ground truth route sheets as input. FB's fixed schema aligns perfectly with GT format, achieving perfect constraint extraction. This confirms the deterministic constraint-building logic is correct.

### CAE: Route Sheet Extraction Accuracy

| Metric | DSL Pipeline | FB Pipeline | Delta |
|--------|-------------|-------------|-------|
| BLEU | 0.382 | **0.885** | +131.7% |
| ROUGE Precision | 0.550 | **0.656** | +19.3% |
| ROUGE Recall | 0.319 | **0.639** | +100.3% |
| ROUGE F1 | 0.404 | **0.647** | +60.1% |

### CPE: Production Plan Accuracy

| Metric | DSL Pipeline | FB Pipeline | Delta |
|--------|-------------|-------------|-------|
| BLEU | 0.391 | **0.866** | +121.5% |
| ROUGE Precision | 0.502 | 0.374 | -25.5% |
| ROUGE Recall | 0.272 | **0.371** | +36.4% |
| ROUGE F1 | 0.353 | **0.373** | +5.7% |

---

## Pipeline Reliability

| Metric | Old FB (SRD) | New FB (Fixed Schema) |
|--------|-------------|----------------------|
| Bad Cases | N/A | **0 / 101** |
| Compile Errors | 5 | **0** |
| Runtime Errors | 0 | 0 |
| SRD_field_mapping.json needed | Yes | **No** |

---

## Architecture Comparison

| Aspect | DSL Pipeline | FB Pipeline (New) |
|--------|-------------|-------------------|
| LLM Calls per order | 2 (translate + compile) | 4 (extract + verify + format + format-verify) + 2 global (normalize + normalize-verify) |
| Schema | Free-form → DSL compilation | Fixed schema from step 1 |
| Normalization | DSL-regulated | LLM-based synonym grouping |
| Constraint Building | DSL program execution | Deterministic field access |
| Total Runtime (ta71) | ~15 min (batch API) | ~37 min (serial, 2 workers) |
| Code Complexity | ~600 lines + DSL infrastructure | ~350 lines, self-contained |

---

## Key Observations

### Strengths of FB Pipeline
1. **Much higher extraction fidelity**: BLEU 0.885 vs 0.382 for CAE — the fixed schema produces structured data very close to ground truth
2. **Zero bad cases**: All 101 orders parsed successfully with valid schemas
3. **Better constraint accuracy**: 37.9% vs 32.5% IoU despite over-normalization
4. **Perfect CSE-1**: When given ground truth routes, produces perfect constraint extraction
5. **Simpler code**: No DSL infrastructure needed, no SRD stage, no field_mapping indirection

### Weakness: Component Type Over-Normalization
The normalization step (Step 3) merges specific component types into generic categories:
- Ground truth: `"SheetMetal"` → `"CutSheetMetal"` → `"LathedPart"`
- FB pipeline: `"RawMaterial"` → `"ProcessedComponent"` → `"ProcessedComponent"`

This causes precedence constraint errors because the type matching can't distinguish between different intermediate products. Despite this, the overall IoU is still higher than DSL because FB correctly captures more resource constraints.

### Makespan Analysis
- Ground Truth: 1411
- Old FB (SRD): 2644 (ratio: 1.87)
- New FB (fixed schema): 13048 (ratio: 9.25)
- DSL Pipeline: 13048 (ratio: 9.25)

The new FB pipeline has worse makespan than the old version due to the over-normalization of component types causing incorrect precedence constraints. The same or_matrix structure produces the same solver behavior as DSL.

---

## Recommendations

1. **Fix component_type normalization**: The Step 3 normalize prompt should be instructed to preserve material-specific types (SheetMetal, CutSheetMetal, LathedPart) rather than collapsing them into generic categories. This is the single highest-impact improvement.

2. **Consider splitting Step 3**: Normalize machine/operation names (safe to merge) separately from component_type (should preserve specificity for precedence constraints).

3. **Increase parallelism**: Current max_workers=2 could be raised to 4-8 to cut runtime from 37min to ~10min.

4. **Batch API migration**: Like DSL pipeline's DashScope batch mode, FB could batch all LLM calls per step for cost reduction.

---

## Conclusion

The refactored FB pipeline demonstrates fundamentally superior extraction quality (+131% BLEU) with a simpler architecture. The main gap vs ground truth is in semantic normalization of component types — a targeted fix to Step 3 prompting would likely recover the old pipeline's makespan advantage while retaining the new architecture's reliability (zero bad cases) and code simplicity.
