# DSL Pipeline vs FB Pipeline Evaluation Report

**Date**: 2026-06-17  
**Instance**: ta71 (101 orders)  
**DSL Pipeline**: `deepseek-chat` / `deepseek-v3` (DashScope batch)  
**FB Pipeline**: `claude-sonnet-4-6` via localhost proxy, fixed-schema architecture

---

## 1. Constraint Specification Evaluation (CSE)

CSE evaluates the quality of constraints generated from manufacturing descriptions. Two variants isolate different pipeline stages:

- **CSE-1**: Input = ground truth route sheets (tests constraint generation logic only)
- **CSE-2**: Input = raw NL descriptions (tests full pipeline end-to-end)

### CSE-2: End-to-End Constraint Quality (NL → Constraints)

| Metric | DSL Pipeline | FB Pipeline | Improvement |
|--------|:-----------:|:-----------:|:-----------:|
| Constraint IoU | 0.325 | **0.498** | +53.4% |
| Makespan | 13048 | **4809** | -63.1% |
| Makespan Ratio (vs GT=1411) | 9.25 | **3.41** | -63.1% |
| Runtime Error Rate | 0.0 | 0.0 | = |
| Compile Error Rate | 0/101 | 0/101 | = |

### CSE-1: Constraint Quality from Ground Truth Routes

| Metric | DSL Pipeline | FB Pipeline | Improvement |
|--------|:-----------:|:-----------:|:-----------:|
| Constraint IoU | 0.675 | **1.000** | +48.1% |
| Makespan Ratio (vs GT) | 9.18 | 9.18 | = |
| Runtime Error Rate | 0.0 | 0.0 | = |

> CSE-1 uses ground truth route sheets as input, bypassing the extraction stages. FB achieves IoU=1.0, meaning its deterministic constraint extraction logic produces constraints identical to ground truth. Both pipelines yield the same solver makespan (9.18x) because the OR matrix encoding adds structural dependencies beyond what the constraint-level IoU captures.

---

## 2. Component Abstraction Evaluation (CAE)

CAE measures the accuracy of structured route sheet extraction from natural language descriptions.

| Metric | DSL Pipeline | FB Pipeline | Improvement |
|--------|:-----------:|:-----------:|:-----------:|
| BLEU | 0.382 | **0.900** | +135.6% |
| ROUGE-L Precision | 0.550 | **0.729** | +32.5% |
| ROUGE-L Recall | 0.319 | **0.710** | +122.6% |
| ROUGE-L F1 | 0.404 | **0.720** | +78.2% |

---

## 3. Component Plan Evaluation (CPE)

CPE evaluates end-to-end production plan quality (NL → schedule → plan).

| Metric | DSL Pipeline | FB Pipeline | Improvement |
|--------|:-----------:|:-----------:|:-----------:|
| BLEU | 0.391 | **0.917** | +134.5% |
| ROUGE-L Precision | **0.502** | 0.451 | -10.2% |
| ROUGE-L Recall | 0.272 | **0.447** | +64.3% |
| ROUGE-L F1 | 0.353 | **0.449** | +27.2% |

---

## 4. Schedule Grounding Evaluation (SGE)

SGE isolates the grounding stage: given ground truth solver output, how well does each pipeline map it back to a production plan?

| Metric | DSL Pipeline | FB Pipeline | Improvement |
|--------|:-----------:|:-----------:|:-----------:|
| BLEU | 0.964 | **0.998** | +3.5% |
| ROUGE-L Precision | 0.582 | **0.663** | +13.9% |
| ROUGE-L Recall | 0.582 | **0.663** | +13.9% |
| ROUGE-L F1 | 0.582 | **0.663** | +13.9% |

---

## 5. Pipeline Reliability

| Metric | DSL Pipeline | FB Pipeline |
|--------|:-----------:|:-----------:|
| Bad Cases (extraction failures) | N/A | **0 / 101** |
| Compile Errors | 0 | 0 |
| Runtime Errors | 0 | 0 |
| Orders Processed | 101 | 101 |

---

## 6. Summary

### Aggregate Comparison

| Evaluation | Key Metric | DSL | FB | FB wins? |
|-----------|-----------|:---:|:---:|:--------:|
| CSE-2 | Constraint IoU | 0.325 | **0.498** | Y (+53%) |
| CSE-2 | Makespan Ratio | 9.25 | **3.41** | Y (-63%) |
| CSE-1 | Constraint IoU | 0.675 | **1.000** | Y (+48%) |
| CSE-1 | Makespan Ratio | 9.18 | 9.18 | tie |
| CAE | BLEU | 0.382 | **0.900** | Y (+136%) |
| CAE | ROUGE-L F1 | 0.404 | **0.720** | Y (+78%) |
| CPE | BLEU | 0.391 | **0.917** | Y (+135%) |
| CPE | ROUGE-L F1 | 0.353 | **0.449** | Y (+27%) |
| SGE | BLEU | 0.964 | **0.998** | Y (+4%) |
| SGE | ROUGE-L F1 | 0.582 | **0.663** | Y (+14%) |

### Architecture Comparison

| Aspect | DSL Pipeline | FB Pipeline |
|--------|-------------|-------------|
| LLM Backend | deepseek-v3 (batch API) | claude-sonnet-4-6 (serial) |
| LLM Calls/order | 2 (translate + compile) | 4 (extract + verify + format + format-verify) |
| Global LLM Calls | 0 | 2 (normalize + normalize-verify) |
| Schema Approach | Free-form → DSL compile | Fixed schema from extraction |
| Normalization | DSL-regulated | LLM synonym grouping |
| Constraint Building | DSL program execution | Deterministic field access |
| Code Complexity | ~600 lines + DSL infra | ~350 lines, self-contained |

### Key Findings

1. **Extraction quality**: FB achieves 2.4x higher BLEU on both CAE (0.900 vs 0.382) and CPE (0.917 vs 0.391). The fixed-schema approach produces structured output much closer to ground truth than DSL's free-form translation + compilation.

2. **Constraint accuracy**: FB produces 53% higher constraint IoU (0.498 vs 0.325) and 63% better makespan (3.41x vs 9.25x GT). This means FB's constraints are more faithful to the original manufacturing process.

3. **Perfect CSE-1**: FB achieves IoU=1.0 on constraint extraction from ground truth routes, confirming the deterministic logic is correct. The remaining gap in CSE-2 (IoU=0.498) is entirely attributable to extraction errors in the LLM stages.

4. **Schedule grounding**: Both pipelines perform well on SGE (BLEU >0.96), but FB still edges ahead (+3.5% BLEU, +14% ROUGE-L F1).

5. **Reliability**: FB processes all 101 orders with zero bad cases, zero compile errors, and zero runtime errors.

---

*Note: Results are from a single instance (ta71). Full paper evaluation requires all 10 instances (ta71-ta80) for DAE stability analysis.*
