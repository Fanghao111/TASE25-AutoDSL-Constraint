# Experiment Results - Instance ta71 (2026-06-15)

Ground Truth makespan: 1411

## CSE-1: Constraint Specification Evaluation (Isolated)
Input: GT route sheets → Output: OR matrix

| Metric | DSL Pipeline | FB Pipeline |
|--------|-------------|-------------|
| Constraint IoU | 0.675 | 1.000 |
| Runtime Error Rate | 0.0 | 0.0 |
| Makespan | 12,960 (9.18x GT) | 3,655 (2.59x GT) |

## CSE-2: Constraint Specification Evaluation (Incorporated, from NL)
Input: NL orders → Output: OR matrix

| Metric | DSL Pipeline | FB Pipeline |
|--------|-------------|-------------|
| Constraint IoU | 0.325 | 0.512 |
| Runtime Error Rate | 0.0 | 0.0 |
| Makespan | 13,048 (9.25x GT) | 2,644 (1.87x GT) |

## CPE: Complete Pipeline Evaluation
Input: NL orders → Output: production plan

| Metric | DSL Pipeline | FB Pipeline |
|--------|-------------|-------------|
| BLEU | 0.391 | 0.887 |
| ROUGE-L Precision | 0.502 | 0.291 |
| ROUGE-L Recall | 0.272 | 0.287 |
| ROUGE-L F1 | 0.353 | 0.289 |

## CAE: Constraint Abstraction Evaluation
Input: NL orders → Output: route sheets

| Metric | DSL Pipeline | FB Pipeline |
|--------|-------------|-------------|
| BLEU | 0.382 | 0.831 |
| ROUGE-L Precision | 0.550 | 0.303 |
| ROUGE-L Recall | 0.319 | 0.296 |
| ROUGE-L F1 | 0.404 | 0.299 |

## SGE: Schedule Grounding Evaluation
Input: GT assigned_jobs → Output: production plan

| Metric | DSL Pipeline | FB Pipeline |
|--------|-------------|-------------|
| BLEU | 0.964 | 0.998 |
| ROUGE-L F1 | 0.582 | 0.663 |

## DAE: Data Adaptation Evaluation
Skipped — requires 10 instances, only tested ta71.

---

## Key Findings

1. **Constraint quality**: FB leads significantly on CSE-1 (IoU 1.0 vs 0.675) and CSE-2 (0.512 vs 0.325)
2. **Scheduling quality**: FB makespan much better (CSE-1: 2.59x vs 9.18x; CSE-2: 1.87x vs 9.25x)
3. **End-to-end BLEU**: FB leads in CPE (0.887 vs 0.391) and CAE (0.831 vs 0.382)
4. **ROUGE-L**: DSL slightly better in CPE/CAE F1 (0.35/0.40 vs 0.29/0.30) — DSL's structure is incomplete but key-value precision is higher
5. **SGE**: Both perform well (BLEU>0.96), FB has higher ROUGE-L (0.663 vs 0.582)

## Configuration

- LLM: gpt-4o via local proxy (localhost:4141)
- DSL pipeline: `orders2dsl_program_no_batch_parallel` (ThreadPoolExecutor)
- FB pipeline: non-streaming gpt-4o
- No baseline (MSL/TSL) was run (missing prompts)
- IoU evaluation uses case-insensitive matching
