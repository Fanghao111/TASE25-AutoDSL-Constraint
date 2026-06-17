# Full Experiment Plan: DSL vs FB Pipeline Evaluation

**Branch**: fb-1  
**Date**: 2026-06-17  
**Instances**: ta71-ta80 (10 instances, each 101 orders)  
**LLM Backend**: localhost:4141 proxy (serial, no parallel)

---

## Execution Order (all serial)

```bash
source .venv/bin/activate

# Phase 0: Prerequisites (deterministic, no LLM)
python main.py --mode groundtruth                    # ~5 min
python main.py --mode autodsl_operation              # ~30 min (SciBERT)
python main.py --mode autodsl_production             # ~5 min (EM)

# Phase 1: Generate NL orders for all 10 instances
python generate_orders.py                            # ~60 min (101×10 LLM calls)

# Phase 2: DSL Pipeline (LLM calls via proxy)
python main.py --mode dsl_pipeline --type CPE_CAE_CSE-2   # ~3h (each instance: extraction + translation)
python main.py --mode dsl_pipeline --type CSE-1           # ~2h (translation from GT routes)
python main.py --mode dsl_pipeline --type SGE             # ~1h (grounding)

# Phase 3: FB Pipeline (LLM calls via proxy)
python main.py --mode fb_pipeline --type CPE_CAE_CSE-2 --force   # ~5h (4 LLM steps × 101 × 10)
python main.py --mode fb_pipeline --type CSE-1 --force           # <5 min (deterministic)
python main.py --mode fb_pipeline --type SGE --force             # <5 min (deterministic)

# Phase 4: All Evaluations (fast, no LLM)
python main.py --mode evaluation --evaluation_type CPE
python main.py --mode evaluation --evaluation_type CAE
python main.py --mode evaluation --evaluation_type CSE-1
python main.py --mode evaluation --evaluation_type CSE-2
python main.py --mode evaluation --evaluation_type SGE
```

---

## Expected Output Structure

```
outputs/
├── GroundTruth/
│   └── instance ta7{1..0}/  (route_sheets.json, or_matrix.json, assigned_jobs.json, makespan.txt, production_plan.json)
├── AutoDSL/
│   ├── total_operation_dsl.json (10 instances)
│   ├── total_production_dsl.json (10 instances)
│   └── EM_results.json (10 instances)
├── DSLPipeline-CPE_CAE_CSE-2/
│   └── instance ta7{1..0}/  (orders.json, operation_programs.json, production_programs.json, or_matrix.json, machines.json, assigned_jobs.json, makespan.txt, production_plan.json, route_sheets.json, err_rate.txt)
├── DSLPipeline-CSE-1/
│   └── instance ta7{1..0}/  (same as above)
├── DSLPipeline-SGE/
│   └── instance ta7{1..0}/  (production_plan.json)
├── FB-CPE_CAE_CSE-2/
│   └── instance ta7{1..0}/  (orders.json, step1_extracted.json, step1_verified.json, step2_formatted.json, step2_verified.json, step3_normalized.json, step3_mappings.json, CGM_or_matrix.json, CGM_machines.json, assigned_jobs.json, makespan.txt, SGM_production_plan.json, err_rate.txt, compile_error_num.txt)
├── FB-CSE-1/
│   └── instance ta7{1..0}/  (CGM_or_matrix.json, CGM_machines.json, assigned_jobs.json, makespan.txt, SGM_production_plan.json)
├── FB-SGE/
│   └── instance ta7{1..0}/  (SGM_production_plan.json)
└── Evaluation/
    ├── CPE_{dsl,fb}_{bleu,rouge}.json
    ├── CAE_{dsl,fb}_{bleu,rouge}.json
    ├── CSE-1_{dsl,fb}.json
    ├── CSE-2_{dsl,fb}.json
    └── SGE_{dsl,fb}_{bleu,rouge}.json
```

---

## Evaluation Metrics (paper format)

| Eval Type | Metrics | Description |
|-----------|---------|-------------|
| CPE | BLEU, ROUGE-L (P/R/F1) | Production plan quality |
| CAE | BLEU, ROUGE-L (P/R/F1) | Route sheet extraction accuracy |
| CSE-1 | Constraint IoU, Makespan Ratio, Runtime Error Rate | Constraint from GT routes |
| CSE-2 | Constraint IoU, Makespan Ratio, Runtime Error Rate | End-to-end constraint quality |
| SGE | BLEU, ROUGE-L (P/R/F1) | Schedule grounding quality |
| Makespan | Raw value per instance, Ratio vs GT | Scheduling quality |

All metrics averaged over 10 instances.

---

## Estimated Total Runtime

| Phase | Duration |
|-------|----------|
| Prerequisites (GT + AutoDSL) | ~40 min |
| Generate orders | ~60 min |
| DSL Pipeline (all types) | ~6 h |
| FB Pipeline (CPE_CAE_CSE-2) | ~5 h |
| FB Pipeline (CSE-1 + SGE) | <10 min |
| Evaluations | <5 min |
| **Total** | **~12 h** |
