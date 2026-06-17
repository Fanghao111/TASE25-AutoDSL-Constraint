# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is the codebase for IEEE T-ASE paper "Automated Constraint Specification for Job Scheduling by Regulating Generative Model with Domain-Specific Representation." It implements a constraint-centric architecture that uses LLMs regulated by domain-specific representations (DSLs) to convert raw manufacturing descriptions into formal scheduling constraints, solved via Google OR-Tools.

## Setup & Commands

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

LLM-based stages require `OPENAI_API_KEY`. The `autodsl_operation` mode downloads `allenai/scibert_scivocab_uncased` on first run.

### Running Modes

All execution goes through `main.py --mode <MODE>`:

```bash
# Deterministic stages (no LLM needed)
python main.py --mode preprocess
python main.py --mode autodsl_operation
python main.py --mode autodsl_production
python main.py --mode groundtruth

# LLM-dependent stages
python main.py --mode baseline --type CPE_CAE_CSE-2
python main.py --mode baseline_2 --type CPE_CAE_CSE-2
python main.py --mode dsl_pipeline --type CPE_CAE_CSE-2
python main.py --mode fb_pipeline --type CPE_CAE_CSE-2

# Evaluation
python main.py --mode evaluation --evaluation_type CPE   # also: CAE, CSE-1, CSE-2, SGE, DAE
```

`--type` selects the experiment variant: `CPE_CAE_CSE-2`, `CSE-1`, `SGE`, `DAE`.

`--force` / `--no-force` controls whether intermediate outputs are overwritten or reused.

There is no test suite or linter configured.

## Architecture

### Pipeline Flow

The system has a four-stage pipeline for converting NL manufacturing descriptions to production schedules:

1. **Raw → Structured**: LLM translates NL descriptions into structured route sheets using DSL templates
2. **Structured → Constrained**: DSL programs are compiled into resource/precedence constraints
3. **Constrained → Scheduled**: OR-Tools CP-SAT solver produces a JSP schedule (via `src/experiment/schedule.py`)
4. **Scheduled → Grounded**: Solver output is mapped back to concrete production plans with execution details

### Two DSL Hierarchies

- **Operation DSL** (`src/dsl_design/operation.py`): Three-level hierarchy built by clustering route sheet steps using DPMM. Levels: (1) precondition/postcondition flow types, (2) machine/argument keys, (3) parameter values. The `Feature` class extracts feature vectors at each level; `Operation` runs recursive clustering, builds a hierarchy tree, then abstracts and merges patterns into DSL entries.

- **Production DSL** (`src/dsl_design/production.py`): Captures material/component flow relationships (Pred → FlowUnit → Succ). Uses EM algorithm to discover production structure templates (how many predecessors/successors per component).

### Experiment Methods (all in `src/experiment/`)

- **Baseline** (`baseline.py`): Direct LLM-only approach — NL → structural info → OR matrix → schedule → plan.
- **Baseline_2** (`baseline2.py`): Variant baseline with different prompting.
- **DSLPipeline** (`dsl_pipeline.py`): The paper's proposed approach — uses both DSL hierarchies to regulate LLM translation. Supports both OpenAI batch API and parallel single-call modes.
- **FBPipeline** (`fb_pipeline.py`): Alternative "free-form + bootstrap" pipeline. LLM stages: (1) CAM-1 extract, (1v) verify extraction, (2) format alignment, (2v) verify format, (3) normalize values → `CAM-3_normalized_jsons.json`. Deterministic stages: (4) CGM build OR matrix → `CGM_or_matrix.json` + `CGM_machines.json`, (5) OR-Tools solve, (6) SGM ground plan → `SGM_production_plan.json`. Uses `schema_validator.py` for structural validation between LLM steps.
- **GroundTruth** (`groundtruth.py`): Generates reference outputs from known-correct route sheet data.

### Clustering

`src/dsl_design/cluster.py` implements a Dirichlet Process Mixture Model (DPMM) using Gibbs sampling with `N_Gaussian_Distribution` from `utils/distribution.py`. This drives the automatic DSL structure discovery.

### Evaluation Metrics (`src/evaluation/evaluation.py`)

- **CPE/SGE**: BLEU + custom ROUGE-L (flattens JSON structures, compares key-value pairs case-insensitively, skips scheduling-specific fields like start/end/job_id)
- **CSE-1/CSE-2**: Constraint-level IoU (resource constraints as operation→machine pairs, precedence constraints as pred→succ pairs), runtime error rate, makespan ratio
- **CAE**: BLEU + ROUGE-L on route sheet structures
- **DAE**: Aggregates CPE results and computes variance-to-mean ratio (VMR) for stability

### Data Flow

- Input data lives in `data/` (route sheets, JSSP instances, machine configs)
- All pipeline outputs go to `outputs/<Method>-<Type>/<instance>/` (gitignored)
- Batch LLM calls use `data/temp_batch/` for staging JSONL files
- `data/embedding_dic.json` caches OpenAI text embeddings (gitignored)
- The system processes 10 instances: `ta71` through `ta80`
- Note: `legal_instance_description_list` in `main.py` is currently reduced to `["instance ta71"]` for development; restore to full list for final experiments

### LLM Configuration

Different pipeline classes use different LLM backends:
- `DSLPipeline`: uses `deepseek-chat` via `api.zhizengzeng.com` (single calls) and `deepseek-v3` via DashScope (batch API)
- `FBPipeline`: uses `claude-sonnet-4-6` via local proxy at `localhost:4141`
- `Baseline`/`Operation`: uses OpenAI API directly

Prompt templates are stored as `.txt` files in `src/prompts/` with `---PLACEHOLDER---` style substitution.

## Key Conventions

- `utils/util.py` provides `read_json`, `write_json`, `read_txt`, `write_txt`, `seed_set` — used everywhere for I/O.
- JSON outputs use 4-space indent and `ensure_ascii=False`.
- All modes share the same `legal_instance_description_list` (ta71–ta80) defined in `main.py`.
- Pipeline classes follow load-then-run pattern: constructor sets up state, `load_data()` reads any existing intermediate files, `run()` orchestrates the stages.
- The `--force` flag (default True) controls whether intermediate artifacts are overwritten; `--no-force` allows resuming from partial runs.

### Evaluation ↔ Pipeline Stage Mapping

Each evaluation type isolates a specific pipeline capability by controlling the input starting point:

| Eval Type | Input | Measures | Metrics |
|-----------|-------|----------|---------|
| CPE | NL orders (full pipeline) | End-to-end plan quality | BLEU + ROUGE-L on `production_plan.json` / `SGM_production_plan.json` |
| CAE | NL orders (intermediate) | NL→structured translation | BLEU + ROUGE-L on `route_sheets.json` / `CAM-3_normalized_jsons.json` |
| CSE-2 | NL orders (full pipeline) | NL→constraint quality | Constraint IoU + runtime error rate + makespan ratio |
| CSE-1 | GT route sheets (skip NL) | Constraint generation only | Same as CSE-2 |
| SGE | GT solver result (skip solve) | Grounding mapping only | BLEU + ROUGE-L on production plan |
| DAE | CPE results aggregated | Cross-instance stability | VMR (variance-to-mean ratio) |

DSL uses `operation_programs.json` + `production_programs.json` for constraint extraction in CSE; FB uses `CGM_or_matrix.json` + `CGM_machines.json`.
