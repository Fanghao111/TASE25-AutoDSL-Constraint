# Naming Migration — FB Pipeline & Evaluation

Renamed on 2026-07-01. Old logs, notes under `note/`, and any evaluation JSON files produced before this date use the old names. New code, new outputs, and this document use the new names.

## Pipeline steps

| Old method name (`FBPipeline.*`) | New method name | Old dump file (per instance) | New dump file |
|---|---|---|---|
| `extract_all()` | `extract()` | `step1_extracted.json` | `s1_extracted.json` |
| `verify_extract()` | `verify()` | `step1_verified.json` | `s2_verified.json` |
| `verify_format_code()` | `apply_format()` | `step1_format_fixed.json` | `s3_formatted.json` |
| `normalize_all()` | `normalize()` | `step3_normalized.json` | `s4_normalized.json` |
| — | — | `step3_mappings.json` | `s4_mappings.json` |
| — | — | `step3_mappings_pre_verify.json` | `s4_mappings_pre.json` |
| — | — | `CAM-3_normalized_jsons.json` | **removed** (evaluation now reads `s4_normalized.json` directly) |
| `build_or_matrix()` | `build_graph()` | `CGM_or_matrix.json` | `s5_or_matrix.json` |
| — | — | `CGM_machines.json` | `s5_machines.json` |
| `solve_jsp()` | `solve()` | `assigned_jobs.json` | `s6_assigned_jobs.json` |
| — | — | `err_rate.txt` | `s6_err_rate.txt` |
| — | — | `makespan.txt` | `s6_makespan.txt` |
| `ground_production_plan()` | `ground()` | `SGM_production_plan.json` | `s7_production_plan.json` |
| — | — | `compile_error_num.txt` | `s7_compile_errors.txt` |

Attribute names inside `FBPipeline`:

| Old | New |
|---|---|
| `self.format_verified_jsons` | `self.formatted_jsons` |
| `self.step1_extract_prompt` | `self.s1_extract_prompt` |
| `self.step1_verify_prompt` | `self.s2_verify_prompt` |
| `self.step3_normalize_prompt` | `self.s4_normalize_prompt` |
| `self.step3_verify_prompt` | `self.s4_verify_prompt` |

## Experiment type (`--type`, `FBPipeline.__init__(experiment_type=…)`)

| Old | New | Meaning |
|---|---|---|
| `CPE_CAE_CSE-2` | `full` | Run all of s1..s7 from NL orders |
| `CSE-1` | `from_gt_route_sheet` | Skip s1..s4, start s5..s7 from GT route sheets |
| `SGE` | `from_gt_schedule` | Skip s1..s6, start s7 from GT assigned_jobs |
| `DAE` | `dae` | Kept as-is (DSL diagnostic; unused by FB) |

**Note**: `baseline`, `baseline_2`, `dsl_pipeline` modes still use the old strings (`CPE_CAE_CSE-2`, `CSE-1`, `SGE`, `DAE`) internally, so `main.py` translates the new `--type` back to legacy names when invoking them (see `_LEGACY_TYPE_MAP`). This preserves DSL/Baseline output layouts for cross-model comparability.

## Evaluation type (`--evaluation_type`)

| Old | New | Reads FB file | Writes eval file (`outputs/Evaluation/`) |
|---|---|---|---|
| `CPE` | `production_plan` | `s7_production_plan.json` | `production_plan_{fb,dsl,baseline,baseline2}_{bleu,rouge}.json` |
| `CAE` | `route_sheet` | `s4_normalized.json` → `build_route_sheets` | `route_sheet_{fb,dsl,baseline}_{bleu,rouge}.json` |
| `CSE-1` | `graph_from_gt` | `s5_or_matrix.json`, `s5_machines.json`, `s6_err_rate.txt`, `s6_makespan.txt` | `graph_from_gt_{fb,dsl,baseline}.json` |
| `CSE-2` | `graph` | `s5_or_matrix.json`, `s5_machines.json`, `s6_err_rate.txt`, `s6_makespan.txt` | `graph_{fb,dsl,baseline}.json` |
| `SGE` | `ground_from_gt` | `s7_production_plan.json` | `ground_from_gt_{fb,dsl}_{bleu,rouge}.json` |
| `DAE` | `dae` | reads prior `production_plan_*` results | `dae.json`, `dae_metric.json`, `dae_vmr.json` |

## Directory layout (unchanged)

- FB pipeline outputs: `outputs/FB-models/<model>/instance taXX/{s1..s7}_*.{json,txt}`
- Evaluation aggregate outputs (all-model): `outputs/Evaluation/{eval_type}_{baseline,baseline2,dsl}_*.json`
- Evaluation per-model FB outputs: `outputs/Evaluation/FB-models_<model>/{eval_type}_fb_*.json`
- Comparison markdown: `outputs/Evaluation/FB-models-comparison-report.md`

## Orders input path

Pipeline now reads orders **directly from the canonical `preprocess/orders/<ta_num>/orders.json`** (e.g. `preprocess/orders/ta71/orders.json`) — no per-model staging copy. The wrapper `run_fb_pipeline_qwen.ps1` used to `Copy-Item` orders into each model's dump dir before running; this step was removed. Dump dirs (`outputs/FB-models/<model>/instance taXX/`) now hold only pipeline outputs (`s1_*.json` .. `s7_*.txt`).

## Directory layout (2026-07-01 refactor)

- `data/` — raw inputs only: `machines.json`, `jssp_data.json`, `arrange.json`
- `preprocess/` — artifacts derived from raw inputs (no LLM method involved):
  - `jssp_mapped.json`, `route_sheet.json`, `route_sheet_reduce.json`
  - `orders/ta{71..80}/orders.json`
- `outputs/` — everything downstream of orders (per-method intermediates and final results): `AutoDSL/`, `DSL-models/`, `FB-models/`, `FB-gt-ceiling/`, `GroundTruth/`, `Evaluation/`

## Environment variables (unchanged)

- `FB_OUTPUT_PREFIX` — overrides FB pipeline dump root (e.g. `FB-models/qwen3-4b`)
- `FB_EVAL_LABEL` — overrides FB evaluation read/write root
- `LLM_BASE_URL`, `LLM_MODEL`, `OPENAI_API_KEY`, `LLM_MAX_WORKERS` (default lowered from 96 → 64)

## What was NOT renamed

- **Prompt files** (`src/prompts/step1_extract.txt`, `step1_verify.txt`, `step3_normalize.txt`, `step3_verify.txt`) — data assets, keeping name for git blame continuity.
- **DSL / Baseline / Baseline2 output layouts** — they use legacy names (`CPE_CAE_CSE-2/instance taXX/or_matrix.json`, etc.) and keep them so cross-model comparability holds.
- **GroundTruth output layout** — external reference data, not touched.
- **Old on-disk outputs** — pre-migration files in `outputs/FB-models/*/instance taXX/` (`step1_*.json`, `CGM_*.json`, `SGM_*.json` etc.) remain on disk. New code only reads new names. To use old data with new code, rename the files or rerun the pipeline.
- **Historical notes** under `note/*.md` — timestamped snapshots, kept as-is.
- **Historical evaluation JSONs** in `outputs/Evaluation/CPE_*.json`, `CAE_*.json`, `CSE-2_*.json` etc. — kept but no longer read by new code.

## Rerunning after migration

The old evaluation JSONs (`CPE_fb_bleu.json` etc.) and old pipeline dumps (`SGM_production_plan.json` etc.) are still on disk but new code writes to `production_plan_fb_bleu.json` and `s7_production_plan.json`. To get a clean run under the new naming:

```powershell
# Full 4-model run + eval + report (new wrapper already uses new names)
.\scripts\run_fb_pipeline_qwen.ps1

# Or just one model
.\scripts\run_fb_pipeline_qwen.ps1 qwen3-4b
```
