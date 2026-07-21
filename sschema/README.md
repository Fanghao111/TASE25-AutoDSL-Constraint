# sschema — self-contained structured-schema pipeline for JSP orders

This folder is a self-contained copy of the "FB 2-stage" pipeline: it maps NL
manufacturing orders into a fixed extraction schema, normalizes and formats
that schema, builds a JSP disjunctive graph from it, solves with CP-SAT, and
projects the schedule back into a production plan.

Everything the pipeline needs lives under this folder — raw data, prompts,
common modules, canonical preprocessed artifacts. Nothing imports from paths
outside `sschema/`, so this directory can be lifted out into its own repo
without further work.

## Layout

```
sschema/
├── preprocess.py        stage 1 entry (reproducibility only — canonical
│                        outputs are already shipped under preprocess_out/)
├── pipeline.py          stage 2 entry — the 7-step FB pipeline (s1..s7)
├── evaluate.py          stage 3 entry — score pipeline artifacts vs GT
├── common/              shared modules imported by all three scripts
├── prompts/             6 LLM prompt templates
├── data/                raw jssp / arrange / machines JSONs (~1 MB)
├── preprocess_out/      canonical preprocessed artifacts (~61 MB, committed)
│   ├── jssp_mapped.json
│   ├── route_sheet.json
│   ├── route_sheet_reduce.json
│   └── orders/ta71..ta80/orders.json
└── artifacts/           runtime outputs (gitignored)
    ├── preprocess/      preprocess.py output when re-run
    ├── pipeline/<experiment_type>/<inst>/  s1..s7 intermediates + makespan
    ├── groundtruth/<inst>/                 route_sheets / or_matrix / plan
    └── evaluate/*_report.json              scoring reports
```

## Environment variables

Set these before running `preprocess.py` or `pipeline.py` (both call an
OpenAI-compatible chat endpoint). `evaluate.py` doesn't need them.

| Variable | Default | Purpose |
|---|---|---|
| `LLM_BASE_URL` | `http://localhost:4142/v1` | OpenAI-compatible endpoint |
| `LLM_MODEL` | `gpt-4o` | Model name passed to the client |
| `OPENAI_API_KEY` | `sk-placeholder` | API key |
| `LLM_MAX_WORKERS` | `100` | HTTP pool size + per-stage concurrency |
| `SKIP_ARRANGE_MAPPING` | *(unset)* | Skip the `arrange.json` remap in preprocess |

## Running

Install dependencies (one time):

```
pip install -r sschema/requirements.txt
```

### Stage 2 — pipeline (the main object of study)

Run the 7-step pipeline for a single instance, reading orders from the shipped
canonical `preprocess_out/`:

```
LLM_MODEL=gpt-4o LLM_BASE_URL=http://localhost:4142/v1 \
  python sschema/pipeline.py --instances ta71 --experiment-type full
```

Ablations that skip early stages by feeding in ground truth:

```
python sschema/pipeline.py --instances ta71 --experiment-type from_gt_route_sheet
python sschema/pipeline.py --instances ta71 --experiment-type from_gt_schedule
```

Outputs land under `artifacts/pipeline/<experiment_type>/<inst>/`
(`s1_extracted.json` .. `s7_production_plan.json`, plus `makespan.txt`).
Ground truth is auto-built under `artifacts/groundtruth/<inst>/` when the
`from_gt_*` variants need it.

### Stage 3 — evaluate

```
python sschema/evaluate.py --instances ta71 --type all
```

Or one metric at a time — see `python sschema/evaluate.py --help` for the full
list. Reports go to `artifacts/evaluate/*_report.json` and a summary is
printed to stdout.

### Stage 1 — preprocess (only if you want to regenerate)

The canonical outputs under `preprocess_out/` were produced with the same
script and are what pipeline.py reads by default. Only re-run this to
reproduce that build:

```
python sschema/preprocess.py --stage all --out-dir sschema/artifacts/preprocess
python sschema/pipeline.py --instances ta71 --preprocess-dir sschema/artifacts/preprocess
```

`--stage` accepts `mapping`, `route_sheet`, `reduce`, `orders`, or `all`.
