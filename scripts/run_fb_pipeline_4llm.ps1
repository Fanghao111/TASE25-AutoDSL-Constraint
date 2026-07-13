# Run FB pipeline + evaluation + comparison report across 4 LLMs served by
# the mi300 cluster's LiteLLM gateway. Single endpoint (localhost:8000/v1),
# LiteLLM routes to the right sglang pool via the `model` field.
#
# Usage:
#   .\scripts\run_fb_pipeline_4llm.ps1                                # all 4 models
#   .\scripts\run_fb_pipeline_4llm.ps1 qwen3-4b                       # one model
#   .\scripts\run_fb_pipeline_4llm.ps1 deepseek-v3 qwen3-235b         # subset
#
# Prereq: SSH tunnel forwarding local :8000 to mi300-1's LiteLLM gateway
#         (bash D:/FB/mi300/tunnel.sh).

Set-Location (Join-Path $PSScriptRoot '..')
New-Item -ItemType Directory -Force -Path logs | Out-Null

$models = @('deepseek-v3','qwen3-235b','qwen3-30b','qwen3-4b')
if ($args.Count -gt 0) {
  $selected = $args
  $models = $models | Where-Object { $selected -contains $_ }
}
if (-not $models) {
  Write-Error "No matching models. Valid names: deepseek-v3, qwen3-235b, qwen3-30b, qwen3-4b"
  exit 1
}

$instances = @('ta71','ta72','ta73','ta74','ta75','ta76','ta77','ta78','ta79','ta80')
$evalTypes = @('production_plan', 'route_sheet', 'graph')

# ---- Gateway env (shared across all invocations) ------------------------
$env:LLM_BASE_URL   = 'http://localhost:8000/v1'
$env:OPENAI_API_KEY = 'sk-mi300-local'
if (-not $env:LLM_MAX_WORKERS) { $env:LLM_MAX_WORKERS = '64' }

# ---- GT ceiling (LLM-free, shared across all models) --------------------
# from_gt_route_sheet: GT route sheet -> s5..s7 (deterministic code + OR-Tools)
# from_gt_schedule:    GT assigned_jobs -> s7 only (deterministic code)
# Idempotent: skip if the last-instance sentinel is already present.
$ceilingSentinel = 'outputs/FB-gt-ceiling/from_gt_sched/instance ta80/s7_production_plan.json'
if (Test-Path $ceilingSentinel) {
  Write-Host "==== [GT ceiling] already computed, skipping ====" -ForegroundColor DarkGray
} else {
  Write-Host "==== [GT ceiling] from_gt_route_sheet + from_gt_schedule ====" -ForegroundColor Cyan

  # from_gt modes never call the LLM, but the code lazily builds an OpenAI client
  # from these env vars; the gateway env above is enough.
  $env:LLM_MODEL = 'gt-ceiling'

  $env:FB_OUTPUT_PREFIX = 'FB-gt-ceiling/from_gt_rs'
  $ceilingArgsRs = @('-u', 'main.py', '--mode', 'fb_pipeline', '--type', 'from_gt_route_sheet', '--instances') + $instances
  & '.\.venv\Scripts\python.exe' @ceilingArgsRs 2>&1 |
    Tee-Object -FilePath 'logs\fb_pipeline_gt_from_gt_rs.log'

  $env:FB_OUTPUT_PREFIX = 'FB-gt-ceiling/from_gt_sched'
  $ceilingArgsSched = @('-u', 'main.py', '--mode', 'fb_pipeline', '--type', 'from_gt_schedule', '--instances') + $instances
  & '.\.venv\Scripts\python.exe' @ceilingArgsSched 2>&1 |
    Tee-Object -FilePath 'logs\fb_pipeline_gt_from_gt_sched.log'

  $env:FB_EVAL_LABEL = 'FB-gt-ceiling/from_gt_rs'
  & '.\.venv\Scripts\python.exe' -u main.py --mode evaluation --evaluation_type graph_from_gt 2>&1 |
    Tee-Object -FilePath 'logs\fb_eval_gt_graph_from_gt.log'

  $env:FB_EVAL_LABEL = 'FB-gt-ceiling/from_gt_sched'
  & '.\.venv\Scripts\python.exe' -u main.py --mode evaluation --evaluation_type ground_from_gt 2>&1 |
    Tee-Object -FilePath 'logs\fb_eval_gt_ground_from_gt.log'

  Remove-Item Env:\FB_EVAL_LABEL
  Remove-Item Env:\FB_OUTPUT_PREFIX
}

foreach ($m in $models) {
  $prefix = "FB-models/$m"
  Write-Host "==== [$m] pipeline + evaluation ====" -ForegroundColor Cyan

  # Pipeline reads orders directly from canonical preprocess/orders/<ta_num>/orders.json
  # (no per-model copy needed — pipeline dump dirs hold only pipeline outputs).

  $env:LLM_MODEL        = $m
  $env:FB_OUTPUT_PREFIX = $prefix

  $pyArgs = @('-u', 'main.py', '--mode', 'fb_pipeline', '--type', 'full', '--instances') + $instances
  & '.\.venv\Scripts\python.exe' @pyArgs 2>&1 |
    Tee-Object -FilePath "logs\fb_pipeline_$m.log"
  if ($LASTEXITCODE -ne 0) {
    Write-Warning "[$m] pipeline exited $LASTEXITCODE — skipping evaluation"
    continue
  }

  # Evaluation (per type). FB_EVAL_LABEL routes FB inputs + result dir.
  $env:FB_EVAL_LABEL = $prefix
  foreach ($t in $evalTypes) {
    & '.\.venv\Scripts\python.exe' -u main.py --mode evaluation --evaluation_type $t 2>&1 |
      Tee-Object -FilePath "logs\fb_eval_${m}_$t.log"
  }
  Remove-Item Env:\FB_EVAL_LABEL
}

# Comparison report — scans FB-models_* and DSL-models_*.
Write-Host "==== generating comparison report ====" -ForegroundColor Cyan
& '.\.venv\Scripts\python.exe' -u scripts/generate_comparison_report.py 2>&1 |
  Tee-Object -FilePath 'logs\comparison_report.log'

Write-Host ""
Write-Host "Done. Report: outputs/Evaluation/FB-models-comparison-report.md" -ForegroundColor Green
