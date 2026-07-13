# Run DSL pipeline + evaluation across 4 LLMs served by the mi300 cluster's
# LiteLLM gateway. Mirrors run_fb_pipeline_4llm.ps1 for the DSL pipeline.
#
# Usage:
#   .\scripts\run_dsl_pipeline_4llm.ps1                                # all 4 models
#   .\scripts\run_dsl_pipeline_4llm.ps1 qwen3-4b                       # one model
#   .\scripts\run_dsl_pipeline_4llm.ps1 deepseek-v3 qwen3-235b         # subset
#
# Prereqs:
#   1) SSH tunnel forwarding local :8000 to mi300-1 (bash D:/FB/mi300/tunnel.sh)
#   2) AutoDSL artifacts present: outputs/AutoDSL/{total_operation_dsl,total_production_dsl,EM_results}.json
#      (run main.py --mode autodsl_operation && --mode autodsl_production first)
#   3) Canonical orders present: preprocess/orders/ta7?/orders.json

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

foreach ($m in $models) {
  $prefix = "DSL-models/$m"
  Write-Host "==== [$m] DSL pipeline + evaluation ====" -ForegroundColor Cyan

  $env:LLM_MODEL         = $m
  $env:DSL_OUTPUT_PREFIX = $prefix

  $pyArgs = @('-u', 'main.py', '--mode', 'dsl_pipeline', '--type', 'full', '--instances') + $instances
  & '.\.venv\Scripts\python.exe' @pyArgs 2>&1 |
    Tee-Object -FilePath "logs\dsl_pipeline_$m.log"
  if ($LASTEXITCODE -ne 0) {
    Write-Warning "[$m] pipeline exited $LASTEXITCODE — skipping evaluation"
    continue
  }

  # Evaluation (per type). DSL_EVAL_LABEL routes DSL inputs + result dir.
  $env:DSL_EVAL_LABEL = $prefix
  foreach ($t in $evalTypes) {
    & '.\.venv\Scripts\python.exe' -u main.py --mode evaluation --evaluation_type $t 2>&1 |
      Tee-Object -FilePath "logs\dsl_eval_${m}_$t.log"
  }
  Remove-Item Env:\DSL_EVAL_LABEL
}

# Comparison report — scans FB-models_* and DSL-models_*.
Write-Host "==== generating comparison report ====" -ForegroundColor Cyan
& '.\.venv\Scripts\python.exe' -u scripts/generate_comparison_report.py 2>&1 |
  Tee-Object -FilePath 'logs\comparison_report.log'

Write-Host ""
Write-Host "Done. Report: outputs/Evaluation/FB-models-comparison-report.md" -ForegroundColor Green
