# Run DSL pipeline against the mi300 LLM endpoint.
# Foreground: stdout streams to terminal AND logs/dsl_pipeline_run.log.
#
# Prereq: AutoDSL outputs must exist (run --mode autodsl_operation + autodsl_production first).
#
# Usage:
#   .\scripts\run_dsl_pipeline.ps1                 # all 10 instances (ta71-ta80)
#   .\scripts\run_dsl_pipeline.ps1 ta73 ta74       # selected instances only
#   .\scripts\run_dsl_pipeline.ps1 'instance ta73' # full form also accepted

Set-Location (Join-Path $PSScriptRoot '..')
New-Item -ItemType Directory -Force -Path logs | Out-Null

$env:LLM_BASE_URL   = 'http://localhost:8000/v1'
$env:LLM_MODEL      = 'deepseek-v3'
$env:OPENAI_API_KEY = 'dummy'
if (-not $env:LLM_MAX_WORKERS) { $env:LLM_MAX_WORKERS = '96' }

$pyArgs = @('-u', 'main.py', '--mode', 'dsl_pipeline', '--type', 'CPE_CAE_CSE-2')
if ($args.Count -gt 0) {
  $pyArgs += '--instances'
  $pyArgs += $args
}

& '.\.venv\Scripts\python.exe' @pyArgs 2>&1 | Tee-Object -FilePath 'logs\dsl_pipeline_run.log'
exit $LASTEXITCODE
