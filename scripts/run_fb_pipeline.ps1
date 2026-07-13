# Run FB-2s pipeline against the mi300 LLM endpoint.
# Foreground: stdout streams to terminal AND logs/fb_pipeline_run.log.
#
# Usage:
#   .\scripts\run_fb_pipeline.ps1                 # all 10 instances (ta71-ta80)
#   .\scripts\run_fb_pipeline.ps1 ta73 ta74       # selected instances only
#   .\scripts\run_fb_pipeline.ps1 'instance ta73' # full form also accepted

Set-Location (Join-Path $PSScriptRoot '..')
New-Item -ItemType Directory -Force -Path logs | Out-Null

$env:LLM_BASE_URL   = 'http://localhost:8000/v1'
$env:LLM_MODEL      = 'deepseek-v3'
$env:OPENAI_API_KEY = 'dummy'
if (-not $env:LLM_MAX_WORKERS) { $env:LLM_MAX_WORKERS = '64' }

$pyArgs = @('-u', 'main.py', '--mode', 'fb_pipeline', '--type', 'full')
if ($args.Count -gt 0) {
  $pyArgs += '--instances'
  $pyArgs += $args
}

& '.\.venv\Scripts\python.exe' @pyArgs 2>&1 | Tee-Object -FilePath 'logs\fb_pipeline_run.log'
exit $LASTEXITCODE
