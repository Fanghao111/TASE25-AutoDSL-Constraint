# Generate NL order descriptions for all 10 instances (ta71-ta80) via LLM.
# Foreground: stdout streams to terminal AND logs/generate_orders_run.log.
#
# Writes preprocess/orders/ta{71..80}/orders.json (canonical). Idempotent —
# skips instances whose orders.json already exists.
#
# Usage:
#   .\scripts\generate_orders.ps1

Set-Location (Join-Path $PSScriptRoot '..')
New-Item -ItemType Directory -Force -Path logs | Out-Null

$env:LLM_BASE_URL   = 'http://localhost:8000/v1'
$env:LLM_MODEL      = 'deepseek-v3'
$env:OPENAI_API_KEY = 'dummy'
if (-not $env:LLM_MAX_WORKERS) { $env:LLM_MAX_WORKERS = '96' }

& '.\.venv\Scripts\python.exe' -u generate_orders.py 2>&1 | Tee-Object -FilePath 'logs\generate_orders_run.log'
exit $LASTEXITCODE
