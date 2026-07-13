# Run per-step `generate_orders.py` in parallel across 4 LLMs served by the
# mi300 LiteLLM gateway. Each model reads its OWN route_sheet_reduce (self-
# consistent per-model pipeline) and writes to its own directory:
#   preprocess/${MODEL}/orders/ta{71..80}/orders.json
#
# generate_orders.py uses ORDERS_OUTPUT_DIR env var. Existing per-model orders
# dirs are wiped before launch so we start from a clean slate.
#
# Usage:
#   .\scripts\run_generate_orders_4llm.ps1                            # all 4 in parallel
#   .\scripts\run_generate_orders_4llm.ps1 qwen3-4b                   # single model
#   .\scripts\run_generate_orders_4llm.ps1 deepseek-v3 qwen3-235b     # subset
#
# Prereq: SSH tunnel forwarding local :8000 to mi300-1's LiteLLM gateway
#         (bash D:/FB/mi300/tunnel.sh).

Set-Location (Join-Path $PSScriptRoot '..')
New-Item -ItemType Directory -Force -Path logs | Out-Null

$allModels = @('deepseek-v3','qwen3-235b','qwen3-30b','qwen3-4b')
if ($args.Count -gt 0) {
  $selected = $args
  $models = @($allModels | Where-Object { $selected -contains $_ })
  if (-not $models) {
    Write-Error "No matching models. Valid names: $($allModels -join ', ')"
    exit 1
  }
} else {
  $models = $allModels
}

# ---- Gateway env (shared) ------------------------------------------------
$env:LLM_BASE_URL   = 'http://localhost:8000/v1'
$env:OPENAI_API_KEY = 'sk-mi300-local'
if (-not $env:LLM_MAX_WORKERS) { $env:LLM_MAX_WORKERS = '64' }

# Localhost tunnel: strip any HTTPS_PROXY that would divert the request off-box.
Remove-Item Env:HTTPS_PROXY  -ErrorAction SilentlyContinue
Remove-Item Env:HTTP_PROXY   -ErrorAction SilentlyContinue
Remove-Item Env:https_proxy  -ErrorAction SilentlyContinue
Remove-Item Env:http_proxy   -ErrorAction SilentlyContinue

# Sanity: each model's route_sheet_reduce must exist.
foreach ($m in $models) {
  $rp = "preprocess/$m/route_sheet_reduce.json"
  if (-not (Test-Path $rp)) {
    Write-Error "$rp not found. Run preprocess for $m first."
    exit 1
  }
}

# Wipe any stale per-model orders dirs before launching.
foreach ($m in $models) {
  $d = "preprocess/$m/orders"
  if (Test-Path $d) {
    Write-Host "  [reset] removing existing $d"
    Remove-Item -Recurse -Force $d
  }
}

$repoRoot   = (Get-Location).Path
$baseUrl    = $env:LLM_BASE_URL
$apiKey     = $env:OPENAI_API_KEY
$maxWorkers = $env:LLM_MAX_WORKERS

$jobs = @()
foreach ($m in $models) {
  Write-Host "==== [$m] generate_orders launched in background -> logs/generate_orders_${m}.log ====" -ForegroundColor Cyan
  $job = Start-Job -Name "orders-$m" -ScriptBlock {
    param($model, $repoRoot, $baseUrl, $apiKey, $maxWorkers)
    Set-Location $repoRoot
    $env:LLM_BASE_URL        = $baseUrl
    $env:OPENAI_API_KEY      = $apiKey
    $env:LLM_MAX_WORKERS     = $maxWorkers
    $env:LLM_MODEL           = $model
    $env:ORDERS_OUTPUT_DIR   = "preprocess/$model/orders"
    Remove-Item Env:HTTPS_PROXY  -ErrorAction SilentlyContinue
    Remove-Item Env:HTTP_PROXY   -ErrorAction SilentlyContinue
    Remove-Item Env:https_proxy  -ErrorAction SilentlyContinue
    Remove-Item Env:http_proxy   -ErrorAction SilentlyContinue
    & '.\.venv\Scripts\python.exe' -u generate_orders.py `
      --reduce ("preprocess/" + $model + "/route_sheet_reduce.json") `
      *> ("logs/generate_orders_" + $model + ".log")
    exit $LASTEXITCODE
  } -ArgumentList $m, $repoRoot, $baseUrl, $apiKey, $maxWorkers
  $jobs += [PSCustomObject]@{ Model = $m; Job = $job }
}

Write-Host ""
Write-Host "==== waiting for $($models.Count) model(s) to finish ====" -ForegroundColor Cyan
Write-Host "     Get-Content -Wait logs\generate_orders_<model>.log   # tail a specific model"
Write-Host ""

$fail = 0
foreach ($entry in $jobs) {
  Wait-Job -Job $entry.Job | Out-Null
  Receive-Job -Job $entry.Job -ErrorAction SilentlyContinue | Out-Null
  $ok = ($entry.Job.State -eq 'Completed') -and (-not $entry.Job.ChildJobs[0].Error)
  if ($ok) {
    Write-Host "  [OK]   $($entry.Model)  ->  preprocess/$($entry.Model)/orders/ta{71..80}/orders.json" -ForegroundColor Green
  } else {
    Write-Host "  [FAIL] $($entry.Model)  (state=$($entry.Job.State); see logs\generate_orders_$($entry.Model).log)" -ForegroundColor Red
    $fail++
  }
  Remove-Job -Job $entry.Job -Force | Out-Null
}

Write-Host ""
if ($fail -gt 0) {
  Write-Error "Done with $fail failure(s)."
  exit 1
}
Write-Host "All $($models.Count) model(s) done." -ForegroundColor Green
