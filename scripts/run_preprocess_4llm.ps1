# Run the `preprocess` stage in parallel across 4 LLMs served by the mi300
# LiteLLM gateway. Each model writes to its own subdirectory:
#   preprocess/${MODEL}/route_sheet.json
#   preprocess/${MODEL}/route_sheet_reduce.json
#
# Deterministic mapping() runs once up front (produces preprocess/jssp_mapped.json)
# to avoid a write race between the 4 background model processes.
#
# Usage:
#   .\scripts\run_preprocess_4llm.ps1                             # all 4 models in parallel
#   .\scripts\run_preprocess_4llm.ps1 qwen3-4b                    # single model
#   .\scripts\run_preprocess_4llm.ps1 deepseek-v3 qwen3-235b      # subset
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

# ---- Gateway env (shared across all invocations) ------------------------
$env:LLM_BASE_URL   = 'http://localhost:8000/v1'
$env:OPENAI_API_KEY = 'sk-mi300-local'
if (-not $env:LLM_MAX_WORKERS) { $env:LLM_MAX_WORKERS = '64' }

# Localhost tunnel: strip any HTTPS_PROXY that would divert the request off-box.
Remove-Item Env:HTTPS_PROXY  -ErrorAction SilentlyContinue
Remove-Item Env:HTTP_PROXY   -ErrorAction SilentlyContinue
Remove-Item Env:https_proxy  -ErrorAction SilentlyContinue
Remove-Item Env:http_proxy   -ErrorAction SilentlyContinue

$py = '.\.venv\Scripts\python.exe'

# ---- Step 1: deterministic mapping() — run once, no LLM -----------------
# create_route_sheet reads preprocess/jssp_mapped.json; producing it here once
# keeps the 4 parallel workers from racing on the same file.
if (-not (Test-Path 'preprocess/jssp_mapped.json')) {
  Write-Host "==== [mapping] building preprocess/jssp_mapped.json ====" -ForegroundColor Cyan
  $mappingPy = @'
from src.preprocess.RouteSheet import RouteSheet
rs = RouteSheet(
    machines_data_path="data/machines.json",
    jssp_data_path="data/jssp_data.json",
    arrange_path="data/arrange.json",
    jssp_mapped_path="preprocess/jssp_mapped.json",
    route_sheet_store_path="preprocess/route_sheet.json",
    route_sheet_reduce_path="preprocess/route_sheet_reduce.json",
)
rs.mapping()
print("mapping done")
'@
  $mappingPy | & $py -u -
  if ($LASTEXITCODE -ne 0) {
    Write-Error "mapping() failed (exit $LASTEXITCODE)"
    exit 1
  }
} else {
  Write-Host "==== [mapping] preprocess/jssp_mapped.json exists, skipping ====" -ForegroundColor DarkGray
}

# ---- Step 2: fan out N preprocess processes in parallel -----------------
# Start-Job spawns a background PowerShell that runs the python entry with its
# own env (LLM_MODEL, PREPROCESS_OUTPUT_DIR). All output is captured to
# logs/preprocess_<model>.log via `*>` (merges stdout+stderr+all streams).
$repoRoot   = (Get-Location).Path
$baseUrl    = $env:LLM_BASE_URL
$apiKey     = $env:OPENAI_API_KEY
$maxWorkers = $env:LLM_MAX_WORKERS

$jobs = @()
foreach ($m in $models) {
  Write-Host "==== [$m] preprocess launched in background -> logs/preprocess_${m}.log ====" -ForegroundColor Cyan
  $job = Start-Job -Name "preprocess-$m" -ScriptBlock {
    param($model, $repoRoot, $baseUrl, $apiKey, $maxWorkers)
    Set-Location $repoRoot
    $env:LLM_BASE_URL         = $baseUrl
    $env:OPENAI_API_KEY       = $apiKey
    $env:LLM_MAX_WORKERS      = $maxWorkers
    $env:LLM_MODEL            = $model
    $env:PREPROCESS_OUTPUT_DIR    = "preprocess/$model"
    $env:PREPROCESS_SKIP_MAPPING  = '1'
    Remove-Item Env:HTTPS_PROXY  -ErrorAction SilentlyContinue
    Remove-Item Env:HTTP_PROXY   -ErrorAction SilentlyContinue
    Remove-Item Env:https_proxy  -ErrorAction SilentlyContinue
    Remove-Item Env:http_proxy   -ErrorAction SilentlyContinue
    & '.\.venv\Scripts\python.exe' -u main.py --mode preprocess `
      *> ("logs/preprocess_" + $model + ".log")
    exit $LASTEXITCODE
  } -ArgumentList $m, $repoRoot, $baseUrl, $apiKey, $maxWorkers
  $jobs += [PSCustomObject]@{ Model = $m; Job = $job }
}

Write-Host ""
Write-Host "==== waiting for $($models.Count) model(s) to finish ====" -ForegroundColor Cyan
Write-Host "     Get-Content -Wait logs\preprocess_<model>.log   # tail a specific model"
Write-Host ""

$fail = 0
foreach ($entry in $jobs) {
  Wait-Job -Job $entry.Job | Out-Null
  # Drain any residual output the job printed to its own pipeline (all real
  # output is already in the log file thanks to `*>`).
  Receive-Job -Job $entry.Job -ErrorAction SilentlyContinue | Out-Null
  $ok = ($entry.Job.State -eq 'Completed') -and (-not $entry.Job.ChildJobs[0].Error)
  if ($ok) {
    Write-Host "  [OK]   $($entry.Model)  ->  preprocess/$($entry.Model)/route_sheet.json + preprocess/$($entry.Model)/route_sheet_reduce.json" -ForegroundColor Green
  } else {
    Write-Host "  [FAIL] $($entry.Model)  (state=$($entry.Job.State); see logs\preprocess_$($entry.Model).log)" -ForegroundColor Red
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
