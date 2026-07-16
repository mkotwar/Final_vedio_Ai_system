param(
    [Parameter(Mandatory = $true)]
    [string]$VideoPath,

    [ValidateSet('inherit', 'disabled', 'api_qwen', 'local_qwen')]
    [string]$VlmBackend = 'inherit',

    [switch]$RunVlmPipeline
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$pythonExe = Join-Path $repoRoot '.venv\Scripts\python.exe'

if (-not (Test-Path $pythonExe)) {
    throw "Python executable not found: $pythonExe"
}

$resolvedVideoPath = (Resolve-Path $VideoPath).Path
$env:TD_CASE2_INPUT_VIDEO = $resolvedVideoPath

if ($VlmBackend -ne 'inherit') {
    $env:TD_CASE2_VLM_BACKEND = $VlmBackend
}

if (-not $env:HF_HUB_OFFLINE) {
    $env:HF_HUB_OFFLINE = '1'
}
if (-not $env:TRANSFORMERS_OFFLINE) {
    $env:TRANSFORMERS_OFFLINE = '1'
}

Write-Host "Repository root: $repoRoot"
Write-Host "Video path: $resolvedVideoPath"
Write-Host "Running readiness check..."

& $pythonExe (Join-Path $repoRoot 'tests\td_case2\check_td_case2_readiness.py') --video-path $resolvedVideoPath
if ($LASTEXITCODE -ne 0) {
    throw "td_case2 readiness check failed. Fix the reported issues before running the pipeline."
}

Write-Host "Running search-ready pipeline..."
$searchOutput = & $pythonExe (Join-Path $repoRoot 'tests\td_case2\run_td_case2_search_ready_pipeline.py') 2>&1
$searchOutput | ForEach-Object { $_ }

$runDir = $null
foreach ($line in $searchOutput) {
    $text = [string]$line
    if ($text -match 'TD_CASE2_RUN_DIR=(.+)$') {
        $runDir = $Matches[1].Trim()
    }
}

if (-not $runDir) {
    throw "Could not determine TD_CASE2_RUN_DIR from the search-ready pipeline output."
}

$env:TD_CASE2_RUN_DIR = $runDir
Write-Host "Search-ready run directory: $runDir"

if ($RunVlmPipeline) {
    Write-Host "Running VLM/event pipeline..."
    & $pythonExe (Join-Path $repoRoot 'tests\td_case2\run_td_case2_vlm_event_pipeline.py')
    if ($LASTEXITCODE -ne 0) {
        throw "VLM/event pipeline failed."
    }
}

Write-Host "Final run directory: $runDir"
