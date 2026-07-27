$ErrorActionPreference = "Stop"

$RepoRoot = "C:\Mukul K\vinfo1\video-search-engine"
$Python = Join-Path $RepoRoot ".venv\Scripts\python.exe"

$PipelineModule = "tests.td_case2.multicamera_vehicle_tracking_pipeline"
$PipelinePath = "tests\td_case2\multicamera_vehicle_tracking_pipeline"

$OutputDir = Join-Path $RepoRoot "debug_runs\multicamera_vehicle_tracking_pipeline"
$MainReport = Join-Path $OutputDir "three_video_full_test_latest.json"
$EnrichmentReport = Join-Path $OutputDir "three_video_enrichment_verification.json"
$GlobalReport = Join-Path $OutputDir "three_video_global_verification.json"

Set-Location $RepoRoot

Write-Host ""
Write-Host "============================================"
Write-Host " Three-camera full video test"
Write-Host "============================================"
Write-Host ""

# Confirm Python exists
if (-not (Test-Path $Python)) {
    throw "Python virtual environment not found at: $Python"
}

# Confirm videos exist
$Videos = @(
    "$RepoRoot\tests\td_case2\multicamera_vehicle_tracking_pipeline\data\testv\1test_20.mp4",
    "$RepoRoot\tests\td_case2\multicamera_vehicle_tracking_pipeline\data\testv\2test_20.mp4",
    "$RepoRoot\tests\td_case2\multicamera_vehicle_tracking_pipeline\data\testv\3test_20.mp4"
)

foreach ($Video in $Videos) {
    if (-not (Test-Path $Video)) {
        throw "Video not found: $Video"
    }

    Write-Host "FOUND: $Video"
}

# Confirm Supabase credentials are available
if (-not $env:SUPABASE_URL) {
    throw "SUPABASE_URL is missing in this PowerShell session."
}

if (-not $env:SUPABASE_SERVICE_ROLE_KEY) {
    throw "SUPABASE_SERVICE_ROLE_KEY is missing in this PowerShell session."
}

Write-Host ""
Write-Host "Supabase configuration: SET"
Write-Host ""

# Make output directory
New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null

# Remove previous latest report to prevent reading an old run code
if (Test-Path $MainReport) {
    Remove-Item $MainReport -Force
}

Write-Host "STEP 1: Running all three videos fully..."
Write-Host ""

& $Python -m "$PipelineModule.scripts.validate_worker_multicamera_tracking" `
    --camera-config "$PipelinePath\config\cameras.yaml" `
    --detection-config "$PipelinePath\config\detection.yaml" `
    --tracking-config "$PipelinePath\config\tracking.yaml" `
    --worker-config "$PipelinePath\config\workers.yaml" `
    --persistence-config "$PipelinePath\config\persistence.yaml" `
    --evidence-config "$PipelinePath\config\evidence.yaml" `
    --florence-config "$PipelinePath\config\florence.yaml" `
    --vehicle-colour-config "$PipelinePath\config\vehicle_colour.yaml" `
    --anpr-config "$PipelinePath\config\anpr.yaml" `
    --camera-codes CAM_001 CAM_002 CAM_003 `
    --persist-to-supabase `
    --save-sample-frames `
    --sample-frame-limit-per-camera 10 `
    --output-report $MainReport `
    --log-level INFO

if ($LASTEXITCODE -ne 0) {
    throw "Main three-camera pipeline failed with exit code $LASTEXITCODE."
}

if (-not (Test-Path $MainReport)) {
    throw "Main output report was not created: $MainReport"
}

Write-Host ""
Write-Host "STEP 2: Reading generated run code..."
Write-Host ""

$Report = Get-Content $MainReport -Raw | ConvertFrom-Json

$RunCode = $null

$PossibleRunFields = @(
    "run_code",
    "pipeline_run_code",
    "processing_run_code"
)

foreach ($Field in $PossibleRunFields) {
    if ($Report.PSObject.Properties.Name -contains $Field) {
        $Value = $Report.$Field

        if ($Value) {
            $RunCode = [string]$Value
            break
        }
    }
}

# Search nested report data when run_code is not at the top level
if (-not $RunCode) {
    $JsonText = Get-Content $MainReport -Raw

    $Match = [regex]::Match(
        $JsonText,
        'RUN_\d{8}_\d{6}'
    )

    if ($Match.Success) {
        $RunCode = $Match.Value
    }
}

if (-not $RunCode) {
    throw "Could not find the generated RUN code inside $MainReport"
}

Write-Host "Generated run code: $RunCode"
Write-Host ""

Write-Host "STEP 3: Verifying enrichment..."
Write-Host ""

& $Python -m "$PipelineModule.scripts.verify_enrichment_run" `
    --run-code $RunCode `
    --strict `
    --json-output $EnrichmentReport

if ($LASTEXITCODE -ne 0) {
    throw "Enrichment verification failed for $RunCode."
}

Write-Host ""
Write-Host "STEP 4: Building global vehicle objects..."
Write-Host ""

& $Python -m "$PipelineModule.scripts.build_global_vehicle_objects" `
    --run-code $RunCode `
    --global-match-config "$PipelinePath\config\global_matching.yaml" `
    --persist

if ($LASTEXITCODE -ne 0) {
    throw "Global vehicle construction failed for $RunCode."
}

Write-Host ""
Write-Host "STEP 5: Verifying global vehicle objects..."
Write-Host ""

& $Python -m "$PipelineModule.scripts.verify_global_vehicle_objects" `
    --run-code $RunCode `
    --strict `
    --json-output $GlobalReport

if ($LASTEXITCODE -ne 0) {
    throw "Global vehicle verification failed for $RunCode."
}

Write-Host ""
Write-Host "============================================"
Write-Host " Full three-video test completed"
Write-Host "============================================"
Write-Host "Run code: $RunCode"
Write-Host "Main report: $MainReport"
Write-Host "Enrichment report: $EnrichmentReport"
Write-Host "Global report: $GlobalReport"
Write-Host ""
Write-Host "Open this run in the UI:"
Write-Host $RunCode
