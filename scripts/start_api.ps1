param(
    [string]$ApiHost = "127.0.0.1",
    [int]$Port = 8000,
    [switch]$Reload
)

$repoRoot = Split-Path -Parent $PSScriptRoot
$pythonExe = Join-Path $repoRoot ".venv\Scripts\python.exe"
$loadEnv = Join-Path $repoRoot "load-env.ps1"

if (-not (Test-Path -LiteralPath $pythonExe)) {
    throw "Python executable not found at $pythonExe"
}

if (-not (Test-Path -LiteralPath $loadEnv)) {
    throw "Environment loader not found at $loadEnv"
}

$listener = Get-NetTCPConnection -LocalPort $Port -ErrorAction SilentlyContinue | Select-Object -First 1
if ($null -ne $listener) {
    $owner = Get-Process -Id $listener.OwningProcess -ErrorAction SilentlyContinue | Select-Object -First 1
    $ownerName = if ($null -ne $owner) { $owner.ProcessName } else { "unknown" }
    throw "Port $Port is already in use by PID $($listener.OwningProcess) ($ownerName). Free the port before starting the API."
}

. $loadEnv

$arguments = @(
    "-m",
    "uvicorn",
    "tests.td_case2.multicamera_vehicle_tracking_pipeline.api.main:app",
    "--host",
    $ApiHost,
    "--port",
    [string]$Port
)

if ($Reload) {
    $arguments += "--reload"
}

Write-Host "Starting multicamera API on http://$ApiHost`:$Port"
Write-Host "Health URL: http://$ApiHost`:$Port/api/v1/health"
& $pythonExe @arguments
if ($LASTEXITCODE -ne 0) {
    throw "FastAPI startup failed with exit code $LASTEXITCODE"
}
