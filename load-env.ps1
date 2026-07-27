param(
    [string]$EnvFile = "tests\td_case2\multicamera_vehicle_tracking_pipeline\.env.example"
)

$resolvedEnvFile = Join-Path -Path $PSScriptRoot -ChildPath $EnvFile

if (-not (Test-Path -LiteralPath $resolvedEnvFile)) {
    throw "Environment file not found: $resolvedEnvFile"
}

function Convert-ToSupabaseProjectUrl {
    param(
        [Parameter(Mandatory = $true)]
        [string]$RawValue
    )

    $value = $RawValue.Trim()
    if (-not $value) {
        return ""
    }

    if ($value.StartsWith("https://") -or $value.StartsWith("http://")) {
        return $value.TrimEnd("/")
    }

    try {
        $uri = [System.Uri]$value
    } catch {
        return $value
    }

    $hostName = if ($uri.Host) { $uri.Host.ToLowerInvariant() } else { "" }
    if ($hostName.StartsWith("db.") -and $hostName.EndsWith(".supabase.co")) {
        $projectRef = $hostName.Substring(3, $hostName.Length - 3 - ".supabase.co".Length)
        if ($projectRef) {
            return "https://$projectRef.supabase.co"
        }
    }

    return $value
}

$loaded = @{}

foreach ($line in Get-Content -LiteralPath $resolvedEnvFile) {
    $trimmed = $line.Trim()
    if (-not $trimmed -or $trimmed.StartsWith("#")) {
        continue
    }

    $separatorIndex = $trimmed.IndexOf("=")
    if ($separatorIndex -lt 1) {
        continue
    }

    $name = $trimmed.Substring(0, $separatorIndex).Trim()
    $value = $trimmed.Substring($separatorIndex + 1).Trim()

    if (
        ($value.StartsWith('"') -and $value.EndsWith('"')) -or
        ($value.StartsWith("'") -and $value.EndsWith("'"))
    ) {
        $value = $value.Substring(1, $value.Length - 2)
    }

    Set-Item -Path "Env:$name" -Value $value
    $loaded[$name] = "SET"
}

if (-not $env:SUPABASE_URL -and $env:supabase_database_url) {
    $normalizedSupabaseUrl = Convert-ToSupabaseProjectUrl -RawValue $env:supabase_database_url
    if ($normalizedSupabaseUrl) {
        Set-Item -Path "Env:SUPABASE_URL" -Value $normalizedSupabaseUrl
        $loaded["SUPABASE_URL"] = "SET (derived from supabase_database_url)"
    }
}

Write-Host "Loaded environment values from $resolvedEnvFile"
foreach ($entry in @("SUPABASE_URL", "SUPABASE_SERVICE_ROLE_KEY", "SUPABASE_ANON_KEY", "SUPABASE_EVIDENCE_BUCKET", "DATABASE_SCHEMA_VERSION", "supabase_database_url")) {
    $status = "MISSING"
    if (Test-Path -LiteralPath "Env:$entry") {
        $status = "SET"
    }
    if ($loaded.ContainsKey($entry)) {
        $status = $loaded[$entry]
    }
    Write-Host "$entry=$status"
}
