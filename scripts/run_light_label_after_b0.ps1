$ErrorActionPreference = "Stop"

Set-Location (Split-Path -Parent $PSScriptRoot)
$python = "D:\program\miniconda\envs\learnDL310\python.exe"
$logPath = "results\light_label\overnight_pipeline.log"
$b0Metrics = "results\light_label\b0_mean\full\metrics.json"

New-Item -ItemType Directory -Force -Path "results\light_label" | Out-Null

function Log($message) {
    $line = "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] $message"
    Write-Host $line
    Add-Content -LiteralPath $logPath -Value $line
}

function Run-Step($label, $arguments) {
    Log "START $label"
    & $python @arguments 2>&1 | Tee-Object -FilePath $logPath -Append
    if ($LASTEXITCODE -ne 0) {
        Log "FAILED $label exit_code=$LASTEXITCODE"
        exit $LASTEXITCODE
    }
    Log "DONE $label"
}

Log "Overnight process01 pipeline started"
Log "Waiting for existing B0: $b0Metrics"

while ($true) {
    $b0Processes = Get-CimInstance Win32_Process -ErrorAction SilentlyContinue |
        Where-Object { $_.CommandLine -match "train_light_label_model\.py.*b0_mean" }
    if ((Test-Path -LiteralPath $b0Metrics) -and -not $b0Processes) {
        break
    }
    if (-not $b0Processes -and -not (Test-Path -LiteralPath $b0Metrics)) {
        Log "WARNING: B0 process is absent and metrics are not present"
    }
    Start-Sleep -Seconds 60
}

Log "B0 completed"

Run-Step "B1 shared attention" @(
    "src\train_light_label_model.py",
    "--config", "configs\light_label\b1_shared_attention.yaml"
)

Run-Step "B2 label attention" @(
    "src\train_light_label_model.py",
    "--config", "configs\light_label\b2_label_attention.yaml"
)

Run-Step "full comparison" @(
    "scripts\analyze_light_label_model.py",
    "--run", "full"
)

Run-Step "B2 attention diagnostics" @(
    "scripts\analyze_label_attention.py",
    "--config", "configs\light_label\b2_label_attention.yaml"
)

Log "Overnight process01 pipeline completed"
