$ErrorActionPreference = "Stop"

Set-Location (Split-Path -Parent $PSScriptRoot)
$python = "D:\program\miniconda\envs\learnDL310\python.exe"
$c1Metrics = "results\light_label\prototype\c1_b2_pairwise\full\metrics.json"
$logPath = "results\light_label\prototype\c1_to_c2.log"

New-Item -ItemType Directory -Force -Path "results\light_label\prototype" | Out-Null

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

Log "Waiting for current C1 training to finish"
while ($true) {
    $c1Processes = Get-CimInstance Win32_Process -ErrorAction SilentlyContinue |
        Where-Object { $_.CommandLine -match "train_label_decoupled_proto\.py.*c1_b2_pairwise" }
    if ((Test-Path -LiteralPath $c1Metrics) -and -not $c1Processes) {
        break
    }
    Start-Sleep -Seconds 60
}

Log "C1 completed and metrics.json is present"

Run-Step "C2 label-decoupled prototype" @(
    "scripts\train_label_decoupled_proto.py",
    "--config", "configs\light_label\c2_b2_prototype.yaml"
)

Run-Step "C0 C1 C2 report" @(
    "scripts\analyze_label_decoupled_proto.py",
    "--config", "configs\light_label\c2_b2_prototype.yaml"
)

Log "C1 to C2 pipeline completed"
