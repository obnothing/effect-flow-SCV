param(
    [ValidateSet("smoke", "full")]
    [string]$Mode = "full",
    [string]$PythonPath = "python"
)

$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)

$configs = @(
    "configs/light_label/extensions/e1_vcfm.yaml",
    "configs/light_label/extensions/e2_vrop.yaml",
    "configs/light_label/extensions/e3_vasm.yaml",
    "configs/light_label/extensions/e4_pgvr.yaml",
    "configs/light_label/extensions/e5_tdvp.yaml"
)

Write-Host "[extensions] mode=$Mode dataset=DIVE_main6_opcode_process01 seed=42 test_checked=false"
foreach ($config in $configs) {
    Write-Host "[extensions] starting $config"
    if ($Mode -eq "smoke") {
        & $PythonPath src/train_light_label_model.py --config $config --smoke
    } else {
        & $PythonPath src/train_light_label_model.py --config $config
    }
    if ($LASTEXITCODE -ne 0) { throw "training failed: $config" }
}

& $PythonPath scripts/analyze_light_extensions.py --run $Mode
if ($LASTEXITCODE -ne 0) { throw "comparison analysis failed" }

& $PythonPath scripts/diagnose_light_extensions.py --run $Mode
if ($LASTEXITCODE -ne 0) { throw "mechanism diagnostics failed" }

& $PythonPath scripts/analyze_light_extensions.py --run $Mode
if ($LASTEXITCODE -ne 0) { throw "final comparison refresh failed" }

Write-Host "[extensions] complete; test remains locked"
