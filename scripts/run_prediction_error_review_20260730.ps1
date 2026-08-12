param(
    [switch]$ReusePredictions,
    [switch]$SkipAllCasePanels,
    [int]$BatchSize = 4,
    [int]$TopK = 12
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

$Python = "D:\anaconda3\envs\pytorch\python.exe"
$Manifest = "docs\prediction_visualization_manifest_20260730.json"
$OutDir = "results\prediction_error_review_20260730"

if (-not (Test-Path $Python)) {
    throw "Python environment not found: $Python"
}
if (-not (Test-Path $Manifest)) {
    throw "Manifest not found: $Manifest"
}

$Arguments = @(
    "analyze_prediction_errors.py",
    "--manifest", $Manifest,
    "--data_dir", ".\dataset_all_filtered",
    "--split", "test",
    "--out_dir", $OutDir,
    "--threshold", "0.5",
    "--img_size", "256",
    "--batch_size", "$BatchSize",
    "--top_k", "$TopK"
)
if ($ReusePredictions) {
    $Arguments += "--reuse_predictions"
}
if ($SkipAllCasePanels) {
    $Arguments += "--skip_all_case_panels"
}

Write-Host "============================================================"
Write-Host "[START] Full prediction and error review"
Write-Host "[OUT]   $OutDir"
Write-Host "============================================================"

& $Python @Arguments
if ($LASTEXITCODE -ne 0) {
    throw "Prediction error review failed with exit code $LASTEXITCODE"
}

Write-Host ""
Write-Host "============================================================"
Write-Host "[ALL DONE]"
Write-Host "HTML:        $OutDir\index.html"
Write-Host "Summary:     $OutDir\analysis_summary.md"
Write-Host "Predictions: $OutDir\predictions"
Write-Host "All cases:   $OutDir\all_cases"
Write-Host "============================================================"
