param(
    [string]$Python = "D:\anaconda3\envs\pytorch\python.exe",
    [int]$BatchSize = 4,
    [string]$Thresholds = "0.30:0.70:0.02",
    [string]$SelectionMetric = "dice"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$ProjectRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $ProjectRoot

$OutDir = Join-Path "results" "threshold_selection_20260621"
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

$TransUNetWeight = "results/experiments/all_filtered/baseline_retrain_20260619/0619_0232/best_model.pth"
$OursWeight = "results/experiments/all_filtered/ours_green_only_mse_only_20260620/0620_0616/best_model.pth"

foreach ($path in @($TransUNetWeight, $OursWeight)) {
    if (-not (Test-Path $path)) {
        throw "Missing weight: $path"
    }
}

& $Python select_threshold_on_val.py `
    --name "TransUNet_baseline_val_selected" `
    --model_type transunet `
    --weight $TransUNetWeight `
    --dataset all_filtered `
    --batch_size $BatchSize `
    --thresholds $Thresholds `
    --selection_metric $SelectionMetric `
    --out_dir (Join-Path $OutDir "transunet")

& $Python select_threshold_on_val.py `
    --name "Ours_green_mse10_grad0_val_selected" `
    --model_type ours `
    --weight $OursWeight `
    --dataset all_filtered `
    --batch_size $BatchSize `
    --thresholds $Thresholds `
    --selection_metric $SelectionMetric `
    --teacher_mode green_only `
    --enhancer basic `
    --joint_model v1 `
    --loss_weighting fixed `
    --lambda_mse 10.0 `
    --lambda_grad 0.0 `
    --seg_loss bce_dice `
    --out_dir (Join-Path $OutDir "ours_green_mse10_grad0")

Write-Host ""
Write-Host "============================================================"
Write-Host "[THRESHOLD SELECTION DONE]"
Write-Host "Results: $OutDir"
Write-Host "============================================================"