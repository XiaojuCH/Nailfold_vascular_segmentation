param(
    [string]$Python = "D:\anaconda3\envs\pytorch\python.exe",
    [string]$Pretrained = "model\vit_checkpoint\imagenet21k\R50+ViT-B_16.npz",
    [string]$TeacherWeight = "results\experiments\all_filtered\direct_green_baseline_20260620\0620_0119\best_model.pth",
    [int]$Seed = 42,
    [int]$Epochs = 50,
    [int]$Patience = 20,
    [int]$BatchSize = 4,
    [double]$Lr = 1e-4,
    [double]$LambdaDecoder = 0.1
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$ProjectRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $ProjectRoot

if (-not (Test-Path $Pretrained)) {
    throw "Missing pretrained weight: $Pretrained"
}
if (-not (Test-Path $TeacherWeight)) {
    throw "Missing decoder teacher weight: $TeacherWeight"
}

$RunRoot = Join-Path "results" "decoder_distill_v2_20260710"
$LogRoot = Join-Path $RunRoot "logs"
$EvalRoot = Join-Path "results" "unified_eval_decoder_distill_v2_20260710"
New-Item -ItemType Directory -Force -Path $RunRoot, $LogRoot, $EvalRoot | Out-Null

$ExpKey = "decoder_distill_v2_direct_green_teacher_pretrained_seed${Seed}_20260710"
$DisplayName = "DecoderDistillV2_direct_green_teacher_pretrained_seed${Seed}"
$ExpName = "all_filtered/$ExpKey"
$TrainLog = Join-Path $LogRoot "$($ExpKey)_train.log"
$EvalLog = Join-Path $LogRoot "$($ExpKey)_eval.log"

function Invoke-LoggedPython {
    param([string]$StepName, [string[]]$Arguments, [string]$LogPath)
    Write-Host ""
    Write-Host "============================================================"
    Write-Host "[START] $StepName"
    Write-Host "[LOG]   $LogPath"
    Write-Host "============================================================"

    $stdoutPath = "$LogPath.stdout"
    $stderrPath = "$LogPath.stderr"
    foreach ($p in @($stdoutPath, $stderrPath, $LogPath)) {
        if (Test-Path $p) { Remove-Item $p -Force }
    }

    $process = Start-Process -FilePath $Python -ArgumentList $Arguments -NoNewWindow -Wait -PassThru -RedirectStandardOutput $stdoutPath -RedirectStandardError $stderrPath

    "===== STDOUT =====" | Set-Content -Encoding UTF8 -Path $LogPath
    if (Test-Path $stdoutPath) { Get-Content $stdoutPath | Add-Content -Encoding UTF8 -Path $LogPath }
    "===== STDERR =====" | Add-Content -Encoding UTF8 -Path $LogPath
    if (Test-Path $stderrPath) { Get-Content $stderrPath | Add-Content -Encoding UTF8 -Path $LogPath }

    if ($process.ExitCode -ne 0) {
        Write-Host "[FAILED] $StepName exit code $($process.ExitCode)"
        if (Test-Path $stderrPath) { Get-Content $stderrPath | Select-Object -Last 100 }
        throw "Step failed: $StepName. See log: $LogPath"
    }
    Write-Host "[DONE] $StepName"
    if (Test-Path $stdoutPath) { Get-Content $stdoutPath | Select-Object -Last 12 }
}

function Get-LatestWeight {
    param([string]$ExperimentName)
    $expRoot = Join-Path "results\experiments" $ExperimentName
    $latest = Get-ChildItem $expRoot -Directory | Sort-Object LastWriteTime -Descending | Select-Object -First 1
    if ($null -eq $latest) { throw "No run directory under $expRoot" }
    $weight = Join-Path $latest.FullName "best_model.pth"
    if (-not (Test-Path $weight)) { throw "Missing weight: $weight" }
    return $weight
}

$TrainArgs = @(
    "train_unified.py",
    "--mode", "ours",
    "--dataset", "all_filtered",
    "--teacher_mode", "green_only",
    "--enhancer", "basic",
    "--enhancer_norm", "bn",
    "--joint_model", "decoder_distill_v2",
    "--loss_weighting", "fixed",
    "--lambda_mse", "10.0",
    "--lambda_grad", "0.0",
    "--lambda_decoder_distill", "$LambdaDecoder",
    "--decoder_distill_layers", "3",
    "--decoder_distill_mode", "cosine_mse",
    "--decoder_teacher_weight", $TeacherWeight,
    "--pretrained", $Pretrained,
    "--intensity_aug", "on",
    "--seg_loss", "bce_dice",
    "--epochs", "$Epochs",
    "--patience", "$Patience",
    "--batch_size", "$BatchSize",
    "--lr", "$Lr",
    "--seed", "$Seed",
    "--exp_name", $ExpName
)
Invoke-LoggedPython -StepName "Train $DisplayName" -Arguments $TrainArgs -LogPath $TrainLog

$Weight = Get-LatestWeight -ExperimentName $ExpName
Write-Host "[WEIGHT] $Weight"

$EvalArgs = @(
    "evaluate_all.py",
    "--name", $DisplayName,
    "--model_type", "ours",
    "--weight", $Weight,
    "--dataset", "all_filtered",
    "--split", "test",
    "--threshold", "0.5",
    "--batch_size", "$BatchSize",
    "--teacher_mode", "green_only",
    "--enhancer", "basic",
    "--enhancer_norm", "bn",
    "--joint_model", "decoder_distill_v2",
    "--loss_weighting", "fixed",
    "--lambda_mse", "10.0",
    "--lambda_grad", "0.0",
    "--lambda_decoder_distill", "$LambdaDecoder",
    "--decoder_distill_layers", "3",
    "--decoder_distill_mode", "cosine_mse",
    "--decoder_teacher_weight", $TeacherWeight,
    "--seg_loss", "bce_dice",
    "--out_dir", $EvalRoot
)
Invoke-LoggedPython -StepName "Evaluate $DisplayName" -Arguments $EvalArgs -LogPath $EvalLog

$Aggregate = Get-ChildItem $EvalRoot -Recurse -Filter "aggregate_results.csv" | Sort-Object LastWriteTime -Descending | Select-Object -First 1
if ($null -eq $Aggregate) {
    throw "No aggregate_results.csv under $EvalRoot"
}

$SummaryPath = Join-Path $RunRoot "metrics_summary.csv"
if (-not (Test-Path $SummaryPath)) {
    "experiment,seed,teacher_weight,lambda_decoder,dice,iou,sensitivity,precision,specificity,accuracy,hd95,cldice,boundary_f1,aggregate_csv,weight" | Set-Content -Encoding UTF8 -Path $SummaryPath
}
$Metrics = Import-Csv $Aggregate.FullName | Select-Object -First 1
$Line = '"{0}",{1},"{2}",{3},{4},{5},{6},{7},{8},{9},{10},{11},{12},"{13}","{14}"' -f $DisplayName, $Seed, $TeacherWeight, $LambdaDecoder, $Metrics.dice, $Metrics.iou, $Metrics.sensitivity, $Metrics.precision, $Metrics.specificity, $Metrics.accuracy, $Metrics.hd95, $Metrics.cldice, $Metrics.boundary_f1, $Aggregate.FullName, $Weight
Add-Content -Encoding UTF8 -Path $SummaryPath -Value $Line

Write-Host ""
Write-Host "============================================================"
Write-Host "[ALL DONE]"
Write-Host "Metrics: $SummaryPath"
Write-Host "Logs:    $LogRoot"
Write-Host "Eval:    $EvalRoot"
Write-Host "============================================================"
