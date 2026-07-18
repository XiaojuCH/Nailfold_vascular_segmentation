param(
    [string]$Python = "D:\anaconda3\envs\pytorch\python.exe",
    [int]$Seed = 42,
    [int]$Epochs = 50,
    [int]$Patience = 20,
    [int]$BatchSize = 4,
    [double]$Lr = 1e-4,
    [switch]$SkipExisting
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$ProjectRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $ProjectRoot

if (-not (Test-Path -LiteralPath $Python)) {
    throw "Python executable not found: $Python"
}

# Model selection stays on validation only. Do not add --evaluate_test_after_training here.
$RunRoot = Join-Path "results" "corrected_baselines_20260714"
$LogRoot = Join-Path $RunRoot "logs"
$SummaryPath = Join-Path $RunRoot "run_summary.csv"
New-Item -ItemType Directory -Force -Path $RunRoot, $LogRoot | Out-Null

if (-not (Test-Path -LiteralPath $SummaryPath)) {
    "experiment,seed,mode,teacher_mode,intensity_aug,seg_loss,epochs,patience,batch_size,lr,experiment_root,external_log" |
        Set-Content -Encoding UTF8 -Path $SummaryPath
}

function Invoke-Training {
    param(
        [string]$Name,
        [string]$ExperimentName,
        [string]$Mode,
        [string]$TeacherMode,
        [string[]]$ExtraArguments
    )

    $ExperimentRoot = Join-Path "results\experiments" $ExperimentName
    $ExistingWeight = Get-ChildItem -LiteralPath $ExperimentRoot -Directory -ErrorAction SilentlyContinue |
        Sort-Object LastWriteTime -Descending |
        ForEach-Object { Join-Path $_.FullName "best_model.pth" } |
        Where-Object { Test-Path -LiteralPath $_ } |
        Select-Object -First 1

    if ($SkipExisting -and $null -ne $ExistingWeight) {
        Write-Host "[SKIP] $Name already has a best weight: $ExistingWeight"
        return
    }

    $LogPath = Join-Path $LogRoot "$Name`_train.log"
    $StdoutPath = "$LogPath.stdout"
    $StderrPath = "$LogPath.stderr"
    foreach ($Path in @($LogPath, $StdoutPath, $StderrPath)) {
        if (Test-Path -LiteralPath $Path) {
            Remove-Item -LiteralPath $Path -Force
        }
    }

    $Arguments = @(
        "train_unified.py",
        "--mode", $Mode,
        "--dataset", "all_filtered",
        "--seg_loss", "bce_dice",
        "--epochs", "$Epochs",
        "--patience", "$Patience",
        "--batch_size", "$BatchSize",
        "--lr", "$Lr",
        "--seed", "$Seed",
        "--exp_name", $ExperimentName
    ) + $ExtraArguments

    Write-Host ""
    Write-Host "============================================================"
    Write-Host "[START] $Name"
    Write-Host "[LOG]   $LogPath"
    Write-Host "[NOTE]  Validation-only selection; no test evaluation is run."
    Write-Host "============================================================"

    $Process = Start-Process -FilePath $Python -ArgumentList $Arguments -NoNewWindow -Wait -PassThru `
        -RedirectStandardOutput $StdoutPath -RedirectStandardError $StderrPath

    "===== STDOUT =====" | Set-Content -Encoding UTF8 -Path $LogPath
    if (Test-Path -LiteralPath $StdoutPath) {
        Get-Content -LiteralPath $StdoutPath | Add-Content -Encoding UTF8 -Path $LogPath
    }
    "===== STDERR =====" | Add-Content -Encoding UTF8 -Path $LogPath
    if (Test-Path -LiteralPath $StderrPath) {
        Get-Content -LiteralPath $StderrPath | Add-Content -Encoding UTF8 -Path $LogPath
    }

    if ($Process.ExitCode -ne 0) {
        Write-Host "[FAILED] $Name (exit code $($Process.ExitCode))"
        if (Test-Path -LiteralPath $StderrPath) {
            Get-Content -LiteralPath $StderrPath | Select-Object -Last 80
        }
        throw "Training failed. Inspect: $LogPath"
    }

    $LatestRun = Get-ChildItem -LiteralPath $ExperimentRoot -Directory |
        Sort-Object LastWriteTime -Descending |
        Select-Object -First 1
    if ($null -eq $LatestRun -or -not (Test-Path -LiteralPath (Join-Path $LatestRun.FullName "best_model.pth"))) {
        throw "Training finished but best_model.pth was not found under: $ExperimentRoot"
    }

    $TrainingLog = Join-Path $LatestRun.FullName "training_log.txt"
    $SummaryLine = '"{0}",{1},"{2}","{3}","on","bce_dice",{4},{5},{6},{7},"{8}","{9}"' -f `
        $Name, $Seed, $Mode, $TeacherMode, $Epochs, $Patience, $BatchSize, $Lr, $LatestRun.FullName, $LogPath
    Add-Content -Encoding UTF8 -Path $SummaryPath -Value $SummaryLine

    Write-Host "[DONE] $Name"
    Write-Host "[WEIGHT] $(Join-Path $LatestRun.FullName 'best_model.pth')"
    Write-Host "[TRAIN LOG] $TrainingLog"
    Get-Content -LiteralPath $LogPath | Select-Object -Last 18
}

Invoke-Training `
    -Name "r0_transunet_corrected_scratch_seed${Seed}_20260714" `
    -ExperimentName "all_filtered/r0_transunet_corrected_scratch_seed${Seed}_20260714" `
    -Mode "baseline" `
    -TeacherMode "" `
    -ExtraArguments @("--intensity_aug", "on")

Invoke-Training `
    -Name "r1_green_mse_corrected_scratch_seed${Seed}_20260714" `
    -ExperimentName "all_filtered/r1_green_mse_corrected_scratch_seed${Seed}_20260714" `
    -Mode "ours" `
    -TeacherMode "green_only" `
    -ExtraArguments @(
        "--teacher_mode", "green_only",
        "--joint_model", "v1",
        "--enhancer", "basic",
        "--enhancer_norm", "bn",
        "--loss_weighting", "fixed",
        "--lambda_mse", "10.0",
        "--lambda_grad", "0.0",
        "--intensity_aug", "on"
    )

Write-Host ""
Write-Host "============================================================"
Write-Host "[ALL DONE]"
Write-Host "Summary: $SummaryPath"
Write-Host "Logs:    $LogRoot"
Write-Host "Next: compare best validation metrics in each run's training_log.txt."
Write-Host "============================================================"
