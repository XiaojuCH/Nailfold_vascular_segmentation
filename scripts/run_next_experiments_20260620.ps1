param(
    [string]$Python = "D:\anaconda3\envs\pytorch\python.exe",
    [int]$Epochs = 50,
    [int]$Patience = 20,
    [int]$BatchSize = 4,
    [double]$Lr = 1e-4
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$ProjectRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $ProjectRoot

if (-not (Test-Path $Python)) {
    throw "Python executable not found: $Python"
}

$RunRoot = Join-Path "results" "overnight_20260620"
$LogRoot = Join-Path $RunRoot "logs"
$EvalOutDir = Join-Path "results" "unified_eval_next_20260620"
New-Item -ItemType Directory -Force -Path $RunRoot, $LogRoot, $EvalOutDir | Out-Null

$SummaryPath = Join-Path $RunRoot "run_summary.csv"
$MetricsSummaryPath = Join-Path $RunRoot "metrics_summary.csv"
"experiment,kind,weight,train_log,eval_log" | Set-Content -Encoding UTF8 -Path $SummaryPath
"experiment,kind,dice,iou,sensitivity,precision,specificity,accuracy,hd95,cldice,boundary_f1,aggregate_csv" | Set-Content -Encoding UTF8 -Path $MetricsSummaryPath

function Assert-VariantDataset {
    param(
        [string]$DataDir,
        [int]$TrainCount = 1838,
        [int]$ValCount = 449,
        [int]$TestCount = 436
    )

    $expected = @{
        train = $TrainCount
        val = $ValCount
        test = $TestCount
    }

    foreach ($split in @("train", "val", "test")) {
        foreach ($sub in @("images", "masks")) {
            $path = Join-Path $DataDir "$split\$sub"
            if (-not (Test-Path $path)) {
                throw "Missing directory: $path"
            }
            $count = (Get-ChildItem $path -File | Measure-Object).Count
            if ($count -ne $expected[$split]) {
                throw "Unexpected file count in $path. Expected $($expected[$split]), got $count"
            }
        }
    }
}

function Invoke-LoggedPython {
    param(
        [string]$StepName,
        [string[]]$Arguments,
        [string]$LogPath
    )

    Write-Host ""
    Write-Host "============================================================"
    Write-Host "[START] $StepName"
    Write-Host "[LOG]   $LogPath"
    Write-Host "============================================================"

    $stdoutPath = "$LogPath.stdout"
    $stderrPath = "$LogPath.stderr"
    if (Test-Path $stdoutPath) { Remove-Item $stdoutPath -Force }
    if (Test-Path $stderrPath) { Remove-Item $stderrPath -Force }
    if (Test-Path $LogPath) { Remove-Item $LogPath -Force }

    $process = Start-Process `
        -FilePath $Python `
        -ArgumentList $Arguments `
        -NoNewWindow `
        -Wait `
        -PassThru `
        -RedirectStandardOutput $stdoutPath `
        -RedirectStandardError $stderrPath

    "===== STDOUT =====" | Set-Content -Encoding UTF8 -Path $LogPath
    if (Test-Path $stdoutPath) { Get-Content $stdoutPath | Add-Content -Encoding UTF8 -Path $LogPath }
    "===== STDERR =====" | Add-Content -Encoding UTF8 -Path $LogPath
    if (Test-Path $stderrPath) { Get-Content $stderrPath | Add-Content -Encoding UTF8 -Path $LogPath }

    if ($process.ExitCode -ne 0) {
        Write-Host "[FAILED] $StepName exit code $($process.ExitCode)"
        if (Test-Path $stderrPath) {
            Write-Host "[STDERR tail]"
            Get-Content $stderrPath | Select-Object -Last 40
        }
        throw "Step failed: $StepName. See log: $LogPath"
    }

    Write-Host "[DONE] $StepName"
    if (Test-Path $stdoutPath) {
        Get-Content $stdoutPath | Select-Object -Last 8
    }
}

function Get-LatestWeight {
    param([string]$ExpName)

    $expRoot = Join-Path "results\experiments" $ExpName
    if (-not (Test-Path $expRoot)) {
        throw "Experiment directory not found: $expRoot"
    }

    $latest = Get-ChildItem $expRoot -Directory | Sort-Object LastWriteTime -Descending | Select-Object -First 1
    if ($null -eq $latest) {
        throw "No timestamp directory found under: $expRoot"
    }

    $weight = Join-Path $latest.FullName "best_model.pth"
    if (-not (Test-Path $weight)) {
        throw "best_model.pth not found: $weight"
    }
    return $weight
}

function Get-LatestAggregateCsv {
    param([string]$OutDir)

    $aggregate = Get-ChildItem $OutDir -Recurse -Filter "aggregate_results.csv" |
        Sort-Object LastWriteTime -Descending |
        Select-Object -First 1
    if ($null -eq $aggregate) {
        throw "No aggregate_results.csv found under: $OutDir"
    }
    return $aggregate.FullName
}

Write-Host "[CHECK] Direct-input datasets"
Assert-VariantDataset -DataDir ".\dataset_all_filtered_direct_green"
Assert-VariantDataset -DataDir ".\dataset_all_filtered_direct_clahe"
Assert-VariantDataset -DataDir ".\dataset_all_filtered_direct_green_clahe"
Write-Host "[CHECK] OK"

$experiments = @(
    @{
        Name = "direct_green_baseline_20260620"
        Display = "TransUNet_direct_green"
        Kind = "direct_input"
        ExpName = "all_filtered/direct_green_baseline_20260620"
        TrainArgs = @("train_unified.py", "--mode", "baseline", "--dataset", "all_filtered", "--data_dir", ".\dataset_all_filtered_direct_green", "--epochs", "$Epochs", "--patience", "$Patience", "--batch_size", "$BatchSize", "--lr", "$Lr", "--exp_name", "all_filtered/direct_green_baseline_20260620")
        EvalArgs = @("evaluate_all.py", "--name", "TransUNet_direct_green", "--model_type", "transunet", "--dataset", "all_filtered", "--data_dir", ".\dataset_all_filtered_direct_green", "--split", "test", "--threshold", "0.5", "--batch_size", "$BatchSize", "--out_dir", $EvalOutDir)
    },
    @{
        Name = "direct_clahe_baseline_20260620"
        Display = "TransUNet_direct_clahe"
        Kind = "direct_input"
        ExpName = "all_filtered/direct_clahe_baseline_20260620"
        TrainArgs = @("train_unified.py", "--mode", "baseline", "--dataset", "all_filtered", "--data_dir", ".\dataset_all_filtered_direct_clahe", "--epochs", "$Epochs", "--patience", "$Patience", "--batch_size", "$BatchSize", "--lr", "$Lr", "--exp_name", "all_filtered/direct_clahe_baseline_20260620")
        EvalArgs = @("evaluate_all.py", "--name", "TransUNet_direct_clahe", "--model_type", "transunet", "--dataset", "all_filtered", "--data_dir", ".\dataset_all_filtered_direct_clahe", "--split", "test", "--threshold", "0.5", "--batch_size", "$BatchSize", "--out_dir", $EvalOutDir)
    },
    @{
        Name = "direct_green_clahe_baseline_20260620"
        Display = "TransUNet_direct_green_clahe"
        Kind = "direct_input"
        ExpName = "all_filtered/direct_green_clahe_baseline_20260620"
        TrainArgs = @("train_unified.py", "--mode", "baseline", "--dataset", "all_filtered", "--data_dir", ".\dataset_all_filtered_direct_green_clahe", "--epochs", "$Epochs", "--patience", "$Patience", "--batch_size", "$BatchSize", "--lr", "$Lr", "--exp_name", "all_filtered/direct_green_clahe_baseline_20260620")
        EvalArgs = @("evaluate_all.py", "--name", "TransUNet_direct_green_clahe", "--model_type", "transunet", "--dataset", "all_filtered", "--data_dir", ".\dataset_all_filtered_direct_green_clahe", "--split", "test", "--threshold", "0.5", "--batch_size", "$BatchSize", "--out_dir", $EvalOutDir)
    },
    @{
        Name = "ours_green_only_mse_only_20260620"
        Display = "Ours_green_only_mse_only"
        Kind = "loss_ablation"
        ExpName = "all_filtered/ours_green_only_mse_only_20260620"
        TrainArgs = @("train_unified.py", "--mode", "ours", "--dataset", "all_filtered", "--teacher_mode", "green_only", "--joint_model", "v1", "--enhancer", "basic", "--loss_weighting", "fixed", "--lambda_mse", "10.0", "--lambda_grad", "0.0", "--epochs", "$Epochs", "--patience", "$Patience", "--batch_size", "$BatchSize", "--lr", "$Lr", "--exp_name", "all_filtered/ours_green_only_mse_only_20260620")
        EvalArgs = @("evaluate_all.py", "--name", "Ours_green_only_mse_only", "--model_type", "ours", "--dataset", "all_filtered", "--split", "test", "--threshold", "0.5", "--batch_size", "$BatchSize", "--teacher_mode", "green_only", "--enhancer", "basic", "--joint_model", "v1", "--loss_weighting", "fixed", "--lambda_mse", "10.0", "--lambda_grad", "0.0", "--out_dir", $EvalOutDir)
    },
    @{
        Name = "ours_green_only_mse10_grad20_20260620"
        Display = "Ours_green_only_mse10_grad20"
        Kind = "loss_ablation"
        ExpName = "all_filtered/ours_green_only_mse10_grad20_20260620"
        TrainArgs = @("train_unified.py", "--mode", "ours", "--dataset", "all_filtered", "--teacher_mode", "green_only", "--joint_model", "v1", "--enhancer", "basic", "--loss_weighting", "fixed", "--lambda_mse", "10.0", "--lambda_grad", "20.0", "--epochs", "$Epochs", "--patience", "$Patience", "--batch_size", "$BatchSize", "--lr", "$Lr", "--exp_name", "all_filtered/ours_green_only_mse10_grad20_20260620")
        EvalArgs = @("evaluate_all.py", "--name", "Ours_green_only_mse10_grad20", "--model_type", "ours", "--dataset", "all_filtered", "--split", "test", "--threshold", "0.5", "--batch_size", "$BatchSize", "--teacher_mode", "green_only", "--enhancer", "basic", "--joint_model", "v1", "--loss_weighting", "fixed", "--lambda_mse", "10.0", "--lambda_grad", "20.0", "--out_dir", $EvalOutDir)
    },
    @{
        Name = "ours_green_only_mse10_grad40_20260620"
        Display = "Ours_green_only_mse10_grad40"
        Kind = "loss_ablation"
        ExpName = "all_filtered/ours_green_only_mse10_grad40_20260620"
        TrainArgs = @("train_unified.py", "--mode", "ours", "--dataset", "all_filtered", "--teacher_mode", "green_only", "--joint_model", "v1", "--enhancer", "basic", "--loss_weighting", "fixed", "--lambda_mse", "10.0", "--lambda_grad", "40.0", "--epochs", "$Epochs", "--patience", "$Patience", "--batch_size", "$BatchSize", "--lr", "$Lr", "--exp_name", "all_filtered/ours_green_only_mse10_grad40_20260620")
        EvalArgs = @("evaluate_all.py", "--name", "Ours_green_only_mse10_grad40", "--model_type", "ours", "--dataset", "all_filtered", "--split", "test", "--threshold", "0.5", "--batch_size", "$BatchSize", "--teacher_mode", "green_only", "--enhancer", "basic", "--joint_model", "v1", "--loss_weighting", "fixed", "--lambda_mse", "10.0", "--lambda_grad", "40.0", "--out_dir", $EvalOutDir)
    }
)

foreach ($exp in $experiments) {
    $trainLog = Join-Path $LogRoot "$($exp.Name)_train.log"
    $evalLog = Join-Path $LogRoot "$($exp.Name)_eval.log"

    Invoke-LoggedPython -StepName "Train $($exp.Display)" -Arguments $exp.TrainArgs -LogPath $trainLog

    $weight = Get-LatestWeight -ExpName $exp.ExpName
    Write-Host "[WEIGHT] $weight"

    $evalArgs = @($exp.EvalArgs)
    $evalArgs += @("--weight", $weight)
    Invoke-LoggedPython -StepName "Evaluate $($exp.Display)" -Arguments $evalArgs -LogPath $evalLog

    $line = '"{0}","{1}","{2}","{3}","{4}"' -f $exp.Display, $exp.Kind, $weight, $trainLog, $evalLog
    Add-Content -Encoding UTF8 -Path $SummaryPath -Value $line

    $aggregateCsv = Get-LatestAggregateCsv -OutDir $EvalOutDir
    $metrics = Import-Csv $aggregateCsv | Select-Object -First 1
    $metricLine = '"{0}","{1}",{2},{3},{4},{5},{6},{7},{8},{9},{10},"{11}"' -f $exp.Display, $exp.Kind, $metrics.dice, $metrics.iou, $metrics.sensitivity, $metrics.precision, $metrics.specificity, $metrics.accuracy, $metrics.hd95, $metrics.cldice, $metrics.boundary_f1, $aggregateCsv
    Add-Content -Encoding UTF8 -Path $MetricsSummaryPath -Value $metricLine
}

Write-Host ""
Write-Host "============================================================"
Write-Host "[ALL DONE]"
Write-Host "Summary: $SummaryPath"
Write-Host "Metrics: $MetricsSummaryPath"
Write-Host "Logs:    $LogRoot"
Write-Host "Eval:    $EvalOutDir"
Write-Host "============================================================"
