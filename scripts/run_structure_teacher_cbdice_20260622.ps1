param(
    [string]$Python = "D:\anaconda3\envs\pytorch\python.exe",
    [int]$Epochs = 50,
    [int]$Patience = 20,
    [int]$BatchSize = 4,
    [double]$Lr = 1e-4,
    [int]$Seed = 42,
    [switch]$UsePretrained,
    [string]$Pretrained = "model\vit_checkpoint\imagenet21k\R50+ViT-B_16.npz"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$ProjectRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $ProjectRoot

if ($UsePretrained -and -not (Test-Path $Pretrained)) {
    throw "Missing pretrained weight: $Pretrained"
}

$RunRoot = Join-Path "results" "structure_teacher_cbdice_20260622"
$LogRoot = Join-Path $RunRoot "logs"
$EvalRoot = Join-Path "results" "unified_eval_structure_teacher_cbdice_20260622"
New-Item -ItemType Directory -Force -Path $RunRoot, $LogRoot, $EvalRoot | Out-Null

$SummaryPath = Join-Path $RunRoot "run_summary.csv"
$MetricsSummaryPath = Join-Path $RunRoot "metrics_summary.csv"
"experiment,teacher_mode,seg_loss,seed,pretrained,lambda_mse,lambda_grad,cbdice_weight,boundary_weight,weight,train_log,eval_log" | Set-Content -Encoding UTF8 -Path $SummaryPath
"experiment,teacher_mode,seg_loss,seed,pretrained,lambda_mse,lambda_grad,cbdice_weight,boundary_weight,dice,iou,sensitivity,precision,specificity,accuracy,hd95,cldice,boundary_f1,aggregate_csv" | Set-Content -Encoding UTF8 -Path $MetricsSummaryPath

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
        if (Test-Path $stderrPath) { Get-Content $stderrPath | Select-Object -Last 80 }
        throw "Step failed: $StepName. See log: $LogPath"
    }
    Write-Host "[DONE] $StepName"
    if (Test-Path $stdoutPath) { Get-Content $stdoutPath | Select-Object -Last 12 }
}

function Get-LatestWeight {
    param([string]$ExpName)
    $expRoot = Join-Path "results\experiments" $ExpName
    $latest = Get-ChildItem $expRoot -Directory | Sort-Object LastWriteTime -Descending | Select-Object -First 1
    if ($null -eq $latest) { throw "No run directory under $expRoot" }
    $weight = Join-Path $latest.FullName "best_model.pth"
    if (-not (Test-Path $weight)) { throw "Missing weight: $weight" }
    return $weight
}

function Get-LatestAggregateCsv {
    param([string]$OutDir)
    $aggregate = Get-ChildItem $OutDir -Recurse -Filter "aggregate_results.csv" | Sort-Object LastWriteTime -Descending | Select-Object -First 1
    if ($null -eq $aggregate) { throw "No aggregate_results.csv under $OutDir" }
    return $aggregate.FullName
}

$teacherModes = @("green_blackhat", "green_clahe_blackhat", "green_frangi")
foreach ($mode in $teacherModes) {
    $teacherDir = Join-Path "dataset_all_filtered\train" ("teacher_priors_" + $mode.Replace("+", "_"))
    if (-not (Test-Path $teacherDir)) {
        Invoke-LoggedPython -StepName "Generate teacher $mode" -Arguments @("scripts\generate_teacher.py", "--dataset", "dataset_all_filtered", "--mode", $mode) -LogPath (Join-Path $LogRoot "generate_${mode}.log")
    } else {
        Write-Host "[CHECK] Teacher exists: $teacherDir"
    }
}

$pretrainedArgs = @()
$pretrainedLabel = ""
if ($UsePretrained) {
    $pretrainedArgs = @("--pretrained", $Pretrained)
    $pretrainedLabel = $Pretrained
}

$experiments = @(
    @{Name="ours_green_blackhat_mse10_grad0_cbdice_boundary_20260622"; Display="Ours_green_blackhat_cbdice_boundary"; Teacher="green_blackhat"; SegLoss="bce_dice_cbdice_boundary"; Cb="0.3"; Boundary="0.3"},
    @{Name="ours_green_clahe_blackhat_mse10_grad0_cbdice_boundary_20260622"; Display="Ours_green_clahe_blackhat_cbdice_boundary"; Teacher="green_clahe_blackhat"; SegLoss="bce_dice_cbdice_boundary"; Cb="0.3"; Boundary="0.3"},
    @{Name="ours_green_frangi_mse10_grad0_cbdice_boundary_20260622"; Display="Ours_green_frangi_cbdice_boundary"; Teacher="green_frangi"; SegLoss="bce_dice_cbdice_boundary"; Cb="0.3"; Boundary="0.3"}
)

foreach ($exp in $experiments) {
    $suffix = if ($UsePretrained) { "_pretrained" } else { "" }
    $expName = "all_filtered/$($exp.Name)$suffix"
    $trainLog = Join-Path $LogRoot "$($exp.Name)$suffix`_train.log"
    $evalLog = Join-Path $LogRoot "$($exp.Name)$suffix`_eval.log"
    $expEvalOutDir = Join-Path $EvalRoot "$($exp.Name)$suffix"
    New-Item -ItemType Directory -Force -Path $expEvalOutDir | Out-Null

    $trainArgs = @(
        "train_unified.py",
        "--mode", "ours",
        "--dataset", "all_filtered",
        "--teacher_mode", $exp.Teacher,
        "--joint_model", "v1",
        "--enhancer", "basic",
        "--loss_weighting", "fixed",
        "--lambda_mse", "10.0",
        "--lambda_grad", "0.0",
        "--seg_loss", $exp.SegLoss,
        "--cbdice_weight", $exp.Cb,
        "--boundary_weight", $exp.Boundary,
        "--epochs", "$Epochs",
        "--patience", "$Patience",
        "--batch_size", "$BatchSize",
        "--lr", "$Lr",
        "--seed", "$Seed",
        "--exp_name", $expName
    ) + $pretrainedArgs

    Invoke-LoggedPython -StepName "Train $($exp.Display)$suffix" -Arguments $trainArgs -LogPath $trainLog
    $weight = Get-LatestWeight -ExpName $expName
    Write-Host "[WEIGHT] $weight"

    $evalArgs = @(
        "evaluate_all.py",
        "--name", "$($exp.Display)$suffix",
        "--model_type", "ours",
        "--weight", $weight,
        "--dataset", "all_filtered",
        "--split", "test",
        "--threshold", "0.5",
        "--batch_size", "$BatchSize",
        "--teacher_mode", $exp.Teacher,
        "--enhancer", "basic",
        "--joint_model", "v1",
        "--loss_weighting", "fixed",
        "--lambda_mse", "10.0",
        "--lambda_grad", "0.0",
        "--seg_loss", $exp.SegLoss,
        "--cbdice_weight", $exp.Cb,
        "--boundary_weight", $exp.Boundary,
        "--out_dir", $expEvalOutDir
    )

    Invoke-LoggedPython -StepName "Evaluate $($exp.Display)$suffix" -Arguments $evalArgs -LogPath $evalLog

    $aggregateCsv = Get-LatestAggregateCsv -OutDir $expEvalOutDir
    $metrics = Import-Csv $aggregateCsv | Select-Object -First 1

    $line = '"{0}","{1}","{2}",{3},"{4}",10.0,0.0,"{5}","{6}","{7}","{8}","{9}"' -f "$($exp.Display)$suffix", $exp.Teacher, $exp.SegLoss, $Seed, $pretrainedLabel, $exp.Cb, $exp.Boundary, $weight, $trainLog, $evalLog
    Add-Content -Encoding UTF8 -Path $SummaryPath -Value $line

    $metricLine = '"{0}","{1}","{2}",{3},"{4}",10.0,0.0,"{5}","{6}",{7},{8},{9},{10},{11},{12},{13},{14},{15},"{16}"' -f "$($exp.Display)$suffix", $exp.Teacher, $exp.SegLoss, $Seed, $pretrainedLabel, $exp.Cb, $exp.Boundary, $metrics.dice, $metrics.iou, $metrics.sensitivity, $metrics.precision, $metrics.specificity, $metrics.accuracy, $metrics.hd95, $metrics.cldice, $metrics.boundary_f1, $aggregateCsv
    Add-Content -Encoding UTF8 -Path $MetricsSummaryPath -Value $metricLine
}

Write-Host ""
Write-Host "============================================================"
Write-Host "[ALL DONE]"
Write-Host "Summary: $SummaryPath"
Write-Host "Metrics: $MetricsSummaryPath"
Write-Host "Logs:    $LogRoot"
Write-Host "Eval:    $EvalRoot"
Write-Host "============================================================"
