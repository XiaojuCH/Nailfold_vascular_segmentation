param(
    [string]$Python = "D:\anaconda3\envs\pytorch\python.exe",
    [string]$Pretrained = "model\vit_checkpoint\imagenet21k\R50+ViT-B_16.npz",
    [int]$Epochs = 50,
    [int]$Patience = 20,
    [int]$BatchSize = 4,
    [double]$Lr = 1e-4,
    [int]$Seed = 42
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$ProjectRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $ProjectRoot

if (-not (Test-Path $Pretrained)) {
    throw "Missing pretrained weight: $Pretrained"
}

$RunRoot = Join-Path "results" "pretrained_p0_20260622"
$LogRoot = Join-Path $RunRoot "logs"
$EvalRoot = Join-Path "results" "unified_eval_pretrained_p0_20260622"
New-Item -ItemType Directory -Force -Path $RunRoot, $LogRoot, $EvalRoot | Out-Null

$SummaryPath = Join-Path $RunRoot "run_summary.csv"
$MetricsSummaryPath = Join-Path $RunRoot "metrics_summary.csv"
"experiment,mode,seg_loss,seed,pretrained,lambda_mse,lambda_grad,cldice_weight,boundary_weight,weight,train_log,eval_log" | Set-Content -Encoding UTF8 -Path $SummaryPath
"experiment,mode,seg_loss,seed,pretrained,lambda_mse,lambda_grad,cldice_weight,boundary_weight,dice,iou,sensitivity,precision,specificity,accuracy,hd95,cldice,boundary_f1,aggregate_csv" | Set-Content -Encoding UTF8 -Path $MetricsSummaryPath

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

$experiments = @(
    @{
        Name="transunet_pretrained_20260622"; Display="TransUNet_pretrained"; Mode="baseline"; ModelType="transunet";
        SegLoss="bce_dice"; LambdaMse=""; LambdaGrad=""; ClDice=""; Boundary="";
        TrainExtra=@(); EvalExtra=@()
    },
    @{
        Name="ours_green_mse10_grad0_pretrained_20260622"; Display="Ours_green_mse10_grad0_pretrained"; Mode="ours"; ModelType="ours";
        SegLoss="bce_dice"; LambdaMse="10.0"; LambdaGrad="0.0"; ClDice=""; Boundary="";
        TrainExtra=@("--teacher_mode","green_only","--joint_model","v1","--enhancer","basic","--loss_weighting","fixed","--lambda_mse","10.0","--lambda_grad","0.0");
        EvalExtra=@("--teacher_mode","green_only","--joint_model","v1","--enhancer","basic","--loss_weighting","fixed","--lambda_mse","10.0","--lambda_grad","0.0")
    },
    @{
        Name="ours_green_mse10_grad0_cldice_boundary_pretrained_20260622"; Display="Ours_green_mse10_grad0_cldice_boundary_pretrained"; Mode="ours"; ModelType="ours";
        SegLoss="bce_dice_cldice_boundary"; LambdaMse="10.0"; LambdaGrad="0.0"; ClDice="0.3"; Boundary="0.3";
        TrainExtra=@("--teacher_mode","green_only","--joint_model","v1","--enhancer","basic","--loss_weighting","fixed","--lambda_mse","10.0","--lambda_grad","0.0","--cldice_weight","0.3","--boundary_weight","0.3");
        EvalExtra=@("--teacher_mode","green_only","--joint_model","v1","--enhancer","basic","--loss_weighting","fixed","--lambda_mse","10.0","--lambda_grad","0.0","--cldice_weight","0.3","--boundary_weight","0.3")
    }
)

foreach ($exp in $experiments) {
    $expName = "all_filtered/$($exp.Name)"
    $trainLog = Join-Path $LogRoot "$($exp.Name)_train.log"
    $evalLog = Join-Path $LogRoot "$($exp.Name)_eval.log"
    $expEvalOutDir = Join-Path $EvalRoot $exp.Name
    New-Item -ItemType Directory -Force -Path $expEvalOutDir | Out-Null

    $trainArgs = @(
        "train_unified.py",
        "--mode", $exp.Mode,
        "--dataset", "all_filtered",
        "--seg_loss", $exp.SegLoss,
        "--epochs", "$Epochs",
        "--patience", "$Patience",
        "--batch_size", "$BatchSize",
        "--lr", "$Lr",
        "--seed", "$Seed",
        "--pretrained", $Pretrained,
        "--exp_name", $expName
    ) + $exp.TrainExtra

    Invoke-LoggedPython -StepName "Train $($exp.Display)" -Arguments $trainArgs -LogPath $trainLog
    $weight = Get-LatestWeight -ExpName $expName
    Write-Host "[WEIGHT] $weight"

    $evalArgs = @(
        "evaluate_all.py",
        "--name", $exp.Display,
        "--model_type", $exp.ModelType,
        "--weight", $weight,
        "--dataset", "all_filtered",
        "--split", "test",
        "--threshold", "0.5",
        "--batch_size", "$BatchSize",
        "--seg_loss", $exp.SegLoss,
        "--out_dir", $expEvalOutDir
    ) + $exp.EvalExtra

    Invoke-LoggedPython -StepName "Evaluate $($exp.Display)" -Arguments $evalArgs -LogPath $evalLog

    $aggregateCsv = Get-LatestAggregateCsv -OutDir $expEvalOutDir
    $metrics = Import-Csv $aggregateCsv | Select-Object -First 1

    $line = '"{0}","{1}","{2}",{3},"{4}","{5}","{6}","{7}","{8}","{9}","{10}","{11}"' -f $exp.Display, $exp.Mode, $exp.SegLoss, $Seed, $Pretrained, $exp.LambdaMse, $exp.LambdaGrad, $exp.ClDice, $exp.Boundary, $weight, $trainLog, $evalLog
    Add-Content -Encoding UTF8 -Path $SummaryPath -Value $line

    $metricLine = '"{0}","{1}","{2}",{3},"{4}","{5}","{6}","{7}","{8}",{9},{10},{11},{12},{13},{14},{15},{16},{17},"{18}"' -f $exp.Display, $exp.Mode, $exp.SegLoss, $Seed, $Pretrained, $exp.LambdaMse, $exp.LambdaGrad, $exp.ClDice, $exp.Boundary, $metrics.dice, $metrics.iou, $metrics.sensitivity, $metrics.precision, $metrics.specificity, $metrics.accuracy, $metrics.hd95, $metrics.cldice, $metrics.boundary_f1, $aggregateCsv
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