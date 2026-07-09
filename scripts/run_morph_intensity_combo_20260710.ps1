param(
    [string]$Python = "D:\anaconda3\envs\pytorch\python.exe",
    [string]$Pretrained = "model\vit_checkpoint\imagenet21k\R50+ViT-B_16.npz",
    [ValidateSet("pretrained", "scratch", "both")]
    [string]$PretrainMode = "both",
    [int]$Epochs = 50,
    [int]$Patience = 20,
    [int]$BatchSize = 4,
    [double]$Lr = 1e-4,
    [int]$Seed = 42,
    [switch]$IncludeStructureLoss,
    [switch]$RunThresholdSelection,
    [string]$Thresholds = "0.35:0.65:0.02",
    [ValidateSet("dice", "iou", "cldice", "boundary_f1", "structure_combo")]
    [string]$SelectionMetric = "dice",
    [switch]$SkipExisting
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$ProjectRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $ProjectRoot

if ($PretrainMode -ne "scratch" -and -not (Test-Path $Pretrained)) {
    throw "Missing pretrained weight: $Pretrained"
}

$RunRoot = Join-Path "results" "morph_intensity_combo_20260710"
$LogRoot = Join-Path $RunRoot "logs"
$EvalRoot = Join-Path "results" "unified_eval_morph_intensity_combo_20260710"
$ThresholdRoot = Join-Path "results" "threshold_selection_morph_intensity_combo_20260710"
New-Item -ItemType Directory -Force -Path $RunRoot, $LogRoot, $EvalRoot, $ThresholdRoot | Out-Null

$SummaryPath = Join-Path $RunRoot "run_summary.csv"
$MetricsSummaryPath = Join-Path $RunRoot "metrics_summary.csv"
$ThresholdSummaryPath = Join-Path $RunRoot "threshold_summary.csv"

if (-not $SkipExisting -or -not (Test-Path $SummaryPath)) {
    "experiment,innovation,pretrain_mode,seed,pretrained,enhancer,enhancer_norm,joint_model,intensity_aug,lambda_mse,lambda_grad,seg_loss,cldice_weight,boundary_weight,batch_size,weight,train_log,eval_log" | Set-Content -Encoding UTF8 -Path $SummaryPath
}
if (-not $SkipExisting -or -not (Test-Path $MetricsSummaryPath)) {
    "experiment,innovation,pretrain_mode,seed,enhancer,enhancer_norm,joint_model,intensity_aug,lambda_mse,lambda_grad,seg_loss,cldice_weight,boundary_weight,dice,iou,sensitivity,precision,specificity,accuracy,hd95,cldice,boundary_f1,decision,aggregate_csv" | Set-Content -Encoding UTF8 -Path $MetricsSummaryPath
}
if ($RunThresholdSelection -and (-not $SkipExisting -or -not (Test-Path $ThresholdSummaryPath))) {
    "experiment,pretrain_mode,selection_metric,selected_threshold,val_selection_score,val_dice_at_selected_threshold,dice,iou,sensitivity,precision,specificity,accuracy,hd95,cldice,boundary_f1,threshold_csv" | Set-Content -Encoding UTF8 -Path $ThresholdSummaryPath
}

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
    param([string]$ExpName)
    $expRoot = Join-Path "results\experiments" $ExpName
    $latest = Get-ChildItem $expRoot -Directory | Sort-Object LastWriteTime -Descending | Select-Object -First 1
    if ($null -eq $latest) { throw "No run directory under $expRoot" }
    $weight = Join-Path $latest.FullName "best_model.pth"
    if (-not (Test-Path $weight)) { throw "Missing weight: $weight" }
    return $weight
}

function Get-LatestCsv {
    param([string]$OutDir, [string]$Name)
    $csv = Get-ChildItem $OutDir -Recurse -Filter $Name | Sort-Object LastWriteTime -Descending | Select-Object -First 1
    if ($null -eq $csv) { throw "No $Name under $OutDir" }
    return $csv.FullName
}

function Get-Decision {
    param($Metrics)
    $dice = [double]$Metrics.dice
    $hd95 = [double]$Metrics.hd95
    $cldice = [double]$Metrics.cldice
    $boundary = [double]$Metrics.boundary_f1
    $precision = [double]$Metrics.precision

    if ($dice -ge 0.7625 -and $hd95 -le 22.69) { return "strong_positive" }
    if ($dice -ge 0.7600) { return "continue" }
    if ($dice -ge 0.7583) { return "beat_current_best_dice" }
    if ($dice -ge 0.7567 -and ($cldice -ge 0.8530 -or $boundary -ge 0.6510)) { return "structure_continue" }
    if ($dice -lt 0.7567 -or $precision -lt 0.70) { return "pause" }
    return "watch"
}

function Add-ContentWithRetry {
    param([string]$Path, [string]$Value, [int]$Retries = 12, [int]$DelaySeconds = 5)
    for ($attempt = 1; $attempt -le $Retries; $attempt++) {
        try {
            Add-Content -Encoding UTF8 -Path $Path -Value $Value
            return
        }
        catch [System.IO.IOException] {
            if ($attempt -eq $Retries) {
                $fallbackPath = "$Path.pending"
                Write-Host "[WARN] Could not write $Path because it is locked. Writing fallback line to $fallbackPath"
                Add-Content -Encoding UTF8 -Path $fallbackPath -Value $Value
                return
            }
            Write-Host "[WARN] $Path is locked. Close Excel/WPS if it is open. Retry $attempt/$Retries in $DelaySeconds seconds..."
            Start-Sleep -Seconds $DelaySeconds
        }
    }
}

function Test-CsvContainsExperiment {
    param([string]$Path, [string]$Experiment)
    if (-not (Test-Path $Path)) { return $false }
    try {
        $matches = @(Import-Csv $Path | Where-Object { $_.experiment -eq $Experiment })
        return $matches.Count -gt 0
    }
    catch {
        return $false
    }
}

$experiments = @(
    @{
        Name = "anisotropic_no_intensity"
        Display = "C1_anisotropic_no_intensity"
        Innovation = "morphology_plus_intensity_prior"
        Enhancer = "anisotropic"
        EnhancerNorm = "bn"
        JointModel = "v1"
        IntensityAug = "off"
        SegLoss = "bce_dice"
        ClDiceWeight = "0.5"
        BoundaryWeight = "0.5"
    },
    @{
        Name = "no_bn_no_intensity"
        Display = "C2_no_bn_no_intensity"
        Innovation = "intensity_prior_preservation"
        Enhancer = "basic"
        EnhancerNorm = "none"
        JointModel = "v1"
        IntensityAug = "off"
        SegLoss = "bce_dice"
        ClDiceWeight = "0.5"
        BoundaryWeight = "0.5"
    },
    @{
        Name = "anisotropic_no_bn_no_intensity"
        Display = "C3_anisotropic_no_bn_no_intensity"
        Innovation = "morphology_plus_norm_aug_preservation"
        Enhancer = "anisotropic"
        EnhancerNorm = "none"
        JointModel = "v1"
        IntensityAug = "off"
        SegLoss = "bce_dice"
        ClDiceWeight = "0.5"
        BoundaryWeight = "0.5"
    }
)

if ($IncludeStructureLoss) {
    $experiments += @{
        Name = "anisotropic_no_intensity_cldice_boundary"
        Display = "C4_anisotropic_no_intensity_cldice_boundary"
        Innovation = "morphology_intensity_structure_loss"
        Enhancer = "anisotropic"
        EnhancerNorm = "bn"
        JointModel = "v1"
        IntensityAug = "off"
        SegLoss = "bce_dice_cldice_boundary"
        ClDiceWeight = "0.3"
        BoundaryWeight = "0.3"
    }
}

$pretrainSettings = @()
if ($PretrainMode -eq "pretrained" -or $PretrainMode -eq "both") {
    $pretrainSettings += @{
        Tag = "pretrained"
        PretrainedValue = $Pretrained
        TrainExtra = @("--pretrained", $Pretrained)
    }
}
if ($PretrainMode -eq "scratch" -or $PretrainMode -eq "both") {
    $pretrainSettings += @{
        Tag = "scratch"
        PretrainedValue = ""
        TrainExtra = @()
    }
}

foreach ($pretrain in $pretrainSettings) {
    foreach ($exp in $experiments) {
        $runName = "$($exp.Name)_$($pretrain.Tag)_20260710"
        $displayName = "$($exp.Display)_$($pretrain.Tag)"
        $expName = "all_filtered/$runName"
        $trainLog = Join-Path $LogRoot "$($runName)_train.log"
        $evalLog = Join-Path $LogRoot "$($runName)_eval.log"
        $thresholdLog = Join-Path $LogRoot "$($runName)_threshold.log"
        $expEvalOutDir = Join-Path $EvalRoot $runName
        $expThresholdOutDir = Join-Path $ThresholdRoot $runName
        New-Item -ItemType Directory -Force -Path $expEvalOutDir, $expThresholdOutDir | Out-Null

        $existingAggregate = Get-ChildItem $expEvalOutDir -Recurse -Filter "aggregate_results.csv" -ErrorAction SilentlyContinue | Sort-Object LastWriteTime -Descending | Select-Object -First 1
        if ($SkipExisting -and $null -ne $existingAggregate) {
            Write-Host "[SKIP TRAIN/EVAL] $displayName already has aggregate: $($existingAggregate.FullName)"
            $aggregateCsv = $existingAggregate.FullName
            $metrics = Import-Csv $aggregateCsv | Select-Object -First 1
            $weight = $metrics.weight
        }
        else {
            $trainArgs = @(
                "train_unified.py",
                "--mode", "ours",
                "--dataset", "all_filtered",
                "--teacher_mode", "green_only",
                "--loss_weighting", "fixed",
                "--lambda_mse", "10.0",
                "--lambda_grad", "0.0",
                "--seg_loss", $exp.SegLoss,
                "--cldice_weight", $exp.ClDiceWeight,
                "--boundary_weight", $exp.BoundaryWeight,
                "--epochs", "$Epochs",
                "--patience", "$Patience",
                "--batch_size", "$BatchSize",
                "--lr", "$Lr",
                "--seed", "$Seed",
                "--exp_name", $expName,
                "--enhancer", $exp.Enhancer,
                "--enhancer_norm", $exp.EnhancerNorm,
                "--joint_model", $exp.JointModel,
                "--intensity_aug", $exp.IntensityAug
            ) + $pretrain.TrainExtra

            Invoke-LoggedPython -StepName "Train $displayName" -Arguments $trainArgs -LogPath $trainLog
            $weight = Get-LatestWeight -ExpName $expName
            Write-Host "[WEIGHT] $weight"

            $evalArgs = @(
                "evaluate_all.py",
                "--name", $displayName,
                "--model_type", "ours",
                "--weight", $weight,
                "--dataset", "all_filtered",
                "--split", "test",
                "--threshold", "0.5",
                "--batch_size", "$BatchSize",
                "--teacher_mode", "green_only",
                "--enhancer", $exp.Enhancer,
                "--enhancer_norm", $exp.EnhancerNorm,
                "--joint_model", $exp.JointModel,
                "--loss_weighting", "fixed",
                "--lambda_mse", "10.0",
                "--lambda_grad", "0.0",
                "--seg_loss", $exp.SegLoss,
                "--cldice_weight", $exp.ClDiceWeight,
                "--boundary_weight", $exp.BoundaryWeight,
                "--out_dir", $expEvalOutDir
            )

            Invoke-LoggedPython -StepName "Evaluate $displayName" -Arguments $evalArgs -LogPath $evalLog
            $aggregateCsv = Get-LatestCsv -OutDir $expEvalOutDir -Name "aggregate_results.csv"
            $metrics = Import-Csv $aggregateCsv | Select-Object -First 1
        }

        $decision = Get-Decision -Metrics $metrics
        if (-not (Test-CsvContainsExperiment -Path $SummaryPath -Experiment $displayName)) {
            $runLine = '"{0}","{1}","{2}",{3},"{4}","{5}","{6}","{7}","{8}",10.0,0.0,"{9}","{10}","{11}",{12},"{13}","{14}","{15}"' -f $displayName, $exp.Innovation, $pretrain.Tag, $Seed, $pretrain.PretrainedValue, $exp.Enhancer, $exp.EnhancerNorm, $exp.JointModel, $exp.IntensityAug, $exp.SegLoss, $exp.ClDiceWeight, $exp.BoundaryWeight, $BatchSize, $weight, $trainLog, $evalLog
            Add-ContentWithRetry -Path $SummaryPath -Value $runLine
        }

        if (-not (Test-CsvContainsExperiment -Path $MetricsSummaryPath -Experiment $displayName)) {
            $metricLine = '"{0}","{1}","{2}",{3},"{4}","{5}","{6}","{7}",10.0,0.0,"{8}","{9}","{10}",{11},{12},{13},{14},{15},{16},{17},{18},{19},"{20}","{21}"' -f $displayName, $exp.Innovation, $pretrain.Tag, $Seed, $exp.Enhancer, $exp.EnhancerNorm, $exp.JointModel, $exp.IntensityAug, $exp.SegLoss, $exp.ClDiceWeight, $exp.BoundaryWeight, $metrics.dice, $metrics.iou, $metrics.sensitivity, $metrics.precision, $metrics.specificity, $metrics.accuracy, $metrics.hd95, $metrics.cldice, $metrics.boundary_f1, $decision, $aggregateCsv
            Add-ContentWithRetry -Path $MetricsSummaryPath -Value $metricLine
        }

        if ($RunThresholdSelection) {
            $existingThresholdCsv = Get-ChildItem $expThresholdOutDir -Recurse -Filter "selected_threshold_test_results.csv" -ErrorAction SilentlyContinue | Sort-Object LastWriteTime -Descending | Select-Object -First 1
            if ($SkipExisting -and $null -ne $existingThresholdCsv) {
                Write-Host "[SKIP THRESHOLD] $displayName already has threshold result: $($existingThresholdCsv.FullName)"
                $thresholdCsv = $existingThresholdCsv.FullName
            }
            else {
                $thresholdArgs = @(
                    "select_threshold_on_val.py",
                    "--name", $displayName,
                    "--model_type", "ours",
                    "--weight", $weight,
                    "--dataset", "all_filtered",
                    "--batch_size", "$BatchSize",
                    "--thresholds", $Thresholds,
                    "--selection_metric", $SelectionMetric,
                    "--teacher_mode", "green_only",
                    "--enhancer", $exp.Enhancer,
                    "--enhancer_norm", $exp.EnhancerNorm,
                    "--joint_model", $exp.JointModel,
                    "--loss_weighting", "fixed",
                    "--lambda_mse", "10.0",
                    "--lambda_grad", "0.0",
                    "--seg_loss", $exp.SegLoss,
                    "--cldice_weight", $exp.ClDiceWeight,
                    "--boundary_weight", $exp.BoundaryWeight,
                    "--out_dir", $expThresholdOutDir
                )
                Invoke-LoggedPython -StepName "Threshold select $displayName" -Arguments $thresholdArgs -LogPath $thresholdLog
                $thresholdCsv = Get-LatestCsv -OutDir $expThresholdOutDir -Name "selected_threshold_test_results.csv"
            }

            $thresholdMetrics = Import-Csv $thresholdCsv | Select-Object -First 1
            if (-not (Test-CsvContainsExperiment -Path $ThresholdSummaryPath -Experiment $displayName)) {
                $thresholdLine = '"{0}","{1}","{2}",{3},{4},{5},{6},{7},{8},{9},{10},{11},{12},{13},{14},"{15}"' -f $displayName, $pretrain.Tag, $SelectionMetric, $thresholdMetrics.selected_threshold, $thresholdMetrics.val_selection_score, $thresholdMetrics.val_dice_at_selected_threshold, $thresholdMetrics.dice, $thresholdMetrics.iou, $thresholdMetrics.sensitivity, $thresholdMetrics.precision, $thresholdMetrics.specificity, $thresholdMetrics.accuracy, $thresholdMetrics.hd95, $thresholdMetrics.cldice, $thresholdMetrics.boundary_f1, $thresholdCsv
                Add-ContentWithRetry -Path $ThresholdSummaryPath -Value $thresholdLine
            }
        }
    }
}

Write-Host ""
Write-Host "============================================================"
Write-Host "[ALL DONE]"
Write-Host "Summary:   $SummaryPath"
Write-Host "Metrics:   $MetricsSummaryPath"
if ($RunThresholdSelection) { Write-Host "Threshold: $ThresholdSummaryPath" }
Write-Host "Logs:      $LogRoot"
Write-Host "Eval:      $EvalRoot"
Write-Host "============================================================"
