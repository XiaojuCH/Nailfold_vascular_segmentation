param(
    [string]$Python = "D:\anaconda3\envs\pytorch\python.exe",
    [string]$Pretrained = "model\vit_checkpoint\imagenet21k\R50+ViT-B_16.npz",
    [ValidateSet("pretrained", "scratch", "both")]
    [string]$PretrainMode = "pretrained",
    [int]$Epochs = 50,
    [int]$Patience = 20,
    [int]$BatchSize = 4,
    [int]$DecoderBatchSize = 2,
    [double]$Lr = 1e-4,
    [int]$Seed = 42,
    [switch]$SkipExisting
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$ProjectRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $ProjectRoot

if ($PretrainMode -ne "scratch" -and -not (Test-Path $Pretrained)) {
    throw "Missing pretrained weight: $Pretrained"
}

$RunRoot = Join-Path "results" "four_innovation_probes_20260710"
$LogRoot = Join-Path $RunRoot "logs"
$EvalRoot = Join-Path "results" "unified_eval_four_innovation_probes_20260710"
New-Item -ItemType Directory -Force -Path $RunRoot, $LogRoot, $EvalRoot | Out-Null

$SummaryPath = Join-Path $RunRoot "run_summary.csv"
$MetricsSummaryPath = Join-Path $RunRoot "metrics_summary.csv"
if (-not $SkipExisting -or -not (Test-Path $SummaryPath)) {
    "experiment,innovation,pretrain_mode,seed,pretrained,enhancer,enhancer_norm,joint_model,intensity_aug,lambda_mse,lambda_grad,lambda_decoder_distill,decoder_distill_layers,batch_size,weight,train_log,eval_log" | Set-Content -Encoding UTF8 -Path $SummaryPath
}
if (-not $SkipExisting -or -not (Test-Path $MetricsSummaryPath)) {
    "experiment,innovation,pretrain_mode,seed,enhancer,enhancer_norm,joint_model,intensity_aug,lambda_mse,lambda_grad,lambda_decoder_distill,decoder_distill_layers,dice,iou,sensitivity,precision,specificity,accuracy,hd95,cldice,boundary_f1,decision,aggregate_csv" | Set-Content -Encoding UTF8 -Path $MetricsSummaryPath
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

function Get-LatestAggregateCsv {
    param([string]$OutDir)
    $aggregate = Get-ChildItem $OutDir -Recurse -Filter "aggregate_results.csv" | Sort-Object LastWriteTime -Descending | Select-Object -First 1
    if ($null -eq $aggregate) { throw "No aggregate_results.csv under $OutDir" }
    return $aggregate.FullName
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
    if ($dice -ge 0.7567 -and ($cldice -ge 0.8539 -or $boundary -ge 0.6464)) { return "structure_continue" }
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

$experiments = @(
    @{
        Name = "anisotropic_enhancer_pretrained_20260710"
        Display = "E1_anisotropic_enhancer_pretrained"
        Innovation = "anisotropic_kernel"
        Enhancer = "anisotropic"
        EnhancerNorm = "bn"
        JointModel = "v1"
        IntensityAug = "on"
        LambdaDecoder = ""
        DecoderLayers = ""
        Batch = $BatchSize
        TrainExtra = @("--enhancer", "anisotropic", "--enhancer_norm", "bn", "--joint_model", "v1", "--intensity_aug", "on")
        EvalExtra = @("--enhancer", "anisotropic", "--enhancer_norm", "bn", "--joint_model", "v1")
    },
    @{
        Name = "decoder_distill_pretrained_20260710"
        Display = "E2_decoder_distill_pretrained"
        Innovation = "decoder_distillation"
        Enhancer = "basic"
        EnhancerNorm = "bn"
        JointModel = "decoder_distill"
        IntensityAug = "on"
        LambdaDecoder = "1.0"
        DecoderLayers = "2,3"
        Batch = $DecoderBatchSize
        TrainExtra = @("--enhancer", "basic", "--enhancer_norm", "bn", "--joint_model", "decoder_distill", "--lambda_decoder_distill", "1.0", "--decoder_distill_layers", "2,3", "--intensity_aug", "on")
        EvalExtra = @("--enhancer", "basic", "--enhancer_norm", "bn", "--joint_model", "decoder_distill", "--lambda_decoder_distill", "1.0", "--decoder_distill_layers", "2,3")
    },
    @{
        Name = "dual_fusion_pretrained_20260710"
        Display = "E3_dual_fusion_pretrained"
        Innovation = "cnn_transunet_dual_fusion"
        Enhancer = "basic"
        EnhancerNorm = "bn"
        JointModel = "dual_fusion"
        IntensityAug = "on"
        LambdaDecoder = ""
        DecoderLayers = ""
        Batch = $BatchSize
        TrainExtra = @("--enhancer", "basic", "--enhancer_norm", "bn", "--joint_model", "dual_fusion", "--intensity_aug", "on")
        EvalExtra = @("--enhancer", "basic", "--enhancer_norm", "bn", "--joint_model", "dual_fusion")
    },
    @{
        Name = "no_enhancer_bn_pretrained_20260710"
        Display = "E4a_no_enhancer_bn_pretrained"
        Innovation = "intensity_prior_norm"
        Enhancer = "basic"
        EnhancerNorm = "none"
        JointModel = "v1"
        IntensityAug = "on"
        LambdaDecoder = ""
        DecoderLayers = ""
        Batch = $BatchSize
        TrainExtra = @("--enhancer", "basic", "--enhancer_norm", "none", "--joint_model", "v1", "--intensity_aug", "on")
        EvalExtra = @("--enhancer", "basic", "--enhancer_norm", "none", "--joint_model", "v1")
    },
    @{
        Name = "no_intensity_aug_pretrained_20260710"
        Display = "E4b_no_intensity_aug_pretrained"
        Innovation = "intensity_prior_augmentation"
        Enhancer = "basic"
        EnhancerNorm = "bn"
        JointModel = "v1"
        IntensityAug = "off"
        LambdaDecoder = ""
        DecoderLayers = ""
        Batch = $BatchSize
        TrainExtra = @("--enhancer", "basic", "--enhancer_norm", "bn", "--joint_model", "v1", "--intensity_aug", "off")
        EvalExtra = @("--enhancer", "basic", "--enhancer_norm", "bn", "--joint_model", "v1")
    }
)

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
        $runName = $exp.Name -replace "_pretrained_20260710$", "_$($pretrain.Tag)_20260710"
        $displayName = $exp.Display -replace "_pretrained$", "_$($pretrain.Tag)"
        $expName = "all_filtered/$runName"
        $trainLog = Join-Path $LogRoot "$($runName)_train.log"
        $evalLog = Join-Path $LogRoot "$($runName)_eval.log"
        $expEvalOutDir = Join-Path $EvalRoot $runName
        New-Item -ItemType Directory -Force -Path $expEvalOutDir | Out-Null

        if ($SkipExisting) {
            $existingAggregate = Get-ChildItem $expEvalOutDir -Recurse -Filter "aggregate_results.csv" -ErrorAction SilentlyContinue | Sort-Object LastWriteTime -Descending | Select-Object -First 1
            if ($null -ne $existingAggregate) {
                Write-Host "[SKIP] $displayName already has aggregate: $($existingAggregate.FullName)"
                continue
            }
        }

        $trainArgs = @(
            "train_unified.py",
            "--mode", "ours",
            "--dataset", "all_filtered",
            "--teacher_mode", "green_only",
            "--loss_weighting", "fixed",
            "--lambda_mse", "10.0",
            "--lambda_grad", "0.0",
            "--seg_loss", "bce_dice",
            "--epochs", "$Epochs",
            "--patience", "$Patience",
            "--batch_size", "$($exp.Batch)",
            "--lr", "$Lr",
            "--seed", "$Seed",
            "--exp_name", $expName
        ) + $pretrain.TrainExtra + $exp.TrainExtra

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
            "--loss_weighting", "fixed",
            "--lambda_mse", "10.0",
            "--lambda_grad", "0.0",
            "--seg_loss", "bce_dice",
            "--out_dir", $expEvalOutDir
        ) + $exp.EvalExtra

        Invoke-LoggedPython -StepName "Evaluate $displayName" -Arguments $evalArgs -LogPath $evalLog

        $aggregateCsv = Get-LatestAggregateCsv -OutDir $expEvalOutDir
        $metrics = Import-Csv $aggregateCsv | Select-Object -First 1
        $decision = Get-Decision -Metrics $metrics

        $line = '"{0}","{1}","{2}",{3},"{4}","{5}","{6}","{7}","{8}",10.0,0.0,"{9}","{10}",{11},"{12}","{13}","{14}"' -f $displayName, $exp.Innovation, $pretrain.Tag, $Seed, $pretrain.PretrainedValue, $exp.Enhancer, $exp.EnhancerNorm, $exp.JointModel, $exp.IntensityAug, $exp.LambdaDecoder, $exp.DecoderLayers, $exp.Batch, $weight, $trainLog, $evalLog
        Add-ContentWithRetry -Path $SummaryPath -Value $line

        $metricLine = '"{0}","{1}","{2}",{3},"{4}","{5}","{6}","{7}",10.0,0.0,"{8}","{9}",{10},{11},{12},{13},{14},{15},{16},{17},{18},"{19}","{20}"' -f $displayName, $exp.Innovation, $pretrain.Tag, $Seed, $exp.Enhancer, $exp.EnhancerNorm, $exp.JointModel, $exp.IntensityAug, $exp.LambdaDecoder, $exp.DecoderLayers, $metrics.dice, $metrics.iou, $metrics.sensitivity, $metrics.precision, $metrics.specificity, $metrics.accuracy, $metrics.hd95, $metrics.cldice, $metrics.boundary_f1, $decision, $aggregateCsv
        Add-ContentWithRetry -Path $MetricsSummaryPath -Value $metricLine
    }
}

Write-Host ""
Write-Host "============================================================"
Write-Host "[ALL DONE]"
Write-Host "Summary: $SummaryPath"
Write-Host "Metrics: $MetricsSummaryPath"
Write-Host "Logs:    $LogRoot"
Write-Host "Eval:    $EvalRoot"
Write-Host "============================================================"
