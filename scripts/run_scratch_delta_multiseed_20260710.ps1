param(
    [string]$Python = "D:\anaconda3\envs\pytorch\python.exe",
    [string]$Pretrained = "model\vit_checkpoint\imagenet21k\R50+ViT-B_16.npz",
    [ValidateSet("scratch", "pretrained", "both")]
    [string]$PretrainMode = "scratch",
    [string]$Seeds = "43,44",
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

if ($PretrainMode -ne "scratch" -and -not (Test-Path $Pretrained)) {
    throw "Missing pretrained weight: $Pretrained"
}

$RunRoot = Join-Path "results" "scratch_delta_multiseed_20260710"
$LogRoot = Join-Path $RunRoot "logs"
$EvalRoot = Join-Path "results" "unified_eval_scratch_delta_multiseed_20260710"
New-Item -ItemType Directory -Force -Path $RunRoot, $LogRoot, $EvalRoot | Out-Null

$SummaryPath = Join-Path $RunRoot "run_summary.csv"
$MetricsSummaryPath = Join-Path $RunRoot "metrics_summary.csv"
if (-not $SkipExisting -or -not (Test-Path $SummaryPath)) {
    "experiment,group,pretrain_mode,seed,pretrained,mode,model_type,enhancer,enhancer_norm,intensity_aug,lambda_mse,lambda_grad,seg_loss,batch_size,weight,train_log,eval_log" | Set-Content -Encoding UTF8 -Path $SummaryPath
}
if (-not $SkipExisting -or -not (Test-Path $MetricsSummaryPath)) {
    "experiment,group,pretrain_mode,seed,dice,iou,sensitivity,precision,specificity,accuracy,hd95,cldice,boundary_f1,delta_vs_transunet_seed42,delta_vs_ours_seed42,aggregate_csv" | Set-Content -Encoding UTF8 -Path $MetricsSummaryPath
}

function Get-SeedList {
    param([string]$Text)
    return @($Text -split "[,;\s]+" | Where-Object { $_.Trim() } | ForEach-Object { [int]$_.Trim() } | Where-Object { $_ -gt 0 })
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
        Key = "transunet"
        Group = "baseline"
        Display = "TransUNet"
        Mode = "baseline"
        ModelType = "transunet"
        Enhancer = ""
        EnhancerNorm = ""
        IntensityAug = ""
        LambdaMse = ""
        LambdaGrad = ""
        SegLoss = "bce_dice"
        TrainExtra = @()
        EvalExtra = @()
    },
    @{
        Key = "ours_green_mse10_grad0"
        Group = "old_ours"
        Display = "Ours_green_mse10_grad0"
        Mode = "ours"
        ModelType = "ours"
        Enhancer = "basic"
        EnhancerNorm = "bn"
        IntensityAug = "on"
        LambdaMse = "10.0"
        LambdaGrad = "0.0"
        SegLoss = "bce_dice"
        TrainExtra = @("--teacher_mode", "green_only", "--enhancer", "basic", "--enhancer_norm", "bn", "--joint_model", "v1", "--intensity_aug", "on", "--loss_weighting", "fixed", "--lambda_mse", "10.0", "--lambda_grad", "0.0")
        EvalExtra = @("--teacher_mode", "green_only", "--enhancer", "basic", "--enhancer_norm", "bn", "--joint_model", "v1", "--loss_weighting", "fixed", "--lambda_mse", "10.0", "--lambda_grad", "0.0")
    },
    @{
        Key = "c3_anisotropic_no_bn_no_intensity"
        Group = "new_c3"
        Display = "C3_anisotropic_no_bn_no_intensity"
        Mode = "ours"
        ModelType = "ours"
        Enhancer = "anisotropic"
        EnhancerNorm = "none"
        IntensityAug = "off"
        LambdaMse = "10.0"
        LambdaGrad = "0.0"
        SegLoss = "bce_dice"
        TrainExtra = @("--teacher_mode", "green_only", "--enhancer", "anisotropic", "--enhancer_norm", "none", "--joint_model", "v1", "--intensity_aug", "off", "--loss_weighting", "fixed", "--lambda_mse", "10.0", "--lambda_grad", "0.0")
        EvalExtra = @("--teacher_mode", "green_only", "--enhancer", "anisotropic", "--enhancer_norm", "none", "--joint_model", "v1", "--loss_weighting", "fixed", "--lambda_mse", "10.0", "--lambda_grad", "0.0")
    }
)

$pretrainSettings = @()
if ($PretrainMode -eq "scratch" -or $PretrainMode -eq "both") {
    $pretrainSettings += @{
        Tag = "scratch"
        PretrainedValue = ""
        TrainExtra = @()
        DeltaTransUNetSeed42 = 0.7521763416398705
        DeltaOursSeed42 = 0.757144138
    }
}
if ($PretrainMode -eq "pretrained" -or $PretrainMode -eq "both") {
    $pretrainSettings += @{
        Tag = "pretrained"
        PretrainedValue = $Pretrained
        TrainExtra = @("--pretrained", $Pretrained)
        DeltaTransUNetSeed42 = 0.7566664561293599
        DeltaOursSeed42 = 0.7583218461767187
    }
}

foreach ($seed in (Get-SeedList -Text $Seeds)) {
    foreach ($pretrain in $pretrainSettings) {
        foreach ($exp in $experiments) {
            $runName = "$($exp.Key)_$($pretrain.Tag)_seed${seed}_20260710"
            $displayName = "$($exp.Display)_$($pretrain.Tag)_seed${seed}"
            $expName = "all_filtered/$runName"
            $trainLog = Join-Path $LogRoot "$($runName)_train.log"
            $evalLog = Join-Path $LogRoot "$($runName)_eval.log"
            $expEvalOutDir = Join-Path $EvalRoot $runName
            New-Item -ItemType Directory -Force -Path $expEvalOutDir | Out-Null

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
                    "--mode", $exp.Mode,
                    "--dataset", "all_filtered",
                    "--seg_loss", $exp.SegLoss,
                    "--epochs", "$Epochs",
                    "--patience", "$Patience",
                    "--batch_size", "$BatchSize",
                    "--lr", "$Lr",
                    "--seed", "$seed",
                    "--exp_name", $expName
                ) + $pretrain.TrainExtra + $exp.TrainExtra

                Invoke-LoggedPython -StepName "Train $displayName" -Arguments $trainArgs -LogPath $trainLog
                $weight = Get-LatestWeight -ExpName $expName
                Write-Host "[WEIGHT] $weight"

                $evalArgs = @(
                    "evaluate_all.py",
                    "--name", $displayName,
                    "--model_type", $exp.ModelType,
                    "--weight", $weight,
                    "--dataset", "all_filtered",
                    "--split", "test",
                    "--threshold", "0.5",
                    "--batch_size", "$BatchSize",
                    "--seg_loss", $exp.SegLoss,
                    "--out_dir", $expEvalOutDir
                ) + $exp.EvalExtra

                Invoke-LoggedPython -StepName "Evaluate $displayName" -Arguments $evalArgs -LogPath $evalLog
                $aggregateCsv = Get-LatestAggregateCsv -OutDir $expEvalOutDir
                $metrics = Import-Csv $aggregateCsv | Select-Object -First 1
            }

            if (-not (Test-CsvContainsExperiment -Path $SummaryPath -Experiment $displayName)) {
                $runLine = '"{0}","{1}","{2}",{3},"{4}","{5}","{6}","{7}","{8}","{9}","{10}","{11}","{12}",{13},"{14}","{15}","{16}"' -f $displayName, $exp.Group, $pretrain.Tag, $seed, $pretrain.PretrainedValue, $exp.Mode, $exp.ModelType, $exp.Enhancer, $exp.EnhancerNorm, $exp.IntensityAug, $exp.LambdaMse, $exp.LambdaGrad, $exp.SegLoss, $BatchSize, $weight, $trainLog, $evalLog
                Add-ContentWithRetry -Path $SummaryPath -Value $runLine
            }

            if (-not (Test-CsvContainsExperiment -Path $MetricsSummaryPath -Experiment $displayName)) {
                $deltaTransUNet = [double]$metrics.dice - [double]$pretrain.DeltaTransUNetSeed42
                $deltaOurs = [double]$metrics.dice - [double]$pretrain.DeltaOursSeed42
                $metricLine = '"{0}","{1}","{2}",{3},{4},{5},{6},{7},{8},{9},{10},{11},{12},{13},{14},"{15}"' -f $displayName, $exp.Group, $pretrain.Tag, $seed, $metrics.dice, $metrics.iou, $metrics.sensitivity, $metrics.precision, $metrics.specificity, $metrics.accuracy, $metrics.hd95, $metrics.cldice, $metrics.boundary_f1, $deltaTransUNet, $deltaOurs, $aggregateCsv
                Add-ContentWithRetry -Path $MetricsSummaryPath -Value $metricLine
            }
        }
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
