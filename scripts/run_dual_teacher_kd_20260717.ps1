param(
    [string]$Python = "D:\anaconda3\envs\pytorch\python.exe",
    [string]$Pretrained = "model\vit_checkpoint\imagenet21k\R50+ViT-B_16.npz",
    [string]$F0Weight = "results\experiments\all_filtered\f0_transunet_corrected_pretrained_seed42_20260715\0715_1907\best_model.pth",
    [string]$F3Weight = "results\experiments\all_filtered\f3_directional_green_multiscale_scratch_seed42_20260715\0715_1556\best_model.pth",
    [int]$Seed = 42,
    [int]$Epochs = 30,
    [int]$Patience = 10,
    [int]$BatchSize = 4,
    [double]$Lr = 3e-5,
    [switch]$SkipExisting,
    [switch]$RunFallback,
    [switch]$SkipDevelopmentTest
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$ProjectRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $ProjectRoot

foreach ($path in @($Python, $Pretrained, $F0Weight, $F3Weight)) {
    if (-not (Test-Path -LiteralPath $path)) {
        throw "Required file is missing: $path"
    }
}

$RunDate = "20260717"
$RunRoot = Join-Path "results" "dual_teacher_kd_$RunDate"
$TargetRoot = Join-Path $RunRoot "soft_targets"
$LogRoot = Join-Path $RunRoot "logs"
$EvalRoot = Join-Path "results" "unified_eval_dual_teacher_kd_$RunDate"
$ManifestPath = Join-Path $RunRoot "development_test_manifest.json"
$RunSummaryPath = Join-Path $RunRoot "run_summary.csv"
$MetricsSummaryPath = Join-Path $RunRoot "metrics_summary.csv"
$DecisionPath = Join-Path $RunRoot "first_night_decision.json"
if ($RunFallback) {
    $ManifestPath = Join-Path $RunRoot "fallback_development_test_manifest.json"
    $RunSummaryPath = Join-Path $RunRoot "fallback_run_summary.csv"
    $MetricsSummaryPath = Join-Path $RunRoot "fallback_metrics_summary.csv"
    $DecisionPath = Join-Path $RunRoot "fallback_decision.json"
}
New-Item -ItemType Directory -Force -Path $RunRoot, $LogRoot, $EvalRoot | Out-Null

function Invoke-LoggedPython {
    param([string]$StepName, [string[]]$Arguments, [string]$LogPath)

    $stdoutPath = "$LogPath.stdout"
    $stderrPath = "$LogPath.stderr"
    foreach ($path in @($LogPath, $stdoutPath, $stderrPath)) {
        if (Test-Path -LiteralPath $path) {
            Remove-Item -LiteralPath $path -Force
        }
    }

    Write-Host ""
    Write-Host "============================================================"
    Write-Host "[START] $StepName"
    Write-Host "[LOG]   $LogPath"
    Write-Host "============================================================"

    $process = Start-Process -FilePath $Python -ArgumentList $Arguments -NoNewWindow -Wait -PassThru `
        -RedirectStandardOutput $stdoutPath -RedirectStandardError $stderrPath
    "===== STDOUT =====" | Set-Content -Encoding UTF8 -Path $LogPath
    if (Test-Path -LiteralPath $stdoutPath) { Get-Content -LiteralPath $stdoutPath | Add-Content -Encoding UTF8 -Path $LogPath }
    "===== STDERR =====" | Add-Content -Encoding UTF8 -Path $LogPath
    if (Test-Path -LiteralPath $stderrPath) { Get-Content -LiteralPath $stderrPath | Add-Content -Encoding UTF8 -Path $LogPath }

    if ($process.ExitCode -ne 0) {
        Write-Host "[FAILED] $StepName (exit code $($process.ExitCode))"
        if (Test-Path -LiteralPath $stderrPath) { Get-Content -LiteralPath $stderrPath | Select-Object -Last 100 }
        throw "Step failed. See: $LogPath"
    }
    Write-Host "[DONE] $StepName"
    Get-Content -LiteralPath $LogPath | Select-Object -Last 18
}

function Test-TrainingCompleted {
    param([string]$LogPath)
    foreach ($path in @($LogPath, "$LogPath.stdout")) {
        if ((Test-Path -LiteralPath $path) -and
            (Select-String -LiteralPath $path -Pattern 'Training complete\. Results saved to:' -Quiet)) {
            return $true
        }
    }
    return $false
}

function Get-LatestRunDirectory {
    param([string]$ExperimentName)
    $root = Join-Path "results\experiments" $ExperimentName
    if (-not (Test-Path -LiteralPath $root)) { throw "Missing experiment root: $root" }
    $run = Get-ChildItem -LiteralPath $root -Directory | Sort-Object LastWriteTime -Descending | Select-Object -First 1
    if ($null -eq $run) { throw "No timestamped run directory under: $root" }
    return $run
}

function Get-BestWeight {
    param([string]$ExperimentName)
    $weight = Join-Path (Get-LatestRunDirectory -ExperimentName $ExperimentName).FullName "best_model.pth"
    if (-not (Test-Path -LiteralPath $weight)) { throw "Missing best weight: $weight" }
    return $weight
}

function Get-BestValidationRow {
    param([string]$ExperimentName)
    $log = Join-Path (Get-LatestRunDirectory -ExperimentName $ExperimentName).FullName "training_log.txt"
    if (-not (Test-Path -LiteralPath $log)) { throw "Missing training log: $log" }
    $rows = @()
    Get-Content -LiteralPath $log | ForEach-Object {
        if ($_ -match '^Ep\s+(\d+).*\| Dice:\s+([0-9.]+)\s+\| HD95:\s+([0-9.]+)') {
            $rows += [pscustomobject]@{ epoch=[int]$Matches[1]; val_dice=[double]$Matches[2]; val_hd95=[double]$Matches[3] }
        }
    }
    if ($rows.Count -eq 0) { throw "No validation metrics found in: $log" }
    return $rows | Sort-Object val_dice -Descending | Select-Object -First 1
}

function Test-TargetGenerationComplete {
    $metadata = Join-Path $TargetRoot "metadata.json"
    $trainTargets = Join-Path $TargetRoot "train\ensemble_probabilities"
    $valTargets = Join-Path $TargetRoot "val\ensemble_probabilities"
    if (-not ((Test-Path -LiteralPath $metadata) -and (Test-Path -LiteralPath $trainTargets) -and (Test-Path -LiteralPath $valTargets))) { return $false }
    return ((Get-ChildItem -LiteralPath $trainTargets -Filter '*.npy').Count -eq 1838 -and
            (Get-ChildItem -LiteralPath $valTargets -Filter '*.npy').Count -eq 449)
}

if (-not ($SkipExisting -and (Test-TargetGenerationComplete))) {
    Invoke-LoggedPython -StepName "Generate sequential F0/F3 soft targets and ensemble audit" `
        -Arguments @(
            "generate_dual_teacher_targets.py",
            "--data_dir", "./dataset_all_filtered",
            "--splits", "train,val,test",
            "--f0_weight", $F0Weight,
            "--f3_weight", $F3Weight,
            "--f3_variant", "directional_multiscale",
            "--out_dir", $TargetRoot,
            "--batch_size", "$BatchSize",
            "--img_size", "256",
            "--threshold", "0.5"
        ) -LogPath (Join-Path $LogRoot "generate_soft_targets.log")
}
else {
    Write-Host "[SKIP COMPLETED] Dual-teacher targets: $TargetRoot"
}

if (-not (Test-Path -LiteralPath (Join-Path $TargetRoot "teacher_ensemble_metrics.csv"))) {
    throw "Missing ensemble audit after target generation."
}

$TargetTrain = Join-Path $TargetRoot "train\ensemble_probabilities"
$TargetVal = Join-Path $TargetRoot "val\ensemble_probabilities"
$DisagreementTrain = Join-Path $TargetRoot "train\disagreement"
$DisagreementVal = Join-Path $TargetRoot "val\disagreement"

if ($RunFallback) {
    $controlExperimentName = "all_filtered/K0_finetune_control_20260717"
    $controlWeight = Get-BestWeight -ExperimentName $controlExperimentName
    $controlValidation = Get-BestValidationRow -ExperimentName $controlExperimentName
    $experiments = @(
        [pscustomobject]@{ key="K3_agreement_lambda0p3"; display="K3_agreement_lambda0p3_seed$Seed"; mode="agreement"; lambda="0.3" },
        [pscustomobject]@{ key="K4_uniform_lambda0p1"; display="K4_uniform_lambda0p1_seed$Seed"; mode="uniform"; lambda="0.1" }
    )
}
else {
    $experiments = @(
        [pscustomobject]@{ key="K0_finetune_control"; display="K0_finetune_control_seed$Seed"; mode="uniform"; lambda="0.0" },
        [pscustomobject]@{ key="K1_uniform_lambda0p3"; display="K1_uniform_lambda0p3_seed$Seed"; mode="uniform"; lambda="0.3" },
        [pscustomobject]@{ key="K2_uniform_lambda1p0"; display="K2_uniform_lambda1p0_seed$Seed"; mode="uniform"; lambda="1.0" }
    )
}

$runRecords = @()
if ($RunFallback) {
    $runRecords += [pscustomobject]@{
        experiment="K0_finetune_control_seed$Seed"; experiment_name=$controlExperimentName; kd_weight_mode="uniform"; lambda_kd=0.0;
        seed=$Seed; best_val_epoch=$controlValidation.epoch; best_val_dice=$controlValidation.val_dice; hd95_at_best_val_dice=$controlValidation.val_hd95; weight=$controlWeight
    }
}
foreach ($experiment in $experiments) {
    $experimentName = "all_filtered/$($experiment.key)_$RunDate"
    $trainLog = Join-Path $LogRoot "$($experiment.key)_train.log"
    $existingWeight = $null
    try { $existingWeight = Get-BestWeight -ExperimentName $experimentName } catch { $existingWeight = $null }
    if ($SkipExisting -and $null -ne $existingWeight -and (Test-TrainingCompleted -LogPath $trainLog)) {
        Write-Host "[SKIP COMPLETED] $($experiment.display): $existingWeight"
    }
    else {
        if ($null -ne $existingWeight) { Write-Host "[RESTART INCOMPLETE] $($experiment.display) has a partial best_model.pth." }
        $args = @(
            "train_unified.py",
            "--mode", "soft_kd",
            "--dataset", "all_filtered",
            "--seg_loss", "bce_dice",
            "--pretrained", $Pretrained,
            "--init_weight", $F0Weight,
            "--soft_target_dir", $TargetTrain,
            "--lambda_kd", $experiment.lambda,
            "--kd_weight_mode", $experiment.mode,
            "--intensity_aug", "on",
            "--epochs", "$Epochs",
            "--patience", "$Patience",
            "--batch_size", "$BatchSize",
            "--lr", "$Lr",
            "--seed", "$Seed",
            "--exp_name", $experimentName
        )
        if ($experiment.mode -eq "agreement") { $args += @("--disagreement_dir", $DisagreementTrain) }
        Invoke-LoggedPython -StepName "Train $($experiment.display)" -Arguments $args -LogPath $trainLog
    }
    $weight = Get-BestWeight -ExperimentName $experimentName
    $validation = Get-BestValidationRow -ExperimentName $experimentName
    $runRecords += [pscustomobject]@{
        experiment=$experiment.display; experiment_name=$experimentName; kd_weight_mode=$experiment.mode; lambda_kd=[double]$experiment.lambda;
        seed=$Seed; best_val_epoch=$validation.epoch; best_val_dice=$validation.val_dice; hd95_at_best_val_dice=$validation.val_hd95; weight=$weight
    }
}
$runRecords | Export-Csv -NoTypeInformation -Encoding UTF8 -Path $RunSummaryPath

if (-not $SkipDevelopmentTest) {
    $manifest = @{ experiments = @() }
    foreach ($record in $runRecords) {
        $manifest.experiments += @{ name=$record.experiment; model_type="transunet"; weight=$record.weight; seg_loss="bce_dice" }
    }
    [System.IO.File]::WriteAllText((Join-Path $ProjectRoot $ManifestPath), ($manifest | ConvertTo-Json -Depth 5), [System.Text.UTF8Encoding]::new($false))
    Invoke-LoggedPython -StepName "One development-test evaluation for KD candidates" `
        -Arguments @("evaluate_all.py", "--manifest", $ManifestPath, "--dataset", "all_filtered", "--split", "test", "--threshold", "0.5", "--img_size", "256", "--batch_size", "$BatchSize", "--out_dir", $EvalRoot) `
        -LogPath (Join-Path $LogRoot "development_test_eval.log")
    $aggregate = Get-ChildItem -LiteralPath $EvalRoot -Recurse -Filter "aggregate_results.csv" | Sort-Object LastWriteTime -Descending | Select-Object -First 1
    if ($null -eq $aggregate) { throw "No aggregate_results.csv found under: $EvalRoot" }
    $candidateNames = ($runRecords | Where-Object { $_.lambda_kd -gt 0.0 } | Select-Object -ExpandProperty experiment) -join ","
    $controlName = $runRecords | Where-Object { $_.lambda_kd -eq 0.0 } | Select-Object -First 1 -ExpandProperty experiment
    if ([string]::IsNullOrWhiteSpace($controlName)) {
        throw "This script needs K0 control before it can make a KD decision. Run without -RunFallback first."
    }
    Invoke-LoggedPython -StepName "Summarize KD deltas and branch decision" `
        -Arguments @(
            "summarize_dual_teacher_kd.py",
            "--aggregate_csv", $aggregate.FullName,
            "--teacher_metrics_csv", (Join-Path $TargetRoot "teacher_ensemble_metrics.csv"),
            "--out_dir", $RunRoot,
            "--control_name", $controlName,
            "--candidate_names", $candidateNames,
            "--summary_name", (Split-Path -Leaf $MetricsSummaryPath),
            "--decision_name", (Split-Path -Leaf $DecisionPath)
        ) `
        -LogPath (Join-Path $LogRoot "summarize_first_night.log")
}

Write-Host ""
Write-Host "============================================================"
Write-Host "[ALL DONE]"
Write-Host "Targets: $TargetRoot"
Write-Host "Summary: $RunSummaryPath"
if (-not $SkipDevelopmentTest) { Write-Host "Metrics: $MetricsSummaryPath"; Write-Host "Decision: $DecisionPath" }
Write-Host "Logs: $LogRoot"
Write-Host "============================================================"
