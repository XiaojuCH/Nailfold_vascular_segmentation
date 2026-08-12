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
    throw "Python executable is missing: $Python"
}

$RunDate = "20260719"
$RunRoot = Join-Path "results" "cgma_2x2_$RunDate"
$LogRoot = Join-Path $RunRoot "logs"
$CompletedRoot = Join-Path $RunRoot "completed"
$RunSummaryPath = Join-Path $RunRoot "run_summary.csv"
$ValManifestPath = Join-Path $RunRoot "val_manifest.json"
$DevManifestPath = Join-Path $RunRoot "development_test_manifest.json"
$ValEvalRoot = Join-Path "results" "unified_eval_cgma_2x2_val_$RunDate"
$DevEvalRoot = Join-Path "results" "unified_eval_cgma_2x2_$RunDate"
$MetricsPath = Join-Path $RunRoot "metrics_summary.csv"
$DecisionPath = Join-Path $RunRoot "decision.json"
New-Item -ItemType Directory -Force -Path $RunRoot, $LogRoot, $CompletedRoot, $ValEvalRoot, $DevEvalRoot | Out-Null

function Invoke-LoggedPython {
    param([string]$StepName, [string[]]$Arguments, [string]$LogPath)
    $stdoutPath = "$LogPath.stdout"
    $stderrPath = "$LogPath.stderr"
    foreach ($path in @($LogPath, $stdoutPath, $stderrPath)) {
        if (Test-Path -LiteralPath $path) { Remove-Item -LiteralPath $path -Force }
    }

    Write-Host ""
    Write-Host "============================================================"
    Write-Host "[START] $StepName"
    Write-Host "[LOG]   $LogPath"
    Write-Host "============================================================"
    $process = Start-Process -FilePath $Python -ArgumentList $Arguments -NoNewWindow -Wait -PassThru -RedirectStandardOutput $stdoutPath -RedirectStandardError $stderrPath

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
    Get-Content -LiteralPath $LogPath | Select-Object -Last 16
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

function Test-CompletedExperiment {
    param([string]$MarkerPath)
    if (-not (Test-Path -LiteralPath $MarkerPath)) { return $false }
    try {
        $record = Get-Content -Raw -LiteralPath $MarkerPath | ConvertFrom-Json
        return Test-Path -LiteralPath $record.weight
    } catch {
        return $false
    }
}

$experiments = @(
    [pscustomobject]@{ key="M0_transunet_no_prior_no_aux"; display="M0_transunet_no_prior_no_aux_seed$Seed"; experiment_name="all_filtered/cgma_m0_transunet_no_prior_no_aux_$RunDate"; prior="off"; auxiliary="off"; variant="baseline_control" },
    [pscustomobject]@{ key="M1_fixed_prior"; display="M1_fixed_prior_seed$Seed"; experiment_name="all_filtered/cgma_m1_fixed_prior_$RunDate"; prior="on"; auxiliary="off"; variant="fixed_green_local_contrast" },
    [pscustomobject]@{ key="M2_structure_aux"; display="M2_structure_aux_seed$Seed"; experiment_name="all_filtered/cgma_m2_structure_aux_$RunDate"; prior="off"; auxiliary="on"; variant="boundary_centerline_auxiliary" },
    [pscustomobject]@{ key="M3_cgma_full"; display="M3_cgma_full_seed$Seed"; experiment_name="all_filtered/cgma_m3_full_$RunDate"; prior="on"; auxiliary="on"; variant="fixed_prior_plus_structure_auxiliary" }
)

$runRecords = @()
foreach ($experiment in $experiments) {
    $markerPath = Join-Path $CompletedRoot "$($experiment.key)_seed$Seed.json"
    $trainLog = Join-Path $LogRoot "$($experiment.key)_train.log"
    if ($SkipExisting -and (Test-CompletedExperiment -MarkerPath $markerPath)) {
        $marker = Get-Content -Raw -LiteralPath $markerPath | ConvertFrom-Json
        Write-Host "[SKIP COMPLETED] $($experiment.display): $($marker.weight)"
        $runRecords += [pscustomobject]$marker
        continue
    }

    $trainArgs = @(
        "train_unified.py", "--mode", "cgma", "--dataset", "all_filtered", "--seg_loss", "bce_dice",
        "--cgma_prior", $experiment.prior, "--cgma_auxiliary", $experiment.auxiliary,
        "--cgma_boundary_weight", "0.10", "--cgma_centerline_weight", "0.10",
        "--intensity_aug", "off", "--epochs", "$Epochs", "--patience", "$Patience",
        "--batch_size", "$BatchSize", "--lr", "$Lr", "--seed", "$Seed", "--exp_name", $experiment.experiment_name
    )
    Invoke-LoggedPython -StepName "Train $($experiment.display)" -Arguments $trainArgs -LogPath $trainLog

    $weight = Get-BestWeight -ExperimentName $experiment.experiment_name
    $validation = Get-BestValidationRow -ExperimentName $experiment.experiment_name
    $record = [ordered]@{
        experiment=$experiment.display; experiment_name=$experiment.experiment_name; variant=$experiment.variant
        cgma_prior=$experiment.prior; cgma_auxiliary=$experiment.auxiliary; seed=$Seed
        best_val_epoch=$validation.epoch; best_val_dice=$validation.val_dice
        hd95_at_best_val_dice=$validation.val_hd95; weight=$weight
    }
    $record | ConvertTo-Json | Set-Content -Encoding UTF8 -Path $markerPath
    $runRecords += [pscustomobject]$record
}
$runRecords | Export-Csv -NoTypeInformation -Encoding UTF8 -Path $RunSummaryPath

function Write-Manifest {
    param([string]$Path)
    $manifest = @{ experiments = @() }
    foreach ($record in $runRecords) {
        $manifest.experiments += @{
            name=$record.experiment; model_type="cgma"; weight=$record.weight; seg_loss="bce_dice"
            cgma_prior=$record.cgma_prior; cgma_auxiliary=$record.cgma_auxiliary
            cgma_boundary_weight=0.10; cgma_centerline_weight=0.10
        }
    }
    $content = $manifest | ConvertTo-Json -Depth 5
    [System.IO.File]::WriteAllText((Join-Path $ProjectRoot $Path), $content, [System.Text.UTF8Encoding]::new($false))
}

Write-Manifest -Path $ValManifestPath
$valArgs = @("evaluate_all.py", "--manifest", $ValManifestPath, "--dataset", "all_filtered", "--split", "val", "--threshold", "0.5", "--img_size", "256", "--batch_size", "$BatchSize", "--out_dir", $ValEvalRoot)
Invoke-LoggedPython -StepName "One unified validation evaluation for M0-M3" -Arguments $valArgs -LogPath (Join-Path $LogRoot "val_eval.log")

Write-Manifest -Path $DevManifestPath
$devArgs = @("evaluate_all.py", "--manifest", $DevManifestPath, "--dataset", "all_filtered", "--split", "test", "--threshold", "0.5", "--img_size", "256", "--batch_size", "$BatchSize", "--out_dir", $DevEvalRoot)
Invoke-LoggedPython -StepName "One development-test evaluation for M0-M3" -Arguments $devArgs -LogPath (Join-Path $LogRoot "development_test_eval.log")

$valAggregate = Get-ChildItem -LiteralPath $ValEvalRoot -Recurse -Filter "aggregate_results.csv" | Sort-Object LastWriteTime -Descending | Select-Object -First 1
$devAggregate = Get-ChildItem -LiteralPath $DevEvalRoot -Recurse -Filter "aggregate_results.csv" | Sort-Object LastWriteTime -Descending | Select-Object -First 1
if ($null -eq $valAggregate -or $null -eq $devAggregate) { throw "Missing unified aggregate results after evaluation." }

$summaryArgs = @("summarize_cgma_2x2.py", "--run_summary", $RunSummaryPath, "--val_aggregate", $valAggregate.FullName, "--development_test_aggregate", $devAggregate.FullName, "--out_csv", $MetricsPath, "--out_decision", $DecisionPath, "--seed", "$Seed")
Invoke-LoggedPython -StepName "Summarize CGMA 2 x 2 deltas" -Arguments $summaryArgs -LogPath (Join-Path $LogRoot "summarize.log")

Write-Host ""
Write-Host "============================================================"
Write-Host "[ALL DONE]"
Write-Host "Run summary: $RunSummaryPath"
Write-Host "Metrics:     $MetricsPath"
Write-Host "Decision:    $DecisionPath"
Write-Host "Logs:        $LogRoot"
Write-Host "Val eval:    $ValEvalRoot"
Write-Host "Dev eval:    $DevEvalRoot"
Write-Host "============================================================"
