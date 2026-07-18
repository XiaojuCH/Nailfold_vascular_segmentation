param(
    [string]$Python = "D:\anaconda3\envs\pytorch\python.exe",
    [int]$Seed = 42,
    [int]$Epochs = 50,
    [int]$Patience = 20,
    [int]$BatchSize = 4,
    [double]$Lr = 1e-4,
    [switch]$SkipExisting,
    [switch]$SkipDevelopmentTest
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$ProjectRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $ProjectRoot

if (-not (Test-Path -LiteralPath $Python)) {
    throw "Python executable not found: $Python"
}

# These names are a preregistered F1-F3 ablation chain. No loss, augmentation,
# optimizer, split, or seed changes are made between variants.
$RunDate = "20260715"
$RunRoot = Join-Path "results" "directional_prior_probes_$RunDate"
$LogRoot = Join-Path $RunRoot "logs"
$EvalRoot = Join-Path "results" "unified_eval_directional_prior_probes_$RunDate"
$ManifestPath = Join-Path $RunRoot "development_test_manifest.json"
$RunSummaryPath = Join-Path $RunRoot "run_summary.csv"
$MetricsSummaryPath = Join-Path $RunRoot "metrics_summary.csv"
New-Item -ItemType Directory -Force -Path $RunRoot, $LogRoot, $EvalRoot | Out-Null

$BaselineExperiment = "all_filtered/r0_transunet_corrected_scratch_seed42_20260714"

function Get-LatestRunDirectory {
    param([string]$ExperimentName)

    $experimentRoot = Join-Path "results\experiments" $ExperimentName
    if (-not (Test-Path -LiteralPath $experimentRoot)) {
        throw "Missing experiment root: $experimentRoot"
    }
    $latest = Get-ChildItem -LiteralPath $experimentRoot -Directory |
        Sort-Object LastWriteTime -Descending |
        Select-Object -First 1
    if ($null -eq $latest) {
        throw "No timestamped run directory under: $experimentRoot"
    }
    return $latest
}

function Get-BestWeight {
    param([string]$ExperimentName)

    $latestRun = Get-LatestRunDirectory -ExperimentName $ExperimentName
    $weight = Join-Path $latestRun.FullName "best_model.pth"
    if (-not (Test-Path -LiteralPath $weight)) {
        throw "Missing best weight: $weight"
    }
    return $weight
}

function Test-TrainingCompleted {
    param([string]$LogPath)

    # A partial run can already have best_model.pth. Only an explicit completion
    # marker makes it eligible for reuse after an interrupted PowerShell session.
    foreach ($path in @($LogPath, "$LogPath.stdout")) {
        if ((Test-Path -LiteralPath $path) -and
            (Select-String -LiteralPath $path -Pattern 'Training complete\. Results saved to:' -Quiet)) {
            return $true
        }
    }
    return $false
}

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
    if (Test-Path -LiteralPath $stdoutPath) {
        Get-Content -LiteralPath $stdoutPath | Add-Content -Encoding UTF8 -Path $LogPath
    }
    "===== STDERR =====" | Add-Content -Encoding UTF8 -Path $LogPath
    if (Test-Path -LiteralPath $stderrPath) {
        Get-Content -LiteralPath $stderrPath | Add-Content -Encoding UTF8 -Path $LogPath
    }

    if ($process.ExitCode -ne 0) {
        Write-Host "[FAILED] $StepName (exit code $($process.ExitCode))"
        if (Test-Path -LiteralPath $stderrPath) {
            Get-Content -LiteralPath $stderrPath | Select-Object -Last 100
        }
        throw "Step failed. See: $LogPath"
    }

    Write-Host "[DONE] $StepName"
    Get-Content -LiteralPath $LogPath | Select-Object -Last 16
}

function Get-BestValidationRow {
    param([string]$ExperimentName)

    $run = Get-LatestRunDirectory -ExperimentName $ExperimentName
    $trainingLog = Join-Path $run.FullName "training_log.txt"
    if (-not (Test-Path -LiteralPath $trainingLog)) {
        throw "Missing training log: $trainingLog"
    }

    $rows = @()
    Get-Content -LiteralPath $trainingLog | ForEach-Object {
        if ($_ -match '^Ep\s+(\d+).*\| Dice:\s+([0-9.]+)\s+\| HD95:\s+([0-9.]+)') {
            $rows += [pscustomobject]@{
                epoch = [int]$Matches[1]
                val_dice = [double]$Matches[2]
                val_hd95 = [double]$Matches[3]
            }
        }
    }
    if ($rows.Count -eq 0) {
        throw "No validation metrics found in: $trainingLog"
    }
    return $rows | Sort-Object val_dice -Descending | Select-Object -First 1
}

$experiments = @(
    [pscustomobject]@{
        key = "f1_plain_green_single"
        display_name = "F1_plain_green_single_scratch_seed$Seed"
        experiment_name = "all_filtered/f1_plain_green_single_scratch_seed${Seed}_$RunDate"
        variant = "plain_single"
        purpose = "Green feature branch only; no directional strip filters."
    },
    [pscustomobject]@{
        key = "f2_directional_green_single"
        display_name = "F2_directional_green_single_scratch_seed$Seed"
        experiment_name = "all_filtered/f2_directional_green_single_scratch_seed${Seed}_$RunDate"
        variant = "directional_single"
        purpose = "F1 plus parallel 1x7, 7x1, 1x21, and 21x1 green prior filters."
    },
    [pscustomobject]@{
        key = "f3_directional_green_multiscale"
        display_name = "F3_directional_green_multiscale_scratch_seed$Seed"
        experiment_name = "all_filtered/f3_directional_green_multiscale_scratch_seed${Seed}_$RunDate"
        variant = "directional_multiscale"
        purpose = "F2 plus gated prior injection at 64x64, 128x128, and 256x256 decoder scales."
    }
)

$runRecords = @()
foreach ($experiment in $experiments) {
    $trainLog = Join-Path $LogRoot "$($experiment.key)_train.log"
    $existingWeight = $null
    try {
        $existingWeight = Get-BestWeight -ExperimentName $experiment.experiment_name
    }
    catch {
        $existingWeight = $null
    }

    $completed = Test-TrainingCompleted -LogPath $trainLog
    if ($SkipExisting -and $completed -and $null -ne $existingWeight) {
        Write-Host "[SKIP COMPLETED] $($experiment.display_name): $existingWeight"
    }
    else {
        if ($null -ne $existingWeight) {
            Write-Host "[RESTART INCOMPLETE] $($experiment.display_name) has a partial best_model.pth but no completion marker."
        }
        $trainArgs = @(
            "train_unified.py",
            "--mode", "prior_fusion",
            "--prior_fusion_variant", $experiment.variant,
            "--dataset", "all_filtered",
            "--seg_loss", "bce_dice",
            "--intensity_aug", "on",
            "--epochs", "$Epochs",
            "--patience", "$Patience",
            "--batch_size", "$BatchSize",
            "--lr", "$Lr",
            "--seed", "$Seed",
            "--exp_name", $experiment.experiment_name
        )
        Invoke-LoggedPython -StepName "Train $($experiment.display_name)" -Arguments $trainArgs -LogPath $trainLog
    }

    $bestWeight = Get-BestWeight -ExperimentName $experiment.experiment_name
    $bestValidation = Get-BestValidationRow -ExperimentName $experiment.experiment_name
    $runRecords += [pscustomobject]@{
        experiment = $experiment.display_name
        experiment_name = $experiment.experiment_name
        variant = $experiment.variant
        purpose = $experiment.purpose
        seed = $Seed
        pretrained = "scratch"
        seg_loss = "bce_dice"
        intensity_aug = "on"
        best_val_epoch = $bestValidation.epoch
        best_val_dice = $bestValidation.val_dice
        hd95_at_best_val_dice = $bestValidation.val_hd95
        weight = $bestWeight
    }
}

$baselineWeight = Get-BestWeight -ExperimentName $BaselineExperiment
$baselineValidation = Get-BestValidationRow -ExperimentName $BaselineExperiment
$allRunRecords = @(
    [pscustomobject]@{
        experiment = "F0_transunet_corrected_scratch_seed$Seed"
        experiment_name = $BaselineExperiment
        variant = "none"
        purpose = "Corrected TransUNet baseline."
        seed = $Seed
        pretrained = "scratch"
        seg_loss = "bce_dice"
        intensity_aug = "on"
        best_val_epoch = $baselineValidation.epoch
        best_val_dice = $baselineValidation.val_dice
        hd95_at_best_val_dice = $baselineValidation.val_hd95
        weight = $baselineWeight
    }
) + $runRecords
$allRunRecords | Export-Csv -NoTypeInformation -Encoding UTF8 -Path $RunSummaryPath

if (-not $SkipDevelopmentTest) {
    # The current test split is explicitly treated as a development test because
    # earlier model choices already used it. Evaluate only once after F1-F3 finish.
    $manifest = @{
        experiments = @(
            @{
                name = "F0_transunet_corrected_scratch_seed$Seed"
                model_type = "transunet"
                weight = $baselineWeight
                seg_loss = "bce_dice"
            }
        )
    }
    foreach ($record in $runRecords) {
        $manifest.experiments += @{
            name = $record.experiment
            model_type = "prior_fusion"
            weight = $record.weight
            prior_fusion_variant = $record.variant
            seg_loss = "bce_dice"
        }
    }
    # Use UTF-8 without BOM so the manifest is portable across Python readers.
    [System.IO.File]::WriteAllText(
        (Join-Path $ProjectRoot $ManifestPath),
        ($manifest | ConvertTo-Json -Depth 5),
        [System.Text.UTF8Encoding]::new($false)
    )

    $evalLog = Join-Path $LogRoot "development_test_eval.log"
    $evalArgs = @(
        "evaluate_all.py",
        "--manifest", $ManifestPath,
        "--dataset", "all_filtered",
        "--split", "test",
        "--threshold", "0.5",
        "--img_size", "256",
        "--batch_size", "$BatchSize",
        "--out_dir", $EvalRoot
    )
    Invoke-LoggedPython -StepName "One development-test evaluation for F0-F3" -Arguments $evalArgs -LogPath $evalLog

    $aggregate = Get-ChildItem -LiteralPath $EvalRoot -Recurse -Filter "aggregate_results.csv" |
        Sort-Object LastWriteTime -Descending |
        Select-Object -First 1
    if ($null -eq $aggregate) {
        throw "Development-test evaluation finished without aggregate_results.csv under: $EvalRoot"
    }
    $testRows = Import-Csv -LiteralPath $aggregate.FullName
    $baselineTest = $testRows | Where-Object { $_.experiment -eq "F0_transunet_corrected_scratch_seed$Seed" } | Select-Object -First 1
    if ($null -eq $baselineTest) {
        throw "F0 is missing from development-test aggregate: $($aggregate.FullName)"
    }
    $metrics = foreach ($testRow in $testRows) {
        $record = $allRunRecords | Where-Object { $_.experiment -eq $testRow.experiment } | Select-Object -First 1
        [pscustomobject]@{
            experiment = $testRow.experiment
            variant = $record.variant
            best_val_epoch = $record.best_val_epoch
            best_val_dice = [double]$record.best_val_dice
            delta_val_dice_vs_f0 = [double]$record.best_val_dice - [double]$baselineValidation.val_dice
            development_test_dice = [double]$testRow.dice
            delta_test_dice_vs_f0 = [double]$testRow.dice - [double]$baselineTest.dice
            development_test_iou = [double]$testRow.iou
            development_test_hd95 = [double]$testRow.hd95
            delta_test_hd95_vs_f0 = [double]$testRow.hd95 - [double]$baselineTest.hd95
            development_test_cldice = [double]$testRow.cldice
            delta_test_cldice_vs_f0 = [double]$testRow.cldice - [double]$baselineTest.cldice
            development_test_boundary_f1 = [double]$testRow.boundary_f1
            delta_test_boundary_f1_vs_f0 = [double]$testRow.boundary_f1 - [double]$baselineTest.boundary_f1
            aggregate_csv = $aggregate.FullName
        }
    }
    $metrics | Export-Csv -NoTypeInformation -Encoding UTF8 -Path $MetricsSummaryPath
}

Write-Host ""
Write-Host "============================================================"
Write-Host "[ALL DONE]"
Write-Host "Run summary: $RunSummaryPath"
if (-not $SkipDevelopmentTest) {
    Write-Host "Metrics:     $MetricsSummaryPath"
    Write-Host "Dev test:    $EvalRoot"
}
Write-Host "Logs:        $LogRoot"
Write-Host "============================================================"
