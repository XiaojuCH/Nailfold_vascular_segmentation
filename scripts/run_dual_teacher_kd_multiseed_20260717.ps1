param(
    [string]$Python = "D:\anaconda3\envs\pytorch\python.exe",
    [string]$Pretrained = "model\vit_checkpoint\imagenet21k\R50+ViT-B_16.npz",
    [string]$F0Weight = "results\experiments\all_filtered\f0_transunet_corrected_pretrained_seed42_20260715\0715_1907\best_model.pth",
    [int[]]$Seeds = @(43, 44),
    [int]$Epochs = 30,
    [int]$Patience = 10,
    [int]$BatchSize = 4,
    [double]$Lr = 3e-5,
    [switch]$SkipExisting
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$ProjectRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $ProjectRoot

foreach ($path in @($Python, $Pretrained, $F0Weight)) {
    if (-not (Test-Path -LiteralPath $path)) { throw "Required file is missing: $path" }
}

$RunDate = "20260717"
$FirstNightRoot = Join-Path "results" "dual_teacher_kd_$RunDate"
$DecisionPath = Join-Path $FirstNightRoot "first_night_decision.json"
$TargetRoot = Join-Path $FirstNightRoot "soft_targets"
$TargetTrain = Join-Path $TargetRoot "train\ensemble_probabilities"
if (-not (Test-Path -LiteralPath $DecisionPath)) { throw "Missing first-night decision: $DecisionPath" }
if (-not (Test-Path -LiteralPath $TargetTrain)) { throw "Missing soft targets: $TargetTrain" }
$decision = Get-Content -Raw -LiteralPath $DecisionPath | ConvertFrom-Json
if ($decision.next_phase -ne "multiseed") {
    throw "First-night KD did not pass the multiseed gate. Run fallback K3/K4 instead."
}

$BestName = [string]$decision.best_candidate
if ($BestName -match '^K([12])_uniform_lambda([0-9]+)p([0-9]+)_seed42$') {
    $lambda = "$($Matches[2]).$($Matches[3])"
    $mode = "uniform"
    $keyPrefix = "K$($Matches[1])_uniform_lambda$($Matches[2])p$($Matches[3])"
}
else {
    throw "Cannot derive the winning KD config from: $BestName"
}

$RunRoot = Join-Path "results" "dual_teacher_kd_multiseed_$RunDate"
$LogRoot = Join-Path $RunRoot "logs"
$EvalRoot = Join-Path "results" "unified_eval_dual_teacher_kd_multiseed_$RunDate"
$ManifestPath = Join-Path $RunRoot "development_test_manifest.json"
$RunSummaryPath = Join-Path $RunRoot "run_summary.csv"
New-Item -ItemType Directory -Force -Path $RunRoot, $LogRoot, $EvalRoot | Out-Null

function Invoke-LoggedPython {
    param([string]$StepName, [string[]]$Arguments, [string]$LogPath)
    $stdoutPath = "$LogPath.stdout"; $stderrPath = "$LogPath.stderr"
    foreach ($path in @($LogPath, $stdoutPath, $stderrPath)) { if (Test-Path -LiteralPath $path) { Remove-Item -LiteralPath $path -Force } }
    Write-Host ""; Write-Host "============================================================"; Write-Host "[START] $StepName"; Write-Host "[LOG]   $LogPath"; Write-Host "============================================================"
    $process = Start-Process -FilePath $Python -ArgumentList $Arguments -NoNewWindow -Wait -PassThru -RedirectStandardOutput $stdoutPath -RedirectStandardError $stderrPath
    "===== STDOUT =====" | Set-Content -Encoding UTF8 -Path $LogPath
    if (Test-Path -LiteralPath $stdoutPath) { Get-Content -LiteralPath $stdoutPath | Add-Content -Encoding UTF8 -Path $LogPath }
    "===== STDERR =====" | Add-Content -Encoding UTF8 -Path $LogPath
    if (Test-Path -LiteralPath $stderrPath) { Get-Content -LiteralPath $stderrPath | Add-Content -Encoding UTF8 -Path $LogPath }
    if ($process.ExitCode -ne 0) { throw "Step failed: $StepName. See: $LogPath" }
    Write-Host "[DONE] $StepName"
}

function Get-LatestRunDirectory { param([string]$ExperimentName) $root=Join-Path "results\experiments" $ExperimentName; if(-not(Test-Path -LiteralPath $root)){throw "Missing experiment root: $root"}; $run=Get-ChildItem -LiteralPath $root -Directory | Sort-Object LastWriteTime -Descending | Select-Object -First 1; if($null -eq $run){throw "No run under $root"}; return $run }
function Get-BestWeight { param([string]$ExperimentName) $weight=Join-Path (Get-LatestRunDirectory $ExperimentName).FullName "best_model.pth"; if(-not(Test-Path -LiteralPath $weight)){throw "Missing best weight: $weight"}; return $weight }
function Test-TrainingCompleted { param([string]$LogPath) foreach($path in @($LogPath,"$LogPath.stdout")){if((Test-Path -LiteralPath $path) -and (Select-String -LiteralPath $path -Pattern 'Training complete\. Results saved to:' -Quiet)){return $true}};return $false }

$records = @()
foreach ($seed in $Seeds) {
    $runs = @(
        [pscustomobject]@{ key="K0_finetune_control"; display="K0_finetune_control_seed$seed"; lambda="0.0"; mode="uniform" },
        [pscustomobject]@{ key=$keyPrefix; display="$keyPrefix`_seed$seed"; lambda=$lambda; mode=$mode }
    )
    foreach ($run in $runs) {
        $name = "all_filtered/$($run.key)_seed$seed`_$RunDate"
        $log = Join-Path $LogRoot "$($run.key)_seed$seed`_train.log"
        $weight = $null
        try { $weight=Get-BestWeight $name } catch { $weight=$null }
        if (-not ($SkipExisting -and $null -ne $weight -and (Test-TrainingCompleted $log))) {
            $args=@("train_unified.py","--mode","soft_kd","--dataset","all_filtered","--seg_loss","bce_dice","--pretrained",$Pretrained,"--init_weight",$F0Weight,"--soft_target_dir",$TargetTrain,"--lambda_kd",$run.lambda,"--kd_weight_mode",$run.mode,"--intensity_aug","on","--epochs","$Epochs","--patience","$Patience","--batch_size","$BatchSize","--lr","$Lr","--seed","$seed","--exp_name",$name)
            Invoke-LoggedPython -StepName "Train $($run.display)" -Arguments $args -LogPath $log
        }
        $records += [pscustomobject]@{ experiment=$run.display; experiment_name=$name; seed=$seed; lambda_kd=[double]$run.lambda; weight=(Get-BestWeight $name) }
    }
}
$records | Export-Csv -NoTypeInformation -Encoding UTF8 -Path $RunSummaryPath

$manifest=@{experiments=@()}
foreach($record in $records){$manifest.experiments += @{name=$record.experiment;model_type="transunet";weight=$record.weight;seg_loss="bce_dice"}}
[System.IO.File]::WriteAllText((Join-Path $ProjectRoot $ManifestPath),($manifest|ConvertTo-Json -Depth 5),[System.Text.UTF8Encoding]::new($false))
Invoke-LoggedPython -StepName "One development-test evaluation for KD multi-seed" -Arguments @("evaluate_all.py","--manifest",$ManifestPath,"--dataset","all_filtered","--split","test","--threshold","0.5","--img_size","256","--batch_size","$BatchSize","--out_dir",$EvalRoot) -LogPath (Join-Path $LogRoot "development_test_eval.log")

$seed42Aggregate = Get-ChildItem -LiteralPath "results\unified_eval_dual_teacher_kd_20260717" -Recurse -Filter "aggregate_results.csv" | Sort-Object LastWriteTime -Descending | Select-Object -First 1
$multiAggregate = Get-ChildItem -LiteralPath $EvalRoot -Recurse -Filter "aggregate_results.csv" | Sort-Object LastWriteTime -Descending | Select-Object -First 1
if ($null -eq $seed42Aggregate -or $null -eq $multiAggregate) { throw "Missing aggregate CSV for multi-seed summary." }
Invoke-LoggedPython -StepName "Summarize KD multi-seed statistics" -Arguments @("summarize_multiseed_kd.py","--seed42_aggregate",$seed42Aggregate.FullName,"--multiseed_aggregate",$multiAggregate.FullName,"--decision_json",$DecisionPath,"--out_dir",$RunRoot) -LogPath (Join-Path $LogRoot "summarize_multiseed.log")

Write-Host "[ALL DONE] Summary: $RunSummaryPath; Eval: $EvalRoot; Logs: $LogRoot"
