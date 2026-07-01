param(
    [string]$Python = "D:\anaconda3\envs\pytorch\python.exe",
    [int]$Epochs = 50,
    [int]$Patience = 20,
    [int]$BatchSize = 4,
    [double]$Lr = 1e-4,
    [int]$Seed = 42,
    [string]$EncoderWeights = "",
    [switch]$IncludeStructure
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$ProjectRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $ProjectRoot

$RunRoot = Join-Path "results" "smp_green_prior_20260623"
$LogRoot = Join-Path $RunRoot "logs"
New-Item -ItemType Directory -Force -Path $RunRoot, $LogRoot | Out-Null

$SummaryPath = Join-Path $RunRoot "metrics_summary.csv"
"experiment,arch,encoder_name,encoder_weights,seed,teacher_mode,enhancer,joint_model,lambda_mse,lambda_grad,seg_loss,cldice_weight,boundary_weight,cbdice_weight,dice,iou,sensitivity,precision,specificity,accuracy,hd95,cldice,boundary_f1,aggregate_csv,train_log" | Set-Content -Encoding UTF8 -Path $SummaryPath

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

function Get-LatestAggregateCsv {
    param([string]$ExpName)
    $expRoot = Join-Path "results\experiments" $ExpName
    $aggregate = Get-ChildItem $expRoot -Recurse -Filter "aggregate_results.csv" | Sort-Object LastWriteTime -Descending | Select-Object -First 1
    if ($null -eq $aggregate) { throw "No aggregate_results.csv under $expRoot" }
    return $aggregate.FullName
}

$experiments = @(
    @{
        Name="smp_joint_deeplabv3plus_resnet34_green_mse10_grad0_20260623";
        Arch="deeplabv3plus"; Encoder="resnet34"; Teacher="green_only"; Enhancer="basic"; Joint="v1";
        SegLoss="bce_dice"; ClDice="0.5"; Boundary="0.5"; CbDice="0.5";
    },
    @{
        Name="smp_joint_fpn_resnet34_green_mse10_grad0_20260623";
        Arch="fpn"; Encoder="resnet34"; Teacher="green_only"; Enhancer="basic"; Joint="v1";
        SegLoss="bce_dice"; ClDice="0.5"; Boundary="0.5"; CbDice="0.5";
    },
    @{
        Name="smp_joint_unetplusplus_efficientnet_b3_green_mse10_grad0_20260623";
        Arch="unetplusplus"; Encoder="efficientnet-b3"; Teacher="green_only"; Enhancer="basic"; Joint="v1";
        SegLoss="bce_dice"; ClDice="0.5"; Boundary="0.5"; CbDice="0.5";
    }
)

if ($IncludeStructure) {
    $experiments += @(
        @{
            Name="smp_joint_fpn_resnet34_green_cldice_boundary_20260623";
            Arch="fpn"; Encoder="resnet34"; Teacher="green_only"; Enhancer="basic"; Joint="v1";
            SegLoss="bce_dice_cldice_boundary"; ClDice="0.3"; Boundary="0.3"; CbDice="0.5";
        },
        @{
            Name="smp_joint_unetplusplus_efficientnet_b3_green_cldice_boundary_20260623";
            Arch="unetplusplus"; Encoder="efficientnet-b3"; Teacher="green_only"; Enhancer="basic"; Joint="v1";
            SegLoss="bce_dice_cldice_boundary"; ClDice="0.3"; Boundary="0.3"; CbDice="0.5";
        }
    )
}

foreach ($exp in $experiments) {
    $weightsTag = if ([string]::IsNullOrWhiteSpace($EncoderWeights)) { "scratch" } else { $EncoderWeights }
    $display = "$($exp.Name)_$weightsTag"
    $expName = "all_filtered/$display"
    $log = Join-Path $LogRoot "$display.log"

    $args = @(
        "train_smp_joint.py",
        "--arch", $exp.Arch,
        "--encoder_name", $exp.Encoder,
        "--dataset", "all_filtered",
        "--teacher_mode", $exp.Teacher,
        "--enhancer", $exp.Enhancer,
        "--joint_model", $exp.Joint,
        "--loss_weighting", "fixed",
        "--lambda_mse", "10.0",
        "--lambda_grad", "0.0",
        "--seg_loss", $exp.SegLoss,
        "--cldice_weight", $exp.ClDice,
        "--boundary_weight", $exp.Boundary,
        "--cbdice_weight", $exp.CbDice,
        "--epochs", "$Epochs",
        "--patience", "$Patience",
        "--batch_size", "$BatchSize",
        "--lr", "$Lr",
        "--seed", "$Seed",
        "--exp_name", $expName
    )
    if (-not [string]::IsNullOrWhiteSpace($EncoderWeights)) {
        $args += @("--encoder_weights", $EncoderWeights)
    }

    Invoke-LoggedPython -StepName "Train $display" -Arguments $args -LogPath $log

    $aggregateCsv = Get-LatestAggregateCsv -ExpName $expName
    $metrics = Import-Csv $aggregateCsv | Select-Object -First 1
    $line = '"{0}","{1}","{2}","{3}",{4},"{5}","{6}","{7}",10.0,0.0,"{8}",{9},{10},{11},{12},{13},{14},{15},{16},{17},{18},{19},{20},"{21}","{22}"' -f $display, $exp.Arch, $exp.Encoder, $weightsTag, $Seed, $exp.Teacher, $exp.Enhancer, $exp.Joint, $exp.SegLoss, $exp.ClDice, $exp.Boundary, $exp.CbDice, $metrics.dice, $metrics.iou, $metrics.sensitivity, $metrics.precision, $metrics.specificity, $metrics.accuracy, $metrics.hd95, $metrics.cldice, $metrics.boundary_f1, $aggregateCsv, $log
    Add-Content -Encoding UTF8 -Path $SummaryPath -Value $line
}

Write-Host ""
Write-Host "============================================================"
Write-Host "[ALL DONE]"
Write-Host "Metrics: $SummaryPath"
Write-Host "Logs:    $LogRoot"
Write-Host "============================================================"
