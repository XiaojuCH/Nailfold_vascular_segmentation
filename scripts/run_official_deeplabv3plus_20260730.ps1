param(
    [ValidateSet("prepare", "smoke", "train", "evaluate")]
    [string]$Stage = "train",
    [string]$Checkpoint = "",
    [switch]$SkipDatasetPreparation,
    [switch]$Resume
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot
# MMEngine collects compiler metadata during Runner startup. On Chinese Windows
# locales, forcing UTF-8 avoids decoding the MSVC version as GBK.
$env:PYTHONUTF8 = "1"

$Python = "D:\anaconda3\envs\mmseg_official\python.exe"
$MMSegRoot = Join-Path $ProjectRoot "TYT_Code\mmsegmentation-main"
$Config = Join-Path $ProjectRoot "configs\deeplabv3plus_mobilenetv2_official_all_filtered.py"
$DataView = Join-Path $ProjectRoot "dataset_all_filtered_mmseg"
$RunRoot = Join-Path $ProjectRoot "results\official_deeplabv3plus_20260730"
$WorkDir = Join-Path $RunRoot "work_dirs\seed42_scratch_10k"

if (-not (Test-Path $Python)) { throw "Python environment not found: $Python" }
if (-not (Test-Path $MMSegRoot)) { throw "MMSeg source not found: $MMSegRoot" }
if (-not (Test-Path $Config)) { throw "Config not found: $Config" }
New-Item -ItemType Directory -Force -Path $RunRoot | Out-Null

& $Python -c "import torch, mmcv, mmengine, mmseg; from mmcv.ops import point_sample"
if ($LASTEXITCODE -ne 0) {
    throw "The isolated MMSeg environment is incomplete. Run .\scripts\setup_official_mmseg_env.ps1 first."
}

function Invoke-PythonStep {
    param([string]$Name, [string[]]$Arguments)
    Write-Host ""
    Write-Host "============================================================"
    Write-Host "[START] $Name"
    Write-Host "============================================================"
    & $Python @Arguments
    if ($LASTEXITCODE -ne 0) { throw "$Name failed with exit code $LASTEXITCODE" }
}

if (-not $SkipDatasetPreparation) {
    Invoke-PythonStep -Name "Prepare MMSeg 0/1 label view" -Arguments @(
        "dataset_tools\prepare_mmseg_binary_dataset.py", "--source", "dataset_all_filtered", "--output", "dataset_all_filtered_mmseg"
    )
}

$env:PYTHONPATH = "$MMSegRoot;$env:PYTHONPATH"
$env:JIABI_MMSEG_DATA_ROOT = $DataView

if ($Stage -eq "prepare") {
    Write-Host "[DONE] Dataset view prepared: $DataView"
    exit 0
}

if ($Stage -eq "smoke") {
    Invoke-PythonStep -Name "MMSeg config/dataset smoke test" -Arguments @(
        "TYT_Code\mmsegmentation-main\tools\misc\print_config.py", $Config
    )
    Write-Host "[DONE] Config parses. Training was not started."
    exit 0
}

if ($Stage -eq "train") {
    $trainArguments = @("TYT_Code\mmsegmentation-main\tools\train.py", $Config, "--work-dir", $WorkDir)
    if ($Resume) { $trainArguments += "--resume" }
    Invoke-PythonStep -Name "Official MMSeg DeepLabV3+ train (seed42, scratch, 10k iters)" -Arguments $trainArguments
    $best = Get-ChildItem -Path $WorkDir -Filter "best_mDice_*.pth" | Sort-Object LastWriteTime -Descending | Select-Object -First 1
    if ($null -eq $best) { throw "No best_mDice checkpoint found in $WorkDir" }
    $Checkpoint = $best.FullName
}

if ($Stage -eq "evaluate") {
    if ([string]::IsNullOrWhiteSpace($Checkpoint)) {
        $best = Get-ChildItem -Path $WorkDir -Filter "best_mDice_*.pth" | Sort-Object LastWriteTime -Descending | Select-Object -First 1
        if ($null -eq $best) { throw "Pass -Checkpoint or train first; no best_mDice checkpoint found." }
        $Checkpoint = $best.FullName
    }
}

if (-not (Test-Path $Checkpoint)) { throw "Checkpoint not found: $Checkpoint" }
Invoke-PythonStep -Name "Unified development-test evaluation" -Arguments @(
    "evaluate_mmseg_deeplabv3plus.py", "--config", $Config, "--checkpoint", $Checkpoint,
    "--data_dir", "dataset_all_filtered", "--split", "test",
    "--out_dir", (Join-Path $RunRoot "unified_eval_seed42"), "--device", "cuda:0"
)

Write-Host ""
Write-Host "============================================================"
Write-Host "[ALL DONE]"
Write-Host "Checkpoint: $Checkpoint"
Write-Host "Metrics:    $RunRoot\unified_eval_seed42\aggregate_metrics.csv"
Write-Host "============================================================"
