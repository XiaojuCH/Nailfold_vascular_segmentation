param(
    [Parameter(Mandatory = $true)]
    [string]$DataDir,
    [string]$Python = "D:\anaconda3\envs\pytorch\python.exe",
    [string]$OutputRoot = "outputs",
    [string]$Pretrained = "reference_weights\R50+ViT-B_16.npz",
    [int]$Seed = 42,
    [int]$F0Epochs = 50,
    [int]$F3Epochs = 50,
    [int]$K2Epochs = 30,
    [int]$Patience = 20,
    [int]$K2Patience = 10,
    [int]$BatchSize = 4,
    [double]$F0F3Lr = 1e-4,
    [double]$K2Lr = 3e-5,
    [switch]$IncludeK0Control,
    [switch]$EvaluateTest,
    [switch]$SkipExisting
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$PackageRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $PackageRoot

function Resolve-K2Path {
    param([string]$Path)
    if ([System.IO.Path]::IsPathRooted($Path)) {
        return [System.IO.Path]::GetFullPath($Path)
    }
    return [System.IO.Path]::GetFullPath((Join-Path $PackageRoot $Path))
}

if (-not (Test-Path -LiteralPath $Python)) { throw "Python not found: $Python" }
if (-not (Test-Path -LiteralPath $DataDir)) { throw "Dataset root not found: $DataDir" }
$Pretrained = Resolve-K2Path $Pretrained
if (-not (Test-Path -LiteralPath $Pretrained)) { throw "ImageNet21k initialization not found: $Pretrained" }

$OutputRoot = Resolve-K2Path $OutputRoot
$DataDir = (Resolve-Path -LiteralPath $DataDir).Path
New-Item -ItemType Directory -Force -Path $OutputRoot | Out-Null
$LogRoot = Join-Path $OutputRoot "logs"
New-Item -ItemType Directory -Force -Path $LogRoot | Out-Null

function ConvertTo-WindowsCommandLineArgument {
    param([string]$Value)
    if ($Value -notmatch '[\s"]') { return $Value }

    # Escape embedded quotes and trailing backslashes according to the Windows command-line rules.
    $escaped = [regex]::Replace($Value, '(\\*)"', '$1$1\"')
    $escaped = [regex]::Replace($escaped, '(\\+)$', '$1$1')
    return '"' + $escaped + '"'
}

function Invoke-Step {
    param([string]$Name, [string[]]$Arguments)
    $logPath = Join-Path $LogRoot ("{0}.log" -f $Name)
    $stdoutPath = Join-Path $LogRoot ("{0}.stdout.log" -f $Name)
    $stderrPath = Join-Path $LogRoot ("{0}.stderr.log" -f $Name)
    $argumentLine = (($Arguments | ForEach-Object { ConvertTo-WindowsCommandLineArgument $_ }) -join " ")
    Write-Host ""
    Write-Host "============================================================"
    Write-Host "[START] $Name"
    Write-Host "[LOG]   $logPath"
    Write-Host "============================================================"
    Write-Host "[RUNNING] Standard output/error are captured separately to avoid PowerShell treating tqdm stderr as a failure."
    $process = Start-Process -FilePath $Python -ArgumentList $argumentLine -NoNewWindow -Wait -PassThru `
        -RedirectStandardOutput $stdoutPath -RedirectStandardError $stderrPath

    @(
        "[K2 pipeline step] $Name"
        "[Exit code] $($process.ExitCode)"
        ""
        "[stdout]"
        if (Test-Path -LiteralPath $stdoutPath) { Get-Content -LiteralPath $stdoutPath }
        ""
        "[stderr]"
        if (Test-Path -LiteralPath $stderrPath) { Get-Content -LiteralPath $stderrPath }
    ) | Set-Content -LiteralPath $logPath -Encoding UTF8

    if ($process.ExitCode -ne 0) { throw "Failed step '$Name' (exit code $($process.ExitCode)). See: $logPath" }
}

function Test-CompleteTrainingRun {
    param([string]$RunDir)
    return (Test-Path -LiteralPath (Join-Path $RunDir "best_model.pth")) -and `
        (Test-Path -LiteralPath (Join-Path $RunDir "config.json")) -and `
        (Test-Path -LiteralPath (Join-Path $RunDir "val_per_image.csv"))
}

function Invoke-TrainingStep {
    param([string]$Name, [string]$RunDir, [string[]]$Arguments)
    if (Test-Path -LiteralPath $RunDir) {
        if (Test-CompleteTrainingRun $RunDir) {
            if ($SkipExisting) {
                Write-Host "[SKIP] $Name already has best_model.pth, config.json, and val_per_image.csv"
                return
            }
            throw "Completed run already exists: $RunDir. Use -SkipExisting to reuse it or choose a new -OutputRoot."
        }
        throw "Incomplete run directory exists: $RunDir. It will not be overwritten. Inspect it, then use a new -OutputRoot or remove it manually after confirmation."
    }
    Invoke-Step -Name $Name -Arguments $Arguments
    if (-not (Test-CompleteTrainingRun $RunDir)) {
        throw "Training step '$Name' exited successfully but its expected outputs are incomplete: $RunDir"
    }
}

function Get-ImageStems {
    param([string]$Split)
    $imageDir = Join-Path $DataDir "$Split\images"
    return @(
        Get-ChildItem -LiteralPath $imageDir -File |
            Where-Object { $_.Extension.ToLowerInvariant() -in @(".png", ".jpg", ".jpeg", ".bmp") } |
            ForEach-Object { [System.IO.Path]::GetFileNameWithoutExtension($_.Name) } |
            Sort-Object
    )
}

function Test-CompleteTargets {
    param([string]$TargetDir)
    $metadataPath = Join-Path $TargetDir "metadata.json"
    if (-not (Test-Path -LiteralPath $metadataPath)) { return $false }
    try { $metadata = Get-Content -LiteralPath $metadataPath -Raw | ConvertFrom-Json }
    catch { return $false }
    if ($metadata.f0_weight -ne [System.IO.Path]::GetFullPath($F0Weight) -or
        $metadata.f3_weight -ne [System.IO.Path]::GetFullPath($F3Weight) -or
        $metadata.f3_variant -ne "directional_multiscale") { return $false }
    foreach ($split in @("train", "val")) {
        $stems = Get-ImageStems $split
        $record = @($metadata.splits | Where-Object { $_.split -eq $split })
        if ($record.Count -ne 1 -or [int]$record[0].images -ne $stems.Count) { return $false }
        foreach ($stem in $stems) {
            if (-not (Test-Path -LiteralPath (Join-Path $TargetDir "$split\ensemble_probabilities\$stem.npy")) -or
                -not (Test-Path -LiteralPath (Join-Path $TargetDir "$split\disagreement\$stem.npy"))) { return $false }
        }
    }
    return $true
}

Invoke-Step -Name "00_dataset_audit" -Arguments @(
    "code\audit_dataset.py", "--data_dir", $DataDir,
    "--out", (Join-Path $OutputRoot "dataset_audit.json")
)

$F0Dir = Join-Path $OutputRoot "F0_seed$Seed"
$F3Dir = Join-Path $OutputRoot "F3_seed$Seed"
$K0Dir = Join-Path $OutputRoot "K0_seed$Seed"
$K2Dir = Join-Path $OutputRoot "K2_seed$Seed"
$TargetRoot = Join-Path $OutputRoot "dual_teacher_targets"
$F0Weight = Join-Path $F0Dir "best_model.pth"
$F3Weight = Join-Path $F3Dir "best_model.pth"
$K2Weight = Join-Path $K2Dir "best_model.pth"

Invoke-TrainingStep -Name "01_train_F0_rgb_teacher" -RunDir $F0Dir -Arguments @(
    "code\train_k2.py", "--stage", "f0", "--data_dir", $DataDir,
    "--output_dir", $OutputRoot, "--exp_name", "F0_seed$Seed",
    "--pretrained", $Pretrained, "--epochs", "$F0Epochs", "--patience", "$Patience",
    "--batch_size", "$BatchSize", "--lr", "$F0F3Lr", "--seed", "$Seed", "--intensity_aug", "on"
)

Invoke-TrainingStep -Name "02_train_F3_green_morphology_teacher" -RunDir $F3Dir -Arguments @(
    "code\train_k2.py", "--stage", "f3", "--data_dir", $DataDir,
    "--output_dir", $OutputRoot, "--exp_name", "F3_seed$Seed",
    "--f3_variant", "directional_multiscale", "--epochs", "$F3Epochs", "--patience", "$Patience",
    "--batch_size", "$BatchSize", "--lr", "$F0F3Lr", "--seed", "$Seed", "--intensity_aug", "on"
)

if (Test-Path -LiteralPath $TargetRoot) {
    if ((Test-CompleteTargets $TargetRoot) -and $SkipExisting) {
        Write-Host "[SKIP] 03_generate_dual_teacher_soft_targets has complete metadata and train/val target maps"
    } elseif (Test-CompleteTargets $TargetRoot) {
        throw "Completed soft targets already exist: $TargetRoot. Use -SkipExisting to reuse them or choose a new -OutputRoot."
    } else {
        throw "Incomplete soft-target directory exists: $TargetRoot. It will not be overwritten automatically."
    }
} else {
    Invoke-Step -Name "03_generate_dual_teacher_soft_targets" -Arguments @(
        "code\generate_dual_teacher_targets.py", "--data_dir", $DataDir, "--splits", "train,val",
        "--f0_weight", $F0Weight, "--f3_weight", $F3Weight,
        "--f3_variant", "directional_multiscale", "--out_dir", $TargetRoot,
        "--batch_size", "$BatchSize", "--img_size", "256", "--threshold", "0.5"
    )
}

if ($IncludeK0Control) {
    Invoke-TrainingStep -Name "04_train_K0_finetune_control" -RunDir $K0Dir -Arguments @(
        "code\train_k2.py", "--stage", "k2", "--data_dir", $DataDir,
        "--output_dir", $OutputRoot, "--exp_name", "K0_seed$Seed",
        "--pretrained", $Pretrained, "--init_weight", $F0Weight,
        "--soft_target_dir", (Join-Path $TargetRoot "train\ensemble_probabilities"),
        "--lambda_kd", "0", "--epochs", "$K2Epochs", "--patience", "$K2Patience",
        "--batch_size", "$BatchSize", "--lr", "$K2Lr", "--seed", "$Seed", "--intensity_aug", "on"
    )
}

Invoke-TrainingStep -Name "05_train_K2_dual_teacher_student" -RunDir $K2Dir -Arguments @(
    "code\train_k2.py", "--stage", "k2", "--data_dir", $DataDir,
    "--output_dir", $OutputRoot, "--exp_name", "K2_seed$Seed",
    "--pretrained", $Pretrained, "--init_weight", $F0Weight,
    "--soft_target_dir", (Join-Path $TargetRoot "train\ensemble_probabilities"),
    "--lambda_kd", "1.0", "--epochs", "$K2Epochs", "--patience", "$K2Patience",
    "--batch_size", "$BatchSize", "--lr", "$K2Lr", "--seed", "$Seed", "--intensity_aug", "on"
)

$EvalSplit = if ($EvaluateTest) { "test" } else { "val" }
Invoke-Step -Name "06_evaluate_K2_$EvalSplit" -Arguments @(
    "code\evaluate_k2.py", "--data_dir", $DataDir, "--weight", $K2Weight,
    "--split", $EvalSplit, "--out_dir", (Join-Path $OutputRoot "evaluation"),
    "--name", "K2_seed$Seed", "--img_size", "256", "--batch_size", "$BatchSize", "--threshold", "0.5"
)

Write-Host ""
Write-Host "[ALL DONE]"
Write-Host "F0 teacher: $F0Weight"
Write-Host "F3 teacher: $F3Weight"
if ($IncludeK0Control) { Write-Host "K0 control:  $(Join-Path $K0Dir 'best_model.pth')" }
Write-Host "K2 student:  $K2Weight"
Write-Host "Targets:     $TargetRoot"
Write-Host "Evaluation:  $(Join-Path $OutputRoot 'evaluation')"
