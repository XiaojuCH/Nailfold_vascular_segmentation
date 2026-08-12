param(
    [Parameter(Mandatory = $true)]
    [string]$DataDir,
    [string]$Split = "test",
    [string]$Python = "D:\anaconda3\envs\pytorch\python.exe",
    [string]$Weight = "reference_weights\K2_uniform_lambda1p0_seed42_best_model.pth",
    [string]$OutputRoot = "outputs\reference_evaluation",
    [int]$BatchSize = 4
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

if ($Split -notin @("train", "val", "test")) { throw "Split must be train, val, or test." }
if (-not (Test-Path -LiteralPath $Python)) { throw "Python not found: $Python" }
if (-not (Test-Path -LiteralPath $DataDir)) { throw "Dataset root not found: $DataDir" }
$Weight = Resolve-K2Path $Weight
if (-not (Test-Path -LiteralPath $Weight)) { throw "Reference K2 checkpoint not found: $Weight" }

$DataDir = (Resolve-Path -LiteralPath $DataDir).Path
$OutputRoot = Resolve-K2Path $OutputRoot
$LogRoot = Join-Path $OutputRoot "logs"
New-Item -ItemType Directory -Force -Path $LogRoot | Out-Null

function ConvertTo-WindowsCommandLineArgument {
    param([string]$Value)
    if ($Value -notmatch '[\s"]') { return $Value }
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
    Write-Host "[START] $Name"
    Write-Host "[LOG]   $logPath"
    $process = Start-Process -FilePath $Python -ArgumentList $argumentLine -NoNewWindow -Wait -PassThru `
        -RedirectStandardOutput $stdoutPath -RedirectStandardError $stderrPath
    @(
        "[K2 reference evaluation step] $Name"
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

Invoke-Step -Name "00_dataset_audit" -Arguments @(
    "code\audit_dataset.py", "--data_dir", $DataDir
)
Invoke-Step -Name "01_evaluate_reference_k2" -Arguments @(
    "code\evaluate_k2.py", "--data_dir", $DataDir, "--weight", $Weight, "--split", $Split,
    "--out_dir", $OutputRoot, "--name", "Reference_K2_seed42", "--img_size", "256",
    "--batch_size", "$BatchSize", "--threshold", "0.5"
)
