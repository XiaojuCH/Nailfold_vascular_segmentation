param(
    [string]$Python = "D:\anaconda3\envs\pytorch\python.exe",
    [switch]$SkipExisting
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$ProjectRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $ProjectRoot

$DecisionPath = "results\dual_teacher_kd_20260717\first_night_decision.json"
if (-not (Test-Path -LiteralPath $DecisionPath)) {
    throw "First-stage decision is missing: $DecisionPath. Finish run_dual_teacher_kd_20260717.ps1 first."
}

$decision = Get-Content -Raw -LiteralPath $DecisionPath | ConvertFrom-Json
if ($decision.next_phase -eq "multiseed") {
    & (Join-Path $PSScriptRoot "run_dual_teacher_kd_multiseed_20260717.ps1") -Python $Python -SkipExisting:$SkipExisting
}
elseif ($decision.next_phase -eq "fallback_k3_k4") {
    & (Join-Path $PSScriptRoot "run_dual_teacher_kd_20260717.ps1") -Python $Python -RunFallback -SkipExisting:$SkipExisting
}
else {
    throw "Unknown next phase in decision file: $($decision.next_phase)"
}
