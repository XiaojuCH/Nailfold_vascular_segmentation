param(
    [string]$EnvironmentName = "mmseg_official"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

$Conda = "D:\anaconda3\Scripts\conda.exe"
$Python = "D:\anaconda3\envs\$EnvironmentName\python.exe"
if (-not (Test-Path $Conda)) { throw "Conda not found: $Conda" }

if (-not (Test-Path $Python)) {
    & $Conda create -y -n $EnvironmentName python=3.10 pip
    if ($LASTEXITCODE -ne 0) { throw "Failed to create Conda environment: $EnvironmentName" }
}

& $Python -m pip install --index-url https://download.pytorch.org/whl/cu118 torch==2.1.2 torchvision==0.16.2
if ($LASTEXITCODE -ne 0) { throw "Failed to install PyTorch" }

& $Python -m pip install "setuptools<81" "numpy<2" "mmengine>=0.5.0,<1.0.0" ftfy prettytable scipy scikit-image opencv-python tqdm
if ($LASTEXITCODE -ne 0) { throw "Failed to install MMSeg runtime dependencies" }

& $Python -m pip install --only-binary=:all: "mmcv==2.1.0" -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.1.0/index.html
if ($LASTEXITCODE -ne 0) { throw "Failed to install MMCV CUDA wheel" }

& $Python -m pip install -e "TYT_Code\mmsegmentation-main" --no-deps
if ($LASTEXITCODE -ne 0) { throw "Failed to install the supplied MMSegmentation source" }

& $Python -c "import torch, mmcv, mmengine, mmseg; from mmcv.ops import point_sample; print('torch', torch.__version__, 'cuda', torch.cuda.is_available()); print('mmcv', mmcv.__version__, 'mmengine', mmengine.__version__, 'mmseg', mmseg.__version__)"
if ($LASTEXITCODE -ne 0) { throw "MMSeg environment validation failed" }

Write-Host "[DONE] Isolated MMSeg environment: $EnvironmentName"
