param(
    [ValidateSet("cuda", "cpu")]
    [string]$Accelerator = "cuda",
    [string]$EnvironmentPath = "",
    [switch]$SkipModels
)

$ErrorActionPreference = "Stop"
$repoRoot = $PSScriptRoot
if (-not $EnvironmentPath) {
    $EnvironmentPath = Join-Path (Split-Path $repoRoot -Parent) ".tools\tennis-racketpose-env"
}
$environmentPython = Join-Path $EnvironmentPath "python.exe"

if (-not (Get-Command conda -ErrorAction SilentlyContinue)) {
    throw "RacketPose 的隔离安装需要 Conda。"
}
if (-not (Test-Path -LiteralPath $environmentPython)) {
    & conda create -p $EnvironmentPath python=3.10 pip -y
    if ($LASTEXITCODE -ne 0) { throw "无法建立 RacketPose Python 3.10 环境。" }
}

$pytorchIndex = if ($Accelerator -eq "cuda") {
    "https://download.pytorch.org/whl/cu121"
} else {
    "https://download.pytorch.org/whl/cpu"
}
$mmcvIndex = if ($Accelerator -eq "cuda") {
    "https://download.openmmlab.com/mmcv/dist/cu121/torch2.1.0/index.html"
} else {
    "https://download.openmmlab.com/mmcv/dist/cpu/torch2.1.0/index.html"
}

& $environmentPython -m pip install torch==2.1.2 torchvision==0.16.2 --index-url $pytorchIndex
if ($LASTEXITCODE -ne 0) { throw "无法安装 RacketPose $Accelerator 版 PyTorch。" }

# chumpy 0.70 predates modern isolated builds and imports pip from setup.py.
& $environmentPython -m pip install "setuptools<81" "numpy==1.26.4"
if ($LASTEXITCODE -ne 0) { throw "无法安装 RacketPose 基础兼容包。" }
& $environmentPython -m pip install "chumpy==0.70" --no-build-isolation
if ($LASTEXITCODE -ne 0) { throw "无法安装 MMPose 的 chumpy 兼容依赖。" }
& $environmentPython -m pip install "mmcv==2.1.0" --find-links $mmcvIndex --no-deps
if ($LASTEXITCODE -ne 0) { throw "无法安装与 $Accelerator 匹配的 MMCV。" }

& $environmentPython -m pip install -r (Join-Path $repoRoot "requirements-racketpose.txt")
if ($LASTEXITCODE -ne 0) { throw "无法安装 RacketPose OpenMMLab 依赖。" }
& $environmentPython -m pip install --editable $repoRoot --no-deps
if ($LASTEXITCODE -ne 0) { throw "无法把 TennisVision 接入 RacketPose 环境。" }

if (-not $SkipModels) {
    & (Join-Path $repoRoot ".venv\Scripts\python.exe") (Join-Path $repoRoot "tools\install_assets.py") --group racket_pose
    if ($LASTEXITCODE -ne 0) { throw "RacketPose 模型下载或校验失败。" }
}

Write-Host "RacketPose $Accelerator 环境已就绪：$EnvironmentPath" -ForegroundColor Green
