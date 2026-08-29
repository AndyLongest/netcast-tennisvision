param(
    [ValidateSet("auto", "cuda", "cpu")]
    [string]$Accelerator = "auto",
    [switch]$RuntimeOnly,
    [string]$AssetSource
)

$ErrorActionPreference = "Stop"
$repoRoot = $PSScriptRoot
$venvDirectory = Join-Path $repoRoot ".venv"
$venvPython = Join-Path $venvDirectory "Scripts\python.exe"

if (-not (Test-Path -LiteralPath $venvPython)) {
    $bootstrap = Get-Command py -ErrorAction SilentlyContinue
    if ($bootstrap) {
        & $bootstrap.Source -3.12 -m venv $venvDirectory
    }
    else {
        $bootstrap = Get-Command python -ErrorAction Stop
        & $bootstrap.Source -m venv $venvDirectory
    }
}

& $venvPython -m pip install --upgrade pip

$selectedAccelerator = $Accelerator
if ($selectedAccelerator -eq "auto") {
    $selectedAccelerator = if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) { "cuda" } else { "cpu" }
}

if ($selectedAccelerator -eq "cuda") {
    & $venvPython -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
}
else {
    & $venvPython -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cpu
}

$requirements = if ($RuntimeOnly) { "requirements.txt" } else { "requirements-dev.txt" }
& $venvPython -m pip install -r (Join-Path $repoRoot $requirements)

$assetArguments = @("tools\install_assets.py", "--group", "runtime")
if ($AssetSource) {
    $assetArguments += @("--source-dir", (Resolve-Path -LiteralPath $AssetSource).Path)
}
& $venvPython @assetArguments
if ($LASTEXITCODE -ne 0) {
    throw "模型参数下载或校验失败。请查看 assets\MODELS.md"
}

& $venvPython "tools\verify_install.py"
if ($LASTEXITCODE -ne 0) {
    throw "环境自检失败，请按上面的提示修复后重试。"
}

Write-Host "Netcast TennisVision 安装完成。运行 .\run_ui.ps1 即可启动。" -ForegroundColor Green
