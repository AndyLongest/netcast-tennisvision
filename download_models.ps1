param(
    [string]$SourceDirectory
)

$ErrorActionPreference = "Stop"
$repoRoot = $PSScriptRoot
$python = Join-Path $repoRoot ".venv\Scripts\python.exe"

if (-not (Test-Path -LiteralPath $python)) {
    throw "尚未建立 Python 环境。请先运行 .\setup.ps1；它也会自动下载模型。"
}

$arguments = @("tools\install_assets.py", "--group", "runtime")
if ($SourceDirectory) {
    $arguments += @("--source-dir", (Resolve-Path -LiteralPath $SourceDirectory).Path)
}

& $python @arguments
if ($LASTEXITCODE -ne 0) {
    throw "模型安装失败。请检查网络后重试，或阅读 assets\MODELS.md 使用离线安装。"
}

Write-Host "三个生产模型均已安装并通过 SHA-256 校验。" -ForegroundColor Green
