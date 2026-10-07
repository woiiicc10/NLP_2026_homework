# Bootstrap.ps1
# 创建项目虚拟环境并安装全部依赖。
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$Venv = Join-Path $Root ".venv-bert"

function Test-PythonVersion {
    param([string]$Executable, [string[]]$Arguments)
    $script = "import sys; print(sys.version_info.major, sys.version_info.minor, sys.executable)"
    $output = & $Executable @Arguments -c $script 2>$null
    if ($LASTEXITCODE -ne 0 -or [string]::IsNullOrWhiteSpace($output)) { return $null }
    $parts = ($output -split '\s+')
    if ($parts.Count -lt 2) { return $null }
    [int]$major = 0; [int]$minor = 0
    if (-not [int]::TryParse($parts[0], [ref]$major)) { return $null }
    if (-not [int]::TryParse($parts[1], [ref]$minor)) { return $null }
    return @{ Major = $major; Minor = $minor; Executable = ($parts[2..($parts.Count-1)] -join ' ') }
}

# 1) 优先使用 Windows Python Launcher 寻找 3.13，其次任意 3.x
$candidates = @()
$py = Get-Command py -ErrorAction SilentlyContinue
if ($py) {
    foreach ($version in @('-3.13','-3')) {
        $candidates += @{ Executable = 'py'; Arguments = @($version) }
    }
}
$python = Get-Command python -ErrorAction SilentlyContinue
if ($python) {
    $candidates += @{ Executable = 'python'; Arguments = @() }
}

$BasePython = $null
$SelectedInfo = $null
foreach ($candidate in $candidates) {
    $info = Test-PythonVersion -Executable $candidate.Executable -Arguments $candidate.Arguments
    if ($null -ne $info -and $info.Major -eq 3 -and $info.Minor -ge 10) {
        $BasePython = $info.Executable
        $SelectedInfo = $info
        break
    }
}

if ($null -eq $BasePython) {
    throw "未找到 Python 3.10+ 解释器。请安装 Python 3.13 后重试，或手动设置 scripts/bootstrap.ps1 中的 BasePython。"
}

Write-Host "Using base Python $($SelectedInfo.Major).$($SelectedInfo.Minor): $BasePython"

if (-not (Test-Path -LiteralPath (Join-Path $Venv "Scripts\python.exe"))) {
    & $BasePython -m venv --copies $Venv
    if ($LASTEXITCODE -ne 0) { throw "venv creation failed" }
}

$Python = Join-Path $Venv "Scripts\python.exe"
& $Python -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) { throw "pip upgrade failed" }

# CPU/通用依赖
& $Python -m pip install -r (Join-Path $Root "requirements.txt")
if ($LASTEXITCODE -ne 0) { throw "requirements install failed" }

# CUDA 版 PyTorch。无 NVIDIA GPU 的机器可改为 CPU 版：
#   & $Python -m pip install "torch==2.11.0"
& $Python -m pip install --index-url https://download.pytorch.org/whl/cu128 "torch==2.11.0+cu128"
if ($LASTEXITCODE -ne 0) { throw "torch install failed" }

Write-Host "Environment ready: $Python"
