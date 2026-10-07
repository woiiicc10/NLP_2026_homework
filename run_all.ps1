# run_all.ps1
# 一键运行：准备环境 -> 下载资源 -> Task1 -> Task2 -> Task3。
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$Venv = Join-Path $Root ".venv-bert"
$Python = Join-Path $Venv "Scripts\python.exe"

if (-not (Test-Path -LiteralPath $Python)) {
    Write-Host "Environment not found; bootstrapping ..."
    & (Join-Path $Root "scripts\bootstrap.ps1")
}

& (Join-Path $Root "scripts\download_assets.ps1")

$env:PYTHONIOENCODING = "utf-8"
$env:HF_ENDPOINT = "https://hf-mirror.com"
$env:HF_HUB_DISABLE_XET = "1"
$env:HF_XET_DISABLE = "1"

Write-Host "`n===== Task 1: Bag-of-Words ====="
& $Python (Join-Path $Root "src\task1_bow.py")
if ($LASTEXITCODE -ne 0) { throw "Task 1 failed" }

Write-Host "`n===== Task 2: Word2Vec / GloVe ====="
& $Python (Join-Path $Root "src\task2_word2vec.py")
if ($LASTEXITCODE -ne 0) { throw "Task 2 failed" }

Write-Host "`n===== Task 3: BERT ====="
$BertLocal = Join-Path $Root "cache\bert-base-uncased-local"
& $Python (Join-Path $Root "src\task3_bert.py") --model-name $BertLocal
if ($LASTEXITCODE -ne 0) { throw "Task 3 failed" }

Write-Host "`nAll experiments completed. Results are in outputs/."
