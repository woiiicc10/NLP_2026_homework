# download_assets.ps1
# 下载 GloVe 100d 和 BERT-base-uncased（经 ModelScope 镜像），并放入 cache。
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$Cache = Join-Path $Root "cache"
$GloveDir = Join-Path $Cache "glove"
$GloveTxt = Join-Path $GloveDir "glove.6B.100d.txt"
$BertDir = Join-Path $Cache "bert-base-uncased-local"

New-Item -ItemType Directory -Force -Path $Cache, $GloveDir, $BertDir | Out-Null

# 1) GloVe 6B zip -> 仅解压 100d
if (-not (Test-Path -LiteralPath $GloveTxt)) {
    $Zip = Join-Path $Cache "glove.6B.zip"
    if (-not (Test-Path -LiteralPath $Zip)) {
        Write-Host "Downloading GloVe 6B ..."
        & curl.exe -L --fail --retry 3 --output $Zip "https://nlp.stanford.edu/data/glove.6B.zip"
        if ($LASTEXITCODE -ne 0) { throw "GloVe download failed" }
    }
    Add-Type -AssemblyName System.IO.Compression.FileSystem
    $archive = [System.IO.Compression.ZipFile]::OpenRead($Zip)
    try {
        $entry = $archive.GetEntry("glove.6B.100d.txt")
        if ($null -eq $entry) { throw "glove.6B.100d.txt missing in zip" }
        [System.IO.Compression.ZipFileExtensions]::ExtractToFile($entry, $GloveTxt, $true)
    } finally { $archive.Dispose() }
    if ((Get-Item -LiteralPath $GloveTxt).Length -lt 300MB) { throw "unexpected GloVe file size" }
    Remove-Item -LiteralPath $Zip -Force
    Write-Host "GloVe ready: $GloveTxt"
} else {
    Write-Host "GloVe already exists: $GloveTxt"
}

# 2) BERT 权重（与 google-bert/bert-base-uncased 相同）
$BertModel = Join-Path $BertDir "model.safetensors"
if (-not (Test-Path -LiteralPath $BertModel)) {
    Write-Host "Downloading bert-base-uncased from ModelScope ..."
    $Base = "https://modelscope.cn/api/v1/models/AI-ModelScope/bert-base-uncased/repo?Revision=master&FilePath="
    foreach ($f in @("config.json","vocab.txt","tokenizer_config.json","tokenizer.json","model.safetensors")) {
        $out = Join-Path $BertDir $f
        if (-not (Test-Path -LiteralPath $out)) {
            & curl.exe -L --fail --retry 3 --output $out ($Base + $f)
            if ($LASTEXITCODE -ne 0) { throw "BERT download failed: $f" }
        }
    }
    Write-Host "BERT ready: $BertModel"
} else {
    Write-Host "BERT already exists: $BertModel"
}
