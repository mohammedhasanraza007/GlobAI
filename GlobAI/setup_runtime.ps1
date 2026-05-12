# =============================================================
# GlobAI — Portable Runtime Bootstrap (CLEAN VERSION)
# =============================================================

$ErrorActionPreference = "Stop"

[Net.ServicePointManager]::SecurityProtocol =
    [Net.SecurityProtocolType]::Tls12 -bor
    [Net.SecurityProtocolType]::Tls11

$Root        = Split-Path -Parent $MyInvocation.MyCommand.Definition
$Runtime     = Join-Path $Root "runtime"
$Python      = Join-Path $Runtime "python.exe"
$PthFile     = Join-Path $Runtime "python310._pth"
$GetPipPath  = Join-Path $Runtime "get-pip.py"
$Requirements= Join-Path $Root "requirements.txt"

$PyVersion   = "3.10.6"
$PyZipName   = "python-3.10.6-embed-amd64.zip"
$PyUrl       = "https://www.python.org/ftp/python/3.10.6/$PyZipName"
$GetPipUrl   = "https://bootstrap.pypa.io/get-pip.py"
$TempZip     = Join-Path $Root "_py.zip"

Write-Host "============================================================"
Write-Host " GlobAI Runtime Setup"
Write-Host "============================================================"
Write-Host ""

if (Test-Path $Python) {
    Write-Host "[OK] Runtime already exists."
}
else {

    Write-Host "[DOWNLOAD] Python 3.10.6..."

    $ProgressPreference = "SilentlyContinue"

    Invoke-WebRequest -Uri $PyUrl -OutFile $TempZip

    if (!(Test-Path $Runtime)) {
        New-Item -ItemType Directory -Path $Runtime | Out-Null
    }

    Expand-Archive -Path $TempZip -DestinationPath $Runtime -Force

    Remove-Item $TempZip -Force -ErrorAction SilentlyContinue

    if (!(Test-Path $Python)) {
        Write-Host "[ERROR] python.exe missing after extract"
        exit 1
    }

    Write-Host "[OK] Python extracted"

    if (Test-Path $PthFile) {
        $txt = Get-Content $PthFile -Raw
        $txt = $txt.Replace("#import site", "import site")
        Set-Content $PthFile $txt -NoNewline
        Write-Host "[OK] site-packages enabled"
    }

    Write-Host "[DOWNLOAD] get-pip.py..."

    Invoke-WebRequest -Uri $GetPipUrl -OutFile $GetPipPath

    & $Python $GetPipPath

    & $Python -m pip install --upgrade pip

    Remove-Item $GetPipPath -Force -ErrorAction SilentlyContinue
}

if (!(Test-Path $Requirements)) {
    Write-Host "[ERROR] requirements.txt missing"
    exit 1
}

Write-Host "[INSTALL] dependencies..."

& $Python -m pip install -r $Requirements

Write-Host "[VERIFY] runtime check..."

$ver = & $Python -c "import sys; print(sys.version.split()[0])"

Write-Host "Python version: $ver"

Write-Host "============================================================"
Write-Host " DONE"
Write-Host "============================================================"