<#
    Build the Simplex release artifacts.

        powershell -ExecutionPolicy Bypass -File packaging\build.ps1
        powershell -ExecutionPolicy Bypass -File packaging\build.ps1 -Version 1.1.0 -Zip

    Produces dist\Simplex-<version>-setup.exe (Inno Setup) and, with -Zip, a
    plain dist\Simplex-<version>.zip for Scoop and for anyone who would rather
    unzip than install. Prints the SHA-256 of each, which is what the winget
    and Scoop manifests need.

    Inno Setup 6 must be installed: https://jrsoftware.org/isdl.php
#>
[CmdletBinding()]
param(
    [string] $Version = "1.0.0",
    [switch] $Zip,
    [switch] $SkipInstaller
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$dist = Join-Path $root "dist"
New-Item -ItemType Directory -Force -Path $dist | Out-Null

# Anything the kit creates for itself must not be shipped: a stale .env would
# override the new user's choices, and .venv is tied to one machine's paths.
$exclude = @(".venv", ".git", ".github", "models", "sessions", "workspace",
             "logs", "dist", "build", ".simplex", "__pycache__")
# keep this list and the Excludes line in packaging\simplex.iss in step
$excludeFiles = @(".env", "providers.json", "bench_vram.json")

function Test-Excluded([string] $fullName) {
    $rel = $fullName.Substring($root.Length).TrimStart('\')
    foreach ($e in $exclude) { if ($rel -eq $e -or $rel.StartsWith("$e\")) { return $true } }
    foreach ($f in $excludeFiles) { if ($rel -eq $f) { return $true } }
    if ($rel.EndsWith(".pyc") -or $rel.EndsWith(".log")) { return $true }
    return $false
}

if ($Zip) {
    Write-Host "Staging the zip ..." -ForegroundColor Cyan
    $stage = Join-Path $dist "simplex"
    if (Test-Path $stage) { Remove-Item -Recurse -Force $stage }
    New-Item -ItemType Directory -Force -Path $stage | Out-Null
    Get-ChildItem -Path $root -Recurse -File | ForEach-Object {
        if (-not (Test-Excluded $_.FullName)) {
            $rel = $_.FullName.Substring($root.Length).TrimStart('\')
            $dest = Join-Path $stage $rel
            New-Item -ItemType Directory -Force -Path (Split-Path -Parent $dest) | Out-Null
            Copy-Item $_.FullName $dest
        }
    }
    $zipPath = Join-Path $dist "Simplex-$Version.zip"
    if (Test-Path $zipPath) { Remove-Item -Force $zipPath }
    Compress-Archive -Path $stage -DestinationPath $zipPath
    Remove-Item -Recurse -Force $stage
    Write-Host "  $zipPath" -ForegroundColor Green
}

if (-not $SkipInstaller) {
    $iscc = @(
        "$env:ProgramFiles\Inno Setup 6\ISCC.exe",
        "${env:ProgramFiles(x86)}\Inno Setup 6\ISCC.exe"
    ) | Where-Object { Test-Path $_ } | Select-Object -First 1
    if (-not $iscc) {
        Write-Warning "Inno Setup 6 not found - skipping the installer. https://jrsoftware.org/isdl.php"
    } else {
        $isccVersion = (Get-Item $iscc).VersionInfo.FileVersion
        Write-Host "  using $iscc ($isccVersion)" -ForegroundColor DarkGray
        Write-Host "Building the installer ..." -ForegroundColor Cyan
        & $iscc "/DAppVersion=$Version" (Join-Path $PSScriptRoot "simplex.iss")
        if ($LASTEXITCODE -ne 0) { throw "ISCC failed with $LASTEXITCODE" }
    }
}

Write-Host ""
Write-Host "Hashes for the winget / Scoop manifests:" -ForegroundColor Cyan
Get-ChildItem -Path $dist -File | Where-Object { $_.Name -like "Simplex-$Version*" } | ForEach-Object {
    $h = (Get-FileHash $_.FullName -Algorithm SHA256).Hash
    "{0,-40} {1}" -f $_.Name, $h
}
