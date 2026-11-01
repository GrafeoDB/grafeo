# Build WASM package with size-optimized profile.
#
# Usage:
#   .\scripts\build-wasm.ps1                           # default: GQL only, web target
#   .\scripts\build-wasm.ps1 -Features ai              # GQL + AI search
#   .\scripts\build-wasm.ps1 -Features full            # all languages + AI
#   .\scripts\build-wasm.ps1 -OutDir path\to\output    # custom output directory
#   .\scripts\build-wasm.ps1 -Target bundler -Scope grafeo-db
#
# Requirements: rustup target wasm32-unknown-unknown, wasm-bindgen-cli

param(
    [string]$Target = "web",
    [string]$Scope = "",
    [string]$Features = "",
    [string]$OutDir = "",
    [string]$Name = "@grafeo-db/wasm",
    [switch]$Release
)

$ErrorActionPreference = "Stop"

$CrateDir = "crates\bindings\wasm"
if (-not $OutDir) { $OutDir = "$CrateDir\pkg" }
$Profile = if ($Release) { "release" } else { "wasm" }

Write-Host "Building WASM (profile: $Profile, target: $Target)"

# Step 1: Cargo build
$cargoArgs = @("build", "--target", "wasm32-unknown-unknown", "--profile", $Profile, "-p", "grafeo-wasm")
if ($Features) {
    $cargoArgs += "--features"
    $cargoArgs += $Features
}
Write-Host "  cargo build..."
& cargo @cargoArgs
if ($LASTEXITCODE -ne 0) { throw "Cargo build failed" }

# Determine output path
$TargetDir = if ($env:CARGO_TARGET_DIR) { $env:CARGO_TARGET_DIR } else { "target" }
$WasmFile = "$TargetDir\wasm32-unknown-unknown\$Profile\grafeo_wasm.wasm"
if (-not (Test-Path $WasmFile)) {
    throw "Error: $WasmFile not found"
}

# Step 2: wasm-bindgen
Write-Host "  wasm-bindgen..."
if (Test-Path $OutDir) { Remove-Item -Recurse -Force $OutDir }
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
& wasm-bindgen --target $Target --out-dir $OutDir $WasmFile
if ($LASTEXITCODE -ne 0) { throw "wasm-bindgen failed" }

# Step 2b: wasm-bindgen does not emit a package.json. Write a minimal one
# so the output is installable via npm `file:` links and symlinked node_modules.
# Version mirrors [workspace.package] in Cargo.toml.
$cargoToml = Get-Content "Cargo.toml"
$pkgVersion = "0.0.0"
$inWorkspacePackage = $false
foreach ($line in $cargoToml) {
    if ($line -match '^\[workspace\.package\]') { $inWorkspacePackage = $true; continue }
    if ($line -match '^\[') { $inWorkspacePackage = $false }
    if ($inWorkspacePackage -and $line -match '^version\s*=\s*"([^"]+)"') {
        $pkgVersion = $Matches[1]; break
    }
}
# wasm-bindgen emits CommonJS for --target nodejs and ESM for web/bundler/deno/esm.
# The package.json "type" field must match the module format of the generated .js shim.
if ($Target -eq "nodejs") {
    $pkgType = "commonjs"
    $moduleField = ""
} else {
    $pkgType = "module"
    $moduleField = "  `"module`": `"grafeo_wasm.js`",`n"
}

$packageJson = @"
{
  "name": "$Name",
  "version": "$pkgVersion",
  "type": "$pkgType",
  "main": "grafeo_wasm.js",
$($moduleField)  "types": "grafeo_wasm.d.ts",
  "files": [
    "grafeo_wasm.js",
    "grafeo_wasm.d.ts",
    "grafeo_wasm_bg.wasm",
    "grafeo_wasm_bg.wasm.d.ts"
  ],
  "sideEffects": [
    "./grafeo_wasm.js",
    "./snippets/*"
  ]
}
"@
$packageJson | Out-File -FilePath (Join-Path $OutDir "package.json") -Encoding utf8 -NoNewline

# Step 3: Report sizes
$wasmPath = Join-Path $OutDir "grafeo_wasm_bg.wasm"
$rawSize = (Get-Item $wasmPath).Length
$gzBytes = [System.IO.File]::ReadAllBytes($wasmPath)
$ms = New-Object System.IO.MemoryStream
$gz = New-Object System.IO.Compression.GZipStream($ms, [System.IO.Compression.CompressionMode]::Compress)
$gz.Write($gzBytes, 0, $gzBytes.Length)
$gz.Close()
$gzSize = $ms.Length

Write-Host ""
Write-Host "Output: $OutDir\"
Write-Host "  Raw:    $([math]::Round($rawSize / 1024)) KB"
Write-Host "  Gzip:   $([math]::Round($gzSize / 1024)) KB"

# Size thresholds (gzipped bytes)
# 660 KB = 675840 bytes: warning threshold for browser profile
# Binary is ~95% essential application code (parser, planner, executor),
# competitive with sql.js (~600 KB). Profiled with twiggy in 0.5.39.
$warnThreshold = 675840
$failThreshold = 716800  # 700 KB: hard limit

if ($gzSize -gt $failThreshold) {
    Write-Host "  ERROR: $gzSize bytes gzipped exceeds 700 KB limit" -ForegroundColor Red
    exit 1
} elseif ($gzSize -gt $warnThreshold) {
    Write-Host "  WARNING: $gzSize bytes gzipped exceeds 660 KB threshold" -ForegroundColor Yellow
} else {
    Write-Host "  OK: $gzSize bytes gzipped (under 660 KB threshold)" -ForegroundColor Green
}
