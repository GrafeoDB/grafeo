# Run all spec test suites locally (mirrors CI spec test jobs)
# Usage: .\scripts\ci-spec.ps1 [-Only rust,python,node,go,csharp,dart]
#
# Requires: cargo, python, node/npm, go, dotnet, dart

param(
    [string]$Only = ""
)

$ErrorActionPreference = "Continue"
$root = (Get-Item $PSScriptRoot).Parent.FullName
$failures = @()
$passed = @()
$skipped = @()

function Write-Header($text) {
    Write-Host "`n========================================" -ForegroundColor Yellow
    Write-Host "  $text" -ForegroundColor Yellow
    Write-Host "========================================`n" -ForegroundColor Yellow
}

function Should-Run($name) {
    if ($Only -eq "") { return $true }
    $list = $Only -split ","
    return $list -contains $name
}

# Determine library extension and cargo output name
$libExt = "dll"
$libPrefix = ""
$libName = "grafeo_c.dll"
if ($IsLinux) {
    $libExt = "so"
    $libPrefix = "lib"
    $libName = "libgrafeo_c.so"
} elseif ($IsMacOS) {
    $libExt = "dylib"
    $libPrefix = "lib"
    $libName = "libgrafeo_c.dylib"
}

$needsNative = @("go", "csharp", "dart") | Where-Object { Should-Run $_ }
$builtNative = $false

function Build-NativeLib {
    if ($script:builtNative) { return $true }
    Write-Header "Building grafeo-c (release, full features)"
    Push-Location $root
    cargo build --release -p grafeo-c --features full
    $ok = $LASTEXITCODE -eq 0
    Pop-Location
    if ($ok) { $script:builtNative = $true }
    return $ok
}

# ── 1. Rust spec tests ────────────────────────────────────────────
if (Should-Run "rust") {
    Write-Header "Rust Spec Tests"
    Push-Location $root
    cargo test -p grafeo-spec-tests --all-features 2>&1 | Tee-Object -Variable rustOut
    if ($LASTEXITCODE -eq 0) { $passed += "rust" } else { $failures += "rust" }
    Pop-Location
}

# ── 2. Python spec tests ─────────────────────────────────────────
if (Should-Run "python") {
    Write-Header "Python Spec Tests"
    Push-Location "$root\crates\bindings\python"
    maturin develop --release --features pyo3/extension-module 2>&1 | Out-Null
    Pop-Location
    if ($LASTEXITCODE -eq 0) {
        Push-Location $root
        pytest tests/spec/ -v 2>&1 | Tee-Object -Variable pyOut
        if ($LASTEXITCODE -eq 0) { $passed += "python" } else { $failures += "python" }
        Pop-Location
    } else {
        Write-Host "  Python build failed" -ForegroundColor Red
        $failures += "python"
    }
}

# ── 3. Node.js spec tests ────────────────────────────────────────
if (Should-Run "node") {
    Write-Header "Node.js Spec Tests"
    Push-Location "$root\crates\bindings\node"
    npm install 2>&1 | Out-Null
    npm run build 2>&1 | Out-Null
    Pop-Location
    if ($LASTEXITCODE -eq 0) {
        Push-Location $root
        npx vitest run tests/spec/runners/node/spec-runner.test.mjs 2>&1 | Tee-Object -Variable nodeOut
        if ($LASTEXITCODE -eq 0) { $passed += "node" } else { $failures += "node" }
        Pop-Location
    } else {
        Write-Host "  Node.js build failed" -ForegroundColor Red
        $failures += "node"
    }
}

# ── 4. Go spec tests ─────────────────────────────────────────────
if (Should-Run "go") {
    Write-Header "Go Spec Tests"
    if (Build-NativeLib) {
        $libSrc = Join-Path $root "target\release\$libName"
        $goDest = Join-Path $root "tests\spec\runners\go"
        Copy-Item $libSrc $goDest -Force
        Push-Location $goDest
        $env:CGO_ENABLED = "1"
        if ($IsLinux) {
            $env:CGO_LDFLAGS = "-L. -lgrafeo_c -lm -ldl -lpthread"
            $env:LD_LIBRARY_PATH = "."
        } elseif ($IsMacOS) {
            $env:CGO_LDFLAGS = "-L. -lgrafeo_c -lm -ldl -lpthread -framework Security"
            $env:DYLD_LIBRARY_PATH = "."
        } else {
            $env:CGO_LDFLAGS = "-L. -lgrafeo_c -lws2_32 -lbcrypt -lntdll -luserenv"
        }
        go test -count=1 -run TestSpec -timeout 120s -v 2>&1 | Tee-Object -Variable goOut
        if ($LASTEXITCODE -eq 0) { $passed += "go" } else { $failures += "go" }
        Pop-Location
    } else {
        Write-Host "  Native lib build failed" -ForegroundColor Red
        $failures += "go"
    }
}

# ── 5. C# spec tests ─────────────────────────────────────────────
if (Should-Run "csharp") {
    Write-Header "C# Spec Tests"
    if (Build-NativeLib) {
        $libSrc = Join-Path $root "target\release\$libName"
        $csDest = Join-Path $root "tests\spec\runners\csharp"
        Copy-Item $libSrc $csDest -Force
        Push-Location $csDest
        $env:LD_LIBRARY_PATH = $csDest
        # Windows finds native DLLs via PATH, not LD_LIBRARY_PATH; without this the
        # runner silently skips every test or loads a stale copy from bin\.
        # Scoped to this block so later steps see the original PATH.
        $originalPath = $env:PATH
        $env:PATH = "$csDest$([System.IO.Path]::PathSeparator)$env:PATH"
        try {
            dotnet test --verbosity minimal 2>&1 | Tee-Object -Variable csOut
            if ($LASTEXITCODE -eq 0) { $passed += "csharp" } else { $failures += "csharp" }
        } finally {
            $env:PATH = $originalPath
        }
        Pop-Location
    } else {
        Write-Host "  Native lib build failed" -ForegroundColor Red
        $failures += "csharp"
    }
}

# ── 6. Dart spec tests ───────────────────────────────────────────
if (Should-Run "dart") {
    Write-Header "Dart Spec Tests"
    if (Build-NativeLib) {
        $libSrc = Join-Path $root "target\release\$libName"
        $dartLib = Join-Path $root "crates\bindings\dart"
        $dartRunner = Join-Path $root "tests\spec\runners\dart"
        Copy-Item $libSrc $dartLib -Force
        Copy-Item $libSrc $dartRunner -Force
        Push-Location $dartRunner
        $env:LD_LIBRARY_PATH = $dartLib
        dart pub get 2>&1 | Out-Null
        dart test spec_runner_test.dart 2>&1 | Tee-Object -Variable dartOut
        if ($LASTEXITCODE -eq 0) { $passed += "dart" } else { $failures += "dart" }
        Pop-Location
    } else {
        Write-Host "  Native lib build failed" -ForegroundColor Red
        $failures += "dart"
    }
}

# ── Summary ───────────────────────────────────────────────────────
Write-Header "Spec Test Summary"
if ($passed.Count -gt 0) {
    Write-Host "  PASSED: $($passed -join ', ')" -ForegroundColor Green
}
if ($failures.Count -gt 0) {
    Write-Host "  FAILED: $($failures -join ', ')" -ForegroundColor Red
    exit 1
} else {
    Write-Host "`n  All spec tests passed!" -ForegroundColor Green
    exit 0
}
