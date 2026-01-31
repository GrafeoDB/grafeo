# Clean CI test script - mimics GitHub Actions environment exactly
# Usage: .\scripts\ci-clean.ps1 [-SkipRust] [-Python <version>]
#
# This creates a fresh venv and installs everything from scratch,
# just like CI does. Use this to catch dependency issues before pushing.
#
# Examples:
#   .\scripts\ci-clean.ps1              # Test all Python versions (3.12, 3.13, 3.14)
#   .\scripts\ci-clean.ps1 -SkipRust    # Skip Rust checks, test all Python versions
#   .\scripts\ci-clean.ps1 -Python 3.12 # Test only Python 3.12

param(
    [switch]$SkipRust,
    [string]$Python  # Specific version to test, or empty for all
)

$ErrorActionPreference = "Stop"
$pythonVersions = @("3.12", "3.13", "3.14")

if ($Python) {
    $pythonVersions = @($Python)
}

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "  Clean CI Test (mimics GitHub Actions)" -ForegroundColor Cyan
Write-Host "========================================`n" -ForegroundColor Cyan
Write-Host "Testing Python versions: $($pythonVersions -join ', ')" -ForegroundColor Cyan

# Create temp directory for clean test
$tempDir = Join-Path $env:TEMP "grafeo-ci-test-$(Get-Random)"
New-Item -ItemType Directory -Path $tempDir -Force | Out-Null
Write-Host "Using temp directory: $tempDir`n" -ForegroundColor Gray

$failed = @()

try {
    # 1. Rust checks (unless skipped)
    if (-not $SkipRust) {
        Write-Host "[1/4] Format check..." -ForegroundColor Yellow
        cargo fmt --all -- --check
        if ($LASTEXITCODE -ne 0) { throw "Format check failed" }
        Write-Host "PASSED" -ForegroundColor Green

        Write-Host "`n[2/4] Clippy..." -ForegroundColor Yellow
        cargo clippy --all-targets --all-features -- -D warnings
        if ($LASTEXITCODE -ne 0) { throw "Clippy failed" }
        Write-Host "PASSED" -ForegroundColor Green

        Write-Host "`n[3/4] Docs..." -ForegroundColor Yellow
        Remove-Item -Recurse -Force target\doc -ErrorAction SilentlyContinue
        $env:RUSTDOCFLAGS = "-D warnings"
        cargo doc --no-deps --all-features
        if ($LASTEXITCODE -ne 0) { throw "Doc check failed" }
        Remove-Item Env:\RUSTDOCFLAGS -ErrorAction SilentlyContinue
        Write-Host "PASSED" -ForegroundColor Green

        Write-Host "`n[4/4] Rust tests..." -ForegroundColor Yellow
        cargo test --all-features --workspace
        if ($LASTEXITCODE -ne 0) { throw "Rust tests failed" }
        Write-Host "PASSED" -ForegroundColor Green
    } else {
        Write-Host "Skipping Rust checks (-SkipRust)" -ForegroundColor Yellow
    }

    # Test each Python version
    foreach ($pyVer in $pythonVersions) {
        Write-Host "`n========================================" -ForegroundColor Cyan
        Write-Host "  Python $pyVer" -ForegroundColor Cyan
        Write-Host "========================================" -ForegroundColor Cyan

        $pyDistDir = Join-Path $tempDir "dist-$pyVer"
        $pyVenvDir = Join-Path $tempDir "venv-$pyVer"

        try {
            # Ensure Python version is available (install if needed)
            Write-Host "`n  Ensuring Python $pyVer is available..." -ForegroundColor Gray
            $findResult = uv python find $pyVer 2>&1
            if ($LASTEXITCODE -ne 0 -or $findResult -match "error:") {
                Write-Host "  Installing Python $pyVer..." -ForegroundColor Yellow
                uv python install $pyVer 2>&1 | Out-Null
                if ($LASTEXITCODE -ne 0) {
                    Write-Host "  Failed to install Python $pyVer, skipping" -ForegroundColor Yellow
                    continue
                }
                $pyInterp = (uv python find $pyVer 2>&1).Trim()
            } else {
                $pyInterp = $findResult.Trim()
            }

            if (-not $pyInterp -or -not (Test-Path $pyInterp)) {
                Write-Host "  Python $pyVer interpreter not found, skipping" -ForegroundColor Yellow
                continue
            }

            # Build wheel for this Python version
            Write-Host "  Building wheel..." -ForegroundColor Yellow
            maturin build --release --out $pyDistDir -m crates/bindings/python/Cargo.toml --interpreter $pyInterp
            if ($LASTEXITCODE -ne 0) { throw "Wheel build failed" }

            # Create venv and install
            Write-Host "  Creating venv..." -ForegroundColor Gray
            uv venv $pyVenvDir --python $pyVer
            if ($LASTEXITCODE -ne 0) { throw "Failed to create venv" }

            $pythonCmd = Join-Path $pyVenvDir "Scripts\python.exe"

            Write-Host "  Installing wheel..." -ForegroundColor Gray
            $wheel = Get-ChildItem -Path $pyDistDir -Filter "*.whl" | Select-Object -First 1
            uv pip install $wheel.FullName --python $pythonCmd
            if ($LASTEXITCODE -ne 0) { throw "Failed to install wheel" }

            Write-Host "  Installing test dependencies..." -ForegroundColor Gray
            uv pip install pytest pytest-asyncio networkx numpy scipy solvor --python $pythonCmd
            if ($LASTEXITCODE -ne 0) { throw "Failed to install dependencies" }

            Write-Host "  Running tests..." -ForegroundColor Gray
            & $pythonCmd -m pytest tests/python/ -v --ignore=tests/python/benchmark_grafeo.py --ignore=tests/python/benchmark_phases.py
            if ($LASTEXITCODE -ne 0) { throw "Tests failed" }

            Write-Host "  Python $pyVer PASSED" -ForegroundColor Green
        } catch {
            Write-Host "  Python $pyVer FAILED: $_" -ForegroundColor Red
            $failed += "Python $pyVer"
        }
    }

    # Summary
    Write-Host "`n========================================" -ForegroundColor Cyan
    Write-Host "  Summary" -ForegroundColor Cyan
    Write-Host "========================================" -ForegroundColor Cyan

    if ($failed.Count -eq 0) {
        Write-Host "`nAll checks passed!" -ForegroundColor Green
    } else {
        Write-Host "`nFailed:" -ForegroundColor Red
        foreach ($f in $failed) {
            Write-Host "  - $f" -ForegroundColor Red
        }
        exit 1
    }

} catch {
    Write-Host "`n========================================" -ForegroundColor Red
    Write-Host "  FAILED: $_" -ForegroundColor Red
    Write-Host "========================================`n" -ForegroundColor Red
    exit 1
} finally {
    # Cleanup
    Write-Host "`nCleaning up temp directory..." -ForegroundColor Gray
    Remove-Item -Recurse -Force $tempDir -ErrorAction SilentlyContinue
}
