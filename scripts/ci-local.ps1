# Local CI check script for Windows - run this before making a PR
# Usage: .\scripts\ci-local.ps1 [-Quick]

param(
    [switch]$Quick
)

$ErrorActionPreference = "Continue"
$failures = @()

function Write-Header($text) {
    Write-Host "`n========================================" -ForegroundColor Yellow
    Write-Host "  $text" -ForegroundColor Yellow
    Write-Host "========================================`n" -ForegroundColor Yellow
}

function Run-Check($name, $command) {
    Write-Host "`n[$name] Running..." -ForegroundColor Yellow
    try {
        Invoke-Expression $command
        if ($LASTEXITCODE -eq 0) {
            Write-Host "[$name] PASSED" -ForegroundColor Green
            return $true
        } else {
            Write-Host "[$name] FAILED" -ForegroundColor Red
            return $false
        }
    } catch {
        Write-Host "[$name] FAILED: $_" -ForegroundColor Red
        return $false
    }
}

Write-Header "Local CI Check"

if ($Quick) {
    Write-Host "Running in quick mode (skipping release tests)" -ForegroundColor Yellow
}

# 1. Format check
if (-not (Run-Check "Format" "cargo fmt --all -- --check")) {
    $failures += "Format"
}

# 2. Clippy
if (-not (Run-Check "Clippy" "cargo clippy --all-targets --all-features -- -D warnings")) {
    $failures += "Clippy"
}

# 3. Documentation
Remove-Item -Recurse -Force target\doc -ErrorAction SilentlyContinue
$env:RUSTDOCFLAGS = "-D warnings"
if (-not (Run-Check "Docs" "cargo doc --no-deps --all-features")) {
    $failures += "Docs"
}
Remove-Item Env:\RUSTDOCFLAGS -ErrorAction SilentlyContinue

# 4. Rust tests
if (-not (Run-Check "Rust Tests" "cargo test --all-features --workspace")) {
    $failures += "Rust Tests"
}

# 5. Rust tests (release) - skip in quick mode
if (-not $Quick) {
    if (-not (Run-Check "Rust Tests (Release)" "cargo test --all-features --workspace --release")) {
        $failures += "Rust Tests (Release)"
    }
}

# 6. Python tests
if (Test-Path ".venv") {
    $pythonCmd = ".venv\Scripts\python.exe"

    # Rebuild Python package
    Write-Host "`n[Python Build] Rebuilding..." -ForegroundColor Yellow
    Push-Location "crates\bindings\python"
    & maturin develop --features pyo3/extension-module,full 2>$null
    Pop-Location

    # Install test deps if needed
    & $pythonCmd -c "import numpy" 2>$null
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Installing numpy..." -ForegroundColor Yellow
        & uv pip install numpy 2>$null
    }
    & $pythonCmd -c "import scipy" 2>$null
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Installing scipy..." -ForegroundColor Yellow
        & uv pip install scipy 2>$null
    }
    & $pythonCmd -c "import networkx" 2>$null
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Installing networkx..." -ForegroundColor Yellow
        & uv pip install networkx 2>$null
    }
    & $pythonCmd -c "import solvor" 2>$null
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Installing solvor..." -ForegroundColor Yellow
        & uv pip install solvor 2>$null
    }

    if (-not (Run-Check "Python Tests" "& $pythonCmd -m pytest tests/python/ -v --ignore=tests/python/benchmark_grafeo.py --ignore=tests/python/benchmark_phases.py")) {
        $failures += "Python Tests"
    }
} else {
    Write-Host "[Python Tests] Skipped (no .venv found)" -ForegroundColor Yellow
}

# Summary
Write-Header "Summary"

if ($failures.Count -eq 0) {
    Write-Host "All checks passed!" -ForegroundColor Green
    exit 0
} else {
    Write-Host "Failed checks:" -ForegroundColor Red
    foreach ($f in $failures) {
        Write-Host "  - $f" -ForegroundColor Red
    }
    exit 1
}
