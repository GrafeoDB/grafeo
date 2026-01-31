#!/bin/bash
# Local CI check script - run this before making a PR to avoid surprises
# Usage: ./scripts/ci-local.sh [--quick]

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

QUICK_MODE=false
if [[ "$1" == "--quick" ]]; then
    QUICK_MODE=true
    echo -e "${YELLOW}Running in quick mode (skipping release tests)${NC}"
fi

echo -e "\n${YELLOW}========================================${NC}"
echo -e "${YELLOW}  Local CI Check${NC}"
echo -e "${YELLOW}========================================${NC}\n"

# Track failures
FAILURES=()

run_check() {
    local name="$1"
    local cmd="$2"
    echo -e "\n${YELLOW}[$name]${NC} Running..."
    if eval "$cmd"; then
        echo -e "${GREEN}[$name] PASSED${NC}"
    else
        echo -e "${RED}[$name] FAILED${NC}"
        FAILURES+=("$name")
    fi
}

# 1. Rust formatting
run_check "Format" "cargo fmt --all -- --check"

# 2. Clippy
run_check "Clippy" "cargo clippy --all-targets --all-features -- -D warnings"

# 3. Documentation
run_check "Docs" "rm -rf target/doc && RUSTDOCFLAGS='-D warnings' cargo doc --no-deps --all-features"

# 4. Rust tests
run_check "Rust Tests" "cargo test --all-features --workspace"

# 5. Rust tests (release) - skip in quick mode
if [[ "$QUICK_MODE" == false ]]; then
    run_check "Rust Tests (Release)" "cargo test --all-features --workspace --release"
fi

# 6. Python tests (if venv exists)
if [[ -d ".venv" ]]; then
    # Rebuild Python package
    echo -e "\n${YELLOW}[Python Build]${NC} Rebuilding..."
    cd crates/bindings/python
    maturin develop --features pyo3/extension-module,full 2>/dev/null || true
    cd ../../..

    # Check if networkx and solvor are installed
    PYTHON_CMD=".venv/Scripts/python.exe"
    if [[ ! -f "$PYTHON_CMD" ]]; then
        PYTHON_CMD=".venv/bin/python"
    fi

    # Install test deps if needed
    $PYTHON_CMD -c "import numpy" 2>/dev/null || {
        echo -e "${YELLOW}Installing numpy...${NC}"
        uv pip install numpy 2>/dev/null || pip install numpy
    }
    $PYTHON_CMD -c "import scipy" 2>/dev/null || {
        echo -e "${YELLOW}Installing scipy...${NC}"
        uv pip install scipy 2>/dev/null || pip install scipy
    }
    $PYTHON_CMD -c "import networkx" 2>/dev/null || {
        echo -e "${YELLOW}Installing networkx...${NC}"
        uv pip install networkx 2>/dev/null || pip install networkx
    }
    $PYTHON_CMD -c "import solvor" 2>/dev/null || {
        echo -e "${YELLOW}Installing solvor...${NC}"
        uv pip install solvor 2>/dev/null || pip install solvor
    }

    run_check "Python Tests" "$PYTHON_CMD -m pytest tests/python/ -v --ignore=tests/python/benchmark_grafeo.py --ignore=tests/python/benchmark_phases.py"
else
    echo -e "${YELLOW}[Python Tests] Skipped (no .venv found)${NC}"
fi

# Summary
echo -e "\n${YELLOW}========================================${NC}"
echo -e "${YELLOW}  Summary${NC}"
echo -e "${YELLOW}========================================${NC}\n"

if [[ ${#FAILURES[@]} -eq 0 ]]; then
    echo -e "${GREEN}All checks passed!${NC}"
    exit 0
else
    echo -e "${RED}Failed checks:${NC}"
    for f in "${FAILURES[@]}"; do
        echo -e "  ${RED}- $f${NC}"
    done
    exit 1
fi
