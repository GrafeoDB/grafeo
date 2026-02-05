#!/bin/bash
# Clean CI test script - mimics GitHub Actions environment exactly
# Usage: ./scripts/ci-clean.sh [--skip-rust] [--python <version>]
#
# This creates a fresh venv and installs everything from scratch,
# just like CI does. Use this to catch dependency issues before pushing.
#
# Examples:
#   ./scripts/ci-clean.sh                 # Test all Python versions (3.12, 3.13, 3.14)
#   ./scripts/ci-clean.sh --skip-rust     # Skip Rust checks, test all Python versions
#   ./scripts/ci-clean.sh --python 3.12   # Test only Python 3.12

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
GRAY='\033[0;90m'
NC='\033[0m'

SKIP_RUST=false
PYTHON_VERSIONS=("3.12" "3.13" "3.14")

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-rust)
            SKIP_RUST=true
            shift
            ;;
        --python)
            PYTHON_VERSIONS=("$2")
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo -e "\n${CYAN}========================================${NC}"
echo -e "${CYAN}  Clean CI Test (mimics GitHub Actions)${NC}"
echo -e "${CYAN}========================================${NC}\n"
echo -e "${CYAN}Testing Python versions: ${PYTHON_VERSIONS[*]}${NC}"

# Create temp directory
TEMP_DIR=$(mktemp -d)
echo -e "${GRAY}Using temp directory: $TEMP_DIR${NC}\n"

FAILED=()

cleanup() {
    echo -e "${GRAY}Cleaning up temp directory...${NC}"
    rm -rf "$TEMP_DIR"
}
trap cleanup EXIT

# 1. Rust checks (unless skipped)
if [[ "$SKIP_RUST" == false ]]; then
    echo -e "${YELLOW}[1/4] Format check...${NC}"
    cargo fmt --all -- --check
    echo -e "${GREEN}PASSED${NC}"

    echo -e "\n${YELLOW}[2/4] Clippy...${NC}"
    cargo clippy --all-targets --all-features -- -D warnings
    echo -e "${GREEN}PASSED${NC}"

    echo -e "\n${YELLOW}[3/4] Docs...${NC}"
    rm -rf target/doc
    RUSTDOCFLAGS="-D warnings" cargo doc --no-deps --all-features
    echo -e "${GREEN}PASSED${NC}"

    echo -e "\n${YELLOW}[4/4] Rust tests...${NC}"
    cargo test --all-features --workspace
    echo -e "${GREEN}PASSED${NC}"
else
    echo -e "${YELLOW}Skipping Rust checks (--skip-rust)${NC}"
fi

# Test each Python version
for PY_VER in "${PYTHON_VERSIONS[@]}"; do
    echo -e "\n${CYAN}========================================${NC}"
    echo -e "${CYAN}  Python $PY_VER${NC}"
    echo -e "${CYAN}========================================${NC}"

    PY_DIST_DIR="$TEMP_DIR/dist-$PY_VER"
    PY_VENV_DIR="$TEMP_DIR/venv-$PY_VER"

    # Ensure Python version is available (install if needed)
    echo -e "${GRAY}  Ensuring Python $PY_VER is available...${NC}"
    PY_INTERP=$(uv python find "$PY_VER" 2>/dev/null || echo "")
    if [[ -z "$PY_INTERP" ]]; then
        echo -e "${YELLOW}  Installing Python $PY_VER...${NC}"
        if ! uv python install "$PY_VER"; then
            echo -e "${YELLOW}  Failed to install Python $PY_VER, skipping${NC}"
            continue
        fi
        PY_INTERP=$(uv python find "$PY_VER")
    fi

    # Build wheel for this Python version
    echo -e "\n${YELLOW}  Building wheel...${NC}"
    if ! maturin build --release --out "$PY_DIST_DIR" -m crates/bindings/python/Cargo.toml --interpreter "$PY_INTERP"; then
        echo -e "${RED}  Python $PY_VER FAILED: Wheel build failed${NC}"
        FAILED+=("Python $PY_VER")
        continue
    fi

    # Create venv and install
    echo -e "${GRAY}  Creating venv...${NC}"
    if ! uv venv "$PY_VENV_DIR" --python "$PY_VER"; then
        echo -e "${RED}  Python $PY_VER FAILED: Venv creation failed${NC}"
        FAILED+=("Python $PY_VER")
        continue
    fi

    PYTHON_CMD="$PY_VENV_DIR/bin/python"

    echo -e "${GRAY}  Installing wheel...${NC}"
    WHEEL=$(ls "$PY_DIST_DIR"/*.whl | head -1)
    if ! uv pip install "$WHEEL" --python "$PYTHON_CMD"; then
        echo -e "${RED}  Python $PY_VER FAILED: Wheel install failed${NC}"
        FAILED+=("Python $PY_VER")
        continue
    fi

    echo -e "${GRAY}  Installing test dependencies...${NC}"
    if ! uv pip install pytest pytest-asyncio networkx numpy scipy solvor --python "$PYTHON_CMD"; then
        echo -e "${RED}  Python $PY_VER FAILED: Dependency install failed${NC}"
        FAILED+=("Python $PY_VER")
        continue
    fi

    echo -e "${GRAY}  Running tests...${NC}"
    if ! "$PYTHON_CMD" -m pytest tests/python/ -v --ignore=tests/python/benchmark_grafeo.py --ignore=tests/python/benchmark_phases.py; then
        echo -e "${RED}  Python $PY_VER FAILED: Tests failed${NC}"
        FAILED+=("Python $PY_VER")
        continue
    fi

    echo -e "${GREEN}  Python $PY_VER PASSED${NC}"
done

# Summary
echo -e "\n${CYAN}========================================${NC}"
echo -e "${CYAN}  Summary${NC}"
echo -e "${CYAN}========================================${NC}"

if [[ ${#FAILED[@]} -eq 0 ]]; then
    echo -e "\n${GREEN}All checks passed!${NC}"
else
    echo -e "\n${RED}Failed:${NC}"
    for f in "${FAILED[@]}"; do
        echo -e "  ${RED}- $f${NC}"
    done
    exit 1
fi
