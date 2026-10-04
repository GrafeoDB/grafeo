#!/usr/bin/env bash
# Build WASM package with size-optimized profile.
#
# Usage:
#   ./scripts/build-wasm.sh                           # default: GQL only, web target
#   ./scripts/build-wasm.sh --features ai              # GQL + AI search
#   ./scripts/build-wasm.sh --features full            # all languages + AI
#   ./scripts/build-wasm.sh --out-dir path/to/output   # custom output directory
#   ./scripts/build-wasm.sh --target bundler --scope grafeo-db
#
# Requirements: rustup target wasm32-unknown-unknown, wasm-bindgen-cli

set -euo pipefail

CRATE_DIR="crates/bindings/wasm"
OUT_DIR=""
PROFILE="minimal-size"
TARGET="web"
SCOPE=""
FEATURES=""
PKG_NAME="@grafeo-db/wasm"

while [[ $# -gt 0 ]]; do
    case $1 in
        --target)   TARGET="$2"; shift 2 ;;
        --scope)    SCOPE="--scope $2"; shift 2 ;;
        --features) FEATURES="--features $2"; shift 2 ;;
        --out-dir)  OUT_DIR="$2"; shift 2 ;;
        --name)     PKG_NAME="$2"; shift 2 ;;
        --release)  PROFILE="release"; shift ;;
        *)          echo "Unknown option: $1"; exit 1 ;;
    esac
done

# Default output directory
if [[ -z "$OUT_DIR" ]]; then
    OUT_DIR="${CRATE_DIR}/pkg"
fi

echo "Building WASM (profile: ${PROFILE}, target: ${TARGET})"

# Step 1: Cargo build
CARGO_CMD="cargo build --target wasm32-unknown-unknown --profile ${PROFILE} -p grafeo-wasm"
if [[ -n "$FEATURES" ]]; then
    CARGO_CMD="${CARGO_CMD} ${FEATURES}"
fi
echo "  cargo build..."
# Filter cargo's chatter to compile/warn/error lines while preserving its exit
# status via PIPESTATUS. errexit + pipefail are disabled briefly so neither
# grep's "no-match" exit (1) nor a cargo failure short-circuits before we read
# PIPESTATUS[0]. Cannot use `|| true` here because that runs `true` as its own
# command and clobbers PIPESTATUS to [0].
set +e +o pipefail
eval "$CARGO_CMD" 2>&1 \
    | grep -E "^error|^   --> |Compiling grafeo-wasm|Finished|warning:"
CARGO_STATUS=${PIPESTATUS[0]}
set -e -o pipefail
if [[ "$CARGO_STATUS" -ne 0 ]]; then
    echo "Error: cargo build failed (exit ${CARGO_STATUS}). Re-run without the filter to see full output:"
    echo "  ${CARGO_CMD}"
    exit "$CARGO_STATUS"
fi

# Determine the output path (profile name maps to directory)
PROFILE_DIR="${PROFILE}"
if [[ "$PROFILE" == "release" ]]; then
    PROFILE_DIR="release"
fi
WASM_FILE="target/wasm32-unknown-unknown/${PROFILE_DIR}/grafeo_wasm.wasm"

if [[ ! -f "$WASM_FILE" ]]; then
    echo "Error: ${WASM_FILE} not found"
    exit 1
fi

# Step 2: wasm-bindgen
echo "  wasm-bindgen..."
rm -rf "$OUT_DIR"
mkdir -p "$OUT_DIR"
wasm-bindgen --target "$TARGET" --out-dir "$OUT_DIR" "$WASM_FILE"

# Step 2b: wasm-bindgen does not emit a package.json. Write a minimal one
# so the output is installable via npm `file:` links and symlinked node_modules.
# Version mirrors [workspace.package] in Cargo.toml.
PKG_VERSION=$(awk '/^\[workspace\.package\]/{p=1;next} /^\[/{p=0} p && /^version[[:space:]]*=/{gsub(/[" ]/,"",$3); print $3; exit}' Cargo.toml)
PKG_VERSION="${PKG_VERSION:-0.0.0}"

# wasm-bindgen emits different module formats per target:
#   nodejs     -> CommonJS
#   no-modules -> plain script that defines a global (no module system)
#   web/bundler/deno -> ESM
# The package.json "type" field must match the module format of the generated .js shim.
case "$TARGET" in
    nodejs|no-modules)
        PKG_TYPE="commonjs"
        PKG_MODULE_FIELD=""
        ;;
    web|bundler|deno)
        PKG_TYPE="module"
        PKG_MODULE_FIELD='  "module": "grafeo_wasm.js",'$'\n'
        ;;
    *)
        echo "Error: unsupported wasm-bindgen target: $TARGET" >&2
        echo "       expected one of: web, bundler, nodejs, no-modules, deno" >&2
        exit 1
        ;;
esac

cat > "$OUT_DIR/package.json" <<EOF
{
  "name": "$PKG_NAME",
  "version": "$PKG_VERSION",
  "type": "$PKG_TYPE",
  "main": "grafeo_wasm.js",
${PKG_MODULE_FIELD}  "types": "grafeo_wasm.d.ts",
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
EOF

# Step 3: Report sizes
RAW_SIZE=$(stat -c%s "$OUT_DIR/grafeo_wasm_bg.wasm" 2>/dev/null || stat -f%z "$OUT_DIR/grafeo_wasm_bg.wasm")
GZ_SIZE=$(gzip -c "$OUT_DIR/grafeo_wasm_bg.wasm" | wc -c)

echo ""
echo "Output: ${OUT_DIR}/"
echo "  Raw:    $(( RAW_SIZE / 1024 )) KB"
echo "  Gzip:   $(( GZ_SIZE / 1024 )) KB"

# Size thresholds (gzipped bytes) depend on feature set.
# Default (browser profile, GQL + regex-lite): competitive with sql.js (~600 KB).
# Full profile (all languages + AI): larger binary, ~1.2 MB gzipped.
if [[ "$FEATURES" == *"full"* ]]; then
    WARN_THRESHOLD=1258291   # 1.2 MB
    FAIL_THRESHOLD=1468006   # 1.4 MB
    LABEL="full profile"
else
    WARN_THRESHOLD=757760    # 740 KB
    FAIL_THRESHOLD=778240    # 760 KB
    LABEL="browser profile"
fi

if [[ "$GZ_SIZE" -gt "$FAIL_THRESHOLD" ]]; then
    echo "  ERROR: ${GZ_SIZE} bytes gzipped exceeds $(( FAIL_THRESHOLD / 1024 )) KB limit (${LABEL})"
    exit 1
elif [[ "$GZ_SIZE" -gt "$WARN_THRESHOLD" ]]; then
    echo "  WARNING: ${GZ_SIZE} bytes gzipped exceeds $(( WARN_THRESHOLD / 1024 )) KB threshold (${LABEL})"
else
    echo "  OK: ${GZ_SIZE} bytes gzipped (under $(( WARN_THRESHOLD / 1024 )) KB threshold, ${LABEL})"
fi
