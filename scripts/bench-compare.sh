#!/usr/bin/env bash
# bench-compare.sh: Compare two Criterion baselines with per-benchmark thresholds
# and optional memory snapshot comparison.
#
# Usage: bench-compare.sh <baseline> <candidate> <config_file> <pr_number>
#
# Requires: critcmp, gh (GitHub CLI), python3 (3.11+ for tomllib)

set -euo pipefail

BASELINE="${1:?Usage: bench-compare.sh <baseline> <candidate> <config_file> <pr_number>}"
CANDIDATE="${2:?}"
CONFIG="${3:?}"
PR_NUMBER="${4:?}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── Load per-benchmark thresholds ──────────────────────────────────────
# Build an associative array: pattern -> "threshold_pct:fail_ci"
declare -A THRESHOLDS
DEFAULT_THRESHOLD=15
DEFAULT_FAIL_CI="false"

while IFS=$'\t' read -r pattern threshold fail_ci; do
  if [[ "$pattern" == "benchmark_pattern" ]]; then
    continue  # skip header
  fi
  if [[ "$pattern" == "__default__" ]]; then
    DEFAULT_THRESHOLD="$threshold"
    DEFAULT_FAIL_CI="$fail_ci"
  else
    THRESHOLDS["$pattern"]="${threshold}:${fail_ci}"
  fi
done < <(python3 "$SCRIPT_DIR/parse-thresholds.py" "$CONFIG")

# Resolve threshold for a given benchmark name.
resolve_threshold() {
  local bench_name="$1"
  for pattern in "${!THRESHOLDS[@]}"; do
    # Use bash pattern matching (fnmatch-style via case)
    # shellcheck disable=SC2254
    case "$bench_name" in
      $pattern)
        echo "${THRESHOLDS[$pattern]}"
        return
        ;;
    esac
  done
  echo "${DEFAULT_THRESHOLD}:${DEFAULT_FAIL_CI}"
}

# ── Run critcmp comparison ─────────────────────────────────────────────
COMPARISON=$(critcmp "$BASELINE" "$CANDIDATE" --color never 2>&1) || true

echo "--- critcmp output ---"
echo "$COMPARISON"
echo "--- end ---"

# ── Parse regressions with per-benchmark thresholds ────────────────────
BLOCKING_FAILURES=0
declare -a REGRESSION_LINES=()

while IFS= read -r line; do
  # Skip header/separator lines
  if [[ "$line" =~ ^group ]] || [[ "$line" =~ ^----- ]] || [[ -z "$line" ]]; then
    continue
  fi
  # Data lines start with a letter or underscore
  if [[ "$line" =~ ^[a-zA-Z_] ]]; then
    bench_name=$(echo "$line" | awk '{print $1}')
    # Find the candidate ratio (second ratio value in line)
    ratio=$(echo "$line" | awk '{
      count = 0
      for (i = 2; i <= NF; i++) {
        if ($i ~ /^[0-9]+\.[0-9]+$/) {
          count++
          if (count == 2) { print $i; exit }
        }
      }
    }')

    if [[ -z "$ratio" ]] || [[ "$ratio" == "0" ]]; then
      continue
    fi

    pct_change=$(awk "BEGIN { printf \"%.1f\", ($ratio - 1.0) * 100 }")

    # Check if positive (regression)
    is_regression=$(awk "BEGIN { print ($ratio > 1.0) ? 1 : 0 }")
    if [[ "$is_regression" != "1" ]]; then
      continue
    fi

    # Resolve per-benchmark threshold
    resolved=$(resolve_threshold "$bench_name")
    threshold="${resolved%%:*}"
    fail_ci="${resolved##*:}"

    exceeds=$(awk "BEGIN { print ($pct_change > $threshold + 0) ? 1 : 0 }")
    if [[ "$exceeds" == "1" ]]; then
      marker=""
      if [[ "$fail_ci" == "true" ]]; then
        marker=" **BLOCKING**"
        BLOCKING_FAILURES=$((BLOCKING_FAILURES + 1))
      fi
      REGRESSION_LINES+=("| \`$bench_name\` | +${pct_change}% | ${threshold}% |${marker}")
    fi
  fi
done <<< "$COMPARISON"

# ── Memory comparison ──────────────────────────────────────────────────
MEMORY_BASE="target/criterion/memory_snapshot_base.json"
MEMORY_PR="target/criterion/memory_snapshot.json"
declare -a MEMORY_LINES=()
MEMORY_FAIL=0

if [[ -f "$MEMORY_BASE" && -f "$MEMORY_PR" ]]; then
  # Read memory bounds from config
  while IFS=$'\t' read -r key value; do
    # Read PR snapshot value for this key
    pr_value=$(python3 -c "
import json, sys
with open('$MEMORY_PR') as f:
    data = json.load(f)
print(data.get('$key', 0))
" 2>/dev/null || echo "0")

    base_value=$(python3 -c "
import json, sys
with open('$MEMORY_BASE') as f:
    data = json.load(f)
print(data.get('$key', 0))
" 2>/dev/null || echo "0")

    if [[ "$pr_value" -gt 0 ]]; then
      # Check against absolute bound from config
      bound=$(python3 -c "
import tomllib
with open('$CONFIG', 'rb') as f:
    c = tomllib.load(f)
print(c.get('memory', {}).get('bounds', {}).get('$key', 0))
" 2>/dev/null || echo "0")

      if [[ "$bound" -gt 0 && "$pr_value" -gt "$bound" ]]; then
        MEMORY_LINES+=("| \`$key\` | $(numfmt --to=iec "$base_value" 2>/dev/null || echo "$base_value") | $(numfmt --to=iec "$pr_value" 2>/dev/null || echo "$pr_value") | $(numfmt --to=iec "$bound" 2>/dev/null || echo "$bound") | EXCEEDED |")
        # Check if memory failures should block
        mem_fail_ci=$(python3 -c "
import tomllib
with open('$CONFIG', 'rb') as f:
    c = tomllib.load(f)
print(str(c.get('memory', {}).get('fail_ci', False)).lower())
" 2>/dev/null || echo "false")
        if [[ "$mem_fail_ci" == "true" ]]; then
          MEMORY_FAIL=1
        fi
      else
        MEMORY_LINES+=("| \`$key\` | $(numfmt --to=iec "$base_value" 2>/dev/null || echo "$base_value") | $(numfmt --to=iec "$pr_value" 2>/dev/null || echo "$pr_value") | $(numfmt --to=iec "$bound" 2>/dev/null || echo "$bound") | OK |")
      fi
    fi
  done < <(python3 -c "
import json, sys
with open('$MEMORY_PR') as f:
    data = json.load(f)
for k, v in data.items():
    print(f'{k}\t{v}')
" 2>/dev/null)
fi

# ── Build markdown body ────────────────────────────────────────────────
BODY_FILE=$(mktemp)
{
  echo "## Benchmark Comparison (base vs PR)"
  echo ""
  echo "Measured on the **same runner** to eliminate hardware variance."
  echo ""
  echo "<details>"
  echo "<summary>Full results</summary>"
  echo ""
  echo '```'
  echo "$COMPARISON"
  echo '```'
  echo ""
  echo "</details>"
  echo ""
} > "$BODY_FILE"

if [[ ${#REGRESSION_LINES[@]} -gt 0 ]]; then
  {
    echo "### Performance regressions"
    echo ""
    echo "| Benchmark | Regression | Threshold |"
    echo "| --- | --- | --- |"
    for line in "${REGRESSION_LINES[@]}"; do
      echo "$line"
    done
    echo ""
  } >> "$BODY_FILE"
else
  echo "No performance regressions above configured thresholds." >> "$BODY_FILE"
  echo "" >> "$BODY_FILE"
fi

if [[ ${#MEMORY_LINES[@]} -gt 0 ]]; then
  {
    echo "### Memory usage"
    echo ""
    echo "| Benchmark | Base | PR | Bound | Status |"
    echo "| --- | --- | --- | --- | --- |"
    for line in "${MEMORY_LINES[@]}"; do
      echo "$line"
    done
    echo ""
  } >> "$BODY_FILE"
fi

# Add marker for idempotent comment updates
echo "" >> "$BODY_FILE"
echo "<!-- grafeo-bench-comparison -->" >> "$BODY_FILE"

BODY=$(cat "$BODY_FILE")
rm -f "$BODY_FILE"

# Keep the report available even when a fork token cannot publish comments.
if [[ -n "${GITHUB_STEP_SUMMARY:-}" ]]; then
  printf '\n%s\n' "$BODY" >> "$GITHUB_STEP_SUMMARY"
fi

# ── Post or update PR comment ──────────────────────────────────────────
REPO="${GITHUB_REPOSITORY:?GITHUB_REPOSITORY must be set}"

COMMENT_ID=$(gh api "repos/${REPO}/issues/${PR_NUMBER}/comments" \
  --jq '.[] | select(.body | contains("<!-- grafeo-bench-comparison -->")) | .id' \
  | head -1) || true

if [ -n "$COMMENT_ID" ]; then
  if gh api --method PATCH "repos/${REPO}/issues/comments/${COMMENT_ID}" -f body="$BODY"; then
    echo "Updated existing comment $COMMENT_ID"
  else
    echo "WARNING: Could not update benchmark comment; continuing with benchmark checks." >&2
  fi
else
  if gh pr comment "$PR_NUMBER" --body "$BODY"; then
    echo "Posted new comment on PR #$PR_NUMBER"
  else
    echo "WARNING: Could not post benchmark comment; continuing with benchmark checks." >&2
  fi
fi

# ── Exit code ──────────────────────────────────────────────────────────
TOTAL_FAILURES=$((BLOCKING_FAILURES + MEMORY_FAIL))
if [[ "$TOTAL_FAILURES" -gt 0 ]]; then
  echo "ERROR: $BLOCKING_FAILURES blocking regression(s), $MEMORY_FAIL memory bound failure(s)"
  exit 1
fi

echo "All benchmarks within configured thresholds."
