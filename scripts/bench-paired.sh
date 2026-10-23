#!/usr/bin/env bash
# bench-paired.sh: a paired benchmark run of two revisions on this machine.
#
# Builds the given benchmarks of both revisions in release (each revision in
# a worktree of its own, with its own target directory), then runs them
# interleaved, base then head, for several rounds, and prints per case each
# side's fastest round, the change, and whether the rounds overlap (noise).
# Interleaving puts any slow phase of the machine on both sides; nothing else
# should build or run while it measures (other cargo or rustc processes make
# rounds of 2x apart).
#
# Usage:
#   scripts/bench-paired.sh [-r ROUNDS] [-o DIR] BASE_REV HEAD_REV CRATE:BENCH[:FILTER]...
#
#   -r ROUNDS  rounds per side (default 3)
#   -o DIR     where the worktrees, executables and outputs go
#              (default target/bench-paired)
#   FILTER     a Criterion filter (a regex over the case names); all cases
#              when left out
#
# Example: the edge reads of the row-group store, and the write path's file
# cases (GRAFEO_BENCH_STORAGE and GRAFEO_BENCH_SCALE reach the benches):
#   GRAFEO_BENCH_SCALE=10 scripts/bench-paired.sh main HEAD \
#     'grafeo-core:store_apply:in_degree|expand' \
#     'grafeo-engine:write_path_bench:cypher_merge_rel'
#
# Requires: git, cargo, python3.

set -euo pipefail

ROUNDS=3
OUT=target/bench-paired
while getopts "r:o:" option; do
  case "$option" in
    r) ROUNDS="$OPTARG" ;;
    o) OUT="$OPTARG" ;;
    *) sed -n '2,27p' "$0"; exit 1 ;;
  esac
done
shift $((OPTIND - 1))
if [ "$#" -lt 3 ]; then
  sed -n '2,27p' "$0"
  exit 1
fi
BASE_REV="$1"
HEAD_REV="$2"
shift 2
SPECS=("$@")

ROOT="$(git rev-parse --show-toplevel)"
mkdir -p "$OUT"
OUT="$(cd "$OUT" && pwd)"
# The first interpreter that runs (on Windows `python3` can be the Store's
# placeholder, which only prints a hint).
PYTHON=""
for candidate in python3 python; do
  if "$candidate" -c "" > /dev/null 2>&1; then
    PYTHON="$candidate"
    break
  fi
done
[ -n "$PYTHON" ] || { echo "bench-paired.sh needs Python 3" >&2; exit 1; }
# The paths Python is given, in its own form under Git Bash on Windows.
native() { cygpath -m "$1" 2>/dev/null || echo "$1"; }

# Builds every benchmark of SPECS at REV and copies the executables to
# $OUT/bin/<bench>-<side>.
build_side() {
  local side="$1" rev="$2" tree="$OUT/src-$1"
  if [ -d "$tree" ]; then
    git -C "$tree" checkout --quiet --detach "$rev"
  else
    git -C "$ROOT" worktree add --quiet --detach "$tree" "$rev"
  fi
  mkdir -p "$OUT/bin"
  local spec crate bench
  for spec in "${SPECS[@]}"; do
    IFS=: read -r crate bench _ <<< "$spec"
    (cd "$tree" && CARGO_TARGET_DIR="$OUT/target-$side" cargo bench --all-features \
      -p "$crate" --bench "$bench" --no-run --message-format json) |
      "$PYTHON" -c "
import json, shutil, sys
for line in sys.stdin:
    try:
        message = json.loads(line)
    except ValueError:
        continue
    if message.get('reason') == 'compiler-artifact' and message.get('executable') \
            and message['target']['name'] == '$bench':
        shutil.copy(message['executable'], '$(native "$OUT/bin/$bench-$side")')
"
    [ -f "$OUT/bin/$bench-$side" ] || { echo "no executable for $bench at $rev" >&2; exit 1; }
  done
}

echo "building $BASE_REV (base) and $HEAD_REV (head)"
build_side base "$BASE_REV"
build_side head "$HEAD_REV"

rm -rf "$OUT/runs"
mkdir -p "$OUT/runs"
BENCHES=()
for round in $(seq 1 "$ROUNDS"); do
  for side in base head; do
    for spec in "${SPECS[@]}"; do
      IFS=: read -r crate bench filter <<< "$spec"
      # Each bench runs from its own crate, as cargo bench runs it.
      (cd "$OUT/src-$side/crates/$crate" &&
        "$OUT/bin/$bench-$side" --bench ${filter:+"$filter"} \
        > "$OUT/runs/$bench-$side-$round.txt" 2>&1)
    done
  done
  echo "round $round of $ROUNDS done"
done
for spec in "${SPECS[@]}"; do
  IFS=: read -r _ bench _ <<< "$spec"
  BENCHES+=("$bench")
done
"$PYTHON" "$(native "$ROOT/scripts/bench_paired_table.py")" "$(native "$OUT/runs")" base head "${BENCHES[@]}"
echo
echo "The revisions stay checked out in $OUT/src-base and $OUT/src-head for the"
echo "next run; drop them with: git worktree remove --force <dir>"
