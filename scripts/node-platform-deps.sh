#!/usr/bin/env bash
# The platform packages of @grafeo-db/js, which hold its native binaries.
#
# The manifest in the repository leaves them out: they reach npm only during a
# release, so before that no lockfile can name them at the release's version and
# `npm ci` would refuse the package. The release adds them just before it publishes.
# Without them the published package has no binaries, so every mode fails unless
# it finds exactly the folders under npm/, at the version.
#
# Run from crates/bindings/node:
#   node-platform-deps.sh add VERSION        adds them to package.json, for publishing
#   node-platform-deps.sh check              packs a temporary copy with them added and
#                                            checks the packed manifest (CI, every push)
#   node-platform-deps.sh published VERSION  checks that version on npm
set -euo pipefail

readonly EXPECTED_COUNT=6

fail() {
  echo "node-platform-deps: $*" >&2
  exit 1
}

# The names of the packages under npm/, one per line, after checking their number.
expected_names() {
  local count=0 dir
  for dir in npm/*/; do
    node -p "require('./${dir}package.json').name"
    count=$((count + 1))
  done
  [ "$count" -eq "$EXPECTED_COUNT" ] ||
    fail "found $count folders under npm/, expected $EXPECTED_COUNT"
}

# Fails unless the optional dependencies on standard input are exactly `names`
# (one per line), each at `version`.
assert_lists_all() {
  local version=$1 names=$2
  NAMES=$names VERSION=$version node -e '
    const listed = JSON.parse(require("fs").readFileSync(0, "utf8") || "{}");
    const wanted = process.env.NAMES.split(/\r?\n/).filter(Boolean);
    const wrong = [
      ...wanted.filter((name) => listed[name] !== process.env.VERSION),
      ...Object.keys(listed).filter((name) => !wanted.includes(name)),
    ];
    if (wrong.length > 0) {
      console.error(`wanted ${wanted.length} at ${process.env.VERSION}, wrong: ${wrong.join(", ")}`);
      console.error(JSON.stringify(listed, null, 2));
      process.exit(1);
    }
    console.log(`${wanted.length} platform packages at ${process.env.VERSION}`);
  ' || fail "the optional dependencies are not the platform packages"
}

# Prints `optionalDependencies` of the manifest at the path given, as JSON.
optional_dependencies_of() {
  node -p "JSON.stringify(require('$1').optionalDependencies ?? {})"
}

add() {
  local version=$1 names dir own
  [ "$(optional_dependencies_of ./package.json)" = "{}" ] ||
    fail "package.json lists optional dependencies already"
  names=$(expected_names)
  for dir in npm/*/; do
    own=$(node -p "require('./${dir}package.json').version")
    [ "$own" = "$version" ] || fail "$dir is at $own, not $version"
    npm pkg set "optionalDependencies.$(node -p "require('./${dir}package.json').name")=$version"
  done
  optional_dependencies_of ./package.json | assert_lists_all "$version" "$names"
}

check() {
  local version source_sum names tarball
  version=$(node -p "require('./package.json').version")
  source_sum=$(cksum package.json)
  names=$(expected_names)
  copy=$(mktemp -d)
  trap 'rm -rf "$copy"' EXIT
  cp package.json "$copy/"
  cp -R npm "$copy/"
  (
    cd "$copy"
    add "$version"
    tarball=$(npm pack --ignore-scripts --silent | tail -n 1)
    tar -xzOf "$tarball" package/package.json >packed.json
    optional_dependencies_of ./packed.json | assert_lists_all "$version" "$names"
  )
  [ "$(cksum package.json)" = "$source_sum" ] || fail "the check changed package.json"
}

# A package just published can take a moment to show up.
published() {
  local version=$1 names listed="" attempt
  names=$(expected_names)
  for attempt in 1 2 3 4 5 6; do
    if listed=$(npm view "@grafeo-db/js@$version" optionalDependencies --json 2>/dev/null) &&
      [ -n "$listed" ]; then
      break
    fi
    listed=""
    [ "$attempt" -eq 6 ] || sleep 20
  done
  [ -n "$listed" ] || fail "@grafeo-db/js@$version lists no optional dependencies on npm"
  printf '%s' "$listed" | assert_lists_all "$version" "$names"
}

case "${1:-}" in
  add) [ $# -eq 2 ] || fail "add needs a version"; add "$2" ;;
  check) check ;;
  published) [ $# -eq 2 ] || fail "published needs a version"; published "$2" ;;
  *) fail "usage: node-platform-deps.sh add VERSION | check | published VERSION" ;;
esac
