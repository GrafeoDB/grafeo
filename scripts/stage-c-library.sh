#!/usr/bin/env bash
# Stages one variant of the grafeo-c library for a release asset, in a
# directory named grafeo-c-<tag>-<target>-<variant> in the current directory.
# The release workflow archives that directory (tar.gz, or zip on Windows).
#
# Usage: scripts/stage-c-library.sh <tag> <target> <variant> <lib-dir> [features]
#
#   tag       release tag, such as v0.6.0
#   target    Rust target triple, such as x86_64-unknown-linux-gnu
#   variant   asset label: "default" or "all"
#   lib-dir   directory that holds the built library (target/<target>/release)
#   features  the cargo features the library was built with, comma separated,
#             on top of the default features; empty for the default variant
#
# The directory holds the shared library (with its import library on Windows),
# grafeo.h, the C binding README, LICENSE and FEATURES.txt, which lists the
# grafeo-c features the library was built with. grafeo.h declares every
# function; a function whose feature is not in FEATURES.txt is not exported.
set -euo pipefail

if [ "$#" -lt 4 ] || [ "$#" -gt 5 ]; then
  echo "usage: $0 <tag> <target> <variant> <lib-dir> [features]" >&2
  exit 2
fi

tag=$1
target=$2
variant=$3
lib_dir=$4
features=${5:-}

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

case "$target" in
  *-windows-*) libs=(grafeo_c.dll grafeo_c.dll.lib) ;;
  *-apple-*) libs=(libgrafeo_c.dylib) ;;
  *) libs=(libgrafeo_c.so) ;;
esac

staging="grafeo-c-${tag}-${target}-${variant}"
rm -rf "$staging"
mkdir -p "$staging"

for lib in "${libs[@]}"; do
  if [ ! -f "${lib_dir}/${lib}" ]; then
    echo "error: ${lib_dir}/${lib} not found; build grafeo-c for ${target} first" >&2
    exit 1
  fi
  cp "${lib_dir}/${lib}" "$staging/"
done

cp "${repo_root}/crates/bindings/c/grafeo.h" "$staging/"
cp "${repo_root}/crates/bindings/c/README.md" "$staging/"
cp "${repo_root}/LICENSE" "$staging/"

# The grafeo-c features this build has, as cargo resolves them for the same
# feature selection (grafeo-c features are not unified with other workspace
# members, since no workspace member depends on grafeo-c).
feature_args=()
if [ -n "$features" ]; then
  feature_args=(--features "$features")
fi
# The ${a[@]+...} form keeps an empty array working under `set -u` in bash 3.2 (macOS).
enabled=$(cd "$repo_root" && cargo tree -p grafeo-c --target "$target" ${feature_args[@]+"${feature_args[@]}"} \
  -e features -i grafeo-c --prefix none \
  | sed -n 's/^grafeo-c feature "\([^"]*\)".*/\1/p' | sort -u | paste -sd ' ' -)
if [ -z "$enabled" ]; then
  echo "error: could not resolve the grafeo-c features for ${target}" >&2
  exit 1
fi

{
  echo "grafeo-c ${tag} for ${target}, variant: ${variant}"
  echo
  if [ -n "$features" ]; then
    echo "Cargo features: default, ${features//,/, }"
  else
    echo "Cargo features: default"
  fi
  echo "Enabled grafeo-c features: ${enabled}"
} > "$staging/FEATURES.txt"

echo "Staged ${staging}:"
ls -l "$staging"
cat "$staging/FEATURES.txt"
