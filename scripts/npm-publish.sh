#!/usr/bin/env bash
# Publishes the npm package in the current directory, or skips it when this
# version is already on npm, so a re-run of a release is safe while a real
# failure still fails the job. Extra arguments go to `npm publish`.
#
# Publishing uses npm trusted publishing: the job needs `id-token: write` and
# npm 11.5.1 or later, and no token.
set -euo pipefail

name=$(node -p "require('./package.json').name")
version=$(node -p "require('./package.json').version")

if [ -n "$(npm view "${name}@${version}" version 2>/dev/null || true)" ]; then
  echo "${name}@${version} is already on npm, skipping"
  exit 0
fi

echo "Publishing ${name}@${version}"
npm publish --access public "$@"
