// Usage: node datasets/tools/set-version.mjs <version> [--root <repo root>]
// Sets the version of every dataset package, editing only the version field.
import { readFileSync, writeFileSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const VERSION = /^\d+\.\d+\.\d+(-[0-9A-Za-z.]+)?$/;

/** Replaces the first `"version": "..."` line, keeping the rest of the file as it is. */
const updateJsonVersion = (text, version) => text.replace(/("version":\s*")[^"]*(")/, `$1${version}$2`);

/** Replaces the `version = "..."` line and the `grafeo>=` floor of the extra in pyproject.toml. */
const updatePyproject = (text, version) =>
  text.replace(/^(version = ")[^"]*(")/m, `$1${version}$2`).replace(/("grafeo>=)[^"]*(")/, `$1${version}$2`);

/** Replaces the `__version__ = "..."` line. */
const updatePyInit = (text, version) => text.replace(/^(__version__ = ")[^"]*(")/m, `$1${version}$2`);

/**
 * One entry per file that carries the version. Add a package here.
 * `required` lists what must be in the file before it is rewritten, as [pattern, name] pairs.
 */
export const TARGETS = [
  {
    file: 'packages/grafeo-datasets-npm/package.json',
    update: updateJsonVersion,
    required: [[/"version":\s*"/, 'version field']],
  },
  {
    file: 'packages/grafeo-datasets-python/pyproject.toml',
    update: updatePyproject,
    required: [
      [/^version = "[^"]*"/m, 'version field'],
      [/"grafeo>=[^"]*"/, 'grafeo>= floor'],
    ],
  },
  {
    file: 'packages/grafeo-datasets-python/grafeo_datasets/__init__.py',
    update: updatePyInit,
    required: [[/^__version__ = "/m, '__version__']],
  },
];

export function setVersion(root, version) {
  if (!VERSION.test(version)) throw new Error(`Version must be X.Y.Z or X.Y.Z-pre, got "${version}"`);
  // Check every target first, so a failure leaves nothing partly updated.
  const texts = TARGETS.map(({ file, required }) => {
    const before = readFileSync(join(root, file), 'utf8');
    for (const [pattern, name] of required) {
      if (!pattern.test(before)) throw new Error(`No ${name} found in ${file}`);
    }
    return before;
  });
  TARGETS.forEach(({ file, update }, i) => writeFileSync(join(root, file), update(texts[i], version)));
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  const args = process.argv.slice(2);
  const rootAt = args.indexOf('--root');
  if (rootAt !== -1 && !args[rootAt + 1]) {
    console.error('--root needs a directory');
    process.exit(1);
  }
  const root = rootAt === -1 ? join(dirname(fileURLToPath(import.meta.url)), '..', '..') : resolve(args[rootAt + 1]);
  const rest = rootAt === -1 ? args : args.filter((a, i) => i !== rootAt && i !== rootAt + 1);
  const version = rest[0] ?? '';
  try {
    setVersion(root, version);
  } catch (error) {
    console.error(error.message);
    process.exit(1);
  }
}
