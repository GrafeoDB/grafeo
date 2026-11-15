import { test } from 'node:test';
import assert from 'node:assert/strict';
import { execFileSync, spawnSync } from 'node:child_process';
import { cpSync, mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const here = dirname(fileURLToPath(import.meta.url));
const script = join(here, '..', 'set-version.mjs');
const repo = join(here, '..', '..', '..');
const realManifest = join(repo, 'packages', 'grafeo-datasets-npm', 'package.json');
const PY_FILES = ['pyproject.toml', 'grafeo_datasets/__init__.py'];

function tempRoot() {
  const root = mkdtempSync(join(tmpdir(), 'set-version-'));
  mkdirSync(join(root, 'packages', 'grafeo-datasets-npm'), { recursive: true });
  cpSync(realManifest, join(root, 'packages', 'grafeo-datasets-npm', 'package.json'));
  mkdirSync(join(root, 'packages', 'grafeo-datasets-python', 'grafeo_datasets'), { recursive: true });
  for (const f of PY_FILES) {
    cpSync(join(repo, 'packages', 'grafeo-datasets-python', f), join(root, 'packages', 'grafeo-datasets-python', f));
  }
  return root;
}

test('set-version rewrites only the version field', () => {
  const root = tempRoot();
  try {
    const target = join(root, 'packages', 'grafeo-datasets-npm', 'package.json');
    const before = readFileSync(target, 'utf8');
    execFileSync(process.execPath, [script, '0.6.2', '--root', root]);
    const after = readFileSync(target, 'utf8');
    assert.equal(JSON.parse(after).version, '0.6.2');
    assert.equal(after, before.replace(/"version": "[^"]+"/, '"version": "0.6.2"'));
    assert.ok(after.endsWith('}\n'));
  } finally {
    rmSync(root, { recursive: true, force: true });
  }
});

test('set-version accepts a pre-release and refuses other forms', () => {
  const root = tempRoot();
  try {
    assert.equal(spawnSync(process.execPath, [script, '0.6.3-rc.1', '--root', root]).status, 0);
    for (const bad of ['v0.6.2', '0.6', '0.6.2.1', 'latest', '']) {
      const r = spawnSync(process.execPath, [script, bad, '--root', root]);
      assert.notEqual(r.status, 0, bad);
    }
  } finally {
    rmSync(root, { recursive: true, force: true });
  }
});

test('set-version fails clearly when a target has no version field', () => {
  const root = tempRoot();
  try {
    const target = join(root, 'packages', 'grafeo-datasets-npm', 'package.json');
    writeFileSync(target, '{\n  "name": "x",\n  "note": "0.6.2"\n}\n');
    const r = spawnSync(process.execPath, [script, '0.6.2', '--root', root], { encoding: 'utf8' });
    assert.notEqual(r.status, 0);
    assert.match(r.stderr, /No version field found in packages\/grafeo-datasets-npm\/package\.json/);
  } finally {
    rmSync(root, { recursive: true, force: true });
  }
});

test('set-version without --root sets the version of the tree it sits in', () => {
  // As the release jobs run it: the root is found from the script's own place.
  const root = tempRoot();
  try {
    const copy = join(root, 'datasets', 'tools', 'set-version.mjs');
    mkdirSync(dirname(copy), { recursive: true });
    cpSync(script, copy);
    const r = spawnSync(process.execPath, [copy, '0.6.9'], { encoding: 'utf8' });
    assert.equal(r.status, 0, r.stderr);
    const manifest = readFileSync(join(root, 'packages', 'grafeo-datasets-npm', 'package.json'), 'utf8');
    assert.equal(JSON.parse(manifest).version, '0.6.9');
    const init = readFileSync(join(root, 'packages', 'grafeo-datasets-python', 'grafeo_datasets', '__init__.py'), 'utf8');
    assert.match(init, /^__version__ = "0\.6\.9"$/m);
  } finally {
    rmSync(root, { recursive: true, force: true });
  }
});

test('set-version prints a clean message when --root has no value', () => {
  const r = spawnSync(process.execPath, [script, '0.6.2', '--root'], { encoding: 'utf8' });
  assert.notEqual(r.status, 0);
  assert.match(r.stderr, /--root needs a directory/);
  assert.doesNotMatch(r.stderr, /at .*\.mjs/);
});

test('set-version rewrites the Python version, the grafeo extra and __version__ only', () => {
  const root = tempRoot();
  try {
    execFileSync(process.execPath, [script, '0.6.2', '--root', root]);
    for (const f of PY_FILES) {
      const before = readFileSync(join(repo, 'packages', 'grafeo-datasets-python', f), 'utf8');
      const after = readFileSync(join(root, 'packages', 'grafeo-datasets-python', f), 'utf8');
      const expected = before
        .replace(/^version = ".*"$/m, 'version = "0.6.2"')
        .replace(/grafeo>=[0-9][^"]*/, 'grafeo>=0.6.2')
        .replace(/^__version__ = ".*"$/m, '__version__ = "0.6.2"');
      assert.equal(after, expected, f);
      assert.notEqual(after, before, f);
    }
    const pyproject = readFileSync(join(root, 'packages', 'grafeo-datasets-python', 'pyproject.toml'), 'utf8');
    assert.match(pyproject, /^version = "0\.6\.2"$/m);
    assert.match(pyproject, /grafeo = \["grafeo>=0\.6\.2"\]/);
  } finally {
    rmSync(root, { recursive: true, force: true });
  }
});

test('set-version fails clearly when the Python files lack their version lines', () => {
  const root = tempRoot();
  try {
    writeFileSync(join(root, 'packages', 'grafeo-datasets-python', 'grafeo_datasets', '__init__.py'), 'x = 1\n');
    const r = spawnSync(process.execPath, [script, '0.6.2', '--root', root], { encoding: 'utf8' });
    assert.notEqual(r.status, 0);
    assert.match(r.stderr, /No __version__ found in packages\/grafeo-datasets-python\/grafeo_datasets\/__init__\.py/);
  } finally {
    rmSync(root, { recursive: true, force: true });
  }
});

test('set-version writes nothing when a later target fails its check', () => {
  const root = tempRoot();
  try {
    writeFileSync(join(root, 'packages', 'grafeo-datasets-python', 'grafeo_datasets', '__init__.py'), 'x = 1\n');
    const files = ['packages/grafeo-datasets-npm/package.json', 'packages/grafeo-datasets-python/pyproject.toml'];
    const before = files.map((f) => readFileSync(join(root, f), 'utf8'));
    const r = spawnSync(process.execPath, [script, '0.6.2', '--root', root], { encoding: 'utf8' });
    assert.notEqual(r.status, 0);
    assert.deepEqual(files.map((f) => readFileSync(join(root, f), 'utf8')), before);
  } finally {
    rmSync(root, { recursive: true, force: true });
  }
});
