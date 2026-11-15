import assert from 'node:assert/strict';
import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from 'node:fs';
import { spawnSync } from 'node:child_process';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { test } from 'node:test';

const build = fileURLToPath(new URL('../build.mjs', import.meta.url));

function run(t, name, setup) {
  const dir = mkdtempSync(join(tmpdir(), 'ds-'));
  t.after(() => rmSync(dir, { recursive: true, force: true }));
  const path = join(dir, name);
  setup(path);
  const r = spawnSync(process.execPath, [build, '--source', path, '--out', dir], { encoding: 'utf8' });
  return { r, path };
}

test('a source that is not valid JSON gives one line and exit 1', (t) => {
  const { r, path } = run(t, 'bad.json', (p) => writeFileSync(p, '{ nope'));
  assert.equal(r.status, 1);
  assert.equal(r.stderr.trim().split('\n').length, 1, r.stderr);
  assert.ok(r.stderr.startsWith(`Cannot parse ${path} as JSON`), r.stderr);
  assert.ok(!r.stderr.includes('    at '));
});

test('a source that cannot be read gives one line and exit 1', (t) => {
  const { r, path } = run(t, 'dir.json', (p) => mkdirSync(p));
  assert.equal(r.status, 1);
  assert.equal(r.stderr.trim().split('\n').length, 1, r.stderr);
  assert.ok(r.stderr.startsWith(`Cannot read ${path}`), r.stderr);
});
