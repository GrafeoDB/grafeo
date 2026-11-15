import { test } from 'node:test';
import assert from 'node:assert/strict';
import { execFileSync } from 'node:child_process';
import { existsSync, readFileSync, rmSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const pkgDir = join(dirname(fileURLToPath(import.meta.url)), '..');
const derived = join(pkgDir, '..', '..', 'datasets', 'silicon-valley', 'derived');
const stage = join(pkgDir, '..', '..', 'datasets', 'tools', 'stage.mjs');
const manifest = JSON.parse(readFileSync(join(pkgDir, 'package.json'), 'utf8'));

execFileSync(process.execPath, [stage], { cwd: pkgDir });

const load = (sub) => import(`@grafeo-db/datasets${sub}`);

test('root has nodes, edges and timeline', async () => {
  const { siliconValley } = await load('');
  assert.equal(siliconValley.name, 'silicon-valley');
  assert.ok(siliconValley.nodes.length >= 60);
  assert.ok(siliconValley.edges.length > 0);
  assert.ok(siliconValley.timeline.length > 0);
  const graph = JSON.parse(readFileSync(join(derived, 'graph.json'), 'utf8'));
  assert.deepEqual(siliconValley.nodes, graph.nodes);
  assert.deepEqual(siliconValley.edges, graph.edges);
});

test('root entry imports no subpath module', () => {
  const text = readFileSync(join(pkgDir, 'dist', 'index.js'), 'utf8');
  assert.doesNotMatch(text, /^\s*import\b/m);
  assert.doesNotMatch(text, /\bimport\s*\(/);
  assert.doesNotMatch(text, /\brequire\s*\(/);
});

for (const [sub, file] of [
  ['gql', 'silicon-valley.gql'],
  ['gql-with-vectors', 'silicon-valley.vectors.gql'],
  ['cypher', 'silicon-valley.cypher'],
]) {
  test(`${sub} statements match the derived file`, async () => {
    const mod = await load(`/silicon-valley/${sub}`);
    const text = readFileSync(join(derived, file), 'utf8');
    assert.equal(mod.default, text);
    assert.deepEqual(mod.statements, text.split('\n').filter((l) => l.trim() !== ''));
  });
}

test('turtle default export matches the derived file', async () => {
  const mod = await load('/silicon-valley/turtle');
  assert.equal(mod.default, readFileSync(join(derived, 'silicon-valley.ttl'), 'utf8'));
});

test('vectors decode to 384 floats per node and vectorOf finds middle-out', async () => {
  const { vectors, vectorOf } = await load('/silicon-valley/vectors');
  const meta = JSON.parse(readFileSync(join(derived, 'vectors.json'), 'utf8'));
  const raw = readFileSync(join(derived, 'vectors.f32'));
  assert.equal(vectors.dimension, 384);
  assert.equal(vectors.model, meta.model);
  assert.equal(vectors.keys.length, meta.nodes.length);
  assert.equal(vectors.data.length, vectors.keys.length * 384);
  assert.ok(vectors.data instanceof Float32Array);
  const expected = new Float32Array(raw.length / 4);
  for (let i = 0; i < expected.length; i++) expected[i] = raw.readFloatLE(i * 4);
  assert.deepEqual(vectors.data, expected);
  const v = vectorOf('middle-out');
  assert.equal(v.length, 384);
  assert.ok(Math.abs(Math.hypot(...v) - 1) < 1e-3);
  assert.equal(vectorOf('no-such-key'), undefined);
});

test('types exist for every export', () => {
  for (const [name, entry] of Object.entries(manifest.exports)) {
    assert.ok(existsSync(join(pkgDir, entry.types)), `${name}: ${entry.types}`);
    assert.ok(existsSync(join(pkgDir, entry.default)), `${name}: ${entry.default}`);
  }
});

const expectedFiles = [
  'LICENSE',
  'README.md',
  'dist/index.js',
  'dist/silicon-valley/cypher.js',
  'dist/silicon-valley/gql-with-vectors.js',
  'dist/silicon-valley/gql.js',
  'dist/silicon-valley/turtle.js',
  'dist/silicon-valley/vectors.js',
  'package.json',
  'types/index.d.ts',
  'types/silicon-valley/cypher.d.ts',
  'types/silicon-valley/gql-with-vectors.d.ts',
  'types/silicon-valley/gql.d.ts',
  'types/silicon-valley/turtle.d.ts',
  'types/silicon-valley/vectors.d.ts',
].sort();

const packedFiles = (extra) => {
  const out = execFileSync('npm', ['pack', '--dry-run', '--json', ...extra], {
    cwd: pkgDir,
    encoding: 'utf8',
    shell: process.platform === 'win32',
  });
  return JSON.parse(out.slice(out.search(/^\[$/m))).at(0).files.map((f) => f.path).sort();
};

test('packed files are exactly the expected set', () => {
  assert.deepEqual(packedFiles(['--ignore-scripts']), expectedFiles);
});

test('prepack stages a complete package on its own', () => {
  rmSync(join(pkgDir, 'dist'), { recursive: true, force: true });
  assert.deepEqual(packedFiles([]), expectedFiles);
});
