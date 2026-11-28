import assert from 'node:assert/strict';
import { cpSync, existsSync, mkdtempSync, readFileSync, rmSync, truncateSync, writeFileSync } from 'node:fs';
import { spawnSync } from 'node:child_process';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { test } from 'node:test';
import { f32Literal, floatLiteral, numberLiteral } from '../lib/numbers.mjs';
import {
  DIMENSION,
  DTYPE,
  MODEL,
  cosine,
  emitGqlWithVectors,
  f32Bytes,
  meanPoolNormalized,
  readVectors,
  reusable,
  sha256,
  vectorProblems,
} from '../lib/vectors.mjs';

const datasets = fileURLToPath(new URL('../..', import.meta.url));
const derived = join(datasets, 'silicon-valley', 'derived');
const graph = JSON.parse(readFileSync(join(derived, 'graph.json'), 'utf8'));
const queries = JSON.parse(readFileSync(join(datasets, 'tests', 'fixtures', 'vector-queries.json'), 'utf8'));
const vectors = readVectors(derived);
const row = (i) => vectors.data.subarray(i * DIMENSION, (i + 1) * DIMENSION);

test('vectors.json lists every node once, in graph order', () => {
  assert.deepEqual(vectors.keys, graph.nodes.map((n) => n.key));
  assert.equal(new Set(vectors.keys).size, vectors.keys.length);
  assert.equal(vectors.dimension, 384);
  assert.equal(vectors.model, 'Xenova/all-MiniLM-L6-v2');
  assert.equal(vectors.data.length, graph.nodes.length * 384);
  const meta = JSON.parse(readFileSync(join(derived, 'vectors.json'), 'utf8'));
  assert.equal(meta.dtype, 'q8');
  assert.equal(vectors.dtype, 'q8');
  assert.deepEqual(meta.nodes.map((n) => n.offset), graph.nodes.map((_, i) => i * 384));
});

test('hashes match the descriptions', () => {
  for (const n of graph.nodes) assert.equal(vectors.hashes.get(n.key), sha256(n.description), n.key);
});

test('vectors are unit length', () => {
  for (let i = 0; i < vectors.keys.length; i++) {
    const norm = Math.sqrt(row(i).reduce((s, x) => s + x * x, 0));
    assert.ok(Math.abs(norm - 1) < 1e-3, `${vectors.keys[i]}: ${norm}`);
  }
});

test('nearest neighbour finds the expected node', () => {
  for (const q of queries) {
    assert.equal(q.vector.length, 384, q.text);
    let best = 0;
    let bestScore = -Infinity;
    for (let i = 0; i < vectors.keys.length; i++) {
      const s = cosine(q.vector, row(i));
      if (s > bestScore) [best, bestScore] = [i, s];
    }
    assert.ok(q.expect.includes(vectors.keys[best]), `${q.text}: got ${vectors.keys[best]}`);
  }
});

test('float32 literals round-trip', () => {
  for (const x of [0, 1, -1, 0.1, 1e-8, 123456.789, 0.03125, -0.0123456789]) {
    const f = Math.fround(x);
    assert.equal(Math.fround(Number(f32Literal(f))), f, String(x));
  }
  assert.equal(f32Literal(Math.fround(0.1)), '1e-1');
  for (const x of vectors.data) assert.equal(Math.fround(Number(f32Literal(x))), x);
});

test('the vectors GQL has plain statements, embeddings and one index per label', () => {
  const source = { name: 'f', nodes: graph.nodes.slice(0, 2).map((n) => ({ ...n, props: undefined })), edges: [] };
  const small = { ...vectors, keys: vectors.keys.slice(0, 2), data: vectors.data.subarray(0, 768) };
  const out = emitGqlWithVectors(source, small);
  assert.equal(out.length, 2 + 2 + new Set(source.nodes.map((n) => n.label)).size);
  assert.match(out[2], /^MATCH \(n:\w+ \{key: '[a-z-]+'\}\) SET n\.embedding = vector\(\[-?\d(?:\.\d+)?e-?\d+(?:, -?\d(?:\.\d+)?e-?\d+){383}\]\)$/);
  assert.match(out.at(-1), /^CREATE VECTOR INDEX sv_\w+_description ON :\w+\(embedding\) DIMENSION 384 METRIC 'cosine'$/);
});

test('build --check fails when a description changed after embedding', (t) => {
  const dir = mkdtempSync(join(tmpdir(), 'ds-vec-'));
  t.after(() => rmSync(dir, { recursive: true, force: true }));
  cpSync(join(datasets, 'silicon-valley'), dir, { recursive: true });
  const path = join(dir, 'source.json');
  const source = JSON.parse(readFileSync(path, 'utf8'));
  source.nodes[0].description += ' Changed.';
  writeFileSync(path, JSON.stringify(source));
  const build = fileURLToPath(new URL('../build.mjs', import.meta.url));
  const r = spawnSync(process.execPath, [build, '--check', '--source', path, '--out', join(dir, 'derived')], { encoding: 'utf8' });
  assert.equal(r.status, 1);
  assert.match(r.stderr, /npm run embed/);
});

test('float32 literals are always in exponent form and round-trip', () => {
  for (const x of [0, 1, -1, 1e-8, -1.5e-10, 5e-7, 123456789, 1e20, 1.17549435e-38, 3.6578045e-33, 3.4028235e38]) {
    const f = Math.fround(x);
    const lit = f32Literal(f);
    assert.match(lit, /^-?\d(\.\d+)?e-?\d+$/, lit);
    assert.equal(Math.fround(Number(lit)), f, lit);
  }
  assert.equal(f32Literal(Math.fround(3.4028235e38)), '3.4028235e38');
  assert.equal(f32Literal(Math.fround(1e-8)), '1e-8');
});

test('integers whose text has an exponent are written in exponent form', () => {
  assert.equal(numberLiteral(1e21), '1e21');
  assert.equal(numberLiteral(123), '123');
  assert.equal(numberLiteral(-4), '-4');
});

test('float literals for properties use the shortest exponent form', () => {
  assert.equal(floatLiteral(5.2), '5.2e0');
  assert.equal(floatLiteral(-0.5), '-5e-1');
  assert.equal(floatLiteral(3.6578045e-8), '3.6578045e-8');
});

test('f32 bytes are little-endian', () => {
  assert.deepEqual([...f32Bytes(Float32Array.of(1))], [0, 0, 0x80, 0x3f]);
});

function tempDerived(t) {
  const dir = mkdtempSync(join(tmpdir(), 'ds-vec-'));
  t.after(() => rmSync(dir, { recursive: true, force: true }));
  cpSync(join(datasets, 'silicon-valley'), dir, { recursive: true });
  return { dir, out: join(dir, 'derived'), path: join(dir, 'source.json') };
}
const readSource = (path) => JSON.parse(readFileSync(path, 'utf8'));
const editMeta = (out, fn) => {
  const p = join(out, 'vectors.json');
  const meta = JSON.parse(readFileSync(p, 'utf8'));
  fn(meta);
  writeFileSync(p, JSON.stringify(meta));
};
const buildCli = fileURLToPath(new URL('../build.mjs', import.meta.url));

test('a truncated vectors.f32 is rejected, naming npm run embed', (t) => {
  const { out, path } = tempDerived(t);
  truncateSync(join(out, 'vectors.f32'), 98304 - 4);
  const problems = vectorProblems(readSource(path), readVectors(out));
  assert.ok(problems.length > 0);
  assert.ok(problems.every((p) => p.includes('npm run embed')), problems.join(', '));
  const r = spawnSync(process.execPath, [buildCli, '--source', path, '--out', out], { encoding: 'utf8' });
  assert.equal(r.status, 1);
  assert.match(r.stderr, /npm run embed/);
});

test('a wrong offset is rejected, naming npm run embed', (t) => {
  const { out, path } = tempDerived(t);
  editMeta(out, (m) => { m.nodes[3].offset += 1; });
  const problems = vectorProblems(readSource(path), readVectors(out));
  assert.ok(problems.some((p) => /offset/i.test(p) && p.includes('npm run embed')), problems.join(', '));
});

test('a wrong dimension is rejected, naming npm run embed', (t) => {
  const { out, path } = tempDerived(t);
  editMeta(out, (m) => { m.dimension = 256; });
  const problems = vectorProblems(readSource(path), readVectors(out));
  assert.ok(problems.some((p) => /dimension/i.test(p) && p.includes('npm run embed')), problems.join(', '));
});

test('a node missing from vectors.json is rejected, naming npm run embed', (t) => {
  const { out, path } = tempDerived(t);
  editMeta(out, (m) => { m.nodes.pop(); });
  const problems = vectorProblems(readSource(path), readVectors(out));
  assert.ok(problems.some((p) => p.startsWith('No vector for node') && p.includes('npm run embed')), problems.join(', '));
});

test('without vectors.json the build skips the vectors file', (t) => {
  const { out, path } = tempDerived(t);
  rmSync(join(out, 'vectors.json'));
  rmSync(join(out, 'vectors.f32'));
  rmSync(join(out, 'silicon-valley.vectors.gql'));
  const r = spawnSync(process.execPath, [buildCli, '--source', path, '--out', out], { encoding: 'utf8' });
  assert.equal(r.status, 0, r.stderr);
  assert.equal(existsSync(join(out, 'silicon-valley.vectors.gql')), false);
  assert.match(r.stdout, /Built 9 file/);
});

test('stored vectors are reused only when both model and dtype match', () => {
  assert.equal(reusable({ model: MODEL, dtype: DTYPE }), true);
  assert.equal(reusable({ model: MODEL, dtype: 'fp32' }), false);
  assert.equal(reusable({ model: MODEL, dtype: undefined }), false);
  assert.equal(reusable({ model: 'Xenova/other', dtype: DTYPE }), false);
  assert.equal(reusable(vectors), true);
});

test('mean pooling averages the unmasked tokens and normalises', () => {
  // Three tokens of dimension 2; the third is padding.
  const hidden = Float32Array.of(0, 0, 6, 8, 100, 100);
  const v = meanPoolNormalized(hidden, [1, 1, 0], 2);
  assert.ok(v instanceof Float32Array && v.length === 2);
  assert.ok(Math.abs(v[0] - 0.6) < 1e-6 && Math.abs(v[1] - 0.8) < 1e-6, String(v));
  assert.throws(() => meanPoolNormalized(hidden, [1, 1], 2), /tokens/);
});
