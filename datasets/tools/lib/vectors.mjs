import { createHash } from 'node:crypto';
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { emitGql } from './emit-gql.mjs';
import { f32Literal } from './numbers.mjs';
import { compareNodes } from './sort.mjs';

export const MODEL = 'Xenova/all-MiniLM-L6-v2';
export const DTYPE = 'q8';
export const DIMENSION = 384;

/** SHA-256 of a description, as lowercase hex. */
export const sha256 = (text) => createHash('sha256').update(text, 'utf8').digest('hex');

/** Cosine similarity of two equal-length number arrays. */
export function cosine(a, b) {
  let dot = 0;
  let na = 0;
  let nb = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    na += a[i] * a[i];
    nb += b[i] * b[i];
  }
  return dot / Math.sqrt(na * nb);
}

/** Little-endian float32 bytes of a Float32Array, whatever the platform's byte order. */
export function f32Bytes(data) {
  const buf = new Uint8Array(data.length * 4);
  const view = new DataView(buf.buffer);
  data.forEach((x, i) => view.setFloat32(i * 4, x, true));
  return buf;
}

/**
 * Read `vectors.json` and `vectors.f32` from a derived directory.
 * @returns {{model: string, dtype: string | undefined, dimension: number, keys: string[], data: Float32Array, hashes: Map<string, string>}}
 */
export function readVectors(dir) {
  const meta = JSON.parse(readFileSync(join(dir, 'vectors.json'), 'utf8'));
  const bytes = readFileSync(join(dir, 'vectors.f32'));
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const data = new Float32Array(bytes.byteLength / 4);
  for (let i = 0; i < data.length; i++) data[i] = view.getFloat32(i * 4, true);
  return {
    model: meta.model,
    dtype: meta.dtype,
    dimension: meta.dimension,
    offsets: meta.nodes.map((n) => n.offset),
    keys: meta.nodes.map((n) => n.key),
    data,
    hashes: new Map(meta.nodes.map((n) => [n.key, n.sha256])),
  };
}

/** Stored vectors are reused only when they come from this model at this dtype. */
export const reusable = (stored) => stored.model === MODEL && stored.dtype === DTYPE;

/**
 * Mean of the token rows of `hidden` (tokens x `dimension`, row-major) whose `mask` entry is 1, scaled to
 * unit length: the pipeline's `pooling: 'mean', normalize: true`.
 */
export function meanPoolNormalized(hidden, mask, dimension) {
  if (hidden.length !== mask.length * dimension) {
    throw new Error(`hidden state holds ${hidden.length} floats, expected ${mask.length} tokens x ${dimension}`);
  }
  const sum = new Float64Array(dimension);
  let count = 0;
  mask.forEach((m, t) => {
    if (Number(m) !== 1) return;
    count++;
    for (let j = 0; j < dimension; j++) sum[j] += hidden[t * dimension + j];
  });
  const norm = Math.sqrt(sum.reduce((s, x) => s + (x / count) ** 2, 0));
  return Float32Array.from(sum, (x) => x / count / norm);
}

/** Problems between a source and its stored vectors; each names `npm run embed`. */
export function vectorProblems(source, vectors) {
  const problems = [];
  if (vectors.dimension !== DIMENSION) {
    problems.push(`Stored dimension is ${vectors.dimension}, expected ${DIMENSION}. Run \`npm run embed\`.`);
  }
  if (vectors.data.length !== vectors.keys.length * vectors.dimension) {
    problems.push(
      `vectors.f32 holds ${vectors.data.length} floats, expected ${vectors.keys.length * vectors.dimension}. Run \`npm run embed\`.`,
    );
  }
  vectors.keys.forEach((k, i) => {
    if (vectors.offsets?.[i] !== i * vectors.dimension) {
      problems.push(`Offset of ${k} is ${vectors.offsets?.[i]}, expected ${i * vectors.dimension}. Run \`npm run embed\`.`);
    }
  });
  const stored = new Set(vectors.keys);
  for (const n of source.nodes) {
    if (!stored.has(n.key)) problems.push(`No vector for node ${n.key}. Run \`npm run embed\`.`);
    else if (vectors.hashes.get(n.key) !== sha256(n.description)) {
      problems.push(`Description of ${n.key} changed after embedding. Run \`npm run embed\`.`);
    }
  }
  const keys = new Set(source.nodes.map((n) => n.key));
  for (const k of vectors.keys) if (!keys.has(k)) problems.push(`Vector for unknown node ${k}. Run \`npm run embed\`.`);
  return problems;
}

/**
 * Plain GQL statements, then one embedding per node, then one vector index per label.
 */
export function emitGqlWithVectors(source, vectors) {
  const index = new Map(vectors.keys.map((k, i) => [k, i]));
  const lines = emitGql(source);
  const labels = new Set();
  for (const n of [...source.nodes].sort(compareNodes)) {
    const i = index.get(n.key);
    if (i === undefined) throw new Error(`no vector for node ${n.key}`);
    const v = Array.from(vectors.data.subarray(i * vectors.dimension, (i + 1) * vectors.dimension), f32Literal);
    lines.push(`MATCH (n:${n.label} {key: '${n.key}'}) SET n.embedding = vector([${v.join(', ')}])`);
    labels.add(n.label);
  }
  for (const label of [...labels].sort()) {
    lines.push(
      `CREATE VECTOR INDEX sv_${label.toLowerCase()}_description ON :${label}(embedding) DIMENSION ${vectors.dimension} METRIC 'cosine'`,
    );
  }
  return lines;
}
