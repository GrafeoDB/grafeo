#!/usr/bin/env node
import { existsSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { parseArgs } from 'node:util';
import { f32Literal } from './lib/numbers.mjs';
import {
  DIMENSION,
  DTYPE,
  MODEL,
  cosine,
  f32Bytes,
  meanPoolNormalized,
  readVectors,
  reusable,
  sha256,
} from './lib/vectors.mjs';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const { values } = parseArgs({
  options: {
    models: { type: 'string', default: process.env.GRAFEO_MINILM_DIR },
    out: { type: 'string', default: join(root, 'silicon-valley', 'derived') },
    queries: { type: 'string', default: join(root, 'tests', 'fixtures', 'vector-queries.json') },
  },
});

if (!values.models) {
  console.error(`Pass --models <dir> (or set GRAFEO_MINILM_DIR); <dir> holds ${MODEL}.`);
  process.exit(1);
}
const modelDir = resolve(values.models);
if (!existsSync(join(modelDir, MODEL, 'onnx', 'model_quantized.onnx'))) {
  console.error(`Missing ${MODEL}/onnx/model_quantized.onnx in ${modelDir}.`);
  process.exit(1);
}

const { AutoTokenizer, ModelRegistry, env, pipeline } = await import('@huggingface/transformers');
env.allowRemoteModels = false;
env.localModelPath = modelDir;
// The playground runs the quantized model; Node defaults to fp32, so dtype is explicit.
const extractor = await pipeline('feature-extraction', MODEL, { dtype: DTYPE });
// The library's own resolution of the files this dtype loads must be the quantized model the playground ships.
const files = await ModelRegistry.get_pipeline_files('feature-extraction', MODEL, { dtype: DTYPE });
if (!files.includes('onnx/model_quantized.onnx') || files.some((f) => f.endsWith('.onnx') && f !== 'onnx/model_quantized.onnx')) {
  console.error(`Wrong model file: dtype ${DTYPE} resolves to ${files.filter((f) => f.endsWith('.onnx')).join(', ')}, expected onnx/model_quantized.onnx.`);
  process.exit(1);
}
const embed = async (text) => {
  const t = await extractor(text, { pooling: 'mean', normalize: true });
  const v = Float32Array.from(t.data);
  if (v.length !== DIMENSION) throw new Error(`expected ${DIMENSION} floats, got ${v.length}`);
  return v;
};

/**
 * The browser runs the same model file on onnxruntime-web (WebAssembly), not on onnxruntime-node. This embeds
 * `text` with onnxruntime-web's WebAssembly backend, on the tokens transformers.js produces, so the stored
 * vectors can be compared with what a browser query gets.
 */
async function embedInBrowserRuntime(text) {
  const ort = await import('onnxruntime-web');
  ort.env.wasm.numThreads = 1;
  const session = await ort.InferenceSession.create(readFileSync(join(modelDir, MODEL, 'onnx', 'model_quantized.onnx')), {
    executionProviders: ['wasm'],
  });
  const tokenizer = await AutoTokenizer.from_pretrained(MODEL);
  const ids = Array.from(tokenizer(text).input_ids.data, BigInt);
  const tensor = (values) => new ort.Tensor('int64', BigInt64Array.from(values), [1, ids.length]);
  const mask = ids.map(() => 1n);
  const feeds = { input_ids: tensor(ids), attention_mask: tensor(mask) };
  if (session.inputNames.includes('token_type_ids')) feeds.token_type_ids = tensor(ids.map(() => 0n));
  const output = await session.run(feeds);
  return meanPoolNormalized(output[session.outputNames[0]].data, mask, DIMENSION);
}

/** Exit when `cosine` is below 0.9999. */
function selfCheck(name, score, failure) {
  console.log(`${name} (${JSON.stringify(queries[0].text)}): cosine ${score.toFixed(6)} to the committed vector.`);
  if (!(score >= 0.9999)) {
    console.error(`${name} failed: ${failure}`);
    process.exit(1);
  }
}

// A committed reference vector must match a fresh embedding (this catches a wrong model or dtype) and the
// browser runtime's embedding of the same text (this catches a runtime that drifts from the browser).
const queries = JSON.parse(readFileSync(values.queries, 'utf8'));
const reference = queries[0]?.vector;
if (Array.isArray(reference) && reference.length === DIMENSION) {
  selfCheck(
    'Reference check',
    cosine(reference, await embed(queries[0].text)),
    'a fresh embedding does not match the committed query vector (wrong model or dtype?).',
  );
  selfCheck(
    'Browser runtime check',
    cosine(reference, await embedInBrowserRuntime(queries[0].text)),
    'onnxruntime-web does not reproduce the committed query vector, so browser queries would not line up.',
  );
} else {
  console.log('Reference checks skipped: no committed query vector yet.');
}

const graph = JSON.parse(readFileSync(join(values.out, 'graph.json'), 'utf8'));
const nodes = graph.nodes;

let old = null;
if (existsSync(join(values.out, 'vectors.json')) && existsSync(join(values.out, 'vectors.f32'))) {
  try {
    old = readVectors(values.out);
    if (!reusable(old)) {
      console.log(`Stored vectors come from ${old.model} at dtype ${old.dtype}, not ${MODEL} at ${DTYPE}; embedding every node again.`);
      old = null;
    }
  } catch (err) {
    console.warn(`Cannot read the stored vectors (${err.message}); embedding every node again.`);
    old = null;
  }
}

const data = new Float32Array(nodes.length * DIMENSION);
let embedded = 0;
for (const [i, n] of nodes.entries()) {
  const hash = sha256(n.description);
  const j = old ? old.keys.indexOf(n.key) : -1;
  if (j >= 0 && old.hashes.get(n.key) === hash) {
    data.set(old.data.subarray(j * DIMENSION, (j + 1) * DIMENSION), i * DIMENSION);
  } else {
    data.set(await embed(n.description), i * DIMENSION);
    embedded++;
  }
}

let minNorm = Infinity;
let maxNorm = -Infinity;
for (let i = 0; i < nodes.length; i++) {
  const norm = Math.sqrt(data.subarray(i * DIMENSION, (i + 1) * DIMENSION).reduce((s, x) => s + x * x, 0));
  minNorm = Math.min(minNorm, norm);
  maxNorm = Math.max(maxNorm, norm);
}
console.log(`Norms ${minNorm.toFixed(6)} to ${maxNorm.toFixed(6)}.`);
if (Math.abs(minNorm - 1) >= 1e-3 || Math.abs(maxNorm - 1) >= 1e-3) {
  console.error('Norm check failed: a vector is not unit length.');
  process.exit(1);
}

mkdirSync(values.out, { recursive: true });
writeFileSync(join(values.out, 'vectors.f32'), f32Bytes(data));
const meta = {
  model: MODEL,
  dtype: DTYPE,
  dimension: DIMENSION,
  nodes: nodes.map((n, i) => ({ key: n.key, offset: i * DIMENSION, sha256: sha256(n.description) })),
};
writeFileSync(join(values.out, 'vectors.json'), `${JSON.stringify(meta, null, 2)}\n`, 'utf8');
console.log(`Embedded ${embedded} of ${nodes.length} node(s); wrote vectors.f32 and vectors.json.`);

// Query vectors for the tests and the playground.
for (const q of queries) {
  const v = await embed(q.text);
  q.vector = Array.from(v, (x) => Number(f32Literal(x)));
  const scored = nodes
    .map((n, i) => ({ key: n.key, score: cosine(v, data.subarray(i * DIMENSION, (i + 1) * DIMENSION)) }))
    .sort((a, b) => b.score - a.score)
    .slice(0, 3);
  console.log(`${JSON.stringify(q.text)}: ${scored.map((s) => `${s.key} ${s.score.toFixed(4)}`).join(', ')}`);
}
// Written with shortest float32 literals, one query per object.
const text = `[\n${queries
  .map((q) => `  {\n    "text": ${JSON.stringify(q.text)},\n    "expect": ${JSON.stringify(q.expect)},\n    "vector": [${q.vector.map((x) => f32Literal(x)).join(', ')}]\n  }`)
  .join(',\n')}\n]\n`;
writeFileSync(values.queries, text, 'utf8');
