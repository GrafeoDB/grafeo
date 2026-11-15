// Stages both packages: reads datasets/silicon-valley/derived and writes
// packages/grafeo-datasets-npm/dist (JavaScript modules) and
// packages/grafeo-datasets-python/grafeo_datasets/data/silicon_valley (verbatim files).
// Run by `prepack`; paths come from this file's location.
import { copyFileSync, mkdirSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = join(dirname(fileURLToPath(import.meta.url)), '..', '..');
const arg = (name) => {
  const at = process.argv.indexOf(name);
  return at === -1 ? undefined : process.argv[at + 1];
};
const derived = arg('--derived') ?? join(root, 'datasets', 'silicon-valley', 'derived');
const dist = arg('--dist') ?? join(root, 'packages', 'grafeo-datasets-npm', 'dist');

// With --dist alone (a test run), the Python package is left alone unless --python-data is given.
const pythonData =
  arg('--python-data') ??
  (arg('--dist') === undefined
    ? join(root, 'packages', 'grafeo-datasets-python', 'grafeo_datasets', 'data', 'silicon_valley')
    : undefined);
const PYTHON_FILES = [
  'graph.json',
  'timeline.json',
  'silicon-valley.gql',
  'silicon-valley.cypher',
  'silicon-valley.ttl',
  'silicon-valley.vectors.gql',
  'vectors.f32',
  'vectors.json',
];

const read = (name) => readFileSync(join(derived, name));
const text = (name) => read(name).toString('utf8');
const json = (name) => JSON.parse(text(name));
// JSON string literals are valid ES2019, U+2028 and U+2029 included.
const lit = (value) => JSON.stringify(value);

const out = (path, body) => {
  const file = join(dist, path);
  mkdirSync(dirname(file), { recursive: true });
  writeFileSync(file, body.replace(/\r\n/g, '\n'));
};

/** One text constant, exported whole and as statements (one per line). */
const statementsModule = (name) =>
  `const text = ${lit(text(name))};
export default text;
export const statements = text.split('\\n').filter((line) => line.trim() !== '');
`;

rmSync(dist, { recursive: true, force: true });

const graph = json('graph.json');
out(
  'index.js',
  `export const siliconValley = ${lit({
    name: graph.name,
    nodes: graph.nodes,
    edges: graph.edges,
    timeline: json('timeline.json'),
  })};\n`,
);

out('silicon-valley/gql.js', statementsModule('silicon-valley.gql'));
out('silicon-valley/gql-with-vectors.js', statementsModule('silicon-valley.vectors.gql'));
out('silicon-valley/cypher.js', statementsModule('silicon-valley.cypher'));
out('silicon-valley/turtle.js', `export default ${lit(text('silicon-valley.ttl'))};\n`);

const meta = json('vectors.json');
const keys = meta.nodes.map((n) => n.key);
out(
  'silicon-valley/vectors.js',
  `const bytes = Uint8Array.from(atob(${lit(read('vectors.f32').toString('base64'))}), (c) => c.charCodeAt(0));
const view = new DataView(bytes.buffer);
const data = new Float32Array(bytes.length / 4);
for (let i = 0; i < data.length; i++) data[i] = view.getFloat32(i * 4, true);

export const vectors = {
  model: ${lit(meta.model)},
  dimension: ${lit(meta.dimension)},
  keys: ${lit(keys)},
  data,
};

const index = new Map(vectors.keys.map((key, i) => [key, i]));

export function vectorOf(key) {
  const i = index.get(key);
  return i === undefined ? undefined : vectors.data.subarray(i * vectors.dimension, (i + 1) * vectors.dimension);
}
`,
);

if (pythonData !== undefined) {
  rmSync(pythonData, { recursive: true, force: true });
  mkdirSync(pythonData, { recursive: true });
  for (const name of PYTHON_FILES) copyFileSync(join(derived, name), join(pythonData, name));
}
