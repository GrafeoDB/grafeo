#!/usr/bin/env node
import { existsSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { parseArgs } from 'node:util';
import { emitCypher } from './lib/emit-cypher.mjs';
import { emitGql } from './lib/emit-gql.mjs';
import { emitGraph } from './lib/emit-graph.mjs';
import { emitTimeline } from './lib/emit-timeline.mjs';
import { emitTurtle } from './lib/emit-turtle.mjs';
import { validate } from './lib/validate.mjs';
import { emitGqlWithVectors, readVectors, vectorProblems } from './lib/vectors.mjs';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '..');

/** Derived files as `{file, emit}` pairs; `emit(source)` returns the file text, with LF line endings. */
const json = (value) => `${JSON.stringify(value, null, 2)}\n`;
const lines = (list) => `${list.join('\n')}\n`;
const emitters = [
  { file: 'graph.json', emit: (s) => json(emitGraph(s)) },
  { file: 'timeline.json', emit: (s) => json(emitTimeline(s)) },
  { file: 'silicon-valley.gql', emit: (s) => lines(emitGql(s)) },
  { file: 'silicon-valley.cypher', emit: (s) => lines(emitCypher(s)) },
  { file: 'silicon-valley.ttl', emit: emitTurtle },
];

const { values } = parseArgs({
  options: {
    source: { type: 'string', default: join(root, 'silicon-valley', 'source.json') },
    out: { type: 'string', default: join(root, 'silicon-valley', 'derived') },
    check: { type: 'boolean', default: false },
  },
});

if (!existsSync(values.source)) {
  console.error(`Source not found: ${values.source}. Pass --source <file> or create the default source.`);
  process.exit(1);
}

let text;
try {
  text = readFileSync(values.source, 'utf8');
} catch (err) {
  console.error(`Cannot read ${values.source}: ${err.message}`);
  process.exit(1);
}
let source;
try {
  source = JSON.parse(text);
} catch (err) {
  console.error(`Cannot parse ${values.source} as JSON: ${err.message}`);
  process.exit(1);
}
const problems = validate(source);
if (problems.length > 0) {
  for (const p of problems) console.error(p);
  console.error(`${problems.length} problem(s) in ${values.source}`);
  process.exit(1);
}

// Vectors GQL exists only where vectors do; when they do, they must cover every node.
if (existsSync(join(values.out, 'vectors.json'))) {
  let vectors;
  try {
    vectors = readVectors(values.out);
  } catch (err) {
    console.error(`Cannot read the vectors in ${values.out}: ${err.message}. Run \`npm run embed\`.`);
    process.exit(1);
  }
  const vp = vectorProblems(source, vectors);
  if (vp.length > 0) {
    for (const p of vp) console.error(p);
    process.exit(1);
  }
  emitters.push({ file: 'silicon-valley.vectors.gql', emit: (s) => lines(emitGqlWithVectors(s, vectors)) });
}

const outputs = emitters.map(({ file, emit }) => ({ path: join(values.out, file), text: emit(source) }));

if (values.check) {
  const stale = outputs.filter(({ path, text }) => !existsSync(path) || readFileSync(path, 'utf8') !== text);
  for (const { path } of stale) console.error(`Stale or missing: ${path}`);
  if (stale.length > 0) process.exit(1);
  console.log(`OK: ${source.nodes.length} nodes, ${source.edges.length} edges, ${outputs.length} derived file(s) current.`);
} else {
  for (const { path, text } of outputs) {
    mkdirSync(dirname(path), { recursive: true });
    writeFileSync(path, text, 'utf8');
  }
  console.log(`Built ${outputs.length} file(s) from ${source.nodes.length} nodes, ${source.edges.length} edges.`);
}
