import { test } from 'node:test';
import assert from 'node:assert/strict';
import { execFileSync } from 'node:child_process';
import { cpSync, mkdirSync, mkdtempSync, readdirSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { dirname, join } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const here = dirname(fileURLToPath(import.meta.url));
const stage = join(here, '..', 'stage.mjs');
const derived = join(here, '..', '..', 'silicon-valley', 'derived');

test('stage keeps U+2028 and U+2029 in text intact', async () => {
  const tmp = mkdtempSync(join(tmpdir(), 'stage-'));
  try {
    const src = join(tmp, 'derived');
    cpSync(derived, src, { recursive: true });
    const odd = 'line\u2028sep\u2029end';
    const graph = JSON.parse(readFileSync(join(src, 'graph.json'), 'utf8'));
    graph.nodes[0].description = odd;
    writeFileSync(join(src, 'graph.json'), JSON.stringify(graph));
    writeFileSync(join(src, 'silicon-valley.ttl'), `# ${odd}\n`);
    writeFileSync(join(tmp, 'package.json'), '{"type":"module"}\n');
    const dist = join(tmp, 'dist');
    execFileSync(process.execPath, [stage, '--derived', src, '--dist', dist]);
    const index = await import(pathToFileURL(join(dist, 'index.js')).href);
    assert.equal(index.siliconValley.nodes[0].description, odd);
    const ttl = await import(pathToFileURL(join(dist, 'silicon-valley', 'turtle.js')).href);
    assert.equal(ttl.default, `# ${odd}\n`);
  } finally {
    rmSync(tmp, { recursive: true, force: true });
  }
});

test('stage writes the Python data folder: the expected files, byte-identical to derived', () => {
  const tmp = mkdtempSync(join(tmpdir(), 'stage-py-'));
  try {
    const target = join(tmp, 'data');
    mkdirSync(target);
    writeFileSync(join(target, 'stale.txt'), 'old');
    execFileSync(process.execPath, [stage, '--derived', derived, '--dist', join(tmp, 'dist'), '--python-data', target]);
    const expected = [
      'graph.json',
      'silicon-valley.cypher',
      'silicon-valley.gql',
      'silicon-valley.ttl',
      'silicon-valley.vectors.gql',
      'timeline.json',
      'vectors.f32',
      'vectors.json',
    ];
    assert.deepEqual(readdirSync(target).sort(), expected);
    for (const name of expected) {
      assert.ok(readFileSync(join(target, name)).equals(readFileSync(join(derived, name))), name);
    }
  } finally {
    rmSync(tmp, { recursive: true, force: true });
  }
});
