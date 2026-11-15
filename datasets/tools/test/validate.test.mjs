import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { test } from 'node:test';
import { compareDates, validate } from '../lib/validate.mjs';

const valid = () => JSON.parse(readFileSync(new URL('./fixtures/valid.json', import.meta.url), 'utf8'));
const node = (key, label = 'Person', extra = {}) => ({ key, label, name: key, description: 'Fine.', ...extra });

/** Assert that some problem mentions `text`. */
function expectProblem(source, text) {
  const problems = validate(source);
  assert.ok(problems.some((p) => p.includes(text)), `expected "${text}" in ${JSON.stringify(problems)}`);
}

test('a valid source has no problems', () => {
  assert.deepEqual(validate(valid()), []);
});

test('duplicate key', () => {
  const s = valid();
  s.nodes.push(node('ann'));
  expectProblem(s, 'duplicate key: ann');
});

test('key not kebab case', () => {
  const s = valid();
  s.nodes.push(node('Big_Head'));
  expectProblem(s, 'key not kebab case');
});

test('unknown label', () => {
  const s = valid();
  s.nodes.push(node('x', 'Robot'));
  expectProblem(s, 'unknown label Robot');
});

test('unknown edge type', () => {
  const s = valid();
  s.edges.push({ type: 'HATES', source: 'ann', target: 'bob' });
  expectProblem(s, 'unknown edge type HATES');
});

test('edge end missing', () => {
  const s = valid();
  s.edges.push({ type: 'KNOWS', source: 'ann', target: 'ghost' });
  expectProblem(s, 'edge end missing: ghost');
});

test('edge end has the wrong label', () => {
  const s = valid();
  s.edges.push({ type: 'WORKS_AT', source: 'acme', target: 'acme' });
  expectProblem(s, 'edge end has the wrong label');
});

test('invalid date', () => {
  for (const date of ['2014-13', '2014-02-30']) {
    const s = valid();
    s.edges.push({ type: 'FOUNDED', source: 'ann', target: 'acme', props: { date } });
    expectProblem(s, `invalid date in date: ${date}`);
  }
  const s = valid();
  s.nodes[2].props.founded = 'soon';
  expectProblem(s, 'invalid date in founded');
});

test('from after to', () => {
  const s = valid();
  s.edges[1].props = { from: '2015-01', to: '2014-12-31' };
  expectProblem(s, 'from after to');
});

test('equal from and to is allowed', () => {
  const s = valid();
  s.edges[1].props = { from: '2014-03-01', to: '2014-03' };
  assert.deepEqual(validate(s), []);
});

test('property not allowed on this edge type', () => {
  const s = valid();
  s.edges[0].props = { role: 'Boss' };
  expectProblem(s, 'property not allowed on this edge type: role');
});

test('missing description', () => {
  const s = valid();
  s.nodes.push({ key: 'x', label: 'Person', name: 'X' });
  expectProblem(s, 'missing description');
});

test('description over two sentences', () => {
  const s = valid();
  s.nodes.push(node('x', 'Person', { description: 'One. Two! Three?' }));
  expectProblem(s, 'description over two sentences');
});

test('abbreviations do not count as sentence ends', () => {
  const s = valid();
  s.nodes.push(node('x', 'Person', { description: 'Works at Acme Inc. as a dev. Likes tea.' }));
  assert.deepEqual(validate(s), []);
});

test('month dates compare by their first day', () => {
  assert.ok(compareDates('2014-03', '2014-03-15') < 0);
  assert.equal(compareDates('2014-03', '2014-03-01'), 0);
  assert.ok(compareDates('2014-04', '2014-03-31') > 0);
});

test('a duplicate key keeps the first node label for edge checks', () => {
  const s = valid();
  s.nodes.push(node('acme', 'Person'));
  s.edges.push({ type: 'WORKS_AT', source: 'ann', target: 'acme' });
  const problems = validate(s);
  assert.ok(problems.includes('duplicate key: acme'));
  assert.ok(!problems.some((p) => p.includes('wrong label')), JSON.stringify(problems));
});

test('control characters in strings are rejected', () => {
  for (const ch of ['\u0000', '\u001f', '\u007f', '\u0085', String.fromCharCode(8232), String.fromCharCode(8233)]) {
    const s = valid();
    s.nodes.push(node('x', 'Person', { name: `a${ch}b` }));
    expectProblem(s, 'node "x": control character in name');
  }
  const d = valid();
  d.nodes.push(node('y', 'Person', { description: `a${String.fromCharCode(8232)}b` }));
  expectProblem(d, 'node "y": control character in description');
  const p = valid();
  p.nodes.push(node('z', 'Person', { props: { title: 'a\u0001b' } }));
  expectProblem(p, 'node "z": control character in title');
  const e = valid();
  e.edges[1].props = { role: 'a\u0007b' };
  expectProblem(e, 'edge WORKS_AT bob -> acme: control character in role');
});

test('tab, newline and carriage return are allowed in strings', () => {
  const s = valid();
  s.nodes.push(node('x', 'Person', { name: 'a\tb\nc\rd' }));
  assert.deepEqual(validate(s), []);
});

test('numbers that are not finite or print in exponent form are rejected', () => {
  for (const n of [1e21, 1e-7, NaN, Infinity]) {
    const s = valid();
    s.nodes.push(node('x', 'Person', { props: { weissman_score: n } }));
    expectProblem(s, 'node "x": number not written as plain decimal in weissman_score');
  }
  const e = valid();
  e.edges.push({ type: 'INVESTED_IN', source: 'ann', target: 'acme', props: { amount_usd: 1e21 } });
  expectProblem(e, 'number not written as plain decimal in amount_usd');
  const ok = valid();
  ok.nodes.push(node('x', 'Person', { props: { weissman_score: 5.2 } }));
  ok.edges.push({ type: 'INVESTED_IN', source: 'ann', target: 'acme', props: { amount_usd: 10000000 } });
  assert.deepEqual(validate(ok), []);
});

test('leap days', () => {
  const s = valid();
  s.edges.push({ type: 'FOUNDED', source: 'ann', target: 'acme', props: { date: '2016-02-29' } });
  assert.deepEqual(validate(s), []);
  s.edges[3].props.date = '2015-02-29';
  expectProblem(s, 'invalid date in date: 2015-02-29');
});
