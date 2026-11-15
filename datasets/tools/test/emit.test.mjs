import assert from 'node:assert/strict';
import { test } from 'node:test';
import { emitCypher } from '../lib/emit-cypher.mjs';
import { emitGql } from '../lib/emit-gql.mjs';
import { emitGraph } from '../lib/emit-graph.mjs';
import { emitTimeline } from '../lib/emit-timeline.mjs';

const source = () => ({
  name: 'fixture',
  nodes: [
    { key: 'richard-hendricks', label: 'Person', name: 'Richard Hendricks', description: 'A founder.' },
    { key: 'pied-piper', label: 'Company', name: 'Pied Piper', description: 'A startup.', props: { founded: '2014-04' } },
    { key: 'erlich-bachman', label: 'Person', name: 'Erlich', description: 'Erlich\'s "palapa" \\ test' },
  ],
  edges: [
    { type: 'WORKS_AT', source: 'richard-hendricks', target: 'pied-piper', props: { role: 'CEO', from: '2014-03', to: '2014-03-15' } },
    { type: 'KNOWS', source: 'richard-hendricks', target: 'erlich-bachman' },
  ],
});

test('gql node statement', () => {
  assert.equal(
    emitGql(source()).find((l) => l.includes("'richard-hendricks', name")),
    "INSERT (:Person {key: 'richard-hendricks', name: 'Richard Hendricks', description: 'A founder.'})",
  );
});

test('gql edge statement with a month date', () => {
  const line = emitGql(source()).find((l) => l.includes('WORKS_AT'));
  assert.equal(
    line,
    "MATCH (a:Person {key: 'richard-hendricks'}), (b:Company {key: 'pied-piper'}) INSERT (a)-[:WORKS_AT {role: 'CEO', from: DATE '2014-03-01', from_precision: 'month', to: DATE '2014-03-15'}]->(b)",
  );
});

test('month-only date gets a precision property', () => {
  const lines = emitGql(source());
  assert.ok(lines.some((l) => l.includes("from_precision: 'month'")));
  assert.ok(lines.some((l) => l.includes("founded: DATE '2014-04-01', founded_precision: 'month'")));
});

test('a day date gets none', () => {
  const line = emitGql(source()).find((l) => l.includes('WORKS_AT'));
  assert.ok(!line.includes('to_precision'));
});

test('quotes and backslashes are escaped', () => {
  const line = emitGql(source()).find((l) => l.includes('erlich-bachman') && l.startsWith('INSERT'));
  assert.ok(line.includes(String.raw`description: 'Erlich\'s "palapa" \\ test'`), line);
});

test('newlines and carriage returns are escaped, numbers and booleans are bare', () => {
  const s = source();
  s.nodes[0].description = 'a\nb\rc';
  s.nodes[0].props = { weissman_score: 5.2, flag: true };
  const line = emitGql(s).find((l) => l.startsWith('INSERT (:Person {key: \'richard'));
  assert.ok(line.includes(String.raw`'a\nb\rc'`));
  assert.ok(line.includes('flag: true, weissman_score: 5.2e0'));
  const cy = emitCypher(s).find((l) => l.startsWith("CREATE (:Person {key: 'richard"));
  assert.ok(cy.includes('flag: true, weissman_score: 5.2e0'));
});

test('a float property is a float literal even when it is a whole number', () => {
  const s = source();
  s.nodes[0].props = { weissman_score: 3, tier: 3 };
  const line = emitGql(s).find((l) => l.startsWith("INSERT (:Person {key: 'richard"));
  assert.ok(line.includes('tier: 3, weissman_score: 3e0'), line);
  const cy = emitCypher(s).find((l) => l.startsWith("CREATE (:Person {key: 'richard"));
  assert.ok(cy.includes('tier: 3, weissman_score: 3e0'), cy);
});

test('cypher uses CREATE and date()', () => {
  const lines = emitCypher(source());
  assert.ok(lines.every((l) => !l.includes('INSERT') && !l.endsWith(';')));
  assert.ok(lines[0].startsWith('CREATE (:'));
  assert.ok(lines.find((l) => l.includes('WORKS_AT')).includes("from: date('2014-03-01')"));
  assert.ok(lines.find((l) => l.includes('WORKS_AT')).includes('CREATE (a)-[:WORKS_AT'));
});

test('nodes come before edges', () => {
  const lines = emitGql(source());
  assert.equal(lines.filter((l) => l.startsWith('INSERT')).length, 3);
  assert.ok(lines.slice(3).every((l) => l.startsWith('MATCH')));
});

test('timeline turns a dated edge into a fact', () => {
  const s = source();
  s.edges.push({ type: 'FOUNDED', source: 'richard-hendricks', target: 'pied-piper', props: { date: '2014-04-01' } });
  assert.deepEqual(emitTimeline(s), [
    { subject: 'richard-hendricks', relation: 'WORKS_AT', object: 'pied-piper', from: '2014-03', to: '2014-03-15' },
    { subject: 'richard-hendricks', relation: 'FOUNDED', object: 'pied-piper', from: '2014-04-01', to: '2014-04-01' },
  ]);
});

test('undated edges are not in the timeline', () => {
  assert.ok(!emitTimeline(source()).some((f) => f.relation === 'KNOWS'));
});

test('graph keeps duplicate edges in input-independent order', () => {
  const s = source();
  s.edges.push({ type: 'WORKS_AT', source: 'richard-hendricks', target: 'pied-piper', props: { role: 'CTO', from: '2015-01-02' } });
  const r = source();
  r.edges.push(s.edges[2]);
  r.nodes.reverse();
  r.edges.reverse();
  assert.deepEqual(emitGraph(r), emitGraph(s));
});

test('graph output is sorted and stable', () => {
  const a = emitGraph(source());
  assert.deepEqual(a, emitGraph(source()));
  const shuffled = source();
  shuffled.nodes.reverse();
  shuffled.edges.reverse();
  assert.deepEqual(emitGraph(shuffled), a);
  assert.deepEqual(a.nodes.map((n) => n.key), ['erlich-bachman', 'pied-piper', 'richard-hendricks']);
  assert.deepEqual(a.nodes[0].props, {});
  assert.equal(a.edges[0].type, 'KNOWS');
});

test('timeline keeps a fact with only a to date, without from', () => {
  const s = source();
  s.edges = [{ type: 'WORKS_AT', source: 'richard-hendricks', target: 'pied-piper', props: { to: '2014-03' } }];
  assert.deepEqual(emitTimeline(s), [
    { subject: 'richard-hendricks', relation: 'WORKS_AT', object: 'pied-piper', to: '2014-03' },
  ]);
});

test('timeline sorts by from (month by first day), a to-only fact by its to, then subject', () => {
  const s = source();
  const e = (subject, props) => ({ type: 'WORKS_AT', source: subject, target: 'pied-piper', props });
  s.edges = [
    e('zed', { from: '2014-05-01' }),
    e('amy', { from: '2014-05-01' }),
    e('zed', { from: '2014-04-15' }),
    e('zed', { from: '2014-04' }),
    e('qux', { to: '2014-03' }),
  ];
  assert.deepEqual(
    emitTimeline(s).map((f) => `${f.subject} ${f.from ?? '-'} ${f.to ?? '-'}`),
    ['qux - 2014-03', 'zed 2014-04 -', 'zed 2014-04-15 -', 'amy 2014-05-01 -', 'zed 2014-05-01 -'],
  );
});

test('cypher node statement is exact', () => {
  assert.equal(
    emitCypher(source()).find((l) => l.includes("'richard-hendricks', name")),
    "CREATE (:Person {key: 'richard-hendricks', name: 'Richard Hendricks', description: 'A founder.'})",
  );
});

test('cypher edge statement is exact', () => {
  assert.equal(
    emitCypher(source()).find((l) => l.includes('WORKS_AT')),
    "MATCH (a:Person {key: 'richard-hendricks'}), (b:Company {key: 'pied-piper'}) CREATE (a)-[:WORKS_AT {role: 'CEO', from: date('2014-03-01'), from_precision: 'month', to: date('2014-03-15')}]->(b)",
  );
});

test('tabs are escaped', () => {
  const s = source();
  s.nodes[0].description = 'a\tb';
  const line = emitGql(s).find((l) => l.includes("'richard-hendricks', name"));
  assert.ok(line.includes(String.raw`'a\tb'`), line);
});

test('to_precision follows a month-only to', () => {
  const s = source();
  s.edges[0].props = { role: 'CEO', from: '2014-03-01', to: '2014-05' };
  const line = emitGql(s).find((l) => l.includes('WORKS_AT'));
  assert.ok(line.includes("from: DATE '2014-03-01', to: DATE '2014-05-01', to_precision: 'month'}"), line);
  assert.ok(!line.includes('from_precision'));
});

test('an edge with empty props has no property map', () => {
  const s = source();
  s.edges[1].props = {};
  assert.ok(
    emitGql(s).includes(
      "MATCH (a:Person {key: 'richard-hendricks'}), (b:Person {key: 'erlich-bachman'}) INSERT (a)-[:KNOWS]->(b)",
    ),
  );
});

test('graph props are written with sorted keys', () => {
  const s = source();
  s.nodes[1].props = { founded: '2014-04', title: 'x', born_as: 'y' };
  s.edges[0].props = { to: '2014-03-15', role: 'CEO', from: '2014-03' };
  const a = emitGraph(s);
  const t = source();
  t.nodes[1].props = { born_as: 'y', title: 'x', founded: '2014-04' };
  t.edges[0].props = { from: '2014-03', role: 'CEO', to: '2014-03-15' };
  assert.equal(JSON.stringify(emitGraph(t)), JSON.stringify(a));
  assert.deepEqual(Object.keys(a.nodes.find((n) => n.key === 'pied-piper').props), ['born_as', 'founded', 'title']);
});

test('an edge end that is not a node throws a clear error', () => {
  const s = source();
  s.edges.push({ type: 'KNOWS', source: 'richard-hendricks', target: 'ghost' });
  assert.throws(() => emitGql(s), /ghost/);
  assert.throws(() => emitCypher(s), /ghost/);
});
