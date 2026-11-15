import assert from 'node:assert/strict';
import { test } from 'node:test';
import { EDGE_TYPES } from '../lib/model.mjs';
import { emitTurtle } from '../lib/emit-turtle.mjs';

const source = () => ({
  name: 'fixture',
  nodes: [
    { key: 'richard-hendricks', label: 'Person', name: 'Richard Hendricks', description: 'A founder.' },
    { key: 'pied-piper', label: 'Company', name: 'Pied Piper', description: 'A startup.', props: { founded: '2014-04' } },
    { key: 'erlich-bachman', label: 'Person', name: 'Erlich', description: 'Erlich\'s "palapa" \\ test' },
    { key: 'middle-out', label: 'Technology', name: 'Middle-out', description: 'Compression.', props: { weissman_score: 5.2, tier: 3, flag: true } },
  ],
  edges: [
    { type: 'WORKS_AT', source: 'richard-hendricks', target: 'pied-piper', props: { role: 'CEO', from: '2014-03', to: '2014-03-15' } },
    { type: 'KNOWS', source: 'richard-hendricks', target: 'erlich-bachman' },
    { type: 'WORKS_AT', source: 'richard-hendricks', target: 'pied-piper', props: { role: 'CTO', from: '2015-01-02' } },
  ],
});
const lines = (s = source()) => emitTurtle(s).split('\n');

test('prefixes are declared', () => {
  const l = lines();
  assert.deepEqual(l.slice(0, 4), [
    '@prefix sv: <https://grafeo.dev/datasets/silicon-valley/vocab#> .',
    '@prefix svr: <https://grafeo.dev/datasets/silicon-valley/id/> .',
    '@prefix schema: <https://schema.org/> .',
    '@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .',
  ]);
});

test('a person is a schema:Person with name and description', () => {
  const text = emitTurtle(source());
  assert.ok(
    text.includes('svr:richard-hendricks\n    a schema:Person ;\n    schema:description "A founder." ;\n    schema:name "Richard Hendricks" .\n'),
    text,
  );
});

test('WORKS_AT states schema:worksFor and an sv:Employment', () => {
  const l = lines();
  assert.ok(l.includes('svr:richard-hendricks sv:worksAt svr:pied-piper .'));
  assert.ok(l.includes('svr:richard-hendricks schema:worksFor svr:pied-piper .'));
  const text = emitTurtle(source());
  assert.ok(
    text.includes(
      'svr:works-at-richard-hendricks-pied-piper-1\n    a sv:Employment ;\n    sv:person svr:richard-hendricks ;\n    sv:company svr:pied-piper ;\n    sv:from "2014-03-01"^^xsd:date ;\n    sv:fromPrecision "month" ;\n    sv:role "CEO" ;\n    sv:to "2014-03-15"^^xsd:date .\n',
    ),
    text,
  );
  assert.ok(text.includes('svr:works-at-richard-hendricks-pied-piper-2\n'));
});

test('a month-only date is the first of the month plus a precision triple', () => {
  const l = lines();
  assert.ok(l.includes('    schema:foundingDate "2014-04-01"^^xsd:date ;'));
  assert.ok(l.includes('    sv:foundedPrecision "month" .'));
  assert.ok(l.includes('    sv:from "2015-01-02"^^xsd:date ;'));
});

test('a day date has no precision triple, and no date is xsd:gYearMonth', () => {
  const text = emitTurtle(source());
  assert.ok(!text.includes('gYearMonth'));
  assert.ok(!text.includes('sv:toPrecision'));
  const second = text.slice(text.indexOf('svr:works-at-richard-hendricks-pied-piper-2\n'));
  assert.ok(!second.split('\n\n')[0].includes('Precision'), second);
});

test('edge without properties is a plain sv: predicate', () => {
  const text = emitTurtle(source());
  assert.ok(text.includes('svr:richard-hendricks sv:knows svr:erlich-bachman .\n'));
  assert.ok(!text.includes('knows-richard'));
});

test('numbers and booleans are bare, facts are camelCase', () => {
  const l = lines();
  assert.ok(l.includes('    sv:weissmanScore 5.2e0 .'));
  assert.ok(l.includes('    sv:tier 3 ;'));
  assert.ok(l.includes('    sv:flag true ;'));
  assert.ok(l.includes('    a sv:Technology ;'));
});

test('string escapes', () => {
  const s = source();
  s.nodes[0].description = 'a"b\\c\nd\re\tf';
  assert.ok(emitTurtle(s).includes(String.raw`schema:description "a\"b\\c\nd\re\tf"`));
  assert.ok(emitTurtle(source()).includes(String.raw`"Erlich's \"palapa\" \\ test"`));
});

test('output is stable', () => {
  const a = emitTurtle(source());
  const s = source();
  s.nodes.reverse();
  s.edges.reverse();
  assert.equal(emitTurtle(s), a);
  assert.ok(a.endsWith('.\n') && !a.endsWith('\n\n') && !a.includes('\r'));
});

test('an edge with only a to date', () => {
  const s = source();
  s.edges = [{ type: 'LIVES_IN', source: 'richard-hendricks', target: 'pied-piper', props: { to: '2016-05' } }];
  const text = emitTurtle(s);
  assert.ok(text.includes('    sv:to "2016-05-01"^^xsd:date ;\n    sv:toPrecision "month" .'), text);
  assert.ok(!text.includes('sv:from'));
});

test('a month-only date on an edge is an xsd:date with sv:datePrecision', () => {
  const s = source();
  s.edges = [{ type: 'FOUNDED', source: 'richard-hendricks', target: 'pied-piper', props: { date: '2014-04' } }];
  assert.ok(emitTurtle(s).includes('    sv:date "2014-04-01"^^xsd:date ;\n    sv:datePrecision "month" .'));
});

test('a float property is a double even when it is a whole number', () => {
  const s = source();
  s.nodes[3].props = { weissman_score: 3, tier: 3 };
  const l = lines(s);
  assert.ok(l.includes('    sv:weissmanScore 3e0 .'), l.join('\n'));
  assert.ok(l.includes('    sv:tier 3 ;'));
});

const QUALIFIED_CASES = [
  ['WORKS_AT', 'sv:Employment', 'sv:person', 'sv:company'],
  ['BASED_IN', 'sv:Base', 'sv:company', 'sv:place'],
  ['INVESTED_IN', 'sv:Investment', 'sv:investor', 'sv:company'],
  ['ATTENDED', 'sv:Attendance', 'sv:attendee', 'sv:event'],
  ['FOUNDED', 'sv:Founding', 'sv:founder', 'sv:founded'],
  ['LIVES_IN', 'sv:Residence', 'sv:person', 'sv:city'],
  ['OWNS', 'sv:Ownership', 'sv:owner', 'sv:owned'],
];

for (const [type, cls, fromRole, toRole] of QUALIFIED_CASES) {
  test(`${type} becomes ${cls}`, () => {
    const prop = EDGE_TYPES[type].props.includes('date') ? { date: '2014-04-01' } : { from: '2014-04-01' };
    const s = { name: 'x', nodes: [], edges: [{ type, source: 'a-1', target: 'b-2', props: prop }] };
    const text = emitTurtle(s);
    const iri = `svr:${type.toLowerCase().replace(/_/g, '-')}-a-1-b-2-1`;
    assert.ok(text.includes(`${iri}\n    a ${cls} ;\n    ${fromRole} svr:a-1 ;\n    ${toRole} svr:b-2 ;\n`), text);
  });
}

test('every edge type with properties has a qualified resource', () => {
  const covered = QUALIFIED_CASES.map(([t]) => t);
  const withProps = Object.entries(EDGE_TYPES).filter(([, d]) => d.props.length > 0).map(([t]) => t);
  assert.deepEqual([...covered].sort(), [...withProps].sort());
});

test('an edge type with properties but no qualified class throws', () => {
  const s = { name: 'x', nodes: [], edges: [{ type: 'WORKS_AT', source: 'a', target: 'b', props: { role: 'x' } }] };
  assert.doesNotThrow(() => emitTurtle(s));
  s.edges[0].type = 'NOPE';
  assert.throws(() => emitTurtle(s), /NOPE/);
});
