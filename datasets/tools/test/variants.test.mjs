import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { test } from 'node:test';
import { emitLpgImport, emitRdfImport } from '../lib/emit-import.mjs';
import { emitOwl, emitRdfs, vocabulary } from '../lib/emit-schema.mjs';
import { dataTriples, emitTurtle } from '../lib/emit-turtle.mjs';

const source = () => ({
  name: 'fixture',
  nodes: [
    { key: 'richard-hendricks', label: 'Person', name: 'Richard Hendricks', description: 'A founder.' },
    { key: 'pied-piper', label: 'Company', name: 'Pied Piper', description: 'A startup.', props: { founded: '2014-04' } },
    { key: 'hooli', label: 'Company', name: 'Hooli', description: 'A giant.' },
    { key: 'middle-out', label: 'Technology', name: 'Middle-out', description: 'An algorithm.', props: { weissman_score: 5.2 } },
    { key: 'peter-gregory', label: 'Person', name: 'Peter Gregory', description: 'An investor.' },
  ],
  edges: [
    { type: 'WORKS_AT', source: 'richard-hendricks', target: 'pied-piper', props: { role: 'CEO', from: '2014-03' } },
    { type: 'INVESTED_IN', source: 'peter-gregory', target: 'pied-piper', props: { amount_usd: 200000 } },
    { type: 'INVESTED_IN', source: 'hooli', target: 'pied-piper', props: { amount_usd: 1 } },
    { type: 'KNOWS', source: 'richard-hendricks', target: 'peter-gregory' },
  ],
});

/** The Turtle blocks of one subject in `text`, up to the blank line after it. */
const blockOf = (text, subject) => {
  const start = text.indexOf(`\n${subject}\n`);
  assert.notEqual(start, -1, `no block for ${subject}`);
  return text.slice(start + 1, text.indexOf('\n\n', start + 1));
};

test('the variants hold the plain Turtle data unchanged', () => {
  const body = emitTurtle(source()).split('\n\n').slice(1).join('\n\n');
  for (const text of [emitRdfs(source()), emitOwl(source())]) assert.ok(text.endsWith(body));
});

test('every predicate and class the data uses is declared, in both variants', () => {
  const { classes, predicates } = vocabulary(source());
  const rdfs = emitRdfs(source());
  const owl = emitOwl(source());
  for (const c of classes.filter((c) => c.startsWith('sv:'))) {
    assert.match(blockOf(rdfs, c), /a rdfs:Class/);
    assert.match(blockOf(owl, c), /a owl:Class/);
  }
  for (const p of predicates.keys()) {
    if (p.startsWith('sv:')) assert.match(blockOf(rdfs, p), /a rdf:Property/);
    assert.match(blockOf(owl, p), /a owl:(Object|Datatype)Property/);
  }
});

test('RDFS states a single domain, OWL a union of several', () => {
  // sv:investor links an investment to a Person or an Organization, so RDFS states no range (it would mean
  // both) and OWL the union.
  const rdfsInvestor = blockOf(emitRdfs(source()), 'sv:investor');
  assert.match(rdfsInvestor, /rdfs:domain sv:Investment/);
  assert.doesNotMatch(rdfsInvestor, /rdfs:range/);
  assert.match(
    blockOf(emitOwl(source()), 'sv:investor'),
    /rdfs:range \[ a owl:Class ; owl:unionOf \( schema:Organization schema:Person \) \]/,
  );
  assert.match(blockOf(emitOwl(source()), 'sv:weissmanScore'), /rdfs:range xsd:double/);
  assert.match(blockOf(emitRdfs(source()), 'sv:worksAt'), /rdfs:subPropertyOf schema:worksFor/);
});

test('the LPG payload links edges by index and tags dates', () => {
  const { nodes, edges } = emitLpgImport(source());
  assert.equal(nodes.length, 5);
  const works = edges.find((e) => e.type === 'WORKS_AT');
  assert.equal(nodes[works.source].properties.key, 'richard-hendricks');
  assert.equal(nodes[works.target].properties.key, 'pied-piper');
  assert.deepEqual(works.properties, { role: 'CEO', from: { $date: '2014-03-01' }, from_precision: 'month' });
  const pp = nodes.find((n) => n.properties.key === 'pied-piper');
  assert.deepEqual(pp.labels, ['Company']);
  assert.deepEqual(pp.properties.founded, { $date: '2014-04-01' });
  assert.equal(nodes.find((n) => n.properties.key === 'middle-out').properties.weissman_score, 5.2);
});

test('the LPG payload refuses values JSON would change', () => {
  const whole = source();
  whole.nodes[3].props.weissman_score = 5;
  assert.throws(() => emitLpgImport(whole), /whole/);
  const big = source();
  big.edges[1].props.amount_usd = 2 ** 53;
  assert.throws(() => emitLpgImport(big), /2\^53/);
});

test('the RDF payload is the Turtle data with full IRIs and datatypes', () => {
  const { triples } = emitRdfImport(source());
  assert.equal(triples.length, dataTriples(source()).length);
  const sv = 'https://docs.grafeo.dev/datasets/silicon-valley/vocab#';
  const of = (predicate) => triples.filter((t) => t.predicate === `${sv}${predicate}`).map((t) => t.object);
  assert.deepEqual(of('weissmanScore'), [{ value: '5.2e0', datatype: 'http://www.w3.org/2001/XMLSchema#double' }]);
  assert.deepEqual(of('role'), [{ value: 'CEO' }]);
  assert.ok(of('knows').every((o) => typeof o === 'string' && o.startsWith('https://docs.grafeo.dev/')));
});

test('the committed payloads have as many nodes, edges and triples as the dataset', () => {
  const derived = new URL('../../silicon-valley/derived/', import.meta.url);
  const graph = JSON.parse(readFileSync(new URL('graph.json', derived), 'utf8'));
  const lpg = JSON.parse(readFileSync(new URL('silicon-valley.lpg.json', derived), 'utf8'));
  assert.equal(lpg.nodes.length, graph.nodes.length);
  assert.equal(lpg.edges.length, graph.edges.length);
  for (const [i, e] of lpg.edges.entries()) {
    assert.equal(lpg.nodes[e.source].properties.key, graph.edges[i].source);
    assert.equal(lpg.nodes[e.target].properties.key, graph.edges[i].target);
  }
});
