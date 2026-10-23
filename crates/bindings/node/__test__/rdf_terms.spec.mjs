// SPARQL results hold exact RDF terms as RDF/JS objects, and a result says
// what it holds (#464).

import { describe, it, expect } from 'vitest'
import { GrafeoDB } from '../grafeo.js'

const XSD = 'http://www.w3.org/2001/XMLSchema#'
const RDF_LANG_STRING = 'http://www.w3.org/1999/02/22-rdf-syntax-ns#langString'

async function fruit() {
  const db = GrafeoDB.create()
  await db.executeSparql(`INSERT DATA {
    <http://ex.org/apple> <http://ex.org/label> "Pomme"@fr ;
      <http://ex.org/price> "1.50"^^<${XSD}decimal> ;
      <http://ex.org/note> "fresh" ;
      <http://ex.org/grower> _:farm .
  }`)
  return db
}

const namedNode = (value) => ({ termType: 'NamedNode', value })
const literal = (value, language, datatype) => ({ termType: 'Literal', value, language, datatype: namedNode(datatype) })

describe('RDF terms', () => {
  it('a SELECT holds every kind of term', async () => {
    const db = await fruit()
    const result = await db.executeSparql('SELECT ?p ?o WHERE { <http://ex.org/apple> ?p ?o }')
    expect(result.kind).toBe('rows')
    expect(result.boolean).toBeNull()
    const objects = Object.fromEntries(result.toArray().map((row) => [row.p.value, row.o]))
    for (const row of result.toArray()) expect(row.p).toEqual(namedNode(row.p.value))
    expect(objects['http://ex.org/label']).toEqual(literal('Pomme', 'fr', RDF_LANG_STRING))
    expect(objects['http://ex.org/price']).toEqual(literal('1.50', '', `${XSD}decimal`))
    expect(objects['http://ex.org/note']).toEqual(literal('fresh', '', `${XSD}string`))
    const grower = objects['http://ex.org/grower']
    expect(Object.keys(grower).sort()).toEqual(['termType', 'value'])
    expect(grower.termType).toBe('BlankNode')
    expect(grower.value).not.toMatch(/^_:/)
  })

  it('ASK is one boolean and CONSTRUCT a graph', async () => {
    const db = await fruit()
    const ask = await db.executeSparql('ASK { ?s <http://ex.org/label> "Pomme"@fr }')
    expect(ask.kind).toBe('boolean')
    expect(ask.boolean).toBe(true)
    expect((await db.executeSparql('ASK { ?s <http://ex.org/label> "Pomme"@en }')).boolean).toBe(false)
    const graph = await db.executeSparql('CONSTRUCT { ?s ?p ?o } WHERE { ?s ?p ?o }')
    expect(graph.kind).toBe('graph')
    expect(graph.columns).toEqual(['subject', 'predicate', 'object'])
    expect(graph.length).toBe(4)
  })

  it('a term round-trips as a parameter and as a property', async () => {
    const db = GrafeoDB.create()
    const terms = [
      namedNode('http://ex.org/Kraków'),
      { termType: 'BlankNode', value: 'b0' },
      literal('Pomme', 'fr', RDF_LANG_STRING),
      literal('2026-01-15', '', `${XSD}date`),
      literal('Apple', '', `${XSD}string`),
    ]
    for (const [n, term] of terms.entries()) {
      const back = await db.execute('RETURN $term AS term', { term })
      expect(back.scalar()).toEqual(term)
      await db.execute('INSERT (:Concept {n: $n, term: $term})', { n, term })
    }
    const stored = await db.execute('MATCH (c:Concept) RETURN c.term AS term ORDER BY c.n')
    expect(stored.toArray().map((row) => row.term)).toEqual(terms)
  })

  it('a term has the RDF/JS equals, not enumerable', async () => {
    const db = await fruit()
    const result = await db.executeSparql('SELECT ?p ?o WHERE { <http://ex.org/apple> ?p ?o }')
    const objects = Object.fromEntries(result.toArray().map((row) => [row.p.value, row.o]))
    const label = objects['http://ex.org/label']
    const price = objects['http://ex.org/price']
    const predicate = result.toArray()[0].p
    expect(typeof label.equals).toBe('function')
    expect(Object.keys(label).sort()).toEqual(['datatype', 'language', 'termType', 'value'])
    expect(JSON.parse(JSON.stringify(label))).toEqual(literal('Pomme', 'fr', RDF_LANG_STRING))
    expect(label.equals(label)).toBe(true)
    expect(label.equals(literal('Pomme', 'fr', RDF_LANG_STRING))).toBe(true)
    // Language tags compare without case.
    expect(label.equals(literal('Pomme', 'FR', RDF_LANG_STRING))).toBe(true)
    expect(label.equals(literal('Pomme', 'nl', RDF_LANG_STRING))).toBe(false)
    expect(label.equals(literal('Apple', 'fr', RDF_LANG_STRING))).toBe(false)
    expect(price.equals(literal('1.50', '', `${XSD}decimal`))).toBe(true)
    expect(price.equals(literal('1.50', '', `${XSD}string`))).toBe(false)
    expect(price.datatype.equals(namedNode(`${XSD}decimal`))).toBe(true)
    expect(predicate.equals(namedNode(predicate.value))).toBe(true)
    expect(predicate.equals({ termType: 'BlankNode', value: predicate.value })).toBe(false)
    for (const other of [null, undefined, 'http://ex.org/label', 42, {}]) {
      expect(label.equals(other)).toBe(false)
      expect(predicate.equals(other)).toBe(false)
    }
    expect(label.equals()).toBe(false)
  })

  it('a parameter takes the terms of other RDF/JS libraries', async () => {
    const db = GrafeoDB.create()
    // As N3.js makes them: termType and value are getters of a class.
    class N3Term {
      constructor(id) { this.id = id }
      get termType() { return 'NamedNode' }
      get value() { return this.id }
      equals(other) { return !!other && other.termType === this.termType && other.value === this.value }
    }
    const fromClass = await db.execute('RETURN $t AS t', { t: new N3Term('http://ex.org/Berlin') })
    expect(fromClass.scalar()).toEqual(namedNode('http://ex.org/Berlin'))
    // Extra fields (rdf-data-factory, @rdfjs/data-model) are ignored.
    const extra = { termType: 'NamedNode', value: 'http://ex.org/a', extra: 1, equals: () => true }
    expect((await db.execute('RETURN $t AS t', { t: extra })).scalar()).toEqual(namedNode('http://ex.org/a'))
    const tagged = { termType: 'Literal', value: 'Pomme', language: 'FR', datatype: { termType: 'NamedNode', value: RDF_LANG_STRING, equals() {} } }
    const back = (await db.execute('RETURN $t AS t', { t: tagged })).scalar()
    expect(back).toEqual(literal('Pomme', 'fr', RDF_LANG_STRING))
    expect(back.equals(tagged)).toBe(true)
    // A literal without a datatype is the simple literal.
    const simple = (await db.execute('RETURN $t AS t', { t: { termType: 'Literal', value: 'fresh' } })).scalar()
    expect(simple).toEqual(literal('fresh', '', `${XSD}string`))
  })

  it('an object of another termType, or with SPARQL JSON keys, is a map', async () => {
    const db = GrafeoDB.create()
    for (const map of [
      { termType: 'Quad', value: 'x' },
      { termType: 'NamedNode' },
      { type: 'uri', value: 'http://ex.org/Paris' },
      { type: 'literal', value: 'Pomme', 'xml:lang': 'fr' },
    ]) {
      const back = await db.execute('RETURN $m AS m', { m: map })
      expect(back.scalar()).toEqual(map)
      expect(back.scalar().equals).toBeUndefined()
    }
  })

  it('executeStream and change events give terms as execute does', async () => {
    const db = GrafeoDB.create()
    db.enableCdc()
    const term = literal('Pomme', 'fr', RDF_LANG_STRING)
    await db.execute('INSERT (:Concept {n: 1, term: $term, terms: [$term]})', { term })
    const rows = []
    const stream = await db.executeStream('MATCH (c:Concept) RETURN c.term AS term, c.terms AS terms')
    let row
    while ((row = await stream.next()) !== null) rows.push(row)
    expect(rows).toEqual([{ term, terms: [term] }])
    expect(rows[0].term.equals(term)).toBe(true)
    const [node] = (await db.execute('MATCH (c:Concept) RETURN id(c) AS id')).toArray()
    const history = await db.nodeHistory(node.id)
    const created = history.find((event) => event.after && event.after.term)
    expect(created.after.term).toEqual(term)
    expect(created.after.term.equals(term)).toBe(true)
    expect(created.after.terms[0].equals(term)).toBe(true)
  })

  it('executeStream gives every kind of term as execute does', async () => {
    const db = GrafeoDB.create()
    const terms = [
      namedNode('http://ex.org/Kraków'),
      { termType: 'BlankNode', value: 'b0' },
      literal('Pomme', 'fr', RDF_LANG_STRING),
      literal('025', '', `${XSD}integer`),
      literal('fresh', '', `${XSD}string`),
      literal('x', '', 'http://ex.org/unknown'),
    ]
    for (const [n, term] of terms.entries()) {
      await db.execute('INSERT (:Concept {n: $n, term: $term})', { n, term })
    }
    // A stream takes no ORDER BY: both sides are sorted by n here.
    const query = 'MATCH (c:Concept) RETURN c.n AS n, c.term AS term'
    const byN = (rows) => rows.sort((a, b) => a.n - b.n)
    const executed = byN((await db.execute(query)).toArray())
    const streamed = []
    const stream = await db.executeStream(query)
    let row
    while ((row = await stream.next()) !== null) streamed.push(row)
    byN(streamed)
    expect(streamed).toEqual(executed)
    expect(streamed.map((r) => r.term)).toEqual(terms)
    for (const [n, r] of streamed.entries()) expect(r.term.equals(terms[n])).toBe(true)
  })
})
