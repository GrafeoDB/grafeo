/**
 * RDF terms in the WASM bindings.
 *
 * Results give each RDF term as an RDF/JS term object with the data model's
 * `equals(other)` (not enumerable); a parameter takes any object of that
 * shape, the terms of other libraries too, and keeps a language tag in
 * lowercase; an object of another termType, or with the keys of a SPARQL JSON
 * term, stays a map. The SPARQL cases run on a build with SPARQL (the full
 * profile) and are skipped on the browser build.
 */

import { describe, it, expect } from 'vitest'
import { readFileSync, existsSync } from 'fs'
import { join, resolve } from 'path'

const pkg = resolve(import.meta.dirname, '..', '..', '..', '..', 'crates', 'bindings', 'wasm', 'pkg')
const available = existsSync(join(pkg, 'grafeo_wasm.js')) && existsSync(join(pkg, 'grafeo_wasm_bg.wasm'))

let Database
if (available) {
  const mod = await import('../../../../crates/bindings/wasm/pkg/grafeo_wasm.js')
  mod.initSync({ module: readFileSync(join(pkg, 'grafeo_wasm_bg.wasm')) })
  Database = mod.Database
}

const XSD = 'http://www.w3.org/2001/XMLSchema#'
const RDF_LANG_STRING = 'http://www.w3.org/1999/02/22-rdf-syntax-ns#langString'
const namedNode = (value) => ({ termType: 'NamedNode', value })
const literal = (value, language, datatype) => ({ termType: 'Literal', value, language, datatype: namedNode(datatype) })

const roundTrip = (db, term) => db.executeWithParams('RETURN $t AS t', { t: term })[0].t

const hasSparql = (() => {
  if (!available) return false
  const db = new Database()
  try {
    db.executeSparql('ASK { ?s ?p ?o }')
    return true
  } catch {
    return false
  } finally {
    db.free()
  }
})()

describe.skipIf(!available)('RDF terms (WASM)', () => {
  it('a term round-trips as a parameter, with equals', () => {
    const db = new Database()
    for (const term of [
      namedNode('http://ex.org/Kraków'),
      { termType: 'BlankNode', value: 'b0' },
      literal('Pomme', 'fr', RDF_LANG_STRING),
      literal('2026-01-15', '', `${XSD}date`),
      literal('Apple', '', `${XSD}string`),
    ]) {
      const back = roundTrip(db, term)
      expect(back).toEqual(term)
      expect(typeof back.equals).toBe('function')
      expect(Object.keys(back)).not.toContain('equals')
      expect(JSON.parse(JSON.stringify(back))).toEqual(term)
      expect(back.equals(term)).toBe(true)
      expect(back.equals(back)).toBe(true)
      for (const other of [null, undefined, 'x', 42, {}]) expect(back.equals(other)).toBe(false)
      expect(back.equals()).toBe(false)
    }
    db.free()
  })

  it('equals compares kind, value, language without case, and datatype', () => {
    const db = new Database()
    const label = roundTrip(db, literal('Pomme', 'fr', RDF_LANG_STRING))
    expect(label.equals(literal('Pomme', 'FR', RDF_LANG_STRING))).toBe(true)
    expect(label.equals(literal('Pomme', 'nl', RDF_LANG_STRING))).toBe(false)
    expect(label.equals(literal('Apple', 'fr', RDF_LANG_STRING))).toBe(false)
    const price = roundTrip(db, literal('1.50', '', `${XSD}decimal`))
    expect(price.equals(literal('1.50', '', `${XSD}string`))).toBe(false)
    expect(price.datatype.equals(namedNode(`${XSD}decimal`))).toBe(true)
    const iri = roundTrip(db, namedNode('http://ex.org/Paris'))
    expect(iri.equals({ termType: 'BlankNode', value: 'http://ex.org/Paris' })).toBe(false)
    db.free()
  })

  it('a language tag is kept in lowercase', () => {
    const db = new Database()
    const back = roundTrip(db, literal('Colour', 'en-GB', RDF_LANG_STRING))
    expect(back.language).toBe('en-gb')
    db.free()
  })

  it('a parameter takes the terms of other RDF/JS libraries', () => {
    const db = new Database()
    class N3Term {
      constructor(id) { this.id = id }
      get termType() { return 'NamedNode' }
      get value() { return this.id }
    }
    expect(roundTrip(db, new N3Term('http://ex.org/Berlin'))).toEqual(namedNode('http://ex.org/Berlin'))
    const extra = { termType: 'NamedNode', value: 'http://ex.org/a', extra: 1, equals: () => true }
    expect(roundTrip(db, extra)).toEqual(namedNode('http://ex.org/a'))
    expect(roundTrip(db, { termType: 'Literal', value: 'fresh' })).toEqual(literal('fresh', '', `${XSD}string`))
    db.free()
  })

  it('an object of another termType, or with SPARQL JSON keys, is a map', () => {
    const db = new Database()
    for (const map of [
      { termType: 'Quad', value: 'x' },
      { termType: 'NamedNode' },
      { type: 'uri', value: 'http://ex.org/Paris' },
      { type: 'literal', value: 'Pomme', 'xml:lang': 'fr' },
    ]) {
      const back = roundTrip(db, map)
      expect(back).toEqual(map)
      expect(back.equals).toBeUndefined()
    }
    db.free()
  })

  it.skipIf(!hasSparql)('a SPARQL result holds terms with equals', () => {
    const db = new Database()
    db.executeSparql(`INSERT DATA { <http://ex.org/apple> <http://ex.org/label> "Pomme"@fr }`)
    const [row] = db.executeSparql('SELECT ?s ?o WHERE { ?s <http://ex.org/label> ?o }')
    expect(row.s).toEqual(namedNode('http://ex.org/apple'))
    expect(row.s.equals(namedNode('http://ex.org/apple'))).toBe(true)
    expect(row.o.equals(literal('Pomme', 'FR', RDF_LANG_STRING))).toBe(true)
    const raw = db.executeRawWithLanguage('SELECT ?o WHERE { ?s ?p ?o }', 'sparql')
    expect(raw.rows[0][0].equals(row.o)).toBe(true)
    db.free()
  })
})
