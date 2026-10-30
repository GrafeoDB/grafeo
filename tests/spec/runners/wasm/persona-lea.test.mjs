/**
 * Persona: Lea builds a browser app on the WASM build (the edge profile).
 *
 * A small graph of notes with Float32Array embeddings: similarity in a
 * query (the edge build has no vector index), a threshold, deleting a note
 * with its links, saving the graph as a snapshot (signed, for local storage)
 * and loading it back, a transaction, and the limits of the edge build, each
 * with an error that says what is missing. Every error carries a GQLSTATUS.
 */

import { describe, it, expect } from 'vitest'
import { readFileSync, existsSync } from 'fs'
import { join, resolve } from 'path'

const pkg = resolve(import.meta.dirname, '..', '..', '..', '..', 'crates', 'bindings', 'wasm', 'pkg')
const available = existsSync(join(pkg, 'grafeo_wasm.js')) && existsSync(join(pkg, 'grafeo_wasm_bg.wasm'))

let Database
// The checks of what the edge build leaves out run on that build only; the
// full package (built into the same pkg/) has those features.
let edge = false
if (available) {
  const mod = await import('../../../../crates/bindings/wasm/pkg/grafeo_wasm.js')
  mod.initSync({ module: readFileSync(join(pkg, 'grafeo_wasm_bg.wasm')) })
  Database = mod.Database
  edge = !new Database().info().features.includes('cypher')
}

function notes() {
  const db = new Database()
  db.executeWithParams(
    "INSERT (:Note {title: 'Canals', emb: $a}), (:Note {title: 'Bikes', emb: $b}), (:Note {title: 'Graphs', emb: $c})",
    { a: new Float32Array([0, 1, 0]), b: new Float32Array([0, 0, 1]), c: new Float32Array([1, 0, 0]) },
  )
  db.execute("MATCH (a:Note {title: 'Canals'}), (b:Note {title: 'Bikes'}) INSERT (a)-[:SEE_ALSO]->(b)")
  return db
}

function thrown(call) {
  try {
    call()
  } catch (error) {
    return error
  }
  throw new Error('expected the call to throw')
}

const titles = (db) => db.execute('MATCH (n:Note) RETURN n.title AS t ORDER BY t').map((row) => row.t)

describe.skipIf(!available)('persona Lea (WASM, edge build)', () => {
  it('finds the nearest note by similarity in a query', () => {
    const db = notes()
    const [best] = db.executeWithParams(
      'MATCH (n:Note) RETURN n.title AS t, cosine_similarity(n.emb, $q) AS s ORDER BY s DESC LIMIT 1',
      { q: new Float32Array([0.9, 0.1, 0]) },
    )
    expect(best.t).toBe('Graphs')
    expect(best.s).toBeGreaterThan(0.99)
    const close = db.executeWithParams(
      'MATCH (n:Note) WHERE cosine_similarity(n.emb, $q) > 0.5 RETURN n.title AS t ORDER BY t',
      { q: new Float32Array([0, 0.8, 0.6]) },
    )
    expect(close.map((row) => row.t)).toEqual(['Bikes', 'Canals'])
  })

  it('reads an embedding back as a Float32Array', () => {
    const [row] = notes().execute("MATCH (n:Note {title: 'Canals'}) RETURN n.emb AS e")
    expect(row.e).toBeInstanceOf(Float32Array)
    expect([...row.e]).toEqual([0, 1, 0])
  })

  it('deletes a note with its links', () => {
    const db = notes()
    const error = thrown(() => db.execute("MATCH (n:Note {title: 'Canals'}) DELETE n"))
    expect(error.message).toContain('Use DETACH DELETE')
    expect(titles(db)).toEqual(['Bikes', 'Canals', 'Graphs'])
    const [{ id }] = db.execute("MATCH (n:Note {title: 'Canals'}) RETURN id(n) AS id")
    expect(db.detachDeleteNode(id)).toBe(true)
    expect(db.detachDeleteNode(id)).toBe(false)
    expect(db.detachDeleteNode(9999)).toBe(false)
    expect(titles(db)).toEqual(['Bikes', 'Graphs'])
    expect([db.nodeCount(), db.edgeCount()]).toEqual([2, 0])
  })

  it('saves the graph as a snapshot and loads it back', () => {
    const db = notes()
    const restored = Database.importSnapshot(db.exportSnapshot())
    expect(titles(restored)).toEqual(['Bikes', 'Canals', 'Graphs'])
    expect(restored.edgeCount()).toBe(1)
    const [row] = restored.execute("MATCH (n:Note {title: 'Graphs'}) RETURN n.emb AS e")
    expect([...row.e]).toEqual([1, 0, 0])
  })

  it('signs a snapshot and refuses one with another key or a changed byte', () => {
    const db = notes()
    const key = new Uint8Array(32).fill(7)
    const signed = db.exportSnapshotSigned(key)
    expect(titles(Database.importSnapshotSigned(signed, key))).toEqual(['Bikes', 'Canals', 'Graphs'])
    const tampered = signed.slice()
    tampered[tampered.length - 40] ^= 1
    for (const [bytes, withKey] of [
      [signed, new Uint8Array(32).fill(8)],
      [tampered, key],
    ]) {
      const error = thrown(() => Database.importSnapshotSigned(bytes, withKey))
      expect(error.message).toContain('HMAC verification failed')
    }
    expect(thrown(() => Database.importSnapshot(signed)).message).toContain('importSnapshotSigned')
  })

  it('rolls back a transaction', () => {
    const db = notes()
    db.beginTransaction()
    db.execute("INSERT (:Note {title: 'Draft'})")
    expect(db.isTransactionActive()).toBe(true)
    db.rollbackTransaction()
    expect(db.isTransactionActive()).toBe(false)
    expect(titles(db)).toEqual(['Bikes', 'Canals', 'Graphs'])
  })

  it('holds a large text and many nodes', () => {
    const db = new Database()
    db.executeWithParams('INSERT (:Blob {s: $s})', { s: 'x'.repeat(5_000_000) })
    for (let batch = 0; batch < 10; batch++) {
      db.executeWithParams('UNWIND range(1, 5000) AS i INSERT (:Item {i: i, batch: $b})', { b: batch })
    }
    expect(db.nodeCount()).toBe(50_001)
    const [{ n }] = db.execute('MATCH (b:Blob) RETURN size(b.s) AS n')
    expect(n).toBe(5_000_000)
    expect(db.memoryUsage().total_bytes).toBeGreaterThan(5_000_000)
  })

  it.skipIf(!edge).each([
    ['a vector index', () => new Database().execute('CREATE VECTOR INDEX idx ON :Note(emb) DIMENSION 3'), 'vector indexes'],
    ['a procedure', () => new Database().execute("CALL grafeo.search.vector('Note', 'emb', [1.0, 0.0, 0.0], 1)"), 'procedures'],
  ])('is told the edge build has no %s', (_what, call, words) => {
    const error = thrown(call)
    expect(error.gqlStatus).toBe('42000')
    expect(error.message).toContain(words)
  })

  it.skipIf(!edge)('is told the edge build has GQL only', () => {
    const db = new Database()
    expect(db.info().features).toEqual(['gql'])
    const error = thrown(() => db.executeWithLanguage('MATCH (n) RETURN count(n) AS c', 'cypher'))
    expect(error.message).toContain("'cypher'")
  })

  it.each([
    ['a commit with no transaction', (db) => db.commitTransaction(), 'No active transaction'],
    ['a closed database', (db) => { db.close(); db.execute('RETURN 1 AS x') }, 'Database is closed'],
    ['a snapshot signed with another key', (db) => Database.importSnapshotSigned(db.exportSnapshotSigned(new Uint8Array(32).fill(1)), new Uint8Array(32).fill(2)), 'HMAC'],
  ])('gets a GQLSTATUS for %s', (_what, call, words) => {
    const error = thrown(() => call(notes()))
    expect(error.message).toContain(words)
    expect(error.gqlStatus).toMatch(/^[0-9A-Z]{5}$/)
  })
})
