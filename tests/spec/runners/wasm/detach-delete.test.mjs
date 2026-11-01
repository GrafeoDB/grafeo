/** The public WASM detachDeleteNode export, including its transaction route. */

import { describe, it, expect } from 'vitest'
import { readFileSync } from 'fs'
import { join, resolve } from 'path'

const pkg = resolve(import.meta.dirname, '..', '..', '..', '..', 'crates', 'bindings', 'wasm', 'pkg')
const mod = await import('../../../../crates/bindings/wasm/pkg/grafeo_wasm.js')
mod.initSync({ module: readFileSync(join(pkg, 'grafeo_wasm_bg.wasm')) })
const { Database } = mod

const before = {
  nodes: ['A', 'B', 'C', 'D', 'V'],
  edges: [
    { source: 'A', kind: 'KNOWS', slot: 1, target: 'V' },
    { source: 'A', kind: 'KNOWS', slot: 2, target: 'V' },
    { source: 'V', kind: 'WORKS_WITH', slot: 3, target: 'B' },
    { source: 'V', kind: 'NOTE', slot: 4, target: 'V' },
    { source: 'C', kind: 'OTHER', slot: 5, target: 'D' },
  ],
}

const after = {
  nodes: ['A', 'B', 'C', 'D'],
  edges: [{ source: 'C', kind: 'OTHER', slot: 5, target: 'D' }],
}

function state(db) {
  return {
    nodes: db.execute('MATCH (n:Node) RETURN n.name AS name ORDER BY n.name').map((row) => row.name),
    edges: db.execute(
      'MATCH (a:Node)-[e]->(b:Node) RETURN a.name AS source, type(e) AS kind, e.slot AS slot, b.name AS target ORDER BY e.slot',
    ),
  }
}

function withFixture(check) {
  const db = new Database()
  try {
    db.execute(
      "INSERT (a:Node {name: 'A'}), (b:Node {name: 'B'}), (c:Node {name: 'C'}), " +
      "(d:Node {name: 'D'}), (v:Node {name: 'V'}), " +
      '(a)-[:KNOWS {slot: 1}]->(v), (a)-[:KNOWS {slot: 2}]->(v), ' +
      '(v)-[:WORKS_WITH {slot: 3}]->(b), (v)-[:NOTE {slot: 4}]->(v), ' +
      '(c)-[:OTHER {slot: 5}]->(d)',
    )
    const rows = db.execute("MATCH (v:Node {name: 'V'}) RETURN id(v) AS id")
    expect(rows).toHaveLength(1)
    const id = rows[0].id
    expect(Number.isSafeInteger(id)).toBe(true)
    expect(id).toBeGreaterThanOrEqual(0)
    expect(state(db)).toEqual(before)
    check(db, id)
  } finally {
    db.free()
  }
}

describe('detachDeleteNode (WASM JavaScript API)', () => {
  it('removes every incident edge once and retains unrelated nodes and edges', () => {
    withFixture((db, id) => {
      expect(() => db.execute("MATCH (v:Node {name: 'V'}) DELETE v")).toThrow()
      expect(state(db)).toEqual(before)

      expect(typeof db.detachDeleteNode).toBe('function')
      expect(db.detachDeleteNode(id)).toBe(true)
      expect(state(db)).toEqual(after)
      expect(db.detachDeleteNode(id)).toBe(false)
      expect(db.detachDeleteNode(Number.MAX_SAFE_INTEGER)).toBe(false)
      expect(state(db)).toEqual(after)
    })
  })

  it('uses the active transaction for pending reads, rollback and commit', () => {
    withFixture((db, id) => {
      db.beginTransaction()
      expect(db.detachDeleteNode(id)).toBe(true)
      expect(state(db)).toEqual(after)
      expect(db.detachDeleteNode(id)).toBe(false)
      db.rollbackTransaction()
      expect(state(db)).toEqual(before)

      db.beginTransaction()
      expect(db.detachDeleteNode(id)).toBe(true)
      expect(state(db)).toEqual(after)
      db.commitTransaction()
      expect(state(db)).toEqual(after)
      expect(db.detachDeleteNode(id)).toBe(false)
    })
  })

  it('rejects invalid numeric IDs without changing the graph', () => {
    withFixture((db) => {
      for (const id of [-1, 0.5, NaN, Infinity, -Infinity, Number.MAX_SAFE_INTEGER + 1]) {
        expect(() => db.detachDeleteNode(id), String(id)).toThrow(/not a node ID/)
        expect(state(db)).toEqual(before)
      }
    })
  })

  it('checks the closed state before accepting or validating an ID', () => {
    withFixture((db, id) => {
      db.close()
      expect(() => db.detachDeleteNode(id)).toThrow(/closed/i)
      expect(() => db.detachDeleteNode(0.5)).toThrow(/closed/i)
    })
  })
})
