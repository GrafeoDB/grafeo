/**
 * schema() on the WASM build gives each label and edge type its properties
 * and value types, and each edge type the label pairs it connects, as plain
 * objects and arrays (no Map), with snake_case keys and `mode: 'lpg'`.
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

const named = (entries, name) => entries.find((entry) => entry.name === name)

describe.skipIf(!available)('schema summary (WASM)', () => {
  it('gives the properties of a label and the label pairs of an edge type', () => {
    const db = new Database()
    db.execute(
      "INSERT (alix:Person:Employee {name: 'Alix', age: 30}), (gus:Person {name: 'Gus', age: '31'}), " +
        "(acme:Company {name: 'Acme'}), ({note: 'loose'}), " +
        '(alix)-[:WORKS_AT {since: 2020}]->(acme), (gus)-[:WORKS_AT]->(acme)',
    )
    db.execute("MATCH (n {note: 'loose'}), (g:Person {name: 'Gus'}) INSERT (g)-[:LINK]->(n)")
    const schema = db.schema()
    expect(schema).not.toBeInstanceOf(Map)
    const lpg = schema
    expect(lpg.mode).toBe('lpg')
    expect(named(lpg.labels, 'Person')).toEqual({
      name: 'Person',
      count: 2,
      properties: [
        { name: 'age', types: ['INT64', 'STRING'] },
        { name: 'name', types: ['STRING'] },
      ],
    })
    expect(named(lpg.edge_types, 'WORKS_AT')).toEqual({
      name: 'WORKS_AT',
      count: 2,
      properties: [{ name: 'since', types: ['INT64'] }],
      endpoints: [
        { source: 'Employee', target: 'Company', count: 1 },
        { source: 'Person', target: 'Company', count: 2 },
      ],
    })
    expect(named(lpg.edge_types, 'LINK').endpoints).toEqual([])
    expect(JSON.parse(JSON.stringify(schema))).toEqual(schema)
  })
})
