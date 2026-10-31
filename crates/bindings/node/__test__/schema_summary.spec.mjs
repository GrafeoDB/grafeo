// schema() gives each label and edge type its properties and value types,
// and each edge type the label pairs it connects (serialized as the Rust
// summary: snake_case keys, tagged `mode: 'lpg'`). A node with several labels counts
// in each pair; an end without a label is in none.

import { describe, it, expect } from 'vitest'
import { GrafeoDB } from '../grafeo.js'

async function seeded() {
  const db = GrafeoDB.create()
  await db.execute(
    "INSERT (alix:Person:Employee {name: 'Alix', age: 30}), (gus:Person {name: 'Gus', age: '31'}), " +
      "(acme:Company:Org {name: 'Acme'}), ({note: 'loose'}), " +
      '(alix)-[:WORKS_AT {since: 2020}]->(acme), (gus)-[:WORKS_AT {since: 2.5e0}]->(acme)',
  )
  await db.execute("MATCH (n {note: 'loose'}), (g:Person {name: 'Gus'}) INSERT (g)-[:LINK]->(n)")
  return db
}

const named = (entries, name) => entries.find((entry) => entry.name === name)

describe('schema summary (Node.js)', () => {
  it('lists the properties of a label with their types', async () => {
    const lpg = (await seeded()).schema()
    expect(lpg.mode).toBe('lpg')
    expect(named(lpg.labels, 'Person')).toEqual({
      name: 'Person',
      count: 2,
      properties: [
        { name: 'age', types: ['INT64', 'STRING'] },
        { name: 'name', types: ['STRING'] },
      ],
    })
  })

  it('lists every label pair an edge type connects', async () => {
    const lpg = (await seeded()).schema()
    expect(lpg.mode).toBe('lpg')
    expect(named(lpg.edge_types, 'WORKS_AT')).toEqual({
      name: 'WORKS_AT',
      count: 2,
      properties: [{ name: 'since', types: ['FLOAT64', 'INT64'] }],
      endpoints: [
        { source: 'Employee', target: 'Company', count: 1 },
        { source: 'Employee', target: 'Org', count: 1 },
        { source: 'Person', target: 'Company', count: 2 },
        { source: 'Person', target: 'Org', count: 2 },
      ],
    })
    expect(named(lpg.edge_types, 'LINK').endpoints).toEqual([])
  })

  it('round-trips through JSON and agrees with db.labels()', async () => {
    const db = await seeded()
    const schema = db.schema()
    expect(JSON.parse(JSON.stringify(schema))).toEqual(schema)
    const labels = (await db.execute('CALL db.labels() YIELD label RETURN label')).toArray().map((row) => row.label)
    expect(labels).toEqual(schema.labels.map((label) => label.name))
    expect(schema.property_keys).toEqual(['age', 'name', 'note', 'since'])
  })
})
