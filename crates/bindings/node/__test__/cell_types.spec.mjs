// A value is a node or an edge where the type its cell was built with says
// so. An object with the keys of a node ({_id: 1, _labels: []}) is a map:
// nodes() used to list it, guessing from the keys. An edge with a property
// named _labels is an edge. cellType(row, column) says where the nodes and
// edges of a row are (experimental, new in 0.6.4).

import { describe, it, expect } from 'vitest'
import { GrafeoDB } from '../grafeo.js'

async function knows() {
  const db = GrafeoDB.create()
  await db.execute(
    "INSERT (:Person {name: 'Alix'})-[:KNOWS {since: 2020, _labels: ['L']}]->(:Person {name: 'Gus'})",
  )
  return db
}

describe('cell types', () => {
  it('reads a map with the keys of a node as a map', async () => {
    const db = await knows()
    for (const run of [(q) => db.execute(q), (q) => db.executeCypher(q)]) {
      const result = await run(
        "MATCH (a:Person {name: 'Alix'}) RETURN a, {_id: 99, _labels: ['Person']} AS fake, " +
          "{_id: 98, _type: 'KNOWS', _source: 1, _target: 2} AS fakeEdge",
      )
      expect(result.nodes().map((node) => node.get('name'))).toEqual(['Alix'])
      expect(result.edges()).toEqual([])
      expect(result.cellType(0, 0)).toBe('node')
      expect(result.cellType(0, 1)).not.toBe('node')
      expect(result.cellType(0, 2)).not.toBe('edge')
    }
    db.close()
  })

  it('reads an edge with a _labels property as an edge', async () => {
    const db = await knows()
    const result = await db.execute('MATCH ()-[r:KNOWS]->() RETURN r')
    expect(result.nodes()).toEqual([])
    const [edge] = result.edges()
    expect(edge.edgeType).toBe('KNOWS')
    expect(edge.get('_labels')).toEqual(['L'])
    expect(result.cellType(0, 0)).toBe('edge')
    db.close()
  })

  it('types nested nodes and each branch of a union', async () => {
    const db = await knows()
    const nested = await db.execute(
      "MATCH p = (a:Person {name: 'Alix'})-[r]->(b) RETURN [a, 1] AS l, {k: b} AS m, p",
    )
    expect(nested.cellType(0, 0).tuple[0]).toBe('node')
    expect(nested.cellType(0, 1)).toEqual({ struct: { k: 'node' } })
    expect(nested.cellType(0, 2)).toBe('path')
    const union = await db.execute(
      'MATCH (a:Person) RETURN a AS x UNION ALL MATCH ()-[r]->() RETURN r AS x',
    )
    expect([0, 1, 2].map((row) => union.cellType(row, 0))).toEqual(['node', 'node', 'edge'])
    db.close()
  })
})
