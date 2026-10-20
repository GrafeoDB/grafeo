// `QueryResult.profile`: the operator tree of a `PROFILE` statement (#461).

import { describe, it, expect } from 'vitest'
import { GrafeoDB } from '../grafeo.js'

async function people() {
  const db = GrafeoDB.create()
  await db.execute(
    "INSERT (:Person {name: 'Alix', age: 30}), (:Person {name: 'Gus', age: 25}), (:Person {name: 'Vincent', age: 35})",
  )
  return db
}

function* operators(node, depth = 0) {
  yield [depth, node]
  for (const child of node.children) yield* operators(child, depth + 1)
}

describe('QueryResult.profile', () => {
  it('holds the tree of the PROFILE text', async () => {
    const db = await people()
    const result = await db.execute('PROFILE MATCH (p:Person) WHERE p.age > 28 RETURN p.name')
    const profile = result.profile
    expect(Object.keys(profile).sort()).toEqual(['root', 'totalTimeMs'])
    expect(profile.root.rows).toBe(2)
    expect(profile.totalTimeMs).toBeLessThanOrEqual(result.executionTimeMs)

    const lines = String(result.scalar())
      .split('\n')
      .filter((line) => line && !line.startsWith('Total time'))
    const all = [...operators(profile.root)]
    expect(lines.length).toBe(all.length)
    all.forEach(([depth, operator], index) => {
      expect(Object.keys(operator).sort()).toEqual(['calls', 'children', 'detail', 'operator', 'rows', 'selfTimeMs', 'timeMs'])
      const prefix = `${'  '.repeat(depth)}${operator.operator} (${operator.detail})  rows=${operator.rows}  `
      expect(lines[index].startsWith(prefix)).toBe(true)
      expect(operator.timeMs).toBeGreaterThanOrEqual(operator.selfTimeMs)
    })
  })

  it('is set in a transaction too', async () => {
    const db = await people()
    const tx = db.beginTransaction()
    const result = await tx.execute('PROFILE MATCH (p:Person) RETURN count(p) AS c')
    expect(result.profile.root.rows).toBe(1)
    tx.rollback()
  })

  it('is null for other statements', async () => {
    const db = await people()
    expect((await db.execute('MATCH (p:Person) RETURN p.name')).profile).toBeNull()
    expect((await db.execute('EXPLAIN MATCH (p:Person) RETURN p.name')).profile).toBeNull()
  })
})
