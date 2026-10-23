// A statement called again with other values answers for those values:
// the engine optimizes it once with its parameters in and fills a copy of
// that plan with each call's values, here through the Node.js binding's
// parameter path.

import { describe, it, expect } from 'vitest'
import { GrafeoDB } from '../grafeo.js'

const PEOPLE = [
  ['Alix', 30],
  ['Gus', 25],
  ['Vincent', 41],
  ['Mia', 25],
]

async function people() {
  const db = GrafeoDB.create()
  for (const [name, age] of PEOPLE) {
    await db.execute('INSERT (:Person {name: $name, age: $age})', { name, age })
  }
  return db
}

async function column(db, query, params, name) {
  return (await db.execute(query, params)).toArray().map((row) => row[name])
}

describe('prepared parameters', () => {
  it('a lookup takes each call\'s value', async () => {
    const db = await people()
    const query = 'MATCH (p:Person) WHERE p.name = $name RETURN p.age AS age'
    for (const [name, ages] of [
      ['Gus', [25]],
      ['Alix', [30]],
      ['Nobody', []],
      ['Gus', [25]],
    ]) {
      expect(await column(db, query, { name }, 'age')).toEqual(ages)
    }
  })

  it('SKIP and LIMIT take each call\'s values', async () => {
    const db = await people()
    const query = 'MATCH (p:Person) RETURN p.name AS name ORDER BY p.name SKIP $skip LIMIT $limit'
    const names = PEOPLE.map(([name]) => name).sort()
    for (const [skip, limit] of [
      [0, 1],
      [1, 2],
      [3, 5],
      [0, 4],
    ]) {
      expect(await column(db, query, { skip, limit }, 'name')).toEqual(names.slice(skip, skip + limit))
    }
  })

  it('a value of another type answers as its literal', async () => {
    const db = await people()
    const query = 'MATCH (p:Person) WHERE p.age = $age RETURN p.name AS name'
    for (const [age, literal] of [
      [25, '25'],
      ['25', "'25'"],
      [25, '25'],
      [true, 'true'],
    ]) {
      const got = (await column(db, query, { age }, 'name')).sort()
      const expected = (await column(db, query.replace('$age', literal), {}, 'name')).sort()
      expect(got).toEqual(expected)
    }
  })

  it('a Cypher MERGE takes each call\'s values', async () => {
    const db = await people()
    const merge =
      'MATCH (a:Person {name: $a}), (b:Person {name: $b}) MERGE (a)-[:LIKES]->(b) RETURN a.name AS a, b.name AS b'
    const count = 'MATCH (:Person)-[r:LIKES]->(:Person) RETURN count(r) AS n'
    for (const [a, b, total] of [
      ['Alix', 'Gus', 1],
      ['Gus', 'Mia', 2],
      ['Alix', 'Gus', 2],
    ]) {
      const rows = (await db.executeCypher(merge, { a, b })).toArray()
      expect(rows.map((row) => [row.a, row.b])).toEqual([[a, b]])
      expect(await column(db, count, {}, 'n')).toEqual([total])
    }
  })
})
