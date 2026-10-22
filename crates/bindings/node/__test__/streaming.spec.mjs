import { describe, it, expect } from 'vitest'
import { GrafeoDB } from '../index.js'

function seedPeople() {
  const db = GrafeoDB.create()
  for (const [name, age] of [
    ['Alix', 32],
    ['Gus', 28],
    ['Vincent', 45],
    ['Jules', 40],
    ['Mia', 24],
  ]) {
    db.createNode(['Person'], { name, age })
  }
  return db
}

async function drainStream(stream) {
  const rows = []
  let row
  while ((row = await stream.next()) !== null) {
    rows.push(row)
  }
  return rows
}

describe('executeStream', () => {
  it('yields the same rows as execute()', async () => {
    const db = seedPeople()
    try {
      const query = 'MATCH (p:Person) RETURN p.name AS name, p.age AS age'
      const materialized = (await db.execute(query)).toArray()
      const streamed = await drainStream(await db.executeStream(query))

      expect(streamed.length).toBe(materialized.length)
      const key = (r) =>
        Object.keys(r)
          .sort()
          .map((k) => `${k}=${r[k]}`)
          .join('|')
      expect(new Set(streamed.map(key))).toEqual(new Set(materialized.map(key)))
    } finally {
      db.close()
    }
  })

  it('exposes column names before iteration', async () => {
    const db = seedPeople()
    try {
      const stream = await db.executeStream(
        'MATCH (p:Person) RETURN p.name AS name, p.age AS age',
      )
      expect(stream.columns).toEqual(['name', 'age'])
    } finally {
      db.close()
    }
  })

  it('yields row objects keyed by column name', async () => {
    const db = seedPeople()
    try {
      const stream = await db.executeStream('MATCH (p:Person) RETURN p.name')
      const rows = await drainStream(stream)
      expect(rows.length).toBe(5)
      expect(rows.every((r) => typeof r === 'object' && 'p.name' in r)).toBe(true)
    } finally {
      db.close()
    }
  })

  it('returns null after exhaustion', async () => {
    const db = seedPeople()
    try {
      const stream = await db.executeStream('MATCH (p:Person) RETURN p.name')
      await drainStream(stream)
      const extra = await stream.next()
      expect(extra).toBe(null)
    } finally {
      db.close()
    }
  })

  it('respects WHERE filters', async () => {
    const db = seedPeople()
    try {
      const rows = await drainStream(
        await db.executeStream('MATCH (p:Person) WHERE p.age > 30 RETURN p.name AS name'),
      )
      const names = new Set(rows.map((r) => r.name))
      expect(names).toEqual(new Set(['Alix', 'Vincent', 'Jules']))
    } finally {
      db.close()
    }
  })

  it('returns no rows for an empty result', async () => {
    const db = seedPeople()
    try {
      const rows = await drainStream(
        await db.executeStream('MATCH (p:Person) WHERE p.age > 999 RETURN p.name'),
      )
      expect(rows).toEqual([])
    } finally {
      db.close()
    }
  })

  it('close() short-circuits the stream', async () => {
    const db = seedPeople()
    try {
      const stream = await db.executeStream('MATCH (p:Person) RETURN p.name')
      const first = await stream.next()
      expect(first).not.toBe(null)
      await stream.close()
      expect(await stream.next()).toBe(null)
    } finally {
      db.close()
    }
  })

  it('rejects mutating queries', async () => {
    const db = seedPeople()
    try {
      await expect(db.executeStream("INSERT (:Person {name: 'Butch'})")).rejects.toThrow()
    } finally {
      db.close()
    }
  })

  it('rejects ORDER BY (push-based pipeline)', async () => {
    const db = seedPeople()
    try {
      await expect(
        db.executeStream('MATCH (p:Person) RETURN p.name AS n ORDER BY n'),
      ).rejects.toThrow()
    } finally {
      db.close()
    }
  })

  it('rejects session commands', async () => {
    const db = GrafeoDB.create()
    try {
      await expect(db.executeStream('SESSION SET GRAPH analytics')).rejects.toThrow()
    } finally {
      db.close()
    }
  })

  it('rejects EXPLAIN', async () => {
    const db = seedPeople()
    try {
      await expect(
        db.executeStream('EXPLAIN MATCH (p:Person) RETURN p.name'),
      ).rejects.toThrow()
    } finally {
      db.close()
    }
  })

  it('supports async iteration via generator wrapper', async () => {
    const db = seedPeople()
    try {
      async function* iterate(stream) {
        let row
        while ((row = await stream.next()) !== null) yield row
      }

      const rows = []
      for await (const row of iterate(await db.executeStream('MATCH (p:Person) RETURN p.name'))) {
        rows.push(row)
      }
      expect(rows.length).toBe(5)
    } finally {
      db.close()
    }
  })
})
