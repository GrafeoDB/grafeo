// A BigInt past 64 bits is an exact DECIMAL, up to 38 digits: a BigInt that
// fits 64 bits stays INT64, one past it but within 38 digits reads as a
// DECIMAL with scale 0 (it used to be an error, "too large for i64"), and one
// past 38 digits is an error naming the property and the value.

import { describe, it, expect } from 'vitest'
import { GrafeoDB } from '../grafeo.js'

const WIDE = [2n ** 63n, 2n ** 64n, -(2n ** 63n) - 1n, 10n ** 38n - 1n, -(10n ** 38n) + 1n]

async function stored(db) {
  const result = await db.execute(
    'MATCH (n:N) RETURN n.v AS v, n.v IS TYPED INT64 AS int, n.v IS TYPED DECIMAL AS decimal',
  )
  return result.toArray()
}

describe('a BigInt property', () => {
  it('stays INT64 when it fits 64 bits', async () => {
    const db = GrafeoDB.create()
    db.createNode(['N'], { v: 2n ** 63n - 1n })
    db.createNode(['N'], { v: -(2n ** 63n) })
    const rows = await stored(db)
    expect(rows.map((row) => [row.int, row.decimal])).toEqual([
      [true, false],
      [true, false],
    ])
  })

  for (const value of WIDE) {
    it(`past 64 bits is an exact DECIMAL: ${value}`, async () => {
      const db = GrafeoDB.create()
      db.createNode(['N'], { v: value })
      const node = db.createNode(['N'], {})
      db.setNodeProperty(node.id, 'v', value)
      expect(await stored(db)).toEqual([
        { v: { $decimal: String(value) }, int: false, decimal: true },
        { v: { $decimal: String(value) }, int: false, decimal: true },
      ])
    })
  }

  for (const value of [10n ** 38n, -(10n ** 38n), 2n ** 200n]) {
    it(`past 38 digits is an error naming the property: ${value}`, () => {
      const db = GrafeoDB.create()
      const pattern = new RegExp(`property 'v'.*${value}.*38 digits`)
      expect(() => db.createNode(['N'], { v: value })).toThrow(pattern)
      const node = db.createNode(['N'], {})
      expect(() => db.setNodeProperty(node.id, 'v', value)).toThrow(pattern)
      expect(db.nodeCount()).toBe(1)
    })
  }

  it('keeps arithmetic over INT64 and DECIMAL rows exact', async () => {
    const db = GrafeoDB.create()
    for (const value of [5n, 2n ** 64n]) db.createNode(['N'], { v: value })
    const next = (await db.execute('MATCH (n:N) RETURN n.v + 1 AS next ORDER BY next')).toArray()
    expect(next.map((row) => row.next)).toEqual([6, { $decimal: String(2n ** 64n + 1n) }])
    const total = (await db.execute('MATCH (n:N) RETURN sum(n.v) AS total')).toArray()
    expect(total).toEqual([{ total: { $decimal: String(2n ** 64n + 5n) } }])
  })
})

// Parameters convert from the JavaScript value as properties do (they used
// to go through JSON, which made a BigInt past 64 bits a string and a whole
// number past 2^32 a float).
describe('a parameter', () => {
  const typed = async (db, value) =>
    (
      await db.execute(
        'RETURN $x AS x, $x IS TYPED INT64 AS int, $x IS TYPED DECIMAL AS decimal, $x IS TYPED FLOAT64 AS float',
        { x: value },
      )
    ).toArray()[0]

  for (const value of WIDE) {
    it(`BigInt past 64 bits is an exact DECIMAL: ${value}`, async () => {
      const db = GrafeoDB.create()
      expect(await typed(db, value)).toEqual({
        x: { $decimal: String(value) },
        int: false,
        decimal: true,
        float: false,
      })
    })
  }

  it('BigInt that fits 64 bits and a whole number past 2^32 are INT64', async () => {
    const db = GrafeoDB.create()
    for (const value of [2n ** 63n - 1n, -(2n ** 63n), 5000000000, -5000000000, 3]) {
      const row = await typed(db, value)
      expect([row.int, row.decimal, row.float], String(value)).toEqual([true, false, false])
    }
  })

  it('a fraction or a float past 2^53 stays FLOAT64 with its value', async () => {
    const db = GrafeoDB.create()
    for (const value of [1.5, -0.25, 1e300, 2 ** 60]) {
      const row = await typed(db, value)
      expect([row.x, row.float], String(value)).toEqual([value, true])
    }
  })

  it('past 38 digits is an error naming the parameter', async () => {
    const db = GrafeoDB.create()
    const pattern = new RegExp(`parameter 'x'.*${10n ** 38n}.*38 digits`)
    await expect(db.execute('RETURN $x AS x', { x: 10n ** 38n })).rejects.toThrow(pattern)
    await expect(db.execute('RETURN $x AS x', { x: [1, 10n ** 38n] })).rejects.toThrow(pattern)
  })

  it('inside a list and a map follows the rule, and a tagged object keeps its type', async () => {
    const db = GrafeoDB.create()
    const rows = (
      await db.execute('UNWIND $xs AS v RETURN v IS TYPED DECIMAL AS decimal', { xs: [3, 2n ** 64n] })
    ).toArray()
    expect(rows.map((row) => row.decimal)).toEqual([false, true])
    const [row] = (
      await db.execute('RETURN $m.big AS big, $d IS TYPED DATE AS date, $n AS n', {
        m: { big: 10n ** 38n - 1n },
        d: { $date: '2024-01-15' },
        n: { $decimal: '1.50' },
      })
    ).toArray()
    expect(row).toEqual({ big: { $decimal: String(10n ** 38n - 1n) }, date: true, n: { $decimal: '1.50' } })
  })

  it('follows the rule in a transaction and in another language', async () => {
    const db = GrafeoDB.create()
    const tx = db.beginTransaction()
    const inTx = (await tx.execute('RETURN $x IS TYPED DECIMAL AS d', { x: 2n ** 64n })).toArray()
    tx.rollback()
    const cypher = (await db.executeCypher('RETURN $x AS x', { x: 5000000000 })).toArray()
    expect([inTx, cypher]).toEqual([[{ d: true }], [{ x: 5000000000 }]])
  })

  it('must be an object', async () => {
    const db = GrafeoDB.create()
    await expect(db.execute('RETURN 1', [1, 2])).rejects.toThrow(/params must be an object/)
    await expect(db.execute('RETURN 1', 5)).rejects.toThrow(/params must be an object/)
  })
})
