// Temporal values keep their GQL type both ways: a `Date` is a TIMESTAMP, and
// a DATE, TIME, DURATION or ZONED DATETIME, which JavaScript has no value for,
// comes back as the tag it goes in as (`{ $time: '12:30:00+02:00' }`), as a
// decimal does, so a value read back goes in again as the same type.

import { describe, it, expect } from 'vitest'
import { GrafeoDB } from '../grafeo.js'

/** Whether `$v` arrived as `gqlType`, and what came back. */
async function typedBack(db, gqlType, value) {
  const [row] = (await db.execute(`RETURN $v IS TYPED ${gqlType} AS typed, $v AS back`, { v: value })).toArray()
  return row
}

const VALUES = [
  ['TIMESTAMP', new Date(Date.UTC(2024, 2, 5, 10, 30, 0, 123))],
  ['LOCAL DATETIME', new Date(Date.UTC(1969, 6, 20, 20, 17, 40, 654))],
  ['ZONED DATETIME', { $zoned_datetime: '2024-03-05T10:30:00.25+01:00' }],
  ['ZONED DATETIME', { $zoned_datetime: '2024-03-05T10:30:00Z' }],
  ['DATE', { $date: '2024-02-29' }],
  ['TIME', { $time: '14:30:15.5' }],
  ['TIME', { $time: '12:30:00+02:00' }],
  ['TIME', { $time: '23:59:59.999999999Z' }],
  ['DURATION', { $duration: 'P1Y2M3DT4H5M6.5S' }],
  ['DURATION', { $duration: '-P1M2DT0.000000003S' }],
  ['DURATION', { $duration: 'P1D' }],
  ['DURATION', { $duration: 'PT0S' }],
]

describe('temporal values', () => {
  it('go in as their type, come back as they went in, and go in again as their type', async () => {
    const db = GrafeoDB.create()
    for (const [gqlType, value] of VALUES) {
      const { typed, back } = await typedBack(db, gqlType, value)
      expect(typed, `${JSON.stringify(value)} as ${gqlType}`).toBe(true)
      expect(back).toEqual(value)
      expect((await typedBack(db, gqlType, back)).typed, `${JSON.stringify(back)} again`).toBe(true)
    }
  })

  it('come back typed from a query, in lists and maps, and go in again', async () => {
    const db = GrafeoDB.create()
    const [row] = (
      await db.execute(
        "RETURN DATE '2024-03-05' AS d, TIME '12:30:00+02:00' AS zt, TIME '08:15:30.25' AS lt, " +
          "DURATION 'P1Y2M3DT4H5M6.5S' AS dur, ZONED DATETIME '2024-03-05T10:30:00.5+05:30' AS zdt, " +
          "local_datetime('2024-03-05T10:30:00.001') AS ldt, " +
          "[DATE '2024-03-05', DATE '2024-03-06'] AS dates, " +
          "{at: TIME '12:30:00Z', took: DURATION 'PT1M'} AS m",
      )
    ).toArray()
    expect(row).toEqual({
      d: { $date: '2024-03-05' },
      zt: { $time: '12:30:00+02:00' },
      lt: { $time: '08:15:30.25' },
      dur: { $duration: 'P1Y2M3DT4H5M6.5S' },
      zdt: { $zoned_datetime: '2024-03-05T10:30:00.5+05:30' },
      ldt: new Date(Date.UTC(2024, 2, 5, 10, 30, 0, 1)),
      dates: [{ $date: '2024-03-05' }, { $date: '2024-03-06' }],
      m: { at: { $time: '12:30:00Z' }, took: { $duration: 'PT1M' } },
    })
    const [again] = (
      await db.execute(
        'RETURN $d IS TYPED DATE AND $zt IS TYPED TIME AND $lt IS TYPED TIME AND $dur IS TYPED DURATION ' +
          'AND $zdt IS TYPED ZONED DATETIME AND $ldt IS TYPED LOCAL DATETIME AND $dates IS TYPED LIST<DATE> ' +
          'AND $m.at IS TYPED TIME AND $m.took IS TYPED DURATION AS typed, ' +
          "$zt = TIME '12:30:00+02:00' AND $zdt = ZONED DATETIME '2024-03-05T10:30:00.5+05:30' AS equal",
        row,
      )
    ).toArray()
    expect(again).toEqual({ typed: true, equal: true })
  })

  it('a timestamp before 1970 is the millisecond at or before it', async () => {
    const db = GrafeoDB.create()
    const [row] = (await db.execute("RETURN local_datetime('1969-12-31T23:59:59.9995') AS t")).toArray()
    expect(row.t.getTime()).toBe(Date.UTC(1969, 11, 31, 23, 59, 59, 999))
  })

  it('stored values come back typed from a node, a stream and an AsOf read', async () => {
    const db = GrafeoDB.create(undefined, { historyRetention: 'forever' })
    const born = { $date: '1994-05-12' }
    const wakes = { $time: '07:30:00+01:00' }
    const seen = [new Date(Date.UTC(2024, 2, 5, 10, 30))]
    const [{ id }] = (
      await db.execute('INSERT (p:Person {name: $name, born: $born, wakes: $wakes, seen: $seen}) RETURN id(p) AS id', {
        name: 'Shosanna',
        born,
        wakes,
        seen,
      })
    ).toArray()
    const want = { born, wakes, seen }
    const node = db.getNode(id).properties()
    expect({ born: node.born, wakes: node.wakes, seen: node.seen }).toEqual(want)

    const stream = await db.executeStream('MATCH (p:Person) RETURN p.born AS born, p.wakes AS wakes')
    const streamed = await stream.next()
    expect(streamed).toEqual({ born, wakes })

    const view = db.asOf(db.currentEpoch())
    expect({
      born: view.getNodeProperty(id, 'born'),
      wakes: view.getNodeProperty(id, 'wakes'),
      seen: view.getNodeProperty(id, 'seen'),
    }).toEqual(want)
  })
})
