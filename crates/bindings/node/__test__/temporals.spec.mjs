// A timestamp comes back as a Date when it is whole milliseconds, which a Date
// holds, and as the tag { $timestamp_us } when it has microseconds, which a
// Date would floor away. Passed back, either is the same instant.

import { describe, it, expect } from 'vitest'
import { GrafeoDB } from '../grafeo.js'

const value = async (db, expression, params) =>
  (await db.execute(`RETURN ${expression} AS v`, params)).toArray()[0].v

describe('temporals', () => {
  it('keeps the microseconds of a timestamp', async () => {
    const db = GrafeoDB.create()
    const expression = "localdatetime('2024-01-15T10:30:00.000001')"
    const v = await value(db, expression)
    expect(v).not.toBeInstanceOf(Date)
    expect(typeof v.$timestamp_us).toBe('number')
    expect(await value(db, 'toString($v)', { v })).toBe(
      await value(db, `toString(${expression})`),
    )
    db.close()
  })

  // Past 2^53 microseconds from 1970 (before 1685, after 2255) a number would
  // round them, so the tag holds a BigInt there.
  for (const text of ['1600-01-01T00:00:00.000001', '2300-06-30T12:00:00.000001']) {
    it(`keeps the microseconds of ${text}`, async () => {
      const db = GrafeoDB.create()
      const expression = `localdatetime('${text}')`
      const v = await value(db, expression)
      expect(typeof v.$timestamp_us).toBe('bigint')
      expect(await value(db, 'toString($v)', { v })).toBe(
        await value(db, `toString(${expression})`),
      )
      db.close()
    })
  }

  it('gives a Date for a timestamp of whole milliseconds', async () => {
    const db = GrafeoDB.create()
    const v = await value(db, "localdatetime('2024-01-15T10:30:00.123')")
    expect(v).toBeInstanceOf(Date)
    expect(v.toISOString()).toBe('2024-01-15T10:30:00.123Z')
    db.close()
  })
})
