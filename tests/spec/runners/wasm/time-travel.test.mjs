/**
 * Time travel and the clock in the WASM bindings.
 *
 * `new Database({ historyRetention })` keeps the history; `asOf(point)` gives
 * a read-only view, `executeAsOf` runs one query there and `epochAsOf` names
 * the epoch. A point is a bigint (an epoch), a Date (a time) or
 * { ago: milliseconds }; a plain number is a TypeError, as it names neither.
 * Before the engine read the host's clock, a commit with history on and
 * { ago } panicked in WASM, and timings were missing.
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

const sleep = (ms) => new Promise((done) => setTimeout(done, ms))
const people = (target) => target.execute('MATCH (p:Person) RETURN p.name AS name ORDER BY name').map((row) => row.name)

function seeded() {
  const db = new Database({ historyRetention: 'forever' })
  db.execute("INSERT (:Person {name: 'Alix'})")
  return db
}

describe.skipIf(!available)('time travel (WASM)', () => {
  it('commits with history on, and reads an epoch through asOf', () => {
    const db = seeded()
    const then = db.currentEpoch()
    expect(typeof then).toBe('bigint')
    db.execute("INSERT (:Person {name: 'Gus'})")
    const view = db.asOf(then)
    expect(view.epoch).toBe(then)
    expect(people(view)).toEqual(['Alix'])
    expect(people(db)).toEqual(['Alix', 'Gus'])
    expect(view.executeRaw('MATCH (p:Person) RETURN count(p) AS n').rows).toEqual([[1]])
    expect(() => view.execute("INSERT (:Person {name: 'Mia'})")).toThrow()
    view.close()
    view.close()
    expect(() => view.execute('RETURN 1')).toThrow(/closed/)
  })

  it('resolves { ago } and a Date', async () => {
    const db = seeded()
    expect(db.epochAsOf({ ago: 0 })).toBe(db.currentEpoch())
    await sleep(5)
    const beforeGus = new Date()
    await sleep(5)
    db.execute("INSERT (:Person {name: 'Gus'})")
    expect(db.executeAsOf('MATCH (p:Person) RETURN count(p) AS n', beforeGus)).toEqual([{ n: 1 }])
    expect(db.epochAsOf(new Date(Date.now() + 1000))).toBe(db.currentEpoch())
  })

  it('runs executeAsOf with params', () => {
    const db = seeded()
    const then = db.currentEpoch()
    db.execute("INSERT (:Person {name: 'Gus'})")
    const rows = db.executeAsOf('MATCH (p:Person) WHERE p.name <> $not RETURN count(p) AS n', then, { not: 'Vincent' })
    expect(rows).toEqual([{ n: 1 }])
    const view = db.asOf(then)
    expect(view.executeWithParams('MATCH (p:Person {name: $name}) RETURN p.name AS name', { name: 'Gus' })).toEqual([])
    view.close()
  })

  it('refuses a plain number and other values as a TypeError', () => {
    const db = seeded()
    for (const point of [3, '3', null, true, {}, { ago: -1 }, { ago: '5' }, -1n, new Date(NaN)]) {
      expect(() => db.epochAsOf(point), String(point)).toThrow(TypeError)
    }
  })

  it('needs history for a time, and keeps only its window', () => {
    expect(() => new Database().asOf(new Date(2000, 0, 1))).toThrow(/needs history/)
    const hour = new Database({ historyRetention: 3_600_000 })
    hour.execute("INSERT (:Person {name: 'Alix'})")
    expect(() => hour.epochAsOf(new Date(2000, 0, 1))).toThrow(/not retained/)
  })

  it('takes the retention as off, forever or milliseconds', () => {
    for (const historyRetention of ['off', 'forever', 0, 60_000]) {
      new Database({ historyRetention })
    }
    new Database({})
    expect(() => new Database({ historyRetention: 'always' })).toThrow(/historyRetention/)
    expect(() => new Database({ historyRetention: -1 })).toThrow(/historyRetention/)
  })
})

describe.skipIf(!available)('the clock (WASM)', () => {
  it('times queries and reads the current time', () => {
    const db = new Database()
    const raw = db.executeRaw('RETURN current_timestamp() AS t')
    expect(raw.executionTimeMs).toBeGreaterThanOrEqual(0)
    expect(raw.rows).toHaveLength(1)
    expect(raw.rows[0][0]).not.toBeNull()
  })
})
