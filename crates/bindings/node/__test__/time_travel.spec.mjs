// Time travel: reads of the database as it was at an earlier epoch or time.
// `historyRetention` keeps the history; `asOf(point)` gives a read-only view,
// `executeAsOf` runs one query there and `epochAsOf` names the epoch. A point
// is a bigint (an epoch), a Date (a time) or { ago: milliseconds }; a plain
// number is a TypeError, as it names neither.

import { mkdtempSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { describe, it, expect } from 'vitest'
import { GrafeoDB } from '../grafeo.js'

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms))

async function people(target) {
  const result = await target.execute('MATCH (p:Person) RETURN p.name AS name ORDER BY name')
  return result.toArray().map((row) => row.name)
}

async function seeded() {
  const db = GrafeoDB.create(undefined, { historyRetention: 'forever' })
  await db.execute("INSERT (:Person {name: 'Alix'})")
  return db
}

describe('time travel', () => {
  it('reads the state of an epoch through asOf', async () => {
    const db = await seeded()
    const then = db.currentEpoch()
    expect(typeof then).toBe('bigint')
    await db.execute("INSERT (:Person {name: 'Gus'})")
    const view = db.asOf(then)
    expect(view.epoch).toBe(then)
    expect(await people(view)).toEqual(['Alix'])
    expect(await people(db)).toEqual(['Alix', 'Gus'])
    view.close()
  })

  it('refuses a write through the view', async () => {
    const db = await seeded()
    const view = db.asOf(db.currentEpoch())
    await expect(view.execute("INSERT (:Person {name: 'Mia'})")).rejects.toThrow()
    expect(await people(db)).toEqual(['Alix'])
    view.close()
  })

  it('reads nodes and properties through the view', async () => {
    const db = await seeded()
    const id = (await db.execute('MATCH (p:Person) RETURN id(p) AS id')).toArray()[0].id
    const then = db.currentEpoch()
    await db.execute("MATCH (p:Person) SET p.name = 'Alix B.'")
    const view = db.asOf(then)
    expect(view.getNode(id).get('name')).toBe('Alix')
    expect(view.getNodeProperty(id, 'name')).toBe('Alix')
    expect(view.getNode(id + 1000)).toBeNull()
    expect(view.getEdge(0)).toBeNull()
    view.close()
  })

  it('runs one query with executeAsOf and params', async () => {
    const db = await seeded()
    const then = db.currentEpoch()
    await db.execute("INSERT (:Person {name: 'Gus'})")
    const result = await db.executeAsOf('MATCH (p:Person) WHERE p.name <> $not RETURN count(p) AS n', then, {
      not: 'Vincent',
    })
    expect(result.toArray()[0].n).toBe(1)
  })

  it('reads the last commit at or before a Date', async () => {
    const db = await seeded()
    await sleep(5)
    const beforeGus = new Date()
    await sleep(5)
    await db.execute("INSERT (:Person {name: 'Gus'})")
    const view = db.asOf(beforeGus)
    expect(await people(view)).toEqual(['Alix'])
    view.close()
    expect(db.epochAsOf(new Date(Date.now() + 1000))).toBe(db.currentEpoch())
  })

  it('reads a duration ago with { ago }', async () => {
    const db = await seeded()
    expect(db.epochAsOf({ ago: 0 })).toBe(db.currentEpoch())
  })

  it('refuses a plain number and other values as a TypeError', () => {
    const db = GrafeoDB.create(undefined, { historyRetention: 'forever' })
    for (const point of [3, '3', null, true, {}, { ago: -1 }, { ago: '5' }, -1n, new Date(NaN)]) {
      expect(() => db.epochAsOf(point), String(point)).toThrow(TypeError)
    }
  })

  it('fails for a time without history', async () => {
    const db = GrafeoDB.create()
    expect(() => db.asOf(new Date(2000, 0, 1))).toThrow(/needs history/)
  })

  it('throws on a closed view', async () => {
    const db = await seeded()
    const view = db.asOf(db.currentEpoch())
    view.close()
    view.close()
    expect(() => view.getNode(0)).toThrow(/closed/)
    await expect(view.execute('MATCH (n) RETURN n')).rejects.toThrow(/closed/)
  })

  it('takes the retention as off, forever or milliseconds', () => {
    for (const historyRetention of ['off', 'forever', 3_600_000]) {
      GrafeoDB.create(undefined, { historyRetention })
    }
    expect(() => GrafeoDB.create(undefined, { historyRetention: 'always' })).toThrow(/historyRetention/)
    expect(() => GrafeoDB.create(undefined, { historyRetention: -1 })).toThrow(/historyRetention/)
  })

  it('stores the retention in the file', async () => {
    const path = join(mkdtempSync(join(tmpdir(), 'grafeo-history-')), 'history.grafeo')
    let db = GrafeoDB.create(path, { historyRetention: 'forever' })
    await db.execute("INSERT (:Person {name: 'Alix'})")
    const then = db.currentEpoch()
    await db.execute("INSERT (:Person {name: 'Gus'})")
    db.close()
    db = GrafeoDB.open(path)
    const view = db.asOf(then)
    expect(await people(view)).toEqual(['Alix'])
    view.close()
    db.close()
  })
})
