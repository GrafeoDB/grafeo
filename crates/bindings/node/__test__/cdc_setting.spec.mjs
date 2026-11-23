// The CDC setting is stored with the database (#679): enableCdc() and
// disableCdc() persist it, so isCdcEnabled after a close and a reopen is what
// it was, and they throw when it cannot be stored (a read-only open). An
// in-memory database toggles as before.

import { mkdtempSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { describe, it, expect } from 'vitest'
import { GrafeoDB } from '../grafeo.js'

function file(name) {
  return join(mkdtempSync(join(tmpdir(), 'grafeo-cdc-')), name)
}

describe('the stored CDC setting', () => {
  it('survives a reopen both ways', () => {
    const path = file('amsterdam.grafeo')
    let db = GrafeoDB.open(path)
    expect(db.isCdcEnabled).toBe(false)
    db.enableCdc()
    db.close()
    db = GrafeoDB.open(path)
    expect(db.isCdcEnabled).toBe(true)
    db.disableCdc()
    db.close()
    db = GrafeoDB.open(path)
    expect(db.isCdcEnabled).toBe(false)
    db.close()
  })

  it('throws on a read-only open and changes nothing', async () => {
    const path = file('berlin.grafeo')
    const db = GrafeoDB.open(path)
    await db.execute("INSERT (:Person {name: 'Alix'})")
    db.close()
    const reader = GrafeoDB.openReadOnly(path)
    expect(() => reader.enableCdc()).toThrow()
    expect(reader.isCdcEnabled).toBe(false)
    reader.close()
  })

  it('toggles on an in-memory database', () => {
    const db = GrafeoDB.create()
    db.enableCdc()
    expect(db.isCdcEnabled).toBe(true)
    db.disableCdc()
    expect(db.isCdcEnabled).toBe(false)
  })
})
