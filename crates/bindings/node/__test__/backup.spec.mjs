// Full and incremental backups and a restore to an epoch: a full backup is
// an image of the database at an epoch, written while commits go on; the
// incremental backups after it hold the WAL after that epoch.

import { copyFileSync, mkdtempSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { describe, it, expect } from 'vitest'
import { GrafeoDB } from '../grafeo.js'

const range = (n) => Array.from({ length: n }, (_, i) => i)

// The seq of every Person and of every City, each sorted.
async function committed(db) {
  const seqs = async (label) =>
    (await db.execute(`MATCH (n:${label}) RETURN n.seq AS s ORDER BY s`))
      .toArray()
      .map((row) => Number(row.s))
  return [await seqs('Person'), await seqs('City')]
}

async function people(db) {
  const result = await db.execute('MATCH (p:Person) RETURN p.name AS name ORDER BY name')
  return result.toArray().map((row) => row.name)
}

describe('backups', () => {
  it('restore full and incremental backups to each epoch', async () => {
    const dir = mkdtempSync(join(tmpdir(), 'grafeo-backup-'))
    const backups = join(dir, 'backups')
    const db = GrafeoDB.create(join(dir, 'berlin.grafeo'))
    await db.execute("INSERT (:Person {name: 'Alix'})")
    db.backupFull(backups)
    const fullEpoch = db.currentEpoch()
    await db.execute("INSERT (:Person {name: 'Gus'})")
    db.backupIncremental(backups)
    const latest = db.currentEpoch()
    db.close()

    const atFull = join(dir, 'full.grafeo')
    GrafeoDB.restoreToEpoch(backups, Number(fullEpoch), atFull)
    let restored = GrafeoDB.open(atFull)
    expect(await people(restored)).toEqual(['Alix'])
    restored.close()

    const atLatest = join(dir, 'latest.grafeo')
    GrafeoDB.restoreToEpoch(backups, Number(latest), atLatest)
    restored = GrafeoDB.open(atLatest)
    expect(await people(restored)).toEqual(['Alix', 'Gus'])
    restored.close()
  })

  it('an incremental backup works across a close', async () => {
    // The chain survives restarts: close() keeps the WAL the next incremental reads.
    const dir = mkdtempSync(join(tmpdir(), 'grafeo-backup-'))
    const path = join(dir, 'prague.grafeo')
    const backups = join(dir, 'backups')
    let db = GrafeoDB.create(path)
    await db.execute("INSERT (:Person {name: 'Alix'})")
    db.backupFull(backups)
    const fullEpoch = db.currentEpoch()
    db.close()

    db = GrafeoDB.open(path)
    await db.execute("INSERT (:Person {name: 'Gus'})")
    db.close()

    db = GrafeoDB.open(path)
    db.backupIncremental(backups)
    const latest = db.currentEpoch()
    db.close()

    for (const [epoch, expected] of [[fullEpoch, ['Alix']], [latest, ['Alix', 'Gus']]]) {
      const at = join(dir, `restored-${epoch}.grafeo`)
      GrafeoDB.restoreToEpoch(backups, Number(epoch), at)
      const restored = GrafeoDB.open(at)
      expect(await people(restored)).toEqual(expected)
      restored.close()
    }
  })

  it('a full backup during concurrent writes holds whole statements', async () => {
    // Writes run on the thread pool while the backup runs on this thread: the
    // image holds every statement whole (a Person and a City each), the ones
    // committed before it, and nothing of a transaction open during it.
    const dir = mkdtempSync(join(tmpdir(), 'grafeo-backup-'))
    const backups = join(dir, 'backups')
    const db = GrafeoDB.create(join(dir, 'lyon.grafeo'))
    const insert = (seq) => db.execute(`INSERT (:Person {seq: ${seq}}), (:City {seq: ${seq}})`)
    for (let seq = 0; seq < 20; seq++) await insert(seq)
    const open = db.beginTransaction()
    await open.execute('INSERT (:Person {seq: -1})')
    const writes = range(280).map((i) => insert(20 + i))
    db.backupFull(backups)
    open.rollback()
    await Promise.all(writes)

    const image = join(dir, 'image.grafeo')
    copyFileSync(join(backups, 'backup_full_0000.grafeo'), image)
    const copy = GrafeoDB.open(image)
    const [people, cities] = await committed(copy)
    copy.close()
    expect(people).toEqual(cities)
    expect(people.slice(0, 20)).toEqual(range(20))
    expect(people.length).toBeLessThanOrEqual(300)
    expect(people).not.toContain(-1)

    db.backupIncremental(backups)
    const latest = db.currentEpoch()
    db.close()
    const restoredPath = join(dir, 'restored.grafeo')
    GrafeoDB.restoreToEpoch(backups, Number(latest), restoredPath)
    for (let reopen = 0; reopen < 2; reopen++) {
      // Opened twice: the restore and its reopen hold every write.
      const restored = GrafeoDB.open(restoredPath)
      expect(await committed(restored)).toEqual([range(300), range(300)])
      restored.close()
    }
  })
})
