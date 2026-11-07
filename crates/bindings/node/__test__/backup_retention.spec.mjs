// The WAL a backup cursor keeps is bounded, and the cursor can be released.
// `walRetentionLimit` caps the WAL a checkpoint keeps beyond itself for the
// backup cursor (1 GiB by default): once incremental backups stop and the
// checkpoints pass it, the next incremental backup fails with GRAFEO-S005,
// naming the limit, and a full backup starts a new chain.
// `releaseBackupCursor()` drops the cursor on purpose and returns whether one
// was released; the next incremental backup fails with GRAFEO-S005 until a
// full backup.

import { mkdtempSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { describe, it, expect } from 'vitest'
import { GrafeoDB } from '../grafeo.js'

const LIMIT = 256 * 1024

async function writeBatch(db, start) {
  await db.execute(
    `UNWIND range(${start}, ${start + 99}) AS i INSERT (:Person {i: i, note: '${'x'.repeat(300)}'})`,
  )
}

async function chainBroken(db, backups) {
  const error = await Promise.resolve()
    .then(() => db.backupIncremental(backups))
    .then(() => null, (e) => e)
  expect(error).not.toBeNull()
  expect(error.errorCode).toBe('GRAFEO-S005')
  return String(error.message)
}

describe('backup retention', () => {
  it('cuts a stopped backup at the limit, and a full backup restarts the chain', async () => {
    const dir = mkdtempSync(join(tmpdir(), 'grafeo-retention-'))
    const backups = join(dir, 'backups')
    const db = GrafeoDB.create(join(dir, 'db.grafeo'), { walRetentionLimit: LIMIT })
    db.backupFull(backups)
    for (let round = 0; round < 30; round++) {
      await writeBatch(db, round * 100)
      db.walCheckpoint()
    }
    expect(await chainBroken(db, backups)).toContain(String(LIMIT))
    db.backupFull(backups)
    await writeBatch(db, 3000)
    db.backupIncremental(backups)
    db.close()
  })

  it('releases the cursor once, and then asks for a full backup', async () => {
    const dir = mkdtempSync(join(tmpdir(), 'grafeo-retention-'))
    const backups = join(dir, 'backups')
    const db = GrafeoDB.create(join(dir, 'db.grafeo'))
    await writeBatch(db, 0)
    db.backupFull(backups)
    await writeBatch(db, 100)
    expect(db.releaseBackupCursor()).toBe(true)
    expect(db.releaseBackupCursor()).toBe(false)
    expect(await chainBroken(db, backups)).toContain('cursor')
    db.backupFull(backups)
    await writeBatch(db, 200)
    db.backupIncremental(backups)
    db.close()
  })
})
