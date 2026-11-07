// Automatic checkpoints are on by default, and the WAL's segment size is a
// setting. `checkpointWalSize` is the WAL size past the last checkpoint at
// which the database checkpoints by itself: 256 MiB when omitted, `null`
// turns it off (the WAL then grows until close or a manual checkpoint).
// `walSegmentSize` is where the WAL rotates, 64 MiB when omitted. A segment
// size outside 1 MiB to 1 GiB and a checkpoint size below 1 MiB are refused,
// naming the setting and its bounds.

import { existsSync, mkdtempSync, readdirSync, statSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { describe, it, expect } from 'vitest'
import { GrafeoDB } from '../grafeo.js'

const MIB = 1024 * 1024

function temp() {
  return mkdtempSync(join(tmpdir(), 'grafeo-checkpoint-'))
}

async function write(db, size = 64 * 1024) {
  await db.execute('INSERT (:Blob {b: $b})', { b: 'x'.repeat(size) })
}

function segments(path) {
  const dir = `${path}.wal`
  return readdirSync(dir)
    .filter((name) => /^wal_\d+\.log$/.test(name))
    .sort()
    .map((name) => statSync(join(dir, name)).size)
}

describe('checkpoint default', () => {
  it('checkpoints at 256 MiB of WAL by default, with no interval', () => {
    const db = GrafeoDB.create(join(temp(), 'amsterdam.grafeo'))
    const status = db.checkpointStatus()
    expect(status.walBytes).toBe(256 * MIB)
    expect(status.intervalMs ?? null).toBeNull()
    db.close()
  })

  it('turns automatic checkpoints off with null', () => {
    const db = GrafeoDB.create(join(temp(), 'berlin.grafeo'), { checkpointWalSize: null })
    expect(db.checkpointStatus().walBytes ?? null).toBeNull()
    db.close()
  })

  it('takes a checkpoint size', () => {
    const db = GrafeoDB.create(join(temp(), 'lisbon.grafeo'), { checkpointWalSize: MIB })
    expect(db.checkpointStatus().walBytes).toBe(MIB)
    db.close()
  })

  it('rotates the WAL at the segment size', async () => {
    const path = join(temp(), 'porto.grafeo')
    const db = GrafeoDB.create(path, { checkpointWalSize: null, walSegmentSize: MIB })
    for (let i = 0; i < 48; i++) {
      await write(db)
    }
    const sizes = segments(path)
    expect(sizes.length).toBeGreaterThanOrEqual(3)
    for (const size of sizes.slice(0, -1)) {
      expect(size).toBeGreaterThan(MIB / 2)
      expect(size).toBeLessThanOrEqual(MIB + 128 * 1024)
    }
    db.close()
  })

  it.each([
    [{ walSegmentSize: MIB - 1 }, ['walSegmentSize', 'wal_segment_size'], true],
    [{ walSegmentSize: 1024 * MIB + 1 }, ['walSegmentSize', 'wal_segment_size'], true],
    [{ checkpointWalSize: MIB - 1 }, ['checkpointWalSize', 'checkpoint_wal_size'], false],
    [{ checkpointWalSize: 0 }, ['checkpointWalSize', 'checkpoint_wal_size'], false],
  ])('refuses %o, naming the setting and its bounds', (options, names, upper) => {
    const path = join(temp(), 'antwerp.grafeo')
    let error = null
    try {
      GrafeoDB.create(path, options)
    } catch (e) {
      error = e
    }
    expect(error).not.toBeNull()
    const message = String(error.message)
    expect(names.some((name) => message.includes(name))).toBe(true)
    expect(message).toContain('1 MiB')
    if (upper) {
      expect(message).toContain('1 GiB')
    }
    expect(existsSync(path)).toBe(false)
  })
})
