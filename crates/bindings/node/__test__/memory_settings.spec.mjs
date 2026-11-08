// The memory settings from Node (#751): GrafeoDB.create(path, { memoryLimit,
// queryMemoryLimit }), read back through memoryUsage(). The default limit is
// enforced; a write past the limit throws GRAFEO-S001 naming the memory limit
// and changes nothing; the query limit defaults to the global one and never
// exceeds it.

import { mkdtempSync, rmSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

import { describe, it, expect } from 'vitest'
import { GrafeoDB } from '../grafeo.js'

const MIB = 1024 * 1024
const BATCH = `UNWIND range(1, 1024) AS i INSERT (:Fill {text: '${'x'.repeat(1000)}'})`

async function count(db, label) {
  return (await db.execute(`MATCH (n:${label}) RETURN n`)).toArray().length
}

describe('memory settings', () => {
  it('read back through memoryUsage()', () => {
    const db = GrafeoDB.create(undefined, { memoryLimit: 64 * MIB, queryMemoryLimit: 8 * MIB })
    const usage = db.memoryUsage()
    expect(usage.limitBytes).toBe(64 * MIB)
    expect(usage.queryLimitBytes).toBe(8 * MIB)
  })

  it('have enforced defaults, the query limit equal to the global one', () => {
    const usage = GrafeoDB.create().memoryUsage()
    expect(usage.limitBytes).toBeGreaterThan(0)
    expect(usage.queryLimitBytes).toBe(usage.limitBytes)
  })

  it('never let the query limit exceed the global one', () => {
    const usage = GrafeoDB.create(undefined, { memoryLimit: 32 * MIB, queryMemoryLimit: 256 * MIB }).memoryUsage()
    expect(usage.queryLimitBytes).toBe(32 * MIB)
  })

  it('refuse a write past the limit and change nothing', async () => {
    const db = GrafeoDB.create(undefined, { memoryLimit: 64 * MIB })
    let refusal
    for (let i = 0; i < 10000 && !refusal; i++) {
      try {
        await db.execute(BATCH)
      } catch (error) {
        refusal = error
      }
    }
    expect(refusal, 'the limit bound').toBeInstanceOf(Error)
    expect(refusal.errorCode).toBe('GRAFEO-S001')
    expect(refusal.message).toContain('memory limit')
    const held = await count(db, 'Fill')
    await expect(db.execute(BATCH)).rejects.toThrow()
    expect(await count(db, 'Fill')).toBe(held)
  })
})

describe('spill settings', () => {
  it('read back through memoryUsage().spill, for a database file', () => {
    const dir = mkdtempSync(join(tmpdir(), 'grafeo-spill-'))
    try {
      const zero = GrafeoDB.create(join(dir, 'zero.grafeo'), { spillLimit: 0 })
      expect(zero.memoryUsage().spill.limitBytes).toBe(0)
      zero.close()
      const unlimited = GrafeoDB.create(join(dir, 'unlimited.grafeo'), {
        spillLimit: 16 * MIB,
        unlimitedSpill: true,
      })
      expect(unlimited.memoryUsage().spill.limitBytes).toBeNull()
      unlimited.close()
    } finally {
      rmSync(dir, { recursive: true, force: true })
    }
  })

  it('spill nothing in memory, whatever the settings', () => {
    const db = GrafeoDB.create(undefined, { unlimitedSpill: true })
    expect(db.memoryUsage().spill.limitBytes).toBe(0)
  })
})

describe('open warnings', () => {
  // Distinct 1 KiB strings: data that stays past the limit once reloaded.
  const FILLER = `UNWIND range(1, 20000) AS i INSERT (:Fill {i: i, text: CAST(i AS STRING) || '${'x'.repeat(1000)}'})`

  it('name an open over the limit, which still reads', async () => {
    const dir = mkdtempSync(join(tmpdir(), 'grafeo-memory-'))
    try {
      const path = join(dir, 'over.grafeo')
      const db = GrafeoDB.create(path)
      await db.execute(FILLER)
      db.close()

      const tight = GrafeoDB.create(path, { memoryLimit: MIB })
      expect(tight.openWarnings().map((warning) => warning.code)).toContain('MEMORY_LIMIT_PASSED')
      expect(await count(tight, 'Fill')).toBe(20000)
      tight.close()

      const roomy = GrafeoDB.create(path)
      expect(roomy.openWarnings().map((warning) => warning.code)).not.toContain('MEMORY_LIMIT_PASSED')
      roomy.close()
    } finally {
      rmSync(dir, { recursive: true, force: true })
    }
  })
})
