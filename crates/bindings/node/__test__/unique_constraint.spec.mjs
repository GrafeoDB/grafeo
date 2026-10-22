// A UNIQUE value written by two concurrent transactions commits once: the
// #400 reporter's insert-if-absent from eight concurrent async transactions,
// and the errors a second writer of a key gets (the retryable write conflict
// GRAFEO-T001 while the first writer's transaction is open, G2000 once it
// committed). The engine's tests of every write path are
// crates/grafeo-engine/tests/unique_constraint_claims.rs.

import { describe, it, expect } from 'vitest'
import { GrafeoDB } from '../grafeo.js'

const CONSTRAINT = 'CREATE CONSTRAINT memory_key FOR (m:Memory) ON (m.key) UNIQUE'

async function keyed() {
  const db = GrafeoDB.create()
  await db.execute(CONSTRAINT)
  return db
}

async function count(db, key) {
  const rows = await db.execute(`MATCH (m:Memory {key: '${key}'}) RETURN count(m) AS c`)
  return Number(rows.toArray()[0].c)
}

async function rejection(promise) {
  try {
    await promise
  } catch (error) {
    return error
  }
  throw new Error('expected the promise to reject')
}

describe('UNIQUE between concurrent transactions', () => {
  it('a second writer gets a conflict, then a violation', async () => {
    const db = await keyed()
    const holder = db.beginTransaction()
    await holder.execute("INSERT (:Memory {key: 'ssi'})")
    const conflict = await rejection(db.execute("INSERT (:Memory {key: 'ssi'})"))
    expect(conflict.message).toContain('GRAFEO-T001')
    expect(await count(db, 'ssi')).toBe(0)
    holder.commit()
    const violation = await rejection(db.execute("INSERT (:Memory {key: 'ssi'})"))
    expect(violation.gqlStatus).toBe('G2000')
    expect(await count(db, 'ssi')).toBe(1)
  })

  it('eight concurrent insert-if-absent transactions commit once', async () => {
    const db = await keyed()
    const insertIfAbsent = async () => {
      const tx = db.beginTransaction()
      try {
        const found = await tx.execute("MATCH (m:Memory {key: 'ssi'}) RETURN m.key AS k")
        if (found.toArray().length > 0) {
          tx.rollback()
          return 'found'
        }
        await tx.execute("INSERT (:Memory {key: 'ssi'})")
        tx.commit()
        return 'committed'
      } catch (error) {
        return error
      }
    }
    const results = await Promise.all(Array.from({ length: 8 }, insertIfAbsent))
    expect(results.filter((result) => result === 'committed')).toHaveLength(1)
    for (const result of results) {
      if (result instanceof Error) {
        expect(result.message.includes('GRAFEO-T001') || result.gqlStatus === 'G2000').toBe(true)
      } else {
        expect(['committed', 'found']).toContain(result)
      }
    }
    expect(await count(db, 'ssi')).toBe(1)
  })
})
