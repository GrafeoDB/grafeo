// The query resource limits from Node (#492): GrafeoDB.create(path, { pathSearchBudget,
// pathExpansionLimit, maxQueryLength, maxIdentifierLength }). Zero is refused; each setting
// binds, with its error code and the option named in the message. The snapshot retention limit, { snapshotRetentionLimit }: below 1 MiB is refused; past it an open
// transaction throws GRAFEO-T012.

import { describe, it, expect } from 'vitest'
import { GrafeoDB } from '../grafeo.js'

const STAR =
  "INSERT (c:Star {n: 0}), (c)-[:E]->(:Leaf {n: 1}), (c)-[:E]->(:Leaf {n: 2}), " +
  "(c)-[:E]->(:Leaf {n: 3}), (c)-[:E]->(:Leaf {n: 4}), (c)-[:E]->(:Leaf {n: 5})"
const PATHS = 'MATCH (s:Star)-[:E*1..2]->(t) RETURN t.n AS n'

async function star(options) {
  const db = GrafeoDB.create(undefined, options)
  await db.execute(STAR)
  return db
}

async function refused(db, query) {
  try {
    await db.execute(query)
  } catch (error) {
    return error
  }
  throw new Error(`expected ${query} to be refused`)
}

describe('query limits', () => {
  for (const option of ['pathSearchBudget', 'pathExpansionLimit', 'maxQueryLength', 'maxIdentifierLength']) {
    it(`refuses a zero ${option}`, () => {
      expect(() => GrafeoDB.create(undefined, { [option]: 0 })).toThrow()
    })
  }

  it('binds the path expansion limit', async () => {
    const error = await refused(await star({ pathExpansionLimit: 4 }), PATHS)
    expect(error.errorCode).toBe('GRAFEO-Q009')
    expect(error.message).toContain('pathExpansionLimit')
    expect((await (await star({ pathExpansionLimit: 5 })).execute(PATHS)).toArray()).toHaveLength(5)
  })

  it('binds the path search budget', async () => {
    const error = await refused(await star({ pathSearchBudget: 1 }), PATHS)
    expect(error.errorCode).toBe('GRAFEO-S001')
    expect(error.message).toContain('pathSearchBudget')
  })

  it('binds the query length', async () => {
    const db = await star({ maxQueryLength: 200 })
    expect((await db.execute(PATHS)).toArray()).toHaveLength(5)
    const error = await refused(db, PATHS + ' '.repeat(200))
    expect(error.errorCode).toBe('GRAFEO-Q002')
    expect(error.message).toContain('maxQueryLength')
  })

  it('binds the identifier length', async () => {
    const db = await star({ maxIdentifierLength: 8 })
    expect((await db.execute('MATCH (abcdefgh:Star) RETURN abcdefgh.n')).toArray()).toHaveLength(1)
    const error = await refused(db, 'MATCH (abcdefghi:Star) RETURN 1')
    expect(error.errorCode).toBe('GRAFEO-Q002')
    expect(error.message).toContain('maxIdentifierLength')
  })

  it('refuses a snapshotRetentionLimit below 1 MiB', () => {
    expect(() => GrafeoDB.create(undefined, { snapshotRetentionLimit: 512 * 1024 })).toThrow(
      'at least 1 MiB',
    )
  })

  // Past the limit (20 commits replace 64 KiB each, over 1 MiB) the open transaction that read
  // Alix throws GRAFEO-T012, and rolls back.
  it('binds an open transaction to the snapshot retention limit', async () => {
    const db = GrafeoDB.create(undefined, { snapshotRetentionLimit: 1 << 20 })
    await db.execute("INSERT (:Person {name: 'Alix', bio: 'Amsterdam'})")
    const readBio = "MATCH (p:Person {name: 'Alix'}) RETURN p.bio AS bio"
    const tx = db.beginTransaction()
    await tx.execute(readBio)
    const text = 'Vincent '.repeat(8 * 1024)
    for (let round = 0; round < 20; round++) {
      await db.execute(`MATCH (p:Person {name: 'Alix'}) SET p.bio = '${round}${text}'`)
    }
    const error = await refused(tx, readBio)
    expect(error.errorCode).toBe('GRAFEO-T012')
    tx.rollback()
  })

  it('refuses a zero historyRetentionLimit and opens with one', () => {
    expect(() => GrafeoDB.create(undefined, { historyRetentionLimit: 0 })).toThrow(
      'history_retention_limit',
    )
    GrafeoDB.create(undefined, { historyRetentionLimit: 1 << 20 }).close()
  })
})
