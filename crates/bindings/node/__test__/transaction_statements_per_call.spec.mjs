// A transaction statement through `db.execute` is refused, as the call has no lasting session:
// `START TRANSACTION` used to open a transaction that ended with the call, the next write
// committed at once, and a `ROLLBACK` found nothing to undo.

import { describe, it, expect } from 'vitest'
import { GrafeoDB } from '../grafeo.js'

async function rejection(promise) {
  try {
    await promise
  } catch (error) {
    return error
  }
  throw new Error('expected the promise to reject')
}

describe('transaction statements through execute', () => {
  for (const statement of ['START TRANSACTION', 'START TRANSACTION READ ONLY', 'COMMIT', 'ROLLBACK']) {
    it(`refuses ${statement} and points to beginTransaction`, async () => {
      const db = GrafeoDB.create()
      const error = await rejection(db.execute(statement))
      expect(error.message).toMatch(/begin_?[tT]ransaction/)
      expect(error.gqlStatus.startsWith('25')).toBe(true)
    })
  }

  it('leaves the writes around the refused calls alone', async () => {
    const db = GrafeoDB.create()
    await rejection(db.execute('START TRANSACTION'))
    await db.execute("INSERT (:Person {name: 'Alix'})")
    await rejection(db.execute('ROLLBACK'))
    const result = await db.execute('MATCH (p:Person) RETURN p.name')
    expect(result.toArray()).toHaveLength(1)
  })

  it('keeps beginTransaction working', async () => {
    const db = GrafeoDB.create()
    const rolledBack = db.beginTransaction()
    await rolledBack.execute("INSERT (:Person {name: 'Gus'})")
    rolledBack.rollback()
    expect((await db.execute('MATCH (p:Person) RETURN p.name')).toArray()).toHaveLength(0)
    const committed = db.beginTransaction()
    await committed.execute("INSERT (:Person {name: 'Gus'})")
    committed.commit()
    expect((await db.execute('MATCH (p:Person) RETURN p.name')).toArray()).toHaveLength(1)
  })
})
