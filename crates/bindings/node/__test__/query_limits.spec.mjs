// The path search budget from Node (#492): GrafeoDB.create(path, { pathSearchBudget }). Zero is
// refused; the budget binds, with its error code and the option named in the message.

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
  it('refuses a zero pathSearchBudget', () => {
    expect(() => GrafeoDB.create(undefined, { pathSearchBudget: 0 })).toThrow()
  })

  it('binds the path search budget', async () => {
    const error = await refused(await star({ pathSearchBudget: 1 }), PATHS)
    expect(error.errorCode).toBe('GRAFEO-S001')
    expect(error.message).toContain('pathSearchBudget')
  })
})
