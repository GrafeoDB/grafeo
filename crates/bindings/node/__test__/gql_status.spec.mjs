// Every error from the database carries its GQLSTATUS as `err.gqlStatus`
// (#643), for a rejected promise and for a sync throw alike, as the other
// bindings do (#639).

import { createRequire } from 'node:module'
import { describe, it, expect } from 'vitest'
import { GrafeoDB } from '../grafeo.js'

const require = createRequire(import.meta.url)
const { MARKER, wrap } = require('../error-status.js')

async function rejection(promise) {
  try {
    await promise
  } catch (error) {
    return error
  }
  throw new Error('expected the promise to reject')
}

function thrown(call) {
  try {
    call()
  } catch (error) {
    return error
  }
  throw new Error('expected the call to throw')
}

describe('GQLSTATUS of database errors', () => {
  it.each([
    ['RETURN 1 / 0 AS x', '22012', 'division by zero: 1 / 0'],
    ['RETURN 1 % 0 AS x', '22012', 'division by zero: 1 % 0'],
    ['RETURN 9223372036854775807 + 1 AS x', '22003', 'numeric value out of range: 9223372036854775807 + 1'],
  ])('a rejected GQL query %s carries %s', async (query, status, message) => {
    const db = GrafeoDB.create()
    const error = await rejection(db.execute(query))
    expect(error).toBeInstanceOf(Error)
    expect(error.gqlStatus).toBe(status)
    expect(error.message).toContain('GRAFEO-Q006')
    expect(error.message).toContain(message)
    expect(error.cause).toBeUndefined()
  })

  it('a syntax error carries 42001', async () => {
    const db = GrafeoDB.create()
    const error = await rejection(db.execute('MATCH (n RETURN n'))
    expect(error.gqlStatus).toBe('42001')
  })

  it('Cypher, a transaction and a graph handle carry it too', async () => {
    const db = GrafeoDB.create()
    expect((await rejection(db.executeCypher('RETURN 10 / 0 AS x'))).gqlStatus).toBe('22012')

    const tx = db.beginTransaction()
    expect((await rejection(tx.execute('RETURN 9223372036854775807 + 1 AS x'))).gqlStatus).toBe('22003')
    tx.rollback()

    await db.execute('CREATE GRAPH berlin')
    const error = await rejection(db.graph('berlin').execute('MATCH (n RETURN n'))
    expect(error.gqlStatus).toBe('42001')
  })

  it('a sync call that fails in the database carries its status', () => {
    const db = GrafeoDB.create()
    const error = thrown(() => db.setNodeProperty(999999, 'key', 'value'))
    expect(error).toBeInstanceOf(Error)
    expect(error.message).toContain('GRAFEO-V002')
    expect(error.gqlStatus).toBe('02000')
    expect(error.cause).toBeUndefined()
  })

  it('an error the binding raises before the database has none', () => {
    const db = GrafeoDB.create()
    const error = thrown(() => db.setNodeProperty(-1, 'key', 'value'))
    expect(error.message).toContain('Invalid node ID')
    expect(error.gqlStatus).toBeUndefined()
  })

  it('a query that succeeds is unchanged', async () => {
    const db = GrafeoDB.create()
    const result = await db.execute('RETURN 1 + 1 AS x')
    expect(result.toArray()).toEqual([{ x: 2 }])
  })
})

describe('the status wrapper', () => {
  it('keeps a cause it did not set', async () => {
    const reason = new Error('from the caller')
    const failing = wrap(() => Promise.reject(new Error('outer', { cause: reason })))
    const error = await rejection(failing())
    expect(error.cause).toBe(reason)
    expect(error.gqlStatus).toBeUndefined()

    const throwing = wrap(() => {
      throw new Error('outer', { cause: reason })
    })
    expect(thrown(throwing).cause).toBe(reason)
  })

  it('reads only its own marker', () => {
    const throwing = wrap(() => {
      throw new Error('outer', { cause: new Error(`${MARKER}22012`) })
    })
    const error = thrown(throwing)
    expect(error.gqlStatus).toBe('22012')
    expect(error.cause).toBeUndefined()
  })

  it('passes a non-Error throw and a result through', () => {
    expect(thrown(wrap(() => {
      throw 'a string'
    }))).toBe('a string')
    expect(wrap((a, b) => a + b)(2, 3)).toBe(5)
  })
})
