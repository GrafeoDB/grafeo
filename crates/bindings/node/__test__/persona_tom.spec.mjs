// Persona: Tom embeds Grafeo in a Node.js service (TypeScript, ESM).
//
// He imports the package as the README shows, moves money between accounts in
// transactions with savepoints, lets concurrent requests race on one row,
// maps every database error to an HTTP answer by its GQLSTATUS, and passes
// request data as parameters. His slips: committing twice, using a finished
// transaction, a savepoint that does not exist, forgetting to commit.

import { execFileSync } from 'node:child_process'
import { describe, it, expect, beforeEach } from 'vitest'
import { GrafeoDB } from '../grafeo.js'

let db

async function balances() {
  const rows = (await db.execute('MATCH (x:Account) RETURN x.id AS id, x.balance AS b ORDER BY id')).toArray()
  return Object.fromEntries(rows.map((row) => [row.id, row.b]))
}

function thrown(call) {
  try {
    call()
  } catch (error) {
    return error
  }
  throw new Error('expected the call to throw')
}

async function rejection(promise) {
  try {
    await promise
  } catch (error) {
    return error
  }
  throw new Error('expected the promise to reject')
}

beforeEach(async () => {
  db = GrafeoDB.create()
  await db.execute("INSERT (:Account {id: 'alix', balance: 100}), (:Account {id: 'gus', balance: 50})")
})

describe('persona Tom: the package', () => {
  it('imports by name in a plain ES module, as the README shows', () => {
    // vitest rewrites imports; a service runs plain Node.
    const url = new URL('../grafeo.js', import.meta.url).href
    const script = `import { GrafeoDB } from ${JSON.stringify(url)}; console.log(typeof GrafeoDB.create)`
    const out = execFileSync(process.execPath, ['--input-type=module', '-e', script], { encoding: 'utf8' })
    expect(out.trim()).toBe('function')
  })
})

describe('persona Tom: transactions', () => {
  it('moves money and keeps a savepoint', async () => {
    const tx = db.beginTransaction()
    await tx.execute("MATCH (x:Account {id: 'alix'}) SET x.balance = x.balance - 30")
    await tx.execute('SAVEPOINT before_credit')
    await tx.execute("MATCH (x:Account {id: 'gus'}) SET x.balance = x.balance + 999")
    await tx.execute('ROLLBACK TO SAVEPOINT before_credit')
    await tx.execute("MATCH (x:Account {id: 'gus'}) SET x.balance = x.balance + 30")
    // Nothing shows outside before the commit.
    expect(await balances()).toEqual({ alix: 100, gus: 50 })
    tx.commit()
    expect(tx.isActive).toBe(false)
    expect(await balances()).toEqual({ alix: 70, gus: 80 })
  })

  it('rolls back on an error in his handler', async () => {
    const tx = db.beginTransaction()
    try {
      await tx.execute("MATCH (x:Account) SET x.balance = 0")
      await tx.execute('MATCH (x:Account RETURN x')
      tx.commit()
    } catch (error) {
      tx.rollback()
      expect(error.gqlStatus).toBe('42001')
    }
    expect(await balances()).toEqual({ alix: 100, gus: 50 })
  })

  it('rolls back a transaction he forgets to commit when it is disposed', async () => {
    const tx = db.beginTransaction()
    await tx.execute("MATCH (x:Account) SET x.balance = -1")
    expect(typeof tx[Symbol.dispose]).toBe('function')
    tx[Symbol.dispose]()
    expect(tx.isActive).toBe(false)
    expect(await balances()).toEqual({ alix: 100, gus: 50 })
    // Its writes no longer block anyone.
    const next = db.beginTransaction()
    await next.execute("MATCH (x:Account {id: 'alix'}) SET x.balance = 1")
    next.commit()
    expect((await balances()).alix).toBe(1)
  })

  it('lets one of many concurrent increments win', async () => {
    const stages = []
    const results = await Promise.allSettled(
      Array.from({ length: 10 }, async () => {
        const tx = db.beginTransaction()
        let stage = 'execute'
        try {
          await tx.execute("MATCH (x:Account {id: 'gus'}) SET x.balance = x.balance + 1")
          stage = 'commit'
          tx.commit()
        } catch (error) {
          stage += ', then rollback'
          if (tx.isActive) tx.rollback()
          throw error
        } finally {
          stages.push(stage)
        }
      }),
    )
    const won = results.filter((r) => r.status === 'fulfilled')
    const lost = results.filter((r) => r.status === 'rejected').map((r) => r.reason)
    expect(won).toHaveLength(1)
    expect((await balances()).gus).toBe(51)
    for (const error of lost) {
      expect(error.message, stages.join('; ')).toContain('Write-write conflict')
      expect(error.gqlStatus).toMatch(/^40/)
    }
  })

  it('retries the losers one at a time until all land', async () => {
    const increment = async () => {
      for (;;) {
        const tx = db.beginTransaction()
        try {
          await tx.execute("MATCH (x:Account {id: 'gus'}) SET x.balance = x.balance + 1")
          tx.commit()
          return
        } catch (error) {
          if (tx.isActive) tx.rollback()
          if (!String(error.gqlStatus).startsWith('40')) throw error
        }
      }
    }
    await Promise.all(Array.from({ length: 10 }, increment))
    expect((await balances()).gus).toBe(60)
  })

  it('a commit that loses a conflict ends the transaction', async () => {
    // The README's pattern: commit in the try, roll back in the catch. The
    // conflict shows at the commit when the other transaction committed
    // first; the catch must see it, not an error of the rollback.
    const first = db.beginTransaction()
    const second = db.beginTransaction()
    await first.execute("MATCH (x:Account {id: 'alix'}) SET x.balance = 1")
    first.commit()
    await second.execute("MATCH (x:Account {id: 'alix'}) SET x.balance = 2")
    let caught
    try {
      second.commit()
    } catch (error) {
      caught = error
      expect(second.isActive).toBe(false)
      second.rollback()
    }
    expect(caught.message).toContain('Write-write conflict')
    expect(caught.gqlStatus).toMatch(/^40/)
    expect((await balances()).alix).toBe(1)
  })

  it('runs parallel reads on one database', async () => {
    const rows = await Promise.all(Array.from({ length: 20 }, (_, i) => db.execute('RETURN $i AS i', { i })))
    expect(rows.map((r) => r.toArray()[0].i)).toEqual(Array.from({ length: 20 }, (_, i) => i))
  })
})

describe('persona Tom: errors mapped to HTTP answers', () => {
  it.each([
    ['MATCH (x:Account RETURN x', '42001', 'Expected RParen'],
    ['MATCH (x:Account) WHERE x.id = $id RETURN x', '42002', 'Missing parameter: $id'],
    ['RETURN 1 / 0 AS x', '22012', 'division by zero'],
  ])('%s carries %s', async (query, status, words) => {
    const error = await rejection(db.execute(query))
    expect(error.gqlStatus).toBe(status)
    expect(error.message).toContain(words)
  })

  it('a missing savepoint is an invalid transaction state', async () => {
    const tx = db.beginTransaction()
    const error = await rejection(tx.execute('ROLLBACK TO SAVEPOINT nope'))
    tx.rollback()
    expect(error.gqlStatus).toBe('25000')
    expect(error.message).toContain("Savepoint 'nope' not found")
  })

  it('committing twice is an invalid transaction state', () => {
    const tx = db.beginTransaction()
    tx.commit()
    const error = thrown(() => tx.commit())
    expect(error.message).toContain('Already committed')
    expect(error.gqlStatus).toBe('25000')
  })

  it('a query on a finished transaction is an invalid transaction state', async () => {
    const tx = db.beginTransaction()
    tx.rollback()
    const error = await rejection(tx.execute('RETURN 1 AS x'))
    expect(error.message).toContain('no longer active')
    expect(error.gqlStatus).toBe('25000')
  })

  it('names the write conflict once', async () => {
    const first = db.beginTransaction()
    const second = db.beginTransaction()
    await first.execute("MATCH (x:Account {id: 'alix'}) SET x.balance = 1")
    const error = await rejection(second.execute("MATCH (x:Account {id: 'alix'}) SET x.balance = 2"))
    first.commit()
    second.rollback()
    expect(error.message.match(/Write conflict/g)).toHaveLength(1)
  })
})

describe('persona Tom: request data as parameters', () => {
  it('takes dates, nested objects and big integers', async () => {
    const row = (
      await db.execute('RETURN $at AS at, $order.items[1].sku AS sku, $big AS big', {
        at: new Date('2026-10-30T10:00:00Z'),
        order: { items: [{ sku: 'a1' }, { sku: 'b2' }] },
        big: 9007199254740993n,
      })
    ).toArray()[0]
    expect(row.at).toEqual(new Date('2026-10-30T10:00:00Z'))
    expect(row.sku).toBe('b2')
    expect(row.big).toBe(9007199254740993n)
  })

  it('reads an undefined field as null', async () => {
    const row = (await db.execute('RETURN $x AS x', { x: undefined })).toArray()[0]
    expect(row.x).toBeNull()
  })

  it.each([
    ['a function', () => 1, 'Function'],
    ['a symbol', Symbol('s'), 'Symbol'],
  ])('refuses %s, naming the parameter', async (_kind, value, type) => {
    const error = await rejection(db.execute('RETURN $x AS x', { x: value }))
    expect(error.message).toContain("parameter 'x'")
    expect(error.message).toContain(type)
  })
})
