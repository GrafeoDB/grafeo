// A session without external access reads and writes nothing outside the
// database. `db.session({ readOnly, externalAccess })` gives a Session; with
// `externalAccess: false` every statement on it that reaches outside (LOAD
// DATA in each form, Cypher LOAD CSV, the load procedures, SPARQL LOAD) is
// refused at planning, before any file is opened, EXPLAIN included: the
// permission-denied error (GQLSTATUS class 42) naming external access. The
// bound holds for every statement on the session: auto-commit, its
// transactions, every language method and the stream. readOnly and
// externalAccess are independent, the defaults are today's `db.execute`, and
// the application's own calls (`db.importCsv`) are unaffected.
//
// The oracle: the same statement on a default session answers as `db.execute`
// does today, and a refused one changes nothing. A file that does not exist
// tells the refusal comes before the open.

import { mkdtempSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { describe, it, expect } from 'vitest'
import { GrafeoDB } from '../grafeo.js'

function files() {
  const dir = mkdtempSync(join(tmpdir(), 'grafeo-external-'))
  const path = (name, text) => {
    const file = join(dir, name)
    if (text !== undefined) writeFileSync(file, text, 'utf8')
    return file.replaceAll('\\', '/')
  }
  return {
    csv: path('people.csv', 'id,name\n1,Alix\n2,Gus\n'),
    bareCsv: path('bare.csv', '1,Alix\n2,Gus\n'),
    jsonl: path('people.jsonl', '{"id": 1, "name": "Alix"}\n{"id": 2, "name": "Gus"}\n'),
    turtle: path('people.ttl', '<urn:alix> <urn:knows> <urn:gus> .\n'),
    missing: path('nobody-wrote-this.csv'),
  }
}

const fileUrl = (path) => 'file:///' + path.replace(/^\/+/, '')
const loadData = (path) => `LOAD DATA FROM '${path}' FORMAT CSV WITH HEADERS AS row RETURN row.name AS name`
const loadProcedure = (path) => `CALL grafeo.load_nodes_csv({path: '${path}', labels: ['Person'], key: 'id'})`

// Each form: its name, the method, the query, and whether a default session
// answers it. SPARQL LOAD is refused as not supported yet even with access.
const FORMS = [
  ['GQL LOAD DATA', 'execute', (f) => loadData(f.csv), true],
  ['GQL LOAD DATA without headers', 'execute', (f) => `LOAD DATA FROM '${f.bareCsv}' FORMAT CSV AS row RETURN row[1] AS name`, true],
  ['GQL LOAD DATA from a file URL', 'execute', (f) => `LOAD DATA FROM '${fileUrl(f.csv)}' FORMAT CSV WITH HEADERS AS row RETURN row.name AS name`, true],
  ['GQL LOAD DATA of a missing file', 'execute', (f) => loadData(f.missing), false],
  ['GQL LOAD DATA JSONL', 'execute', (f) => `LOAD DATA FROM '${f.jsonl}' FORMAT JSONL AS row RETURN row.name AS name`, true],
  ['GQL load_nodes_csv', 'execute', (f) => loadProcedure(f.csv), true],
  ['GQL load_nodes_csv then MATCH', 'execute', (f) => `${loadProcedure(f.csv)} YIELD rows RETURN rows NEXT MATCH (p:Person) RETURN count(p) AS people`, true],
  // With access the next check is the arguments', so this one only shows the access check comes first.
  ['GQL EXPLAIN load_nodes_csv without arguments', 'execute', () => 'EXPLAIN CALL grafeo.load_nodes_csv()', false],
  ['GQL EXPLAIN LOAD DATA', 'execute', (f) => 'EXPLAIN ' + loadData(f.csv), true],
  ['GQL PROFILE LOAD DATA', 'execute', (f) => 'PROFILE ' + loadData(f.csv), true],
  ['GQL through the stream', 'executeStream', (f) => loadData(f.csv), true],
  ['GQL through executeLanguage', 'executeLanguage:gql', (f) => loadData(f.csv), true],
  ['Cypher LOAD CSV', 'executeCypher', (f) => `LOAD CSV WITH HEADERS FROM '${fileUrl(f.csv)}' AS row RETURN row.name AS name`, true],
  ['Cypher LOAD CSV from a bare path', 'executeCypher', (f) => `LOAD CSV FROM '${f.bareCsv}' AS row RETURN row[1] AS name`, true],
  ['Cypher load_nodes_csv then MATCH', 'executeCypher', (f) => `${loadProcedure(f.csv)} YIELD rows MATCH (p:Person) RETURN p.name AS name`, true],
  ['SQL/PGQ load_nodes_csv', 'executeSql', (f) => loadProcedure(f.csv), true],
  ['SPARQL LOAD', 'executeSparql', (f) => `LOAD <${fileUrl(f.turtle)}>`, false],
  ['SPARQL LOAD INTO GRAPH', 'executeSparql', (f) => `LOAD <${fileUrl(f.turtle)}> INTO GRAPH <urn:people>`, false],
]

// Runs `query` through `method` of a session, transaction or database and
// reads every row, so a stream runs too.
async function call(target, method, query) {
  if (method.startsWith('executeLanguage:')) {
    return (await target.executeLanguage(method.split(':')[1], query)).toArray()
  }
  if (method === 'executeStream') {
    const stream = await target.executeStream(query)
    const rows = []
    let row
    while ((row = await stream.next()) !== null) rows.push(row)
    return rows
  }
  return (await target[method](query)).toArray()
}

// The error `run` throws or rejects with; fails the test when it succeeds.
async function failure(run) {
  try {
    await run()
  } catch (error) {
    return error
  }
  throw new Error('expected the call to fail')
}

// `run` fails with the external-access refusal, thrown or rejected: class 42,
// naming external access.
async function expectRefused(run) {
  const error = await failure(run)
  expect(error, 'expected the external-access refusal').toBeInstanceOf(Error)
  expect(error.message).toContain('external access')
  expect(error.gqlStatus).toMatch(/^42/)
}

async function people(db) {
  return (await db.execute('MATCH (p:Person) RETURN p')).toArray().length
}

describe('a session without external access', () => {
  it.each(FORMS)('refuses %s', async (_name, method, query) => {
    const f = files()
    const db = GrafeoDB.create()
    const session = db.session({ externalAccess: false })
    await expectRefused(() => call(session, method, query(f)))
    expect(await people(db)).toBe(0)
  })

  it.each(FORMS.filter(([, , , runs]) => runs))('runs %s on a default session', async (_name, method, query) => {
    const f = files()
    // A database per statement, so one load does not meet another's nodes.
    await call(GrafeoDB.create().session(), method, query(f))
    await call(GrafeoDB.create().session({ externalAccess: true }), method, query(f))
    // A load procedure also writes: a read-only session refuses it as a write, which the order
    // of refusals test pins.
    if (!query(f).startsWith('CALL')) await call(GrafeoDB.create().session({ readOnly: true }), method, query(f))
    await call(GrafeoDB.create(), method, query(f))
  })

  it('fails differently on a missing file with access', async () => {
    const f = files()
    const db = GrafeoDB.create()
    const error = await failure(() => db.session().execute(loadData(f.missing)))
    expect(error).toBeInstanceOf(Error)
    expect(error.message).not.toContain('external access')
    expect(error.gqlStatus).not.toMatch(/^42/)
  })
})

describe('the session options', () => {
  it.each([
    ['defaults', undefined, true, true],
    ['explicit defaults', { readOnly: false, externalAccess: true }, true, true],
    ['read-only', { readOnly: true }, false, true],
    ['no external access', { externalAccess: false }, true, false],
    ['both', { readOnly: true, externalAccess: false }, false, false],
  ])('%s', async (_name, options, writes, loads) => {
    const f = files()
    const db = GrafeoDB.create()
    const session = options === undefined ? db.session() : db.session(options)
    const write = "INSERT (:Person {name: 'Mia'})"
    if (writes) {
      await session.execute(write)
      expect(await people(db)).toBe(1)
    } else {
      const error = await failure(() => session.execute(write))
      expect(error).toBeInstanceOf(Error)
      expect(error.message).not.toContain('external access')
      expect(await people(db)).toBe(0)
    }
    if (loads) {
      expect((await session.execute(loadData(f.csv))).toArray().map((row) => row.name)).toEqual(['Alix', 'Gus'])
    } else {
      await expectRefused(() => session.execute(loadData(f.csv)))
    }
  })

  it.each([
    ['external', { external: false }],
    ['external_access', { external_access: false }],
    ['read_only', { read_only: true }],
    ['a misspelt key next to a right one', { externalAccess: false, readonly: true }],
  ])('refuses the unknown option %s', (_name, options) => {
    expect(() => GrafeoDB.create().session(options)).toThrow(/external|read_only|readonly|option/i)
  })
})

describe('every statement on the session is bound', () => {
  it('in its transactions', async () => {
    const f = files()
    const db = GrafeoDB.create()
    const session = db.session({ externalAccess: false })
    const tx = session.beginTransaction()
    await expectRefused(() => tx.execute(loadData(f.csv)))
    await expectRefused(() => tx.execute(loadProcedure(f.csv)))
    await expectRefused(() =>
      tx.executeCypher(`LOAD CSV WITH HEADERS FROM '${fileUrl(f.csv)}' AS row RETURN row.name AS name`),
    )
    await tx.execute("INSERT (:Person {name: 'Mia'})")
    tx.rollback()
    await expectRefused(() => session.execute(loadData(f.csv)))
    const readCommitted = session.beginTransaction('read_committed')
    await expectRefused(() => readCommitted.execute(loadData(f.csv)))
    readCommitted.rollback()
    expect(await people(db)).toBe(0)
  })

  it('for a plan another session cached', async () => {
    const f = files()
    const db = GrafeoDB.create()
    const query = "CALL grafeo.load_nodes_csv({path: $path, labels: ['Person'], key: 'id'})"
    await db.execute(query, { path: f.csv })
    expect(await people(db)).toBe(2)
    const restricted = db.session({ externalAccess: false })
    for (let i = 0; i < 2; i++) await expectRefused(() => restricted.execute(query, { path: f.csv }))
    expect(await people(db)).toBe(2)
  })

  it('composed after a write, so nothing runs', async () => {
    const f = files()
    const db = GrafeoDB.create()
    await expectRefused(() =>
      db.session({ externalAccess: false }).execute(`INSERT (:Marker) RETURN 1 AS one NEXT ${loadData(f.csv)}`),
    )
    expect((await db.execute('MATCH (m:Marker) RETURN m')).toArray()).toHaveLength(0)
  })

  it('after SESSION RESET', async () => {
    const f = files()
    const session = GrafeoDB.create().session({ externalAccess: false })
    await session.execute('SESSION RESET')
    await expectRefused(() => session.execute(loadData(f.csv)))
  })

  it.each([
    ['execute', 'MATCH (n) RETURN count(n) AS n'],
    ['executeCypher', 'MATCH (n) RETURN count(n) AS n'],
    ['executeSql', 'SELECT * FROM GRAPH_TABLE (MATCH (n) COLUMNS (n.name AS name))'],
    ['executeGremlin', 'g.V().count()'],
    ['executeGraphql', '{ Person { name } }'],
    ['executeSparql', 'SELECT ?s WHERE { ?s ?p ?o }'],
    ['executeStream', 'MATCH (n:Person) RETURN n.name'],
    ['executeLanguage:gql', 'CALL grafeo.procedures() YIELD name RETURN name'],
  ])('and %s still runs what stays inside', async (method, query) => {
    const db = GrafeoDB.create()
    await db.execute("INSERT (:Person {name: 'Alix'})")
    await call(db.session({ externalAccess: false }), method, query)
  })
})

describe('outside the session', () => {
  it("leaves the application's own calls alone", async () => {
    const f = files()
    const db = GrafeoDB.create()
    const restricted = db.session({ externalAccess: false })
    expect(await db.importCsv(f.csv, { label: 'Person' })).toBe(2)
    expect((await db.execute(loadData(f.csv))).toArray().map((row) => row.name)).toEqual(['Alix', 'Gus'])
    await expectRefused(() => restricted.execute(loadData(f.csv)))
  })

  it('refuses statements on a closed session, and others go on', async () => {
    const f = files()
    const db = GrafeoDB.create()
    const closed = db.session()
    const other = db.session({ externalAccess: false })
    closed.close()
    expect(await failure(() => closed.execute('RETURN 1 AS one'))).toBeInstanceOf(Error)
    expect(await failure(() => closed.beginTransaction())).toBeInstanceOf(Error)
    expect((await other.execute('RETURN 1 AS one')).toArray()).toEqual([{ one: 1 }])
    await expectRefused(() => other.execute(loadData(f.csv)))
    expect((await db.execute('RETURN 1 AS one')).toArray()).toEqual([{ one: 1 }])
  })
})

describe('the order of refusals', () => {
  // A load procedure reaches outside and writes: a read-only session without
  // access is told about external access; once access is granted, the write is
  // what it refuses.
  it('names external access before the write', async () => {
    const f = files()
    const db = GrafeoDB.create()
    await expectRefused(() => db.session({ readOnly: true, externalAccess: false }).execute(loadProcedure(f.csv)))
    const error = await failure(() => db.session({ readOnly: true }).execute(loadProcedure(f.csv)))
    expect(error).toBeInstanceOf(Error)
    expect(error.message).not.toContain('external access')
    expect(await people(db)).toBe(0)
  })
})
