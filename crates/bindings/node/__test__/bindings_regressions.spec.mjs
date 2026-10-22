// User-reported bugs, checked through the Node.js binding:
// - #405: one writer per database file, so a second open fails "locked";
// - #638: a property index survives a close and a reopen, and is used;
// - #245: setSchema of a missing schema throws and keeps the schema;
// - #268 and #335: nodes and edges in results carry their values, an edge
//   its type and stored endpoints, and ORDER BY ... LIMIT returns the node;
// - #302: nodes and edges written after compact() are visible, also after
//   a reopen.

import { mkdtempSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { describe, it, expect } from 'vitest'
import { GrafeoDB } from '../grafeo.js'

const tempPath = (name) => join(mkdtempSync(join(tmpdir(), 'grafeo-node-')), name)

async function rows(db, query) {
  return (await db.execute(query)).toArray()
}

async function names(db) {
  return (await rows(db, 'MATCH (p:Person) RETURN p.name AS n ORDER BY n')).map((row) => row.n)
}

describe('one writer per database file (#405)', () => {
  for (const name of ['people', 'people.grafeo']) {
    it(`refuses a second open while the first is open (${name})`, async () => {
      const path = tempPath(name)
      const first = GrafeoDB.create(path)
      await first.execute("INSERT (:Person {name: 'A'})")
      expect(() => GrafeoDB.open(path)).toThrow(/locked/)
      await first.execute("INSERT (:Person {name: 'B'})")
      first.close()

      const reopened = GrafeoDB.open(path)
      expect(await names(reopened)).toEqual(['A', 'B'])
      reopened.close()
    })
  }
})

describe('a property index after a reopen (#638)', () => {
  it('is there and an equality looks it up', async () => {
    const path = tempPath('indexed.grafeo')
    const db = GrafeoDB.create(path)
    await db.execute("INSERT (:Person {name: 'Alix'}), (:Person {name: 'Gus'})")
    await db.execute('CREATE INDEX person_name FOR (p:Person) ON (p.name)')
    await db.execute("INSERT (:Person {name: 'Vincent'})")
    db.close()

    const reopened = GrafeoDB.open(path)
    const query = "MATCH (p:Person {name: 'Vincent'}) RETURN p.name AS n"
    const plan = Object.values((await rows(reopened, `EXPLAIN ${query}`))[0])[0]
    expect(plan).toContain('[index: name]')
    expect(await rows(reopened, query)).toEqual([{ n: 'Vincent' }])
    reopened.close()
  })
})

describe('setSchema (#245)', () => {
  it('throws for a missing schema and keeps the schema', () => {
    const db = GrafeoDB.create()
    expect(() => db.setSchema('missing')).toThrow("Schema 'missing' does not exist")
    expect(db.currentSchema()).toBeNull()
  })
})

describe('nodes and edges in results (#268, #335)', () => {
  for (const query of [
    "MATCH (a:Person {name: 'Alix'})-[r:KNOWS]->(b) RETURN a, r, b",
    "MATCH (b)<-[r:KNOWS]-(a:Person {name: 'Alix'}) RETURN a, r, b",
  ]) {
    it(`an edge names its type and stored endpoints: ${query}`, async () => {
      const db = GrafeoDB.create()
      await db.execute("INSERT (:Person {name: 'Alix'})-[:KNOWS {since: 2020}]->(:Person {name: 'Gus'})")
      const result = await rows(db, query)
      expect(result).toHaveLength(1)
      const { a, r, b } = result[0]
      expect([a.name, a._labels]).toEqual(['Alix', ['Person']])
      expect([b.name, b._labels]).toEqual(['Gus', ['Person']])
      expect([r._type, r.since]).toEqual(['KNOWS', 2020])
      expect([r._source, r._target]).toEqual([a._id, b._id])
    })
  }

  it('ORDER BY ... LIMIT returns the nodes', async () => {
    const db = GrafeoDB.create()
    await db.execute("INSERT (:Article {title: 'A2'}), (:Article {title: 'A1'}), (:Article {title: 'A3'})")
    const result = await rows(db, 'MATCH (n:Article) RETURN n ORDER BY n.title LIMIT 2')
    expect(result.map((row) => [row.n.title, row.n._labels])).toEqual([
      ['A1', ['Article']],
      ['A2', ['Article']],
    ])
  })
})

describe('writes after compact() (#302)', () => {
  it('are visible in memory', async () => {
    const db = GrafeoDB.create()
    await db.execute("INSERT (:Person {name: 'Alix'})")
    db.compact()
    await db.execute("INSERT (:Person {name: 'Gus'})")
    await db.execute("MATCH (a:Person {name: 'Alix'}), (b:Person {name: 'Gus'}) INSERT (a)-[:KNOWS]->(b)")
    expect(await names(db)).toEqual(['Alix', 'Gus'])
    expect(await rows(db, 'MATCH (a)-[:KNOWS]->(b) RETURN a.name AS a, b.name AS b')).toEqual([
      { a: 'Alix', b: 'Gus' },
    ])
  })

  it('survive a reopen', async () => {
    const path = tempPath('compacted.grafeo')
    const db = GrafeoDB.create(path)
    await db.execute("INSERT (:Person {name: 'Alix'})")
    db.compact()
    await db.execute("INSERT (:Person {name: 'Gus'})")
    await db.execute("MATCH (p:Person {name: 'Alix'}) SET p.city = 'Amsterdam'")
    db.close()

    const reopened = GrafeoDB.open(path)
    expect(await rows(reopened, 'MATCH (p:Person) RETURN p.name AS n, p.city AS c ORDER BY n')).toEqual([
      { n: 'Alix', c: 'Amsterdam' },
      { n: 'Gus', c: null },
    ])
    reopened.close()
  })
})
