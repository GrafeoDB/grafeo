import { describe, it, expect, beforeEach, afterEach } from 'vitest'
import { GrafeoDB, version, simdSupport } from '../index.js'

// ── Helpers ──────────────────────────────────────────────────────────

/** Create a fresh in-memory database with some seed data. */
function seedDb() {
  const db = GrafeoDB.create()
  // People
  const alix = db.createNode(['Person'], { name: 'Alix', age: 30 })
  const gus = db.createNode(['Person'], { name: 'Gus', age: 25 })
  const vincent = db.createNode(['Person'], { name: 'Vincent', age: 35 })
  // Company
  const acme = db.createNode(['Company'], { name: 'Acme Corp', founded: 2010 })
  // Relationships
  const knows1 = db.createEdge(alix.id, gus.id, 'KNOWS', { since: 2020 })
  const knows2 = db.createEdge(gus.id, vincent.id, 'KNOWS', { since: 2021 })
  const worksAt = db.createEdge(alix.id, acme.id, 'WORKS_AT', { role: 'Engineer' })
  return { db, alix, gus, vincent, acme, knows1, knows2, worksAt }
}

// ── Module-level exports ─────────────────────────────────────────────

describe('module exports', () => {
  it('should export version()', () => {
    expect(version()).toMatch(/^\d+\.\d+\.\d+$/)
  })

  it('should export simdSupport()', () => {
    const simd = simdSupport()
    expect(typeof simd).toBe('string')
    expect(simd.length).toBeGreaterThan(0)
  })
})

// ── Database lifecycle ───────────────────────────────────────────────

describe('database lifecycle', () => {
  it('should create in-memory database', () => {
    const db = GrafeoDB.create()
    expect(db.nodeCount()).toBe(0)
    expect(db.edgeCount()).toBe(0)
    db.close()
  })

  it('should create persistent database', async () => {
    const fs = await import('fs')
    const os = await import('os')
    const path = await import('path')
    const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'grafeo-test-'))
    const dbPath = path.join(dir, 'test.db')

    const db = GrafeoDB.create(dbPath)
    db.createNode(['Test'], { val: 42 })
    expect(db.nodeCount()).toBe(1)
    db.close()

    // Reopen
    const db2 = GrafeoDB.open(dbPath)
    expect(db2.nodeCount()).toBe(1)
    db2.close()

    // Cleanup — best-effort; Windows may hold WAL file locks briefly after close
    try { fs.rmSync(dir, { recursive: true, force: true }) } catch { /* ignore */ }
  })

  it('should close without error', () => {
    const db = GrafeoDB.create()
    expect(() => db.close()).not.toThrow()
  })
})

// ── Node CRUD ────────────────────────────────────────────────────────

describe('node CRUD', () => {
  let db

  beforeEach(() => {
    db = GrafeoDB.create()
  })

  afterEach(() => {
    db.close()
  })

  it('should create a node with labels', () => {
    const node = db.createNode(['Person'])
    expect(node.id).toBeGreaterThanOrEqual(0)
    expect(node.labels).toEqual(['Person'])
    expect(db.nodeCount()).toBe(1)
  })

  it('should create a node with multiple labels', () => {
    const node = db.createNode(['Person', 'Employee'])
    expect(node.labels).toContain('Person')
    expect(node.labels).toContain('Employee')
  })

  it('should create a node with properties', () => {
    const node = db.createNode(['Person'], { name: 'Alix', age: 30 })
    expect(node.get('name')).toBe('Alix')
    expect(node.get('age')).toBe(30)
  })

  it('should get a node by ID', () => {
    const created = db.createNode(['Person'], { name: 'Alix' })
    const fetched = db.getNode(created.id)
    expect(fetched).not.toBeNull()
    expect(fetched.id).toBe(created.id)
    expect(fetched.get('name')).toBe('Alix')
  })

  it('should return null for nonexistent node', () => {
    expect(db.getNode(99999)).toBeNull()
  })

  it('should delete a node', () => {
    const node = db.createNode(['Person'])
    expect(db.deleteNode(node.id)).toBe(true)
    expect(db.getNode(node.id)).toBeNull()
    expect(db.nodeCount()).toBe(0)
  })

  it('should return false when deleting nonexistent node', () => {
    expect(db.deleteNode(99999)).toBe(false)
  })

  it('should hasLabel work correctly', () => {
    const node = db.createNode(['Person', 'Employee'])
    expect(node.hasLabel('Person')).toBe(true)
    expect(node.hasLabel('Company')).toBe(false)
  })

  it('should toString produce readable output', () => {
    const node = db.createNode(['Person'])
    const str = node.toString()
    expect(str).toContain('Person')
  })
})

// ── Edge CRUD ────────────────────────────────────────────────────────

describe('edge CRUD', () => {
  let db, alix, gus

  beforeEach(() => {
    db = GrafeoDB.create()
    alix = db.createNode(['Person'], { name: 'Alix' })
    gus = db.createNode(['Person'], { name: 'Gus' })
  })

  afterEach(() => {
    db.close()
  })

  it('should create an edge', () => {
    const edge = db.createEdge(alix.id, gus.id, 'KNOWS')
    expect(edge.id).toBeGreaterThanOrEqual(0)
    expect(edge.edgeType).toBe('KNOWS')
    expect(edge.sourceId).toBe(alix.id)
    expect(edge.targetId).toBe(gus.id)
    expect(db.edgeCount()).toBe(1)
  })

  it('should create an edge with properties', () => {
    const edge = db.createEdge(alix.id, gus.id, 'KNOWS', { since: 2020 })
    expect(edge.get('since')).toBe(2020)
  })

  it('should get an edge by ID', () => {
    const created = db.createEdge(alix.id, gus.id, 'KNOWS', { weight: 0.5 })
    const fetched = db.getEdge(created.id)
    expect(fetched).not.toBeNull()
    expect(fetched.edgeType).toBe('KNOWS')
    expect(fetched.get('weight')).toBeCloseTo(0.5)
  })

  it('should return null for nonexistent edge', () => {
    expect(db.getEdge(99999)).toBeNull()
  })

  it('should delete an edge', () => {
    const edge = db.createEdge(alix.id, gus.id, 'KNOWS')
    expect(db.deleteEdge(edge.id)).toBe(true)
    expect(db.getEdge(edge.id)).toBeNull()
    expect(db.edgeCount()).toBe(0)
  })

  it('should toString produce readable output', () => {
    const edge = db.createEdge(alix.id, gus.id, 'KNOWS')
    const str = edge.toString()
    expect(str).toContain('KNOWS')
  })
})

// ── Properties ───────────────────────────────────────────────────────

describe('properties', () => {
  let db

  beforeEach(() => {
    db = GrafeoDB.create()
  })

  afterEach(() => {
    db.close()
  })

  it('should set and get node property', () => {
    const node = db.createNode(['Person'])
    db.setNodeProperty(node.id, 'name', 'Alix')
    const updated = db.getNode(node.id)
    expect(updated.get('name')).toBe('Alix')
  })

  it('should overwrite node property', () => {
    const node = db.createNode(['Person'], { name: 'Alix' })
    db.setNodeProperty(node.id, 'name', 'Gus')
    const updated = db.getNode(node.id)
    expect(updated.get('name')).toBe('Gus')
  })

  it('should set and get edge property', () => {
    const a = db.createNode(['A'])
    const b = db.createNode(['B'])
    const edge = db.createEdge(a.id, b.id, 'REL')
    db.setEdgeProperty(edge.id, 'weight', 3.14)
    const updated = db.getEdge(edge.id)
    expect(updated.get('weight')).toBeCloseTo(3.14)
  })

  it('should handle multiple property types', () => {
    const node = db.createNode(['Test'], {
      str: 'hello',
      int: 42,
      float: 3.14,
      bool: true,
      nil: null,
    })
    expect(node.get('str')).toBe('hello')
    expect(node.get('int')).toBe(42)
    expect(node.get('float')).toBeCloseTo(3.14)
    expect(node.get('bool')).toBe(true)
    expect(node.get('nil')).toBeNull()
  })

  it('should return undefined for missing property', () => {
    const node = db.createNode(['Person'])
    expect(node.get('nonexistent')).toBeUndefined()
  })

  it('should return all properties as object', () => {
    const node = db.createNode(['Person'], { name: 'Alix', age: 30 })
    const props = node.properties()
    expect(props.name).toBe('Alix')
    expect(props.age).toBe(30)
  })
})

// ── GQL Queries ──────────────────────────────────────────────────────

describe('GQL queries', () => {
  it('should execute INSERT and MATCH', async () => {
    const db = GrafeoDB.create()
    await db.execute("INSERT (:Person {name: 'Alix', age: 30})")
    await db.execute("INSERT (:Person {name: 'Gus', age: 25})")
    const result = await db.execute('MATCH (p:Person) RETURN p.name, p.age')

    expect(result.length).toBe(2)
    expect(result.columns.length).toBe(2)

    const rows = result.toArray()
    const names = rows.map((r) => r[result.columns[0]])
    expect(names).toContain('Alix')
    expect(names).toContain('Gus')
    db.close()
  })

  it('should execute with parameters', async () => {
    const db = GrafeoDB.create()
    await db.execute("INSERT (:Person {name: 'Alix', age: 30})")
    await db.execute("INSERT (:Person {name: 'Gus', age: 25})")
    const result = await db.execute(
      'MATCH (p:Person) WHERE p.age > $minAge RETURN p.name',
      { minAge: 28 }
    )

    expect(result.length).toBe(1)
    const name = result.scalar()
    expect(name).toBe('Alix')
    db.close()
  })

  it('should return scalar value', async () => {
    const db = GrafeoDB.create()
    await db.execute("INSERT (:Person {name: 'Alix'})")
    const result = await db.execute('MATCH (p:Person) RETURN p.name')
    expect(result.scalar()).toBe('Alix')
    db.close()
  })

  it('should return execution time', async () => {
    const db = GrafeoDB.create()
    const result = await db.execute('MATCH (n) RETURN n')
    expect(result.executionTimeMs).not.toBeNull()
    expect(result.executionTimeMs).toBeGreaterThanOrEqual(0)
    db.close()
  })

  it('should match relationships', async () => {
    const { db } = seedDb()
    const result = await db.execute(
      "MATCH (a:Person)-[:KNOWS]->(b:Person) WHERE a.name = 'Alix' RETURN b.name"
    )
    expect(result.length).toBe(1)
    expect(result.scalar()).toBe('Gus')
    db.close()
  })

  it('should return rows as arrays', async () => {
    const db = GrafeoDB.create()
    await db.execute("INSERT (:Person {name: 'Alix'})")
    const result = await db.execute('MATCH (p:Person) RETURN p.name')
    const rows = result.rows()
    expect(rows.length).toBe(1)
    expect(rows[0][0]).toBe('Alix')
    db.close()
  })

  it('should get row by index', async () => {
    const db = GrafeoDB.create()
    await db.execute("INSERT (:Person {name: 'Alix'})")
    const result = await db.execute('MATCH (p:Person) RETURN p.name')
    const row = result.get(0)
    expect(Object.values(row)).toContain('Alix')
    db.close()
  })

  it('should throw on invalid query', async () => {
    const db = GrafeoDB.create()
    await expect(db.execute('THIS IS NOT VALID')).rejects.toThrow()
    db.close()
  })
})

// ── Previously undeclared methods ───────────────────────────────────

describe('previously undeclared methods', () => {
  it('should call clearPlanCache without error', () => {
    const db = GrafeoDB.create()
    expect(() => db.clearPlanCache()).not.toThrow()
    db.close()
  })

  it('should set and get schema', async () => {
    const db = GrafeoDB.create()
    await db.execute('CREATE SCHEMA test')
    db.setSchema('test')
    expect(db.currentSchema()).toBe('test')
    db.resetSchema()
    expect(db.currentSchema()).toBeNull()
    db.close()
  })

  it('should call toString on QueryResult', async () => {
    const db = GrafeoDB.create()
    await db.execute("INSERT (:Person {name: 'Alix'})")
    const result = await db.execute('MATCH (p:Person) RETURN p.name')
    const str = result.toString()
    expect(typeof str).toBe('string')
    expect(str.length).toBeGreaterThan(0)
    db.close()
  })
})

// ── Transactions ─────────────────────────────────────────────────────

describe('transactions', () => {
  it('should commit transaction', async () => {
    const db = GrafeoDB.create()
    const tx = db.beginTransaction()
    expect(tx.isActive).toBe(true)

    await tx.execute("INSERT (:Person {name: 'Alix'})")
    tx.commit()

    expect(tx.isActive).toBe(false)
    expect(db.nodeCount()).toBe(1)
    db.close()
  })

  it('should rollback transaction', async () => {
    const db = GrafeoDB.create()
    const tx = db.beginTransaction()
    await tx.execute("INSERT (:Person {name: 'Alix'})")
    tx.rollback()

    expect(tx.isActive).toBe(false)
    expect(db.nodeCount()).toBe(0)
    db.close()
  })

  it('should error on double commit', async () => {
    const db = GrafeoDB.create()
    const tx = db.beginTransaction()
    await tx.execute("INSERT (:Person {name: 'Alix'})")
    tx.commit()
    expect(() => tx.commit()).toThrow(/Already committed/)
    db.close()
  })

  it('should error on commit after rollback', async () => {
    const db = GrafeoDB.create()
    const tx = db.beginTransaction()
    tx.rollback()
    expect(() => tx.commit()).toThrow(/Already rolled back/)
    db.close()
  })

  it('should execute multiple operations', async () => {
    const db = GrafeoDB.create()
    const tx = db.beginTransaction()
    await tx.execute("INSERT (:Person {name: 'Alix'})")
    await tx.execute("INSERT (:Person {name: 'Gus'})")
    await tx.execute("INSERT (:Person {name: 'Vincent'})")
    tx.commit()

    expect(db.nodeCount()).toBe(3)
    db.close()
  })

  it('should execute with parameters in transaction', async () => {
    const db = GrafeoDB.create()
    await db.execute("INSERT (:Person {name: 'Alix', age: 30})")
    await db.execute("INSERT (:Person {name: 'Gus', age: 25})")

    const tx = db.beginTransaction()
    const result = await tx.execute(
      'MATCH (p:Person) WHERE p.age > $minAge RETURN p.name',
      { minAge: 28 }
    )
    tx.commit()

    expect(result.length).toBe(1)
    expect(result.scalar()).toBe('Alix')
    db.close()
  })
})

// ── QueryResult metadata & entity extraction ────────────────────────

describe('QueryResult metadata', () => {
  it('should return rowsScanned', async () => {
    const { db } = seedDb()
    const result = await db.execute('MATCH (p:Person) RETURN p.name')
    // rowsScanned may be null or a number depending on the query
    if (result.rowsScanned !== null) {
      expect(typeof result.rowsScanned).toBe('number')
      expect(result.rowsScanned).toBeGreaterThanOrEqual(0)
    }
    db.close()
  })

  it('should extract nodes from MATCH result', async () => {
    const { db } = seedDb()
    const result = await db.execute('MATCH (p:Person) RETURN p')
    const nodes = result.nodes()
    expect(nodes.length).toBe(3)
    const names = nodes.map((n) => n.get('name'))
    expect(names).toContain('Alix')
    expect(names).toContain('Gus')
    expect(names).toContain('Vincent')
    db.close()
  })

  it('should return edges() accessor without error', async () => {
    const { db } = seedDb()
    const result = await db.execute(
      'MATCH (a:Person)-[r:KNOWS]->(b:Person) RETURN a.name, b.name'
    )
    // edges() should be callable even when no edge columns are returned
    const edges = result.edges()
    expect(Array.isArray(edges)).toBe(true)
    db.close()
  })

  it('should deduplicate extracted nodes', async () => {
    const { db } = seedDb()
    // Query that returns same nodes in multiple rows
    const result = await db.execute(
      'MATCH (a:Person)-[:KNOWS]->(b:Person) RETURN a, b'
    )
    const nodes = result.nodes()
    const ids = nodes.map((n) => n.id)
    const uniqueIds = [...new Set(ids)]
    expect(ids.length).toBe(uniqueIds.length)
    db.close()
  })

  it('should return empty nodes/edges for scalar queries', async () => {
    const db = GrafeoDB.create()
    await db.execute("INSERT (:Person {name: 'Alix'})")
    const result = await db.execute('MATCH (p:Person) RETURN p.name')
    expect(result.nodes().length).toBe(0)
    expect(result.edges().length).toBe(0)
    db.close()
  })
})

// ── Advanced type round-trips ───────────────────────────────────────

describe('advanced type round-trips', () => {
  let db

  beforeEach(() => {
    db = GrafeoDB.create()
  })

  afterEach(() => {
    db.close()
  })

  it('should round-trip array/list values', () => {
    const node = db.createNode(['Test'])
    db.setNodeProperty(node.id, 'tags', [1, 'two', true])
    const fetched = db.getNode(node.id)
    const tags = fetched.get('tags')
    expect(Array.isArray(tags)).toBe(true)
    expect(tags).toEqual([1, 'two', true])
  })

  it('should round-trip nested object/map values', () => {
    const node = db.createNode(['Test'])
    db.setNodeProperty(node.id, 'meta', { a: 1, b: 'two' })
    const fetched = db.getNode(node.id)
    const meta = fetched.get('meta')
    expect(typeof meta).toBe('object')
    expect(meta.a).toBe(1)
    expect(meta.b).toBe('two')
  })

  it('should round-trip Date values', () => {
    const node = db.createNode(['Test'])
    const date = new Date('2024-01-15T12:00:00.000Z')
    db.setNodeProperty(node.id, 'created', date)
    const fetched = db.getNode(node.id)
    const result = fetched.get('created')
    expect(result instanceof Date).toBe(true)
    // Millisecond precision
    expect(result.getTime()).toBe(date.getTime())
  })

  it('should round-trip Buffer values', () => {
    const node = db.createNode(['Test'])
    const buf = Buffer.from([1, 2, 3, 4, 5])
    db.setNodeProperty(node.id, 'data', buf)
    const fetched = db.getNode(node.id)
    const result = fetched.get('data')
    expect(Buffer.isBuffer(result)).toBe(true)
    expect([...result]).toEqual([1, 2, 3, 4, 5])
  })

  it('should round-trip Float32Array/vector values', () => {
    const node = db.createNode(['Test'])
    const vec = new Float32Array([1.0, 2.0, 3.0])
    db.setNodeProperty(node.id, 'embedding', vec)
    const fetched = db.getNode(node.id)
    const result = fetched.get('embedding')
    // napi-rs returns the vector data as a buffer; reconstruct Float32Array
    const floats = new Float32Array(
      result.buffer ?? result,
      result.byteOffset ?? 0,
      3
    )
    expect(floats.length).toBe(3)
    expect(floats[0]).toBeCloseTo(1.0)
    expect(floats[1]).toBeCloseTo(2.0)
    expect(floats[2]).toBeCloseTo(3.0)
  })

  it('should round-trip BigInt values', () => {
    const node = db.createNode(['Test'])
    db.setNodeProperty(node.id, 'big', 42n)
    const fetched = db.getNode(node.id)
    // BigInt gets truncated to i64, returned as number if in safe range
    expect(fetched.get('big')).toBe(42)
  })

  it('should handle MAX_SAFE_INTEGER boundary', () => {
    const node = db.createNode(['Test'])
    db.setNodeProperty(node.id, 'big', Number.MAX_SAFE_INTEGER)
    const fetched = db.getNode(node.id)
    expect(fetched.get('big')).toBe(Number.MAX_SAFE_INTEGER)
  })

  it('should return edge properties() as object', () => {
    const a = db.createNode(['A'])
    const b = db.createNode(['B'])
    const edge = db.createEdge(a.id, b.id, 'REL', { w: 1.5, tag: 'x' })
    const props = edge.properties()
    expect(props.w).toBeCloseTo(1.5)
    expect(props.tag).toBe('x')
  })
})

// ── Cypher queries ───────────────────────────────────────────────────

describe('Cypher queries', () => {
  it('should execute Cypher CREATE and MATCH', async () => {
    const db = GrafeoDB.create()
    await db.executeCypher("CREATE (a:Person {name: 'Alix'})")
    const result = await db.executeCypher('MATCH (p:Person) RETURN p.name')
    expect(result.scalar()).toBe('Alix')
    db.close()
  })

  it('should execute Cypher with parameters', async () => {
    const db = GrafeoDB.create()
    await db.executeCypher("CREATE (:Person {name: 'Alix', age: 30})")
    await db.executeCypher("CREATE (:Person {name: 'Gus', age: 25})")
    const result = await db.executeCypher(
      'MATCH (p:Person) WHERE p.age > $min RETURN p.name',
      { min: 28 }
    )
    expect(result.length).toBe(1)
    expect(result.scalar()).toBe('Alix')
    db.close()
  })
})

// ── Gremlin queries ─────────────────────────────────────────────────

describe('Gremlin queries', () => {
  it('should execute basic Gremlin traversal', async () => {
    const db = GrafeoDB.create()
    await db.execute("INSERT (:Person {name: 'Alix'})")
    await db.execute("INSERT (:Person {name: 'Gus'})")
    const result = await db.executeGremlin(
      "g.V().hasLabel('Person').values('name')"
    )
    expect(result.length).toBe(2)
    db.close()
  })
})

// ── SPARQL queries ──────────────────────────────────────────────────

describe('SPARQL queries', () => {
  it('should execute basic SPARQL SELECT', async () => {
    const db = GrafeoDB.create()
    // SPARQL works against the RDF triple store
    const result = await db.executeSparql('SELECT ?x WHERE { ?x ?y ?z }')
    // Empty triple store returns 0 rows
    expect(result.length).toBe(0)
    db.close()
  })
})

// ── Transaction edge cases ──────────────────────────────────────────

describe('transaction edge cases', () => {
  it('should error on execute after commit', async () => {
    const db = GrafeoDB.create()
    const tx = db.beginTransaction()
    await tx.execute("INSERT (:Person {name: 'Alix'})")
    tx.commit()
    await expect(
      tx.execute("INSERT (:Person {name: 'Gus'})")
    ).rejects.toThrow(/no longer active/)
    db.close()
  })

  it('should error on execute after rollback', async () => {
    const db = GrafeoDB.create()
    const tx = db.beginTransaction()
    tx.rollback()
    await expect(
      tx.execute("INSERT (:Person {name: 'Alix'})")
    ).rejects.toThrow(/no longer active/)
    db.close()
  })

  it('should error on double rollback', () => {
    const db = GrafeoDB.create()
    const tx = db.beginTransaction()
    tx.rollback()
    expect(() => tx.rollback()).toThrow(/Already rolled back/)
    db.close()
  })

  it('should error on rollback after commit', async () => {
    const db = GrafeoDB.create()
    const tx = db.beginTransaction()
    await tx.execute("INSERT (:Person {name: 'Alix'})")
    tx.commit()
    expect(() => tx.rollback()).toThrow(/Already committed/)
    db.close()
  })

  it('should never run an in-flight query outside a committed transaction', async () => {
    for (let i = 0; i < 20; i++) {
      const db = GrafeoDB.create()
      const tx = db.beginTransaction()
      const pending = tx.execute("INSERT (:Person {name: 'Vincent'})")
      let commitError = null
      try {
        tx.commit()
      } catch (e) {
        commitError = e
      }
      if (commitError) {
        // The query held the session: commit refused instead of blocking.
        expect(commitError.message).toMatch(/still running/)
        expect(tx.isActive).toBe(true)
        await pending
        tx.commit()
        const r = await db.execute('MATCH (p:Person) RETURN p.name')
        expect(r.length).toBe(1)
      } else {
        // Commit won the race: the queued query must fail, not run auto-committed.
        await expect(pending).rejects.toThrow(/no longer active/)
        const r = await db.execute('MATCH (p:Person) RETURN p.name')
        expect(r.length).toBe(0)
      }
      db.close()
    }
  })

  it('should refuse rollback while a query is running, then allow it', async () => {
    const db = GrafeoDB.create()
    const tx = db.beginTransaction()
    const pending = tx.execute("INSERT (:Person {name: 'Jules'})")
    try {
      tx.rollback()
    } catch (e) {
      expect(e.message).toMatch(/still running/)
      await pending
      tx.rollback()
    }
    await pending.catch(() => {})
    expect(tx.isActive).toBe(false)
    const r = await db.execute('MATCH (p:Person) RETURN p.name')
    expect(r.length).toBe(0)
    db.close()
  })
})

// ── Persistence and checkpoint ──────────────────────────────────────

describe('persistence and checkpoint', () => {
  it('should checkpoint and reopen with data intact', async () => {
    const fs = await import('fs')
    const os = await import('os')
    const path = await import('path')
    const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'grafeo-ckpt-'))
    const dbPath = path.join(dir, 'ckpt.grafeo')

    const db = GrafeoDB.create(dbPath)
    db.createNode(['Person'], { name: 'Alix', age: 30 })
    db.createNode(['Person'], { name: 'Gus', age: 25 })
    db.walCheckpoint()
    db.close()

    const db2 = GrafeoDB.open(dbPath)
    expect(db2.nodeCount()).toBe(2)

    const result = await db2.execute('MATCH (p:Person) RETURN p.name')
    const names = result.toArray().map((r) => r['p.name']).sort()
    expect(names).toEqual(['Alix', 'Gus'])
    db2.close()

    try { fs.rmSync(dir, { recursive: true, force: true }) } catch { /* ignore */ }
  })

  it('should save in-memory database to file', async () => {
    const fs = await import('fs')
    const os = await import('os')
    const path = await import('path')
    const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'grafeo-save-'))
    const dbPath = path.join(dir, 'saved.grafeo')

    const db = GrafeoDB.create()
    db.createNode(['Person'], { name: 'Alix', age: 30 })
    db.createNode(['Person'], { name: 'Gus', age: 25 })
    db.save(dbPath)

    const db2 = GrafeoDB.open(dbPath)
    expect(db2.nodeCount()).toBe(2)

    const result = await db2.execute('MATCH (p:Person) RETURN p.name')
    const names = result.toArray().map((r) => r['p.name']).sort()
    expect(names).toEqual(['Alix', 'Gus'])
    db2.close()

    try { fs.rmSync(dir, { recursive: true, force: true }) } catch { /* ignore */ }
  })

  it('should handle multiple checkpoints', async () => {
    const fs = await import('fs')
    const os = await import('os')
    const path = await import('path')
    const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'grafeo-multi-'))
    const dbPath = path.join(dir, 'multi.grafeo')

    const db = GrafeoDB.create(dbPath)

    // Phase 1
    db.createNode(['Person'], { name: 'Alix' })
    db.walCheckpoint()

    // Phase 2
    db.createNode(['Person'], { name: 'Gus' })
    db.createNode(['Person'], { name: 'Vincent' })
    db.walCheckpoint()

    db.close()

    const db2 = GrafeoDB.open(dbPath)
    expect(db2.nodeCount()).toBe(3)

    const result = await db2.execute('MATCH (p:Person) RETURN p.name')
    const names = result.toArray().map((r) => r['p.name']).sort()
    expect(names).toEqual(['Alix', 'Gus', 'Vincent'])
    db2.close()

    try { fs.rmSync(dir, { recursive: true, force: true }) } catch { /* ignore */ }
  })

  it('should checkpoint empty database without error', async () => {
    const fs = await import('fs')
    const os = await import('os')
    const path = await import('path')
    const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'grafeo-empty-'))
    const dbPath = path.join(dir, 'empty.grafeo')

    const db = GrafeoDB.create(dbPath)
    expect(() => db.walCheckpoint()).not.toThrow()
    db.close()

    try { fs.rmSync(dir, { recursive: true, force: true }) } catch { /* ignore */ }
  })

  it('should create, close, and reopen with data intact', async () => {
    const fs = await import('fs')
    const os = await import('os')
    const path = await import('path')
    const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'grafeo-reopen-'))
    const dbPath = path.join(dir, 'reopen.grafeo')

    const db = GrafeoDB.create(dbPath)
    await db.execute("INSERT (:Person {name: 'Alix', age: 30})")
    await db.execute("INSERT (:Person {name: 'Gus', age: 25})")
    await db.execute(
      "MATCH (a:Person {name: 'Alix'}), (b:Person {name: 'Gus'}) INSERT (a)-[:KNOWS]->(b)"
    )
    expect(db.nodeCount()).toBe(2)
    expect(db.edgeCount()).toBe(1)
    db.close()

    const db2 = GrafeoDB.open(dbPath)
    expect(db2.nodeCount()).toBe(2)
    expect(db2.edgeCount()).toBe(1)

    const result = await db2.execute('MATCH (p:Person) RETURN p.name')
    const names = result.toArray().map((r) => r['p.name']).sort()
    expect(names).toEqual(['Alix', 'Gus'])
    db2.close()

    try { fs.rmSync(dir, { recursive: true, force: true }) } catch { /* ignore */ }
  })

  it('should accumulate data across multiple reopen cycles', async () => {
    const fs = await import('fs')
    const os = await import('os')
    const path = await import('path')
    const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'grafeo-cycles-'))
    const dbPath = path.join(dir, 'cycles.grafeo')

    // Cycle 1
    let db = GrafeoDB.create(dbPath)
    await db.execute("INSERT (:Person {name: 'Alix'})")
    db.close()

    // Cycle 2
    db = GrafeoDB.open(dbPath)
    expect(db.nodeCount()).toBe(1)
    await db.execute("INSERT (:Person {name: 'Gus'})")
    db.close()

    // Cycle 3
    db = GrafeoDB.open(dbPath)
    expect(db.nodeCount()).toBe(2)
    await db.execute("INSERT (:Person {name: 'Vincent'})")
    db.close()

    // Final check
    db = GrafeoDB.open(dbPath)
    expect(db.nodeCount()).toBe(3)
    const result = await db.execute('MATCH (p:Person) RETURN p.name')
    const names = result.toArray().map((r) => r['p.name']).sort()
    expect(names).toEqual(['Alix', 'Gus', 'Vincent'])
    db.close()

    try { fs.rmSync(dir, { recursive: true, force: true }) } catch { /* ignore */ }
  })

  it('should persist edge properties across close/reopen', async () => {
    const fs = await import('fs')
    const os = await import('os')
    const path = await import('path')
    const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'grafeo-edgeprops-'))
    const dbPath = path.join(dir, 'edgeprops.grafeo')

    const db = GrafeoDB.create(dbPath)
    await db.execute("INSERT (:Person {name: 'Alix'})")
    await db.execute("INSERT (:Person {name: 'Gus'})")
    await db.execute(
      "MATCH (a:Person {name: 'Alix'}), (b:Person {name: 'Gus'}) " +
      "INSERT (a)-[:KNOWS {since: 2020}]->(b)"
    )
    db.close()

    const db2 = GrafeoDB.open(dbPath)
    const result = await db2.execute('MATCH ()-[e:KNOWS]->() RETURN e.since')
    const rows = result.toArray()
    expect(rows).toHaveLength(1)
    expect(rows[0]['e.since']).toBe(2020)
    db2.close()

    try { fs.rmSync(dir, { recursive: true, force: true }) } catch { /* ignore */ }
  })

  it('should produce a single .grafeo file, not a directory', async () => {
    const fs = await import('fs')
    const os = await import('os')
    const path = await import('path')
    const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'grafeo-single-'))
    const dbPath = path.join(dir, 'single.grafeo')

    const db = GrafeoDB.create(dbPath)
    await db.execute("INSERT (:Node {x: 1})")
    db.close()

    const stat = fs.statSync(dbPath)
    expect(stat.isFile()).toBe(true)

    try { fs.rmSync(dir, { recursive: true, force: true }) } catch { /* ignore */ }
  })
})

// ── Error handling ───────────────────────────────────────────────────

describe('error handling', () => {
  it('should throw on out-of-range row index', async () => {
    const db = GrafeoDB.create()
    const result = await db.execute('MATCH (n) RETURN n')
    expect(() => result.get(999)).toThrow()
    db.close()
  })

  it('should throw on scalar with no rows', async () => {
    const db = GrafeoDB.create()
    const result = await db.execute('MATCH (n:NonExistent) RETURN n')
    expect(() => result.scalar()).toThrow()
    db.close()
  })

  it('should throw on invalid params type', async () => {
    const db = GrafeoDB.create()
    // Passing a non-object as params
    await expect(
      db.execute('MATCH (n) RETURN n', 'not-an-object')
    ).rejects.toThrow()
    db.close()
  })
})

// ── Counts ──────────────────────────────────────────────────────────

describe('database counts', () => {
  it('should track nodeCount and edgeCount', () => {
    const { db } = seedDb()
    expect(db.nodeCount()).toBe(4) // Alix, Gus, Vincent, Acme
    expect(db.edgeCount()).toBe(3) // knows1, knows2, worksAt
    db.close()
  })
})

// ── Vector operations ───────────────────────────────────────────────

describe('vector operations', () => {
  it('should create vector index and search', async () => {
    const db = GrafeoDB.create()
    await db.batchCreateNodes('Doc', 'embedding', [
      [1, 0, 0],
      [0, 1, 0],
      [0, 0, 1],
    ])

    await db.createVectorIndex('Doc', 'embedding', 3, 'cosine')
    const results = await db.vectorSearch('Doc', 'embedding', [1, 0, 0], 3)

    expect(results.length).toBe(3)
    // Each result is [nodeId, distance]
    expect(results[0].length).toBe(2)
    // Closest should have near-zero distance
    expect(results[0][1]).toBeLessThan(0.01)
    db.close()
  })

  it('should search with explicit ef parameter', async () => {
    const db = GrafeoDB.create()
    await db.batchCreateNodes('Doc', 'embedding', [
      [1, 0, 0],
      [0, 1, 0],
    ])

    await db.createVectorIndex('Doc', 'embedding', 3, 'cosine')
    const results = await db.vectorSearch('Doc', 'embedding', [1, 0, 0], 2, 200)

    expect(results.length).toBe(2)
    db.close()
  })

  it('should create vector index with HNSW tuning params', async () => {
    const db = GrafeoDB.create()
    await db.batchCreateNodes('Doc', 'embedding', [[1, 0, 0]])

    // Pass m and ef_construction
    await db.createVectorIndex('Doc', 'embedding', 3, 'cosine', 32, 200)
    const results = await db.vectorSearch('Doc', 'embedding', [1, 0, 0], 1)
    expect(results.length).toBe(1)
    db.close()
  })

  it('should create vector index with euclidean metric', async () => {
    const db = GrafeoDB.create()
    await db.batchCreateNodes('Doc', 'embedding', [
      [1, 0, 0],
      [0, 1, 0],
    ])

    await db.createVectorIndex('Doc', 'embedding', 3, 'euclidean')
    const results = await db.vectorSearch('Doc', 'embedding', [1, 0, 0], 2)
    expect(results.length).toBe(2)
    // Identical vector should have distance ~0
    expect(results[0][1]).toBeLessThan(0.01)
    db.close()
  })

  it('should batch create nodes with vectors', async () => {
    const db = GrafeoDB.create()
    const vectors = [
      [1, 0, 0],
      [0, 1, 0],
      [0, 0, 1],
    ]
    const ids = await db.batchCreateNodes('Doc', 'embedding', vectors)
    expect(ids.length).toBe(3)
    expect(db.nodeCount()).toBe(3)
    // All unique IDs
    expect(new Set(ids).size).toBe(3)
    db.close()
  })

  it('should batch create empty list', async () => {
    const db = GrafeoDB.create()
    const ids = await db.batchCreateNodes('Doc', 'embedding', [])
    expect(ids.length).toBe(0)
    db.close()
  })

  it('should batch vector search', async () => {
    const db = GrafeoDB.create()
    const vectors = [
      [1, 0, 0],
      [0, 1, 0],
      [0, 0, 1],
    ]
    await db.batchCreateNodes('Doc', 'embedding', vectors)
    await db.createVectorIndex('Doc', 'embedding', 3, 'cosine')

    const queries = [
      [1, 0, 0],
      [0, 1, 0],
    ]
    const results = await db.batchVectorSearch('Doc', 'embedding', queries, 2)
    expect(results.length).toBe(2)
    for (const result of results) {
      expect(result.length).toBe(2)
      // Each result entry is [nodeId, distance]
      expect(result[0].length).toBe(2)
    }
    db.close()
  })

  it('should batch search closest match correctly', async () => {
    const db = GrafeoDB.create()
    await db.batchCreateNodes('Doc', 'embedding', [
      [1, 0, 0],
      [0, 1, 0],
      [0, 0, 1],
    ])
    await db.createVectorIndex('Doc', 'embedding', 3, 'cosine')

    const queries = [
      [1, 0, 0],
      [0, 1, 0],
      [0, 0, 1],
    ]
    const results = await db.batchVectorSearch('Doc', 'embedding', queries, 1)
    expect(results.length).toBe(3)
    for (const result of results) {
      expect(result.length).toBe(1)
      // Each query matches its vector exactly
      expect(result[0][1]).toBeLessThan(0.01)
    }
    db.close()
  })

  it('should batch search with explicit ef', async () => {
    const db = GrafeoDB.create()
    await db.batchCreateNodes('Doc', 'embedding', [
      [1, 0, 0],
      [0, 1, 0],
    ])
    await db.createVectorIndex('Doc', 'embedding', 3, 'cosine')

    const results = await db.batchVectorSearch(
      'Doc',
      'embedding',
      [[1, 0, 0]],
      2,
      200
    )
    expect(results.length).toBe(1)
    expect(results[0].length).toBe(2)
    db.close()
  })

  it('should error on vector search without index', async () => {
    const db = GrafeoDB.create()
    db.createNode(['Doc'], { embedding: new Float32Array([1, 0, 0]) })
    await expect(
      db.vectorSearch('Doc', 'embedding', [1, 0, 0], 1)
    ).rejects.toThrow()
    db.close()
  })
})

// ── GraphQL queries ─────────────────────────────────────────────────

describe('GraphQL queries', () => {
  it('should execute basic GraphQL query', async () => {
    const db = GrafeoDB.create()
    await db.execute("INSERT (:Person {name: 'Alix', age: 30})")
    await db.execute("INSERT (:Person {name: 'Gus', age: 25})")
    const result = await db.executeGraphql('{ Person { name } }')
    expect(result.length).toBeGreaterThanOrEqual(1)
    db.close()
  })
})

// ── Text search ──────────────────────────────────────────────────────

describe('text search', () => {
  it('should create text index and search', async () => {
    const db = GrafeoDB.create()
    db.createNode(['Article'], { title: 'Rust graph database engine' })
    db.createNode(['Article'], { title: 'Python machine learning' })
    db.createNode(['Article'], { title: 'Rust systems programming' })

    await db.createTextIndex('Article', 'title')
    const results = await db.textSearch('Article', 'title', 'Rust', 10)
    expect(results.length).toBeGreaterThanOrEqual(2)
    db.close()
  })

  it('should return empty for no matches', async () => {
    const db = GrafeoDB.create()
    db.createNode(['Article'], { title: 'Rust graph database' })
    await db.createTextIndex('Article', 'title')

    const results = await db.textSearch('Article', 'title', 'nonexistentxyz', 10)
    expect(results.length).toBe(0)
    db.close()
  })

  it('should error without text index', async () => {
    const db = GrafeoDB.create()
    db.createNode(['Article'], { title: 'test' })
    await expect(
      db.textSearch('Article', 'title', 'test', 10)
    ).rejects.toThrow()
    db.close()
  })

  it('should find new nodes after mutation', async () => {
    const db = GrafeoDB.create()
    db.createNode(['Article'], { title: 'Rust graph' })
    await db.createTextIndex('Article', 'title')

    db.createNode(['Article'], { title: 'Rust web framework' })

    const results = await db.textSearch('Article', 'title', 'Rust', 10)
    expect(results.length).toBeGreaterThanOrEqual(2)
    db.close()
  })
})

// ── Hybrid search ────────────────────────────────────────────────────

describe('hybrid search', () => {
  it('should combine text and vector search', async () => {
    const db = GrafeoDB.create()
    db.createNode(['Doc'], {
      content: 'Rust graph database',
      emb: new Float32Array([1, 0, 0]),
    })
    db.createNode(['Doc'], {
      content: 'Python machine learning',
      emb: new Float32Array([0, 1, 0]),
    })
    db.createNode(['Doc'], {
      content: 'Rust systems programming',
      emb: new Float32Array([0.9, 0.1, 0]),
    })

    await db.createTextIndex('Doc', 'content')
    await db.createVectorIndex('Doc', 'emb', 3, 'cosine')

    const results = await db.hybridSearch(
      'Doc', 'content', 'emb', 'Rust graph', 4, [1, 0, 0]
    )
    expect(results.length).toBeGreaterThan(0)
    db.close()
  })

  it('should work with text only (no vector query)', async () => {
    const db = GrafeoDB.create()
    db.createNode(['Doc'], {
      content: 'Rust graph database',
      emb: new Float32Array([1, 0, 0]),
    })
    db.createNode(['Doc'], {
      content: 'Python machine learning',
      emb: new Float32Array([0, 1, 0]),
    })

    await db.createTextIndex('Doc', 'content')
    await db.createVectorIndex('Doc', 'emb', 3, 'cosine')

    const results = await db.hybridSearch(
      'Doc', 'content', 'emb', 'Rust', 4
    )
    expect(results.length).toBeGreaterThan(0)
    db.close()
  })
})

// ── CDC operations ───────────────────────────────────────────────────

describe('CDC operations', () => {
  it('should track node creation history', async () => {
    const db = GrafeoDB.create()
    db.enableCdc()
    const node = db.createNode(['Person'], { name: 'Alix' })

    const history = await db.nodeHistory(node.id)
    expect(history.length).toBeGreaterThanOrEqual(1)
    db.close()
  })

  it('should track node update history', async () => {
    const db = GrafeoDB.create()
    db.enableCdc()
    const node = db.createNode(['Person'], { name: 'Alix' })
    db.setNodeProperty(node.id, 'age', 30)

    const history = await db.nodeHistory(node.id)
    expect(history.length).toBeGreaterThanOrEqual(2)
    db.close()
  })

  it('should track edge creation history', async () => {
    const db = GrafeoDB.create()
    db.enableCdc()
    const a = db.createNode(['N'])
    const b = db.createNode(['N'])
    const edge = db.createEdge(a.id, b.id, 'R')

    const history = await db.edgeHistory(edge.id)
    expect(history.length).toBeGreaterThanOrEqual(1)
    db.close()
  })

  it('should return changes between epochs', async () => {
    const db = GrafeoDB.create()
    db.enableCdc()
    db.createNode(['Person'], { name: 'Alix' })
    db.createNode(['Person'], { name: 'Gus' })

    const changes = await db.changesBetween(0, 1000)
    expect(changes.length).toBeGreaterThanOrEqual(2)
    db.close()
  })

  it('should return empty history for nonexistent node', async () => {
    const db = GrafeoDB.create()
    const history = await db.nodeHistory(9999)
    expect(history.length).toBe(0)
    db.close()
  })
})

// ── Label management ────────────────────────────────────────────────

describe('label management', () => {
  let db

  beforeEach(() => {
    db = GrafeoDB.create()
  })

  afterEach(() => {
    db.close()
  })

  it('should add a label to an existing node', () => {
    const node = db.createNode(['Person'])
    const added = db.addNodeLabel(node.id, 'Employee')
    expect(added).toBe(true)
    const labels = db.getNodeLabels(node.id)
    expect(labels).toContain('Person')
    expect(labels).toContain('Employee')
  })

  it('should return false when adding duplicate label', () => {
    const node = db.createNode(['Person'])
    const added = db.addNodeLabel(node.id, 'Person')
    expect(added).toBe(false)
  })

  it('should remove a label from a node', () => {
    const node = db.createNode(['Person', 'Employee'])
    const removed = db.removeNodeLabel(node.id, 'Employee')
    expect(removed).toBe(true)
    const labels = db.getNodeLabels(node.id)
    expect(labels).toContain('Person')
    expect(labels).not.toContain('Employee')
  })

  it('should return false when removing nonexistent label', () => {
    const node = db.createNode(['Person'])
    const removed = db.removeNodeLabel(node.id, 'NoSuchLabel')
    expect(removed).toBe(false)
  })

  it('should return null for labels of nonexistent node', () => {
    const labels = db.getNodeLabels(99999)
    expect(labels).toBeNull()
  })
})

// ── Property removal ────────────────────────────────────────────────

describe('property removal', () => {
  let db

  beforeEach(() => {
    db = GrafeoDB.create()
  })

  afterEach(() => {
    db.close()
  })

  it('should remove a node property', () => {
    const node = db.createNode(['Person'], { name: 'Alix', age: 30 })
    const removed = db.removeNodeProperty(node.id, 'age')
    expect(removed).toBe(true)
    const fetched = db.getNode(node.id)
    expect(fetched.get('name')).toBe('Alix')
    expect(fetched.get('age')).toBeUndefined()
  })

  it('should return false when removing nonexistent property', () => {
    const node = db.createNode(['Person'])
    const removed = db.removeNodeProperty(node.id, 'noSuchProp')
    expect(removed).toBe(false)
  })

  it('should remove an edge property', () => {
    const a = db.createNode(['A'])
    const b = db.createNode(['B'])
    const edge = db.createEdge(a.id, b.id, 'REL', { weight: 1.5, tag: 'x' })
    const removed = db.removeEdgeProperty(edge.id, 'weight')
    expect(removed).toBe(true)
    const fetched = db.getEdge(edge.id)
    expect(fetched.get('tag')).toBe('x')
    expect(fetched.get('weight')).toBeUndefined()
  })
})

// ── SPARQL with parameters ──────────────────────────────────────────

describe('SPARQL with parameters', () => {
  it('should execute SPARQL with params argument', async () => {
    const db = GrafeoDB.create()
    // Even if params aren't used in this query, the API should accept them
    const result = await db.executeSparql(
      'SELECT ?x WHERE { ?x ?y ?z }',
      { limit: 10 }
    )
    expect(result.length).toBe(0) // empty triple store
    db.close()
  })

  it('should execute SPARQL without params (backward compat)', async () => {
    const db = GrafeoDB.create()
    const result = await db.executeSparql('SELECT ?x WHERE { ?x ?y ?z }')
    expect(result.length).toBe(0)
    db.close()
  })
})

// ── SQL/PGQ queries ─────────────────────────────────────────────────

describe('SQL/PGQ queries', () => {
  it('should execute basic SQL/PGQ query', async () => {
    const db = GrafeoDB.create()
    await db.execute("INSERT (:Person {name: 'Alix', age: 30})")
    const result = await db.executeSql(
      "SELECT * FROM GRAPH_TABLE (MATCH (p:Person) COLUMNS (p.name AS name))"
    )
    expect(result.length).toBe(1)
    expect(result.scalar()).toBe('Alix')
    db.close()
  })

  it('should execute SQL/PGQ relationship query', async () => {
    const db = GrafeoDB.create()
    await db.execute("INSERT (:Person {name: 'Alix', age: 30})")
    await db.execute("INSERT (:Person {name: 'Gus', age: 25})")
    await db.execute(
      "MATCH (a:Person), (b:Person) WHERE a.name = 'Alix' AND b.name = 'Gus' INSERT (a)-[:KNOWS]->(b)"
    )
    const result = await db.executeSql(
      "SELECT * FROM GRAPH_TABLE (MATCH (a:Person)-[:KNOWS]->(b:Person) COLUMNS (a.name AS person, b.name AS friend))"
    )
    expect(result.length).toBe(1)
    db.close()
  })
})

// ── Admin operations ─────────────────────────────────────────────────

describe('admin operations', () => {
  it('should return node and edge counts', () => {
    const { db } = seedDb()
    expect(db.nodeCount()).toBeGreaterThan(0)
    expect(db.edgeCount()).toBeGreaterThan(0)
    db.close()
  })

  it('should close database without error', () => {
    const db = GrafeoDB.create()
    db.createNode(['Person'], { name: 'Test' })
    expect(() => db.close()).not.toThrow()
  })

  it('should return info as JSON', () => {
    const { db } = seedDb()
    const info = db.info()
    expect(info).toBeDefined()
    expect(typeof info).toBe('object')
    expect(info.node_count).toBe(4)
    expect(info.edge_count).toBe(3)
    db.close()
  })

  it('should return schema as JSON', () => {
    const { db } = seedDb()
    const schema = db.schema()
    expect(schema).toBeDefined()
    expect(typeof schema).toBe('object')
    db.close()
  })

  it('should return version string', () => {
    const db = GrafeoDB.create()
    const ver = db.version()
    expect(ver).toMatch(/^\d+\.\d+\.\d+$/)
    db.close()
  })
})

// ── ID validation ───────────────────────────────────────────────────

describe('ID validation', () => {
  let db

  beforeEach(() => {
    db = GrafeoDB.create()
  })

  afterEach(() => {
    db.close()
  })

  it('should reject negative node ID', () => {
    expect(() => db.getNode(-1)).toThrow(/Invalid node ID/)
  })

  it('should reject NaN node ID', () => {
    expect(() => db.getNode(NaN)).toThrow(/Invalid node ID/)
  })

  it('should reject Infinity node ID', () => {
    expect(() => db.getNode(Infinity)).toThrow(/Invalid node ID/)
  })

  it('should reject negative edge ID', () => {
    expect(() => db.getEdge(-1)).toThrow(/Invalid edge ID/)
  })
})

// ── Concurrent database instances ───────────────────────────────────

describe('concurrent instances', () => {
  it('should support multiple independent databases', async () => {
    const db1 = GrafeoDB.create()
    const db2 = GrafeoDB.create()

    db1.createNode(['A'], { val: 1 })
    db2.createNode(['B'], { val: 2 })

    expect(db1.nodeCount()).toBe(1)
    expect(db2.nodeCount()).toBe(1)

    const r1 = await db1.execute('MATCH (n:A) RETURN n.val')
    const r2 = await db2.execute('MATCH (n:B) RETURN n.val')
    expect(r1.scalar()).toBe(1)
    expect(r2.scalar()).toBe(2)
    db1.close()
    db2.close()
  })
})
