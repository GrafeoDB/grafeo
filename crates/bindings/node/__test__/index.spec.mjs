import { describe, it, expect, beforeEach } from 'vitest'
import { GrafeoDB, version, simdSupport } from '../index.js'

// ── Helpers ──────────────────────────────────────────────────────────

/** Create a fresh in-memory database with some seed data. */
function seedDb() {
  const db = GrafeoDB.create()
  // People
  const alice = db.createNode(['Person'], { name: 'Alice', age: 30 })
  const bob = db.createNode(['Person'], { name: 'Bob', age: 25 })
  const charlie = db.createNode(['Person'], { name: 'Charlie', age: 35 })
  // Company
  const acme = db.createNode(['Company'], { name: 'Acme Corp', founded: 2010 })
  // Relationships
  const knows1 = db.createEdge(alice.id, bob.id, 'KNOWS', { since: 2020 })
  const knows2 = db.createEdge(bob.id, charlie.id, 'KNOWS', { since: 2021 })
  const worksAt = db.createEdge(alice.id, acme.id, 'WORKS_AT', { role: 'Engineer' })
  return { db, alice, bob, charlie, acme, knows1, knows2, worksAt }
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
    expect(db.nodeCount).toBe(0)
    expect(db.edgeCount).toBe(0)
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
    expect(db.nodeCount).toBe(1)
    db.close()

    // Reopen
    const db2 = GrafeoDB.open(dbPath)
    expect(db2.nodeCount).toBe(1)
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

  it('should create a node with labels', () => {
    const node = db.createNode(['Person'])
    expect(node.id).toBeGreaterThanOrEqual(0)
    expect(node.labels).toEqual(['Person'])
    expect(db.nodeCount).toBe(1)
  })

  it('should create a node with multiple labels', () => {
    const node = db.createNode(['Person', 'Employee'])
    expect(node.labels).toContain('Person')
    expect(node.labels).toContain('Employee')
  })

  it('should create a node with properties', () => {
    const node = db.createNode(['Person'], { name: 'Alice', age: 30 })
    expect(node.get('name')).toBe('Alice')
    expect(node.get('age')).toBe(30)
  })

  it('should get a node by ID', () => {
    const created = db.createNode(['Person'], { name: 'Alice' })
    const fetched = db.getNode(created.id)
    expect(fetched).not.toBeNull()
    expect(fetched.id).toBe(created.id)
    expect(fetched.get('name')).toBe('Alice')
  })

  it('should return null for nonexistent node', () => {
    expect(db.getNode(99999)).toBeNull()
  })

  it('should delete a node', () => {
    const node = db.createNode(['Person'])
    expect(db.deleteNode(node.id)).toBe(true)
    expect(db.getNode(node.id)).toBeNull()
    expect(db.nodeCount).toBe(0)
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
  let db, alice, bob

  beforeEach(() => {
    db = GrafeoDB.create()
    alice = db.createNode(['Person'], { name: 'Alice' })
    bob = db.createNode(['Person'], { name: 'Bob' })
  })

  it('should create an edge', () => {
    const edge = db.createEdge(alice.id, bob.id, 'KNOWS')
    expect(edge.id).toBeGreaterThanOrEqual(0)
    expect(edge.edgeType).toBe('KNOWS')
    expect(edge.sourceId).toBe(alice.id)
    expect(edge.targetId).toBe(bob.id)
    expect(db.edgeCount).toBe(1)
  })

  it('should create an edge with properties', () => {
    const edge = db.createEdge(alice.id, bob.id, 'KNOWS', { since: 2020 })
    expect(edge.get('since')).toBe(2020)
  })

  it('should get an edge by ID', () => {
    const created = db.createEdge(alice.id, bob.id, 'KNOWS', { weight: 0.5 })
    const fetched = db.getEdge(created.id)
    expect(fetched).not.toBeNull()
    expect(fetched.edgeType).toBe('KNOWS')
    expect(fetched.get('weight')).toBeCloseTo(0.5)
  })

  it('should return null for nonexistent edge', () => {
    expect(db.getEdge(99999)).toBeNull()
  })

  it('should delete an edge', () => {
    const edge = db.createEdge(alice.id, bob.id, 'KNOWS')
    expect(db.deleteEdge(edge.id)).toBe(true)
    expect(db.getEdge(edge.id)).toBeNull()
    expect(db.edgeCount).toBe(0)
  })

  it('should toString produce readable output', () => {
    const edge = db.createEdge(alice.id, bob.id, 'KNOWS')
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

  it('should set and get node property', () => {
    const node = db.createNode(['Person'])
    db.setNodeProperty(node.id, 'name', 'Alice')
    const updated = db.getNode(node.id)
    expect(updated.get('name')).toBe('Alice')
  })

  it('should overwrite node property', () => {
    const node = db.createNode(['Person'], { name: 'Alice' })
    db.setNodeProperty(node.id, 'name', 'Bob')
    const updated = db.getNode(node.id)
    expect(updated.get('name')).toBe('Bob')
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
    const node = db.createNode(['Person'], { name: 'Alice', age: 30 })
    const props = node.properties()
    expect(props.name).toBe('Alice')
    expect(props.age).toBe(30)
  })
})

// ── GQL Queries ──────────────────────────────────────────────────────

describe('GQL queries', () => {
  it('should execute INSERT and MATCH', async () => {
    const db = GrafeoDB.create()
    await db.execute("INSERT (:Person {name: 'Alice', age: 30})")
    await db.execute("INSERT (:Person {name: 'Bob', age: 25})")
    const result = await db.execute('MATCH (p:Person) RETURN p.name, p.age')

    expect(result.length).toBe(2)
    expect(result.columns.length).toBe(2)

    const rows = result.toArray()
    const names = rows.map((r) => r[result.columns[0]])
    expect(names).toContain('Alice')
    expect(names).toContain('Bob')
  })

  it('should execute with parameters', async () => {
    const db = GrafeoDB.create()
    await db.execute("INSERT (:Person {name: 'Alice', age: 30})")
    await db.execute("INSERT (:Person {name: 'Bob', age: 25})")
    const result = await db.execute(
      'MATCH (p:Person) WHERE p.age > $minAge RETURN p.name',
      { minAge: 28 }
    )

    expect(result.length).toBe(1)
    const name = result.scalar()
    expect(name).toBe('Alice')
  })

  it('should return scalar value', async () => {
    const db = GrafeoDB.create()
    await db.execute("INSERT (:Person {name: 'Alice'})")
    const result = await db.execute('MATCH (p:Person) RETURN p.name')
    expect(result.scalar()).toBe('Alice')
  })

  it('should return execution time', async () => {
    const db = GrafeoDB.create()
    const result = await db.execute('MATCH (n) RETURN n')
    expect(result.executionTimeMs).not.toBeNull()
    expect(result.executionTimeMs).toBeGreaterThanOrEqual(0)
  })

  it('should match relationships', async () => {
    const { db } = seedDb()
    const result = await db.execute(
      "MATCH (a:Person)-[:KNOWS]->(b:Person) WHERE a.name = 'Alice' RETURN b.name"
    )
    expect(result.length).toBe(1)
    expect(result.scalar()).toBe('Bob')
  })

  it('should return rows as arrays', async () => {
    const db = GrafeoDB.create()
    await db.execute("INSERT (:Person {name: 'Alice'})")
    const result = await db.execute('MATCH (p:Person) RETURN p.name')
    const rows = result.rows()
    expect(rows.length).toBe(1)
    expect(rows[0][0]).toBe('Alice')
  })

  it('should get row by index', async () => {
    const db = GrafeoDB.create()
    await db.execute("INSERT (:Person {name: 'Alice'})")
    const result = await db.execute('MATCH (p:Person) RETURN p.name')
    const row = result.get(0)
    expect(Object.values(row)).toContain('Alice')
  })

  it('should throw on invalid query', async () => {
    const db = GrafeoDB.create()
    await expect(db.execute('THIS IS NOT VALID')).rejects.toThrow()
  })
})

// ── Aggregations ─────────────────────────────────────────────────────

describe('aggregations', () => {
  it('should count nodes', async () => {
    const { db } = seedDb()
    const result = await db.execute('MATCH (p:Person) RETURN COUNT(p)')
    expect(result.scalar()).toBe(3)
  })

  it('should compute SUM and AVG', async () => {
    const { db } = seedDb()
    const result = await db.execute(
      'MATCH (p:Person) RETURN SUM(p.age), AVG(p.age)'
    )
    const row = result.toArray()[0]
    const values = Object.values(row)
    expect(values).toContain(90) // 30 + 25 + 35
    expect(values).toContain(30) // 90 / 3
  })
})

// ── Transactions ─────────────────────────────────────────────────────

describe('transactions', () => {
  it('should commit transaction', async () => {
    const db = GrafeoDB.create()
    const tx = db.beginTransaction()
    expect(tx.isActive).toBe(true)

    await tx.execute("INSERT (:Person {name: 'Alice'})")
    tx.commit()

    expect(tx.isActive).toBe(false)
    expect(db.nodeCount).toBe(1)
  })

  it('should rollback transaction', async () => {
    const db = GrafeoDB.create()
    const tx = db.beginTransaction()
    await tx.execute("INSERT (:Person {name: 'Alice'})")
    tx.rollback()

    expect(tx.isActive).toBe(false)
    expect(db.nodeCount).toBe(0)
  })

  it('should error on double commit', async () => {
    const db = GrafeoDB.create()
    const tx = db.beginTransaction()
    await tx.execute("INSERT (:Person {name: 'Alice'})")
    tx.commit()
    expect(() => tx.commit()).toThrow(/Already committed/)
  })

  it('should error on commit after rollback', async () => {
    const db = GrafeoDB.create()
    const tx = db.beginTransaction()
    tx.rollback()
    expect(() => tx.commit()).toThrow(/Already rolled back/)
  })

  it('should execute multiple operations', async () => {
    const db = GrafeoDB.create()
    const tx = db.beginTransaction()
    await tx.execute("INSERT (:Person {name: 'Alice'})")
    await tx.execute("INSERT (:Person {name: 'Bob'})")
    await tx.execute("INSERT (:Person {name: 'Charlie'})")
    tx.commit()

    expect(db.nodeCount).toBe(3)
  })

  it('should execute with parameters in transaction', async () => {
    const db = GrafeoDB.create()
    await db.execute("INSERT (:Person {name: 'Alice', age: 30})")
    await db.execute("INSERT (:Person {name: 'Bob', age: 25})")

    const tx = db.beginTransaction()
    const result = await tx.execute(
      'MATCH (p:Person) WHERE p.age > $minAge RETURN p.name',
      { minAge: 28 }
    )
    tx.commit()

    expect(result.length).toBe(1)
    expect(result.scalar()).toBe('Alice')
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
  })

  it('should extract nodes from MATCH result', async () => {
    const { db } = seedDb()
    const result = await db.execute('MATCH (p:Person) RETURN p')
    const nodes = result.nodes()
    expect(nodes.length).toBe(3)
    const names = nodes.map((n) => n.get('name'))
    expect(names).toContain('Alice')
    expect(names).toContain('Bob')
    expect(names).toContain('Charlie')
  })

  it('should return edges() accessor without error', async () => {
    const { db } = seedDb()
    const result = await db.execute(
      'MATCH (a:Person)-[r:KNOWS]->(b:Person) RETURN a.name, b.name'
    )
    // edges() should be callable even when no edge columns are returned
    const edges = result.edges()
    expect(Array.isArray(edges)).toBe(true)
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
  })

  it('should return empty nodes/edges for scalar queries', async () => {
    const db = GrafeoDB.create()
    await db.execute("INSERT (:Person {name: 'Alice'})")
    const result = await db.execute('MATCH (p:Person) RETURN p.name')
    expect(result.nodes().length).toBe(0)
    expect(result.edges().length).toBe(0)
  })
})

// ── Advanced type round-trips ───────────────────────────────────────

describe('advanced type round-trips', () => {
  let db

  beforeEach(() => {
    db = GrafeoDB.create()
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
    await db.executeCypher("CREATE (a:Person {name: 'Alice'})")
    const result = await db.executeCypher('MATCH (p:Person) RETURN p.name')
    expect(result.scalar()).toBe('Alice')
  })

  it('should execute Cypher with parameters', async () => {
    const db = GrafeoDB.create()
    await db.executeCypher("CREATE (:Person {name: 'Alice', age: 30})")
    await db.executeCypher("CREATE (:Person {name: 'Bob', age: 25})")
    const result = await db.executeCypher(
      'MATCH (p:Person) WHERE p.age > $min RETURN p.name',
      { min: 28 }
    )
    expect(result.length).toBe(1)
    expect(result.scalar()).toBe('Alice')
  })
})

// ── Gremlin queries ─────────────────────────────────────────────────

describe('Gremlin queries', () => {
  it('should execute basic Gremlin traversal', async () => {
    const db = GrafeoDB.create()
    await db.execute("INSERT (:Person {name: 'Alice'})")
    await db.execute("INSERT (:Person {name: 'Bob'})")
    const result = await db.executeGremlin(
      "g.V().hasLabel('Person').count()"
    )
    expect(result.scalar()).toBe(2)
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
  })
})

// ── Transaction edge cases ──────────────────────────────────────────

describe('transaction edge cases', () => {
  it('should error on execute after commit', async () => {
    const db = GrafeoDB.create()
    const tx = db.beginTransaction()
    await tx.execute("INSERT (:Person {name: 'Alice'})")
    tx.commit()
    await expect(
      tx.execute("INSERT (:Person {name: 'Bob'})")
    ).rejects.toThrow(/no longer active/)
  })

  it('should error on execute after rollback', async () => {
    const db = GrafeoDB.create()
    const tx = db.beginTransaction()
    tx.rollback()
    await expect(
      tx.execute("INSERT (:Person {name: 'Alice'})")
    ).rejects.toThrow(/no longer active/)
  })

  it('should error on double rollback', () => {
    const db = GrafeoDB.create()
    const tx = db.beginTransaction()
    tx.rollback()
    expect(() => tx.rollback()).toThrow(/Already rolled back/)
  })

  it('should error on rollback after commit', async () => {
    const db = GrafeoDB.create()
    const tx = db.beginTransaction()
    await tx.execute("INSERT (:Person {name: 'Alice'})")
    tx.commit()
    expect(() => tx.rollback()).toThrow(/Already committed/)
  })
})

// ── Error handling ───────────────────────────────────────────────────

describe('error handling', () => {
  it('should throw on out-of-range row index', async () => {
    const db = GrafeoDB.create()
    const result = await db.execute('MATCH (n) RETURN n')
    expect(() => result.get(999)).toThrow()
  })

  it('should throw on scalar with no rows', async () => {
    const db = GrafeoDB.create()
    const result = await db.execute('MATCH (n:NonExistent) RETURN n')
    expect(() => result.scalar()).toThrow()
  })

  it('should throw on invalid params type', async () => {
    const db = GrafeoDB.create()
    // Passing a non-object as params
    await expect(
      db.execute('MATCH (n) RETURN n', 'not-an-object')
    ).rejects.toThrow()
  })
})

// ── Counts / getters ─────────────────────────────────────────────────

describe('database getters', () => {
  it('should track nodeCount and edgeCount', () => {
    const { db } = seedDb()
    expect(db.nodeCount).toBe(4) // Alice, Bob, Charlie, Acme
    expect(db.edgeCount).toBe(3) // knows1, knows2, worksAt
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
  })

  it('should create vector index with HNSW tuning params', async () => {
    const db = GrafeoDB.create()
    await db.batchCreateNodes('Doc', 'embedding', [[1, 0, 0]])

    // Pass m and ef_construction
    await db.createVectorIndex('Doc', 'embedding', 3, 'cosine', 32, 200)
    const results = await db.vectorSearch('Doc', 'embedding', [1, 0, 0], 1)
    expect(results.length).toBe(1)
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
    expect(db.nodeCount).toBe(3)
    // All unique IDs
    expect(new Set(ids).size).toBe(3)
  })

  it('should batch create empty list', async () => {
    const db = GrafeoDB.create()
    const ids = await db.batchCreateNodes('Doc', 'embedding', [])
    expect(ids.length).toBe(0)
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
  })

  it('should error on vector search without index', async () => {
    const db = GrafeoDB.create()
    db.createNode(['Doc'], { embedding: new Float32Array([1, 0, 0]) })
    await expect(
      db.vectorSearch('Doc', 'embedding', [1, 0, 0], 1)
    ).rejects.toThrow()
  })
})

// ── GraphQL queries ─────────────────────────────────────────────────

describe('GraphQL queries', () => {
  it('should execute basic GraphQL query', async () => {
    const db = GrafeoDB.create()
    await db.execute("INSERT (:Person {name: 'Alice', age: 30})")
    await db.execute("INSERT (:Person {name: 'Bob', age: 25})")
    const result = await db.executeGraphql('{ Person { name } }')
    expect(result.length).toBeGreaterThanOrEqual(1)
  })
})
