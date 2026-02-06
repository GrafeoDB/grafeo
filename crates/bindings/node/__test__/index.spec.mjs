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
    expect(version()).toBe('0.4.0')
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

    // Cleanup
    fs.rmSync(dir, { recursive: true })
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

// ── Cypher queries ───────────────────────────────────────────────────

describe('Cypher queries', () => {
  it('should execute Cypher CREATE and MATCH', async () => {
    const db = GrafeoDB.create()
    await db.executeCypher("CREATE (a:Person {name: 'Alice'})")
    const result = await db.executeCypher('MATCH (p:Person) RETURN p.name')
    expect(result.scalar()).toBe('Alice')
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
})

// ── Counts / getters ─────────────────────────────────────────────────

describe('database getters', () => {
  it('should track nodeCount and edgeCount', () => {
    const { db } = seedDb()
    expect(db.nodeCount).toBe(4) // Alice, Bob, Charlie, Acme
    expect(db.edgeCount).toBe(3) // knows1, knows2, worksAt
  })
})
