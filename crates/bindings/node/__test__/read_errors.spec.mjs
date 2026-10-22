// A read that fails throws; an absent node or edge reads as null. A read of a
// graph that was dropped is a failing read: a graph handle throws instead of
// answering null.

import { describe, it, expect } from 'vitest'
import { GrafeoDB } from '../grafeo.js'

describe('read errors', () => {
  it('reads an absent node or edge as null', () => {
    const db = GrafeoDB.create()
    const mia = db.createNode(['Person'], { name: 'Mia' })
    db.deleteNode(mia.id)
    expect(db.getNode(mia.id)).toBeNull()
    expect(db.getNodeLabels(mia.id)).toBeNull()
    expect(db.getNode(424242)).toBeNull()
    expect(db.getEdge(424242)).toBeNull()
    expect(db.nodeCount()).toBe(0)
  })

  it('throws for a read of a dropped graph, with its GQLSTATUS', async () => {
    const db = GrafeoDB.create()
    await db.execute('CREATE GRAPH trips')
    const trips = db.graph('trips')
    const berlin = trips.createNode(['City'], { name: 'Berlin' })
    expect(trips.getNode(424242)).toBeNull()
    expect(trips.getNode(berlin.id).get('name')).toBe('Berlin')
    await db.execute('DROP GRAPH trips')
    let thrown
    try {
      trips.getNode(berlin.id)
    } catch (error) {
      thrown = error
    }
    expect(thrown).toBeDefined()
    expect(typeof thrown.gqlStatus).toBe('string')
    expect(thrown.gqlStatus).toHaveLength(5)
  })
})
