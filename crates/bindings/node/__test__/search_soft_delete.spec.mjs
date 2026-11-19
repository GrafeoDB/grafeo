// Leaving soft-deleted nodes out of text search from Node (#742): filters
// { deleted: { $ne: true } } keep a node that has no deleted flag, and a null
// filter value is refused with an error pointing to $ne.

import { describe, it, expect } from 'vitest'
import { GrafeoDB } from '../grafeo.js'

async function documents() {
  const db = GrafeoDB.create()
  await db.execute(
    "INSERT (:Doc {owner: 'Alix', text: 'canals of Amsterdam', deleted: false}), " +
      "(:Doc {owner: 'Gus', text: 'canals and bridges', deleted: true}), " +
      "(:Doc {owner: 'Vincent', text: 'canals at night'}), " +
      "(:Doc {owner: 'Jules', text: 'Amsterdam canals in winter', deleted: false})",
  )
  await db.createTextIndex('Doc', 'text')
  return db
}

async function owners(db, hits) {
  const found = new Set()
  for (const [id] of hits) {
    const rows = (await db.execute(`MATCH (d:Doc) WHERE id(d) = ${id} RETURN d.owner AS o`)).toArray()
    found.add(rows[0].o)
  }
  return found
}

describe('soft delete in text search', () => {
  it('keeps a node without the flag under $ne', async () => {
    const db = await documents()
    const hits = await db.textSearch('Doc', 'text', 'canals', 10, { deleted: { $ne: true } })
    expect(await owners(db, hits)).toEqual(new Set(['Alix', 'Vincent', 'Jules']))
    const live = await db.textSearch('Doc', 'text', 'canals', 10, { deleted: false })
    expect(await owners(db, live)).toEqual(new Set(['Alix', 'Jules']))
  })

  it('refuses a null filter value, pointing to $ne', async () => {
    const db = await documents()
    for (const filters of [{ deleted: null }, { deleted: { $ne: null } }]) {
      await expect(db.textSearch('Doc', 'text', 'canals', 10, filters)).rejects.toThrow(/\$ne/)
    }
  })
})
