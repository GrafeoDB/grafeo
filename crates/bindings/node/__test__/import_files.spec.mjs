// importCsv and importJsonl keep every column and key under its own name.
// The binding built its own LOAD DATA statement with sanitized names, so a
// header with a space, a byte order mark, a quoted header holding the
// delimiter or a JSONL key that first appears on a later line lost its values
// without an error. Both calls now run the engine's import_nodes.

import { mkdtempSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { describe, it, expect } from 'vitest'
import { GrafeoDB } from '../grafeo.js'

function write(name, text) {
  const dir = mkdtempSync(join(tmpdir(), 'grafeo-import-'))
  const path = join(dir, name)
  writeFileSync(path, text, 'utf8')
  return path
}

async function nodes(db, label) {
  const quoted = label.replaceAll('`', '``')
  const result = await db.execute(`MATCH (n:\`${quoted}\`) RETURN properties(n) AS p`)
  return result.toArray().map((row) => row.p).sort((a, b) => JSON.stringify(a).localeCompare(JSON.stringify(b)))
}

describe('importCsv and importJsonl', () => {
  it('keeps the values of a header with a space', async () => {
    const path = write('people.csv', 'name,home city\nAlix,Amsterdam\nGus,Berlin\n')
    const db = GrafeoDB.create()
    expect(await db.importCsv(path, { label: 'Person' })).toBe(2)
    expect(await nodes(db, 'Person')).toEqual([
      { name: 'Alix', 'home city': 'Amsterdam' },
      { name: 'Gus', 'home city': 'Berlin' },
    ])
  })

  it('reads a byte order mark and a quoted header', async () => {
    const path = write('excel.csv', '\ufeffid,"Amsterdam, NL"\n3,yes\n')
    const db = GrafeoDB.create()
    expect(await db.importCsv(path)).toBe(1)
    expect(await nodes(db, 'Row')).toEqual([{ id: '3', 'Amsterdam, NL': 'yes' }])
  })

  it('numbers the columns of a CSV without headers', async () => {
    const path = write('raw.csv', 'Alix,Amsterdam\nGus,Berlin,3\n')
    const db = GrafeoDB.create()
    expect(await db.importCsv(path, { label: 'Raw', headers: false })).toBe(2)
    expect(await nodes(db, 'Raw')).toEqual([
      { column0: 'Alix', column1: 'Amsterdam' },
      { column0: 'Gus', column1: 'Berlin', column2: '3' },
    ])
  })

  it('keeps JSONL keys of later lines with their types', async () => {
    const path = write('events.jsonl', '{"id": 1}\n{"id": 2, "geo.lat": 52.5}\n')
    const db = GrafeoDB.create()
    expect(await db.importJsonl(path, { label: 'Event' })).toBe(2)
    expect(await nodes(db, 'Event')).toEqual([{ id: 2, 'geo.lat': 52.5 }, { id: 1 }])
  })

  it('uses the label as given', async () => {
    const path = write('fans.csv', 'name\nMia\n')
    const db = GrafeoDB.create()
    expect(await db.importCsv(path, { label: 'Film Fan' })).toBe(1)
    expect(await nodes(db, 'Film Fan')).toEqual([{ name: 'Mia' }])
  })

  it('counts only the nodes it created', async () => {
    const path = write('people.csv', 'name\nVincent\nJules\n')
    const db = GrafeoDB.create()
    await db.execute("INSERT (:Person {name: 'Butch'})")
    expect(await db.importCsv(path, { label: 'Person' })).toBe(2)
    expect(await nodes(db, 'Person')).toHaveLength(3)
  })

  it('rejects a missing file', async () => {
    const db = GrafeoDB.create()
    await expect(db.importCsv(join(tmpdir(), 'grafeo-no-such-file.csv'))).rejects.toThrow()
  })
})
