// A Node.js process that exits without close() keeps its commits (#395): a
// child process opens a database, creates a node, optionally calls
// walCheckpoint(), and exits with process.exit(0); the parent reopens the
// database and finds the node with its property. The reporter used a path
// without the .grafeo extension; both kinds of path are checked.

import { spawnSync } from 'node:child_process'
import { mkdtempSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join, dirname } from 'node:path'
import { fileURLToPath } from 'node:url'
import { describe, it, expect } from 'vitest'
import { GrafeoDB } from '../grafeo.js'

const here = dirname(fileURLToPath(import.meta.url))

// Runs `body` in a child Node.js process that exits without close().
function crashAfter(path, body) {
  const script = `
    const { GrafeoDB } = require(${JSON.stringify(join(here, '..', 'grafeo.js'))})
    const db = GrafeoDB.create(${JSON.stringify(path)})
    ${body}
    process.exit(0)
  `
  const child = spawnSync(process.execPath, ['-e', script], { encoding: 'utf8' })
  expect(child.status, child.stderr).toBe(0)
}

async function titles(path) {
  const db = GrafeoDB.open(path)
  const result = await db.execute('MATCH (d:Document) RETURN d.title AS title ORDER BY title')
  const rows = result.toArray().map((row) => row.title)
  const count = db.nodeCount()
  db.close()
  return { rows, count }
}

describe('a process that exits without close()', () => {
  for (const name of ['notes', 'notes.grafeo']) {
    it(`keeps createNode after walCheckpoint (${name})`, async () => {
      const path = join(mkdtempSync(join(tmpdir(), 'grafeo-crash-')), name)
      crashAfter(path, "db.createNode(['Document'], { title: 'test' }); db.walCheckpoint()")
      expect(await titles(path)).toEqual({ rows: ['test'], count: 1 })
    })

    it(`keeps createNode without walCheckpoint (${name})`, async () => {
      const path = join(mkdtempSync(join(tmpdir(), 'grafeo-crash-')), name)
      crashAfter(path, "db.createNode(['Document'], { title: 'test' })")
      expect(await titles(path)).toEqual({ rows: ['test'], count: 1 })
    })
  }

  it('keeps the writes before and after walCheckpoint', async () => {
    const path = join(mkdtempSync(join(tmpdir(), 'grafeo-crash-')), 'mixed')
    crashAfter(
      path,
      `db.createNode(['Document'], { title: 'first' })
       db.walCheckpoint()
       db.createNode(['Document'], { title: 'second' })`,
    )
    expect(await titles(path)).toEqual({ rows: ['first', 'second'], count: 2 })
  })
})
