// A parameter nested deeper than a property value may be (128 levels of
// arrays and objects) is refused with an error that names the limit, and the
// process lives on: no stack overflow while the binding converts it or the
// engine reads it, also at depth 100,000. The deep cases run in a child
// process, since a stack overflow ends it.

import { spawnSync } from 'node:child_process'
import { join, dirname } from 'node:path'
import { fileURLToPath } from 'node:url'
import { describe, it, expect } from 'vitest'
import { GrafeoDB } from '../grafeo.js'

const here = dirname(fileURLToPath(import.meta.url))
const LIMIT = 128

function nested(shape, depth) {
  let value = 1
  for (let i = 0; i < depth; i++) value = shape === 'array' ? [value] : { a: value }
  return value
}

// Runs the query with a `shape` parameter `depth` deep in a child process:
// its exit status and output. 19: refused, and the database still answers.
function inChild(shape, depth, query) {
  const script = `
    const { GrafeoDB } = require(${JSON.stringify(join(here, '..', 'grafeo.js'))})
    let value = 1
    for (let i = 0; i < ${depth}; i++) value = ${JSON.stringify(shape)} === 'array' ? [value] : { a: value }
    const db = GrafeoDB.create()
    db.execute(${JSON.stringify(query)}, { p: value }).then(
      () => { console.log('ACCEPTED'); process.exit(23) },
      async (error) => {
        console.log('REFUSED', error.message)
        const one = await db.execute('RETURN 1 AS one')
        process.exit(one.toArray()[0].one === 1 ? 19 : 29)
      },
    )
  `
  const child = spawnSync(process.execPath, ['-e', script], { encoding: 'utf8' })
  return { status: child.status, signal: child.signal, output: `${child.stdout}\n${child.stderr}` }
}

describe('a parameter nested past the limit', () => {
  for (const depth of [LIMIT + 1, 100_000]) {
    for (const shape of ['array', 'object']) {
      for (const query of ['RETURN $p AS p', 'INSERT (:N {p: $p})']) {
        it(`is refused: ${shape} ${depth} deep, ${query}`, () => {
          const { status, signal, output } = inChild(shape, depth, query)
          expect({ status, signal }, output).toEqual({ status: 19, signal: null })
          expect(output).toContain(String(LIMIT))
        })
      }
    }
  }
})

describe('a parameter at the limit', () => {
  for (const shape of ['array', 'object']) {
    it(`works: ${shape}`, async () => {
      const db = GrafeoDB.create()
      const value = nested(shape, LIMIT)
      const same = await db.execute('RETURN $p = $p AS same', { p: value })
      expect(same.toArray()[0].same).toBe(true)
      await db.execute('INSERT (:N {p: $p})', { p: value })
      const count = await db.execute('MATCH (n:N) RETURN count(n) AS n')
      expect(count.toArray()[0].n).toBe(1)
      db.close()
    })
  }
})
