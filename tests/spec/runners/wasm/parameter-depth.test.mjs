/**
 * A parameter nested deeper than a property value may be (128 levels of
 * arrays and objects) is refused with an error that names the limit, and the
 * module keeps working: no WASM trap from a stack overflow while the binding
 * converts the value or the engine reads it, also at depth 100,000. A trap
 * leaves the module unusable, so each case runs a query afterwards, and the
 * controls run first.
 */

import { describe, it, expect } from 'vitest'
import { readFileSync, existsSync } from 'fs'
import { join, resolve } from 'path'

const pkg = resolve(import.meta.dirname, '..', '..', '..', '..', 'crates', 'bindings', 'wasm', 'pkg')
const available = existsSync(join(pkg, 'grafeo_wasm.js')) && existsSync(join(pkg, 'grafeo_wasm_bg.wasm'))

let Database
if (available) {
  const mod = await import('../../../../crates/bindings/wasm/pkg/grafeo_wasm.js')
  mod.initSync({ module: readFileSync(join(pkg, 'grafeo_wasm_bg.wasm')) })
  Database = mod.Database
}

const LIMIT = 128

function nested(shape, depth) {
  let value = 1
  for (let i = 0; i < depth; i++) value = shape === 'array' ? [value] : { a: value }
  return value
}

describe.skipIf(!available)('parameter depth (WASM)', () => {
  for (const shape of ['array', 'object']) {
    it(`takes a ${shape} at the limit`, () => {
      const db = new Database()
      const [row] = db.executeWithParams('RETURN $p = $p AS same', { p: nested(shape, LIMIT) })
      expect(row.same).toBe(true)
      db.executeWithParams('INSERT (:N {p: $p})', { p: nested(shape, LIMIT) })
      expect(db.execute('MATCH (n:N) RETURN count(n) AS n')[0].n).toBe(1)
    })
  }

  for (const depth of [LIMIT + 1, 100_000]) {
    for (const shape of ['array', 'object']) {
      for (const query of ['RETURN $p AS p', 'INSERT (:N {p: $p})']) {
        it(`refuses a ${shape} ${depth} deep: ${query}`, () => {
          const db = new Database()
          let message = null
          try {
            db.executeWithParams(query, { p: nested(shape, depth) })
          } catch (error) {
            message = String(error?.message ?? error)
          }
          expect(message, 'the call fails').not.toBeNull()
          expect(message).toContain(String(LIMIT))
          // The module is not trapped: this and a new database still answer.
          expect(db.execute('RETURN 1 AS one')[0].one).toBe(1)
          expect(new Database().execute('RETURN 2 AS two')[0].two).toBe(2)
        })
      }
    }
  }
})
