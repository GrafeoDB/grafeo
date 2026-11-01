/**
 * Typed array parameters in the WASM bindings (#586).
 *
 * A Float32Array or Float64Array is a VECTOR, of 32-bit floats as vectors
 * are stored: a Float64Array element becomes the nearest one. A Uint8Array
 * stays BYTES. Before 0.6.1 a Float64Array became a map of its indexes.
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

// 0.1 is exact in neither width, so a Float64Array element must be rounded
// to the 32-bit float a Float32Array holds.
const elements = [0.5, 0.1, 2]
const asStored = [...new Float32Array(elements)]

describe.skipIf(!available)('typed array parameters (WASM)', () => {
  it.each([
    ['Float32Array', new Float32Array(elements)],
    ['Float64Array', new Float64Array(elements)],
  ])('takes a %s parameter as a vector', (_kind, vector) => {
    const db = new Database()
    const [row] = db.executeWithParams('RETURN $v AS v, size($v) AS n', { v: vector })
    expect(row.v).toBeInstanceOf(Float32Array)
    expect([...row.v]).toEqual(asStored)
    expect(row.n).toBe(3)
  })

  it.each([
    ['Float32Array', new Float32Array(elements)],
    ['Float64Array', new Float64Array(elements)],
  ])('stores a %s property as a vector', (_kind, vector) => {
    const db = new Database()
    db.executeWithParams("INSERT (:Doc {name: 'Alix', e: $v})", { v: vector })
    const [row] = db.execute("MATCH (d:Doc {name: 'Alix'}) RETURN d.e AS e, size(d.e) AS n")
    expect(row.e).toBeInstanceOf(Float32Array)
    expect([...row.e]).toEqual(asStored)
    expect(row.n).toBe(3)
  })

  it('keeps a Uint8Array parameter as bytes', () => {
    const db = new Database()
    const [row] = db.executeWithParams('RETURN $b AS b, $b IS TYPED BYTES AS isBytes', { b: new Uint8Array([1, 2, 3]) })
    expect(row.isBytes).toBe(true)
    expect(row.b).toBeInstanceOf(Uint8Array)
    expect([...row.b]).toEqual([1, 2, 3])
  })
})
