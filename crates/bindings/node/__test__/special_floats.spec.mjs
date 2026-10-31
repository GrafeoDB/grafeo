// NaN and the infinities as parameters round-trip as native numbers (Node.js
// passes JS values, not JSON), and the strings 'NaN' and 'Infinity' stay
// strings (0.6.1, the $float tag of the JSON bindings).

import { describe, it, expect } from 'vitest'
import { GrafeoDB } from '../grafeo.js'

describe('special floats', () => {
  it('NaN and the infinities round-trip as FLOAT64', async () => {
    const db = GrafeoDB.create()
    for (const value of [NaN, Infinity, -Infinity]) {
      const [row] = (await db.execute('RETURN $x AS x, $x IS TYPED FLOAT64 AS float', { x: value })).toArray()
      expect(row.float, String(value)).toBe(true)
      expect(Object.is(row.x, value), `${value} came back as ${row.x}`).toBe(true)
    }
  })

  it('the strings NaN and Infinity stay strings', async () => {
    const db = GrafeoDB.create()
    for (const text of ['NaN', 'Infinity', '-Infinity']) {
      const [row] = (await db.execute('RETURN $x AS x, $x IS TYPED STRING AS str', { x: text })).toArray()
      expect(row).toEqual({ x: text, str: true })
    }
  })
})
