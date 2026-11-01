// A tagged parameter object is its typed value only when the tag is its only
// key: `{ $date: '2026-10-18' }` is a date, `{ $date: '2026-10-18', note: 'x' }`
// is a map of both keys, as `{ $decimal }` already was (0.6.1). The shared
// conversion is grafeo-bindings-common's tagged_value; its own tests are
// crates/bindings/common/tests/json_temporal_tags.rs.

import { describe, it, expect } from 'vitest'
import { GrafeoDB } from '../grafeo.js'

const TAGS = [
  ['$date', '2026-10-18', 'DATE'],
  ['$time', '14:30:00', 'TIME'],
  ['$duration', 'P1D', 'DURATION'],
  ['$zoned_datetime', '2026-10-18T14:30:00+02:00', 'ZONED DATETIME'],
]

describe('tagged parameter objects', () => {
  it('a tag alone is its typed value', async () => {
    const db = GrafeoDB.create()
    for (const [tag, text] of TAGS) {
      const [row] = (await db.execute('RETURN $v IS TYPED MAP AS map', { v: { [tag]: text } })).toArray()
      expect(row.map, tag).toBe(false)
    }
  })

  it('a tag beside another key is a map of both keys', async () => {
    const db = GrafeoDB.create()
    for (const [tag, text] of TAGS) {
      const [row] = (
        await db.execute('RETURN $v IS TYPED MAP AS map, size(keys($v)) AS n, $v.note AS note', {
          v: { [tag]: text, note: 'x' },
        })
      ).toArray()
      expect(row, tag).toEqual({ map: true, n: 2, note: 'x' })
    }
  })
})
