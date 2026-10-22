/**
 * Result comparison logic for .gtest spec tests.
 *
 * Mirrors the assertion helpers in grafeo-spec-tests/src/lib.rs so that
 * the Node.js runner validates results identically to the Rust runner.
 */

import { createHash } from 'crypto'

/**
 * Convert a duration from {months, days, nanos} to ISO 8601 format.
 * Matches Rust's Display impl for Duration.
 */
function durationToIso(totalMonths, days, nanos) {
  const years = Math.floor(totalMonths / 12)
  const months = totalMonths % 12
  const hours = Math.floor(nanos / 3_600_000_000_000)
  let rem = nanos % 3_600_000_000_000
  const minutes = Math.floor(rem / 60_000_000_000)
  rem = rem % 60_000_000_000
  const seconds = Math.floor(rem / 1_000_000_000)
  const subNanos = rem % 1_000_000_000

  let result = 'P'
  if (years) result += `${years}Y`
  if (months) result += `${months}M`
  if (days) result += `${days}D`

  let timePart = ''
  if (hours) timePart += `${hours}H`
  if (minutes) timePart += `${minutes}M`
  if (seconds || subNanos) {
    if (subNanos) {
      const frac = String(subNanos).padStart(9, '0').replace(/0+$/, '')
      timePart += `${seconds}.${frac}S`
    } else {
      timePart += `${seconds}S`
    }
  }
  if (timePart) result += 'T' + timePart

  return result === 'P' ? 'P0D' : result
}

/**
 * Convert a JS value to its canonical string for comparison.
 * Must match Rust's value_to_string in lib.rs.
 */
export function valueToString(val) {
  if (val === null || val === undefined) return 'null'
  if (typeof val === 'boolean') return val ? 'true' : 'false'
  if (typeof val === 'bigint') return val.toString()
  if (typeof val === 'number') {
    if (isNaN(val)) return 'NaN'
    if (!isFinite(val)) return val > 0 ? 'Infinity' : '-Infinity'
    // Rust: format!("{}", 15.0_f64) -> "15" (no trailing .0)
    if (Number.isInteger(val)) return val.toString()
    return val.toString()
  }
  if (Array.isArray(val)) {
    const inner = val.map(valueToString).join(', ')
    return `[${inner}]`
  }
  if (typeof val === 'object' && val !== null) {
    // Date-like objects
    if (val instanceof Date) return val.toISOString()
    // Duration: {months, days, nanos} -> ISO 8601
    const keys = Object.keys(val)
    if (keys.length === 3 && 'months' in val && 'days' in val && 'nanos' in val) {
      return durationToIso(val.months, val.days, val.nanos)
    }
    // Plain object (map)
    const entries = Object.entries(val)
      .map(([k, v]) => `${k}: ${valueToString(v)}`)
      .sort()
    return `{${entries.join(', ')}}`
  }
  return String(val)
}

/**
 * Convert a GrafeoDB QueryResult to rows of canonical strings.
 * @param {object} result - QueryResult from db.execute()
 * @returns {string[][]}
 */
export function resultToRows(result) {
  const columns = result.columns
  const rows = []
  const arr = result.toArray()
  for (const row of arr) {
    const r = []
    for (const col of columns) {
      r.push(valueToString(row[col]))
    }
    rows.push(r)
  }
  return rows
}

/**
 * Assert rows match after sorting both sides.
 */
export function assertRowsSorted(result, expected) {
  const actual = resultToRows(result)
  const sortedActual = [...actual].sort((a, b) => a.join('|').localeCompare(b.join('|')))
  const sortedExpected = [...expected].sort((a, b) => a.join('|').localeCompare(b.join('|')))

  if (sortedActual.length !== sortedExpected.length) {
    throw new Error(
      `Row count mismatch: got ${sortedActual.length}, expected ${sortedExpected.length}\n` +
      `Actual: ${JSON.stringify(sortedActual)}\nExpected: ${JSON.stringify(sortedExpected)}`
    )
  }
  for (let i = 0; i < sortedActual.length; i++) {
    for (let j = 0; j < sortedActual[i].length; j++) {
      if (sortedActual[i][j] !== sortedExpected[i][j]) {
        throw new Error(
          `Mismatch at sorted row ${i}, col ${j}: got '${sortedActual[i][j]}', expected '${sortedExpected[i][j]}'\n` +
          `Actual row: ${JSON.stringify(sortedActual[i])}\nExpected row: ${JSON.stringify(sortedExpected[i])}`
        )
      }
    }
  }
}

/**
 * Assert rows match with floating-point tolerance.
 * Cells that parse as numbers on both sides are compared within 10^(-precision).
 */
export function assertRowsWithPrecision(result, expected, precision) {
  const actual = resultToRows(result)
  const tolerance = Math.pow(10, -precision)
  if (actual.length !== expected.length) {
    throw new Error(
      `Row count mismatch: got ${actual.length}, expected ${expected.length}`
    )
  }
  for (let i = 0; i < actual.length; i++) {
    for (let j = 0; j < actual[i].length; j++) {
      const a = actual[i][j]
      const e = expected[i][j]
      const af = parseFloat(a)
      const ef = parseFloat(e)
      if (!isNaN(af) && !isNaN(ef)) {
        if (Math.abs(af - ef) >= tolerance) {
          throw new Error(
            `Float mismatch at row ${i}, col ${j}: got ${af}, expected ${ef} (tolerance ${tolerance})`
          )
        }
      } else if (a !== e) {
        throw new Error(
          `Mismatch at row ${i}, col ${j}: got '${a}', expected '${e}'`
        )
      }
    }
  }
}

/**
 * Assert that the MD5 hash of sorted, pipe-delimited rows matches.
 * Mirrors assert_hash in the Rust runner.
 */
export function assertHash(result, expectedHash) {
  const rows = resultToRows(result)
  rows.sort((a, b) => a.join('|').localeCompare(b.join('|')))
  const hasher = createHash('md5')
  for (const row of rows) {
    hasher.update(row.join('|'))
    hasher.update('\n')
  }
  const actualHash = hasher.digest('hex')
  if (actualHash !== expectedHash) {
    throw new Error(
      `Hash mismatch: got '${actualHash}', expected '${expectedHash}'\nRows: ${JSON.stringify(rows)}`
    )
  }
}

/**
 * Assert rows match in exact order.
 */
export function assertRowsOrdered(result, expected) {
  const actual = resultToRows(result)
  if (actual.length !== expected.length) {
    throw new Error(
      `Row count mismatch: got ${actual.length}, expected ${expected.length}\n` +
      `Actual: ${JSON.stringify(actual)}\nExpected: ${JSON.stringify(expected)}`
    )
  }
  for (let i = 0; i < actual.length; i++) {
    for (let j = 0; j < actual[i].length; j++) {
      if (actual[i][j] !== expected[i][j]) {
        throw new Error(
          `Mismatch at row ${i}, col ${j}: got '${actual[i][j]}', expected '${expected[i][j]}'\n` +
          `Actual row: ${JSON.stringify(actual[i])}\nExpected row: ${JSON.stringify(expected[i])}`
        )
      }
    }
  }
}
