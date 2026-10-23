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
    // Decimal: {$decimal: '1.50'} -> its text, as Rust prints it
    if (keys.length === 1 && typeof val.$decimal === 'string') return val.$decimal
    // RDF/JS term: its lexical form, a blank node as `_:` and its label
    if (typeof val.termType === 'string' && typeof val.value === 'string') {
      return val.termType === 'BlankNode' ? `_:${val.value}` : val.value
    }
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

/**
 * What an error case expects, for its failure message.
 */
export function describeError(expect) {
  const parts = []
  if (expect.error != null) parts.push(`containing '${expect.error}'`)
  if (expect.gqlStatus != null) parts.push(`with GQLSTATUS ${expect.gqlStatus}`)
  return parts.join(' ')
}

/**
 * Assert that an error carries the GQLSTATUS `expected` (its `gqlStatus`). An
 * error without one fails the case: a status the binding does not expose is a
 * gap, not a pass.
 */
export function assertGqlStatus(err, expected) {
  const status = err == null ? undefined : err.gqlStatus
  if (status == null) {
    throw new Error(`Error '${err?.message ?? String(err)}' has no GQLSTATUS`)
  }
  if (String(status) !== expected) {
    throw new Error(`Error '${err.message}' has GQLSTATUS ${status}, expected ${expected}`)
  }
}

// ---------------------------------------------------------------------------
// RDF terms (mirrors term_to_string, normalize_expected_term and assert_terms
// in grafeo-spec-tests/src/lib.rs)
// ---------------------------------------------------------------------------

const XSD = 'http://www.w3.org/2001/XMLSchema#'
const RDF = 'http://www.w3.org/1999/02/22-rdf-syntax-ns#'
const ESCAPES = { '\\': '\\\\', '"': '\\"', '\n': '\\n', '\r': '\\r', '\t': '\\t' }

/** A literal's text as N-Triples writes it, between the quotes. */
function escapeLexical(text) {
  return [...text].map((c) => ESCAPES[c] ?? c).join('')
}

/**
 * A result cell as the N-Triples form of the RDF/JS term it holds: `<iri>`,
 * `_:label`, `"text"` (an xsd:string), `"text"@tag` or `"text"^^<datatype>`;
 * `null` when unbound. Anything that is not an RDF/JS term fails: a `terms:`
 * expectation compares terms only.
 */
export function termToString(val) {
  if (val === null || val === undefined) return 'null'
  switch (val.termType) {
    case 'NamedNode':
      return `<${val.value}>`
    case 'BlankNode':
      return `_:${val.value}`
    case 'Literal': {
      const text = escapeLexical(val.value)
      if (val.language) return `"${text}"@${val.language.toLowerCase()}`
      const datatype = val.datatype?.value ?? `${XSD}string`
      if (datatype === `${XSD}string`) return `"${text}"`
      return `"${text}"^^<${datatype}>`
    }
    default:
      throw new Error(`a terms: expectation got a value that is not an RDF term: ${JSON.stringify(val)}`)
  }
}

/**
 * An expected term in the form `termToString` gives: an `xsd:` or `rdf:`
 * datatype expanded, an explicit xsd:string dropped and a language tag
 * lowercased.
 */
export function normalizeExpectedTerm(expected) {
  if (expected === null || expected === undefined || expected === 'null') return 'null'
  if (!expected.startsWith('"')) return expected
  const end = expected.lastIndexOf('"')
  if (end === 0) return expected
  const text = expected.slice(1, end)
  const suffix = expected.slice(end + 1)
  if (suffix.startsWith('@')) return `"${text}"@${suffix.slice(1).toLowerCase()}`
  if (suffix.startsWith('^^')) {
    const datatype = suffix.slice(2)
    let iri
    if (datatype.startsWith('xsd:')) iri = XSD + datatype.slice(4)
    else if (datatype.startsWith('rdf:')) iri = RDF + datatype.slice(4)
    else iri = datatype.replace(/^</, '').replace(/>$/, '')
    if (iri === `${XSD}string`) return `"${text}"`
    return `"${text}"^^<${iri}>`
  }
  return expected
}

/**
 * Assert that each cell holds the exact RDF term `expected` names, in order
 * when `ordered`, as a multiset otherwise. A blank node matches a blank node;
 * in order, the labels must also correspond one to one.
 */
export function assertTerms(result, expected, ordered) {
  const columns = result.columns
  const actual = result.toArray().map((row) => columns.map((col) => termToString(row[col])))
  const want = expected.map((row) => row.map(normalizeExpectedTerm))
  const withoutLabels = (rows) => rows.map((row) => row.map((cell) => (cell.startsWith('_:') ? '_:' : cell)))
  let got = withoutLabels(actual)
  let exp = withoutLabels(want)
  if (!ordered) {
    const key = (row) => JSON.stringify(row)
    got = got.map(key).sort()
    exp = exp.map(key).sort()
  }
  if (JSON.stringify(got) !== JSON.stringify(exp)) {
    throw new Error(`terms differ\nactual:   ${JSON.stringify(actual)}\nexpected: ${JSON.stringify(want)}`)
  }
  if (ordered) {
    const pairs = new Map()
    const back = new Map()
    actual.forEach((row, i) => {
      row.forEach((a, j) => {
        if (!a.startsWith('_:')) return
        const e = want[i][j]
        if ((pairs.get(e) ?? a) !== a) throw new Error(`${e} names two blank nodes\nactual: ${JSON.stringify(actual)}`)
        if ((back.get(a) ?? e) !== e) throw new Error(`blank node ${a} has two names\nactual: ${JSON.stringify(actual)}`)
        pairs.set(e, a)
        back.set(a, e)
      })
    })
  }
}
