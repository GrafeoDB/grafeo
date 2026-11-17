/**
 * Vitest spec runner for .gtest files.
 *
 * Discovers all .gtest files under tests/spec/, parses them, and creates
 * vitest tests that execute queries through the Node.js GrafeoDB bindings.
 */

import { afterAll, describe, it, expect, vi } from 'vitest'

// A test that loads a dataset (the LDBC files) takes seconds through the binding,
// close to vitest's 5 s default on a busy machine: give each test a minute.
vi.setConfig({ testTimeout: 60_000 })
import { readFileSync, readdirSync, statSync, existsSync } from 'fs'
import { join, relative, resolve } from 'path'
import { parseGtestFile, coerceParams, paramValue } from './parser.mjs'
import { assertRowsSorted, assertRowsOrdered, assertRowsWithPrecision, assertHash, describeError, assertGqlStatus, assertTerms, assertValues, assertSideEffects, observation } from './comparator.mjs'
import { SpecFailure, runMarked, runProbe, specRoot, summary, summaryLine } from './xfail.mjs'
import { matches as matchCell, parse as parseCell, toNeutral } from './tck_values.mjs'

// ---------------------------------------------------------------------------
// Import GrafeoDB. A binding that does not load fails the run: it must not pass
// as thousands of skipped tests.
// ---------------------------------------------------------------------------

const { GrafeoDB } = await import('../../../../crates/bindings/node/grafeo.js')

// ---------------------------------------------------------------------------
// Paths
// ---------------------------------------------------------------------------

const SPEC_DIR = resolve(import.meta.dirname, '..', '..')
// The .gtest files the run reads: tests/spec, or GRAFEO_SPEC_ROOT (the runner probes).
const SPEC_ROOT = specRoot(SPEC_DIR)
const DATASETS_DIR = join(SPEC_DIR, 'datasets')

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/** Recursively find all .gtest files. */
function findGtestFiles(dir) {
  const results = []
  if (!existsSync(dir)) return results
  for (const entry of readdirSync(dir)) {
    const full = join(dir, entry)
    const stat = statSync(full)
    if (stat.isDirectory()) {
      results.push(...findGtestFiles(full))
    } else if (entry.endsWith('.gtest')) {
      results.push(full)
    }
  }
  return results.sort()
}

/** Load a .setup file and execute each line as GQL. */
async function loadDataset(db, datasetName) {
  const setupPath = join(DATASETS_DIR, `${datasetName}.setup`)
  if (!existsSync(setupPath)) {
    throw new Error(`Dataset file not found: ${setupPath}`)
  }
  const content = readFileSync(setupPath, 'utf-8')
  for (const line of content.split(/\r?\n/)) {
    const trimmed = line.trim()
    if (!trimmed || trimmed.startsWith('#')) continue
    await db.execute(trimmed)
  }
}

/** Execute a query in the specified language. */
async function executeQuery(db, language, query, params) {
  switch (language) {
    case 'gql':
    case '':
      return params ? db.execute(query, params) : db.execute(query)
    case 'cypher':
      if (!db.executeCypher) throw new Error('Cypher not available')
      return params ? db.executeCypher(query, params) : db.executeCypher(query)
    case 'gremlin':
      if (!db.executeGremlin) throw new Error('Gremlin not available')
      return params ? db.executeGremlin(query, params) : db.executeGremlin(query)
    case 'graphql':
      if (!db.executeGraphql) throw new Error('GraphQL not available')
      return params ? db.executeGraphql(query, params) : db.executeGraphql(query)
    case 'sparql':
      if (!db.executeSparql) throw new Error('SPARQL not available')
      return params ? db.executeSparql(query, params) : db.executeSparql(query)
    case 'sql-pgq':
    case 'sql_pgq':
      if (!db.executeSql) throw new Error('SQL/PGQ not available')
      return params ? db.executeSql(query, params) : db.executeSql(query)
    default:
      if (db.executeLanguage) return db.executeLanguage(language, query)
      throw new Error(`Unsupported language: ${language}`)
  }
}

const RUNNER_CAPABILITIES = new Set(['int64-safe'])

/** Cached set of compiled feature flags, populated on first use. */
let _features = null
function getFeatures(db) {
  if (!_features) {
    const info = db.info()
    if (!info.features) throw new Error('db.info() lists no features')
    _features = new Set(info.features)
  }
  return _features
}

/** Check if a language or feature requirement is available. */
function isAvailable(db, requirement) {
  if (requirement === 'gql' || requirement === '') return true
  const key = requirement.replace(/_/g, '-')
  // Compound language keys: "graphql-rdf" requires both "graphql" and "rdf"
  if (key === 'graphql-rdf') {
    const f = getFeatures(db)
    return f.has('graphql') && f.has('rdf')
  }
  return getFeatures(db).has(key) || RUNNER_CAPABILITIES.has(key)
}

// ---------------------------------------------------------------------------
// Parameter typing (shared with the WASM runner). A .gtest file cannot pass a
// number past the f64 range (the Rust runner refuses it when it builds), so
// that case is tested here.
// ---------------------------------------------------------------------------

describe('parameter values', () => {
  it('keeps a JSON integer that fits 64 bits exact', () => {
    expect(paramValue('[3, -5000000000, 9007199254740993, 9223372036854775807]')).toEqual([
      3,
      -5000000000n,
      9007199254740993n,
      9223372036854775807n,
    ])
    expect(paramValue('{"min": -9223372036854775808}')).toEqual({ min: -9223372036854775808n })
  })

  it('passes a JSON integer past 64 bits as a BigInt, which the binding types', () => {
    expect(paramValue('[9223372036854775808, {"big": 18446744073709551616}]')).toEqual([
      9223372036854775808n,
      { big: 18446744073709551616n },
    ])
    expect(paramValue(`1${'0'.repeat(38)}`)).toBe(10n ** 38n)
  })

  it('refuses a JSON number past the f64 range', () => {
    for (const text of ['[1e400]', '{"tiny": [-1e400]}', `[1${'0'.repeat(400)}.0]`]) {
      expect(() => paramValue(text)).toThrow('out of the f64 range')
    }
  })

  it('refuses a bare number past the f64 range', () => {
    expect(() => paramValue('1e999')).toThrow('out of the f64 range')
  })
})

// ---------------------------------------------------------------------------
// Discover and register tests
// ---------------------------------------------------------------------------

const gtestFiles = findGtestFiles(SPEC_ROOT)

// One line with the run's skipped and expected-failure counts, written past vitest's
// console capture, which keeps a passing run's logs back.
afterAll(() => process.stdout.write(`${summaryLine()}
`))

for (const filePath of gtestFiles) {
  // Skip runner directories
  if (filePath.includes('runners')) continue

  const relPath = relative(SPEC_ROOT, filePath).replace(/\\/g, '/')
  let parsed

  try {
    parsed = parseGtestFile(filePath)
  } catch (err) {
    describe(relPath, () => {
      it('should parse without errors', () => {
        throw new Error(`Parse error: ${err.message}`)
      })
    })
    continue
  }

  const { meta, tests } = parsed

  describe(relPath, () => {
    for (const tc of tests) {
      // Handle rosetta variants
      if (tc.variants && Object.keys(tc.variants).length > 0) {
        for (const [lang, query] of Object.entries(tc.variants)) {
          it(`${tc.name}_${lang}`, async (ctx) => {
            if (tc.skip || tc.runnerGap?.runners.includes('node')) {
              summary.skipped++
              return ctx.skip()
            }
            const db = GrafeoDB.create()
            try {
              if (!isAvailable(db, lang)) return ctx.skip()
              // Check the file's and the test's requires, as single tests do
              for (const req of [...(meta.requires || []), ...(tc.requires || [])]) {
                if (!isAvailable(db, req)) return ctx.skip()
              }
              const effectiveDataset = tc.dataset || meta.dataset
              // The dataset loads inside the case, as in the other runners: a binding error there
              // is the case's failure, which an `xfail:` absorbs; a missing file is the runner's.
              await runMarked(tc, async () => {
                if (effectiveDataset && effectiveDataset !== 'empty') {
                  await loadDataset(db, effectiveDataset)
                }
                await runTestCase(db, { ...tc, query }, lang, meta.language || 'gql', meta.valuesDialect)
              })
            } finally {
              db.close()
            }
          })
        }
        continue
      }

      it(tc.name, async (ctx) => {
        // Skip by field
        // A gap of this binding (`runner_gap:`) skips the case here, as `skip:` does
        if (tc.skip || tc.runnerGap?.runners.includes('node')) {
          summary.skipped++
          return ctx.skip()
        }

        const db = GrafeoDB.create()
        try {
          // Check language availability (file-level and per-test)
          if (!isAvailable(db, meta.language)) return ctx.skip()
          if (tc.language && !isAvailable(db, tc.language)) return ctx.skip()

          // Check requires: skip if the compiled build lacks the feature
          for (const req of meta.requires) {
            if (!isAvailable(db, req)) return ctx.skip()
          }

          // Check per-test requires
          for (const req of (tc.requires || [])) {
            if (!isAvailable(db, req)) return ctx.skip()
          }

          // Load dataset (per-test override takes priority)
          const effectiveDataset = tc.dataset || meta.dataset
          // The dataset loads inside the case, as in the other runners: a binding error there
          // is the case's failure, which an `xfail:` absorbs; a missing file is the runner's.
          await runMarked(tc, async () => {
            if (effectiveDataset && effectiveDataset !== 'empty') {
              await loadDataset(db, effectiveDataset)
            }
            await runTestCase(db, tc, tc.language || meta.language, meta.language || 'gql', meta.valuesDialect)
          })
        } finally {
          db.close()
        }
      })
    }
  })
}

const TEMPORAL_TAGS = ['$date', '$time', '$duration', '$zoned_datetime']

/** Whether the Node binding gave `v` for a temporal: a tag object, or a Date. */
function isTemporal(v) {
  if (v instanceof Date) return true
  if (v === null || typeof v !== 'object' || Array.isArray(v)) return false
  const keys = Object.keys(v)
  return keys.length === 1 && TEMPORAL_TAGS.includes(keys[0])
}

/** `v` with each temporal in it written as its Cypher `toString()`, the engine's own text. */
async function temporalsAsText(db, v) {
  if (isTemporal(v)) {
    const result = await db.executeCypher('RETURN toString($v) AS s', { v })
    return result.toArray()[0].s
  }
  if (Array.isArray(v)) return Promise.all(v.map((x) => temporalsAsText(db, x)))
  if (v !== null && typeof v === 'object' && !(v instanceof Date)) {
    const out = {}
    for (const [k, x] of Object.entries(v)) out[k] = await temporalsAsText(db, x)
    return out
  }
  return v
}

/** What the graph holds, as the openCypher TCK observes side effects. */
async function observeGraph(db) {
  const values = async (query) => (await db.execute(query)).toArray().map((row) => Object.values(row)[0])
  return observation(await values('MATCH (n) RETURN n'), await values('MATCH ()-[r]->() RETURN r'))
}

/** Execute a single test case and assert the expected result. */
async function runTestCase(db, tc, language, setupLanguage, valuesDialect) {
  runProbe(tc)
  // GQL statements first (the TCK's test procedures), then the setup queries in the file's
  // declared language (not the variant language)
  for (const statement of tc.setupGql) {
    await executeQuery(db, 'gql', statement)
  }
  for (const setupQ of tc.setup) {
    await executeQuery(db, setupLanguage || language, setupQ)
  }

  const exp = tc.expect

  // Coerce params (only applied to the final query)
  const params = coerceParams(tc.params)

  // Determine queries
  const queries = tc.statements.length > 0 ? tc.statements : (tc.query || exp.error != null || exp.gqlStatus != null) ? [tc.query ?? ''] : []
  if (queries.length === 0) throw new Error(`No query or statements in test '${tc.name}'`)

  // The graph before the statements, when their side effects are checked
  const before = exp.sideEffects ? await observeGraph(db) : null

  // Error case: execute all-but-last normally, only last should fail
  if (exp.error != null || exp.gqlStatus != null) {
    for (let i = 0; i < queries.length - 1; i++) {
      await executeQuery(db, language, queries[i], params)
    }
    let failure = null
    try {
      await executeQuery(db, language, queries[queries.length - 1], params)
    } catch (err) {
      failure = err
    }
    if (failure === null) {
      throw new SpecFailure(`Expected error ${describeError(exp)} but query succeeded`)
    }
    if (exp.error != null) expect(failure.message || String(failure)).toContain(exp.error)
    if (exp.gqlStatus != null) assertGqlStatus(failure, exp.gqlStatus)
    if (before) assertSideEffects(before, await observeGraph(db), exp.sideEffects)
    return
  }

  // Execute all queries, capture last result
  let result
  for (let i = 0; i < queries.length; i++) {
    const isLast = i === queries.length - 1
    result = await executeQuery(db, language, queries[i], params)
  }

  if (before) assertSideEffects(before, await observeGraph(db), exp.sideEffects)

  // The warnings, also when none are expected
  expect(result.warnings.map(w => w.code)).toEqual(exp.warnings)

  // Typed rows in the TCK's value notation
  if (exp.values != null) {
    const toText = valuesDialect === 'tck' ? (v) => temporalsAsText(db, v) : async (v) => v
    await assertValues(result, exp.values, exp.ordered, exp.unorderedLists, toText)
    return
  }

  // Column assertion (checked before value assertions)
  if (exp.columns && exp.columns.length > 0) {
    const actualCols = result.columns ? [...result.columns] : []
    expect(actualCols).toEqual(exp.columns)
  }

  // Empty check
  if (exp.empty) {
    expect(result.length).toBe(0)
    return
  }

  // Count check
  if (exp.count !== null && exp.count !== undefined) {
    expect(result.length).toBe(exp.count)
    return
  }

  // Hash check
  if (exp.hash) {
    assertHash(result, exp.hash)
    return
  }

  // Terms check
  if (exp.terms.length > 0) {
    assertTerms(result, exp.terms, exp.ordered)
    return
  }

  // Rows check
  if (exp.rows.length > 0) {
    if (exp.precision !== null && exp.precision !== undefined) {
      assertRowsWithPrecision(result, exp.rows, exp.precision)
    } else if (exp.ordered) {
      assertRowsOrdered(result, exp.rows)
    } else {
      assertRowsSorted(result, exp.rows)
    }
  }
}

// ---------------------------------------------------------------------------
// The shared value cases (tests/spec/runners/values_cases.json): this runner's
// `values:` comparison, temporal round trips and parameter typing agree with
// every other runner's, a gap of this binding giving the opposite verdict.
// ---------------------------------------------------------------------------

const CASE_FILE = JSON.parse(readFileSync(join(import.meta.dirname, '..', 'values_cases.json'), 'utf-8'))

/** A neutral value as the Node binding returns it. */
async function nativeValue(db, actual) {
  if (actual === null || typeof actual === 'boolean') return actual
  if (Array.isArray(actual)) return Promise.all(actual.map((a) => nativeValue(db, a)))
  const [[kind, inner]] = Object.entries(actual)
  const props = async (p) => Object.fromEntries(await Promise.all(Object.entries(p).map(async ([k, v]) => [k, await nativeValue(db, v)])))
  switch (kind) {
    case 'int': { const n = BigInt(inner); return n >= -(2n ** 53n) && n <= 2n ** 53n ? Number(n) : n }
    case 'float': return { NaN, Inf: Infinity, '-Inf': -Infinity }[inner] ?? Number(inner)
    case 'string': return inner
    case 'map': return props(inner)
    case 'node': return { _id: inner.id, _labels: inner.labels, ...(await props(inner.properties)) }
    case 'relationship':
      return { _id: inner.id, _type: inner.type, _source: inner.source, _target: inner.target, ...(await props(inner.properties)) }
    case 'path':
      return {
        nodes: await Promise.all(inner.nodes.map((n) => nativeValue(db, { node: n }))),
        edges: await Promise.all(inner.relationships.map((r) => nativeValue(db, { relationship: r }))),
      }
    case 'cypher': return (await db.executeCypher(`RETURN ${inner} AS v`)).toArray()[0].v
    default: throw new Error(`unknown neutral kind ${kind}`)
  }
}

describe('values_cases.json', () => {
  for (const c of CASE_FILE.cases) {
    it(`${c.cell} vs ${JSON.stringify(c.actual)}`, async () => {
      const db = GrafeoDB.create()
      try {
        let value = await nativeValue(db, c.actual)
        if (c.dialect === 'tck') value = await temporalsAsText(db, value)
        const verdict = matchCell(parseCell(c.cell), toNeutral(value), { unorderedLists: !!c.unordered_lists })
        // A known limitation, or a gap of this binding, gives the opposite verdict until fixed
        const gap = 'limitation' in c || 'node' in (c.runner_gaps ?? {})
        expect(verdict).toBe(c.match !== gap)
      } finally {
        db.close()
      }
    })
  }
  for (const trip of CASE_FILE.round_trips) {
    it(`round trip ${trip.cypher}`, async () => {
      const db = GrafeoDB.create()
      try {
        const text = await temporalsAsText(db, await nativeValue(db, { cypher: trip.cypher }))
        expect(text === trip.text).toBe(!('node' in trip.runner_gaps))
      } finally {
        db.close()
      }
    })
  }
  for (const p of CASE_FILE.params) {
    it(`param ${p.param}`, () => {
      expect(paramValue(p.param)).toEqual(p.value === null ? null : p.value.string)
    })
  }
})
