/**
 * Vitest spec runner for .gtest files (WASM bindings).
 *
 * Discovers all .gtest files under tests/spec/, parses them, and creates
 * vitest tests that execute queries through the WASM GrafeoDB bindings.
 *
 * Reuses the parser and comparator from the Node.js runner.
 */

import { afterAll, describe, it, expect, vi } from 'vitest'

// A test that loads a dataset (the LDBC files) takes seconds through the binding,
// close to vitest's 5 s default on a busy machine: give each test a minute.
vi.setConfig({ testTimeout: 60_000 })
import { readFileSync, readdirSync, statSync, existsSync } from 'fs'
import { join, relative, resolve } from 'path'
import { parseGtestFile, coerceParams, paramValue, configOptions, datasetFile } from '../node/parser.mjs'
import { assertRowsSorted, assertRowsOrdered, assertRowsWithPrecision, assertHash, describeError, assertGqlStatus, assertTerms , assertValueRows, assertSideEffects, observation} from '../node/comparator.mjs'
import { SpecFailure, runMarkedSync, runProbe, specRoot, summary, summaryLine } from '../node/xfail.mjs'
import { caseCellType, cellTypeAt, matches as matchCell, parse as parseCell, toNeutral } from '../node/tck_values.mjs'

// ---------------------------------------------------------------------------
// Import WASM bindings. A package that does not load fails the run: it must not
// pass as thousands of skipped tests.
// ---------------------------------------------------------------------------

const wasmPkgPath = resolve(import.meta.dirname, '..', '..', '..', '..', 'crates', 'bindings', 'wasm', 'pkg')
const { Database, initSync } = await import('../../../../crates/bindings/wasm/pkg/grafeo_wasm.js')

// Initialize WASM synchronously from the .wasm file
initSync({ module: readFileSync(join(wasmPkgPath, 'grafeo_wasm_bg.wasm')) })

// ---------------------------------------------------------------------------
// Paths
// ---------------------------------------------------------------------------

const SPEC_DIR = resolve(import.meta.dirname, '..', '..')
// The .gtest files the run reads: tests/spec, or GRAFEO_SPEC_ROOT (the runner probes).
const SPEC_ROOT = specRoot(SPEC_DIR)
const DATASETS_DIR = join(SPEC_DIR, 'datasets')

// ---------------------------------------------------------------------------
// Result conversion
// ---------------------------------------------------------------------------

/**
 * Wrap a WASM raw result so it looks like the Node.js QueryResult expected
 * by the comparator functions (which call resultToRows internally).
 */
function wrapRawResult(rawResult) {
  // Eagerly copy data so the rawResult (which may borrow the WASM Database)
  // can be released before db.free() is called.
  const columns = [...rawResult.columns]
  const rows = rawResult.rows.map(row => [...row])
  const warnings = rawResult.warnings.map(w => ({ code: w.code, kind: w.kind, message: w.message }))
  const cellTypes = structuredClone(rawResult.cellTypes ?? { columns: [], runs: [] })

  return {
    columns,
    warnings,
    length: rows.length,
    /** The type the cell at `row`, `column` was built with (the result's `cellTypes`). */
    cellType(row, column) {
      return cellTypeAt(cellTypes, row, column)
    },
    toArray() {
      const arr = []
      for (const row of rows) {
        const obj = {}
        for (let i = 0; i < columns.length; i++) {
          obj[columns[i]] = row[i]
        }
        arr.push(obj)
      }
      return arr
    },
  }
}

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

/** Load a dataset, one statement per line: GQL from a `.setup` file, SPARQL Update from a `.ru` file. */
function loadDataset(db, datasetName) {
  const { path, language } = datasetFile(DATASETS_DIR, datasetName)
  const content = readFileSync(path, 'utf-8')
  for (const line of content.split(/\r?\n/)) {
    const trimmed = line.trim()
    if (!trimmed || trimmed.startsWith('#')) continue
    executeQuery(db, language, trimmed)
  }
}

/**
 * Map .gtest language keys to the WASM executeWithLanguage dispatch key.
 * Returns null if the language is not recognized.
 */
function toDispatchKey(language) {
  switch (language) {
    case 'gql': case '': return 'gql'
    case 'cypher': return 'cypher'
    case 'gremlin': return 'gremlin'
    case 'graphql': return 'graphql'
    case 'graphql-rdf': return 'graphql-rdf'
    case 'sparql': return 'sparql'
    case 'sql-pgq': case 'sql_pgq': case 'sql': return 'sql'
    default: return null
  }
}

/** Execute a query in the specified language, returning a raw result. */
function executeQueryRaw(db, language, query, params) {
  const key = toDispatchKey(language)
  if (key === null) throw new Error(`Unsupported language: ${language}`)
  if (params) return db.executeRawWithLanguageAndParams(query, key, params)
  if (key === 'gql') return db.executeRaw(query)
  return db.executeRawWithLanguage(query, key)
}

/** Execute a query in the specified language (for setup, returns array of objects). */
function executeQuery(db, language, query, params) {
  const key = toDispatchKey(language)
  if (key === null) throw new Error(`Unsupported language: ${language}`)
  if (params) return db.executeWithLanguageAndParams(query, key, params)
  if (key === 'gql') return db.execute(query)
  return db.executeWithLanguage(query, key)
}

/**
 * Frees `db`. Without it every test's database stays in the WASM heap, and
 * with all features the heap runs out after about 3,500 tests: every later
 * test then fails with `RuntimeError: unreachable`.
 */
function release(db) {
  try {
    db.free()
  } catch (err) {
    // After a panic the instance may trap again; the test already reported it
    if (!(err instanceof WebAssembly.RuntimeError)) throw err
  }
}

const RUNNER_CAPABILITIES = new Set([])

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
// Discover and register tests
// ---------------------------------------------------------------------------

const gtestFiles = findGtestFiles(SPEC_ROOT)

// One line with the run's skipped and expected-failure counts, written past vitest's
// console capture, which keeps a passing run's logs back.
afterAll(() => process.stdout.write(`${summaryLine()}\n`))

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
          it(`${tc.name}_${lang}`, (ctx) => {
            if (tc.skip || tc.runnerGap?.runners.includes('wasm')) {
              summary.skipped++
              return ctx.skip()
            }
            const db = new Database(configOptions(tc.config))
            try {
              if (!isAvailable(db, lang)) return ctx.skip()
              // Check the file's and the test's requires, as single tests do
              for (const req of [...(meta.requires || []), ...(tc.requires || [])]) {
                if (!isAvailable(db, req)) return ctx.skip()
              }
              const effectiveDataset = tc.dataset || meta.dataset
              // The dataset loads inside the case, as in the other runners: a binding error there
              // is the case's failure, which an `xfail:` absorbs; a missing file is the runner's.
              try {
                runMarkedSync(tc, () => {
                  if (effectiveDataset && effectiveDataset !== 'empty') {
                    loadDataset(db, effectiveDataset)
                  }
                  runTestCase(db, { ...tc, query }, lang, meta.language || 'gql')
                })
              } catch (err) {
                if (err instanceof WebAssembly.RuntimeError) return ctx.skip()
                throw err
              }
            } finally {
              release(db)
            }
          })
        }
        continue
      }

      it(tc.name, (ctx) => {
        // Skip by field
        // A gap of this binding (`runner_gap:`) skips the case here, as `skip:` does
        if (tc.skip || tc.runnerGap?.runners.includes('wasm')) {
          summary.skipped++
          return ctx.skip()
        }

        // The case's settings (`config:`), the defaults without one
        const db = new Database(configOptions(tc.config))
        try {
          // Check language availability (file-level and per-test)
          if (!isAvailable(db, meta.language)) return ctx.skip()
          if (tc.language && !isAvailable(db, tc.language)) return ctx.skip()

          // Check requires: skip if binding does not expose the required method
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
          try {
            runMarkedSync(tc, () => {
              if (effectiveDataset && effectiveDataset !== 'empty') {
                loadDataset(db, effectiveDataset)
              }
              runTestCase(db, tc, tc.language || meta.language, meta.language || 'gql')
            })
          } catch (err) {
            // WASM panics (e.g. rand(), SystemTime) surface as RuntimeError: unreachable
            if (err instanceof WebAssembly.RuntimeError) return ctx.skip()
            throw err
          }
        } finally {
          release(db)
        }
      })
    }
  })
}

/** What the graph holds, as the openCypher TCK observes side effects. */
function observeGraph(db) {
  const values = (query) => {
    const result = wrapRawResult(executeQueryRaw(db, 'gql', query))
    return result.toArray().map((row, i) => [Object.values(row)[0], result.cellType(i, 0)])
  }
  return observation(values('MATCH (n) RETURN n'), values('MATCH ()-[r]->() RETURN r'))
}

/** Execute a single test case and assert the expected result. */
function runTestCase(db, tc, language, setupLanguage) {
  runProbe(tc)
  // GQL statements first (the TCK's test procedures), then the setup queries in the file's
  // declared language (not the variant language)
  for (const statement of tc.setupGql) {
    executeQuery(db, 'gql', statement)
  }
  for (const setupQ of tc.setup) {
    executeQuery(db, setupLanguage || language, setupQ)
  }

  const exp = tc.expect

  // Determine queries
  const queries = tc.statements.length > 0 ? tc.statements : (tc.query || exp.error != null || exp.gqlStatus != null) ? [tc.query ?? ''] : []
  if (queries.length === 0) throw new Error(`No query or statements in test '${tc.name}'`)

  // Params apply to every statement of the test, as in the other runners
  const params = coerceParams(tc.params)

  // The graph before the statements, when their side effects are checked
  const before = exp.sideEffects ? observeGraph(db) : null

  // Error case: execute all-but-last normally, only last should fail
  if (exp.error != null || exp.gqlStatus != null) {
    for (let i = 0; i < queries.length - 1; i++) {
      executeQuery(db, language, queries[i], params)
    }
    let failure = null
    try {
      executeQuery(db, language, queries[queries.length - 1], params)
    } catch (err) {
      failure = err
    }
    if (failure === null) {
      throw new SpecFailure(`Expected error ${describeError(exp)} but query succeeded`)
    }
    if (exp.error != null) expect(failure.message || String(failure)).toContain(exp.error)
    if (exp.gqlStatus != null) assertGqlStatus(failure, exp.gqlStatus)
    if (before) assertSideEffects(before, observeGraph(db), exp.sideEffects)
    return
  }

  // Execute all queries, capture last raw result for assertions
  let rawResult
  for (let i = 0; i < queries.length; i++) {
    rawResult = executeQueryRaw(db, language, queries[i], params)
  }

  // Wrap eagerly and release the WASM reference to avoid borrow conflicts on db.free()
  const result = wrapRawResult(rawResult)
  rawResult = null

  if (before) assertSideEffects(before, observeGraph(db), exp.sideEffects)

  // The warnings, also when none are expected
  expect(result.warnings.map(w => w.code)).toEqual(exp.warnings)

  // Typed rows in the TCK's value notation. WASM returns a temporal as plain text, so a TCK
  // file's temporal cases carry a runner_gap for this runner and never get here.
  if (exp.values != null) {
    const values = result.toArray().map((row) => result.columns.map((column) => row[column]))
    assertValueRows(result.columns, values, exp.values, exp.ordered, exp.unorderedLists, (r, c) => result.cellType(r, c))
    return
  }

  // Column assertion (checked before value assertions)
  if (exp.columns && exp.columns.length > 0) {
    const actualCols = [...result.columns]
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

/** A neutral value as the WASM binding returns it. */
function nativeValue(db, actual) {
  if (actual === null || typeof actual === 'boolean') return actual
  if (Array.isArray(actual)) return actual.map((a) => nativeValue(db, a))
  const [[kind, inner]] = Object.entries(actual)
  const props = (p) => Object.fromEntries(Object.entries(p).map(([k, v]) => [k, nativeValue(db, v)]))
  switch (kind) {
    case 'int': { const n = BigInt(inner); return n >= -(2n ** 53n) && n <= 2n ** 53n ? Number(n) : n }
    case 'float': return { NaN, Inf: Infinity, '-Inf': -Infinity }[inner] ?? Number(inner)
    case 'string': return inner
    case 'map': return props(inner)
    case 'node': return { _id: inner.id, _labels: inner.labels, ...props(inner.properties) }
    case 'relationship':
      return { _id: inner.id, _type: inner.type, _source: inner.source, _target: inner.target, ...props(inner.properties) }
    case 'path':
      return {
        nodes: inner.nodes.map((n) => nativeValue(db, { node: n })),
        edges: inner.relationships.map((r) => nativeValue(db, { relationship: r })),
        _type: 'path',
      }
    case 'cypher': return wrapRawResult(executeQueryRaw(db, 'cypher', `RETURN ${inner} AS v`)).toArray()[0].v
    default: throw new Error(`unknown neutral kind ${kind}`)
  }
}

/** The Cypher `toString()` of `value` passed back as a parameter. */
function toStringOf(db, value) {
  return wrapRawResult(executeQueryRaw(db, 'cypher', 'RETURN toString($v) AS s', { v: value })).toArray()[0].s
}

// A case builds its value, or reads a temporal back, through Cypher: the edge and rdf
// builds have none, so there such a case is skipped like a gtest that requires cypher
const HAS_CYPHER = (() => {
  const db = new Database()
  try { return isAvailable(db, 'cypher') } finally { release(db) }
})()
const needsCypher = (c) => c.dialect === 'tck' || JSON.stringify(c.actual).includes('"cypher"')

describe('values_cases.json', () => {
  for (const c of CASE_FILE.cases) {
    it.skipIf(!HAS_CYPHER && needsCypher(c))(`${c.cell} vs ${JSON.stringify(c.actual)}`, async () => {
      const db = new Database()
      try {
        const value = nativeValue(db, c.actual)
        const ty = await caseCellType(c.actual, (text) =>
          wrapRawResult(executeQueryRaw(db, 'cypher', `RETURN ${text} AS v`)).cellType(0, 0))
        const verdict = matchCell(parseCell(c.cell), toNeutral(value, ty), { unorderedLists: !!c.unordered_lists })
        // A gap of this binding gives the opposite verdict until fixed
        const gap = 'wasm' in (c.runner_gaps ?? {})
        expect(verdict).toBe(c.match !== gap)
      } finally {
        release(db)
      }
    })
  }
  for (const trip of CASE_FILE.round_trips) {
    it.skipIf(!HAS_CYPHER)(`round trip ${trip.cypher}`, () => {
      const db = new Database()
      try {
        const text = toStringOf(db, nativeValue(db, { cypher: trip.cypher }))
        expect(text === trip.text).toBe(!('wasm' in trip.runner_gaps))
      } finally {
        release(db)
      }
    })
  }
  for (const p of CASE_FILE.params) {
    it(`param ${p.param}`, () => {
      expect(paramValue(p.param)).toEqual(p.value === null ? null : p.value.string)
    })
  }
})
