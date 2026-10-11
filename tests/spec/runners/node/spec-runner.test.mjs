/**
 * Vitest spec runner for .gtest files.
 *
 * Discovers all .gtest files under tests/spec/, parses them, and creates
 * vitest tests that execute queries through the Node.js GrafeoDB bindings.
 */

import { describe, it, expect, vi } from 'vitest'

// A test that loads a dataset (the LDBC files) takes seconds through the binding,
// close to vitest's 5 s default on a busy machine: give each test a minute.
vi.setConfig({ testTimeout: 60_000 })
import { readFileSync, readdirSync, statSync, existsSync } from 'fs'
import { join, relative, resolve } from 'path'
import { parseGtestFile, coerceParams, paramValue } from './parser.mjs'
import { assertRowsSorted, assertRowsOrdered, assertRowsWithPrecision, assertHash, resultToRows } from './comparator.mjs'

// ---------------------------------------------------------------------------
// Import GrafeoDB (skip all tests gracefully if unavailable)
// ---------------------------------------------------------------------------

let GrafeoDB
let GRAFEO_AVAILABLE = false

try {
  const mod = await import('../../../../crates/bindings/node/index.js')
  GrafeoDB = mod.GrafeoDB
  GRAFEO_AVAILABLE = true
} catch {
  // Bindings not built; all tests will be skipped
}

// ---------------------------------------------------------------------------
// Paths
// ---------------------------------------------------------------------------

const SPEC_DIR = resolve(import.meta.dirname, '..', '..')
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
    _features = new Set(info.features || [])
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

  it('makes a JSON integer past 64 bits a float', () => {
    expect(paramValue('[9223372036854775808, {"big": 18446744073709551616}]')).toEqual([
      9223372036854775808,
      { big: 18446744073709551616 },
    ])
  })

  it('refuses a JSON number past the f64 range', () => {
    for (const text of ['[1e400]', '{"tiny": [-1e400]}', `[1${'0'.repeat(400)}]`]) {
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

const gtestFiles = findGtestFiles(SPEC_DIR)

for (const filePath of gtestFiles) {
  // Skip runner directories
  if (filePath.includes('runners')) continue

  const relPath = relative(SPEC_DIR, filePath).replace(/\\/g, '/')
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
            if (!GRAFEO_AVAILABLE) return ctx.skip()
            if (tc.skip) return ctx.skip()
            const db = GrafeoDB.create()
            try {
              if (!isAvailable(db, lang)) return ctx.skip()
              // Check the file's and the test's requires, as single tests do
              for (const req of [...(meta.requires || []), ...(tc.requires || [])]) {
                if (!isAvailable(db, req)) return ctx.skip()
              }
              const effectiveDataset = tc.dataset || meta.dataset
              if (effectiveDataset && effectiveDataset !== 'empty') {
                await loadDataset(db, effectiveDataset)
              }
              await runTestCase(db, { ...tc, query }, lang, meta.language || 'gql')
            } finally {
              db.close()
            }
          })
        }
        continue
      }

      it(tc.name, async (ctx) => {
        if (!GRAFEO_AVAILABLE) return ctx.skip()

        // Skip by field
        if (tc.skip) return ctx.skip()

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
          if (effectiveDataset && effectiveDataset !== 'empty') {
            await loadDataset(db, effectiveDataset)
          }

          await runTestCase(db, tc, tc.language || meta.language, meta.language || 'gql')
        } finally {
          db.close()
        }
      })
    }
  })
}

/** Execute a single test case and assert the expected result. */
async function runTestCase(db, tc, language, setupLanguage) {
  // Run setup queries in the file's declared language (not the variant language)
  for (const setupQ of tc.setup) {
    await executeQuery(db, setupLanguage || language, setupQ)
  }

  const exp = tc.expect

  // Coerce params (only applied to the final query)
  const params = coerceParams(tc.params)

  // Determine queries
  const queries = tc.statements.length > 0 ? tc.statements : (tc.query || exp.error != null) ? [tc.query ?? ''] : []
  if (queries.length === 0) throw new Error(`No query or statements in test '${tc.name}'`)

  // Error case: execute all-but-last normally, only last should fail
  if (exp.error != null) {
    for (let i = 0; i < queries.length - 1; i++) {
      await executeQuery(db, language, queries[i], params)
    }
    try {
      await executeQuery(db, language, queries[queries.length - 1], params)
      throw new Error(`Expected error containing '${exp.error}' but query succeeded`)
    } catch (err) {
      if (err.message.startsWith('Expected error')) throw err
      expect(err.message || String(err)).toContain(exp.error)
    }
    return
  }

  // Execute all queries, capture last result
  let result
  for (let i = 0; i < queries.length; i++) {
    const isLast = i === queries.length - 1
    result = await executeQuery(db, language, queries[i], params)
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
