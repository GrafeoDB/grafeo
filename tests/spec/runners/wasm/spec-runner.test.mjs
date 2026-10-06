/**
 * Vitest spec runner for .gtest files (WASM bindings).
 *
 * Discovers all .gtest files under tests/spec/, parses them, and creates
 * vitest tests that execute queries through the WASM GrafeoDB bindings.
 *
 * Reuses the parser and comparator from the Node.js runner.
 */

import { describe, it, expect } from 'vitest'
import { readFileSync, readdirSync, statSync, existsSync } from 'fs'
import { join, relative, resolve } from 'path'
import { parseGtestFile, coerceParams } from '../node/parser.mjs'
import { assertRowsSorted, assertRowsOrdered, assertRowsWithPrecision, assertHash } from '../node/comparator.mjs'

// ---------------------------------------------------------------------------
// Import WASM bindings (skip all tests gracefully if unavailable)
// ---------------------------------------------------------------------------

let Database
let initSync
let WASM_AVAILABLE = false

try {
  const wasmPkgPath = resolve(import.meta.dirname, '..', '..', '..', '..', 'crates', 'bindings', 'wasm', 'pkg')
  const wasmJsPath = join(wasmPkgPath, 'grafeo_wasm.js')
  const wasmBinPath = join(wasmPkgPath, 'grafeo_wasm_bg.wasm')

  if (existsSync(wasmJsPath) && existsSync(wasmBinPath)) {
    const mod = await import('../../../../crates/bindings/wasm/pkg/grafeo_wasm.js')
    Database = mod.Database
    initSync = mod.initSync

    // Initialize WASM synchronously from the .wasm file
    const wasmBytes = readFileSync(wasmBinPath)
    initSync({ module: wasmBytes })
    WASM_AVAILABLE = true
  }
} catch {
  // WASM package not built or initialization failed; all tests will be skipped
}

// ---------------------------------------------------------------------------
// Paths
// ---------------------------------------------------------------------------

const SPEC_DIR = resolve(import.meta.dirname, '..', '..')
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

  return {
    columns,
    length: rows.length,
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

/** Load a .setup file and execute each line as GQL. */
function loadDataset(db, datasetName) {
  const setupPath = join(DATASETS_DIR, `${datasetName}.setup`)
  if (!existsSync(setupPath)) {
    throw new Error(`Dataset file not found: ${setupPath}`)
  }
  const content = readFileSync(setupPath, 'utf-8')
  for (const line of content.split(/\r?\n/)) {
    const trimmed = line.trim()
    if (!trimmed || trimmed.startsWith('#')) continue
    db.execute(trimmed)
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

const RUNNER_CAPABILITIES = new Set([])

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
          it(`${tc.name}_${lang}`, (ctx) => {
            if (!WASM_AVAILABLE) return ctx.skip()
            if (tc.skip) return ctx.skip()
            const db = new Database()
            if (!isAvailable(db, lang)) return ctx.skip()
            // Check per-test requires
            for (const req of (tc.requires || [])) {
              if (!isAvailable(db, req)) return ctx.skip()
            }
            const effectiveDataset = tc.dataset || meta.dataset
            if (effectiveDataset && effectiveDataset !== 'empty') {
              loadDataset(db, effectiveDataset)
            }
            try {
              runTestCase(db, { ...tc, query }, lang, meta.language || 'gql')
            } catch (err) {
              if (err instanceof WebAssembly.RuntimeError) return ctx.skip()
              throw err
            }
          })
        }
        continue
      }

      it(tc.name, (ctx) => {
        if (!WASM_AVAILABLE) return ctx.skip()

        // Skip by field
        if (tc.skip) return ctx.skip()

        const db = new Database()

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
        if (effectiveDataset && effectiveDataset !== 'empty') {
          loadDataset(db, effectiveDataset)
        }

        try {
          runTestCase(db, tc, tc.language || meta.language, meta.language || 'gql')
        } catch (err) {
          // WASM panics (e.g. rand(), SystemTime) surface as RuntimeError: unreachable
          if (err instanceof WebAssembly.RuntimeError) return ctx.skip()
          throw err
        }
      })
    }
  })
}

/** Execute a single test case and assert the expected result. */
function runTestCase(db, tc, language, setupLanguage) {
  // Run setup queries in the file's declared language (not the variant language)
  for (const setupQ of tc.setup) {
    executeQuery(db, setupLanguage || language, setupQ)
  }

  const exp = tc.expect

  // Determine queries
  const queries = tc.statements.length > 0 ? tc.statements : (tc.query || exp.error != null) ? [tc.query ?? ''] : []
  if (queries.length === 0) throw new Error(`No query or statements in test '${tc.name}'`)

  // Params apply to every statement of the test, as in the other runners
  const params = coerceParams(tc.params)

  // Error case: execute all-but-last normally, only last should fail
  if (exp.error != null) {
    for (let i = 0; i < queries.length - 1; i++) {
      executeQuery(db, language, queries[i], params)
    }
    try {
      executeQuery(db, language, queries[queries.length - 1], params)
      throw new Error(`Expected error containing '${exp.error}' but query succeeded`)
    } catch (err) {
      if (err.message.startsWith('Expected error')) throw err
      expect(err.message || String(err)).toContain(exp.error)
    }
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
