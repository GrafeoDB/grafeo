/**
 * Parse .gtest YAML files into structured test cases.
 *
 * Line-based parser (no YAML dependency) that mirrors the Rust build.rs parser
 * in crates/grafeo-spec-tests/build.rs.
 */

import { readFileSync } from 'fs'

/**
 * @typedef {Object} Meta
 * @property {string} language
 * @property {string} model
 * @property {string} section
 * @property {string} title
 * @property {string} dataset
 * @property {string[]} requires
 * @property {string[]} tags
 */

/**
 * @typedef {Object} Expect
 * @property {string[][]} rows
 * @property {boolean} ordered
 * @property {number|null} count
 * @property {boolean} empty
 * @property {string|null} error
 * @property {string|null} hash
 * @property {number|null} precision
 * @property {string[]} columns
 * @property {string[]} warnings The codes of the warnings the statement raises,
 *   in the result's order (by code): none without the key, so every test that
 *   runs to a result also checks that it raises no other warning.
 */

/**
 * @typedef {Object} TestCase
 * @property {string} name
 * @property {string|null} query
 * @property {string[]} statements
 * @property {string[]} setup
 * @property {Object<string,string>} params Each value as the file writes it,
 *   quotes included: whether it was quoted decides its type (see `paramValue`).
 * @property {string[]} tags
 * @property {string|null} skip
 * @property {{spec: string, issue: string}|null} xfail The spec and the issue a case expected
 *   to fail is held against (`xfail:`)
 * @property {string|null} probe A runner probe (`probe:`), only under `spec-probes`
 * @property {Expect} expect
 * @property {Object<string,string>} variants
 */

/**
 * @typedef {Object} GtestFile
 * @property {Meta} meta
 * @property {TestCase[]} tests
 */

// The number syntax of the Rust reference runner (crates/grafeo-spec-tests/
// build.rs): decimal only. `Number` alone would also read `0x1F`, `0b11`,
// `Infinity` and `''`, which the reference runner keeps as strings.
const DECIMAL_INTEGER = /^[+-]?[0-9]+$/
const DECIMAL_NUMBER = /^[+-]?([0-9]+(\.[0-9]*)?|\.[0-9]+)([eE][+-]?[0-9]+)?$/
// A small integer goes as a JS number, any other as a BigInt, which both
// bindings type by the integer rule: an INT64 when it fits 64 bits, else an
// exact DECIMAL up to 38 digits, and an error past that.
const NUMBER_INTEGER_MIN = -(2n ** 31n)
const NUMBER_INTEGER_MAX = 2n ** 32n - 1n

/**
 * The value of a parameter written as `text` in a .gtest file, by the rule of
 * the Rust reference runner that every runner follows: a quoted value is a
 * string, whatever it reads like; a bare value starting with `[` or `{` is
 * JSON, whose numbers follow the rule of bare ones; a bare decimal integer
 * is an integer (a BigInt past 32 bits, which a number would round), which
 * the binding types, and any other bare decimal number a float, an error
 * past the f64 range; a bare `true` or `false` is a boolean; any other
 * bare value is a string.
 */
export function paramValue(text) {
  text = text.trim()
  if (isQuoted(text)) return unquote(text)
  if (text.startsWith('[') || text.startsWith('{')) return JSON.parse(text, jsonNumber)
  if (DECIMAL_INTEGER.test(text) || DECIMAL_NUMBER.test(text)) return numberValue(text)
  if (text === 'true' || text === 'false') return text === 'true'
  // A bare `null` is null; a quoted 'null' is the string
  if (text === 'null') return null
  return text
}

/**
 * A decimal number written as `text`: an integer when it is one, of any
 * width (the binding types it), else a float, which must be finite.
 */
function numberValue(text) {
  if (DECIMAL_INTEGER.test(text)) {
    const integer = BigInt(text)
    return integer >= NUMBER_INTEGER_MIN && integer <= NUMBER_INTEGER_MAX ? Number(integer) : integer
  }
  const number = Number(text)
  if (!Number.isFinite(number)) throw new Error(`parameter ${text} is out of the f64 range`)
  return number
}

/**
 * `JSON.parse` reviver: a number is read from its source text by the rule of
 * a bare one, as `JSON.parse` alone rounds an integer past 2^53 and reads
 * `1e400` as Infinity.
 */
function jsonNumber(key, value, context) {
  if (typeof value !== 'number') return value
  if (context?.source === undefined) {
    throw new Error('a JSON parameter needs JSON.parse source text access (Node.js 22 or later)')
  }
  return numberValue(context.source)
}

/** Type each parameter by `paramValue`; undefined when there are none. */
export function coerceParams(rawParams) {
  if (!rawParams || Object.keys(rawParams).length === 0) return undefined
  const result = {}
  for (const [key, text] of Object.entries(rawParams)) {
    result[key] = paramValue(text)
  }
  return result
}

/** @param {string} filePath @returns {GtestFile} */
export function parseGtestFile(filePath) {
  const content = readFileSync(filePath, 'utf-8')
  const lines = content.split(/\r?\n/)
  const ctx = { lines, idx: 0 }

  skipBlankAndComments(ctx)
  const meta = parseMeta(ctx)
  skipBlankAndComments(ctx)
  const tests = parseTests(ctx)
  for (const tc of tests) checkMarks(tc, filePath)
  return { meta, tests }
}

/** The runner probes a `probe:` key may name. */
const PROBES = new Set(['runner_error'])

/**
 * The rules of the marks, as every runner checks them: an `xfail:` names its spec and its
 * issue, a case is never both `skip:` and `xfail:`, and `probe:` is for the runner probes only,
 * in a file under a `spec-probes` directory.
 */
function checkMarks(tc, filePath) {
  if (tc.xfail != null) {
    for (const key of ['spec', 'issue']) {
      if (!tc.xfail[key]?.trim()) throw new Error(`test '${tc.name}': an xfail names its \`${key}:\``)
    }
  }
  if (tc.skip != null && tc.xfail != null) {
    throw new Error(`test '${tc.name}': a case is either skipped or expected to fail, not both`)
  }
  if (tc.probe != null) {
    if (!filePath.split(/[\\/]/).includes('spec-probes')) {
      throw new Error(
        `test '${tc.name}': \`probe:\` is for the runner probes only, under a \`spec-probes\` directory, not in ${filePath}`,
      )
    }
    if (!PROBES.has(tc.probe)) throw new Error(`test '${tc.name}': unknown probe '${tc.probe}' (known: runner_error)`)
  }
}

/** The `spec:` and `issue:` under an `xfail:` at `ctx.lines[ctx.idx]`, checked by `checkMarks`. */
function parseXfail(ctx) {
  const indent = (line) => line.length - line.trimStart().length
  const own = indent(ctx.lines[ctx.idx])
  ctx.idx++
  // `passes_in:` names the runners where the case passes all the same and must
  const xfail = { spec: '', issue: '', passesIn: [] }
  while (ctx.idx < ctx.lines.length) {
    const line = ctx.lines[ctx.idx]
    const trimmed = line.trim()
    if (!trimmed || trimmed.startsWith('#')) { ctx.idx++; continue }
    if (indent(line) <= own) break
    const kv = parseKV(trimmed)
    if (kv && kv[0] === 'passes_in') {
      xfail.passesIn = parseYamlList(kv[1])
      const unknown = xfail.passesIn.find((r) => !RUNNERS.includes(r))
      if (unknown) throw new Error(`unknown runner '${unknown}' in passes_in`)
    } else if (!kv || (kv[0] !== 'spec' && kv[0] !== 'issue')) {
      throw new Error(`unknown xfail key at line ${ctx.idx + 1}: '${trimmed}'`)
    } else {
      xfail[kv[0]] = unquote(kv[1])
    }
    ctx.idx++
  }
  return xfail
}

// ---------------------------------------------------------------------------
// Meta block
// ---------------------------------------------------------------------------

function parseMeta(ctx) {
  const meta = {
    language: 'gql', model: '', section: '', title: '',
    dataset: 'empty', requires: [], tags: [], valuesDialect: 'strict',
  }
  expectLine(ctx, 'meta:')
  while (ctx.idx < ctx.lines.length) {
    skipBlankAndComments(ctx)
    if (ctx.idx >= ctx.lines.length) break
    const line = ctx.lines[ctx.idx]
    if (!line.startsWith(' ') && !line.startsWith('\t')) break
    const kv = parseKV(line.trim())
    if (!kv) { ctx.idx++; continue }
    const [key, value] = kv
    switch (key) {
      case 'language': meta.language = value; break
      case 'model': meta.model = value; break
      case 'section': meta.section = unquote(value); break
      case 'title': meta.title = value; break
      case 'dataset': meta.dataset = value; break
      case 'requires': meta.requires = parseYamlList(value); break
      case 'tags': meta.tags = parseYamlList(value); break
      case 'values_dialect':
        meta.valuesDialect = unquote(value)
        if (!['strict', 'tck'].includes(meta.valuesDialect)) {
          throw new Error(`unknown values_dialect '${meta.valuesDialect}' (known: strict, tck)`)
        }
        break
    }
    ctx.idx++
  }
  return meta
}

// ---------------------------------------------------------------------------
// Tests list
// ---------------------------------------------------------------------------

function parseTests(ctx) {
  skipBlankAndComments(ctx)
  expectLine(ctx, 'tests:')
  const tests = []
  while (ctx.idx < ctx.lines.length) {
    skipBlankAndComments(ctx)
    if (ctx.idx >= ctx.lines.length) break
    const trimmed = ctx.lines[ctx.idx].trim()
    if (trimmed.startsWith('- name:')) {
      tests.push(parseSingleTest(ctx))
    } else {
      break
    }
  }
  return tests
}

function parseSingleTest(ctx) {
  const tc = {
    name: '', query: null, statements: [], setup: [], setupGql: [],
    params: {}, tags: [], skip: null, xfail: null, runnerGap: null, config: {}, probe: null, language: null,
    expect: makeExpect(), variants: {}, requires: [],
  }

  // First line: "- name: xxx"
  const first = ctx.lines[ctx.idx].trim()
  const kv = parseKV(first.slice(2)) // strip "- "
  if (kv) tc.name = unquote(kv[1])
  ctx.idx++

  while (ctx.idx < ctx.lines.length) {
    const line = ctx.lines[ctx.idx]
    const trimmed = line.trim()
    if (trimmed.startsWith('#')) { ctx.idx++; continue }
    if (trimmed.startsWith('- name:')) break
    if (!trimmed) { ctx.idx++; continue }

    const kv2 = parseKV(trimmed)
    if (!kv2) { ctx.idx++; continue }
    const [key, value] = kv2
    switch (key) {
      case 'query':
        tc.query = value === '|' ? parseBlockScalar(ctx) : unquote(value)
        if (value !== '|') ctx.idx++
        break
      case 'skip':
        tc.skip = unquote(value); ctx.idx++; break
      case 'xfail':
        tc.xfail = parseXfail(ctx); break
      case 'probe':
        tc.probe = unquote(value); ctx.idx++; break
      case 'runner_gap':
        tc.runnerGap = parseRunnerGap(ctx); break
      case 'config':
        tc.config = parseConfig(value); ctx.idx++; break
      case 'setup':
        ctx.idx++; tc.setup = parseStringList(ctx); break
      case 'setup_gql':
        ctx.idx++; tc.setupGql = parseStringList(ctx); break
      case 'statements':
        ctx.idx++; tc.statements = parseStringList(ctx); break
      case 'language':
        tc.language = unquote(value); ctx.idx++; break
      case 'dataset':
        tc.dataset = unquote(value); ctx.idx++; break
      case 'tags':
        tc.tags = parseYamlList(value); ctx.idx++; break
      case 'requires':
        tc.requires = parseYamlList(value); ctx.idx++; break
      case 'params':
        ctx.idx++; tc.params = parseParams(ctx); break
      case 'expect':
        ctx.idx++; tc.expect = parseExpectBlock(ctx); break
      case 'variants':
        ctx.idx++; tc.variants = parseMap(ctx, 6); break
      default:
        ctx.idx++
    }
  }
  return tc
}

// ---------------------------------------------------------------------------
// Expect block
// ---------------------------------------------------------------------------

function makeExpect() {
  return {
    rows: [], ordered: false, count: null, empty: false, error: null, gqlStatus: null, hash: null,
    precision: null, columns: [], terms: [], warnings: [],
    values: null, unorderedLists: false, sideEffects: null,
  }
}

/** The runners a `runner_gap:` may name. */
export const RUNNERS = ['rust', 'python', 'node', 'wasm', 'go', 'csharp', 'dart']

/**
 * The keys a `config:` may set: Config settings every binding exposes, each with its option
 * name in the JavaScript bindings (Node.js `GrafeoDB.create` and the WASM `Database`).
 */
export const CONFIG_KEYS = { path_search_budget: 'pathSearchBudget' }

/**
 * A `config:` value as the file writes it: an inline map of `CONFIG_KEYS` to positive
 * integers (`{path_search_budget: 1048576}`). An unknown or repeated key is refused, never
 * ignored.
 */
export function parseConfig(text) {
  const inner = text.trim()
  if (!inner.startsWith('{') || !inner.endsWith('}') || !inner.slice(1, -1).trim()) {
    throw new Error(`config must be an inline map of settings, got '${text}'`)
  }
  const settings = {}
  for (const entry of inner.slice(1, -1).split(',')) {
    const colon = entry.indexOf(':')
    const key = colon < 0 ? '' : entry.slice(0, colon).trim()
    const value = colon < 0 ? '' : entry.slice(colon + 1).trim()
    if (!key) throw new Error(`config entry '${entry.trim()}' is not 'key: value'`)
    if (Object.hasOwn(settings, key)) throw new Error(`config sets '${key}' twice`)
    if (!Object.hasOwn(CONFIG_KEYS, key)) {
      throw new Error(`unknown config key '${key}' (known: ${Object.keys(CONFIG_KEYS).join(', ')})`)
    }
    if (!/^[0-9]+$/.test(value) || Number(value) === 0) {
      throw new Error(`config '${key}' needs a positive integer, got '${value}'`)
    }
    settings[key] = Number(value)
  }
  return settings
}

/** The JavaScript binding options of a case's parsed `config:`. */
export function configOptions(config) {
  return Object.fromEntries(Object.entries(config).map(([key, value]) => [CONFIG_KEYS[key], value]))
}

/** The keys of `side_effects:`, as the openCypher TCK names them. */
export const SIDE_EFFECT_KEYS = [
  '+nodes', '-nodes', '+relationships', '-relationships',
  '+labels', '-labels', '+properties', '-properties',
]

/** The `key: value` lines indented under the key at `ctx.lines[ctx.idx]`; leaves `ctx.idx` past them. */
function parseNested(ctx) {
  const indent = (line) => line.length - line.trimStart().length
  const own = indent(ctx.lines[ctx.idx])
  ctx.idx++
  const pairs = []
  while (ctx.idx < ctx.lines.length) {
    const line = ctx.lines[ctx.idx]
    const trimmed = line.trim()
    if (!trimmed || trimmed.startsWith('#')) { ctx.idx++; continue }
    if (indent(line) <= own) break
    const kv = parseKV(trimmed)
    if (!kv) throw new Error(`not a key at line ${ctx.idx + 1}: '${trimmed}'`)
    pairs.push(kv)
    ctx.idx++
  }
  return pairs
}

/** A `runner_gap:` at `ctx.lines[ctx.idx]`: its runners, issue and reason, all required. */
function parseRunnerGap(ctx) {
  const gap = { runners: [], issue: '', reason: '' }
  for (const [key, value] of parseNested(ctx)) {
    if (key === 'runners') gap.runners = parseYamlList(value)
    else if (key === 'issue' || key === 'reason') gap[key] = unquote(value)
    else throw new Error(`unknown runner_gap key '${key}'`)
  }
  if (!gap.runners.length || !gap.issue.trim() || !gap.reason.trim()) {
    throw new Error('a runner_gap names its `runners:`, `issue:` and `reason:`')
  }
  const unknown = gap.runners.find((r) => !RUNNERS.includes(r))
  if (unknown) throw new Error(`unknown runner '${unknown}' (known: ${RUNNERS.join(', ')})`)
  return gap
}

/** The counts of a `side_effects:` key whose value is `value`: `{}`, or one per line below. */
function parseSideEffects(ctx, value) {
  if (value === '{}') { ctx.idx++; return [] }
  if (value) throw new Error(`\`side_effects:\` is \`{}\` or one key per line, not '${value}'`)
  return parseNested(ctx).map(([key, count]) => {
    if (!SIDE_EFFECT_KEYS.includes(key)) throw new Error(`unknown side effect '${key}'`)
    if (!/^-?[0-9]+$/.test(count)) throw new Error(`side effect '${key}' needs a count, not '${count}'`)
    return [key, parseInt(count, 10)]
  })
}

function parseExpectBlock(ctx) {
  const expect = makeExpect()
  // The indent of the expectation's own keys: a line indented less is the
  // test's next key (a `skip:` or `note:` after `expect:`).
  let childIndent = null
  while (ctx.idx < ctx.lines.length) {
    const line = ctx.lines[ctx.idx]
    const trimmed = line.trim()
    if (trimmed.startsWith('#')) { ctx.idx++; continue }
    if (trimmed.startsWith('- name:')) break
    if (!line.startsWith(' ') && !line.startsWith('\t') && trimmed) break
    if (!trimmed) { ctx.idx++; continue }
    const indent = line.length - line.trimStart().length
    if (childIndent === null) childIndent = indent
    if (indent < childIndent) break

    const kv = parseKV(trimmed)
    if (!kv) break
    const [key, value] = kv
    switch (key) {
      case 'ordered': expect.ordered = value === 'true'; ctx.idx++; break
      case 'count': expect.count = parseInt(value, 10); ctx.idx++; break
      case 'empty': expect.empty = value === 'true'; ctx.idx++; break
      case 'error': expect.error = unquote(value); ctx.idx++; break
      case 'gql_status': expect.gqlStatus = unquote(value); ctx.idx++; break
      case 'hash': expect.hash = unquote(value); ctx.idx++; break
      case 'precision': expect.precision = parseInt(value, 10); ctx.idx++; break
      case 'columns': expect.columns = parseYamlList(value); ctx.idx++; break
      case 'warnings': expect.warnings = parseYamlList(value); ctx.idx++; break
      case 'rows': ctx.idx++; expect.rows = parseRows(ctx); break
      case 'terms': ctx.idx++; expect.terms = parseRows(ctx); break
      case 'values':
        if (value !== '|') throw new Error('`values:` is a block scalar (`|`)')
        expect.values = parseBlockScalar(ctx).trim(); break
      case 'unordered_lists': expect.unorderedLists = value === 'true'; ctx.idx++; break
      case 'side_effects': expect.sideEffects = parseSideEffects(ctx, value); break
      // A key no runner reads would leave a test that checks nothing.
      default: throw new Error(`unknown expect key '${key}'`)
    }
  }
  const others = expect.rows.length || expect.terms.length || expect.count !== null || expect.empty
  if (expect.values !== null && (others || expect.error !== null || expect.gqlStatus !== null)) {
    throw new Error('`values:` excludes `rows`, `terms`, `count`, `empty`, `error` and `gql_status`')
  }
  return expect
}

function parseRows(ctx) {
  const rows = []
  while (ctx.idx < ctx.lines.length) {
    const trimmed = ctx.lines[ctx.idx].trim()
    if (trimmed.startsWith('#')) { ctx.idx++; continue }
    if (!trimmed) { ctx.idx++; continue }
    if (trimmed.startsWith('- [')) {
      rows.push(parseInlineList(trimmed.slice(2)))
      ctx.idx++
    } else {
      break
    }
  }
  return rows
}

// ---------------------------------------------------------------------------
// YAML primitives
// ---------------------------------------------------------------------------

function parseKV(s) {
  let inSingle = false, inDouble = false
  for (let i = 0; i < s.length; i++) {
    const c = s[i]
    if (c === "'" && !inDouble) inSingle = !inSingle
    else if (c === '"' && !inSingle) inDouble = !inDouble
    else if (c === ':' && !inSingle && !inDouble) {
      const key = s.slice(0, i).trim()
      const value = s.slice(i + 1).trim()
      if (key) return [key, value]
    }
  }
  return null
}

/** Whether `s` is written in single or double quotes. */
function isQuoted(s) {
  s = s.trim()
  return s.length >= 2 &&
    ((s.startsWith('"') && s.endsWith('"')) || (s.startsWith("'") && s.endsWith("'")))
}

function unquote(s) {
  s = s.trim()
  if (isQuoted(s)) {
    // Only unescape YAML-level escapes (quotes and backslashes).
    // Do NOT process \n or \t: those are GQL string escapes handled by the engine.
    return s.slice(1, -1)
      .replace(/\\\\/g, '\x00').replace(/\\"/g, '"').replace(/\\'/g, "'").replace(/\x00/g, '\\')
  }
  return s
}

function parseYamlList(s) {
  s = s.trim()
  if (s === '[]' || !s) return []
  if (s.startsWith('[') && s.endsWith(']')) {
    return s.slice(1, -1).split(',').map(v => unquote(v.trim())).filter(Boolean)
  }
  return [unquote(s)]
}

function parseInlineList(s) {
  s = s.trim()
  if (!s.startsWith('[') || !s.endsWith(']')) return [unquote(s)]
  const inner = s.slice(1, -1)
  const items = []
  let current = '', depth = 0, inSingle = false, inDouble = false
  for (const c of inner) {
    if (c === "'" && !inDouble && depth === 0) { inSingle = !inSingle; current += c }
    else if (c === '"' && !inSingle && depth === 0) { inDouble = !inDouble; current += c }
    else if ((c === '[' || c === '{') && !inSingle && !inDouble) { depth++; current += c }
    else if ((c === ']' || c === '}') && !inSingle && !inDouble) { depth--; current += c }
    else if (c === ',' && depth === 0 && !inSingle && !inDouble) {
      items.push(unquote(current.trim()))
      current = ''
    } else { current += c }
  }
  if (current.trim()) items.push(unquote(current.trim()))
  return items
}

function parseStringList(ctx) {
  const items = []
  while (ctx.idx < ctx.lines.length) {
    const trimmed = ctx.lines[ctx.idx].trim()
    if (trimmed.startsWith('#')) { ctx.idx++; continue }
    if (!trimmed) { ctx.idx++; continue }
    if (trimmed.startsWith('- ')) {
      const value = trimmed.slice(2)
      if (value === '|') {
        items.push(parseBlockScalar(ctx))
      } else {
        items.push(unquote(value))
        ctx.idx++
      }
    } else { break }
  }
  return items
}

function parseMap(ctx, minIndent) {
  const map = {}
  while (ctx.idx < ctx.lines.length) {
    const line = ctx.lines[ctx.idx]
    const trimmed = line.trim()
    if (trimmed.startsWith('#') || !trimmed) { ctx.idx++; continue }
    if (trimmed.startsWith('- name:')) break
    const indent = line.length - line.trimStart().length
    if (indent < minIndent) break
    const kv = parseKV(trimmed)
    if (kv) {
      if (kv[1] === '|') {
        map[kv[0]] = parseBlockScalar(ctx)
      } else {
        map[kv[0]] = unquote(kv[1])
        ctx.idx++
      }
    } else {
      break
    }
  }
  return map
}

/** Params entries sit at indent 6 or deeper; each value keeps its quotes. */
function parseParams(ctx) {
  const params = {}
  while (ctx.idx < ctx.lines.length) {
    const line = ctx.lines[ctx.idx]
    const trimmed = line.trim()
    if (trimmed.startsWith('#') || !trimmed) { ctx.idx++; continue }
    const indent = line.length - line.trimStart().length
    if (indent < 6) break
    const kv = parseKV(trimmed)
    if (!kv) break
    params[kv[0]] = kv[1]
    ctx.idx++
  }
  return params
}

function parseBlockScalar(ctx) {
  ctx.idx++ // skip the "|" line
  if (ctx.idx >= ctx.lines.length) return ''
  const blockIndent = ctx.lines[ctx.idx].length - ctx.lines[ctx.idx].trimStart().length
  const parts = []
  while (ctx.idx < ctx.lines.length) {
    const line = ctx.lines[ctx.idx]
    const trimmed = line.trim()
    if (!trimmed) { parts.push(''); ctx.idx++; continue }
    const indent = line.length - line.trimStart().length
    if (indent < blockIndent) break
    parts.push(line.slice(blockIndent))
    ctx.idx++
  }
  return parts.join('\n').trimEnd()
}

function skipBlankAndComments(ctx) {
  while (ctx.idx < ctx.lines.length) {
    const trimmed = ctx.lines[ctx.idx].trim()
    if (!trimmed || trimmed.startsWith('#')) ctx.idx++
    else break
  }
}

function expectLine(ctx, expected) {
  skipBlankAndComments(ctx)
  if (ctx.idx >= ctx.lines.length || ctx.lines[ctx.idx].trim() !== expected) {
    throw new Error(`Expected '${expected}' at line ${ctx.idx + 1}, got '${ctx.lines[ctx.idx]?.trim() ?? '<EOF>'}'`)
  }
  ctx.idx++
}
