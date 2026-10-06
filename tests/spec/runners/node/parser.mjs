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
 */

/**
 * @typedef {Object} TestCase
 * @property {string} name
 * @property {string|null} query
 * @property {string[]} statements
 * @property {string[]} setup
 * @property {Object<string,string>} params
 * @property {string[]} tags
 * @property {string|null} skip
 * @property {Expect} expect
 * @property {Object<string,string>} variants
 */

/**
 * @typedef {Object} GtestFile
 * @property {Meta} meta
 * @property {TestCase[]} tests
 */

/**
 * Coerce raw param string values to proper JS types.
 * Mirrors the type coercion in crates/grafeo-spec-tests/build.rs: a value
 * written as a JSON list or map (`[1, 2]`, `{"city": "Amsterdam"}`) becomes
 * that list or object, then booleans, numbers, and the rest stays a string.
 */
export function coerceParams(rawParams) {
  if (!rawParams || Object.keys(rawParams).length === 0) return undefined
  const result = {}
  for (const [key, val] of Object.entries(rawParams)) {
    const trimmed = val.trim()
    if (trimmed.startsWith('[') || trimmed.startsWith('{')) {
      result[key] = JSON.parse(trimmed)
    } else if (val === 'true') {
      result[key] = true
    } else if (val === 'false') {
      result[key] = false
    } else {
      const num = Number(val)
      if (!isNaN(num) && val.trim() !== '') {
        result[key] = num
      } else {
        result[key] = val
      }
    }
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
  return { meta, tests }
}

// ---------------------------------------------------------------------------
// Meta block
// ---------------------------------------------------------------------------

function parseMeta(ctx) {
  const meta = {
    language: 'gql', model: '', section: '', title: '',
    dataset: 'empty', requires: [], tags: [],
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
    name: '', query: null, statements: [], setup: [],
    params: {}, tags: [], skip: null, language: null,
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
      case 'setup':
        ctx.idx++; tc.setup = parseStringList(ctx); break
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
        ctx.idx++; tc.params = parseMap(ctx, 6); break
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
  return { rows: [], ordered: false, count: null, empty: false, error: null, hash: null, precision: null, columns: [] }
}

function parseExpectBlock(ctx) {
  const expect = makeExpect()
  while (ctx.idx < ctx.lines.length) {
    const line = ctx.lines[ctx.idx]
    const trimmed = line.trim()
    if (trimmed.startsWith('#')) { ctx.idx++; continue }
    if (trimmed.startsWith('- name:')) break
    if (!line.startsWith(' ') && !line.startsWith('\t') && trimmed) break
    if (!trimmed) { ctx.idx++; continue }

    const kv = parseKV(trimmed)
    if (!kv) break
    const [key, value] = kv
    switch (key) {
      case 'ordered': expect.ordered = value === 'true'; ctx.idx++; break
      case 'count': expect.count = parseInt(value, 10); ctx.idx++; break
      case 'empty': expect.empty = value === 'true'; ctx.idx++; break
      case 'error': expect.error = unquote(value); ctx.idx++; break
      case 'hash': expect.hash = unquote(value); ctx.idx++; break
      case 'precision': expect.precision = parseInt(value, 10); ctx.idx++; break
      case 'columns': expect.columns = parseYamlList(value); ctx.idx++; break
      case 'rows': ctx.idx++; expect.rows = parseRows(ctx); break
      default: ctx.idx++
    }
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

function unquote(s) {
  s = s.trim()
  if ((s.startsWith('"') && s.endsWith('"')) || (s.startsWith("'") && s.endsWith("'"))) {
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
