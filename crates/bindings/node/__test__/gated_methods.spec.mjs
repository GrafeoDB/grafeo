// A method the package compiles only with a feature is there exactly when
// `features()` reports that feature. The gated methods are read from the
// binding's source: napi-rs gates whole `#[napi] impl` blocks (a `#[cfg]` on
// one method of a block does not work), so every method of a block under a
// `#[cfg(...)]` counts, by its JavaScript name, and so does a gated free
// function. A custom profile (lpg, rdf, ai, analytics, edge) builds the
// package without some features; their methods must be absent, not present
// and failing. The package's profiles turn on what the Python package's do,
// so `features()` reports the same for the same profile.

import { readFileSync, readdirSync, statSync } from 'node:fs'
import { dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'
import { describe, it, expect } from 'vitest'
import grafeo from '../grafeo.js'

const HERE = dirname(fileURLToPath(import.meta.url))
const NODE = join(HERE, '..')
const PYTHON = join(NODE, '..', 'python')

function rustFiles(dir) {
  return readdirSync(dir).flatMap((name) => {
    const path = join(dir, name)
    if (statSync(path).isDirectory()) return rustFiles(path)
    return name.endsWith('.rs') ? [path] : []
  })
}

/** Whether a `cfg(...)` expression of feature terms holds for `features`. */
function cfgHolds(expression, features) {
  const text = expression.trim()
  const term = text.match(/^feature\s*=\s*"([^"]+)"$/)
  if (term) return features.has(term[1])
  const call = text.match(/^(all|any|not)\s*\(([\s\S]*)\)$/)
  if (!call) throw new Error(`a cfg the test cannot read: ${text}`)
  const parts = []
  let depth = 0
  let start = 0
  for (let i = 0; i < call[2].length; i++) {
    const char = call[2][i]
    if (char === '(') depth++
    if (char === ')') depth--
    if (char === ',' && depth === 0) {
      parts.push(call[2].slice(start, i))
      start = i + 1
    }
  }
  parts.push(call[2].slice(start))
  const values = parts.filter((part) => part.trim()).map((part) => cfgHolds(part, features))
  if (call[1] === 'not') return !values[0]
  return call[1] === 'all' ? values.every(Boolean) : values.some(Boolean)
}

function cfgFeatures(expression) {
  return new Set([...expression.matchAll(/feature\s*=\s*"([^"]+)"/g)].map((m) => m[1]))
}

function camelCase(name) {
  return name.replace(/_([a-z0-9])/g, (_, char) => char.toUpperCase())
}

/** Rust struct name to JavaScript class name, from `#[napi(js_name = ...)]`. */
function classNames(source) {
  const names = {}
  const declaration = /#\[napi(?:\(([^\]]*)\))?\]\s*(?:#\[[^\]]*\]\s*)*pub struct (\w+)/g
  for (const match of source.matchAll(declaration)) {
    const rename = (match[1] || '').match(/js_name\s*=\s*"(\w+)"/)
    names[match[2]] = rename ? rename[1] : match[2]
  }
  return names
}

/** The JavaScript name a `#[napi(...)]` attribute gives the function after it. */
function jsName(attributes, rustName) {
  const rename = attributes.match(/js_name\s*=\s*"(\w+)"/)
  if (rename) return rename[1]
  const getter = attributes.match(/#\[napi\([^\]]*\bgetter\b/)
  const name = getter ? rustName.replace(/^get_/, '') : rustName
  return camelCase(name)
}

/**
 * `[{ owner, name, cfg }]`: the methods of every `#[cfg(...)] #[napi] impl X`
 * block (owner the class name) and every gated `#[napi]` free function (owner
 * null).
 */
function gatedItems(source, names) {
  const items = []
  const block = /^#\[cfg\((.*)\)\][ \t]*\n(?:\/\/[^\n]*\n|#\[[^\n]*\]\s*\n)*#\[napi\]\s*\nimpl (\w+) \{\n([\s\S]*?)^\}/gm
  for (const match of source.matchAll(block)) {
    const owner = names[match[2]] || match[2]
    let pending = []
    for (const line of match[3].split('\n')) {
      const stripped = line.trim()
      const method = line.match(/^ {4}(?:pub(?:\(\w+\))?\s+)?(?:async\s+)?fn\s+(\w+)/)
      if (method) {
        const attributes = pending.join('\n')
        pending = []
        if (attributes.includes('#[napi')) {
          items.push({ owner, name: jsName(attributes, method[1]), cfg: match[1] })
        }
      } else if (/^ {4}[#/]/.test(line) || (pending.length && !line.startsWith('    }'))) {
        if (stripped) pending.push(stripped)
      } else if (line.startsWith('    }')) {
        pending = []
      }
    }
  }
  const free = /^#\[cfg\((.*)\)\][ \t]*\n((?:#\[[^\n]*\]\s*\n|\/\/\/[^\n]*\n)*)pub (?:async )?fn (\w+)/gm
  for (const match of source.matchAll(free)) {
    if (match[2].includes('#[napi')) {
      items.push({ owner: null, name: jsName(match[2], match[3]), cfg: match[1] })
    }
  }
  return items
}

function bindingItems() {
  const texts = rustFiles(join(NODE, 'src')).map((path) => readFileSync(path, 'utf8'))
  const names = Object.assign({}, ...texts.map(classNames))
  return texts.flatMap((text) => gatedItems(text, names))
}

/** The names a binding's `features()` can report, read from its source. */
/** The names `features()` can report: the pairs of the shared `reported_features!` macro. */
function reportable() {
  const source = readFileSync(join(NODE, '..', 'common', 'src', 'lib.rs'), 'utf8')
  const body = source.slice(source.indexOf('macro_rules! reported_features'))
  const list = body.slice(0, body.indexOf('.into_iter()'))
  return [...list.matchAll(/\("([a-z-]+)", cfg!\(feature = "\1"\)\)/g)].map((m) => m[1])
}

/** The body of a binding's `features()` function. */
function featuresBody(binding) {
  const source = readFileSync(join(binding, 'src', 'lib.rs'), 'utf8').replace(/\r\n/g, '\n')
  const body = source.slice(source.indexOf('fn features()'))
  return body.slice(0, body.indexOf('\n}\n'))
}

/** A manifest's `[features]` table: each feature and what it turns on. */
function featureTable(binding) {
  const text = readFileSync(join(binding, 'Cargo.toml'), 'utf8').replace(/\r\n/g, '\n')
  const section = text.split('\n[features]\n')[1].split('\n[')[0]
  const table = {}
  for (const line of section.split('\n')) {
    const entry = line.match(/^([a-z0-9-]+)\s*=\s*\[([^\]]*)\]/)
    if (entry) table[entry[1]] = [...entry[2].matchAll(/"([^"]+)"/g)].map((m) => m[1])
  }
  return table
}

/** The binding's own features a feature turns on, transitively. */
function expand(table, feature, seen = new Set()) {
  for (const item of table[feature] || []) {
    if (item.includes('/') || seen.has(item)) continue
    seen.add(item)
    expand(table, item, seen)
  }
  return seen
}

/**
 * Every method of each `#[napi] impl` block, gated or not:
 * `Map<"Class.name", Set<feature>>`, the features its block's cfg reads.
 */
function nodeMethods() {
  const methods = new Map()
  const texts = rustFiles(join(NODE, 'src')).map((path) => readFileSync(path, 'utf8'))
  const names = Object.assign({}, ...texts.map(classNames))
  const block = /^((?:#\[cfg\((.*)\)\][ \t]*\n)?)(?:\/\/[^\n]*\n|#\[(?!cfg)[^\n]*\]\s*\n)*#\[napi\]\s*\nimpl (\w+) \{\n([\s\S]*?)^\}/gm
  for (const text of texts) {
    for (const match of text.matchAll(block)) {
      const owner = names[match[3]] || match[3]
      const features = match[2] ? cfgFeatures(match[2]) : new Set()
      let pending = []
      for (const line of match[4].split('\n')) {
        const method = line.match(/^ {4}(?:pub(?:\(\w+\))?\s+)?(?:async\s+)?fn\s+(\w+)/)
        if (method) {
          const attributes = pending.join('\n')
          pending = []
          if (attributes.includes('#[napi')) methods.set(`${owner}.${jsName(attributes, method[1])}`, features)
        } else if (/^ {4}[#/]/.test(line)) {
          pending.push(line.trim())
        } else if (line.startsWith('    }')) {
          pending = []
        }
      }
    }
  }
  return methods
}

/**
 * Every method of each `#[pymethods]` block of the Python binding, by its name
 * in camel case: `Map<"Class.name", Set<feature>>`, the features its own cfgs
 * read (Python gates single methods).
 */
function pythonMethods() {
  const methods = new Map()
  const texts = rustFiles(join(PYTHON, 'src')).map((path) => readFileSync(path, 'utf8'))
  const names = {}
  for (const text of texts) {
    const declaration = /#\[pyclass(?:\(([^\]]*)\))?\]\s*(?:#\[[^\]]*\]\s*)*pub struct (\w+)/g
    for (const match of text.matchAll(declaration)) {
      const rename = (match[1] || '').match(/name\s*=\s*"(\w+)"/)
      names[match[2]] = rename ? rename[1] : match[2]
    }
  }
  const block = /^#\[pymethods\]\s*\nimpl (\w+) \{\n([\s\S]*?)^\}/gm
  for (const text of texts) {
    for (const match of text.matchAll(block)) {
      const owner = names[match[1]] || match[1]
      let pending = []
      for (const line of match[2].split('\n')) {
        const method = line.match(/^ {4}(?:pub(?:\(\w+\))?\s+)?(?:async\s+)?fn\s+(\w+)/)
        if (method) {
          const attributes = pending.join('\n')
          pending = []
          if (attributes.includes('#[setter')) continue
          const rename = attributes.match(/#\[pyo3\([^\]]*?\bname\s*=\s*"(\w+)"/)
          const getter = attributes.match(/#\[getter(?:\((\w+)\))?\]/)
          const name = rename ? rename[1] : getter ? getter[1] || method[1].replace(/^get_/, '') : method[1]
          const cfgs = [...attributes.matchAll(/#\[cfg\((.*)\)\]/g)].map((m) => m[1])
          const key = `${owner}.${camelCase(name)}`
          const features = new Set(cfgs.flatMap((cfg) => [...cfgFeatures(cfg)]))
          // A method defined under a cfg and its not(): there either way.
          methods.set(key, methods.has(key) && cfgs.length ? new Set() : features)
        } else if (/^ {4}[#/]/.test(line) || (pending.length && /^ {5,}/.test(line))) {
          pending.push(line.trim())
        } else if (line.startsWith('    }')) {
          pending = []
        }
      }
    }
  }
  return methods
}

const SAMPLE = `
#[napi(js_name = "GrafeoDB")]
pub struct JsGrafeoDB {
    inner: u8,
}

#[napi]
impl JsGrafeoDB {
    #[napi(js_name = "execute")]
    pub fn execute(&self) {}
}

// Cypher methods live in a separate impl block.
#[cfg(feature = "cypher")]
#[napi]
impl JsGrafeoDB {
    /// Execute a Cypher query.
    #[napi(js_name = "executeCypher")]
    pub async fn execute_cypher(
        &self,
        query: String,
    ) -> Result<u8> {
        let text = format!("{}", 1);
        Ok(0)
    }

    #[napi]
    pub fn explain_cypher(&self) {}

    fn helper(&self) {}
}

#[cfg(all(feature = "embed", feature = "vector-index"))]
#[napi]
pub fn embed_model_names() -> Vec<String> {
    vec![]
}
`

describe('gated methods', () => {
  it('the reader finds each gated method by its JavaScript name', () => {
    const names = classNames(SAMPLE)
    expect(names).toEqual({ JsGrafeoDB: 'GrafeoDB' })
    expect(gatedItems(SAMPLE, names)).toEqual([
      { owner: 'GrafeoDB', name: 'executeCypher', cfg: 'feature = "cypher"' },
      { owner: 'GrafeoDB', name: 'explainCypher', cfg: 'feature = "cypher"' },
      { owner: null, name: 'embedModelNames', cfg: 'all(feature = "embed", feature = "vector-index")' },
    ])
    expect(cfgHolds('all(feature = "embed", feature = "vector-index")', new Set(['embed']))).toBe(false)
    expect(cfgHolds('not(feature = "algos")', new Set())).toBe(true)
  })

  it('every gated method names features the package reports', () => {
    const items = bindingItems()
    expect(items.length).toBeGreaterThanOrEqual(20)
    const known = new Set(reportable())
    const unknown = items
      .filter((item) => [...cfgFeatures(item.cfg)].some((feature) => !known.has(feature)))
      .map((item) => `${item.owner}.${item.name}: ${item.cfg}`)
    expect(unknown).toEqual([])
  })

  it('a gated method is there exactly when its features are', () => {
    const features = new Set(grafeo.features())
    const wrong = []
    for (const { owner, name, cfg } of bindingItems()) {
      const target = owner === null ? grafeo : grafeo[owner]
      expect(target, `${owner} is exported`).toBeDefined()
      const there = owner === null ? name in target : name in target.prototype || name in target
      if (there !== cfgHolds(cfg, features)) {
        wrong.push(`${owner}.${name} should be ${there ? 'absent' : 'there'}: ${cfg}`)
      }
    }
    expect(wrong, `features ${[...features].sort()}`).toEqual([])
  })

  it('reports the same features as the Python package for the same profile', () => {
    // Both call the one macro, so they report the same names from the same cfgs.
    for (const binding of [NODE, PYTHON]) {
      expect(featuresBody(binding)).toContain('grafeo_bindings_common::reported_features!()')
    }
    expect(reportable().length).toBeGreaterThanOrEqual(15)
    const known = new Set(reportable())
    const node = featureTable(NODE)
    const python = featureTable(PYTHON)
    const profiles = ['default', 'lpg', 'rdf', 'ai', 'analytics', 'edge', 'embedded', 'full']
    for (const profile of profiles) {
      expect(profile in node, `node defines ${profile}`).toBe(profile in python)
      const of = (table) => [...expand(table, profile)].filter((f) => known.has(f)).sort()
      expect(of(node), `the ${profile} profile`).toEqual(of(python))
    }
  })
  it('gates a method both packages have on the same features in both', () => {
    const node = nodeMethods()
    const python = pythonMethods()
    expect(node.size).toBeGreaterThanOrEqual(100)
    expect(python.size).toBeGreaterThanOrEqual(100)
    const sorted = (set) => [...set].sort().join(', ') || 'none'
    const differ = [...node]
      .filter(([key, features]) => python.has(key) && sorted(features) !== sorted(python.get(key)))
      .map(([key, features]) => `${key}: Node ${sorted(features)}, Python ${sorted(python.get(key))}`)
    expect(differ).toEqual([])
  })
})
