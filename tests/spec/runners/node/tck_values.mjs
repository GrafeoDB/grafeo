/**
 * The openCypher TCK's value notation, for `values:`: parse a cell and compare it to a
 * result value. Mirrors tests/spec/runners/python/tck_values.py and the Rust runner's
 * src/tck.rs; every runner checks itself against tests/spec/runners/values_cases.json.
 *
 * JavaScript has one number type, so a number cell matches by value here: `1` and
 * `1.0` both match the number 1. The Rust runner keeps them apart.
 */

/** A float cell, kept apart from an integer cell. */
export class Float {
  constructor(value) { this.value = value }
}

export class Node {
  constructor(labels, properties) { this.labels = labels; this.properties = properties }
}

export class Relationship {
  constructor(type, properties) { this.type = type; this.properties = properties }
}

export class Path {
  /** `forward[i]` is true when relationship i starts at node i (`-[]->`). */
  constructor(nodes, relationships, forward) {
    this.nodes = nodes; this.relationships = relationships; this.forward = forward
  }
}

const ESCAPES = { "'": "'", '"': '"', '\\': '\\', n: '\n', t: '\t', r: '\r', b: '\b', f: '\f' }

class Parser {
  constructor(text) { this.text = text; this.at = 0 }
  fail(what) { return new Error(`${what} at ${this.at} in '${this.text}'`) }
  skip() { while (this.at < this.text.length && /\s/.test(this.text[this.at])) this.at++ }
  peek(token) { this.skip(); return this.text.startsWith(token, this.at) }
  take(token) {
    if (!this.peek(token)) throw this.fail(`expected '${token}'`)
    this.at += token.length
  }

  name() {
    this.skip()
    if (this.peek('`')) {
      const end = this.text.indexOf('`', this.at + 1)
      if (end < 0) throw this.fail('unterminated name')
      const name = this.text.slice(this.at + 1, end)
      this.at = end + 1
      return name
    }
    // By code point: a letter outside the BMP is two code units
    const name = /[\p{L}\p{N}_]*/uy
    name.lastIndex = this.at
    const found = name.exec(this.text)[0]
    if (!found) throw this.fail('expected a name')
    this.at += found.length
    return found
  }

  word(word) {
    const rest = this.text.slice(this.at)
    if (rest.startsWith(word) && !/^[\p{L}\p{N}]/u.test(rest.slice(word.length))) {
      this.at += word.length
      return true
    }
    return false
  }

  value() {
    this.skip()
    if (this.word('null')) return null
    if (this.word('true')) return true
    if (this.word('false')) return false
    for (const [word, value] of [['NaN', NaN], ['-Inf', -Infinity], ['Inf', Infinity]]) {
      if (this.text.startsWith(word, this.at)) { this.at += word.length; return new Float(value) }
    }
    const rest = this.text.slice(this.at)
    if (rest.startsWith("'")) return this.string()
    if (rest.startsWith('{')) return this.map()
    if (rest.startsWith('(')) return this.node()
    if (rest.startsWith('<')) return this.path()
    if (rest.startsWith('[:')) return this.relationship()
    if (rest.startsWith('[')) return this.list()
    return this.number()
  }

  string() {
    this.take("'")
    let out = ''
    while (this.at < this.text.length) {
      const c = this.text[this.at]
      if (c === "'") { this.at++; return out }
      if (c === '\\') {
        const code = this.text[this.at + 1]
        if (code === 'u') {
          out += String.fromCharCode(parseInt(this.text.slice(this.at + 2, this.at + 6), 16))
          this.at += 6
          continue
        }
        out += ESCAPES[code] ?? `\\${code}`
        this.at += 2
        continue
      }
      out += c
      this.at++
    }
    throw this.fail('unterminated string')
  }

  number() {
    const match = /^[+\-0-9.eE]+/.exec(this.text.slice(this.at))
    if (!match) throw this.fail('expected a value')
    const text = match[0]
    this.at += text.length
    if (/[.eE]/.test(text)) {
      const value = Number(text)
      if (Number.isNaN(value)) throw this.fail('bad float')
      return new Float(value)
    }
    if (!/^[+-]?[0-9]+$/.test(text)) throw this.fail('bad integer')
    return BigInt(text)
  }

  items(close, item) {
    const out = []
    if (this.peek(close)) { this.take(close); return out }
    for (;;) {
      out.push(item())
      if (this.peek(',')) { this.take(','); continue }
      this.take(close)
      return out
    }
  }

  list() { this.take('['); return this.items(']', () => this.value()) }

  map() {
    this.take('{')
    const entries = this.items('}', () => {
      const key = this.name()
      this.take(':')
      return [key, this.value()]
    })
    return Object.fromEntries(entries)
  }

  properties() { return this.peek('{') ? this.map() : {} }

  node() {
    this.take('(')
    const labels = new Set()
    while (this.peek(':')) { this.take(':'); labels.add(this.name()) }
    const properties = this.properties()
    this.take(')')
    return new Node(labels, properties)
  }

  relationship() {
    this.take('[')
    this.take(':')
    const type = this.name()
    const properties = this.properties()
    this.take(']')
    return new Relationship(type, properties)
  }

  path() {
    this.take('<')
    const nodes = [this.node()]
    const relationships = []
    const forward = []
    while (!this.peek('>')) {
      if (this.peek('<-')) {
        this.take('<-'); relationships.push(this.relationship()); this.take('-'); forward.push(false)
      } else {
        this.take('-'); relationships.push(this.relationship()); this.take('->'); forward.push(true)
      }
      nodes.push(this.node())
    }
    this.take('>')
    return new Path(nodes, relationships, forward)
  }
}

/** The value a TCK cell writes; throws on anything else. An integer is a BigInt. */
export function parse(text) {
  const parser = new Parser(text)
  const value = parser.value()
  parser.skip()
  if (parser.at !== text.length) throw parser.fail('unexpected text')
  return value
}

const isPlainObject = (v) => v !== null && typeof v === 'object' && !Array.isArray(v)
const isInteger = (v) => typeof v === 'bigint' || Number.isInteger(v)

function isNodeShape(v) {
  return isPlainObject(v) && isInteger(v._id) && Array.isArray(v._labels)
}

function isRelationshipShape(v) {
  return isPlainObject(v) && isInteger(v._id) && typeof v._type === 'string' &&
    isInteger(v._source) && isInteger(v._target)
}

function entity(v, reserved) {
  const properties = {}
  for (const [k, p] of Object.entries(v)) if (!reserved.includes(k)) properties[k] = toNeutral(p)
  return { id: Number(v._id), properties }
}

/**
 * A result value as the Node or WASM binding gives it, in the neutral form `matches`
 * compares. The one place that recognises entities, by the shape the bindings give them:
 * an object with an integer `_id` and a `_labels` array is a node; one with `_id`, a string
 * `_type`, `_source` and `_target` a relationship; `{nodes, edges}` of those (WASM adds
 * `_type: 'path'`) a path. A map the query builds with those keys comes back as an entity too.
 * A temporal (Node's `{$date: ...}` tags, a Date) becomes `{other}`, which no cell matches:
 * the TCK dialect writes temporals as their Cypher `toString()` first.
 */
export function toNeutral(value) {
  if (value === null || value === undefined || typeof value === 'boolean') return value ?? null
  if (typeof value === 'bigint' || typeof value === 'number') return { number: value }
  if (typeof value === 'string') return { string: value }
  if (Array.isArray(value)) return value.map(toNeutral)
  if (isPlainObject(value)) {
    if (isNodeShape(value)) {
      return { node: { ...entity(value, ['_id', '_labels']), labels: [...value._labels] } }
    }
    if (isRelationshipShape(value)) {
      const rel = entity(value, ['_id', '_type', '_source', '_target'])
      return { relationship: { ...rel, type: value._type, source: Number(value._source), target: Number(value._target) } }
    }
    const keys = Object.keys(value).filter((k) => k !== '_type' || value._type !== 'path').sort()
    if (keys.join(',') === 'edges,nodes' && value.nodes.every(isNodeShape) && value.edges.every(isRelationshipShape)) {
      return {
        path: {
          nodes: value.nodes.map((n) => toNeutral(n).node),
          relationships: value.edges.map((e) => toNeutral(e).relationship),
        },
      }
    }
    const map = {}
    for (const [k, v] of Object.entries(value)) map[k] = toNeutral(v)
    return { map }
  }
  return { other: String(value) }
}

function kindOf(actual) {
  if (actual === null) return 'null'
  if (typeof actual === 'boolean') return 'bool'
  if (Array.isArray(actual)) return 'list'
  return Object.keys(actual)[0]
}

function numberMatches(expected, actual) {
  if (expected instanceof Float) {
    const n = Number(actual)
    return n === expected.value || (Number.isNaN(n) && Number.isNaN(expected.value))
  }
  // An integer cell: a BigInt, compared exactly against a BigInt or a safe integer
  return typeof actual === 'bigint' ? actual === expected : Number.isInteger(actual) && BigInt(actual) === expected
}

function propertiesMatch(expected, actual, flags) {
  const keys = Object.keys(expected)
  return keys.length === Object.keys(actual).length &&
    keys.every((k) => k in actual && matches(expected[k], actual[k], flags))
}

function sameLabels(expected, actual) {
  return expected.size === actual.length && actual.every((l) => expected.has(l))
}

function pathMatches(expected, actual, flags) {
  const { nodes, relationships } = actual
  if (nodes.length !== expected.nodes.length || relationships.length !== expected.relationships.length) return false
  for (let i = 0; i < relationships.length; i++) {
    const [start, end] = expected.forward[i] ? [nodes[i], nodes[i + 1]] : [nodes[i + 1], nodes[i]]
    const rel = relationships[i]
    if (rel.source !== start.id || rel.target !== end.id) return false
    if (rel.type !== expected.relationships[i].type) return false
    if (!propertiesMatch(expected.relationships[i].properties, rel.properties, flags)) return false
  }
  return expected.nodes.every((n, i) =>
    sameLabels(n.labels, nodes[i].labels) && propertiesMatch(n.properties, nodes[i].properties, flags))
}

function bagMatches(expected, actual, flags) {
  if (expected.length !== actual.length) return false
  const left = [...actual]
  for (const item of expected) {
    const hit = left.findIndex((a) => matches(item, a, flags))
    if (hit < 0) return false
    left.splice(hit, 1)
  }
  return true
}

/** Whether a parsed TCK cell matches a result value in the neutral form. */
export function matches(expected, actual, flags = { unorderedLists: false }) {
  const kind = kindOf(actual)
  if (expected === null || typeof expected === 'boolean') return actual === expected
  if (typeof expected === 'bigint' || expected instanceof Float) {
    return kind === 'number' && numberMatches(expected, actual.number)
  }
  if (typeof expected === 'string') return kind === 'string' && actual.string === expected
  if (Array.isArray(expected)) {
    if (kind !== 'list') return false
    if (flags.unorderedLists) return bagMatches(expected, actual, flags)
    return expected.length === actual.length && expected.every((e, i) => matches(e, actual[i], flags))
  }
  if (expected instanceof Node) {
    return kind === 'node' && sameLabels(expected.labels, actual.node.labels) &&
      propertiesMatch(expected.properties, actual.node.properties, flags)
  }
  if (expected instanceof Relationship) {
    return kind === 'relationship' && actual.relationship.type === expected.type &&
      propertiesMatch(expected.properties, actual.relationship.properties, flags)
  }
  if (expected instanceof Path) return kind === 'path' && pathMatches(expected, actual.path, flags)
  if (isPlainObject(expected)) return kind === 'map' && propertiesMatch(expected, actual.map, flags)
  throw new Error(`not a TCK value: ${expected}`)
}

/** The rows of a `values:` table, each a list of cell texts (`\|` is a pipe in a cell). */
export function tableRows(table) {
  return table.split('\n').map((l) => l.trim()).filter(Boolean).map((line) => {
    const inner = line.startsWith('|') && line.endsWith('|') ? line.slice(1, -1) : line
    const cells = []
    let cell = ''
    for (let i = 0; i < inner.length; i++) {
      if (inner[i] === '\\' && inner[i + 1] === '|') { cell += '|'; i++; continue }
      if (inner[i] === '|') { cells.push(cell.trim()); cell = ''; continue }
      cell += inner[i]
    }
    cells.push(cell.trim())
    return cells
  })
}
