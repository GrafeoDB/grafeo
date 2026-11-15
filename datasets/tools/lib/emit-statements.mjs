import { EDGE_DATE_FIELDS, EDGE_TYPES, FLOAT_PROPS, NODE_DATE_FIELDS } from './model.mjs';
import { floatLiteral, numberLiteral } from './numbers.mjs';
import { cmp, compareEdges, compareNodes } from './sort.mjs';

/** Single-quoted string literal with backslash, single quote, tab, newline and carriage return escaped. */
function quote(s) {
  const body = s
    .replace(/\\/g, '\\\\')
    .replace(/'/g, "\\'")
    .replace(/\t/g, '\\t')
    .replace(/\n/g, '\\n')
    .replace(/\r/g, '\\r');
  return `'${body}'`;
}

/** `[name, literal]` pairs for one property; month-only dates add a precision pair. */
function pairs(name, value, dateFields, dateLiteral) {
  if (dateFields.includes(name)) {
    if (value.length === 7) return [[name, dateLiteral(`${value}-01`)], [`${name}_precision`, quote('month')]];
    return [[name, dateLiteral(value)]];
  }
  if (typeof value === 'string') return [[name, quote(value)]];
  if (typeof value === 'number') return [[name, FLOAT_PROPS.includes(name) ? floatLiteral(value) : numberLiteral(value)]];
  return [[name, String(value)]];
}

const render = (list) => `{${list.map(([k, v]) => `${k}: ${v}`).join(', ')}}`;

/**
 * Build statements. `insert` is the keyword, `dateLiteral` wraps an ISO day.
 * @param {import('./validate.mjs').Source} source
 */
export function emitStatements(source, { insert, dateLiteral }) {
  const labels = new Map();
  for (const n of source.nodes) if (!labels.has(n.key)) labels.set(n.key, n.label);
  const lines = [];

  for (const n of [...source.nodes].sort(compareNodes)) {
    const list = [['key', quote(n.key)], ['name', quote(n.name)], ['description', quote(n.description)]];
    const props = n.props ?? {};
    for (const name of Object.keys(props).sort(cmp)) {
      list.push(...pairs(name, props[name], NODE_DATE_FIELDS, dateLiteral));
    }
    lines.push(`${insert} (:${n.label} ${render(list)})`);
  }

  for (const e of [...source.edges].sort(compareEdges)) {
    const props = e.props ?? {};
    const list = [];
    for (const name of EDGE_TYPES[e.type].props) {
      if (props[name] !== undefined) list.push(...pairs(name, props[name], EDGE_DATE_FIELDS, dateLiteral));
    }
    const body = list.length > 0 ? ` ${render(list)}` : '';
    for (const key of [e.source, e.target]) {
      if (!labels.has(key)) throw new Error(`edge ${e.type} ${e.source} -> ${e.target}: no node with key ${key}`);
    }
    const a = `(a:${labels.get(e.source)} {key: ${quote(e.source)}})`;
    const b = `(b:${labels.get(e.target)} {key: ${quote(e.target)}})`;
    lines.push(`MATCH ${a}, ${b} ${insert} (a)-[:${e.type}${body}]->(b)`);
  }
  return lines;
}
