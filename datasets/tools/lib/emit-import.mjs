import { EDGE_DATE_FIELDS, EDGE_TYPES, FLOAT_PROPS, NODE_DATE_FIELDS } from './model.mjs';
import { NAMESPACES, dataTriples } from './emit-turtle.mjs';
import { cmp, compareEdges, compareNodes } from './sort.mjs';

const RDF_TYPE = 'http://www.w3.org/1999/02/22-rdf-syntax-ns#type';
const XSD_STRING = `${NAMESPACES.xsd}string`;

/**
 * A property as JSON, as the GQL file writes it: a date is `{"$date": day}` (a month-only date is the first
 * of the month plus `<name>_precision: "month"`); other values stay as they are.
 * JSON has one number type, and the importer reads a whole number as an integer, so a float prop must not be
 * whole and an integer must stay exact.
 */
function properties(name, value, dateFields) {
  if (dateFields.includes(name)) {
    if (value.length === 7) return [[name, { $date: `${value}-01` }], [`${name}_precision`, 'month']];
    return [[name, { $date: value }]];
  }
  if (FLOAT_PROPS.includes(name) && Number.isInteger(value)) {
    throw new Error(`float prop ${name} = ${value} is whole: the JSON import would read it as an integer`);
  }
  if (typeof value === 'number' && Number.isInteger(value) && !Number.isSafeInteger(value)) {
    throw new Error(`prop ${name} = ${value} is past 2^53: JSON cannot hold it exactly`);
  }
  return [[name, value]];
}

/**
 * The payload of `importLpg` in @grafeo-db/wasm: the same nodes, edges and properties as the GQL file, with
 * edge ends as indexes into `nodes`.
 * @param {import('./validate.mjs').Source} source
 */
export function emitLpgImport(source) {
  const nodes = [...source.nodes].sort(compareNodes);
  const index = new Map(nodes.map((n, i) => [n.key, i]));
  return {
    nodes: nodes.map((n) => {
      const props = n.props ?? {};
      const list = [['key', n.key], ['name', n.name], ['description', n.description]];
      for (const name of Object.keys(props).sort(cmp)) list.push(...properties(name, props[name], NODE_DATE_FIELDS));
      return { labels: [n.label], properties: Object.fromEntries(list) };
    }),
    edges: [...source.edges].sort(compareEdges).map((e) => {
      for (const key of [e.source, e.target]) {
        if (!index.has(key)) throw new Error(`edge ${e.type} ${e.source} -> ${e.target}: no node with key ${key}`);
      }
      const props = e.props ?? {};
      const list = [];
      for (const name of EDGE_TYPES[e.type].props) {
        if (props[name] !== undefined) list.push(...properties(name, props[name], EDGE_DATE_FIELDS));
      }
      return { source: index.get(e.source), target: index.get(e.target), type: e.type, properties: Object.fromEntries(list) };
    }),
  };
}

/** `svr:anton` to its full IRI. */
function expand(name) {
  const [prefix, rest] = [name.slice(0, name.indexOf(':')), name.slice(name.indexOf(':') + 1)];
  if (!NAMESPACES[prefix]) throw new Error(`no namespace for ${name}`);
  return `${NAMESPACES[prefix]}${rest}`;
}

/**
 * The payload of `importRdf` in @grafeo-db/wasm: the triples of the Turtle file with full IRIs. A plain
 * string has no datatype; every other literal names its XSD type.
 * @param {import('./validate.mjs').Source} source
 */
export function emitRdfImport(source) {
  return {
    triples: dataTriples(source).map(([s, p, o]) => {
      let object;
      if ('iri' in o) {
        object = expand(o.iri);
      } else {
        const datatype = expand(o.datatype);
        object = datatype === XSD_STRING ? { value: o.lexical } : { value: o.lexical, datatype };
      }
      return { subject: expand(s), predicate: p === 'a' ? RDF_TYPE : expand(p), object };
    }),
  };
}
