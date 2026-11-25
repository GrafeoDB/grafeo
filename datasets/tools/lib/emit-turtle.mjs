import { floatLiteral, numberLiteral } from './numbers.mjs';
import { EDGE_DATE_FIELDS, FLOAT_PROPS, NODE_DATE_FIELDS } from './model.mjs';
import { cmp, compareEdges, compareNodes, sortedProps } from './sort.mjs';

/** Prefix to namespace, in the order the Turtle header lists them. */
export const NAMESPACES = {
  sv: 'https://docs.grafeo.dev/datasets/silicon-valley/vocab#',
  svr: 'https://docs.grafeo.dev/datasets/silicon-valley/id#',
  schema: 'https://schema.org/',
  xsd: 'http://www.w3.org/2001/XMLSchema#',
};

/** `@prefix` lines for `names`, in that order. */
export const prefixLines = (names, namespaces = NAMESPACES) => names.map((p) => `@prefix ${p}: <${namespaces[p]}> .`);

export const NODE_CLASSES = {
  Person: 'schema:Person',
  Company: 'schema:Organization',
  Product: 'schema:Product',
  Place: 'schema:Place',
  City: 'schema:City',
  Event: 'schema:Event',
  Technology: 'sv:Technology',
  AI: 'sv:AI',
};

const NODE_PREDICATES = { name: 'schema:name', description: 'schema:description', founded: 'schema:foundingDate' };

/** Qualified resource per edge type: class and the predicates for source and target. */
const QUALIFIED = {
  WORKS_AT: ['sv:Employment', 'sv:person', 'sv:company'],
  BASED_IN: ['sv:Base', 'sv:company', 'sv:place'],
  INVESTED_IN: ['sv:Investment', 'sv:investor', 'sv:company'],
  ATTENDED: ['sv:Attendance', 'sv:attendee', 'sv:event'],
  FOUNDED: ['sv:Founding', 'sv:founder', 'sv:founded'],
  LIVES_IN: ['sv:Residence', 'sv:person', 'sv:city'],
  OWNS: ['sv:Ownership', 'sv:owner', 'sv:owned'],
};

/** `weissman_score` or `WORKS_AT` to `weissmanScore` or `worksAt`. */
export const camel = (s) =>
  s
    .toLowerCase()
    .split('_')
    .map((w, i) => (i === 0 ? w : w.charAt(0).toUpperCase() + w.slice(1)))
    .join('');

const kebab = (s) => s.toLowerCase().replace(/_/g, '-');

/** Double-quoted string literal with Turtle escapes. */
export function quote(s) {
  const body = s
    .replace(/\\/g, '\\\\')
    .replace(/"/g, '\\"')
    .replace(/\n/g, '\\n')
    .replace(/\r/g, '\\r')
    .replace(/\t/g, '\\t');
  return `"${body}"`;
}

/**
 * An RDF object: `{iri}` (a prefixed name) or `{lexical, datatype}` (`xsd:string` for a plain string).
 * @typedef {{iri: string} | {lexical: string, datatype: string}} RdfObject
 */

const iri = (name) => ({ iri: name });
const literal = (lexical, datatype) => ({ lexical, datatype });

/**
 * Predicate-object pairs for one property. Dates are `xsd:date`; a month-only date is the first of the month
 * plus `sv:<field>Precision "month"`, as the graph side's `<field>_precision`. Integers are `xsd:integer`,
 * booleans `xsd:boolean`; other numbers, and every float prop, are `xsd:double` in exponent form.
 * @returns {[string, RdfObject][]}
 */
function literals(predicate, name, value, dateFields) {
  if (dateFields.includes(name)) {
    if (value.length === 7) {
      return [
        [predicate, literal(`${value}-01`, 'xsd:date')],
        [`sv:${camel(name)}Precision`, literal('month', 'xsd:string')],
      ];
    }
    return [[predicate, literal(value, 'xsd:date')]];
  }
  if (typeof value === 'string') return [[predicate, literal(value, 'xsd:string')]];
  if (typeof value === 'number') {
    const text = FLOAT_PROPS.includes(name) ? floatLiteral(value) : numberLiteral(value);
    return [[predicate, literal(text, /e/.test(text) ? 'xsd:double' : 'xsd:integer')]];
  }
  return [[predicate, literal(String(value), 'xsd:boolean')]];
}

/** An object in Turtle: a prefixed name, a quoted string, a typed date, or a bare number or boolean. */
export function turtleObject(o) {
  if ('iri' in o) return o.iri;
  if (o.datatype === 'xsd:string') return quote(o.lexical);
  if (o.datatype === 'xsd:date') return `${quote(o.lexical)}^^xsd:date`;
  return o.lexical;
}

/**
 * The data as subject blocks, in output order: one per node, one block of plain edge triples, then one per
 * qualified resource (an edge with properties). A block is `{subject, cls, pairs}`; the plain block has
 * `cls` null and its `triples` as `[subject, predicate, object]`.
 * @param {import('./validate.mjs').Source} source
 */
export function rdfData(source) {
  const nodes = [];
  for (const n of [...source.nodes].sort(compareNodes)) {
    const pairs = [
      ['schema:name', literal(n.name, 'xsd:string')],
      ['schema:description', literal(n.description, 'xsd:string')],
    ];
    for (const [name, value] of Object.entries(sortedProps(n.props))) {
      pairs.push(...literals(NODE_PREDICATES[name] ?? `sv:${camel(name)}`, name, value, NODE_DATE_FIELDS));
    }
    pairs.sort(([a], [b]) => cmp(a, b));
    nodes.push({ subject: `svr:${n.key}`, cls: NODE_CLASSES[n.label], pairs });
  }

  const edges = [...source.edges].sort(compareEdges);

  const plain = new Map();
  const add = (s, p, o) => plain.set(`${s} ${p} ${o}`, [s, p, iri(o)]);
  for (const e of edges) {
    add(`svr:${e.source}`, `sv:${camel(e.type)}`, `svr:${e.target}`);
    if (e.type === 'WORKS_AT') add(`svr:${e.source}`, 'schema:worksFor', `svr:${e.target}`);
  }

  const qualified = [];
  const counts = new Map();
  for (const e of edges) {
    const props = sortedProps(e.props);
    if (Object.keys(props).length === 0) continue;
    if (!QUALIFIED[e.type]) throw new Error(`edge type ${e.type} has properties but no qualified resource class`);
    const [cls, fromRole, toRole] = QUALIFIED[e.type];
    const triple = `${e.type} ${e.source} ${e.target}`;
    const n = (counts.get(triple) ?? 0) + 1;
    counts.set(triple, n);
    const pairs = [
      [fromRole, iri(`svr:${e.source}`)],
      [toRole, iri(`svr:${e.target}`)],
      ...Object.entries(props).flatMap(([name, value]) => literals(`sv:${camel(name)}`, name, value, EDGE_DATE_FIELDS)),
    ];
    qualified.push({ subject: `svr:${kebab(e.type)}-${e.source}-${e.target}-${n}`, cls, pairs });
  }

  return { nodes, plain: [...plain.values()], qualified };
}

/** Every data triple as `[subject, predicate, object]`, `rdf:type` written as `a`, in output order. */
export function dataTriples(source) {
  const { nodes, plain, qualified } = rdfData(source);
  const expand = ({ subject, cls, pairs }) => [[subject, 'a', iri(cls)], ...pairs.map(([p, o]) => [subject, p, o])];
  return [...nodes.flatMap(expand), ...plain, ...qualified.flatMap(expand)];
}

/** One block: subject line, then `a <class>` and the predicate-object pairs, ending with a full stop. */
export function block(subject, items) {
  return `${subject}\n${items.map((i) => `    ${i}`).join(' ;\n')} .`;
}

/** The data blocks as Turtle text, without the prefix header. */
export function turtleBody(source) {
  const { nodes, plain, qualified } = rdfData(source);
  const typed = ({ subject, cls, pairs }) =>
    block(subject, [`a ${cls}`, ...pairs.map(([p, o]) => `${p} ${turtleObject(o)}`)]);
  return [
    ...nodes.map(typed),
    plain.map(([s, p, o]) => `${s} ${p} ${turtleObject(o)} .`).join('\n'),
    ...qualified.map(typed),
  ].join('\n\n');
}

/**
 * Turtle text: nodes, plain edge triples, then qualified resources for edges with properties.
 * @param {import('./validate.mjs').Source} source
 * @returns {string}
 */
export function emitTurtle(source) {
  return `${prefixLines(['sv', 'svr', 'schema', 'xsd']).join('\n')}\n\n${turtleBody(source)}\n`;
}
