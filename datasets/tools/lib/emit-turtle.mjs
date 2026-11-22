import { floatLiteral, numberLiteral } from './numbers.mjs';
import { EDGE_DATE_FIELDS, FLOAT_PROPS, NODE_DATE_FIELDS } from './model.mjs';
import { cmp, compareEdges, compareNodes, sortedProps } from './sort.mjs';

const PREFIXES = [
  '@prefix sv: <https://docs.grafeo.dev/datasets/silicon-valley/vocab#> .',
  '@prefix svr: <https://docs.grafeo.dev/datasets/silicon-valley/id#> .',
  '@prefix schema: <https://schema.org/> .',
  '@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .',
];

const NODE_CLASSES = {
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
const camel = (s) =>
  s
    .toLowerCase()
    .split('_')
    .map((w, i) => (i === 0 ? w : w.charAt(0).toUpperCase() + w.slice(1)))
    .join('');

const kebab = (s) => s.toLowerCase().replace(/_/g, '-');

/** Double-quoted string literal with Turtle escapes. */
function quote(s) {
  const body = s
    .replace(/\\/g, '\\\\')
    .replace(/"/g, '\\"')
    .replace(/\n/g, '\\n')
    .replace(/\r/g, '\\r')
    .replace(/\t/g, '\\t');
  return `"${body}"`;
}

/**
 * Predicate-object pairs for one property. Dates are `xsd:date`; a month-only date is the first of the month
 * plus `sv:<field>Precision "month"`, as the graph side's `<field>_precision`. Integers and booleans are bare;
 * other numbers, and every float prop, are exponent-form doubles.
 */
function literals(predicate, name, value, dateFields) {
  if (dateFields.includes(name)) {
    if (value.length === 7) {
      return [
        [predicate, `${quote(`${value}-01`)}^^xsd:date`],
        [`sv:${camel(name)}Precision`, quote('month')],
      ];
    }
    return [[predicate, `${quote(value)}^^xsd:date`]];
  }
  if (typeof value === 'string') return [[predicate, quote(value)]];
  if (typeof value === 'number') return [[predicate, FLOAT_PROPS.includes(name) ? floatLiteral(value) : numberLiteral(value)]];
  return [[predicate, String(value)]];
}

/** One block: subject line, then `a <class>` and the predicate-object pairs, ending with a full stop. */
function block(subject, cls, pairs) {
  const items = [`a ${cls}`, ...pairs.map(([p, o]) => `${p} ${o}`)];
  return `${subject}\n${items.map((i) => `    ${i}`).join(' ;\n')} .`;
}

/**
 * Turtle text: nodes, plain edge triples, then qualified resources for edges with properties.
 * @param {import('./validate.mjs').Source} source
 * @returns {string}
 */
export function emitTurtle(source) {
  const blocks = [PREFIXES.join('\n')];

  for (const n of [...source.nodes].sort(compareNodes)) {
    const pairs = [
      ['schema:name', quote(n.name)],
      ['schema:description', quote(n.description)],
    ];
    for (const [name, value] of Object.entries(sortedProps(n.props))) {
      pairs.push(...literals(NODE_PREDICATES[name] ?? `sv:${camel(name)}`, name, value, NODE_DATE_FIELDS));
    }
    pairs.sort(([a], [b]) => cmp(a, b));
    blocks.push(block(`svr:${n.key}`, NODE_CLASSES[n.label], pairs));
  }

  const edges = [...source.edges].sort(compareEdges);

  const plain = new Set();
  for (const e of edges) {
    plain.add(`svr:${e.source} sv:${camel(e.type)} svr:${e.target} .`);
    if (e.type === 'WORKS_AT') plain.add(`svr:${e.source} schema:worksFor svr:${e.target} .`);
  }
  blocks.push([...plain].join('\n'));

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
      [fromRole, `svr:${e.source}`],
      [toRole, `svr:${e.target}`],
      ...Object.entries(props).flatMap(([name, value]) => literals(`sv:${camel(name)}`, name, value, EDGE_DATE_FIELDS)),
    ];
    blocks.push(block(`svr:${kebab(e.type)}-${e.source}-${e.target}-${n}`, cls, pairs));
  }

  return `${blocks.join('\n\n')}\n`;
}
