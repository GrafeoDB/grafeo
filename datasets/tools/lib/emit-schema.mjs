import { NAMESPACES, block, dataTriples, prefixLines, quote, turtleBody } from './emit-turtle.mjs';
import { cmp } from './sort.mjs';

const SCHEMA_NAMESPACES = {
  rdf: 'http://www.w3.org/1999/02/22-rdf-syntax-ns#',
  rdfs: 'http://www.w3.org/2000/01/rdf-schema#',
  owl: 'http://www.w3.org/2002/07/owl#',
  ...NAMESPACES,
};

/** The ontology itself: the vocabulary namespace without its `#`. */
const ONTOLOGY = `<${NAMESPACES.sv.slice(0, -1)}>`;

/** Facts the data cannot show: schema.org's City is a Place, and sv:worksAt is a kind of schema:worksFor. */
const SUBCLASSES = [['schema:City', 'schema:Place']];
const SUBPROPERTIES = [['sv:worksAt', 'schema:worksFor']];

/** `worksAt` to `works at`. */
const words = (name) => name.replace(/([a-z])([A-Z])/g, '$1 $2').toLowerCase();

const local = (name) => name.slice(name.indexOf(':') + 1);
const ours = (name) => name.startsWith('sv:');
const sorted = (set) => [...set].sort(cmp);

/**
 * The vocabulary as the data uses it: every class, and per predicate whether it links resources and which
 * classes or datatypes it has as domain and range.
 * @param {import('./validate.mjs').Source} source
 */
export function vocabulary(source) {
  const triples = dataTriples(source);
  const classOf = new Map(triples.filter(([, p]) => p === 'a').map(([s, , o]) => [s, o.iri]));
  const classes = new Set(classOf.values());
  const predicates = new Map();
  for (const [s, p, o] of triples) {
    if (p === 'a') continue;
    const entry = predicates.get(p) ?? { object: 'iri' in o, domain: new Set(), range: new Set() };
    if (entry.object !== 'iri' in o) throw new Error(`${p} has both resource and literal objects`);
    entry.domain.add(classOf.get(s));
    entry.range.add('iri' in o ? classOf.get(o.iri) : o.datatype);
    predicates.set(p, entry);
  }
  for (const [sub, sup] of SUBCLASSES) classes.add(sub).add(sup);
  return { classes: sorted(classes), predicates: new Map([...predicates].sort(([a], [b]) => cmp(a, b))) };
}

/** A class keeps its name (`Employment`, `AI`); a property reads as words (`works at`). */
const label = (name) => `rdfs:label ${quote(ours(name) && /^[a-z]/.test(local(name)) ? words(local(name)) : local(name))}`;

/** Turtle text with the vocabulary in RDFS before the data. A domain or range is stated only when it is one
 *  class or datatype: RDFS reads several as their intersection. */
export function emitRdfs(source) {
  const { classes, predicates } = vocabulary(source);
  const blocks = [prefixLines(['rdf', 'rdfs', 'sv', 'svr', 'schema', 'xsd'], SCHEMA_NAMESPACES).join('\n')];
  for (const c of classes.filter(ours)) blocks.push(block(c, ['a rdfs:Class', label(c)]));
  blocks.push(SUBCLASSES.map(([sub, sup]) => `${sub} rdfs:subClassOf ${sup} .`).join('\n'));
  for (const [p, { domain, range }] of predicates) {
    if (!ours(p)) continue;
    const items = ['a rdf:Property', label(p)];
    if (domain.size === 1) items.push(`rdfs:domain ${sorted(domain)[0]}`);
    if (range.size === 1) items.push(`rdfs:range ${sorted(range)[0]}`);
    for (const [sub, sup] of SUBPROPERTIES) if (sub === p) items.push(`rdfs:subPropertyOf ${sup}`);
    blocks.push(block(p, items));
  }
  blocks.push(turtleBody(source));
  return `${blocks.join('\n\n')}\n`;
}

/** A class or datatype expression: the name itself, or the union of several. */
function union(set, kind) {
  const names = sorted(set);
  return names.length === 1 ? names[0] : `[ a ${kind} ; owl:unionOf ( ${names.join(' ')} ) ]`;
}

/** Turtle text with the vocabulary in OWL before the data: every class and property declared, several
 *  domains or ranges stated as their union. */
export function emitOwl(source) {
  const { classes, predicates } = vocabulary(source);
  const blocks = [prefixLines(['rdf', 'rdfs', 'owl', 'sv', 'svr', 'schema', 'xsd'], SCHEMA_NAMESPACES).join('\n')];
  blocks.push(block(ONTOLOGY, ['a owl:Ontology', `rdfs:label ${quote('Silicon Valley')}`]));
  for (const c of classes) {
    const items = ['a owl:Class'];
    if (ours(c)) items.push(label(c));
    for (const [sub, sup] of SUBCLASSES) if (sub === c) items.push(`rdfs:subClassOf ${sup}`);
    blocks.push(block(c, items));
  }
  for (const [p, { object, domain, range }] of predicates) {
    const items = [`a ${object ? 'owl:ObjectProperty' : 'owl:DatatypeProperty'}`];
    if (ours(p)) {
      items.push(label(p), `rdfs:domain ${union(domain, 'owl:Class')}`);
      items.push(`rdfs:range ${union(range, object ? 'owl:Class' : 'rdfs:Datatype')}`);
    }
    for (const [sub, sup] of SUBPROPERTIES) if (sub === p) items.push(`rdfs:subPropertyOf ${sup}`);
    blocks.push(block(p, items));
  }
  blocks.push(turtleBody(source));
  return `${blocks.join('\n\n')}\n`;
}
