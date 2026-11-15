import { EDGE_DATE_FIELDS, EDGE_TYPES, LABELS, NODE_DATE_FIELDS } from './model.mjs';

/**
 * @typedef {Record<string, string|number|boolean>} Props
 * @typedef {{key: string, label: string, name: string, description: string, props?: Props}} SourceNode
 * @typedef {{type: string, source: string, target: string, props?: Props}} SourceEdge
 * @typedef {{name: string, nodes: SourceNode[], edges: SourceEdge[]}} Source
 */

const KEBAB = /^[a-z0-9]+(-[a-z0-9]+)*$/;
const DATE = /^(\d{4})-(\d{2})(?:-(\d{2}))?$/;
const ABBREVIATIONS = ['inc', 'ltd', 'co', 'corp', 'mr', 'mrs', 'ms', 'dr', 'st', 'vs', 'jr', 'sr', 'etc'];

/** Whether `s` is a real `YYYY-MM` or `YYYY-MM-DD` date. */
function isValidDate(s) {
  const m = typeof s === 'string' ? DATE.exec(s) : null;
  if (!m) return false;
  const year = Number(m[1]);
  const month = Number(m[2]);
  const day = m[3] === undefined ? 1 : Number(m[3]);
  if (month < 1 || month > 12 || day < 1) return false;
  return day <= new Date(Date.UTC(year, month, 0)).getUTCDate();
}

/** C0 controls other than tab, LF and CR, plus DEL, NEL, LS and PS: they break one statement per line. */
const isBadChar = (ch) => {
  const c = ch.codePointAt(0);
  return (c < 32 && c !== 9 && c !== 10 && c !== 13) || c === 127 || c === 133 || c === 8232 || c === 8233;
};
const hasBadChar = (s) => [...s].some(isBadChar);

/** Problems in the string and number values of `values`, a map of field name to value. */
function checkValues(id, values) {
  const problems = [];
  for (const [field, value] of Object.entries(values)) {
    if (typeof value === 'string' && hasBadChar(value)) {
      problems.push(`${id}: control character in ${field}`);
    } else if (typeof value === 'number' && (!Number.isFinite(value) || /e/i.test(String(value)))) {
      problems.push(`${id}: number not written as plain decimal in ${field}`);
    }
  }
  return problems;
}

/** Compare two valid dates by their first day; a month stands for its first day. */
export function compareDates(a, b) {
  const first = (s) => (s.length === 7 ? `${s}-01` : s);
  const [x, y] = [first(a), first(b)];
  return x < y ? -1 : x > y ? 1 : 0;
}

/** Number of sentence ends in `text`, not counting known abbreviations. */
function sentenceCount(text) {
  let count = 0;
  for (const m of text.matchAll(/[.!?](\s|$)/g)) {
    const word = /([A-Za-z]+)$/.exec(text.slice(0, m.index));
    if (m[0][0] === '.' && word && ABBREVIATIONS.includes(word[1].toLowerCase())) continue;
    count++;
  }
  return count;
}

/**
 * Check a source; returns the problems, empty when valid.
 * @param {Source} source
 * @returns {string[]}
 */
export function validate(source) {
  const problems = [];
  const labels = new Map();

  for (const node of source.nodes ?? []) {
    const id = `node "${node.key}"`;
    if (labels.has(node.key)) problems.push(`duplicate key: ${node.key}`);
    if (!labels.has(node.key)) labels.set(node.key, node.label);
    if (typeof node.key !== 'string' || !KEBAB.test(node.key)) {
      problems.push(`${id}: key not kebab case`);
    }
    problems.push(
      ...checkValues(id, { name: node.name, description: node.description, ...(node.props ?? {}) }),
    );
    if (!LABELS.includes(node.label)) problems.push(`${id}: unknown label ${node.label}`);
    if (typeof node.description !== 'string' || node.description.trim() === '') {
      problems.push(`${id}: missing description`);
    } else if (sentenceCount(node.description) > 2) {
      problems.push(`${id}: description over two sentences`);
    }
    for (const field of NODE_DATE_FIELDS) {
      const value = node.props?.[field];
      if (value !== undefined && !isValidDate(value)) {
        problems.push(`${id}: invalid date in ${field}: ${value}`);
      }
    }
  }

  for (const edge of source.edges ?? []) {
    const id = `edge ${edge.type} ${edge.source} -> ${edge.target}`;
    const spec = EDGE_TYPES[edge.type];
    if (!spec) problems.push(`${id}: unknown edge type ${edge.type}`);
    const ends = [
      ['source', edge.source, spec?.from],
      ['target', edge.target, spec?.to],
    ];
    for (const [end, key, allowed] of ends) {
      if (!labels.has(key)) {
        problems.push(`${id}: edge end missing: ${key}`);
      } else if (allowed && !allowed.includes(labels.get(key))) {
        problems.push(
          `${id}: edge end has the wrong label: ${end} ${key} is ${labels.get(key)}, expected ${allowed.join(' or ')}`,
        );
      }
    }
    const props = edge.props ?? {};
    problems.push(...checkValues(id, props));
    for (const name of Object.keys(props)) {
      if (spec && !spec.props.includes(name)) {
        problems.push(`${id}: property not allowed on this edge type: ${name}`);
      }
    }
    let datesOk = true;
    for (const field of EDGE_DATE_FIELDS) {
      if (props[field] !== undefined && !isValidDate(props[field])) {
        problems.push(`${id}: invalid date in ${field}: ${props[field]}`);
        datesOk = false;
      }
    }
    if (datesOk && props.from !== undefined && props.to !== undefined && compareDates(props.from, props.to) > 0) {
      problems.push(`${id}: from after to: ${props.from} > ${props.to}`);
    }
  }
  return problems;
}
