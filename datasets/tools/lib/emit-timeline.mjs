import { cmp } from './sort.mjs';
import { compareDates } from './validate.mjs';

/**
 * One fact per edge that has `from`, `to` or `date`; a `date` gives `from = to = date`.
 * A fact with only `to` has no `from` and sorts as if `from` were its `to`.
 * @param {import('./validate.mjs').Source} source
 * @returns {{subject: string, relation: string, object: string, from?: string, to?: string}[]}
 */
export function emitTimeline(source) {
  const facts = [];
  for (const e of source.edges) {
    const p = e.props ?? {};
    const fact = { subject: e.source, relation: e.type, object: e.target };
    if (p.date !== undefined) {
      fact.from = p.date;
      fact.to = p.date;
    } else if (p.from !== undefined || p.to !== undefined) {
      if (p.from !== undefined) fact.from = p.from;
      if (p.to !== undefined) fact.to = p.to;
    } else {
      continue;
    }
    facts.push(fact);
  }
  const start = (f) => f.from ?? f.to;
  return facts.sort(
    (a, b) =>
      compareDates(start(a), start(b)) ||
      cmp(a.subject, b.subject) ||
      cmp(a.relation, b.relation) ||
      cmp(a.object, b.object),
  );
}
