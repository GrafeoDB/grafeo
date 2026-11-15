/** Three-way comparison of two strings by code unit. */
export const cmp = (x, y) => (x < y ? -1 : x > y ? 1 : 0);

/** Order nodes by key. */
export const compareNodes = (a, b) => cmp(a.key, b.key);

/** Order edges by type, then source, then target, then props. */
export const compareEdges = (a, b) =>
  cmp(a.type, b.type) ||
  cmp(a.source, b.source) ||
  cmp(a.target, b.target) ||
  cmp(JSON.stringify(sortedProps(a.props)), JSON.stringify(sortedProps(b.props)));

/** A copy of `props` with its keys in sorted order. */
export function sortedProps(props) {
  return Object.fromEntries(Object.entries(props ?? {}).sort(([a], [b]) => cmp(a, b)));
}
