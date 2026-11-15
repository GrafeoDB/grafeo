import { compareEdges, compareNodes, sortedProps } from './sort.mjs';

/**
 * The graph as the npm package ships it: sorted, `props` always present with sorted keys.
 * @param {import('./validate.mjs').Source} source
 */
export function emitGraph(source) {
  return {
    name: source.name,
    nodes: [...source.nodes].sort(compareNodes).map((n) => ({
      key: n.key,
      label: n.label,
      name: n.name,
      description: n.description,
      props: sortedProps(n.props),
    })),
    edges: [...source.edges].sort(compareEdges).map((e) => ({
      type: e.type,
      source: e.source,
      target: e.target,
      props: sortedProps(e.props),
    })),
  };
}
