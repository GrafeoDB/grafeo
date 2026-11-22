/** The data model shared by the validator and the emitters. */

export const LABELS = ['Person', 'Company', 'Product', 'Technology', 'AI', 'Place', 'City', 'Event'];

/** Optional node facts that are dates. */
export const NODE_DATE_FIELDS = ['founded'];

/** Props that are always floats: written as float literals even when the value is a whole number. */
export const FLOAT_PROPS = ['weissman_score'];

/** Edge props that are dates. */
export const EDGE_DATE_FIELDS = ['from', 'to', 'date'];

/**
 * Edge types: allowed labels at each end and allowed property names.
 * @type {Record<string, {from: string[], to: string[], props: string[]}>}
 */
export const EDGE_TYPES = {
  KNOWS: { from: ['Person'], to: ['Person'], props: [] },
  RIVAL_OF: { from: ['Person'], to: ['Person'], props: [] },
  MENTORS: { from: ['Person'], to: ['Person'], props: [] },
  WORKS_AT: { from: ['Person'], to: ['Company'], props: ['role', 'from', 'to'] },
  FOUNDED: { from: ['Person'], to: ['Company', 'Product'], props: ['date'] },
  LIVES_IN: { from: ['Person'], to: ['City'], props: ['from', 'to'] },
  BASED_IN: { from: ['Company'], to: ['Place', 'City'], props: ['from', 'to'] },
  INVESTED_IN: {
    from: ['Person', 'Company'],
    to: ['Company'],
    props: ['round', 'amount_usd', 'date', 'note'],
  },
  MAKES: { from: ['Company'], to: ['Product'], props: [] },
  USES: { from: ['Product'], to: ['Technology'], props: [] },
  COMPETES_WITH: { from: ['Product'], to: ['Product'], props: [] },
  BUILT: { from: ['Person'], to: ['AI', 'Product'], props: [] },
  TRAINED_ON: { from: ['AI'], to: ['AI'], props: [] },
  OWNS: { from: ['Person', 'Company'], to: ['AI', 'Place', 'Product'], props: ['from', 'to'] },
  ATTENDED: { from: ['Person', 'Company'], to: ['Event'], props: ['date', 'outcome'] },
  LOCATED_IN: { from: ['Place'], to: ['City'], props: [] },
};
