export type NodeLabel = 'Person' | 'Company' | 'Product' | 'Technology' | 'AI' | 'Place' | 'City' | 'Event';

export type EdgeType =
  | 'KNOWS'
  | 'RIVAL_OF'
  | 'MENTORS'
  | 'WORKS_AT'
  | 'FOUNDED'
  | 'LIVES_IN'
  | 'BASED_IN'
  | 'INVESTED_IN'
  | 'MAKES'
  | 'USES'
  | 'COMPETES_WITH'
  | 'BUILT'
  | 'OWNS'
  | 'ATTENDED'
  | 'LOCATED_IN';

export type PropValue = string | number | boolean;

export interface DatasetNode {
  /** Kebab-case key, unique in the dataset. */
  key: string;
  label: NodeLabel;
  name: string;
  description: string;
  props: Record<string, PropValue>;
}

export interface DatasetEdge {
  type: EdgeType;
  /** Key of the source node. */
  source: string;
  /** Key of the target node. */
  target: string;
  props: Record<string, PropValue>;
}

export interface TimelineFact {
  /** Key of the subject node. */
  subject: string;
  relation: EdgeType;
  /** Key of the object node. */
  object: string;
  /** ISO `YYYY-MM-DD` or `YYYY-MM`. Absent when only `to` is known. */
  from?: string;
  /** ISO `YYYY-MM-DD` or `YYYY-MM`. Absent while the fact still holds. */
  to?: string;
}

export interface Dataset {
  name: string;
  nodes: DatasetNode[];
  edges: DatasetEdge[];
  timeline: TimelineFact[];
}

export const siliconValley: Dataset;
