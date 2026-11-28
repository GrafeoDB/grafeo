/** A property value; a date is `{ $date: "YYYY-MM-DD" }`. */
export type Value = string | number | boolean | { $date: string };

/** The payload of `importLpg` in @grafeo-db/wasm: edge `source` and `target` index `nodes`. */
declare const payload: {
  nodes: { labels: string[]; properties: Record<string, Value> }[];
  edges: { source: number; target: number; type: string; properties: Record<string, Value> }[];
};
export default payload;
