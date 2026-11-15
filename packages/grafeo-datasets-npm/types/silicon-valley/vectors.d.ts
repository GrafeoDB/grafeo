export interface Vectors {
  /** Embedding model, `Xenova/all-MiniLM-L6-v2`. */
  model: string;
  dimension: number;
  /** Node keys, in the order of the rows in `data`. */
  keys: string[];
  /** `keys.length * dimension` floats, one unit-length row per key. */
  data: Float32Array;
}
export const vectors: Vectors;
/** The embedding of a node description, or `undefined` for an unknown key. */
export function vectorOf(key: string): Float32Array | undefined;
