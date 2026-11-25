/** The payload of `importRdf` in @grafeo-db/wasm: full IRIs; an object is an IRI or a literal. */
declare const payload: {
  triples: {
    subject: string;
    predicate: string;
    object: string | { value: string; datatype?: string };
  }[];
};
export default payload;
