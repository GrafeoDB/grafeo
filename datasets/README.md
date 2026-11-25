# Datasets

Demo datasets for Grafeo. Each dataset is hand-made, original text, and ships as plain files that load into a database through GQL, Cypher or Turtle, with an optional set of description embeddings for vector search.

## Layout

- `silicon-valley/source.json`: the one source of truth for the Silicon Valley dataset.
- `silicon-valley/notes.md`: notes on the content.
- `silicon-valley/derived/`: files generated from the source (graph, timeline, GQL, Cypher, Turtle, Turtle with its vocabulary in RDFS and in OWL, the `importLpg` and `importRdf` payloads of `@grafeo-db/wasm`, vectors). Never edited by hand.
- `tools/`: the validator, the emitters, `build.mjs` and `embed.mjs`.
- `tests/`: the Python engine checks (every format loads into Grafeo and answers the story's questions) and their fixtures.

## Commands

Run these from this folder. Only `embed` needs `npm install`, which installs the model runtime (about 480 MB); the other commands run on Node alone.

- `npm run build`: validates the source and rewrites everything in `derived/`.
- `npm run check`: fails when a derived file is missing or stale.
- `npm test`: runs the tools tests.
- `npm run embed -- --models <dir>`: embeds every node description and writes `vectors.f32` and `vectors.json`.

The engine checks in `tests/` need the Python binding built from this repository. From the repository root, in a virtual environment:

```bash
pip install maturin -r datasets/tests/requirements.txt
maturin develop --release -m crates/bindings/python/Cargo.toml
pytest datasets/tests
```

## Vectors

`<dir>` holds `Xenova/all-MiniLM-L6-v2`, the quantized ONNX files the Grafeo playground ships. The environment variable `GRAFEO_MINILM_DIR` can replace the flag. The script uses the playground's settings (quantized model, mean pooling, normalised output, 384 dimensions), so a query embedded in the browser matches the stored vectors. The vectors come from onnxruntime-node at `q8`; the browser runs the same model file on onnxruntime-web, and `embed` checks that both runtimes give the committed vector of a reference query (cosine at least 0.9999) before it writes anything. Only nodes whose description changed are embedded again, unless the stored vectors come from another model or dtype, which re-embeds every node. After editing `source.json`, run `npm run build`, then `npm run embed`, then `npm run build` again (`embed` reads the built `graph.json`).

The vectors are committed. `npm run build` and `npm run check` fail, and name `npm run embed`, when a description changed after embedding. CI never needs the model.
