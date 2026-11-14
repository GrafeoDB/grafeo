# Files written by the `grafeo` crate

Used by `tests/default_features_open.rs` (#763): the CLI's default build must open every file the
`grafeo` crate's default profile writes, and an open it must refuse names the feature to install.

| File | Written by | Holds |
| --- | --- | --- |
| `default-profile.grafeo` | `grafeo` with its default features | three `:Document` nodes (Amsterdam, Utrecht, Rotterdam) with `title`, `body` and a 3-dimensional `embedding`; a cosine vector index on `Document.embedding` and a text index on `Document.body` |
| `rdf-profile.grafeo` | `grafeo` with `--no-default-features --features rdf` | one RDF triple (`<http://ex.org/alix> <http://ex.org/knows> <http://ex.org/gus>`) and one `:Person` node |

Both were written with `GrafeoDB::open`, the data above, and `close()`, so each is a single closed
file with no WAL beside it. The tests open copies, never the files themselves.
