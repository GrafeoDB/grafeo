# Databases closed while their embeddings were spilled

`tests/legacy_spill.rs` opens these. Before 0.6, spilling a vector index moved the embeddings of its column into
`<file>.spill/vectors_<label>%3A<property>.bin` and out of the database, so a database closed while spilled holds
them only there. Each directory holds one such database, `spilled.grafeo` with its `spilled.grafeo.spill/`:

- `0.5.44/`: written by `grafeo-engine` 0.5.44 from crates.io (a 0.5.x file).
- `0.6.0-dev/`: written by a 0.6 development build before #594 (commit `09b5c45b`, a 0.6 file).

Both went through the same steps, in a persistent database with `TierOverride::ForceDisk` on the vector section:

1. Four `:Item` nodes with a `name` and a 3-dimension `embedding`: Alix `[3, 19, 88]`, Gus `[19, 88, 3]`, Vincent
   `[88, 3, 19]`, Jules `[3.19, 19.88, 88.3]`; then a vector index on `:Item(embedding)` (3 dimensions).
2. `buffer_manager().spill_all()`: the spill file holds the four embeddings, the column none.
3. While spilled: Gus's embedding set to `[1988, 3, 19]` (the database file holds it), Vincent deleted, Jules's
   embedding removed (the old spill file does not record removals).
4. `close()`.

So an open that folds the spill file back in finds Alix `[3, 19, 88]`, Gus `[1988, 3, 19]` (the newer value wins),
Jules `[3.19, 19.88, 88.3]` (a removal while spilled before 0.6 comes back, as its reload did) and no Vincent.

The 0.5.44 database was written by this program (`grafeo-engine = { version = "=0.5.44", features = ["full"] }`,
`grafeo-common = "=0.5.44"`), with the database path as its argument:

```rust
use grafeo_common::storage::{SectionMemoryConfig, SectionType, TierOverride};
use grafeo_common::types::Value;
use grafeo_engine::{Config, GrafeoDB};

fn main() {
    let path = std::path::PathBuf::from(std::env::args().nth(1).expect("the database path"));
    let db = GrafeoDB::with_config(Config::persistent(&path).with_section_config(
        SectionType::VectorStore,
        SectionMemoryConfig { max_ram: None, tier: TierOverride::ForceDisk },
    ))
    .unwrap();
    let names = ["Alix", "Gus", "Vincent", "Jules"];
    let ids: Vec<_> = names
        .iter()
        .zip([[3.0_f32, 19.0, 88.0], [19.0, 88.0, 3.0], [88.0, 3.0, 19.0], [3.19, 19.88, 88.3]])
        .map(|(name, embedding)| {
            db.create_node_with_props(
                &["Item"],
                [("name", Value::from(*name)), ("embedding", Value::Vector(embedding.to_vec().into()))],
            )
            .unwrap()
        })
        .collect();
    db.create_vector_index("Item", "embedding", Some(3), None, None, None, None).unwrap();
    assert!(db.buffer_manager().spill_all() > 0, "nothing spilled");
    db.set_node_property(ids[1], "embedding", Value::Vector(vec![1988.0, 3.0, 19.0].into())).unwrap();
    db.delete_node(ids[2]).unwrap();
    db.remove_node_property(ids[3], "embedding").unwrap();
    db.close().unwrap();
}
```

The 0.6.0-dev database ran the same steps as a test on commit `09b5c45b`. Regenerate either only to add content; the
tests describe what each one holds.
