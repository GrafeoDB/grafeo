//! The schema survives a reopen through the catalog section's version 2
//! (#517): every kind of catalog entry is written as a typed record, cut into
//! stream pieces, and read back one record at a time, with the property
//! types of #569.
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test catalog_v2
//! ```

#![cfg(all(
    feature = "lpg",
    feature = "gql",
    feature = "grafeo-file",
    feature = "vector-index",
    feature = "text-index"
))]

use std::path::Path;

use grafeo_common::storage::{
    ChunkCaps, ChunkMeta, ChunkStreamReader, SectionType, read_catalog_records,
};
use grafeo_common::testing::chunk_caps::with_chunk_caps;
use grafeo_common::types::{NodeId, Value};
use grafeo_engine::{Config, GrafeoDB};
use grafeo_storage::file::GrafeoFileManager;

/// Three rows and 128 bytes per chunk: the records of [`SCHEMA`] fill many
/// stream pieces.
const SMALL: ChunkCaps = ChunkCaps {
    max_rows: 3,
    max_bytes: 128,
};

/// The DDL of `scripts/released_fixtures.py` (types with a default, a parent
/// type, endpoints, a graph type, a typed graph, named and unnamed
/// constraints, property, vector and text indexes with names), plus a schema,
/// a procedure, the `KEY` labels of an inline graph type and the property
/// types of #569. The documents give the vector index its dimensions.
const SCHEMA: &[&str] = &[
    "CREATE NODE TYPE City (name STRING NOT NULL, country STRING DEFAULT 'NL')",
    "CREATE NODE TYPE Capital EXTENDS City (since INT64)",
    "CREATE EDGE TYPE ROUTE CONNECTING (City) TO (City) (km INT64 DEFAULT 88)",
    "CREATE GRAPH TYPE travel (NODE TYPE City, EDGE TYPE ROUTE)",
    "CREATE CONSTRAINT person_email FOR (p:Person) ON (p.email) UNIQUE",
    "CREATE CONSTRAINT FOR (p:Person) ON (p.name) NOT NULL",
    "CREATE INDEX person_name FOR (p:Person) ON (p.name)",
    "INSERT (:Document {title: 'Canals', content: 'boats on the canals of Amsterdam', \
     embedding: vector([3.0, 19.0, 88.0])}), \
     (:Document {title: 'Bridges', content: 'bridges over the river in Prague', \
     embedding: vector([88.0, 19.0, 3.0])})",
    "CREATE VECTOR INDEX doc_embedding ON :Document(embedding)",
    "CREATE INDEX doc_content FOR (d:Document) ON (d.content) USING TEXT",
    "CREATE GRAPH trips TYPED travel",
    "CREATE SCHEMA archive",
    "CREATE PROCEDURE capitals() RETURNS (name STRING) AS { MATCH (c:Capital) RETURN c.name AS name }",
    "CREATE GRAPH TYPE itinerary (NODE TYPE Stop KEY (StopKey) (arrives ZONED DATETIME), \
     EDGE TYPE LEG KEY (LegKey) (departs LIST<LOCAL DATETIME>))",
    "CREATE NODE TYPE Event (begins ZONED DATETIME, departs LOCAL DATETIME, \
     tags LIST<STRING>, scores LIST<LIST<INT64>>)",
    "CREATE EDGE TYPE BOOKED CONNECTING (Event) TO (City) (stamps LIST<ZONED DATETIME>)",
];

/// The `SHOW` statements whose output a reopen must keep.
const SHOWN: &[&str] = &[
    "SHOW NODE TYPES",
    "SHOW EDGE TYPES",
    "SHOW GRAPH TYPES",
    "SHOW CONSTRAINTS",
    "SHOW INDEXES",
    "SHOW GRAPHS",
    "SHOW SCHEMAS",
];

fn rows(db: &GrafeoDB, query: &str) -> Vec<Vec<Value>> {
    db.execute(query)
        .unwrap_or_else(|error| panic!("{query}: {error}"))
        .rows()
        .to_vec()
}

/// The value of the only row and column of `query`.
fn single(db: &GrafeoDB, query: &str) -> Value {
    let rows = rows(db, query);
    assert_eq!(rows.len(), 1, "{query}: {rows:?}");
    rows[0][0].clone()
}

/// Every row of every statement of [`SHOWN`], one line each, the rows of a
/// statement sorted (`SHOW INDEXES` lists them in hash-map order).
fn show_everything(db: &GrafeoDB) -> Vec<String> {
    SHOWN
        .iter()
        .flat_map(|query| {
            let mut lines: Vec<String> = rows(db, query)
                .iter()
                .map(|row| format!("{query}: {row:?}"))
                .collect();
            lines.sort();
            lines
        })
        .collect()
}

/// The row of `query` whose first column is `name`.
fn row_of(db: &GrafeoDB, query: &str, name: &str) -> Vec<Value> {
    rows(db, query)
        .into_iter()
        .find(|row| row[0] == Value::from(name))
        .unwrap_or_else(|| panic!("{query} lists no {name}"))
}

/// Creates the database at `path` with the chunk caps [`SMALL`], runs
/// [`SCHEMA`] and closes it (the final checkpoint runs on this thread, so
/// with these caps). Returns what [`show_everything`] showed before the
/// close.
fn write_schema(path: &Path) -> Vec<String> {
    with_chunk_caps(SMALL, || {
        let db = GrafeoDB::with_config(Config::persistent(path)).unwrap();
        for statement in SCHEMA {
            db.execute(statement)
                .unwrap_or_else(|error| panic!("{statement}: {error}"));
        }
        let shown = show_everything(&db);
        db.close().unwrap();
        shown
    })
}

#[test]
fn the_schema_survives_a_reopen_after_a_checkpoint_in_small_chunks() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("schema.grafeo");
    let before = write_schema(&path);

    let db = GrafeoDB::with_config(Config::persistent(&path)).unwrap();
    assert_eq!(show_everything(&db), before);

    // What the comparison covers holds what the statements made, so the
    // comparison cannot pass on empty output.
    assert_eq!(
        row_of(&db, "SHOW NODE TYPES", "Event")[1],
        Value::from(
            "begins ZONED DATETIME, departs LOCAL DATETIME, tags LIST<STRING>, \
             scores LIST<LIST<INT64>>"
        ),
        "the property types of #569"
    );
    assert_eq!(
        row_of(&db, "SHOW NODE TYPES", "Stop")[1..4],
        [
            Value::from("arrives ZONED DATETIME"),
            Value::from(""),
            Value::from("StopKey")
        ],
        "an inline element type, its key label also its parent"
    );
    assert_eq!(
        row_of(&db, "SHOW NODE TYPES", "Capital")[3],
        Value::from("City"),
        "the parent type"
    );
    assert_eq!(
        row_of(&db, "SHOW EDGE TYPES", "ROUTE")[1..4],
        [
            Value::from("km INT64"),
            Value::from("City"),
            Value::from("City")
        ],
        "the endpoints"
    );
    assert_eq!(
        row_of(&db, "SHOW EDGE TYPES", "BOOKED")[1..4],
        [
            Value::from("stamps LIST<ZONED DATETIME>"),
            Value::from("Event"),
            Value::from("City")
        ]
    );
    assert_eq!(
        row_of(&db, "SHOW EDGE TYPES", "LEG")[1],
        Value::from("departs LIST<LOCAL DATETIME>")
    );
    assert_eq!(
        row_of(&db, "SHOW GRAPH TYPES", "travel")[2..4],
        [Value::from("City"), Value::from("ROUTE")]
    );
    let names = |query: &str| -> Vec<Value> {
        let mut names: Vec<Value> = rows(&db, query)
            .into_iter()
            .map(|row| row[0].clone())
            .collect();
        names.sort_by_key(ToString::to_string);
        names
    };
    assert_eq!(
        names("SHOW CONSTRAINTS"),
        ["Person_name_not_null", "person_email"].map(Value::from)
    );
    assert_eq!(
        names("SHOW INDEXES"),
        ["doc_content", "person_name"].map(Value::from),
        "SHOW INDEXES leaves out vector indexes"
    );
    assert_eq!(names("SHOW GRAPHS"), [Value::from("trips")]);
    assert_eq!(names("SHOW SCHEMAS"), [Value::from("archive")]);

    // The entries work, not only show.
    db.execute("INSERT (:City {name: 'Prague'})").unwrap();
    assert_eq!(
        single(&db, "MATCH (c:City {name: 'Prague'}) RETURN c.country"),
        Value::from("NL"),
        "the default"
    );
    let duplicate = db
        .execute(
            "INSERT (:Person {name: 'Gus', email: 'gus@example.org'}), \
                  (:Person {name: 'Mia', email: 'gus@example.org'})",
        )
        .unwrap_err()
        .to_string();
    assert!(
        duplicate.contains("UNIQUE"),
        "the named constraint: {duplicate}"
    );
    let unnamed = db
        .execute("INSERT (:Person {email: 'gus@example.org'})")
        .unwrap_err()
        .to_string();
    assert!(
        unnamed.contains("NOT NULL"),
        "the unnamed NOT NULL constraint: {unnamed}"
    );
    db.execute("DROP CONSTRAINT person_email").unwrap();
    db.execute("DROP CONSTRAINT Person_name_not_null").unwrap();
    db.execute("INSERT (:Person {email: 'gus@example.org'}), (:Person {email: 'gus@example.org'})")
        .unwrap();
    db.execute("DROP INDEX person_name").unwrap();
    db.execute("DROP PROCEDURE capitals").unwrap();
    db.execute("DROP SCHEMA archive").unwrap();

    // The vector and text indexes came back from their definitions.
    let id = |title: &str| match single(
        &db,
        &format!("MATCH (d:Document {{title: '{title}'}}) RETURN id(d)"),
    ) {
        Value::Int64(id) => NodeId::new(u64::try_from(id).unwrap()),
        other => panic!("an id: {other:?}"),
    };
    let nearest = db
        .vector_search("Document", "embedding", &[3.0, 19.0, 88.0], 1, None, None)
        .unwrap();
    assert_eq!(
        nearest.iter().map(|(node, _)| *node).collect::<Vec<_>>(),
        [id("Canals")],
        "the vector index"
    );
    let matches = db
        .text_search("Document", "content", "bridges", 3, None)
        .unwrap();
    assert_eq!(
        matches.iter().map(|(node, _)| *node).collect::<Vec<_>>(),
        [id("Bridges")],
        "the text index"
    );

    // `trips` is still bound to `travel`: a graph type made LIKE it copies
    // the types of `travel`. Without the binding, LIKE takes every node and
    // edge type of the catalog.
    db.execute("CREATE GRAPH TYPE trips_copy LIKE trips")
        .unwrap();
    assert_eq!(
        row_of(&db, "SHOW GRAPH TYPES", "trips_copy")[2..4],
        [Value::from("City"), Value::from("ROUTE")],
        "the binding of trips"
    );
    db.close().unwrap();
}

/// After a close, the catalog section of the file is version 2: its
/// metadata chunk (the layout byte and the byte cap), then the framed
/// records as pieces of stream 0 of graph 0, each piece but the last as long
/// as the cap. The records come grouped by kind, in the order of their
/// kinds, and every kind of this release is there.
#[test]
fn the_catalog_section_is_version_2_and_has_no_raw_chunk() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("schema.grafeo");
    write_schema(&path);

    let file = GrafeoFileManager::open_read_only(&path, None).unwrap();
    let kinds = file
        .read_image(|image| {
            let source = image
                .section_source(SectionType::Catalog)
                .expect("a catalog section");
            assert_eq!(source.section_version(), 2);
            let chunks = source.chunks();
            assert_eq!(chunks[0], ChunkMeta::meta(), "the metadata chunk first");
            assert_eq!(
                source.fetch(0).unwrap().as_ref(),
                [1, 128],
                "layout 1 and the byte cap, as bincode"
            );
            assert!(chunks.len() > 3, "{} chunks", chunks.len());
            let mut offset = 0;
            for (index, chunk) in chunks.iter().enumerate().skip(1) {
                assert_eq!(
                    *chunk,
                    ChunkMeta::stream_piece(0, 0, offset),
                    "chunk {index}"
                );
                let length = source.fetch(index).unwrap().len();
                if index + 1 < chunks.len() {
                    assert_eq!(length, 128, "piece {index} is as long as the cap");
                } else {
                    assert!((1..=128).contains(&length), "the last piece: {length}");
                }
                offset += u64::try_from(length).unwrap();
            }

            let mut kinds = Vec::new();
            let mut reader = ChunkStreamReader::new(source.as_ref(), 0, 0);
            read_catalog_records(&mut reader, &mut |record| {
                kinds.push(record.kind());
                Ok(())
            })?;
            Ok(kinds)
        })
        .unwrap();
    assert!(
        kinds.is_sorted(),
        "the records come grouped by kind, in the order of their kinds: {kinds:?}"
    );
    let mut distinct = kinds.clone();
    distinct.dedup();
    assert_eq!(distinct, (1..=9).collect::<Vec<u8>>(), "{kinds:?}");
}
