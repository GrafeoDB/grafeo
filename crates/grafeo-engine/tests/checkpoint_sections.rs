//! Every checkpoint writes the database's complete state.
//!
//! A checkpoint writes a new `.grafeo` container that holds only the sections
//! it was given, so a checkpoint that leaves a section out drops it from the
//! file. Compacted databases lost their catalog (schema) and RDF data this way,
//! and the periodic timer kept writing the pre-compaction store.
//!
//! ```bash
//! cargo test -p grafeo-engine --features full --test checkpoint_sections
//! ```

#![cfg(all(
    feature = "compact-store",
    feature = "lpg",
    feature = "grafeo-file",
    feature = "gql"
))]

use std::time::{Duration, Instant};

use grafeo_common::storage::{ChunkCaps, SectionType};
use grafeo_common::testing::chunk_caps::with_chunk_caps;
use grafeo_common::types::Value;
use grafeo_engine::config::StorageFormat;
use grafeo_engine::{Config, GrafeoDB};
use grafeo_storage::file::GrafeoFileManager;

fn config(path: &std::path::Path) -> Config {
    Config::persistent(path).with_storage_format(StorageFormat::Auto)
}

fn single_value(db: &GrafeoDB, query: &str) -> Value {
    db.session().execute(query).unwrap().rows()[0][0].clone()
}

/// Whether the active image of the database file holds `section_type`.
fn holds(fm: &GrafeoFileManager, section_type: SectionType) -> bool {
    fm.read_image(|image| Ok(image.section_source(section_type).is_some()))
        .unwrap()
}

#[test]
fn a_compacted_database_keeps_its_schema() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("db.grafeo");
    {
        let mut db = GrafeoDB::with_config(config(&path)).unwrap();
        db.session()
            .execute("CREATE NODE TYPE Person (name STRING)")
            .unwrap();
        db.session()
            .execute("INSERT (:Person {name: 'Alix'}), (:Person {name: 'Gus'})")
            .unwrap();
        db.compact().unwrap();
        db.close().unwrap();
    }

    let db = GrafeoDB::with_config(config(&path)).unwrap();
    assert_eq!(
        single_value(&db, "MATCH (p:Person) RETURN count(p)"),
        Value::Int64(2)
    );
    let types = db.session().execute("SHOW NODE TYPES").unwrap();
    assert_eq!(types.rows().len(), 1, "the node type survives: {types:?}");
}

/// The file keeps the RDF section with its triples. (Querying the triples
/// after `compact()` is a separate problem: they are hidden in memory already.)
#[test]
#[cfg(feature = "sparql")]
fn a_compacted_database_keeps_its_rdf_section() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("db.grafeo");
    {
        let mut db = GrafeoDB::with_config(config(&path)).unwrap();
        db.session()
            .execute_sparql("INSERT DATA { <http://ex/alix> <http://ex/knows> <http://ex/gus> }")
            .unwrap();
        db.session()
            .execute("INSERT (:Person {name: 'Alix'})")
            .unwrap();
        db.compact().unwrap();
        db.close().unwrap();
    }

    let db = GrafeoDB::with_config(config(&path)).unwrap();
    let fm = db.file_manager().unwrap();
    assert!(holds(fm, SectionType::RdfStore));
    // Read the store itself: SPARQL over a compacted database does not see
    // RDF data yet, a separate problem.
    assert_eq!(db.rdf_store().len(), 1, "the triple is in the section");
}

/// The header of a compacted database's checkpoint counts the compacted
/// base as well as the overlay.
#[test]
fn a_compacted_checkpoint_counts_the_base() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("db.grafeo");
    let mut db = GrafeoDB::with_config(config(&path)).unwrap();
    db.session()
        .execute("INSERT (:Person {name: 'Alix'})-[:KNOWS]->(:Person {name: 'Gus'})")
        .unwrap();
    db.compact().unwrap();
    db.session()
        .execute("INSERT (:Person {name: 'Vincent'})")
        .unwrap();
    db.wal_checkpoint().unwrap();

    let header = db.file_manager().unwrap().active_header();
    assert_eq!((header.node_count, header.edge_count), (3, 1));
    db.close().unwrap();
}

/// The timer captured the store when the database opened, so after
/// `compact()` its checkpoints wrote the old store without the compacted
/// base. A crash after such a checkpoint lost the base: the WAL files that
/// held its data were deleted.
#[test]
fn the_checkpoint_timer_writes_the_compacted_base() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("db.grafeo");
    let mut db =
        GrafeoDB::with_config(config(&path).with_checkpoint_interval(Duration::from_millis(100)))
            .unwrap();
    db.session()
        .execute("INSERT (:Person {name: 'Alix'})-[:KNOWS]->(:Person {name: 'Gus'})")
        .unwrap();
    db.compact().unwrap();

    let fm = std::sync::Arc::clone(db.file_manager().unwrap());
    let compacted_at = fm.active_header().iteration;
    let deadline = Instant::now() + Duration::from_secs(10);
    while fm.active_header().iteration <= compacted_at {
        assert!(Instant::now() < deadline, "the timer did not checkpoint");
        std::thread::sleep(Duration::from_millis(20));
    }

    assert!(
        holds(&fm, SectionType::CompactStore),
        "the timer's checkpoint contains the compacted base"
    );
    assert!(holds(&fm, SectionType::Catalog));
    db.close().unwrap();
}

/// Writes data into every kind of section: the default graph and a named
/// graph (LPG store), a triple (RDF store), a vector and a text index, and a
/// constraint (catalog).
#[cfg(all(feature = "sparql", feature = "vector-index", feature = "text-index"))]
fn write_every_kind_of_section(db: &GrafeoDB) {
    db.execute("CREATE CONSTRAINT person_email FOR (p:Person) ON (p.email) UNIQUE")
        .unwrap();
    db.execute(
        "INSERT (:Person {name: 'Alix', email: 'alix@example.org', bio: 'jazz in Amsterdam', \
         embedding: vector([3.0, 19.0, 88.0])})-[:KNOWS]->\
         (:Person {name: 'Gus', email: 'gus@example.org', bio: 'cycling in Berlin', \
         embedding: vector([88.0, 19.0, 3.0])})",
    )
    .unwrap();
    db.create_vector_index("Person", "embedding", None, None, None, None, None)
        .unwrap();
    db.create_text_index("Person", "bio").unwrap();
    db.create_graph("trips").unwrap();
    db.graph("trips")
        .unwrap()
        .execute("INSERT (:City {name: 'Paris'})-[:ROUTE {km: 1030}]->(:City {name: 'Prague'})")
        .unwrap();
    db.execute_sparql(
        "INSERT DATA { <http://example.org/alix> <http://example.org/knows> \
         <http://example.org/gus> }",
    )
    .unwrap();
}

/// Checks everything [`write_every_kind_of_section`] wrote. `what` names the
/// database in failures.
#[cfg(all(feature = "sparql", feature = "vector-index", feature = "text-index"))]
fn assert_every_kind_of_section(db: &GrafeoDB, what: &str) {
    let rows = |query: &str| {
        db.execute(query)
            .unwrap_or_else(|error| panic!("{what}: {query}: {error}"))
            .rows()
            .to_vec()
    };
    assert_eq!(
        rows("MATCH (a:Person)-[:KNOWS]->(b:Person) RETURN a.name, b.name"),
        [vec![Value::from("Alix"), Value::from("Gus")]],
        "{what}: the default graph"
    );
    let trips = db
        .graph("trips")
        .unwrap_or_else(|error| panic!("{what}: trips: {error}"));
    assert_eq!(
        trips
            .execute("MATCH (a)-[r:ROUTE]->(b) RETURN a.name, r.km, b.name")
            .unwrap()
            .rows(),
        [vec![
            Value::from("Paris"),
            Value::Int64(1030),
            Value::from("Prague")
        ]],
        "{what}: the named graph"
    );
    assert_eq!(
        db.execute_sparql("SELECT ?s ?o WHERE { ?s <http://example.org/knows> ?o }")
            .unwrap()
            .rows(),
        [vec![
            Value::from("http://example.org/alix"),
            Value::from("http://example.org/gus")
        ]],
        "{what}: the triple"
    );

    let alix = rows("MATCH (p:Person {name: 'Alix'}) RETURN id(p)")[0][0].clone();
    let ids = |found: Vec<grafeo_common::types::NodeId>| -> Vec<Value> {
        found
            .into_iter()
            .map(|node| Value::Int64(i64::try_from(node.as_u64()).unwrap()))
            .collect()
    };
    let nearest = db
        .vector_search("Person", "embedding", &[3.0, 19.0, 88.0], 1, None, None)
        .unwrap_or_else(|error| panic!("{what}: vector search: {error}"));
    assert_eq!(
        ids(nearest.into_iter().map(|(node, _)| node).collect()),
        std::slice::from_ref(&alix),
        "{what}: vector search"
    );
    let matches = db
        .text_search("Person", "bio", "jazz", 3)
        .unwrap_or_else(|error| panic!("{what}: text search: {error}"));
    assert_eq!(
        ids(matches.into_iter().map(|(node, _)| node).collect()),
        [alix],
        "{what}: text search"
    );

    // A read-only database refuses writes and the SHOW commands (they run as
    // schema commands): check the constraint on a copy of what it loaded.
    let copy = db.is_read_only().then(|| db.to_memory().unwrap());
    let unrestricted = copy.as_ref().unwrap_or(db);
    assert_eq!(
        unrestricted
            .execute("SHOW CONSTRAINTS")
            .unwrap()
            .rows()
            .iter()
            .map(|row| row[0].clone())
            .collect::<Vec<_>>(),
        [Value::from("person_email")],
        "{what}: constraints"
    );
    let duplicate = unrestricted
        .execute("INSERT (:Person {name: 'Vincent', email: 'alix@example.org'})")
        .map(|_| ())
        .unwrap_err()
        .to_string();
    assert!(
        duplicate.contains("UNIQUE"),
        "{what}: the constraint holds: {duplicate}"
    );
}

/// Writes every kind of section into a new database file with `caps` (the
/// final checkpoint, which `close()` runs on this thread, cuts the sections
/// with them) and reopens it with the default caps: the file is in container
/// v3 and every kind of section comes back from it. Returns how many chunks
/// each section has in the file.
#[cfg(all(feature = "sparql", feature = "vector-index", feature = "text-index"))]
fn every_kind_of_section_after_a_reopen(caps: ChunkCaps) -> Vec<(SectionType, usize)> {
    use grafeo_storage::file::detect::{OnDisk, detect};

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("db.grafeo");
    with_chunk_caps(caps, || {
        let db = GrafeoDB::with_config(config(&path)).unwrap();
        write_every_kind_of_section(&db);
        db.close().unwrap();
    });
    assert_eq!(detect(&path).unwrap(), OnDisk::Current, "a v3 file");

    let db = GrafeoDB::with_config(config(&path)).unwrap();
    let fm = db.file_manager().unwrap();
    let chunks: Vec<(SectionType, usize)> = [
        SectionType::Catalog,
        SectionType::LpgStore,
        SectionType::RdfStore,
        SectionType::VectorStore,
        SectionType::TextIndex,
    ]
    .into_iter()
    .map(|section_type| {
        let count = fm
            .read_image(|image| {
                Ok(image
                    .section_source(section_type)
                    .map_or(0, |source| source.chunks().len()))
            })
            .unwrap();
        (section_type, count)
    })
    .collect();
    for (section_type, count) in &chunks {
        assert!(*count > 0, "the file holds the {section_type:?} section");
    }
    assert_every_kind_of_section(&db, "reopened");
    db.close().unwrap();
    chunks
}

/// A database file is written in container v3, and every kind of section
/// comes back from it.
#[test]
#[cfg(all(feature = "sparql", feature = "vector-index", feature = "text-index"))]
fn every_kind_of_section_survives_a_reopen() {
    every_kind_of_section_after_a_reopen(ChunkCaps::DEFAULT);
}

/// Every kind of section written in chunks of one row and 64 bytes (each
/// row, and each 64 bytes of an index stream or of the catalog's records, in
/// a chunk of its own) comes back with the default caps. The small caps cut
/// the LPG store, the index sections and the catalog into more chunks than
/// the default caps do. (Chunks of three rows and 1 KiB would cut none of
/// them: each graph has two nodes, each stream is shorter. One triple is one
/// row of the RDF store; `chunked_sections.rs` cuts RDF graphs.)
#[test]
#[cfg(all(feature = "sparql", feature = "vector-index", feature = "text-index"))]
fn every_kind_of_section_survives_a_reopen_in_small_chunks() {
    let small = every_kind_of_section_after_a_reopen(ChunkCaps {
        max_rows: 1,
        max_bytes: 64,
    });
    let default = every_kind_of_section_after_a_reopen(ChunkCaps::DEFAULT);
    for ((section_type, small), (_, default)) in small.iter().zip(&default) {
        if *section_type != SectionType::RdfStore {
            assert!(
                small > default,
                "{section_type:?}: {small} chunks with the small caps, {default} with the \
                 default caps"
            );
        }
    }
}

/// A compacted database's v3 file holds the compacted base and the
/// overlay's deletions next to the other sections, and they come back.
///
/// `compact()` drops the vector and text indexes (the database has none
/// right after it), so this file has no sections for them;
/// `every_kind_of_section_survives_a_reopen` covers those.
#[test]
#[cfg(all(feature = "sparql", feature = "vector-index", feature = "text-index"))]
fn a_compacted_base_and_its_deletions_survive_a_reopen() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("db.grafeo");
    {
        let mut db = GrafeoDB::with_config(config(&path)).unwrap();
        write_every_kind_of_section(&db);
        // Mia goes into the compacted base, and her deletion into the
        // overlay's deletions.
        db.execute("INSERT (:Person {name: 'Mia', email: 'mia@example.org'})")
            .unwrap();
        db.compact().unwrap();
        db.execute("MATCH (p:Person {name: 'Mia'}) DELETE p")
            .unwrap();
        db.close().unwrap();
    }

    let db = GrafeoDB::with_config(config(&path)).unwrap();
    let fm = db.file_manager().unwrap();
    for section_type in [
        SectionType::Catalog,
        SectionType::LpgStore,
        SectionType::RdfStore,
        SectionType::CompactStore,
        SectionType::OverlayDeletions,
    ] {
        assert!(
            holds(fm, section_type),
            "the file holds the {section_type:?} section"
        );
    }
    let rows = |query: &str| db.execute(query).unwrap().rows().to_vec();
    assert_eq!(
        rows("MATCH (p:Person) RETURN p.name ORDER BY p.name"),
        [vec![Value::from("Alix")], vec![Value::from("Gus")]],
        "the base holds Alix and Gus; Mia, deleted from it, stays deleted"
    );
    assert_eq!(
        rows("MATCH (a:Person)-[:KNOWS]->(b:Person) RETURN a.name, b.name"),
        [vec![Value::from("Alix"), Value::from("Gus")]],
        "the base's edge"
    );
    assert_eq!(
        db.graph("trips")
            .unwrap()
            .execute("MATCH (a)-[r:ROUTE]->(b) RETURN a.name, r.km, b.name")
            .unwrap()
            .rows(),
        [vec![
            Value::from("Paris"),
            Value::Int64(1030),
            Value::from("Prague")
        ]],
        "the named graph"
    );
    // SPARQL over a compacted database does not see RDF data yet (see
    // `a_compacted_database_keeps_its_rdf_section`): read the store.
    assert_eq!(db.rdf_store().len(), 1, "the triple");
    assert_eq!(
        rows("SHOW CONSTRAINTS")
            .into_iter()
            .map(|row| row[0].clone())
            .collect::<Vec<_>>(),
        [Value::from("person_email")],
        "the constraint"
    );
    db.close().unwrap();
}

/// `save()` to a `.grafeo` path writes a v3 file that holds everything,
/// without a sidecar WAL, and opens read-write and read-only.
#[test]
#[cfg(all(
    feature = "wal",
    feature = "sparql",
    feature = "vector-index",
    feature = "text-index"
))]
fn save_writes_a_v3_file_with_every_section() {
    use grafeo_storage::file::detect::{OnDisk, detect, sidecar_wal_path};

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("saved.grafeo");
    let db = GrafeoDB::new_in_memory();
    write_every_kind_of_section(&db);
    db.save(&path).unwrap();
    assert_eq!(detect(&path).unwrap(), OnDisk::Current, "a v3 file");
    assert!(
        !sidecar_wal_path(&path).exists(),
        "the saved file holds everything, without a WAL"
    );

    {
        let reader = GrafeoDB::open_read_only(&path).unwrap();
        assert!(
            std::fs::File::open(&path).unwrap().try_lock().is_err(),
            "a read-only open of a v3 file holds its shared lock"
        );
        assert_every_kind_of_section(&reader, "saved, read-only");
        reader.close().unwrap();
    }
    let saved = GrafeoDB::with_config(config(&path)).unwrap();
    assert_every_kind_of_section(&saved, "saved");
    saved.close().unwrap();
}
