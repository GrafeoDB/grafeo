//! The index definitions of a 0.5.x container v1 file.
//!
//! A 0.5.x container v1 file holds its database as one snapshot blob
//! (version 4): its nodes and edges, its schema and the definitions of its
//! property, vector and text indexes. An open of such a file reads the
//! snapshot with `persistence::load_snapshot_into_store`, which restored the
//! data and the schema but never read the index definitions (the import of
//! the same snapshot does, `restore_row_group_indexes`): the migration wrote
//! a 0.6 file without the indexes, and a read-only open had none either, so
//! vector and text searches failed until the user created them again.
//!
//! A build without `vector-index` or `text-index` cannot build such an index,
//! so it refuses such a file, as it refuses other data it cannot read, and
//! leaves it as it is.
//!
//! The snapshot is encoded by hand, as 0.5.x encoded it (bincode of the
//! version 4 snapshot, see `tests/fixtures/snapshots/README.md` for one 0.5.x
//! exported); [`the_hand_encoded_snapshot_holds_the_three_indexes`] checks
//! that an import reads it with its indexes.
//!
//! ```bash
//! cargo test -p grafeo-engine --features full --test snapshot_file_indexes
//! cargo test -p grafeo-engine --no-default-features --features gql,wal,grafeo-file \
//!     --test snapshot_file_indexes
//! ```

#![cfg(all(feature = "gql", feature = "wal", feature = "grafeo-file", not(miri)))]

use std::path::{Path, PathBuf};

use grafeo_common::types::Value;
use grafeo_engine::GrafeoDB;
use grafeo_storage::file::detect::{OnDisk, detect};

/// The documents of the snapshot: title, content, embedding and author.
const DOCUMENTS: [(&str, &str, [f32; 3], &str); 3] = [
    (
        "Amsterdam",
        "boats on the canals of Amsterdam",
        [1.0, 0.0, 0.0],
        "Alix",
    ),
    (
        "Berlin",
        "museums and parks of Berlin",
        [0.0, 1.0, 0.0],
        "Gus",
    ),
    (
        "Paris",
        "cafes along the river in Paris",
        [0.0, 0.0, 1.0],
        "Vincent",
    ),
];

/// A bincode varint of `value`: one byte below 251, as every count and
/// length here is.
fn varint(blob: &mut Vec<u8>, value: usize) {
    blob.push(u8::try_from(value).ok().filter(|byte| *byte < 251).unwrap());
}

/// A bincode string: its length, then its bytes.
fn string(blob: &mut Vec<u8>, text: &str) {
    varint(blob, text.len());
    blob.extend_from_slice(text.as_bytes());
}

/// A property with one version, at epoch 1: the name, a list of one
/// `(epoch, value)` pair, then the value's variant and contents.
fn property(blob: &mut Vec<u8>, name: &str, value: impl FnOnce(&mut Vec<u8>)) {
    string(blob, name);
    blob.extend_from_slice(&[1, 1]);
    value(blob);
}

/// A version 4 snapshot, as 0.5.x wrote it into a container v1 file: three
/// `:Document` nodes (ids 0 to 2) with a title, a content, an embedding of
/// 3 dimensions and an author, no edges, no named graphs, no RDF, an empty
/// schema, and three index definitions: a property index on `author`, a
/// vector index on `:Document(embedding)` (3 dimensions, Euclidean, `m` 16,
/// `ef_construction` 64) and a text index on `:Document(content)`.
fn snapshot_with_indexes() -> Vec<u8> {
    let mut blob = vec![4];
    varint(&mut blob, DOCUMENTS.len());
    for (id, (title, content, embedding, author)) in DOCUMENTS.iter().enumerate() {
        varint(&mut blob, id);
        varint(&mut blob, 1);
        string(&mut blob, "Document");
        // The properties, sorted by name; Int64 is variant 2, String 4 and
        // Vector 13.
        varint(&mut blob, 4);
        property(&mut blob, "author", |blob| {
            blob.push(4);
            string(blob, author);
        });
        property(&mut blob, "content", |blob| {
            blob.push(4);
            string(blob, content);
        });
        property(&mut blob, "embedding", |blob| {
            blob.push(13);
            varint(blob, embedding.len());
            for value in embedding {
                blob.extend_from_slice(&value.to_le_bytes());
            }
        });
        property(&mut blob, "title", |blob| {
            blob.push(4);
            string(blob, title);
        });
    }
    // The edges, the named graphs, the RDF triples and named graphs, and the
    // six lists of the schema.
    blob.extend_from_slice(&[0; 10]);
    // The property indexes.
    varint(&mut blob, 1);
    string(&mut blob, "author");
    // The vector indexes: label, property, dimensions, metric (1 is
    // Euclidean), `m` and `ef_construction`.
    varint(&mut blob, 1);
    string(&mut blob, "Document");
    string(&mut blob, "embedding");
    blob.extend_from_slice(&[3, 1, 16, 64]);
    // The text indexes.
    varint(&mut blob, 1);
    string(&mut blob, "Document");
    string(&mut blob, "content");
    // The epoch.
    blob.push(1);
    blob
}

/// Writes at `path` a 0.5.x container v1 file: the database as one snapshot
/// blob after the headers, the form 0.5.x used before container v2.
fn write_0_5_snapshot_file(path: &Path, snapshot: &[u8]) {
    use std::io::{Seek, SeekFrom, Write};

    use grafeo_storage::file::format::DATA_OFFSET;
    use grafeo_storage::file::header::{write_db_header, write_file_header};
    use grafeo_storage::file::{DbHeader, FileHeader};

    let mut file = std::fs::File::create(path).unwrap();
    write_file_header(&mut file, &FileHeader::new()).unwrap();
    write_db_header(&mut file, 0, &DbHeader::EMPTY).unwrap();
    write_db_header(
        &mut file,
        1,
        &DbHeader {
            iteration: 1,
            checksum: crc32fast::hash(snapshot),
            snapshot_length: u64::try_from(snapshot.len()).unwrap(),
            epoch: 1,
            ..DbHeader::EMPTY
        },
    )
    .unwrap();
    file.seek(SeekFrom::Start(DATA_OFFSET)).unwrap();
    file.write_all(snapshot).unwrap();
}

/// A container v1 file around [`snapshot_with_indexes`] in a new directory,
/// and its path.
fn snapshot_file() -> (tempfile::TempDir, PathBuf) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("documents.grafeo");
    write_0_5_snapshot_file(&path, &snapshot_with_indexes());
    assert_eq!(detect(&path).unwrap(), OnDisk::LegacyFile);
    (dir, path)
}

/// `<path><suffix>`, next to the database file.
#[cfg(all(feature = "vector-index", feature = "text-index"))]
fn with_suffix(path: &Path, suffix: &str) -> PathBuf {
    let mut name = path.as_os_str().to_owned();
    name.push(suffix);
    PathBuf::from(name)
}

/// The titles of the documents, sorted.
fn titles(db: &GrafeoDB) -> Vec<Value> {
    db.execute("MATCH (d:Document) RETURN d.title AS title ORDER BY title")
        .unwrap()
        .rows()
        .iter()
        .map(|row| row[0].clone())
        .collect()
}

/// The titles of [`DOCUMENTS`], sorted.
#[cfg(all(feature = "vector-index", feature = "text-index"))]
fn expected_titles() -> Vec<Value> {
    DOCUMENTS
        .iter()
        .map(|(title, ..)| Value::from(*title))
        .collect()
}

/// Checks that `db` holds the three indexes of the snapshot and that each
/// works: the property index finds Gus's document, the vector search the
/// nearest document (also with a filter), the text search the one document
/// with the word.
#[cfg(all(feature = "vector-index", feature = "text-index"))]
fn assert_indexes(db: &GrafeoDB, context: &str) {
    use std::collections::HashMap;

    let id_of = |title: &str| {
        let rows = db
            .execute(&format!(
                "MATCH (d:Document {{title: '{title}'}}) RETURN id(d)"
            ))
            .unwrap();
        assert_eq!(rows.rows().len(), 1, "{context}: the document {title}");
        match &rows.rows()[0][0] {
            Value::Int64(id) => grafeo_common::types::NodeId::new(u64::try_from(*id).unwrap()),
            other => panic!("{context}: id(d) is an integer, got {other:?}"),
        }
    };
    let berlin = id_of("Berlin");
    // Each index is checked before the test fails, so its message lists
    // every one that is missing or wrong.
    let mut problems = Vec::new();

    if !db.has_property_index("author") {
        problems.push("no property index on author".to_string());
    }
    let by_author = db
        .find_nodes_by_property("author", &Value::from("Gus"))
        .unwrap();
    if by_author != [berlin] {
        problems.push(format!(
            "the property lookup found {by_author:?}, not Gus's document {berlin:?}"
        ));
    }

    let nodes = |found: grafeo_common::utils::error::Result<Vec<_>>| {
        found.map(|found| found.into_iter().map(|(node, _)| node).collect::<Vec<_>>())
    };
    match nodes(db.vector_search("Document", "embedding", &[0.0, 1.0, 0.0], 1, None, None)) {
        Ok(nearest) if nearest == [berlin] => {}
        other => problems.push(format!(
            "the vector search did not find the nearest document {berlin:?}: {other:?}"
        )),
    }
    let filters: HashMap<String, Value> = [("author".to_string(), Value::from("Gus"))].into();
    match nodes(db.vector_search(
        "Document",
        "embedding",
        &[0.0, 0.0, 1.0],
        3,
        None,
        Some(&filters),
    )) {
        Ok(filtered) if filtered == [berlin] => {}
        other => problems.push(format!(
            "a filtered vector search did not find {berlin:?} alone: {other:?}"
        )),
    }

    let found = db
        .text_search("Document", "content", "museums", 3, None)
        .map(|found| found.into_iter().map(|(node, _)| node).collect::<Vec<_>>());
    match found {
        Ok(found) if found == [berlin] => {}
        other => problems.push(format!(
            "the text search did not find {berlin:?} alone: {other:?}"
        )),
    }
    assert!(problems.is_empty(), "{context}: {}", problems.join("; "));
}

/// Control: the hand-encoded snapshot is one 0.5.x wrote, with its three
/// indexes: `import_snapshot` reads it, and the import restores them.
#[cfg(all(feature = "vector-index", feature = "text-index"))]
#[test]
fn the_hand_encoded_snapshot_holds_the_three_indexes() {
    let db = GrafeoDB::import_snapshot(&snapshot_with_indexes()).unwrap();
    assert_eq!(titles(&db), expected_titles(), "imported");
    assert_indexes(&db, "imported");
}

/// Control: a read-write open migrates the container v1 file with its data,
/// keeps it as `<path>.pre-0.6` byte for byte, and the migrated file opens
/// again with the same data.
#[cfg(all(feature = "vector-index", feature = "text-index"))]
#[test]
fn a_0_5_snapshot_file_with_indexes_migrates_with_its_data() {
    let (_dir, path) = snapshot_file();
    let before = std::fs::read(&path).unwrap();

    let db = GrafeoDB::open(&path).unwrap();
    assert_eq!(titles(&db), expected_titles(), "migrated");
    db.close().unwrap();
    drop(db);
    assert_eq!(detect(&path).unwrap(), OnDisk::Current);
    assert_eq!(
        std::fs::read(with_suffix(&path, ".pre-0.6")).unwrap(),
        before,
        "the 0.5.x file is kept as it was"
    );

    let db = GrafeoDB::open(&path).unwrap();
    assert_eq!(titles(&db), expected_titles(), "reopened");
    db.close().unwrap();
}

/// The migration of a container v1 file brings its property, vector and
/// text indexes into the 0.6 file: they are there and work right after the
/// migration, and after a close and a reopen of the migrated file.
#[cfg(all(feature = "vector-index", feature = "text-index"))]
#[test]
fn a_0_5_snapshot_file_keeps_its_indexes_through_the_migration() {
    let (_dir, path) = snapshot_file();

    let db = GrafeoDB::open(&path).unwrap();
    assert_indexes(&db, "migrated");
    db.close().unwrap();
    drop(db);

    let db = GrafeoDB::open(&path).unwrap();
    assert_indexes(&db, "the migrated file reopened");
    db.close().unwrap();
}

/// A read-only open of a container v1 file (which reads it in place, with
/// the same loader) has its indexes too, and changes nothing on disk.
#[cfg(all(feature = "vector-index", feature = "text-index"))]
#[test]
fn a_read_only_open_of_a_0_5_snapshot_file_has_its_indexes() {
    let (_dir, path) = snapshot_file();
    let before = std::fs::read(&path).unwrap();

    let db = GrafeoDB::open_read_only(&path).unwrap();
    assert_eq!(titles(&db), expected_titles(), "read-only");
    assert_indexes(&db, "read-only");
    db.close().unwrap();
    drop(db);
    assert_eq!(
        std::fs::read(&path).unwrap(),
        before,
        "the file is unchanged"
    );
}

/// A build without `vector-index` or `text-index` refuses a container v1
/// file that defines such an index (a read-write and a read-only open),
/// naming the feature, and changes nothing on disk: it cannot build the
/// index, and a migration would write the 0.6 file without its definition.
#[cfg(not(all(feature = "vector-index", feature = "text-index")))]
#[test]
fn a_build_without_the_index_features_refuses_a_0_5_snapshot_file_with_such_indexes() {
    let (dir, path) = snapshot_file();
    let before = std::fs::read(&path).unwrap();
    let missing: Vec<&str> = [
        (!cfg!(feature = "vector-index")).then_some("vector-index"),
        (!cfg!(feature = "text-index")).then_some("text-index"),
    ]
    .into_iter()
    .flatten()
    .collect();

    type Open = fn(&Path) -> grafeo_common::utils::error::Result<GrafeoDB>;
    let opens: [(&str, Open); 2] = [
        ("read-write", |path| GrafeoDB::open(path)),
        ("read-only", |path| GrafeoDB::open_read_only(path)),
    ];
    for (name, open) in opens {
        let error = match open(&path) {
            Ok(db) => {
                let titles = titles(&db);
                drop(db);
                panic!(
                    "{name}: the open of a file with indexes this build cannot build succeeded \
                     (documents {titles:?})"
                );
            }
            Err(error) => error.to_string(),
        };
        for feature in &missing {
            assert!(
                error.contains(&format!("`{feature}`")),
                "{name}: the error names the {feature} feature: {error}"
            );
        }
        assert_eq!(
            std::fs::read(&path).unwrap(),
            before,
            "{name}: the file is unchanged"
        );
        let names: Vec<_> = std::fs::read_dir(dir.path())
            .unwrap()
            .map(|entry| entry.unwrap().path())
            .filter(|path| {
                path.extension()
                    .is_none_or(|extension| extension != "spill")
            })
            .map(|path| path.file_name().unwrap().to_string_lossy().into_owned())
            .collect();
        assert_eq!(
            names,
            ["documents.grafeo"],
            "{name}: nothing next to the file"
        );
    }
}
