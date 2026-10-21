//! The paths of earlier formats, snapshots and copies on a row-group root
//! (#412 P5): every database a released Grafeo wrote opens into the
//! row-group store, the engine's own read-only open of each fixture being
//! the reference, and checkpoints to
//! version 4 holding the same; snapshots export, import and restore; a graph
//! copy holds what a reader at its epoch sees; embeddings a 0.5.x build
//! spilled come back.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use grafeo_common::change::StandaloneOp;
use grafeo_common::types::{EpochId, PropertyKey, Value};
use grafeo_common::utils::error::Result;
use grafeo_core::graph::GraphStore;
use grafeo_storage::file::GrafeoFileManager;
use grafeo_storage::file::detect::{OnDisk, detect};

use super::super::super::catalog_section::CatalogStore;
use super::super::super::persistence::row_groups::{self, Parts};
use super::super::super::sections::LoadedSections;
use super::super::super::standalone;
use super::super::{continue_row_group_epochs, load_row_group_sections, restore_row_group_indexes};
use super::{
    Database, EdgeView, NodeView, assert_holds, committed_of, images, open_writes, travels,
};
use crate::GrafeoDB;
use crate::transaction::StandaloneChange;

/// A copy of the fixture directory `relative` (under `tests/fixtures`) in a
/// new temporary directory, and the path of `name` in it: a fixture is read
/// in place, and is never changed.
fn fixture(relative: &str, name: &str) -> (tempfile::TempDir, PathBuf) {
    fn copy_tree(from: &Path, to: &Path) {
        std::fs::create_dir_all(to).unwrap();
        for entry in std::fs::read_dir(from).unwrap() {
            let entry = entry.unwrap();
            let target = to.join(entry.file_name());
            if entry.file_type().unwrap().is_dir() {
                copy_tree(&entry.path(), &target);
            } else {
                std::fs::copy(entry.path(), target).unwrap();
            }
        }
    }
    let dir = tempfile::tempdir().unwrap();
    copy_tree(
        &Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures")
            .join(relative),
        dir.path(),
    );
    let path = dir.path().join(name);
    (dir, path)
}

/// Every file under `dir` with its bytes, by path.
fn bytes_under(dir: &Path) -> BTreeMap<PathBuf, Vec<u8>> {
    let mut files = BTreeMap::new();
    for entry in std::fs::read_dir(dir).unwrap() {
        let entry = entry.unwrap();
        if entry.file_type().unwrap().is_dir() {
            files.extend(bytes_under(&entry.path()));
        } else {
            files.insert(entry.path(), std::fs::read(entry.path()).unwrap());
        }
    }
    files
}

/// The parts of `database` a snapshot reads and writes.
fn parts(database: &Database) -> Parts<'_> {
    Parts {
        root: &database.root,
        catalog: &database.catalog,
        #[cfg(feature = "triple-store")]
        rdf: &database.rdf_store,
    }
}

/// A database on a row-group root holding the 0.5.x database at `path`,
/// read as the engine's read-only open reads it (its file
/// and sidecar WAL, or the WAL of its directory, through the same
/// functions), with every index built and the epochs continued.
fn read_0_5(path: &Path) -> Result<Database> {
    let database = Database::new();
    let mut loaded = LoadedSections::default();
    if detect(path)? == OnDisk::WalDirectory {
        GrafeoDB::load_legacy_directory(
            path,
            &database.root,
            &database.catalog,
            #[cfg(feature = "triple-store")]
            &database.rdf_store,
            false,
        )?;
    } else {
        GrafeoDB::load_legacy_file(
            path,
            &database.root,
            &database.catalog,
            #[cfg(feature = "triple-store")]
            &database.rdf_store,
            &mut loaded,
            false,
        )?;
    }
    restore_row_group_indexes(&database.root, std::mem::take(loaded.unbuilt_mut()));
    continue_row_group_epochs(&database.root, &database.transaction_manager);
    Ok(database)
}

/// A database on a row-group root holding the 0.6 file at `path`, read as
/// the engine's open reads it: its sections (an LPG section of version 3
/// migrated) and the epoch of its header. The fixtures have no sidecar WAL
/// (the replay of a WAL of this release onto the row-group store comes with
/// the switch, H2d).
fn read_0_6(path: &Path) -> Result<Database> {
    let database = Database::new();
    let fm = GrafeoFileManager::open_read_only(path, None)?;
    assert!(!fm.has_sidecar_wal(), "{}: no sidecar WAL", path.display());
    fm.read_image(|image| {
        load_row_group_sections(
            image,
            Some(path),
            &database.root,
            &database.catalog,
            #[cfg(feature = "triple-store")]
            &database.rdf_store,
        )
        .map(|history| assert_eq!(history, None, "a fixture keeps no history"))
    })?;
    database
        .root
        .sync_epoch(EpochId::new(fm.active_header().epoch));
    continue_row_group_epochs(&database.root, &database.transaction_manager);
    Ok(database)
}

/// The indexes of a graph: its property index keys, and the keys of its
/// vector and text indexes, sorted.
fn indexes_of<S: CatalogStore>(store: &S) -> [Vec<String>; 3] {
    let mut property = store.property_index_keys();
    #[cfg(feature = "vector-index")]
    let mut vector: Vec<String> = store
        .vector_index_entries()
        .into_iter()
        .map(|(key, _)| key)
        .collect();
    #[cfg(not(feature = "vector-index"))]
    let mut vector: Vec<String> = Vec::new();
    #[cfg(feature = "text-index")]
    let mut text: Vec<String> = store
        .text_index_entries()
        .into_iter()
        .map(|(key, _)| key)
        .collect();
    #[cfg(not(feature = "text-index"))]
    let mut text: Vec<String> = Vec::new();
    for keys in [&mut property, &mut vector, &mut text] {
        keys.sort();
    }
    [property, vector, text]
}

/// What a catalog defines, as names.
fn definitions(catalog: &crate::catalog::Catalog) -> Vec<String> {
    let mut names: Vec<String> = catalog
        .all_node_type_defs()
        .into_iter()
        .map(|def| format!("node type {}", def.name))
        .chain(
            catalog
                .all_edge_type_defs()
                .into_iter()
                .map(|def| format!("edge type {}", def.name)),
        )
        .chain(
            catalog
                .schema_names()
                .into_iter()
                .map(|name| format!("schema {name}")),
        )
        .chain(
            catalog
                .constraints()
                .into_iter()
                .map(|def| format!("constraint {}", def.name)),
        )
        .collect();
    names.push(format!("{} index names", catalog.index_count()));
    names.sort();
    names
}

/// What a query sees of a graph of the engine's store: its rows, without
/// the edges whose end the graph does not hold (a 0.5.43 WAL directory
/// replays some into a named graph), which 0.5.x kept and the row-group
/// store leaves out, and without null values, which 0.5.x kept and the
/// row-group store never holds (a null value is no value, #412 decision
/// Q8).
fn seen(store: &dyn GraphStore) -> (BTreeMap<u64, NodeView>, BTreeMap<u64, EdgeView>) {
    let (mut nodes, mut edges) = committed_of(store);
    edges.retain(|_, (src, dst, _, _)| nodes.contains_key(src) && nodes.contains_key(dst));
    for (_, values) in nodes.values_mut() {
        values.retain(|(_, value)| !value.is_null());
    }
    for (_, _, _, values) in edges.values_mut() {
        values.retain(|(_, value)| !value.is_null());
    }
    (nodes, edges)
}

/// Asserts the row-group `database` holds what the engine's `db` holds:
/// every graph's rows and indexes, the catalog's definitions, the RDF
/// triples, and the epoch the database continues at.
fn assert_same_database(db: &GrafeoDB, database: &Database, what: &str) {
    let store = db.lpg_store();
    assert_eq!(
        committed_of(&*database.root),
        seen(&*store),
        "{what}: the default graph"
    );
    assert_eq!(
        indexes_of(&*database.root),
        indexes_of(&*store),
        "{what}: the default graph's indexes"
    );
    let mut names = store.graph_names();
    names.sort();
    let mut row_names = database.root.graph_names();
    row_names.sort();
    assert_eq!(row_names, names, "{what}: the named graphs");
    for name in &names {
        let (graph, reference) = (
            database.root.graph(name).unwrap(),
            store.graph(name).unwrap(),
        );
        assert_eq!(
            committed_of(&*graph),
            seen(&*reference),
            "{what}: graph {name}"
        );
        assert_eq!(
            indexes_of(&*graph),
            indexes_of(&*reference),
            "{what}: the indexes of graph {name}"
        );
    }
    assert_eq!(
        definitions(&database.catalog),
        definitions(&db.catalog),
        "{what}: the catalog"
    );
    #[cfg(feature = "triple-store")]
    assert_eq!(
        database.rdf_store.len(),
        db.rdf_store.len(),
        "{what}: the RDF triples"
    );
    // Both give a replayed commit an epoch of its own.
    assert_eq!(
        database.transaction_manager.current_epoch(),
        db.transaction_manager.current_epoch(),
        "{what}: the epoch"
    );
}

/// Both reads of a fixture, or `None` when this build lacks a feature the
/// fixture's data needs (RDF triples without `triple-store`, an index without
/// its feature): then the row-group read refuses it as the engine does, with
/// the error that names the feature, and drops nothing silently.
fn both_read(
    read: Result<Database>,
    engine: Result<GrafeoDB>,
    what: &str,
) -> Option<(Database, GrafeoDB)> {
    const NEEDS_A_FEATURE: &str = "only a build with the";
    match (read, engine) {
        (Ok(database), Ok(db)) => Some((database, db)),
        (Err(read), Err(engine)) => {
            let (read, engine) = (read.to_string(), engine.to_string());
            assert!(
                read.contains(NEEDS_A_FEATURE) && engine.contains(NEEDS_A_FEATURE),
                "{what}: both refuse, for a feature this build lacks: {read} / {engine}"
            );
            None
        }
        (read, engine) => panic!(
            "{what}: the row-group read and the engine disagree: {:?} / {:?}",
            read.err().map(|error| error.to_string()),
            engine.err().map(|error| error.to_string())
        ),
    }
}

/// Asserts what `read` loads of the database at `path` matches the engine's
/// read-only open of it, leaves every byte under `dir` as it was, and that
/// a checkpoint of it (the migration's write, version 4) reopens with the
/// same, its versions costing what a loaded root's do.
fn assert_migrates(dir: &Path, path: &Path, read: fn(&Path) -> Result<Database>, what: &str) {
    let before = bytes_under(dir);
    let read = read(path);
    assert_eq!(
        bytes_under(dir),
        before,
        "{what}: the files stay as they were"
    );
    let engine = GrafeoDB::open_read_only(path);
    let Some((database, db)) = both_read(read, engine, what) else {
        return;
    };
    assert_same_database(&db, &database, what);
    assert!(db.node_count() > 0, "{what}: the fixture holds data");

    let (image, _) = database.image();
    let reopened = Database::load(&image, None).unwrap();
    assert_same_database(&db, &reopened, &format!("{what}, checkpointed"));
    assert_eq!(
        database.root.version_bytes(),
        reopened.root.version_bytes(),
        "{what}: the migrated versions are collapsed"
    );
}

/// Every database 0.5.43 and 0.5.44 wrote in each layout (closed, with
/// changes only in its sidecar WAL, a WAL directory, a compacted base with
/// the writes after it) opens into the row-group store as the engine opens
/// it, through the replay writer.
#[test]
fn released_0_5_databases_open_into_row_groups() {
    for (version, name) in [
        ("0.5.43", "closed.grafeo"),
        ("0.5.43", "unflushed.grafeo"),
        ("0.5.43", "directory"),
        ("0.5.44", "closed.grafeo"),
        ("0.5.44", "unflushed.grafeo"),
        ("0.5.44", "directory"),
        ("0.5.44", "compacted.grafeo"),
    ] {
        let (dir, path) = fixture(&format!("released/{version}"), name);
        assert_migrates(dir.path(), &path, read_0_5, &format!("{version}/{name}"));
    }
}

/// A 0.5.44 file compacted, written to after, then left with commits in its
/// sidecar WAL by a crash; and one whose compacted base joined labels: the
/// fold and the replay of the first log format on the row-group store.
#[test]
fn compacted_0_5_files_fold_into_row_groups() {
    for (relative, name) in [
        ("compacted-crashed/0.5.44", "crashed.grafeo"),
        ("compacted-labels/0.5.44", "labels.grafeo"),
    ] {
        let (dir, path) = fixture(relative, name);
        assert_migrates(dir.path(), &path, read_0_5, relative);
    }
}

/// Files of 0.6.0-beta (LPG section version 3) with RDF triples, and with
/// vector and text indexes, migrate into the row-group store.
#[test]
fn beta_files_of_version_3_open_into_row_groups() {
    for (relative, name) in [
        ("rdf/0.6.0-dev", "triples.grafeo"),
        ("search-indexes/0.6.0-dev", "documents.grafeo"),
    ] {
        let (dir, path) = fixture(relative, name);
        assert_migrates(dir.path(), &path, read_0_6, relative);
    }
}

/// Embeddings 0.5.44 spilled to `<file>.spill` come back into the row-group
/// store's columns as 0.5.44 read them back: the newer value in
/// the file wins, a node deleted while spilled gets none (see the fixture's
/// README).
#[cfg(feature = "vector-index")]
#[test]
fn embeddings_a_0_5_build_spilled_fold_into_row_groups() {
    use super::super::super::legacy_spill;

    let (dir, path) = fixture("closed-while-spilled/0.5.44", "spilled.grafeo");
    let database = read_0_5(&path).unwrap();
    let files = legacy_spill::find(&dir.path().join("spilled.grafeo.spill"), true);
    assert_eq!(files.len(), 1, "the spill file of :Item(embedding)");
    let folded = legacy_spill::fold_in(&*database.root, &files);
    assert_eq!(folded.files.len(), 1, "read through, with nothing kept");

    let db = GrafeoDB::open_read_only(&path).unwrap();
    assert_eq!(
        committed_of(&*database.root),
        committed_of(&*db.lpg_store()),
        "the embeddings as the engine folds them"
    );
    let key = PropertyKey::new("embedding");
    let embeddings: Vec<Option<Value>> = database
        .root
        .nodes_by_label("Item")
        .into_iter()
        .map(|id| database.root.get_node_property(id, &key))
        .collect();
    assert_eq!(embeddings.len(), 3, "Alix, Gus and Jules; Vincent is gone");
    assert!(embeddings.iter().all(Option::is_some));
    let index = database.root.get_vector_index("Item", "embedding").unwrap();
    assert_eq!(index.len(), 3, "the index holds every folded embedding");
}

/// Snapshots of version 4 (0.5.x, 0.6.0-beta.1) and 5 import into a
/// row-group root as `import_snapshot` imports them.
#[test]
fn snapshot_fixtures_import_into_row_groups() {
    for relative in [
        "snapshot_v4.bin",
        "snapshot_v5.bin",
        "snapshots/0.6.0-dev/documents.snapshot",
    ] {
        let data = std::fs::read(
            Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("tests/fixtures")
                .join(relative),
        )
        .unwrap();
        let database = Database::new();
        let read = row_groups::restore(&parts(&database), &data, "import").map(|()| database);
        let Some((database, db)) = both_read(read, GrafeoDB::import_snapshot(&data), relative)
        else {
            continue;
        };
        continue_row_group_epochs(&database.root, &database.transaction_manager);
        assert_same_database(&db, &database, relative);
    }
}

/// A snapshot exported while transactions are open in the default graph
/// and in a named graph holds each graph's committed state, read at the
/// export's epoch without a committed copy; it imports into a row-group
/// root with every index, and into a database alike.
#[test]
fn a_snapshot_export_during_open_writes_round_trips() {
    let database = Database::new();
    let travellers = travels(&database);
    let committed = images(&database);
    let open = open_writes(&database, travellers);
    let data = {
        let _commits = database.transaction_manager.hold_commits().unwrap();
        row_groups::export(
            &parts(&database),
            database.transaction_manager.current_epoch(),
        )
        .unwrap()
    };
    drop(open);

    let imported = Database::new();
    row_groups::restore(&parts(&imported), &data, "import").unwrap();
    continue_row_group_epochs(&imported.root, &imported.transaction_manager);
    assert_holds(&imported, &committed, "the import");
    assert!(
        imported.transaction_manager.current_epoch() >= database.root.current_epoch(),
        "the import continues at the export's epoch"
    );

    let db = GrafeoDB::import_snapshot(&data).unwrap();
    assert_eq!(committed_of(&*db.lpg_store()), committed[0], "the database");
    assert_eq!(
        committed_of(&*db.lpg_store().graph("trips").unwrap()),
        committed[1],
        "the database, trips"
    );
    assert_eq!(
        indexes_of(&*db.lpg_store()),
        indexes_of(&*imported.root),
        "the indexes"
    );
}

/// A restore replaces the whole database, named graphs and indexes
/// included; a snapshot that does not decode changes nothing.
#[test]
fn a_snapshot_restore_replaces_the_row_group_database() {
    let source = Database::new();
    source.write(None, |writer| {
        writer
            .create_node(
                &["Person".to_string()],
                vec![("name".to_string(), Value::from("Mia"))],
            )
            .unwrap();
    });
    let data =
        row_groups::export(&parts(&source), source.transaction_manager.current_epoch()).unwrap();

    let database = Database::new();
    travels(&database);
    let before = images(&database);
    assert!(row_groups::restore(&parts(&database), &[5, 3, 19], "restore").is_err());
    assert_eq!(
        images(&database),
        before,
        "a refused restore changes nothing"
    );

    row_groups::restore(&parts(&database), &data, "restore").unwrap();
    assert_eq!(committed_of(&*database.root), committed_of(&*source.root));
    assert!(database.root.graph_names().is_empty(), "trips is gone");
    assert!(
        indexes_of(&*database.root).iter().all(Vec::is_empty),
        "the indexes of the snapshot only: none"
    );
}

/// `CREATE GRAPH ... AS COPY OF` on a row-group root: the copy holds every
/// node and edge a reader at the published epoch sees, through the replay
/// writer, with nothing of the transactions still open.
#[test]
fn a_graph_copy_holds_the_committed_state() {
    let database = Database::new();
    let travellers = travels(&database);
    let committed = images(&database);
    let open = open_writes(&database, travellers);
    for (source, dest, expected) in [
        (None, "copy", &committed[0]),
        (Some("trips"), "trips copy", &committed[1]),
    ] {
        let held = standalone::hold(&database.transaction_manager, false).unwrap();
        let mut change = StandaloneChange::new();
        change.push(StandaloneOp::CreateGraph {
            name: dest.to_string(),
        });
        standalone::commit_copy(
            change,
            &held,
            None,
            &database.root,
            &database.catalog,
            &database.transaction_manager,
            (&*database.store(source), dest),
        )
        .unwrap();
        drop(held);
        let copy = database.root.graph(dest).unwrap();
        assert_eq!(&committed_of(&*copy), expected, "{dest}");
        assert_eq!(
            copy.current_epoch(),
            database.transaction_manager.current_epoch()
        );
    }
    // The open transactions commit in the source; the copies are graphs of
    // their own, which keep the state they were copied with.
    database.commit(open.root);
    database.commit(open.trips);
    let bio = PropertyKey::new("bio");
    assert_eq!(
        database.root.get_node_property(travellers.alix, &bio),
        Some(Value::from("Alix moved to Berlin"))
    );
    assert_eq!(
        database
            .root
            .graph("copy")
            .unwrap()
            .get_node_property(travellers.alix, &bio),
        Some(Value::from("Alix lives in Amsterdam"))
    );
}
