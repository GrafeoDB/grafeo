//! Databases written by released versions open with today's code (#427).
//!
//! `scripts/released_fixtures.py` wrote every fixture under `fixtures/released/` with a
//! release from PyPI: one database, changed in two sessions, in three layouts. These tests
//! open a copy of each and check what it holds. Where a fixture holds less than was
//! written, its expectation says why: the release did not write it, or today's code does
//! not read it yet (and the issue that will).
//!
//! Every check runs twice: on a read-write open of every fixture, and on a read-only open
//! of every fixture, which reads the 0.5.x database (its file and sidecar WAL, or the WAL
//! of its directory) in place and must leave every byte of it as it was. A read-write
//! open first migrates the database to a single file in the 0.6 format at the same path
//! and keeps the 0.5.x files as `<name>.pre-0.6` (a file with `<name>.pre-0.6.wal/`, or
//! the whole directory), so its checks read the migrated file.
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test released_formats
//! ```

#![cfg(all(
    feature = "grafeo-file",
    feature = "wal",
    feature = "triple-store",
    feature = "sparql",
    feature = "vector-index",
    feature = "text-index"
))]

use std::collections::{BTreeMap, BTreeSet};
use std::panic::AssertUnwindSafe;
use std::path::{Path, PathBuf};

use grafeo_common::types::{Date, Duration, NodeId, PropertyKey, Value, ZonedDatetime};
use grafeo_engine::GrafeoDB;
use grafeo_storage::file::detect::{OnDisk, detect};
use grafeo_storage::lock::DirectoryLock;

/// How the release left the database on disk.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Layout {
    /// `closed.grafeo`: closed cleanly, so everything is in the file.
    Closed,
    /// `unflushed.grafeo`: the second session is only in the sidecar WAL, because the
    /// process exited without `close()`.
    Unflushed,
    /// `directory/`: a WAL-directory database, closed cleanly.
    Directory,
}

#[derive(Clone, Copy)]
struct Fixture {
    version: &'static str,
    layout: Layout,
}

const FIXTURES: [Fixture; 6] = [
    Fixture {
        version: "0.5.43",
        layout: Layout::Closed,
    },
    Fixture {
        version: "0.5.43",
        layout: Layout::Unflushed,
    },
    Fixture {
        version: "0.5.43",
        layout: Layout::Directory,
    },
    Fixture {
        version: "0.5.44",
        layout: Layout::Closed,
    },
    Fixture {
        version: "0.5.44",
        layout: Layout::Unflushed,
    },
    Fixture {
        version: "0.5.44",
        layout: Layout::Directory,
    },
];

impl Fixture {
    fn file_name(self) -> &'static str {
        match self.layout {
            Layout::Closed => "closed.grafeo",
            Layout::Unflushed => "unflushed.grafeo",
            Layout::Directory => "directory",
        }
    }

    fn name(self) -> String {
        format!("{}/{}", self.version, self.file_name())
    }

    /// Whether the release wrote a WAL directory (`directory/`), not a single `.grafeo`
    /// file (with its sidecar WAL for `unflushed.grafeo`).
    fn is_directory(self) -> bool {
        self.layout == Layout::Directory
    }

    /// The directory of the release's fixtures.
    fn source(self) -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/released")
            .join(self.version)
    }

    /// Copies the fixture into a new directory, so the committed files stay as the
    /// release wrote them.
    fn copy(self) -> tempfile::TempDir {
        let dir = tempfile::tempdir().unwrap();
        // The database and, for `unflushed.grafeo`, its sidecar WAL directory.
        for entry in std::fs::read_dir(self.source()).unwrap() {
            let entry = entry.unwrap();
            if entry
                .file_name()
                .to_string_lossy()
                .starts_with(self.file_name())
            {
                copy(&entry.path(), &dir.path().join(entry.file_name()));
            }
        }
        dir
    }

    /// Opens a copy for reading and writing, which migrates it to a single file in the
    /// 0.6 format first.
    fn open(self) -> (tempfile::TempDir, GrafeoDB) {
        let dir = self.copy();
        let db = GrafeoDB::open(dir.path().join(self.file_name()))
            .unwrap_or_else(|error| panic!("{}: {error}", self.name()));
        (dir, db)
    }

    /// 0.5.43 did not log a switch back to the default graph, so its WAL directory
    /// replays every change after the first switch to a named graph into that graph (as
    /// 0.5.43 itself does): the default graph holds the first session up to that switch.
    fn replays_into_a_named_graph(self) -> bool {
        self.version == "0.5.43" && self.layout == Layout::Directory
    }

    /// Whether the first session's schema is only in WAL records, which carry no default
    /// values, parent types or edge endpoints (#517 adds them to the WAL).
    fn first_session_schema_in_wal(self) -> bool {
        self.layout == Layout::Directory
    }

    /// Whether the second session's schema is only in WAL records.
    fn second_session_schema_in_wal(self) -> bool {
        self.layout != Layout::Closed
    }

    /// The constraint names `SHOW CONSTRAINTS` lists. 0.5.43 files store no constraint
    /// names (0.5.44 added them, #420), but its WAL records do.
    fn constraint_names(self) -> &'static [&'static str] {
        match (self.version, self.layout) {
            ("0.5.43", Layout::Closed) => &[],
            ("0.5.43", Layout::Unflushed) => &["museum_name"],
            ("0.5.43", Layout::Directory) => &["Person_not_null", "museum_name", "person_email"],
            _ => &["Person_name_not_null", "museum_name", "person_email"],
        }
    }

    /// The indexes `SHOW INDEXES` lists (it leaves out vector indexes). 0.5.43 files hold
    /// no index definitions, and today's code does not replay the index records of a WAL
    /// (#401), so an index created in a WAL is missing.
    fn index_names(self) -> &'static [&'static str] {
        match (self.version, self.layout) {
            ("0.5.44", Layout::Closed) => &["doc_content", "museum_name_index", "person_name"],
            ("0.5.44", Layout::Unflushed) => &["doc_content", "person_name"],
            _ => &[],
        }
    }

    /// Whether the vector and text indexes of the first session are there. 0.5.43 wrote
    /// them to the file but lost them on reopen, so the checkpoint at the end of its second
    /// session wrote the file without them; in a WAL directory they are only index records
    /// (#401).
    fn has_search_indexes(self) -> bool {
        match self.layout {
            Layout::Closed => self.version != "0.5.43",
            Layout::Unflushed => true,
            Layout::Directory => false,
        }
    }
}

fn copy(from: &Path, to: &Path) {
    if from.is_dir() {
        std::fs::create_dir_all(to).unwrap();
        for entry in std::fs::read_dir(from).unwrap() {
            let entry = entry.unwrap();
            copy(&entry.path(), &to.join(entry.file_name()));
        }
    } else {
        std::fs::copy(from, to).unwrap();
    }
}

/// Every file and directory under `root`, with the bytes of each file.
fn files(root: &Path) -> BTreeMap<PathBuf, Option<Vec<u8>>> {
    let mut found = BTreeMap::new();
    let mut pending = vec![root.to_path_buf()];
    while let Some(dir) = pending.pop() {
        for entry in std::fs::read_dir(&dir).unwrap() {
            let path = entry.unwrap().path();
            let relative = path.strip_prefix(root).unwrap().to_path_buf();
            if path.is_dir() {
                found.insert(relative, None);
                pending.push(path);
            } else {
                found.insert(relative, Some(std::fs::read(&path).unwrap()));
            }
        }
    }
    found
}

/// Runs `test` on each fixture, then fails with the list of fixtures it failed on, so a
/// layout that fails does not hide how the others fare. The output above the failure
/// holds each fixture's panic message.
fn each(fixtures: impl IntoIterator<Item = Fixture>, test: impl Fn(Fixture)) {
    let failed: Vec<String> = fixtures
        .into_iter()
        .filter(|&fixture| std::panic::catch_unwind(AssertUnwindSafe(|| test(fixture))).is_err())
        .map(Fixture::name)
        .collect();
    assert!(failed.is_empty(), "failed on {failed:?}");
}

/// Runs `check` on a read-write open of every fixture.
fn read_write(check: fn(Fixture, &GrafeoDB)) {
    each(FIXTURES, |fixture| {
        let (_dir, db) = fixture.open();
        check(fixture, &db);
    });
}

/// Runs `check` on a read-only open of every fixture (0.5.44 refused read-only opens of
/// WAL directories; 0.6 reads them in place), and checks that the open wrote nothing:
/// afterwards every file of the copy holds the same bytes, and no file or directory
/// came or went (a read-only open spills to the system temp directory, never beside
/// the database).
fn read_only(check: fn(Fixture, &GrafeoDB)) {
    each(FIXTURES, |fixture| {
        let name = fixture.name();
        let dir = fixture.copy();
        let database_files = || files(dir.path());
        let before = database_files();
        let db = GrafeoDB::open_read_only(dir.path().join(fixture.file_name()))
            .unwrap_or_else(|error| panic!("{name}: read-only: {error}"));
        assert!(
            db.execute("INSERT (:Person {name: 'Jules'})").is_err(),
            "{name}: a read-only database refuses writes"
        );
        check(fixture, &db);
        db.close().unwrap();
        drop(db);

        let after = database_files();
        let changed: BTreeSet<&PathBuf> = before
            .keys()
            .chain(after.keys())
            .filter(|path| before.get(*path) != after.get(*path))
            .collect();
        assert!(
            changed.is_empty(),
            "{name}: the read-only open changed, added or removed {changed:?}"
        );
    });
}

/// `open_in_memory` of a 0.5.x database reads it as a read-only open does (the sidecar
/// WAL of a file replayed, the WAL of a directory replayed) and changes nothing on disk:
/// no migration, no kept copy, no lock file. Every check runs on its own in-memory copy,
/// which takes writes.
#[test]
fn open_in_memory_reads_a_0_5_database_without_changing_it() {
    let checks: [fn(Fixture, &GrafeoDB); 6] = [
        check_default_graph,
        check_named_graphs,
        check_rdf_triples,
        check_constraints,
        check_schema,
        check_indexes,
    ];
    each(FIXTURES, |fixture| {
        let name = fixture.name();
        let dir = fixture.copy();
        let path = dir.path().join(fixture.file_name());
        let spill = PathBuf::from(format!("{}.spill", fixture.file_name()));
        let database_files = || {
            let mut found = files(dir.path());
            found.retain(|file, _| !file.starts_with(&spill));
            found
        };
        let before = database_files();
        for check in checks {
            let db = GrafeoDB::open_in_memory(&path)
                .unwrap_or_else(|error| panic!("{name}: open_in_memory: {error}"));
            assert!(!db.is_read_only(), "{name}: the copy takes writes");
            check(fixture, &db);
        }
        let after = database_files();
        let changed: BTreeSet<&PathBuf> = before
            .keys()
            .chain(after.keys())
            .filter(|file| before.get(*file) != after.get(*file))
            .collect();
        assert!(
            changed.is_empty(),
            "{name}: open_in_memory changed, added or removed {changed:?}"
        );
    });
}

/// An in-memory copy of what a read-only `db` loaded, for the statements a read-only
/// database refuses: writes, and the `SHOW` commands, which run as schema commands.
/// `None` for a read-write `db`, which runs them itself.
fn unrestricted_copy(db: &GrafeoDB) -> Option<GrafeoDB> {
    db.is_read_only().then(|| db.to_memory().unwrap())
}

fn rows(db: &GrafeoDB, query: &str) -> Vec<Vec<Value>> {
    db.execute(query)
        .unwrap_or_else(|error| panic!("{query}: {error}"))
        .rows()
        .to_vec()
}

fn strings(values: &[&str]) -> Value {
    Value::List(values.iter().map(|v| Value::from(*v)).collect())
}

/// The first column of `query`, sorted, as strings.
fn names(db: &GrafeoDB, query: &str) -> Vec<String> {
    let mut names: Vec<String> = rows(db, query)
        .into_iter()
        .map(|row| match &row[0] {
            Value::String(name) => name.to_string(),
            other => panic!("{query}: expected a name, got {other:?}"),
        })
        .collect();
    names.sort();
    names
}

/// A read-only open of a 0.5.x database reads it into memory and holds no lock once it
/// is loaded: while the database is open, another handle locks the file exclusively, or
/// takes the `LOCK` of the directory as a 0.5.44 writer does.
#[test]
fn a_0_5_database_opened_read_only_is_not_locked_once_loaded() {
    each(FIXTURES, |fixture| {
        let name = fixture.name();
        let dir = fixture.copy();
        let path = dir.path().join(fixture.file_name());
        let db = GrafeoDB::open_read_only(&path)
            .unwrap_or_else(|error| panic!("{name}: read-only: {error}"));

        if fixture.is_directory() {
            let writer = DirectoryLock::acquire(&path);
            assert!(
                writer.is_ok(),
                "{name}: the directory is locked while the read-only database is open: {:?}",
                writer.err()
            );
        } else {
            let file = std::fs::File::open(&path).unwrap();
            assert!(
                file.try_lock().is_ok(),
                "{name}: the file is locked while the read-only database is open"
            );
            file.unlock().unwrap();
        }
        assert!(db.node_count() > 0, "{name}: the database holds the data");
        db.close().unwrap();
    });
}

/// `<path><suffix>`, next to the database file.
fn with_suffix(path: &Path, suffix: &str) -> PathBuf {
    let mut name = path.as_os_str().to_owned();
    name.push(suffix);
    PathBuf::from(name)
}

/// A read-write open of a 0.5.x database migrates it to a single file in the 0.6 format
/// at the same path, and keeps the old database next to it, byte for byte: a file as
/// `<name>.pre-0.6` and its sidecar WAL as `<name>.pre-0.6.wal/`, a WAL directory as
/// the directory `<name>.pre-0.6/` (its `wal/` files, and its `LOCK` where 0.5.44 left
/// one). Nothing of the migration is left behind, and a second read-write open finds a
/// 0.6 file and changes nothing but the database file.
#[test]
fn a_read_write_open_migrates_a_0_5_database_and_keeps_the_old_one() {
    each(FIXTURES, |fixture| {
        let name = fixture.name();
        let dir = fixture.copy();
        let path = dir.path().join(fixture.file_name());
        let original = fixture.source().join(fixture.file_name());
        let original_wal = with_suffix(&original, ".wal");

        let db = GrafeoDB::open(&path).unwrap_or_else(|error| panic!("{name}: {error}"));
        assert!(db.node_count() > 0, "{name}: the database holds the data");
        db.close().unwrap();
        drop(db);

        assert!(path.is_file(), "{name}: the database is a single file");
        assert_eq!(
            detect(&path).unwrap(),
            OnDisk::Current,
            "{name}: the database file is in the 0.6 format"
        );
        let kept = with_suffix(&path, ".pre-0.6");
        if fixture.is_directory() {
            assert!(
                kept.is_dir(),
                "{name}: the 0.5.x directory is kept as <name>.pre-0.6/"
            );
            assert!(
                files(&kept) == files(&original),
                "{name}: <name>.pre-0.6/ holds the files of the 0.5.x directory: kept {:?}, \
                 original {:?}",
                files(&kept).keys().collect::<Vec<_>>(),
                files(&original).keys().collect::<Vec<_>>()
            );
        } else {
            assert!(
                std::fs::read(&kept).unwrap() == std::fs::read(&original).unwrap(),
                "{name}: <name>.pre-0.6 holds the bytes of the 0.5.x file"
            );
        }
        let kept_wal = with_suffix(&path, ".pre-0.6.wal");
        if original_wal.exists() {
            assert!(
                files(&kept_wal) == files(&original_wal),
                "{name}: <name>.pre-0.6.wal holds the files of the 0.5.x sidecar WAL"
            );
        } else {
            assert!(
                !kept_wal.exists(),
                "{name}: there was no sidecar WAL to keep"
            );
        }
        for leftover in [".migrating", ".migrating.creating", ".migrate.lock"] {
            assert!(
                !with_suffix(&path, leftover).exists(),
                "{name}: the migration left <name>{leftover} behind"
            );
        }

        // A second open finds a 0.6 file: the kept copies stay as they are, and no
        // file comes or goes (the database file itself takes a checkpoint on close).
        let spill = PathBuf::from(format!("{}.spill", fixture.file_name()));
        let others = || {
            let mut found = files(dir.path());
            found.retain(|file, _| {
                !file.starts_with(&spill) && file.as_path() != Path::new(fixture.file_name())
            });
            found
        };
        let before = others();
        let db = GrafeoDB::open(&path).unwrap_or_else(|error| panic!("{name}: reopen: {error}"));
        db.close().unwrap();
        drop(db);
        let after = others();
        let changed: BTreeSet<&PathBuf> = before
            .keys()
            .chain(after.keys())
            .filter(|file| before.get(*file) != after.get(*file))
            .collect();
        assert!(
            changed.is_empty(),
            "{name}: the second open migrated again: it changed, added or removed {changed:?}"
        );
    });
}

#[test]
fn the_default_graph_holds_what_was_written() {
    read_write(check_default_graph);
}

#[test]
fn the_default_graph_holds_what_was_written_read_only() {
    read_only(check_default_graph);
}

fn check_default_graph(fixture: Fixture, db: &GrafeoDB) {
    let name = fixture.name();
    let complete = !fixture.replays_into_a_named_graph();

    let (labels, edges, people) = if complete {
        (
            vec![
                vec![strings(&["City"]), Value::Int64(2)],
                vec![strings(&["Document"]), Value::Int64(3)],
                vec![strings(&["Employee", "Manager", "Person"]), Value::Int64(1)],
                vec![strings(&["Museum"]), Value::Int64(1)],
                vec![strings(&["Person"]), Value::Int64(2)],
            ],
            vec![
                vec![Value::from("IN_CITY"), Value::Int64(1)],
                vec![Value::from("KNOWS"), Value::Int64(1)],
                vec![Value::from("LIVES_IN"), Value::Int64(1)],
            ],
            ["Alix", "Gus", "Mia"],
        )
    } else {
        (
            vec![
                vec![strings(&["City"]), Value::Int64(2)],
                vec![strings(&["Document"]), Value::Int64(2)],
                vec![strings(&["Employee", "Person"]), Value::Int64(1)],
                vec![strings(&["Person"]), Value::Int64(2)],
            ],
            vec![
                vec![Value::from("KNOWS"), Value::Int64(1)],
                vec![Value::from("LIVES_IN"), Value::Int64(1)],
            ],
            ["Alix", "Gus", "Vincent"],
        )
    };
    assert_eq!(
        rows(
            db,
            "MATCH (n) RETURN labels(n) AS l, count(*) AS c ORDER BY l"
        ),
        labels,
        "{name}: nodes"
    );
    assert_eq!(
        rows(
            db,
            "MATCH ()-[r]->() RETURN type(r) AS t, count(*) AS c ORDER BY t"
        ),
        edges,
        "{name}: edges"
    );
    assert_eq!(
        names(db, "MATCH (p:Person) RETURN p.name"),
        people,
        "{name}: people"
    );

    // Every value type, and the second session's update and removal.
    let (age, score) = if complete {
        (Value::Int64(31), Value::Null)
    } else {
        (Value::Int64(30), Value::Int64(-3))
    };
    let address = BTreeMap::from([
        (PropertyKey::new("city"), Value::from("Amsterdam")),
        (PropertyKey::new("number"), Value::Int64(19)),
    ]);
    assert_eq!(
        rows(
            db,
            "MATCH (p:Person {name: 'Alix'}) RETURN p.email, p.age, p.score, p.height, \
             p.active, p.tags, p.address, p.born, p.seen, p.stay, p.embedding"
        ),
        [vec![
            Value::from("alix@example.org"),
            age,
            score,
            Value::Float64(1.88),
            Value::Bool(true),
            strings(&["amsterdam", "jazz"]),
            Value::Map(address.into()),
            Value::Date(Date::parse("1994-03-19").unwrap()),
            Value::ZonedDatetime(ZonedDatetime::parse("2024-03-19T08:30:00+01:00").unwrap()),
            Value::Duration(Duration::parse("P3D").unwrap()),
            Value::Vector(vec![3.0, 19.0, 88.0].into()),
        ]],
        "{name}: Alix"
    );
}

#[test]
fn named_graphs_hold_what_was_written() {
    read_write(check_named_graphs);
}

#[test]
fn named_graphs_hold_what_was_written_read_only() {
    read_only(check_named_graphs);
}

fn check_named_graphs(fixture: Fixture, db: &GrafeoDB) {
    let name = fixture.name();
    let mut graphs = db.list_graphs();
    graphs.sort();
    assert_eq!(graphs, ["museums", "trips"], "{name}: graphs");

    let museums = db.graph("museums").unwrap();
    assert_eq!(
        museums
            .execute("MATCH (n) RETURN labels(n), n.name")
            .unwrap()
            .rows(),
        [vec![strings(&["Museum"]), Value::from("Louvre")]],
        "{name}: museums"
    );

    let trips = db.graph("trips").unwrap();
    let expected_trips = if fixture.replays_into_a_named_graph() {
        // Everything 0.5.43 wrote after its first switch to `trips` is replayed
        // into it. The second session's changes address nodes by id, so Gus's
        // `Manager` label lands on Prague, the node with his id in `trips`.
        vec![
            vec![strings(&["Person"]), Value::from("Mia"), Value::Null],
            vec![strings(&["City"]), Value::from("Paris"), Value::from("FR")],
            vec![
                strings(&["City", "Manager"]),
                Value::from("Prague"),
                Value::from("CZ"),
            ],
            vec![
                strings(&["Museum"]),
                Value::from("Rijksmuseum"),
                Value::Null,
            ],
            vec![strings(&["Document"]), Value::Null, Value::Null],
        ]
    } else {
        vec![
            vec![strings(&["City"]), Value::from("Paris"), Value::from("FR")],
            vec![strings(&["City"]), Value::from("Prague"), Value::from("CZ")],
        ]
    };
    assert_eq!(
        trips
            .execute("MATCH (c) RETURN labels(c), c.name, c.country ORDER BY c.name")
            .unwrap()
            .rows(),
        expected_trips,
        "{name}: trips nodes"
    );
    assert_eq!(
        trips
            .execute("MATCH ()-[r]->() RETURN type(r), r.km")
            .unwrap()
            .rows(),
        [vec![Value::from("ROUTE"), Value::Int64(1030)]],
        "{name}: trips edges"
    );
}

#[test]
fn rdf_triples_hold_what_was_written() {
    read_write(check_rdf_triples);
}

#[test]
fn rdf_triples_hold_what_was_written_read_only() {
    read_only(check_rdf_triples);
}

fn check_rdf_triples(fixture: Fixture, db: &GrafeoDB) {
    let iri = |name: &str| Value::from(format!("http://example.org/{name}"));
    assert_eq!(
        db.execute_sparql("SELECT ?s ?p ?o WHERE { ?s ?p ?o } ORDER BY ?s ?p ?o")
            .unwrap()
            .rows(),
        [
            vec![iri("alix"), iri("knows"), iri("gus")],
            vec![iri("gus"), iri("knows"), iri("mia")],
        ],
        "{}: triples",
        fixture.name()
    );
}

#[test]
fn constraints_hold() {
    read_write(check_constraints);
}

#[test]
fn constraints_hold_read_only() {
    read_only(check_constraints);
}

fn check_constraints(fixture: Fixture, db: &GrafeoDB) {
    let name = fixture.name();
    let copy = unrestricted_copy(db);
    let db = copy.as_ref().unwrap_or(db);
    let error = |query: &str| match db.execute(query) {
        Ok(result) => panic!("{name}: {query} succeeded: {:?}", result.rows()),
        Err(error) => error.to_string(),
    };
    let duplicate = error("INSERT (:Person {name: 'Jules', email: 'alix@example.org'})");
    assert!(duplicate.contains("UNIQUE"), "{name}: {duplicate}");
    let unnamed = error("INSERT (:Person {email: 'jules@example.org'})");
    assert!(unnamed.contains("NOT NULL"), "{name}: {unnamed}");
    if !fixture.replays_into_a_named_graph() {
        let museum = error("INSERT (:Museum {name: 'Rijksmuseum'})");
        assert!(museum.contains("UNIQUE"), "{name}: {museum}");
    }
    assert_eq!(
        names(db, "SHOW CONSTRAINTS"),
        fixture.constraint_names(),
        "{name}: constraint names"
    );
}

#[test]
fn the_schema_holds_what_was_written() {
    read_write(check_schema);
}

#[test]
fn the_schema_holds_what_was_written_read_only() {
    read_only(check_schema);
}

fn check_schema(fixture: Fixture, db: &GrafeoDB) {
    let name = fixture.name();
    let copy = unrestricted_copy(db);
    let db = copy.as_ref().unwrap_or(db);
    assert_eq!(
        names(db, "SHOW NODE TYPES"),
        ["Capital", "City", "Museum", "Person"],
        "{name}: node types"
    );
    assert_eq!(
        names(db, "SHOW GRAPH TYPES"),
        ["culture", "travel"],
        "{name}: graph types"
    );

    let first = !fixture.first_session_schema_in_wal();
    let second = !fixture.second_session_schema_in_wal();
    let column = |query: &str, type_name: &str, columns: std::ops::Range<usize>| {
        rows(db, query)
            .into_iter()
            .find(|row| row[0] == Value::from(type_name))
            .map_or_else(
                || panic!("{name}: no {type_name} in {query}"),
                |row| row[columns].to_vec(),
            )
    };
    let kept = |kept: bool, values: &[&str]| -> Vec<Value> {
        values
            .iter()
            .map(|value| Value::from(if kept { *value } else { "" }))
            .collect()
    };
    assert_eq!(
        column("SHOW NODE TYPES", "Capital", 3..4),
        kept(first, &["City"]),
        "{name}: parent of Capital"
    );
    assert_eq!(
        column("SHOW EDGE TYPES", "ROUTE", 2..4),
        kept(first, &["City", "City"]),
        "{name}: endpoints of ROUTE"
    );
    assert_eq!(
        column("SHOW EDGE TYPES", "IN_CITY", 2..4),
        kept(second, &["Museum", "City"]),
        "{name}: endpoints of IN_CITY"
    );

    // Default values apply to new nodes.
    db.execute("INSERT (:City {name: 'Barcelona'})").unwrap();
    assert_eq!(
        rows(db, "MATCH (c:City {name: 'Barcelona'}) RETURN c.country"),
        [vec![if first {
            Value::from("NL")
        } else {
            Value::Null
        }]],
        "{name}: default of City.country"
    );
    db.execute("INSERT (:Museum {name: 'Stedelijk'})").unwrap();
    assert_eq!(
        rows(db, "MATCH (m:Museum {name: 'Stedelijk'}) RETURN m.open"),
        [vec![if second {
            Value::Bool(true)
        } else {
            Value::Null
        }]],
        "{name}: default of Museum.open"
    );
}

#[test]
fn indexes_hold_what_was_written() {
    read_write(check_indexes);
}

#[test]
fn indexes_hold_what_was_written_read_only() {
    read_only(check_indexes);
}

fn check_indexes(fixture: Fixture, db: &GrafeoDB) {
    let as_value = |node: NodeId| Value::Int64(i64::try_from(node.as_u64()).unwrap());
    let name = fixture.name();
    let copy = unrestricted_copy(db);
    assert_eq!(
        names(copy.as_ref().unwrap_or(db), "SHOW INDEXES"),
        fixture.index_names(),
        "{name}: indexes"
    );
    let vector = db.vector_search("Document", "embedding", &[1.0, 0.0, 0.0], 1, None, None);
    let text = db.text_search("Document", "content", "canals", 3);
    if !fixture.has_search_indexes() {
        assert!(vector.is_err(), "{name}: a vector index");
        assert!(text.is_err(), "{name}: a text index");
        return;
    }
    let id = |title: &str| {
        let found = rows(
            db,
            &format!("MATCH (d:Document {{title: '{title}'}}) RETURN id(d)"),
        );
        assert_eq!(found.len(), 1, "{name}: documents titled {title}");
        found[0][0].clone()
    };
    let nearest: Vec<Value> = vector
        .unwrap_or_else(|error| panic!("{name}: vector search: {error}"))
        .into_iter()
        .map(|(node, _)| as_value(node))
        .collect();
    assert_eq!(nearest, [id("Canals")], "{name}: vector search");
    let mut matches: Vec<Value> = text
        .unwrap_or_else(|error| panic!("{name}: text search: {error}"))
        .into_iter()
        .map(|(node, _)| as_value(node))
        .collect();
    let mut expected = vec![id("Canals"), id("Bridges")];
    matches.sort_by_key(Value::as_int64);
    expected.sort_by_key(Value::as_int64);
    assert_eq!(matches, expected, "{name}: text search");
}
