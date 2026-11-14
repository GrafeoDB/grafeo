//! The fixtures `scripts/downgrade_check.py` opens with the 0.6.2 wheel (#764): files a 0.6.3 build
//! wrote that hold a standalone node type and a standalone edge type a graph type lists, which the
//! catalog writes as record kinds 12 and 13. 0.6.2 must refuse them cleanly, leaving every byte.
//!
//! - `standalone-section.grafeo`: closed, so the two records are in the checkpointed catalog
//!   section.
//! - `standalone-wal.grafeo` with `standalone-wal.grafeo.wal/`: the process exited without
//!   `close()`, so the two records are only in the WAL.
//!
//! The test here opens copies of both with this build and checks that they hold what the downgrade
//! check needs: each type is marked (its drop is refused, naming the graph type that lists it). The
//! writer, run by hand to make the fixtures again (README beside them):
//!
//! ```bash
//! GRAFEO_WRITE_DOWNGRADE_FIXTURES=1 cargo test -p grafeo-engine --all-features --test main \
//!     downgrade_fixtures::write_the_fixtures -- --ignored --exact
//! ```
//!
//! It writes the section fixture, then the WAL fixture, and ends the process without closing the
//! second database, as a crash would.

#![cfg(all(feature = "gql", feature = "grafeo-file", feature = "wal"))]

use std::path::{Path, PathBuf};

use grafeo_engine::GrafeoDB;

/// The statements both fixtures run: a standalone node type and edge type, each listed by a graph
/// type (so each is marked: kinds 12 and 13), and a node of the type.
const STATEMENTS: [&str; 5] = [
    "CREATE NODE TYPE A (h FLOAT64)",
    "CREATE EDGE TYPE ROAD (km INT64)",
    "CREATE GRAPH TYPE t (NODE TYPE A)",
    "CREATE GRAPH TYPE u (EDGE TYPE ROAD)",
    "INSERT (:A {h: 1.5e0})",
];

fn fixtures() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/downgrade/0.6.3")
}

fn run_all(db: &GrafeoDB) {
    for statement in STATEMENTS {
        db.execute(statement)
            .unwrap_or_else(|error| panic!("{statement}: {error}"));
    }
}

/// `from` and everything beside it of the same name (its WAL directory), copied into `to`.
fn copy_database(from: &Path, to: &Path) -> PathBuf {
    let name = from.file_name().unwrap();
    let copy = to.join(name);
    std::fs::copy(from, &copy).unwrap_or_else(|error| panic!("{}: {error}", from.display()));
    let wal = from.with_file_name(format!("{}.wal", name.to_string_lossy()));
    if wal.is_dir() {
        let target = to.join(wal.file_name().unwrap());
        std::fs::create_dir_all(&target).unwrap();
        for entry in std::fs::read_dir(&wal).unwrap() {
            let entry = entry.unwrap();
            std::fs::copy(entry.path(), target.join(entry.file_name())).unwrap();
        }
    }
    copy
}

/// Both fixtures, opened by this build, hold the two marked types: the drop of each is refused,
/// naming the graph type that lists it. The WAL fixture holds them in its WAL.
#[test]
fn the_fixtures_hold_the_marked_standalone_types() {
    for (name, in_wal) in [
        ("standalone-section.grafeo", false),
        ("standalone-wal.grafeo", true),
    ] {
        let source = fixtures().join(name);
        assert_eq!(
            source.with_file_name(format!("{name}.wal")).is_dir(),
            in_wal,
            "{name}: a WAL beside it only for the WAL fixture"
        );
        let dir = tempfile::tempdir().unwrap();
        let copy = copy_database(&source, dir.path());
        let db = GrafeoDB::open(&copy).unwrap_or_else(|error| panic!("{name}: {error}"));
        for (statement, graph_type) in [
            ("DROP NODE TYPE A", "graph type 't'"),
            ("DROP EDGE TYPE ROAD", "graph type 'u'"),
        ] {
            let error = db
                .execute(statement)
                .expect_err(&format!("{name}: {statement} is refused"));
            assert!(
                error.to_string().contains(graph_type),
                "{name}: {statement}: {error}"
            );
        }
        let rows = db.execute("MATCH (n:A) RETURN n.h").unwrap();
        assert_eq!(rows.rows().len(), 1, "{name}: the node of type A");
        db.close().unwrap();
    }
}

/// Writes the two fixtures (see the module docs). Ends the process.
#[test]
#[ignore = "writes the downgrade fixtures into the source tree and ends the process; run by hand"]
fn write_the_fixtures() {
    if std::env::var_os("GRAFEO_WRITE_DOWNGRADE_FIXTURES").is_none() {
        return;
    }
    let dir = fixtures();
    std::fs::create_dir_all(&dir).unwrap();
    for name in ["standalone-section.grafeo", "standalone-wal.grafeo"] {
        let _ = std::fs::remove_file(dir.join(name));
        let _ = std::fs::remove_dir_all(dir.join(format!("{name}.wal")));
    }

    let db = GrafeoDB::open(dir.join("standalone-section.grafeo")).unwrap();
    run_all(&db);
    db.close().unwrap();
    drop(db);

    let db = GrafeoDB::open(dir.join("standalone-wal.grafeo")).unwrap();
    run_all(&db);
    // A crash: the process ends with the database open, so the records stay in the WAL only.
    std::process::exit(0);
}
