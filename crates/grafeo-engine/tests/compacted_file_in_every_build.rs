//! A database file compacted by 0.5.x opens in every build that opens files.
//!
//! Up to 0.5.44, `compact()` wrote the default graph's data into a compacted
//! base (the `CompactStore` section), the deletes of base nodes and edges
//! since into the `OverlayDeletions` section (from 0.5.42 on), and the writes
//! since into the LPG section, the overlay. An open folds the base into the
//! one store in every build with the `grafeo-file` feature: the
//! `compact-store` feature, which 0.5.x needed to read the base, is
//! deprecated and enables nothing. So a build without it, such as the
//! `grafeo` crate's default or the CLI, reads and migrates such a file as it
//! does any 0.5.x file. CI runs these tests in a build with only `lpg`, `gql`
//! and `grafeo-file`, and with all features.
//!
//! Two fixtures, each written by the released 0.5.44 wheel with the
//! `write_fixture.py` next to it:
//!
//! - `fixtures/compacted-labels/0.5.44/labels.grafeo`: a base and an overlay,
//!   closed cleanly (`compacted_file_labels.rs` checks its labels).
//! - `fixtures/compacted-crashed/0.5.44/crashed.grafeo`: a base checkpointed
//!   into the file, then direct calls that changed nodes and edges of the
//!   base, in its sidecar WAL, because the process exited without `close()`.
//!   0.5.44 replayed that WAL before it wired the base, so its own reopen
//!   lost every change to a base node or edge.
//!
//! ```bash
//! cargo test -p grafeo-engine --no-default-features --features lpg,gql,grafeo-file \
//!     --test compacted_file_in_every_build
//! cargo test -p grafeo-engine --all-features --test compacted_file_in_every_build
//! ```

#![cfg(all(feature = "lpg", feature = "gql", feature = "grafeo-file", not(miri)))]

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use grafeo_common::types::Value;
use grafeo_engine::{Config, GrafeoDB};
use grafeo_storage::file::detect::{OnDisk, detect};

/// An open of an existing database file.
type Open = fn(&Path) -> grafeo_common::utils::error::Result<GrafeoDB>;

/// Every open of an existing database file: read-write (which migrates a
/// 0.5.x file), read-only and, with the `wal` feature, `open_in_memory`
/// (which reads the file as a read-only open does).
fn opens() -> Vec<(&'static str, Open)> {
    #[cfg_attr(
        not(feature = "wal"),
        expect(unused_mut, reason = "only the `wal` feature adds `open_in_memory`")
    )]
    let mut opens: Vec<(&'static str, Open)> = vec![
        ("read-write", |path| {
            GrafeoDB::with_config(Config::persistent(path))
        }),
        ("read-only", |path| GrafeoDB::open_read_only(path)),
    ];
    #[cfg(feature = "wal")]
    opens.push(("in-memory", |path| GrafeoDB::open_in_memory(path)));
    opens
}

/// The directory of a fixture.
fn fixture_dir(fixture: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(fixture)
        .join("0.5.44")
}

/// Copies every file and directory under `from` to `to`.
fn copy_dir(from: &Path, to: &Path) {
    std::fs::create_dir_all(to).unwrap();
    for entry in std::fs::read_dir(from).unwrap() {
        let path = entry.unwrap().path();
        let target = to.join(path.file_name().unwrap());
        if path.is_dir() {
            copy_dir(&path, &target);
        } else {
            std::fs::copy(&path, &target).unwrap();
        }
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

/// `<path><suffix>`, next to the database file.
fn with_suffix(path: &Path, suffix: &str) -> PathBuf {
    let mut name = path.as_os_str().to_owned();
    name.push(suffix);
    PathBuf::from(name)
}

fn text(value: &Value) -> Option<String> {
    match value {
        Value::String(text) => Some(text.to_string()),
        Value::Null => None,
        other => panic!("expected a string or null, got {other:?}"),
    }
}

/// Every node: its name, its labels (sorted) and its city.
fn nodes(db: &GrafeoDB) -> Vec<(String, Vec<String>, Option<String>)> {
    db.execute("MATCH (n) RETURN n.name, labels(n), n.city ORDER BY n.name")
        .unwrap()
        .rows()
        .iter()
        .map(|row| {
            let Value::List(labels) = &row[1] else {
                panic!("labels(n) is a list, got {:?}", row[1]);
            };
            let mut labels: Vec<String> = labels
                .iter()
                .map(|label| text(label).expect("a label"))
                .collect();
            labels.sort();
            (text(&row[0]).expect("a name"), labels, text(&row[2]))
        })
        .collect()
}

/// Every edge of type `edge_type`: its source's and target's names and its
/// `since`.
fn edges(db: &GrafeoDB, edge_type: &str) -> Vec<(String, String, Value)> {
    db.execute(&format!(
        "MATCH (a)-[e:{edge_type}]->(b) RETURN a.name, b.name, e.since ORDER BY a.name, b.name"
    ))
    .unwrap()
    .rows()
    .iter()
    .map(|row| {
        (
            text(&row[0]).expect("a source"),
            text(&row[1]).expect("a target"),
            row[2].clone(),
        )
    })
    .collect()
}

/// What a fixture holds once open.
struct Expected {
    /// Every node: its name, its labels and its city.
    nodes: Vec<(&'static str, Vec<&'static str>, Option<&'static str>)>,
    /// The type of its edges.
    edge_type: &'static str,
    /// Every edge: its source's and target's names and its `since`.
    edges: Vec<(&'static str, &'static str, i64)>,
}

impl Expected {
    fn check(&self, db: &GrafeoDB, stage: &str) {
        let want_nodes: Vec<(String, Vec<String>, Option<String>)> = self
            .nodes
            .iter()
            .map(|(name, labels, city)| {
                (
                    (*name).to_string(),
                    labels.iter().map(|label| (*label).to_string()).collect(),
                    city.map(str::to_string),
                )
            })
            .collect();
        assert_eq!(nodes(db), want_nodes, "{stage}: the nodes");
        let want_edges: Vec<(String, String, Value)> = self
            .edges
            .iter()
            .map(|(source, target, since)| {
                (
                    (*source).to_string(),
                    (*target).to_string(),
                    Value::Int64(*since),
                )
            })
            .collect();
        assert_eq!(edges(db, self.edge_type), want_edges, "{stage}: the edges");
    }
}

/// Opens a copy of `fixture`'s `file` in every way, and checks that each
/// open serves `expected`: a read-write open migrates the file to the 0.6
/// format, keeps the 0.5.x file and its sidecar WAL as `<name>.pre-0.6` and
/// `<name>.pre-0.6.wal/`, and serves the same after a reopen; a read-only
/// open and `open_in_memory` change nothing on disk.
fn assert_every_open_serves(fixture: &str, file: &str, expected: &Expected) {
    let original = fixture_dir(fixture);
    for (kind, open) in opens() {
        let dir = tempfile::tempdir().unwrap();
        copy_dir(&original, dir.path());
        let path = dir.path().join(file);
        assert_eq!(
            detect(&path).unwrap(),
            OnDisk::LegacyFile,
            "{fixture}: the file is a 0.5.x file"
        );
        let before = files(dir.path());

        let db = open(&path).unwrap_or_else(|error| panic!("{kind} open of {fixture}: {error}"));
        expected.check(&db, &format!("{kind} open of {fixture}"));
        db.close().unwrap();
        drop(db);

        if kind != "read-write" {
            assert!(
                files(dir.path()) == before,
                "{kind} open of {fixture}: every file stays as it was"
            );
            continue;
        }
        assert_eq!(
            detect(&path).unwrap(),
            OnDisk::Current,
            "{fixture}: the read-write open migrated the file to the 0.6 format"
        );
        assert!(
            std::fs::read(with_suffix(&path, ".pre-0.6")).unwrap()
                == std::fs::read(original.join(file)).unwrap(),
            "{fixture}: <name>.pre-0.6 holds the bytes of the 0.5.x file"
        );
        let original_wal = with_suffix(&original.join(file), ".wal");
        if original_wal.exists() {
            assert!(
                files(&with_suffix(&path, ".pre-0.6.wal")) == files(&original_wal),
                "{fixture}: <name>.pre-0.6.wal holds the files of the 0.5.x sidecar WAL"
            );
        }
        let db = GrafeoDB::with_config(Config::persistent(&path)).unwrap();
        expected.check(&db, &format!("reopen of the migrated {fixture}"));
        db.close().unwrap();
    }
}

/// A compacted 0.5.x file opens with its base folded in, in a build without
/// the old `compact-store` feature too: every node and edge of the base comes
/// back, a change made after `compact()` wins over the base's version
/// (Alix's city and `Starred`), and a node deleted after it (Butch) stays
/// deleted.
#[test]
fn every_open_of_a_compacted_0_5_file_folds_its_base() {
    let expected = Expected {
        nodes: vec![
            (
                "Alix",
                vec!["Graph", "Repository", "Starred"],
                Some("Amsterdam"),
            ),
            ("Gus", vec!["Graph"], None),
            ("Jules", vec!["Graph", "Repository"], None),
            ("Mia", vec!["Archive", "Graph", "Repository"], None),
            ("Vincent", vec!["Graph", "Repository"], None),
        ],
        edge_type: "FORKED_FROM",
        edges: vec![("Alix", "Vincent", 2019), ("Jules", "Mia", 2088)],
    };
    assert_every_open_serves("compacted-labels", "labels.grafeo", &expected);
}

/// A compacted 0.5.44 file whose process exited without `close()` replays
/// its WAL onto the folded base: the direct calls after `compact()` that
/// changed base nodes and edges (Alix's city, Gus's `Employee` label,
/// Vincent's removed city, the deleted edge from Gus to Vincent and the
/// deleted Butch) all hold, as do the creates (Jules and his edge to Mia).
/// The query after `compact()` (Django) is not there: 0.5.44 never logged it
/// (#558), and no reader can bring it back.
#[cfg(feature = "wal")]
#[test]
fn a_crashed_compacted_0_5_file_replays_its_wal_onto_the_base() {
    let expected = Expected {
        nodes: vec![
            ("Alix", vec!["Person"], Some("Berlin")),
            ("Gus", vec!["Employee", "Person"], Some("Berlin")),
            ("Jules", vec!["Person"], Some("Amsterdam")),
            ("Mia", vec!["Person"], Some("Prague")),
            ("Vincent", vec!["Person"], None),
        ],
        edge_type: "KNOWS",
        edges: vec![
            ("Alix", "Gus", 2019),
            ("Jules", "Mia", 2088),
            ("Vincent", "Mia", 2019),
        ],
    };
    assert_every_open_serves("compacted-crashed", "crashed.grafeo", &expected);
}
