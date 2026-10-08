//! Migration of 0.5.x database files in a build without the `wal` feature,
//! such as the default `embedded` profile of the Node.js, C, Go, C# and Dart
//! bindings.
//!
//! A file 0.5.x closed cleanly has no sidecar WAL: a read-write open migrates
//! it as in any build (when it holds nothing else this build refuses, see
//! [`closed_0_5_file`]). A file whose sidecar WAL holds changes is refused, by a
//! read-write and a read-only open alike: this build cannot replay the WAL, and
//! the file alone would lack its changes. For the same reason a read-write and
//! a read-only open refuse a 0.6 file whose sidecar WAL holds commits. These
//! tests run in CI with:
//!
//! ```bash
//! cargo test -p grafeo-engine --no-default-features --features lpg,gql,grafeo-file \
//!     --test migration_without_wal
//! ```

#![cfg(all(
    feature = "lpg",
    feature = "gql",
    feature = "grafeo-file",
    not(feature = "wal")
))]

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

#[path = "common/legacy_file.rs"]
mod legacy_file;

use grafeo_common::storage::section::SectionType;
use grafeo_common::types::Value;
use grafeo_engine::{Config, GrafeoDB};
use grafeo_storage::file::detect::{OnDisk, detect};

/// The released versions with fixtures (#427).
const VERSIONS: [&str; 2] = ["0.5.43", "0.5.44"];

/// The fixture `name` written by `version`.
fn fixture(version: &str, name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/released")
        .join(version)
        .join(name)
}

/// `<path><suffix>`, next to the database file.
fn with_suffix(path: &Path, suffix: &str) -> PathBuf {
    let mut name = path.as_os_str().to_owned();
    name.push(suffix);
    PathBuf::from(name)
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

/// Every file under `root` with its bytes (`None` for a directory), leaving
/// out the spill directories an open creates next to a database (scratch
/// space, not part of it).
fn files(root: &Path) -> BTreeMap<PathBuf, Option<Vec<u8>>> {
    let mut found = BTreeMap::new();
    let mut pending = vec![root.to_path_buf()];
    while let Some(dir) = pending.pop() {
        for entry in std::fs::read_dir(&dir).unwrap() {
            let path = entry.unwrap().path();
            let relative = path.strip_prefix(root).unwrap().to_path_buf();
            if relative.to_string_lossy().ends_with(".spill") {
                continue;
            }
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

/// A read-write open of `path` (`GrafeoDB::open` needs the `wal` feature).
fn open(path: &Path) -> grafeo_common::utils::error::Result<GrafeoDB> {
    GrafeoDB::with_config(Config::persistent(path))
}

fn rows(db: &GrafeoDB, query: &str) -> Vec<Vec<Value>> {
    db.execute(query)
        .unwrap_or_else(|error| panic!("{query}: {error}"))
        .rows()
        .to_vec()
}

/// What a database holds, as queries see it.
#[derive(Debug, PartialEq)]
struct Contents {
    /// The label sets, with their node counts.
    nodes: Vec<Vec<Value>>,
    /// The edge types, with their counts.
    edges: Vec<Vec<Value>>,
    /// Every person with some properties, by name.
    people: Vec<Vec<Value>>,
    /// The named graphs, sorted.
    graphs: Vec<String>,
}

fn contents(db: &GrafeoDB) -> Contents {
    let mut graphs = db.list_graphs();
    graphs.sort();
    Contents {
        nodes: rows(
            db,
            "MATCH (n) RETURN labels(n) AS l, count(*) AS c ORDER BY l",
        ),
        edges: rows(
            db,
            "MATCH ()-[r]->() RETURN type(r) AS t, count(*) AS c ORDER BY t",
        ),
        people: rows(
            db,
            "MATCH (p:Person) RETURN p.name AS name, p.email, p.age ORDER BY name",
        ),
        graphs,
    }
}

/// Writes at `path` the released file `version/closed.grafeo` (closed
/// cleanly) without what a build of these tests may refuse (see
/// `rdf_file_without_triple_store` and `search_indexes_without_their_features`):
/// its triples (the `RdfStore` section), the index sections, and the catalog
/// of 0.5.44, which defines a vector and a text index (0.5.43 kept no index
/// definitions). What is left is the LPG data, and the schema of 0.5.43.
fn closed_0_5_file(version: &str, path: &Path) {
    legacy_file::write_0_5_file(
        path,
        version,
        |section_type| match section_type {
            SectionType::RdfStore | SectionType::VectorStore | SectionType::TextIndex => false,
            SectionType::Catalog => version == "0.5.43",
            _ => true,
        },
        &[],
    );
}

/// A read-write open migrates a 0.5.x file 0.5.x closed cleanly: the data is
/// there, the old file is kept as `<path>.pre-0.6` byte for byte, the new file
/// is in the 0.6 format, nothing of the migration is left, and a reopen finds
/// the same data.
#[test]
fn a_closed_0_5_file_migrates_without_the_wal_feature() {
    for version in VERSIONS {
        let dir = tempfile::tempdir().unwrap();
        let original = dir.path().join("original.grafeo");
        closed_0_5_file(version, &original);
        let expected = {
            let reference = dir.path().join("reference.grafeo");
            copy(&original, &reference);
            let db = GrafeoDB::open_read_only(&reference).unwrap();
            let found = contents(&db);
            db.close().unwrap();
            found
        };
        let people: Vec<&Value> = expected.people.iter().map(|person| &person[0]).collect();
        assert_eq!(
            people,
            [
                &Value::from("Alix"),
                &Value::from("Gus"),
                &Value::from("Mia")
            ],
            "{version}: the fixture holds its people"
        );

        let path = dir.path().join("db.grafeo");
        copy(&original, &path);
        let db = open(&path).unwrap_or_else(|error| panic!("{version}: {error}"));
        assert_eq!(contents(&db), expected, "{version}: the migrated data");
        db.close().unwrap();
        drop(db);

        assert!(
            std::fs::read(with_suffix(&path, ".pre-0.6")).unwrap()
                == std::fs::read(&original).unwrap(),
            "{version}: <path>.pre-0.6 holds the bytes of the 0.5.x file"
        );
        assert!(
            !with_suffix(&path, ".pre-0.6.wal").exists(),
            "{version}: there was no sidecar WAL to keep"
        );
        assert_eq!(
            detect(&path).unwrap(),
            OnDisk::Current,
            "{version}: the database file is in the 0.6 format"
        );
        for leftover in [".migrating", ".migrating.creating", ".migrate.lock"] {
            assert!(
                !with_suffix(&path, leftover).exists(),
                "{version}: the migration left <path>{leftover} behind"
            );
        }

        let db = open(&path).unwrap_or_else(|error| panic!("{version}: {error}"));
        assert_eq!(contents(&db), expected, "{version}: the reopened data");
        db.close().unwrap();
    }
}

/// A 0.5.x file whose sidecar WAL holds changes is refused by a read-write and
/// a read-only open, with an error that names the WAL and the `wal` feature,
/// and every file stays as it was.
#[test]
fn a_0_5_file_with_a_sidecar_wal_is_refused_without_the_wal_feature() {
    for version in VERSIONS {
        let original = fixture(version, "unflushed.grafeo");
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("db.grafeo");
        copy(&original, &path);
        copy(&with_suffix(&original, ".wal"), &with_suffix(&path, ".wal"));
        let before = files(dir.path());
        let wal = with_suffix(&path, ".wal").display().to_string();

        for (open, result) in [
            ("read-write", open(&path)),
            ("read-only", GrafeoDB::open_read_only(&path)),
        ] {
            let error = match result {
                Ok(_) => panic!("{version}: the {open} open succeeded without replaying the WAL"),
                Err(error) => error.to_string(),
            };
            assert!(
                error.contains(&wal)
                    && error.contains("only a build with the `wal` feature can replay"),
                "{version}: the {open} error names the WAL and the feature: {error}"
            );
            assert!(
                files(dir.path()) == before,
                "{version}: the refused {open} open changes nothing"
            );
        }
    }
}

/// A 0.5.x WAL directory holds its data only in its WAL, which this build
/// cannot replay: a read-write and a read-only open refuse it with an error
/// that names the directory and the `wal` feature, and every file stays as it
/// was.
#[test]
fn a_0_5_wal_directory_is_refused_without_the_wal_feature() {
    for version in VERSIONS {
        let original = fixture(version, "directory");
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("db");
        copy(&original, &path);
        let before = files(dir.path());

        for (open, result) in [
            ("read-write", open(&path)),
            ("read-only", GrafeoDB::open_read_only(&path)),
        ] {
            let error = match result {
                Ok(_) => panic!("{version}: the {open} open succeeded without replaying the WAL"),
                Err(error) => error.to_string(),
            };
            assert!(
                error.contains(&path.display().to_string()) && error.contains("`wal` feature"),
                "{version}: the {open} error names the directory and the feature: {error}"
            );
            assert!(
                files(dir.path()) == before,
                "{version}: the refused {open} open changes nothing"
            );
        }
    }
}

/// A 0.6 file whose sidecar WAL holds files has commits only the WAL holds (a
/// writer with the `wal` feature exited without `close()`). This build cannot
/// replay them: a read-write open (whose `close()` would remove the WAL with
/// them) and a read-only open refuse the file with an error that names the
/// database, its WAL and the `wal` feature, and nothing on disk changes, the
/// WAL byte for byte, so a build with `wal` still finds its commits. A missing
/// or empty sidecar WAL is no obstacle.
#[test]
fn a_0_6_file_with_a_sidecar_wal_is_refused_without_the_wal_feature() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("amsterdam.grafeo");
    {
        let db = open(&path).unwrap();
        db.execute("INSERT (:Person {name: 'Alix'})").unwrap();
        db.close().unwrap();
    }
    let people = |db: &GrafeoDB| rows(db, "MATCH (p:Person) RETURN p.name AS name");
    let wal = with_suffix(&path, ".wal");

    // No sidecar WAL, an empty one, and one with nothing to replay (the
    // checkpoint metadata and an empty log, as a writer leaves that
    // checkpointed and crashed before its next commit): both opens work. A
    // read-write `close()` removes the sidecar, so it is made again before
    // the read-only open.
    let arrange = |state: &str| match state {
        "missing" => {}
        "empty" => std::fs::create_dir_all(&wal).unwrap(),
        "nothing to replay" => {
            std::fs::create_dir_all(&wal).unwrap();
            std::fs::write(wal.join("checkpoint.meta"), [3_u8; 12]).unwrap();
            std::fs::write(wal.join("wal_00000019.log"), []).unwrap();
        }
        other => panic!("unknown sidecar state {other}"),
    };
    for state in ["missing", "empty", "nothing to replay"] {
        arrange(state);
        let db = open(&path)
            .unwrap_or_else(|error| panic!("sidecar {state}: a read-write open works: {error}"));
        assert_eq!(people(&db), vec![vec![Value::from("Alix")]]);
        db.close().unwrap();
        drop(db);
        arrange(state);
        let db = GrafeoDB::open_read_only(&path)
            .unwrap_or_else(|error| panic!("sidecar {state}: a read-only open works: {error}"));
        assert_eq!(people(&db), vec![vec![Value::from("Alix")]]);
        db.close().unwrap();
        drop(db);
        if wal.exists() {
            std::fs::remove_dir_all(&wal).unwrap();
        }
    }

    // A log file stands in for the records of a writer with the `wal`
    // feature (this build cannot write a WAL).
    let log = fixture("0.5.44", "unflushed.grafeo.wal").join("wal_00000000.log");
    std::fs::create_dir_all(&wal).unwrap();
    std::fs::copy(&log, wal.join("wal_00000000.log")).unwrap();
    let before = files(dir.path());

    for read_write in [true, false] {
        let kind = if read_write {
            "read-write"
        } else {
            "read-only"
        };
        let opened = if read_write {
            open(&path)
        } else {
            GrafeoDB::open_read_only(&path)
        };
        let error = match opened {
            Ok(db) => {
                db.close().unwrap();
                drop(db);
                let left = wal.join("wal_00000000.log").exists();
                panic!(
                    "the {kind} open succeeded without replaying the sidecar WAL; after close() \
                     the WAL's log file {}",
                    if left {
                        "is still there"
                    } else {
                        "is gone, with its commits"
                    }
                );
            }
            Err(error) => error.to_string(),
        };
        assert!(
            error.contains(&path.display().to_string())
                && error.contains(&wal.display().to_string())
                && error.contains("wal_00000000.log")
                && error.contains("`wal` feature"),
            "the {kind} error names the database, its WAL, the log file it found and the feature: \
             {error}"
        );
        assert!(
            files(dir.path()) == before,
            "the refused {kind} open changes nothing"
        );
    }
    assert!(
        std::fs::read(wal.join("wal_00000000.log")).unwrap() == std::fs::read(&log).unwrap(),
        "the WAL keeps its commits, byte for byte, for a build with the `wal` feature"
    );
}
