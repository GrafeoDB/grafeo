//! The `rdf` profile on its own keeps its data across a reopen, after a
//! `close()` and after a crash. Without the LPG store it kept nothing (#544):
//! loading the file, replaying the WAL and checkpoints need it, so the profile
//! includes it.
//!
//! ```bash
//! cargo test -p grafeo --no-default-features --features rdf --test rdf_profile
//! ```

#![cfg(feature = "rdf")]

use std::path::Path;

use grafeo::{Config, GrafeoDB};
use grafeo_common::testing::child_process;

/// Tells [`crash_child`] where its database is.
const CRASH_PATH_VAR: &str = "GRAFEO_RDF_PROFILE_CRASH_PATH";

fn triples(db: &GrafeoDB) -> usize {
    db.execute_sparql("SELECT ?s ?p ?o WHERE { ?s ?p ?o }")
        .unwrap()
        .rows()
        .len()
}

/// Opens a new database at `path` and inserts one triple.
fn insert(path: &Path) -> GrafeoDB {
    let db = GrafeoDB::with_config(Config::persistent(path)).unwrap();
    db.execute_sparql(
        "INSERT DATA { <http://ex.org/alix> <http://ex.org/knows> <http://ex.org/gus> }",
    )
    .unwrap();
    assert_eq!(triples(&db), 1, "{}", path.display());
    db
}

/// Child-process entry for [`triples_survive_a_reopen`]; a no-op when run
/// directly.
#[test]
fn crash_child() {
    let Some(path) = std::env::var_os(CRASH_PATH_VAR) else {
        return;
    };
    let _db = insert(Path::new(&path));
    // Crash: no close(), no checkpoint, no destructors.
    std::process::exit(0);
}

#[test]
fn triples_survive_a_reopen() {
    let dir = std::env::temp_dir().join(format!("grafeo-rdf-profile-{}", std::process::id()));
    // A run that failed leaves its files behind.
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    // `close()` checkpoints the file; a process that exits without it leaves
    // its changes in the sidecar WAL, which the reopen replays.
    for (name, crash) in [("closed.grafeo", false), ("crashed.grafeo", true)] {
        let path = dir.join(name);
        if crash {
            let status = child_process::run(
                std::process::Command::new(std::env::current_exe().unwrap())
                    .args(["--exact", "crash_child", "--nocapture"])
                    .env(CRASH_PATH_VAR, &path),
            )
            .unwrap();
            assert!(status.success(), "{name}: the child process failed");
        } else {
            insert(&path).close().unwrap();
        }
        let db = GrafeoDB::with_config(Config::persistent(&path)).unwrap();
        assert_eq!(triples(&db), 1, "{name} after reopen");
        db.close().unwrap();
    }
    std::fs::remove_dir_all(&dir).unwrap();
}
