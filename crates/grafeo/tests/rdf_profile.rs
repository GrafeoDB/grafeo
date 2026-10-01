//! The `rdf` profile on its own keeps its data across a reopen, in both
//! storage formats. Without the LPG store it kept nothing (#544): loading
//! the file, replaying the WAL and checkpoints need it, so the profile
//! includes it.
//!
//! ```bash
//! cargo test -p grafeo --no-default-features --features rdf --test rdf_profile
//! ```

#![cfg(feature = "rdf")]

use grafeo::{Config, GrafeoDB};

fn triples(db: &GrafeoDB) -> usize {
    db.execute_sparql("SELECT ?s ?p ?o WHERE { ?s ?p ?o }")
        .unwrap()
        .rows()
        .len()
}

#[test]
fn triples_survive_a_reopen() {
    let dir = std::env::temp_dir().join(format!("grafeo-rdf-profile-{}", std::process::id()));
    // A run that failed leaves its files behind.
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    // A `.grafeo` path is a single file, any other path a WAL directory.
    for name in ["single.grafeo", "wal-directory"] {
        let path = dir.join(name);
        {
            let db = GrafeoDB::with_config(Config::persistent(&path)).unwrap();
            db.execute_sparql(
                "INSERT DATA { <http://ex.org/alix> <http://ex.org/knows> <http://ex.org/gus> }",
            )
            .unwrap();
            assert_eq!(triples(&db), 1, "{name}");
            db.close().unwrap();
        }
        let db = GrafeoDB::with_config(Config::persistent(&path)).unwrap();
        assert_eq!(triples(&db), 1, "{name} after reopen");
        db.close().unwrap();
    }
    std::fs::remove_dir_all(&dir).unwrap();
}
