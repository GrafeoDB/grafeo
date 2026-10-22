//! The `grafeo` binary, run as a user runs it: one process per command, each
//! opening and closing the database file.
//!
//! #308: a text index made by one command answers `text_match` in the next.
//! #267: `grafeo backup full` and `grafeo backup restore-to-epoch` give back
//! exactly the people of the backup's epoch, and `grafeo backup incremental`
//! extends the chain. The tests check what the commands wrote through the
//! engine, and read the CLI's own output only for which rows a query returned.
//!
//! ```bash
//! cargo test -p grafeo-cli --all-features --test cli_runs
//! ```

use std::path::Path;
use std::process::Command;

use grafeo_common::types::Value;
use grafeo_engine::GrafeoDB;

/// Runs the `grafeo` binary with `args`; its standard output, after
/// checking that it succeeded.
fn grafeo(args: &[&str]) -> String {
    let output = Command::new(env!("CARGO_BIN_EXE_grafeo"))
        .arg("--no-color")
        .args(args)
        .output()
        .expect("run grafeo");
    assert!(
        output.status.success(),
        "grafeo {args:?} failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    String::from_utf8(output.stdout).unwrap()
}

/// The rows `grafeo --format json query` printed for `query`.
fn query_rows(db: &Path, query: &str) -> Vec<serde_json::Value> {
    let out = grafeo(&["--format", "json", "query", db.to_str().unwrap(), query]);
    match serde_json::from_str(&out) {
        Ok(serde_json::Value::Array(rows)) => rows,
        other => panic!("`{query}` printed {out:?}: {other:?}"),
    }
}

fn names(db: &Path) -> Vec<Value> {
    let db = GrafeoDB::open_read_only(db).unwrap();
    db.execute("MATCH (p:Person) RETURN p.name AS name ORDER BY name")
        .unwrap()
        .rows()
        .iter()
        .map(|row| row[0].clone())
        .collect()
}

fn strings(values: &[&str]) -> Vec<Value> {
    values.iter().map(|value| Value::from(*value)).collect()
}

/// #308 through the CLI: documents and a text index written by separate
/// commands, then `text_match` in a new process finds the document, and a
/// document written after the index is found too.
#[cfg(feature = "text-index")]
#[test]
fn a_text_index_made_by_one_command_answers_the_next() {
    let dir = tempfile::tempdir().unwrap();
    let db = dir.path().join("docs.grafeo");
    let path = db.to_str().unwrap();
    grafeo(&["init", path]);
    grafeo(&[
        "query",
        path,
        "INSERT (:Doc {title: 'Canals', body: 'boats on the canals of Amsterdam'}), \
         (:Doc {title: 'Bridges', body: 'bridges over the river in Prague'})",
    ]);
    grafeo(&[
        "query",
        path,
        "CREATE INDEX doc_body FOR (d:Doc) ON (d.body) USING TEXT",
    ]);
    let search = "MATCH (d:Doc) WHERE text_match(d.body, 'canals') RETURN d.title AS t";
    let rows = query_rows(&db, search);
    assert_eq!(rows.len(), 1, "{rows:?}");
    assert!(rows[0]["t"].to_string().contains("Canals"), "{rows:?}");

    grafeo(&[
        "query",
        path,
        "INSERT (:Doc {title: 'Gondolas', body: 'gondolas on the canals of Venice'})",
    ]);
    let rows = query_rows(&db, search);
    let mut titles: Vec<String> = rows.iter().map(|row| row["t"].to_string()).collect();
    titles.sort();
    assert_eq!(titles.len(), 2, "{rows:?}");
    assert!(
        titles[0].contains("Canals") && titles[1].contains("Gondolas"),
        "{titles:?}"
    );
    let db = GrafeoDB::open_read_only(&db).unwrap();
    assert_eq!(
        db.text_search("Doc", "body", "canals", 5, None)
            .unwrap()
            .len(),
        2,
        "the index the CLI made holds both documents"
    );
}

/// The epoch the newest full backup in `backups` ends at.
fn latest_full_epoch(backups: &Path) -> String {
    GrafeoDB::read_backup_manifest(backups)
        .unwrap()
        .unwrap()
        .latest_full()
        .unwrap()
        .end_epoch
        .as_u64()
        .to_string()
}

/// #267 through the CLI: two full backups, each restored to its epoch by
/// `grafeo backup restore-to-epoch`, give back exactly the people of that
/// epoch.
#[test]
fn full_backups_restore_to_their_epochs() {
    let dir = tempfile::tempdir().unwrap();
    let db = dir.path().join("people.grafeo");
    let backups = dir.path().join("backups");
    let (path, backups_arg) = (db.to_str().unwrap(), backups.to_str().unwrap());
    grafeo(&["init", path]);
    grafeo(&["query", path, "INSERT (:Person {name: 'Alix'})"]);
    grafeo(&["backup", "full", path, "-o", backups_arg]);
    let alix = latest_full_epoch(&backups);
    grafeo(&["query", path, "INSERT (:Person {name: 'Gus'})"]);
    grafeo(&["backup", "full", path, "-o", backups_arg]);
    let gus = latest_full_epoch(&backups);

    for (epoch, expected) in [(&alix, &["Alix"][..]), (&gus, &["Alix", "Gus"][..])] {
        let restored = dir.path().join(format!("restored-{epoch}.grafeo"));
        grafeo(&[
            "backup",
            "restore-to-epoch",
            backups_arg,
            "--epoch",
            epoch,
            "-o",
            restored.to_str().unwrap(),
        ]);
        assert_eq!(names(&restored), strings(expected), "epoch {epoch}");
    }
    assert_eq!(names(&db), strings(&["Alix", "Gus"]), "the database itself");
}

/// #267 through the CLI: `grafeo backup incremental` after `grafeo backup
/// full`, each in its own process, extends the chain, and the restore to
/// its epoch has every person.
#[test]
fn an_incremental_backup_extends_the_chain() {
    let dir = tempfile::tempdir().unwrap();
    let db = dir.path().join("people.grafeo");
    let backups = dir.path().join("backups");
    let (path, backups_arg) = (db.to_str().unwrap(), backups.to_str().unwrap());
    grafeo(&["init", path]);
    grafeo(&["query", path, "INSERT (:Person {name: 'Alix'})"]);
    grafeo(&["backup", "full", path, "-o", backups_arg]);
    grafeo(&["query", path, "INSERT (:Person {name: 'Gus'})"]);
    grafeo(&["backup", "incremental", path, "-o", backups_arg]);
    let manifest = GrafeoDB::read_backup_manifest(&backups).unwrap().unwrap();
    let last = manifest
        .segments
        .last()
        .unwrap()
        .end_epoch
        .as_u64()
        .to_string();
    let restored = dir.path().join("restored.grafeo");
    grafeo(&[
        "backup",
        "restore-to-epoch",
        backups_arg,
        "--epoch",
        &last,
        "-o",
        restored.to_str().unwrap(),
    ]);
    assert_eq!(names(&restored), strings(&["Alix", "Gus"]));
}
