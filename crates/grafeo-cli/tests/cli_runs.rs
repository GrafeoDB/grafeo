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
#[cfg(feature = "text-index")]
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
    assert_eq!(
        query_rows(&db, search),
        [serde_json::json!({"t": "Canals"})]
    );

    grafeo(&[
        "query",
        path,
        "INSERT (:Doc {title: 'Gondolas', body: 'gondolas on the canals of Venice'})",
    ]);
    let ordered = "MATCH (d:Doc) WHERE text_match(d.body, 'canals') RETURN d.title AS t ORDER BY t";
    assert_eq!(
        query_rows(&db, ordered),
        [
            serde_json::json!({"t": "Canals"}),
            serde_json::json!({"t": "Gondolas"})
        ]
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

/// `--format json` prints each row as an object of JSON values, its keys in
/// the order of the columns: numbers, booleans and null as themselves,
/// strings without the quotes of their display text, lists as arrays, a
/// date as its text.
#[test]
fn json_output_keeps_value_types_and_column_order() {
    let dir = tempfile::tempdir().unwrap();
    let db = dir.path().join("people.grafeo");
    let path = db.to_str().unwrap();
    grafeo(&["init", path]);
    grafeo(&[
        "query",
        path,
        "INSERT (:Person {name: 'Alix', age: 30, score: 1.5, active: true, \
         tags: ['canals', 'bridges'], born: DATE '1988-03-19'})",
    ]);
    let query = "MATCH (p:Person) RETURN p.name AS name, p.age AS age, p.score AS score, \
                 p.active AS active, p.tags AS tags, p.born AS born, p.missing AS missing";
    let out = grafeo(&["--format", "json", "query", path, query]);
    let rows: serde_json::Value = serde_json::from_str(&out).unwrap();
    assert_eq!(
        rows,
        serde_json::json!([{
            "name": "Alix",
            "age": 30,
            "score": 1.5,
            "active": true,
            "tags": ["canals", "bridges"],
            "born": "1988-03-19",
            "missing": null
        }])
    );
    let at = |key: &str| out.find(&format!("\"{key}\"")).unwrap();
    assert!(
        at("name") < at("age") && at("age") < at("score") && at("born") < at("missing"),
        "the keys follow the columns: {out}"
    );
}

/// `--format csv` writes each value as its text: a string as it is, quoted
/// only when it holds a comma or a quote, and a date as its value.
#[test]
fn csv_output_writes_values_as_their_text() {
    let dir = tempfile::tempdir().unwrap();
    let db = dir.path().join("people.grafeo");
    let path = db.to_str().unwrap();
    grafeo(&["init", path]);
    grafeo(&[
        "query",
        path,
        "INSERT (:Person {name: 'Alix', age: 30, born: DATE '1988-03-19'}), \
         (:Person {name: 'Gus, Jr.', age: 25, born: DATE '1999-12-31'})",
    ]);
    let out = grafeo(&[
        "--format",
        "csv",
        "query",
        path,
        "MATCH (p:Person) RETURN p.name AS name, p.age AS age, p.born AS born ORDER BY age",
    ]);
    assert_eq!(
        out.replace("\r\n", "\n"),
        "name,age,born\n\"Gus, Jr.\",25,1999-12-31\nAlix,30,1988-03-19\n"
    );
}

/// `grafeo index list` names an index as `CREATE INDEX` named it, as
/// `SHOW INDEXES` does.
#[test]
fn index_list_shows_the_name_the_index_was_created_with() {
    let dir = tempfile::tempdir().unwrap();
    let db = dir.path().join("people.grafeo");
    let path = db.to_str().unwrap();
    grafeo(&["init", path]);
    grafeo(&["query", path, "INSERT (:Person {name: 'Alix'})"]);
    grafeo(&[
        "query",
        path,
        "CREATE INDEX person_name FOR (p:Person) ON (p.name)",
    ]);
    let out = grafeo(&["--format", "json", "index", "list", path]);
    let indexes: serde_json::Value = serde_json::from_str(&out).unwrap();
    assert_eq!(indexes[0]["name"], "person_name", "{out}");
    assert_eq!(indexes[0]["target"], "Person:name", "{out}");
    assert_eq!(indexes.as_array().unwrap().len(), 1, "{out}");
}

/// Runs the `grafeo` binary with `args`: its standard output and standard
/// error, after checking that it succeeded.
fn grafeo_streams(args: &[&str]) -> (String, String) {
    let output = Command::new(env!("CARGO_BIN_EXE_grafeo"))
        .arg("--no-color")
        .args(args)
        .output()
        .expect("run grafeo");
    let stderr = String::from_utf8(output.stderr).unwrap();
    assert!(output.status.success(), "grafeo {args:?} failed: {stderr}");
    (String::from_utf8(output.stdout).unwrap(), stderr)
}

/// `stdout` as JSON, failing the test with the command and its output.
fn parsed(args: &[&str], stdout: &str) -> serde_json::Value {
    serde_json::from_str(stdout)
        .unwrap_or_else(|error| panic!("grafeo {args:?} printed no JSON ({error}): {stdout:?}"))
}

/// With `--format json` every command prints JSON on stdout and nothing
/// else, the status lines of the commands that act go to stderr, and `-q`
/// removes those lines and keeps the results.
#[test]
fn every_command_prints_json_on_stdout_with_status_on_stderr() {
    let dir = tempfile::tempdir().unwrap();
    let db = dir.path().join("people.grafeo");
    let backups = dir.path().join("backups");
    let csv = dir.path().join("people.csv");
    let jsonl = dir.path().join("people.jsonl");
    let dump = dir.path().join("dump.jsonl");
    let copy = dir.path().join("copy.grafeo");
    let restored = dir.path().join("restored.grafeo");
    let loaded = dir.path().join("loaded.grafeo");
    std::fs::write(&csv, "name\nMia\n").unwrap();
    std::fs::write(&jsonl, "{\"name\": \"Butch\"}\n").unwrap();
    let p = |path: &Path| path.to_str().unwrap().to_string();
    let (path, backups_arg) = (p(&db), p(&backups));

    // Each command, with what its JSON must hold (`None`: any JSON value).
    let mut commands: Vec<(Vec<String>, Option<serde_json::Value>)> = vec![
        (vec!["init".into(), path.clone()], None),
        (
            vec![
                "query".into(),
                path.clone(),
                "INSERT (:Person {name: 'Alix'})".into(),
            ],
            Some(serde_json::json!([])),
        ),
        (
            vec![
                "query".into(),
                path.clone(),
                "MATCH (p:Person) RETURN p.name AS name".into(),
            ],
            Some(serde_json::json!([{"name": "Alix"}])),
        ),
        (vec!["info".into(), path.clone()], None),
        (vec!["stats".into(), path.clone()], None),
        (vec!["schema".into(), path.clone()], None),
        (vec!["validate".into(), path.clone()], None),
        (
            vec!["index".into(), "list".into(), path.clone()],
            Some(serde_json::json!([])),
        ),
        (vec!["index".into(), "stats".into(), path.clone()], None),
        (vec!["wal".into(), "status".into(), path.clone()], None),
        (
            vec!["wal".into(), "checkpoint".into(), path.clone()],
            Some(serde_json::json!({"database": path.clone(), "checkpointed": true})),
        ),
        (
            vec![
                "import".into(),
                "csv".into(),
                p(&csv),
                "--path".into(),
                path.clone(),
                "--label".into(),
                "Person".into(),
            ],
            Some(
                serde_json::json!({"file": p(&csv), "database": path.clone(), "label": "Person", "nodes": 1}),
            ),
        ),
        (
            vec![
                "import".into(),
                "jsonl".into(),
                p(&jsonl),
                "--path".into(),
                path.clone(),
                "--label".into(),
                "Person".into(),
            ],
            Some(
                serde_json::json!({"file": p(&jsonl), "database": path.clone(), "label": "Person", "nodes": 1}),
            ),
        ),
        (
            vec![
                "backup".into(),
                "full".into(),
                path.clone(),
                "-o".into(),
                backups_arg.clone(),
            ],
            None,
        ),
        (
            vec![
                "query".into(),
                path.clone(),
                "INSERT (:Person {name: 'Gus'})".into(),
            ],
            None,
        ),
        (
            vec![
                "backup".into(),
                "incremental".into(),
                path.clone(),
                "-o".into(),
                backups_arg.clone(),
            ],
            None,
        ),
        (
            vec!["backup".into(), "status".into(), backups_arg.clone()],
            None,
        ),
        (
            vec![
                "backup".into(),
                "create".into(),
                path.clone(),
                "-o".into(),
                p(&copy),
            ],
            Some(serde_json::json!({"database": path.clone(), "backup": p(&copy)})),
        ),
        (
            vec![
                "data".into(),
                "dump".into(),
                path.clone(),
                "-o".into(),
                p(&dump),
            ],
            Some(serde_json::json!({"output": p(&dump), "nodes": 4, "edges": 0})),
        ),
        (
            vec!["data".into(), "load".into(), p(&dump), p(&loaded)],
            Some(serde_json::json!({"database": p(&loaded), "nodes": 4, "edges": 0})),
        ),
        (
            vec!["compact".into(), path.clone(), "--dry-run".into()],
            None,
        ),
    ];
    let restore = vec!["backup".into(), "restore".into(), p(&copy), p(&restored)];
    commands.push((
        restore,
        Some(serde_json::json!({"backup": p(&copy), "database": p(&restored)})),
    ));

    for (args, expected) in &commands {
        let mut full: Vec<&str> = vec!["--format", "json"];
        full.extend(args.iter().map(String::as_str));
        let (stdout, _) = grafeo_streams(&full);
        let value = parsed(&full, &stdout);
        if let Some(expected) = expected {
            assert_eq!(&value, expected, "grafeo {full:?}");
        }
    }

    // The backup records, and a restore to the incremental's epoch.
    let (stdout, stderr) = grafeo_streams(&["--format", "json", "backup", "status", &backups_arg]);
    let status = parsed(&["backup", "status"], &stdout);
    assert_eq!(status["segments"][0]["kind"], "full", "{status}");
    assert_eq!(status["segments"][1]["kind"], "incremental", "{status}");
    assert_eq!(stderr, "", "backup status prints no status line");
    let epoch = status["segments"][1]["end_epoch"]
        .as_u64()
        .unwrap()
        .to_string();
    let (stdout, stderr) = grafeo_streams(&[
        "--format",
        "json",
        "backup",
        "restore-to-epoch",
        &backups_arg,
        "--epoch",
        &epoch,
        "-o",
        &p(&dir.path().join("at-epoch.grafeo")),
    ]);
    assert_eq!(
        parsed(&["backup", "restore-to-epoch"], &stdout)["epoch"],
        epoch.parse::<u64>().unwrap()
    );
    assert!(
        stderr.contains(&format!("Database restored to epoch {epoch}")),
        "the status line goes to stderr: {stderr:?}"
    );

    // -q keeps the results and drops the status lines.
    let (stdout, stderr) = grafeo_streams(&[
        "--format",
        "json",
        "-q",
        "query",
        &path,
        "MATCH (p:Person) RETURN p.name AS name ORDER BY name",
    ]);
    assert_eq!(
        parsed(&["-q", "query"], &stdout),
        serde_json::json!([
            {"name": "Alix"},
            {"name": "Butch"},
            {"name": "Gus"},
            {"name": "Mia"}
        ])
    );
    assert_eq!(stderr, "");
    let (stdout, stderr) = grafeo_streams(&[
        "--format",
        "json",
        "-q",
        "backup",
        "full",
        &path,
        "-o",
        &backups_arg,
    ]);
    assert_eq!(parsed(&["-q", "backup", "full"], &stdout)["kind"], "full");
    assert_eq!(stderr, "", "-q drops the status lines");
    let (stdout, stderr) = grafeo_streams(&["--format", "json", "-q", "info", &path]);
    assert!(parsed(&["-q", "info"], &stdout).is_object());
    assert_eq!(stderr, "");
}

/// In a table the results go to stdout and the row count to stderr; `-q`
/// keeps the table and drops the count.
#[test]
fn a_table_keeps_results_on_stdout_and_the_row_count_on_stderr() {
    let dir = tempfile::tempdir().unwrap();
    let db = dir.path().join("people.grafeo");
    let path = db.to_str().unwrap();
    grafeo(&["init", path]);
    grafeo(&["query", path, "INSERT (:Person {name: 'Alix'})"]);
    let query = "MATCH (p:Person) RETURN p.name AS name";
    let (stdout, stderr) = grafeo_streams(&["--format", "table", "query", path, query]);
    assert!(stdout.contains("Alix"), "{stdout}");
    assert!(!stdout.contains("1 row"), "{stdout}");
    assert!(stderr.contains("1 row"), "{stderr}");
    let (stdout, stderr) = grafeo_streams(&["--format", "table", "-q", "query", path, query]);
    assert!(stdout.contains("Alix"), "{stdout}");
    assert_eq!(stderr, "");
}
