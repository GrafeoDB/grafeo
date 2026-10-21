//! CLI command implementations.

use std::path::Path;

use anyhow::{Context, Result};
use grafeo_engine::GrafeoDB;

pub mod backup;
pub mod compact;
pub mod data;
pub mod import;
pub mod index;
pub mod info;
pub mod init;
pub mod query;
pub mod schema;
pub mod stats;
pub mod validate;
pub mod version;
pub mod wal;

/// Open an existing database, returning a clear error if the path does not exist.
///
/// Use this instead of `GrafeoDB::open` for commands that expect a pre-existing database.
/// The `init` command should use `GrafeoDB::open` directly since it intentionally creates.
pub fn open_existing(path: &Path) -> Result<GrafeoDB> {
    let path = existing_path(path)?;
    GrafeoDB::open(&path).with_context(|| format!("Failed to open database at {}", path.display()))
}

/// Open an existing database to read it only, for the commands that do not
/// write (`info`, `stats`, `schema`, `validate`, the exports, `index list`
/// and `index stats`, `wal status`): a 0.5.x file is read in place, not
/// migrated, and closing it writes nothing, so the file stays as it was.
pub fn open_existing_read_only(path: &Path) -> Result<GrafeoDB> {
    let path = existing_path(path)?;
    GrafeoDB::open_read_only(&path)
        .with_context(|| format!("Failed to open database at {} to read it", path.display()))
}

/// `path` without trailing separators, or an error when nothing is there:
/// `db/` names the database file `db` too, but no longer reaches it as a
/// path once a 0.5.x directory became a file.
fn existing_path(path: &Path) -> Result<std::path::PathBuf> {
    let path: std::path::PathBuf = path.components().collect();
    if !path.exists() {
        anyhow::bail!(
            "Database not found: {}\n\
             Use `grafeo init {}` to create a new database.",
            path.display(),
            path.display()
        );
    }
    Ok(path)
}

#[cfg(test)]
mod tests {
    use super::open_existing;

    /// `db/` names the database file `db`: the CLI opens it, as a 0.5.x WAL
    /// directory named `db/` becomes the file `db` when 0.6 migrates it.
    #[test]
    fn a_path_with_a_trailing_separator_opens_the_database_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("db");
        let db = grafeo_engine::GrafeoDB::open(&path).unwrap();
        db.execute("INSERT (:Person {name: 'Alix'})").unwrap();
        db.close().unwrap();
        drop(db);

        let written = dir.path().join("db/");
        let db = open_existing(&written).unwrap_or_else(|error| panic!("{error:#}"));
        assert_eq!(db.node_count(), 1, "the database at {}", written.display());
        db.close().unwrap();
    }

    /// The names and bytes of the files in `dir`, in name order.
    fn snapshot(dir: &std::path::Path) -> Vec<(String, Vec<u8>)> {
        let mut files: Vec<(String, Vec<u8>)> = std::fs::read_dir(dir)
            .unwrap()
            .map(|entry| {
                let entry = entry.unwrap();
                let bytes = if entry.path().is_file() {
                    std::fs::read(entry.path()).unwrap()
                } else {
                    Vec::new()
                };
                (entry.file_name().to_string_lossy().into_owned(), bytes)
            })
            .collect();
        files.sort();
        files
    }

    /// Runs every command that only reads on the database at `path`, and
    /// checks after each one that the files next to it are as they were.
    fn every_reading_command_leaves_the_files(path: &std::path::Path) {
        use crate::{DataCommands, IndexCommands, OutputFormat, WalCommands};
        let dir = path.parent().unwrap();
        let out = tempfile::tempdir().unwrap();
        let before = snapshot(dir);
        let run = |name: &str, result: anyhow::Result<()>| {
            result.unwrap_or_else(|error| panic!("{name}: {error:#}"));
            assert_eq!(snapshot(dir), before, "{name} changed the database files");
        };
        let path_buf = path.to_path_buf();
        run("info", super::info::run(path, OutputFormat::Json, true));
        run("stats", super::stats::run(path, OutputFormat::Json, true));
        run("schema", super::schema::run(path, OutputFormat::Json, true));
        run(
            "validate",
            super::validate::run(path, OutputFormat::Json, true),
        );
        run(
            "index list",
            super::index::run(
                IndexCommands::List {
                    path: path_buf.clone(),
                },
                OutputFormat::Json,
                true,
            ),
        );
        run(
            "index stats",
            super::index::run(
                IndexCommands::Stats {
                    path: path_buf.clone(),
                },
                OutputFormat::Json,
                true,
            ),
        );
        run(
            "wal status",
            super::wal::run(
                WalCommands::Status {
                    path: path_buf.clone(),
                },
                OutputFormat::Json,
                true,
            ),
        );
        // `grafeo query` with a statement that does not write.
        let mut reading = vec![
            (crate::QueryLanguage::Gql, "MATCH (n) RETURN count(n) AS c"),
            (
                crate::QueryLanguage::Gql,
                "EXPLAIN INSERT (:Person {name: 'Mia'})",
            ),
        ];
        #[cfg(feature = "cypher")]
        reading.push((
            crate::QueryLanguage::Cypher,
            "MATCH (n) RETURN count(n) AS c",
        ));
        for (lang, query) in reading {
            run(&format!("query {query}"), run_query(path, lang, query));
        }
        for format in ["json", "gexf", "graphml"] {
            run(
                &format!("data dump --export-format {format}"),
                super::data::run(
                    DataCommands::Dump {
                        path: path_buf.clone(),
                        output: out.path().join(format!("dump.{format}")),
                        export_format: Some(format.to_string()),
                    },
                    OutputFormat::Json,
                    true,
                ),
            );
        }
    }

    /// `grafeo query` of `query` in `lang` on the database at `path`, quietly.
    fn run_query(
        path: &std::path::Path,
        lang: crate::QueryLanguage,
        query: &str,
    ) -> anyhow::Result<()> {
        super::query::run(super::query::QueryRunOptions {
            path,
            query: Some(query.to_string()),
            file: None,
            stdin: false,
            params: &[],
            lang,
            format: crate::OutputFormat::Json,
            quiet: true,
            timing: false,
            max_width: None,
        })
    }

    /// `grafeo query` opens the database read-write for a statement that
    /// writes, and its changes stay.
    #[test]
    fn a_writing_query_writes() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("people.grafeo");
        let db = grafeo_engine::GrafeoDB::open(&path).unwrap();
        db.execute("INSERT (:Person {name: 'Alix'})").unwrap();
        db.close().unwrap();
        drop(db);

        run_query(
            &path,
            crate::QueryLanguage::Gql,
            "INSERT (:Person {name: 'Gus'})",
        )
        .unwrap();
        run_query(
            &path,
            crate::QueryLanguage::Gql,
            "MATCH (n:Person {name: 'Alix'}) SET n.city = 'Amsterdam'",
        )
        .unwrap();
        let db = grafeo_engine::GrafeoDB::open_read_only(&path).unwrap();
        let result = db
            .execute("MATCH (n:Person) RETURN n.name AS name, n.city AS city ORDER BY name")
            .unwrap();
        assert_eq!(
            format!("{:?}", result.rows()),
            "[[String(\"Alix\"), String(\"Amsterdam\")], [String(\"Gus\"), Null]]"
        );
    }

    /// `--lang cypher` with parameters runs the statement as Cypher: it ran
    /// as GQL, through the GQL call with parameters.
    #[cfg(feature = "cypher")]
    #[test]
    fn a_cypher_query_with_parameters_runs_as_cypher() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("people.grafeo");
        let db = grafeo_engine::GrafeoDB::open(&path).unwrap();
        db.execute("INSERT (:Person {name: 'Alix'})").unwrap();
        db.close().unwrap();
        drop(db);

        let query = |query: &str| {
            super::query::run(super::query::QueryRunOptions {
                path: &path,
                query: Some(query.to_string()),
                file: None,
                stdin: false,
                params: &["name=Gus".to_string()],
                lang: crate::QueryLanguage::Cypher,
                format: crate::OutputFormat::Json,
                quiet: true,
                timing: false,
                max_width: None,
            })
        };
        query("CREATE (:Person {name: $name})").unwrap();
        let error = query("MATCH (n {name: $name}) RETURN n.name AS").unwrap_err();
        assert!(format!("{error:#}").contains("[Cypher]"), "{error:#}");
        let db = grafeo_engine::GrafeoDB::open_read_only(&path).unwrap();
        let names = db
            .execute("MATCH (n:Person) RETURN n.name AS name ORDER BY name")
            .unwrap();
        assert_eq!(
            format!("{:?}", names.rows()),
            "[[String(\"Alix\")], [String(\"Gus\")]]"
        );
    }

    /// A statement that does not parse fails without touching the files.
    #[test]
    fn a_query_that_does_not_parse_leaves_the_files() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("people.grafeo");
        let db = grafeo_engine::GrafeoDB::open(&path).unwrap();
        db.execute("INSERT (:Person {name: 'Alix'})").unwrap();
        db.close().unwrap();
        drop(db);

        let before = snapshot(dir.path());
        let error = run_query(&path, crate::QueryLanguage::Gql, "MATCH (n RETURN n").unwrap_err();
        assert!(
            format!("{error:#}").contains("Query execution failed"),
            "{error:#}"
        );
        assert_eq!(snapshot(dir.path()), before);
    }

    /// The commands that only read leave a database file byte for byte as it
    /// was: they used to open it read-write, which writes a checkpoint at
    /// close.
    #[test]
    fn the_reading_commands_leave_a_database_file_unchanged() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("people.grafeo");
        let db = grafeo_engine::GrafeoDB::open(&path).unwrap();
        db.execute("INSERT (:Person {name: 'Alix'})-[:KNOWS]->(:Person {name: 'Gus'})")
            .unwrap();
        db.close().unwrap();
        drop(db);
        every_reading_command_leaves_the_files(&path);
    }

    /// On a 0.5.x database they read the file in place: `grafeo info`
    /// migrated it to 0.6, a one-way change, and left a `.pre-0.6` copy. The
    /// released 0.5.44 fixtures hold RDF triples, which a CLI built without
    /// the triple store refuses to open at all.
    #[cfg(feature = "triple-store")]
    #[test]
    fn the_reading_commands_do_not_migrate_a_0_5_database() {
        let fixture = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../grafeo-engine/tests/fixtures/released/0.5.44/closed.grafeo");
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("closed.grafeo");
        std::fs::copy(&fixture, &path).unwrap();
        every_reading_command_leaves_the_files(&path);
    }
}
