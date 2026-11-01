//! A persistent database replays an RDF load after a crash (0.6.1): the load
//! writes its triples to the WAL as one group, also when the file is parsed
//! in parallel (1 MiB or more) and the indexes are built in parallel (4,096
//! triples or more). A child process loads and exits without closing the
//! database, as a crash does; the reopened database holds what an in-memory
//! load of the same file holds, with the writes that followed the load.
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test rdf_load_crash_replay
//! ```

#[cfg(all(
    feature = "wal",
    feature = "grafeo-file",
    feature = "sparql",
    feature = "triple-store"
))]
mod tests {
    use std::collections::BTreeSet;
    use std::fmt::Write;
    use std::path::{Path, PathBuf};

    use grafeo_common::testing::child_process;
    use grafeo_engine::config::StorageFormat;
    use grafeo_engine::{Config, GrafeoDB};

    const SCENARIO_VAR: &str = "GRAFEO_RDF_LOAD_CRASH_SCENARIO";
    const PATH_VAR: &str = "GRAFEO_RDF_LOAD_CRASH_PATH";
    const FILE_VAR: &str = "GRAFEO_RDF_LOAD_CRASH_FILE";
    const PEOPLE: usize = 5_000;

    fn open(path: &Path) -> GrafeoDB {
        GrafeoDB::with_config(Config::persistent(path).with_storage_format(StorageFormat::Auto))
            .unwrap()
    }

    fn sidecar_wal(path: &Path) -> PathBuf {
        let mut sidecar = path.as_os_str().to_owned();
        sidecar.push(".wal");
        PathBuf::from(sidecar)
    }

    /// Five triples per person, 25,000 in all: over 1 MiB of N-Triples.
    fn ntriples() -> String {
        let mut text = String::from("# people\n");
        for i in 0..PEOPLE {
            let person = format!("<http://ex.org/person/{i}>");
            writeln!(
                text,
                "{person} <http://ex.org/name> \"Person number {i}\" ."
            )
            .unwrap();
            writeln!(
                text,
                "{person} <http://ex.org/age> \"{}\"^^<http://www.w3.org/2001/XMLSchema#integer> .",
                20 + i % 50
            )
            .unwrap();
            writeln!(
                text,
                "{person} <http://ex.org/livesIn> <http://ex.org/city/{}> .",
                ["Amsterdam", "Berlin", "Paris", "Prague", "Barcelona"][i % 5]
            )
            .unwrap();
            writeln!(
                text,
                "{person} <http://ex.org/knows> <http://ex.org/person/{}> .",
                (i + 1) % PEOPLE
            )
            .unwrap();
            writeln!(
                text,
                "{person} <http://ex.org/motto> \"line one\\nline \\\"two\\\"\"@en ."
            )
            .unwrap();
        }
        text
    }

    const AFTER_THE_LOAD: [&str; 2] = [
        "INSERT DATA { <http://ex.org/person/0> <http://ex.org/nick> \"Alix\" . \
         <http://ex.org/person/1> <http://ex.org/nick> \"Gus\" }",
        "DELETE DATA { <http://ex.org/person/2> <http://ex.org/name> \"Person number 2\" }",
    ];

    /// Runs `scenario` in a child process that exits without closing the
    /// database, like a crash.
    fn crash_after(scenario: &str, path: &Path, file: &Path) {
        let status = child_process::run(
            std::process::Command::new(std::env::current_exe().unwrap())
                .args(["--exact", "tests::crash_child", "--nocapture"])
                .env(SCENARIO_VAR, scenario)
                .env(PATH_VAR, path)
                .env(FILE_VAR, file),
        )
        .unwrap();
        assert!(status.success(), "scenario {scenario} failed");
    }

    /// Child-process entry for [`crash_after`]; a no-op when run directly.
    #[test]
    fn crash_child() {
        let Ok(scenario) = std::env::var(SCENARIO_VAR) else {
            return;
        };
        let path = PathBuf::from(std::env::var_os(PATH_VAR).unwrap());
        let file = PathBuf::from(std::env::var_os(FILE_VAR).unwrap());
        let db = open(&path);
        assert_eq!(db.load_rdf(&file, None).unwrap(), 5 * PEOPLE);
        if scenario == "load_then_write" {
            for update in AFTER_THE_LOAD {
                db.execute_sparql(update).unwrap();
            }
        }
        // Exit without closing: no checkpoint, the WAL stays.
        std::process::exit(0);
    }

    fn triples(db: &GrafeoDB) -> BTreeSet<String> {
        db.rdf_store()
            .triples()
            .iter()
            .map(ToString::to_string)
            .collect()
    }

    /// What an in-memory database holds after the same steps.
    fn expected(file: &Path, write_after: bool) -> BTreeSet<String> {
        let db = GrafeoDB::new_in_memory();
        db.load_rdf(file, None).unwrap();
        if write_after {
            for update in AFTER_THE_LOAD {
                db.execute_sparql(update).unwrap();
            }
        }
        triples(&db)
    }

    fn replays(scenario: &str, write_after: bool) {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("people.grafeo");
        let file = dir.path().join("people.nt");
        let text = ntriples();
        assert!(text.len() > 1 << 20, "{} bytes", text.len());
        std::fs::write(&file, text).unwrap();

        crash_after(scenario, &path, &file);
        assert!(sidecar_wal(&path).exists(), "the crash left the WAL");

        let db = open(&path);
        let want = expected(&file, write_after);
        assert_eq!(db.rdf_store().len(), want.len());
        assert_eq!(triples(&db), want);
        let rows = db
            .execute_sparql(
                "SELECT ?n WHERE { <http://ex.org/person/4999> <http://ex.org/knows> ?f . \
                 ?f <http://ex.org/name> ?n }",
            )
            .unwrap();
        assert_eq!(rows.rows().len(), 1, "person 4999 knows person 0");
        db.close().unwrap();

        // Closed cleanly, it opens with the same triples.
        let db = open(&path);
        assert_eq!(triples(&db), want);
        db.close().unwrap();
    }

    #[test]
    fn a_load_is_replayed_after_a_crash() {
        replays("load", false);
    }

    #[test]
    fn a_load_and_the_writes_after_it_are_replayed_after_a_crash() {
        replays("load_then_write", true);
    }
}
