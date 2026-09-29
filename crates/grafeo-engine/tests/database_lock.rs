//! Only one `GrafeoDB` can open a persistent database for writing (#405).
//!
//! Without a lock, two processes could open the same WAL-directory database
//! and the one that closed last silently overwrote the other's commits. A
//! second open now fails with a "locked" error until the first one closes.
//!
//! ```bash
//! cargo test -p grafeo-engine --features full --test database_lock
//! ```

#![allow(missing_docs)]

#[cfg(feature = "wal")]
mod tests {
    use grafeo_common::testing::child_process;
    use grafeo_common::types::Value;
    use grafeo_engine::config::StorageFormat;
    use grafeo_engine::{Config, GrafeoDB};
    use std::path::{Path, PathBuf};

    /// Environment variable that turns [`child_open_fails`] into a helper run
    /// in a separate process.
    const CHILD_PATH_VAR: &str = "GRAFEO_LOCK_TEST_CHILD_PATH";
    const CHILD_FORMAT_VAR: &str = "GRAFEO_LOCK_TEST_CHILD_FORMAT";

    fn formats(dir: &Path) -> Vec<(&'static str, PathBuf, StorageFormat)> {
        let mut formats = vec![(
            "wal-directory",
            dir.join("dir-db"),
            StorageFormat::WalDirectory,
        )];
        #[cfg(feature = "grafeo-file")]
        formats.push((
            "single-file",
            dir.join("single.grafeo"),
            StorageFormat::SingleFile,
        ));
        formats
    }

    fn format_from_name(name: &str) -> StorageFormat {
        match name {
            "wal-directory" => StorageFormat::WalDirectory,
            "single-file" => StorageFormat::SingleFile,
            other => panic!("unknown format {other}"),
        }
    }

    fn open(path: &Path, format: StorageFormat) -> grafeo_common::utils::error::Result<GrafeoDB> {
        GrafeoDB::with_config(Config::persistent(path).with_storage_format(format))
    }

    fn names(db: &GrafeoDB) -> Vec<Value> {
        let result = db
            .session()
            .execute("MATCH (n:Item) RETURN n.name ORDER BY n.name")
            .unwrap();
        result.rows().iter().map(|row| row[0].clone()).collect()
    }

    fn insert(db: &GrafeoDB, name: &str) {
        db.session()
            .execute(&format!("INSERT (:Item {{name: '{name}'}})"))
            .unwrap();
    }

    #[test]
    fn second_open_fails_while_first_is_open() {
        let dir = tempfile::tempdir().unwrap();
        for (name, path, format) in formats(dir.path()) {
            let first = open(&path, format).unwrap();
            insert(&first, "Alix");

            let Err(err) = open(&path, format) else {
                panic!("{name}: second open of a database in use must fail");
            };
            assert!(
                err.to_string().contains("locked"),
                "{name}: expected a locked error, got: {err}"
            );

            // The failed open must not disturb the first handle or its data.
            insert(&first, "Gus");
            first.close().unwrap();

            let reopened = open(&path, format).unwrap();
            assert_eq!(
                names(&reopened),
                vec![Value::String("Alix".into()), Value::String("Gus".into())],
                "{name}: data after reopen"
            );
            reopened.close().unwrap();
        }
    }

    #[test]
    fn close_releases_the_lock() {
        let dir = tempfile::tempdir().unwrap();
        for (name, path, format) in formats(dir.path()) {
            let first = open(&path, format).unwrap();
            insert(&first, "Alix");
            first.close().unwrap();

            // `first` is still alive: close() alone must release the lock.
            let second = open(&path, format)
                .unwrap_or_else(|e| panic!("{name}: open after close failed: {e}"));
            assert_eq!(names(&second), vec![Value::String("Alix".into())]);
            second.close().unwrap();
            drop(first);
        }
    }

    #[test]
    fn drop_releases_the_lock() {
        let dir = tempfile::tempdir().unwrap();
        for (name, path, format) in formats(dir.path()) {
            {
                let first = open(&path, format).unwrap();
                insert(&first, "Alix");
            }
            let second = open(&path, format)
                .unwrap_or_else(|e| panic!("{name}: open after drop failed: {e}"));
            assert_eq!(names(&second), vec![Value::String("Alix".into())]);
            second.close().unwrap();
        }
    }

    /// The scenario from the issue: another process opens the database while
    /// this one has it open. It must fail instead of later overwriting it.
    #[test]
    fn open_from_another_process_fails() {
        let dir = tempfile::tempdir().unwrap();
        for (name, path, format) in formats(dir.path()) {
            let first = open(&path, format).unwrap();
            insert(&first, "Alix");

            let child = child_process::output(
                std::process::Command::new(std::env::current_exe().unwrap())
                    .args(["--exact", "tests::child_open_fails", "--nocapture"])
                    .env(CHILD_PATH_VAR, &path)
                    .env(CHILD_FORMAT_VAR, name),
            )
            .unwrap();
            let stdout = String::from_utf8_lossy(&child.stdout);
            assert!(
                child.status.success(),
                "{name}: the other process could open the database\n{}",
                String::from_utf8_lossy(&child.stderr)
            );
            // A child that matched no test would also exit successfully.
            assert!(
                stdout.contains("1 passed"),
                "{name}: the child ran no test:\n{stdout}"
            );

            insert(&first, "Gus");
            first.close().unwrap();
            let reopened = open(&path, format).unwrap();
            assert_eq!(
                names(&reopened),
                vec![Value::String("Alix".into()), Value::String("Gus".into())],
                "{name}: data after reopen"
            );
            reopened.close().unwrap();
        }
    }

    /// Helper for [`open_from_another_process_fails`]; a no-op when run
    /// directly.
    #[test]
    fn child_open_fails() {
        let (Some(path), Ok(format)) = (
            std::env::var_os(CHILD_PATH_VAR),
            std::env::var(CHILD_FORMAT_VAR),
        ) else {
            return;
        };
        let format = format_from_name(&format);
        match open(Path::new(&path), format) {
            Ok(db) => {
                // Do not write anything on the way out.
                std::mem::forget(db);
                panic!("opened a database that another process holds");
            }
            Err(err) => assert!(err.to_string().contains("locked"), "{err}"),
        }
    }
}
