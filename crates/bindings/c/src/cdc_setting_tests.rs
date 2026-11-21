//! `grafeo_set_cdc(db, enabled)` stores the CDC setting with the database (#679) and returns a
//! status: after a close and a reopen `grafeo_is_cdc_enabled` is what it was, and a read-only
//! open gets an error status with the setting unchanged. `grafeo_set_cdc_enabled`, the old
//! function without a status, is deprecated naming it and still turns capture on and off.

use std::ffi::CString;

use crate::database::{
    grafeo_close, grafeo_execute, grafeo_free_database, grafeo_free_result, grafeo_is_cdc_enabled,
    grafeo_open, grafeo_open_memory, grafeo_open_read_only, grafeo_set_cdc, grafeo_set_cdc_enabled,
};
use crate::error::GrafeoStatus;

/// A database file in a directory of its own, removed with the guard.
struct File(std::path::PathBuf);

impl File {
    fn new(name: &str) -> Self {
        let dir = std::env::temp_dir().join(format!("grafeo-c-cdc-{}-{name}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        Self(dir)
    }

    fn path(&self) -> CString {
        CString::new(self.0.join("db.grafeo").to_str().unwrap()).unwrap()
    }
}

impl Drop for File {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn close(db: *mut crate::types::GrafeoDatabase) {
    assert_eq!(grafeo_close(db), GrafeoStatus::Ok);
    grafeo_free_database(db);
}

fn insert(db: *mut crate::types::GrafeoDatabase) {
    let query = CString::new("INSERT (:Person {name: 'Alix'})").unwrap();
    let result = grafeo_execute(db, query.as_ptr());
    assert!(!result.is_null());
    grafeo_free_result(result);
}

#[test]
fn the_setting_survives_a_reopen_both_ways() {
    let file = File::new("reopen");
    let path = file.path();
    let open = |path: &CString| -> *mut crate::types::GrafeoDatabase {
        let db = grafeo_open(path.as_ptr());
        assert!(!db.is_null());
        db
    };
    let db = open(&path);
    assert!(!grafeo_is_cdc_enabled(db));
    assert_eq!(grafeo_set_cdc(db, true), GrafeoStatus::Ok);
    close(db);
    let db = open(&path);
    assert!(grafeo_is_cdc_enabled(db), "on after the reopen");
    assert_eq!(grafeo_set_cdc(db, false), GrafeoStatus::Ok);
    close(db);
    let db = open(&path);
    assert!(!grafeo_is_cdc_enabled(db), "off after the reopen");
    close(db);
}

#[test]
fn a_read_only_open_gets_an_error_status() {
    let file = File::new("read-only");
    let path = file.path();
    let db = grafeo_open(path.as_ptr());
    insert(db);
    close(db);
    let reader = grafeo_open_read_only(path.as_ptr());
    assert!(!reader.is_null());
    assert_ne!(grafeo_set_cdc(reader, true), GrafeoStatus::Ok);
    assert!(!grafeo_is_cdc_enabled(reader));
    grafeo_free_database(reader);
}

#[test]
#[allow(
    deprecated,
    reason = "the deprecated function must keep working until 0.7.0"
)]
fn the_deprecated_function_still_works() {
    let db = grafeo_open_memory();
    grafeo_set_cdc_enabled(db, true);
    assert!(grafeo_is_cdc_enabled(db));
    grafeo_set_cdc_enabled(db, false);
    assert!(!grafeo_is_cdc_enabled(db));
    grafeo_free_database(db);
}

/// The deprecated function keeps its old behaviour, as Rust's `set_cdc_enabled` does: the
/// setting holds for the open, and nothing is stored.
#[test]
fn the_deprecated_function_stores_nothing() {
    let file = File::new("deprecated");
    let path = file.path();
    let db = grafeo_open(path.as_ptr());
    assert!(!db.is_null());
    grafeo_set_cdc_enabled(db, true);
    assert!(grafeo_is_cdc_enabled(db), "on for this open");
    close(db);
    let db = grafeo_open(path.as_ptr());
    assert!(!grafeo_is_cdc_enabled(db), "nothing was stored");
    close(db);
}
