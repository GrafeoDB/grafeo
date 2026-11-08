//! The memory settings through C (#751): a typed options object, `grafeo_options_new` and
//! `grafeo_options_free`, setters that return a status (`set_memory_limit`,
//! `set_query_memory_limit`, `set_spill_limit`, `set_unlimited_spill`), and
//! `grafeo_open_with_options(path or NULL, options or NULL)`. Unset means the default, with no
//! sentinels; a zero memory limit is refused at the setter; a zero spill limit is accepted and
//! means nothing spills. The settings read back through `grafeo_memory_usage_json`.

use std::ffi::{CStr, CString};

use crate::database::{
    grafeo_execute, grafeo_free_database, grafeo_free_result, grafeo_memory_usage_json,
    grafeo_open_memory,
};
use crate::error::GrafeoStatus;
use crate::options::{
    grafeo_open_with_options, grafeo_options_free, grafeo_options_new,
    grafeo_options_set_memory_limit, grafeo_options_set_query_memory_limit,
    grafeo_options_set_spill_limit, grafeo_options_set_unlimited_spill,
};
use crate::types::GrafeoDatabase;

const MIB: u64 = 1024 * 1024;

/// The database's memory report as JSON.
fn usage(db: *mut GrafeoDatabase) -> serde_json::Value {
    let json = grafeo_memory_usage_json(db);
    assert!(!json.is_null());
    // SAFETY: a string the binding returned, freed below.
    let text = unsafe { CStr::from_ptr(json) }
        .to_string_lossy()
        .into_owned();
    crate::database::grafeo_free_string(json);
    serde_json::from_str(&text).unwrap()
}

/// A unique path under the system's temporary directory, for a database file; removed with its
/// sidecars by [`remove_database`].
fn database_path(name: &str) -> std::path::PathBuf {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    std::env::temp_dir().join(format!(
        "grafeo-c-{name}-{}-{nanos}.grafeo",
        std::process::id()
    ))
}

fn remove_database(path: &std::path::Path) {
    let _ = std::fs::remove_file(path);
    for suffix in [".wal", ".spill"] {
        let mut sidecar = path.as_os_str().to_owned();
        sidecar.push(suffix);
        let _ = std::fs::remove_dir_all(std::path::PathBuf::from(sidecar));
    }
}

/// Each setter reads back through the report, for a database file (which has a spill directory,
/// so the spill settings apply): a zero spill limit reports 0, unlimited spill reports none.
#[test]
fn each_setter_reads_back_through_the_report() {
    let path = database_path("setters");
    let c_path = std::ffi::CString::new(path.to_string_lossy().into_owned()).unwrap();
    let options = grafeo_options_new();
    assert!(!options.is_null());
    assert_eq!(
        grafeo_options_set_memory_limit(options, 64 * MIB),
        GrafeoStatus::Ok
    );
    assert_eq!(
        grafeo_options_set_query_memory_limit(options, 8 * MIB),
        GrafeoStatus::Ok
    );
    assert_eq!(
        grafeo_options_set_spill_limit(options, 0),
        GrafeoStatus::Ok,
        "zero is valid"
    );
    let db = grafeo_open_with_options(c_path.as_ptr(), options);
    assert!(!db.is_null());
    let report = usage(db);
    assert_eq!(report["limit_bytes"], 64 * MIB);
    assert_eq!(report["query_limit_bytes"], 8 * MIB);
    assert_eq!(report["spill"]["limit_bytes"], 0);
    grafeo_free_database(db);

    assert_eq!(
        grafeo_options_set_unlimited_spill(options),
        GrafeoStatus::Ok
    );
    let db = grafeo_open_with_options(c_path.as_ptr(), options);
    assert!(!db.is_null());
    assert!(usage(db)["spill"]["limit_bytes"].is_null(), "unlimited");
    grafeo_free_database(db);
    grafeo_options_free(options);
    remove_database(&path);
}

/// An in-memory open has no spill directory, so nothing spills whatever the settings say: the
/// report's spill limit is 0, also with unlimited spill set.
#[test]
fn an_in_memory_open_spills_nothing_whatever_the_settings() {
    let options = grafeo_options_new();
    assert_eq!(
        grafeo_options_set_unlimited_spill(options),
        GrafeoStatus::Ok
    );
    let db = grafeo_open_with_options(std::ptr::null(), options);
    assert!(!db.is_null());
    assert_eq!(usage(db)["spill"]["limit_bytes"], 0);
    grafeo_free_database(db);
    grafeo_options_free(options);
}

#[test]
fn a_zero_memory_limit_is_refused_at_the_setter() {
    let options = grafeo_options_new();
    assert_ne!(
        grafeo_options_set_memory_limit(options, 0),
        GrafeoStatus::Ok
    );
    assert_ne!(
        grafeo_options_set_query_memory_limit(options, 0),
        GrafeoStatus::Ok
    );
    grafeo_options_free(options);
}

#[test]
fn null_options_are_the_defaults() {
    let defaults = grafeo_open_memory();
    let null = grafeo_open_with_options(std::ptr::null(), std::ptr::null());
    assert!(!null.is_null());
    assert_eq!(usage(null)["limit_bytes"], usage(defaults)["limit_bytes"]);
    assert_eq!(
        usage(null)["query_limit_bytes"],
        usage(defaults)["query_limit_bytes"]
    );
    grafeo_free_database(null);
    grafeo_free_database(defaults);
}

#[test]
fn one_options_object_opens_two_databases_alike_and_may_be_freed_after() {
    let options = grafeo_options_new();
    assert_eq!(
        grafeo_options_set_memory_limit(options, 48 * MIB),
        GrafeoStatus::Ok
    );
    let first = grafeo_open_with_options(std::ptr::null(), options);
    let second = grafeo_open_with_options(std::ptr::null(), options);
    grafeo_options_free(options);
    assert_eq!(usage(first)["limit_bytes"], 48 * MIB);
    assert_eq!(usage(second)["limit_bytes"], 48 * MIB);
    let query = CString::new("INSERT (:Person {name: 'Alix'})").unwrap();
    for db in [first, second] {
        let result = grafeo_execute(db, query.as_ptr());
        assert!(
            !result.is_null(),
            "the database works after the options are freed"
        );
        grafeo_free_result(result);
        grafeo_free_database(db);
    }
}

#[test]
fn null_handles_are_errors() {
    assert_eq!(
        grafeo_options_set_memory_limit(std::ptr::null_mut(), MIB),
        GrafeoStatus::ErrorNullPointer
    );
    grafeo_options_free(std::ptr::null_mut());
}

/// The open warnings as JSON: a list of `{code, message, context}`.
fn open_warning_codes(db: *mut GrafeoDatabase) -> Vec<String> {
    let json = crate::database::grafeo_open_warnings(db);
    assert!(!json.is_null());
    // SAFETY: a string the binding returned, freed below.
    let text = unsafe { CStr::from_ptr(json) }
        .to_string_lossy()
        .into_owned();
    crate::database::grafeo_free_string(json);
    let warnings: Vec<serde_json::Value> = serde_json::from_str(&text).unwrap();
    warnings
        .iter()
        .map(|warning| warning["code"].as_str().unwrap_or_default().to_string())
        .collect()
}

/// A file whose data passes the limit it is opened with opens, reads, and says so in its open
/// warnings; opened with room, it says nothing of the kind.
#[test]
fn an_open_over_the_limit_warns_and_reads() {
    let dir = std::env::temp_dir().join(format!("grafeo-c-over-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let path = CString::new(dir.join("over.grafeo").to_string_lossy().into_owned()).unwrap();

    let db = crate::database::grafeo_open(path.as_ptr());
    assert!(!db.is_null());
    // Distinct 1 KiB strings: data that stays past the limit once reloaded (a repeated string
    // and dense integers load back as small compressed columns).
    let fill = CString::new(format!(
        "UNWIND range(1, 20000) AS i INSERT (:Fill {{i: i, text: CAST(i AS STRING) || '{}'}})",
        "x".repeat(1_000)
    ))
    .unwrap();
    let result = grafeo_execute(db, fill.as_ptr());
    assert!(!result.is_null());
    grafeo_free_result(result);
    assert_eq!(crate::database::grafeo_close(db), GrafeoStatus::Ok);
    grafeo_free_database(db);

    let options = grafeo_options_new();
    assert_eq!(
        grafeo_options_set_memory_limit(options, MIB),
        GrafeoStatus::Ok
    );
    let tight = grafeo_open_with_options(path.as_ptr(), options);
    grafeo_options_free(options);
    assert!(!tight.is_null(), "a database over its limit opens");
    assert!(open_warning_codes(tight).contains(&"MEMORY_LIMIT_PASSED".to_string()));
    let read = CString::new("MATCH (f:Fill) RETURN count(f) AS c").unwrap();
    let result = grafeo_execute(tight, read.as_ptr());
    assert!(!result.is_null(), "and reads");
    grafeo_free_result(result);
    assert_eq!(crate::database::grafeo_close(tight), GrafeoStatus::Ok);
    grafeo_free_database(tight);

    let roomy = crate::database::grafeo_open(path.as_ptr());
    assert!(!open_warning_codes(roomy).contains(&"MEMORY_LIMIT_PASSED".to_string()));
    assert_eq!(crate::database::grafeo_close(roomy), GrafeoStatus::Ok);
    grafeo_free_database(roomy);
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn a_clean_open_lists_no_warning() {
    let db = grafeo_open_memory();
    // Automatic memory detection may legitimately record its fallback warning
    // on hosts where detection is unavailable; every other warning is a bug.
    let codes = open_warning_codes(db);
    assert!(
        codes.iter().all(|code| code == "MEMORY_DETECTION_FALLBACK"),
        "{codes:?}"
    );
    grafeo_free_database(db);
}
