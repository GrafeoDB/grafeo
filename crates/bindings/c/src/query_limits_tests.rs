//! The path search budget through C (#492): `grafeo_options_set_path_search_budget` on the
//! options object. Zero is refused at the setter; the budget binds, with its error naming its
//! code and the setting.

use std::ffi::{CStr, CString};

use crate::database::{
    grafeo_begin_transaction, grafeo_execute, grafeo_free_database, grafeo_free_result,
    grafeo_free_transaction, grafeo_rollback, grafeo_transaction_execute,
};
use crate::error::{GrafeoStatus, grafeo_last_error};
use crate::options::{
    GrafeoOptions, grafeo_open_with_options, grafeo_options_free, grafeo_options_new,
    grafeo_options_set_path_search_budget, grafeo_options_set_snapshot_retention_limit,
};
use crate::types::GrafeoDatabase;

const STAR: &str = "INSERT (c:Star {n: 0}), (c)-[:E]->(:Leaf {n: 1}), (c)-[:E]->(:Leaf {n: 2}), \
                    (c)-[:E]->(:Leaf {n: 3}), (c)-[:E]->(:Leaf {n: 4}), (c)-[:E]->(:Leaf {n: 5})";
const PATHS: &str = "MATCH (s:Star)-[:E*1..2]->(t) RETURN t.n AS n";

/// An in-memory database opened with the setting `set` makes, holding the star.
fn opened(set: impl FnOnce(*mut GrafeoOptions) -> GrafeoStatus) -> *mut GrafeoDatabase {
    let options = grafeo_options_new();
    assert_eq!(set(options), GrafeoStatus::Ok);
    let db = grafeo_open_with_options(std::ptr::null(), options);
    grafeo_options_free(options);
    assert!(!db.is_null());
    assert_eq!(run(db, STAR), Ok(()));
    db
}

/// Runs `query`: `Ok` or the last error's text.
fn run(db: *mut GrafeoDatabase, query: &str) -> Result<(), String> {
    let query = CString::new(query).unwrap();
    let result = grafeo_execute(db, query.as_ptr());
    if result.is_null() {
        // SAFETY: the binding's last error, a C string it owns.
        Err(unsafe { CStr::from_ptr(grafeo_last_error()) }
            .to_string_lossy()
            .into_owned())
    } else {
        grafeo_free_result(result);
        Ok(())
    }
}

#[test]
fn zero_is_refused_at_the_setter() {
    let options = grafeo_options_new();
    assert_ne!(
        grafeo_options_set_path_search_budget(options, 0),
        GrafeoStatus::Ok
    );
    grafeo_options_free(options);
}

#[test]
fn the_path_search_budget_binds() {
    let db = opened(|o| grafeo_options_set_path_search_budget(o, 1));
    let error = run(db, PATHS).expect_err("no path fits in one byte");
    assert!(
        error.contains("GRAFEO-S001") && error.contains("path_search_budget"),
        "{error}"
    );
    grafeo_free_database(db);
}

/// The last error's text.
fn last_error() -> String {
    // SAFETY: the binding's last error, a C string it owns.
    unsafe { CStr::from_ptr(grafeo_last_error()) }
        .to_string_lossy()
        .into_owned()
}

/// The snapshot retention limit reaches the open: below 1 MiB the open fails naming the minimum;
/// at 1 MiB an open transaction that read Alix fails with GRAFEO-T012 once 20 commits replaced
/// 64 KiB each, and rolls back.
#[test]
fn the_snapshot_retention_limit_binds() {
    let options = grafeo_options_new();
    assert_eq!(
        grafeo_options_set_snapshot_retention_limit(options, 512 * 1024),
        GrafeoStatus::Ok
    );
    let refused = grafeo_open_with_options(std::ptr::null(), options);
    grafeo_options_free(options);
    assert!(refused.is_null());
    assert!(last_error().contains("at least 1 MiB"), "{}", last_error());

    let options = grafeo_options_new();
    assert_eq!(
        grafeo_options_set_snapshot_retention_limit(options, 1 << 20),
        GrafeoStatus::Ok
    );
    let db = grafeo_open_with_options(std::ptr::null(), options);
    grafeo_options_free(options);
    assert!(!db.is_null());
    assert_eq!(
        run(db, "INSERT (:Person {name: 'Alix', bio: 'Amsterdam'})"),
        Ok(())
    );
    let tx = grafeo_begin_transaction(db);
    assert!(!tx.is_null());
    let read = CString::new("MATCH (p:Person {name: 'Alix'}) RETURN p.bio AS bio").unwrap();
    let result = grafeo_transaction_execute(tx, read.as_ptr());
    assert!(!result.is_null(), "{}", last_error());
    grafeo_free_result(result);
    for round in 0..20 {
        let bio = format!("{round}{}", "Vincent ".repeat(8 * 1024));
        assert_eq!(
            run(
                db,
                &format!("MATCH (p:Person {{name: 'Alix'}}) SET p.bio = '{bio}'")
            ),
            Ok(())
        );
    }
    assert!(grafeo_transaction_execute(tx, read.as_ptr()).is_null());
    assert!(last_error().contains("GRAFEO-T012"), "{}", last_error());
    assert_eq!(grafeo_rollback(tx), GrafeoStatus::Ok);
    grafeo_free_transaction(tx);
    grafeo_free_database(db);
}
