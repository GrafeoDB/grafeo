//! The path search budget through C (#492): `grafeo_options_set_path_search_budget` on the
//! options object. Zero is refused at the setter; the budget binds, with its error naming its
//! code and the setting.

use std::ffi::{CStr, CString};

use crate::database::{grafeo_execute, grafeo_free_database, grafeo_free_result};
use crate::error::{GrafeoStatus, grafeo_last_error};
use crate::options::{
    GrafeoOptions, grafeo_open_with_options, grafeo_options_free, grafeo_options_new,
    grafeo_options_set_path_search_budget,
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
