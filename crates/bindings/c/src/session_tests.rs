//! `grafeo_session_open(db, flags)`: a session whose flags bound every statement on it.
//! `GRAFEO_SESSION_READ_ONLY` (1) refuses writes, `GRAFEO_SESSION_NO_EXTERNAL_ACCESS` (2) refuses,
//! at planning and before any file is opened, every statement that reaches outside the database
//! (LOAD DATA in each form, Cypher LOAD CSV, the load procedures, SPARQL LOAD), with the
//! permission-denied error (GQLSTATUS class 42) naming external access. 0 is today's
//! `grafeo_execute`; a bit no flag names is refused at open. The bound holds for the session's
//! statements, its transactions and its streams; the database's own calls are unaffected.

use std::ffi::{CStr, CString, c_char};

use crate::database::{
    grafeo_execute, grafeo_free_database, grafeo_free_result, grafeo_free_string,
    grafeo_free_transaction, grafeo_open_memory, grafeo_result_row_count, grafeo_rollback,
    grafeo_transaction_execute,
};
use crate::error::{
    GrafeoStatus, grafeo_clear_error, grafeo_last_error, grafeo_last_error_gql_status,
};
use crate::session::{
    GRAFEO_SESSION_NO_EXTERNAL_ACCESS, GRAFEO_SESSION_READ_ONLY, GrafeoSession,
    grafeo_free_session, grafeo_session_begin_transaction, grafeo_session_close,
    grafeo_session_execute, grafeo_session_execute_language, grafeo_session_execute_with_params,
    grafeo_session_open, grafeo_session_stream_open,
};
use crate::stream::{grafeo_stream_free, grafeo_stream_next_row_json};
use crate::types::{GrafeoDatabase, GrafeoResult};

const NO_ACCESS: u32 = 2;

fn text(pointer: *const c_char) -> String {
    if pointer.is_null() {
        return String::new();
    }
    // SAFETY: the pointer is valid until the next call on this thread.
    unsafe { CStr::from_ptr(pointer) }
        .to_string_lossy()
        .into_owned()
}

/// The last error and its GQLSTATUS.
fn last() -> (String, String) {
    (
        text(grafeo_last_error()),
        text(grafeo_last_error_gql_status()),
    )
}

/// The rows of `result`, or the last error when it is NULL; frees the result.
fn rows(result: *mut GrafeoResult) -> Result<usize, (String, String)> {
    if result.is_null() {
        return Err(last());
    }
    let count = grafeo_result_row_count(result);
    grafeo_free_result(result);
    Ok(count)
}

fn execute(session: *mut GrafeoSession, query: &str) -> Result<usize, (String, String)> {
    grafeo_clear_error();
    let query = CString::new(query).unwrap();
    rows(grafeo_session_execute(session, query.as_ptr()))
}

fn execute_language(
    session: *mut GrafeoSession,
    language: &str,
    query: &str,
) -> Result<usize, (String, String)> {
    grafeo_clear_error();
    let language = CString::new(language).unwrap();
    let query = CString::new(query).unwrap();
    rows(grafeo_session_execute_language(
        session,
        language.as_ptr(),
        query.as_ptr(),
        std::ptr::null(),
    ))
}

/// Streams every row of `query`, or the error from the open or from a row.
fn stream(session: *mut GrafeoSession, query: &str) -> Result<usize, (String, String)> {
    grafeo_clear_error();
    let query = CString::new(query).unwrap();
    let stream = grafeo_session_stream_open(session, query.as_ptr());
    if stream.is_null() {
        return Err(last());
    }
    let mut count = 0;
    let outcome = loop {
        let mut row: *mut c_char = std::ptr::null_mut();
        if grafeo_stream_next_row_json(stream, &raw mut row) != GrafeoStatus::Ok {
            break Err(last());
        }
        if row.is_null() {
            break Ok(count);
        }
        grafeo_free_string(row);
        count += 1;
    };
    grafeo_stream_free(stream);
    outcome
}

/// `result` is the external-access refusal: class 42, naming external access.
fn assert_refused(result: Result<usize, (String, String)>, what: &str) {
    match result {
        Ok(found) => panic!("{what}: ran ({found} rows), want the external-access refusal"),
        Err((message, status)) => assert!(
            message.contains("external access") && status.starts_with("42"),
            "{what}: {status} {message}"
        ),
    }
}

/// The files a test reads, in a directory of its own that is removed with it.
struct Files {
    dir: std::path::PathBuf,
    csv: String,
    missing: String,
    #[cfg(feature = "sparql")]
    turtle: String,
}

impl Drop for Files {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.dir);
    }
}

fn files() -> Files {
    static NEXT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    let dir = std::env::temp_dir().join(format!(
        "grafeo-c-external-{}-{}",
        std::process::id(),
        NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
    ));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let path = |name: &str| dir.join(name).to_string_lossy().replace('\\', "/");
    std::fs::write(dir.join("people.csv"), "id,name\n1,Alix\n2,Gus\n").unwrap();
    #[cfg(feature = "sparql")]
    std::fs::write(
        dir.join("people.ttl"),
        "<urn:alix> <urn:knows> <urn:gus> .\n",
    )
    .unwrap();
    Files {
        csv: path("people.csv"),
        missing: path("nobody-wrote-this.csv"),
        #[cfg(feature = "sparql")]
        turtle: path("people.ttl"),
        dir,
    }
}

fn load_data(path: &str) -> String {
    format!("LOAD DATA FROM '{path}' FORMAT CSV WITH HEADERS AS row RETURN row.name AS name")
}

fn load_procedure(path: &str) -> String {
    format!("CALL grafeo.load_nodes_csv({{path: '{path}', labels: ['Person'], key: 'id'}})")
}

fn file_url(path: &str) -> String {
    format!("file:///{}", path.trim_start_matches('/'))
}

/// Every form, as (name, language, query, whether a session with access answers it).
fn forms(f: &Files) -> Vec<(&'static str, &'static str, String, bool)> {
    let mut all = vec![
        ("GQL LOAD DATA", "gql", load_data(&f.csv), true),
        (
            "GQL LOAD DATA of a missing file",
            "gql",
            load_data(&f.missing),
            false,
        ),
        ("GQL load_nodes_csv", "gql", load_procedure(&f.csv), true),
        // With access the next check is the arguments', so this one only shows the access
        // check comes first.
        (
            "GQL EXPLAIN load_nodes_csv without arguments",
            "gql",
            "EXPLAIN CALL grafeo.load_nodes_csv()".to_string(),
            false,
        ),
        (
            "GQL PROFILE LOAD DATA",
            "gql",
            format!("PROFILE {}", load_data(&f.csv)),
            true,
        ),
    ];
    #[cfg(feature = "cypher")]
    all.push((
        "Cypher LOAD CSV",
        "cypher",
        format!(
            "LOAD CSV WITH HEADERS FROM '{}' AS row RETURN row.name AS name",
            file_url(&f.csv)
        ),
        true,
    ));
    #[cfg(feature = "sql-pgq")]
    all.push((
        "SQL/PGQ load_nodes_csv",
        "sql",
        load_procedure(&f.csv),
        true,
    ));
    #[cfg(feature = "sparql")]
    all.push((
        "SPARQL LOAD (not supported yet, even with access)",
        "sparql",
        format!("LOAD <{}>", file_url(&f.turtle)),
        false,
    ));
    let _ = file_url;
    all
}

fn people(db: *mut GrafeoDatabase) -> usize {
    let query = CString::new("MATCH (p:Person) RETURN p").unwrap();
    rows(grafeo_execute(db, query.as_ptr())).unwrap()
}

#[test]
fn the_flags_have_their_header_values() {
    assert_eq!(GRAFEO_SESSION_READ_ONLY, 1);
    assert_eq!(GRAFEO_SESSION_NO_EXTERNAL_ACCESS, 2);
}

#[test]
fn every_form_is_refused_without_external_access() {
    let f = files();
    let db = grafeo_open_memory();
    let session = grafeo_session_open(db, NO_ACCESS);
    assert!(!session.is_null(), "{:?}", last());
    for (name, language, query, _) in forms(&f) {
        assert_refused(execute_language(session, language, &query), name);
    }
    assert_refused(
        execute(session, &load_data(&f.csv)),
        "grafeo_session_execute",
    );
    assert_refused(stream(session, &load_data(&f.csv)), "the session's stream");
    let query =
        CString::new("CALL grafeo.load_nodes_csv({path: $path, labels: ['Person'], key: 'id'})")
            .unwrap();
    let params = CString::new(format!("{{\"path\": \"{}\"}}", f.csv)).unwrap();
    grafeo_clear_error();
    assert_refused(
        rows(grafeo_session_execute_with_params(
            session,
            query.as_ptr(),
            params.as_ptr(),
        )),
        "with a parameter",
    );
    assert_eq!(people(db), 0, "a refused load changed the graph");
    grafeo_free_session(session);
    grafeo_free_database(db);
}

#[test]
fn every_form_runs_with_access_and_on_a_read_only_session() {
    let f = files();
    for flags in [0, GRAFEO_SESSION_READ_ONLY] {
        for (name, language, query, runs) in forms(&f) {
            if !runs {
                continue;
            }
            // A load procedure writes, so a read-only session may refuse it as a write.
            if flags == GRAFEO_SESSION_READ_ONLY && query.starts_with("CALL") {
                continue;
            }
            let db = grafeo_open_memory();
            let session = grafeo_session_open(db, flags);
            if let Err(error) = execute_language(session, language, &query) {
                panic!("flags {flags}, {name}: {error:?}");
            }
            grafeo_free_session(session);
            grafeo_free_database(db);
        }
    }
}

#[test]
fn a_missing_file_fails_differently_with_access() {
    let f = files();
    let db = grafeo_open_memory();
    let session = grafeo_session_open(db, 0);
    let (message, status) = execute(session, &load_data(&f.missing)).unwrap_err();
    assert!(
        !message.contains("external access") && !status.starts_with("42"),
        "{status} {message}"
    );
    grafeo_free_session(session);
    grafeo_free_database(db);
}

/// Each flag alone and both: READ ONLY refuses writes and keeps file access, NO EXTERNAL
/// ACCESS refuses loads and keeps writes.
#[test]
fn each_flag_bounds_only_its_own_kind() {
    let f = files();
    for flags in [0, 1, 2, 3] {
        let db = grafeo_open_memory();
        let session = grafeo_session_open(db, flags);
        assert!(!session.is_null(), "flags {flags}: {:?}", last());
        let write = execute(session, "INSERT (:Person {name: 'Mia'})");
        if flags & GRAFEO_SESSION_READ_ONLY == 0 {
            assert_eq!(write, Ok(0), "flags {flags}");
            assert_eq!(people(db), 1);
        } else {
            let (message, _) = write.expect_err("a read-only session refuses a write");
            assert!(!message.contains("external access"), "{message}");
            assert_eq!(people(db), 0);
        }
        let load = execute(session, &load_data(&f.csv));
        if flags & GRAFEO_SESSION_NO_EXTERNAL_ACCESS == 0 {
            assert_eq!(load, Ok(2), "flags {flags}");
        } else {
            assert_refused(load, &format!("flags {flags}"));
        }
        grafeo_free_session(session);
        grafeo_free_database(db);
    }
}

/// A bit no flag names is refused at open, not ignored: NULL, with the error set.
#[test]
fn an_unknown_flag_bit_is_refused_at_open() {
    let db = grafeo_open_memory();
    for flags in [4, 8, 1 << 31, 3 | 4, u32::MAX] {
        grafeo_clear_error();
        let session = grafeo_session_open(db, flags);
        assert!(session.is_null(), "flags {flags:#x} opened a session");
        assert!(!last().0.is_empty(), "flags {flags:#x}: no error set");
    }
    assert!(grafeo_session_open(std::ptr::null_mut(), 0).is_null());
    grafeo_free_database(db);
}

#[test]
fn the_transactions_of_the_session_are_bound() {
    let f = files();
    let db = grafeo_open_memory();
    let session = grafeo_session_open(db, NO_ACCESS);
    let tx = grafeo_session_begin_transaction(session);
    assert!(!tx.is_null(), "{:?}", last());
    for query in [load_data(&f.csv), load_procedure(&f.csv)] {
        grafeo_clear_error();
        let query = CString::new(query).unwrap();
        assert_refused(
            rows(grafeo_transaction_execute(tx, query.as_ptr())),
            "in a transaction",
        );
    }
    assert_eq!(grafeo_rollback(tx), GrafeoStatus::Ok);
    grafeo_free_transaction(tx);
    assert_refused(execute(session, &load_data(&f.csv)), "after the rollback");
    assert_eq!(people(db), 0);
    grafeo_free_session(session);
    grafeo_free_database(db);
}

#[test]
fn a_composed_statement_runs_nothing() {
    let f = files();
    let db = grafeo_open_memory();
    let session = grafeo_session_open(db, NO_ACCESS);
    assert_refused(
        execute(
            session,
            &format!(
                "INSERT (:Marker) RETURN 1 AS one NEXT {}",
                load_data(&f.csv)
            ),
        ),
        "composed after a write",
    );
    let markers = CString::new("MATCH (m:Marker) RETURN m").unwrap();
    assert_eq!(rows(grafeo_execute(db, markers.as_ptr())), Ok(0));
    grafeo_free_session(session);
    grafeo_free_database(db);
}

/// The database's own calls are not bound by a restricted session on it, and a closed session
/// refuses further statements while the others go on.
#[test]
fn a_closed_session_refuses_and_the_rest_goes_on() {
    let f = files();
    let db = grafeo_open_memory();
    let closed = grafeo_session_open(db, 0);
    let restricted = grafeo_session_open(db, NO_ACCESS);
    assert_eq!(grafeo_session_close(closed), GrafeoStatus::Ok);
    assert!(execute(closed, "RETURN 1 AS one").is_err());
    assert!(grafeo_session_begin_transaction(closed).is_null());
    assert_eq!(execute(restricted, "RETURN 1 AS one"), Ok(1));
    assert_refused(execute(restricted, &load_data(&f.csv)), "the other session");
    let load = CString::new(load_data(&f.csv)).unwrap();
    assert_eq!(rows(grafeo_execute(db, load.as_ptr())), Ok(2));
    grafeo_free_session(closed);
    grafeo_free_session(restricted);
    grafeo_free_database(db);
}

#[test]
fn a_null_session_is_an_error() {
    let query = CString::new("RETURN 1 AS one").unwrap();
    assert!(grafeo_session_execute(std::ptr::null_mut(), query.as_ptr()).is_null());
    assert!(grafeo_session_begin_transaction(std::ptr::null_mut()).is_null());
    assert_eq!(
        grafeo_session_close(std::ptr::null_mut()),
        GrafeoStatus::ErrorNullPointer
    );
    grafeo_free_session(std::ptr::null_mut());
}

/// A load procedure reaches outside and writes: a read-only session without access is told
/// about external access; once access is granted, the write is what it refuses.
#[test]
fn external_access_is_refused_before_the_write() {
    let f = files();
    let db = grafeo_open_memory();
    let both = grafeo_session_open(
        db,
        GRAFEO_SESSION_READ_ONLY | GRAFEO_SESSION_NO_EXTERNAL_ACCESS,
    );
    assert_refused(
        execute(both, &load_procedure(&f.csv)),
        "read-only without access",
    );
    let read_only = grafeo_session_open(db, GRAFEO_SESSION_READ_ONLY);
    let (message, _) = execute(read_only, &load_procedure(&f.csv))
        .expect_err("a read-only session refuses the load as a write");
    assert!(!message.contains("external access"), "{message}");
    assert_eq!(people(db), 0);
    grafeo_free_session(both);
    grafeo_free_session(read_only);
    grafeo_free_database(db);
}
