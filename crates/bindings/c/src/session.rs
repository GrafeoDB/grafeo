//! Sessions: `grafeo_session_open(db, flags)` gives a handle whose limits bind every statement
//! run through it, auto-commit ones, its transactions and its streams alike.
//!
//! `GRAFEO_SESSION_READ_ONLY` refuses writes. `GRAFEO_SESSION_NO_EXTERNAL_ACCESS` refuses, at
//! planning and before any file is opened, every statement that reaches outside the database
//! (`LOAD DATA`, `LOAD CSV`, the load procedures, SPARQL `LOAD`). The two are independent: a
//! read-only session keeps file access unless it is turned off. 0 runs as `grafeo_execute` does.
//! A flag bit this library does not know is refused at open, never ignored.

use std::ffi::c_char;
use std::sync::Arc;

use grafeo_engine::auth::{Identity, Role};
use grafeo_engine::database::GrafeoDB;
use parking_lot::{Mutex, RwLock};

use crate::database::{build_result, new_transaction};
use crate::error::{GrafeoStatus, set_error, set_last_error, str_from_ptr};
use crate::stream::GrafeoStream;
use crate::types::{GrafeoDatabase, GrafeoResult, GrafeoTransaction};

/// Refuses writes: inserts, updates, deletes and schema changes.
pub const GRAFEO_SESSION_READ_ONLY: u32 = 1;

/// Refuses every statement that reads or writes outside the database: files, URLs, models.
pub const GRAFEO_SESSION_NO_EXTERNAL_ACCESS: u32 = 2;

/// Every flag this library knows.
const KNOWN_FLAGS: u32 = GRAFEO_SESSION_READ_ONLY | GRAFEO_SESSION_NO_EXTERNAL_ACCESS;

/// Opaque session handle. Created by [`grafeo_session_open`], freed by [`grafeo_free_session`].
pub struct GrafeoSession {
    database: Arc<RwLock<GrafeoDB>>,
    /// What the session may do: each transaction it begins gets the same.
    identity: Identity,
    /// The session's statements run here; `None` once closed.
    session: Mutex<Option<grafeo_engine::session::Session>>,
}

impl GrafeoSession {
    /// Runs `run` on the open session; NULL, with the error set, when it is closed.
    fn with_open<T>(&self, run: impl FnOnce(&grafeo_engine::session::Session) -> *mut T) -> *mut T {
        match self.session.lock().as_ref() {
            Some(session) => run(session),
            None => {
                set_last_error("the session is closed");
                std::ptr::null_mut()
            }
        }
    }
}

/// The session behind `session`, or NULL from the caller with the error set.
macro_rules! session_ref_or_null {
    ($session:expr) => {{
        // SAFETY: the caller passes a pointer from grafeo_session_open, or NULL.
        match unsafe { $session.as_ref() } {
            Some(session) => session,
            None => {
                set_last_error("Null session pointer");
                return std::ptr::null_mut();
            }
        }
    }};
}

/// Opens a session on `db` with `flags` (`GRAFEO_SESSION_*`, or 0 for none). Returns NULL, with
/// the error set, for a NULL database or a flag bit no `GRAFEO_SESSION_*` names. Free it with
/// [`grafeo_free_session`].
#[unsafe(no_mangle)]
pub extern "C" fn grafeo_session_open(db: *mut GrafeoDatabase, flags: u32) -> *mut GrafeoSession {
    // SAFETY: the caller passes a pointer from grafeo_open*, or NULL.
    let Some(db) = (unsafe { db.as_ref() }) else {
        set_last_error("Null database pointer");
        return std::ptr::null_mut();
    };
    let unknown = flags & !KNOWN_FLAGS;
    if unknown != 0 {
        set_last_error(&format!(
            "unknown session flags {unknown:#x}: the flags are GRAFEO_SESSION_READ_ONLY (1) and \
             GRAFEO_SESSION_NO_EXTERNAL_ACCESS (2)"
        ));
        return std::ptr::null_mut();
    }
    // Explicit, so the flag decides, not the role's default.
    let identity = if flags & GRAFEO_SESSION_READ_ONLY == 0 {
        Identity::anonymous()
    } else {
        Identity::new("anonymous", [Role::ReadOnly])
    }
    .with_external_access(flags & GRAFEO_SESSION_NO_EXTERNAL_ACCESS == 0);
    let session = db.inner.read().session_with_identity(identity.clone());
    Box::into_raw(Box::new(GrafeoSession {
        database: Arc::clone(&db.inner),
        identity,
        session: Mutex::new(Some(session)),
    }))
}

/// Executes a GQL statement on the session. Returns a result, or NULL with the error set.
#[unsafe(no_mangle)]
pub extern "C" fn grafeo_session_execute(
    session: *mut GrafeoSession,
    query: *const c_char,
) -> *mut GrafeoResult {
    run(session, query, "gql", std::ptr::null())
}

/// Executes a GQL statement on the session with JSON-encoded parameters (NULL for none).
#[unsafe(no_mangle)]
pub extern "C" fn grafeo_session_execute_with_params(
    session: *mut GrafeoSession,
    query: *const c_char,
    params_json: *const c_char,
) -> *mut GrafeoResult {
    run(session, query, "gql", params_json)
}

/// Executes a statement in the named language (`"gql"`, `"cypher"`, `"sparql"` and the others
/// `grafeo_execute_language` takes) on the session, with JSON-encoded parameters (NULL for none).
#[unsafe(no_mangle)]
pub extern "C" fn grafeo_session_execute_language(
    session: *mut GrafeoSession,
    language: *const c_char,
    query: *const c_char,
    params_json: *const c_char,
) -> *mut GrafeoResult {
    let Ok(language) = str_from_ptr(language) else {
        return std::ptr::null_mut();
    };
    run(session, query, language, params_json)
}

/// What the execute functions share.
fn run(
    session: *mut GrafeoSession,
    query: *const c_char,
    language: &str,
    params_json: *const c_char,
) -> *mut GrafeoResult {
    let session = session_ref_or_null!(session);
    let Ok(query) = str_from_ptr(query) else {
        return std::ptr::null_mut();
    };
    let Ok(params) = crate::types::parse_params(params_json) else {
        return std::ptr::null_mut();
    };
    session.with_open(
        |open| match open.execute_language(query, language, params) {
            Ok(result) => build_result(&result),
            Err(error) => {
                set_error(&error);
                std::ptr::null_mut()
            }
        },
    )
}

/// Begins a transaction with the session's limits (default isolation). Returns a transaction,
/// used and freed as one from `grafeo_begin_transaction`, or NULL with the error set.
#[unsafe(no_mangle)]
pub extern "C" fn grafeo_session_begin_transaction(
    session: *mut GrafeoSession,
) -> *mut GrafeoTransaction {
    let session = session_ref_or_null!(session);
    session.with_open(|_| {
        let mut transaction = session
            .database
            .read()
            .session_with_identity(session.identity.clone());
        match transaction.begin_transaction() {
            Ok(()) => Box::into_raw(Box::new(new_transaction(transaction))),
            Err(error) => {
                set_error(&error);
                std::ptr::null_mut()
            }
        }
    })
}

/// Opens a read-only streaming query on the session, read as one from `grafeo_stream_open`.
/// Returns NULL with the error set.
#[cfg(feature = "gql")]
#[unsafe(no_mangle)]
pub extern "C" fn grafeo_session_stream_open(
    session: *mut GrafeoSession,
    query: *const c_char,
) -> *mut GrafeoStream {
    let session = session_ref_or_null!(session);
    let Ok(query) = str_from_ptr(query) else {
        return std::ptr::null_mut();
    };
    session.with_open(|open| match open.execute_streaming_owned(query) {
        Ok(stream) => crate::stream::stream_handle(&session.database, stream),
        Err(error) => {
            set_error(&error);
            std::ptr::null_mut()
        }
    })
}

/// Closes the session: its statements fail from now on; the database and other sessions are
/// unaffected. Closing a closed session does nothing.
#[unsafe(no_mangle)]
pub extern "C" fn grafeo_session_close(session: *mut GrafeoSession) -> GrafeoStatus {
    // SAFETY: the caller passes a pointer from grafeo_session_open, or NULL.
    let Some(session) = (unsafe { session.as_ref() }) else {
        set_last_error("Null session pointer");
        return GrafeoStatus::ErrorNullPointer;
    };
    drop(session.session.lock().take());
    GrafeoStatus::Ok
}

/// Frees a session handle (closing it first). NULL does nothing.
#[unsafe(no_mangle)]
pub extern "C" fn grafeo_free_session(session: *mut GrafeoSession) {
    if !session.is_null() {
        // SAFETY: the pointer came from Box::into_raw in grafeo_session_open, freed once.
        drop(unsafe { Box::from_raw(session) });
    }
}
