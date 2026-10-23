//! C FFI for lazy query result streaming.
//!
//! `GrafeoStream` exposes the Rust `OwnedRowIterator` over a C-compatible API.
//! Callers:
//! 1. open a stream with [`grafeo_stream_open`]
//! 2. read column names once with [`grafeo_stream_columns_json`] (returns a
//!    JSON array; caller frees with `grafeo_free_string`)
//! 3. pull rows with [`grafeo_stream_next_row_json`] (returns JSON objects
//!    via an out-pointer; the caller frees each row string)
//! 4. release the stream with [`grafeo_stream_free`] when done
//!
//! `grafeo_stream_next_row_json` returns [`GrafeoStatus::Ok`] and sets
//! `*out_json` to the row on success, or to NULL on clean exhaustion. Any
//! other status code indicates an error; call [`grafeo_last_error`] for
//! details.

use std::ffi::CString;
use std::os::raw::c_char;
use std::sync::Arc;

use parking_lot::{Mutex, RwLock};

use grafeo_engine::database::GrafeoDB;
use grafeo_engine::{OwnedResultStream, OwnedRowIterator};

use crate::error::{GrafeoStatus, set_error, set_last_error, str_from_ptr};
use crate::types::GrafeoDatabase;

/// Opaque stream handle. Created by [`grafeo_stream_open`]; freed by
/// [`grafeo_stream_free`].
pub struct GrafeoStream {
    /// Keepalive for the underlying stores.
    _database: Arc<RwLock<GrafeoDB>>,
    columns: Vec<String>,
    /// Behind a `Mutex` because `OwnedRowIterator` is not `Sync`. Taking the
    /// mutex also lets `grafeo_stream_next_row_json` be called from different
    /// threads (one at a time) when the stream handle is shared.
    iter: Mutex<Option<OwnedRowIterator>>,
}

/// Opens a read-only streaming query. Returns null on error; call
/// `grafeo_last_error` for details.
#[unsafe(no_mangle)]
pub extern "C" fn grafeo_stream_open(
    db: *mut GrafeoDatabase,
    query: *const c_char,
) -> *mut GrafeoStream {
    if db.is_null() {
        set_last_error("Null database pointer");
        return std::ptr::null_mut();
    }
    // SAFETY: Caller guarantees db is a valid pointer from grafeo_open*.
    let db = unsafe { &*db };
    let Ok(query_str) = str_from_ptr(query) else {
        return std::ptr::null_mut();
    };
    let stream: OwnedResultStream = match db.inner.read().execute_streaming(query_str) {
        Ok(s) => s,
        Err(e) => {
            set_error(&e);
            return std::ptr::null_mut();
        }
    };
    let columns = stream.columns().to_vec();
    let handle = Box::new(GrafeoStream {
        _database: Arc::clone(&db.inner),
        columns,
        iter: Mutex::new(Some(stream.into_row_iter())),
    });
    Box::into_raw(handle)
}

/// Returns the column names as a JSON array string. The caller owns the
/// returned pointer and must free it with `grafeo_free_string`.
#[unsafe(no_mangle)]
pub extern "C" fn grafeo_stream_columns_json(stream: *const GrafeoStream) -> *mut c_char {
    if stream.is_null() {
        set_last_error("Null stream pointer");
        return std::ptr::null_mut();
    }
    // SAFETY: Caller guarantees stream was produced by grafeo_stream_open.
    let s = unsafe { &*stream };
    let json = serde_json::to_string(&s.columns).unwrap_or_else(|_| "[]".to_string());
    CString::new(json).map_or(std::ptr::null_mut(), CString::into_raw)
}

/// Pulls the next row as a JSON object into `*out_json`.
///
/// Returns [`GrafeoStatus::Ok`] on both "row available" and "exhausted":
/// - row available: `*out_json` is a heap-allocated JSON string that the
///   caller must free with `grafeo_free_string`
/// - exhausted: `*out_json` is set to NULL
///
/// Any non-Ok status indicates an error; `*out_json` is null and
/// `grafeo_last_error()` carries the message.
#[unsafe(no_mangle)]
pub extern "C" fn grafeo_stream_next_row_json(
    stream: *mut GrafeoStream,
    out_json: *mut *mut c_char,
) -> GrafeoStatus {
    if out_json.is_null() {
        set_last_error("Null out_json pointer");
        return GrafeoStatus::ErrorNullPointer;
    }
    // SAFETY: caller-owned out-pointer; we write null before any early return.
    unsafe { *out_json = std::ptr::null_mut() };

    if stream.is_null() {
        set_last_error("Null stream pointer");
        return GrafeoStatus::ErrorNullPointer;
    }
    // SAFETY: Caller guarantees stream was produced by grafeo_stream_open.
    let s = unsafe { &*stream };

    let mut guard = s.iter.lock();
    let Some(iter) = guard.as_mut() else {
        // Stream was closed or exhausted earlier; report clean exhaustion.
        return GrafeoStatus::Ok;
    };

    match iter.next() {
        Some(Ok(values)) => {
            let obj = match crate::types::json_object(s.columns.iter().cloned().zip(values.iter()))
            {
                Ok(obj) => obj,
                Err(status) => {
                    *guard = None;
                    return status;
                }
            };
            let json_str = serde_json::to_string(&obj).unwrap_or_else(|_| "{}".to_string());
            let Ok(c) = CString::new(json_str) else {
                set_last_error("Row contained embedded NUL byte; cannot serialize to C string");
                return GrafeoStatus::ErrorSerialization;
            };
            // SAFETY: out_json was null-checked above.
            unsafe { *out_json = c.into_raw() };
            GrafeoStatus::Ok
        }
        Some(Err(err)) => {
            *guard = None;
            set_error(&err)
        }
        None => {
            *guard = None;
            GrafeoStatus::Ok
        }
    }
}

/// Frees a stream handle. Safe to call on null.
#[unsafe(no_mangle)]
pub extern "C" fn grafeo_stream_free(stream: *mut GrafeoStream) {
    if !stream.is_null() {
        // SAFETY: pointer originated from Box::into_raw in grafeo_stream_open.
        drop(unsafe { Box::from_raw(stream) });
    }
}
