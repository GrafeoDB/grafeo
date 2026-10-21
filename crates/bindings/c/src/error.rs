//! Thread-local error handling for the C FFI layer.
//!
//! Follows the same pattern as SQLite and libgit2: functions return a status
//! code and store a detailed error message in thread-local storage.

use std::cell::RefCell;
use std::ffi::{CStr, CString};
use std::os::raw::c_char;

use grafeo_common::utils::GqlStatus;

/// Status codes returned by C FFI functions.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GrafeoStatus {
    Ok = 0,
    ErrorDatabase = 1,
    ErrorQuery = 2,
    ErrorTransaction = 3,
    ErrorStorage = 4,
    ErrorIo = 5,
    ErrorSerialization = 6,
    ErrorInternal = 7,
    ErrorNullPointer = 8,
    ErrorInvalidUtf8 = 9,
}

impl From<&grafeo_common::utils::error::Error> for GrafeoStatus {
    fn from(err: &grafeo_common::utils::error::Error) -> Self {
        use grafeo_bindings_common::error::{ErrorCategory, classify_error};
        match classify_error(err) {
            ErrorCategory::Query => GrafeoStatus::ErrorQuery,
            ErrorCategory::Transaction => GrafeoStatus::ErrorTransaction,
            // Not the transaction status, which wrappers retry on conflicts:
            // no retry opens a closed database. No status of its own, so the
            // C ABI stays as it is; the message names it (GRAFEO-T007).
            ErrorCategory::DatabaseClosed | ErrorCategory::Database => GrafeoStatus::ErrorDatabase,
            // The message starts with GRAFEO-S002.
            ErrorCategory::Storage | ErrorCategory::Corruption => GrafeoStatus::ErrorStorage,
            ErrorCategory::Io => GrafeoStatus::ErrorIo,
            ErrorCategory::Serialization => GrafeoStatus::ErrorSerialization,
            ErrorCategory::Internal => GrafeoStatus::ErrorInternal,
        }
    }
}

thread_local! {
    static LAST_ERROR: RefCell<Option<CString>> = const { RefCell::new(None) };
    static LAST_GQL_STATUS: RefCell<Option<CString>> = const { RefCell::new(None) };
}

/// Store an error message for later retrieval via [`grafeo_last_error`], for
/// an error the binding raises itself: its GQLSTATUS is `22000` (data
/// exception), as the database reports an internal error.
pub fn set_last_error(msg: &str) {
    store(msg, GqlStatus::DATA_EXCEPTION);
}

/// Store an error from a [`grafeo_common::utils::error::Error`] with its
/// GQLSTATUS and return the corresponding status code.
pub fn set_error(err: &grafeo_common::utils::error::Error) -> GrafeoStatus {
    store(&err.to_string(), GqlStatus::from(err));
    GrafeoStatus::from(err)
}

fn store(message: &str, gql_status: GqlStatus) {
    LAST_ERROR.with(|cell| {
        *cell.borrow_mut() = CString::new(message).ok();
    });
    LAST_GQL_STATUS.with(|cell| {
        *cell.borrow_mut() = CString::new(gql_status.as_str()).ok();
    });
}

/// Returns the last error message, or null if no error.
///
/// The returned pointer is valid until the next FFI call on this thread.
/// The caller must NOT free this pointer.
#[unsafe(no_mangle)]
pub extern "C" fn grafeo_last_error() -> *const c_char {
    LAST_ERROR.with(|cell| {
        cell.borrow()
            .as_ref()
            .map_or(std::ptr::null(), |s| s.as_ptr())
    })
}

/// Returns the GQLSTATUS of the last error, five characters such as
/// `"22012"` (division by zero) or `"42001"` (syntax error), or null if no
/// error.
///
/// The returned pointer is valid until the next FFI call on this thread.
/// The caller must NOT free this pointer.
#[unsafe(no_mangle)]
pub extern "C" fn grafeo_last_error_gql_status() -> *const c_char {
    LAST_GQL_STATUS.with(|cell| {
        cell.borrow()
            .as_ref()
            .map_or(std::ptr::null(), |s| s.as_ptr())
    })
}

/// Clears the last error and its GQLSTATUS.
#[unsafe(no_mangle)]
pub extern "C" fn grafeo_clear_error() {
    LAST_ERROR.with(|cell| {
        *cell.borrow_mut() = None;
    });
    LAST_GQL_STATUS.with(|cell| {
        *cell.borrow_mut() = None;
    });
}

/// Extract a `&str` from a C string pointer, returning an error status if null
/// or invalid UTF-8.
pub fn str_from_ptr<'a>(ptr: *const c_char) -> Result<&'a str, GrafeoStatus> {
    if ptr.is_null() {
        set_last_error("Null string pointer");
        return Err(GrafeoStatus::ErrorNullPointer);
    }
    // SAFETY: Caller guarantees ptr is a valid, null-terminated C string.
    unsafe { CStr::from_ptr(ptr) }.to_str().map_err(|_| {
        set_last_error("Invalid UTF-8 in string");
        GrafeoStatus::ErrorInvalidUtf8
    })
}

#[cfg(test)]
mod tests {
    use grafeo_common::utils::error::{Error, TransactionError};

    use super::GrafeoStatus;

    /// A closed database or a commit that did not complete is no transaction
    /// error: wrappers (C#, Dart) raise a transaction exception for that
    /// status, which conflict-retry loops retry forever. Both get the generic
    /// database status, and the message names them.
    #[test]
    fn errors_a_retry_cannot_fix_get_the_database_status() {
        for error in [
            Error::Transaction(TransactionError::DatabaseClosed),
            Error::Transaction(TransactionError::IncompleteCommit),
            Error::Transaction(TransactionError::CommitOutcomeUnknown("fsync".into())),
            Error::Transaction(TransactionError::LogUnavailable("poisoned".into())),
        ] {
            assert_eq!(
                GrafeoStatus::from(&error),
                GrafeoStatus::ErrorDatabase,
                "{error}"
            );
        }
        assert_eq!(
            GrafeoStatus::from(&Error::Transaction(TransactionError::Conflict)),
            GrafeoStatus::ErrorTransaction,
            "a conflict stays a transaction error"
        );
        assert_eq!(
            GrafeoStatus::from(&Error::Transaction(TransactionError::LogWriteFailed(
                "disk full".into()
            ))),
            GrafeoStatus::ErrorTransaction,
            "a commit that was rolled back can be retried"
        );
    }

    /// The GQLSTATUS of the last error: the status of a database error
    /// (22012, 22003, 42001, 42002, 42000), 22000 for one the binding raises
    /// itself (a null pointer), and none once the error is cleared (#639).
    #[test]
    fn the_last_error_carries_its_gql_status() {
        use std::ffi::{CStr, CString};

        use super::{grafeo_clear_error, grafeo_last_error, grafeo_last_error_gql_status};
        use crate::database::{grafeo_execute, grafeo_free_database, grafeo_open_memory};

        let status = || {
            let pointer = grafeo_last_error_gql_status();
            assert!(!pointer.is_null(), "no status after an error");
            // SAFETY: the status lives until the next call on this thread.
            unsafe { CStr::from_ptr(pointer) }
                .to_str()
                .unwrap()
                .to_string()
        };
        let message = || {
            // SAFETY: the message lives until the next call on this thread.
            unsafe { CStr::from_ptr(grafeo_last_error()) }
                .to_string_lossy()
                .into_owned()
        };
        let db = grafeo_open_memory();
        for (query, expected, text) in [
            ("RETURN 1 / 0 AS x", "22012", "division by zero: 1 / 0"),
            (
                "RETURN 9223372036854775807 + 1 AS x",
                "22003",
                "numeric value out of range",
            ),
            ("MATCH (n RETURN n", "42001", "GRAFEO-Q001"),
            ("MATCH (n) RETURN m", "42002", "Undefined variable 'm'"),
            ("RETURN abs(1, 2) AS v", "42000", "takes 1 argument, got 2"),
        ] {
            let query_text = CString::new(query).unwrap();
            assert!(grafeo_execute(db, query_text.as_ptr()).is_null(), "{query}");
            assert_eq!(status(), expected, "{query}");
            assert!(message().contains(text), "{query}: {}", message());
        }
        assert!(grafeo_execute(db, std::ptr::null()).is_null());
        assert_eq!(status(), "22000", "an error of the binding itself");
        grafeo_clear_error();
        assert!(grafeo_last_error_gql_status().is_null());
        grafeo_free_database(db);
    }
}
