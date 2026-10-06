//! Thread-local error handling for the C FFI layer.
//!
//! Follows the same pattern as SQLite and libgit2: functions return a status
//! code and store a detailed error message in thread-local storage.

use std::cell::RefCell;
use std::ffi::{CStr, CString};
use std::os::raw::c_char;

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
            ErrorCategory::Storage => GrafeoStatus::ErrorStorage,
            ErrorCategory::Io => GrafeoStatus::ErrorIo,
            ErrorCategory::Serialization => GrafeoStatus::ErrorSerialization,
            ErrorCategory::Internal => GrafeoStatus::ErrorInternal,
        }
    }
}

thread_local! {
    static LAST_ERROR: RefCell<Option<CString>> = const { RefCell::new(None) };
}

/// Store an error message for later retrieval via [`grafeo_last_error`].
pub fn set_last_error(msg: &str) {
    LAST_ERROR.with(|cell| {
        *cell.borrow_mut() = CString::new(msg).ok();
    });
}

/// Store an error from a [`grafeo_common::utils::error::Error`] and return
/// the corresponding status code.
pub fn set_error(err: &grafeo_common::utils::error::Error) -> GrafeoStatus {
    set_last_error(&err.to_string());
    GrafeoStatus::from(err)
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

/// Clears the last error.
#[unsafe(no_mangle)]
pub extern "C" fn grafeo_clear_error() {
    LAST_ERROR.with(|cell| {
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
    }
}
