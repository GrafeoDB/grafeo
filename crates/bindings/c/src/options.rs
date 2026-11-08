//! The open options: the memory and spill settings a database opens with, in an options object
//! the caller builds and frees (`grafeo_options_new`, the `grafeo_options_set_*` setters,
//! `grafeo_options_free`) and passes to `grafeo_open_with_options`. Unset means the default, with
//! no sentinel values; the object may be freed once the open returns, and one object may open
//! several databases.

use std::os::raw::c_char;
use std::sync::Arc;

use parking_lot::RwLock;

use grafeo_engine::config::Config;
use grafeo_engine::database::GrafeoDB;

use crate::error::{GrafeoStatus, set_error, set_last_error, str_from_ptr};
use crate::types::GrafeoDatabase;

/// The settings a database opens with (opaque to C).
#[derive(Debug, Clone, Default)]
pub struct GrafeoOptions {
    memory_limit: Option<usize>,
    query_memory_limit: Option<usize>,
    spill_limit: Option<u64>,
    unlimited_spill: bool,
}

impl GrafeoOptions {
    /// `config` with these settings.
    fn apply(&self, mut config: Config) -> Config {
        if let Some(bytes) = self.memory_limit {
            config = config.with_memory_limit(bytes);
        }
        if let Some(bytes) = self.query_memory_limit {
            config = config.with_query_memory_limit(bytes);
        }
        if self.unlimited_spill {
            config = config.with_unlimited_spill();
        } else if let Some(bytes) = self.spill_limit {
            config = config.with_spill_limit(bytes);
        }
        config
    }
}

/// The options behind `options`, or `ErrorNullPointer` for a null one.
fn options_mut<'a>(options: *mut GrafeoOptions) -> Result<&'a mut GrafeoOptions, GrafeoStatus> {
    if options.is_null() {
        set_last_error("Null options pointer");
        return Err(GrafeoStatus::ErrorNullPointer);
    }
    // SAFETY: the caller passes an object from `grafeo_options_new` not freed yet.
    Ok(unsafe { &mut *options })
}

/// A size in bytes that must be above zero, as a `usize`.
fn positive(bytes: u64, what: &str) -> Result<usize, GrafeoStatus> {
    match usize::try_from(bytes) {
        Ok(bytes) if bytes > 0 => Ok(bytes),
        Ok(_) => {
            set_last_error(&format!("{what} must be greater than zero"));
            Err(GrafeoStatus::ErrorDatabase)
        }
        Err(_) => {
            set_last_error(&format!(
                "{what} of {bytes} bytes does not fit this platform"
            ));
            Err(GrafeoStatus::ErrorDatabase)
        }
    }
}

/// A new options object with every setting at its default. Free it with
/// `grafeo_options_free`.
#[unsafe(no_mangle)]
pub extern "C" fn grafeo_options_new() -> *mut GrafeoOptions {
    Box::into_raw(Box::new(GrafeoOptions::default()))
}

/// Frees an options object; a null pointer is ignored.
#[unsafe(no_mangle)]
pub extern "C" fn grafeo_options_free(options: *mut GrafeoOptions) {
    if !options.is_null() {
        // SAFETY: an object from `grafeo_options_new`, freed once.
        drop(unsafe { Box::from_raw(options) });
    }
}

/// Sets the memory limit in bytes, above zero (the default is 75% of the memory the process may
/// use). A write past it fails with `GRAFEO-S001`.
#[unsafe(no_mangle)]
pub extern "C" fn grafeo_options_set_memory_limit(
    options: *mut GrafeoOptions,
    bytes: u64,
) -> GrafeoStatus {
    match options_mut(options).and_then(|options| {
        options.memory_limit = Some(positive(bytes, "memory_limit")?);
        Ok(())
    }) {
        Ok(()) => GrafeoStatus::Ok,
        Err(status) => status,
    }
}

/// Sets the memory one statement's operators may hold, in bytes, above zero (the default is the
/// memory limit, which it never exceeds).
#[unsafe(no_mangle)]
pub extern "C" fn grafeo_options_set_query_memory_limit(
    options: *mut GrafeoOptions,
    bytes: u64,
) -> GrafeoStatus {
    match options_mut(options).and_then(|options| {
        options.query_memory_limit = Some(positive(bytes, "query_memory_limit")?);
        Ok(())
    }) {
        Ok(()) => GrafeoStatus::Ok,
        Err(status) => status,
    }
}

/// Sets how many bytes may spill to disk; zero means nothing spills (the default is half the
/// spill volume's free space when the database opens). Unlimited spill, once set, wins.
#[unsafe(no_mangle)]
pub extern "C" fn grafeo_options_set_spill_limit(
    options: *mut GrafeoOptions,
    bytes: u64,
) -> GrafeoStatus {
    match options_mut(options) {
        Ok(options) => {
            options.spill_limit = Some(bytes);
            GrafeoStatus::Ok
        }
        Err(status) => status,
    }
}

/// Lets spilling use the spill volume without a limit of its own (it still keeps the room the
/// database needs).
#[unsafe(no_mangle)]
pub extern "C" fn grafeo_options_set_unlimited_spill(options: *mut GrafeoOptions) -> GrafeoStatus {
    match options_mut(options) {
        Ok(options) => {
            options.unlimited_spill = true;
            GrafeoStatus::Ok
        }
        Err(status) => status,
    }
}

/// Opens a database with `options`: in memory when `path` is null, else the file at `path`
/// (created when it does not exist). Null `options` are the defaults. The options object is not
/// kept: free it whenever suits.
///
/// Returns an opaque pointer, or null on error (check `grafeo_last_error()`).
#[unsafe(no_mangle)]
pub extern "C" fn grafeo_open_with_options(
    path: *const c_char,
    options: *const GrafeoOptions,
) -> *mut GrafeoDatabase {
    let config = if path.is_null() {
        Config::in_memory()
    } else {
        match str_from_ptr(path) {
            Ok(path) => Config::persistent(path),
            Err(_) => return std::ptr::null_mut(),
        }
    };
    let config = if options.is_null() {
        config
    } else {
        // SAFETY: the caller passes an object from `grafeo_options_new` not freed yet.
        unsafe { &*options }.apply(config)
    };
    match GrafeoDB::with_config(config) {
        Ok(db) => Box::into_raw(Box::new(GrafeoDatabase {
            inner: Arc::new(RwLock::new(db)),
        })),
        Err(error) => {
            set_error(&error);
            std::ptr::null_mut()
        }
    }
}
