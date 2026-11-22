//! The open options: the memory and spill settings a database opens with, in an options object
//! the caller builds and frees (`grafeo_options_new`, the `grafeo_options_set_*` setters,
//! `grafeo_options_free`) and passes to `grafeo_open_with_options`. Unset means the default, with
//! no sentinel values; the object may be freed once the open returns, and one object may open
//! several databases.

use std::os::raw::c_char;

use grafeo_engine::config::Config;
use grafeo_engine::database::GrafeoDB;

use crate::error::{GrafeoStatus, set_error, set_last_error, str_from_ptr};
use crate::types::GrafeoDatabase;

/// The settings a database opens with (opaque to C).
#[derive(Debug, Clone, Default)]
pub struct GrafeoOptions {
    memory_limit: Option<usize>,
    query_memory_limit: Option<usize>,
    path_search_budget: Option<usize>,
    path_expansion_limit: Option<u64>,
    max_query_length: Option<usize>,
    max_identifier_length: Option<usize>,
    snapshot_retention_limit: Option<u64>,
    history_retention_limit: Option<u64>,
    spill_limit: Option<u64>,
    unlimited_spill: bool,
    plan_cache_limit: Option<usize>,
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
        if let Some(bytes) = self.path_search_budget {
            config = config.with_path_search_budget(bytes);
        }
        if let Some(expansions) = self.path_expansion_limit {
            config = config.with_path_expansion_limit(expansions);
        }
        if let Some(bytes) = self.max_query_length {
            config = config.with_max_query_length(bytes);
        }
        if let Some(bytes) = self.max_identifier_length {
            config = config.with_max_identifier_length(bytes);
        }
        if let Some(bytes) = self.snapshot_retention_limit {
            config = config.with_snapshot_retention_limit(bytes);
        }
        if let Some(bytes) = self.history_retention_limit {
            config = config.with_history_retention_limit(bytes);
        }
        if self.unlimited_spill {
            config = config.with_unlimited_spill();
        } else if let Some(bytes) = self.spill_limit {
            config = config.with_spill_limit(bytes);
        }
        if let Some(bytes) = self.plan_cache_limit {
            config = config.with_plan_cache_limit(bytes);
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

/// Sets the memory one path search may hold, in bytes, above zero: everything a variable-length,
/// shortest-path or RDF property path search holds at once (the nodes it visits, its queues and the
/// paths it keeps), which counts against the query memory limit too. The default is a quarter
/// of the memory limit, at most 256 MiB; never more than the memory limit. A search over it fails
/// with GRAFEO-S001.
#[unsafe(no_mangle)]
pub extern "C" fn grafeo_options_set_path_search_budget(
    options: *mut GrafeoOptions,
    bytes: u64,
) -> GrafeoStatus {
    match options_mut(options).and_then(|options| {
        options.path_search_budget = Some(positive(bytes, "path_search_budget")?);
        Ok(())
    }) {
        Ok(()) => GrafeoStatus::Ok,
        Err(status) => status,
    }
}

/// Sets how many path expansions one statement's variable-length and shortest-path searches may
/// make together, above zero: each edge they read to extend a path, and each node and edge they
/// copy out of a path into a row. The default is 100 million. A statement over it fails with
/// GRAFEO-Q009.
#[unsafe(no_mangle)]
pub extern "C" fn grafeo_options_set_path_expansion_limit(
    options: *mut GrafeoOptions,
    expansions: u64,
) -> GrafeoStatus {
    match options_mut(options).and_then(|options| {
        if expansions == 0 {
            set_last_error("path_expansion_limit must be greater than zero");
            return Err(GrafeoStatus::ErrorDatabase);
        }
        options.path_expansion_limit = Some(expansions);
        Ok(())
    }) {
        Ok(()) => GrafeoStatus::Ok,
        Err(status) => status,
    }
}

/// Sets how many bytes a statement's text may have, above zero; the values passed as parameters
/// do not count. The default is 16 MiB. A longer statement fails with GRAFEO-Q002 before it is
/// read.
#[unsafe(no_mangle)]
pub extern "C" fn grafeo_options_set_max_query_length(
    options: *mut GrafeoOptions,
    bytes: u64,
) -> GrafeoStatus {
    match options_mut(options).and_then(|options| {
        options.max_query_length = Some(positive(bytes, "max_query_length")?);
        Ok(())
    }) {
        Ok(()) => GrafeoStatus::Ok,
        Err(status) => status,
    }
}

/// Sets how many bytes one name in a statement may have, as written between its delimiters,
/// above zero: an identifier, a parameter name, a SPARQL variable or prefixed name. String
/// literals and IRIs are not names. The default is 16 KiB. A statement with a longer name fails
/// with GRAFEO-Q002.
#[unsafe(no_mangle)]
pub extern "C" fn grafeo_options_set_max_identifier_length(
    options: *mut GrafeoOptions,
    bytes: u64,
) -> GrafeoStatus {
    match options_mut(options).and_then(|options| {
        options.max_identifier_length = Some(positive(bytes, "max_identifier_length")?);
        Ok(())
    }) {
        Ok(()) => GrafeoStatus::Ok,
        Err(status) => status,
    }
}

/// Sets how many bytes of replaced data an open transaction's snapshot may keep: what the commits
/// since it began replaced that it can still read. Past it the transaction's next call fails with
/// the retryable GRAFEO-T012 and rolls it back. The default is a quarter of the memory limit, at
/// least 1 MiB; below 1 MiB `grafeo_open_with_options` fails.
#[unsafe(no_mangle)]
pub extern "C" fn grafeo_options_set_snapshot_retention_limit(
    options: *mut GrafeoOptions,
    bytes: u64,
) -> GrafeoStatus {
    match options_mut(options) {
        Ok(options) => {
            options.snapshot_retention_limit = Some(bytes);
            GrafeoStatus::Ok
        }
        Err(status) => status,
    }
}

/// How many bytes of replaced data the history window may keep: once the commits since its start
/// replaced more, the start moves forward, whichever of time or bytes comes first, and a read
/// before it is refused. No limit by default; zero fails at the open. Not stored with the
/// database: pass it at every open.
#[unsafe(no_mangle)]
pub extern "C" fn grafeo_options_set_history_retention_limit(
    options: *mut GrafeoOptions,
    bytes: u64,
) -> GrafeoStatus {
    match options_mut(options) {
        Ok(options) => {
            options.history_retention_limit = Some(bytes);
            GrafeoStatus::Ok
        }
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

/// Sets how many bytes of plans the database keeps for repeated statements, by a deep estimate of
/// what each plan holds; zero keeps none, so every statement plans. The default is a sixty-fourth
/// of the memory limit, from 4 MiB to 64 MiB; never more than the memory limit. The report's
/// `caches.plan_cache_limit_bytes` reads it back.
#[unsafe(no_mangle)]
pub extern "C" fn grafeo_options_set_plan_cache_limit(
    options: *mut GrafeoOptions,
    bytes: u64,
) -> GrafeoStatus {
    match options_mut(options) {
        Ok(options) => {
            // Past the platform's size, the bound is the memory limit anyway.
            options.plan_cache_limit = Some(usize::try_from(bytes).unwrap_or(usize::MAX));
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
            inner: crate::types::SharedDb::new(db),
        })),
        Err(error) => {
            set_error(&error);
            std::ptr::null_mut()
        }
    }
}
