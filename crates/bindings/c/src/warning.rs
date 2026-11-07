//! Query warnings over the C API: a count and the code, kind and message of each, on a result
//! and on a stream. Strings belong to the result or the stream.

use std::ffi::CString;
use std::os::raw::c_char;

use grafeo_bindings_common::warning::Warning;

use crate::stream::GrafeoStream;
use crate::types::GrafeoResult;

/// A warning as C reads it, its strings owned with it.
pub(crate) struct CWarning {
    code: CString,
    kind: CString,
    message: CString,
}

impl From<&Warning> for CWarning {
    fn from(warning: &Warning) -> Self {
        // A NUL cannot end a C string early: the registry's texts have none, and one is
        // replaced if it ever came.
        let text = |text: &str| CString::new(text.replace('\0', " ")).unwrap_or_default();
        Self {
            code: text(&warning.code),
            kind: text(warning.kind),
            message: text(&warning.message),
        }
    }
}

/// `warnings` with their strings for C.
pub(crate) fn c_warnings(warnings: &[Warning]) -> Vec<CWarning> {
    warnings.iter().map(CWarning::from).collect()
}

/// Which string of a warning an accessor reads.
#[derive(Clone, Copy)]
enum Part {
    Code,
    Kind,
    Message,
}

/// The `part` of the `index`th of `warnings`, or NULL past the end.
fn part(warnings: &[CWarning], index: usize, part: Part) -> *const c_char {
    warnings.get(index).map_or(std::ptr::null(), |warning| {
        match part {
            Part::Code => &warning.code,
            Part::Kind => &warning.kind,
            Part::Message => &warning.message,
        }
        .as_ptr()
    })
}

/// The `part` of a result's `index`th warning.
fn result_part(result: *const GrafeoResult, index: usize, which: Part) -> *const c_char {
    if result.is_null() {
        return std::ptr::null();
    }
    // SAFETY: Caller guarantees a valid pointer from grafeo_execute*.
    part(&unsafe { &*result }.warnings, index, which)
}

/// The number of warnings the statement raised (0 for NULL).
#[unsafe(no_mangle)]
pub extern "C" fn grafeo_result_warning_count(result: *const GrafeoResult) -> usize {
    if result.is_null() {
        return 0;
    }
    // SAFETY: Caller guarantees a valid pointer from grafeo_execute*.
    unsafe { &*result }.warnings.len()
}

/// The GQLSTATUS code of the `index`th warning (class `01`, for example
/// `01X01`), ordered by code; NULL past the last one. Valid until
/// `grafeo_free_result`.
#[unsafe(no_mangle)]
pub extern "C" fn grafeo_result_warning_code(
    result: *const GrafeoResult,
    index: usize,
) -> *const c_char {
    result_part(result, index, Part::Code)
}

/// The kind of the `index`th warning: `deprecation` or `notice`; NULL past
/// the last one. Valid until `grafeo_free_result`.
#[unsafe(no_mangle)]
pub extern "C" fn grafeo_result_warning_kind(
    result: *const GrafeoResult,
    index: usize,
) -> *const c_char {
    result_part(result, index, Part::Kind)
}

/// The message of the `index`th warning, naming the replacement of a
/// deprecated form; NULL past the last one. Valid until
/// `grafeo_free_result`.
#[unsafe(no_mangle)]
pub extern "C" fn grafeo_result_warning_message(
    result: *const GrafeoResult,
    index: usize,
) -> *const c_char {
    result_part(result, index, Part::Message)
}

/// The number of warnings the stream's statement raised so far, taking
/// them for `grafeo_stream_warning_code` and the other accessors: its
/// translation's from the open, its run's as the rows that raise them are
/// pulled, all of them once the stream is exhausted (0 for NULL).
#[unsafe(no_mangle)]
pub extern "C" fn grafeo_stream_warning_count(stream: *const GrafeoStream) -> usize {
    if stream.is_null() {
        return 0;
    }
    // SAFETY: Caller guarantees stream was produced by grafeo_stream_open.
    unsafe { &*stream }.take_warnings()
}

/// The `part` of a stream's `index`th warning, as the last count took them.
fn stream_part(stream: *const GrafeoStream, index: usize, which: Part) -> *const c_char {
    if stream.is_null() {
        return std::ptr::null();
    }
    // SAFETY: Caller guarantees stream was produced by grafeo_stream_open.
    unsafe { &*stream }.with_warnings(|warnings| part(warnings, index, which))
}

/// The code of the `index`th warning `grafeo_stream_warning_count` took;
/// NULL past the last one. Valid until the next `grafeo_stream_warning_count`
/// or `grafeo_stream_free`.
#[unsafe(no_mangle)]
pub extern "C" fn grafeo_stream_warning_code(
    stream: *const GrafeoStream,
    index: usize,
) -> *const c_char {
    stream_part(stream, index, Part::Code)
}

/// The kind of the `index`th warning `grafeo_stream_warning_count` took;
/// NULL past the last one. Valid until the next `grafeo_stream_warning_count`
/// or `grafeo_stream_free`.
#[unsafe(no_mangle)]
pub extern "C" fn grafeo_stream_warning_kind(
    stream: *const GrafeoStream,
    index: usize,
) -> *const c_char {
    stream_part(stream, index, Part::Kind)
}

/// The message of the `index`th warning `grafeo_stream_warning_count` took;
/// NULL past the last one. Valid until the next `grafeo_stream_warning_count`
/// or `grafeo_stream_free`.
#[unsafe(no_mangle)]
pub extern "C" fn grafeo_stream_warning_message(
    stream: *const GrafeoStream,
    index: usize,
) -> *const c_char {
    stream_part(stream, index, Part::Message)
}
