//! WASM utility helpers.

/// Sets up better panic messages in the browser console.
///
/// Call this once during initialization. Subsequent calls are no-ops.
pub fn set_panic_hook() {
    console_error_panic_hook::set_once();
}
