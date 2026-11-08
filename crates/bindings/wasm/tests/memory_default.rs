//! The default memory budget in the browser (#751): wasm32 has no memory figure to read, so the
//! platform's fixed figure stands by intent, 1 GiB, and the default budget is 0.75 of it,
//! 768 MiB, as in 0.6.1.
//!
//! Run via: `wasm-pack test crates/bindings/wasm --node`

#![cfg(target_arch = "wasm32")]

use wasm_bindgen_test::*;

use grafeo_common::memory::{
    DEFAULT_MEMORY_FRACTION, MemorySource, WASM_PLATFORM_MEMORY, detect_memory,
};
use grafeo_wasm::Database;

const MIB: usize = 1024 * 1024;

/// Detection on wasm32 gives the platform's figure, by intent and not as a fallback.
#[wasm_bindgen_test]
fn detection_gives_the_platform_figure() {
    let found = detect_memory();
    assert_eq!(found.source, MemorySource::Platform, "{found:?}");
    assert_eq!(found.bytes, WASM_PLATFORM_MEMORY);
    assert_eq!(WASM_PLATFORM_MEMORY, 1024 * MIB);
    assert!((DEFAULT_MEMORY_FRACTION - 0.75).abs() < f64::EPSILON);
    assert_eq!(found.default_budget(), 768 * MIB);
}

/// An open with no option runs under the 768 MiB budget, as the memory report shows.
#[wasm_bindgen_test]
fn an_open_without_options_has_a_768_mib_budget() {
    let db = Database::new().unwrap();
    let usage = db.memory_usage().unwrap();
    let buffer_manager = js_sys::Reflect::get(&usage, &"buffer_manager".into()).unwrap();
    let budget = js_sys::Reflect::get(&buffer_manager, &"budget_bytes".into()).unwrap();
    assert_eq!(budget.as_f64(), Some(805_306_368.0), "768 MiB");
}
