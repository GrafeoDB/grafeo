//! The memory limit in the browser (#751): `memoryLimit` in the constructor's options, and in
//! the options `importSnapshot` and `importSnapshotSigned` take, so a reloaded database keeps
//! its limit. Options given at an import win over what the snapshot carries; an import without
//! options takes the 768 MiB platform default. A zero limit is refused.
//!
//! Run via: `wasm-pack test crates/bindings/wasm --node`

#![cfg(target_arch = "wasm32")]

use wasm_bindgen_test::*;

use grafeo_wasm::Database;

const MIB: f64 = 1024.0 * 1024.0;

/// The 768 MiB default on wasm32.
const DEFAULT: f64 = 805_306_368.0;

fn options(memory_limit: f64) -> js_sys::Object {
    let options = js_sys::Object::new();
    js_sys::Reflect::set(&options, &"memoryLimit".into(), &memory_limit.into()).unwrap();
    options
}

/// The limit the database reports.
fn limit(db: &Database) -> Option<f64> {
    let usage = db.memory_usage().unwrap();
    js_sys::Reflect::get(&usage, &"limit_bytes".into())
        .unwrap()
        .as_f64()
}

fn a_snapshot() -> Vec<u8> {
    let db = Database::new().unwrap();
    db.execute("INSERT (:Person {name: 'Alix'})").unwrap();
    db.export_snapshot().unwrap()
}

#[wasm_bindgen_test]
fn the_constructor_takes_a_memory_limit() {
    let db = Database::create(Some(options(64.0 * MIB))).unwrap();
    assert_eq!(limit(&db), Some(64.0 * MIB));
    assert_eq!(limit(&Database::new().unwrap()), Some(DEFAULT));
}

#[wasm_bindgen_test]
fn an_import_with_a_memory_limit_reports_it() {
    let snapshot = a_snapshot();
    let db = Database::import_snapshot(&snapshot, Some(options(48.0 * MIB))).unwrap();
    assert_eq!(limit(&db), Some(48.0 * MIB));
    assert_eq!(db.node_count().unwrap(), 1, "the data came along");
}

#[wasm_bindgen_test]
fn an_import_without_options_reports_the_default() {
    let snapshot = a_snapshot();
    let db = Database::import_snapshot(&snapshot, None).unwrap();
    assert_eq!(limit(&db), Some(DEFAULT));
}

/// The limit of the database that exported the snapshot does not ride along: the import's
/// options decide, or the default.
#[wasm_bindgen_test]
fn an_import_takes_its_own_options_over_the_exporting_databases() {
    let source = Database::create(Some(options(32.0 * MIB))).unwrap();
    source.execute("INSERT (:Person {name: 'Gus'})").unwrap();
    let snapshot = source.export_snapshot().unwrap();
    let db = Database::import_snapshot(&snapshot, Some(options(96.0 * MIB))).unwrap();
    assert_eq!(limit(&db), Some(96.0 * MIB));
    let db = Database::import_snapshot(&snapshot, None).unwrap();
    assert_eq!(limit(&db), Some(DEFAULT));
}

#[wasm_bindgen_test]
fn a_signed_import_takes_the_options_too() {
    let key = b"a key for the tests";
    let db = Database::new().unwrap();
    db.execute("INSERT (:Person {name: 'Vincent'})").unwrap();
    let signed = db.export_snapshot_signed(key).unwrap();
    let restored =
        Database::import_snapshot_signed(&signed, key, Some(options(40.0 * MIB))).unwrap();
    assert_eq!(limit(&restored), Some(40.0 * MIB));
    let restored = Database::import_snapshot_signed(&signed, key, None).unwrap();
    assert_eq!(limit(&restored), Some(DEFAULT));
}

#[wasm_bindgen_test]
fn a_zero_memory_limit_is_refused() {
    assert!(Database::create(Some(options(0.0))).is_err());
    assert!(Database::import_snapshot(&a_snapshot(), Some(options(0.0))).is_err());
}
