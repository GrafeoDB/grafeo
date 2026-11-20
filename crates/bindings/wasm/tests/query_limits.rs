//! The path search budget in the browser (#492): `pathSearchBudget` in the constructor's
//! options. Zero is refused; the budget binds, with its error code and the setting named in the
//! message. The snapshot retention limit, `snapshotRetentionLimit`: below 1 MiB is refused, 1 MiB
//! opens.
//!
//! Run via: `wasm-pack test crates/bindings/wasm --node`

#![cfg(target_arch = "wasm32")]

use wasm_bindgen_test::*;

use grafeo_wasm::Database;

const STAR: &str = "INSERT (c:Star {n: 0}), (c)-[:E]->(:Leaf {n: 1}), (c)-[:E]->(:Leaf {n: 2}), \
                    (c)-[:E]->(:Leaf {n: 3}), (c)-[:E]->(:Leaf {n: 4}), (c)-[:E]->(:Leaf {n: 5})";
const PATHS: &str = "MATCH (s:Star)-[:E*1..2]->(t) RETURN t.n AS n";

fn options(key: &str, value: f64) -> js_sys::Object {
    let options = js_sys::Object::new();
    js_sys::Reflect::set(&options, &key.into(), &value.into()).unwrap();
    options
}

#[wasm_bindgen_test]
fn zero_is_refused() {
    assert!(Database::create(Some(options("pathSearchBudget", 0.0))).is_err());
}

#[wasm_bindgen_test]
fn the_path_search_budget_binds() {
    let db = Database::create(Some(options("pathSearchBudget", 1.0))).unwrap();
    db.execute(STAR).unwrap();
    let error = format!("{:?}", db.execute(PATHS).expect_err(PATHS));
    assert!(error.contains("GRAFEO-S001"), "{error}");
    assert!(error.contains("pathSearchBudget"), "{error}");
}

#[wasm_bindgen_test]
fn a_snapshot_retention_limit_below_1_mib_is_refused() {
    let refused = Database::create(Some(options("snapshotRetentionLimit", 512.0 * 1024.0)));
    let error = format!("{:?}", refused.err().expect("512 KiB is refused"));
    assert!(error.contains("at least 1 MiB"), "{error}");
    assert!(Database::create(Some(options("snapshotRetentionLimit", 1_048_576.0))).is_ok());
}

#[wasm_bindgen_test]
fn a_history_retention_limit_opens_and_zero_is_refused() {
    assert!(Database::create(Some(options("historyRetentionLimit", 0.0))).is_err());
    assert!(Database::create(Some(options("historyRetentionLimit", 1_048_576.0))).is_ok());
}
