//! The query resource limits in the browser (#492): `pathSearchBudget`, `pathExpansionLimit`,
//! `maxQueryLength` and `maxIdentifierLength` in the constructor's options. Zero is refused;
//! each setting binds, with its error code and the setting named in the message. The snapshot retention limit, `snapshotRetentionLimit`: below 1 MiB is refused, 1 MiB
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

fn star(key: &str, value: f64) -> Database {
    let db = Database::create(Some(options(key, value))).unwrap();
    db.execute(STAR).unwrap();
    db
}

fn assert_refused(db: &Database, query: &str, code: &str, setting: &str) {
    let error = format!("{:?}", db.execute(query).expect_err(query));
    assert!(error.contains(code), "{query}: {error}");
    assert!(error.contains(setting), "{query}: {error}");
}

#[wasm_bindgen_test]
fn zero_is_refused() {
    for key in [
        "pathSearchBudget",
        "pathExpansionLimit",
        "maxQueryLength",
        "maxIdentifierLength",
    ] {
        assert!(Database::create(Some(options(key, 0.0))).is_err(), "{key}");
    }
}

#[wasm_bindgen_test]
fn each_setting_binds() {
    assert_refused(
        &star("pathExpansionLimit", 4.0),
        PATHS,
        "GRAFEO-Q009",
        "pathExpansionLimit",
    );
    star("pathExpansionLimit", 5.0).execute(PATHS).unwrap();
    assert_refused(
        &star("pathSearchBudget", 1.0),
        PATHS,
        "GRAFEO-S001",
        "pathSearchBudget",
    );
    let db = star("maxQueryLength", 200.0);
    db.execute(PATHS).unwrap();
    assert_refused(
        &db,
        &format!("{PATHS}{}", " ".repeat(200)),
        "GRAFEO-Q002",
        "maxQueryLength",
    );
    let db = star("maxIdentifierLength", 8.0);
    db.execute("MATCH (abcdefgh:Star) RETURN abcdefgh.n")
        .unwrap();
    assert_refused(
        &db,
        "MATCH (abcdefghi:Star) RETURN 1",
        "GRAFEO-Q002",
        "maxIdentifierLength",
    );
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
