//! `=` finds the same rows with and without a property index, whatever plan
//! runs it (#535).
//!
//! The filter's `=` is lenient: a number equals a string it parses as
//! (`'042' = 42`), numbers compare within `f64::EPSILON` (`0.1 + 0.2 = 0.3`)
//! and lists compare element by element. A property index finds values by
//! exact value, so every plan that looks a key up in it (the lookup of a
//! constant key, an `IN` list, the seek of a key from the row, and `id()`)
//! has to find every value `=` finds equal. These tests run every pair of
//! value kinds through each plan, with and without an index, and compare the
//! rows with those of the filter itself.
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test equality_with_index
//! ```

use std::collections::{BTreeSet, HashMap};

use grafeo_common::types::Value;
use grafeo_engine::GrafeoDB;

/// One literal of each kind and of each spelling `=` treats specially, as GQL
/// writes it. Each is stored as the `p` of a `:Doc` and looked up as a key.
const LITERALS: &[&str] = &[
    "42",
    "-42",
    "42.0",
    "42.5",
    "'42'",
    "'-42'",
    "'042'",
    "'+42'",
    "'42.0'",
    "'4.2e1'",
    "'abc'",
    "0.1 + 0.2",
    "0.3",
    "'0.3'",
    "0",
    "0.0",
    "-0.0",
    "1",
    "1.0",
    "0.9999999999999999",
    "1.9999999999999998",
    "2",
    "'2'",
    "true",
    "'true'",
    "9007199254740993",
    "9007199254740992.0",
    "'9007199254740993'",
    "'NaN'",
    "'inf'",
    "[42]",
    "['42']",
    "[0.1 + 0.2]",
    "{a: 42}",
    "{a: '42'}",
    "DATE '2024-03-15'",
    "ZONED DATETIME '2024-03-15T10:00:00+01:00'",
    "ZONED DATETIME '2024-03-15T09:00:00Z'",
];

/// A database holding one `:Doc {k: i, p: LITERALS[i]}` per literal, one
/// `:Other` node, and an index on `p` when `indexed`.
fn docs(indexed: bool) -> GrafeoDB {
    let db = GrafeoDB::new_in_memory();
    for (k, literal) in LITERALS.iter().enumerate() {
        db.execute(&format!("INSERT (:Doc {{k: {k}, p: {literal}}})"))
            .unwrap_or_else(|e| panic!("store {literal}: {e}"));
    }
    db.execute("INSERT (:Other {k: -1})").unwrap();
    if indexed {
        db.create_property_index("p").unwrap();
    }
    db
}

/// The `k` of the rows of `query`, which returns `k`.
fn keys(db: &GrafeoDB, query: &str, params: Option<HashMap<String, Value>>) -> BTreeSet<i64> {
    let result = match params {
        Some(params) => db.execute_with_params(query, params),
        None => db.execute(query),
    }
    .unwrap_or_else(|e| panic!("{query}: {e}"));
    result
        .rows()
        .iter()
        .map(|row| match &row[0] {
            Value::Int64(k) => *k,
            other => panic!("{query}: k is {other:?}"),
        })
        .collect()
}

/// The value a literal evaluates to.
fn value_of(db: &GrafeoDB, literal: &str) -> Value {
    db.execute(&format!("RETURN {literal} AS v"))
        .unwrap()
        .rows()[0][0]
        .clone()
}

/// Every plan that runs `d.p = key` for a `:Doc` `d`, as (name, query,
/// parameters).
fn plans(
    literal: &str,
    key: &Value,
) -> Vec<(&'static str, String, Option<HashMap<String, Value>>)> {
    let param = HashMap::from([("key".to_string(), key.clone())]);
    vec![
        (
            "constant key",
            format!("MATCH (d:Doc) WHERE d.p = {literal} RETURN d.k"),
            None,
        ),
        (
            "constant key, no label",
            format!("MATCH (d) WHERE d.p = {literal} RETURN d.k"),
            None,
        ),
        (
            "IN list",
            format!("MATCH (d:Doc) WHERE d.p IN [{literal}] RETURN d.k"),
            None,
        ),
        (
            "parameter",
            "MATCH (d:Doc) WHERE d.p = $key RETURN d.k".to_string(),
            Some(param),
        ),
        (
            "key from the row",
            format!("UNWIND [{literal}] AS key MATCH (d:Doc) WHERE d.p = key RETURN d.k"),
            None,
        ),
        (
            "constant key after a clause",
            format!("MATCH (m:Other) MATCH (d:Doc) WHERE d.p = {literal} RETURN d.k"),
            None,
        ),
    ]
}

/// The rows the filter's own `=` finds: `(d.p = key) = true` is no
/// conjunct a lookup can take, so it runs as a filter over a scan.
fn filtered(db: &GrafeoDB, literal: &str) -> BTreeSet<i64> {
    keys(
        db,
        &format!("MATCH (d:Doc) WHERE (d.p = {literal}) = true RETURN d.k"),
        None,
    )
}

#[test]
fn equality_finds_the_same_rows_with_and_without_an_index_in_every_plan() {
    let plain = docs(false);
    let indexed = docs(true);
    let mut differences = Vec::new();
    for literal in LITERALS {
        let key = value_of(&plain, literal);
        let expected = filtered(&plain, literal);
        assert_eq!(
            expected,
            filtered(&indexed, literal),
            "{literal}: the filter itself does not depend on the index"
        );
        for (plan, query, params) in plans(literal, &key) {
            for (db, index) in [(&plain, "without an index"), (&indexed, "with an index")] {
                let found = keys(db, &query, params.clone());
                if found != expected {
                    differences.push(format!(
                        "{literal} ({plan}, {index}): {found:?}, the filter finds {expected:?}"
                    ));
                }
            }
        }
    }
    assert!(
        differences.is_empty(),
        "`=` found other rows than the filter:\n{}",
        differences.join("\n")
    );
}

#[test]
fn the_reported_rows_are_found_with_an_index() {
    // The reproduction of #535: '42.0' and 0.1 + 0.2, keys from the row.
    for indexed in [false, true] {
        let db = GrafeoDB::new_in_memory();
        db.execute("INSERT (:Doc {k: 3, p: '42.0'}), (:Doc {k: 19, p: 0.1 + 0.2})")
            .unwrap();
        if indexed {
            db.create_property_index("p").unwrap();
        }
        let found = keys(
            &db,
            "UNWIND [42.0, 0.3] AS key MATCH (d:Doc) WHERE d.p = key RETURN d.k",
            None,
        );
        assert_eq!(found, BTreeSet::from([3, 19]), "indexed: {indexed}");
    }
}

#[test]
fn id_finds_the_node_its_key_equals() {
    let db = GrafeoDB::new_in_memory();
    db.execute("INSERT (:N {k: 0}), (:N {k: 1}), (:N {k: 2}), (:N {k: 3})")
        .unwrap();
    let ids = keys(&db, "MATCH (n:N) RETURN id(n) AS k", None);
    let third = *ids.iter().nth(2).unwrap();
    let k_of = |key: &str| -> BTreeSet<i64> {
        let seek = keys(
            &db,
            &format!("MATCH (n:N) WHERE id(n) = {key} RETURN n.k"),
            None,
        );
        let filter = keys(
            &db,
            &format!("MATCH (n:N) WHERE (id(n) = {key}) = true RETURN n.k"),
            None,
        );
        assert_eq!(
            seek, filter,
            "id(n) = {key}: the seek finds what the filter finds"
        );
        seek
    };
    assert_eq!(k_of(&third.to_string()), BTreeSet::from([2]));
    assert_eq!(k_of(&format!("'{third}'")), BTreeSet::from([2]));
    assert_eq!(k_of(&format!("{third}.0")), BTreeSet::from([2]));
    assert_eq!(k_of(&format!("'{third}x'")), BTreeSet::new());
    assert_eq!(k_of("NULL"), BTreeSet::new());
}
