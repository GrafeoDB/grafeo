//! `UNWIND $rows AS r MATCH (n:Label {id: r.id}) SET ...` (the bulk update
//! shape) reads every key before its first write: a write above the join
//! reads its input whole first, so a hash join on `n.id = r.id` gives the
//! rows of reading the clause at a time, and is what runs (no pass over
//! every row and node pair). The rows and the values written:
//!
//! - SET of other properties, and of the key itself (a later row's key
//!   equal to an earlier row's new key still matches the old key);
//! - DETACH DELETE with repeated keys;
//! - CREATE of a node a later row's key names (not matched);
//! - MERGE (pins today's rows) and a scan whose input writes (pins today's
//!   rows);
//! - an `id` index (the index lookup stays);
//! - null and repeated keys; a transaction's own writes before the
//!   statement.
//!
//! In GQL and Cypher.
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test value_join_under_writes
//! ```

#![cfg(all(feature = "gql", feature = "cypher"))]

use std::collections::HashMap;

use grafeo_common::types::Value;
use grafeo_engine::GrafeoDB;

const LANGUAGES: [&str; 2] = ["gql", "cypher"];

/// Methods.
const METHODS: i64 = 300;

/// `METHODS` Graph:Method nodes, `id` 0 to METHODS - 1, `p0: 'old'`, and 10
/// Graph:File nodes with `id` 0 to 9 (a key a Method shares, under another
/// label).
fn database(index: bool) -> GrafeoDB {
    let db = GrafeoDB::new_in_memory();
    if index {
        db.create_property_index("id").unwrap();
    }
    db.execute(&format!(
        "UNWIND range(0, {}) AS i INSERT (:Graph:Method {{id: i, p0: 'old'}})",
        METHODS - 1
    ))
    .unwrap();
    db.execute("UNWIND range(0, 9) AS i INSERT (:Graph:File {id: i, p0: 'old'})")
        .unwrap();
    db
}

fn run(
    db: &GrafeoDB,
    language: &str,
    query: &str,
    params: Option<HashMap<String, Value>>,
) -> Vec<Vec<Value>> {
    db.execute_language(query, language, params)
        .unwrap_or_else(|error| panic!("{language} `{query}`: {error}"))
        .rows()
        .to_vec()
}

fn ints(values: &[i64]) -> Vec<Vec<Value>> {
    values.iter().map(|v| vec![Value::Int64(*v)]).collect()
}

/// `{id, p0}` maps for `ids`, `p0` the text `new-<id>`.
fn update_rows(ids: &[i64]) -> HashMap<String, Value> {
    let rows = ids
        .iter()
        .map(|&id| {
            let mut row = std::collections::BTreeMap::new();
            row.insert(
                "id".into(),
                if id < 0 {
                    Value::Null
                } else {
                    Value::Int64(id)
                },
            );
            row.insert("p0".into(), Value::from(format!("new-{id}").as_str()));
            Value::Map(std::sync::Arc::new(row))
        })
        .collect::<Vec<_>>();
    HashMap::from([("rows".to_string(), Value::List(rows.into()))])
}

/// The `(id, p0)` of every node with `label`, by id.
fn state(db: &GrafeoDB, label: &str) -> Vec<(i64, String)> {
    run(
        db,
        "gql",
        &format!("MATCH (n:{label}) RETURN n.id AS id, n.p0 AS p0 ORDER BY id"),
        None,
    )
    .into_iter()
    .map(|row| match (&row[0], &row[1]) {
        (Value::Int64(id), Value::String(p0)) => (*id, p0.to_string()),
        other => panic!("{other:?}"),
    })
    .collect()
}

const BULK: &str = "UNWIND $rows AS r MATCH (n:Graph:Method {id: r.id}) SET n.p0 = r.p0 \
                    RETURN n.id AS id ORDER BY id";

/// The bench shape: every even id and two missing ones. The matched
/// Methods get their new `p0`, the rest keep `old`, and the Files (same
/// ids, other label) are untouched.
#[test]
fn the_bulk_update_writes_what_it_matches() {
    let ids: Vec<i64> = (0..METHODS)
        .step_by(2)
        .chain([METHODS + 5, METHODS + 7])
        .collect();
    let matched: Vec<i64> = (0..METHODS).step_by(2).collect();
    for language in LANGUAGES {
        let db = database(false);
        assert_eq!(
            run(&db, language, BULK, Some(update_rows(&ids))),
            ints(&matched),
            "{language}"
        );
        let expected: Vec<(i64, String)> = (0..METHODS)
            .map(|id| {
                let p0 = if id % 2 == 0 {
                    format!("new-{id}")
                } else {
                    "old".to_string()
                };
                (id, p0)
            })
            .collect();
        assert_eq!(state(&db, "Method"), expected, "{language}");
        assert!(
            state(&db, "File").iter().all(|(_, p0)| p0 == "old"),
            "{language}: the Files are untouched"
        );
    }
}

/// The bulk update never pairs every row with every Method: no operator in
/// PROFILE produces rows × Methods rows (a hash join, or an index seek).
#[test]
fn the_bulk_update_does_not_pair_every_row_with_every_node() {
    let ids: Vec<i64> = (0..METHODS).collect();
    let pairs = u64::try_from(ids.len() * usize::try_from(METHODS).unwrap()).unwrap();
    for language in LANGUAGES {
        let db = database(false);
        let profile = run(
            &db,
            language,
            &format!("PROFILE {BULK}"),
            Some(update_rows(&ids)),
        );
        let Value::String(text) = &profile[0][0] else {
            panic!("{language}: {profile:?}");
        };
        let widest = text
            .lines()
            .filter_map(|line| {
                line.split("rows=")
                    .nth(1)?
                    .split_whitespace()
                    .next()?
                    .parse::<u64>()
                    .ok()
            })
            .max()
            .unwrap_or(0);
        assert!(
            widest < pairs,
            "{language}: an operator produced {widest} rows ({pairs} pairs): {text}"
        );
    }
}

/// SET of the key: a later row's key (100001) equals an earlier row's new
/// key, and still matches nothing, since every key is read before the
/// first write.
#[test]
fn setting_the_key_matches_the_old_keys() {
    for language in LANGUAGES {
        let db = database(false);
        let query = "UNWIND [1, 2, 100001] AS x MATCH (n:Graph:Method {id: x}) \
                     SET n.id = x + 100000 RETURN x ORDER BY x";
        assert_eq!(run(&db, language, query, None), ints(&[1, 2]), "{language}");
        let moved = run(
            &db,
            language,
            "MATCH (n:Graph:Method) WHERE n.id > 99999 RETURN n.id ORDER BY n.id",
            None,
        );
        assert_eq!(moved, ints(&[100_001, 100_002]), "{language}");
    }
}

/// DETACH DELETE with a key repeated: three matches, two Methods gone.
#[test]
fn delete_with_repeated_keys() {
    for language in LANGUAGES {
        let db = database(false);
        let query = "UNWIND [3, 3, 4] AS x MATCH (n:Graph:Method {id: x}) DETACH DELETE n \
                     RETURN count(*) AS c";
        assert_eq!(run(&db, language, query, None), ints(&[3]), "{language}");
        let left: Vec<i64> = state(&db, "Method").into_iter().map(|(id, _)| id).collect();
        let expected: Vec<i64> = (0..METHODS).filter(|id| *id != 3 && *id != 4).collect();
        assert_eq!(left, expected, "{language}");
    }
}

/// CREATE above the join: the second row's key names the node the first
/// row creates, and does not match it.
#[test]
fn a_node_created_above_is_not_matched_by_a_later_row() {
    for (language, create) in [("gql", "INSERT"), ("cypher", "CREATE")] {
        let db = database(false);
        let query = format!(
            "UNWIND [5, 9001] AS x MATCH (n:Graph:Method {{id: x}}) \
             {create} (:Graph:Method {{id: 9001}}) RETURN x"
        );
        assert_eq!(run(&db, language, &query, None), ints(&[5]), "{language}");
        let created = run(
            &db,
            language,
            "MATCH (n:Graph:Method {id: 9001}) RETURN count(n) AS c",
            None,
        );
        assert_eq!(created, ints(&[1]), "{language}");
    }
}

/// MERGE above the match: today's rows, pinned (the second row does not
/// see the node the first merged).
#[test]
fn merge_above_keeps_todays_rows() {
    for language in LANGUAGES {
        let db = database(false);
        let query = "UNWIND [7, 50007] AS x MATCH (n:Graph:Method {id: x}) \
                     MERGE (m:Graph:Method {id: x + 50000}) RETURN x, m.id AS m";
        assert_eq!(
            run(&db, language, query, None),
            vec![vec![Value::Int64(7), Value::Int64(50_007)]],
            "{language}"
        );
        let merged = run(
            &db,
            language,
            "MATCH (n:Graph:Method) WHERE n.id >= 50000 RETURN n.id ORDER BY n.id",
            None,
        );
        assert_eq!(merged, ints(&[50_007]), "{language}");
    }
}

/// The scan's input writes: the rows see the nodes their input created,
/// today's rows pinned.
#[test]
fn a_scan_whose_input_writes_keeps_todays_rows() {
    for (language, create) in [("gql", "INSERT"), ("cypher", "CREATE")] {
        let db = database(false);
        let query = format!(
            "UNWIND [8, 9] AS x {create} (:Graph:Method {{id: x + 70000}}) WITH x \
             MATCH (n:Graph:Method {{id: x + 70000}}) RETURN n.id ORDER BY n.id"
        );
        assert_eq!(
            run(&db, language, &query, None),
            ints(&[70_008, 70_009]),
            "{language}"
        );
    }
}

fn explain_text(db: &GrafeoDB, language: &str, ids: &[i64]) -> String {
    let plan = run(
        db,
        language,
        &format!("EXPLAIN {BULK}"),
        Some(update_rows(ids)),
    );
    match &plan[0][0] {
        Value::String(text) => text.to_string(),
        other => panic!("{language}: {other:?}"),
    }
}

/// With an `id` index the plan is chosen by cost: the hash join when the
/// rows times 3 exceed the scanned label's nodes, the seek otherwise. 3
/// rows seek; against 1500 Methods, 500 rows (1500, not more) seek, and 501
/// and 1500 rows build one hash join. The rows are the same either way.
#[test]
fn an_id_index_seeks_for_few_rows_and_hash_joins_for_many() {
    for language in LANGUAGES {
        let db = database(true);
        let few = [1, 2, 3];
        assert_eq!(
            run(&db, language, BULK, Some(update_rows(&few))),
            ints(&few),
            "{language}"
        );
        let text = explain_text(&db, language, &few);
        assert!(
            text.contains("[index: id]"),
            "{language}: 3 rows seek: {text}"
        );

        let big = GrafeoDB::new_in_memory();
        big.create_property_index("id").unwrap();
        big.execute("UNWIND range(0, 1499) AS i INSERT (:Graph:Method {id: i, p0: 'old'})")
            .unwrap();
        for (count, hash) in [(500, false), (501, true), (1500, true)] {
            let ids: Vec<i64> = (0..count).collect();
            assert_eq!(
                run(&big, language, BULK, Some(update_rows(&ids))),
                ints(&ids),
                "{language} {count} rows"
            );
            let text = explain_text(&big, language, &ids);
            if hash {
                assert!(
                    text.contains("[hash join: n.id = r.id]") && !text.contains("[index: id]"),
                    "{language}: {count} rows against 1500 Methods hash join: {text}"
                );
            } else {
                assert!(
                    text.contains("[index: id]"),
                    "{language}: {count} rows against 1500 Methods seek: {text}"
                );
            }
        }
    }
}

/// A null key matches nothing; a repeated key matches its node once per
/// row.
#[test]
fn null_and_repeated_keys() {
    for language in LANGUAGES {
        let db = database(false);
        assert_eq!(
            run(&db, language, BULK, Some(update_rows(&[-1, 10, 10, 11]))),
            ints(&[10, 10, 11]),
            "{language}"
        );
    }
}

/// A transaction's own new Method is matched by the update in the same
/// transaction, and gone after the rollback.
#[test]
fn a_transactions_own_writes() {
    for language in LANGUAGES {
        let db = database(false);
        let mut session = db.session();
        session.begin_transaction().unwrap();
        session
            .execute("INSERT (:Graph:Method {id: 80000, p0: 'old'})")
            .unwrap();
        let rows = session
            .execute_language(BULK, language, Some(update_rows(&[80_000, 1])))
            .unwrap()
            .rows()
            .to_vec();
        assert_eq!(rows, ints(&[1, 80_000]), "{language}");
        session.rollback().unwrap();
        assert_eq!(
            run(
                &db,
                language,
                "MATCH (n:Graph:Method {id: 80000}) RETURN n.id",
                None
            ),
            ints(&[]),
            "{language}"
        );
        assert_eq!(state(&db, "Method")[1].1, "old", "{language}: rolled back");
    }
}
