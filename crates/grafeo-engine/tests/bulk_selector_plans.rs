//! The plans of a code-graph client's bulk paths (Deriva's): an upsert with
//! `UNWIND $rows ... MERGE` seeks the merge key's index, a selector narrowed
//! with `IN $ids` seeks an indexed property as `IN [list]` and `=` do, a
//! label test the scan already guarantees leaves no filter behind, and
//! `collect(x)` sliced gives the right elements on a large input. EXPLAIN
//! shows each plan; every query also checks its rows.
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test bulk_selector_plans
//! ```

#![cfg(all(feature = "gql", feature = "cypher"))]

use std::collections::HashMap;

use grafeo_common::types::Value;
use grafeo_engine::GrafeoDB;

/// 500 Methods (`:Graph:Method`, ids `m1` to `m500`) and 300 Files, with an
/// index on `Method.id` when `indexed`.
fn database(indexed: bool) -> GrafeoDB {
    let db = GrafeoDB::new_in_memory();
    db.execute("UNWIND range(1, 500) AS i INSERT (:Graph:Method {id: 'm' + toString(i), n: i})")
        .unwrap();
    db.execute("UNWIND range(1, 300) AS i INSERT (:Graph:File {id: 'f' + toString(i), n: i})")
        .unwrap();
    if indexed {
        db.execute("CREATE INDEX method_id FOR (m:Method) ON (m.id)")
            .unwrap();
    }
    db
}

fn params(pairs: &[(&str, Value)]) -> HashMap<String, Value> {
    pairs
        .iter()
        .map(|(name, value)| ((*name).to_string(), value.clone()))
        .collect()
}

fn rows(
    db: &GrafeoDB,
    language: &str,
    query: &str,
    with: &HashMap<String, Value>,
) -> Vec<Vec<Value>> {
    db.execute_language(query, language, Some(with.clone()))
        .unwrap_or_else(|error| panic!("{language} `{query}`: {error}"))
        .rows()
        .to_vec()
}

fn explain(db: &GrafeoDB, language: &str, query: &str, with: &HashMap<String, Value>) -> String {
    let result = rows(db, language, &format!("EXPLAIN {query}"), with);
    match &result[0][0] {
        Value::String(text) => text.to_string(),
        other => panic!("{language} EXPLAIN `{query}` returned {other:?}"),
    }
}

fn texts(values: &[&str]) -> Value {
    Value::List(values.iter().map(|text| Value::from(*text)).collect())
}

/// `UNWIND $rows AS r MERGE (m:Method {id: r.id})` seeks the index on the
/// merge key, updates the existing Methods and creates the new ones once.
#[test]
fn merge_under_unwind_seeks_the_merge_key() {
    let rows_param = Value::List(
        ["m1", "m2", "m501", "m502"]
            .iter()
            .enumerate()
            .map(|(i, id)| {
                Value::Map(
                    std::collections::BTreeMap::from([
                        ("id".into(), Value::from(*id)),
                        ("n".into(), Value::Int64(1000 + i64::try_from(i).unwrap())),
                    ])
                    .into(),
                )
            })
            .collect::<Vec<_>>()
            .into(),
    );
    let with = params(&[("rows", rows_param)]);
    for language in ["gql", "cypher"] {
        let db = database(true);
        let query = "UNWIND $rows AS r MERGE (m:Method {id: r.id}) SET m.n = r.n";
        let plan = explain(&db, language, query, &with);
        assert!(plan.contains("Merge (m) [index: id]"), "{language}: {plan}");
        rows(&db, language, query, &with);
        let count = rows(
            &db,
            language,
            "MATCH (m:Method) RETURN count(m) AS c",
            &with,
        );
        assert_eq!(
            count,
            [[Value::Int64(502)]],
            "{language}: two new Methods, no duplicate"
        );
        let updated = rows(
            &db,
            language,
            "MATCH (m:Method) WHERE m.n >= 1000 RETURN m.id AS id ORDER BY id",
            &with,
        );
        assert_eq!(
            updated,
            ["m1", "m2", "m501", "m502"].map(|id| vec![Value::from(id)]),
            "{language}"
        );
    }
}

/// Control: without an index the merge plans no seek.
#[test]
fn merge_without_an_index_plans_no_seek() {
    let db = database(false);
    let with = params(&[]);
    let plan = explain(
        &db,
        "gql",
        "UNWIND ['m1'] AS id MERGE (m:Method {id: id})",
        &with,
    );
    assert!(!plan.contains("[index:"), "{plan}");
}

/// `m.id IN $ids` seeks the index as `m.id IN ['m1', ...]` does.
#[test]
fn in_a_parameter_list_seeks_the_index() {
    let db = database(true);
    let with = params(&[("ids", texts(&["m3", "m7", "f1", "m999"]))]);
    for language in ["gql", "cypher"] {
        let literal = explain(
            &db,
            language,
            "MATCH (m:Method) WHERE m.id IN ['m3', 'm7'] RETURN m.n AS n",
            &with,
        );
        assert!(
            literal.contains("[index: id]"),
            "{language} control: {literal}"
        );
        let query = "MATCH (m:Method) WHERE m.id IN $ids RETURN m.n AS n ORDER BY n";
        let plan = explain(&db, language, query, &with);
        assert!(plan.contains("[index: id]"), "{language}: {plan}");
        assert_eq!(
            rows(&db, language, query, &with),
            [[Value::Int64(3)], [Value::Int64(7)]],
            "{language}: the Methods among the ids, a File id and a missing one left out"
        );
    }
}

/// `IN $ids` with an empty list or null: no rows, and no error.
#[test]
fn in_an_empty_or_null_parameter_finds_nothing() {
    let db = database(true);
    for ids in [texts(&[]), Value::Null] {
        let with = params(&[("ids", ids.clone())]);
        let found = rows(
            &db,
            "gql",
            "MATCH (m:Method) WHERE m.id IN $ids RETURN m.n AS n",
            &with,
        );
        assert_eq!(found, Vec::<Vec<Value>>::new(), "{ids:?}");
    }
}

/// `'Method' IN labels(n)` on a scan of Method is true for every row: the
/// plan keeps no filter for it.
#[test]
fn a_label_test_the_scan_guarantees_leaves_no_filter() {
    let db = database(false);
    let with = params(&[]);
    for language in ["gql", "cypher"] {
        let query = "MATCH (n:Method) WHERE 'Method' IN labels(n) RETURN count(n) AS c";
        assert_eq!(
            rows(&db, language, query, &with),
            [[Value::Int64(500)]],
            "{language}"
        );
        let plan = explain(&db, language, query, &with);
        assert!(
            !plan
                .lines()
                .any(|line| line.trim_start().starts_with("Filter")),
            "{language}: {plan}"
        );
    }
}

/// `'Method' IN labels(n)` on a scan of Graph is the label test `n:Method`:
/// it plans as `(n:Graph:Method)` does, a scan of the smaller label.
#[test]
fn a_label_test_on_another_label_plans_as_a_label() {
    let db = database(false);
    let with = params(&[]);
    for language in ["gql", "cypher"] {
        let query = "MATCH (n:Graph) WHERE 'File' IN labels(n) RETURN count(n) AS c";
        assert_eq!(
            rows(&db, language, query, &with),
            [[Value::Int64(300)]],
            "{language}"
        );
        let plan = explain(&db, language, query, &with);
        assert!(
            plan.contains("NodeScan (n:File)") && !plan.contains("labels(n)"),
            "{language}: {plan}"
        );
    }
}

/// `collect(x)` sliced on a large input: the elements of the slice, in the
/// order collected. EXPLAIN has no bounded collect (an Aggregate, then a
/// slice), so this checks the result only. Negative bounds are in
/// tests/spec/rosetta/list_slices.gtest.
#[test]
fn a_sliced_collect_gives_its_elements_on_a_large_input() {
    let db = GrafeoDB::new_in_memory();
    let with = params(&[]);
    let ints = |values: &[i64]| Value::List(values.iter().map(|v| Value::Int64(*v)).collect());
    for language in ["gql", "cypher"] {
        for (slice, expected) in [
            ("[..3]", ints(&[1, 2, 3])),
            ("[0..3]", ints(&[1, 2, 3])),
            ("[5..8]", ints(&[6, 7, 8])),
            ("[199998..]", ints(&[199_999, 200_000])),
            ("[0..0]", ints(&[])),
        ] {
            let query = format!(
                "UNWIND range(1, 200000) AS x WITH x ORDER BY x RETURN collect(x){slice} AS xs"
            );
            assert_eq!(
                rows(&db, language, &query, &with),
                [[expected.clone()]],
                "{language} {slice}"
            );
        }
        let query = "UNWIND range(1, 200000) AS x WITH x % 3 AS k, x ORDER BY x \
                     RETURN k, collect(x)[..2] AS xs ORDER BY k";
        assert_eq!(
            rows(&db, language, query, &with),
            [
                vec![Value::Int64(0), ints(&[3, 6])],
                vec![Value::Int64(1), ints(&[1, 4])],
                vec![Value::Int64(2), ints(&[2, 5])],
            ],
            "{language} per group"
        );
    }
}
