//! Grouping by `labels(n)` (the label-mask key, 0.6.1): the groups are the
//! label sets, with the counts, sums and lists of grouping the plain label
//! lists. Each grouped query is checked against rows written by hand and
//! against the same rows grouped here from `RETURN labels(n)` per node:
//!
//! - count, sum and collect per label set; a node without labels is the
//!   empty list;
//! - a filter on the grouped label list (WITH ... WHERE), ORDER BY the list;
//! - labels(n) as the key and inside a returned expression;
//! - two label keys (labels(a), labels(b));
//! - more than 64 labels in the store (no mask for the high ones): a set
//!   with a high label still groups with its equal sets;
//! - at a past epoch, with a transaction's own label changes;
//! - on a projection (a store without masks).
//!
//! In GQL and Cypher.
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test group_by_labels
//! ```

#![cfg(all(feature = "gql", feature = "cypher"))]

use std::collections::BTreeMap;

use grafeo_common::types::Value;
use grafeo_engine::{Config, GrafeoDB, HistoryRetention, ProjectionSpec};

const LANGUAGES: [&str; 2] = ["gql", "cypher"];

/// Alix, Gus and Vincent are Persons (v 1, 2, 3), Mia a Person and Admin
/// (v 10), Paris and Prague Cities (v 100, 200), and one node without
/// labels (v 1000). R: Alix and Gus to Paris, Mia to Prague, Vincent to Mia,
/// the unlabelled node to Alix.
fn database() -> GrafeoDB {
    let db = GrafeoDB::with_config(
        Config::in_memory().with_history_retention(HistoryRetention::Forever),
    )
    .unwrap();
    db.execute(
        "INSERT (alix:Person {name: 'Alix', v: 1}), (gus:Person {name: 'Gus', v: 2}), \
         (vincent:Person {name: 'Vincent', v: 3}), (mia:Person:Admin {name: 'Mia', v: 10}), \
         (paris:City {name: 'Paris', v: 100}), (prague:City {name: 'Prague', v: 200}), \
         (none {name: 'none', v: 1000}), \
         (alix)-[:R]->(paris), (gus)-[:R]->(paris), (mia)-[:R]->(prague), \
         (vincent)-[:R]->(mia), (none)-[:R]->(alix)",
    )
    .unwrap();
    db
}

fn list(names: &[&str]) -> Value {
    Value::List(
        names
            .iter()
            .map(|n| Value::from(*n))
            .collect::<Vec<_>>()
            .into(),
    )
}

fn rows(db: &GrafeoDB, language: &str, query: &str) -> Vec<Vec<Value>> {
    db.execute_language(query, language, None)
        .unwrap_or_else(|error| panic!("{language} `{query}`: {error}"))
        .rows()
        .to_vec()
}

/// `(labels, count, sum of v)` per label set, from the per-node label lists
/// of `pattern` (which binds `n`), grouped here.
fn grouped_here(db: &GrafeoDB, language: &str, pattern: &str) -> Vec<Vec<Value>> {
    let mut groups: BTreeMap<String, (Value, i64, i64)> = BTreeMap::new();
    for row in rows(
        db,
        language,
        &format!("{pattern} RETURN labels(n) AS l, n.v AS v"),
    ) {
        let key = format!("{:?}", row[0]);
        let v = match row[1] {
            Value::Int64(v) => v,
            _ => 0,
        };
        let entry = groups.entry(key).or_insert((row[0].clone(), 0, 0));
        entry.1 += 1;
        entry.2 += v;
    }
    groups
        .into_values()
        .map(|(l, c, s)| vec![l, Value::Int64(c), Value::Int64(s)])
        .collect()
}

/// The grouped query's `(l, c, s)` rows in the order of [`grouped_here`].
fn grouped(db: &GrafeoDB, language: &str, pattern: &str) -> Vec<Vec<Value>> {
    let mut got = rows(
        db,
        language,
        &format!("{pattern} RETURN labels(n) AS l, count(*) AS c, sum(n.v) AS s"),
    );
    got.sort_by_key(|row| format!("{:?}", row[0]));
    got
}

#[test]
fn count_and_sum_per_label_set() {
    let db = database();
    for language in LANGUAGES {
        let got = grouped(&db, language, "MATCH (n)");
        assert_eq!(got, grouped_here(&db, language, "MATCH (n)"), "{language}");
        let mut expected = vec![
            vec![list(&[]), Value::Int64(1), Value::Int64(1000)],
            vec![
                list(&["Admin", "Person"]),
                Value::Int64(1),
                Value::Int64(10),
            ],
            vec![list(&["City"]), Value::Int64(2), Value::Int64(300)],
            vec![list(&["Person"]), Value::Int64(3), Value::Int64(6)],
        ];
        expected.sort_by_key(|row| format!("{:?}", row[0]));
        assert_eq!(got, expected, "{language}");
    }
}

#[test]
fn collect_per_label_set() {
    let db = database();
    for language in LANGUAGES {
        let got = rows(
            &db,
            language,
            "MATCH (n:Person) WITH labels(n) AS l, collect(n.name) AS names \
             RETURN l, size(names) AS k ORDER BY k",
        );
        assert_eq!(
            got,
            [
                vec![list(&["Admin", "Person"]), Value::Int64(1)],
                vec![list(&["Person"]), Value::Int64(3)]
            ],
            "{language}"
        );
    }
}

/// A filter on the grouped list, after the grouping.
#[test]
fn a_filter_on_the_grouped_list() {
    let db = database();
    for language in LANGUAGES {
        let got = rows(
            &db,
            language,
            "MATCH (n) WITH labels(n) AS l, count(*) AS c WHERE 'Person' IN l \
             RETURN l, c ORDER BY c",
        );
        assert_eq!(
            got,
            [
                vec![list(&["Admin", "Person"]), Value::Int64(1)],
                vec![list(&["Person"]), Value::Int64(3)]
            ],
            "{language}"
        );
    }
}

/// ORDER BY the grouped list: the empty list first, then by element.
#[test]
fn order_by_the_grouped_list() {
    let db = database();
    for language in LANGUAGES {
        let got: Vec<Value> = rows(
            &db,
            language,
            "MATCH (n) RETURN labels(n) AS l, count(*) AS c ORDER BY l",
        )
        .into_iter()
        .map(|row| row[0].clone())
        .collect();
        assert_eq!(
            got,
            [
                list(&[]),
                list(&["Admin", "Person"]),
                list(&["City"]),
                list(&["Person"])
            ],
            "{language}"
        );
    }
}

/// labels(n) as the key and inside a returned expression of it.
#[test]
fn the_key_inside_a_returned_expression() {
    let db = database();
    for language in LANGUAGES {
        let got = rows(
            &db,
            language,
            "MATCH (n) WITH labels(n) AS l, count(*) AS c \
             RETURN size(l) AS k, l[0] AS first, c ORDER BY k, c",
        );
        assert_eq!(
            got,
            [
                vec![Value::Int64(0), Value::Null, Value::Int64(1)],
                vec![Value::Int64(1), Value::from("City"), Value::Int64(2)],
                vec![Value::Int64(1), Value::from("Person"), Value::Int64(3)],
                vec![Value::Int64(2), Value::from("Admin"), Value::Int64(1)]
            ],
            "{language}"
        );
    }
}

/// Two label keys, of both ends of R.
#[test]
fn two_label_keys() {
    let db = database();
    for language in LANGUAGES {
        let mut got = rows(
            &db,
            language,
            "MATCH (a)-[:R]->(b) RETURN labels(a) AS la, labels(b) AS lb, count(*) AS c",
        );
        got.sort_by_key(|row| format!("{row:?}"));
        let mut expected = vec![
            vec![list(&["Person"]), list(&["City"]), Value::Int64(2)],
            vec![list(&["Admin", "Person"]), list(&["City"]), Value::Int64(1)],
            vec![
                list(&["Person"]),
                list(&["Admin", "Person"]),
                Value::Int64(1),
            ],
            vec![list(&[]), list(&["Person"]), Value::Int64(1)],
        ];
        expected.sort_by_key(|row| format!("{row:?}"));
        assert_eq!(got, expected, "{language}");
    }
}

/// 70 labels in the store: X0 to X69 one node each, and two nodes with both
/// X0 and X69 (created in either order). The high label has no mask bit:
/// those nodes still group by their set, the two together.
#[test]
fn more_than_64_labels_still_group_by_set() {
    let db = GrafeoDB::new_in_memory();
    for i in 0..70 {
        db.execute(&format!("INSERT (:X{i} {{v: {i}}})")).unwrap();
    }
    db.execute("INSERT (:X0:X69 {v: 1000}), (:X69:X0 {v: 2000})")
        .unwrap();
    for language in LANGUAGES {
        let got = grouped(&db, language, "MATCH (n)");
        assert_eq!(got, grouped_here(&db, language, "MATCH (n)"), "{language}");
        assert_eq!(got.len(), 71, "{language}: 70 single sets and one pair");
        let pair = got
            .iter()
            .find(|row| row[0] == list(&["X0", "X69"]))
            .unwrap_or_else(|| panic!("{language}: no X0, X69 group"));
        assert_eq!(
            pair[1..],
            [Value::Int64(2), Value::Int64(3000)],
            "{language}"
        );
    }
}

/// At a past epoch the groups are those of then; a transaction sees its
/// own label changes, and the rollback undoes them.
#[test]
fn a_past_epoch_and_a_transactions_label_changes() {
    let db = database();
    let before = db.current_epoch();
    db.execute("MATCH (n {name: 'Gus'}) SET n:Admin").unwrap();
    for language in LANGUAGES {
        assert_eq!(
            grouped(&db, language, "MATCH (n:Person)"),
            grouped_here(&db, language, "MATCH (n:Person)"),
            "{language} now"
        );
    }
    let then = db
        .execute_at_epoch(
            "MATCH (n:Person) RETURN labels(n) AS l, count(*) AS c ORDER BY c",
            before,
        )
        .unwrap();
    assert_eq!(
        then.rows(),
        [
            [list(&["Admin", "Person"]), Value::Int64(1)],
            [list(&["Person"]), Value::Int64(3)]
        ]
    );

    let mut session = db.session();
    session.begin_transaction().unwrap();
    session
        .execute("MATCH (n {name: 'Vincent'}) SET n:Admin")
        .unwrap();
    session
        .execute("MATCH (n {name: 'Mia'}) REMOVE n:Admin")
        .unwrap();
    let inside = session
        .execute("MATCH (n:Person) RETURN labels(n) AS l, count(*) AS c")
        .unwrap();
    let mut got = inside.rows().to_vec();
    got.sort_by_key(|row| format!("{row:?}"));
    assert_eq!(
        got,
        [
            vec![list(&["Admin", "Person"]), Value::Int64(2)],
            vec![list(&["Person"]), Value::Int64(2)]
        ],
        "Gus and Vincent are Admins, Mia no longer"
    );
    session.rollback().unwrap();
    let mut after = rows(
        &db,
        "gql",
        "MATCH (n:Person) RETURN labels(n) AS l, count(*) AS c",
    );
    after.sort_by_key(|row| format!("{row:?}"));
    assert_eq!(
        after,
        [
            vec![list(&["Admin", "Person"]), Value::Int64(2)],
            vec![list(&["Person"]), Value::Int64(2)]
        ],
        "Gus (committed) and Mia are Admins"
    );
}

/// On a projection (a store without label masks) the groups are those of
/// its label lists.
#[test]
fn on_a_projection() {
    let db = database();
    db.create_projection("people", ProjectionSpec::new().with_node_labels(["Person"]))
        .unwrap();
    let view = GrafeoDB::with_read_store(
        db.projection("people").unwrap().shared(),
        Config::in_memory(),
    )
    .unwrap();
    for language in LANGUAGES {
        let got = grouped(&view, language, "MATCH (n)");
        assert_eq!(
            got,
            grouped_here(&view, language, "MATCH (n)"),
            "{language}"
        );
        let total: i64 = got
            .iter()
            .map(|row| match row[1] {
                Value::Int64(c) => c,
                _ => 0,
            })
            .sum();
        assert_eq!(total, 4, "{language}: the four Persons");
    }
}
