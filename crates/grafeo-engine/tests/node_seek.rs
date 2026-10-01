//! A node pinned by `id(n)` or by an indexed property is looked up per input
//! row, also when the key comes from the row (`UNWIND`, an earlier `MATCH`):
//! the results are those of a scan, only without scanning.

#![cfg(all(feature = "lpg", feature = "gql"))]

use std::collections::HashMap;

use grafeo_common::types::Value;
use grafeo_engine::GrafeoDB;

/// Twelve `Doc` nodes with ids `d0`..`d11`, numbers 0..11 and a few edges;
/// the same data with and without a property index on `id`.
fn docs(indexed: bool) -> GrafeoDB {
    let db = GrafeoDB::new_in_memory();
    if indexed {
        db.create_property_index("id");
        db.create_property_index("n");
    }
    db.execute("UNWIND range(0, 11) AS i INSERT (:Doc {id: 'd' + toString(i), n: i})")
        .unwrap();
    db.execute("INSERT (:Other {id: 'd1'})").unwrap();
    db.execute(
        "MATCH (a:Doc), (b:Doc) WHERE b.n = a.n + 1 AND a.n < 11 INSERT (a)-[:NEXT {w: a.n}]->(b)",
    )
    .unwrap();
    db
}

fn rows(db: &GrafeoDB, query: &str, params: &[(&str, Value)]) -> Vec<Vec<Value>> {
    let params: HashMap<String, Value> = params
        .iter()
        .map(|(name, value)| ((*name).to_string(), value.clone()))
        .collect();
    let mut rows = db
        .execute_with_params(query, params)
        .unwrap()
        .rows()
        .to_vec();
    rows.sort_by_key(|row| format!("{row:?}"));
    rows
}

fn plan(db: &GrafeoDB, query: &str) -> String {
    let result = db.execute(&format!("EXPLAIN {query}")).unwrap();
    result
        .rows()
        .iter()
        .map(|row| format!("{:?}", row[0]))
        .collect::<Vec<_>>()
        .join("\n")
}

fn list(items: &[&str]) -> Value {
    Value::List(
        items
            .iter()
            .map(|s| Value::from(*s))
            .collect::<Vec<_>>()
            .into(),
    )
}

/// Every query returns the same rows with the seek (index) as with a scan.
#[test]
fn a_seek_returns_what_a_scan_returns() {
    let (sought, scanned) = (docs(true), docs(false));
    let keys = list(&["d3", "d7", "missing", "d3", "d1"]);
    let cases: Vec<(&str, Vec<(&str, Value)>)> = vec![
        (
            "UNWIND $keys AS k MATCH (n:Doc {id: k}) RETURN k, n.n",
            vec![("keys", keys.clone())],
        ),
        (
            "UNWIND $keys AS k MATCH (n:Doc) WHERE n.id = k AND n.n > 2 RETURN k, n.n",
            vec![("keys", keys.clone())],
        ),
        (
            "UNWIND $keys AS k MATCH (n) WHERE n.id = k RETURN k, labels(n)",
            vec![("keys", keys.clone())],
        ),
        (
            "MATCH (a:Doc {id: 'd2'}) MATCH (b:Doc) WHERE b.n = a.n * 2 RETURN b.id",
            vec![],
        ),
        (
            "UNWIND $groups AS g MATCH (n:Doc) WHERE n.id IN g RETURN n.id",
            vec![(
                "groups",
                Value::List(vec![list(&["d1", "d2"]), list(&["d2", "x"])].into()),
            )],
        ),
        (
            "UNWIND [null, 4, 4.0, '4'] AS k MATCH (n:Doc {n: k}) RETURN k, n.id",
            vec![],
        ),
        (
            "UNWIND $keys AS k MATCH (n:Doc {id: k})-[r:NEXT]->(m) RETURN n.id, r.w, m.id",
            vec![("keys", keys)],
        ),
    ];
    for (query, params) in cases {
        assert_eq!(
            rows(&sought, query, &params),
            rows(&scanned, query, &params),
            "{query}"
        );
    }
    assert_eq!(
        rows(
            &sought,
            "UNWIND ['d3', 'missing'] AS k MATCH (n:Doc {id: k}) RETURN k, n.n",
            &[]
        ),
        [vec![Value::from("d3"), Value::Int64(3)]]
    );
}

/// A literal key is looked up in the index once; of what it finds, only the
/// visible nodes with the pattern's label count (`:Other {id: 'd1'}` shares the
/// key), also for labels set or removed in the transaction.
#[test]
fn a_literal_key_keeps_the_nodes_with_the_label() {
    let (sought, scanned) = (docs(true), docs(false));
    for query in [
        "MATCH (n:Doc {id: 'd1'}) RETURN n.n",
        "MATCH (n:Other {id: 'd1'}) RETURN n.id",
        "MATCH (n:Doc) WHERE n.id IN ['d1', 'd2', 'x'] RETURN n.id",
        "MATCH (n:Other) WHERE n.id IN ['d1', 'd2'] RETURN n.id",
    ] {
        assert_eq!(
            rows(&sought, query, &[]),
            rows(&scanned, query, &[]),
            "{query}"
        );
    }
    assert_eq!(
        rows(&sought, "MATCH (n:Doc {id: 'd1'}) RETURN n.n", &[]),
        [vec![Value::Int64(1)]]
    );

    let mut session = sought.session();
    session.begin_transaction().unwrap();
    session
        .execute("MATCH (n:Other {id: 'd1'}) SET n:Doc")
        .unwrap();
    session
        .execute("MATCH (n:Doc {id: 'd2'}) REMOVE n:Doc")
        .unwrap();
    let count = |query: &str| session.execute(query).unwrap().rows().len();
    assert_eq!(count("MATCH (n:Doc {id: 'd1'}) RETURN n"), 2);
    assert_eq!(count("MATCH (n:Doc {id: 'd2'}) RETURN n"), 0);
    assert_eq!(
        count("MATCH (n:Doc) WHERE n.id IN ['d1', 'd2'] RETURN n"),
        2
    );
    session.rollback().unwrap();
    assert_eq!(
        rows(&sought, "MATCH (n:Doc {id: 'd2'}) RETURN n.n", &[]),
        [vec![Value::Int64(2)]]
    );
}

/// A labeled point lookup costs about what an unlabeled one does, however many
/// nodes have the label: the label is checked on the index's results, not by
/// collecting every node with it (which took 3 ms per lookup at 60,000 nodes).
#[cfg(not(debug_assertions))]
#[test]
fn a_labeled_point_lookup_does_not_grow_with_the_label() {
    use std::time::{Duration, Instant};

    let db = GrafeoDB::new_in_memory();
    db.execute("UNWIND range(0, 59999) AS i INSERT (:Graph:File {id: 'n' + toString(i)})")
        .unwrap();
    db.create_property_index("id");
    // The fastest of five batches, to keep a busy machine out of the ratio.
    let time = |query: &str| -> Duration {
        (0..5)
            .map(|_| {
                let start = Instant::now();
                for _ in 0..100 {
                    db.execute(query).unwrap();
                }
                start.elapsed()
            })
            .min()
            .unwrap()
    };
    let unlabeled = time("MATCH (s {id: 'n10'}) RETURN s.id");
    let labeled = time("MATCH (s:File {id: 'n10'}) RETURN s.id");
    assert!(
        labeled < unlabeled * 5,
        "labeled {labeled:?} vs unlabeled {unlabeled:?} per 100 lookups"
    );
}

#[test]
fn a_key_from_the_row_is_looked_up_in_the_index() {
    let db = docs(true);
    let unwind = "UNWIND $keys AS k MATCH (n:Doc {id: k}) RETURN n";
    assert!(
        plan(&db, unwind).contains("[index: id]"),
        "{}",
        plan(&db, unwind)
    );
    // Without an index there is no lookup to claim.
    let plain = docs(false);
    assert!(
        !plan(&plain, unwind).contains("[index"),
        "{}",
        plan(&plain, unwind)
    );
}

#[test]
fn id_lookups_do_not_scan() {
    let (db, scanned) = (docs(true), docs(false));
    let edge = "MATCH (s)-[r:NEXT]->(d) WHERE id(s) = $s AND id(d) = $d RETURN r.w";
    let ids = rows(
        &db,
        "MATCH (n:Doc) WHERE n.n IN [4, 5] RETURN id(n), n.n",
        &[],
    );
    let (four, five) = (ids[0][0].clone(), ids[1][0].clone());
    let params = [("s", four.clone()), ("d", five.clone())];
    assert_eq!(rows(&db, edge, &params), [vec![Value::Int64(4)]]);
    assert_eq!(rows(&db, edge, &params), rows(&scanned, edge, &params));
    assert!(rows(&db, edge, &[("s", five), ("d", four.clone())]).is_empty());

    let explained = plan(
        &db,
        "MATCH (s)-[r:NEXT]->(d) WHERE id(s) = 1 AND id(d) = 2 RETURN r",
    );
    assert!(explained.contains("[seek: id]"), "{explained}");
    let single = "MATCH (n) WHERE id(n) = $id RETURN n.id";
    assert_eq!(
        rows(&db, single, &[("id", four)]),
        [vec![Value::from("d4")]]
    );
    assert!(rows(&db, single, &[("id", Value::Int64(-1))]).is_empty());
    assert!(rows(&db, single, &[("id", Value::Null)]).is_empty());
}

/// A key that reads the clock is not looked up: the lookup and the filter
/// above it would each read the clock, and could see different days.
#[test]
fn a_clock_key_is_not_looked_up() {
    let db = docs(true);
    let lookup = |key: &str| {
        plan(
            &db,
            &format!("UNWIND ['d'] AS p MATCH (n:Doc {{id: {key}}}) RETURN n"),
        )
    };
    // Keys that read the clock or randomness, in any spelling or arity.
    for changing in [
        "p + toString(date())",
        "p + toString(zoneddatetime())",
        "p + toString(current_date())",
        "p + toString(timestamp())",
        "p + toString(randomUUID())",
    ] {
        let plan = lookup(changing);
        assert!(!plan.contains("[index"), "{changing}: {plan}");
    }
    // Keys built from their arguments alone.
    for fixed in [
        "p + toString(date('2026-09-30'))",
        "toLower(p)",
        "coalesce(p, 'x')",
    ] {
        let plan = lookup(fixed);
        assert!(plan.contains("[index: id]"), "{fixed}: {plan}");
    }
}

#[test]
fn a_seek_sees_what_the_transaction_sees() {
    let db = docs(true);
    let lookup = "UNWIND ['new'] AS k MATCH (n:Doc {id: k}) RETURN n.id";
    let mut session = db.session();
    session.begin_transaction().unwrap();
    session.execute("INSERT (:Doc {id: 'new'})").unwrap();
    session
        .execute("MATCH (n:Doc {id: 'd1'}) DETACH DELETE n")
        .unwrap();
    assert_eq!(session.execute(lookup).unwrap().rows().len(), 1);
    assert!(
        session
            .execute("UNWIND ['d1'] AS k MATCH (n:Doc {id: k}) RETURN n")
            .unwrap()
            .rows()
            .is_empty()
    );
    assert!(
        db.execute(lookup).unwrap().rows().is_empty(),
        "not committed"
    );
    session.rollback().unwrap();
    assert!(db.execute(lookup).unwrap().rows().is_empty());
    assert_eq!(
        db.execute("UNWIND ['d1'] AS k MATCH (n:Doc {id: k}) RETURN n")
            .unwrap()
            .rows()
            .len(),
        1
    );
}

#[test]
fn edge_upserts_between_sought_endpoints() {
    let db = docs(true);
    let upsert = "UNWIND $rows AS row \
                  MATCH (s:Doc {id: row.src}), (d:Doc {id: row.dst}) \
                  MERGE (s)-[r:LINK {id: row.id}]->(d) SET r.w = row.w RETURN count(r)";
    let row = |src: &str, dst: &str, id: &str, w: i64| {
        Value::Map(
            [
                ("src".into(), Value::from(src)),
                ("dst".into(), Value::from(dst)),
                ("id".into(), Value::from(id)),
                ("w".into(), Value::Int64(w)),
            ]
            .into_iter()
            .collect::<std::collections::BTreeMap<_, _>>()
            .into(),
        )
    };
    let written = rows(
        &db,
        upsert,
        &[(
            "rows",
            Value::List(
                vec![
                    row("d0", "d1", "l1", 1),
                    row("d0", "d1", "l1", 5),
                    row("d0", "missing", "l2", 1),
                ]
                .into(),
            ),
        )],
    );
    assert_eq!(written, [vec![Value::Int64(2)]]);
    assert_eq!(
        rows(&db, "MATCH ()-[r:LINK]->() RETURN r.id, r.w", &[]),
        [vec![Value::from("l1"), Value::Int64(5)]]
    );
}

/// PROFILE shows the operators that ran: the seek, not a scan.
#[test]
fn profile_shows_the_seek() {
    let db = docs(true);
    let profile = |query: &str| {
        let result = db.execute(&format!("PROFILE {query}")).unwrap();
        result
            .rows()
            .iter()
            .map(|row| format!("{row:?}"))
            .collect::<Vec<_>>()
            .join("\n")
    };
    let unwind = profile("UNWIND ['d3', 'd4'] AS k MATCH (n:Doc {id: k}) RETURN n.n");
    assert!(unwind.contains("NodeSeek"), "{unwind}");
    assert!(!unwind.contains("NodeScan"), "{unwind}");
    let by_id = profile("MATCH (n) WHERE id(n) = 3 RETURN n.n");
    assert!(by_id.contains("NodeSeek"), "{by_id}");
}
