//! A node pattern with several labels scans the label with the fewest nodes
//! and checks the others: `MATCH (e:Graph:Environment)`, where every code
//! element is a Graph and a few are Environments, reads the Environments,
//! whichever order the labels are written in, and the other way round when
//! Graph is the small one. The same inside an EXISTS. EXPLAIN shows the scan.
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test multi_label_scan_order
//! ```

#![cfg(feature = "gql")]

use grafeo_common::types::Value;
use grafeo_engine::GrafeoDB;

/// 300 nodes `broad` alone, 4 nodes with both labels, each with an `R` edge
/// to a Thing.
fn database(broad: &str, small: &str) -> GrafeoDB {
    let db = GrafeoDB::new_in_memory();
    db.execute(&format!(
        "UNWIND range(1, 300) AS i INSERT (:{broad} {{i: i}})"
    ))
    .unwrap();
    db.execute(&format!(
        "UNWIND range(301, 304) AS i INSERT (:{broad}:{small} {{i: i}})-[:R]->(:Thing {{i: i}})"
    ))
    .unwrap();
    db
}

fn explain(db: &GrafeoDB, query: &str) -> String {
    let rows = db
        .execute(&format!("EXPLAIN {query}"))
        .unwrap()
        .rows()
        .to_vec();
    match &rows[0][0] {
        Value::String(text) => text.to_string(),
        other => panic!("EXPLAIN `{query}` returned {other:?}"),
    }
}

fn ids(db: &GrafeoDB, query: &str) -> Vec<Value> {
    db.execute(query)
        .unwrap()
        .rows()
        .iter()
        .map(|row| row[0].clone())
        .collect()
}

const FOUR: [i64; 4] = [301, 302, 303, 304];

/// Graph broad, Environment small: the scan reads Environment, in both
/// orders of the labels.
#[test]
fn a_two_label_pattern_scans_the_smaller_label() {
    let db = database("Graph", "Environment");
    for pattern in ["e:Graph:Environment", "e:Environment:Graph"] {
        let query = format!("MATCH ({pattern}) RETURN e.i AS i ORDER BY i");
        assert_eq!(ids(&db, &query), FOUR.map(Value::Int64), "{query}");
        let plan = explain(&db, &query);
        assert!(plan.contains("NodeScan (e:Environment)"), "{query}: {plan}");
    }
}

/// The other way round: Environment broad, Graph small, the scan reads
/// Graph. The choice follows the counts, not the order or the names.
#[test]
fn the_choice_follows_the_label_counts() {
    let db = database("Environment", "Graph");
    for pattern in ["e:Graph:Environment", "e:Environment:Graph"] {
        let query = format!("MATCH ({pattern}) RETURN e.i AS i ORDER BY i");
        assert_eq!(ids(&db, &query), FOUR.map(Value::Int64), "{query}");
        let plan = explain(&db, &query);
        assert!(plan.contains("NodeScan (e:Graph)"), "{query}: {plan}");
    }
}

/// Inside an EXISTS the subquery's scan reads the smaller label too.
#[test]
fn a_two_label_pattern_in_an_exists_scans_the_smaller_label() {
    for (broad, small) in [("Graph", "Environment"), ("Environment", "Graph")] {
        let db = database(broad, small);
        for pattern in ["e:Graph:Environment", "e:Environment:Graph"] {
            let query = format!(
                "MATCH (t:Thing) WHERE EXISTS {{ MATCH ({pattern})-[:R]->(t) }} RETURN t.i AS i ORDER BY i"
            );
            assert_eq!(ids(&db, &query), FOUR.map(Value::Int64), "{query}");
            let plan = explain(&db, &query);
            let small_scan = format!("NodeScan (e:{small})");
            let broad_scan = format!("NodeScan (e:{broad})");
            assert!(
                plan.contains(&small_scan) && !plan.contains(&broad_scan),
                "{broad} broad, {small} small, {query}: {plan}"
            );
        }
    }
}

/// Three labels: the smallest of the three is scanned.
#[test]
fn three_labels_scan_the_smallest() {
    let db = GrafeoDB::new_in_memory();
    db.execute("UNWIND range(1, 200) AS i INSERT (:Graph {i: i})")
        .unwrap();
    db.execute("UNWIND range(201, 240) AS i INSERT (:Graph:Method {i: i})")
        .unwrap();
    db.execute("UNWIND range(241, 242) AS i INSERT (:Graph:Method:Test {i: i})")
        .unwrap();
    let query = "MATCH (e:Graph:Method:Test) RETURN e.i AS i ORDER BY i";
    assert_eq!(ids(&db, query), [241, 242].map(Value::Int64));
    let plan = explain(&db, query);
    assert!(plan.contains("NodeScan (e:Test)"), "{plan}");
}
