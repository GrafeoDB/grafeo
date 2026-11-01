//! A doubled backtick inside a backtick-delimited identifier is one backtick
//! (#579, ISO GQL and openCypher): for labels, property keys, variables and
//! graph names, in GQL and Cypher.
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test delimited_identifiers
//! ```

#![cfg(all(feature = "gql", feature = "cypher"))]

use grafeo_common::types::Value;
use grafeo_engine::GrafeoDB;

fn strings(value: &Value) -> Vec<String> {
    match value {
        Value::List(items) => items
            .iter()
            .map(|item| match item {
                Value::String(s) => s.to_string(),
                other => panic!("a string, not {other:?}"),
            })
            .collect(),
        other => panic!("a list, not {other:?}"),
    }
}

#[test]
fn a_doubled_backtick_is_one_backtick_in_labels_keys_and_variables() {
    for (language, insert, read) in [
        (
            "gql",
            "INSERT (:`MATCH``me` {`a``b`: 1})",
            "MATCH (`n``m`:`MATCH``me`) RETURN labels(`n``m`) AS l, keys(`n``m`) AS k, `n``m`.`a``b` AS v",
        ),
        (
            "cypher",
            "CREATE (:`MATCH``me` {`a``b`: 1})",
            "MATCH (`n``m`:`MATCH``me`) RETURN labels(`n``m`) AS l, keys(`n``m`) AS k, `n``m`.`a``b` AS v",
        ),
    ] {
        let db = GrafeoDB::new_in_memory();
        let run = |query: &str| match language {
            "gql" => db.execute(query),
            _ => db.execute_cypher(query),
        };
        run(insert).unwrap_or_else(|e| panic!("{language}: {insert}: {e}"));
        let result = run(read).unwrap_or_else(|e| panic!("{language}: {read}: {e}"));
        let row = &result.rows()[0];
        assert_eq!(strings(&row[0]), ["MATCH`me"], "{language}: labels");
        assert_eq!(strings(&row[1]), ["a`b"], "{language}: keys");
        assert_eq!(row[2], Value::Int64(1), "{language}: the property");
        assert_eq!(result.columns, ["l", "k", "v"], "{language}");
    }
}

#[test]
fn a_doubled_backtick_is_one_backtick_in_graph_names() {
    let db = GrafeoDB::new_in_memory();
    db.execute("CREATE GRAPH `my``graph` ANY").unwrap();
    assert!(
        db.list_graphs().contains(&"my`graph".to_string()),
        "{:?}",
        db.list_graphs()
    );
    db.execute("USE GRAPH `my``graph` INSERT (:P {v: 1})")
        .unwrap();
    let result = db
        .execute("USE GRAPH `my``graph` MATCH (p:P) RETURN p.v")
        .unwrap();
    assert_eq!(result.rows(), &[vec![Value::Int64(1)]]);
}
