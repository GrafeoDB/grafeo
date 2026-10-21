//! A filter or projection reads one property of a node or edge without
//! building the entity, and sees exactly what the read sees: nothing of a
//! node another open transaction created, the node a transaction created
//! itself, and nothing at a past epoch before the node existed. Also for
//! edge properties.
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test property_read_visibility
//! cargo test -p grafeo-engine --features full --test property_read_visibility
//! ```

#![cfg(all(feature = "lpg", feature = "gql"))]

use grafeo_common::types::Value;
use grafeo_engine::{GrafeoDB, Session};

fn names(session: &Session, query: &str) -> Vec<Value> {
    let mut rows: Vec<Value> = session
        .execute(query)
        .unwrap()
        .rows()
        .iter()
        .map(|row| row[0].clone())
        .collect();
    rows.sort_by_key(ToString::to_string);
    rows
}

const BY_CITY: &str = "MATCH (p:Person) WHERE p.city = 'Prague' RETURN p.name";
const BY_KEY: &str = "MATCH (p:Person {city: 'Prague'}) RETURN p.name";

#[test]
fn a_property_filter_sees_only_what_the_read_sees() {
    let db = GrafeoDB::new_in_memory();
    db.execute("INSERT (:Person {name: 'Alix', city: 'Prague'})")
        .unwrap();

    let mut writer = db.session();
    writer.begin_transaction().unwrap();
    writer
        .execute("INSERT (:Person {name: 'Gus', city: 'Prague'})")
        .unwrap();

    // Another session sees only the committed node, through a filter, a
    // key in the pattern and a projection.
    let reader = db.session();
    for query in [BY_CITY, BY_KEY] {
        assert_eq!(names(&reader, query), [Value::from("Alix")], "{query}");
    }
    // The writing transaction sees its own node.
    for query in [BY_CITY, BY_KEY] {
        assert_eq!(
            names(&writer, query),
            [Value::from("Alix"), Value::from("Gus")],
            "{query}"
        );
    }
    writer.commit().unwrap();
    assert_eq!(
        names(&reader, BY_CITY),
        [Value::from("Alix"), Value::from("Gus")]
    );
}

#[test]
fn a_rolled_back_node_is_not_read() {
    let db = GrafeoDB::new_in_memory();
    let mut writer = db.session();
    writer.begin_transaction().unwrap();
    writer
        .execute("INSERT (:Person {name: 'Vincent', city: 'Prague'})")
        .unwrap();
    writer.rollback().unwrap();
    assert_eq!(names(&db.session(), BY_CITY), Vec::<Value>::new());
}

#[test]
fn a_deleted_node_is_not_read() {
    let db = GrafeoDB::new_in_memory();
    db.execute(
        "INSERT (:Person {name: 'Mia', city: 'Prague'}), (:Person {name: 'Jules', city: 'Prague'})",
    )
    .unwrap();
    db.execute("MATCH (p:Person {name: 'Mia'}) DELETE p")
        .unwrap();
    assert_eq!(names(&db.session(), BY_CITY), [Value::from("Jules")]);
}

#[test]
fn an_edge_property_filter_sees_only_what_the_read_sees() {
    let db = GrafeoDB::new_in_memory();
    db.execute("INSERT (:Person {name: 'Alix'})-[:KNOWS {since: 3}]->(:Person {name: 'Gus'})")
        .unwrap();
    let mut writer = db.session();
    writer.begin_transaction().unwrap();
    writer
        .execute(
            "MATCH (a:Person {name: 'Alix'}) INSERT (a)-[:KNOWS {since: 3}]->(:Person {name: 'Butch'})",
        )
        .unwrap();
    let query = "MATCH (:Person)-[k:KNOWS]->(b) WHERE k.since = 3 RETURN b.name";
    assert_eq!(names(&db.session(), query), [Value::from("Gus")]);
    assert_eq!(
        names(&writer, query),
        [Value::from("Butch"), Value::from("Gus")]
    );
    let projected = "MATCH (:Person)-[k:KNOWS]->() RETURN k.since";
    assert_eq!(names(&db.session(), projected), [Value::Int64(3)]);
    writer.rollback().unwrap();
}

#[test]
fn a_missing_property_reads_as_null() {
    let db = GrafeoDB::new_in_memory();
    db.execute("INSERT (:Person {name: 'Hans'})").unwrap();
    let session = db.session();
    assert_eq!(names(&session, BY_CITY), Vec::<Value>::new());
    assert_eq!(
        names(&session, "MATCH (p:Person) RETURN p.city"),
        [Value::Null]
    );
}

#[test]
fn a_read_at_a_past_epoch_sees_the_values_of_then() {
    // History kept: a read at an earlier epoch needs it.
    let db = GrafeoDB::with_config(
        grafeo_engine::Config::in_memory()
            .with_history_retention(grafeo_engine::HistoryRetention::Forever),
    )
    .unwrap();
    db.execute("INSERT (:Person {name: 'Shosanna', city: 'Paris'})")
        .unwrap();
    let before = db.current_epoch();
    db.execute("MATCH (p:Person) SET p.city = 'Prague'")
        .unwrap();
    let then = db
        .session()
        .execute_at_epoch(BY_CITY, before)
        .unwrap()
        .rows()
        .len();
    assert_eq!(then, 0, "Shosanna lived in Paris then");
    assert_eq!(names(&db.session(), BY_CITY), [Value::from("Shosanna")]);
}
