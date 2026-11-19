//! The edge type filter an expand and an intersection prepare once, when they open
//! (`GraphStore::edge_type_filter`): it holds while the store's types do and resolves again
//! when they change, so a type created within the statement is found on the next row, a
//! type a later clause creates reads no edges before it, one text on two graphs reads each
//! graph's own, and statements on many threads, each with its own filters, read alike while
//! another creates types. The store's own cases (a filter against another store, the oracle
//! of the filter and name reads) are in the conformance suite.
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test main edge_type_filter::
//! ```

#![cfg(all(feature = "gql", feature = "cypher"))]

use grafeo_common::types::Value;
use grafeo_engine::GrafeoDB;
use grafeo_engine::session::Session;

fn rows(session: &Session, query: &str) -> Vec<Vec<Value>> {
    let mut rows = session.execute(query).unwrap().rows().to_vec();
    rows.sort_by_key(|row| format!("{row:?}"));
    rows
}

fn cypher_rows(db: &GrafeoDB, query: &str) -> Vec<Vec<Value>> {
    let mut rows = db.execute_cypher(query).unwrap().rows().to_vec();
    rows.sort_by_key(|row| format!("{row:?}"));
    rows
}

fn people(db: &GrafeoDB) {
    db.execute("INSERT (:Person {name: 'Alix'}), (:Person {name: 'Gus'})")
        .unwrap();
}

/// The expand after `WITH` was planned, its filter resolved, before the type `NEW`
/// existed: the `CREATE` before it makes the type, and the expand finds each new edge.
#[test]
fn a_type_created_mid_statement_is_found_on_the_next_row() {
    let db = GrafeoDB::new_in_memory();
    people(&db);
    let query = "MATCH (a:Person) CREATE (a)-[:NEW]->(:X) WITH a \
                 MATCH (a)-[:NEW]->(b) RETURN a.name, labels(b)";
    let x = Value::List(vec![Value::from("X")].into());
    assert_eq!(
        cypher_rows(&db, query),
        [
            vec![Value::from("Alix"), x.clone()],
            vec![Value::from("Gus"), x.clone()],
        ]
    );
    // Again, with the type known when the filters resolve: one edge more each.
    assert_eq!(cypher_rows(&db, query).len(), 4);
}

/// A type no edge has when the statement starts reads no edges, until a later clause
/// creates it; a second run, with the type known from the start, counts one edge more.
#[test]
fn an_unknown_type_reads_no_edges_until_a_later_clause_creates_it() {
    let db = GrafeoDB::new_in_memory();
    people(&db);
    let query = "MATCH (a:Person {name: 'Alix'}) OPTIONAL MATCH (a)-[:LATER]->(x) \
                 WITH a, count(x) AS before CREATE (a)-[:LATER]->(:Y) WITH a, before \
                 MATCH (a)-[:LATER]->(y) RETURN before, count(y) AS after";
    assert_eq!(
        cypher_rows(&db, query),
        [vec![Value::Int64(0), Value::Int64(1)]]
    );
    assert_eq!(
        cypher_rows(&db, query),
        [vec![Value::Int64(1), Value::Int64(2)]]
    );
}

/// The same texts on two graphs whose stores number the types differently, in turn and in
/// one transaction: each reads its own graph's edges.
#[test]
fn one_text_on_two_graphs_reads_each_graphs_own_types() {
    let db = GrafeoDB::new_in_memory();
    db.execute("CREATE GRAPH alpha").unwrap();
    db.execute("CREATE GRAPH beta").unwrap();
    let mut session = db.session();
    session
        .execute("USE alpha INSERT (:P {name: 'Alix'})-[:SPARE]->(:P {name: 'Gus'})")
        .unwrap();
    session
        .execute("USE alpha INSERT (:P {name: 'Mia'})-[:KNOWS]->(:P {name: 'Vincent'})")
        .unwrap();
    session
        .execute("USE beta INSERT (:P {name: 'Jules'})-[:KNOWS]->(:P {name: 'Butch'})")
        .unwrap();
    let pair = |a: &str, b: &str| vec![Value::from(a), Value::from(b)];
    let check = |session: &Session| {
        for _ in 0..2 {
            for (graph, knows, spare) in [
                (
                    "alpha",
                    vec![pair("Mia", "Vincent")],
                    vec![pair("Alix", "Gus")],
                ),
                ("beta", vec![pair("Jules", "Butch")], vec![]),
            ] {
                assert_eq!(
                    rows(
                        session,
                        &format!("USE {graph} MATCH (a)-[:KNOWS]->(b) RETURN a.name, b.name")
                    ),
                    knows,
                    "{graph}"
                );
                assert_eq!(
                    rows(
                        session,
                        &format!("USE {graph} MATCH (a)-[:SPARE]->(b) RETURN a.name, b.name")
                    ),
                    spare,
                    "{graph}"
                );
            }
        }
    };
    check(&session);
    session.begin_transaction().unwrap();
    check(&session);
    session.commit().unwrap();
}

/// Statements on several threads, each operator with its own filters, read the same
/// expands and intersections while another thread keeps creating new edge types, each of
/// which changes the stamp every filter checks.
#[test]
fn statements_on_many_threads_read_alike_while_types_are_created() {
    let db = GrafeoDB::new_in_memory();
    db.execute(
        "INSERT (a:Person {name: 'Alix'}), (g:Person {name: 'Gus'}), (m:Person {name: 'Mia'}), \
         (a)-[:KNOWS]->(g), (g)-[:KNOWS]->(m), (m)-[:KNOWS]->(a), (a)-[:KNOWS]->(m), \
         (:Spare {name: 'Django'})",
    )
    .unwrap();
    let hops = "MATCH (a:Person)-[:KNOWS]->(b:Person) RETURN count(*)";
    let triangles = "MATCH (a:Person)-[:KNOWS]->(b:Person)-[:KNOWS]->(c:Person)-[:KNOWS]->(a) \
                     RETURN count(*)";
    let count = |query: &str| db.execute(query).unwrap().rows()[0][0].clone();
    let (hop_count, triangle_count) = (count(hops), count(triangles));
    assert_eq!(hop_count, Value::Int64(4));
    assert_eq!(triangle_count, Value::Int64(3));
    std::thread::scope(|scope| {
        scope.spawn(|| {
            for index in 0..40 {
                db.execute(&format!(
                    "MATCH (d:Spare {{name: 'Django'}}) INSERT (d)-[:TYPE_{index}]->(:Spare)"
                ))
                .unwrap();
            }
        });
        for _ in 0..4 {
            scope.spawn(|| {
                for _ in 0..20 {
                    assert_eq!(count(hops), hop_count);
                    assert_eq!(count(triangles), triangle_count);
                }
            });
        }
    });
}
