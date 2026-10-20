//! `rows_scanned` counts the rows a statement reads from storage, before any
//! filter (#461): the nodes a scan, a label scan, an index lookup or a range
//! scan reads, and the edges an expand reads. It no longer repeats the rows
//! the statement returns, so a full scan behind a single count shows.
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test rows_scanned
//! ```

use grafeo_engine::GrafeoDB;
use grafeo_engine::database::QueryResult;

/// 1,000 `:Person {id, age}` (`age` is `id % 50`), 10 `:City {id}`, and a
/// `KNOWS` chain from Person 0 to Person 10 (10 edges).
fn graph() -> GrafeoDB {
    let db = GrafeoDB::new_in_memory();
    db.execute("UNWIND range(0, 999) AS i INSERT (:Person {id: i, age: i % 50})")
        .unwrap();
    db.execute("UNWIND range(0, 9) AS i INSERT (:City {id: i})")
        .unwrap();
    db.execute(
        "MATCH (a:Person), (b:Person) WHERE a.id < 10 AND b.id = a.id + 1 \
         INSERT (a)-[:KNOWS]->(b)",
    )
    .unwrap();
    db
}

fn scanned(result: &QueryResult) -> u64 {
    result
        .rows_scanned
        .expect("every query reports rows_scanned")
}

#[test]
fn a_count_reports_the_rows_it_read_not_the_one_it_returns() {
    let db = graph();
    let result = db.execute("MATCH (n:Person) RETURN count(n)").unwrap();
    assert_eq!(result.row_count(), 1);
    assert_eq!(scanned(&result), 1_000, "the label scan reads every Person");

    let result = db.execute("MATCH (n) RETURN count(n)").unwrap();
    assert_eq!(
        scanned(&result),
        1_010,
        "a scan without a label reads every node"
    );
}

#[test]
fn an_index_lookup_reads_its_hits_where_a_scan_reads_the_label() {
    let db = graph();
    let query = "MATCH (n:Person {id: 5}) RETURN n.id";
    assert_eq!(
        scanned(&db.execute(query).unwrap()),
        1_000,
        "no index: a scan"
    );

    db.create_property_index("id").unwrap();
    let result = db.execute(query).unwrap();
    assert_eq!(result.row_count(), 1);
    // The index holds `id` of every label: Person 5 and City 5 are read, and
    // the label check drops the city after the read.
    assert_eq!(scanned(&result), 2);

    let result = db
        .execute("MATCH (n:Person) WHERE n.id IN [1, 2, 3] RETURN n.id")
        .unwrap();
    assert_eq!(result.row_count(), 3);
    assert_eq!(
        scanned(&result),
        6,
        "each value's lookup reads a person and a city"
    );
}

#[test]
fn an_expand_counts_the_edges_it_reads() {
    let db = graph();
    let result = db
        .execute("MATCH (a:Person {id: 0})-[:KNOWS]->(b) RETURN b.id")
        .unwrap();
    assert_eq!(result.row_count(), 1);
    assert_eq!(
        scanned(&result),
        1_001,
        "the scan, then Person 0's one edge"
    );

    let result = db
        .execute("MATCH (a:Person)-[:KNOWS]->(b) RETURN count(b)")
        .unwrap();
    assert_eq!(
        scanned(&result),
        1_010,
        "the scan, then the 10 edges of the chain"
    );

    let result = db
        .execute("MATCH (a:Person {id: 0})-[:KNOWS]->{1,2}(b) RETURN b.id")
        .unwrap();
    assert_eq!(result.row_count(), 2);
    assert_eq!(scanned(&result), 1_002, "the scan, then one edge per hop");
}

#[test]
fn a_range_scan_counts_the_rows_the_range_returns() {
    let db = graph();
    db.create_property_index("id").unwrap();
    let result = db
        .execute("MATCH (n:Person) WHERE n.id >= 10 AND n.id < 20 RETURN n.id")
        .unwrap();
    assert_eq!(result.row_count(), 10);
    assert_eq!(scanned(&result), 10);
}

#[test]
fn a_statement_that_reads_nothing_reports_zero() {
    let db = graph();
    assert_eq!(scanned(&db.execute("RETURN 1").unwrap()), 0);
    assert_eq!(scanned(&db.execute("INSERT (:Tmp)").unwrap()), 0);
}

#[test]
fn a_transaction_reports_its_reads_too() {
    let db = graph();
    let mut session = db.session();
    session.begin_transaction().unwrap();
    session.execute("INSERT (:Person {id: 1000})").unwrap();
    let result = session.execute("MATCH (n:Person) RETURN count(n)").unwrap();
    assert_eq!(scanned(&result), 1_001, "its own pending node included");
    session.rollback().unwrap();
}

#[cfg(feature = "cypher")]
#[test]
fn cypher_counts_the_same_reads() {
    let db = graph();
    let session = db.session();
    let gql = session.execute("MATCH (n:Person) RETURN count(n)").unwrap();
    let cypher = session
        .execute_cypher("MATCH (n:Person) RETURN count(n)")
        .unwrap();
    assert_eq!(scanned(&cypher), scanned(&gql));
    assert_eq!(scanned(&cypher), 1_000);
}

#[cfg(feature = "sql-pgq")]
#[test]
fn sql_pgq_counts_its_reads() {
    let db = graph();
    let result = db
        .session()
        .execute_sql("SELECT * FROM GRAPH_TABLE (MATCH (c:City) COLUMNS (c.id AS id))")
        .unwrap();
    assert_eq!(result.row_count(), 10);
    assert_eq!(scanned(&result), 10);
}

#[cfg(all(feature = "sparql", feature = "triple-store"))]
#[test]
fn sparql_counts_the_triples_its_scans_read() {
    let db = GrafeoDB::new_in_memory();
    db.execute_sparql(
        r#"INSERT DATA {
            <http://ex.org/alix> <http://ex.org/age> 30 .
            <http://ex.org/gus> <http://ex.org/age> 25 .
            <http://ex.org/vincent> <http://ex.org/age> 35 .
            <http://ex.org/alix> <http://ex.org/name> "Alix" .
        }"#,
    )
    .unwrap();

    let result = db
        .execute_sparql("SELECT ?s WHERE { ?s <http://ex.org/age> ?a FILTER(?a > 28) }")
        .unwrap();
    assert_eq!(result.row_count(), 2);
    assert_eq!(
        scanned(&result),
        3,
        "the scan reads every age before the filter"
    );

    let result = db
        .execute_sparql("SELECT ?p WHERE { <http://ex.org/alix> ?p ?o }")
        .unwrap();
    assert_eq!(result.row_count(), 2);
    assert_eq!(scanned(&result), 2, "a bound subject reads its own triples");
}
