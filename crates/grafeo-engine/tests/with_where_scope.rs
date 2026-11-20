//! The WHERE of a Cypher WITH sees the variables bound before the WITH as
//! well as the WITH's own: `WITH a.name AS name WHERE a.age > 30` filters by
//! `a`, which the WITH does not pass on (openCypher TCK 2024.3 WithWhere1,
//! WithWhere2, WithWhere7, TriadicSelection1, Comparison2). The WHERE ran
//! after the projection, so it failed with "Undefined variable 'a'". What the
//! WITH does not project stays out of scope after it.
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test main with_where_scope::
//! ```

#![cfg(feature = "cypher")]

use grafeo_common::types::Value;
use grafeo_engine::GrafeoDB;

/// `(:X {name: 'A', k: 1})`, the same with `B` and 2, and `C` and 3; `A`
/// knows `B`.
fn database() -> GrafeoDB {
    let db = GrafeoDB::new_in_memory();
    db.execute_cypher(
        "CREATE (a:X {name: 'A', k: 1})-[:KNOWS]->(:X {name: 'B', k: 2}), (:X {name: 'C', k: 3})",
    )
    .unwrap();
    db
}

fn rows(db: &GrafeoDB, query: &str) -> (Vec<String>, Vec<Vec<Value>>) {
    let result = db
        .execute_cypher(query)
        .unwrap_or_else(|error| panic!("{query}: {error}"));
    (result.columns.clone(), result.rows().to_vec())
}

fn names(values: &[&str]) -> Vec<Vec<Value>> {
    values.iter().map(|name| vec![Value::from(*name)]).collect()
}

/// The WHERE reads a variable the WITH leaves out, alone and beside one it
/// projects; `RETURN *` gives only the WITH's columns.
#[test]
fn a_with_where_reads_variables_bound_before_it() {
    let db = database();
    for (query, expected) in [
        (
            "MATCH (a:X) WITH a.name AS name WHERE a.k = 2 RETURN *",
            names(&["B"]),
        ),
        (
            "MATCH (a:X) WITH a.name AS name WHERE name = 'B' OR a.k = 3 RETURN * ORDER BY name",
            names(&["B", "C"]),
        ),
        (
            "MATCH (other:X) OPTIONAL MATCH (a)-[r:KNOWS]->(other) WITH other WHERE r IS NULL \
             RETURN other.name AS name ORDER BY name",
            names(&["A", "C"]),
        ),
        (
            "MATCH (a:X) WITH DISTINCT a.k > 1 AS big WHERE a.k <> 1 RETURN *",
            vec![vec![Value::Bool(true)]],
        ),
    ] {
        let (columns, found) = rows(&db, query);
        assert_eq!(found, expected, "{query}");
        assert!(columns.len() == 1, "{query}: {columns:?}");
    }
}

/// An ORDER BY, SKIP or LIMIT after such a WITH still sorts by what the WHERE
/// read or by another variable of the input, and what follows sees only the
/// WITH's columns.
#[test]
fn order_by_skip_and_limit_after_it() {
    let db = database();
    for (query, expected) in [
        (
            "MATCH (a:X) WITH a.name AS name WHERE a.k > 1 ORDER BY a.k DESC RETURN *",
            names(&["C", "B"]),
        ),
        (
            "MATCH (a:X), (b:X {name: 'A'}) WITH a.name AS name WHERE a.k > 1 \
             ORDER BY b.k, a.k DESC RETURN *",
            names(&["C", "B"]),
        ),
        (
            "MATCH (a:X) WITH a.name AS name WHERE a.k > 1 ORDER BY a.k LIMIT 1 RETURN *",
            names(&["B"]),
        ),
        (
            "MATCH (a:X) WITH a.name AS name WHERE a.k >= 1 ORDER BY a.k SKIP 2 RETURN *",
            names(&["C"]),
        ),
    ] {
        let (columns, found) = rows(&db, query);
        assert_eq!(columns, ["name"], "{query}");
        assert_eq!(found, expected, "{query}");
    }
}

/// After the WITH, the variables it left out are gone, also those its WHERE
/// read.
#[test]
fn what_the_with_leaves_out_stays_out_of_scope() {
    let db = database();
    let error = db
        .execute_cypher("MATCH (a:X) WITH a.name AS name WHERE a.k = 2 RETURN a")
        .expect_err("a is not in scope after the WITH");
    assert_eq!(
        grafeo_common::utils::gqlstatus::GqlStatus::from(&error).as_str(),
        "42002",
        "{error}"
    );
    assert!(error.to_string().contains("'a'"), "{error}");
}

/// In a CALL body, such a WITH keeps the body's imports: the second MATCH
/// reads the imported `p`, with and without DISTINCT.
#[test]
fn a_with_where_in_a_call_body_keeps_the_imports() {
    let db = GrafeoDB::new_in_memory();
    db.execute_cypher(
        "CREATE (:X {name: 'A'})-[:KNOWS]->(:X {name: 'B', k: 2}), (:X {name: 'C'})-[:KNOWS]->(:X {name: 'D', k: 4})",
    )
    .unwrap();
    for distinct in ["", "DISTINCT "] {
        let query = format!(
            "MATCH (p:X {{name: 'A'}}) CALL {{ WITH p MATCH (p)-->(f) WITH {distinct}f.name AS n WHERE f.k > 1 MATCH (p)-->(c) RETURN n, c.name AS c }} RETURN n, c"
        );
        let (_, found) = rows(&db, &query);
        assert_eq!(
            found,
            vec![vec![Value::from("B"), Value::from("B")]],
            "{query}"
        );
    }
}

/// A subquery in the WHERE (a pattern comprehension, an EXISTS) reads a
/// variable the WITH leaves out as the row's: only `A` knows anyone. Run on
/// the projection alone, the pattern matched for every row.
#[test]
fn a_subquery_in_the_where_reads_the_rows_variables() {
    let db = database();
    for query in [
        "MATCH (a:X) WITH a.name AS k WHERE size([(a)-->(:X) | 1]) > 0 RETURN k",
        "MATCH (a:X) WITH a.name AS k WHERE EXISTS { (a)-->() } RETURN k",
        "MATCH (a:X) WITH DISTINCT a.name AS k WHERE EXISTS { (a)-->() } RETURN k",
        // A subquery inside a list predicate: which names it reads is not
        // known, so the WITH keeps every variable of its input.
        "MATCH (a:X) WITH a.name AS k WHERE any(x IN [1] WHERE EXISTS { (a)-->() }) RETURN k",
    ] {
        let (columns, found) = rows(&db, query);
        assert_eq!(columns, ["k"], "{query}");
        assert_eq!(found, names(&["A"]), "{query}");
    }
}

/// A parameter named by a number (`$1`, openCypher's decimal-integer
/// parameter name) is read as any other (TCK WithWhere2 [2]).
#[test]
fn a_parameter_named_by_a_number() {
    let db = database();
    let params = std::collections::HashMap::from([("1".to_string(), Value::Int64(2))]);
    let result = db
        .execute_cypher_with_params(
            "MATCH (a:X) WITH a WHERE a.k = $1 RETURN a.name AS name",
            params,
        )
        .unwrap();
    assert_eq!(result.rows(), names(&["B"]).as_slice());
}
