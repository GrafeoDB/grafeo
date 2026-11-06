//! Each language's rule for comparing values of two kinds (GA04 step b, 2026-11-05): GQL orders
//! them by rank, Cypher answers null for an ordering, Gremlin takes Cypher's rule (TinkerPop:
//! "returns FALSE for comparisons across types", and a filter drops false as it drops null), and
//! SQL/PGQ raises "values not comparable" (22G04) for every comparison across kinds, `=` included
//! (ISO/IEC 9075-2 8.2 Syntax Rule 3 and 9.11: the comparands must be comparable). The rule follows
//! the statement's language, so one stored graph answers each language by its own rule.
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test main comparison_by_language::
//! ```

use std::collections::HashMap;

use grafeo_common::types::Value;
use grafeo_common::utils::gqlstatus::GqlStatus;
use grafeo_engine::GrafeoDB;
use grafeo_engine::session::Session;

/// `k` of each kind: a string that spells no number, numbers, strings that spell numbers, a date,
/// and a node without `k`.
const MIXED: &str = "INSERT (:L {row: 0, k: 'a'}), (:L {row: 1, k: 3}), (:L {row: 2, k: '9'}), \
     (:L {row: 3, k: 7}), (:L {row: 4}), (:L {row: 5, k: DATE '2024-01-01'}), (:L {row: 6, k: '3'})";

/// Numbers only, of two numeric kinds, and a node without `k`.
const NUMBERS: &str =
    "INSERT (:N {row: 0, k: 2}), (:N {row: 1, k: 3.5e0}), (:N {row: 2}), (:N {row: 3, k: 7})";

/// Two properties of one kind (`P`) and of two kinds (`Q`), and nodes with no `k` (`E`). Each
/// label is its own graph part, so no case depends on the order an `AND` is evaluated in, which SQL
/// leaves open.
const PAIRS: &str = "INSERT (:P {row: 0, a: 1, b: 2}), (:Q {row: 0, a: 1, b: '1'}), (:E {row: 0}),      (:E {row: 1})";

fn graph() -> GrafeoDB {
    let db = GrafeoDB::new_in_memory();
    for statement in [MIXED, NUMBERS, PAIRS] {
        db.execute(statement).unwrap();
    }
    db
}

/// The sorted `row` numbers a query returns, or the status of its error.
fn rows(
    result: grafeo_common::utils::error::Result<grafeo_engine::database::QueryResult>,
) -> Result<Vec<i64>, String> {
    match result {
        Ok(result) => {
            let mut found: Vec<i64> = result
                .rows()
                .iter()
                .map(|row| match &row[0] {
                    Value::Int64(n) => *n,
                    other => panic!("a row number, not {other:?}"),
                })
                .collect();
            found.sort_unstable();
            Ok(found)
        }
        Err(error) => Err(GqlStatus::from(&error).as_str().to_string()),
    }
}

fn gql(session: &Session, query: &str) -> Result<Vec<i64>, String> {
    rows(session.execute(query))
}

fn cypher(session: &Session, query: &str) -> Result<Vec<i64>, String> {
    rows(session.execute_cypher(query))
}

/// The baseline the other languages differ from: on the same graph, GQL orders a string and a
/// date before every number and Cypher drops them.
#[test]
fn gql_and_cypher_answer_the_mixed_graph_by_their_rules() {
    let db = graph();
    let session = db.session();
    assert_eq!(
        gql(&session, "MATCH (n:L) WHERE n.k < 5 RETURN n.row"),
        Ok(vec![0, 1, 2, 5, 6])
    );
    assert_eq!(
        cypher(&session, "MATCH (n:L) WHERE n.k < 5 RETURN n.row"),
        Ok(vec![1])
    );
    assert_eq!(
        gql(&session, "MATCH (n:L) WHERE n.k <= 'b' RETURN n.row"),
        Ok(vec![0, 2, 5, 6])
    );
    assert_eq!(
        cypher(&session, "MATCH (n:L) WHERE n.k <= 'b' RETURN n.row"),
        Ok(vec![0, 2, 6])
    );
}

#[cfg(feature = "gremlin")]
mod gremlin {
    use super::*;

    fn gremlin(session: &Session, query: &str) -> Result<Vec<i64>, String> {
        rows(session.execute_gremlin(query))
    }

    fn gremlin_with(session: &Session, query: &str, p: Value) -> Result<Vec<i64>, String> {
        rows(session.execute_gremlin_with_params(query, HashMap::from([("p".to_string(), p)])))
    }

    /// Gremlin follows Cypher's rule: a value of another kind never passes an ordering step, where
    /// GQL's rank would let a string or a date below a number.
    #[test]
    fn gremlin_orders_only_within_a_kind() {
        let db = graph();
        let session = db.session();
        for (query, want) in [
            ("g.V().hasLabel('L').has('k', lt(5)).values('row')", vec![1]),
            (
                "g.V().hasLabel('L').has('k', lte(3)).values('row')",
                vec![1],
            ),
            (
                "g.V().hasLabel('L').has('k', gt(1)).values('row')",
                vec![1, 3],
            ),
            (
                "g.V().hasLabel('L').has('k', gte(3)).values('row')",
                vec![1, 3],
            ),
            (
                "g.V().hasLabel('L').has('k', lte('b')).values('row')",
                vec![0, 2, 6],
            ),
            (
                "g.V().hasLabel('L').has('k', gt('5')).values('row')",
                vec![0, 2],
            ),
        ] {
            assert_eq!(gremlin(&session, query), Ok(want), "`{query}`");
        }
    }

    /// Equality across kinds is false: `3` finds the number and not the string `'3'`, and the
    /// reverse, as a literal and as a parameter.
    #[test]
    fn gremlin_equality_keeps_kinds_apart() {
        let db = graph();
        let session = db.session();
        assert_eq!(
            gremlin(&session, "g.V().hasLabel('L').has('k', 3).values('row')"),
            Ok(vec![1])
        );
        assert_eq!(
            gremlin(&session, "g.V().hasLabel('L').has('k', '3').values('row')"),
            Ok(vec![6])
        );
        let query = "g.V().hasLabel('L').has('k', $p).values('row')";
        assert_eq!(gremlin_with(&session, query, Value::Int64(3)), Ok(vec![1]));
        assert_eq!(gremlin_with(&session, query, Value::from("3")), Ok(vec![6]));
        // A parameter inside a predicate step (`lt($p)`) is read as text today, a separate bug
        // (Gremlin parameters, Low): its ordering case joins this test with that fix.
    }

    /// Numbers of two kinds are one kind, ordered and equal by value.
    #[test]
    fn gremlin_compares_numbers_across_numeric_kinds() {
        let db = graph();
        let session = db.session();
        assert_eq!(
            gremlin(
                &session,
                "g.V().hasLabel('N').has('k', lt(3.6)).values('row')"
            ),
            Ok(vec![0, 1])
        );
        assert_eq!(
            gremlin(&session, "g.V().hasLabel('N').has('k', 2.0).values('row')"),
            Ok(vec![0])
        );
    }
}

#[cfg(feature = "sql-pgq")]
mod sql_pgq {
    use super::*;

    const NOT_COMPARABLE: &str = "22G04";

    fn sql(session: &Session, label: &str, condition: &str) -> Result<Vec<i64>, String> {
        rows(session.execute_sql(&format!(
            "SELECT * FROM GRAPH_TABLE (MATCH (n:{label}) WHERE {condition} COLUMNS (n.row AS row))"
        )))
    }

    /// Every comparison operator across kinds raises 22G04 when a row brings two kinds together,
    /// `=` and `<>` included, with a literal on either side.
    #[test]
    fn sql_pgq_raises_for_every_comparison_across_kinds() {
        let db = graph();
        let session = db.session();
        for op in ["=", "<>", "<", "<=", ">", ">="] {
            for condition in [
                format!("n.k {op} 5"),
                format!("5 {op} n.k"),
                format!("n.k {op} 'b'"),
            ] {
                assert_eq!(
                    sql(&session, "L", &condition),
                    Err(NOT_COMPARABLE.to_string()),
                    "`{condition}`"
                );
            }
        }
    }

    /// A property against a property: of one kind it compares, of two kinds it raises.
    #[test]
    fn sql_pgq_raises_for_two_properties_of_two_kinds() {
        let db = graph();
        let session = db.session();
        assert_eq!(sql(&session, "P", "n.a < n.b"), Ok(vec![0]));
        assert_eq!(
            sql(&session, "Q", "n.a = n.b"),
            Err(NOT_COMPARABLE.to_string())
        );
        assert_eq!(
            sql(&session, "Q", "n.a < n.b"),
            Err(NOT_COMPARABLE.to_string())
        );
    }

    /// A parameter of another kind raises too: SQL types a dynamic parameter from its context,
    /// and a property has no declared type to give it.
    #[test]
    fn sql_pgq_raises_for_a_parameter_of_another_kind() {
        let db = graph();
        let session = db.session();
        let query = "SELECT * FROM GRAPH_TABLE (MATCH (n:N) WHERE n.k = $p COLUMNS (n.row AS row))";
        let with = |p: Value| {
            rows(session.execute_sql_with_params(query, HashMap::from([("p".to_string(), p)])))
        };
        assert_eq!(with(Value::Int64(2)), Ok(vec![0]));
        assert_eq!(with(Value::from("2")), Err(NOT_COMPARABLE.to_string()));
    }

    /// No error where no two kinds meet: numbers of two numeric kinds are comparable (4.5.1), a
    /// null is unknown against anything.
    #[test]
    fn sql_pgq_compares_numbers_and_nulls_without_an_error() {
        let db = graph();
        let session = db.session();
        for (condition, want) in [
            ("n.k = 2.0", vec![0]),
            ("n.k < 3.6", vec![0, 1]),
            ("n.k >= 3", vec![1, 3]),
            ("n.k <> 7", vec![0, 1]),
        ] {
            assert_eq!(sql(&session, "N", condition), Ok(want), "`{condition}`");
        }
        // A missing `k` is unknown against a string, a number and a date, never an error.
        for condition in [
            "n.k < 'a'",
            "n.k = 5",
            "n.k > CAST('2024-01-01' AS DATE)",
            "n.k <> 5",
        ] {
            assert_eq!(sql(&session, "E", condition), Ok(vec![]), "`{condition}`");
        }
        assert_eq!(sql(&session, "L", "n.missing = 'a'"), Ok(vec![]));
    }

    /// Grouping and DISTINCT over a column of mixed kinds raise nothing and keep `'3'` and `3`
    /// apart (ISO/IEC 39075 22.12, which SQL/PGQ follows here: a SQL column has one type).
    #[test]
    fn sql_pgq_groups_a_mixed_column_without_an_error() {
        let db = graph();
        let session = db.session();
        let distinct = session
            .execute_sql("SELECT DISTINCT k FROM GRAPH_TABLE (MATCH (n:L) COLUMNS (n.k AS k))")
            .unwrap();
        assert_eq!(distinct.rows().len(), 7);
        let grouped = session
            .execute_sql(
                "SELECT k, COUNT(*) AS c FROM GRAPH_TABLE (MATCH (n:L) COLUMNS (n.k AS k)) GROUP BY k",
            )
            .unwrap();
        assert_eq!(grouped.rows().len(), 7);
        assert!(grouped.rows().iter().all(|row| row[1] == Value::Int64(1)));
    }
}
