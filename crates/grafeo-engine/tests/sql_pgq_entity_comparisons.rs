//! SQL/PGQ refuses a comparison of a graph element with a value of another kind, `=` and `<>`
//! included, with "values not comparable" (22G04), as it refuses every comparison across kinds
//! (GA04; ISO/IEC 9075-2 8.2 Syntax Rule 3: the comparands must be comparable). A null on either
//! side gives null, so the row is left out without an error. Two nodes compare by identity.
//! 0.6.1 refused these too; 0.6.2 answered `n = 1` with false and `n <> 1` with true.
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test main sql_pgq_entity_comparisons::
//! ```

#![cfg(all(feature = "sql-pgq", feature = "gql"))]

use grafeo_common::utils::gqlstatus::GqlStatus;
use grafeo_engine::GrafeoDB;

const NOT_COMPARABLE: &str = "22G04";

/// Alix knows Gus.
fn graph() -> GrafeoDB {
    let db = GrafeoDB::new_in_memory();
    db.execute("INSERT (:P {k: 1, name: 'Alix'})-[:R {w: 1}]->(:P {k: 2, name: 'Gus'})")
        .unwrap();
    db
}

/// An operand of a comparison in the `WHERE` of `MATCH (n:P)-[e:R]->(m:P)`.
#[derive(Clone, Copy, Debug)]
enum Operand {
    /// The node `n`.
    Node,
    /// The other node, `m`.
    OtherNode,
    /// The edge `e`.
    Edge,
    Integer,
    Float,
    Text,
    Boolean,
    Date,
    /// A property of a node, `m.name`.
    Property,
    Null,
}

const OPERANDS: [Operand; 10] = [
    Operand::Node,
    Operand::OtherNode,
    Operand::Edge,
    Operand::Integer,
    Operand::Float,
    Operand::Text,
    Operand::Boolean,
    Operand::Date,
    Operand::Property,
    Operand::Null,
];

impl Operand {
    fn text(self) -> &'static str {
        match self {
            Self::Node => "n",
            Self::OtherNode => "m",
            Self::Edge => "e",
            Self::Integer => "1",
            Self::Float => "1.5",
            Self::Text => "'Alix'",
            Self::Boolean => "TRUE",
            // SQL/PGQ has no DATE literal; a CAST gives a date value.
            Self::Date => "CAST('2024-01-01' AS DATE)",
            Self::Property => "m.name",
            Self::Null => "NULL",
        }
    }

    /// The kind GA04 compares by: a node, an edge, or a value (whose kind differs from either).
    fn kind(self) -> Option<&'static str> {
        match self {
            Self::Node | Self::OtherNode => Some("node"),
            Self::Edge => Some("edge"),
            Self::Null => None,
            Self::Integer
            | Self::Float
            | Self::Text
            | Self::Boolean
            | Self::Date
            | Self::Property => Some("value"),
        }
    }

    fn is_element(self) -> bool {
        matches!(self, Self::Node | Self::OtherNode | Self::Edge)
    }
}

/// What a `WHERE left op right` over the one path gives: the names of the rows it keeps, or the
/// GQLSTATUS of its error.
fn outcome(db: &GrafeoDB, left: Operand, op: &str, right: Operand) -> Result<Vec<String>, String> {
    let query = format!(
        "SELECT * FROM GRAPH_TABLE (MATCH (n:P)-[e:R]->(m:P) WHERE {} {op} {} \
         COLUMNS (n.name AS name))",
        left.text(),
        right.text()
    );
    match db.execute_sql(&query) {
        Ok(result) => Ok(result
            .rows()
            .iter()
            .map(|row| row[0].as_str().unwrap_or("not a name").to_string())
            .collect()),
        Err(error) => Err(GqlStatus::from(&error).as_str().to_string()),
    }
}

/// The case: a node against an integer, by `=` and `<>`.
#[test]
fn a_node_compared_with_an_integer_is_refused() {
    let db = graph();
    for op in ["=", "<>"] {
        assert_eq!(
            outcome(&db, Operand::Node, op, Operand::Integer),
            Err(NOT_COMPARABLE.to_string()),
            "n {op} 1"
        );
    }
}

/// Every element against every operand, both ways round, by `=` and `<>`: refused where the kinds
/// differ, null (no row, no error) against a null, and two nodes compare by identity.
#[test]
fn an_element_compared_across_kinds_is_refused() {
    let db = graph();
    let mut wrong = Vec::new();
    for element in OPERANDS.into_iter().filter(|o| o.is_element()) {
        for other in OPERANDS {
            for (left, right) in [(element, other), (other, element)] {
                for op in ["=", "<>"] {
                    let got = outcome(&db, left, op, right);
                    let want = match (left.kind(), right.kind()) {
                        (None, _) | (_, None) => Ok(Vec::new()),
                        (Some(a), Some(b)) if a != b => Err(NOT_COMPARABLE.to_string()),
                        // One kind: only two elements share one here.
                        _ => {
                            let same = left.text() == right.text();
                            Ok(if same == (op == "=") {
                                vec!["Alix".to_string()]
                            } else {
                                Vec::new()
                            })
                        }
                    };
                    if got != want {
                        wrong.push(format!(
                            "{} {op} {}: {got:?}, want {want:?}",
                            left.text(),
                            right.text()
                        ));
                    }
                }
            }
        }
    }
    assert!(
        wrong.is_empty(),
        "{} wrong:\n{}",
        wrong.len(),
        wrong.join("\n")
    );
}

/// GQL keeps its own rule for the same comparison: no error, and a node equals no integer, so
/// `=` holds for no row and `<>` for every row.
#[test]
fn gql_answers_a_node_compared_with_an_integer() {
    let db = graph();
    for (op, wanted) in [("=", vec![]), ("<>", vec!["Alix", "Gus"])] {
        let result = db
            .execute(&format!(
                "MATCH (n:P) WHERE n {op} 1 RETURN n.name AS name ORDER BY name"
            ))
            .unwrap_or_else(|error| panic!("n {op} 1: {error}"));
        let names: Vec<grafeo_common::types::Value> =
            result.rows().iter().map(|row| row[0].clone()).collect();
        let wanted: Vec<grafeo_common::types::Value> = wanted
            .into_iter()
            .map(grafeo_common::types::Value::from)
            .collect();
        assert_eq!(names, wanted, "n {op} 1");
    }
}
