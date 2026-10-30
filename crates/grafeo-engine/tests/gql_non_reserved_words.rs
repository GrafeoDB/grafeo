//! ISO GQL's non-reserved words (ISO/IEC 39075, the `<non-reserved word>`
//! list, as the openGQL grammar has it) are identifiers wherever a name
//! goes: an alias after AS, a LET variable, a MATCH node and edge variable.
//! `RETURN 1 AS trail` failed with "Expected alias name".
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test gql_non_reserved_words
//! ```

#![cfg(feature = "gql")]

use grafeo_common::types::Value;
use grafeo_engine::GrafeoDB;

const NON_RESERVED: [&str; 47] = [
    "acyclic",
    "binding",
    "bindings",
    "connecting",
    "destination",
    "different",
    "directed",
    "edge",
    "edges",
    "element",
    "elements",
    "first",
    "graph",
    "groups",
    "keep",
    "label",
    "labeled",
    "labels",
    "last",
    "nfc",
    "nfd",
    "nfkc",
    "nfkd",
    "no",
    "node",
    "normalized",
    "only",
    "ordinality",
    "property",
    "read",
    "relationship",
    "relationships",
    "repeatable",
    "shortest",
    "simple",
    "source",
    "table",
    "to",
    "trail",
    "transaction",
    "type",
    "undirected",
    "vertex",
    "walk",
    "without",
    "write",
    "zone",
];

/// Every word as an alias, a LET variable, a node variable and an edge
/// variable, lower and upper case: the cases that fail.
#[test]
fn every_non_reserved_word_is_a_name() {
    let db = GrafeoDB::new_in_memory();
    db.execute("INSERT (:P {v: 7})-[:R {w: 3}]->(:P {v: 8})")
        .unwrap();
    let mut wrong = Vec::new();
    for word in NON_RESERVED {
        for name in [word.to_string(), word.to_uppercase()] {
            for (shape, query, expected) in [
                ("an alias", format!("RETURN 1 AS {name}"), Value::Int64(1)),
                (
                    "a LET variable",
                    format!("LET {name} = 2 RETURN {name}"),
                    Value::Int64(2),
                ),
                (
                    "a node variable",
                    format!("MATCH ({name}:P) RETURN min({name}.v) AS v"),
                    Value::Int64(7),
                ),
                (
                    "an edge variable",
                    format!("MATCH ()-[{name}:R]->() RETURN {name}.w AS w"),
                    Value::Int64(3),
                ),
                (
                    "a FOR variable",
                    format!("FOR {name} IN [4] RETURN {name}"),
                    Value::Int64(4),
                ),
            ] {
                match db.execute(&query) {
                    Ok(result) => {
                        let got = result.rows().first().map(|row| row[0].clone());
                        if got != Some(expected.clone()) {
                            wrong.push(format!("{name} as {shape}: {got:?}"));
                        }
                    }
                    Err(error) => wrong.push(format!(
                        "{name} as {shape}: {}",
                        error.to_string().lines().next().unwrap_or("")
                    )),
                }
            }
        }
    }
    assert!(
        wrong.is_empty(),
        "{} of {} cases fail:\n{}",
        wrong.len(),
        NON_RESERVED.len() * 10,
        wrong.join("\n")
    );
}

/// Control: a reserved word stays refused as a bare alias (MATCH is not a
/// name), so the test above checks the list, not a parser that takes
/// anything.
#[test]
fn a_reserved_word_is_not_a_bare_alias() {
    let db = GrafeoDB::new_in_memory();
    for word in ["MATCH", "RETURN", "WHERE"] {
        assert!(
            db.execute(&format!("RETURN 1 AS {word}")).is_err(),
            "{word} was taken as an alias"
        );
    }
}
