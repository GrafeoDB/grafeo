//! An operator or a list index given an operand of the wrong kind raises "invalid value type" (22G12)
//! in 0.6.2, where it gave null (ISO/IEC 39075 4.14.4: the type tests generated for dynamically typed
//! operands raise 22G12 in their ELSE arm; openCypher raises a type error). Cypher's `+` of a list and
//! one value appends it, as openCypher defines. A null operand stays null. Each case runs with its
//! operands written as literals, read from a stored node and passed as parameters, in GQL and Cypher.
//! Functions keep null until 0.7.0 and are not here.
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test main wrong_kind_operators::
//! ```

use std::collections::HashMap;

use grafeo_common::types::{Date, Value};
use grafeo_common::utils::gqlstatus::GqlStatus;
use grafeo_engine::GrafeoDB;
use grafeo_engine::session::Session;

#[derive(Clone, Copy, Debug)]
enum Language {
    Gql,
    Cypher,
}

/// What a case gives.
#[derive(Debug, PartialEq)]
enum Want {
    InvalidValueType,
    Rows(Value),
}

/// One operand: its literal in GQL and Cypher, its stored property, its value as a parameter.
struct Operand {
    gql: &'static str,
    cypher: &'static str,
    property: &'static str,
    value: Value,
}

fn operands() -> HashMap<&'static str, Operand> {
    let date = Value::Date(Date::parse("2024-01-01").unwrap());
    let list = |items: &[i64]| Value::List(items.iter().map(|i| Value::Int64(*i)).collect());
    HashMap::from([
        (
            "true",
            Operand {
                gql: "true",
                cypher: "true",
                property: "t",
                value: Value::Bool(true),
            },
        ),
        (
            "1",
            Operand {
                gql: "1",
                cypher: "1",
                property: "one",
                value: Value::Int64(1),
            },
        ),
        (
            "2",
            Operand {
                gql: "2",
                cypher: "2",
                property: "two",
                value: Value::Int64(2),
            },
        ),
        (
            "3",
            Operand {
                gql: "3",
                cypher: "3",
                property: "three",
                value: Value::Int64(3),
            },
        ),
        (
            "date",
            Operand {
                gql: "DATE '2024-01-01'",
                cypher: "date('2024-01-01')",
                property: "d",
                value: date,
            },
        ),
        (
            "[1]",
            Operand {
                gql: "[1]",
                cypher: "[1]",
                property: "l1",
                value: list(&[1]),
            },
        ),
        (
            "[1, 2, 3]",
            Operand {
                gql: "[1, 2, 3]",
                cypher: "[1, 2, 3]",
                property: "l3",
                value: list(&[1, 2, 3]),
            },
        ),
        (
            "2.0",
            Operand {
                gql: "2.0e0",
                cypher: "2.0",
                property: "f",
                value: Value::Float64(2.0),
            },
        ),
        (
            "'a'",
            Operand {
                gql: "'a'",
                cypher: "'a'",
                property: "s",
                value: Value::from("a"),
            },
        ),
        (
            "null",
            Operand {
                gql: "null",
                cypher: "null",
                property: "missing",
                value: Value::Null,
            },
        ),
    ])
}

const STORED: &str = "INSERT (:W {t: true, one: 1, two: 2, three: 3, d: DATE '2024-01-01', \
     l1: [1], l3: [1, 2, 3], f: 2.0e0, s: 'a'})";

/// The template with `{0}` and `{1}` for the operands, the operands, and what GQL and Cypher give.
fn cases() -> Vec<(&'static str, Vec<&'static str>, Want, Want)> {
    use Want::{InvalidValueType as Bad, Rows};
    let list = |items: &[i64]| Value::List(items.iter().map(|i| Value::Int64(*i)).collect());
    vec![
        ("{0} + {1}", vec!["true", "1"], Bad, Bad),
        ("{0} + {1}", vec!["1", "true"], Bad, Bad),
        ("{0} + {1}", vec!["date", "1"], Bad, Bad),
        ("{0} + {1}", vec!["[1]", "3"], Bad, Rows(list(&[1, 3]))),
        ("{0}[{1}]", vec!["[1, 2, 3]", "2.0"], Bad, Bad),
        ("{0}[{1}]", vec!["[1, 2, 3]", "'a'"], Bad, Bad),
        ("-({0})", vec!["'a'"], Bad, Bad),
        ("{0} * {1}", vec!["'a'", "2"], Bad, Bad),
        ("{0} - {1}", vec!["[1]", "[1]"], Bad, Bad),
        // A null operand stays null, also beside a wrong kind.
        (
            "{0} + {1}",
            vec!["null", "1"],
            Rows(Value::Null),
            Rows(Value::Null),
        ),
        (
            "{0} + {1}",
            vec!["true", "null"],
            Rows(Value::Null),
            Rows(Value::Null),
        ),
        (
            "{0}[{1}]",
            vec!["[1, 2, 3]", "null"],
            Rows(Value::Null),
            Rows(Value::Null),
        ),
        ("-({0})", vec!["null"], Rows(Value::Null), Rows(Value::Null)),
    ]
}

fn answer(
    session: &Session,
    language: Language,
    query: &str,
    params: HashMap<String, Value>,
) -> Result<Value, String> {
    let result = match language {
        Language::Gql => session.execute_with_params(query, params),
        Language::Cypher => session.execute_cypher_with_params(query, params),
    };
    match result {
        Ok(result) => Ok(result.rows()[0][0].clone()),
        Err(error) => Err(GqlStatus::from(&error).as_str().to_string()),
    }
}

fn judged(found: &Result<Value, String>, want: &Want) -> bool {
    match (found, want) {
        (Err(status), Want::InvalidValueType) => status == "22G12",
        (Ok(value), Want::Rows(expected)) => value == expected,
        _ => false,
    }
}

#[test]
fn a_wrong_kind_raises_invalid_value_type_and_a_null_stays_null() {
    let db = GrafeoDB::new_in_memory();
    db.execute(STORED).unwrap();
    let session = db.session();
    let operands = operands();
    let mut wrong = Vec::new();
    for (template, names, gql_want, cypher_want) in cases() {
        for (language, want) in [(Language::Gql, &gql_want), (Language::Cypher, &cypher_want)] {
            let fill = |pieces: &dyn Fn(&Operand, usize) -> String| -> String {
                let mut text = template.to_string();
                for (at, name) in names.iter().enumerate() {
                    text = text.replace(&format!("{{{at}}}"), &pieces(&operands[name], at));
                }
                text
            };
            let literal = fill(&|operand, _| {
                match language {
                    Language::Gql => operand.gql,
                    Language::Cypher => operand.cypher,
                }
                .to_string()
            });
            let stored = fill(&|operand, _| format!("w.{}", operand.property));
            let parametric = fill(&|_, at| format!("$p{at}"));
            let params: HashMap<String, Value> = names
                .iter()
                .enumerate()
                .map(|(at, name)| (format!("p{at}"), operands[name].value.clone()))
                .collect();
            for (form, query, params) in [
                ("literal", format!("RETURN {literal} AS r"), HashMap::new()),
                (
                    "stored",
                    format!("MATCH (w:W) RETURN {stored} AS r"),
                    HashMap::new(),
                ),
                ("parameter", format!("RETURN {parametric} AS r"), params),
            ] {
                let found = answer(&session, language, &query, params);
                if !judged(&found, want) {
                    wrong.push(format!(
                        "{language:?}, {form}: {query}: {found:?}, want {want:?}"
                    ));
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
