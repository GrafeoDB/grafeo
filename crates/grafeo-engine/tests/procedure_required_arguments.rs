//! A procedure's required argument is required: a CALL that omits it, or passes null for it,
//! positionally or in the map form, is refused with 22023 "invalid parameter value" naming the
//! procedure and the argument, never run on a default. `grafeo.subgraph_isomorphism()` without
//! its pattern counted the empty pattern, `{count: 1}`, as a success. One check in the shared
//! argument path (the one the domain check runs in), driven by the catalogue, so a new procedure's
//! required parameter is covered without a row here.
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test main procedure_required_arguments::
//! ```

#![cfg(all(feature = "gql", feature = "algos"))]

use grafeo_adapters::plugins::{ParameterDef, ParameterType};
use grafeo_common::types::Value;
use grafeo_common::utils::gqlstatus::GqlStatus;
use grafeo_engine::GrafeoDB;
use grafeo_engine::procedures::BuiltinProcedures;

const INVALID_PARAMETER_VALUE: &str = "22023";

/// Two linked nodes, and the id of the first.
fn graph() -> (GrafeoDB, i64) {
    let db = GrafeoDB::new_in_memory();
    db.execute("INSERT (:Doc {name: 'Alix'})-[:E {w: 1}]->(:Doc {name: 'Gus'})")
        .unwrap();
    let node = match db
        .execute("MATCH (d:Doc {name: 'Alix'}) RETURN id(d) AS id")
        .unwrap()
        .rows()[0][0]
    {
        Value::Int64(id) => id,
        ref other => panic!("{other:?}"),
    };
    (db, node)
}

/// A value of the parameter's type that its domain accepts.
fn sample(parameter: &ParameterDef, node: i64) -> String {
    match parameter.param_type {
        ParameterType::Integer => "1".to_string(),
        ParameterType::Float => "0.5".to_string(),
        ParameterType::String => "'Doc'".to_string(),
        ParameterType::Boolean => "true".to_string(),
        ParameterType::NodeId => node.to_string(),
        ParameterType::List => "[1.0, 0.0]".to_string(),
        ParameterType::Map => "{}".to_string(),
    }
}

/// The CALLs that leave `missing` out or pass null for it, positionally and in the map form.
fn calls_without(
    procedure: &str,
    parameters: &[ParameterDef],
    missing: &str,
    node: i64,
) -> Vec<String> {
    let at = parameters
        .iter()
        .position(|p| p.name == missing)
        .expect("a parameter of the procedure");
    let last_required = parameters.iter().rposition(|p| p.required).unwrap_or(at);
    let before: Vec<String> = parameters[..at].iter().map(|p| sample(p, node)).collect();
    let with_null: Vec<String> = parameters[..=last_required.max(at)]
        .iter()
        .map(|p| {
            if p.name == missing {
                "null".to_string()
            } else {
                sample(p, node)
            }
        })
        .collect();
    let others: Vec<String> = parameters
        .iter()
        .filter(|p| p.required && p.name != missing)
        .map(|p| format!("{}: {}", p.name, sample(p, node)))
        .collect();
    let mut nulled = others.clone();
    nulled.push(format!("{missing}: null"));
    vec![
        format!("CALL {procedure}({})", before.join(", ")),
        format!("CALL {procedure}({})", with_null.join(", ")),
        format!("CALL {procedure}({{{}}})", others.join(", ")),
        format!("CALL {procedure}({{{}}})", nulled.join(", ")),
    ]
}

fn refusal(db: &GrafeoDB, query: &str) -> Option<(String, String)> {
    match db.execute(query) {
        Err(error) => Some((
            GqlStatus::from(&error).as_str().to_string(),
            error.to_string(),
        )),
        Ok(_) => None,
    }
}

/// The case: `grafeo.subgraph_isomorphism` without its pattern, or with half of it, is refused
/// naming what is missing; with both it runs.
#[test]
fn subgraph_isomorphism_without_its_pattern_is_refused() {
    let (db, _) = graph();
    for (query, missing) in [
        ("CALL grafeo.subgraph_isomorphism()", "pattern_edges"),
        (
            "CALL grafeo.subgraph_isomorphism({pattern_nodes: 2})",
            "pattern_edges",
        ),
        (
            "CALL grafeo.subgraph_isomorphism({pattern_edges: '0-1'})",
            "pattern_nodes",
        ),
        ("CALL grafeo.subgraph_isomorphism('0-1')", "pattern_nodes"),
        ("CALL grafeo.subgraph_isomorphism(null, 2)", "pattern_edges"),
    ] {
        let (status, message) =
            refusal(&db, query).unwrap_or_else(|| panic!("{query}: ran without {missing}"));
        assert_eq!(status, INVALID_PARAMETER_VALUE, "{query}: {message}");
        assert!(
            message.contains("grafeo.subgraph_isomorphism") && message.contains(missing),
            "{query}: names the procedure and {missing}: {message}"
        );
    }
    db.execute("CALL grafeo.subgraph_isomorphism('0-1', 2)")
        .unwrap();
}

/// The class: every required parameter of every procedure in the catalogue, left out and given
/// null, positionally and in the map form, is refused with 22023 naming the procedure and the
/// argument.
#[test]
fn every_missing_required_argument_is_refused_naming_it() {
    let (db, node) = graph();
    let mut wrong = Vec::new();
    let mut covered = Vec::new();
    for procedure in BuiltinProcedures::new().list() {
        for parameter in procedure.parameters.iter().filter(|p| p.required) {
            covered.push(format!("{}.{}", procedure.name, parameter.name));
            for query in calls_without(
                &procedure.name,
                &procedure.parameters,
                &parameter.name,
                node,
            ) {
                match refusal(&db, &query) {
                    Some((status, message))
                        if status == INVALID_PARAMETER_VALUE
                            && message.contains(&procedure.name)
                            && message.contains(&parameter.name) => {}
                    other => wrong.push(format!(
                        "{query}: {other:?}, want 22023 naming {} and {}",
                        procedure.name, parameter.name
                    )),
                }
            }
        }
    }
    for known in [
        "grafeo.subgraph_isomorphism.pattern_edges",
        "grafeo.subgraph_isomorphism.pattern_nodes",
        "grafeo.bfs.start",
        "grafeo.max_flow.sink",
        "grafeo.search.vector.label",
    ] {
        assert!(covered.iter().any(|c| c == known), "{known} in {covered:?}");
    }
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}

/// The message names the argument's type, and says when a null was given.
#[test]
fn the_message_names_the_type_and_a_null() {
    let (db, _) = graph();
    let (_, omitted) = refusal(&db, "CALL grafeo.subgraph_isomorphism()").unwrap();
    assert!(
        omitted.contains("missing required argument 'pattern_edges'") && omitted.contains("STRING"),
        "{omitted}"
    );
    assert!(!omitted.contains("got null"), "{omitted}");
    let (_, nulled) = refusal(&db, "CALL grafeo.subgraph_isomorphism(null, 2)").unwrap();
    assert!(nulled.ends_with(", got null"), "{nulled}");
    let (_, node) = refusal(&db, "CALL grafeo.bfs()").unwrap();
    assert!(
        node.contains("grafeo.bfs: missing required argument 'start' (NODE ID)"),
        "{node}"
    );
}
