//! Every numeric procedure parameter has a domain, and a value outside it is refused with 22023
//! "invalid parameter value" naming the procedure and the argument (ISO/IEC 9075-2:2023 13.3 GR 7),
//! never run silently (the design of 2026-11-06: one check, from the catalogue). 0.6.1 ran
//! `grafeo.pagerank` with a damping of 2.0, -1.0 or NaN and returned meaningless scores as a
//! success. The table here is the decided domains; the class test holds it equal to the registry
//! both ways, so a new numeric parameter fails until it gets a row, and every probe comes from the
//! row: just outside each bound, NaN and both infinities for floats (refused), each inclusive bound
//! (accepted), positionally and in the map form. A null argument, however it is computed, takes the
//! default of an optional parameter.
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test main procedure_parameter_domains::
//! ```

#![cfg(all(feature = "gql", feature = "algos"))]

use std::collections::HashMap;

use grafeo_common::types::Value;
use grafeo_common::utils::gqlstatus::GqlStatus;
use grafeo_engine::GrafeoDB;
use grafeo_engine::procedures::BuiltinProcedures;

const INVALID_PARAMETER_VALUE: &str = "22023";

#[derive(Clone, Copy, Debug, PartialEq)]
enum Domain {
    /// `>= 0`, finite for a float.
    NonNegative,
    /// `[0, 1]`.
    Unit,
    /// `>=` the value of another argument.
    AtLeast(&'static str),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Kind {
    Integer,
    Float,
}

/// Procedure, parameter, kind, domain: the decided domains, each confirmed against its algorithm.
const DOMAINS: &[(&str, &str, Kind, Domain)] = &[
    ("grafeo.pagerank", "damping", Kind::Float, Domain::Unit),
    (
        "grafeo.pagerank",
        "max_iterations",
        Kind::Integer,
        Domain::NonNegative,
    ),
    (
        "grafeo.pagerank",
        "tolerance",
        Kind::Float,
        Domain::NonNegative,
    ),
    (
        "grafeo.louvain",
        "resolution",
        Kind::Float,
        Domain::NonNegative,
    ),
    (
        "grafeo.label_propagation",
        "max_iterations",
        Kind::Integer,
        Domain::NonNegative,
    ),
    (
        "grafeo.clustering_coefficient",
        "parallel_threshold",
        Kind::Integer,
        Domain::NonNegative,
    ),
    ("grafeo.kcore", "k", Kind::Integer, Domain::NonNegative),
    ("grafeo.ktruss", "k", Kind::Integer, Domain::NonNegative),
    (
        "grafeo.total_triangles",
        "parallel_threshold",
        Kind::Integer,
        Domain::NonNegative,
    ),
    (
        "grafeo.subgraph_isomorphism",
        "pattern_nodes",
        Kind::Integer,
        Domain::NonNegative,
    ),
    (
        "grafeo.search.vector",
        "k",
        Kind::Integer,
        Domain::NonNegative,
    ),
    (
        "grafeo.search.text",
        "k",
        Kind::Integer,
        Domain::NonNegative,
    ),
    ("grafeo.search.mmr", "k", Kind::Integer, Domain::NonNegative),
    (
        "grafeo.search.mmr",
        "fetch_k",
        Kind::Integer,
        Domain::AtLeast("k"),
    ),
    ("grafeo.search.mmr", "lambda", Kind::Float, Domain::Unit),
];

/// A valid value for each parameter a call must name before the one it probes, or that the
/// procedure requires.
fn valid(procedure: &str, parameter: &str) -> Option<&'static str> {
    Some(match (procedure, parameter) {
        (_, "label") => "'Doc'",
        ("grafeo.search.text", "property") => "'body'",
        (_, "property") => "'emb'",
        ("grafeo.search.text", "query") => "'graph'",
        (_, "query") => "[1.0, 0.0]",
        ("grafeo.search.mmr", "k") => "2",
        ("grafeo.search.mmr", "fetch_k") => "3",
        ("grafeo.kcore", "k") => "1",
        // No edges, so every node count from 0 holds the pattern (`'0-1'` with fewer than two
        // nodes is refused).
        ("grafeo.subgraph_isomorphism", "pattern_edges") => "''",
        _ => return None,
    })
}

/// A graph for the algorithms (a triangle and a separate edge) and documents with a vector and a
/// text index for the search procedures.
fn graph() -> GrafeoDB {
    let db = GrafeoDB::new_in_memory();
    db.execute(
        "INSERT (a:Person {name: 'Alix'})-[:KNOWS]->(b:Person {name: 'Gus'})-[:KNOWS]->\
         (c:Person {name: 'Vincent'})-[:KNOWS]->(a), \
         (:Person {name: 'Jules'})-[:KNOWS]->(:Person {name: 'Mia'})",
    )
    .unwrap();
    db.execute(
        "INSERT (:Doc {name: 'Amsterdam', emb: vector([1.0, 0.0]), body: 'graph database'}), \
         (:Doc {name: 'Berlin', emb: vector([0.0, 1.0]), body: 'rust compiler'}), \
         (:Doc {name: 'Paris', emb: vector([0.8, 0.6]), body: 'graph theory'})",
    )
    .unwrap();
    db.create_vector_index("Doc", "emb", Some(2), Some("euclidean"), None, None, None)
        .unwrap();
    db.create_text_index("Doc", "body").unwrap();
    db
}

/// The registry's parameters of a procedure, in order: name, type, required, default.
fn parameters(procedure: &str) -> Vec<(String, String, bool, Option<String>)> {
    let info = BuiltinProcedures::new()
        .list()
        .into_iter()
        .find(|info| info.name == procedure)
        .unwrap_or_else(|| panic!("no procedure {procedure}"));
    info.parameters
        .iter()
        .map(|p| {
            (
                p.name.clone(),
                format!("{:?}", p.param_type),
                p.required,
                p.default.clone(),
            )
        })
        .collect()
}

/// The call with `$x` for `parameter`, positionally (every parameter before it named with a valid
/// value or its default) or in the map form (only the required ones and it).
fn call(procedure: &str, parameter: &str, positional: bool) -> String {
    let params = parameters(procedure);
    let mut args = Vec::new();
    let mut entries = Vec::new();
    for (name, _, required, default) in &params {
        if name == parameter {
            args.push("$x".to_string());
            entries.push(format!("{name}: $x"));
            if positional {
                break;
            }
            continue;
        }
        let value = valid(procedure, name)
            .map(str::to_string)
            .or_else(|| default.clone());
        if *required {
            let value = value.unwrap_or_else(|| panic!("{procedure}.{name}: no valid value"));
            args.push(value.clone());
            entries.push(format!("{name}: {value}"));
        } else if positional {
            args.push(value.unwrap_or_else(|| panic!("{procedure}.{name}: nothing to fill")));
        } else if let Some(value) = valid(procedure, name) {
            // An optional argument another one's domain depends on keeps its valid value.
            entries.push(format!("{name}: {value}"));
        }
    }
    let inside = if positional {
        args.join(", ")
    } else {
        format!("{{{}}}", entries.join(", "))
    };
    format!("CALL {procedure}({inside})")
}

fn run(db: &GrafeoDB, query: &str, x: Value) -> Result<usize, (String, String)> {
    let params = HashMap::from([("x".to_string(), x)]);
    db.execute_with_params(query, params)
        .map(|result| result.rows().len())
        .map_err(|error| {
            (
                GqlStatus::from(&error).as_str().to_string(),
                error.to_string(),
            )
        })
}

/// Values just outside each bound (and NaN and the infinities for a float), and values on each
/// inclusive bound.
fn probes(procedure: &str, kind: Kind, domain: Domain) -> (Vec<Value>, Vec<Value>) {
    let int = |v: i64| Value::Int64(v);
    let float = |v: f64| Value::Float64(v);
    match (kind, domain) {
        (Kind::Integer, Domain::NonNegative) => {
            (vec![int(-1), int(i64::MIN)], vec![int(0), int(1)])
        }
        (Kind::Float, Domain::NonNegative) => (
            vec![
                float(-1e-9),
                float(-1.0),
                float(f64::NAN),
                float(f64::INFINITY),
                float(f64::NEG_INFINITY),
            ],
            vec![float(0.0), float(1.0)],
        ),
        (Kind::Float, Domain::Unit) => (
            vec![
                float(-1e-9),
                float(1.0 + 1e-9),
                float(2.0),
                float(f64::NAN),
                float(f64::INFINITY),
                float(f64::NEG_INFINITY),
            ],
            vec![float(0.0), float(1.0), float(0.5)],
        ),
        (Kind::Integer, Domain::AtLeast(other)) => {
            let floor: i64 = valid(procedure, other).unwrap().parse().unwrap();
            (
                vec![int(floor - 1), int(-1)],
                vec![int(floor), int(floor + 1)],
            )
        }
        (kind, domain) => panic!("no probes for {kind:?} {domain:?}"),
    }
}

fn holds(kind: Kind, domain: Domain, text: &str) -> bool {
    let value: f64 = text.parse().unwrap_or_else(|_| panic!("a default {text}"));
    if kind == Kind::Integer && value.fract() != 0.0 {
        return false;
    }
    match domain {
        Domain::NonNegative => value.is_finite() && value >= 0.0,
        Domain::Unit => (0.0..=1.0).contains(&value),
        // Checked against the other argument's default by the caller.
        Domain::AtLeast(_) => value.is_finite(),
    }
}

/// The table and the registry name the same numeric parameters, with the same kinds, and every
/// default lies in its domain.
#[test]
fn every_numeric_parameter_has_a_domain_holding_its_default() {
    let mut wrong = Vec::new();
    let mut seen = Vec::new();
    for info in BuiltinProcedures::new().list() {
        for p in &info.parameters {
            let kind = match format!("{:?}", p.param_type).as_str() {
                "Integer" => Kind::Integer,
                "Float" => Kind::Float,
                _ => continue,
            };
            seen.push((info.name.clone(), p.name.clone()));
            let Some((_, _, want_kind, domain)) = DOMAINS
                .iter()
                .find(|(proc, param, _, _)| *proc == info.name && *param == p.name)
            else {
                wrong.push(format!("{}.{}: no domain in the table", info.name, p.name));
                continue;
            };
            if kind != *want_kind {
                wrong.push(format!(
                    "{}.{}: {kind:?}, the table says {want_kind:?}",
                    info.name, p.name
                ));
            }
            if let Some(default) = &p.default
                && !holds(kind, *domain, default)
            {
                wrong.push(format!(
                    "{}.{}: the default {default} lies outside {domain:?}",
                    info.name, p.name
                ));
            }
        }
    }
    for (proc, param, _, _) in DOMAINS {
        if !seen.iter().any(|(p, q)| p == proc && q == param) {
            wrong.push(format!("{proc}.{param}: in the table, not in the registry"));
        }
    }
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}

/// Every probe of every row, positionally and in the map form.
#[test]
fn a_value_outside_its_domain_is_refused_and_each_bound_runs() {
    let db = graph();
    let mut wrong = Vec::new();
    let mut checked = 0;
    for (procedure, parameter, kind, domain) in DOMAINS {
        let (outside, inside) = probes(procedure, *kind, *domain);
        for positional in [true, false] {
            let query = call(procedure, parameter, positional);
            let form = if positional { "positional" } else { "map" };
            for value in &outside {
                checked += 1;
                match run(&db, &query, value.clone()) {
                    Err((status, message))
                        if status == INVALID_PARAMETER_VALUE
                            && message.contains(procedure)
                            && message.contains(parameter) => {}
                    other => wrong.push(format!(
                        "{procedure}.{parameter} = {value:?} ({form}): {other:?}, want 22023 naming both\n  {query}"
                    )),
                }
            }
            for value in &inside {
                checked += 1;
                if let Err(error) = run(&db, &query, value.clone()) {
                    wrong.push(format!(
                        "{procedure}.{parameter} = {value:?} ({form}): {error:?}, want a result\n  {query}"
                    ));
                }
            }
        }
    }
    assert!(checked > 100, "{checked} checked");
    assert!(
        wrong.is_empty(),
        "{} of {checked} wrong:\n{}",
        wrong.len(),
        wrong.join("\n")
    );
}

/// A literal out of its domain is refused as a parameter is, an integer and a float alike.
#[test]
fn a_literal_outside_its_domain_is_refused() {
    let db = graph();
    for query in [
        "CALL grafeo.pagerank(2.0) YIELD score",
        "CALL grafeo.pagerank(-1.0) YIELD score",
        "CALL grafeo.pagerank(2) YIELD score",
        "CALL grafeo.pagerank({damping: 1.5}) YIELD score",
        "CALL grafeo.pagerank(0.85, -1) YIELD score",
        "CALL grafeo.pagerank(0.85, 100, -0.5) YIELD score",
        "CALL grafeo.louvain(-0.1) YIELD community_id",
        "CALL grafeo.label_propagation(-1)",
        "CALL grafeo.kcore(-1)",
        "CALL grafeo.search.vector('Doc', 'emb', [1.0, 0.0], -1)",
        "CALL grafeo.search.mmr('Doc', 'emb', [1.0, 0.0], 2, 1, 0.5)",
        "CALL grafeo.search.mmr('Doc', 'emb', [1.0, 0.0], 2, 3, 1.5)",
    ] {
        let error = db.execute(query).expect_err(query);
        assert_eq!(
            GqlStatus::from(&error).as_str(),
            INVALID_PARAMETER_VALUE,
            "{query}: {error}"
        );
    }
}

fn scores(db: &GrafeoDB, arguments: &str, params: HashMap<String, Value>) -> Vec<f64> {
    let query = format!(
        "CALL grafeo.pagerank({arguments}) YIELD node_id, score \
         RETURN node_id, score ORDER BY node_id"
    );
    db.execute_with_params(&query, params)
        .unwrap_or_else(|error| panic!("{query}: {error}"))
        .rows()
        .iter()
        .map(|row| row[1].as_float64().unwrap())
        .collect()
}

/// The damping, by name: the bounds run, 0 gives every node the same score, an integer runs as its
/// float, and the old silent answers are refused.
#[test]
fn pagerank_damping_runs_inside_zero_to_one_only() {
    let db = graph();
    let none = HashMap::new;
    let zero = scores(&db, "0.0, 20", none());
    assert!(
        zero.windows(2)
            .all(|pair| (pair[0] - pair[1]).abs() < 1e-12),
        "damping 0: {zero:?}"
    );
    // Damping 1 has no teleport and may not converge: still a result, not an error.
    assert_eq!(scores(&db, "1.0, 5", none()).len(), zero.len());
    assert_eq!(scores(&db, "1, 5", none()), scores(&db, "1.0, 5", none()));
    // The default is 0.85, inside the domain.
    assert_eq!(scores(&db, "0.85, 100", none()), scores(&db, "", none()));
    for damping in [2.0, -1.0, f64::NAN, f64::INFINITY] {
        let params = HashMap::from([("d".to_string(), Value::Float64(damping))]);
        let error = db
            .execute_with_params("CALL grafeo.pagerank($d) YIELD score", params)
            .expect_err("outside [0, 1]");
        assert_eq!(
            GqlStatus::from(&error).as_str(),
            INVALID_PARAMETER_VALUE,
            "{damping}: {error}"
        );
        assert!(
            error.to_string().contains("damping") && error.to_string().contains("grafeo.pagerank"),
            "{damping}: {error}"
        );
    }
}

/// `max_iterations` 0 returns the starting scores (1 / n each); `tolerance` 0 runs every iteration.
#[test]
fn pagerank_at_zero_iterations_gives_the_starting_scores() {
    let db = graph();
    let start = scores(&db, "0.85, 0", HashMap::new());
    let n = start.len() as f64;
    assert!(
        start.iter().all(|score| (score - 1.0 / n).abs() < 1e-12),
        "{start:?}"
    );
    assert_eq!(
        scores(&db, "0.85, 7, 0.0", HashMap::new()).len(),
        start.len()
    );
}

/// A null argument takes the default of an optional parameter, however the null is computed, in both
/// forms; for a required parameter it is the error of an omitted argument, which then says it got
/// null.
#[test]
fn a_null_argument_takes_the_default() {
    let db = graph();
    let default = scores(&db, "0.85, 3", HashMap::new());
    for null in [
        "null",
        "null + 1",
        "abs(null)",
        "[1][5]",
        "{a: 1}.b",
        "head([])",
        "toInteger('x')",
        "coalesce(null)",
        "$p",
    ] {
        let params = HashMap::from([("p".to_string(), Value::Null)]);
        assert_eq!(
            scores(&db, &format!("{null}, 3"), params.clone()),
            default,
            "positional {null}"
        );
        assert_eq!(
            scores(
                &db,
                &format!("{{damping: {null}, max_iterations: 3}}"),
                params
            ),
            default,
            "map {null}"
        );
    }
    let omitted = db
        .execute("CALL grafeo.search.vector({property: 'emb', query: [1.0, 0.0], k: 1})")
        .expect_err("label is required")
        .to_string();
    for null in ["null", "head([])", "null + 'x'"] {
        let query = format!(
            "CALL grafeo.search.vector({{label: {null}, property: 'emb', query: [1.0, 0.0], k: 1}})"
        );
        let error = db.execute(&query).expect_err(&query).to_string();
        assert_eq!(error, format!("{omitted}, got null"), "{query}");
    }
}

/// Resolution 0 rewards only the edges inside a community: one community per connected component.
#[test]
fn louvain_at_resolution_zero_keeps_each_component_whole() {
    let db = GrafeoDB::new_in_memory();
    db.execute(
        "INSERT (a:P {name: 'Alix'})-[:K]->(b:P {name: 'Gus'})-[:K]->(c:P {name: 'Vincent'})-[:K]->(a), \
         (:P {name: 'Jules'})-[:K]->(:P {name: 'Mia'})-[:K]->(:P {name: 'Butch'})",
    )
    .unwrap();
    let names: HashMap<String, String> = db
        .execute("MATCH (n:P) RETURN id(n), n.name")
        .unwrap()
        .rows()
        .iter()
        .map(|row| {
            (
                format!("{:?}", row[0]),
                row[1].as_str().unwrap().to_string(),
            )
        })
        .collect();
    let community: HashMap<String, Value> = db
        .execute("CALL grafeo.louvain(0.0) YIELD node_id, community_id")
        .unwrap()
        .rows()
        .iter()
        .map(|row| (names[&format!("{:?}", row[0])].clone(), row[1].clone()))
        .collect();
    assert_eq!(community.len(), 6, "{community:?}");
    assert_eq!(community["Alix"], community["Gus"]);
    assert_eq!(community["Gus"], community["Vincent"]);
    assert_eq!(community["Jules"], community["Mia"]);
    assert_eq!(community["Mia"], community["Butch"]);
    assert_ne!(community["Alix"], community["Jules"], "{community:?}");
}

/// One session runs a parameterized CALL again and again, by position and by name: each run reads its
/// own `$d`, not the one the cached plan first saw, and a later value outside the domain is refused.
#[test]
fn a_cached_call_reads_each_runs_parameter() {
    let db = graph();
    let session = db.session();
    for call in [
        "grafeo.pagerank($d, 50)",
        "grafeo.pagerank({damping: $d, max_iterations: 50})",
    ] {
        let query =
            format!("CALL {call} YIELD node_id, score RETURN node_id, score ORDER BY node_id");
        let run = |d: f64| {
            session.execute_with_params(
                &query,
                HashMap::from([("d".to_string(), Value::Float64(d))]),
            )
        };
        let scores = |d: f64| -> Vec<f64> {
            run(d)
                .unwrap_or_else(|error| panic!("{call} with {d}: {error}"))
                .rows()
                .iter()
                .map(|row| row[1].as_float64().unwrap())
                .collect()
        };
        let even = scores(0.0);
        let damped = scores(0.85);
        assert_ne!(
            even, damped,
            "{call}: the second run read the first one's $d"
        );
        assert_eq!(scores(0.0), even, "{call}");
        assert_eq!(
            damped,
            self::scores(&db, "0.85, 50", HashMap::new()),
            "{call}: the cached plan answers as a fresh one"
        );
        let error = run(2.0).expect_err("2.0 lies outside [0, 1]");
        assert_eq!(
            GqlStatus::from(&error).as_str(),
            INVALID_PARAMETER_VALUE,
            "{call}: {error}"
        );
    }
}

/// Every string parameter's declared default lies in its domain (one character for a delimiter,
/// one of the listed words for `missing`), as the numeric ones do: prepare applies the defaults
/// before the domain check, so a default outside it would fail every call that leaves it out.
#[test]
fn every_string_default_lies_in_its_domain() {
    use grafeo_adapters::plugins::Domain;
    let mut wrong = Vec::new();
    let mut checked = 0;
    for info in BuiltinProcedures::new().list() {
        for p in &info.parameters {
            if format!("{:?}", p.param_type) != "String" {
                continue;
            }
            let Some(default) = &p.default else {
                continue;
            };
            checked += 1;
            let holds = match p.domain {
                Domain::OfType => true,
                Domain::OneCharacter | Domain::OneOf(_) => p.domain.admits_text(default),
                other => {
                    wrong.push(format!(
                        "{}.{}: a string parameter with the domain {other:?}",
                        info.name, p.name
                    ));
                    continue;
                }
            };
            if !holds {
                wrong.push(format!(
                    "{}.{}: the default {default:?} lies outside {:?}",
                    info.name, p.name, p.domain
                ));
            }
        }
    }
    assert!(
        checked >= 2,
        "the loads' delimiter and missing: {checked} checked"
    );
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}

/// A load called with `projection` in its map, a value or null, written in the call or given
/// as a parameter's map, is refused with 22023 naming the procedure and `projection`, before
/// any other argument is checked (the calls below omit the required `path`, which would be
/// named otherwise).
#[test]
fn a_load_refuses_a_projection_before_its_arguments() {
    let db = GrafeoDB::new_in_memory();
    let mut wrong = Vec::new();
    for load in ["grafeo.load_nodes_csv", "grafeo.load_edges_csv"] {
        for (value, given) in [("'people'", Value::from("people")), ("null", Value::Null)] {
            let written = format!("CALL {load}({{projection: {value}}})");
            let options = HashMap::from([(
                "opts".to_string(),
                Value::Map(std::sync::Arc::new(
                    [("projection".into(), given)].into_iter().collect(),
                )),
            )]);
            let in_a_parameter = format!("CALL {load}($opts)");
            for (call, result) in [
                (written.clone(), db.execute(&written)),
                (
                    format!("{in_a_parameter} with projection {value}"),
                    db.execute_with_params(&in_a_parameter, options),
                ),
            ] {
                match result {
                    Ok(_) => wrong.push(format!("{call}: ran")),
                    Err(error) => {
                        let text = error.to_string();
                        let status = GqlStatus::from(&error).as_str().to_string();
                        if status != "22023"
                            || !text.contains(load.trim_start_matches("grafeo."))
                            || !text.contains("projection")
                            || text.contains("'path'")
                        {
                            wrong.push(format!("{call}: {status} {text}"));
                        }
                    }
                }
            }
        }
    }
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}
