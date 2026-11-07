//! A procedure's optional argument takes the default the catalogue declares: a CALL that leaves it
//! out, one that passes the declared default and one that passes null give the same result. The
//! defaults come from `ParameterDef` alone (one place, `prepare`), so an algorithm that kept a
//! literal of its own (`unwrap_or(0.85)`) that drifted from its declaration fails here. A direct
//! `Procedure::execute` (and `execute_parallel`) prepares its arguments as CALL does: it gives
//! CALL's rows with the optional arguments left out, and refuses a missing, null or out-of-domain
//! argument with CALL's 22023. CALL checks its arguments when it plans, so `EXPLAIN` refuses them
//! too, before the `YIELD` is checked. Driven by the catalogue `grafeo.procedures()` lists, so a
//! new procedure is covered without a row.
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test main procedure_optional_defaults::
//! ```

#![cfg(all(
    feature = "gql",
    feature = "algos",
    feature = "vector-index",
    feature = "text-index"
))]

use grafeo_adapters::plugins::algorithms::{
    ClusteringCoefficientAlgorithm, GraphAlgorithm, ParallelGraphAlgorithm,
};
use grafeo_adapters::plugins::{Domain, ParameterDef, ParameterType, Parameters};
use grafeo_common::types::Value;
use grafeo_common::utils::error::Error;
use grafeo_common::utils::gqlstatus::GqlStatus;
use grafeo_engine::GrafeoDB;
use grafeo_engine::procedures::{BuiltinProcedures, ProcedureContext};

const INVALID_PARAMETER_VALUE: &str = "22023";

/// A graph every procedure runs on: a weighted cycle and a pair, and documents with a vector and a
/// text index for the search procedures.
fn graph() -> (GrafeoDB, i64) {
    let db = GrafeoDB::new_in_memory();
    db.execute(
        "INSERT (a:Person {name: 'Alix'})-[:KNOWS {w: 2, c: 3, k: 1}]->(b:Person {name: 'Gus'})\
         -[:KNOWS {w: 1, c: 2, k: 2}]->(c:Person {name: 'Vincent'})-[:KNOWS {w: 3, c: 1, k: 1}]->(a), \
         (a)-[:KNOWS {w: 5, c: 1, k: 3}]->(c), \
         (:Person {name: 'Jules'})-[:KNOWS {w: 1, c: 1, k: 1}]->(:Person {name: 'Mia'})",
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
    let alix = match db
        .execute("MATCH (p:Person {name: 'Alix'}) RETURN id(p) AS id")
        .unwrap()
        .rows()[0][0]
    {
        Value::Int64(id) => id,
        ref other => panic!("{other:?}"),
    };
    (db, alix)
}

/// A value a required argument takes on [`graph`], which every domain accepts.
fn sample(procedure: &str, parameter: &ParameterDef, node: i64) -> String {
    match (procedure, parameter.name.as_str()) {
        ("grafeo.search.text", "property") => return "'body'".to_string(),
        (_, "property") => return "'emb'".to_string(),
        (_, "label") => return "'Doc'".to_string(),
        ("grafeo.search.text", "query") => return "'graph'".to_string(),
        ("grafeo.subgraph_isomorphism", "pattern_edges") => return "'0-1'".to_string(),
        _ => {}
    }
    match parameter.param_type {
        ParameterType::Integer => "2".to_string(),
        ParameterType::Float => "0.5".to_string(),
        ParameterType::String => "'w'".to_string(),
        ParameterType::Boolean => "true".to_string(),
        ParameterType::NodeId => node.to_string(),
        ParameterType::List => "[1.0, 0.0]".to_string(),
        ParameterType::Map => "{}".to_string(),
    }
}

/// A declared default as a GQL literal of the parameter's type.
fn literal(parameter: &ParameterDef, default: &str) -> String {
    match parameter.param_type {
        ParameterType::String => format!("'{default}'"),
        _ => default.to_string(),
    }
}

/// The result of `query`, rows or error, as text to compare.
fn outcome(db: &GrafeoDB, query: &str) -> String {
    match db.execute(query) {
        Ok(result) => format!("rows {:?}", result.rows()),
        Err(error) => format!("error {error}"),
    }
}

/// Every optional parameter of every procedure: left out, the declared default and null give the
/// same result, in the map form with the required arguments given.
#[test]
fn an_optional_argument_left_out_is_its_declared_default_and_so_is_null() {
    let (db, node) = graph();
    let mut wrong = Vec::new();
    let mut ran = Vec::new();
    for procedure in BuiltinProcedures::new().list() {
        let required: Vec<String> = procedure
            .parameters
            .iter()
            .filter(|p| p.required)
            .map(|p| format!("{}: {}", p.name, sample(&procedure.name, p, node)))
            .collect();
        for parameter in procedure.parameters.iter().filter(|p| !p.required) {
            let call = |extra: Option<String>| {
                let mut entries = required.clone();
                entries.extend(extra);
                format!("CALL {}({{{}}})", procedure.name, entries.join(", "))
            };
            let omitted_query = call(None);
            let omitted = outcome(&db, &omitted_query);
            if omitted.starts_with("rows") {
                ran.push(format!("{}.{}", procedure.name, parameter.name));
            }
            let mut forms = vec![call(Some(format!("{}: null", parameter.name)))];
            if let Some(default) = &parameter.default {
                forms.push(call(Some(format!(
                    "{}: {}",
                    parameter.name,
                    literal(parameter, default)
                ))));
            }
            for query in forms {
                let got = outcome(&db, &query);
                if got != omitted {
                    wrong.push(format!(
                        "{query}\n  gives {got}\n  {omitted_query}\n  gives {omitted}"
                    ));
                }
            }
        }
    }
    for known in [
        "grafeo.pagerank.damping",
        "grafeo.pagerank.max_iterations",
        "grafeo.louvain.resolution",
        "grafeo.betweenness_centrality.normalized",
        "grafeo.search.vector.k",
        "grafeo.search.text.k",
        "grafeo.search.mmr.k",
        "grafeo.search.mmr.fetch_k",
        "grafeo.search.mmr.lambda",
    ] {
        assert!(
            ran.iter().any(|r| r == known),
            "{known} ran on the graph: {ran:?}"
        );
    }
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}

/// The test reads the catalogue `grafeo.procedures()` shows.
#[test]
fn the_registry_is_the_listed_catalogue() {
    let (db, _) = graph();
    let mut listed: Vec<String> = db
        .execute("CALL grafeo.procedures() YIELD name RETURN name")
        .unwrap()
        .rows()
        .iter()
        .map(|row| match &row[0] {
            Value::String(name) => name.to_string(),
            other => panic!("{other:?}"),
        })
        .collect();
    listed.sort();
    let registry: Vec<String> = BuiltinProcedures::new()
        .list()
        .into_iter()
        .map(|p| p.name)
        .collect();
    assert_eq!(listed, registry);
}

/// `value` (a [`sample`] text) set for `parameter` as its type takes it.
fn set(params: &mut Parameters, parameter: &ParameterDef, value: &str) {
    let name = parameter.name.clone();
    match parameter.param_type {
        ParameterType::Integer | ParameterType::NodeId => {
            params.set_int(name, value.parse().unwrap());
        }
        ParameterType::Float => params.set_float(name, value.parse().unwrap()),
        ParameterType::String => params.set_string(name, value.trim_matches('\'')),
        ParameterType::Boolean => params.set_bool(name, value == "true"),
        ParameterType::List => {
            params.set_list(name, vec![Value::Float64(1.0), Value::Float64(0.0)]);
        }
        ParameterType::Map => params.set_map(name, std::sync::Arc::default()),
    }
}

/// The required arguments of `procedure` on [`graph`], as parameters, leaving `except` out.
fn required(
    procedure: &grafeo_engine::procedures::ProcedureInfo,
    node: i64,
    except: &str,
) -> Parameters {
    let mut params = Parameters::new();
    for parameter in procedure
        .parameters
        .iter()
        .filter(|p| p.required && p.name != except)
    {
        set(
            &mut params,
            parameter,
            &sample(&procedure.name, parameter, node),
        );
    }
    params
}

/// The rows of a direct `Procedure::execute` of `procedure` with `params`, or its error.
/// The procedures a direct execute cannot run on a snapshot: they read the store's own indexes,
/// which the public API gives only through `db.store()` (deprecated: its writes skip the WAL).
/// Their optional arguments are covered through `CALL`, as every procedure's are.
const NEEDS_THE_STORE: [&str; 3] = [
    "grafeo.search.vector",
    "grafeo.search.mmr",
    "grafeo.search.text",
];

/// The procedures a direct execute does not run: they are planned with the statement's writer, so
/// `Procedure::execute` refuses them by design and names the CALL and the API call instead.
/// Their arguments are covered through `CALL` and in the external-access and loader tests.
const PLANNED_WITH_THE_WRITER: [&str; 2] = ["grafeo.load_nodes_csv", "grafeo.load_edges_csv"];

/// Whether the direct-execute rows leave `name` out (see [`NEEDS_THE_STORE`] and
/// [`PLANNED_WITH_THE_WRITER`]).
fn left_out_of_direct_execute(name: &str) -> bool {
    NEEDS_THE_STORE.contains(&name) || PLANNED_WITH_THE_WRITER.contains(&name)
}

fn direct(db: &GrafeoDB, name: &str, params: &Parameters) -> Result<Vec<Vec<Value>>, Error> {
    let path: Vec<String> = name.split('.').map(String::from).collect();
    let procedure = BuiltinProcedures::new().get(&path).expect(name);
    let snapshot = db.snapshot().unwrap();
    let ctx = ProcedureContext::new(&snapshot);
    procedure.execute(&ctx, params).map(|result| result.rows)
}

/// The names the direct-execute rows leave out are registered procedures: a renamed or removed one
/// fails here instead of turning the exclusion into a silent gap.
#[test]
fn the_procedures_left_out_of_direct_execute_are_registered() {
    let registered: Vec<String> = BuiltinProcedures::new()
        .list()
        .into_iter()
        .map(|procedure| procedure.name)
        .collect();
    for name in NEEDS_THE_STORE.into_iter().chain(PLANNED_WITH_THE_WRITER) {
        assert!(
            registered.iter().any(|registered| registered == name),
            "{name} is not registered: {registered:?}"
        );
    }
}

fn status(error: &Error) -> String {
    GqlStatus::from(error).as_str().to_string()
}

/// A direct execute with only the required arguments gives the rows of the CALL that leaves every
/// optional argument out: both take the declared defaults.
#[test]
fn a_direct_execute_without_optional_arguments_gives_what_call_gives() {
    let (db, node) = graph();
    let mut wrong = Vec::new();
    for procedure in BuiltinProcedures::new().list() {
        if left_out_of_direct_execute(&procedure.name) {
            continue;
        }
        let given: Vec<String> = procedure
            .parameters
            .iter()
            .filter(|p| p.required)
            .map(|p| format!("{}: {}", p.name, sample(&procedure.name, p, node)))
            .collect();
        let query = format!("CALL {}({{{}}})", procedure.name, given.join(", "));
        let called = db.execute(&query).map(|result| result.rows().to_vec());
        let executed = direct(&db, &procedure.name, &required(&procedure, node, ""));
        let text = |r: &Result<Vec<Vec<Value>>, Error>| match r {
            Ok(rows) => format!("rows {rows:?}"),
            Err(error) => format!("error {}", status(error)),
        };
        if text(&called) != text(&executed) {
            wrong.push(format!(
                "{}: CALL gives {}, execute gives {}",
                procedure.name,
                text(&called),
                text(&executed)
            ));
        }
    }
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}

/// A value outside `domain`, as an integer and as a float, with the floor an `AtLeast` names given
/// as 3 (`None` for a parameter that is not a number, or a domain every number is inside).
fn outside(domain: Domain, param_type: ParameterType) -> Option<(i64, f64)> {
    let value = match domain {
        Domain::OfType | Domain::OneCharacter | Domain::OneOf(_) | Domain::ListOrOne => {
            return None;
        }
        Domain::NonNegative => (-1, -1.0),
        Domain::Positive => (0, 0.0),
        Domain::Unit => (2, 2.0),
        Domain::AtLeast(_) => (1, 1.0),
    };
    matches!(param_type, ParameterType::Integer | ParameterType::Float).then_some(value)
}

/// A direct execute refuses what CALL refuses, with 22023: a required argument left out (naming
/// it), given as null (saying so) and a numeric argument outside its domain.
#[test]
fn a_direct_execute_refuses_a_missing_null_or_out_of_domain_argument() {
    let (db, node) = graph();
    let mut wrong = Vec::new();
    let mut refused = 0;
    for procedure in BuiltinProcedures::new().list() {
        if left_out_of_direct_execute(&procedure.name) {
            continue;
        }
        let mut cases: Vec<(String, Parameters, String)> = Vec::new();
        for parameter in procedure.parameters.iter().filter(|p| p.required) {
            let missing = required(&procedure, node, &parameter.name);
            cases.push((
                format!("{} left out", parameter.name),
                missing,
                format!("missing required argument '{}'", parameter.name),
            ));
            let mut null = required(&procedure, node, &parameter.name);
            null.set_null(parameter.name.clone());
            cases.push((
                format!("{} null", parameter.name),
                null,
                ", got null".to_string(),
            ));
        }
        for parameter in &procedure.parameters {
            let Some((integer, float)) = outside(parameter.domain, parameter.param_type) else {
                continue;
            };
            let mut params = required(&procedure, node, &parameter.name);
            if let Domain::AtLeast(floor) = parameter.domain {
                params.set_int(floor, 3);
            }
            match parameter.param_type {
                ParameterType::Integer => params.set_int(parameter.name.clone(), integer),
                _ => params.set_float(parameter.name.clone(), float),
            }
            cases.push((
                format!("{} = {float}", parameter.name),
                params,
                parameter.name.clone(),
            ));
        }
        for (case, params, says) in cases {
            match direct(&db, &procedure.name, &params) {
                Err(error)
                    if status(&error) == INVALID_PARAMETER_VALUE
                        && error.to_string().contains(&says) =>
                {
                    refused += 1;
                }
                other => wrong.push(format!(
                    "{} with {case}: {:?}, want 22023 saying {says:?}",
                    procedure.name,
                    other
                        .map(|rows| rows.len())
                        .map_err(|error| error.to_string())
                )),
            }
        }
    }
    // Every case is refused, and the catalogue gives the cases: two per required argument (left
    // out, null) and one per bounded one, counted from the catalogue rather than a fixed floor.
    let expected: usize = BuiltinProcedures::new()
        .list()
        .iter()
        .filter(|procedure| !left_out_of_direct_execute(&procedure.name))
        .map(|procedure| {
            let required = procedure.parameters.iter().filter(|p| p.required).count();
            let bounded = procedure
                .parameters
                .iter()
                .filter(|p| outside(p.domain, p.param_type).is_some())
                .count();
            2 * required + bounded
        })
        .sum();
    assert!(
        expected > 0,
        "the catalogue has required and bounded arguments"
    );
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
    assert_eq!(refused, expected, "every case of the catalogue is refused");
}

/// `execute_parallel` prepares as `execute` does: without optional arguments it gives the
/// sequential rows, and it refuses an argument outside its domain before any work.
#[test]
fn execute_parallel_prepares_its_arguments_as_execute_does() {
    let (db, _) = graph();
    let snapshot = db.snapshot().unwrap();
    let algorithm = ClusteringCoefficientAlgorithm;
    let none = Parameters::new();
    let sequential = algorithm
        .execute(snapshot.store(), snapshot.view(), &none)
        .unwrap();
    let parallel = algorithm
        .execute_parallel(snapshot.store(), snapshot.view(), &none, 2)
        .unwrap();
    assert_eq!(parallel.rows, sequential.rows);
    for parameter in algorithm.parameters() {
        let Some((integer, float)) = outside(parameter.domain, parameter.param_type) else {
            continue;
        };
        let mut params = Parameters::new();
        match parameter.param_type {
            ParameterType::Integer => params.set_int(parameter.name.clone(), integer),
            _ => params.set_float(parameter.name.clone(), float),
        }
        let error = algorithm
            .execute_parallel(snapshot.store(), snapshot.view(), &params, 2)
            .expect_err("outside its domain");
        assert_eq!(
            status(&error),
            INVALID_PARAMETER_VALUE,
            "{}: {error}",
            parameter.name
        );
    }
}

/// CALL checks its arguments when it plans: `EXPLAIN` of a CALL with a required argument left out,
/// or one outside its domain, is refused with 22023, and the argument error comes before an
/// unknown `YIELD` column's.
#[test]
fn a_call_refuses_its_arguments_when_it_plans() {
    let (db, _) = graph();
    let mut wrong = Vec::new();
    for procedure in BuiltinProcedures::new().list() {
        if !procedure.parameters.iter().any(|p| p.required) {
            continue;
        }
        for query in [
            format!("EXPLAIN CALL {}()", procedure.name),
            format!("CALL {}() YIELD no_such_column", procedure.name),
        ] {
            match db.execute(&query) {
                Err(error) if status(&error) == INVALID_PARAMETER_VALUE => {}
                other => wrong.push(format!(
                    "{query}: {:?}",
                    other.map(|r| r.rows().len()).map_err(|e| e.to_string())
                )),
            }
        }
    }
    for query in [
        "EXPLAIN CALL grafeo.pagerank({damping: 2.0})",
        "CALL grafeo.pagerank({damping: 2.0}) YIELD no_such_column",
    ] {
        match db.execute(query) {
            Err(error) if status(&error) == INVALID_PARAMETER_VALUE => {}
            other => wrong.push(format!(
                "{query}: {:?}",
                other.map(|r| r.rows().len()).map_err(|e| e.to_string())
            )),
        }
    }
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}

/// `AtLeast` compares integers exactly: past 2^53, where `k` and the integer below it are one
/// float, a `fetch_k` one below `k` is still refused, and one equal to it is not. Planned under
/// `EXPLAIN`, so no search of that size runs.
#[test]
fn at_least_compares_integers_exactly() {
    let (db, _) = graph();
    let explain = |k: i64, fetch_k: i64| {
        db.execute(&format!(
            "EXPLAIN CALL grafeo.search.mmr({{label: 'Doc', property: 'emb', query: [1.0, 0.0], \
             k: {k}, fetch_k: {fetch_k}}})"
        ))
    };
    let k = 9_007_199_254_740_994;
    let error = explain(k, k - 1).expect_err("a fetch_k one below k");
    assert_eq!(status(&error), INVALID_PARAMETER_VALUE, "{error}");
    assert!(error.to_string().contains("fetch_k"), "{error}");
    explain(k, k).unwrap_or_else(|error| panic!("a fetch_k equal to k: {error}"));
}
