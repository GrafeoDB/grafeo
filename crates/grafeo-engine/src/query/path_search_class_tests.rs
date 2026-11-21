//! Every path search the planner builds holds its state in its path search
//! budget and as a grant of the statement's memory (0.6.4): the variable-length
//! expand, the shortest path and the SPARQL property path each refuse with
//! `GRAFEO-S001` past a small path search budget and past a small query memory
//! limit, and the statement gives back what it held either way. The list of
//! path searches is checked against the planner's own operators: every
//! logical operator is classified below by name, so a new one does not compile
//! until it is classified, every path search has a query here, and PROFILE
//! shows each query running the physical operator it is listed with.
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --lib query::path_search_class_tests::
//! ```
//!
//! A unit test of the crate, as `LogicalOperator` is `#[non_exhaustive]`: only here can the
//! classification name every operator without a catch-all arm.

use std::fmt::Write;

use grafeo_adapters::query::limits::ParseLimits;
use grafeo_common::utils::error::{Error, LimitBound, LimitRefusal};

use crate::query::plan::{LogicalOperator, LogicalPlan};
use crate::query::translators::{cypher, gql, sparql};
use crate::{Config, GrafeoDB};

const KIB: usize = 1024;

/// The hub's leaves, in the graph and in the RDF store: each search holds
/// about 5,000 paths or terms, well past 64 KiB.
const LEAVES: usize = 5_000;

/// A kind of path search the planner builds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PathSearch {
    /// A variable-length expand (`VariableLengthExpandOperator`).
    VariableLength,
    /// A shortest path search (`ShortestPathOperator`).
    Shortest,
    /// A SPARQL property path (`RdfPathOperator`).
    PropertyPath,
}

const EVERY_PATH_SEARCH: [PathSearch; 3] = [
    PathSearch::VariableLength,
    PathSearch::Shortest,
    PathSearch::PropertyPath,
];

/// The physical operators the planner builds a path search as, by the names
/// PROFILE shows: a property path is searched once over its pairs, or once
/// per row of the pattern that binds its subject.
const EVERY_PATH_OPERATOR: [&str; 4] = [
    "VariableLengthExpand",
    "ShortestPath",
    "RdfPropertyPath",
    "RdfPropertyPathPerRow",
];

/// The path search the planner builds from `op`, if any. Every operator is
/// named, without a catch-all arm, so a new logical operator fails to compile
/// here until it is classified.
fn path_search(op: &LogicalOperator) -> Option<PathSearch> {
    use LogicalOperator as Op;
    match op {
        Op::Expand(expand) => (expand.is_variable_length() || expand.path_alias.is_some())
            .then_some(PathSearch::VariableLength),
        Op::ShortestPath(_) => Some(PathSearch::Shortest),
        Op::PropertyPath(_) => Some(PathSearch::PropertyPath),
        Op::NodeScan(_)
        | Op::EdgeScan(_)
        | Op::Filter(_)
        | Op::Project(_)
        | Op::Join(_)
        | Op::Aggregate(_)
        | Op::Limit(_)
        | Op::Skip(_)
        | Op::Sort(_)
        | Op::Distinct(_)
        | Op::CreateNode(_)
        | Op::CreateEdge(_)
        | Op::Create(_)
        | Op::DeleteNode(_)
        | Op::DeleteEdge(_)
        | Op::SetProperty(_)
        | Op::AddLabel(_)
        | Op::RemoveLabel(_)
        | Op::Return(_)
        | Op::Empty
        | Op::TripleScan(_)
        | Op::NamedGraphs(_)
        | Op::RowValues(_)
        | Op::InNamedGraph(_)
        | Op::Union(_)
        | Op::LeftJoin(_)
        | Op::AntiJoin(_)
        | Op::MarkJoin(_)
        | Op::Construct(_)
        | Op::Bind(_)
        | Op::Unwind(_)
        | Op::MapCollect(_)
        | Op::Merge(_)
        | Op::MergeRelationship(_)
        | Op::InsertTriple(_)
        | Op::DeleteTriple(_)
        | Op::Modify(_)
        | Op::ClearGraph(_)
        | Op::CreateGraph(_)
        | Op::DropGraph(_)
        | Op::LoadGraph(_)
        | Op::CopyGraph(_)
        | Op::MoveGraph(_)
        | Op::AddGraph(_)
        | Op::HorizontalAggregate(_)
        | Op::VectorScan(_)
        | Op::VectorJoin(_)
        | Op::TextScan(_)
        | Op::Except(_)
        | Op::Intersect(_)
        | Op::Otherwise(_)
        | Op::Apply(_)
        | Op::ParameterScan(_)
        | Op::DelimScan(_)
        | Op::CreatePropertyGraph(_)
        | Op::DropPropertyGraph(_)
        | Op::MultiWayJoin(_)
        | Op::CallProcedure(_)
        | Op::LoadData(_) => None,
    }
}

/// The path searches of `op` and everything under it.
fn path_searches(op: &LogicalOperator, found: &mut Vec<PathSearch>) {
    found.extend(path_search(op));
    for child in op.children() {
        path_searches(child, found);
    }
}

/// One query per path search and language, with the physical operator it
/// runs and the rows it answers.
const QUERIES: [(&str, &str, PathSearch, &str, usize); 6] = [
    (
        "gql",
        "MATCH (s:Hub)-[:E]->{1,2}(t) RETURN t",
        PathSearch::VariableLength,
        "VariableLengthExpand",
        LEAVES + 1,
    ),
    (
        "cypher",
        "MATCH (s:Hub)-[:E*1..2]->(t) RETURN t",
        PathSearch::VariableLength,
        "VariableLengthExpand",
        LEAVES + 1,
    ),
    (
        "gql",
        "MATCH p = ANY SHORTEST (s:Hub)-[:E]->+(t:Target) RETURN length(p) AS l",
        PathSearch::Shortest,
        "ShortestPath",
        1,
    ),
    (
        "cypher",
        "MATCH p = shortestPath((s:Hub)-[:E*]->(t:Target)) RETURN length(p) AS l",
        PathSearch::Shortest,
        "ShortestPath",
        1,
    ),
    (
        "sparql",
        "PREFIX : <urn:> SELECT ?x WHERE { :hub :e+ ?x }",
        PathSearch::PropertyPath,
        "RdfPropertyPath",
        LEAVES + 1,
    ),
    (
        "sparql",
        "PREFIX : <urn:> SELECT ?x WHERE { ?h :e :leaf0 . ?h :e+ ?x }",
        PathSearch::PropertyPath,
        "RdfPropertyPathPerRow",
        LEAVES + 1,
    ),
];

fn translate(language: &str, query: &str) -> LogicalPlan {
    let limits = ParseLimits::default();
    match language {
        "gql" => gql::translate(query, &limits),
        "cypher" => cypher::translate(query, &limits),
        "sparql" => sparql::translate(query, &limits),
        other => panic!("no translator for {other}"),
    }
    .unwrap_or_else(|error| panic!("{language} `{query}`: {error}"))
}

/// A hub with an edge to each of [`LEAVES`] leaves, the first of which (also
/// `First`) leads on to the target, as a graph and as triples, in the default
/// graph and in the named graph `<urn:g>`.
fn hub(config: Config) -> GrafeoDB {
    let db = GrafeoDB::with_config(config).unwrap();
    let hub = db.create_node(&["Hub"]).unwrap();
    let target = db.create_node(&["Target"]).unwrap();
    let mut triples = String::new();
    for leaf in 0..LEAVES {
        let labels: &[&str] = if leaf == 0 {
            &["Leaf", "First"]
        } else {
            &["Leaf"]
        };
        let node = db.create_node(labels).unwrap();
        db.create_edge(hub, node, "E").unwrap();
        write!(triples, " :hub :e :leaf{leaf} .").unwrap();
        if leaf == 0 {
            db.create_edge(node, target, "E").unwrap();
            triples.push_str(" :leaf0 :e :target .");
        }
    }
    let insert = format!("PREFIX : <urn:> INSERT DATA {{{triples} GRAPH <urn:g> {{{triples} }} }}");
    db.execute_language(&insert, "sparql", None).unwrap();
    db
}

/// The bytes the statements of `db` hold.
fn execution_bytes(db: &GrafeoDB) -> usize {
    db.memory_usage().buffer_manager.execution_buffers_bytes
}

/// The bound `error` names: `GRAFEO-S001` of a memory bound.
fn refused_by(error: &Error) -> LimitBound {
    assert_eq!(error.error_code().as_str(), "GRAFEO-S001", "{error}");
    match error.limit_refusal() {
        Some(LimitRefusal::Memory { bound, .. }) => *bound,
        other => panic!("a memory bound's refusal, got {other:?}: {error}"),
    }
}

/// Whether PROFILE's `profile` shows a line of the operator `name` (not of
/// another whose name it starts).
fn runs(profile: &str, name: &str) -> bool {
    profile.lines().any(|line| {
        line.trim_start_matches(|c: char| !c.is_ascii_alphanumeric())
            .strip_prefix(name)
            .is_some_and(|rest| !rest.starts_with(|c: char| c.is_ascii_alphanumeric()))
    })
}

/// The PROFILE text of `query` in `language` on `db`.
fn profile(db: &GrafeoDB, language: &str, query: &str) -> String {
    let profiled = format!("PROFILE {query}");
    let result = db
        .execute_language(&profiled, language, None)
        .unwrap_or_else(|error| panic!("{language} `{profiled}`: {error}"));
    match &result.rows()[0][0] {
        grafeo_common::types::Value::String(text) => text.to_string(),
        other => panic!("{language} `{profiled}`: a plan's text, got {other:?}"),
    }
}

#[test]
fn every_path_search_the_planner_builds_has_a_query() {
    let db = hub(Config::in_memory());
    for (language, query, search, operator, _) in QUERIES {
        let mut found = Vec::new();
        path_searches(&translate(language, query).root, &mut found);
        assert!(
            found.contains(&search),
            "{language} `{query}` plans {search:?}: {found:?}"
        );
        let profile = profile(&db, language, query);
        assert!(
            runs(&profile, operator),
            "{language} `{query}` runs {operator}:\n{profile}"
        );
    }
    for search in EVERY_PATH_SEARCH {
        assert!(
            QUERIES.iter().any(|&(_, _, of, _, _)| of == search),
            "a query plans {search:?}"
        );
    }
    for operator in EVERY_PATH_OPERATOR {
        assert!(
            QUERIES.iter().any(|&(_, _, _, of, _)| of == operator),
            "a query runs {operator}"
        );
    }
}

#[test]
fn every_path_search_answers_within_its_bounds() {
    let db = hub(Config::in_memory());
    for (language, query, _, _, rows) in QUERIES {
        let result = db
            .execute_language(query, language, None)
            .unwrap_or_else(|error| panic!("{language} `{query}`: {error}"));
        assert_eq!(result.row_count(), rows, "{language} `{query}`");
    }
}

#[test]
fn every_path_search_refuses_past_its_budget() {
    let db = hub(Config::in_memory().with_path_search_budget(64 * KIB));
    for (language, query, _, _, _) in QUERIES {
        let error = db.execute_language(query, language, None).expect_err(query);
        assert_eq!(
            refused_by(&error),
            LimitBound::PathSearchBudget,
            "{language} `{query}`: {error}"
        );
        assert!(
            error.to_string().contains("64 KiB"),
            "{language} `{query}` names the budget: {error}"
        );
        assert_eq!(execution_bytes(&db), 0, "{language} `{query}`: given back");
    }
}

#[test]
fn every_path_search_refuses_past_the_query_memory_limit() {
    let db = hub(Config::in_memory().with_query_memory_limit(64 * KIB));
    for (language, query, _, _, _) in QUERIES {
        let error = db.execute_language(query, language, None).expect_err(query);
        assert_eq!(
            refused_by(&error),
            LimitBound::QueryMemoryLimit,
            "{language} `{query}`: {error}"
        );
        assert_eq!(execution_bytes(&db), 0, "{language} `{query}`: given back");
    }
}

/// More shapes of the three searches, each refused past a small path search
/// budget: the selections that keep more than one path, Cypher's
/// `allShortestPaths`, a search inside `EXISTS` (the `WHERE` on its far node
/// keeps it a search, not an edge check) and under `LIMIT 1` (the hub's 5,000
/// paths are queued before the first row), SPARQL's `*`, a property path
/// searched once per row of the pattern that binds its subject, and one
/// inside `FILTER EXISTS`, planned for each row as it is filtered.
const BUDGET_MATRIX: [(&str, &str); 9] = [
    (
        "gql",
        "MATCH p = ALL SHORTEST (s:Hub)-[:E]->+(t:Target) RETURN length(p) AS l",
    ),
    (
        "gql",
        "MATCH p = SHORTEST 2 (s:Hub)-[:E]->+(t:Target) RETURN length(p) AS l",
    ),
    (
        "gql",
        "MATCH p = SHORTEST 2 GROUPS (s:Hub)-[:E]->+(t:Target) RETURN length(p) AS l",
    ),
    (
        "cypher",
        "MATCH p = allShortestPaths((s:Hub)-[:E*]->(t:Target)) RETURN length(p) AS l",
    ),
    (
        "gql",
        "MATCH (s:Hub) WHERE EXISTS { MATCH (s)-[:E]->{1,2}(t) WHERE t.id = -1 } RETURN s",
    ),
    ("gql", "MATCH (s:Hub)-[:E]->{1,2}(t) RETURN t LIMIT 1"),
    ("sparql", "PREFIX : <urn:> SELECT ?x WHERE { :hub :e* ?x }"),
    (
        "sparql",
        "PREFIX : <urn:> SELECT ?x WHERE { ?h :e :leaf0 . ?h :e+ ?x }",
    ),
    (
        "sparql",
        "PREFIX : <urn:> SELECT ?h WHERE { ?h :e :leaf0          FILTER EXISTS { :hub :e+ ?x { FILTER (?h = :nowhere) } } }",
    ),
];

/// The restrictive path modes of a variable-length pattern and the property
/// path searched per row, past a small query memory limit.
const QUERY_LIMIT_MATRIX: [(&str, &str); 4] = [
    ("gql", "MATCH TRAIL (s:Hub)-[:E]->{1,2}(t) RETURN t"),
    ("gql", "MATCH SIMPLE (s:Hub)-[:E]->{1,2}(t) RETURN t"),
    ("gql", "MATCH ACYCLIC (s:Hub)-[:E]->{1,2}(t) RETURN t"),
    (
        "sparql",
        "PREFIX : <urn:> SELECT ?x WHERE { ?h :e :leaf0 . ?h :e+ ?x }",
    ),
];

/// Each of `queries` on `db` is refused naming `bound`, the statements
/// holding nothing before and after it.
fn assert_refused_and_given_back(db: &GrafeoDB, queries: &[(&str, &str)], bound: LimitBound) {
    for &(language, query) in queries {
        assert_eq!(
            execution_bytes(db),
            0,
            "{language} `{query}`: nothing held before"
        );
        let error = db.execute_language(query, language, None).expect_err(query);
        assert_eq!(refused_by(&error), bound, "{language} `{query}`: {error}");
        assert_eq!(execution_bytes(db), 0, "{language} `{query}`: given back");
    }
}

/// The `FILTER EXISTS` of [`BUDGET_MATRIX`] is tested per row: its path
/// search is planned for each row as it is filtered.
#[test]
fn the_exists_shape_runs_its_search_per_row() {
    let db = hub(Config::in_memory());
    let (language, query) = BUDGET_MATRIX[BUDGET_MATRIX.len() - 1];
    let profile = profile(&db, language, query);
    assert!(profile.contains("[ExistsPerRow]"), "{profile}");
}

#[test]
fn every_shape_of_a_path_search_refuses_past_its_budget() {
    let db = hub(Config::in_memory().with_path_search_budget(64 * KIB));
    assert_refused_and_given_back(&db, &BUDGET_MATRIX, LimitBound::PathSearchBudget);
}

#[test]
fn the_restrictive_modes_and_a_per_row_property_path_refuse_past_the_query_memory_limit() {
    let db = hub(Config::in_memory().with_query_memory_limit(64 * KIB));
    assert_refused_and_given_back(&db, &QUERY_LIMIT_MATRIX, LimitBound::QueryMemoryLimit);
}

#[cfg(feature = "sql-pgq")]
#[test]
fn a_sql_pgq_path_search_refuses_past_its_budget() {
    let db = hub(Config::in_memory().with_path_search_budget(64 * KIB));
    assert_refused_and_given_back(
        &db,
        &[(
            "sql",
            "SELECT * FROM GRAPH_TABLE (MATCH (s:Hub)-[:E*1..3]->(t) COLUMNS (t.id AS id))",
        )],
        LimitBound::PathSearchBudget,
    );
}

/// A search that fails in the middle of its second input row, after the
/// first row's search answered (from the first leaf, a path or two):
/// the statements hold nothing once it is refused.
const MID_ROW_MATRIX: [(&str, &str); 3] = [
    (
        "gql",
        "UNWIND ['First', 'Hub'] AS l MATCH (s) WHERE l IN labels(s) \
         MATCH (s)-[:E]->{0,2}(t) RETURN t",
    ),
    (
        "gql",
        "UNWIND ['First', 'Hub'] AS l MATCH (s) WHERE l IN labels(s) \
         MATCH p = ANY SHORTEST (s)-[:E]->+(t:Target) RETURN length(p) AS n",
    ),
    (
        "sparql",
        "PREFIX : <urn:> SELECT ?x WHERE { VALUES ?h { :leaf0 :hub } ?h :e+ ?x }",
    ),
];

#[test]
fn a_search_that_fails_after_a_row_that_answered_gives_everything_back() {
    let db = hub(Config::in_memory());
    for (language, query) in MID_ROW_MATRIX {
        let first = query.replace(" :hub", "").replace(", 'Hub'", "");
        let answered = db
            .execute_language(&first, language, None)
            .unwrap_or_else(|error| panic!("{language} `{first}`: {error}"));
        assert!(answered.row_count() > 0, "{language} `{first}` answers");
    }
    let db = hub(Config::in_memory().with_path_search_budget(64 * KIB));
    assert_refused_and_given_back(&db, &MID_ROW_MATRIX, LimitBound::PathSearchBudget);
    let db = hub(Config::in_memory().with_query_memory_limit(64 * KIB));
    assert_refused_and_given_back(&db, &MID_ROW_MATRIX, LimitBound::QueryMemoryLimit);
}

/// A property path inside `GRAPH <g>`, over the named graph's copy of the
/// hub's triples.
const NAMED_GRAPH_PATH: (&str, &str) = (
    "sparql",
    "PREFIX : <urn:> SELECT ?x WHERE { GRAPH <urn:g> { :hub :e+ ?x } }",
);

#[test]
fn a_property_path_in_a_named_graph_refuses_at_both_bounds() {
    let (language, query) = NAMED_GRAPH_PATH;
    let db = hub(Config::in_memory());
    let answered = db
        .execute_language(query, language, None)
        .unwrap_or_else(|error| panic!("{query}: {error}"));
    assert_eq!(answered.row_count(), LEAVES + 1, "{query}");
    let db = hub(Config::in_memory().with_path_search_budget(64 * KIB));
    assert_refused_and_given_back(&db, &[NAMED_GRAPH_PATH], LimitBound::PathSearchBudget);
    let db = hub(Config::in_memory().with_query_memory_limit(64 * KIB));
    assert_refused_and_given_back(&db, &[NAMED_GRAPH_PATH], LimitBound::QueryMemoryLimit);
}
