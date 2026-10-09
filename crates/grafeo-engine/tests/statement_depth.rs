//! A statement nested deeper than the stack allows fails with an error that
//! names the limit, and nothing overflows the stack below the limit (#573).
//!
//! Each shape runs on a thread with a small stack, so a test proves the
//! limits rather than the size of the machine's stack: the largest statement
//! of each shape that the limits accept runs on it, the next size up fails
//! with the limit's error, and so does a statement of the shape 10,000 levels
//! deep. A stack overflow aborts the test process.
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test statement_depth
//! ```

use grafeo_adapters::query::limits::MAX_NESTING_DEPTH;
use grafeo_common::types::Value;
use grafeo_engine::GrafeoDB;
use grafeo_engine::query::limits::MAX_PLAN_DEPTH;

/// The stack every statement runs on: well below the smallest stack Grafeo
/// runs on (the 1 MiB main thread on Windows, the default WebAssembly stack),
/// with room left for the frames of the caller.
const STACK: usize = 640 * 1024;

/// The largest size the shapes are tried at; far beyond every limit.
const HUGE: usize = 10_000;

/// The nesting limit, as a size.
const NESTING: usize = MAX_NESTING_DEPTH as usize;

#[derive(Clone, Copy, Debug)]
enum Language {
    Gql,
    #[cfg(feature = "cypher")]
    Cypher,
    #[cfg(feature = "sparql")]
    Sparql,
    #[cfg(feature = "gremlin")]
    Gremlin,
    #[cfg(feature = "graphql")]
    GraphQl,
    #[cfg(feature = "sql-pgq")]
    SqlPgq,
}

/// Runs `statement` on a fresh database on a thread with [`STACK`], and
/// returns its row count or its error.
fn run(language: Language, statement: String) -> Result<usize, String> {
    on_small_stack(move || {
        let db = GrafeoDB::new_in_memory();
        execute(&db, language, &statement)
            .map(|rows| rows.len())
            .map_err(|e| e.to_string())
    })
}

fn execute(
    db: &GrafeoDB,
    language: Language,
    statement: &str,
) -> grafeo_common::utils::error::Result<Vec<Vec<Value>>> {
    let result = match language {
        Language::Gql => db.execute(statement),
        #[cfg(feature = "cypher")]
        Language::Cypher => db.execute_cypher(statement),
        #[cfg(feature = "sparql")]
        Language::Sparql => db.execute_sparql(statement),
        #[cfg(feature = "gremlin")]
        Language::Gremlin => db.execute_gremlin(statement),
        #[cfg(feature = "graphql")]
        Language::GraphQl => db.execute_graphql(statement),
        #[cfg(feature = "sql-pgq")]
        Language::SqlPgq => db.execute_sql(statement),
    };
    result.map(|r| r.rows().to_vec())
}

fn on_small_stack<T: Send + 'static>(work: impl FnOnce() -> T + Send + 'static) -> T {
    std::thread::Builder::new()
        .name("statement on a small stack".to_string())
        .stack_size(STACK)
        .spawn(work)
        .expect("spawn the statement thread")
        .join()
        .expect("the statement thread does not panic")
}

/// Whether `error` is the error of the limit `limit`: it names the limit.
fn names_limit(error: &str, limit: usize) -> bool {
    error.contains(&format!(" {limit} ")) && (error.contains("depth") || error.contains("limit"))
}

/// The largest size from 1 to [`HUGE`] at which `shape` runs, found by
/// bisection; every size tried runs on the small stack. Sizes beyond it fail
/// with the error of `limit`, which the next size up and [`HUGE`] show.
fn largest_accepted(language: Language, shape: fn(usize) -> String, limit: usize) -> usize {
    let at = |size: usize| run(language, shape(size));
    let huge = at(HUGE).expect_err("a statement of size 10,000 is beyond the limits");
    assert!(
        names_limit(&huge, limit),
        "{language:?}, size {HUGE}: the error names the limit {limit}: {huge}"
    );
    if let Err(error) = at(1) {
        panic!(
            "{language:?}: the smallest statement of the shape runs: {}: {error}",
            shape(1)
        );
    }
    let (mut accepted, mut refused) = (1, HUGE);
    while refused - accepted > 1 {
        let size = accepted + (refused - accepted) / 2;
        match at(size) {
            Ok(_) => accepted = size,
            Err(error) => {
                // A nested shape may reach the plan depth limit first.
                assert!(
                    names_limit(&error, limit) || names_limit(&error, MAX_PLAN_DEPTH),
                    "{language:?}, size {size}: only a limit refuses: {error}"
                );
                refused = size;
            }
        }
    }
    accepted
}

/// A shape of statement: its name, the statement of each size, the limit
/// that bounds it and the size it runs up to at least.
type Shape = (&'static str, fn(usize) -> String, usize, usize);

/// Asserts each of `shapes` in `language` (see [`largest_accepted`]).
fn assert_bounded(language: Language, shapes: &[Shape]) {
    for &(name, shape, limit, floor) in shapes {
        let largest = largest_accepted(language, shape, limit);
        eprintln!("{language:?}, {name}: runs up to size {largest}");
        assert!(
            largest >= floor,
            "{language:?}, {name}: runs up to size {floor} at least, not only {largest}"
        );
    }
}

fn joined(size: usize, separator: &str, term: impl Fn(usize) -> String) -> String {
    (0..size).map(term).collect::<Vec<_>>().join(separator)
}

fn nested(size: usize, leaf: &str, wrap: impl Fn(String) -> String) -> String {
    (0..size).fold(leaf.to_string(), |inner, _| wrap(inner))
}

// ---------------------------------------------------------------------------
// GQL and Cypher
// ---------------------------------------------------------------------------

fn parentheses(size: usize) -> String {
    format!("RETURN {}1{} AS v", "(".repeat(size), ")".repeat(size))
}

fn parenthesized_sums(size: usize) -> String {
    let sum = nested(size, "1", |inner| format!("({inner} + 3)"));
    format!("RETURN {sum} AS v")
}

fn or_chain(size: usize) -> String {
    let terms = joined(size, " OR ", |i| format!("n.v = {i}"));
    format!("MATCH (n:Person) WHERE {terms} RETURN n.v")
}

fn and_chain(size: usize) -> String {
    let terms = joined(size, " AND ", |i| format!("n.v <> {i}"));
    format!("MATCH (n:Person) WHERE {terms} RETURN n.v")
}

fn sum_chain(size: usize) -> String {
    format!(
        "RETURN {} AS v",
        joined(size, " + ", |i| (i % 88).to_string())
    )
}

fn key_chain(size: usize) -> String {
    let map = nested(size, "3", |inner| format!("{{a: {inner}}}"));
    format!("UNWIND [{map}] AS m RETURN m{} AS v", ".a".repeat(size))
}

fn nested_lists(size: usize) -> String {
    let list = nested(size, "3", |inner| format!("[{inner}]"));
    format!("RETURN {list} AS v")
}

fn nested_maps(size: usize) -> String {
    let map = nested(size, "3", |inner| format!("{{a: {inner}}}"));
    format!("RETURN {map} AS v")
}

fn nested_case(size: usize) -> String {
    let case = nested(size, "19", |inner| {
        format!("CASE WHEN 3 > 1 THEN {inner} ELSE 88 END")
    });
    format!("RETURN {case} AS v")
}

fn nested_calls(size: usize) -> String {
    let call = nested(size, "-3", |inner| format!("abs({inner})"));
    format!("RETURN {call} AS v")
}

fn nested_list_comprehensions(size: usize) -> String {
    let list = nested(size, "[3]", |inner| format!("[x IN {inner} | x]"));
    format!("RETURN {list} AS v")
}

fn not_chain(size: usize) -> String {
    format!("RETURN {}true AS v", "NOT ".repeat(size))
}

fn nested_exists(size: usize) -> String {
    let exists = nested(size, "MATCH (z)", |inner| {
        format!("MATCH (x) WHERE EXISTS {{ {inner} }}")
    });
    format!("{exists} RETURN count(*) AS c")
}

fn nested_call_subqueries(size: usize) -> String {
    nested(size, "RETURN 1 AS x", |inner| {
        format!("CALL {{ {inner} }} RETURN x")
    })
}

fn with_chain(size: usize) -> String {
    format!(
        "MATCH (n:Person) {}RETURN count(n) AS c",
        "WITH n ".repeat(size)
    )
}

fn path_chain(size: usize) -> String {
    let hops = joined(size, "", |i| format!("-[:KNOWS]->(p{i})"));
    format!("MATCH (start){hops} RETURN start")
}

fn match_chain(size: usize) -> String {
    let clauses = joined(size, " ", |i| format!("MATCH (p{i}:Person)"));
    format!("{clauses} RETURN count(*) AS c")
}

fn unwind_chain(size: usize) -> String {
    let clauses = joined(size, " ", |i| format!("UNWIND [3] AS u{i}"));
    format!("{clauses} RETURN count(*) AS c")
}

fn inline_properties(size: usize) -> String {
    let properties = joined(size, ", ", |i| format!("p{i}: {i}"));
    format!("MATCH (n:Person {{{properties}}}) RETURN n")
}

fn set_items(size: usize) -> String {
    let items = joined(size, ", ", |i| format!("n.p{i} = {i}"));
    format!("MATCH (n:Person) SET {items} RETURN n")
}

fn union_chain(size: usize) -> String {
    joined(size, " UNION ALL ", |i| format!("RETURN {i} AS v"))
}

fn gql_insert(size: usize) -> String {
    let patterns = joined(size, ", ", |i| format!("(:Person {{v: {i}}})"));
    format!("INSERT {patterns}")
}

fn gql_insert_path(size: usize) -> String {
    let hops = joined(size, "", |i| format!("-[:KNOWS]->(:Person {{v: {i}}})"));
    format!("INSERT (:Person {{v: -1}}){hops}")
}

fn gql_nested_let(size: usize) -> String {
    let value = nested(size, "3", |inner| format!("LET x = {inner} IN x END"));
    format!("RETURN {value} AS v")
}

fn gql_parenthesized_patterns(size: usize) -> String {
    format!(
        "MATCH {}(a)-[:KNOWS]->(b){} RETURN a",
        "(".repeat(size),
        ")".repeat(size)
    )
}

#[cfg(feature = "cypher")]
fn cypher_create(size: usize) -> String {
    let patterns = joined(size, ", ", |i| format!("(:Person {{v: {i}}})"));
    format!("CREATE {patterns}")
}

#[cfg(feature = "cypher")]
fn cypher_nested_reduce(size: usize) -> String {
    let value = nested(size, "0", |inner| {
        format!("reduce(a = {inner}, x IN [1] | a + x)")
    });
    format!("RETURN {value} AS v")
}

#[cfg(feature = "cypher")]
fn cypher_nested_foreach(size: usize) -> String {
    let body = nested(size, "SET n.x = 3", |inner| {
        format!("FOREACH (i IN [1] | {inner})")
    });
    format!("MATCH (n:Person) {body}")
}

/// The expression shapes GQL and Cypher share, each bounded by the nesting
/// limit, with the size each one runs up to at least: parentheses, lists and
/// chains nest one level per size, calls, `CASE` and subqueries more.
fn shared_expression_shapes() -> Vec<Shape> {
    vec![
        ("parentheses", parentheses, NESTING, NESTING - 2),
        (
            "parenthesized sums",
            parenthesized_sums,
            NESTING,
            NESTING / 2 - 2,
        ),
        ("OR chain", or_chain, NESTING, NESTING - 2),
        ("AND chain", and_chain, NESTING, NESTING - 2),
        ("sum chain", sum_chain, NESTING, NESTING - 2),
        ("key access chain", key_chain, NESTING, NESTING / 2 - 2),
        ("nested lists", nested_lists, NESTING, NESTING - 2),
        ("nested maps", nested_maps, NESTING, NESTING - 2),
        ("nested CASE", nested_case, NESTING, NESTING / 2 - 2),
        (
            "nested function calls",
            nested_calls,
            NESTING,
            NESTING / 2 - 2,
        ),
        (
            "nested list comprehensions",
            nested_list_comprehensions,
            NESTING,
            NESTING / 3 - 2,
        ),
        ("NOT chain", not_chain, NESTING, NESTING - 2),
        ("nested EXISTS", nested_exists, NESTING, NESTING / 4 - 2),
        (
            "nested CALL subqueries",
            nested_call_subqueries,
            NESTING,
            NESTING / 4 - 2,
        ),
    ]
}

/// The clause shapes GQL and Cypher share, each bounded by the plan depth
/// limit.
fn shared_clause_shapes() -> Vec<Shape> {
    vec![
        ("WITH chain", with_chain, MAX_PLAN_DEPTH, MAX_PLAN_DEPTH - 8),
        ("path chain", path_chain, MAX_PLAN_DEPTH, MAX_PLAN_DEPTH - 8),
        (
            "MATCH chain",
            match_chain,
            MAX_PLAN_DEPTH,
            MAX_PLAN_DEPTH - 8,
        ),
        (
            "UNWIND chain",
            unwind_chain,
            MAX_PLAN_DEPTH,
            MAX_PLAN_DEPTH - 8,
        ),
        (
            "inline properties",
            inline_properties,
            MAX_PLAN_DEPTH,
            MAX_PLAN_DEPTH - 8,
        ),
        ("SET items", set_items, MAX_PLAN_DEPTH, MAX_PLAN_DEPTH - 8),
    ]
}

#[test]
fn gql_expressions_nest_up_to_the_nesting_limit_and_no_deeper() {
    let mut shapes = shared_expression_shapes();
    shapes.push(("nested LET", gql_nested_let, NESTING, NESTING / 2 - 2));
    shapes.push((
        "parenthesized patterns",
        gql_parenthesized_patterns,
        NESTING,
        NESTING / 2 - 2,
    ));
    // A UNION chain nests in the parser: each set operator is a level.
    shapes.push(("UNION chain", union_chain, NESTING, NESTING - 2));
    assert_bounded(Language::Gql, &shapes);
}

#[test]
fn gql_clauses_nest_up_to_the_plan_depth_limit_and_no_deeper() {
    let mut shapes = shared_clause_shapes();
    shapes.push((
        "INSERT patterns",
        gql_insert,
        MAX_PLAN_DEPTH,
        MAX_PLAN_DEPTH - 8,
    ));
    shapes.push((
        "INSERT path",
        gql_insert_path,
        MAX_PLAN_DEPTH,
        MAX_PLAN_DEPTH / 2 - 8,
    ));
    assert_bounded(Language::Gql, &shapes);
}

#[cfg(feature = "cypher")]
#[test]
fn cypher_expressions_nest_up_to_the_nesting_limit_and_no_deeper() {
    let mut shapes = shared_expression_shapes();
    shapes.push((
        "nested reduce",
        cypher_nested_reduce,
        NESTING,
        NESTING / 3 - 2,
    ));
    shapes.push((
        "nested FOREACH",
        cypher_nested_foreach,
        NESTING,
        NESTING / 4 - 2,
    ));
    assert_bounded(Language::Cypher, &shapes);
}

#[cfg(feature = "cypher")]
#[test]
fn cypher_clauses_nest_up_to_the_plan_depth_limit_and_no_deeper() {
    let mut shapes = shared_clause_shapes();
    shapes.push((
        "CREATE patterns",
        cypher_create,
        MAX_PLAN_DEPTH,
        MAX_PLAN_DEPTH - 8,
    ));
    assert_bounded(Language::Cypher, &shapes);
    // A Cypher UNION is flat: its branches are inputs of one operator.
    assert_eq!(run(Language::Cypher, union_chain(HUGE)), Ok(HUGE));
}

// ---------------------------------------------------------------------------
// SPARQL, Gremlin, GraphQL and SQL/PGQ
// ---------------------------------------------------------------------------

#[cfg(feature = "sparql")]
#[test]
fn sparql_queries_nest_up_to_their_limits_and_no_deeper() {
    fn filter(expression: &str) -> String {
        format!("SELECT ?x WHERE {{ ?x ?p ?o FILTER({expression}) }}")
    }
    // The group of the WHERE clause nests two levels, the FILTER one.
    let shapes: Vec<Shape> = vec![
        (
            "parentheses",
            |size| filter(&format!("{}?o > 3{}", "(".repeat(size), ")".repeat(size))),
            NESTING,
            NESTING - 6,
        ),
        (
            "OR chain",
            |size| filter(&joined(size, " || ", |i| format!("?o = {i}"))),
            NESTING,
            NESTING - 6,
        ),
        (
            "sum chain",
            |size| filter(&format!("?o = {}", joined(size, " + ", |i| i.to_string()))),
            NESTING,
            NESTING - 6,
        ),
        (
            "nested groups",
            |size| {
                format!(
                    "SELECT ?x WHERE {}?x ?p ?o{}",
                    "{ ".repeat(size),
                    " }".repeat(size)
                )
            },
            NESTING,
            NESTING / 2 - 2,
        ),
        (
            "nested path groups",
            |size| {
                format!(
                    "SELECT ?x WHERE {{ ?x {}<p>{} ?y }}",
                    "(".repeat(size),
                    ")".repeat(size)
                )
            },
            NESTING,
            NESTING - 3,
        ),
        (
            "triple patterns",
            |size| {
                let triples = joined(size, " . ", |i| format!("?x <p{i}> ?o{i}"));
                format!("SELECT ?x WHERE {{ {triples} }}")
            },
            MAX_PLAN_DEPTH,
            MAX_PLAN_DEPTH / 2 - 8,
        ),
        (
            "OPTIONAL chain",
            |size| {
                let optionals = joined(size, " ", |i| format!("OPTIONAL {{ ?x <p{i}> ?o{i} }}"));
                format!("SELECT ?x WHERE {{ ?x ?p ?o {optionals} }}")
            },
            MAX_PLAN_DEPTH,
            MAX_PLAN_DEPTH / 2 - 8,
        ),
    ];
    assert_bounded(Language::Sparql, &shapes);
}

#[cfg(feature = "gremlin")]
#[test]
fn gremlin_traversals_nest_up_to_their_limits_and_no_deeper() {
    // A nested traversal nests two levels.
    let shapes: Vec<Shape> = vec![
        (
            "nested not",
            |size| {
                format!(
                    "g.V().where({}out(){})",
                    "not(".repeat(size),
                    ")".repeat(size)
                )
            },
            NESTING,
            NESTING / 2 - 4,
        ),
        (
            "nested union",
            |size| {
                format!(
                    "g.V().union({}out(){})",
                    "union(".repeat(size),
                    ")".repeat(size)
                )
            },
            NESTING,
            NESTING / 2 - 4,
        ),
        (
            "step chain",
            |size| format!("g.V(){}", ".out()".repeat(size)),
            MAX_PLAN_DEPTH,
            MAX_PLAN_DEPTH / 2 - 8,
        ),
        (
            "has chain",
            |size| {
                let steps = joined(size, "", |i| format!(".has('p{i}', {i})"));
                format!("g.V(){steps}")
            },
            MAX_PLAN_DEPTH,
            MAX_PLAN_DEPTH / 2 - 8,
        ),
    ];
    assert_bounded(Language::Gremlin, &shapes);
}

#[cfg(feature = "graphql")]
#[test]
fn graphql_queries_nest_up_to_their_limits_and_no_deeper() {
    let shapes: Vec<Shape> = vec![
        (
            "nested selections",
            |size| {
                format!(
                    "{{ {}name{} }}",
                    "person { ".repeat(size),
                    " }".repeat(size)
                )
            },
            NESTING,
            NESTING / 2 - 2,
        ),
        (
            "nested list argument",
            |size| {
                format!(
                    "{{ person(tags: {}3{}) {{ name }} }}",
                    "[".repeat(size),
                    "]".repeat(size)
                )
            },
            NESTING,
            NESTING - 3,
        ),
        (
            "arguments",
            |size| {
                let arguments = joined(size, ", ", |i| format!("p{i}: {i}"));
                format!("{{ person({arguments}) {{ name }} }}")
            },
            MAX_PLAN_DEPTH,
            MAX_PLAN_DEPTH / 2 - 8,
        ),
    ];
    assert_bounded(Language::GraphQl, &shapes);
}

#[cfg(feature = "sql-pgq")]
#[test]
fn sql_pgq_queries_nest_up_to_their_limits_and_no_deeper() {
    fn filter(expression: &str) -> String {
        format!("SELECT * FROM GRAPH_TABLE (MATCH (n) WHERE {expression} COLUMNS (n.v AS v))")
    }
    let shapes: Vec<Shape> = vec![
        (
            "parentheses",
            |size| filter(&format!("{}n.v > 3{}", "(".repeat(size), ")".repeat(size))),
            NESTING,
            NESTING - 3,
        ),
        (
            "OR chain",
            |size| filter(&joined(size, " OR ", |i| format!("n.v = {i}"))),
            NESTING,
            NESTING - 3,
        ),
        (
            "nested CASE",
            |size| {
                let case = nested(size, "1", |inner| {
                    format!("CASE WHEN true THEN {inner} ELSE 0 END")
                });
                filter(&format!("{case} = 1"))
            },
            NESTING,
            NESTING / 3 - 2,
        ),
        (
            "nested function calls",
            |size| {
                let call = nested(size, "n.v", |inner| format!("abs({inner})"));
                filter(&format!("{call} > 3"))
            },
            NESTING,
            NESTING / 2 - 2,
        ),
        (
            "path chain",
            |size| {
                let hops = joined(size, "", |i| format!("-[:KNOWS]->(p{i})"));
                format!("SELECT * FROM GRAPH_TABLE (MATCH (a){hops} COLUMNS (a.v AS v))")
            },
            MAX_PLAN_DEPTH,
            MAX_PLAN_DEPTH - 8,
        ),
    ];
    assert_bounded(Language::SqlPgq, &shapes);
    // A query may hold any number of comments.
    let commented = format!(
        "{}SELECT * FROM GRAPH_TABLE (MATCH (n) COLUMNS (n.v AS v))",
        "-- Alix\n".repeat(100_000)
    );
    assert_eq!(run(Language::SqlPgq, commented), Ok(0));
}

// ---------------------------------------------------------------------------
// The statements of #573
// ---------------------------------------------------------------------------

#[test]
fn the_reported_statements_fail_with_the_limit_error_or_run() {
    // A 400-pattern INSERT and a 10,000-term OR chain.
    match run(Language::Gql, gql_insert(400)) {
        Ok(_) => {}
        Err(error) => assert!(names_limit(&error, MAX_PLAN_DEPTH), "{error}"),
    }
    let error = run(Language::Gql, or_chain(10_000)).expect_err("beyond the nesting limit");
    assert!(names_limit(&error, NESTING), "{error}");
}

#[test]
fn an_insert_at_the_plan_depth_limit_creates_every_node_and_a_refused_one_none() {
    let largest = largest_accepted(Language::Gql, gql_insert, MAX_PLAN_DEPTH);
    let count = on_small_stack(move || {
        let db = GrafeoDB::new_in_memory();
        db.execute(&gql_insert(largest))
            .expect("the largest INSERT runs");
        let refused = db.execute(&gql_insert(largest + 1));
        assert!(refused.is_err(), "one pattern more is refused");
        let result = db.execute("MATCH (n:Person) RETURN count(n) AS c").unwrap();
        result.rows()[0][0].clone()
    });
    assert_eq!(
        count,
        Value::Int64(i64::try_from(largest).unwrap()),
        "every pattern of the largest INSERT creates its node, the refused one none"
    );
}

#[test]
fn a_long_flat_list_is_not_deep() {
    let flat = |size: usize| {
        format!(
            "RETURN size([{}]) AS v",
            joined(size, ", ", |i| i.to_string())
        )
    };
    assert_eq!(run(Language::Gql, flat(HUGE)), Ok(1));
    #[cfg(feature = "cypher")]
    assert_eq!(run(Language::Cypher, flat(HUGE)), Ok(1));
}
