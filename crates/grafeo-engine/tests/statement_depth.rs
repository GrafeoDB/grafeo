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

fn union_chain(size: usize) -> String {
    joined(size, " UNION ALL ", |i| format!("RETURN {i} AS v"))
}

fn except_chain(size: usize) -> String {
    joined(size, " EXCEPT ", |i| format!("RETURN {i} AS v"))
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
/// chains of `+` nest one level per size, calls, `CASE` and subqueries more.
/// (A chain of `AND`, `OR` or `XOR` is a balanced tree: see
/// [`chains_of_and_or_and_xor_of_any_length_run_on_a_small_stack`].)
fn shared_expression_shapes() -> Vec<Shape> {
    vec![
        ("parentheses", parentheses, NESTING, NESTING - 2),
        (
            "parenthesized sums",
            parenthesized_sums,
            NESTING,
            NESTING / 2 - 2,
        ),
        // `+` is not associative (strings, overflow), so its chain is built
        // from the left, one level per operator.
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
/// limit: each clause is an operator over the one before it.
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
    // EXCEPT is not associative: each one nests a level.
    shapes.push(("EXCEPT chain", except_chain, NESTING, NESTING - 2));
    assert_bounded(Language::Gql, &shapes);
    // A UNION of any number of queries is a balanced tree.
    assert_eq!(run(Language::Gql, union_chain(HUGE)), Ok(HUGE));
}

#[test]
fn gql_clauses_nest_up_to_the_plan_depth_limit_and_no_deeper() {
    assert_bounded(Language::Gql, &shared_clause_shapes());
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
    assert_bounded(Language::Cypher, &shared_clause_shapes());
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
fn the_reported_statements_run() {
    // A 400-pattern INSERT and a 10,000-term OR chain.
    assert_eq!(
        persons_after(Language::Gql, None, gql_insert(400)),
        Ok(inserted(400))
    );
    assert_eq!(
        selected(Some(PERSONS), Language::Gql, or_chain(10_000)),
        Ok(vec![3, 19, 88])
    );
}

#[test]
fn a_statement_refused_at_the_plan_depth_limit_writes_nothing() {
    let deep = |size: usize| {
        format!(
            "INSERT (n:Person {{v: 3}}) {}RETURN n.v",
            "WITH n ".repeat(size)
        )
    };
    let mut languages = vec![Language::Gql];
    #[cfg(feature = "cypher")]
    languages.push(Language::Cypher);
    for language in languages {
        let statement = match language {
            Language::Gql => deep(MAX_PLAN_DEPTH),
            _ => deep(MAX_PLAN_DEPTH).replacen("INSERT", "CREATE", 1),
        };
        let shallow = statement.replacen(&"WITH n ".repeat(MAX_PLAN_DEPTH), "WITH n ", 1);
        let (error, count) = on_small_stack(move || {
            let db = GrafeoDB::new_in_memory();
            let error = execute(&db, language, &statement).expect_err("beyond the plan's limit");
            let count = db.execute("MATCH (n:Person) RETURN count(n) AS c").unwrap();
            (error.to_string(), count.rows()[0][0].clone())
        });
        assert!(names_limit(&error, MAX_PLAN_DEPTH), "{language:?}: {error}");
        assert_eq!(
            count,
            Value::Int64(0),
            "{language:?}: the refused statement writes nothing"
        );
        // The same statement within the limit creates its node.
        assert_eq!(
            persons_after(language, None, shallow),
            Ok((1, 3)),
            "{language:?}"
        );
    }
}

// ---------------------------------------------------------------------------
// Long chains, many patterns and long lists are not deep
// ---------------------------------------------------------------------------

/// The persons the long chains select from: `v` is 3, 19, 88 or 100,088.
const PERSONS: &str =
    "INSERT (:Person {v: 3}), (:Person {v: 19}), (:Person {v: 88}), (:Person {v: 100088})";

/// Runs `setup` (GQL, when given) and then `statement` on one database on the
/// small stack, and returns the integers of the first column of its rows in
/// ascending order, or its error.
fn selected(
    setup: Option<&str>,
    language: Language,
    statement: String,
) -> Result<Vec<i64>, String> {
    let setup = setup.map(str::to_string);
    on_small_stack(move || {
        let db = GrafeoDB::new_in_memory();
        if let Some(setup) = setup {
            db.execute(&setup).map_err(|e| e.to_string())?;
        }
        let rows = execute(&db, language, &statement).map_err(|e| e.to_string())?;
        let mut values: Vec<i64> = rows
            .iter()
            .map(|row| match row.first() {
                Some(Value::Int64(v)) => *v,
                other => panic!("{language:?}: an integer, not {other:?}"),
            })
            .collect();
        values.sort_unstable();
        Ok(values)
    })
}

fn xor_chain(size: usize) -> String {
    let terms = joined(size, " XOR ", |i| format!("n.v = {i}"));
    format!("MATCH (n:Person) WHERE {terms} RETURN n.v")
}

/// `n.v > 18 AND n.v > 17 AND ...`: true for a `v` above 18.
fn range_chain(size: usize) -> String {
    let terms = joined(size, " AND ", |i| {
        format!("n.v > {}", 18 - i64::try_from(i).unwrap())
    });
    format!("MATCH (n:Person) WHERE {terms} RETURN n.v")
}

/// An OR chain computed in a projection, then filtered on.
fn projected_or_chain(size: usize) -> String {
    let terms = joined(size, " OR ", |i| format!("n.v = {i}"));
    format!("MATCH (n:Person) WITH n, ({terms}) AS hit WHERE hit RETURN n.v")
}

#[test]
fn chains_of_and_or_and_xor_of_any_length_run_on_a_small_stack() {
    let mut languages = vec![Language::Gql];
    #[cfg(feature = "cypher")]
    languages.push(Language::Cypher);
    for language in languages {
        let run = |shape: fn(usize) -> String| selected(Some(PERSONS), language, shape(HUGE));
        assert_eq!(run(or_chain), Ok(vec![3, 19, 88]), "{language:?}: OR");
        assert_eq!(run(xor_chain), Ok(vec![3, 19, 88]), "{language:?}: XOR");
        assert_eq!(run(and_chain), Ok(vec![100088]), "{language:?}: AND");
        assert_eq!(
            run(range_chain),
            Ok(vec![19, 88, 100088]),
            "{language:?}: AND of ranges"
        );
        assert_eq!(
            run(projected_or_chain),
            Ok(vec![3, 19, 88]),
            "{language:?}: projected OR"
        );
    }
}

#[cfg(feature = "sql-pgq")]
#[test]
fn sql_pgq_chains_of_any_length_run_on_a_small_stack() {
    let query = |separator: &str, term: fn(usize) -> String| {
        let terms = joined(HUGE, separator, term);
        format!("SELECT * FROM GRAPH_TABLE (MATCH (n:Person) WHERE {terms} COLUMNS (n.v AS v))")
    };
    assert_eq!(
        selected(
            Some(PERSONS),
            Language::SqlPgq,
            query(" OR ", |i| format!("n.v = {i}"))
        ),
        Ok(vec![3, 19, 88])
    );
    assert_eq!(
        selected(
            Some(PERSONS),
            Language::SqlPgq,
            query(" AND ", |i| format!("n.v <> {i}"))
        ),
        Ok(vec![100088])
    );
}

#[cfg(feature = "sparql")]
#[test]
fn sparql_chains_of_any_length_run_on_a_small_stack() {
    let filter = |separator: &str, term: fn(usize) -> String| {
        let terms = joined(HUGE, separator, term);
        format!("SELECT ?v WHERE {{ ?x <http://ex/v> ?v FILTER({terms}) }}")
    };
    let data = "INSERT DATA { <http://ex/alix> <http://ex/v> 3 . <http://ex/gus> <http://ex/v> 19 . \
                <http://ex/vincent> <http://ex/v> 88 . <http://ex/mia> <http://ex/v> 100088 }";
    let run = move |statement: String| {
        on_small_stack(move || {
            let db = GrafeoDB::new_in_memory();
            db.execute_sparql(data).map_err(|e| e.to_string())?;
            let rows = execute(&db, Language::Sparql, &statement).map_err(|e| e.to_string())?;
            Ok::<_, String>(rows.len())
        })
    };
    assert_eq!(run(filter(" || ", |i| format!("?v = {i}"))), Ok(3));
    assert_eq!(run(filter(" && ", |i| format!("?v != {i}"))), Ok(1));
}

/// The node count and the sum of `v` of the persons after `statement`.
fn persons_after(
    language: Language,
    setup: Option<&'static str>,
    statement: String,
) -> Result<(i64, i64), String> {
    on_small_stack(move || {
        let db = GrafeoDB::new_in_memory();
        if let Some(setup) = setup {
            db.execute(setup).map_err(|e| e.to_string())?;
        }
        execute(&db, language, &statement).map_err(|e| e.to_string())?;
        let rows = db
            .execute("MATCH (n:Person) RETURN count(n) AS c, sum(n.v) AS s")
            .map_err(|e| e.to_string())?;
        match rows.rows() {
            [row] => match (&row[0], &row[1]) {
                (Value::Int64(count), Value::Int64(sum)) => Ok((*count, *sum)),
                other => panic!("a count and a sum, not {other:?}"),
            },
            other => panic!("one row, not {other:?}"),
        }
    })
}

/// The count and the sum of `v` of the persons [`gql_insert`] of `size` creates.
fn inserted(size: usize) -> (i64, i64) {
    let size = i64::try_from(size).unwrap();
    (size, size * (size - 1) / 2)
}

#[test]
fn an_insert_of_any_number_of_patterns_runs_on_a_small_stack() {
    assert_eq!(
        persons_after(Language::Gql, None, gql_insert(HUGE)),
        Ok(inserted(HUGE))
    );
    // After a MATCH: each pattern also creates an edge from the anchor.
    let after_match = format!(
        "MATCH (a:Anchor) INSERT {}",
        joined(HUGE, ", ", |i| format!("(a)-[:HAS]->(:Person {{v: {i}}})"))
    );
    assert_eq!(
        persons_after(Language::Gql, Some("INSERT (:Anchor)"), after_match),
        Ok(inserted(HUGE))
    );
    // A path of any length: the nodes are created in order, then counted.
    let path = gql_insert_path(HUGE);
    assert_eq!(
        persons_after(Language::Gql, None, path),
        Ok((inserted(HUGE).0 + 1, inserted(HUGE).1 - 1))
    );
    #[cfg(feature = "cypher")]
    assert_eq!(
        persons_after(Language::Cypher, None, cypher_create(HUGE)),
        Ok(inserted(HUGE))
    );
}

#[test]
fn long_property_maps_and_set_lists_run_on_a_small_stack() {
    // A property map of 10,000 entries is 10,000 conditions joined by AND.
    // (The same key each time: a filter reads a property of a node in time
    // that grows with the node's property count.)
    let entries = joined(HUGE, ", ", |_| "v: 3".to_string());
    let items = joined(HUGE, ", ", |i| format!("n.q{i} = {i}"));
    let mut languages = vec![Language::Gql];
    #[cfg(feature = "cypher")]
    languages.push(Language::Cypher);
    for language in languages {
        let matched = format!("MATCH (n:Person {{{entries}}}) RETURN n.v");
        assert_eq!(
            selected(Some(PERSONS), language, matched),
            Ok(vec![3]),
            "{language:?}: map"
        );
        // A SET of 10,000 constants on one node is one operator.
        let set = format!("MATCH (n:Person {{v: 19}}) SET {items} RETURN n.q88");
        assert_eq!(
            selected(Some(PERSONS), language, set),
            Ok(vec![88]),
            "{language:?}: SET"
        );
    }
    #[cfg(feature = "cypher")]
    {
        let person = format!(
            "INSERT (:Person {{{}}})",
            joined(HUGE, ", ", |i| format!("p{i}: {i}"))
        );
        let removed = joined(HUGE - 1, ", ", |i| format!("n.p{i}"));
        let remove = format!("MATCH (n:Person) REMOVE {removed} RETURN size(keys(n))");
        assert_eq!(
            selected(Some(&person), Language::Cypher, remove),
            Ok(vec![1])
        );
    }
    #[cfg(feature = "graphql")]
    {
        let query = format!("{{ person({entries}) {{ v }} }}");
        assert_eq!(
            selected(Some(PERSONS), Language::GraphQl, query),
            Ok(vec![3])
        );
    }
}

#[test]
fn a_long_in_list_runs_on_a_small_stack() {
    let list = joined(100_000, ", ", |i| i.to_string());
    let query = format!("MATCH (n:Person) WHERE n.v IN [{list}] RETURN n.v");
    assert_eq!(
        selected(Some(PERSONS), Language::Gql, query.clone()),
        Ok(vec![3, 19, 88])
    );
    #[cfg(feature = "cypher")]
    assert_eq!(
        selected(Some(PERSONS), Language::Cypher, query.clone()),
        Ok(vec![3, 19, 88])
    );
    // Through a property index: one lookup per item.
    let indexed = on_small_stack(move || {
        let db = GrafeoDB::new_in_memory();
        db.execute(PERSONS).unwrap();
        db.create_property_index("v").unwrap();
        let mut values: Vec<Value> = db
            .execute(&query)
            .unwrap()
            .rows()
            .iter()
            .map(|row| row[0].clone())
            .collect();
        values.sort_by_key(|value| value.as_int64());
        values
    });
    assert_eq!(
        indexed,
        [Value::Int64(3), Value::Int64(19), Value::Int64(88)]
    );
}

#[test]
fn a_flat_insert_or_set_keeps_the_order_of_its_effects() {
    let mut languages = vec![Language::Gql];
    #[cfg(feature = "cypher")]
    languages.push(Language::Cypher);
    for language in languages {
        // A pattern reads the nodes the patterns before it in the same clause
        // created, and an edge connects them.
        let insert = "INSERT (a:Person {v: 3}), (b:Person {v: a.v + 16}), \
                      (a)-[:KNOWS]->(b)-[:KNOWS]->(:Person {v: b.v + 69})";
        let insert = match language {
            Language::Gql => insert.to_string(),
            _ => insert.replacen("INSERT", "CREATE", 1),
        };
        let path = "MATCH (a:Person)-[:KNOWS]->(b:Person)-[:KNOWS]->(c:Person) \
                    RETURN a.v * 10000 + b.v * 100 + c.v";
        let rows = on_small_stack(move || {
            let db = GrafeoDB::new_in_memory();
            execute(&db, language, &insert).unwrap();
            db.execute(path).unwrap().rows().to_vec()
        });
        assert_eq!(
            rows,
            [vec![Value::Int64(31988)]],
            "{language:?}: 3 -> 19 -> 88"
        );
        // A SET item reads what the items before it wrote.
        let set = "MATCH (n:Person {v: 3}) SET n.a = 19, n.b = n.a + 69, n.c = 3 RETURN n.b";
        assert_eq!(
            selected(Some(PERSONS), language, set.to_string()),
            Ok(vec![88]),
            "{language:?}"
        );
    }
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
