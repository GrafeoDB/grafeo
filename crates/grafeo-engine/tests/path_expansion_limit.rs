//! The path expansion limit (#492): `Config::with_path_expansion_limit(expansions)`, 100
//! million by default, bounds the work one statement's path searches do together. An expansion
//! is an edge a search reads to extend a path, plus each node and edge a search copies out of
//! a path into a row (a path variable: `2 * length + 1`; an edge list: `length`). A search past
//! the limit is refused with its own code (`ErrorCode::QueryLimit`, not retryable), naming the
//! limit, its setting and what to do instead. Reachability (`DISTINCT` targets) and RDF
//! property paths are not charged.
//!
//! The star of these tests has a centre with five edges to five leaves and none out of them,
//! so its counts are exact: five edges read, five rows of length one.
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test main path_expansion_limit::
//! ```

#![cfg(feature = "gql")]

use grafeo_common::types::Value;
use grafeo_common::utils::error::{Error, ErrorCode, LimitBound};
use grafeo_engine::config::ConfigError;
use grafeo_engine::{Config, GrafeoDB};

/// The code of the expansion limit's refusal: the next free query code (Q007 and Q008 are the
/// permission refusals).
const EXPANSION_CODE: &str = "GRAFEO-Q009";

fn limited(expansions: u64) -> GrafeoDB {
    let db =
        GrafeoDB::with_config(Config::in_memory().with_path_expansion_limit(expansions)).unwrap();
    db.execute(
        "INSERT (c:Star {n: 0}), (c)-[:E]->(:Leaf {n: 1}), (c)-[:E]->(:Leaf {n: 2}), \
         (c)-[:E]->(:Leaf {n: 3}), (c)-[:E]->(:Leaf {n: 4}), (c)-[:E]->(:Leaf {n: 5})",
    )
    .unwrap();
    db
}

/// `error` is the expansion limit's refusal, with `advice` in it.
fn assert_over_the_limit(case: &str, error: &Error, limit: u64, advice: &str) {
    let message = error.to_string();
    assert_eq!(
        error.error_code(),
        ErrorCode::QueryLimit,
        "{case}: {message}"
    );
    assert_eq!(error.error_code().as_str(), EXPANSION_CODE, "{case}");
    assert!(!error.error_code().is_retryable(), "{case}");
    assert!(
        message.contains(LimitBound::PathExpansionLimit.setting()),
        "{case}: names the setting: {message}"
    );
    assert!(
        message.contains(&limit.to_string()),
        "{case}: names the limit {limit}: {message}"
    );
    assert!(
        message.contains(advice),
        "{case}: says what to do: {message}"
    );
}

/// At exactly the expansions a statement makes it runs; one fewer refuses it.
#[test]
fn a_search_runs_at_its_exact_count_and_fails_one_below() {
    for (case, query, expansions, rows) in [
        (
            "five edges read",
            "MATCH (s:Star)-[:E*1..2]->(t) RETURN t.n AS n",
            5,
            5,
        ),
        (
            "and a path of 3 elements copied per row",
            "MATCH p = (s:Star)-[:E*1..2]->(t) RETURN p",
            5 + 5 * 3,
            5,
        ),
        (
            // GQL binds a named edge of a variable-length pattern with its path, whose edges
            // the edge's property map and WHERE read: the row copies the path's 3 elements.
            // The charge follows the copies made; if the hidden path stops being built for
            // an edge list alone, this count falls with it.
            "and an edge list, from its path of 3 elements per row",
            "MATCH (s:Star)-[es:E*1..2]->(t) RETURN es",
            5 + 5 * 3,
            5,
        ),
    ] {
        let result = limited(expansions).execute(query);
        assert_eq!(
            result
                .unwrap_or_else(|e| panic!("{case}: {e}"))
                .rows()
                .len(),
            rows,
            "{case}"
        );
        let error = limited(expansions - 1).execute(query).expect_err(case);
        assert_over_the_limit(case, &error, expansions - 1, "bound the quantifier");
    }
}

#[cfg(feature = "cypher")]
#[test]
fn cypher_is_held_to_the_same_limit() {
    let query = "MATCH (s:Star)-[:E*1..2]->(t) RETURN t.n AS n";
    assert_eq!(limited(5).execute_cypher(query).unwrap().rows().len(), 5);
    let error = limited(4).execute_cypher(query).expect_err("one below");
    assert_over_the_limit("cypher", &error, 4, "bound the quantifier");
}

/// A reachability search visits each node once: it is not charged.
#[test]
fn distinct_targets_are_not_charged() {
    let db = limited(1);
    let rows = db
        .execute("MATCH (s:Star)-[:E*1..5]->(t) RETURN count(DISTINCT t) AS c")
        .unwrap();
    assert_eq!(rows.rows()[0][0], Value::Int64(5));
}

/// The searches of one statement share its limit: two searches of 5 each pass a limit of 5
/// alone and fail a limit of 9 together.
#[test]
fn the_searches_of_one_statement_share_its_limit() {
    let both = "MATCH (s:Star)-[:E*1..2]->(t) WITH count(t) AS first \
                MATCH (c:Star)-[:E*1..2]->(u) RETURN first, count(u) AS second";
    let db = limited(9);
    db.execute("MATCH (s:Star)-[:E*1..2]->(t) RETURN count(t)")
        .unwrap();
    let error = db.execute(both).expect_err("10 together, over 9");
    assert_over_the_limit("two searches", &error, 9, "bound the quantifier");
    assert_eq!(
        limited(10).execute(both).unwrap().rows(),
        [vec![Value::Int64(5), Value::Int64(5)]]
    );
    // Each statement has its own count: the next one runs again.
    db.execute("MATCH (s:Star)-[:E*1..2]->(t) RETURN count(t)")
        .unwrap();
}

/// ALL SHORTEST over many ties is refused with the shortest-path advice; under the default it
/// returns every tie.
#[test]
fn all_shortest_over_many_ties_is_refused_with_its_advice() {
    let setup = |db: &GrafeoDB| {
        // Eleven layers of two nodes, each linked to both of the next: from one end to
        // the other, 2^9 shortest paths.
        db.execute(
            "UNWIND range(0, 10) AS layer UNWIND [0, 1] AS side \
             INSERT (:Grid {layer: layer, side: side})",
        )
        .unwrap();
        db.execute("MATCH (a:Grid), (b:Grid) WHERE b.layer = a.layer + 1 INSERT (a)-[:E]->(b)")
            .unwrap();
    };
    let query = "MATCH p = ALL SHORTEST (a:Grid {layer: 0, side: 0})-[:E]->+(b:Grid {layer: 10, side: 0}) \
                 RETURN count(p) AS c";
    let roomy = GrafeoDB::new_in_memory();
    setup(&roomy);
    assert_eq!(
        roomy.execute(query).unwrap().rows()[0][0],
        Value::Int64(512)
    );
    let tight =
        GrafeoDB::with_config(Config::in_memory().with_path_expansion_limit(1_000)).unwrap();
    setup(&tight);
    let error = tight.execute(query).expect_err("512 paths of 21 elements");
    assert_over_the_limit("all shortest", &error, 1_000, "keep fewer paths");
}

/// A shortest-path search is charged the edges it reads, not only the path it returns: the
/// one path from the centre to a leaf has 3 elements, but the search reads the centre's 5 edges.
#[test]
fn a_shortest_path_search_is_charged_the_edges_it_reads() {
    let query = "MATCH p = ANY SHORTEST (s:Star)-[:E]->+(t:Leaf {n: 5}) RETURN length(p) AS l";
    assert_eq!(
        limited(100).execute(query).unwrap().rows(),
        [vec![Value::Int64(1)]]
    );
    let error = limited(3).execute(query).expect_err("5 edges read over 3");
    assert_over_the_limit("any shortest", &error, 3, "keep fewer paths");
}

/// A refusal in a transaction leaves the transaction usable, its earlier write in place.
#[test]
fn a_refusal_in_a_transaction_changes_nothing_else() {
    let db = limited(4);
    let mut session = db.session();
    session.begin_transaction().unwrap();
    session.execute("INSERT (:Note {n: 1})").unwrap();
    let error = session
        .execute("MATCH (s:Star)-[:E*1..2]->(t) RETURN t.n AS n")
        .expect_err("5 over 4");
    assert_over_the_limit("in a transaction", &error, 4, "bound the quantifier");
    session.execute("INSERT (:Note {n: 2})").unwrap();
    session.commit().unwrap();
    assert_eq!(
        db.execute("MATCH (n:Note) RETURN count(n) AS c")
            .unwrap()
            .rows()[0][0],
        Value::Int64(2)
    );
}

#[test]
fn the_default_is_100_million_and_zero_is_refused() {
    assert_eq!(Config::in_memory().path_expansion_limit(), 100_000_000);
    assert!(matches!(
        Config::in_memory().with_path_expansion_limit(0).validate(),
        Err(ConfigError::ZeroPathExpansionLimit)
    ));
}

/// RDF property paths are not charged: a limit of 1 lets one run.
#[cfg(all(feature = "sparql", feature = "triple-store"))]
#[test]
fn rdf_property_paths_are_not_charged() {
    let db = limited(1);
    db.execute_sparql(
        "PREFIX ex: <http://example.org/> INSERT DATA { ex:a ex:p ex:b . ex:b ex:p ex:c . \
         ex:c ex:p ex:d }",
    )
    .unwrap();
    let rows = db
        .execute_sparql(
            "PREFIX ex: <http://example.org/> SELECT (COUNT(?x) AS ?c) WHERE { ex:a ex:p+ ?x }",
        )
        .unwrap();
    assert_eq!(rows.rows().len(), 1);
}

/// An empty in-memory database with a path expansion limit of `expansions`.
fn empty(expansions: u64) -> GrafeoDB {
    GrafeoDB::with_config(Config::in_memory().with_path_expansion_limit(expansions)).unwrap()
}

/// Cypher binds a named variable-length edge as the list of its edges alone, with no hidden
/// path: the 5 edges read, and 1 edge copied for each of the 5 rows, 5 + 5.
#[cfg(feature = "cypher")]
#[test]
fn a_cypher_edge_list_is_charged_its_edges_alone() {
    let query = "MATCH (s:Star)-[es:E*1..2]->(t) RETURN es";
    assert_eq!(limited(10).execute_cypher(query).unwrap().rows().len(), 5);
    let error = limited(9).execute_cypher(query).expect_err("10 over 9");
    assert_over_the_limit("cypher edge list", &error, 9, "bound the quantifier");
}

/// A shortest-path search above another path search has only what the statement has left
/// once the search below it has run. The variable-length search reads the centre's 5 edges
/// (5); for each of its 5 rows the shortest path from the centre to that leaf reads the
/// centre's 5 edges and copies a path of 2 nodes and 1 edge (5 * (5 + 3) = 40). Each fits a
/// limit of 44 alone; together they make 45.
#[test]
fn a_shortest_path_search_above_another_search_keeps_to_the_limit() {
    let query = "MATCH (s:Star)-[:E*1..2]->(t) MATCH p = ANY SHORTEST (s)-[:E]->+(t) \
                 RETURN length(p) AS l";
    assert_eq!(limited(45).execute(query).unwrap().rows().len(), 5);
    let error = limited(44)
        .execute(query)
        .expect_err("45 together, over 44");
    assert_over_the_limit("a search above a search", &error, 44, "keep fewer paths");
}

/// A star whose centre has 2 edges of weight 1 and 3 of weight 0.
fn weighted(expansions: u64) -> GrafeoDB {
    let db = empty(expansions);
    db.execute(
        "INSERT (c:Star {n: 0}), (c)-[:E {w: 1}]->(:Leaf {n: 1}), \
         (c)-[:E {w: 1}]->(:Leaf {n: 2}), (c)-[:E {w: 0}]->(:Leaf {n: 3}), \
         (c)-[:E {w: 0}]->(:Leaf {n: 4}), (c)-[:E {w: 0}]->(:Leaf {n: 5})",
    )
    .unwrap();
    db
}

/// An edge condition checked during the search (`all(e IN edges(p) WHERE ...)` over the
/// expand that binds `p`) does not hide the edges it rejects: the search reads the centre's 5
/// edges (5), and the 2 that pass are rows copying a path of 3 elements each (6), 11 in all.
/// Charging only the edges that pass would make it 8.
#[test]
fn an_edge_condition_is_charged_the_edges_it_rejects() {
    let query = "MATCH p = (s:Star)-[:E*1..2]->(t) WHERE all(e IN edges(p) WHERE e.w = 1) \
                 RETURN p";
    // At 11 it runs, which also shows the condition is checked during the search: filtered
    // after it, the search would copy all 5 paths (5 + 15 = 20).
    assert_eq!(weighted(11).execute(query).unwrap().rows().len(), 2);
    let error = weighted(10).execute(query).expect_err("11 over 10");
    assert_over_the_limit("an edge condition", &error, 10, "bound the quantifier");
}

/// A bounded walk that keeps its path does work quadratic in its length, which the limit
/// bounds where the edges it reads would not. On a 3-cycle a walk of up to 300 edges from one
/// node reads 300 edges (one for each path it extends) and copies 2L + 1 elements for each of
/// its 300 rows, L from 1 to 300 (300 * 301 + 300 = 90,600): 90,900 in all, of which a limit
/// on the edges read would see 300.
#[test]
fn a_bounded_walk_keeping_its_path_is_charged_its_copies() {
    let cycle = |expansions| {
        let db = empty(expansions);
        db.execute(
            "INSERT (a:Cycle {n: 0}), (b:Cycle {n: 1}), (c:Cycle {n: 2}), \
             (a)-[:E]->(b), (b)-[:E]->(c), (c)-[:E]->(a)",
        )
        .unwrap();
        db
    };
    let query = "MATCH p = (a:Cycle {n: 0})-[:E*1..300]->(b) RETURN p";
    assert_eq!(cycle(90_900).execute(query).unwrap().rows().len(), 300);
    let error = cycle(90_899)
        .execute(query)
        .expect_err("90,900 over 90,899");
    assert_over_the_limit("a long walk", &error, 90_899, "bound the quantifier");
}

/// A stream that crosses the limit returns its first rows, then the refusal, and no row after
/// it. Each of 3,000 sources has one edge, so each is charged 1: with a limit of 2,500 the
/// first chunk is delivered and a later one is refused.
#[test]
fn a_stream_crossing_the_limit_returns_its_first_rows_then_the_refusal() {
    let db = empty(2_500);
    db.execute("UNWIND range(1, 3000) AS i INSERT (:Src {n: i})-[:E]->(:Dst {n: i})")
        .unwrap();
    let session = db.session();
    let mut stream = session
        .execute_streaming("MATCH (s:Src)-[:E*1..2]->(t) RETURN t.n AS n")
        .unwrap();
    let mut delivered = 0;
    let error = loop {
        match stream.next_chunk() {
            Ok(Some(chunk)) => delivered += chunk.selected_indices().count(),
            Ok(None) => panic!("the stream ended after {delivered} rows without the refusal"),
            Err(error) => break error,
        }
    };
    assert!(delivered > 0, "the first rows come before the refusal");
    assert!(delivered <= 2_500, "{delivered} rows, more than the limit");
    assert_over_the_limit("a stream", &error, 2_500, "bound the quantifier");
    // However often it is read after the refusal, the stream gives no more rows.
    for _ in 0..4_000 {
        match stream.next_chunk() {
            Ok(None) => break,
            Ok(Some(chunk)) => assert_eq!(
                chunk.selected_indices().count(),
                0,
                "a row after the refusal, {delivered} before it"
            ),
            Err(_) => {}
        }
    }
}

/// A TRAIL search over the limit is refused: it reads the centre's 5 edges.
#[test]
fn a_trail_search_over_the_limit_is_refused() {
    let query = "MATCH TRAIL (s:Star)-[:E*1..2]->(t) RETURN t.n AS n";
    assert_eq!(limited(100).execute(query).unwrap().rows().len(), 5);
    let error = limited(4).execute(query).expect_err("5 edges read over 4");
    assert_over_the_limit("trail", &error, 4, "bound the quantifier");
}

/// A SHORTEST k GROUPS search over the limit is refused: it reads the centre's 5 edges.
#[test]
fn a_shortest_groups_search_over_the_limit_is_refused() {
    let query = "MATCH p = SHORTEST 2 GROUPS (s:Star)-[:E]->+(t) RETURN length(p) AS l";
    limited(100).execute(query).unwrap();
    let error = limited(4).execute(query).expect_err("5 edges read over 4");
    assert_over_the_limit("shortest 2 groups", &error, 4, "keep fewer paths");
}

/// The searches an EXISTS runs for its rows share the statement's count: four hubs of 5 edges
/// each are 5 a row, 20 in all. Each row fits a limit of 19 alone; together they do not. The
/// inner WHERE keeps the subquery a path search: a bare pattern is answered by an edge check
/// on each row's node, which reaches nodes and is not charged.
#[test]
fn exists_run_per_row_shares_the_statement_count() {
    let hubs = |expansions| {
        let db = empty(expansions);
        for n in 1..=4 {
            db.execute(&format!(
                "INSERT (h:Hub {{n: {n}}}), (h)-[:E]->(:Leaf {{k: 1}}), \
                 (h)-[:E]->(:Leaf {{k: 1}}), (h)-[:E]->(:Leaf {{k: 1}}), \
                 (h)-[:E]->(:Leaf {{k: 1}}), (h)-[:E]->(:Leaf {{k: 1}})"
            ))
            .unwrap();
        }
        db
    };
    let query = "MATCH (h:Hub) WHERE EXISTS { MATCH (h)-[:E*1..2]->(t) WHERE t.k = 1 } \
                 RETURN h.n AS n";
    assert_eq!(hubs(20).execute(query).unwrap().rows().len(), 4);
    let error = hubs(19).execute(query).expect_err("20 over 19");
    assert_over_the_limit("exists per row", &error, 19, "bound the quantifier");
}

/// A SQL/PGQ `GRAPH_TABLE` is held to the same limit: its search reads the centre's 5 edges.
#[cfg(feature = "sql-pgq")]
#[test]
fn sql_pgq_is_held_to_the_same_limit() {
    let query = "SELECT * FROM GRAPH_TABLE (MATCH (s:Star)-[:E*1..2]->(t) COLUMNS (t.n AS n))";
    assert_eq!(limited(5).execute_sql(query).unwrap().rows().len(), 5);
    let error = limited(4).execute_sql(query).expect_err("5 over 4");
    assert_over_the_limit("sql-pgq", &error, 4, "bound the quantifier");
}
