//! An aggregate over a chain of two or more expands without GROUP BY runs
//! factorized (FactorizedAggregateOperator over the lazy chain): COUNT, SUM,
//! AVG, MIN and MAX of a variable. Factorizing must not change the answer, so
//! each query is checked against the same query on a database with factorized
//! execution off, on every variable of the chain: the start node, a middle
//! node, the last node and each edge, with and without DISTINCT.
//!
//! A grouped count over one expand (`MATCH (a)-[:E]->(:V) RETURN a.id,
//! count(*)`) counts each node's edges from the adjacency (EdgeCount)
//! instead of making a row of each edge: checked the same way on random
//! graphs with self-loops, multi-edges, isolated nodes, several labels and
//! edge types, each direction, labels and filters on either end, and the
//! rows a transaction or an older snapshot sees. A count over one such
//! expand without GROUP BY (`MATCH (a)-[:E]->() RETURN count(*)`) totals the
//! same counts in one row, 0 when no edge matches; over a chain the last
//! hop's edges are counted for each row of the others, in place of the
//! factorized count, which keeps the counts of earlier variables.
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test main factorized_aggregates_match_the_regular_path::
//! ```

#![cfg(feature = "gql")]

use std::collections::BTreeMap;

use grafeo_common::types::Value;
use grafeo_engine::{Config, GrafeoDB, HistoryRetention, Session};

/// Alix reaches Gus and Vincent; Gus reaches Jules and Mia, Vincent reaches
/// Mia and Butch: two paths end at Mia, so DISTINCT counts differ from the
/// plain ones.
fn database(config: Config) -> GrafeoDB {
    let db = GrafeoDB::with_config(config).unwrap();
    db.execute(
        "INSERT (alix:P {name: 'Alix'}), (gus:P {name: 'Gus'}), (vincent:P {name: 'Vincent'}), \
         (jules:P {name: 'Jules'}), (mia:P {name: 'Mia'}), (butch:P {name: 'Butch'}), \
         (alix)-[:R {w: 1}]->(gus), (alix)-[:R {w: 2}]->(vincent), \
         (gus)-[:R {w: 3}]->(jules), (gus)-[:R {w: 4}]->(mia), \
         (vincent)-[:R {w: 5}]->(mia), (vincent)-[:R {w: 6}]->(butch)",
    )
    .unwrap();
    db
}

fn rows(db: &GrafeoDB, query: &str) -> String {
    match db.execute(query) {
        Ok(result) => format!("{:?}", result.rows()),
        Err(error) => format!("error: {error}"),
    }
}

const CHAIN: &str = "MATCH (a:P {name: 'Alix'})-[e1:R]->(b)-[e2:R]->(c)";

#[test]
fn each_aggregate_of_each_chain_variable_matches_the_regular_path() {
    let factorized = database(Config::in_memory());
    let regular = database(Config::in_memory().without_factorized_execution());
    let mut differ = Vec::new();
    for function in ["count", "min", "max", "sum", "avg"] {
        for variable in ["a", "b", "c", "e1", "e2"] {
            for distinct in ["", "DISTINCT "] {
                let query = format!("{CHAIN} RETURN {function}({distinct}{variable}) AS x");
                let (got, want) = (rows(&factorized, &query), rows(&regular, &query));
                if got != want {
                    differ.push(format!(
                        "{query}\n    factorized {got}\n    regular    {want}"
                    ));
                }
            }
        }
    }
    assert_eq!(
        differ.len(),
        0,
        "{} queries differ:\n{}",
        differ.len(),
        differ.join("\n")
    );
}

#[test]
fn several_aggregates_in_one_return_match_the_regular_path() {
    let factorized = database(Config::in_memory());
    let regular = database(Config::in_memory().without_factorized_execution());
    for query in [
        format!("{CHAIN} RETURN count(*) AS n, count(b) AS nb, count(DISTINCT c) AS dc"),
        format!("{CHAIN} RETURN min(b) AS x, max(c) AS y"),
        format!("{CHAIN} RETURN count(*) AS n, sum(c) AS s"),
    ] {
        assert_eq!(rows(&factorized, &query), rows(&regular, &query), "{query}");
    }
}

#[test]
fn counts_over_the_chain_by_hand() {
    // Four paths; three distinct ends (Mia twice), two distinct middles.
    let db = database(Config::in_memory());
    assert_eq!(
        rows(&db, &format!("{CHAIN} RETURN count(*) AS n")),
        "[[Int64(4)]]"
    );
    assert_eq!(
        rows(&db, &format!("{CHAIN} RETURN count(c) AS n")),
        "[[Int64(4)]]"
    );
    assert_eq!(
        rows(&db, &format!("{CHAIN} RETURN count(DISTINCT c) AS n")),
        "[[Int64(3)]]"
    );
    assert_eq!(
        rows(&db, &format!("{CHAIN} RETURN count(DISTINCT b) AS n")),
        "[[Int64(2)]]"
    );
    assert_eq!(
        rows(&db, &format!("{CHAIN} RETURN count(DISTINCT a) AS n")),
        "[[Int64(1)]]"
    );
}

/// A small deterministic generator (xorshift), so each seed gives one graph.
struct Draws(u64);

impl Draws {
    fn below(&mut self, bound: u64) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0 % bound
    }
}

/// Nodes 0 to 39 have edges, 40 to 44 none.
const NODES: u64 = 40;
const ISOLATED: u64 = 5;

/// The edges of the graph of `seed`, as (source id, target id, type): a
/// self-loop of each type and two parallel edges on node 3, then 150
/// random ones of types R and S, one in ten a self-loop.
fn random_edges(seed: u64) -> Vec<(u64, u64, &'static str)> {
    let mut draws = Draws(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1);
    let mut edges = vec![(3, 3, "R"), (3, 3, "S"), (3, 7, "R"), (3, 7, "R")];
    for _ in 0..150 {
        let source = draws.below(NODES);
        let target = if draws.below(10) == 0 {
            source
        } else {
            draws.below(NODES)
        };
        edges.push((source, target, if draws.below(3) == 0 { "S" } else { "R" }));
    }
    edges
}

/// The labels of node `id`: A, B, both or none, by `id % 4`.
fn labels_of(id: u64) -> &'static str {
    match id % 4 {
        0 => ":A",
        1 => ":B",
        2 => ":A:B",
        _ => "",
    }
}

/// The graph of `seed` on a new database: the nodes, each with `id`, `grp`
/// (`id % 3`) and the labels of [`labels_of`], and the edges of
/// [`random_edges`].
fn random_database(config: Config, seed: u64) -> GrafeoDB {
    let db = GrafeoDB::with_config(config).unwrap();
    for id in 0..NODES + ISOLATED {
        db.execute(&format!(
            "INSERT ({}{{id: {id}, grp: {}}})",
            labels_of(id),
            id % 3
        ))
        .unwrap();
    }
    for edge_type in ["R", "S"] {
        let pairs: Vec<String> = random_edges(seed)
            .iter()
            .filter(|(_, _, of_type)| *of_type == edge_type)
            .map(|(source, target, _)| format!("[{source}, {target}]"))
            .collect();
        db.execute(&format!(
            "UNWIND [{}] AS p MATCH (x {{id: p[0]}}), (y {{id: p[1]}}) \
             INSERT (x)-[:{edge_type}]->(y)",
            pairs.join(", ")
        ))
        .unwrap();
    }
    db
}

/// Whether the plan of `query` counts edges with EdgeCount.
fn counts_edges(db: &GrafeoDB, query: &str) -> bool {
    rows(db, &format!("PROFILE {query}")).contains("EdgeCount")
}

/// Grouped counts over one expand, over every shape of edge: no type, one
/// or two; out, in and either way; no label, one or two on the far end,
/// which may be named and counted.
fn degree_queries() -> Vec<String> {
    let mut queries = Vec::new();
    for start in ["(a)", "(a:A)", "(a:B)"] {
        for types in ["", ":R", ":S", ":R|S"] {
            for (left, right) in [("-", "->"), ("<-", "-"), ("-", "-")] {
                for far in ["()", "(:A)", "(:A:B)", "(b:B)", "(b)"] {
                    let pattern = format!("MATCH {start}{left}[e{types}]{right}{far}");
                    queries.push(format!(
                        "{pattern} RETURN a.id AS id, count(*) AS n ORDER BY id"
                    ));
                    if far.starts_with("(b") {
                        queries.push(format!(
                            "{pattern} RETURN a.grp AS g, count(b) AS nb, count(e) AS ne, \
                             count(a) AS na ORDER BY g"
                        ));
                    }
                }
            }
        }
    }
    queries
}

#[test]
fn a_degree_count_matches_the_regular_path_on_random_graphs() {
    for seed in 1..=4 {
        let counted = random_database(Config::in_memory(), seed);
        let regular = random_database(Config::in_memory().without_factorized_execution(), seed);
        let mut differ = Vec::new();
        for query in degree_queries() {
            assert!(counts_edges(&counted, &query), "not counted: {query}");
            assert!(!counts_edges(&regular, &query), "counted: {query}");
            let (got, want) = (rows(&counted, &query), rows(&regular, &query));
            if got != want {
                differ.push(format!("{query}\n    counted {got}\n    regular {want}"));
            }
        }
        assert!(
            differ.is_empty(),
            "seed {seed}: {} queries differ:\n{}",
            differ.len(),
            differ.join("\n")
        );
    }
}

/// The counts of `(a)-[:edge_type]-(far)` per start id from the edge list,
/// `far` the nodes with `label` (any node for `None`): an edge counts once
/// from each end whose other end qualifies, and a self-loop once.
fn undirected_counts(seed: u64, edge_type: &str, label: Option<char>) -> Vec<Vec<Value>> {
    let qualifies = |id: u64| label.is_none_or(|label| labels_of(id).contains(label));
    let mut counts = BTreeMap::new();
    for (source, target, of_type) in random_edges(seed) {
        if of_type != edge_type {
            continue;
        }
        if qualifies(target) {
            *counts.entry(source).or_insert(0) += 1;
        }
        if source != target && qualifies(source) {
            *counts.entry(target).or_insert(0) += 1;
        }
    }
    counts
        .into_iter()
        .map(|(id, count)| {
            vec![
                Value::Int64(i64::try_from(id).unwrap()),
                Value::Int64(count),
            ]
        })
        .collect()
}

#[test]
fn a_degree_count_counts_each_edge_once_from_each_end() {
    for seed in 1..=4 {
        let db = random_database(Config::in_memory(), seed);
        for (query, label) in [
            (
                "MATCH (a)-[:R]-(:A) RETURN a.id AS id, count(*) AS n ORDER BY id",
                Some('A'),
            ),
            (
                "MATCH (a)-[:S]-() RETURN a.id AS id, count(*) AS n ORDER BY id",
                None,
            ),
        ] {
            assert!(counts_edges(&db, query), "{query}");
            let edge_type = if query.contains(":R]") { "R" } else { "S" };
            assert_eq!(
                db.execute(query).unwrap().rows(),
                undirected_counts(seed, edge_type, label),
                "seed {seed}: {query}"
            );
        }
    }
    // Node 3 has a self-loop of each type and two R edges to node 7: each
    // loop counts once either way, each parallel edge once.
    let db = random_database(Config::in_memory(), 1);
    let query = "MATCH (a {id: 3})-[:R]-() RETURN a.id AS id, count(*) AS n";
    let expected: Vec<Vec<Value>> = undirected_counts(1, "R", None)
        .into_iter()
        .filter(|row| row[0] == Value::Int64(3))
        .collect();
    assert!(counts_edges(&db, query), "{query}");
    assert_eq!(db.execute(query).unwrap().rows(), expected);
    let Value::Int64(count) = expected[0][1] else {
        panic!("a count")
    };
    assert!(count >= 3, "the loop and both parallel edges: {count}");
}

/// The counts of [`degree_queries`] without groups, and a few more starts:
/// one with a filter, one anonymous, one bound by a property.
fn ungrouped_count_queries() -> Vec<String> {
    let mut queries = Vec::new();
    for start in ["(a)", "(a:A)", "(a:B)"] {
        for types in ["", ":R", ":S", ":R|S"] {
            for (left, right) in [("-", "->"), ("<-", "-"), ("-", "-")] {
                for far in ["()", "(:A)", "(:A:B)", "(b:B)", "(b)"] {
                    let pattern = format!("MATCH {start}{left}[e{types}]{right}{far}");
                    queries.push(format!("{pattern} RETURN count(*) AS n"));
                    if far.starts_with("(b") {
                        queries.push(format!(
                            "{pattern} RETURN count(b) AS nb, count(e) AS ne, count(a) AS na, \
                             count(*)"
                        ));
                    }
                }
            }
        }
    }
    queries.extend(
        [
            "MATCH (a)-[:R]->(:A) WHERE a.id > 10 AND a.grp <> 1 RETURN count(*) AS n",
            "MATCH ()-[:S]->() RETURN count(*) AS n",
            "MATCH (a {id: 3})-[:R]-() RETURN count(*) AS n",
            // Over a chain the last hop is counted, also where the
            // factorized count would count the chain.
            "MATCH (a)-[:R]->()-[:R]->(:B) RETURN count(*) AS n",
            "MATCH (a)-[:R]->()-[:S]->() RETURN count(*) AS n",
            "MATCH (a)-[:R]->(b)-[e:R]-(c) RETURN count(c) AS nc, count(e) AS ne, count(b) AS nb",
            "MATCH (a:A)-[:R]->()-[:R]->()<-[:S]-() RETURN count(*) AS n",
            // A condition on the count keeps the row or drops it.
            "MATCH (a)-[:R]->() RETURN count(*) AS n HAVING n > 10",
            "MATCH (a)-[:R]->() RETURN count(*) AS n HAVING n > 100000",
        ]
        .map(String::from),
    );
    queries
}

#[test]
fn an_ungrouped_degree_count_matches_the_regular_path_on_random_graphs() {
    for seed in 1..=4 {
        let counted = random_database(Config::in_memory(), seed);
        let regular = random_database(Config::in_memory().without_factorized_execution(), seed);
        let mut differ = Vec::new();
        for query in ungrouped_count_queries() {
            assert!(counts_edges(&counted, &query), "not counted: {query}");
            assert!(!counts_edges(&regular, &query), "counted: {query}");
            let (got, want) = (rows(&counted, &query), rows(&regular, &query));
            if got != want {
                differ.push(format!("{query}\n    counted {got}\n    regular {want}"));
            }
        }
        assert!(
            differ.is_empty(),
            "seed {seed}: {} queries differ:\n{}",
            differ.len(),
            differ.join("\n")
        );
    }
}

#[test]
fn an_ungrouped_degree_count_totals_the_degrees() {
    for seed in 1..=4 {
        let db = random_database(Config::in_memory(), seed);
        for (query, edge_type, label) in [
            ("MATCH (a)-[:R]-(:A) RETURN count(*) AS n", "R", Some('A')),
            ("MATCH (a)-[e:S]-() RETURN count(e) AS n", "S", None),
        ] {
            assert!(counts_edges(&db, query), "{query}");
            let total: i64 = undirected_counts(seed, edge_type, label)
                .iter()
                .map(|row| match row[1] {
                    Value::Int64(count) => count,
                    ref other => panic!("a count, not {other:?}"),
                })
                .sum();
            assert!(total > 0, "seed {seed}: {query} counts some edges");
            assert_eq!(
                db.execute(query).unwrap().rows(),
                [[Value::Int64(total)]],
                "seed {seed}: {query}"
            );
        }
        // Directed: each edge of the type once.
        let edges = random_edges(seed)
            .iter()
            .filter(|(_, _, of_type)| *of_type == "R")
            .count();
        let query = "MATCH (a)-[:R]->() RETURN count(*)";
        assert!(counts_edges(&db, query), "{query}");
        assert_eq!(
            db.execute(query).unwrap().rows(),
            [[Value::Int64(i64::try_from(edges).unwrap())]],
            "seed {seed}"
        );
    }
}

#[test]
fn profile_shows_an_ungrouped_degree_count_in_place_of_the_expand_and_the_aggregate() {
    let db = random_database(Config::in_memory(), 1);
    let result = db
        .execute("PROFILE MATCH (a:A)-[:R]->(:B) RETURN count(*) AS n")
        .unwrap();
    let text = result.rows()[0][0].as_str().expect("the profile text");
    let lines: Vec<&str> = text.lines().filter(|line| line.contains("rows=")).collect();
    let names: Vec<&str> = lines
        .iter()
        .filter_map(|line| line.split_whitespace().next())
        .collect();
    // The RETURN and the aggregate (with the label check the count does)
    // show the count they pass on, and the expand the count in its place,
    // over the scan of the start: one row, no row of each edge.
    assert_eq!(
        names,
        ["EdgeCount", "EdgeCount", "EdgeCount", "Scan"],
        "{text}"
    );
    for line in &lines[..3] {
        assert!(line.contains("rows=1 "), "{text}");
    }
    assert!(lines[1].contains("group: []"), "{text}");
    assert!(lines[1].contains("hasLabel"), "{text}");
    assert!(lines[2].contains("(a)->[:R]->"), "{text}");
}

#[test]
fn an_ungrouped_degree_count_of_no_edges_is_one_row_of_zero() {
    let empty = GrafeoDB::with_config(Config::in_memory()).unwrap();
    let empty_regular =
        GrafeoDB::with_config(Config::in_memory().without_factorized_execution()).unwrap();
    let counted = random_database(Config::in_memory(), 1);
    let regular = random_database(Config::in_memory().without_factorized_execution(), 1);
    for (db, reference, query) in [
        (
            &empty,
            &empty_regular,
            "MATCH (a)-[:R]->() RETURN count(*) AS n",
        ),
        (
            &empty,
            &empty_regular,
            "MATCH (a:A)-[e]-(b:B) RETURN count(e) AS n, count(*)",
        ),
        (
            &counted,
            &regular,
            "MATCH (a)-[:NONE]->() RETURN count(*) AS n",
        ),
        (
            &counted,
            &regular,
            "MATCH (a)-[:R]->(:Nobody) RETURN count(*) AS n",
        ),
        (
            &counted,
            &regular,
            "MATCH (a:Nobody)-[:R]->() RETURN count(*) AS n",
        ),
        // Nodes 40 to 44 have no edge.
        (
            &counted,
            &regular,
            "MATCH (a)-[]-() WHERE a.id >= 40 RETURN count(*) AS n",
        ),
    ] {
        assert!(counts_edges(db, query), "not counted: {query}");
        let zeros = if query.ends_with("count(*)") {
            "[[Int64(0), Int64(0)]]"
        } else {
            "[[Int64(0)]]"
        };
        assert_eq!(rows(db, query), zeros, "{query}");
        assert_eq!(rows(reference, query), zeros, "regular: {query}");
    }
}

#[test]
fn a_count_that_reads_the_far_end_or_the_edge_is_not_a_degree_count() {
    let counted = random_database(Config::in_memory(), 2);
    let regular = random_database(Config::in_memory().without_factorized_execution(), 2);
    for query in [
        // A property of the far end or of the edge, or one of them as a key.
        "MATCH (a)-[:R]->(b) WHERE b.id > 20 RETURN a.id AS id, count(*) AS n ORDER BY id",
        "MATCH (a)-[e:R]->(b) WHERE e.w IS NULL RETURN a.id AS id, count(*) AS n ORDER BY id",
        "MATCH (a)-[:R]->(b) RETURN b.grp AS g, count(*) AS n ORDER BY g",
        "MATCH (a)-[e]->(b) RETURN type(e) AS t, count(*) AS n ORDER BY t",
        // DISTINCT, a count of a value, another aggregate.
        "MATCH (a)-[:R]->(b) RETURN a.id AS id, count(DISTINCT b) AS n ORDER BY id",
        "MATCH (a)-[:R]->(b) RETURN a.id AS id, count(b.id) AS n ORDER BY id",
        "MATCH (a)-[:R]->(b) RETURN a.id AS id, count(*) AS n, sum(b.id) AS s ORDER BY id",
        // A far end that is the start.
        "MATCH (a)-[:R]->(a) RETURN a.id AS id, count(*) AS n ORDER BY id",
        // The same without groups.
        "MATCH (a)-[:R]->(b) WHERE b.id > 20 RETURN count(*) AS n",
        "MATCH (a)-[e:R]->(b) WHERE e.w IS NULL RETURN count(*) AS n",
        "MATCH (a)-[:R]->(b) RETURN count(DISTINCT b) AS n",
        "MATCH (a)-[:R]->(b) RETURN count(b.id) AS n",
        "MATCH (a)-[:R]->(b) RETURN count(*) AS n, sum(b.id) AS s",
        "MATCH (a)-[:R]->(a) RETURN count(*) AS n",
        "MATCH (a {id: 3})-[:R]->({id: 7}) RETURN count(*) AS n",
        // A trail of two edges, whose last edge must differ from the first.
        "MATCH TRAIL (a)-[:R]->()-[:R]->() RETURN count(*) AS n",
    ] {
        assert!(!counts_edges(&counted, query), "counted: {query}");
        assert_eq!(rows(&counted, query), rows(&regular, query), "{query}");
    }
}

#[test]
fn a_degree_count_keeps_filters_on_the_start_and_conditions_on_the_count() {
    let counted = random_database(Config::in_memory(), 3);
    let regular = random_database(Config::in_memory().without_factorized_execution(), 3);
    for query in [
        "MATCH (a)-[:R]->(:A) WHERE a.id > 10 AND a.grp <> 1 \
         RETURN a.id AS id, count(*) AS n ORDER BY id",
        "MATCH (a:B)-[:R|S]-() WITH a.id AS id, count(*) AS n WHERE n > 6 \
         RETURN id, n ORDER BY id",
        "MATCH (a)<-[:S]-(:B) RETURN a.id AS id, count(*) AS degree \
         ORDER BY degree DESC, id ASC LIMIT 5",
        "MATCH (a)-[]->() RETURN a.grp AS g, a.id % 2 AS odd, count(*) AS n ORDER BY g, odd",
        // Two hops: the second one's edges are counted for each row of the
        // first.
        "MATCH (a)-[:R]->()-[:R]->() RETURN a.id AS id, count(*) AS n ORDER BY id",
        "MATCH (a:A)-[:R]->(m)<-[:S]-(:B) RETURN a.id AS id, m.grp AS g, count(*) AS n          ORDER BY id, g",
    ] {
        assert!(counts_edges(&counted, query), "not counted: {query}");
        assert_eq!(rows(&counted, query), rows(&regular, query), "{query}");
    }
}

const DEGREES: &str = "MATCH (a)-[:R]-(:A) RETURN a.id AS id, count(*) AS n ORDER BY id";
const NODE_ZERO: &str = "MATCH (a {id: 0})-[:R]-(:A) RETURN a.id AS id, count(*) AS n";
const TOTAL: &str = "MATCH (a)-[:R]-(:A) RETURN count(*) AS n";
const FROM_THREE: &str = "MATCH (a {id: 3})-[:R]->() RETURN count(*) AS n";

/// The one count of `query`'s one row.
fn count_in(session: &Session, query: &str) -> i64 {
    match session.execute(query).unwrap().rows() {
        [row] => match row.as_slice() {
            [Value::Int64(count)] => *count,
            other => panic!("{query}: one count, not {other:?}"),
        },
        other => panic!("{query}: one row, not {other:?}"),
    }
}

fn rows_in(session: &Session, query: &str) -> String {
    format!("{:?}", session.execute(query).unwrap().rows())
}

#[test]
fn a_degree_count_reads_what_its_transaction_and_its_snapshot_see() {
    let config = || Config::in_memory().with_history_retention(HistoryRetention::Forever);
    let counted = random_database(config(), 4);
    let regular = random_database(config().without_factorized_execution(), 4);
    let before = rows(&counted, DEGREES);
    assert_eq!(before, rows(&regular, DEGREES));
    let (total_before, from_three_before) = (
        count_in(&counted.session(), TOTAL),
        count_in(&counted.session(), FROM_THREE),
    );
    assert_eq!(total_before, count_in(&regular.session(), TOTAL));
    let Value::Int64(node_zero_before) = counted
        .execute(NODE_ZERO)
        .unwrap()
        .rows()
        .first()
        .map_or(Value::Int64(0), |row| row[1].clone())
    else {
        panic!("a count")
    };

    // Each transaction inserts an R edge from node 0 (`:A`) to node 2
    // (`:A:B`) and deletes the two R edges from node 3 to node 7.
    let mut sessions = Vec::new();
    for db in [&counted, &regular] {
        let mut session = db.session();
        session.begin_transaction().unwrap();
        session
            .execute("MATCH (x {id: 0}), (y {id: 2}) INSERT (x)-[:R]->(y)")
            .unwrap();
        session
            .execute("MATCH ({id: 3})-[e:R]->({id: 7}) DELETE e")
            .unwrap();
        sessions.push(session);
    }
    assert!(counts_edges(&counted, DEGREES));
    let in_transaction = rows_in(&sessions[0], DEGREES);
    assert_eq!(in_transaction, rows_in(&sessions[1], DEGREES));
    assert_ne!(in_transaction, before);
    // Its own new edge counts for node 0.
    assert_eq!(
        rows_in(&sessions[0], NODE_ZERO),
        format!("[[Int64(0), Int64({})]]", node_zero_before + 1)
    );
    assert_eq!(
        rows_in(
            &sessions[0],
            "MATCH (a {id: 3})-[:R]->({id: 7}) RETURN count(*) AS n"
        ),
        "[[Int64(0)]]"
    );
    // Without groups: the new edge counts from both of its ends (both
    // `:A`), the deleted ones (between nodes without a label) only from
    // node 3.
    assert!(counts_edges(&counted, TOTAL) && counts_edges(&counted, FROM_THREE));
    for session in &sessions {
        assert_eq!(count_in(session, TOTAL), total_before + 2);
        assert_eq!(count_in(session, FROM_THREE), from_three_before - 2);
    }
    // Another session sees neither change before the commit.
    assert_eq!(rows(&counted, DEGREES), before);
    assert_eq!(count_in(&counted.session(), TOTAL), total_before);
    assert_eq!(count_in(&counted.session(), FROM_THREE), from_three_before);

    let epoch = counted.current_epoch();
    for session in &mut sessions {
        session.commit().unwrap();
    }
    assert_eq!(rows(&counted, DEGREES), in_transaction);
    assert_eq!(rows(&regular, DEGREES), in_transaction);
    // The snapshot before the commit still sees the edges as they were.
    assert_eq!(
        format!(
            "{:?}",
            counted.execute_at_epoch(DEGREES, epoch).unwrap().rows()
        ),
        before
    );
    assert_eq!(count_in(&counted.session(), TOTAL), total_before + 2);
    assert_eq!(
        counted.execute_at_epoch(TOTAL, epoch).unwrap().rows(),
        [[Value::Int64(total_before)]]
    );
    assert_eq!(
        counted.execute_at_epoch(FROM_THREE, epoch).unwrap().rows(),
        [[Value::Int64(from_three_before)]]
    );
}

/// The lines of the profile of `query` that show an operator: its name and
/// the whole line.
fn profile_lines(db: &GrafeoDB, query: &str) -> Vec<(String, String)> {
    let result = db
        .execute(&format!("PROFILE {query}"))
        .unwrap_or_else(|error| panic!("PROFILE {query}: {error}"));
    let text = result.rows()[0][0].as_str().expect("the profile text");
    text.lines()
        .filter(|line| line.contains("rows="))
        .map(|line| {
            let name = line.split_whitespace().next().unwrap_or("").to_string();
            (name, line.to_string())
        })
        .collect()
}

#[test]
fn profile_of_a_factorized_count_names_the_expands_it_absorbs() {
    let db = database(Config::in_memory());
    // Mia reaches Butch: two paths of three hops from Alix.
    db.execute("MATCH (m:P {name: 'Mia'}), (b:P {name: 'Butch'}) INSERT (m)-[:R {w: 7}]->(b)")
        .unwrap();
    // Counts of a variable before the last hop run factorized over the
    // chain (the edge count answers only counts of the last hop's rows):
    // over two hops and over three, of the start, a middle node and the
    // first edge, alone and next to a count of the rows.
    // From every node the degrees answer it, so no aggregate shows the hops either, and no
    // expand has a line of its own.
    for (query, count) in [
        ("MATCH (a)-[:R]->(b)-[:R]->(c) RETURN count(a)", 6),
        ("MATCH (a)-[:R]->(b)-[:R]->(c)-[:R]->(d) RETURN count(a)", 2),
    ] {
        let lines = profile_lines(&db, query);
        assert!(
            lines.iter().all(|(name, _)| !name.contains("Expand")),
            "{query}: {lines:?}"
        );
        assert_eq!(db.execute(query).unwrap().rows()[0][0], Value::Int64(count));
    }
    // The one that answers it, a factorized aggregate or a count over the hops, names them.
    for (query, count) in [
        (
            "MATCH (a:P {name: 'Alix'})-[e1:R]->(b)-[e2:R]->(c)-[e3:R]->(d) RETURN count(b) AS n",
            2,
        ),
        (
            "MATCH (a:P {name: 'Alix'})-[e1:R]->(b)-[e2:R]->(c) RETURN count(e1) AS n",
            4,
        ),
        (
            "MATCH (a:P {name: 'Alix'})-[e1:R]->(b)-[e2:R]->(c) \
             RETURN count(a) AS n, count(*) AS m",
            4,
        ),
    ] {
        let lines = profile_lines(&db, query);
        let names: Vec<&str> = lines.iter().map(|(name, _)| name.as_str()).collect();
        let text = lines
            .iter()
            .map(|(_, line)| line.as_str())
            .collect::<Vec<_>>()
            .join("\n");
        // The factorized aggregate shows in the aggregate's place, with the
        // expands it runs as part of it named in its detail (one per hop),
        // over the scan of the start: no expand has a line of its own. (The
        // RETURN above it can show the aggregate's name too: the last line
        // with it is the aggregate's.)
        let aggregate = names
            .iter()
            .rposition(|name| *name == "FactorizedAggregate" || *name == "EdgeCount")
            .unwrap_or_else(|| panic!("{query}: no factorized aggregate or count\n{text}"));
        let hops = query.matches("-[").count();
        assert_eq!(
            lines[aggregate].1.matches("[:R]").count(),
            hops,
            "{query}\n{text}"
        );
        assert!(
            names.iter().all(|name| !name.contains("Expand")),
            "{query}\n{text}"
        );
        assert!(
            names[aggregate + 1..]
                .iter()
                .all(|name| name.contains("Scan")
                    || name.contains("Filter")
                    || name.contains("NodeList")),
            "{query}\n{text}"
        );
        assert!(lines[aggregate].1.contains("rows=1 "), "{query}\n{text}");
        // And it counts what the query counts without PROFILE.
        assert_eq!(
            db.execute(query).unwrap().rows()[0][0],
            Value::Int64(count),
            "{query}"
        );
    }
}
