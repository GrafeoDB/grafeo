//! SQL/PGQ paths since its MATCH is read by the GQL parser and translator
//! (0.6.1): a path variable with ANY SHORTEST and PATH_LENGTH, the path modes,
//! the path functions of a quantified edge's variable, graph-bench's IC13 text
//! verbatim, GQL's quantifier rules, and where a syntax error inside a long
//! GRAPH_TABLE is reported. Each SQL/PGQ form is checked against the same
//! pattern in GQL, and the counts and lengths against the graph by hand (IC13
//! against a breadth-first search in the test).
//!
//! The graph: Alix (1) knows Gus (2) and Mia (5), Gus knows Vincent (3),
//! Vincent knows Jules (4) and Alix, Mia knows Jules, Jules knows Butch (6);
//! Django (7) knows no one.
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test sql_pgq_paths_through_gql
//! ```

#![cfg(all(feature = "gql", feature = "sql-pgq"))]

use std::collections::{HashMap, VecDeque};

use grafeo_common::types::Value;
use grafeo_engine::GrafeoDB;

const PEOPLE: [(i64, &str); 7] = [
    (1, "Alix"),
    (2, "Gus"),
    (3, "Vincent"),
    (4, "Jules"),
    (5, "Mia"),
    (6, "Butch"),
    (7, "Django"),
];
const KNOWS: [(i64, i64); 7] = [(1, 2), (2, 3), (3, 4), (1, 5), (5, 4), (4, 6), (3, 1)];

/// Runs `statement` in the database or in one of its graphs.
fn fill(execute: impl Fn(&str)) {
    let people: Vec<String> = PEOPLE
        .iter()
        .map(|(id, name)| format!("(:Person {{id: {id}, name: '{name}'}})"))
        .collect();
    execute(&format!("INSERT {}", people.join(", ")));
    for (a, b) in KNOWS {
        execute(&format!(
            "MATCH (a:Person {{id: {a}}}), (b:Person {{id: {b}}}) INSERT (a)-[:KNOWS]->(b)"
        ));
    }
}

fn database() -> GrafeoDB {
    let db = GrafeoDB::new_in_memory();
    fill(|statement| {
        db.execute(statement).unwrap();
    });
    db
}

fn rows(db: &GrafeoDB, language: &str, query: &str) -> Vec<String> {
    let mut rows: Vec<String> = db
        .execute_language(query, language, None)
        .unwrap_or_else(|error| panic!("{language} `{query}`: {error}"))
        .rows()
        .iter()
        .map(|row| format!("{row:?}"))
        .collect();
    rows.sort();
    rows
}

/// SQL/PGQ's rows equal GQL's, and there are `count` of them.
fn agree(db: &GrafeoDB, sql: &str, gql: &str, count: usize) {
    let from_sql = rows(db, "sql-pgq", sql);
    assert_eq!(
        from_sql,
        rows(db, "gql", gql),
        "SQL/PGQ `{sql}` against GQL `{gql}`"
    );
    assert_eq!(from_sql.len(), count, "`{sql}`");
}

#[test]
fn any_shortest_with_a_path_variable_matches_gql() {
    let db = database();
    agree(
        &db,
        "SELECT * FROM GRAPH_TABLE (MATCH p = ANY SHORTEST (x:Person)-[:KNOWS]->+(y:Person) \
         WHERE x.name = 'Alix' COLUMNS (y.name AS b, PATH_LENGTH(p) AS len))",
        "MATCH p = ANY SHORTEST (x:Person)-[:KNOWS]->+(y:Person) WHERE x.name = 'Alix' \
         RETURN y.name, path_length(p)",
        6,
    );
    // By hand: Gus and Mia one hop, Vincent and Jules two, Butch three, and
    // Alix herself three (round the cycle through Vincent): `+` takes one hop
    // at least.
    let lengths: HashMap<String, i64> = db
        .execute_language(
            "SELECT * FROM GRAPH_TABLE (MATCH p = ANY SHORTEST (x:Person)-[:KNOWS]->+(y:Person) \
             WHERE x.name = 'Alix' COLUMNS (y.name AS b, PATH_LENGTH(p) AS len))",
            "sql-pgq",
            None,
        )
        .unwrap()
        .rows()
        .iter()
        .map(|row| match (&row[0], &row[1]) {
            (Value::String(name), Value::Int64(len)) => (name.to_string(), *len),
            other => panic!("{other:?}"),
        })
        .collect();
    let want: HashMap<String, i64> = [
        ("Gus", 1),
        ("Mia", 1),
        ("Vincent", 2),
        ("Jules", 2),
        ("Butch", 3),
        ("Alix", 3),
    ]
    .into_iter()
    .map(|(name, len)| (name.to_string(), len))
    .collect();
    assert_eq!(lengths, want);
}

#[test]
fn the_path_modes_match_gql() {
    // From Alix over one to four hops: ten walks; TRAIL drops the one that
    // takes Alix -> Gus twice; ACYCLIC drops the three back through Alix;
    // SIMPLE keeps the one that ends where it starts.
    let db = database();
    for (mode, count) in [("WALK", 10), ("TRAIL", 9), ("ACYCLIC", 7), ("SIMPLE", 8)] {
        agree(
            &db,
            &format!(
                "SELECT * FROM GRAPH_TABLE (MATCH p = {mode} (x:Person)-[:KNOWS]->{{1,4}}(y:Person) \
                 WHERE x.name = 'Alix' COLUMNS (y.name AS b, PATH_LENGTH(p) AS len, NODES(p) AS n))"
            ),
            &format!(
                "MATCH p = {mode} (x:Person)-[:KNOWS]->{{1,4}}(y:Person) WHERE x.name = 'Alix' \
                 RETURN y.name, path_length(p), nodes(p)"
            ),
            count,
        );
    }
}

#[test]
fn a_path_variable_in_columns() {
    let db = database();
    agree(
        &db,
        "SELECT * FROM GRAPH_TABLE (MATCH p = (x:Person)-[:KNOWS]->{1,3}(y:Person) \
         WHERE x.name = 'Alix' COLUMNS (y.name AS b, LENGTH(p) AS l, NODES(p) AS n, EDGES(p) AS e))",
        "MATCH p = (x:Person)-[:KNOWS]->{1,3}(y:Person) WHERE x.name = 'Alix' \
         RETURN y.name, path_length(p), nodes(p), edges(p)",
        7,
    );
}

#[test]
fn the_path_functions_of_a_quantified_edge_variable_read_its_path() {
    // SQL/PGQ's own form: the variable of a quantified edge stands for its
    // path in LENGTH, PATH_LENGTH, SIZE, NODES and EDGES.
    let db = database();
    agree(
        &db,
        "SELECT * FROM GRAPH_TABLE (MATCH (x:Person)-[k:KNOWS]->{1,3}(y:Person) \
         WHERE x.name = 'Alix' COLUMNS (y.name AS b, LENGTH(k) AS l, PATH_LENGTH(k) AS pl, \
         SIZE(k) AS s, NODES(k) AS n, EDGES(k) AS e))",
        "MATCH p = (x:Person)-[:KNOWS]->{1,3}(y:Person) WHERE x.name = 'Alix' \
         RETURN y.name, path_length(p) AS l, path_length(p) AS pl, path_length(p) AS s, nodes(p) AS n, edges(p) AS e",
        7,
    );
}

/// The hops between `a` and `b` over KNOWS in either direction, by a
/// breadth-first search: 0 for the same person, -1 without a path.
fn distance(a: i64, b: i64) -> i64 {
    let mut hops = HashMap::from([(a, 0)]);
    let mut queue = VecDeque::from([a]);
    while let Some(at) = queue.pop_front() {
        for &(x, y) in &KNOWS {
            let next = if x == at {
                y
            } else if y == at {
                x
            } else {
                continue;
            };
            if !hops.contains_key(&next) {
                hops.insert(next, hops[&at] + 1);
                queue.push_back(next);
            }
        }
    }
    hops.get(&b).copied().unwrap_or(-1)
}

/// graph-bench's IC13 (graph_bench/suites/snb_sql.py), verbatim, on a graph
/// named `snb`, for every pair of people: the same person, a path, and the
/// person who knows no one.
#[test]
fn graph_bench_ic13_verbatim() {
    const IC13: &str = "\
SELECT CASE WHEN $person1Id = $person2Id THEN 0 ELSE COALESCE(MIN(len), -1) END AS shortestPathLength
FROM GRAPH_TABLE (snb
  MATCH path = ANY SHORTEST (p1:Person)-[k:KNOWS]-+(p2:Person)
  WHERE p1.id = $person1Id AND p2.id = $person2Id
  COLUMNS (PATH_LENGTH(path) AS len)
)";
    let db = GrafeoDB::new_in_memory();
    db.execute("CREATE GRAPH snb").unwrap();
    let snb = db.graph("snb").unwrap();
    fill(|statement| {
        snb.execute(statement).unwrap();
    });
    for (a, _) in PEOPLE {
        for (b, _) in PEOPLE {
            let params = HashMap::from([
                ("person1Id".to_string(), Value::Int64(a)),
                ("person2Id".to_string(), Value::Int64(b)),
            ]);
            let got = db
                .execute_language(IC13, "sql-pgq", Some(params))
                .unwrap_or_else(|error| panic!("IC13 {a} {b}: {error}"))
                .rows()
                .to_vec();
            assert_eq!(got, [vec![Value::Int64(distance(a, b))]], "IC13 {a} {b}");
        }
    }
    assert_eq!(distance(1, 7), -1, "Django knows no one");
}

#[test]
fn gql_quantifier_rules() {
    let db = database();
    // {,2} is {0,2}: Alix herself, then one and two hops.
    let max_only = rows(
        &db,
        "gql",
        "MATCH (a:Person)-[:KNOWS]->{,2}(b:Person) WHERE a.name = 'Alix' RETURN b.name",
    );
    assert_eq!(
        max_only,
        rows(
            &db,
            "gql",
            "MATCH (a:Person)-[:KNOWS]->{0,2}(b:Person) WHERE a.name = 'Alix' RETURN b.name",
        )
    );
    assert_eq!(max_only.len(), 5);
    for (pattern, says) in [
        (
            "(a:Person)-[:KNOWS]->{3,1}(b:Person)",
            "upper bound is below its lower bound",
        ),
        ("(a:Person)-[:KNOWS*1..2]->{2}(b:Person)", "one quantifier"),
        ("(a:Person)-[:KNOWS*1..2]->+(b:Person)", "one quantifier"),
    ] {
        let error = db
            .execute(&format!("MATCH {pattern} RETURN b.name"))
            .expect_err(pattern)
            .to_string();
        assert!(error.contains(says), "{pattern}: {error}");
    }
}

/// Where a syntax error inside a long GRAPH_TABLE is reported: its line and
/// column in the whole query, under the SQL/PGQ name.
#[test]
fn a_syntax_error_inside_graph_table_names_its_line_and_column() {
    let db = database();
    for (line, mark) in [
        ("  WHERE a.name = 'Alix' AND AND b.name = 'Gus'", "AND b."),
        ("    -[j:KNOWS]->(c:Person WHERE)", ")"),
        ("      -[k:KNOWS]->{3,1}(c:Person)", "1}"),
    ] {
        let query = format!(
            "SELECT * FROM GRAPH_TABLE (\n  MATCH (a:Person)-[k:KNOWS]->(b:Person)\n{line}\n  \
             COLUMNS (a.name AS a)\n)"
        );
        let error = db
            .execute_language(&query, "sql-pgq", None)
            .expect_err(line)
            .to_string();
        let column = line.rfind(mark).unwrap() + 1;
        assert!(error.contains("[SQL/PGQ]"), "{line}: {error}");
        assert!(
            error.contains(&format!("query:3:{column}")),
            "{line}: column {column} expected: {error}"
        );
    }
}
