//! A RETURN column whose parts are of different kinds (`CASE WHEN c THEN n
//! ELSE 1 END`, `coalesce(b, 1)`, a computed index into `[n, r, 1]`, a list
//! `+` of nodes and plain values) gives each row the type of the part it
//! took: a node where the row took the node, the plain value where it took
//! that. It gave every row plain values, so a node came back as its id.
//! A mixed column that a WITH passes on still gives ids, until 0.7.0's entity
//! values.
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test main mixed_kind_columns::
//! ```

#![cfg(all(feature = "gql", feature = "cypher"))]

use grafeo_common::types::{LogicalType, PropertyKey, Value};
use grafeo_common::utils::gqlstatus::GqlStatus;
use grafeo_engine::database::QueryResult;
use grafeo_engine::{Config, GrafeoDB};

/// `(:A {n: 1})-[:K {w: 1}]->(:B {n: 2})`, and the same with n 3, 4 and w 2,
/// and n 5, 6 and w 3.
fn three_edges() -> GrafeoDB {
    let db = GrafeoDB::with_config(Config::in_memory()).unwrap();
    for (a, w, b) in [(1, 1, 2), (3, 2, 4), (5, 3, 6)] {
        db.execute(&format!(
            "INSERT (:A {{n: {a}}})-[:K {{w: {w}}}]->(:B {{n: {b}}})"
        ))
        .unwrap();
    }
    db
}

fn run(db: &GrafeoDB, language: &str, query: &str) -> QueryResult {
    match language {
        "gql" => db.execute(query),
        "cypher" => db.execute_cypher(query),
        other => panic!("unknown language {other}"),
    }
    .unwrap_or_else(|error| panic!("{language}: {query}: {error}"))
}

fn key(map: &std::collections::BTreeMap<PropertyKey, Value>, name: &str) -> Value {
    map.get(&PropertyKey::new(name))
        .cloned()
        .unwrap_or(Value::Null)
}

/// What a cell is by its type: `n=3` for a node, `w=2` for an edge, a list
/// of what its items are, and a plain value as it is.
fn read(value: &Value, ty: &LogicalType) -> String {
    match (ty, value) {
        (LogicalType::Node, Value::Map(map)) => format!("n={:?}", key(map, "n")),
        (LogicalType::Edge, Value::Map(map)) => format!("w={:?}", key(map, "w")),
        (LogicalType::List(item), Value::List(items)) => {
            let items: Vec<String> = items.iter().map(|value| read(value, item)).collect();
            format!("[{}]", items.join(", "))
        }
        (LogicalType::Tuple(types), Value::List(items)) => {
            let items: Vec<String> = items
                .iter()
                .zip(types)
                .map(|(value, ty)| read(value, ty))
                .collect();
            format!("[{}]", items.join(", "))
        }
        (LogicalType::Path, Value::Path { nodes, edges }) => {
            let nodes: Vec<String> = nodes
                .iter()
                .map(|node| read(node, &LogicalType::Node))
                .collect();
            let edges: Vec<String> = edges
                .iter()
                .map(|edge| read(edge, &LogicalType::Edge))
                .collect();
            format!("path[{}|{}]", nodes.join(","), edges.join(","))
        }
        (LogicalType::Any, Value::List(items)) => {
            let items: Vec<String> = items
                .iter()
                .map(|value| read(value, &LogicalType::Any))
                .collect();
            format!("[{}]", items.join(", "))
        }
        (_, value) => format!("{value:?}"),
    }
}

/// Column `v` of each row by its type, keyed by column `k`, sorted by `k`.
fn by_key(result: &QueryResult) -> Vec<(Value, String)> {
    let mut rows: Vec<(Value, String)> = result
        .rows()
        .iter()
        .enumerate()
        .map(|(row, values)| {
            (
                values[0].clone(),
                read(&values[1], result.cell_type(row, 1)),
            )
        })
        .collect();
    rows.sort_by(|left, right| format!("{:?}", left.0).cmp(&format!("{:?}", right.0)));
    rows
}

/// Each form, in both languages: the rows that take a node or an edge read
/// as that node or edge, the rows that take a plain value read as it.
#[test]
fn each_row_has_the_type_of_the_part_it_took() {
    let db = three_edges();
    let cases: Vec<(&str, Vec<String>)> = vec![
        (
            "MATCH (a:A) RETURN a.n AS k, CASE WHEN a.n > 2 THEN a ELSE a.n END AS v",
            vec![int(1), node(3), node(5)],
        ),
        (
            "MATCH (a:A) OPTIONAL MATCH (a)-[:K]->(b:B) WHERE b.n > 3 \
             RETURN a.n AS k, coalesce(b, a.n) AS v",
            vec![int(1), node(4), node(6)],
        ),
        (
            "MATCH (a:A)-[r:K]->() RETURN a.n AS k, [a, r, a.n][a.n % 3] AS v",
            vec![edge(1), node(3), int(5)],
        ),
        (
            "MATCH (a:A) RETURN a.n AS k, [x IN [a] WHERE x.n > 2] + [a.n] AS v",
            vec![
                format!("[{}]", int(1)),
                format!("[{}, {}]", node(3), int(3)),
                format!("[{}, {}]", node(5), int(5)),
            ],
        ),
        (
            "MATCH (a:A) RETURN a.n AS k, [a] + [a.n] AS v",
            vec![
                format!("[{}, {}]", node(1), int(1)),
                format!("[{}, {}]", node(3), int(3)),
                format!("[{}, {}]", node(5), int(5)),
            ],
        ),
    ];
    for (query, expected) in cases {
        for language in ["gql", "cypher"] {
            let result = run(&db, language, query);
            let got: Vec<String> = by_key(&result).into_iter().map(|(_, v)| v).collect();
            assert_eq!(got, expected, "{language}: {query}");
        }
    }
}

/// A CASE that mixes a node with a plain value, on its own.
#[test]
fn a_case_of_a_node_or_a_number_gives_the_node() {
    let db = three_edges();
    for language in ["gql", "cypher"] {
        let result = run(
            &db,
            language,
            "MATCH (a:A {n: 3}) RETURN CASE WHEN a.n > 2 THEN a ELSE 1 END AS v",
        );
        assert_eq!(*result.cell_type(0, 0), LogicalType::Node, "{language}");
        let Value::Map(map) = &result.rows()[0][0] else {
            panic!("{language}: {:?}", result.rows()[0][0]);
        };
        assert_eq!(key(map, "n"), Value::Int64(3), "{language}");
    }
}

/// Many rows that alternate between a node and a number: every value is
/// right, and the result starts a run at each change of kind, no more.
#[test]
fn an_alternating_column_has_a_run_per_change_of_kind() {
    let db = three_edges();
    for language in ["gql", "cypher"] {
        let result = run(
            &db,
            language,
            "MATCH (a:A {n: 1}) UNWIND range(1, 200) AS i \
             RETURN i AS k, CASE WHEN i % 2 = 0 THEN a ELSE i END AS v",
        );
        assert_eq!(result.rows().len(), 200, "{language}");
        for (row, values) in result.rows().iter().enumerate() {
            let Value::Int64(i) = values[0] else {
                panic!("{language}: {values:?}");
            };
            let expected = if i % 2 == 0 {
                format!("n={:?}", Value::Int64(1))
            } else {
                format!("{:?}", Value::Int64(i))
            };
            assert_eq!(
                read(&values[1], result.cell_type(row, 1)),
                expected,
                "{language}: {i}"
            );
        }
        assert_eq!(result.cell_types().runs().len(), 199, "{language}");
    }
}

/// A mixed form whose rows all take the same part has one type and no run.
#[test]
fn a_mixed_form_whose_rows_take_one_part_has_no_run() {
    let db = three_edges();
    for language in ["gql", "cypher"] {
        let result = run(
            &db,
            language,
            "MATCH (a:A) RETURN CASE WHEN a.n > 0 THEN a ELSE 1 END AS v",
        );
        assert_eq!(result.rows().len(), 3, "{language}");
        assert!(result.cell_types().runs().is_empty(), "{language}");
        assert_eq!(
            result.cell_types().columns(),
            [LogicalType::Node],
            "{language}"
        );
    }
}

/// Column `v` of each row by its type, sorted by column `k`.
fn column_v(result: &QueryResult) -> Vec<String> {
    by_key(result).into_iter().map(|(_, v)| v).collect()
}

fn int(i: i64) -> String {
    format!("{:?}", Value::Int64(i))
}

fn node(n: i64) -> String {
    format!("n={:?}", Value::Int64(n))
}

fn edge(w: i64) -> String {
    format!("w={:?}", Value::Int64(w))
}

/// The id of the `A` node with `n`.
fn id_of(db: &GrafeoDB, n: i64) -> i64 {
    match run(
        db,
        "cypher",
        &format!("MATCH (a:A {{n: {n}}}) RETURN id(a) AS i"),
    )
    .rows()[0][0]
    {
        Value::Int64(id) => id,
        ref other => panic!("{other:?}"),
    }
}

/// 200 rows that alternate between the node with n 1 and the row number.
const ALTERNATING: &str = "MATCH (a:A {n: 1}) UNWIND range(1, 200) AS i \
                           RETURN i AS k, CASE WHEN i % 2 = 0 THEN a ELSE i END AS v";

/// Each row of an alternating result is right: the node on even rows, the
/// number on odd ones.
fn assert_alternating(result: &QueryResult, rows: &[i64], label: &str) {
    let got: Vec<(Value, String)> = by_key(result);
    let mut expected: Vec<(Value, String)> = rows
        .iter()
        .map(|&i| {
            let v = if i % 2 == 0 { node(1) } else { int(i) };
            (Value::Int64(i), v)
        })
        .collect();
    expected.sort_by(|left, right| format!("{:?}", left.0).cmp(&format!("{:?}", right.0)));
    assert_eq!(got, expected, "{label}");
}

/// A LIMIT that ends inside the chunks a split made, a SKIP, and a WHERE
/// before the RETURN (a selection vector) keep every row right, and a RETURN
/// DISTINCT of the alternating column keeps its 200 rows and their types.
#[test]
fn limit_skip_where_and_distinct_cut_a_split_result_right() {
    let db = three_edges();
    for language in ["gql", "cypher"] {
        let result = run(&db, language, &format!("{ALTERNATING} LIMIT 3"));
        assert_alternating(&result, &[1, 2, 3], &format!("{language}: LIMIT 3"));
        assert_eq!(result.cell_types().runs().len(), 2, "{language}: LIMIT 3");

        let result = run(&db, language, &format!("{ALTERNATING} SKIP 2"));
        assert_alternating(
            &result,
            &(3..=200).collect::<Vec<_>>(),
            &format!("{language}: SKIP 2"),
        );

        let result = run(
            &db,
            language,
            "MATCH (a:A {n: 1}) UNWIND range(1, 200) AS i WITH a, i WHERE i % 3 <> 0 \
             RETURN i AS k, CASE WHEN i % 2 = 0 THEN a ELSE i END AS v",
        );
        let kept: Vec<i64> = (1..=200).filter(|i| i % 3 != 0).collect();
        assert_alternating(&result, &kept, &format!("{language}: WHERE"));

        let result = run(
            &db,
            language,
            &ALTERNATING.replace("RETURN", "RETURN DISTINCT"),
        );
        assert_alternating(
            &result,
            &(1..=200).collect::<Vec<_>>(),
            &format!("{language}: DISTINCT"),
        );
        assert_eq!(
            result.cell_types().runs().len(),
            199,
            "{language}: DISTINCT"
        );
    }
}

/// More forms: two per-row columns in one RETURN, an edge and a path as the
/// part a row takes, a `coalesce` of three parts, and a null as the part
/// (a CASE without ELSE, a `coalesce` whose parts are all null), which joins
/// the run of its neighbours.
#[test]
fn more_forms_give_each_row_its_part() {
    let db = three_edges();
    for language in ["gql", "cypher"] {
        let result = run(
            &db,
            language,
            "MATCH (a:A) OPTIONAL MATCH (a)-[:K]->(b:B) WHERE b.n > 3 \
             RETURN a.n AS k, CASE WHEN a.n > 2 THEN a ELSE a.n END AS v, \
             coalesce(b, a.n) AS w",
        );
        assert_eq!(column_v(&result), [int(1), node(3), node(5)], "{language}");
        let mut w: Vec<(Value, String)> = result
            .rows()
            .iter()
            .enumerate()
            .map(|(row, values)| {
                (
                    values[0].clone(),
                    read(&values[2], result.cell_type(row, 2)),
                )
            })
            .collect();
        w.sort_by(|left, right| format!("{:?}", left.0).cmp(&format!("{:?}", right.0)));
        let w: Vec<String> = w.into_iter().map(|(_, w)| w).collect();
        assert_eq!(w, [int(1), node(4), node(6)], "{language}: two columns");

        let result = run(
            &db,
            language,
            "MATCH (a:A)-[r:K]->() RETURN a.n AS k, CASE WHEN a.n > 2 THEN r ELSE a.n END AS v",
        );
        assert_eq!(
            column_v(&result),
            [int(1), edge(2), edge(3)],
            "{language}: edge"
        );

        let result = run(
            &db,
            language,
            "MATCH p = (a:A)-[:K]->() RETURN a.n AS k, CASE WHEN a.n > 2 THEN p ELSE a.n END AS v",
        );
        assert_eq!(
            column_v(&result),
            [
                int(1),
                format!("path[{},{}|{}]", node(3), node(4), edge(2)),
                format!("path[{},{}|{}]", node(5), node(6), edge(3)),
            ],
            "{language}: path"
        );

        let result = run(
            &db,
            language,
            "MATCH (a:A) OPTIONAL MATCH (a)-[:K]->(b:B) WHERE b.n > 5 \
             OPTIONAL MATCH (a)-[r:K]->() WHERE a.n = 3 \
             RETURN a.n AS k, coalesce(b, r, a.n) AS v",
        );
        assert_eq!(
            column_v(&result),
            [int(1), edge(2), node(6)],
            "{language}: three parts"
        );

        for query in [
            "MATCH (a:A) RETURN a.n AS k, CASE WHEN a.n > 2 THEN a WHEN a.n > 9 THEN 0 END AS v",
            "MATCH (a:A) OPTIONAL MATCH (a)-[:K]->(b:B) WHERE b.n > 3 \
             RETURN a.n AS k, coalesce(b, a.missing) AS v",
        ] {
            let result = run(&db, language, query);
            let v = column_v(&result);
            assert_eq!(v[0], format!("{:?}", Value::Null), "{language}: {query}");
            assert!(
                v[1..].iter().all(|v| v.starts_with("n=")),
                "{language}: {query}: {v:?}"
            );
            assert!(
                result.cell_types().runs().is_empty(),
                "{language}: {query}: a null joins the node run"
            );
        }
    }
}

/// A comparison of an item taken from a mixed list by a computed index
/// compares the item, not its id, by each language's rule for a node against
/// a number: `=` is false in both; `<` is null in Cypher (openCypher) and
/// raises values not comparable (22G04) in GQL.
#[test]
fn a_comparison_of_a_mixed_item_compares_the_item() {
    let db = three_edges();
    let result = run(
        &db,
        "cypher",
        "MATCH (a:A {n: 3}) WITH a, toInteger('0') AS i \
         RETURN [a, 1][i] = id(a) AS eq, [a, 1][i] < id(a) + 1 AS lt",
    );
    assert_eq!(result.rows()[0], [Value::Bool(false), Value::Null]);
    let result = run(
        &db,
        "gql",
        "MATCH (a:A {n: 3}) LET i = toInteger('0') RETURN [a, 1][i] = id(a) AS eq",
    );
    assert_eq!(result.rows()[0], [Value::Bool(false)]);
    let error = db
        .execute("MATCH (a:A {n: 3}) LET i = toInteger('0') RETURN [a, 1][i] < id(a) + 1 AS lt")
        .expect_err("GQL orders a node against a number");
    assert_eq!(GqlStatus::from(&error).as_str(), "22G04", "{error}");
}

/// A strict expected failure that pins what happens instead: `observed`
/// must be `wrong`, the outcome until `issue` fixes it. When it is `right`
/// the test fails so the mark is removed, and so does any other outcome.
fn xfail<T: PartialEq + std::fmt::Debug>(spec: &str, issue: &str, observed: T, right: T, wrong: T) {
    assert_ne!(
        observed, right,
        "XPASS: {spec} ({issue}) holds now; remove the xfail"
    );
    assert_eq!(
        observed, wrong,
        "{spec} ({issue}): neither the known outcome nor the right one"
    );
}

/// A cell as its type and value say it: the type, then `map n=3` for a map
/// or the value.
fn typed_cells(result: &QueryResult) -> Vec<String> {
    let mut rows: Vec<(String, String)> = result
        .rows()
        .iter()
        .enumerate()
        .map(|(row, values)| {
            let cell = match &values[1] {
                Value::Map(map) => format!("map n={:?}", key(map, "n")),
                other => format!("{other:?}"),
            };
            (
                format!("{:?}", values[0]),
                format!("{:?} {cell}", result.cell_type(row, 1)),
            )
        })
        .collect();
    rows.sort();
    rows.into_iter().map(|(_, cell)| cell).collect()
}

/// The limits of 0.6.4, until 0.7.0's entity values: a mixed column that a
/// WITH (also a WITH DISTINCT) passes on gives ids, so does a mixed CASE
/// inside another, and an ORDER BY gives the nodes as maps of no node type
/// (the sort keeps one type per column).
#[test]
fn the_forms_left_for_entity_values_keep_their_known_outcome() {
    let db = three_edges();
    let (three, five) = (id_of(&db, 3), id_of(&db, 5));
    let ids = vec![
        format!("Any {}", int(1)),
        format!("Any {}", int(three)),
        format!("Any {}", int(five)),
    ];
    let typed = vec![
        format!("Any {}", int(1)),
        format!("Node map n={:?}", Value::Int64(3)),
        format!("Node map n={:?}", Value::Int64(5)),
    ];
    for language in ["gql", "cypher"] {
        xfail(
            "openCypher: a value taken from a mixed form keeps its kind through WITH \
             (Comparison2 [3])",
            "fix in 0.7.0",
            typed_cells(&run(
                &db,
                language,
                "MATCH (a:A) WITH a.n AS k, CASE WHEN a.n > 2 THEN a ELSE a.n END AS v \
                 RETURN k, v",
            )),
            typed.clone(),
            ids.clone(),
        );
        xfail(
            "a value of a mixed column keeps its kind through WITH DISTINCT",
            "fix in 0.7.0",
            typed_cells(&run(
                &db,
                language,
                "MATCH (a:A) WITH DISTINCT a.n AS k, CASE WHEN a.n > 2 THEN a ELSE a.n END AS v \
                 RETURN k, v",
            )),
            typed.clone(),
            ids.clone(),
        );
        xfail(
            "a value of a mixed RETURN column keeps its type through ORDER BY",
            "fix in 0.7.0",
            typed_cells(&run(
                &db,
                language,
                "MATCH (a:A) RETURN a.n AS k, CASE WHEN a.n > 2 THEN a ELSE a.n END AS v \
                 ORDER BY k DESC",
            )),
            typed.clone(),
            vec![
                format!("Any {}", int(1)),
                format!("Any map n={:?}", Value::Int64(3)),
                format!("Any map n={:?}", Value::Int64(5)),
            ],
        );
        xfail(
            "a mixed form inside another gives each row its part",
            "fix in 0.7.0",
            typed_cells(&run(
                &db,
                language,
                "MATCH (a:A) RETURN a.n AS k, \
                 CASE WHEN a.n > 2 THEN CASE WHEN a.n > 4 THEN a ELSE 0 END ELSE a.n END AS v",
            )),
            vec![
                format!("Any {}", int(1)),
                format!("Any {}", int(0)),
                format!("Node map n={:?}", Value::Int64(5)),
            ],
            vec![
                format!("Any {}", int(1)),
                format!("Any {}", int(0)),
                format!("Any {}", int(five)),
            ],
        );
    }
}

/// DISTINCT, SKIP and UNION ALL on the RETURN keep each row's type.
#[test]
fn distinct_skip_and_union_keep_the_rows_types() {
    let db = three_edges();
    let mixed = "MATCH (a:A) RETURN a.n AS k, CASE WHEN a.n > 2 THEN a ELSE a.n END AS v";
    for (query, expected) in [
        (
            mixed.replace("RETURN", "RETURN DISTINCT"),
            vec![int(1), node(3), node(5)],
        ),
        (format!("{mixed} SKIP 0"), vec![int(1), node(3), node(5)]),
        (
            format!("{mixed} UNION ALL MATCH (b:B {{n: 2}}) RETURN b.n AS k, b AS v"),
            vec![int(1), node(2), node(3), node(5)],
        ),
    ] {
        for language in ["gql", "cypher"] {
            let result = run(&db, language, &query);
            assert_eq!(column_v(&result), expected, "{language}: {query}");
        }
    }
}
