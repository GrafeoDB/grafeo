//! Results over quantified path patterns (ISO/IEC 39075:2024 16.7 <path
//! pattern expression>): an element pattern `WHERE` of a quantified edge holds
//! for each edge of the path, an aggregate over a group variable is computed
//! per path (horizontal aggregation), a SIMPLE path goes no further once it
//! is back at its start (16.6 <path mode>), and `length`, `nodes` and
//! `relationships` read a path value that the pattern of the same clause did
//! not bind.
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test quantified_path_results
//! ```

use grafeo_common::types::Value;
use grafeo_engine::GrafeoDB;

/// Alix, Gus, Vincent and Mia (`:Person`), with `:KNOWS` edges Alix->Gus
/// (`w` 3), Gus->Vincent (19) and Vincent->Mia (88), all `kind` 'road', and a
/// shortcut Alix->Vincent (38, `kind` 'air'). The `w` of an edge names it.
///
/// From Alix the paths of one to three edges are [3] to Gus, [3, 19] and
/// [38] to Vincent, and [3, 19, 88] and [38, 88] to Mia.
fn people() -> GrafeoDB {
    let db = GrafeoDB::new_in_memory();
    db.execute(
        "INSERT (alix:Person {name: 'Alix'}), (gus:Person {name: 'Gus'}), \
         (vincent:Person {name: 'Vincent'}), (mia:Person {name: 'Mia'}), \
         (alix)-[:KNOWS {w: 3, kind: 'road'}]->(gus), \
         (gus)-[:KNOWS {w: 19, kind: 'road'}]->(vincent), \
         (vincent)-[:KNOWS {w: 88, kind: 'road'}]->(mia), \
         (alix)-[:KNOWS {w: 38, kind: 'air'}]->(vincent)",
    )
    .unwrap();
    db
}

/// A value as text: strings bare, lists in brackets, a path as its node
/// names.
fn text(value: &Value) -> String {
    match value {
        Value::String(text) => text.to_string(),
        Value::Int64(number) => number.to_string(),
        Value::Float64(number) => format!("{number:.1}"),
        Value::Bool(flag) => flag.to_string(),
        Value::Null => "null".to_string(),
        Value::List(items) => format!(
            "[{}]",
            items.iter().map(text).collect::<Vec<_>>().join(", ")
        ),
        Value::Map(map) => map
            .get(&grafeo_common::types::PropertyKey::new("name"))
            .map_or_else(|| format!("{map:?}"), text),
        other => panic!("unexpected value {other:?}"),
    }
}

/// The rows of `result` as text, sorted.
fn rows(
    result: grafeo_common::utils::error::Result<grafeo_engine::database::QueryResult>,
) -> Vec<Vec<String>> {
    let mut rows: Vec<Vec<String>> = result
        .unwrap()
        .rows()
        .iter()
        .map(|row| row.iter().map(text).collect())
        .collect();
    rows.sort();
    rows
}

/// `rows` for literal text cells.
fn expected(rows: &[&[&str]]) -> Vec<Vec<String>> {
    let mut rows: Vec<Vec<String>> = rows
        .iter()
        .map(|row| row.iter().map(|cell| (*cell).to_string()).collect())
        .collect();
    rows.sort();
    rows
}

/// Checks the GQL `query` against `want`.
fn assert_gql(db: &GrafeoDB, query: &str, want: &[&[&str]]) {
    assert_eq!(rows(db.execute(query)), expected(want), "GQL: {query}");
}

/// Checks the Cypher `query` against `want`.
#[cfg(feature = "cypher")]
fn assert_cypher(db: &GrafeoDB, query: &str, want: &[&[&str]]) {
    assert_eq!(
        rows(db.execute_cypher(query)),
        expected(want),
        "Cypher: {query}"
    );
}

/// The column names of the GQL `query`.
fn columns(db: &GrafeoDB, query: &str) -> Vec<String> {
    db.execute(query).unwrap().columns
}

// ---------------------------------------------------------------------------
// An element pattern WHERE of a quantified edge pattern holds for each edge
// (16.7: inside the quantified pattern the edge variable is a singleton, one
// edge per iteration; only outside it is it the group list of the edges)
// ---------------------------------------------------------------------------

/// Only the edges heavier than 10 count: Gus->Vincent (19), Vincent->Mia
/// (88) and Alix->Vincent (38), so no path takes Alix->Gus (3).
#[test]
fn a_quantified_edge_where_holds_for_each_edge() {
    let db = people();
    assert_gql(
        &db,
        "MATCH (a:Person)-[e:KNOWS WHERE e.w > 10]->{1,3}(b) \
         RETURN a.name, b.name, [x IN e | x.w]",
        &[
            &["Alix", "Mia", "[38, 88]"],
            &["Alix", "Vincent", "[38]"],
            &["Gus", "Mia", "[19, 88]"],
            &["Gus", "Vincent", "[19]"],
            &["Vincent", "Mia", "[88]"],
        ],
    );
}

/// The WHERE may compare the edge with a variable bound before the pattern.
#[test]
fn a_quantified_edge_where_reads_an_earlier_variable() {
    let db = people();
    assert_gql(
        &db,
        "UNWIND [19, 50] AS lim \
         MATCH (a:Person {name: 'Alix'})-[e:KNOWS WHERE e.w > lim OR e.w = 3]->{1,3}(b) \
         RETURN lim, b.name, [x IN e | x.w]",
        &[
            &["19", "Gus", "[3]"],
            &["19", "Mia", "[38, 88]"],
            &["19", "Vincent", "[38]"],
            &["50", "Gus", "[3]"],
        ],
    );
}

/// The WHERE may read the pattern's own start node too.
#[test]
fn a_quantified_edge_where_reads_the_start_node() {
    let db = people();
    assert_gql(
        &db,
        "MATCH (a:Person)-[e:KNOWS WHERE e.w > 10 AND a.name <> 'Alix']->{1,3}(b) \
         RETURN a.name, b.name, [x IN e | x.w]",
        &[
            &["Gus", "Mia", "[19, 88]"],
            &["Gus", "Vincent", "[19]"],
            &["Vincent", "Mia", "[88]"],
        ],
    );
}

/// A path of no edges has no edge to check: it matches.
#[test]
fn a_quantified_edge_where_keeps_the_empty_path() {
    let db = people();
    assert_gql(
        &db,
        "MATCH (a:Person {name: 'Alix'})-[e:KNOWS WHERE e.w > 10]->{0,2}(b) \
         RETURN b.name, [x IN e | x.w]",
        &[&["Alix", "[]"], &["Mia", "[38, 88]"], &["Vincent", "[38]"]],
    );
}

/// The WHERE of an anonymous quantified edge holds for each edge as well, so
/// the empty path, which has no edge, matches even when it is false.
#[test]
fn an_anonymous_quantified_edge_where_holds_for_each_edge() {
    let db = people();
    assert_gql(
        &db,
        "MATCH (a:Person {name: 'Alix'})-[:KNOWS WHERE a.name = 'Gus']->{0,3}(b) \
         RETURN b.name",
        &[&["Alix"]],
    );
    assert_gql(
        &db,
        "MATCH (a:Person {name: 'Alix'})-[:KNOWS WHERE a.name = 'Alix']->{1,3}(b) \
         RETURN b.name",
        &[&["Gus"], &["Mia"], &["Mia"], &["Vincent"], &["Vincent"]],
    );
}

/// A property map of a quantified edge holds for each edge too, in GQL and
/// in Cypher (`-[e:T*1..3 {k: v}]->`, openCypher: every relationship of the
/// variable-length pattern has the properties).
#[test]
fn a_quantified_edge_property_map_holds_for_each_edge() {
    let db = people();
    let want: &[&[&str]] = &[
        &["Gus", "[3]"],
        &["Mia", "[3, 19, 88]"],
        &["Vincent", "[3, 19]"],
    ];
    assert_gql(
        &db,
        "MATCH (a:Person {name: 'Alix'})-[e:KNOWS {kind: 'road'}]->{1,3}(b) \
         RETURN b.name, [x IN e | x.w]",
        want,
    );
    #[cfg(feature = "cypher")]
    assert_cypher(
        &db,
        "MATCH (a:Person {name: 'Alix'})-[e:KNOWS*1..3 {kind: 'road'}]->(b) \
         RETURN b.name, [x IN e | x.w]",
        want,
    );
}

/// The per-edge WHERE and the group list outside the pattern work together:
/// a statement WHERE after the pattern reads the list.
#[test]
fn a_statement_where_reads_the_group_list_after_the_per_edge_where() {
    let db = people();
    assert_gql(
        &db,
        "MATCH (a:Person)-[e:KNOWS WHERE e.w > 10]->{1,3}(b) WHERE size(e) = 2 \
         RETURN a.name, b.name",
        &[&["Alix", "Mia"], &["Gus", "Mia"]],
    );
}

// ---------------------------------------------------------------------------
// Horizontal aggregation: an aggregate over a group variable is computed per
// path, not over the rows (ISO/IEC 39075:2024 20.9 <aggregate function>; a
// group variable degree of reference, 16.7)
// ---------------------------------------------------------------------------

/// The sum of the weights of each path, beside the path's end.
#[test]
fn a_sum_over_a_group_variable_is_per_path() {
    let db = people();
    let want: &[&[&str]] = &[
        &["Gus", "3"],
        &["Mia", "110"],
        &["Mia", "126"],
        &["Vincent", "22"],
        &["Vincent", "38"],
    ];
    assert_gql(
        &db,
        "MATCH (a:Person {name: 'Alix'})-[e:KNOWS]->{1,3}(b) RETURN b.name, sum(e.w)",
        want,
    );
    assert_gql(
        &db,
        "MATCH (a:Person {name: 'Alix'})-[e:KNOWS]->{1,3}(b) \
         RETURN b.name AS name, sum(e.w) AS total",
        want,
    );
}

/// The result has the columns the RETURN names, and no internal ones.
#[test]
fn a_horizontal_aggregate_returns_only_its_columns() {
    let db = people();
    assert_eq!(
        columns(
            &db,
            "MATCH (a:Person {name: 'Alix'})-[e:KNOWS]->{1,3}(b) RETURN sum(e.w)"
        ),
        ["sum(e.w)"]
    );
    assert_eq!(
        columns(
            &db,
            "MATCH (a:Person {name: 'Alix'})-[e:KNOWS]->{1,3}(b) \
             RETURN a.name, sum(e.w) AS total, b.name"
        ),
        ["a.name", "total", "b.name"]
    );
    assert_gql(
        &db,
        "MATCH (a:Person {name: 'Alix'})-[e:KNOWS]->{1,3}(b) RETURN sum(e.w) AS total",
        &[&["110"], &["126"], &["22"], &["3"], &["38"]],
    );
}

/// Other aggregate functions over a property of the edges.
#[test]
fn other_horizontal_aggregates_are_per_path() {
    let db = people();
    assert_gql(
        &db,
        "MATCH (a:Person {name: 'Alix'})-[e:KNOWS]->{1,3}(b) \
         RETURN b.name, count(e.w) AS hops, min(e.w) AS low, max(e.w) AS high, \
         avg(e.w) AS mean, collect(e.kind) AS kinds",
        &[
            &["Gus", "1", "3", "3", "3.0", "[road]"],
            &["Mia", "2", "38", "88", "63.0", "[air, road]"],
            &["Mia", "3", "3", "88", "36.7", "[road, road, road]"],
            &["Vincent", "1", "38", "38", "38.0", "[air]"],
            &["Vincent", "2", "3", "19", "11.0", "[road, road]"],
        ],
    );
}

/// A horizontal aggregate inside an expression, ordered by its alias.
#[test]
fn a_horizontal_aggregate_in_an_expression() {
    let db = people();
    let result = db
        .execute(
            "MATCH (a:Person {name: 'Alix'})-[e:KNOWS]->{1,3}(b) \
             RETURN b.name AS name, sum(e.w) * 2 + 1 AS score ORDER BY score DESC",
        )
        .unwrap();
    let got: Vec<Vec<String>> = result
        .rows()
        .iter()
        .map(|row| row.iter().map(text).collect())
        .collect();
    let want: Vec<Vec<String>> = [
        ["Mia", "253"],
        ["Mia", "221"],
        ["Vincent", "77"],
        ["Vincent", "45"],
        ["Gus", "7"],
    ]
    .iter()
    .map(|row| row.iter().map(|cell| (*cell).to_string()).collect())
    .collect();
    assert_eq!(got, want, "ordered by the score, highest first");
}

/// Beside a regular aggregate, a horizontal aggregate is a grouping key: the
/// paths are grouped by whether their sum is over 50.
#[test]
fn a_horizontal_aggregate_beside_a_regular_aggregate_groups_the_rows() {
    let db = people();
    assert_gql(
        &db,
        "MATCH (a:Person {name: 'Alix'})-[e:KNOWS]->{1,3}(b) \
         RETURN sum(e.w) > 50 AS long, count(*) AS paths",
        &[&["false", "3"], &["true", "2"]],
    );
    // HAVING reads the sum of each path too: only Mia has a path over 100
    assert_gql(
        &db,
        "MATCH (a:Person {name: 'Alix'})-[e:KNOWS]->{1,3}(b) \
         RETURN b.name, count(*) AS paths GROUP BY b.name HAVING max(sum(e.w)) > 100",
        &[&["Mia", "2"]],
    );
    // A float and a list as grouping keys: the lightest edge of each path,
    // halved, and the kinds of its edges
    assert_gql(
        &db,
        "MATCH (a:Person {name: 'Alix'})-[e:KNOWS]->{1,3}(b) \
         RETURN min(e.w) / 2.0 AS half, count(*) AS paths",
        &[&["1.5", "3"], &["19.0", "2"]],
    );
    assert_gql(
        &db,
        "MATCH (a:Person {name: 'Alix'})-[e:KNOWS]->{1,3}(b) \
         RETURN collect(e.kind) AS kinds, count(*) AS paths",
        &[
            &["[air, road]", "1"],
            &["[air]", "1"],
            &["[road, road, road]", "1"],
            &["[road, road]", "1"],
            &["[road]", "1"],
        ],
    );
    assert_gql(
        &db,
        "MATCH (a:Person {name: 'Alix'})-[e:KNOWS]->{1,3}(b) \
         RETURN b.name AS name, count(*) AS paths, max(sum(e.w)) AS longest",
        &[
            &["Gus", "1", "3"],
            &["Mia", "2", "126"],
            &["Vincent", "2", "38"],
        ],
    );
}

/// A WITH computes a horizontal aggregate per path too, and its WHERE reads
/// it; DISTINCT reads each value of a path once.
#[test]
fn a_horizontal_aggregate_in_with() {
    let db = people();
    assert_gql(
        &db,
        "MATCH (a:Person {name: 'Alix'})-[e:KNOWS]->{1,3}(b) \
         WITH b, sum(e.w) AS total, count(DISTINCT e.kind) AS kinds WHERE total > 50 \
         RETURN b.name, total, kinds",
        &[&["Mia", "110", "1"], &["Mia", "126", "2"]],
    );
}

/// A group variable that a WITH passes on stays one; a later pattern that
/// binds its name to one edge makes the aggregate a regular one again.
#[test]
fn a_horizontal_aggregate_reads_the_group_variable_in_scope() {
    let db = people();
    assert_gql(
        &db,
        "MATCH (a:Person {name: 'Alix'})-[e:KNOWS]->{1,2}(b) WITH b, e \
         RETURN b.name, sum(e.w) AS total",
        &[
            &["Gus", "3"],
            &["Mia", "126"],
            &["Vincent", "22"],
            &["Vincent", "38"],
        ],
    );
    assert_gql(
        &db,
        "MATCH (a:Person {name: 'Alix'})-[e:KNOWS]->{1,1}(b) WITH b \
         MATCH (b)-[e:KNOWS]->(c) RETURN sum(e.w) AS total",
        &[&["107"]],
    );
}

/// ORDER BY the aggregate itself, and its unaliased column names.
#[test]
fn ordering_by_an_unaliased_horizontal_aggregate() {
    let db = people();
    let query = "MATCH (a:Person {name: 'Alix'})-[e:KNOWS]->{1,3}(b) \
                 RETURN b.name, sum(e.w), sum(e.w) * 2 ORDER BY sum(e.w)";
    assert_eq!(columns(&db, query), ["b.name", "sum(e.w)", "sum(e.w) * 2"]);
    let got: Vec<Vec<String>> = db
        .execute(query)
        .unwrap()
        .rows()
        .iter()
        .map(|row| row.iter().map(text).collect())
        .collect();
    let want: Vec<Vec<String>> = [
        ["Gus", "3", "6"],
        ["Vincent", "22", "44"],
        ["Vincent", "38", "76"],
        ["Mia", "110", "220"],
        ["Mia", "126", "252"],
    ]
    .iter()
    .map(|row| row.iter().map(|cell| (*cell).to_string()).collect())
    .collect();
    assert_eq!(got, want, "ordered by the sum of each path");

    // Ordered by a sum the RETURN leaves out, with a path's last weight as
    // the tie breaker that tells the two Vincent paths apart
    let got: Vec<Value> = db
        .execute(
            "MATCH (a:Person {name: 'Alix'})-[e:KNOWS]->{1,3}(b) \
             RETURN b.name, max(e.w) AS high ORDER BY sum(e.w) DESC",
        )
        .unwrap()
        .rows()
        .iter()
        .map(|row| row[1].clone())
        .collect();
    assert_eq!(
        got,
        [88, 88, 38, 19, 3].map(Value::Int64),
        "the heaviest edge of the paths, from the longest sum (126) down"
    );
}

/// A horizontal aggregate over the edges a shortest path search binds.
#[test]
fn a_horizontal_aggregate_over_a_shortest_path() {
    let db = people();
    assert_gql(
        &db,
        "MATCH ANY SHORTEST (a:Person {name: 'Alix'})-[e:KNOWS]->{1,3}(b:Person {name: 'Mia'}) \
         RETURN sum(e.w) AS total",
        &[&["126"]],
    );
}

/// A horizontal aggregate sees the writes of its own transaction.
#[test]
fn a_horizontal_aggregate_reads_the_open_transaction() {
    let db = people();
    let mut session = db.session();
    session.begin_transaction().unwrap();
    session
        .execute("MATCH (:Person {name: 'Alix'})-[k:KNOWS {w: 3}]->() SET k.w = 4")
        .unwrap();
    let result = session
        .execute(
            "MATCH (a:Person {name: 'Alix'})-[e:KNOWS]->{1,1}(b:Person {name: 'Gus'}) \
             RETURN sum(e.w) AS total",
        )
        .unwrap();
    assert_eq!(
        rows(Ok(result)),
        expected(&[&["4"]]),
        "the sum reads the w the transaction set"
    );
    session.rollback().unwrap();
}

// ---------------------------------------------------------------------------
// SIMPLE (16.6 <path mode>): no node repeats, except that the path may end
// where it started; once back at the start, it goes no further
// ---------------------------------------------------------------------------

/// x - y and x - w, walked in both directions: from x, the simple paths are
/// x y, x w, and x y x and x w x (back at the start); never x y x w or
/// x w x y, which go on from the start.
#[test]
fn a_simple_path_stops_once_back_at_its_start() {
    let db = GrafeoDB::new_in_memory();
    db.execute(
        "INSERT (x:Spot {name: 'x'}), (y:Spot {name: 'y'}), (w:Spot {name: 'w'}), \
         (x)-[:L]->(y), (x)-[:L]->(w)",
    )
    .unwrap();
    assert_gql(
        &db,
        "MATCH p = SIMPLE (x:Spot {name: 'x'})-[:L]-{1,5}(z) \
         RETURN [n IN nodes(p) | n.name]",
        &[&["[x, w, x]"], &["[x, w]"], &["[x, y, x]"], &["[x, y]"]],
    );
    assert_gql(
        &db,
        "MATCH SIMPLE (x:Spot {name: 'x'})-[:L]-{1,5}(z) RETURN z.name",
        &[&["w"], &["x"], &["x"], &["y"]],
    );
}

/// A cycle that comes back to its start is one simple path; going on from
/// the start to another node is not.
#[test]
fn a_simple_cycle_ends_at_its_start() {
    let db = GrafeoDB::new_in_memory();
    db.execute(
        "INSERT (a:Spot {name: 'a'}), (b:Spot {name: 'b'}), (c:Spot {name: 'c'}), \
         (d:Spot {name: 'd'}), (a)-[:L]->(b), (b)-[:L]->(c), (c)-[:L]->(a), (a)-[:L]->(d)",
    )
    .unwrap();
    assert_gql(
        &db,
        "MATCH p = SIMPLE (s:Spot {name: 'a'})-[:L]->{1,6}(t) \
         RETURN [n IN nodes(p) | n.name]",
        &[&["[a, b, c, a]"], &["[a, b, c]"], &["[a, b]"], &["[a, d]"]],
    );
    // A path that is back at its start before the minimum is too short and
    // cannot go on: none of four or more edges
    assert_gql(
        &db,
        "MATCH SIMPLE (s:Spot {name: 'a'})-[:L]->{4,6}(t) RETURN t.name",
        &[],
    );
}

// ---------------------------------------------------------------------------
// length, nodes and relationships of a path value that the pattern of the
// same clause did not bind
// ---------------------------------------------------------------------------

/// After a WITH that passes the path on, as the inbox note found it.
#[test]
fn the_length_of_a_path_after_with() {
    let db = people();
    let want: &[&[&str]] = &[&["Mia", "2"], &["Mia", "3"], &["Vincent", "2"]];
    assert_gql(
        &db,
        "MATCH p = (a:Person {name: 'Alix'})-[:KNOWS]->{1,3}(b) WITH b, p \
         WHERE length(p) >= 2 RETURN b.name, length(p)",
        want,
    );
    #[cfg(feature = "cypher")]
    assert_cypher(
        &db,
        "MATCH p = (a:Person {name: 'Alix'})-[:KNOWS*1..3]->(b) WITH b, p \
         WHERE length(p) >= 2 RETURN b.name, length(p)",
        want,
    );
}

/// A renamed path, after an OPTIONAL MATCH as in the inbox note.
#[test]
fn the_length_of_an_optional_path_after_with() {
    let db = people();
    let want: &[&[&str]] = &[&["Gus", "null"], &["Mia", "null"], &["Vincent", "1"]];
    assert_gql(
        &db,
        "MATCH (a:Person {name: 'Alix'})-[:KNOWS]->{1,2}(b) \
         OPTIONAL MATCH p = (b)-[:KNOWS WHERE b.name <> 'Gus']->(c) \
         WITH DISTINCT b, p AS q RETURN b.name, length(q)",
        want,
    );
}

/// A path unwound from a list of paths: its length, nodes and edges.
#[test]
fn the_length_nodes_and_edges_of_an_unwound_path() {
    let db = people();
    let want: &[&[&str]] = &[
        &["1", "[Alix, Gus]", "[3]"],
        &["2", "[Alix, Vincent, Mia]", "[38, 88]"],
    ];
    assert_gql(
        &db,
        "MATCH p = (a:Person {name: 'Alix'})-[:KNOWS]->{1,2}(b) WHERE b.name <> 'Vincent' \
         WITH collect(p) AS ps UNWIND ps AS u \
         RETURN length(u), [n IN nodes(u) | n.name], [r IN relationships(u) | r.w]",
        want,
    );
    #[cfg(feature = "cypher")]
    assert_cypher(
        &db,
        "MATCH p = (a:Person {name: 'Alix'})-[:KNOWS*1..2]->(b) WHERE b.name <> 'Vincent' \
         WITH collect(p) AS ps UNWIND ps AS u \
         RETURN length(u), [n IN nodes(u) | n.name], [r IN relationships(u) | r.w]",
        want,
    );
}

/// `nodes` and `relationships` of an unwound path returned as they are.
#[test]
fn the_nodes_and_edges_of_an_unwound_path_are_returned() {
    let db = people();
    let result = db
        .execute(
            "MATCH p = (a:Person {name: 'Alix'})-[:KNOWS]->(b:Person {name: 'Gus'}) \
             WITH collect(p) AS ps UNWIND ps AS u \
             RETURN length(u) AS len, nodes(u) AS ns, relationships(u) AS rs",
        )
        .unwrap();
    assert_eq!(result.rows().len(), 1, "one path Alix to Gus");
    let row = &result.rows()[0];
    assert_eq!(row[0], Value::Int64(1), "the path has one edge");
    match (&row[1], &row[2]) {
        (Value::List(nodes), Value::List(edges)) => {
            assert_eq!(nodes.len(), 2, "two nodes: {nodes:?}");
            assert_eq!(edges.len(), 1, "one edge: {edges:?}");
        }
        other => panic!("expected lists of nodes and edges, got {other:?}"),
    }
}

/// `length` of a variable that holds a string or a list measures it: the
/// variable is not taken for a path.
#[test]
fn the_length_of_a_string_or_list_variable() {
    let db = people();
    let want: &[&[&str]] = &[&["Amsterdam", "9"], &["Paris", "5"]];
    let query = "UNWIND ['Amsterdam', 'Paris'] AS s RETURN s, length(s)";
    assert_gql(&db, query, want);
    #[cfg(feature = "cypher")]
    assert_cypher(&db, query, want);
    assert_gql(
        &db,
        "UNWIND [[3, 19, 88], [3]] AS xs WITH xs WHERE length(xs) > 1 RETURN length(xs)",
        &[&["3"]],
    );
    assert_eq!(
        columns(
            &db,
            "MATCH p = (a:Person {name: 'Alix'})-[:KNOWS]->(b) RETURN length(p)"
        ),
        ["length(p)"],
        "an unaliased length is named after the call, not an internal column"
    );
}

/// A length filter on an unwound path.
#[test]
fn a_filter_reads_the_length_of_an_unwound_path() {
    let db = people();
    assert_gql(
        &db,
        "MATCH p = (a:Person {name: 'Alix'})-[:KNOWS]->{1,3}(b) \
         WITH collect(p) AS ps UNWIND ps AS u WITH u WHERE length(u) = 3 \
         RETURN [n IN nodes(u) | n.name]",
        &[&["[Alix, Gus, Vincent, Mia]"]],
    );
}

// ---------------------------------------------------------------------------
// EXPLAIN of a variable-length expand
// ---------------------------------------------------------------------------

/// EXPLAIN names a horizontal aggregate and what it reads.
#[test]
fn explain_shows_a_horizontal_aggregate() {
    let db = people();
    let plan = db
        .execute(
            "EXPLAIN MATCH (a:Person {name: 'Alix'})-[e:KNOWS]->{1,3}(b) RETURN sum(e.w) AS total",
        )
        .unwrap()
        .rows()[0][0]
        .clone();
    let Value::String(plan) = plan else {
        panic!("EXPLAIN returns its plan as text, got {plan:?}");
    };
    assert!(
        plan.contains("HorizontalAggregate (") && plan.contains("Sum(e.w))"),
        "{plan}"
    );
    assert!(
        !plan.contains("Discriminant"),
        "every operator is named: {plan}"
    );
}

/// An untyped expand prints its hop range once: `[*1..2]`, not `[:**1..2]`.
#[test]
fn explain_prints_an_untyped_variable_length_expand_once() {
    let db = people();
    let plan = db
        .execute("EXPLAIN MATCH (n)-[*1..2]-(m) RETURN m")
        .unwrap()
        .rows()[0][0]
        .clone();
    let Value::String(plan) = plan else {
        panic!("EXPLAIN returns its plan as text, got {plan:?}");
    };
    assert!(plan.contains("--[*1..2]--"), "untyped: {plan}");
    assert!(
        !plan.contains("[:*"),
        "no type marker before the range: {plan}"
    );
    let plan = db
        .execute("EXPLAIN MATCH (n)-[:KNOWS*1..2]-(m) RETURN m")
        .unwrap()
        .rows()[0][0]
        .clone();
    let Value::String(plan) = plan else {
        panic!("EXPLAIN returns its plan as text, got {plan:?}");
    };
    assert!(plan.contains("[:KNOWS*1..2]"), "typed: {plan}");
}
