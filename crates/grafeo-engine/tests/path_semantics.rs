//! Path semantics of ISO/IEC 39075:2024 (GQL) pattern matching: a path
//! variable binds the whole path of its path pattern (#590), match modes and
//! path modes constrain every edge and node a binding uses (#591), and a
//! shortest-path search binds its edge variable and keeps an element pattern
//! `WHERE` inside the search (#572).
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test path_semantics
//! ```

use grafeo_common::types::Value;
use grafeo_engine::GrafeoDB;

/// Alix, Gus, Vincent and Mia (`:Person`), with `:KNOWS` edges Alix->Gus
/// (`w` 3), Gus->Vincent (19) and Vincent->Mia (88), all `kind` 'road', and a
/// shortcut Alix->Vincent (38, `kind` 'air'). The `w` of an edge names it.
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

/// A value as text: strings bare, lists in brackets.
fn text(value: &Value) -> String {
    match value {
        Value::String(text) => text.to_string(),
        Value::Int64(number) => number.to_string(),
        Value::Bool(flag) => flag.to_string(),
        Value::Null => "null".to_string(),
        Value::List(items) => format!(
            "[{}]",
            items.iter().map(text).collect::<Vec<_>>().join(", ")
        ),
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

/// The single count a query returns.
fn count(result: grafeo_common::utils::error::Result<grafeo_engine::database::QueryResult>) -> i64 {
    let result = result.unwrap();
    assert_eq!(result.rows().len(), 1, "a count has one row");
    result.rows()[0][0]
        .as_int64()
        .unwrap_or_else(|| panic!("expected a count, got {:?}", result.rows()[0][0]))
}

// ---------------------------------------------------------------------------
// #590: a path variable binds every node and edge of its path pattern
// (ISO/IEC 39075:2024 16.7 <path pattern>: the path variable is bound to the
// path that the whole <path pattern expression> matches)
// ---------------------------------------------------------------------------

/// Two fixed hops: the path is the three nodes and two edges, not the last
/// hop alone.
#[test]
fn a_named_two_hop_path_holds_both_hops() {
    let db = people();
    let want: &[&[&str]] = &[
        &["[Alix, Gus, Vincent]", "[3, 19]", "2"],
        &["[Alix, Vincent, Mia]", "[38, 88]", "2"],
        &["[Gus, Vincent, Mia]", "[19, 88]", "2"],
    ];
    assert_gql(
        &db,
        "MATCH p = (a:Person)-[:KNOWS]->(b:Person)-[:KNOWS]->(c:Person) \
         RETURN [n IN nodes(p) | n.name], [e IN edges(p) | e.w], length(p)",
        want,
    );
    #[cfg(feature = "cypher")]
    assert_cypher(
        &db,
        "MATCH p = (a:Person)-[:KNOWS]->(b:Person)-[:KNOWS]->(c:Person) \
         RETURN [n IN nodes(p) | n.name], [e IN relationships(p) | e.w], length(p)",
        want,
    );
}

/// Without labels the hops run as one fused chain; the path is still whole,
/// and `length`, `nodes` and `edges` read it directly.
#[test]
fn a_named_chain_without_labels_holds_every_hop() {
    let db = people();
    let want: &[&[&str]] = &[
        &["Alix", "Mia", "2", "3", "2"],
        &["Alix", "Vincent", "2", "3", "2"],
        &["Gus", "Mia", "2", "3", "2"],
    ];
    assert_gql(
        &db,
        "MATCH p = (a)-[:KNOWS]->()-[:KNOWS]->(c) \
         RETURN a.name, c.name, length(p), size(nodes(p)), size(edges(p))",
        want,
    );
    #[cfg(feature = "cypher")]
    assert_cypher(
        &db,
        "MATCH p = (a)-[:KNOWS]->()-[:KNOWS]->(c) \
         RETURN a.name, c.name, length(p), size(nodes(p)), size(relationships(p))",
        want,
    );
}

/// The three-edge pattern of Microsoft Fabric's GQL documentation.
#[test]
fn a_named_three_hop_path_holds_all_three_hops() {
    let db = people();
    let want: &[&[&str]] = &[&["[Alix, Gus, Vincent, Mia]", "[3, 19, 88]"]];
    assert_gql(
        &db,
        "MATCH p = (a:Person)-[:KNOWS]->(b:Person)-[:KNOWS]->(c:Person)-[:KNOWS]->(d:Person) \
         RETURN [n IN nodes(p) | n.name], [e IN edges(p) | e.w]",
        want,
    );
    #[cfg(feature = "cypher")]
    assert_cypher(
        &db,
        "MATCH p = (a:Person)-[:KNOWS]->(b:Person)-[:KNOWS]->(c:Person)-[:KNOWS]->(d:Person) \
         RETURN [n IN nodes(p) | n.name], [e IN relationships(p) | e.w]",
        want,
    );
}

/// A variable-length hop after a fixed one: the path runs from the first
/// node through every hop of the quantified edge.
#[test]
fn a_named_path_joins_a_fixed_hop_and_a_variable_length_one() {
    let db = people();
    let want: &[&[&str]] = &[
        &["[Alix, Gus, Vincent, Mia]", "[3, 19, 88]", "3"],
        &["[Alix, Gus, Vincent]", "[3, 19]", "2"],
        &["[Alix, Vincent, Mia]", "[38, 88]", "2"],
    ];
    assert_gql(
        &db,
        "MATCH p = (a:Person {name: 'Alix'})-[:KNOWS]->(b)-[:KNOWS]->{1,2}(c) \
         RETURN [n IN nodes(p) | n.name], [e IN edges(p) | e.w], length(p)",
        want,
    );
    #[cfg(feature = "cypher")]
    assert_cypher(
        &db,
        "MATCH p = (a:Person {name: 'Alix'})-[:KNOWS]->(b)-[:KNOWS*1..2]->(c) \
         RETURN [n IN nodes(p) | n.name], [e IN relationships(p) | e.w], length(p)",
        want,
    );
}

/// A variable-length hop before a fixed one.
#[test]
fn a_named_path_joins_a_variable_length_hop_and_a_fixed_one() {
    let db = people();
    let want: &[&[&str]] = &[
        &["[Alix, Gus, Vincent, Mia]", "[3, 19, 88]"],
        &["[Alix, Gus, Vincent]", "[3, 19]"],
        &["[Alix, Vincent, Mia]", "[38, 88]"],
    ];
    assert_gql(
        &db,
        "MATCH p = (a:Person {name: 'Alix'})-[:KNOWS]->{1,2}(b)-[:KNOWS]->(c) \
         RETURN [n IN nodes(p) | n.name], [e IN edges(p) | e.w]",
        want,
    );
    #[cfg(feature = "cypher")]
    assert_cypher(
        &db,
        "MATCH p = (a:Person {name: 'Alix'})-[:KNOWS*1..2]->(b)-[:KNOWS]->(c) \
         RETURN [n IN nodes(p) | n.name], [e IN relationships(p) | e.w]",
        want,
    );
}

/// The path value itself, and its edges unwound to rows.
#[test]
fn a_named_multi_hop_path_is_one_path_value() {
    let db = people();
    let result = db
        .execute("MATCH p = (a:Person {name: 'Alix'})-[:KNOWS]->(b)-[:KNOWS]->(c:Person {name: 'Vincent'}) RETURN p")
        .unwrap();
    assert_eq!(result.rows().len(), 1, "one path Alix, Gus, Vincent");
    match &result.rows()[0][0] {
        Value::Path { nodes, edges } => {
            assert_eq!(nodes.len(), 3, "the path has three nodes: {nodes:?}");
            assert_eq!(edges.len(), 2, "the path has two edges: {edges:?}");
        }
        other => panic!("expected a path, got {other:?}"),
    }
    #[cfg(feature = "cypher")]
    assert_cypher(
        &db,
        "MATCH p = (a:Person)-[:KNOWS]->(b:Person)-[:KNOWS]->(c:Person) \
         UNWIND relationships(p) AS r RETURN r.w",
        &[&["19"], &["19"], &["3"], &["38"], &["88"], &["88"]],
    );
}

/// A filter on the length of the whole path keeps only the longer ones.
#[test]
fn a_filter_reads_the_length_of_the_whole_path() {
    let db = people();
    assert_gql(
        &db,
        "MATCH p = (a:Person)-[:KNOWS]->(b)-[:KNOWS]->{1,2}(c) WHERE length(p) = 3 \
         RETURN a.name, c.name",
        &[&["Alix", "Mia"]],
    );
}

/// A node without a two-hop path keeps its row with a null path.
#[test]
fn an_optional_named_multi_hop_path_is_null_without_a_match() {
    let db = people();
    let want: &[&[&str]] = &[&["Alix", "2"], &["Alix", "2"], &["Mia", "null"]];
    assert_gql(
        &db,
        "MATCH (x:Person WHERE x.name IN ['Alix', 'Mia']) \
         OPTIONAL MATCH p = (x)-[:KNOWS]->()-[:KNOWS]->() RETURN x.name, length(p)",
        want,
    );
    #[cfg(feature = "cypher")]
    assert_cypher(
        &db,
        "MATCH (x:Person) WHERE x.name IN ['Alix', 'Mia'] \
         OPTIONAL MATCH p = (x)-[:KNOWS]->()-[:KNOWS]->() RETURN x.name, length(p)",
        want,
    );
}

/// A questioned edge (`->?`) that matches nothing leaves its hop out of the
/// path, and a path mode then checks the hops that are there.
#[test]
fn a_missing_questioned_hop_is_left_out_of_the_path() {
    let db = people();
    assert_gql(
        &db,
        "MATCH p = (a:Person {name: 'Gus'})-[:KNOWS]->(b)-[:KNOWS]->?(c) \
         RETURN [n IN nodes(p) | n.name], length(p)",
        &[&["[Gus, Vincent, Mia]", "2"]],
    );
    assert_gql(
        &db,
        "MATCH p = (a:Person {name: 'Vincent'})-[:KNOWS]->(b)-[:KNOWS]->?(c) \
         RETURN [n IN nodes(p) | n.name], length(p), c.name",
        &[&["[Vincent, Mia]", "1", "null"]],
    );
    assert_gql(
        &db,
        "MATCH TRAIL (a:Person {name: 'Vincent'})-[:KNOWS]->(b)-[:KNOWS]->?(c) \
         RETURN b.name, c.name",
        &[&["Mia", "null"]],
    );
    // Two questioned edges that both match nothing bind no edge twice
    assert_gql(
        &db,
        "MATCH DIFFERENT EDGES (a:Person {name: 'Mia'})-[:KNOWS]->?(b), (a)-[:KNOWS]->?(c) \
         RETURN a.name, b.name, c.name",
        &[&["Mia", "null", "null"]],
    );
}

/// `isTrail(p)` sees both edges of a two-hop path: it is false exactly for
/// the walks that go back over the edge they came on.
#[test]
fn is_trail_of_a_two_hop_path_compares_both_edges() {
    let db = people();
    let repeated =
        count(db.execute("MATCH (a)-[e1:KNOWS]-(b)-[e2:KNOWS]-(c) WHERE e1 = e2 RETURN count(*)"));
    assert_eq!(repeated, 8, "each of the four edges, back and forth");
    assert_eq!(
        count(db.execute(
            "MATCH p = (a)-[:KNOWS]-(b)-[:KNOWS]-(c) WHERE NOT isTrail(p) RETURN count(*)"
        )),
        repeated
    );
}

// ---------------------------------------------------------------------------
// #591: match modes (ISO/IEC 39075:2024 16.4 <graph pattern>: with DIFFERENT
// EDGES no two edge patterns of the graph pattern bind the same edge, an edge
// of a quantified pattern included; REPEATABLE ELEMENTS, the default, lets
// them repeat) and path modes (16.6 <path mode>: TRAIL, ACYCLIC and SIMPLE
// hold for the whole path of the path pattern, not for each edge pattern)
// ---------------------------------------------------------------------------

/// Two patterns from the same node: with DIFFERENT EDGES only the rows that
/// bind two different edges stay, as `WHERE e1 <> e2` keeps.
#[test]
fn different_edges_binds_no_edge_twice_across_patterns() {
    let db = people();
    let pattern = "(a)-[e1:KNOWS]->(b), (a)-[e2:KNOWS]->(c)";
    let different = count(db.execute(&format!("MATCH {pattern} WHERE e1 <> e2 RETURN count(*)")));
    assert_eq!(different, 2, "Alix's two edges, in both orders");
    assert_eq!(
        count(db.execute(&format!("MATCH DIFFERENT EDGES {pattern} RETURN count(*)"))),
        different
    );
}

/// Anonymous edge patterns bind edges too: DIFFERENT EDGES covers them.
#[test]
fn different_edges_covers_anonymous_edges() {
    let db = people();
    assert_gql(
        &db,
        "MATCH DIFFERENT EDGES (a)-[:KNOWS]->(b), (a)-[:KNOWS]->(c) \
         RETURN a.name, b.name, c.name",
        &[&["Alix", "Gus", "Vincent"], &["Alix", "Vincent", "Gus"]],
    );
}

/// The edges of one path pattern are edges of the graph pattern as well.
#[test]
fn different_edges_covers_the_edges_of_one_path() {
    let db = people();
    let different =
        count(db.execute("MATCH (a)-[e1:KNOWS]-(b)-[e2:KNOWS]-(c) WHERE e1 <> e2 RETURN count(*)"));
    assert_eq!(
        different, 10,
        "18 two-hop walks, 8 of them back over the same edge"
    );
    assert_eq!(
        count(db.execute("MATCH DIFFERENT EDGES (a)-[:KNOWS]-(b)-[:KNOWS]-(c) RETURN count(*)")),
        different
    );
}

/// The edges of a quantified edge pattern (a group variable) differ from the
/// edge of another pattern.
#[test]
fn different_edges_covers_the_edges_of_a_quantified_edge() {
    let db = people();
    let query = "(a:Person {name: 'Alix'})-[e:KNOWS]->{1,3}(b), (a)-[f:KNOWS]->(c) \
                 RETURN [x IN e | x.w], f.w";
    assert_eq!(
        rows(db.execute(&format!("MATCH {query}"))).len(),
        10,
        "five paths from Alix, each with both of Alix's edges"
    );
    assert_gql(
        &db,
        &format!("MATCH DIFFERENT EDGES {query}"),
        &[
            &["[3, 19, 88]", "38"],
            &["[3, 19]", "38"],
            &["[3]", "38"],
            &["[38, 88]", "3"],
            &["[38]", "3"],
        ],
    );
}

/// REPEATABLE ELEMENTS, and no match mode, keep the rows that bind an edge
/// twice.
#[test]
fn repeatable_elements_and_the_default_let_an_edge_repeat() {
    let db = people();
    let pattern = "(a)-[e1:KNOWS]->(b), (a)-[e2:KNOWS]->(c)";
    assert_eq!(
        count(db.execute(&format!("MATCH {pattern} RETURN count(*)"))),
        6
    );
    assert_eq!(
        count(db.execute(&format!(
            "MATCH REPEATABLE ELEMENTS {pattern} RETURN count(*)"
        ))),
        6
    );
}

/// TRAIL holds for the whole path of fixed edge patterns: no walk goes back
/// over the edge it came on.
#[test]
fn trail_holds_for_the_whole_path_of_fixed_hops() {
    let db = people();
    assert_eq!(
        count(db.execute("MATCH TRAIL (a)-[:KNOWS]-(b)-[:KNOWS]-(c) RETURN count(*)")),
        10
    );
    assert_eq!(
        count(db.execute("MATCH TRAIL (a)-[:KNOWS]-(b)-[:KNOWS]-(c)-[:KNOWS]-(d) RETURN count(*)")),
        10
    );
}

/// TRAIL over a fixed edge pattern and a quantified one: the edges of both
/// differ, also from each other.
#[test]
fn trail_holds_across_a_fixed_and_a_quantified_edge() {
    let db = people();
    assert_eq!(
        count(db.execute("MATCH TRAIL (a)-[:KNOWS]-(b)-[:KNOWS]-{1,2}(c) RETURN count(*)")),
        20,
        "56 walks, 20 of them without a repeated edge"
    );
}

/// ACYCLIC holds for the whole path: no node twice.
#[test]
fn acyclic_holds_for_the_whole_path_of_fixed_hops() {
    let db = people();
    assert_eq!(
        count(
            db.execute("MATCH ACYCLIC (a)-[:KNOWS]-(b)-[:KNOWS]-(c)-[:KNOWS]-(d) RETURN count(*)")
        ),
        4
    );
    assert_eq!(
        count(db.execute("MATCH ACYCLIC (a)-[:KNOWS]-(b)-[:KNOWS]-{1,2}(c) RETURN count(*)")),
        14
    );
}

/// ACYCLIC also holds for one edge pattern: a self-loop repeats its node.
#[test]
fn acyclic_excludes_a_self_loop() {
    let db = GrafeoDB::new_in_memory();
    db.execute(
        "INSERT (alix:Person {name: 'Alix'}), (gus:Person {name: 'Gus'}), \
         (alix)-[:KNOWS]->(alix), (alix)-[:KNOWS]->(gus)",
    )
    .unwrap();
    assert_gql(
        &db,
        "MATCH ACYCLIC (a)-[:KNOWS]->(b) RETURN a.name, b.name",
        &[&["Alix", "Gus"]],
    );
    assert_gql(
        &db,
        "MATCH SIMPLE (a)-[:KNOWS]->(b) RETURN a.name, b.name",
        &[&["Alix", "Alix"], &["Alix", "Gus"]],
    );
}

/// SIMPLE holds for the whole path: no node twice, except that the last
/// node may be the first.
#[test]
fn simple_holds_for_the_whole_path_of_fixed_hops() {
    let db = people();
    assert_eq!(
        count(
            db.execute("MATCH SIMPLE (a)-[:KNOWS]-(b)-[:KNOWS]-(c)-[:KNOWS]-(d) RETURN count(*)")
        ),
        10
    );
    assert_eq!(
        count(db.execute("MATCH SIMPLE (a)-[:KNOWS]-(b)-[:KNOWS]-(c) RETURN count(*)")),
        18,
        "a two-hop walk back to its start is simple"
    );
}

/// A match mode and a path mode are independent: REPEATABLE ELEMENTS does
/// not lift TRAIL, and DIFFERENT EDGES keeps ACYCLIC, in the order of the
/// standard (match mode first) and in the other one.
#[test]
fn a_match_mode_and_a_path_mode_apply_together() {
    let db = people();
    for query in [
        "MATCH REPEATABLE ELEMENTS TRAIL (a)-[:KNOWS]-(b)-[:KNOWS]-(c) RETURN count(*)",
        "MATCH TRAIL REPEATABLE ELEMENTS (a)-[:KNOWS]-(b)-[:KNOWS]-(c) RETURN count(*)",
    ] {
        assert_eq!(count(db.execute(query)), 10, "{query}");
    }
    for query in [
        "MATCH DIFFERENT EDGES ACYCLIC (a)-[:KNOWS]-(b)-[:KNOWS]-(c)-[:KNOWS]-(d) RETURN count(*)",
        "MATCH ACYCLIC DIFFERENT EDGES (a)-[:KNOWS]-(b)-[:KNOWS]-(c)-[:KNOWS]-(d) RETURN count(*)",
    ] {
        assert_eq!(count(db.execute(query)), 4, "{query}");
    }
}

/// KEEP DIFFERENT EDGES on a path pattern: no edge twice in that pattern.
#[test]
fn keep_different_edges_binds_no_edge_twice_in_its_pattern() {
    let db = people();
    assert_eq!(
        count(
            db.execute("MATCH (a)-[:KNOWS]-(b)-[:KNOWS]-(c) KEEP DIFFERENT EDGES RETURN count(*)")
        ),
        10
    );
    assert_eq!(
        count(db.execute(
            "MATCH (a)-[:KNOWS]-(b)-[:KNOWS]-(c) KEEP REPEATABLE ELEMENTS RETURN count(*)"
        )),
        18
    );
}

// ---------------------------------------------------------------------------
// #572: a shortest-path search binds the variables of its path pattern and
// selects among the paths that pattern matches (ISO/IEC 39075:2024 16.6
// <path search prefix>: the selector picks from the paths of the path
// pattern, so an element pattern WHERE or property map holds for every edge
// before the selection; the variable of a quantified edge pattern is a group
// variable, bound to the list of the selected path's edges)
// ---------------------------------------------------------------------------

/// Amsterdam, Berlin, Paris, Prague and Barcelona (`:City`), with `:ROUTE`
/// edges Amsterdam->Berlin (`w` 3), Berlin->Prague (19), Amsterdam->Paris
/// (88) and Paris->Prague (38): two shortest routes from Amsterdam to Prague.
/// Barcelona has no route.
fn cities() -> GrafeoDB {
    let db = GrafeoDB::new_in_memory();
    db.execute(
        "INSERT (ams:City {name: 'Amsterdam'}), (ber:City {name: 'Berlin'}), \
         (par:City {name: 'Paris'}), (pra:City {name: 'Prague'}), (:City {name: 'Barcelona'}), \
         (ams)-[:ROUTE {w: 3}]->(ber), (ber)-[:ROUTE {w: 19}]->(pra), \
         (ams)-[:ROUTE {w: 88}]->(par), (par)-[:ROUTE {w: 38}]->(pra)",
    )
    .unwrap();
    db
}

/// The edge variable of the quantified edge pattern is the list of the
/// path's edges: the Fabric example `RETURN size(e)` failed with "Undefined
/// variable 'e'".
#[test]
fn any_shortest_binds_the_edges_of_its_path_to_the_edge_variable() {
    let db = people();
    let want: &[&[&str]] = &[&["2", "[38, 88]", "2"]];
    assert_gql(
        &db,
        "MATCH p = ANY SHORTEST (a:Person WHERE a.name = 'Alix')-[e:KNOWS]->{1,4}\
         (b:Person WHERE b.name = 'Mia') RETURN size(e), [x IN e | x.w], length(p)",
        want,
    );
    #[cfg(feature = "cypher")]
    assert_cypher(
        &db,
        "MATCH p = shortestPath((a:Person {name: 'Alix'})-[e:KNOWS*1..4]->(b:Person {name: 'Mia'})) \
         RETURN size(e), [x IN e | x.w], length(p)",
        want,
    );
}

/// The path variable of a shortest-path search is the selected path: its
/// nodes and edges, and the path value itself.
#[test]
fn any_shortest_binds_the_whole_path() {
    let db = people();
    let want: &[&[&str]] = &[&["[Alix, Vincent, Mia]", "[38, 88]", "false"]];
    assert_gql(
        &db,
        "MATCH p = ANY SHORTEST (a:Person {name: 'Alix'})-[:KNOWS]->+(b:Person {name: 'Mia'}) \
         RETURN [n IN nodes(p) | n.name], [r IN edges(p) | r.w], p IS NULL",
        want,
    );
    #[cfg(feature = "cypher")]
    assert_cypher(
        &db,
        "MATCH p = shortestPath((a:Person {name: 'Alix'})-[:KNOWS*]->(b:Person {name: 'Mia'})) \
         RETURN [n IN nodes(p) | n.name], [r IN relationships(p) | r.w], p IS NULL",
        want,
    );
    let result = db
        .execute(
            "MATCH p = ANY SHORTEST (a:Person {name: 'Alix'})-[:KNOWS]->+(b:Person {name: 'Mia'}) \
             RETURN p",
        )
        .unwrap();
    match result.rows() {
        [row] => match &row[0] {
            Value::Path { nodes, edges } => {
                assert_eq!((nodes.len(), edges.len()), (3, 2), "Alix, Vincent, Mia");
            }
            other => panic!("expected a path, got {other:?}"),
        },
        other => panic!("expected one row, got {other:?}"),
    }
}

/// An element pattern WHERE on the quantified edge holds for every edge of
/// the path during the search: the shortest road route is the three-hop one.
/// Checked after the selection instead, it would drop the only shortest path
/// (which takes the 'air' edge) and return nothing. The Fabric example with
/// such a WHERE failed with "Variable 'p' not found in input".
#[test]
fn an_edge_where_holds_for_every_edge_during_the_search() {
    let db = people();
    let want: &[&[&str]] = &[&["[3, 19, 88]", "3", "[Alix, Gus, Vincent, Mia]"]];
    assert_gql(
        &db,
        "MATCH p = ANY SHORTEST (a:Person WHERE a.name = 'Alix')\
         -[e:KNOWS WHERE e.kind = 'road']->{1,4}(b:Person WHERE b.name = 'Mia') \
         RETURN [x IN e | x.w], length(p), [n IN nodes(p) | n.name]",
        want,
    );
    assert_gql(
        &db,
        "MATCH p = ANY SHORTEST (a:Person {name: 'Alix'})-[e:KNOWS {kind: 'road'}]->{1,4}\
         (b:Person {name: 'Mia'}) RETURN [x IN e | x.w], length(p), [n IN nodes(p) | n.name]",
        want,
    );
    #[cfg(feature = "cypher")]
    assert_cypher(
        &db,
        "MATCH p = shortestPath((a:Person {name: 'Alix'})-[e:KNOWS* {kind: 'road'}]->\
         (b:Person {name: 'Mia'})) RETURN [x IN e | x.w], length(p), [n IN nodes(p) | n.name]",
        want,
    );
    let result = db
        .execute(
            "MATCH p = ANY SHORTEST (a:Person WHERE a.name = 'Alix')\
             -[c:KNOWS WHERE c.kind = 'road']->{1,4}(b:Person WHERE b.name = 'Mia') RETURN p",
        )
        .unwrap();
    assert_eq!(result.rows().len(), 1, "the path of the road route");
}

/// The edge pattern's WHERE reads parameters too.
#[test]
fn an_edge_where_with_a_parameter_holds_during_the_search() {
    let db = people();
    let query = "MATCH p = ANY SHORTEST (a:Person {name: 'Alix'})-[e:KNOWS WHERE e.w <> $skip]->+\
                 (b:Person {name: 'Mia'}) RETURN [x IN e | x.w]";
    for (skip, want) in [(38, "[3, 19, 88]"), (3, "[38, 88]")] {
        let params = std::collections::HashMap::from([("skip".to_string(), Value::Int64(skip))]);
        assert_eq!(
            rows(db.execute_with_params(query, params)),
            expected(&[&[want]]),
            "skipping {skip}"
        );
    }
}

/// An anonymous endpoint is searched like a named one (it failed with
/// "Undefined variable '_anon_..'").
#[test]
fn any_shortest_from_or_to_an_anonymous_node() {
    let db = people();
    assert_gql(
        &db,
        "MATCH p = ANY SHORTEST (:Person {name: 'Alix'})-[:KNOWS]->+(b:Person {name: 'Mia'}) \
         RETURN length(p)",
        &[&["2"]],
    );
    assert_gql(
        &db,
        "MATCH p = ANY SHORTEST (a:Person {name: 'Gus'})-[:KNOWS]->+() RETURN length(p)",
        &[&["1"], &["2"]],
    );
}

/// A shortest-path search over two edge patterns searched the first alone
/// and ignored the node between them; it is an error until it works.
#[test]
fn a_shortest_path_over_two_edge_patterns_is_an_error() {
    let db = people();
    let error = db
        .execute(
            "MATCH p = ANY SHORTEST (a:Person {name: 'Alix'})-[:KNOWS]->(b)-[:KNOWS]->+(c) \
             RETURN length(p)",
        )
        .unwrap_err();
    assert!(
        error.to_string().contains("more than one edge pattern"),
        "{error}"
    );
    #[cfg(feature = "cypher")]
    {
        let error = db
            .execute_cypher(
                "MATCH p = shortestPath((a:Person {name: 'Alix'})-[:KNOWS]->(b)-[:KNOWS*]->(c)) \
                 RETURN length(p)",
            )
            .unwrap_err();
        assert!(
            error.to_string().contains("exactly one relationship"),
            "{error}"
        );
    }
}

/// A filter after the MATCH reads the edge list of the selected path; it
/// does not change which path is selected.
#[test]
fn a_filter_after_the_search_reads_the_edge_list() {
    let db = people();
    let search = "MATCH p = ANY SHORTEST (a:Person {name: 'Alix'})-[e:KNOWS]->{1,4}\
                  (b:Person {name: 'Mia'})";
    assert_gql(
        &db,
        &format!("{search} FILTER ALL(c IN e WHERE c.w > 30) RETURN length(p)"),
        &[&["2"]],
    );
    assert_gql(
        &db,
        &format!("{search} FILTER ALL(c IN e WHERE c.kind = 'road') RETURN length(p)"),
        &[],
    );
}

/// ALL SHORTEST binds each shortest path, with its own edges.
#[test]
fn all_shortest_binds_each_path() {
    let db = cities();
    let want: &[&[&str]] = &[
        &["[Amsterdam, Berlin, Prague]", "[3, 19]"],
        &["[Amsterdam, Paris, Prague]", "[88, 38]"],
    ];
    assert_gql(
        &db,
        "MATCH p = ALL SHORTEST (a:City {name: 'Amsterdam'})-[r:ROUTE]->+(b:City {name: 'Prague'}) \
         RETURN [n IN nodes(p) | n.name], [x IN r | x.w]",
        want,
    );
    #[cfg(feature = "cypher")]
    assert_cypher(
        &db,
        "MATCH p = allShortestPaths((a:City {name: 'Amsterdam'})-[r:ROUTE*]->(b:City {name: 'Prague'})) \
         RETURN [n IN nodes(p) | n.name], [x IN r | x.w]",
        want,
    );
}

/// An edge bound by an earlier MATCH is the edge of the shortest path: every
/// row keeps its own edge, instead of pairing with every path.
#[test]
fn a_bound_edge_is_the_edge_of_the_shortest_path() {
    let db = people();
    assert_gql(
        &db,
        "MATCH ()-[e:KNOWS]->() MATCH ANY SHORTEST (x)-[e]->(y) RETURN e.w, x.name, y.name",
        &[
            &["19", "Gus", "Vincent"],
            &["3", "Alix", "Gus"],
            &["38", "Alix", "Vincent"],
            &["88", "Vincent", "Mia"],
        ],
    );
    // A quantified edge pattern binds a list of edges, which one edge bound
    // before cannot be
    let error = db
        .execute("MATCH ()-[e:KNOWS]->() MATCH ANY SHORTEST (x)-[e]->+(y) RETURN x.name")
        .unwrap_err();
    assert!(
        error.to_string().contains("quantified edge pattern"),
        "{error}"
    );
}

/// An edge bound by an earlier part of the same MATCH works the same way.
#[test]
fn an_edge_of_an_earlier_part_is_the_edge_of_the_shortest_path() {
    let db = GrafeoDB::new_in_memory();
    db.execute(
        "INSERT (alix:Person {name: 'Alix'}), (gus:Person {name: 'Gus'}), \
         (alix)-[:KNOWS {w: 3}]->(gus), (alix)-[:KNOWS {w: 19}]->(gus)",
    )
    .unwrap();
    for w in [3, 19] {
        assert_gql(
            &db,
            &format!(
                "MATCH (a)-[e:KNOWS {{w: {w}}}]->(b), p = ANY SHORTEST (x)-[e]->(y) \
                 RETURN x.name, y.name, [r IN edges(p) | r.w]"
            ),
            &[&["Alix", "Gus", &format!("[{w}]")]],
        );
    }
}

/// With parallel edges the bound edge is still the one the path takes: the
/// search is limited to it, not done first and joined after.
#[test]
fn a_bound_parallel_edge_is_the_edge_of_the_shortest_path() {
    let db = GrafeoDB::new_in_memory();
    db.execute(
        "INSERT (alix:Person {name: 'Alix'}), (gus:Person {name: 'Gus'}), \
         (alix)-[:KNOWS {w: 3}]->(gus), (alix)-[:KNOWS {w: 19}]->(gus)",
    )
    .unwrap();
    for w in [3, 19] {
        assert_gql(
            &db,
            &format!(
                "MATCH ()-[e:KNOWS {{w: {w}}}]->() MATCH p = ANY SHORTEST (x)-[e]->(y) \
                 RETURN x.name, y.name, [r IN edges(p) | r.w]"
            ),
            &[&["Alix", "Gus", &format!("[{w}]")]],
        );
    }
}
