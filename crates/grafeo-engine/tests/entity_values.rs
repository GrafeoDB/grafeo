//! Nodes and edges stay nodes and edges inside the values that hold them: a
//! list or map literal, what `collect`, grouping and `UNWIND` make of such a
//! list or map, a whole path and what `startNode` and `endNode` return. A
//! property read through any of them (`x.msg.name`, `l[0].name`,
//! `startNode(r).name`) reads the entity's property, and RETURN gives the
//! node or edge, with its labels or type and its properties. openCypher (a
//! list or map holds values of any kind, a path value holds its nodes and
//! relationships, `startNode` and `endNode` return nodes) and ISO/IEC 39075
//! (list and record values hold node and edge references, 4.4 and 20.21)
//! define it.
//!
//! These values used to hold the bare ID of each entity: `{msg: m}.msg.id`
//! was null (LDBC SNB IC7), `[n, 1]` returned `[0, 1]`, `RETURN p` gave a
//! path of IDs and `startNode(r).name` was an error.
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test entity_values
//! ```

#![cfg(all(feature = "lpg", feature = "gql", feature = "cypher"))]

use grafeo_common::types::{PropertyKey, Value};
use grafeo_engine::GrafeoDB;

/// `(:Person {name: 'Alix', age: 19})-[:KNOWS {w: 3}]->(:Person {name:
/// 'Gus', age: 88})-[:LIVES_IN {w: 19}]->(:City {name: 'Amsterdam'})`.
fn people() -> GrafeoDB {
    let db = GrafeoDB::new_in_memory();
    db.execute(
        "INSERT (:Person {name: 'Alix', age: 19})-[:KNOWS {w: 3}]->\
         (:Person {name: 'Gus', age: 88})-[:LIVES_IN {w: 19}]->(:City {name: 'Amsterdam'})",
    )
    .unwrap();
    db
}

/// A short, exact form of a returned value: `(:Person {age: 19, name:
/// 'Alix'})` for a node, `[:KNOWS {w: 3}]` for an edge, `<node, edge,
/// node>` for a path, and maps, lists and plain values written out. An ID
/// where an entity belongs shows as a bare number.
fn describe(value: &Value) -> String {
    match value {
        Value::Map(map) => {
            let get = |key: &str| map.get(&PropertyKey::new(key));
            let properties = || {
                let mut entries: Vec<String> = map
                    .iter()
                    .filter(|(key, _)| !key.as_str().starts_with('_'))
                    .map(|(key, value)| format!("{}: {}", key.as_str(), describe(value)))
                    .collect();
                entries.sort();
                entries.join(", ")
            };
            match (get("_id"), get("_labels"), get("_type")) {
                (Some(Value::Int64(_)), Some(Value::List(labels)), _) => {
                    let labels: Vec<String> = labels.iter().map(text).collect();
                    format!("(:{} {{{}}})", labels.join(":"), properties())
                }
                (Some(Value::Int64(_)), None, Some(edge_type)) => {
                    format!("[:{} {{{}}}]", text(edge_type), properties())
                }
                _ => {
                    let mut entries: Vec<String> = map
                        .iter()
                        .map(|(key, value)| format!("{}: {}", key.as_str(), describe(value)))
                        .collect();
                    entries.sort();
                    format!("{{{}}}", entries.join(", "))
                }
            }
        }
        Value::List(items) => {
            let items: Vec<String> = items.iter().map(describe).collect();
            format!("[{}]", items.join(", "))
        }
        Value::Path { nodes, edges } => {
            let mut parts = Vec::new();
            for (i, node) in nodes.iter().enumerate() {
                parts.push(describe(node));
                if let Some(edge) = edges.get(i) {
                    parts.push(describe(edge));
                }
            }
            format!("<{}>", parts.join(", "))
        }
        Value::String(s) => format!("'{s}'"),
        Value::Int64(n) => n.to_string(),
        Value::Null => "null".to_string(),
        other => format!("{other:?}"),
    }
}

fn text(value: &Value) -> String {
    match value {
        Value::String(s) => s.to_string(),
        other => format!("{other:?}"),
    }
}

/// The rows of `query` in `language`, each value described.
fn rows_in(db: &GrafeoDB, language: &str, query: &str) -> Vec<Vec<String>> {
    let result = match language {
        "gql" => db.execute(query),
        _ => db.execute_cypher(query),
    }
    .unwrap_or_else(|error| panic!("{language}: {query}: {error}"));
    result
        .rows()
        .iter()
        .map(|row| row.iter().map(describe).collect())
        .collect()
}

/// The rows of `query`, which reads the same in GQL and Cypher: both give
/// the same rows.
fn rows(db: &GrafeoDB, query: &str) -> Vec<Vec<String>> {
    let gql = rows_in(db, "gql", query);
    let cypher = rows_in(db, "cypher", query);
    assert_eq!(gql, cypher, "GQL and Cypher differ for {query}");
    gql
}

/// The one row of `query`, in GQL and Cypher.
fn row(db: &GrafeoDB, query: &str) -> Vec<String> {
    let mut rows = rows(db, query);
    assert_eq!(rows.len(), 1, "{query}: {rows:?}");
    rows.remove(0)
}

const ALIX: &str = "(:Person {age: 19, name: 'Alix'})";
const GUS: &str = "(:Person {age: 88, name: 'Gus'})";
const AMSTERDAM: &str = "(:City {name: 'Amsterdam'})";
const KNOWS: &str = "[:KNOWS {w: 3}]";
const LIVES_IN: &str = "[:LIVES_IN {w: 19}]";

// ---------------------------------------------------------------------------
// List literals
// ---------------------------------------------------------------------------

/// A node in a list literal is the node, next to a plain number that must
/// stay a number (node IDs start at 0, so `3` would read as a node).
#[test]
fn a_node_in_a_list_literal_stays_a_node() {
    let db = people();
    assert_eq!(
        row(&db, "MATCH (n:Person {name: 'Alix'}) RETURN [n, 3] AS l"),
        [format!("[{ALIX}, 3]")]
    );
}

#[test]
fn a_list_literal_of_a_node_an_edge_and_a_node_keeps_each_kind() {
    let db = people();
    assert_eq!(
        row(
            &db,
            "MATCH (a:Person {name: 'Alix'})-[r:KNOWS]->(b) RETURN [a, r, b] AS t"
        ),
        [format!("[{ALIX}, {KNOWS}, {GUS}]")]
    );
}

/// The list goes through WITH and keeps its kinds; an item's property is the
/// entity's.
#[test]
fn a_list_literal_keeps_its_kinds_through_with() {
    let db = people();
    assert_eq!(
        row(
            &db,
            "MATCH (a:Person {name: 'Alix'})-[r:KNOWS]->(b) WITH [a, r, 3] AS l \
             RETURN l, l[0].name AS a, l[1].w AS w, l[-1] AS last"
        ),
        [
            format!("[{ALIX}, {KNOWS}, 3]"),
            "'Alix'".into(),
            "3".into(),
            "3".into()
        ]
    );
    assert_eq!(
        row(
            &db,
            "MATCH (n:Person {name: 'Gus'}) WITH [n] AS l RETURN l, l[0].age AS age"
        ),
        [format!("[{GUS}]"), "88".into()]
    );
}

/// The list holds the node, not a copy: membership still compares nodes,
/// and a property set later reads its new value.
#[test]
fn a_list_literal_holds_the_node_itself() {
    let db = people();
    assert_eq!(
        rows_in(
            &db,
            "cypher",
            "MATCH (n:Person {name: 'Alix'}) WITH [n] AS l MATCH (m:Person) WHERE m IN l \
             RETURN m.name AS name"
        ),
        [["'Alix'"]]
    );
    assert_eq!(
        row(
            &db,
            "MATCH (n:Person {name: 'Alix'}), (m:Person) WITH m, [n] AS l, {k: n} AS x \
             WHERE m IN l AND x.k = m AND l[0] = m RETURN m.name AS name"
        ),
        ["'Alix'"]
    );
    let set = db
        .execute_cypher(
            "MATCH (n:Person {name: 'Alix'}) WITH n, {p: n} AS x SET n.age = 3 \
             RETURN x.p.age AS age",
        )
        .unwrap();
    assert_eq!(set.rows(), [vec![Value::Int64(3)]]);
}

/// UNWIND of a list literal of nodes gives nodes.
#[test]
fn unwind_of_a_list_literal_of_nodes_gives_nodes() {
    let db = people();
    assert_eq!(
        rows(
            &db,
            "MATCH (a:Person {name: 'Alix'}), (b:City) UNWIND [a, b] AS x \
             RETURN x, x.name AS name ORDER BY name"
        ),
        [
            vec![ALIX.to_string(), "'Alix'".into()],
            vec![AMSTERDAM.to_string(), "'Amsterdam'".into()],
        ]
    );
}

// ---------------------------------------------------------------------------
// Map literals
// ---------------------------------------------------------------------------

#[test]
fn a_node_and_an_edge_in_a_map_literal_keep_their_kind() {
    let db = people();
    assert_eq!(
        row(
            &db,
            "MATCH (a:Person {name: 'Alix'})-[r:KNOWS]->(b) RETURN {k: a, e: r, n: 3} AS m"
        ),
        [format!("{{e: {KNOWS}, k: {ALIX}, n: 3}}")]
    );
}

/// The case of LDBC SNB IC7: `{msg: m}.msg.id`, through WITH.
#[test]
fn a_property_of_a_node_in_a_map_reads_the_node() {
    let db = people();
    assert_eq!(
        row(
            &db,
            "MATCH (m:Person {name: 'Gus'}) WITH {msg: m, t: 19} AS x \
             RETURN x.msg.name AS name, x.msg AS msg, x.t AS t, x.msg.age + x.t AS sum"
        ),
        ["'Gus'".to_string(), GUS.into(), "19".into(), "107".into()]
    );
    assert_eq!(
        row(
            &db,
            "MATCH ()-[r:LIVES_IN]->() WITH {e: r} AS x RETURN x.e.w AS w, x.e AS e"
        ),
        ["19".to_string(), LIVES_IN.into()]
    );
}

/// A map inside a map, read in one expression and through WITH.
#[test]
fn a_node_in_a_nested_map_stays_a_node() {
    let db = people();
    assert_eq!(
        row(
            &db,
            "MATCH (n:City) WITH n, {outer: {inner: n}} AS x \
             RETURN x, x.outer.inner.name AS name, {k: n}.k.name AS direct"
        ),
        [
            format!("{{outer: {{inner: {AMSTERDAM}}}}}"),
            "'Amsterdam'".into(),
            "'Amsterdam'".into()
        ]
    );
}

/// `collect` of maps keeps the nodes in them, also the first one (`head`),
/// as LDBC SNB IC7 reads it.
#[test]
fn collect_of_maps_keeps_their_nodes() {
    let db = people();
    assert_eq!(
        row(
            &db,
            "MATCH (n:Person {name: 'Alix'}) WITH collect({p: n, t: 3}) AS xs \
             RETURN xs, head(xs).p.name AS name, head(xs).t AS t, xs[0].p AS first"
        ),
        [
            format!("[{{p: {ALIX}, t: 3}}]"),
            "'Alix'".into(),
            "3".into(),
            ALIX.into()
        ]
    );
}

/// The IC7 query shape: the latest like per liker as a map of the message
/// and the time, read after grouping.
#[test]
fn the_head_of_collected_maps_per_group_reads_its_node() {
    let db = people();
    let query = "MATCH (a:Person)-[k]->(b) WITH a, b, k.w AS w ORDER BY w DESC \
                 WITH a, head(collect({msg: b, w: w})) AS latest \
                 RETURN a.name AS a, latest.msg.name AS msg, latest.w AS w, latest.msg AS node \
                 ORDER BY a";
    assert_eq!(
        rows_in(&db, "cypher", query),
        [
            vec!["'Alix'".to_string(), "'Gus'".into(), "3".into(), GUS.into()],
            vec![
                "'Gus'".to_string(),
                "'Amsterdam'".into(),
                "19".into(),
                AMSTERDAM.into()
            ],
        ]
    );
}

/// Grouping on a map keeps its node, in the group key and after WITH.
#[test]
fn grouping_on_a_map_keeps_its_node() {
    let db = people();
    assert_eq!(
        rows(
            &db,
            "MATCH (a:Person)-[]->(b) RETURN {p: a} AS x, count(b) AS c ORDER BY c"
        ),
        [
            vec![format!("{{p: {ALIX}}}"), "1".into()],
            vec![format!("{{p: {GUS}}}"), "1".into()],
        ]
    );
    assert_eq!(
        rows(
            &db,
            "MATCH (a:Person)-[]->(b) WITH {p: a} AS x, count(b) AS c \
             RETURN x.p.name AS name, c ORDER BY name"
        ),
        [
            vec!["'Alix'".to_string(), "1".into()],
            vec!["'Gus'".to_string(), "1".into()],
        ]
    );
}

/// UNWIND of a collected list of maps gives maps whose nodes stay nodes.
#[test]
fn unwind_of_collected_maps_keeps_their_nodes() {
    let db = people();
    assert_eq!(
        rows(
            &db,
            "MATCH (n:Person) WITH collect({p: n}) AS xs UNWIND xs AS x \
             RETURN x, x.p.name AS name ORDER BY name"
        ),
        [
            vec![format!("{{p: {ALIX}}}"), "'Alix'".into()],
            vec![format!("{{p: {GUS}}}"), "'Gus'".into()],
        ]
    );
}

// ---------------------------------------------------------------------------
// startNode and endNode
// ---------------------------------------------------------------------------

#[test]
fn startnode_and_endnode_return_nodes() {
    let db = people();
    assert_eq!(
        row(
            &db,
            "MATCH ()-[r:KNOWS]->() RETURN startNode(r) AS s, endNode(r) AS e, \
             startNode(r).name AS sn, endNode(r).age AS ea"
        ),
        [ALIX.to_string(), GUS.into(), "'Alix'".into(), "88".into()]
    );
    assert_eq!(
        row(
            &db,
            "MATCH ()-[r:LIVES_IN]->() WITH endNode(r) AS e RETURN e, e.name AS name"
        ),
        [AMSTERDAM.to_string(), "'Amsterdam'".into()]
    );
}

// ---------------------------------------------------------------------------
// Whole paths
// ---------------------------------------------------------------------------

/// `RETURN p` gives the path's nodes and edges, the same values `nodes(p)`
/// and `relationships(p)` give.
#[test]
fn a_returned_path_holds_its_nodes_and_edges() {
    let db = people();
    let path = format!("<{ALIX}, {KNOWS}, {GUS}>");
    assert_eq!(
        row(
            &db,
            "MATCH p = (:Person {name: 'Alix'})-[:KNOWS]->() \
             RETURN p, nodes(p) AS n, relationships(p) AS r"
        ),
        [
            path.clone(),
            format!("[{ALIX}, {GUS}]"),
            format!("[{KNOWS}]")
        ]
    );
    let two_hops = format!("<{ALIX}, {KNOWS}, {GUS}, {LIVES_IN}, {AMSTERDAM}>");
    assert_eq!(
        row(
            &db,
            "MATCH p = (:Person {name: 'Alix'})-[:KNOWS]->()-[:LIVES_IN]->() RETURN p"
        ),
        std::slice::from_ref(&two_hops)
    );
    assert_eq!(
        rows_in(
            &db,
            "cypher",
            "MATCH p = (:Person {name: 'Alix'})-[*1..2]->() RETURN p ORDER BY length(p)"
        ),
        [vec![path.clone()], vec![two_hops.clone()]]
    );
    assert_eq!(
        rows_in(
            &db,
            "gql",
            "MATCH p = (:Person {name: 'Alix'})-[]->{1,2}() RETURN p ORDER BY length(p)"
        ),
        [vec![path.clone()], vec![two_hops.clone()]]
    );
    assert_eq!(
        rows_in(
            &db,
            "cypher",
            "MATCH p = shortestPath((:Person {name: 'Alix'})-[*]->(:City)) RETURN p"
        ),
        [vec![two_hops.clone()]]
    );
    assert_eq!(
        rows_in(
            &db,
            "gql",
            "MATCH p = ANY SHORTEST (:Person {name: 'Alix'})-[]->*(:City) RETURN p"
        ),
        [vec![two_hops]]
    );
}

/// A path keeps its nodes and edges through WITH, `collect`, UNWIND and
/// grouping (a grouped path used to come back as the text `Path(2 nodes, 1
/// edges)`).
#[test]
fn a_path_keeps_its_nodes_and_edges_through_with_and_collect() {
    let db = people();
    let path = format!("<{ALIX}, {KNOWS}, {GUS}>");
    assert_eq!(
        row(
            &db,
            "MATCH p = (:Person {name: 'Alix'})-[:KNOWS]->() WITH p AS q, collect(p) AS ps \
             RETURN q, ps, [q] AS l, {path: q} AS m"
        ),
        [
            path.clone(),
            format!("[{path}]"),
            format!("[{path}]"),
            format!("{{path: {path}}}")
        ]
    );
    assert_eq!(
        row(
            &db,
            "MATCH p = (:Person {name: 'Alix'})-[:KNOWS]->() WITH collect(p) AS ps \
             UNWIND ps AS u RETURN u"
        ),
        std::slice::from_ref(&path)
    );
    assert_eq!(
        row(
            &db,
            "MATCH p = (:Person {name: 'Alix'})-[:KNOWS]->() RETURN p, count(*) AS c"
        ),
        [path, "1".into()]
    );
}

/// The functions that read a node or an edge read one taken from a list, a
/// map or `startNode`, as they read a pattern variable.
#[test]
fn functions_read_a_node_or_edge_taken_from_a_value() {
    let db = people();
    assert_eq!(
        row(
            &db,
            "MATCH (a:Person {name: 'Alix'})-[r:KNOWS]->(b) WITH r, [r] AS rs, {e: r, n: b} AS x \
             RETURN labels(startNode(r)) AS l, type(head(rs)) AS t, type(x.e) AS te, \
             labels(x.n) AS ln, endNode(rs[0]).name AS e"
        ),
        ["['Person']", "'KNOWS'", "'KNOWS'", "['Person']", "'Gus'"]
    );
    assert_eq!(
        rows_in(
            &db,
            "cypher",
            "MATCH p = (:Person {name: 'Alix'})-[*2]->() \
             WITH p, collect(startNode(last(relationships(p)))) AS mids \
             RETURN nodes(p)[0].name AS first, [n IN nodes(p) | n.name] AS names, \
             mids[0].name AS mid, head(mids) AS node, last(nodes(p)).name AS target"
        ),
        [[
            "'Alix'".to_string(),
            "['Alix', 'Gus', 'Amsterdam']".into(),
            "'Gus'".into(),
            GUS.into(),
            "'Amsterdam'".into()
        ]]
    );
}

/// A key read of an aggregate in the same clause (`head(collect(n)).name`)
/// is an error that says how to write it, never a null per row: the
/// aggregate inside it would not be computed.
#[test]
fn a_key_read_of_an_aggregate_in_the_same_clause_is_an_error() {
    let db = people();
    for language in ["gql", "cypher"] {
        for query in [
            "MATCH (n:Person) RETURN head(collect(n)).name AS first",
            "MATCH (n:Person) RETURN collect(n)[0].name AS first",
        ] {
            let result = match language {
                "gql" => db.execute(query),
                _ => db.execute_cypher(query),
            };
            let error = result.expect_err(query).to_string();
            assert!(
                error.contains("reads an aggregate, so .name cannot read from it"),
                "{language}: {query}: {error}"
            );
        }
    }
    assert_eq!(
        row(
            &db,
            "MATCH (n:Person {name: 'Gus'}) WITH collect(n) AS ns RETURN head(ns).name AS first"
        ),
        ["'Gus'"]
    );
}

// ---------------------------------------------------------------------------
// size() of a string
// ---------------------------------------------------------------------------

/// `size` of a string counts its characters, as `char_length` does
/// (openCypher `size()`, ISO/IEC 39075 20.22 CHAR_LENGTH), not its UTF-8
/// bytes; `octet_length` counts bytes. A negative index into a string
/// counts characters from the end.
#[test]
fn size_of_a_string_counts_characters() {
    let db = people();
    assert_eq!(
        row(
            &db,
            "RETURN size('\u{1F337}') AS tulip, size('Amsterdam') AS city, size('Praha \u{10D}') AS c, \
             char_length('\u{1F337}') AS chars, octet_length('\u{1F337}') AS bytes, \
             '\u{1F337}b'[-1] AS last"
        ),
        ["1", "9", "7", "1", "4", "'b'"]
    );
}
