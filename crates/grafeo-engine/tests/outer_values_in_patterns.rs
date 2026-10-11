//! A value of an earlier clause read inside a pattern is the value of the row
//! the pattern goes on from, in a property map or a WHERE, in every part of a
//! comma-separated MATCH and in an OPTIONAL MATCH. An OPTIONAL MATCH keeps every
//! row it gets, with nulls where nothing matches (ISO GQL, openCypher).
//!
//! Each query runs in GQL and Cypher, on a graph with and without a property
//! index on `id`.
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test outer_values_in_patterns
//! ```

#![cfg(all(feature = "lpg", feature = "gql", feature = "cypher"))]

use grafeo_common::types::Value;
use grafeo_engine::GrafeoDB;

/// People Alix (`id` 3), Gus (19) and Vincent (88); Alix KNOWS Gus (`since`
/// 3), Gus KNOWS Vincent (`since` 3). Forum Amsterdam (`id` 3) contains posts
/// 19 and 88 by Gus; forum Berlin (`id` 19) contains post 3 by Vincent; forum
/// Paris (`id` 88) is empty. Post 19 has the tags Prague (`id` 3, `w` 3) and
/// Barcelona (`id` 19, `w` 19), post 88 has Prague (`w` 88). Posts 19 and 3
/// are located in Amsterdam (the place, `id` 3), post 88 in Berlin (`id` 19).
fn graph(indexed: bool) -> GrafeoDB {
    let db = GrafeoDB::new_in_memory();
    if indexed {
        db.create_property_index("id").unwrap();
    }
    db.execute(
        "INSERT (alix:Person {id: 3, name: 'Alix'}), (gus:Person {id: 19, name: 'Gus'}), \
         (vincent:Person {id: 88, name: 'Vincent'}), \
         (alix)-[:KNOWS {since: 3}]->(gus), (gus)-[:KNOWS {since: 3}]->(vincent), \
         (ams:Forum {id: 3, title: 'Amsterdam'}), (ber:Forum {id: 19, title: 'Berlin'}), \
         (par:Forum {id: 88, title: 'Paris'}), \
         (p19:Post {id: 19}), (p88:Post {id: 88}), (p3:Post {id: 3}), \
         (ams)-[:CONTAINER_OF]->(p19), (ams)-[:CONTAINER_OF]->(p88), \
         (ber)-[:CONTAINER_OF]->(p3), \
         (p19)-[:HAS_CREATOR]->(gus), (p88)-[:HAS_CREATOR]->(gus), \
         (p3)-[:HAS_CREATOR]->(vincent), \
         (prague:Tag {id: 3, name: 'Prague'}), (barcelona:Tag {id: 19, name: 'Barcelona'}), \
         (p19)-[:HAS_TAG {w: 3}]->(prague), (p19)-[:HAS_TAG {w: 19}]->(barcelona), \
         (p88)-[:HAS_TAG {w: 88}]->(prague), \
         (amsterdam:Place {id: 3, name: 'Amsterdam'}), (berlin:Place {id: 19, name: 'Berlin'}), \
         (p19)-[:IS_LOCATED_IN]->(amsterdam), (p3)-[:IS_LOCATED_IN]->(amsterdam), \
         (p88)-[:IS_LOCATED_IN]->(berlin)",
    )
    .unwrap();
    db
}

/// One cell as text.
fn cell(value: &Value) -> String {
    match value {
        Value::String(text) => text.to_string(),
        Value::Int64(number) => number.to_string(),
        Value::Null => "null".to_string(),
        other => format!("{other:?}"),
    }
}

/// The rows of `query` as text, sorted.
fn rows(db: &GrafeoDB, cypher: bool, query: &str) -> Vec<Vec<String>> {
    let result = if cypher {
        db.execute_cypher(query)
    } else {
        db.execute(query)
    };
    let language = if cypher { "Cypher" } else { "GQL" };
    let mut rows: Vec<Vec<String>> = result
        .unwrap_or_else(|error| panic!("{language} `{query}` failed: {error}"))
        .rows()
        .iter()
        .map(|row| row.iter().map(cell).collect())
        .collect();
    rows.sort();
    rows
}

/// Literal text rows, sorted.
fn expected(want: &[&[&str]]) -> Vec<Vec<String>> {
    let mut rows: Vec<Vec<String>> = want
        .iter()
        .map(|row| row.iter().map(|text| (*text).to_string()).collect())
        .collect();
    rows.sort();
    rows
}

/// Checks `query` in GQL against `want`, without and with the index.
fn assert_gql(query: &str, want: &[&[&str]]) {
    for indexed in [false, true] {
        assert_eq!(
            rows(&graph(indexed), false, query),
            expected(want),
            "GQL (index: {indexed}): {query}"
        );
    }
}

/// Checks `query` in Cypher against `want`, without and with the index.
fn assert_cypher(query: &str, want: &[&[&str]]) {
    for indexed in [false, true] {
        assert_eq!(
            rows(&graph(indexed), true, query),
            expected(want),
            "Cypher (index: {indexed}): {query}"
        );
    }
}

/// Checks `query` in GQL and Cypher against `want`.
fn assert_both(query: &str, want: &[&[&str]]) {
    assert_gql(query, want);
    assert_cypher(query, want);
}

// ---------------------------------------------------------------------------
// OPTIONAL MATCH: a property map that reads an earlier value
// ---------------------------------------------------------------------------

/// The property map reads the unwound value of its row: each row gets the
/// person with that `id`, and a value without a person gets null.
#[test]
fn an_optional_property_map_reads_the_unwound_value() {
    let want: &[&[&str]] = &[&["19", "Gus"], &["3", "Alix"], &["319", "null"]];
    assert_both(
        "UNWIND [3, 19, 319] AS x OPTIONAL MATCH (p:Person {id: x}) RETURN x, p.name",
        want,
    );
    // Without a label: only people have a name, every kind of node has an id
    assert_both(
        "UNWIND ['Alix', 'Gus', 'Mia'] AS n OPTIONAL MATCH (p {name: n}) RETURN n, p.id",
        &[&["Alix", "3"], &["Gus", "19"], &["Mia", "null"]],
    );
    assert_both(
        "UNWIND [88, 319] AS x OPTIONAL MATCH (p {id: x}) RETURN x, count(p)",
        &[&["319", "0"], &["88", "3"]],
    );
    // An expression of the value
    assert_both(
        "UNWIND [0, 16, 316] AS x OPTIONAL MATCH (p:Person {id: x + 3}) RETURN x + 3, p.name",
        want,
    );
}

/// The value is a property of a node an earlier MATCH bound.
#[test]
fn an_optional_property_map_reads_a_property_of_an_earlier_node() {
    assert_both(
        "MATCH (f:Forum) OPTIONAL MATCH (p:Person {id: f.id}) RETURN f.title, p.name",
        &[
            &["Amsterdam", "Alix"],
            &["Berlin", "Gus"],
            &["Paris", "Vincent"],
        ],
    );
    // On an edge: only Alix knows someone since her own id
    assert_both(
        "MATCH (a:Person) OPTIONAL MATCH (a)-[:KNOWS {since: a.id}]->(b) RETURN a.name, b.name",
        &[&["Alix", "Gus"], &["Gus", "null"], &["Vincent", "null"]],
    );
}

// ---------------------------------------------------------------------------
// OPTIONAL MATCH: a WHERE that reads an earlier value
// ---------------------------------------------------------------------------

/// A WHERE on a list from the WITH before it keeps the rows without a match,
/// with a count of 0.
#[test]
fn an_optional_where_on_an_earlier_list_keeps_every_row() {
    let query = "MATCH (f:Forum) WITH f, [19] AS xs \
                 OPTIONAL MATCH (p:Person)<-[:HAS_CREATOR]-(post)<-[:CONTAINER_OF]-(f) \
                 WHERE p.id IN xs RETURN f.title, count(post)";
    let want: &[&[&str]] = &[&["Amsterdam", "2"], &["Berlin", "0"], &["Paris", "0"]];
    assert_cypher(query, want);
    // No forum has a post by Alix
    assert_cypher(
        &query.replace("[19]", "[3]"),
        &[&["Amsterdam", "0"], &["Berlin", "0"], &["Paris", "0"]],
    );
}

/// A WHERE on a list of nodes from the WITH before it (LDBC IC5).
#[test]
fn an_optional_where_on_an_earlier_list_of_nodes_keeps_every_row() {
    let want: &[&[&str]] = &[&["Amsterdam", "2"], &["Berlin", "0"], &["Paris", "0"]];
    assert_cypher(
        "MATCH (f:Forum) MATCH (g:Person {name: 'Gus'}) WITH f, collect(g) AS friends \
         OPTIONAL MATCH (friend)<-[:HAS_CREATOR]-(post)<-[:CONTAINER_OF]-(f) \
         WHERE friend IN friends RETURN f.title, count(post)",
        want,
    );
    // GQL: the condition inside the element pattern
    assert_gql(
        "MATCH (f:Forum) MATCH (g:Person {name: 'Gus'}) WITH f, collect(g) AS friends \
         OPTIONAL MATCH (friend WHERE friend IN friends)<-[:HAS_CREATOR]-(post)<-[:CONTAINER_OF]-(f) \
         RETURN f.title, count(post)",
        want,
    );
}

/// An optional shortest path whose end node's property map reads the unwound
/// value: Alix reaches Vincent in two hops, Gus in one, nobody has id 319.
#[test]
fn an_optional_shortest_path_reads_the_unwound_value() {
    let want: &[&[&str]] = &[&["19", "1"], &["3", "2"], &["319", "null"]];
    assert_gql(
        "UNWIND [3, 19, 319] AS x \
         OPTIONAL MATCH p = ANY SHORTEST (a:Person {id: x})-[:KNOWS]->+(b:Person {id: 88}) \
         RETURN x, length(p)",
        want,
    );
    assert_cypher(
        "UNWIND [3, 19, 319] AS x \
         OPTIONAL MATCH p = shortestPath((a:Person {id: x})-[:KNOWS*]->(b:Person {id: 88})) \
         RETURN x, length(p)",
        want,
    );
}

/// The OPTIONAL MATCH names a variable that a clause before it bound and the
/// WITH between them dropped (LDBC IC5: `WITH forum, collect(friend) AS
/// friends OPTIONAL MATCH (friend)<-...`): it is a new variable of the
/// optional part, so its WHERE decides which matches count and every row of
/// the WITH stays.
#[test]
fn an_optional_where_may_read_a_name_the_with_before_it_dropped() {
    assert_cypher(
        "MATCH (forum:Forum) MATCH (friend:Person {name: 'Gus'}) \
         WITH forum, collect(friend) AS friends \
         OPTIONAL MATCH (friend)<-[:HAS_CREATOR]-(post)<-[:CONTAINER_OF]-(forum) \
         WHERE friend IN friends RETURN forum.title, count(post)",
        &[&["Amsterdam", "2"], &["Berlin", "0"], &["Paris", "0"]],
    );
}

/// GQL: the condition on the reused name inside its element pattern.
#[test]
fn an_optional_element_where_may_read_a_name_the_with_before_it_dropped() {
    assert_gql(
        "MATCH (forum:Forum) MATCH (friend:Person {name: 'Gus'}) \
         WITH forum, collect(friend) AS friends \
         OPTIONAL MATCH (friend WHERE friend IN friends)<-[:HAS_CREATOR]-(post)\
         <-[:CONTAINER_OF]-(forum) RETURN forum.title, count(post)",
        &[&["Amsterdam", "2"], &["Berlin", "0"], &["Paris", "0"]],
    );
}

/// A condition on the reused name alone, after an aggregate: the name is the
/// optional part's, so the condition chooses its matches.
#[test]
fn an_optional_condition_on_a_name_an_aggregate_dropped_keeps_every_row() {
    let want: &[&[&str]] = &[
        &["Amsterdam", "3", "2"],
        &["Berlin", "3", "0"],
        &["Paris", "3", "0"],
    ];
    assert_gql(
        "MATCH (f:Forum), (p:Person) WITH f, count(p) AS people \
         OPTIONAL MATCH (f)-[:CONTAINER_OF]->(post)-[:HAS_CREATOR]->(p WHERE p.id = 19) \
         RETURN f.title, people, count(post)",
        want,
    );
    assert_cypher(
        "MATCH (f:Forum), (p:Person) WITH f, count(p) AS people \
         OPTIONAL MATCH (f)-[:CONTAINER_OF]->(post)-[:HAS_CREATOR]->(p) WHERE p.id = 19 \
         RETURN f.title, people, count(post)",
        want,
    );
}

/// The same after a WITH DISTINCT that drops the name.
#[test]
fn an_optional_condition_on_a_name_a_distinct_dropped_keeps_every_row() {
    let want: &[&[&str]] = &[&["Amsterdam", "2"], &["Berlin", "0"], &["Paris", "0"]];
    assert_gql(
        "MATCH (f:Forum), (p:Person) WITH DISTINCT f \
         OPTIONAL MATCH (f)-[:CONTAINER_OF]->(post)-[:HAS_CREATOR]->(p WHERE p.id = 19) \
         RETURN f.title, count(post)",
        want,
    );
    assert_cypher(
        "MATCH (f:Forum), (p:Person) WITH DISTINCT f \
         OPTIONAL MATCH (f)-[:CONTAINER_OF]->(post)-[:HAS_CREATOR]->(p) WHERE p.id = 19 \
         RETURN f.title, count(post)",
        want,
    );
}

/// Inside a CALL subquery the OPTIONAL MATCH reads the imported value, named
/// by an importing `WITH` or a scope clause or imported with `*`, in its
/// property map and in its WHERE.
#[test]
fn an_optional_match_in_a_subquery_reads_the_imported_value() {
    let want: &[&[&str]] = &[&["19", "Gus"], &["3", "Alix"], &["319", "null"]];
    for call in ["CALL { WITH x", "CALL { WITH *", "CALL (x) {"] {
        assert_both(
            &format!(
                "UNWIND [3, 19, 319] AS x {call} \
                 OPTIONAL MATCH (p:Person {{id: x}}) RETURN p.name AS name }} RETURN x, name"
            ),
            want,
        );
        assert_cypher(
            &format!(
                "UNWIND [3, 19, 319] AS x {call} \
                 OPTIONAL MATCH (p:Person) WHERE p.id = x RETURN p.name AS name }} RETURN x, name"
            ),
            want,
        );
        assert_gql(
            &format!(
                "UNWIND [3, 19, 319] AS x {call} \
                 OPTIONAL MATCH (p:Person WHERE p.id = x) RETURN p.name AS name }} RETURN x, name"
            ),
            want,
        );
    }
}

/// The WHERE of an OPTIONAL MATCH is part of its pattern: a condition on an
/// earlier node that the optional part also names decides which matches
/// count, and never removes a row of the clauses before it.
#[test]
fn an_optional_where_on_a_shared_earlier_node_keeps_every_row() {
    assert_both(
        "MATCH (f:Forum) OPTIONAL MATCH (f)-[:CONTAINER_OF]->(post) WHERE f.id = 19 \
         RETURN f.title, count(post)",
        &[&["Amsterdam", "0"], &["Berlin", "1"], &["Paris", "0"]],
    );
    assert_both(
        "MATCH (f:Forum) OPTIONAL MATCH (f)-[:CONTAINER_OF]->(post)-[:HAS_CREATOR]->(p) \
         WHERE p.name = 'Gus' AND f.id = 3 RETURN f.title, count(post)",
        &[&["Amsterdam", "2"], &["Berlin", "0"], &["Paris", "0"]],
    );
}

/// A condition only on earlier values (a node the optional part does not
/// name, an unwound value) keeps the row too, with nulls where it fails.
#[test]
fn an_optional_where_only_on_earlier_values_keeps_every_row() {
    assert_both(
        "MATCH (f:Forum) OPTIONAL MATCH (t:Tag) WHERE f.id = 3 RETURN f.title, count(t)",
        &[&["Amsterdam", "2"], &["Berlin", "0"], &["Paris", "0"]],
    );
    assert_both(
        "UNWIND [3, 19] AS x OPTIONAL MATCH (t:Tag {id: 3}) WHERE x = 3 RETURN x, t.name",
        &[&["19", "null"], &["3", "Prague"]],
    );
}

/// A condition that reads no variable is part of the optional pattern too.
#[test]
fn an_optional_where_without_variables_keeps_every_row() {
    assert_both(
        "MATCH (f:Forum) OPTIONAL MATCH (f)-[:CONTAINER_OF]->(post) WHERE 3 = 19 \
         RETURN f.title, count(post)",
        &[&["Amsterdam", "0"], &["Berlin", "0"], &["Paris", "0"]],
    );
    assert_both(
        "OPTIONAL MATCH (t:Tag) WHERE 3 = 19 RETURN t.name",
        &[&["null"]],
    );
}

/// GQL: FILTER is a statement of its own, not the WHERE of the OPTIONAL MATCH
/// before it: it filters every row, on earlier values and on the optional
/// part's alike.
#[test]
fn a_filter_after_an_optional_match_filters_the_rows() {
    assert_gql(
        "MATCH (f:Forum) OPTIONAL MATCH (f)-[:CONTAINER_OF]->(post) FILTER f.id = 19 \
         RETURN f.title, count(post)",
        &[&["Berlin", "1"]],
    );
    assert_gql(
        "MATCH (f:Forum) OPTIONAL MATCH (f)-[:CONTAINER_OF]->(post) FILTER post.id = 3 \
         RETURN f.title, post.id",
        &[&["Berlin", "3"]],
    );
}

/// A condition after the OPTIONAL MATCH, in a WITH (or GQL's FILTER), filters
/// the rows: the forums without a post.
#[test]
fn a_where_after_a_with_filters_the_optional_rows() {
    let want: &[&[&str]] = &[&["Paris"]];
    assert_cypher(
        "MATCH (f:Forum) OPTIONAL MATCH (f)-[:CONTAINER_OF]->(post) \
         WITH f, post WHERE post IS NULL RETURN f.title",
        want,
    );
    assert_cypher(
        "MATCH (f:Forum) OPTIONAL MATCH (f)-[:CONTAINER_OF]->(post) \
         WITH * WHERE post IS NULL RETURN f.title",
        want,
    );
    assert_gql(
        "MATCH (f:Forum) OPTIONAL MATCH (f)-[:CONTAINER_OF]->(post) \
         FILTER post IS NULL RETURN f.title",
        want,
    );
}

/// GQL: a WHERE after a MATCH whose last edge is questioned (`->?`) is the
/// MATCH's own and filters its rows: the left join of the questioned edge is
/// no OPTIONAL MATCH.
#[test]
fn a_where_after_a_questioned_edge_filters_the_rows() {
    assert_gql(
        "MATCH (a:Person)-[:KNOWS]->?(b) WHERE a.id = 3 RETURN a.name, b.name",
        &[&["Alix", "Gus"]],
    );
    assert_gql(
        "MATCH (a:Person)-[:KNOWS]->?(b) WHERE b.id = 19 RETURN a.name, b.name",
        &[&["Alix", "Gus"]],
    );
}

/// A WHERE that compares with an earlier node and an earlier scalar.
#[test]
fn an_optional_where_on_an_earlier_node_or_scalar_keeps_every_row() {
    assert_both(
        "MATCH (f:Forum), (g:Person) \
         OPTIONAL MATCH (p)<-[:HAS_CREATOR]-(post)<-[:CONTAINER_OF]-(f) WHERE p = g \
         RETURN f.title, g.name, count(post)",
        &[
            &["Amsterdam", "Alix", "0"],
            &["Amsterdam", "Gus", "2"],
            &["Amsterdam", "Vincent", "0"],
            &["Berlin", "Alix", "0"],
            &["Berlin", "Gus", "0"],
            &["Berlin", "Vincent", "1"],
            &["Paris", "Alix", "0"],
            &["Paris", "Gus", "0"],
            &["Paris", "Vincent", "0"],
        ],
    );
    assert_both(
        "UNWIND [3, 19, 319] AS x OPTIONAL MATCH (p:Person) WHERE p.id = x RETURN x, p.name",
        &[&["19", "Gus"], &["3", "Alix"], &["319", "null"]],
    );
}

// ---------------------------------------------------------------------------
// Comma-separated MATCH: a later part reads an earlier value
// ---------------------------------------------------------------------------

/// The second part's property map reads the unwound value.
#[test]
fn a_second_part_property_map_reads_the_unwound_value() {
    assert_both(
        "UNWIND [3] AS x MATCH (p:Person {id: x}), (p)-[:KNOWS]-(f:Person {id: x + 16}) \
         RETURN f.name",
        &[&["Gus"]],
    );
}

/// The second part's property map reads a value of the WITH before it, on a
/// node and on an edge, and so does a third part.
#[test]
fn a_later_part_property_map_reads_a_with_value() {
    assert_both(
        "MATCH (k:Tag {name: 'Barcelona'}) WITH k.id AS tid \
         MATCH (post:Post {id: 19}), (post)-[:HAS_TAG]->(t:Tag {id: tid}) RETURN t.name",
        &[&["Barcelona"]],
    );
    assert_both(
        "MATCH (k:Tag {name: 'Barcelona'}) WITH k.id AS tid \
         MATCH (post:Post {id: 19}), (post)-[:HAS_TAG {w: tid}]->(t) RETURN t.name",
        &[&["Barcelona"]],
    );
    // A third part (LDBC IC6); GQL puts the condition in the element pattern
    assert_cypher(
        "MATCH (k:Tag {name: 'Barcelona'}) WITH k.id AS tid \
         MATCH (f:Person)<-[:HAS_CREATOR]-(post:Post), (post)-[:HAS_TAG]->(t:Tag {id: tid}), \
         (post)-[:HAS_TAG]->(other:Tag) WHERE NOT t = other RETURN f.name, other.name",
        &[&["Gus", "Prague"]],
    );
    assert_gql(
        "MATCH (k:Tag {name: 'Barcelona'}) WITH k.id AS tid \
         MATCH (f:Person)<-[:HAS_CREATOR]-(post:Post), (post)-[:HAS_TAG]->(t:Tag {id: tid}), \
         (post)-[:HAS_TAG]->(other:Tag WHERE NOT t = other) RETURN f.name, other.name",
        &[&["Gus", "Prague"]],
    );
    // The value in the third part. Cypher binds a relationship once per
    // MATCH, so `other` is reached over another HAS_TAG than `t`; GQL lets the
    // two patterns bind one edge
    let third = "MATCH (k:Tag {name: 'Barcelona'}) WITH k.id AS tid \
                 MATCH (f:Person)<-[:HAS_CREATOR]-(post:Post), (post)-[:HAS_TAG]->(other:Tag), \
                 (post)-[:HAS_TAG {w: tid}]->(t:Tag) RETURN f.name, other.name";
    assert_gql(third, &[&["Gus", "Barcelona"], &["Gus", "Prague"]]);
    assert_cypher(third, &[&["Gus", "Prague"]]);
}

/// A WHERE that reads only the second part and earlier values.
#[test]
fn a_where_on_a_second_part_and_earlier_values_keeps_its_rows() {
    assert_cypher(
        "MATCH (x:Place {id: 3}), (y:Place {id: 19}), (friend:Person {name: 'Gus'}) \
         WITH friend, x, y \
         MATCH (friend)<-[:HAS_CREATOR]-(message), (message)-[:IS_LOCATED_IN]->(country) \
         WHERE country IN [x, y] RETURN friend.name, message.id, country.name",
        &[&["Gus", "19", "Amsterdam"], &["Gus", "88", "Berlin"]],
    );
}
