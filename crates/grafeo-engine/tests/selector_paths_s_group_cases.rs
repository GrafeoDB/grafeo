//! More cases for the S fixes of the selector-gap plan (0.6.1), each by its
//! rows (the companion of selector_paths_s_group.rs):
//!
//! - 2a: `labels()` and `type()` of expressions (`head(...)`, a map key),
//!   sorted labels, null for an entity deleted earlier in the statement;
//! - 2b: nodes and edges with more properties than fit inline, every key
//!   and value now, at a past epoch and inside a transaction;
//! - 3a: edge checks with several types, a type the store does not have, an
//!   undirected self-loop counted once, a bound end, COUNT and a
//!   variable-length check with a type;
//! - 1a0: each rule that keeps the label scan, rows identical to the
//!   reference in GQL and Cypher, at a past epoch and in a transaction, a
//!   union label no node has;
//! - 2d: the count read when the query runs, count(n) and two counts, the
//!   regular path for count(DISTINCT n) and with other aggregates, labels
//!   with no node and labels the store does not know.
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test selector_paths_s_group_cases
//! ```

#![cfg(all(feature = "gql", feature = "cypher"))]

use std::collections::BTreeMap;

use grafeo_common::types::{PropertyKey, Value};
use grafeo_engine::{Config, GrafeoDB, HistoryRetention};

const LANGUAGES: [&str; 2] = ["gql", "cypher"];

fn rows_in(db: &GrafeoDB, language: &str, query: &str) -> Vec<Vec<Value>> {
    db.execute_language(query, language, None)
        .unwrap_or_else(|error| panic!("{language} `{query}`: {error}"))
        .rows()
        .to_vec()
}

fn rows(db: &GrafeoDB, query: &str) -> Vec<Vec<Value>> {
    rows_in(db, "gql", query)
}

fn plan(db: &GrafeoDB, keyword: &str, query: &str) -> String {
    match &rows(db, &format!("{keyword} {query}"))[0][0] {
        Value::String(text) => text.to_string(),
        other => panic!("{keyword} {query}: {other:?}"),
    }
}

fn history() -> GrafeoDB {
    GrafeoDB::with_config(Config::in_memory().with_history_retention(HistoryRetention::Forever))
        .unwrap()
}

fn texts(names: &[&str]) -> Value {
    Value::List(names.iter().map(|name| Value::from(*name)).collect())
}

fn int(value: i64) -> Vec<Vec<Value>> {
    vec![vec![Value::Int64(value)]]
}

// ---------------------------------------------------------------------------
// 2a
// ---------------------------------------------------------------------------

#[test]
fn labels_and_type_of_expressions() {
    let db = GrafeoDB::new_in_memory();
    db.execute("INSERT (:Zeta:Alpha:Mid {id: 'a'})-[:USES {w: 1}]->(:Graph {id: 'b'})")
        .unwrap();
    assert_eq!(
        rows(&db, "MATCH (n {id: 'a'}) RETURN labels(n) AS l"),
        [[texts(&["Alpha", "Mid", "Zeta"])]],
        "sorted by name, not as written"
    );
    for language in LANGUAGES {
        assert_eq!(
            rows_in(
                &db,
                language,
                "MATCH (n {id: 'a'}) WITH collect(n) AS ns RETURN labels(head(ns)) AS l"
            ),
            [[texts(&["Alpha", "Mid", "Zeta"])]],
            "{language}"
        );
        assert_eq!(
            rows_in(
                &db,
                language,
                "MATCH ()-[r]->() WITH collect(r) AS rs RETURN type(head(rs)) AS t"
            ),
            [[Value::from("USES")]],
            "{language}"
        );
        assert_eq!(
            rows_in(
                &db,
                language,
                "MATCH ()-[r:USES]->() WITH {e: r} AS x RETURN type(x.e) AS t"
            ),
            [[Value::from("USES")]],
            "{language}"
        );
    }
}

#[test]
fn labels_and_type_of_an_entity_deleted_in_the_statement_are_null() {
    let db = GrafeoDB::new_in_memory();
    db.execute("INSERT (:Graph:File {id: 'a'})-[:USES]->(:Graph {id: 'b'})")
        .unwrap();
    assert_eq!(
        rows(
            &db,
            "MATCH (n {id: 'a'})-[r]->() WITH n, r, n AS m, r AS e DETACH DELETE n RETURN labels(m) AS l, type(e) AS t"
        ),
        [[Value::Null, Value::Null]]
    );
}

// ---------------------------------------------------------------------------
// 2b
// ---------------------------------------------------------------------------

/// `RETURN n` (or `r`) as a map of its properties, without the `_` keys.
fn properties_of(value: &Value) -> BTreeMap<String, Value> {
    let Value::Map(map) = value else {
        panic!("an entity map: {value:?}");
    };
    map.iter()
        .filter(|(key, _)| !key.as_str().starts_with('_'))
        .map(|(key, value)| (key.as_str().to_string(), value.clone()))
        .collect()
}

fn wide(count: usize, base: i64) -> (String, BTreeMap<String, Value>) {
    let mut expected = BTreeMap::new();
    let mut parts = Vec::new();
    for i in 0..count {
        let key = format!("k{i:02}");
        let value = base + i64::try_from(i).unwrap();
        parts.push(format!("{key}: {value}"));
        expected.insert(key, Value::Int64(value));
    }
    (parts.join(", "), expected)
}

#[test]
fn wide_nodes_and_edges_come_back_whole_now_then_and_in_a_transaction() {
    let db = history();
    for count in [5, 31] {
        let (node_props, node_expected) = wide(count, 0);
        let (edge_props, edge_expected) = wide(count, 1000);
        db.execute(&format!(
            "INSERT (:Wide{count} {{{node_props}}})-[:LINK{count} {{{edge_props}}}]->(:End)"
        ))
        .unwrap();
        let node = format!("MATCH (n:Wide{count}) RETURN n");
        let edge = format!("MATCH ()-[r:LINK{count}]->() RETURN r");
        assert_eq!(
            properties_of(&rows(&db, &node)[0][0]),
            node_expected,
            "{count}"
        );
        assert_eq!(
            properties_of(&rows(&db, &edge)[0][0]),
            edge_expected,
            "{count}"
        );
        let Value::Map(map) = &rows(&db, &node)[0][0] else {
            panic!("a node map");
        };
        let keys: Vec<&str> = map.keys().map(PropertyKey::as_str).collect();
        let mut sorted = keys.clone();
        sorted.sort_unstable();
        assert_eq!(keys, sorted, "in key order");

        let before = db.current_epoch();
        db.execute(&format!(
            "MATCH (n:Wide{count}) SET n.k00 = -1 REMOVE n.k01"
        ))
        .unwrap();
        db.execute(&format!("MATCH ()-[r:LINK{count}]->() SET r.k00 = -1"))
            .unwrap();
        let then = db.execute_at_epoch(&node, before).unwrap();
        assert_eq!(
            properties_of(&then.rows()[0][0]),
            node_expected,
            "{count} then"
        );
        let then = db.execute_at_epoch(&edge, before).unwrap();
        assert_eq!(
            properties_of(&then.rows()[0][0]),
            edge_expected,
            "{count} edge then"
        );

        let mut session = db.session();
        session.begin_transaction().unwrap();
        session
            .execute(&format!("MATCH (n:Wide{count}) SET n.added = 'x'"))
            .unwrap();
        let inside = session.execute(&node).unwrap();
        let inside = properties_of(&inside.rows()[0][0]);
        assert_eq!(inside.get("added"), Some(&Value::from("x")));
        assert_eq!(inside.get("k00"), Some(&Value::Int64(-1)));
        assert_eq!(inside.get("k01"), None);
        assert_eq!(inside.len(), count, "{count}: one removed, one added");
        let outside = properties_of(&rows(&db, &node)[0][0]);
        assert_eq!(outside.get("added"), None);
        session.rollback().unwrap();
    }
}

// ---------------------------------------------------------------------------
// 3a
// ---------------------------------------------------------------------------

/// A, B and C edges from `n1`, a self-loop L on `n2`, nothing on `n3`; an
/// A chain n1 -> m1 -> m2 -> t (Target).
fn typed_edges() -> GrafeoDB {
    let db = GrafeoDB::new_in_memory();
    db.execute(
        "INSERT (n1:N {id: 1}), (n2:N {id: 2}), (n3:N {id: 3}), (m1:M), (m2:M), (t:Target {id: 9}), \
         (n1)-[:A]->(m1), (n1)-[:A]->(m2), (n1)-[:B]->(m1), (n1)-[:C]->(n2), (n2)-[:L]->(n2), \
         (m1)-[:A]->(m2), (m2)-[:A]->(t)",
    )
    .unwrap();
    db
}

#[test]
fn typed_edge_checks() {
    let db = typed_edges();
    for language in LANGUAGES {
        let ids = |query: &str| -> Vec<Vec<Value>> { rows_in(&db, language, query) };
        assert_eq!(
            ids("MATCH (n:N) WHERE EXISTS { MATCH (n)-[:B|C]->() } RETURN n.id AS id"),
            int(1),
            "{language}: several types"
        );
        assert_eq!(
            ids("MATCH (n:N) WHERE EXISTS { MATCH (n)-[:NOPE]->() } RETURN n.id AS id"),
            Vec::<Vec<Value>>::new(),
            "{language}: a type no edge has"
        );
        assert_eq!(
            ids("MATCH (n:N) WHERE EXISTS { MATCH (n)-[:NOPE|L]->() } RETURN n.id AS id"),
            int(2),
            "{language}: an unknown type beside a known one"
        );
        assert_eq!(
            ids("MATCH (n:N {id: 2}) RETURN COUNT { MATCH (n)-[:L]-() } AS c"),
            int(1),
            "{language}: an undirected self-loop counts once"
        );
        assert_eq!(
            ids("MATCH (n:N {id: 1}) RETURN COUNT { MATCH (n)-[:A]->() } AS c"),
            int(2),
            "{language}: COUNT with a type, among other types"
        );
        assert_eq!(
            ids(
                "MATCH (n:N), (t:Target) WHERE EXISTS { MATCH (n)-[:A*1..3]->(t) } RETURN n.id AS id"
            ),
            int(1),
            "{language}: a variable-length check with a type"
        );
        assert_eq!(
            ids(
                "MATCH (n:N), (t:Target) WHERE EXISTS { MATCH (n)-[:B*1..3]->(t) } RETURN n.id AS id"
            ),
            Vec::<Vec<Value>>::new(),
            "{language}: another type does not reach it"
        );
        assert_eq!(
            ids(
                "MATCH (n:N {id: 1}), (m:M) WHERE EXISTS { MATCH (n)-[:B]->(m) } RETURN count(m) AS c"
            ),
            int(1),
            "{language}: the bound end"
        );
    }
}

// ---------------------------------------------------------------------------
// 1a0
// ---------------------------------------------------------------------------

fn code_graph(db: &GrafeoDB) {
    for (label, count) in [
        ("File", 40),
        ("TypeDefinition", 30),
        ("Method", 300),
        ("Test", 100),
    ] {
        db.execute(&format!(
            "UNWIND range(1, {count}) AS i INSERT (:Graph:{label} {{id: '{label}' + toString(i), x: (i * 37) % 100}})"
        ))
        .unwrap();
    }
}

const SELECTOR: &str = "MATCH (n:Graph) WHERE (n:File OR n:TypeDefinition) AND n.x >= 90 RETURN n.id AS id ORDER BY id";
const REFERENCE: &str = "MATCH (n) WHERE n:Graph AND (n:File OR n:TypeDefinition) AND n.x >= 90 RETURN n.id AS id ORDER BY id";

#[test]
fn an_or_of_labels_gives_the_rows_of_the_reference_everywhere() {
    let db = history();
    code_graph(&db);
    for language in LANGUAGES {
        let expected = rows_in(&db, language, REFERENCE);
        assert_ne!(expected.len(), 0);
        assert_eq!(rows_in(&db, language, SELECTOR), expected, "{language}");
    }
    let before = db.current_epoch();
    let then = rows(&db, REFERENCE);
    db.execute("MATCH (n:File) WHERE n.x >= 90 SET n.x = 0")
        .unwrap();
    db.execute("MATCH (n:Test) WHERE n.x >= 95 SET n:File")
        .unwrap();
    assert_eq!(
        rows(&db, SELECTOR),
        rows(&db, REFERENCE),
        "after the writes"
    );
    assert_eq!(
        db.execute_at_epoch(SELECTOR, before).unwrap().rows(),
        then.as_slice()
    );

    let mut session = db.session();
    session.begin_transaction().unwrap();
    session
        .execute("MATCH (n:Method) WHERE n.x >= 98 SET n:TypeDefinition")
        .unwrap();
    let inside = session.execute(SELECTOR).unwrap().rows().to_vec();
    let inside_reference = session.execute(REFERENCE).unwrap().rows().to_vec();
    assert_eq!(inside, inside_reference, "the transaction's own labels");
    assert_ne!(
        inside,
        rows(&db, SELECTOR),
        "another session does not see them"
    );
    session.rollback().unwrap();
}

#[test]
fn a_union_label_no_node_has() {
    let db = GrafeoDB::new_in_memory();
    code_graph(&db);
    let query = "MATCH (n:Graph) WHERE (n:File OR n:Nope) RETURN count(*) AS c";
    for language in LANGUAGES {
        assert_eq!(rows_in(&db, language, query), int(40), "{language}");
    }
}

/// The label scan stays when an id seek, an indexed equality or a union over
/// more than half of the label answers better; the rows stay right.
#[test]
fn the_label_scan_stays_where_it_answers_better() {
    let db = GrafeoDB::new_in_memory();
    code_graph(&db);
    let wide_union =
        "MATCH (n:Graph) WHERE (n:Method OR n:Test) AND n.x >= 90 RETURN n.id AS id ORDER BY id";
    assert_eq!(
        rows(&db, wide_union),
        rows(
            &db,
            "MATCH (n) WHERE n:Graph AND (n:Method OR n:Test) AND n.x >= 90 RETURN n.id AS id ORDER BY id"
        )
    );
    let text = plan(&db, "EXPLAIN", wide_union);
    assert!(
        text.contains("NodeScan (n:Graph)") && !text.contains("[labels:"),
        "over half of Graph: {text}"
    );

    let file3 = rows(&db, "MATCH (n:File {id: 'File3'}) RETURN id(n) AS i")[0][0].clone();
    let Value::Int64(file3) = file3 else {
        panic!("an id");
    };
    let by_id = format!(
        "MATCH (n:Graph) WHERE (n:File OR n:TypeDefinition) AND id(n) = {file3} RETURN n.id AS id"
    );
    assert_eq!(rows(&db, &by_id), [[Value::from("File3")]]);
    let text = plan(&db, "EXPLAIN", &by_id);
    assert!(!text.contains("[labels:"), "an id seek: {text}");

    db.create_property_index("id").unwrap();
    let by_key =
        "MATCH (n:Graph) WHERE (n:File OR n:TypeDefinition) AND n.id = 'File3' RETURN n.id AS id";
    assert_eq!(rows(&db, by_key), [[Value::from("File3")]]);
    let text = plan(&db, "EXPLAIN", by_key);
    assert!(
        text.contains("[index: id]") && !text.contains("[labels:"),
        "an index seek: {text}"
    );
}

#[cfg(feature = "vector-index")]
#[test]
fn the_label_scan_stays_for_a_vector_index_on_the_label() {
    let db = GrafeoDB::new_in_memory();
    db.execute("INSERT (:Graph:File {id: 'f', emb: vector([1.0, 0.0])}), (:Graph:TypeDefinition {id: 't', emb: vector([0.0, 1.0])}), (:Graph:Method {id: 'm', emb: vector([1.0, 0.0])})")
        .unwrap();
    for i in 0..20 {
        db.execute(&format!(
            "INSERT (:Graph:Method {{id: 'm{i}', emb: vector([0.5, 0.5])}})"
        ))
        .unwrap();
    }
    db.execute("CREATE VECTOR INDEX graph_emb ON :Graph(emb) DIMENSION 2 METRIC 'cosine'")
        .unwrap();
    let query = "MATCH (n:Graph) WHERE (n:File OR n:TypeDefinition) AND cosine_similarity(n.emb, vector([1.0, 0.0])) > 0.9 RETURN n.id AS id";
    assert_eq!(rows(&db, query), [[Value::from("f")]]);
    let text = plan(&db, "EXPLAIN", query);
    assert!(
        !text.contains("[labels:"),
        "the vector index on Graph: {text}"
    );
}

// ---------------------------------------------------------------------------
// 2d
// ---------------------------------------------------------------------------

#[test]
fn a_repeated_count_reads_the_count_when_it_runs() {
    let db = GrafeoDB::new_in_memory();
    let query = "MATCH (n:Graph) RETURN count(*) AS c";
    for expected in [0, 3, 7] {
        if expected > 0 {
            let more = if expected == 3 { 3 } else { 4 };
            db.execute(&format!(
                "UNWIND range(1, {more}) AS i INSERT (:Graph {{i: i}})"
            ))
            .unwrap();
        }
        assert_eq!(rows(&db, query), int(expected), "the plan may be cached");
    }
}

#[test]
fn count_forms_on_a_label() {
    let db = GrafeoDB::new_in_memory();
    db.execute("UNWIND range(1, 5) AS i INSERT (:Graph {x: i})")
        .unwrap();
    db.execute("INSERT (:Other {x: 100})").unwrap();
    for language in LANGUAGES {
        let q = |query: &str| rows_in(&db, language, query);
        assert_eq!(
            q("MATCH (n:Graph) RETURN count(n) AS c"),
            int(5),
            "{language}"
        );
        assert_eq!(
            q("MATCH (n:Graph) RETURN count(n) AS a, count(*) AS b"),
            [[Value::Int64(5), Value::Int64(5)]],
            "{language}"
        );
        assert_eq!(
            q("MATCH (n:Graph) RETURN count(DISTINCT n) AS c"),
            int(5),
            "{language}"
        );
        assert_eq!(
            q("MATCH (n:Graph) RETURN count(*) AS c, sum(n.x) AS s"),
            [[Value::Int64(5), Value::Int64(15)]],
            "{language}"
        );
        assert_eq!(
            q("MATCH (n:Graph) WHERE n.x > 2 RETURN count(*) AS c"),
            int(3),
            "{language}"
        );
        assert_eq!(
            q("MATCH (n:Unknown) RETURN count(*) AS c"),
            int(0),
            "{language}"
        );
    }
    db.execute("MATCH (n:Other) DELETE n").unwrap();
    assert_eq!(
        rows(&db, "MATCH (n:Other) RETURN count(*) AS c"),
        int(0),
        "no node left"
    );
    let mut session = db.session();
    session.begin_transaction().unwrap();
    session
        .execute("INSERT (:Graph {x: 6}), (:Graph {x: 7})")
        .unwrap();
    assert_eq!(
        session
            .execute("MATCH (n:Graph) RETURN count(n) AS a, count(*) AS b")
            .unwrap()
            .rows(),
        [[Value::Int64(7), Value::Int64(7)]]
    );
    session.rollback().unwrap();
}
