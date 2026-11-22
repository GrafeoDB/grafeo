//! The way each filter is planned, as PROFILE shows its operators, and the rows it gives: one
//! statement per route of the filter's planning, in the order the planner tries them, and the
//! routes whose data or index declines and hands the filter to the next. They hold before and
//! after the planner decides the route in one function. The planner's own record of the routes
//! is pinned by the unit test `each_filter_records_its_route`.
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test main filter_routes::
//! ```

#![cfg(all(
    feature = "gql",
    feature = "cypher",
    feature = "vector-index",
    feature = "text-index"
))]

use grafeo_common::types::Value;
use grafeo_engine::GrafeoDB;

/// People `k` 1 to 4 with names, bios and embeddings, a chain of `R` steps with weights, and
/// things `Q` whose `k` matches some people's; a property index on `name`, a text index on
/// `bio` and a vector index on `embedding`.
fn graph() -> GrafeoDB {
    let db = GrafeoDB::new_in_memory();
    db.execute(
        "INSERT (a:P {name: 'Alix', k: 1, t: 1, bio: 'graph canals', embedding: [1.0, 0.0]}), \
                (g:P {name: 'Gus', k: 2, t: 1, bio: 'canals', embedding: [0.0, 1.0]}), \
                (v:P {name: 'Vincent', k: 3, t: 1, bio: 'bikes', embedding: [1.0, 1.0]}), \
                (j:P {name: 'Jules', k: 4, t: 1, bio: 'graph', embedding: [0.5, 0.5]}), \
                (a)-[:R {w: 1}]->(g), (g)-[:R {w: 2}]->(v), (v)-[:R {w: 3}]->(a), (g)-[:R {w: 4}]->(j), \
                (:Q {k: 2, name: 'Gus'}), (:Q {k: 4, name: 'Jules'}), (:Q {k: 9, name: 'Mia'})",
    )
    .unwrap();
    for ddl in [
        "CREATE INDEX p_name FOR (p:P) ON (p.name)",
        "CREATE INDEX p_bio FOR (p:P) ON (p.bio) USING TEXT",
        "CREATE VECTOR INDEX p_embedding ON :P(embedding) DIMENSION 2 METRIC 'cosine'",
    ] {
        db.execute(ddl).unwrap();
    }
    db
}

/// The operator names of `query`'s PROFILE, from the top down.
fn operators(db: &GrafeoDB, cypher: bool, query: &str) -> Vec<String> {
    let text = format!("PROFILE {query}");
    let profile = if cypher {
        db.execute_cypher(&text)
    } else {
        db.execute(&text)
    }
    .unwrap_or_else(|error| panic!("PROFILE {query}: {error}"));
    let Value::String(text) = &profile.rows()[0][0] else {
        panic!("PROFILE {query} gave no text");
    };
    text.lines()
        .filter_map(|line| line.split_whitespace().next())
        .filter(|name| *name != "Total")
        .map(str::to_string)
        .collect()
}

/// The rows of `query`, each as its values joined by `, ` (a string without its quotes),
/// sorted.
fn rows(db: &GrafeoDB, cypher: bool, query: &str) -> Vec<String> {
    let result = if cypher {
        db.execute_cypher(query)
    } else {
        db.execute(query)
    }
    .unwrap_or_else(|error| panic!("{query}: {error}"));
    let cell = |value: &Value| match value {
        Value::String(text) => text.to_string(),
        other => other.to_string(),
    };
    let mut rows: Vec<String> = result
        .rows()
        .iter()
        .map(|row| row.iter().map(cell).collect::<Vec<_>>().join(", "))
        .collect();
    rows.sort();
    rows
}

/// One statement per route, in the planner's order, and the declining cases.
const ROUTES: &[(&str, bool, &str)] = &[
    (
        "exists as a semi join",
        false,
        "MATCH (a:P) WHERE EXISTS { MATCH (a)-[:R]->(b)-[:R]->(c) } RETURN a.name AS name",
    ),
    (
        "subquery per row",
        false,
        "MATCH (a:P) WHERE COUNT { MATCH (a)-[:R]->(b) } > 1 RETURN a.name AS name",
    ),
    (
        "intersection closing",
        true,
        "MATCH (a:P {k: 1})-[:R]-(b:P)-[:R]-(c:P)-[:R]-(a) RETURN count(*) AS n",
    ),
    (
        "zone map excludes",
        false,
        "MATCH (a:P)-[:R]->(b) WHERE b.k > 100 RETURN a.name AS name",
    ),
    (
        "zone map keeps",
        false,
        "MATCH (a:P)-[:R]->(b) WHERE b.k > 2 RETURN a.name AS name",
    ),
    (
        "zone map, not equal",
        false,
        "MATCH (a:P)-[:R]->(b) WHERE b.k <> 2 RETURN a.name AS name",
    ),
    (
        "zone map, not equal to every value",
        false,
        "MATCH (a:P)-[:R]->(b) WHERE b.t <> 1 RETURN a.name AS name",
    ),
    (
        "edge condition into a path",
        false,
        "MATCH p = (a:P {k: 1})-[:R]->{1,3}(b) WHERE all(e IN edges(p) WHERE e.w > 0) RETURN b.name AS name",
    ),
    (
        "per-row seek",
        false,
        "MATCH (q:Q) MATCH (a:P) WHERE a.k = q.k RETURN a.name AS name",
    ),
    (
        "per-row seek by an indexed key",
        false,
        "MATCH (q:Q) MATCH (a:P) WHERE a.name = q.name RETURN a.k AS k",
    ),
    (
        "value join",
        false,
        "MATCH (q:Q), (a:P) WHERE a.k = q.k RETURN a.name AS name",
    ),
    (
        "property index",
        false,
        "MATCH (a:P) WHERE a.name = 'Gus' RETURN a.k AS k",
    ),
    (
        "property index, nothing",
        false,
        "MATCH (a:P) WHERE a.name = 'Nobody' RETURN a.k AS k",
    ),
    (
        "in-list index",
        false,
        "MATCH (a:P) WHERE a.name IN ['Gus', 'Jules'] RETURN a.k AS k",
    ),
    (
        "in-list index, empty",
        false,
        "MATCH (a:P) WHERE a.name IN [] RETURN a.k AS k",
    ),
    (
        "range index",
        false,
        "MATCH (a:P) WHERE a.k > 2 RETURN a.name AS name",
    ),
    (
        "compound hybrid",
        false,
        "MATCH (a:P) WHERE text_score(a.bio, 'graph') > 0.0 AND cosine_similarity(a.embedding, [1.0, 0.0]) > 0.5 RETURN a.name AS name",
    ),
    (
        "vector index",
        false,
        "MATCH (a:P) WHERE cosine_similarity(a.embedding, [1.0, 0.0]) > 0.5 RETURN a.name AS name",
    ),
    (
        "text index",
        false,
        "MATCH (a:P) WHERE text_score(a.bio, 'canals') > 0.0 RETURN a.name AS name",
    ),
    (
        "label union",
        false,
        "MATCH (n) WHERE n:P OR n:Q RETURN n.k AS k",
    ),
    (
        "generic",
        false,
        "MATCH (a:P)-[:R]->(b) WHERE b.k < a.k RETURN a.name AS name",
    ),
    (
        "generic, edge uniqueness",
        true,
        "MATCH (a:P)-[:R]->(b)-[:R]->(c) RETURN c.name AS name",
    ),
];

/// The operators each route's statement runs, as PROFILE shows them on today's planner.
const PINNED: &[(&str, &[&str])] = &[
    (
        "exists as a semi join",
        &[
            "Project",
            "HashJoin",
            "Scan",
            "Expand",
            "Expand",
            "DelimScan",
        ],
    ),
    ("subquery per row", &["Project", "Filter", "Scan"]),
    (
        "intersection closing",
        &[
            "SimpleAggregate",
            "SimpleAggregate",
            "Filter",
            "Filter",
            "IntersectExpand",
            "IntersectExpand",
            "Filter",
            "Expand",
            "KeyColumnScan",
        ],
    ),
    ("zone map excludes", &["Project", "Empty", "Expand", "Scan"]),
    ("zone map keeps", &["Project", "Filter", "Expand", "Scan"]),
    (
        "zone map, not equal",
        &["Project", "Filter", "Expand", "Scan"],
    ),
    (
        "zone map, not equal to every value",
        &["Project", "Empty", "Expand", "Scan"],
    ),
    (
        "edge condition into a path",
        &[
            "Project",
            "VariableLengthExpand",
            "VariableLengthExpand",
            "KeyColumnScan",
        ],
    ),
    ("per-row seek", &["Project", "Filter", "HashJoin", "Scan"]),
    (
        "per-row seek by an indexed key",
        &["Project", "Filter", "NodeSeek", "Scan"],
    ),
    ("value join", &["Project", "Filter", "HashJoin", "Scan"]),
    ("property index", &["Project", "NodeList"]),
    ("property index, nothing", &["Project", "NodeList"]),
    ("in-list index", &["Project", "NodeList"]),
    ("in-list index, empty", &["Project", "NodeList"]),
    ("range index", &["Project", "RangeScan"]),
    ("compound hybrid", &["Project", "Project", "HashJoin"]),
    ("vector index", &["Project", "Filter", "VectorScan"]),
    (
        "text index",
        &["Project", "TextScan(BM25)", "TextScan(BM25)"],
    ),
    ("label union", &["Project", "Filter", "Scan"]),
    ("generic", &["Project", "Filter", "Expand", "Scan"]),
    (
        "generic, edge uniqueness",
        &["Project", "Filter", "Expand", "Expand", "Scan"],
    ),
];

/// The rows each route's statement gives, sorted: worked out by hand from [`graph`].
const ROWS: &[(&str, &[&str])] = &[
    // Alix, Gus and Vincent start two hops; Jules has no edge out.
    ("exists as a semi join", &["Alix", "Gus", "Vincent"]),
    // Gus alone has two edges out.
    ("subquery per row", &["Gus"]),
    // The cycle Alix, Gus, Vincent, both ways round.
    ("intersection closing", &["2"]),
    ("zone map excludes", &[]),
    // Gus to Vincent (3) and to Jules (4).
    ("zone map keeps", &["Gus", "Gus"]),
    // Every edge but Alix to Gus (2).
    ("zone map, not equal", &["Gus", "Gus", "Vincent"]),
    // Every `t` is 1.
    ("zone map, not equal to every value", &[]),
    // From Alix in one to three steps.
    (
        "edge condition into a path",
        &["Alix", "Gus", "Jules", "Vincent"],
    ),
    // `Q`'s `k` 2 and 4 are Gus's and Jules's; 9 is nobody's.
    ("per-row seek", &["Gus", "Jules"]),
    ("per-row seek by an indexed key", &["2", "4"]),
    ("value join", &["Gus", "Jules"]),
    ("property index", &["2"]),
    ("property index, nothing", &[]),
    ("in-list index", &["2", "4"]),
    ("in-list index, empty", &[]),
    ("range index", &["Jules", "Vincent"]),
    // `graph` in Alix's and Jules's bios, both embeddings near [1, 0].
    ("compound hybrid", &["Alix", "Jules"]),
    // Similarity 1 for Alix, 0.71 for Vincent and Jules, 0 for Gus.
    ("vector index", &["Alix", "Jules", "Vincent"]),
    ("text index", &["Alix", "Gus"]),
    // The people's `k` and the things'.
    ("label union", &["1", "2", "2", "3", "4", "4", "9"]),
    // Vincent (3) to Alix (1) alone goes down.
    ("generic", &["Vincent"]),
    // The ends of the two-step walks from Alix, Gus and Vincent.
    (
        "generic, edge uniqueness",
        &["Alix", "Gus", "Jules", "Vincent"],
    ),
];

/// Each statement runs the operators of its route and gives its rows, and the pins, the rows
/// and the statements name the same routes.
#[test]
fn each_filter_takes_its_route() {
    let db = graph();
    let names = ROUTES.iter().map(|(name, ..)| *name).collect::<Vec<_>>();
    assert_eq!(
        names,
        PINNED.iter().map(|(name, _)| *name).collect::<Vec<_>>()
    );
    assert_eq!(
        names,
        ROWS.iter().map(|(name, _)| *name).collect::<Vec<_>>()
    );
    for (((name, cypher, query), (_, pinned)), (_, expected)) in ROUTES.iter().zip(PINNED).zip(ROWS)
    {
        assert_eq!(operators(&db, *cypher, query), *pinned, "{name}: {query}");
        assert_eq!(rows(&db, *cypher, query), *expected, "{name}: {query}");
    }
}
