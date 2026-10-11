//! Each conjunct of a WHERE filters the rows right where the variables it
//! reads are bound, and a node it pins by ID or by an indexed property is
//! looked up, not scanned:
//!
//! - below a write: `UNWIND $rows AS row MATCH (s), (d) WHERE id(s) = row.src
//!   AND id(d) = row.dst MERGE (s)-[:LINK]->(d)` seeks both endpoints (it
//!   scanned every `s` for each row);
//! - above an OPTIONAL MATCH: a conjunct on the variables of the clauses before
//!   it filters their rows first, one on the optional side stays above it;
//! - on the target of an edge: `MATCH (src)-[r]->(tgt) WHERE id(tgt) IN $ids`
//!   seeks `tgt` and follows its edges back (it expanded every edge of every
//!   node).
//!
//! Every plan returns the rows of the plan written: each query is checked
//! against a form the change leaves as it was, and against rows written out.
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test filters_reach_their_scans
//! ```

#![cfg(all(feature = "lpg", feature = "gql"))]

use std::collections::HashMap;

use grafeo_common::types::Value;
use grafeo_engine::GrafeoDB;

/// The languages a query runs in: GQL, and Cypher with its feature.
fn languages() -> Vec<&'static str> {
    let mut languages = vec!["gql"];
    if cfg!(feature = "cypher") {
        languages.push("cypher");
    }
    languages
}

/// People Alix, Gus, Vincent, Mia and Jules, and the cities Amsterdam, Berlin
/// and Paris, each with a `key` (its name in lower case) and the label `N`.
/// KNOWS (with `w`): Alix to Gus (3), Gus to Vincent (19), Vincent to Alix
/// (88), Mia to Gus (33) and Jules to himself (38). LIVES_IN: Alix and Mia in
/// Amsterdam, Gus in Berlin, Jules in Paris (`w` 1, 2, 4, 5). With `indexed`,
/// a property index on `key`.
fn graph(indexed: bool) -> GrafeoDB {
    let db = GrafeoDB::new_in_memory();
    if indexed {
        db.create_property_index("key").unwrap();
    }
    db.execute(
        "INSERT (alix:Person:N {name: 'Alix', key: 'alix'}), \
         (gus:Person:N {name: 'Gus', key: 'gus'}), \
         (vincent:Person:N {name: 'Vincent', key: 'vincent'}), \
         (mia:Person:N {name: 'Mia', key: 'mia'}), \
         (jules:Person:N {name: 'Jules', key: 'jules'}), \
         (ams:City:N {name: 'Amsterdam', key: 'amsterdam'}), \
         (ber:City:N {name: 'Berlin', key: 'berlin'}), \
         (par:City:N {name: 'Paris', key: 'paris'}), \
         (alix)-[:KNOWS {w: 3}]->(gus), (gus)-[:KNOWS {w: 19}]->(vincent), \
         (vincent)-[:KNOWS {w: 88}]->(alix), (mia)-[:KNOWS {w: 33}]->(gus), \
         (jules)-[:KNOWS {w: 38}]->(jules), \
         (alix)-[:LIVES_IN {w: 1}]->(ams), (mia)-[:LIVES_IN {w: 2}]->(ams), \
         (gus)-[:LIVES_IN {w: 4}]->(ber), (jules)-[:LIVES_IN {w: 5}]->(par)",
    )
    .unwrap();
    db
}

/// The ID of the node named `name`.
fn id_of(db: &GrafeoDB, name: &str) -> Value {
    let result = db
        .execute(&format!("MATCH (n {{name: '{name}'}}) RETURN id(n)"))
        .unwrap();
    result.rows()[0][0].clone()
}

fn list(values: Vec<Value>) -> Value {
    Value::List(values.into())
}

fn text(value: &Value) -> String {
    match value {
        Value::String(text) => text.to_string(),
        Value::Int64(number) => number.to_string(),
        Value::Null => "null".to_string(),
        other => format!("{other:?}"),
    }
}

fn params(entries: &[(&str, Value)]) -> HashMap<String, Value> {
    entries
        .iter()
        .map(|(name, value)| ((*name).to_string(), value.clone()))
        .collect()
}

/// The rows of `query` in `language`, as text, sorted.
fn rows(db: &GrafeoDB, language: &str, query: &str, entries: &[(&str, Value)]) -> Vec<Vec<String>> {
    let result = db
        .execute_language(query, language, Some(params(entries)))
        .unwrap_or_else(|error| panic!("{language}: {query}: {error}"));
    let mut rows: Vec<Vec<String>> = result
        .rows()
        .iter()
        .map(|row| row.iter().map(text).collect())
        .collect();
    rows.sort();
    rows
}

/// The plan of `query` in `language`, as EXPLAIN prints it.
fn plan(db: &GrafeoDB, language: &str, query: &str, entries: &[(&str, Value)]) -> String {
    db.execute_language(&format!("EXPLAIN {query}"), language, Some(params(entries)))
        .unwrap_or_else(|error| panic!("{language}: EXPLAIN {query}: {error}"))
        .rows()
        .iter()
        .map(|row| text(&row[0]))
        .collect::<Vec<_>>()
        .join("\n")
}

fn expected(rows: &[&[&str]]) -> Vec<Vec<String>> {
    let mut rows: Vec<Vec<String>> = rows
        .iter()
        .map(|row| row.iter().map(|cell| (*cell).to_string()).collect())
        .collect();
    rows.sort();
    rows
}

/// The source of each pattern in two forms: unlabeled, where a target the
/// WHERE pins starts the expand, and with the label `N` every node has,
/// whose label scan keeps the expand as written. Both find the same rows.
fn both_starts(query: &str) -> (String, String) {
    (query.to_string(), query.replace("(src)", "(src:N)"))
}

/// Queries on the target of an edge pattern, each with the parameters it
/// reads: directed both ways and undirected, a self-loop, IN lists with
/// duplicates, NULL and an ID no node has, a key from UNWIND, other
/// conjuncts on the source, the edge and both ends, and the target's label.
fn target_queries(db: &GrafeoDB) -> Vec<(&'static str, Vec<(&'static str, Value)>)> {
    let (alix, gus, jules) = (id_of(db, "Alix"), id_of(db, "Gus"), id_of(db, "Jules"));
    let amsterdam = id_of(db, "Amsterdam");
    let ids = || list(vec![gus.clone(), alix.clone(), jules.clone()]);
    vec![
        (
            "MATCH (src)-[r:KNOWS]->(tgt) WHERE id(tgt) IN $ids \
             RETURN src.name, r.w, tgt.name",
            vec![("ids", ids())],
        ),
        (
            "MATCH (src)<-[r:KNOWS]-(tgt) WHERE id(tgt) IN $ids \
             RETURN src.name, r.w, tgt.name",
            vec![("ids", ids())],
        ),
        (
            "MATCH (src)-[r:KNOWS]-(tgt) WHERE id(tgt) IN $ids \
             RETURN src.name, r.w, tgt.name",
            vec![("ids", ids())],
        ),
        (
            "MATCH (src)-[r]->(tgt) WHERE id(tgt) = $id RETURN src.name, type(r), r.w",
            vec![("id", amsterdam.clone())],
        ),
        (
            "MATCH (src)-[r]-(tgt) WHERE id(tgt) IN $ids RETURN src.name, type(r), r.w",
            vec![(
                "ids",
                list(vec![
                    jules.clone(),
                    jules.clone(),
                    Value::Null,
                    Value::Int64(88_888),
                ]),
            )],
        ),
        (
            "UNWIND $ids AS k MATCH (src)-[r:KNOWS]->(tgt) WHERE id(tgt) = k \
             RETURN k, src.name, tgt.name",
            vec![("ids", list(vec![gus.clone(), gus.clone(), alix.clone()]))],
        ),
        (
            "MATCH (src)-[r]->(tgt) WHERE id(tgt) IN $ids AND src.name <> 'Mia' \
             AND r.w > 2 AND tgt.name <> src.name RETURN src.name, r.w, tgt.name",
            vec![(
                "ids",
                list(vec![gus.clone(), amsterdam.clone(), jules.clone()]),
            )],
        ),
        (
            "MATCH (src)-[r]->(tgt:City) WHERE id(tgt) IN $ids RETURN src.name, tgt.name",
            vec![("ids", list(vec![gus.clone(), amsterdam.clone()]))],
        ),
        (
            "MATCH (src)-->(tgt) WHERE id(tgt) IN $ids RETURN src.name, tgt.name",
            vec![("ids", ids())],
        ),
        (
            "MATCH (src)-[r:KNOWS]->(tgt)-[l:LIVES_IN]->(c) WHERE id(tgt) IN $ids \
             RETURN src.name, tgt.name, c.name, l.w",
            vec![("ids", ids())],
        ),
        (
            "MATCH (c)<-[l:LIVES_IN]-(src)-[r:KNOWS]->(tgt) WHERE id(tgt) IN $ids \
             RETURN src.name, tgt.name, c.name",
            vec![("ids", ids())],
        ),
        (
            "MATCH (src)-[r:KNOWS]->(tgt)-[l:LIVES_IN]->(c) WHERE id(tgt) IN $ids \
             RETURN count(l)",
            vec![("ids", ids())],
        ),
        (
            "MATCH (src)-[r:KNOWS]->(tgt) WHERE id(tgt) IN $ids RETURN count(r)",
            vec![("ids", ids())],
        ),
        (
            "MATCH (src)-[r]->(tgt) WHERE tgt.key IN ['gus', 'amsterdam', 'nowhere'] \
             RETURN src.name, r.w, tgt.name",
            vec![],
        ),
        (
            "MATCH (src)-[r]-(tgt) WHERE tgt.key = $key RETURN src.name, r.w",
            vec![("key", Value::from("jules"))],
        ),
        (
            "UNWIND ['gus', 'paris'] AS k MATCH (src)-[r]->(tgt) WHERE tgt.key = k \
             RETURN k, src.name, r.w",
            vec![],
        ),
    ]
}

/// Each query on the target of an edge finds what the same query finds from a
/// labeled source, which keeps the expand as written; with and without an
/// index on the key, in GQL and Cypher.
#[test]
fn a_sought_target_finds_the_rows_of_the_expand_as_written() {
    for indexed in [false, true] {
        let db = graph(indexed);
        for (query, entries) in target_queries(&db) {
            for language in languages() {
                let (turned, written) = both_starts(query);
                assert_eq!(
                    rows(&db, language, &turned, &entries),
                    rows(&db, language, &written, &entries),
                    "{language} (index: {indexed}): {query}"
                );
            }
        }
    }
}

/// The rows themselves, written out.
#[test]
fn a_sought_target_finds_its_edges() {
    let db = graph(false);
    let (alix, gus, jules) = (id_of(&db, "Alix"), id_of(&db, "Gus"), id_of(&db, "Jules"));
    let ids = [("ids", list(vec![gus, alix, jules.clone()]))];
    for language in languages() {
        assert_eq!(
            rows(
                &db,
                language,
                "MATCH (src)-[r:KNOWS]->(tgt) WHERE id(tgt) IN $ids \
                 RETURN src.name, r.w, tgt.name",
                &ids
            ),
            expected(&[
                &["Alix", "3", "Gus"],
                &["Jules", "38", "Jules"],
                &["Mia", "33", "Gus"],
                &["Vincent", "88", "Alix"],
            ]),
            "{language}: outgoing"
        );
        assert_eq!(
            rows(
                &db,
                language,
                "MATCH (src)<-[r:KNOWS]-(tgt) WHERE id(tgt) IN $ids \
                 RETURN src.name, r.w, tgt.name",
                &ids
            ),
            expected(&[
                &["Gus", "3", "Alix"],
                &["Jules", "38", "Jules"],
                &["Vincent", "19", "Gus"],
            ]),
            "{language}: incoming"
        );
        assert_eq!(
            rows(
                &db,
                language,
                "MATCH (src)-[r]-(tgt) WHERE id(tgt) = $id RETURN src.name, r.w",
                &[("id", jules.clone())]
            ),
            rows(
                &db,
                language,
                "MATCH (src:N)-[r]-(tgt) WHERE id(tgt) = $id RETURN src.name, r.w",
                &[("id", jules.clone())]
            ),
            "{language}: undirected, a self-loop as often from either end"
        );
        assert_eq!(
            rows(
                &db,
                language,
                "MATCH (src)-[r]->(tgt) WHERE tgt.key = 'amsterdam' RETURN src.name, r.w",
                &[]
            ),
            expected(&[&["Alix", "1"], &["Mia", "2"]]),
            "{language}: a key"
        );
        assert_eq!(
            rows(
                &db,
                language,
                "MATCH (src)-[:KNOWS]->(tgt)-[:LIVES_IN]->(c) WHERE id(tgt) IN $ids \
                 RETURN src.name, tgt.name, c.name",
                &ids
            ),
            expected(&[
                &["Alix", "Gus", "Berlin"],
                &["Jules", "Jules", "Paris"],
                &["Mia", "Gus", "Berlin"],
                &["Vincent", "Alix", "Amsterdam"],
            ]),
            "{language}: the next edge goes on from the sought target"
        );
    }
}

/// EXPLAIN shows where the expand starts: at the sought target, from a
/// source without a label, as written from a labeled one.
#[test]
fn explain_shows_the_expand_starting_at_the_sought_target() {
    for indexed in [false, true] {
        let db = graph(indexed);
        let ids = [("ids", list(vec![id_of(&db, "Gus")]))];
        for language in languages() {
            let context = format!("{language} (index: {indexed})");
            let turned = plan(
                &db,
                language,
                "MATCH (src)-[r:KNOWS]->(tgt) WHERE id(tgt) IN $ids RETURN src.name",
                &ids,
            );
            assert!(
                turned.contains("Expand (tgt)<-[:KNOWS]<-(src)") && turned.contains("[seek: id]"),
                "{context}:\n{turned}"
            );
            let written = plan(
                &db,
                language,
                "MATCH (src:N)-[r:KNOWS]->(tgt) WHERE id(tgt) IN $ids RETURN src.name",
                &ids,
            );
            assert!(
                written.contains("Expand (src)->[:KNOWS]->(tgt)") && !written.contains("[seek"),
                "{context}:\n{written}"
            );
            let keyed = plan(
                &db,
                language,
                "MATCH (src)-[r]->(tgt) WHERE tgt.key = 'gus' RETURN src.name",
                &[],
            );
            assert_eq!(
                keyed.contains("Expand (tgt)<-[:*]<-(src)"),
                indexed,
                "{context}: a key turns the expand only when it is indexed:\n{keyed}"
            );
        }
    }
}

/// `RETURN *` returns the columns in the order the pattern binds them, so
/// the expand keeps its start there.
#[test]
fn return_star_keeps_the_order_of_the_columns() {
    let db = graph(false);
    let ids = [("ids", list(vec![id_of(&db, "Gus")]))];
    for language in languages() {
        let query = "MATCH (src)-[r:KNOWS]->(tgt) WHERE id(tgt) IN $ids RETURN *";
        let result = db
            .execute_language(query, language, Some(params(&ids)))
            .unwrap();
        assert_eq!(result.columns, ["src", "r", "tgt"], "{language}");
        assert_eq!(result.rows().len(), 2, "{language}");
        let explained = plan(&db, language, query, &ids);
        assert!(
            explained.contains("Expand (src)->[:KNOWS]->(tgt)"),
            "{language}:\n{explained}"
        );
    }
}

/// A transaction's own edges and deletions count from either end.
#[test]
fn a_sought_target_sees_what_the_transaction_sees() {
    let db = graph(false);
    let gus = id_of(&db, "Gus");
    let query = "MATCH (src)-[r:KNOWS]->(tgt) WHERE id(tgt) = $id RETURN src.name";
    let mut session = db.session();
    session.begin_transaction().unwrap();
    session
        .execute(
            "MATCH (v:Person {name: 'Vincent'}), (g:Person {name: 'Gus'}) \
             INSERT (v)-[:KNOWS {w: 88}]->(g)",
        )
        .unwrap();
    session
        .execute("MATCH (:Person {name: 'Mia'})-[r:KNOWS]->() DELETE r")
        .unwrap();
    let seen = |session: &grafeo_engine::Session| -> Vec<String> {
        let mut names: Vec<String> = session
            .execute_with_params(query, params(&[("id", gus.clone())]))
            .unwrap()
            .rows()
            .iter()
            .map(|row| text(&row[0]))
            .collect();
        names.sort();
        names
    };
    assert_eq!(seen(&session), ["Alix", "Vincent"], "in the transaction");
    session.rollback().unwrap();
    assert_eq!(seen(&db.session()), ["Alix", "Mia"], "after the rollback");
}

/// The upserts of edges between endpoints pinned by ID: each write finds its
/// endpoints, and the plan seeks each of them (it scanned every source for
/// each row, the filter above both scans).
#[test]
fn a_filter_below_a_write_seeks_each_endpoint() {
    let writes = [
        (
            "gql",
            "UNWIND $rows AS row MATCH (s), (d) WHERE id(s) = row.src AND id(d) = row.dst \
             INSERT (s)-[l:LINK]->(d) RETURN count(l)",
        ),
        (
            "gql",
            "UNWIND $rows AS row MATCH (s), (d) WHERE id(s) = row.src AND id(d) = row.dst \
             INSERT (s)-[:LINK]->(d)",
        ),
        (
            "cypher",
            "UNWIND $rows AS row MATCH (s), (d) WHERE id(s) = row.src AND id(d) = row.dst \
             MERGE (s)-[l:LINK]->(d) RETURN count(l)",
        ),
        (
            "cypher",
            "UNWIND $rows AS row MATCH (s), (d) WHERE id(s) = row.src AND id(d) = row.dst \
             CREATE (s)-[:LINK]->(d)",
        ),
    ];
    for (language, write) in writes {
        if !languages().contains(&language) {
            continue;
        }
        let db = graph(false);
        let row = |src: &str, dst: &str| {
            Value::Map(
                [
                    ("src".into(), id_of(&db, src)),
                    ("dst".into(), id_of(&db, dst)),
                ]
                .into_iter()
                .collect::<std::collections::BTreeMap<_, _>>()
                .into(),
            )
        };
        let entries = [(
            "rows",
            list(vec![
                row("Alix", "Paris"),
                row("Mia", "Berlin"),
                Value::Map(
                    [
                        ("src".into(), id_of(&db, "Gus")),
                        ("dst".into(), Value::Int64(88_888)),
                    ]
                    .into_iter()
                    .collect::<std::collections::BTreeMap<_, _>>()
                    .into(),
                ),
            ]),
        )];
        let explained = plan(&db, language, write, &entries);
        assert_eq!(
            explained.matches("[seek: id]").count(),
            2,
            "{language}: {write}\n{explained}"
        );
        assert!(
            !explained.contains(" And "),
            "{language}: each conjunct on its own scan: {write}\n{explained}"
        );
        db.execute_language(write, language, Some(params(&entries)))
            .unwrap();
        assert_eq!(
            rows(
                &db,
                "gql",
                "MATCH (a)-[:LINK]->(b) RETURN a.name, b.name",
                &[]
            ),
            expected(&[&["Alix", "Paris"], &["Mia", "Berlin"]]),
            "{language}: {write}"
        );
    }
}

/// A WHERE after an OPTIONAL MATCH and a WITH filters the rows: a conjunct on
/// the clauses before it filters their rows before the join, and one on the
/// optional side stays above it. Moved into the optional side, `c IS NULL`
/// and `c.name = 'Amsterdam'` would keep every person, with nulls; inside
/// the OPTIONAL MATCH a WHERE does that.
#[test]
fn a_where_after_an_optional_match_filters_its_rows() {
    let db = graph(false);
    for language in languages() {
        let after = |condition: &str| {
            rows(
                &db,
                language,
                &format!(
                    "MATCH (p:Person) OPTIONAL MATCH (p)-[:LIVES_IN]->(c) \
                     WITH p, c WHERE {condition} RETURN p.name, c.name"
                ),
                &[],
            )
        };
        assert_eq!(
            after("p.name <> 'Gus' AND c IS NULL"),
            expected(&[&["Vincent", "null"]]),
            "{language}"
        );
        assert_eq!(
            after("c.name = 'Amsterdam' AND p.name <> 'Mia'"),
            expected(&[&["Alix", "Amsterdam"]]),
            "{language}"
        );
        assert_eq!(
            after("c.name = 'Amsterdam'"),
            expected(&[&["Alix", "Amsterdam"], &["Mia", "Amsterdam"]]),
            "{language}"
        );
        assert_eq!(
            rows(
                &db,
                language,
                "MATCH (p:Person) OPTIONAL MATCH (p)-[:LIVES_IN]->(c) \
                 WHERE c.name = 'Amsterdam' RETURN p.name, c.name",
                &[]
            ),
            expected(&[
                &["Alix", "Amsterdam"],
                &["Gus", "null"],
                &["Jules", "null"],
                &["Mia", "Amsterdam"],
                &["Vincent", "null"],
            ]),
            "{language}: a WHERE of the OPTIONAL MATCH keeps every person"
        );

        let explained = plan(
            &db,
            language,
            "MATCH (p:Person) OPTIONAL MATCH (p)-[:LIVES_IN]->(c) \
             WITH p, c WHERE p.name <> 'Gus' AND c IS NULL RETURN p.name",
            &[],
        );
        let line = |needle: &str| {
            explained
                .lines()
                .position(|line| line.contains(needle))
                .unwrap_or_else(|| panic!("{language}: no {needle}:\n{explained}"))
        };
        assert!(
            line("IsNull") < line("LeftJoin") && line("LeftJoin") < line("p.name Ne"),
            "{language}: the conjunct on p filters below the join:\n{explained}"
        );
    }
}

/// Prints the time of each query of the planner-performance notes, at 20,000
/// nodes and 40,000 edges.
#[test]
#[ignore = "bench: prints timings, run with --release -- --ignored --nocapture"]
fn bench_filters_reach_their_scans() {
    use std::time::Instant;

    let size: usize = 20_000;
    let db = GrafeoDB::new_in_memory();
    db.execute(&format!(
        "UNWIND range(0, {}) AS i INSERT (:N {{k: i}})",
        size - 1
    ))
    .unwrap();
    let ids: Vec<Value> = db
        .execute("MATCH (n:N) RETURN id(n) ORDER BY n.k")
        .unwrap()
        .rows()
        .iter()
        .map(|row| row[0].clone())
        .collect();
    let pair = |src: usize, dst: usize, i: usize| {
        Value::Map(
            [
                ("src".into(), ids[src].clone()),
                ("dst".into(), ids[dst].clone()),
                ("i".into(), Value::Int64(i64::try_from(i).unwrap())),
            ]
            .into_iter()
            .collect::<std::collections::BTreeMap<_, _>>()
            .into(),
        )
    };
    // Each node to the next and to the third after it.
    let edges: Vec<Value> = (0..size)
        .flat_map(|i| [i + 1, i + 3].map(|j| (i, j)))
        .filter(|&(_, j)| j < size)
        .map(|(i, j)| pair(i, j, i))
        .collect();
    db.execute_with_params(
        "UNWIND $edges AS e MATCH (s), (d) WHERE id(s) = e.src AND id(d) = e.dst \
         INSERT (s)-[:R]->(d)",
        params(&[("edges", list(edges))]),
    )
    .unwrap();
    let rows: Vec<Value> = (0..size).map(|i| pair(i, (i * 7 + 3) % size, i)).collect();
    let entries = [
        ("rows", list(rows)),
        (
            "ids",
            list(vec![ids[3].clone(), ids[19].clone(), ids[88].clone()]),
        ),
    ];
    for (language, query) in [
        (
            "gql",
            "UNWIND $rows AS item MATCH (s), (d) WHERE id(s) = item.src AND id(d) = item.dst \
             RETURN count(item.i)",
        ),
        (
            "gql",
            "MATCH (src)-[r]->(tgt) WHERE id(src) IN $ids RETURN count(r)",
        ),
        (
            "gql",
            "MATCH (src)-[r]->(tgt) WHERE id(tgt) IN $ids RETURN count(r)",
        ),
        (
            "gql",
            "MATCH (src)<-[r]-(tgt) WHERE id(tgt) IN $ids RETURN count(r)",
        ),
        (
            "gql",
            "MATCH (src)-[r]-(tgt) WHERE id(tgt) IN $ids RETURN count(r)",
        ),
        (
            "gql",
            "MATCH (a:N) OPTIONAL MATCH (a)-[:R]->(b) WITH a, b WHERE a.k = 3 AND b.k > 3 \
             RETURN count(a)",
        ),
        (
            "gql",
            "MATCH (a:N) OPTIONAL MATCH (a)-[:R]->(b) WITH a, b WHERE a.k = 3 AND b IS NULL \
             RETURN count(a)",
        ),
        // The writes last, so that the reads above see the same graph.
        (
            "cypher",
            "UNWIND $rows AS item MATCH (s), (d) WHERE id(s) = item.src AND id(d) = item.dst \
             MERGE (s)-[r:LINK]->(d) RETURN count(r)",
        ),
        (
            "gql",
            "UNWIND $rows AS item MATCH (s), (d) WHERE id(s) = item.src AND id(d) = item.dst \
             INSERT (s)-[r:LINK2]->(d) RETURN count(r)",
        ),
    ] {
        if !languages().contains(&language) {
            continue;
        }
        let start = Instant::now();
        let result = db
            .execute_language(query, language, Some(params(&entries)))
            .unwrap();
        println!(
            "{:>10.2?}  {:?}  {language}: {query}",
            start.elapsed(),
            result.rows().first()
        );
    }
}
