//! A `MATCH` after another one without a shared variable, joined to it by
//! equal values (`MATCH (f:File) MATCH (t:TypeDefinition) WHERE t.filePath =
//! f.path`), runs as a hash join on those values instead of a scan for each
//! row of the first (#455). `EXPLAIN` marks its filter `[hash join: ...]`,
//! which still decides every row: the rows are those of the scan per row, in
//! the same order. A write before the scan, an indexed key and subqueries
//! keep their plans. The planner's tests (`value_join.rs`) compare the rows
//! of the two plans value by value.
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test value_hash_join
//! ```

#![cfg(all(feature = "lpg", feature = "gql", feature = "cypher"))]

use grafeo_common::types::Value;
use grafeo_engine::{GrafeoDB, Session};

#[derive(Debug, Clone, Copy)]
enum Language {
    Gql,
    Cypher,
}

const LANGUAGES: [Language; 2] = [Language::Gql, Language::Cypher];

fn run(session: &Session, language: Language, query: &str) -> Vec<Vec<Value>> {
    let result = match language {
        Language::Gql => session.execute(query),
        Language::Cypher => session.execute_cypher(query),
    };
    result
        .unwrap_or_else(|error| panic!("{language:?} `{query}` failed: {error}"))
        .rows()
        .to_vec()
}

/// The text of the plan that `EXPLAIN` or `PROFILE` returns for `query`.
fn plan_text(session: &Session, language: Language, query: &str) -> String {
    run(session, language, query)
        .iter()
        .map(|row| match &row[0] {
            Value::String(text) => text.to_string(),
            other => format!("{other:?}"),
        })
        .collect::<Vec<_>>()
        .join("\n")
}

fn explain(session: &Session, language: Language, query: &str) -> String {
    plan_text(session, language, &format!("EXPLAIN {query}"))
}

/// The lines of a plan, trimmed.
fn lines(plan: &str) -> Vec<&str> {
    plan.lines().map(str::trim).collect()
}

/// Asserts that `query` returns `expected`, in this order, and that its
/// plan joins by hash on `keys` (`Some("t.k = f.k")`), or not at all.
fn assert_rows_and_plan(
    session: &Session,
    language: Language,
    query: &str,
    expected: &[Vec<Value>],
    keys: Option<&str>,
) {
    assert_eq!(
        run(session, language, query),
        expected,
        "{language:?} `{query}`"
    );
    let plan = explain(session, language, query);
    match keys {
        Some(keys) => assert!(
            plan.contains(&format!(" [hash join: {keys}]")),
            "{language:?} `{query}`:\n{plan}"
        ),
        None => assert!(
            !plan.contains("[hash join"),
            "{language:?} `{query}`:\n{plan}"
        ),
    }
}

/// Rows of strings, `-` for NULL.
fn table(rows: &[&[&str]]) -> Vec<Vec<Value>> {
    rows.iter()
        .map(|row| {
            row.iter()
                .map(|item| match *item {
                    "-" => Value::Null,
                    item => Value::from(item),
                })
                .collect()
        })
        .collect()
}

/// Files `a.rs` (5.0 `lines`), `b.rs` (7), `c.rs` (5) and one without a
/// path (1), then type definitions: Alix in `b.rs` at `line` 5, Gus in
/// `a.rs` at 5, Vincent in `b.rs` at '7' (a string), Jules without a path at
/// 1, and Mia in `z.rs`, which no file has, at 5.0.
fn files() -> GrafeoDB {
    let db = GrafeoDB::new_in_memory();
    db.execute(
        "INSERT (:File {path: 'a.rs', lines: 5.0}), (:File {path: 'b.rs', lines: 7}), \
                (:File {path: 'c.rs', lines: 5}), (:File {lines: 1})",
    )
    .unwrap();
    db.execute(
        "INSERT (:TypeDefinition {name: 'Alix', filePath: 'b.rs', line: 5}), \
                (:TypeDefinition {name: 'Gus', filePath: 'a.rs', line: 5}), \
                (:TypeDefinition {name: 'Vincent', filePath: 'b.rs', line: '7'}), \
                (:TypeDefinition {name: 'Jules', line: 1}), \
                (:TypeDefinition {name: 'Mia', filePath: 'z.rs', line: 5.0})",
    )
    .unwrap();
    db
}

/// The type definitions per file, in the order of the scan per row: the
/// files in order, and for each its type definitions in order.
fn definitions() -> Vec<Vec<Value>> {
    table(&[&["a.rs", "Gus"], &["b.rs", "Alix"], &["b.rs", "Vincent"]])
}

const PATH_JOIN: &str =
    "MATCH (f:File) MATCH (t:TypeDefinition) WHERE t.filePath = f.path RETURN f.path, t.name";

/// The forms of the join of files and type definitions on the path.
fn path_joins() -> Vec<(Language, &'static str)> {
    let mut queries = Vec::new();
    for language in LANGUAGES {
        for query in [
            PATH_JOIN,
            "MATCH (f:File) MATCH (t:TypeDefinition) WHERE f.path = t.filePath RETURN f.path, t.name",
            "MATCH (f:File) MATCH (t:TypeDefinition {filePath: f.path}) RETURN f.path, t.name",
        ] {
            queries.push((language, query));
        }
    }
    for query in [
        "MATCH (f:File) WITH f MATCH (t:TypeDefinition) WHERE t.filePath = f.path \
         RETURN f.path, t.name",
        "MATCH (f:File) WHERE f.lines > 0 MATCH (t:TypeDefinition) WHERE t.filePath = f.path \
         RETURN f.path, t.name",
    ] {
        queries.push((Language::Cypher, query));
    }
    queries
}

#[test]
fn a_join_on_a_property_is_a_hash_join_with_the_rows_of_the_scan_per_row() {
    let db = files();
    let session = db.session();
    for (language, query) in path_joins() {
        assert_rows_and_plan(
            &session,
            language,
            query,
            &definitions(),
            Some("t.filePath = f.path"),
        );
    }
}

/// `=` compares an integer with a float, and with a string that reads as
/// it: 7 meets '7', 5 meets 5.0. The rows are in the order of the scan per
/// row.
#[test]
fn numbers_meet_across_kinds() {
    let db = files();
    let session = db.session();
    let query = "MATCH (f:File) MATCH (t:TypeDefinition) WHERE t.line = f.lines \
                 RETURN f.path, t.name";
    let expected = table(&[
        &["a.rs", "Alix"],
        &["a.rs", "Gus"],
        &["a.rs", "Mia"],
        &["b.rs", "Vincent"],
        &["c.rs", "Alix"],
        &["c.rs", "Gus"],
        &["c.rs", "Mia"],
        &["-", "Jules"],
    ]);
    for language in LANGUAGES {
        assert_rows_and_plan(
            &session,
            language,
            query,
            &expected,
            Some("t.line = f.lines"),
        );
    }
}

/// More keys, keys computed from properties, and conjuncts beside the keys,
/// which the filter above the join checks on each pair it finds. A key in a
/// disjunction is no key.
#[test]
fn keys_and_the_conjuncts_beside_them() {
    let db = files();
    let session = db.session();
    let cases: Vec<(&str, Option<&str>, Vec<Vec<Value>>)> = vec![
        (
            "MATCH (f:File) MATCH (t:TypeDefinition) \
             WHERE t.filePath = f.path AND t.line = f.lines RETURN f.path, t.name",
            Some("t.filePath = f.path, t.line = f.lines"),
            table(&[&["a.rs", "Gus"], &["b.rs", "Vincent"]]),
        ),
        (
            "MATCH (f:File) MATCH (t:TypeDefinition) \
             WHERE toUpper(t.filePath) = toUpper(f.path) AND t.name <> 'Vincent' \
             RETURN f.path, t.name",
            Some("toUpper(t.filePath) = toUpper(f.path)"),
            table(&[&["a.rs", "Gus"], &["b.rs", "Alix"]]),
        ),
        (
            "MATCH (f:File) MATCH (t:TypeDefinition) \
             WHERE t.line + 2 = f.lines RETURN f.path, t.name",
            Some("t.line Add 2 = f.lines"),
            table(&[&["b.rs", "Alix"], &["b.rs", "Gus"], &["b.rs", "Mia"]]),
        ),
        (
            "MATCH (f:File) MATCH (t:TypeDefinition) \
             WHERE t.filePath = f.path AND (t.line > 6 OR f.lines > 6) RETURN f.path, t.name",
            Some("t.filePath = f.path"),
            table(&[&["b.rs", "Alix"], &["b.rs", "Vincent"]]),
        ),
        (
            "MATCH (f:File) MATCH (t:TypeDefinition) \
             WHERE t.filePath = f.path OR t.line = f.lines RETURN f.path, t.name",
            None,
            table(&[
                &["a.rs", "Alix"],
                &["a.rs", "Gus"],
                &["a.rs", "Mia"],
                &["b.rs", "Alix"],
                &["b.rs", "Vincent"],
                &["c.rs", "Alix"],
                &["c.rs", "Gus"],
                &["c.rs", "Mia"],
                &["-", "Jules"],
            ]),
        ),
    ];
    for (query, keys, expected) in cases {
        for language in LANGUAGES {
            assert_rows_and_plan(&session, language, query, &expected, keys);
        }
    }
}

/// Keys are values read from properties of both nodes: a node itself, its
/// `id()` and a value from the row that is not a property are no keys.
#[test]
fn a_key_reads_a_property_of_each_side() {
    let db = files();
    let session = db.session();
    for (query, expected) in [
        (
            "UNWIND ['a.rs', 'b.rs'] AS path MATCH (t:TypeDefinition) WHERE t.filePath = path \
             RETURN path, t.name",
            definitions(),
        ),
        (
            "MATCH (f:File) MATCH (t:TypeDefinition) WHERE t.line = id(f) + 1000 RETURN t.name",
            Vec::new(),
        ),
        (
            "MATCH (f:File) MATCH (t:TypeDefinition) WHERE t = f RETURN t.name",
            Vec::new(),
        ),
    ] {
        for language in LANGUAGES {
            assert_rows_and_plan(&session, language, query, &expected, None);
        }
    }
}

/// Functions Alix (`f1`), Gus (`f2`), Vincent (`f3`, not active), Jules (`f4`)
/// and Mia (`f5`) with `CALLS` edges, and `Model` nodes that point at them by
/// `source_identifier`: one for `f1`, `f3` and `f4`, two for `f2`, none for
/// `f5`.
fn calls() -> GrafeoDB {
    let db = GrafeoDB::new_in_memory();
    db.execute(
        "INSERT (:Function {id: 'f1', name: 'Alix', active: true}), \
                (:Function {id: 'f2', name: 'Gus', active: true}), \
                (:Function {id: 'f3', name: 'Vincent', active: false}), \
                (:Function {id: 'f4', name: 'Jules', active: true}), \
                (:Function {id: 'f5', name: 'Mia', active: true})",
    )
    .unwrap();
    db.execute(
        "UNWIND [['f1', 'f2'], ['f1', 'f3'], ['f2', 'f4'], ['f4', 'f1'], ['f5', 'f2'], ['f2', 'f5']] AS pair \
         MATCH (a:Function {id: pair[0]}), (b:Function {id: pair[1]}) \
         INSERT (a)-[:CALLS]->(b)",
    )
    .unwrap();
    db.execute(
        "INSERT (:Model {source_identifier: 'f1', city: 'Amsterdam'}), \
                (:Model {source_identifier: 'f2', city: 'Berlin'}), \
                (:Model {source_identifier: 'f2', city: 'Paris'}), \
                (:Model {source_identifier: 'f3', city: 'Prague'}), \
                (:Model {source_identifier: 'f4', city: 'Barcelona'})",
    )
    .unwrap();
    db
}

/// The query of #455, with a `WHERE` after each `MATCH` (Cypher), and with
/// one `WHERE` after both.
fn issue_queries() -> Vec<(Language, &'static str)> {
    let one_where = "MATCH (graph_src)-[edge:CALLS]->(graph_tgt) \
         MATCH (model_src:Model), (model_tgt:Model) \
         WHERE graph_src.active = true AND graph_tgt.active = true \
           AND model_src.source_identifier = graph_src.id \
           AND model_tgt.source_identifier = graph_tgt.id \
         RETURN graph_src.id, graph_tgt.id, model_src.city, model_tgt.city";
    vec![
        (
            Language::Cypher,
            "MATCH (graph_src)-[edge:CALLS]->(graph_tgt) \
             WHERE graph_src.active = true AND graph_tgt.active = true \
             MATCH (model_src:Model), (model_tgt:Model) \
             WHERE model_src.source_identifier = graph_src.id \
               AND model_tgt.source_identifier = graph_tgt.id \
             RETURN graph_src.id, graph_tgt.id, model_src.city, model_tgt.city",
        ),
        (Language::Cypher, one_where),
        (Language::Gql, one_where),
    ]
}

/// Each `Model` scan of the query of #455 is a hash join of its own, and the
/// rows are those of the scans per row, in their order.
#[test]
fn each_model_scan_of_the_issue_query_is_a_hash_join() {
    let db = calls();
    let session = db.session();
    for (language, query) in issue_queries() {
        assert_eq!(
            run(&session, language, query),
            table(&[
                &["f1", "f2", "Amsterdam", "Berlin"],
                &["f1", "f2", "Amsterdam", "Paris"],
                &["f2", "f4", "Berlin", "Barcelona"],
                &["f2", "f4", "Paris", "Barcelona"],
                &["f4", "f1", "Barcelona", "Amsterdam"],
            ]),
            "{language:?}"
        );
        let plan = explain(&session, language, query);
        let lines = lines(&plan);
        for hint in [
            "Filter (model_src.source_identifier Eq graph_src.id) \
             [hash join: model_src.source_identifier = graph_src.id]",
            "Filter (model_tgt.source_identifier Eq graph_tgt.id) \
             [hash join: model_tgt.source_identifier = graph_tgt.id]",
        ] {
            assert!(lines.contains(&hint), "{language:?}:\n{plan}");
        }
    }
}

/// `PROFILE` lists the scan the join absorbs as `HashJoin`, and the filter
/// above it with the rows it returns.
#[test]
fn profile_shows_the_hash_join() {
    let db = files();
    let session = db.session();
    for language in LANGUAGES {
        let profile = plan_text(&session, language, &format!("PROFILE {PATH_JOIN}"));
        let lines = lines(&profile);
        assert!(
            lines
                .iter()
                .any(|line| line.starts_with("HashJoin (t:TypeDefinition)  rows=")),
            "{language:?}:\n{profile}"
        );
        assert!(
            lines
                .iter()
                .any(|line| line.starts_with("Filter (t.filePath Eq f.path)  rows=3 ")),
            "{language:?}:\n{profile}"
        );
        assert!(
            !lines.iter().any(|line| line.starts_with("NestedLoopJoin")),
            "{language:?}:\n{profile}"
        );
    }
}

/// `files` with Alix and Gus also `Graph`, and Django `Graph` only, in
/// `a.rs`: of the type definitions in `a.rs` and `b.rs`, only Gus and Alix
/// have both labels.
fn graph_files() -> GrafeoDB {
    let db = files();
    db.execute_cypher("MATCH (t:TypeDefinition {name: 'Alix'}) SET t:Graph")
        .unwrap();
    db.execute_cypher("MATCH (t:TypeDefinition {name: 'Gus'}) SET t:Graph")
        .unwrap();
    db.execute("INSERT (:Graph {name: 'Django', filePath: 'a.rs'})")
        .unwrap();
    db
}

/// The other labels of a pattern are checks of the scanned node. Where the
/// key's filter sits above them (a key from an `UNWIND` row), the join runs
/// them on the nodes it scans: the filter above the join reads only the
/// key, so a node without one of the labels (Django, Vincent) must not get
/// there. `EXPLAIN` shows no hint of their own.
#[test]
fn the_checks_of_more_labels_run_on_the_scanned_nodes() {
    let db = graph_files();
    let session = db.session();
    for pattern in ["t:Graph:TypeDefinition", "t:TypeDefinition:Graph"] {
        let query = format!(
            "UNWIND [{{path: 'a.rs'}}, {{path: 'b.rs'}}, {{path: 'c.rs'}}] AS r \
             MATCH ({pattern}) WHERE t.filePath = r.path RETURN r.path, t.name"
        );
        for language in LANGUAGES {
            assert_rows_and_plan(
                &session,
                language,
                &query,
                &table(&[&["a.rs", "Gus"], &["b.rs", "Alix"]]),
                Some("t.filePath = r.path"),
            );
            let plan = explain(&session, language, &query);
            let lines = lines(&plan);
            let join = lines
                .iter()
                .position(|line| line.ends_with("[hash join: t.filePath = r.path]"))
                .expect("a hash join");
            let check = lines[join + 1];
            assert!(
                check.starts_with("Filter (hasLabel(t, ") && check.ends_with(')'),
                "{language:?} `{query}`:\n{plan}"
            );
            assert!(
                lines[join + 2].starts_with("NodeScan (t:"),
                "{language:?} `{query}`:\n{plan}"
            );
        }
    }
}

/// Where the optimizer moves the key's filter below the checks of the other
/// labels (a key from an earlier `MATCH`), the checks filter the joined rows.
#[test]
fn the_labels_of_a_later_pattern_check_the_joined_rows() {
    let db = graph_files();
    let session = db.session();
    for pattern in ["t:Graph:TypeDefinition", "t:TypeDefinition:Graph"] {
        let query = format!(
            "MATCH (f:File) MATCH ({pattern}) WHERE t.filePath = f.path RETURN f.path, t.name"
        );
        for language in LANGUAGES {
            assert_rows_and_plan(
                &session,
                language,
                &query,
                &table(&[&["a.rs", "Gus"], &["b.rs", "Alix"]]),
                Some("t.filePath = f.path"),
            );
            let plan = explain(&session, language, &query);
            let lines = lines(&plan);
            let check = lines
                .iter()
                .position(|line| line.starts_with("Filter (hasLabel(t, ") && line.ends_with(')'));
            let join = lines.iter().position(|line| line.contains("[hash join"));
            assert!(
                check.is_some() && check < join,
                "{language:?} `{query}`:\n{plan}"
            );
        }
    }
}

/// A statement that writes keeps the scan per row: a write above the join
/// can change keys the later input rows read (the scan per row reads them
/// when each chunk of rows arrives, a hash join would key them once). Here
/// the first 2048 input rows move the keys of the ten nodes by 2048, and the
/// next rows meet them there again: 20 matches.
#[test]
fn a_statement_that_writes_keeps_the_scan_per_row() {
    let db = GrafeoDB::new_in_memory();
    db.execute("UNWIND range(0, 9) AS k INSERT (:T {k: k})")
        .unwrap();
    let session = db.session();
    let query = "UNWIND range(0, 2999) AS i WITH {k: i} AS r MATCH (t:T) WHERE t.k = r.k \
                 SET t.k = t.k + 2048 RETURN count(*) AS n";
    assert!(!explain(&session, Language::Cypher, query).contains("[hash join"));
    assert_eq!(run(&session, Language::Cypher, query), [[Value::Int64(20)]]);
    let mut keys: Vec<Value> = run(&session, Language::Cypher, "MATCH (t:T) RETURN t.k")
        .into_iter()
        .map(|row| row[0].clone())
        .collect();
    keys.sort_by_key(|key| format!("{key:?}"));
    assert_eq!(keys, (4096..4106).map(Value::Int64).collect::<Vec<_>>());

    // Writes that do not touch the keys keep the scan per row too.
    for (language, query, expected) in [
        (
            Language::Cypher,
            "MATCH (f:File) MATCH (t:TypeDefinition) WHERE t.filePath = f.path \
             CREATE (:Seen {path: f.path}) RETURN f.path, t.name",
            definitions(),
        ),
        (
            Language::Gql,
            "MATCH (f:File) MATCH (t:TypeDefinition) WHERE t.filePath = f.path \
             INSERT (:Seen {path: f.path}) RETURN f.path, t.name",
            definitions(),
        ),
        (
            Language::Cypher,
            "MATCH (f:File) MATCH (t:TypeDefinition) WHERE t.filePath = f.path \
             DETACH DELETE t RETURN f.path",
            table(&[&["a.rs"], &["b.rs"], &["b.rs"]]),
        ),
    ] {
        let db = files();
        let session = db.session();
        assert_rows_and_plan(&session, language, query, &expected, None);
    }
}

/// A function whose value changes between calls stays in a filter of its
/// own above the join, which keeps its key; the rows are those of the scan
/// per row.
#[test]
fn a_volatile_conjunct_stays_above_the_join() {
    let db = files();
    let session = db.session();
    let query = "MATCH (f:File) MATCH (t:TypeDefinition) WHERE t.filePath = f.path \
                 AND rand() < 2 RETURN f.path, t.name";
    for language in LANGUAGES {
        assert_rows_and_plan(
            &session,
            language,
            query,
            &definitions(),
            Some("t.filePath = f.path"),
        );
        let plan = explain(&session, language, query);
        let lines = lines(&plan);
        let volatile = lines
            .iter()
            .position(|line| line.starts_with("Filter (rand()"));
        let join = lines.iter().position(|line| line.contains("[hash join"));
        assert!(
            volatile.is_some() && volatile < join,
            "{language:?}:\n{plan}"
        );
    }
}

/// A scan after a write reads what the write did, so it stays a scan per row
/// that reads its whole input first: here the input creates a file and a
/// type definition, changes a key, and removes a label.
#[test]
fn a_write_before_the_scan_keeps_the_scan_per_row() {
    let cases = [
        (
            Language::Gql,
            "INSERT (:File {path: 'n.rs'}), (:TypeDefinition {name: 'Butch', filePath: 'n.rs'}) \
             WITH 1 AS one MATCH (f:File) MATCH (t:TypeDefinition {filePath: f.path}) \
             RETURN f.path, t.name",
            table(&[
                &["a.rs", "Gus"],
                &["b.rs", "Alix"],
                &["b.rs", "Vincent"],
                &["n.rs", "Butch"],
            ]),
        ),
        (
            Language::Cypher,
            "CREATE (:File {path: 'n.rs'}), (:TypeDefinition {name: 'Butch', filePath: 'n.rs'}) \
             WITH 1 AS one MATCH (f:File) MATCH (t:TypeDefinition) WHERE t.filePath = f.path \
             RETURN f.path, t.name",
            table(&[
                &["a.rs", "Gus"],
                &["b.rs", "Alix"],
                &["b.rs", "Vincent"],
                &["n.rs", "Butch"],
            ]),
        ),
        (
            Language::Cypher,
            "MATCH (g:TypeDefinition {name: 'Mia'}) SET g.filePath = 'c.rs' \
             WITH g MATCH (f:File) MATCH (t:TypeDefinition) WHERE t.filePath = f.path \
             RETURN f.path, t.name",
            table(&[
                &["a.rs", "Gus"],
                &["b.rs", "Alix"],
                &["b.rs", "Vincent"],
                &["c.rs", "Mia"],
            ]),
        ),
        (
            Language::Cypher,
            "MATCH (f:File {path: 'b.rs'}) MATCH (g:TypeDefinition {name: 'Alix'}) \
             REMOVE g:TypeDefinition \
             WITH f MATCH (t:TypeDefinition) WHERE t.filePath = f.path RETURN f.path, t.name",
            table(&[&["b.rs", "Vincent"]]),
        ),
    ];
    for (language, query, expected) in cases {
        let db = files();
        let session = db.session();
        assert_rows_and_plan(&session, language, query, &expected, None);
    }
}

/// With an index on the key, the node is looked up per row instead.
#[test]
fn an_indexed_key_is_looked_up() {
    let db = files();
    db.create_property_index("filePath").unwrap();
    let session = db.session();
    for (language, query) in path_joins() {
        assert_rows_and_plan(&session, language, query, &definitions(), None);
        let plan = explain(&session, language, query);
        assert!(
            plan.contains("[index: filePath]"),
            "{language:?} `{query}`:\n{plan}"
        );
    }
}

/// In an open transaction the join sees the nodes, labels and values the
/// transaction changed, on both sides; after a rollback, those from before.
/// Django and a draft of `b.rs` get the labels of a type definition and a
/// file in it, Alix and `a.rs` lose them, Jules and `c.rs` get other keys.
#[test]
fn a_transaction_sees_what_it_changed() {
    let db = files();
    db.execute("INSERT (:Graph {name: 'Django', filePath: 'b.rs'}), (:Draft {path: 'b.rs'})")
        .unwrap();
    let mut session = db.session();
    session.begin_transaction().unwrap();
    for change in [
        "CREATE (:File {path: 'n.rs'}), (:TypeDefinition {name: 'Butch', filePath: 'n.rs'})",
        "MATCH (t:TypeDefinition {name: 'Alix'}) REMOVE t:TypeDefinition",
        "MATCH (t {name: 'Django'}) SET t:TypeDefinition",
        "MATCH (t {name: 'Jules'}) SET t.filePath = 'b.rs'",
        "MATCH (f:File {path: 'c.rs'}) SET f.path = 'z.rs'",
        "MATCH (f:File {path: 'a.rs'}) REMOVE f:File",
        "MATCH (f:Draft) SET f:File",
    ] {
        session.execute_cypher(change).unwrap();
    }
    let changed = table(&[
        &["b.rs", "Vincent"],
        &["b.rs", "Jules"],
        &["b.rs", "Django"],
        &["z.rs", "Mia"],
        &["b.rs", "Vincent"],
        &["b.rs", "Jules"],
        &["b.rs", "Django"],
        &["n.rs", "Butch"],
    ]);
    for language in LANGUAGES {
        assert_rows_and_plan(
            &session,
            language,
            PATH_JOIN,
            &changed,
            Some("t.filePath = f.path"),
        );
    }
    session.rollback().unwrap();
    for language in LANGUAGES {
        assert_eq!(
            run(&session, language, PATH_JOIN),
            definitions(),
            "{language:?} after the rollback"
        );
    }
}

/// No rows before the scan, no nodes to scan, or no node with the key: no
/// rows, also when one side's key is always NULL.
#[test]
fn an_empty_side_joins_nothing() {
    let db = files();
    let session = db.session();
    for query in [
        "MATCH (f:Missing) MATCH (t:TypeDefinition) WHERE t.filePath = f.path RETURN f.path, t.name",
        "MATCH (f:File) MATCH (t:Missing) WHERE t.filePath = f.path RETURN f.path, t.name",
        "MATCH (f:File) WHERE f.lines > 100 MATCH (t:TypeDefinition) WHERE t.filePath = f.path \
         RETURN f.path, t.name",
        "MATCH (f:File) MATCH (t:TypeDefinition) WHERE t.filePath = f.missing RETURN f.path, t.name",
        "MATCH (f:File) MATCH (t:TypeDefinition) WHERE t.missing = f.path RETURN f.path, t.name",
    ] {
        for language in LANGUAGES {
            if matches!(language, Language::Gql) && query.contains("WHERE f.lines") {
                continue;
            }
            assert!(
                run(&session, language, query).is_empty(),
                "{language:?} `{query}`"
            );
            let plan = explain(&session, language, query);
            assert!(
                plan.contains("[hash join"),
                "{language:?} `{query}`:\n{plan}"
            );
        }
    }
}

/// A subquery beside the key stays where it is written, in a filter above
/// the join on the key; the rows are those of the scan per row.
#[test]
fn a_subquery_beside_the_key_filters_the_joined_rows() {
    let db = files();
    db.execute_cypher(
        "MATCH (a:TypeDefinition {name: 'Alix'}), (g:TypeDefinition {name: 'Gus'}) \
         CREATE (a)-[:USES]->(g)",
    )
    .unwrap();
    let session = db.session();
    for query in [
        "MATCH (f:File) MATCH (t:TypeDefinition) \
         WHERE t.filePath = f.path AND EXISTS { MATCH (t)-[:USES]->(u) WHERE u.line = 5 } \
         RETURN f.path, t.name",
        "MATCH (f:File) MATCH (t:TypeDefinition) \
         WHERE t.filePath = f.path AND COUNT { MATCH (t)-[:USES]->() } = 1 RETURN f.path, t.name",
    ] {
        for language in LANGUAGES {
            assert_rows_and_plan(
                &session,
                language,
                query,
                &table(&[&["b.rs", "Alix"]]),
                Some("t.filePath = f.path"),
            );
            let plan = explain(&session, language, query);
            let lines = lines(&plan);
            let subquery = lines.iter().position(|line| line.contains("Subquery("));
            let join = lines.iter().position(|line| line.contains("[hash join"));
            assert!(
                subquery.is_some() && subquery < join,
                "{language:?} `{query}`:\n{plan}"
            );
        }
    }
}

/// A read of an earlier epoch joins the nodes and values of that epoch.
#[cfg(feature = "temporal")]
#[test]
fn a_read_of_an_earlier_epoch_joins_the_values_of_then() {
    let db = files();
    let before = db.current_epoch();
    db.execute_cypher("MATCH (t:TypeDefinition {name: 'Mia'}) SET t.filePath = 'a.rs'")
        .unwrap();
    db.execute_cypher("MATCH (t:TypeDefinition {name: 'Alix'}) SET t.filePath = 'z.rs'")
        .unwrap();
    db.execute("INSERT (:TypeDefinition {name: 'Butch', filePath: 'c.rs'})")
        .unwrap();
    let session = db.session();
    let query = PATH_JOIN;
    assert_eq!(
        session
            .execute_at_epoch(query, before)
            .unwrap()
            .rows()
            .to_vec(),
        definitions()
    );
    // No index is involved, so the join runs at the earlier epoch too.
    let plan = session
        .execute_at_epoch(&format!("EXPLAIN {query}"), before)
        .unwrap();
    assert!(
        format!("{:?}", plan.rows()).contains("[hash join: t.filePath = f.path]"),
        "{:?}",
        plan.rows()
    );
    assert_eq!(
        run(&session, Language::Gql, query),
        table(&[
            &["a.rs", "Gus"],
            &["a.rs", "Mia"],
            &["b.rs", "Vincent"],
            &["c.rs", "Butch"],
        ])
    );
}

// ---------------------------------------------------------------------------
// EXPLAIN names the path that runs
// ---------------------------------------------------------------------------

/// The hint of a filter over a scan per row: the label-first scan and the
/// range scan read a whole label once and do not run for a scan per row.
#[test]
fn a_scan_per_row_shows_no_label_first_or_range_hint() {
    let db = files();
    let session = db.session();
    for indexed in [false, true] {
        if indexed {
            db.create_property_index("line").unwrap();
        }
        for query in [
            "MATCH (f:File) MATCH (t:TypeDefinition) WHERE t.line > 3 RETURN f.path, t.name",
            "MATCH (f:File) MATCH (t:TypeDefinition) WHERE t.name STARTS WITH 'A' \
             RETURN f.path, t.name",
        ] {
            for language in LANGUAGES {
                let plan = explain(&session, language, query);
                assert!(
                    lines(&plan)
                        .iter()
                        .filter(|line| line.starts_with("Filter (t."))
                        .all(|line| line.ends_with(')')),
                    "{language:?} `{query}` (index: {indexed}):\n{plan}"
                );
            }
        }
    }
}

/// Documents with an embedding and a body. With `indexed`, a vector index on
/// the embedding and a text index on the body.
#[cfg(all(feature = "vector-index", feature = "text-index"))]
fn documents(indexed: bool) -> GrafeoDB {
    let db = GrafeoDB::new_in_memory();
    for (name, embedding, body) in [
        ("Alix", [1.0_f32, 0.0], "graph databases"),
        ("Gus", [0.0, 1.0], "vector search"),
        ("Vincent", [0.9, 0.1], "graph queries"),
    ] {
        db.create_node_with_props(
            &["Doc"],
            [
                ("name", Value::from(name)),
                ("emb", Value::Vector(embedding.to_vec().into())),
                ("body", Value::from(body)),
            ],
        )
        .unwrap();
    }
    if indexed {
        db.create_vector_index("Doc", "emb", Some(2), Some("cosine"), None, None, None)
            .unwrap();
        db.create_text_index("Doc", "body").unwrap();
    }
    db
}

/// A vector or text predicate that an index search takes shows no
/// `[label-first]`: the search runs instead of the scan (`PROFILE` names it).
/// Without the indexes the label is scanned and the filter runs above it.
#[cfg(all(feature = "vector-index", feature = "text-index"))]
#[test]
fn an_index_search_shows_no_label_first() {
    let queries = [
        (
            "MATCH (d:Doc) WHERE cosine_similarity(d.emb, [1.0, 0.0]) > 0.5 RETURN d.name",
            "VectorScan",
        ),
        (
            "MATCH (d:Doc) WHERE text_score(d.body, 'graph') > 0.0 RETURN d.name",
            "TextScan",
        ),
        (
            "MATCH (d:Doc) WHERE cosine_similarity(d.emb, [1.0, 0.0]) > 0.5 \
             AND text_score(d.body, 'graph') > 0.0 RETURN d.name",
            "VectorScan",
        ),
    ];
    for indexed in [true, false] {
        let db = documents(indexed);
        let session = db.session();
        for (query, search) in queries {
            let plan = explain(&session, Language::Gql, query);
            let filter = lines(&plan)
                .into_iter()
                .find(|line| line.starts_with("Filter ("))
                .expect("a filter")
                .to_string();
            let profile = plan_text(&session, Language::Gql, &format!("PROFILE {query}"));
            let searched = lines(&profile).iter().any(|line| line.starts_with(search));
            if indexed {
                assert!(filter.ends_with(')'), "`{query}`:\n{plan}");
                assert!(searched, "`{query}`:\n{profile}");
            } else {
                assert!(filter.ends_with(") [label-first]"), "`{query}`:\n{plan}");
                assert!(!searched, "`{query}`:\n{profile}");
            }
        }
    }
}

/// A range of constants on a scan without input runs as a range scan, with
/// or without an index, and `EXPLAIN` says so; with an equality of a
/// constant beside it, the label-first scan runs.
#[test]
fn a_range_without_an_index_shows_the_range_scan() {
    let db = files();
    let session = db.session();
    for language in LANGUAGES {
        let range = "MATCH (t:TypeDefinition) WHERE t.line > 3 RETURN t.name";
        let plan = explain(&session, language, range);
        assert!(
            lines(&plan).contains(&"Filter (t.line Gt 3) [range: line]"),
            "{language:?}:\n{plan}"
        );
        let profile = plan_text(&session, language, &format!("PROFILE {range}"));
        assert!(
            lines(&profile)
                .iter()
                .any(|line| line.starts_with("RangeScan (t.line Gt 3)")),
            "{language:?}:\n{profile}"
        );

        let both = "MATCH (t:TypeDefinition) WHERE t.line > 3 AND t.name = 'Alix' RETURN t.name";
        let plan = explain(&session, language, both);
        assert!(plan.contains("[label-first]"), "{language:?}:\n{plan}");
    }
}
