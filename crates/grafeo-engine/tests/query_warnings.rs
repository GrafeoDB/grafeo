//! Query warnings (0.6.2): one registry (`grafeo_common::utils::warning`), GQLSTATUS class 01 with
//! an implementation-defined subclass (ISO/IEC 39075 23.1), one warning per kind per statement,
//! on `QueryResult::warnings`. Translate-time forms warn when written, run-time forms when
//! evaluated; a statement that fails returns its error only.
//!
//! The catalogue tests run over `Warning::ALL`, so a new warning fails here until it has a code,
//! a message, a docs row and a trigger: one registry entry and one row in [`TRIGGERS`].
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test main query_warnings::
//! ```

#![cfg(all(feature = "gql", feature = "cypher"))]

use std::collections::HashMap;
use std::sync::Arc;

use grafeo_common::types::Value;
use grafeo_common::utils::warning::{QueryWarning, RETIRED, Warning, WarningKind};
use grafeo_engine::GrafeoDB;

/// A query that raises each live warning exactly once, by its code, after its setup (GQL, empty
/// for none) on a new database; a warning added to the registry fails
/// `every_warning_has_a_trigger_that_raises_it_once` until it has a row.
const TRIGGERS: &[(&str, &str, &str, &str)] = &[
    (
        "01X01",
        "gql",
        "",
        "RETURN ZONED DATETIME '2024-01-01T10:00:00+01:00' AS v",
    ),
    ("01X02", "gql", "", "RETURN [1] + [2] AS v"),
    (
        "01X03",
        "gql",
        "",
        "RETURN ZONED TIME '10:00:00+01:00' AS v",
    ),
    // A graph whose scores move after one iteration (an empty one has nothing to iterate).
    (
        "01Y01",
        "gql",
        "INSERT (a:V)-[:E]->(b:V)-[:E]->(c:V)-[:E]->(a), (a)-[:E]->(c)",
        "CALL grafeo.pagerank({max_iterations: 1, tolerance: 1e-15}) YIELD node_id",
    ),
];

/// The tracked page of deprecations, one row per live deprecation's code.
const DEPRECATIONS_PAGE: &str = include_str!("../../../docs/upgrading/deprecations.md");

/// The warning codes of `query` in `language`, in the order the result gives them.
fn codes(
    db: &GrafeoDB,
    language: &str,
    query: &str,
    params: Option<HashMap<String, Value>>,
) -> Vec<String> {
    let result = db
        .execute_language(query, language, params)
        .unwrap_or_else(|error| panic!("{language} `{query}`: {error}"));
    result
        .warnings()
        .iter()
        .map(|w| w.code.as_str().to_string())
        .collect()
}

fn gql(db: &GrafeoDB, query: &str) -> Vec<String> {
    codes(db, "gql", query, None)
}

/// The live warnings, the test probe left out.
fn live() -> Vec<Warning> {
    Warning::ALL
        .iter()
        .copied()
        .filter(|w| !w.spec().code.as_str().starts_with("01Z"))
        .collect()
}

// ---------------------------------------------------------------------------
// The catalogue
// ---------------------------------------------------------------------------

/// Each code is a warning (class 01) with an implementation-defined subclass (its first character
/// `5` to `9` or `I` to `Z`), and no code is used twice, live or retired.
#[test]
fn every_code_is_an_implementation_defined_warning_used_once() {
    let mut seen: Vec<String> = Vec::new();
    for warning in Warning::ALL {
        let code = warning.spec().code.as_str().to_string();
        assert!(
            code.starts_with("01"),
            "{warning:?}: {code} is not class 01"
        );
        let subclass = code.as_bytes()[2];
        assert!(
            matches!(subclass, b'5'..=b'9' | b'I'..=b'Z'),
            "{warning:?}: {code} is not implementation-defined"
        );
        seen.push(code);
    }
    for retired in RETIRED {
        seen.push(retired.as_str().to_string());
    }
    let mut unique = seen.clone();
    unique.sort();
    unique.dedup();
    assert_eq!(unique.len(), seen.len(), "a code used twice: {seen:?}");
}

/// Each message formats with no placeholder left; a deprecation names its release, the release
/// that refuses it, and its replacement.
#[test]
fn every_message_is_complete() {
    for warning in Warning::ALL {
        let spec = warning.spec();
        let message = warning.message();
        assert!(!message.is_empty(), "{warning:?}");
        assert!(
            !message.contains('{') && !message.contains('}'),
            "{warning:?}: a placeholder is left in `{message}`"
        );
        assert!(
            message.contains(spec.form),
            "{warning:?}: `{message}` names no form"
        );
        if let Some(deprecation) = &spec.deprecation {
            assert_eq!(spec.kind, WarningKind::Deprecation, "{warning:?}");
            for part in [
                deprecation.since,
                deprecation.removed_in,
                deprecation.replacement,
            ] {
                assert!(
                    message.contains(part),
                    "{warning:?}: `{message}` lacks `{part}`"
                );
            }
        }
    }
}

/// Each live deprecation has a row on the tracked deprecations page (a notice, such as
/// `NotConverged`, has none).
#[test]
fn every_deprecation_has_a_row_on_the_deprecations_page() {
    for warning in live()
        .into_iter()
        .filter(|w| w.spec().kind == WarningKind::Deprecation)
    {
        let code = warning.spec().code.as_str().to_string();
        assert!(
            DEPRECATIONS_PAGE
                .lines()
                .any(|line| line.starts_with('|') && line.contains(&code)),
            "{warning:?}: no row for {code} in docs/upgrading/deprecations.md"
        );
    }
}

/// Each live warning has a trigger in [`TRIGGERS`], which raises it and nothing else, once.
#[test]
fn every_warning_has_a_trigger_that_raises_it_once() {
    for warning in live() {
        let code = warning.spec().code.as_str().to_string();
        let (_, language, setup, query) = TRIGGERS
            .iter()
            .find(|(c, _, _, _)| *c == code)
            .unwrap_or_else(|| panic!("{warning:?}: no trigger for {code} in TRIGGERS"));
        let db = GrafeoDB::new_in_memory();
        if !setup.is_empty() {
            db.execute(setup)
                .unwrap_or_else(|error| panic!("{code}'s setup: {error}"));
        }
        assert_eq!(
            codes(&db, language, query, None),
            std::slice::from_ref(&code),
            "{query}"
        );
    }
    assert_eq!(
        TRIGGERS.len(),
        live().len(),
        "a trigger for a warning that is not live"
    );
}

/// The result's warning carries the registry's code, kind and message.
#[test]
fn a_warning_carries_its_code_kind_and_message() {
    let db = GrafeoDB::new_in_memory();
    let result = db
        .execute("RETURN ZONED DATETIME '2024-01-01T10:00:00+01:00' AS v")
        .unwrap();
    let [warning] = result.warnings() else {
        panic!("one warning: {:?}", result.warnings());
    };
    let zoned = Warning::ALL
        .iter()
        .find(|w| w.spec().code.as_str() == "01X01")
        .expect("ZONED in the registry");
    assert_eq!(warning.code.as_str(), "01X01");
    assert_eq!(warning.kind, WarningKind::Deprecation);
    assert_eq!(warning.message, zoned.message());
    // The status of a statement with a warning stays as before (a 0.7.0 question).
    assert_eq!(result.gql_status.as_str(), "00000");
}

/// `ZONED TIME '...'` is a deprecation of its own (01X03), apart from `ZONED DATETIME` (01X01):
/// its own message, and its replacement a `TIME '...'` with an offset, which gives the very value.
#[test]
fn a_zoned_time_literal_has_its_own_warning_and_equals_its_replacement() {
    let find = |code: &str| {
        Warning::ALL
            .iter()
            .copied()
            .find(|w| w.spec().code.as_str() == code)
            .unwrap_or_else(|| panic!("{code} in the registry"))
    };
    let (datetime, time) = (find("01X01"), find("01X03"));
    assert_eq!(time.spec().kind, WarningKind::Deprecation);
    assert_ne!(datetime.message(), time.message(), "each its own message");
    assert!(time.message().contains("TIME"), "{}", time.message());
    let deprecation = time.spec().deprecation.expect("01X03 is a deprecation");
    assert_eq!(deprecation.since, "0.6.2");
    assert_eq!(deprecation.removed_in, "0.7.0");
    assert!(
        deprecation.replacement.contains("TIME '"),
        "{}",
        deprecation.replacement
    );

    let db = GrafeoDB::new_in_memory();
    let result = db
        .execute(
            "RETURN ZONED TIME '10:00:00+01:00' AS zoned, TIME '10:00:00+01:00' AS time, \
             ZONED TIME '10:00:00+01:00' = TIME '10:00:00+01:00' AS same",
        )
        .unwrap();
    let row = &result.rows()[0];
    assert_eq!(row[0], row[1], "the same value");
    assert_eq!(row[2], Value::Bool(true));
    let warnings: Vec<&str> = result.warnings().iter().map(|w| w.code.as_str()).collect();
    assert_eq!(warnings, ["01X03"], "the zoned time alone warns");
}

// ---------------------------------------------------------------------------
// The edge cases of the design
// ---------------------------------------------------------------------------

const ZONED: &str = "ZONED DATETIME '2024-01-01T10:00:00+01:00'";

#[test]
fn one_warning_per_kind_per_statement() {
    let db = GrafeoDB::new_in_memory();
    assert_eq!(gql(&db, &format!("RETURN {ZONED} AS v")), ["01X01"]);
    assert_eq!(
        gql(&db, &format!("RETURN {ZONED} AS a, {ZONED} AS b")),
        ["01X01"]
    );
    assert_eq!(gql(&db, "RETURN [1] + [2] AS a, [3] + [4] AS b"), ["01X02"]);
    // Both forms: two, ordered by code.
    assert_eq!(
        gql(&db, &format!("RETURN [1] + [2] AS a, {ZONED} AS b")),
        ["01X01", "01X02"]
    );
}

#[test]
fn cypher_list_plus_is_no_deprecation() {
    let db = GrafeoDB::new_in_memory();
    assert_eq!(
        codes(&db, "cypher", "RETURN [1] + [2] AS v", None),
        Vec::<String>::new()
    );
}

/// The run-time form warns from a property and from a parameter, and when folded at plan time.
#[test]
fn list_plus_warns_from_a_property_a_parameter_and_a_fold() {
    let db = GrafeoDB::new_in_memory();
    db.execute("INSERT (:L {l: [1, 2]})").unwrap();
    assert_eq!(gql(&db, "MATCH (n:L) RETURN n.l + [3] AS v"), ["01X02"]);
    let params = HashMap::from([("p".to_string(), Value::List(vec![Value::Int64(1)].into()))]);
    assert_eq!(
        codes(&db, "gql", "RETURN $p + [2] AS v", Some(params)),
        ["01X02"]
    );
    assert_eq!(
        gql(&db, "RETURN [1] + [2] AS v"),
        ["01X02"],
        "both literal, folded"
    );
}

/// A run-time form warns when it is evaluated, a static form when it is written.
#[test]
fn a_branch_never_evaluated() {
    let db = GrafeoDB::new_in_memory();
    assert_eq!(
        gql(&db, "RETURN CASE WHEN false THEN [1] + [2] END AS v"),
        Vec::<String>::new()
    );
    assert_eq!(
        gql(
            &db,
            &format!("RETURN CASE WHEN false THEN {ZONED} END AS v")
        ),
        ["01X01"]
    );
    // A statement with no rows, the run-time form only in its filter.
    db.execute("INSERT (:L {l: [1]})").unwrap();
    assert_eq!(
        gql(&db, "MATCH (n:Nothing) WHERE n.l + [1] = [1] RETURN n"),
        Vec::<String>::new()
    );
}

/// A statement that fails returns its error only.
#[test]
fn an_error_carries_no_warnings() {
    let db = GrafeoDB::new_in_memory();
    let error = db
        .execute(&format!("RETURN {ZONED} AS a, [1] + [2] AS b, 1 / 0 AS c"))
        .expect_err("division by zero");
    assert!(
        error.to_string().to_lowercase().contains("division"),
        "{error}"
    );
}

/// The translate-time warning on each result of a cached plan, the run-time one on each run.
#[test]
fn a_cached_plan_warns_on_every_run() {
    let db = GrafeoDB::new_in_memory();
    db.execute("INSERT (:L {l: [1]})").unwrap();
    let query = format!("MATCH (n:L) RETURN {ZONED} AS z, n.l + [2] AS v");
    for run in 0..3 {
        assert_eq!(gql(&db, &query), ["01X01", "01X02"], "run {run}");
    }
}

#[test]
fn explain_warns_at_translate_time_only_and_profile_both() {
    let db = GrafeoDB::new_in_memory();
    db.execute("INSERT (:L {l: [1]})").unwrap();
    let query = format!("MATCH (n:L) RETURN {ZONED} AS z, n.l + [2] AS v");
    assert_eq!(gql(&db, &format!("EXPLAIN {query}")), ["01X01"]);
    assert_eq!(gql(&db, &format!("PROFILE {query}")), ["01X01", "01X02"]);
}

/// A form in a nested body warns on the outer statement.
#[test]
fn a_nested_body_warns_on_the_outer_statement() {
    let db = GrafeoDB::new_in_memory();
    db.execute("INSERT (:L {l: [1]}), (:L {l: [2]})").unwrap();
    assert_eq!(
        gql(
            &db,
            "MATCH (n:L) WHERE EXISTS { MATCH (m:L) WHERE m.l + [1] = [1, 1] } RETURN n.l"
        ),
        ["01X02"]
    );
    assert_eq!(
        gql(
            &db,
            "MATCH (n:L) CALL { WITH n RETURN n.l + [9] AS k } RETURN k"
        ),
        ["01X02"]
    );
}

/// In one transaction each statement has its own warnings.
#[test]
fn each_statement_of_a_transaction_has_its_own() {
    let db = GrafeoDB::new_in_memory();
    let mut session = db.session();
    session.begin_transaction().unwrap();
    let first = session.execute(&format!("RETURN {ZONED} AS v")).unwrap();
    let second = session.execute("RETURN 1 AS v").unwrap();
    let third = session.execute("RETURN [1] + [2] AS v").unwrap();
    session.commit().unwrap();
    let codes = |r: &grafeo_engine::database::QueryResult| -> Vec<String> {
        r.warnings()
            .iter()
            .map(|w| w.code.as_str().to_string())
            .collect()
    };
    assert_eq!(codes(&first), ["01X01"]);
    assert_eq!(codes(&second), Vec::<String>::new());
    assert_eq!(codes(&third), ["01X02"]);
}

/// Every worker raising gives one warning.
#[test]
fn many_rows_and_workers_give_one_warning() {
    let db = GrafeoDB::new_in_memory();
    assert_eq!(
        gql(
            &db,
            "UNWIND range(1, 200000) AS i RETURN count([i] + [i]) AS n"
        ),
        ["01X02"]
    );
}

#[test]
fn no_warning_is_an_empty_list() {
    let db = GrafeoDB::new_in_memory();
    for query in [
        "RETURN 1 AS v",
        "RETURN [1] || [2] AS v",
        "RETURN DATETIME '2024-01-01T10:00:00+01:00' AS v",
    ] {
        assert_eq!(gql(&db, query), Vec::<String>::new(), "{query}");
    }
}

// ---------------------------------------------------------------------------
// SQL/PGQ, the processor and streams
// ---------------------------------------------------------------------------

/// The registry's warning with `code`.
fn registered(code: &str) -> Warning {
    Warning::ALL
        .iter()
        .copied()
        .find(|w| w.spec().code.as_str() == code)
        .unwrap_or_else(|| panic!("{code} in the registry"))
}

fn codes_of(warnings: &[QueryWarning]) -> Vec<String> {
    warnings
        .iter()
        .map(|w| w.code.as_str().to_string())
        .collect()
}

fn list(items: &[i64]) -> Value {
    Value::List(
        items
            .iter()
            .map(|i| Value::Int64(*i))
            .collect::<Vec<_>>()
            .into(),
    )
}

/// `+` on two lists is deprecated in GQL and in SQL/PGQ, and the message says so.
#[test]
fn the_list_plus_deprecation_names_gql_and_sql_pgq() {
    let message = registered("01X02").message();
    assert!(message.contains("GQL and SQL/PGQ"), "{message}");
    assert!(message.contains("`||`"), "{message}");
}

/// SQL/PGQ joins lists and strings with `||` (SQL:2023 concatenation): a null operand gives
/// null, and `||` binds tighter than `=`. None of that warns. `+` on two lists still joins them,
/// with 01X02.
#[cfg(feature = "sql-pgq")]
#[test]
fn sql_pgq_joins_with_bars_and_warns_on_list_plus() {
    let db = GrafeoDB::new_in_memory();
    db.execute("INSERT (:L {l: [1, 2], s: 'Amster'})").unwrap();
    let sql =
        |columns: &str| format!("SELECT * FROM GRAPH_TABLE (MATCH (n:L) COLUMNS ({columns}))");
    let result = db
        .execute_sql(&sql(
            "n.l || [3] AS l, n.s || 'dam' AS s, n.l || n.none AS ln, \
             n.s || n.none AS sn, n.s || 'dam' = 'Amsterdam' AS p",
        ))
        .unwrap();
    assert_eq!(
        result.rows()[0],
        [
            list(&[1, 2, 3]),
            Value::from("Amsterdam"),
            Value::Null,
            Value::Null,
            Value::Bool(true)
        ]
    );
    assert_eq!(codes_of(result.warnings()), Vec::<String>::new());
    let result = db.execute_sql(&sql("n.l + [3] AS l")).unwrap();
    assert_eq!(result.rows()[0], [list(&[1, 2, 3])]);
    assert_eq!(codes_of(result.warnings()), ["01X02"]);
}

/// The query processor, without a session, gives a statement's warnings too: 01X01 from its
/// translation, 01X02 from its run, both ordered by code.
#[test]
fn the_processor_gives_translate_and_run_time_warnings() {
    use grafeo_core::graph::rowgroup::RowGroupStore;
    use grafeo_engine::query::{QueryLanguage, QueryProcessor};
    use grafeo_engine::transaction::TransactionManager;

    let store = Arc::new(RowGroupStore::new());
    store.create_node_with_props(&["L"], &[("l".into(), list(&[1]))]);
    let manager = Arc::new(TransactionManager::new());
    manager.sync_epoch(store.current_epoch());
    let processor = QueryProcessor::for_lpg_with_transaction(store, manager);
    for (query, expected) in [
        (format!("RETURN {ZONED} AS v"), vec!["01X01"]),
        (
            "MATCH (n:L) RETURN n.l + [3] AS v".to_string(),
            vec!["01X02"],
        ),
        (
            format!("MATCH (n:L) RETURN {ZONED} AS z, n.l + [3] AS v"),
            vec!["01X01", "01X02"],
        ),
    ] {
        let result = processor
            .process(&query, QueryLanguage::Gql, None)
            .unwrap_or_else(|error| panic!("{query}: {error}"));
        assert_eq!(result.rows().len(), 1, "{query}");
        assert_eq!(codes_of(result.warnings()), expected, "{query}");
    }
}

/// A stream shows a translate-time warning from the start and a run-time one complete once it
/// is exhausted, through the database's stream, the session's and the row iterators; after an
/// error it keeps what was raised before it.
#[test]
fn a_stream_shows_both_stages_and_keeps_them_after_an_error() {
    let db = GrafeoDB::new_in_memory();
    db.execute("UNWIND range(1, 10000) AS i INSERT (:L {i: i, l: [i]})")
        .unwrap();

    let stream = db
        .execute_streaming(&format!("MATCH (n:L) RETURN {ZONED} AS z"))
        .unwrap();
    assert_eq!(
        codes_of(&stream.warnings()),
        ["01X01"],
        "translate time, before a row is pulled"
    );
    drop(stream);

    let mut stream = db
        .execute_streaming("MATCH (n:L) RETURN n.l + [3] AS v")
        .unwrap();
    let mut rows = 0;
    while let Some(chunk) = stream.next_chunk().unwrap() {
        rows += chunk.len();
    }
    assert_eq!(rows, 10000);
    assert_eq!(
        codes_of(&stream.warnings()),
        ["01X02"],
        "run time, once exhausted"
    );

    let mut owned_rows = db
        .execute_streaming(&format!("MATCH (n:L) RETURN {ZONED} AS z, n.l + [3] AS v"))
        .unwrap()
        .into_row_iter();
    for row in &mut owned_rows {
        row.unwrap();
    }
    assert_eq!(
        codes_of(&owned_rows.warnings()),
        ["01X01", "01X02"],
        "the database stream's row iterator"
    );

    let session = db.session();
    let mut session_rows = session
        .execute_streaming(&format!("MATCH (n:L) RETURN {ZONED} AS z, n.l + [3] AS v"))
        .unwrap()
        .into_row_iter();
    for row in &mut session_rows {
        row.unwrap();
    }
    assert_eq!(
        codes_of(&session_rows.warnings()),
        ["01X01", "01X02"],
        "the session stream's row iterator"
    );
    drop(session_rows);

    // The last row divides by zero: every row before it has evaluated the run-time form.
    let mut stream = session
        .execute_streaming("MATCH (n:L) RETURN n.l + [3] AS v, 1 / (10000 - n.i) AS e")
        .unwrap();
    let error = loop {
        match stream.next_chunk() {
            Ok(Some(_)) => {}
            Ok(None) => panic!("the last row divides by zero"),
            Err(error) => break error,
        }
    };
    assert!(
        error.to_string().to_lowercase().contains("division"),
        "{error}"
    );
    assert_eq!(
        codes_of(&stream.warnings()),
        ["01X02"],
        "kept after the error"
    );
}

/// No form the engine builds for itself raises a deprecation the query did not write (named paths
/// were built with list `+`, so every 2-hop path raised 01X02): a query with no deprecated form
/// gives no warning, across named paths of every shape, lists and strings, in GQL and Cypher.
/// Each query returns a row, so every run-time form in it is evaluated.
#[test]
fn a_query_without_a_deprecated_form_raises_no_warning() {
    let db = GrafeoDB::new_in_memory();
    db.execute(
        "INSERT (a:V {i: 1, l: [1], s: 'a'})-[:E]->(b:V {i: 2, l: [2], s: 'b'})\
         -[:E]->(c:V {i: 3, l: [3], s: 'c'})-[:E]->(a)",
    )
    .unwrap();
    let gql = [
        "MATCH p = (a:V)-[:E]->(b:V) RETURN p",
        "MATCH p = (a:V)-[:E]->(b:V)-[:E]->(c:V) RETURN p",
        "MATCH p = (a:V)-[:E]->(b:V)-[:E]->(c:V)-[:E]->(d:V) \
         RETURN nodes(p) AS n, relationships(p) AS r, length(p) AS k",
        "MATCH p = (a:V {i: 1})-[:E]->{1,3}(b:V) RETURN p",
        "MATCH p = (a:V {i: 1})-[:E]->{0,1}(b:V) RETURN p",
        "MATCH p = ALL SHORTEST (a:V {i: 1})-[:E]->+(b:V {i: 3}) RETURN p",
        "MATCH p = ANY SHORTEST (a:V {i: 1})-[:E]->+(b:V {i: 3}) RETURN p",
        "MATCH p = (a:V)-[:E]->(b:V) RETURN [n IN nodes(p) | n.i] AS ids",
        "MATCH (n:V) RETURN n.l || [9] AS l, n.s || 'z' AS s, [x IN n.l | x + 1] AS m",
        "MATCH (n:V) RETURN collect(n.i) AS c",
        "MATCH (n:V) RETURN size(n.l) AS z, reverse(n.l) AS r",
    ];
    let cypher = [
        "MATCH p = (a:V)-[:E]->(b:V)-[:E]->(c:V) RETURN p",
        "MATCH p = (a:V {i: 1})-[:E*1..3]->(b:V) RETURN p",
        "MATCH p = (a:V {i: 1})-[:E*2]->(b:V) RETURN nodes(p) AS n, relationships(p) AS r",
        "MATCH p = shortestPath((a:V {i: 1})-[:E*]->(b:V {i: 3})) RETURN p",
        "MATCH (n:V) RETURN n.l + [9] AS l, n.s + 'z' AS s",
        "MATCH (n:V) RETURN [x IN n.l | x + 1] AS m",
    ];
    for (language, queries) in [("gql", &gql[..]), ("cypher", &cypher[..])] {
        for query in queries {
            let result = db
                .execute_language(query, language, None)
                .unwrap_or_else(|error| panic!("{language} `{query}`: {error}"));
            assert!(
                !result.rows().is_empty(),
                "{language} `{query}`: a row, so every run-time form is evaluated"
            );
            assert_eq!(
                codes_of(result.warnings()),
                Vec::<String>::new(),
                "{language} `{query}`"
            );
        }
    }
}
