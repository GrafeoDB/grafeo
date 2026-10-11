//! A statement that ends with a write and no `RETURN` has no result: no
//! columns and no rows. In ISO/IEC 39075:2024 the `<primitive result
//! statement>` (a `RETURN`, or `FINISH`) of a `<linear data-modifying
//! statement>` is optional (13.1), and without one, as with `FINISH` (14.10),
//! the statement completes with an omitted result (GQLSTATUS 00001,
//! "successful completion: omitted result", Clause 23). openCypher has the
//! same rule: a query that ends with an update returns nothing. Its write
//! counters still say what it changed. A Cypher query ending with `CREATE`,
//! `MERGE`, `SET`, `REMOVE`, `DELETE`, `FOREACH` or a unit `CALL` returned the
//! planner's internal columns (`__list__`, `i`, `_anon_0`) and a row of raw
//! IDs for each row it wrote; a GQL one (`FOR ... INSERT`, `MATCH ... SET`)
//! returned a row without columns for each, and `FINISH` returned no rows but
//! the internal columns. A GQL statement of one `INSERT` (or `CREATE`)
//! returned the last node it created in a column such as `_anon_0` (#580),
//! through every entry point the bindings use; its `EXPLAIN` and `PROFILE`
//! plans ended in that column, and a procedure whose body it is returned the
//! node in a row without columns. After `NEXT`, such a statement could not
//! read the rows of the one before, and wrote nothing after a statement
//! without a result, where it now reads one empty row, as at the start of a
//! statement (also after `FINISH`). A statement that ends
//! with a `RETURN` still returns it, and a standalone procedure call its
//! output (see `call_procedures`). The differential test corpus has these
//! cases too (section BR).
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test statement_without_a_result
//! ```

#![cfg(all(feature = "lpg", feature = "gql", feature = "cypher"))]

use std::collections::HashMap;

use grafeo_common::types::Value;
use grafeo_common::utils::error::Result;
use grafeo_engine::database::QueryResult;
use grafeo_engine::{GrafeoDB, Session};

#[derive(Debug, Clone, Copy)]
enum Language {
    Gql,
    Cypher,
}

fn execute(session: &Session, language: Language, query: &str) -> QueryResult {
    let result = match language {
        Language::Gql => session.execute(query),
        Language::Cypher => session.execute_cypher(query),
    };
    result.unwrap_or_else(|error| panic!("{language:?} `{query}` failed: {error}"))
}

/// Two `W` nodes with `k` 3 and 19.
const SETUP: &str = "INSERT (:W {k: 3}), (:W {k: 19})";

/// What a write changed, in the order of `WriteCounters`: nodes created and
/// deleted, edges created and deleted, properties set, labels added and
/// removed.
type Changes = [u64; 7];

fn changes(result: &QueryResult) -> Changes {
    let c = &result.counters;
    [
        c.nodes_created,
        c.nodes_deleted,
        c.edges_created,
        c.edges_deleted,
        c.properties_set,
        c.labels_added,
        c.labels_removed,
    ]
}

/// Each statement runs on a new database with the two `W` nodes: it returns
/// no columns and no rows, and its counters say what it wrote.
#[test]
fn a_write_without_return_has_no_result() {
    for (language, query, expected) in [
        // The case of the bug report.
        (
            Language::Cypher,
            "UNWIND [1, 2] AS i CREATE (:V)",
            [2, 0, 0, 0, 0, 2, 0],
        ),
        // A GQL statement of one INSERT returned the last node it created
        // (#580), also when GQL reads it as CREATE.
        (Language::Gql, "INSERT (:V {k: 88})", [1, 0, 0, 0, 1, 1, 0]),
        (Language::Gql, "INSERT (a:V), (b:V)", [2, 0, 0, 0, 0, 2, 0]),
        (
            Language::Gql,
            "INSERT (:V {k: 3})-[:R]->(:V)",
            [2, 0, 1, 0, 1, 2, 0],
        ),
        (Language::Gql, "CREATE (:V)", [1, 0, 0, 0, 0, 1, 0]),
        // Statements over several lines (#580: such a CREATE returned a row
        // without columns).
        (
            Language::Gql,
            "CREATE (a:V)\nCREATE (b:V)\nCREATE (a)-[:R]->(b)",
            [2, 0, 1, 0, 0, 2, 0],
        ),
        (
            Language::Cypher,
            "CREATE (a:V)\nCREATE (b:V)\nCREATE (a)-[:R]->(b)",
            [2, 0, 1, 0, 0, 2, 0],
        ),
        (Language::Cypher, "CREATE (:V)", [1, 0, 0, 0, 0, 1, 0]),
        (
            Language::Cypher,
            "CREATE (:V {k: 3})-[:R]->(:V)",
            [2, 0, 1, 0, 1, 2, 0],
        ),
        (
            Language::Cypher,
            "MERGE (:V {k: 88})",
            [1, 0, 0, 0, 1, 1, 0],
        ),
        (
            Language::Cypher,
            "UNWIND [3, 88] AS i MERGE (:W {k: i})",
            [1, 0, 0, 0, 1, 1, 0],
        ),
        (
            Language::Cypher,
            "MATCH (a:W {k: 3}), (b:W {k: 19}) MERGE (a)-[:R]->(b)",
            [0, 0, 1, 0, 0, 0, 0],
        ),
        (
            Language::Cypher,
            "MATCH (w:W) SET w.k = 88",
            [0, 0, 0, 0, 2, 0, 0],
        ),
        (
            Language::Cypher,
            "MATCH (w:W) SET w:V",
            [0, 0, 0, 0, 0, 2, 0],
        ),
        (
            Language::Cypher,
            "MATCH (w:W) REMOVE w.k",
            [0, 0, 0, 0, 2, 0, 0],
        ),
        (
            Language::Cypher,
            "MATCH (w:W) REMOVE w:W",
            [0, 0, 0, 0, 0, 0, 2],
        ),
        (
            Language::Cypher,
            "MATCH (w:W) DELETE w",
            [0, 2, 0, 0, 0, 0, 0],
        ),
        (
            Language::Cypher,
            "MATCH (w:W) DETACH DELETE w",
            [0, 2, 0, 0, 0, 0, 0],
        ),
        (
            Language::Cypher,
            "UNWIND [1, 2] AS i FOREACH (x IN [i] | CREATE (:V))",
            [2, 0, 0, 0, 0, 2, 0],
        ),
        (
            Language::Cypher,
            "UNWIND [1, 2] AS i CALL { CREATE (:V) }",
            [2, 0, 0, 0, 0, 2, 0],
        ),
        (
            Language::Gql,
            "FOR i IN [1, 2] INSERT (:V)",
            [2, 0, 0, 0, 0, 2, 0],
        ),
        (
            Language::Gql,
            "MATCH (w:W) SET w.k = 88",
            [0, 0, 0, 0, 2, 0, 0],
        ),
        (
            Language::Gql,
            "MATCH (w:W) REMOVE w.k",
            [0, 0, 0, 0, 2, 0, 0],
        ),
        (
            Language::Gql,
            "MATCH (w:W) DETACH DELETE w",
            [0, 2, 0, 0, 0, 0, 0],
        ),
        (
            Language::Gql,
            "FOR i IN [1, 2] INSERT (:V) FINISH",
            [2, 0, 0, 0, 0, 2, 0],
        ),
        (Language::Gql, "MATCH (w:W) FINISH", [0; 7]),
        // A GQL statement may end with a CALL that writes (it used to be
        // refused for want of a RETURN), also one whose body returns rows.
        (
            Language::Gql,
            "FOR i IN [1, 2] CALL (i) { INSERT (:V {i: i}) }",
            [2, 0, 0, 0, 2, 2, 0],
        ),
        (
            Language::Gql,
            "MATCH (w:W) CALL { INSERT (:V) }",
            [2, 0, 0, 0, 0, 2, 0],
        ),
        (
            Language::Gql,
            "MATCH (w:W) CALL (w) { SET w.k = 88 }",
            [0, 0, 0, 0, 2, 0, 0],
        ),
        (
            Language::Gql,
            "MATCH (w:W) CALL (w) { INSERT (v:V) RETURN v }",
            [2, 0, 0, 0, 0, 2, 0],
        ),
    ] {
        let session = GrafeoDB::new_in_memory().session();
        execute(&session, Language::Gql, SETUP);
        let result = execute(&session, language, query);
        assert_eq!(
            result.columns,
            Vec::<String>::new(),
            "{language:?} `{query}` has no columns"
        );
        assert_eq!(
            result.rows(),
            &[] as &[Vec<Value>],
            "{language:?} `{query}` has no rows"
        );
        assert_eq!(
            changes(&result),
            expected,
            "{language:?} `{query}` counts its writes"
        );
    }
}

/// The write runs to its end: every row of a long input writes, also past
/// the first chunk of rows, and in a transaction.
#[test]
fn a_write_without_return_writes_every_row() {
    for (language, query) in [
        (
            Language::Cypher,
            "UNWIND range(1, 3000) AS i CREATE (:V {i: i})",
        ),
        (Language::Gql, "FOR i IN range(1, 3000) INSERT (:V {i: i})"),
        (
            Language::Gql,
            "FOR i IN range(1, 3000) INSERT (:V {i: i}) FINISH",
        ),
    ] {
        let db = GrafeoDB::new_in_memory();
        let mut session = db.session();
        session.begin_transaction().unwrap();
        let result = execute(&session, language, query);
        assert!(
            result.columns.is_empty() && result.rows().is_empty(),
            "{language:?} `{query}` has no result"
        );
        assert_eq!(
            result.counters.nodes_created, 3000,
            "{language:?} `{query}`"
        );
        session.commit().unwrap();
        assert_eq!(
            execute(
                &session,
                Language::Gql,
                "MATCH (v:V) RETURN count(v), sum(v.i)"
            )
            .rows(),
            [[Value::Int64(3000), Value::Int64(4_501_500)]],
            "{language:?} `{query}` writes every row"
        );
    }
}

/// A statement that ends with a `RETURN` returns it, after a write too.
#[test]
fn a_write_with_return_keeps_its_result() {
    for (language, query) in [
        (
            Language::Cypher,
            "UNWIND [3, 19] AS i CREATE (v:V {k: i}) RETURN v.k AS k",
        ),
        (
            Language::Gql,
            "FOR i IN [3, 19] INSERT (v:V {k: i}) RETURN v.k AS k",
        ),
    ] {
        let session = GrafeoDB::new_in_memory().session();
        let result = execute(&session, language, query);
        assert_eq!(result.columns, ["k"], "{language:?} `{query}`");
        assert_eq!(
            result.rows(),
            [[Value::Int64(3)], [Value::Int64(19)]],
            "{language:?} `{query}`"
        );
        assert_eq!(result.counters.nodes_created, 2, "{language:?} `{query}`");
    }
}

/// The `k` of every node with `label`, in order.
fn ks(session: &Session, label: &str) -> Vec<Value> {
    execute(
        session,
        Language::Gql,
        &format!("MATCH (n:{label}) RETURN n.k AS k ORDER BY k"),
    )
    .rows()
    .iter()
    .map(|row| row[0].clone())
    .collect()
}

fn ints(values: &[i64]) -> Vec<Value> {
    values.iter().copied().map(Value::Int64).collect()
}

fn three() -> HashMap<String, Value> {
    HashMap::from([("k".to_string(), Value::Int64(3))])
}

/// Every entry point has no result for a statement of one INSERT (#580):
/// the Rust API, `execute_language` (which the Python, Node.js, C, Go, C#,
/// Dart and WebAssembly bindings call), with parameters, and in a
/// transaction, begun through the API or with GQL's `START TRANSACTION`.
#[test]
fn every_entry_point_has_no_result() {
    type Run = fn(&GrafeoDB) -> Result<QueryResult>;
    let runs: [(&str, Run); 13] = [
        ("GrafeoDB::execute", |db| db.execute("INSERT (:V {k: 3})")),
        ("GrafeoDB::execute_with_params", |db| {
            db.execute_with_params("INSERT (:V {k: $k})", three())
        }),
        ("GrafeoDB::execute_language gql", |db| {
            db.execute_language("INSERT (:V {k: 3})", "gql", None)
        }),
        ("GrafeoDB::execute_language gql with parameters", |db| {
            db.execute_language("INSERT (:V {k: $k})", "gql", Some(three()))
        }),
        ("GrafeoDB::execute_language cypher", |db| {
            db.execute_language("CREATE (:V {k: 3})", "cypher", None)
        }),
        ("GrafeoDB::execute_cypher", |db| {
            db.execute_cypher("CREATE (:V {k: 3})")
        }),
        ("Session::execute", |db| {
            db.session().execute("CREATE (:V {k: 3})")
        }),
        ("Session::execute_with_params", |db| {
            db.session()
                .execute_with_params("INSERT (:V {k: $k})", three())
        }),
        ("Session::execute_language gql", |db| {
            db.session()
                .execute_language("INSERT (:V {k: 3})", "gql", None)
        }),
        ("Session::execute_cypher_with_params", |db| {
            db.session()
                .execute_cypher_with_params("CREATE (:V {k: $k})", three())
        }),
        ("a transaction", |db| {
            let mut session = db.session();
            session.begin_transaction()?;
            let result = session.execute("INSERT (:V {k: 3})")?;
            session.commit()?;
            Ok(result)
        }),
        ("a GQL transaction", |db| {
            let session = db.session();
            session.execute("START TRANSACTION")?;
            let result = session.execute("INSERT (:V {k: 3})")?;
            session.execute("COMMIT")?;
            Ok(result)
        }),
        ("a Cypher transaction", |db| {
            let mut session = db.session();
            session.begin_transaction()?;
            let result = session.execute_cypher("CREATE (:V {k: 3})")?;
            session.commit()?;
            Ok(result)
        }),
    ];
    for (entry, run) in runs {
        let db = GrafeoDB::new_in_memory();
        let result = run(&db).unwrap_or_else(|error| panic!("{entry} failed: {error}"));
        assert_eq!(
            result.columns,
            Vec::<String>::new(),
            "{entry} has no columns"
        );
        assert_eq!(result.rows(), &[] as &[Vec<Value>], "{entry} has no rows");
        assert_eq!(
            changes(&result),
            [1, 0, 0, 0, 1, 1, 0],
            "{entry} counts its write"
        );
        assert_eq!(ks(&db.session(), "V"), ints(&[3]), "{entry} writes");
    }
}

/// GQL `NEXT`: the statement after it reads the rows the one before
/// returns, also when it is a lone `INSERT` or `DELETE`, and a statement that
/// ends with one has no result. Each runs on a new database with the two `W`
/// nodes: the `k` of the `V` and `W` nodes it leaves, and the number of `R`
/// edges from the `W` with `k` 3.
#[test]
fn a_write_after_next_has_no_result() {
    for (query, v, w, edges) in [
        // It returned the last node of each INSERT.
        (
            "INSERT (:V {k: 3}) NEXT INSERT (:V {k: 19})",
            &[3, 19][..],
            &[3, 19][..],
            0,
        ),
        // The INSERT created a new node `w` instead of reading the row's.
        (
            "MATCH (w:W {k: 3}) RETURN w NEXT INSERT (w)-[:R]->(:V {k: 88})",
            &[88],
            &[3, 19],
            1,
        ),
        // It failed with "Undefined variable 'k'".
        (
            "MATCH (w:W) RETURN w.k AS k NEXT INSERT (:V {k: k})",
            &[3, 19],
            &[3, 19],
            0,
        ),
        // It failed with "duplicate column name 'w'".
        (
            "MATCH (w:W {k: 3}) RETURN w NEXT DETACH DELETE w",
            &[],
            &[19],
            0,
        ),
    ] {
        let session = GrafeoDB::new_in_memory().session();
        execute(&session, Language::Gql, SETUP);
        let result = execute(&session, Language::Gql, query);
        assert_eq!(
            result.columns,
            Vec::<String>::new(),
            "`{query}` has no columns"
        );
        assert_eq!(result.rows(), &[] as &[Vec<Value>], "`{query}` has no rows");
        assert_eq!(ks(&session, "V"), ints(v), "the V nodes of `{query}`");
        assert_eq!(ks(&session, "W"), ints(w), "the W nodes of `{query}`");
        assert_eq!(
            execute(
                &session,
                Language::Gql,
                "MATCH (:W {k: 3})-[r:R]->(:V) RETURN count(r)"
            )
            .rows(),
            [[Value::Int64(edges)]],
            "the R edges of `{query}`"
        );
    }
}

/// After a statement without a result (a write without `RETURN`, or one that
/// ends with `FINISH`), the statement after `NEXT` reads one empty row, as at
/// the start of a statement, however many rows the one before wrote or
/// matched: a `RETURN count(*)` after it counts one row, and an `INSERT`
/// after it writes once, with or without `FINISH`. The statement before
/// writes first, all of it, and what it binds is no variable after it. A
/// lone `INSERT` wrote nothing after `MATCH ... SET`, and nothing after
/// `FINISH` ran. Each case runs on a new database with the two `W` nodes:
/// the `V` nodes without `k` and the `k` of the `W` nodes it leaves.
#[test]
fn after_a_statement_without_a_result_next_reads_one_empty_row() {
    // The statement before, a variable it binds, the V nodes without `k` and
    // the `k` of the W nodes it leaves.
    for (before, bound, plain_v, w) in [
        ("MATCH (w:W) SET w.k = 88", "w", 0, &[88, 88][..]),
        ("MATCH (z:Z) SET z.k = 88", "z", 0, &[3, 19][..]),
        ("FOR i IN [1, 2] INSERT (:V)", "i", 2, &[3, 19]),
        ("FOR i IN [1, 2] INSERT (:V) FINISH", "i", 2, &[3, 19]),
        ("MATCH (w:W) FINISH", "w", 0, &[3, 19]),
        ("INSERT (v:V)", "v", 1, &[3, 19]),
    ] {
        for after in [
            "INSERT (:V {k: 3})",
            "INSERT (:V {k: 3}) FINISH",
            "RETURN count(*) AS rows",
        ] {
            let query = format!("{before} NEXT {after}");
            let session = GrafeoDB::new_in_memory().session();
            execute(&session, Language::Gql, SETUP);
            let result = execute(&session, Language::Gql, &query);
            let (rows, with_k) = if after.starts_with("RETURN") {
                (vec![vec![Value::Int64(1)]], 0)
            } else {
                (Vec::new(), 1)
            };
            assert_eq!(result.rows(), rows, "the result of `{query}`");
            assert_eq!(
                execute(
                    &session,
                    Language::Gql,
                    "MATCH (v:V) RETURN count(v) - count(v.k), count(v.k)",
                )
                .rows(),
                [[Value::Int64(plain_v), Value::Int64(with_k)]],
                "the V nodes of `{query}`"
            );
            assert_eq!(ks(&session, "W"), ints(w), "the W nodes of `{query}`");
        }
        let query = format!("{before} NEXT RETURN count({bound}) AS bound");
        let error = GrafeoDB::new_in_memory()
            .session()
            .execute(&query)
            .expect_err("nothing the statement before binds is a variable after NEXT");
        assert!(
            error.to_string().contains(&format!("'{bound}'")),
            "`{query}`: {error}"
        );
    }
}

/// A procedure call after `NEXT` runs once after a lone `INSERT`, which
/// passes on one empty row and adds no column of its own (it added
/// `_anon_0`, the node it created).
#[test]
fn a_procedure_call_after_a_lone_insert_runs_once() {
    let session = GrafeoDB::new_in_memory().session();
    let result = execute(
        &session,
        Language::Gql,
        "INSERT (:V {k: 3}) NEXT CALL db.labels()",
    );
    assert_eq!(result.columns, ["label"]);
    assert_eq!(result.rows(), [[Value::from("V")]]);
    assert_eq!(ks(&session, "V"), ints(&[3]));
}

/// `EXPLAIN` shows the plan of a statement without a result, and `PROFILE`
/// runs it: the plan of a GQL `INSERT` is that of the Cypher `CREATE`, whose
/// top returns no rows. GQL's planned a `RETURN` of the last node.
#[test]
fn explain_and_profile_plan_no_result() {
    let plan = |language: Language, query: &str| -> String {
        let session = GrafeoDB::new_in_memory().session();
        let result = execute(&session, language, query);
        assert_eq!(result.columns.len(), 1, "{language:?} `{query}`");
        assert_eq!(result.rows().len(), 1, "{language:?} `{query}`");
        let written = ks(&session, "V").len();
        let expected = usize::from(query.starts_with("PROFILE"));
        assert_eq!(written, expected, "{language:?} `{query}` writes");
        match &result.rows()[0][0] {
            Value::String(text) => text.to_string(),
            other => panic!("{language:?} `{query}` returned {other:?}"),
        }
    };
    assert_eq!(
        plan(Language::Gql, "EXPLAIN INSERT (:V {k: 3})"),
        plan(Language::Cypher, "EXPLAIN CREATE (:V {k: 3})"),
    );
    // The top line of a profile, without its time.
    let top = |language: Language, query: &str| -> String {
        let profile = plan(language, query);
        let line = profile.lines().next().unwrap_or_default();
        line.split("  time=").next().unwrap_or_default().to_string()
    };
    let gql = top(Language::Gql, "PROFILE INSERT (:V {k: 3})");
    assert_eq!(gql, top(Language::Cypher, "PROFILE CREATE (:V {k: 3})"));
    assert!(gql.ends_with("  rows=0"), "the top returns no rows: {gql}");
}

/// A procedure whose body ends with a write has no result either: it
/// returned the node its INSERT created, in a row longer than its columns.
#[test]
fn a_procedure_that_ends_with_a_write_has_no_result() {
    let session = GrafeoDB::new_in_memory().session();
    execute(
        &session,
        Language::Gql,
        "CREATE PROCEDURE add_city(name STRING) AS { INSERT (:City {name: $name}) }",
    );
    let result = execute(&session, Language::Gql, "CALL add_city('Amsterdam')");
    assert_eq!(result.columns, Vec::<String>::new(), "no columns");
    assert_eq!(result.rows(), &[] as &[Vec<Value>], "no rows");
    assert_eq!(changes(&result), [1, 0, 0, 0, 1, 1, 0], "the counters");
    assert_eq!(
        execute(&session, Language::Gql, "MATCH (c:City) RETURN c.name").rows(),
        [[Value::from("Amsterdam")]],
        "the procedure writes"
    );
}
