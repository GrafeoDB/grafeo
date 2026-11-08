//! `EXPLAIN` builds the physical plan and drops it without running it, so every refusal of
//! planning holds under it (procedure arguments, the `YIELD` check, external access), and nothing
//! else changes. Planning has no side effects, and `EXPLAIN` is where that is checked: it opens no
//! file (a `LOAD` opens its file when it runs, also on a session with external access), takes no
//! write state (a write statement under `EXPLAIN` leaves no change, takes no epoch, and does not
//! wait on another session's commit), and its output stays the logical plan it gave before.
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test main explain_plans_physically::
//! ```

#![cfg(all(feature = "gql", feature = "cypher"))]

use std::collections::HashMap;

use grafeo_common::types::Value;
use grafeo_common::utils::gqlstatus::GqlStatus;
use grafeo_engine::{Config, GrafeoDB, HistoryRetention};

#[derive(Clone, Copy, Debug)]
enum Language {
    Gql,
    Cypher,
}

/// Three cities, one of them where Alix lives, and an index on the city name.
fn graph() -> GrafeoDB {
    let db = GrafeoDB::new_in_memory();
    for statement in [
        "INSERT (:City {name: 'Amsterdam', pop: 900})",
        "INSERT (:City {name: 'Utrecht', pop: 360})",
        "INSERT (:Person {name: 'Alix', age: 30})-[:LIVES_IN]->(:City {name: 'Leiden', pop: 125})",
        "CREATE INDEX city_name FOR (c:City) ON (c.name)",
    ] {
        db.execute(statement).unwrap();
    }
    db
}

/// The plan `EXPLAIN` gives, as text.
fn explain(db: &GrafeoDB, language: Language, query: &str) -> Result<String, String> {
    let text = format!("EXPLAIN {query}");
    let result = match language {
        Language::Gql => db.execute(&text),
        Language::Cypher => db.execute_cypher(&text),
    }
    .map_err(|error| error.to_string())?;
    match result.rows() {
        [row] => match &row[..] {
            [Value::String(plan)] => Ok(plan.to_string()),
            other => Err(format!("a plan row of one string: {other:?}")),
        },
        other => Err(format!("one plan row: {other:?}")),
    }
}

/// The output 0.6.2 gave before EXPLAIN planned physically (54beb354), statement by statement.
const TODAY: &[(Language, &str, &str)] = &[
    (
        Language::Gql,
        "MATCH (c:City) WHERE c.name = 'Amsterdam' RETURN c.pop AS pop",
        "Return (c.pop AS pop)\n  Filter (c.name Eq \"Amsterdam\") [index: name]\n    NodeScan (c:City)\n",
    ),
    (
        Language::Gql,
        "MATCH (c:City) WHERE c.pop > 300 RETURN c.name AS name ORDER BY name LIMIT 2",
        "Limit (2)\n  Sort (name ASC)\n    Return (c.name AS name)\n      Filter (c.pop Gt 300) [range: pop]\n        NodeScan (c:City)\n",
    ),
    (
        Language::Gql,
        "MATCH (p:Person)-[:LIVES_IN]->(c:City) RETURN p.name AS who, c.name AS city",
        "Return (p.name AS who, c.name AS city)\n  Filter (hasLabel(c, \"City\"))\n    Expand (p)->[:LIVES_IN]->(c)\n      NodeScan (p:Person)\n",
    ),
    (
        Language::Gql,
        "MATCH (c:City) RETURN count(c) AS cities",
        "Return (cities AS cities)\n  LabelCount (c:City, aggs: [countnonnull(...) AS cities])\n",
    ),
    (
        Language::Gql,
        "MATCH (p:Person) OPTIONAL MATCH (p)-[:KNOWS]->(q) RETURN p.name, q.name",
        "Return (p.name, q.name)\n  LeftJoin\n    NodeScan (p:Person)\n    Expand (p)->[:KNOWS]->(q)\n      NodeScan (p:*)\n",
    ),
    (
        Language::Gql,
        "INSERT (:City {name: 'Delft', pop: 105})",
        "Return ()\n  Create (_anon_0:City)\n",
    ),
    (
        Language::Gql,
        "MERGE (c:City {name: 'Gouda'}) RETURN c.name AS name",
        "Return (c.name AS name) [after the write]\n  Merge (c) [index: name]\n    Empty\n",
    ),
    (
        Language::Gql,
        "MATCH (c:City {name: 'Utrecht'}) SET c.pop = 361",
        "Return ()\n  SetProperty (c.pop)\n    Filter (c.name Eq \"Utrecht\") [index: name]\n      NodeScan (c:City)\n",
    ),
    (
        Language::Gql,
        "MATCH (c:City {name: 'Utrecht'}) DETACH DELETE c",
        "Return ()\n  DeleteNode (c)\n    Filter (c.name Eq \"Utrecht\") [index: name]\n      NodeScan (c:City)\n",
    ),
    (
        Language::Gql,
        "CALL grafeo.pagerank({damping: 0.85}) YIELD node_id, score RETURN node_id, score",
        "Return (node_id, score)\n  CallProcedure (grafeo.pagerank)\n",
    ),
    (
        Language::Cypher,
        "MATCH (c:City) WHERE c.pop < 500 RETURN c.name ORDER BY c.name",
        "Sort (c.name ASC)\n  Return (c.name)\n    Filter (c.pop Lt 500) [range: pop]\n      NodeScan (c:City)\n",
    ),
    (
        Language::Cypher,
        "MATCH (p:Person)-[:LIVES_IN]->(c) RETURN p.name, c.name",
        "Return (p.name, c.name)\n  Expand (p)->[:LIVES_IN]->(c)\n    NodeScan (p:Person)\n",
    ),
];

/// A statement `EXPLAIN` plans gives the same output as before: the logical plan. Reads, writes, a
/// procedure call and Cypher, on a graph with an index.
#[test]
fn a_successful_explain_gives_the_plan_it_gave_before() {
    let mut wrong = Vec::new();
    for &(language, query, plan) in TODAY {
        // A database per statement: an EXPLAIN that wrote would show in the next one's plan.
        match explain(&graph(), language, query) {
            Ok(found) if found == plan => {}
            other => wrong.push(format!(
                "{language:?} {query}\n  want {plan:?}\n  got  {other:?}"
            )),
        }
    }
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}

/// `EXPLAIN` of a statement that reads a file opens none, on a session with external access: a
/// file that does not exist gives its plan, while running the statement fails on the open.
#[test]
fn explain_of_a_load_opens_no_file() {
    let dir = tempfile::tempdir().unwrap();
    let missing = dir
        .path()
        .join("nobody-wrote-this.csv")
        .to_string_lossy()
        .replace('\\', "/");
    let db = GrafeoDB::new_in_memory();
    for (language, query) in [
        (
            Language::Gql,
            format!(
                "LOAD DATA FROM '{missing}' FORMAT CSV WITH HEADERS AS row RETURN row.name AS n"
            ),
        ),
        (
            Language::Gql,
            format!(
                "CALL grafeo.load_nodes_csv({{path: '{missing}', labels: ['Person'], key: 'id'}})"
            ),
        ),
        (
            Language::Cypher,
            format!("LOAD CSV WITH HEADERS FROM '{missing}' AS row RETURN row.name AS n"),
        ),
    ] {
        explain(&db, language, &query)
            .unwrap_or_else(|error| panic!("EXPLAIN {query} opened the file: {error}"));
        let ran = match language {
            Language::Gql => db.execute(&query),
            Language::Cypher => db.execute_cypher(&query),
        };
        assert!(
            ran.is_err(),
            "{query}: running it opens the file, which does not exist"
        );
    }
    assert!(
        std::fs::read_dir(dir.path()).unwrap().next().is_none(),
        "nothing was written beside the missing file"
    );
}

fn cities(db: &GrafeoDB) -> usize {
    db.execute("MATCH (c:City) RETURN c").unwrap().rows().len()
}

/// The write statements the rows below explain.
const WRITES: [&str; 4] = [
    "INSERT (:City {name: 'Delft', pop: 105})",
    "MERGE (c:City {name: 'Gouda'}) RETURN c.name AS name",
    "MATCH (c:City {name: 'Utrecht'}) SET c.pop = 361",
    "MATCH (c:City {name: 'Utrecht'}) DETACH DELETE c",
];

/// `EXPLAIN` of a write, outside a transaction and inside one, changes nothing and takes no epoch:
/// the cities, Utrecht's population and the database's epoch are as before, and a transaction
/// that explained writes commits with nothing to show for them.
#[test]
fn explain_of_a_write_changes_nothing_and_takes_no_epoch() {
    let db = graph();
    let epoch = db.current_epoch();
    for write in WRITES {
        explain(&db, Language::Gql, write).unwrap_or_else(|error| panic!("{write}: {error}"));
    }
    assert_eq!(
        db.current_epoch(),
        epoch,
        "EXPLAIN outside a transaction took an epoch"
    );
    assert_eq!(cities(&db), 3);

    let mut session = db.session();
    session.begin_transaction().unwrap();
    for write in WRITES {
        session
            .execute(&format!("EXPLAIN {write}"))
            .unwrap_or_else(|error| panic!("{write} in a transaction: {error}"));
    }
    let inside = session
        .execute("MATCH (c:City) RETURN c")
        .unwrap()
        .rows()
        .len();
    assert_eq!(inside, 3, "the transaction sees no explained write");
    assert_eq!(
        db.current_epoch(),
        epoch,
        "EXPLAIN in a transaction took an epoch"
    );
    session.commit().unwrap();
    assert_eq!(cities(&db), 3);
    let utrecht = db
        .execute("MATCH (c:City {name: 'Utrecht'}) RETURN c.pop AS pop")
        .unwrap();
    assert_eq!(utrecht.rows(), [vec![Value::Int64(360)]]);
}

/// `EXPLAIN` of a write does not wait on another session's commit: one commit is held right after
/// its epoch is assigned (it holds the commit lock), and a write explained meanwhile answers,
/// while the same write run for real waits until the commit goes on.
#[cfg(feature = "testing-statement-injection")]
#[test]
fn explain_of_a_write_does_not_wait_on_a_commit() {
    use std::sync::Arc;
    use std::sync::mpsc;
    use std::time::Duration;

    use grafeo_common::testing::commit_hook;

    let db = Arc::new(graph());
    let (held_sender, held) = mpsc::channel();
    let (release, released) = mpsc::channel::<()>();
    let committer_db = Arc::clone(&db);
    let committer = std::thread::spawn(move || {
        commit_hook::after_next_commit_epoch(move || {
            held_sender.send(()).unwrap();
            released
                .recv_timeout(Duration::from_secs(30))
                .expect("the commit was released");
        });
        committer_db
            .execute("INSERT (:City {name: 'Haarlem', pop: 160})")
            .unwrap();
    });
    held.recv_timeout(Duration::from_secs(10))
        .expect("the commit reached its hook");

    let explained = {
        let (sender, answer) = mpsc::channel();
        let explainer_db = Arc::clone(&db);
        std::thread::spawn(move || {
            sender
                .send(explain(&explainer_db, Language::Gql, WRITES[0]))
                .unwrap();
        });
        answer.recv_timeout(Duration::from_secs(5))
    };

    // The same write for real waits on the held commit: the hold is real.
    let (sender, ran) = mpsc::channel();
    let writer_db = Arc::clone(&db);
    let writer = std::thread::spawn(move || {
        sender.send(writer_db.execute(WRITES[0]).is_ok()).unwrap();
    });
    let waited = ran.recv_timeout(Duration::from_millis(500)).is_err();

    release.send(()).unwrap();
    committer.join().unwrap();
    writer.join().unwrap();
    let explained = explained.expect("EXPLAIN of a write waited on the held commit");
    explained.unwrap_or_else(|error| panic!("{error}"));
    assert!(
        waited,
        "the real write did not wait: the commit was not held"
    );
    assert_eq!(
        cities(&db),
        5,
        "Haarlem and the real Delft, not the explained one"
    );
}

/// An EXPLAIN whose plan still holds a parameter shows the plan without planning physically: a
/// refusal that depends on values waits for them. A CALL with an out-of-domain literal (a damping
/// of 2.0) beside a parameter gives its plan without values, and is refused with 22023 once they
/// are bound, as the statement that runs would be.
#[test]
fn explain_without_values_plans_logically_and_with_them_refuses() {
    let db = graph();
    let query = "EXPLAIN CALL grafeo.pagerank({damping: 2.0, max_iterations: $n}) \
                 YIELD node_id, score RETURN node_id, score";
    db.execute(query)
        .unwrap_or_else(|error| panic!("without values, the plan: {error}"));
    let error = db
        .execute_with_params(query, HashMap::from([("n".to_string(), Value::Int64(10))]))
        .expect_err("with the values bound, the damping is out of its domain");
    assert_eq!(GqlStatus::from(&error).as_str(), "22023", "{error}");
}

/// EXPLAIN at an expired history epoch plans without pinning it, as planning reads no rows, and
/// never reads versioned data there unpinned: a collection running meanwhile cannot free what it
/// reads. A comma pattern (its order from sampled degrees) and a range (column bounds), explained
/// again and again at the expired epoch while another thread writes and collects: each time a
/// plan, with no error and no crash, holding the statement's operators. Not their order: the
/// planner may read its estimates at a current pin or skip them here, and either gives a valid
/// plan whose order can differ.
#[test]
fn explain_at_an_expired_epoch_plans_while_a_collection_runs() {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, Ordering};

    let db = Arc::new(
        GrafeoDB::with_config(Config::in_memory().with_history_retention(HistoryRetention::Off))
            .unwrap(),
    );
    db.execute("INSERT (:Person {age: 1}), (:Person {age: 2})")
        .unwrap();
    let earlier = db.current_epoch();
    db.execute("MATCH (p:Person) SET p.age = p.age + 4")
        .unwrap();
    db.gc();
    let query = "EXPLAIN MATCH (p:Person), (q:Person) WHERE p.age > 3 RETURN p.age, q.age";
    // The operators of the statement, in whatever order the plan puts them.
    let operators = [
        "Return (p.age, q.age)",
        "NodeScan (p:Person)",
        "NodeScan (q:Person)",
        "Filter (p.age Gt 3)",
    ];

    let stop = Arc::new(AtomicBool::new(false));
    let collector = {
        let (db, stop) = (Arc::clone(&db), Arc::clone(&stop));
        std::thread::spawn(move || {
            while !stop.load(Ordering::Acquire) {
                db.execute("MATCH (p:Person) SET p.age = p.age + 1")
                    .unwrap();
                db.gc();
            }
        })
    };
    let mut session = db.session();
    session.begin_transaction().unwrap();
    session.set_viewing_epoch(earlier);
    let mut wrong = None;
    for round in 0..200 {
        let plan = match session.execute(query) {
            Ok(result) => match result.rows() {
                [row] => match &row[..] {
                    [Value::String(plan)] => plan.to_string(),
                    other => format!("not a plan: {other:?}"),
                },
                other => format!("not one plan row: {other:?}"),
            },
            Err(error) => format!("an error: {error}"),
        };
        let lacking: Vec<&str> = operators
            .iter()
            .copied()
            .filter(|operator| !plan.contains(operator))
            .collect();
        if !lacking.is_empty() {
            wrong = Some(format!("round {round}: lacks {lacking:?} in {plan}"));
            break;
        }
    }
    stop.store(true, Ordering::Release);
    collector.join().expect("the collection did not crash");
    session.clear_viewing_epoch();
    session.rollback().unwrap();
    assert!(wrong.is_none(), "{}", wrong.unwrap_or_default());
}

/// The estimates EXPLAIN plans with are read at the latest data (the degree samples) or need no
/// view (the column bounds), whatever epoch the session views. So with nothing changing, EXPLAIN
/// at an expired epoch gives exactly the plan it gives at the current one.
#[test]
fn explain_at_an_expired_epoch_gives_the_current_plan_when_nothing_changes() {
    let db =
        GrafeoDB::with_config(Config::in_memory().with_history_retention(HistoryRetention::Off))
            .unwrap();
    db.execute("INSERT (:Person {age: 1}), (:Person {age: 2})")
        .unwrap();
    let earlier = db.current_epoch();
    db.execute("MATCH (p:Person) SET p.age = p.age + 4")
        .unwrap();
    db.gc();
    let query = "EXPLAIN MATCH (p:Person), (q:Person) WHERE p.age > 3 RETURN p.age, q.age";
    let current = db.execute(query).unwrap().rows().to_vec();
    let mut session = db.session();
    session.begin_transaction().unwrap();
    session.set_viewing_epoch(earlier);
    let expired = session.execute(query).unwrap().rows().to_vec();
    session.clear_viewing_epoch();
    session.rollback().unwrap();
    assert_eq!(expired, current);
}

/// Three documents near `[1, 0, 0]` under a cosine vector index, and the threshold filter that
/// plans a vector scan without k (it counts its candidates instead of taking k of them).
#[cfg(feature = "vector-index")]
const NEAR: &str = "MATCH (d:D) WHERE cosine_similarity(d.emb, [1.0, 0.0, 0.0]) >= 0.5e0 \
                    RETURN d.i AS i ORDER BY i";

#[cfg(feature = "vector-index")]
fn documents(db: &GrafeoDB) {
    db.execute("CREATE VECTOR INDEX d_emb ON :D(emb) DIMENSION 3 METRIC 'cosine'")
        .unwrap();
    for (i, emb) in ["[1.0, 0.0, 0.0]", "[0.9, 0.1, 0.0]", "[0.8, 0.2, 0.0]"]
        .iter()
        .enumerate()
    {
        db.execute(&format!("INSERT (:D {{i: {i}, emb: vector({emb})}})"))
            .unwrap();
    }
}

/// The vector scan without k counts its candidates when it first runs, not at planning, where it
/// read at the statement's view: EXPLAIN of it at an expired epoch, again and again while another
/// thread writes and collects, gives its plan every time, with no error. EXPLAIN prints the
/// logical plan (the rewrite to a vector scan is the physical planner's), so PROFILE shows that the
/// filter plans one, and each EXPLAIN gives the logical plan it gives at the current epoch.
#[cfg(feature = "vector-index")]
#[test]
fn explain_of_a_vector_scan_without_k_at_an_expired_epoch_plans() {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, Ordering};

    let db = Arc::new(
        GrafeoDB::with_config(Config::in_memory().with_history_retention(HistoryRetention::Off))
            .unwrap(),
    );
    documents(&db);
    let earlier = db.current_epoch();
    db.execute("MATCH (d:D) SET d.seen = true").unwrap();
    db.gc();
    let query = format!("EXPLAIN {NEAR}");
    assert!(
        format!(
            "{:?}",
            db.execute(&format!("PROFILE {NEAR}")).unwrap().rows()
        )
        .contains("VectorScan"),
        "the filter plans a vector scan"
    );
    let current = db.execute(&query).unwrap().rows().to_vec();
    let stop = Arc::new(AtomicBool::new(false));
    let collector = {
        let (db, stop) = (Arc::clone(&db), Arc::clone(&stop));
        std::thread::spawn(move || {
            while !stop.load(Ordering::Acquire) {
                db.execute("MATCH (d:D) SET d.seen = NOT d.seen").unwrap();
                db.gc();
            }
        })
    };
    let mut session = db.session();
    session.begin_transaction().unwrap();
    session.set_viewing_epoch(earlier);
    let mut wrong = None;
    for round in 0..200 {
        match session.execute(&query) {
            Ok(plan) if plan.rows() == current.as_slice() => {}
            other => {
                wrong = Some(format!(
                    "round {round}: {:?}",
                    other.map(|plan| plan.rows().to_vec())
                ));
                break;
            }
        }
    }
    stop.store(true, Ordering::Release);
    collector.join().expect("the collection did not crash");
    session.clear_viewing_epoch();
    session.rollback().unwrap();
    assert!(wrong.is_none(), "{}", wrong.unwrap_or_default());
}

/// A vector scan without k counts at its own snapshot: a document another transaction inserts
/// after a transaction began is neither counted nor returned in it, and is returned once the
/// transaction is over.
#[cfg(feature = "vector-index")]
#[test]
fn a_vector_scan_without_k_counts_at_its_snapshot() {
    let db = GrafeoDB::new_in_memory();
    documents(&db);
    let near = |result: grafeo_engine::database::QueryResult| -> Vec<Value> {
        result.rows().iter().map(|row| row[0].clone()).collect()
    };
    let before = near(db.execute(NEAR).unwrap());
    assert_eq!(before.len(), 3, "{before:?}");
    let mut session = db.session();
    session.begin_transaction().unwrap();
    assert_eq!(near(session.execute(NEAR).unwrap()), before);
    db.execute("INSERT (:D {i: 3, emb: vector([0.95, 0.05, 0.0])})")
        .unwrap();
    assert_eq!(
        near(session.execute(NEAR).unwrap()),
        before,
        "the transaction's snapshot holds three documents"
    );
    session.commit().unwrap();
    let after = near(db.execute(NEAR).unwrap());
    assert_eq!(after.len(), 4, "{after:?}");
}
