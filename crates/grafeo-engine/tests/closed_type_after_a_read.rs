//! A closed graph type refuses what it refuses whatever ran before the write: a read in the session,
//! a read in the transaction, a change of the type by another session, and a statement whose write
//! has no row to write. A read need not build what only a write uses, and these hold however that
//! is built or when.
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test main closed_type_after_a_read::
//! ```

use grafeo_engine::GrafeoDB;
use grafeo_engine::session::Session;

const SETUP: &[&str] = &[
    "CREATE GRAPH TYPE t (NODE TYPE P (id INT64 NOT NULL, name STRING, spare STRING), \
     NODE TYPE Q (id INT64 NOT NULL), EDGE TYPE R (), EDGE TYPE S (w INT64 NOT NULL))",
    "CREATE GRAPH g TYPED t",
    "USE GRAPH g",
    "INSERT (:P {id: 1, name: 'Alix'}), (:Q {id: 1})",
];

/// What the type refuses, with the rows to write it on there.
const REFUSED: &[(&str, &str)] = &[
    ("a_label_it_does_not_declare", "INSERT (:Other {id: 2})"),
    (
        "a_label_it_does_not_declare_beside_one_it_does",
        "INSERT (:P:Other {id: 2})",
    ),
    ("no_label", "INSERT ({id: 2})"),
    (
        "a_property_it_does_not_declare",
        "INSERT (:P {id: 2, other: 1})",
    ),
    ("a_value_of_another_type", "INSERT (:P {id: 'two'})"),
    ("a_missing_required_property", "INSERT (:P {name: 'Gus'})"),
    ("no_properties_with_one_required", "INSERT (:P)"),
    (
        "an_edge_type_it_does_not_declare",
        "MATCH (a:P {id: 1}), (b:Q {id: 1}) INSERT (a)-[:Other]->(b)",
    ),
    (
        "an_edge_property_it_does_not_declare",
        "MATCH (a:P {id: 1}), (b:Q {id: 1}) INSERT (a)-[:R {w: 1}]->(b)",
    ),
    (
        "an_edge_missing_a_required_property",
        "MATCH (a:P {id: 1}), (b:Q {id: 1}) INSERT (a)-[:S]->(b)",
    ),
    (
        "set_a_property_it_does_not_declare",
        "MATCH (a:P {id: 1}) SET a.other = 1",
    ),
    (
        "set_a_value_of_another_type",
        "MATCH (a:P {id: 1}) SET a.name = 1",
    ),
    (
        "remove_a_required_property",
        "MATCH (a:P {id: 1}) REMOVE a.id",
    ),
    (
        "set_a_label_it_does_not_declare",
        "MATCH (a:P {id: 1}) SET a:Other",
    ),
    ("remove_the_last_label", "MATCH (a:P {id: 1}) REMOVE a:P"),
    (
        "merge_a_label_it_does_not_declare",
        "MERGE (:Other {id: 2})",
    ),
    (
        "a_write_after_a_read_clause",
        "MATCH (a:P {id: 1}) WITH a.id AS k INSERT (:Other {id: k})",
    ),
];

/// What the type allows.
const ALLOWED: &[(&str, &str)] = &[
    ("a_declared_node", "INSERT (:P {id: 3, name: 'Mia'})"),
    (
        "a_declared_edge",
        "MATCH (a:P {id: 1}), (b:Q {id: 1}) INSERT (a)-[:R]->(b)",
    ),
    (
        "set_a_declared_property",
        "MATCH (a:P {id: 1}) SET a.name = 'Gus'",
    ),
    (
        "remove_a_property_not_required",
        "MATCH (a:P {id: 1}) REMOVE a.name",
    ),
];

const READS: &[&str] = &[
    "RETURN 1 AS one",
    "MATCH (n) RETURN count(n) AS nodes",
    "MATCH (n:P {id: 1}) RETURN n.name",
    "MATCH (n:Other) RETURN n",
];

fn typed() -> (GrafeoDB, Session) {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    for statement in SETUP {
        session.execute(statement).unwrap();
    }
    (db, session)
}

fn on_the_graph(db: &GrafeoDB) -> Session {
    let session = db.session();
    session.execute("USE GRAPH g").unwrap();
    session
}

fn count(session: &Session) -> i64 {
    match session
        .execute("MATCH (n) RETURN count(n) AS c")
        .unwrap()
        .rows()[0][0]
    {
        grafeo_common::types::Value::Int64(nodes) => nodes,
        ref other => panic!("{other:?}"),
    }
}

/// The first statement of a session.
#[test]
fn a_closed_type_refuses_as_a_sessions_first_statement() {
    for (name, write) in REFUSED {
        let (db, _setup) = typed();
        let session = on_the_graph(&db);
        assert!(session.execute(write).is_err(), "{name}");
    }
    for (name, write) in ALLOWED {
        let (db, _setup) = typed();
        let session = on_the_graph(&db);
        session
            .execute(write)
            .unwrap_or_else(|error| panic!("{name}: {error}"));
    }
}

/// Right after a read of each kind in the same session, and with the same words again.
#[test]
fn a_closed_type_refuses_right_after_a_read() {
    for read in READS {
        for (name, write) in REFUSED {
            let (db, _setup) = typed();
            let session = on_the_graph(&db);
            session.execute(read).unwrap();
            let before = count(&session);
            assert!(session.execute(write).is_err(), "{name} after {read}");
            assert!(session.execute(write).is_err(), "{name} again after {read}");
            session.execute(read).unwrap();
            assert!(session.execute(write).is_err(), "{name}, a read between");
            assert_eq!(count(&session), before, "{name}: nothing written");
        }
        for (name, write) in ALLOWED {
            let (db, _setup) = typed();
            let session = on_the_graph(&db);
            session.execute(read).unwrap();
            session
                .execute(write)
                .unwrap_or_else(|error| panic!("{name} after {read}: {error}"));
        }
    }
}

/// In a transaction, after its read: the statement fails, and the transaction's other writes stand or
/// go with it as they do for any failed statement.
#[test]
fn a_closed_type_refuses_in_a_transaction_after_a_read() {
    for (name, write) in REFUSED {
        let (db, _setup) = typed();
        let mut session = on_the_graph(&db);
        session.begin_transaction().unwrap();
        session
            .execute("MATCH (n) RETURN count(n) AS nodes")
            .unwrap();
        assert!(session.execute(write).is_err(), "{name}");
        let _ = session.rollback();
        assert_eq!(count(&session), 2, "{name}: nothing written");
    }
    let (db, _setup) = typed();
    let mut session = on_the_graph(&db);
    session.begin_transaction().unwrap();
    session.execute("RETURN 1 AS one").unwrap();
    session.execute("INSERT (:P {id: 7})").unwrap();
    session.execute("MATCH (n:P {id: 7}) RETURN n.id").unwrap();
    assert!(session.execute("INSERT (:Other {id: 8})").is_err());
    let _ = session.rollback();
    assert_eq!(count(&session), 2, "the transaction went");
}

/// A write with no row to write on: what the plan's type check says does not wait for a row.
#[test]
fn a_write_with_no_rows_answers_as_it_does_after_a_read() {
    let forms = [
        "MATCH (a:P {id: 99}) SET a.other = 1",
        "MATCH (a:P {id: 99}) SET a.name = 1",
        "MATCH (a:P {id: 99}) REMOVE a.id",
        "MATCH (a:P {id: 99}) SET a:Other",
        "MATCH (a:P {id: 99}) REMOVE a:P",
        "MATCH (a:P {id: 99}), (b:Q {id: 1}) INSERT (a)-[:Other]->(b)",
        "MATCH (a:P {id: 99}) INSERT (:Other {id: a.id})",
        "MATCH (a:P {id: 99}) DELETE a",
        "MATCH (a:P {id: 99}) SET a.name = 'Gus'",
    ];
    for write in forms {
        let (db, _setup) = typed();
        let first = on_the_graph(&db)
            .execute(write)
            .map(|_| ())
            .map_err(|error| error.to_string());
        let session = on_the_graph(&db);
        session.execute("RETURN 1 AS one").unwrap();
        let after_a_read = session
            .execute(write)
            .map(|_| ())
            .map_err(|error| error.to_string());
        assert_eq!(first, after_a_read, "{write}");
        assert_eq!(count(&session), 2, "{write}");
    }
}

/// A change of the type by another session counts at the next statement, after a read, for words the
/// session has run before.
#[test]
fn a_type_changed_by_another_session_counts_at_the_next_write() {
    let (db, other) = typed();
    let session = on_the_graph(&db);
    let added = "INSERT (:P {id: 10, age: 1})";
    let spare = "INSERT (:P {id: 11, spare: 'x'})";
    let new_label = "INSERT (:New {id: 12})";
    let new_edge = "MATCH (a:P {id: 1}), (b:Q {id: 1}) INSERT (a)-[:Added]->(b)";

    session.execute("RETURN 1 AS one").unwrap();
    assert!(session.execute(added).is_err(), "not declared yet");
    assert!(session.execute(new_label).is_err(), "not declared yet");
    assert!(session.execute(new_edge).is_err(), "not declared yet");

    other
        .execute("ALTER NODE TYPE P ADD PROPERTY age INT64")
        .unwrap();
    session.execute("MATCH (n:P) RETURN n.id").unwrap();
    session.execute(added).unwrap();

    other.execute("CREATE NODE TYPE New (id INT64)").unwrap();
    other
        .execute("ALTER GRAPH TYPE t ADD NODE TYPE New")
        .unwrap();
    session.execute("RETURN 1 AS one").unwrap();
    session.execute(new_label).unwrap();

    other.execute("CREATE EDGE TYPE Added").unwrap();
    other
        .execute("ALTER GRAPH TYPE t ADD EDGE TYPE Added")
        .unwrap();
    session.execute("RETURN 1 AS one").unwrap();
    session.execute(new_edge).unwrap();

    // A property no node holds goes, and with it the write of it, for the same words.
    let first = on_the_graph(&db);
    first.execute(spare).unwrap();
    first
        .execute("MATCH (n:P {id: 11}) REMOVE n.spare")
        .unwrap();
    other
        .execute("ALTER NODE TYPE P DROP PROPERTY spare")
        .unwrap();
    first.execute("RETURN 1 AS one").unwrap();
    assert!(first.execute("INSERT (:P {id: 13, spare: 'x'})").is_err());
    assert!(first.execute(spare).is_err(), "the same words as before");
}

/// A transaction open across another session's change reads the type at each statement as it does
/// without a read before it: the read changes nothing.
#[test]
fn a_type_changed_during_a_transaction_answers_alike_with_a_read_first() {
    let outcome = |read_first: bool| -> (bool, bool) {
        let (db, other) = typed();
        let mut session = on_the_graph(&db);
        session.begin_transaction().unwrap();
        if read_first {
            session.execute("RETURN 1 AS one").unwrap();
            session.execute("MATCH (n) RETURN n").unwrap();
        }
        other
            .execute("ALTER NODE TYPE P ADD PROPERTY age INT64")
            .unwrap();
        let written = session.execute("INSERT (:P {id: 10, age: 1})").is_ok();
        let committed = session.commit().is_ok();
        (written, committed)
    };
    assert_eq!(
        outcome(false),
        (true, true),
        "the write and its commit go through"
    );
    assert_eq!(outcome(false), outcome(true));
}

/// A Cypher write, the same way.
#[test]
fn a_cypher_write_is_refused_after_a_read() {
    let (db, _setup) = typed();
    let session = on_the_graph(&db);
    session.execute_cypher("RETURN 1 AS one").unwrap();
    for write in [
        "CREATE (:Other {id: 2})",
        "CREATE (:P {id: 2, other: 1})",
        "CREATE (:P {name: 'Gus'})",
        "MATCH (a:P {id: 1}) SET a.other = 1",
        "MATCH (a:P {id: 1}) REMOVE a.id",
    ] {
        assert!(session.execute_cypher(write).is_err(), "{write}");
    }
    assert_eq!(count(&session), 2);
}

/// A graph on no closed type takes what it took, after a read too, in the same database.
#[test]
fn an_untyped_graph_beside_it_takes_any_write() {
    let (db, _setup) = typed();
    let session = db.session();
    session.execute("CREATE GRAPH free").unwrap();
    session.execute("USE GRAPH free").unwrap();
    session.execute("RETURN 1 AS one").unwrap();
    session
        .execute("INSERT (:Other {id: 'x', anything: 1})")
        .unwrap();
    session.execute("USE GRAPH g").unwrap();
    assert!(
        session.execute("INSERT (:Other {id: 2})").is_err(),
        "back on the closed graph"
    );
    session.execute("USE GRAPH free").unwrap();
    session.execute("INSERT (:Other {id: 2})").unwrap();
}

/// While another session adds and drops a property, a write of it answers by the type at one moment:
/// each statement either writes or is refused, whichever it sees, and a value that got in keeps the
/// property from going. This race checks that last clause only when a write got in; the test after it
/// checks it on every run.
#[test]
fn a_type_changing_under_writes_gives_one_answer_per_statement() {
    let (db, other) = typed();
    let stop = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    // The writes start once the first ADD has gone through, so the type has changed at least once
    // before the writes start: on a loaded machine the changer could otherwise get no turn before
    // they end. A first ADD refused, or a panic, drops the sender and fails the test at once.
    let (added, first_add) = std::sync::mpsc::channel();
    let changer = {
        let stop = std::sync::Arc::clone(&stop);
        std::thread::spawn(move || {
            let mut changes = 0_u32;
            let mut added = Some(added);
            while !stop.load(std::sync::atomic::Ordering::Relaxed) {
                if other
                    .execute("ALTER NODE TYPE P ADD PROPERTY age INT64")
                    .is_ok()
                {
                    changes += 1;
                    if let Some(added) = added.take() {
                        added.send(()).unwrap();
                    }
                } else if added.is_some() {
                    return changes;
                }
                let _ = other.execute("ALTER NODE TYPE P DROP PROPERTY age");
                std::thread::yield_now();
            }
            changes
        })
    };
    let session = on_the_graph(&db);
    first_add.recv().expect("the first ADD went through");
    let mut written = 0_i64;
    for at in 0..2_000 {
        session.execute("RETURN 1 AS one").unwrap();
        if session
            .execute(&format!("INSERT (:P {{id: {}, age: 1}})", 1_000 + at))
            .is_ok()
        {
            written += 1;
        }
    }
    stop.store(true, std::sync::atomic::Ordering::Relaxed);
    assert!(changer.join().unwrap() > 0, "the type did change");
    let held = session
        .execute("MATCH (n:P) WHERE n.age = 1 RETURN count(n) AS c")
        .unwrap();
    assert_eq!(
        held.rows()[0][0],
        grafeo_common::types::Value::Int64(written)
    );
    if written > 0 {
        assert!(
            on_the_graph(&db)
                .execute("ALTER NODE TYPE P DROP PROPERTY age")
                .is_err(),
            "{written} nodes hold it"
        );
    }
}

/// The pair the race above turns on, every run: a transaction that wrote `age` commits while another
/// session drops `age`, the two started together, so either may reach the commit lock first. Exactly
/// one goes through, and a node holds `age` exactly when the commit did.
#[test]
fn a_write_of_a_property_and_its_drop_give_one_answer_together() {
    for round in 0..50 {
        let (db, other) = typed();
        other
            .execute("ALTER NODE TYPE P ADD PROPERTY age INT64")
            .unwrap();
        let mut session = on_the_graph(&db);
        session.begin_transaction().unwrap();
        session.execute("INSERT (:P {id: 1, age: 1})").unwrap();
        let start = std::sync::Arc::new(std::sync::Barrier::new(2));
        let dropper = {
            let start = std::sync::Arc::clone(&start);
            std::thread::spawn(move || {
                start.wait();
                other
                    .execute("ALTER NODE TYPE P DROP PROPERTY age")
                    .map(|_| ())
            })
        };
        start.wait();
        // Statement work, not a wait: with none the commit tends to win, with one or more the DROP,
        // so both orders occur across the rounds; the test asserts neither order nor how often.
        for _ in 0..round % 4 {
            session.execute("RETURN 1 AS one").unwrap();
        }
        let committed = session.commit();
        let dropped = dropper.join().unwrap();
        let after = on_the_graph(&db);
        match (&committed, &dropped) {
            (Ok(()), Err(_)) => assert!(
                after
                    .execute("ALTER NODE TYPE P DROP PROPERTY age")
                    .is_err(),
                "round {round}: the held value keeps the property"
            ),
            (Err(error), Ok(())) => assert!(
                error.to_string().contains("GRAFEO-V001"),
                "round {round}: {error}"
            ),
            _ => panic!("round {round}: commit {committed:?}, drop {dropped:?}"),
        }
        let held = after
            .execute("MATCH (n:P) WHERE n.age IS NOT NULL RETURN count(n) AS c")
            .unwrap();
        let expected = i64::from(committed.is_ok());
        assert_eq!(
            held.rows()[0][0],
            grafeo_common::types::Value::Int64(expected),
            "round {round}: a node holds age exactly when the commit went through"
        );
        assert_eq!(
            after.execute("INSERT (:P {id: 2, age: 2})").is_ok(),
            committed.is_ok(),
            "round {round}: age is declared exactly when a node holds it"
        );
    }
}
