//! #743: what a `NODE KEY` constraint does, each claim its docs make pinned here. A node key is the
//! values of its properties taken together: no two nodes of the label hold the same values, and every
//! node of the label holds a value for each property, so it is UNIQUE over the tuple plus NOT NULL on
//! each property. It holds on insert, on update, on a label added, over existing data, across
//! transactions, in the direct API and after a reopen or a crash; it is checked at each write, binds
//! by label and holds in each graph on its own; it builds an index on its first property, which a
//! lookup by the key reads, and reads the same in GQL and Cypher.
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test main node_key_claims::
//! ```

use grafeo_common::types::Value;
use grafeo_common::utils::gqlstatus::GqlStatus;
use grafeo_engine::GrafeoDB;
use grafeo_engine::session::Session;

const KEY: &str = "CREATE CONSTRAINT person_key FOR (p:Person) ON (p.first, p.last) NODE KEY";

fn keyed() -> GrafeoDB {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    session.execute(KEY).unwrap();
    session
        .execute("INSERT (:Person {first: 'Alix', last: 'Gus', city: 'Amsterdam'})")
        .unwrap();
    db
}

fn outcome(session: &Session, statement: &str) -> Result<(), String> {
    session
        .execute(statement)
        .map(|_| ())
        .map_err(|error| format!("{} {error}", GqlStatus::from(&error).as_str()))
}

fn count(session: &Session) -> i64 {
    match session
        .execute("MATCH (p:Person) RETURN count(p) AS c")
        .unwrap()
        .rows()[0][0]
    {
        Value::Int64(n) => n,
        ref other => panic!("{other:?}"),
    }
}

/// Composite: the tuple is unique, not each property.
#[test]
fn the_tuple_is_the_key() {
    let db = keyed();
    let session = db.session();
    for (statement, accepted) in [
        ("INSERT (:Person {first: 'Alix', last: 'Gus'})", false),
        ("INSERT (:Person {first: 'Alix', last: 'Mia'})", true),
        ("INSERT (:Person {first: 'Vincent', last: 'Gus'})", true),
        (
            "INSERT (:Person {last: 'Gus', first: 'Alix', city: 'Berlin'})",
            false,
        ),
        ("INSERT (:Other {first: 'Alix', last: 'Gus'})", true),
    ] {
        assert_eq!(
            outcome(&session, statement).is_ok(),
            accepted,
            "{statement}"
        );
    }
    assert_eq!(count(&session), 3);
}

/// Every node of the label holds each key property: missing or null is refused, on insert and update.
#[test]
fn each_key_property_is_required() {
    let db = keyed();
    let session = db.session();
    for statement in [
        "INSERT (:Person {first: 'Mia'})",
        "INSERT (:Person {last: 'Mia'})",
        "INSERT (:Person {first: 'Mia', last: null})",
        "INSERT (:Person)",
        "MATCH (p:Person {first: 'Alix'}) SET p.last = null",
        "MATCH (p:Person {first: 'Alix'}) REMOVE p.last",
        "MATCH (p:Person {first: 'Alix'}) SET p = {first: 'Alix'}",
    ] {
        assert!(outcome(&session, statement).is_err(), "{statement}");
    }
    let rows = session
        .execute("MATCH (p:Person) RETURN p.first AS f, p.last AS l")
        .unwrap();
    assert_eq!(
        rows.rows(),
        [[Value::from("Alix"), Value::from("Gus")]],
        "nothing changed"
    );
}

/// A key property may change, to a tuple no other node holds; to one another node holds it may not.
#[test]
fn an_update_keeps_the_key_unique() {
    let db = keyed();
    let session = db.session();
    session
        .execute("INSERT (:Person {first: 'Vincent', last: 'Mia'})")
        .unwrap();
    assert!(
        outcome(
            &session,
            "MATCH (p:Person {first: 'Vincent'}) SET p.first = 'Alix', p.last = 'Gus'"
        )
        .is_err()
    );
    assert!(
        outcome(
            &session,
            "MATCH (p:Person {first: 'Vincent'}) SET p.last = 'Gus'"
        )
        .is_ok(),
        "Vincent Gus is free"
    );
    assert!(
        outcome(
            &session,
            "MATCH (p:Person {first: 'Vincent'}) SET p.first = 'Alix'"
        )
        .is_err(),
        "Alix Gus is taken"
    );
    assert!(
        outcome(
            &session,
            "MATCH (p:Person {first: 'Vincent'}) SET p += {first: 'Alix'}"
        )
        .is_err()
    );
    assert!(
        outcome(
            &session,
            "MATCH (p:Person {first: 'Alix'}) SET p.first = 'Alix'"
        )
        .is_ok(),
        "the same value again"
    );
}

/// A node that takes the label must hold a free key.
#[test]
fn a_label_added_must_bring_a_free_key() {
    let db = keyed();
    let session = db.session();
    session
        .execute("INSERT (:Guest {first: 'Alix', last: 'Gus'}), (:Guest {first: 'Mia'}), (:Guest {first: 'Jules', last: 'Hans'})")
        .unwrap();
    assert!(
        outcome(&session, "MATCH (g:Guest {first: 'Alix'}) SET g:Person").is_err(),
        "a taken key"
    );
    assert!(
        outcome(&session, "MATCH (g:Guest {first: 'Mia'}) SET g:Person").is_err(),
        "a missing property"
    );
    assert!(outcome(&session, "MATCH (g:Guest {first: 'Jules'}) SET g:Person").is_ok());
    assert_eq!(count(&session), 2);
}

/// Over data that already breaks it the constraint is refused, naming it, and nothing is kept.
#[test]
fn existing_data_that_breaks_it_refuses_the_constraint() {
    for (data, why) in [
        (
            "INSERT (:Person {first: 'Alix', last: 'Gus'}), (:Person {first: 'Alix', last: 'Gus'})",
            "a duplicate",
        ),
        ("INSERT (:Person {first: 'Alix'})", "a missing property"),
    ] {
        let db = GrafeoDB::new_in_memory();
        let session = db.session();
        session.execute(data).unwrap();
        let refused = outcome(&session, KEY);
        assert!(
            matches!(&refused, Err(text) if text.contains("person_key")),
            "{why}: {refused:?}"
        );
        let shown = session.execute("SHOW CONSTRAINTS").unwrap();
        assert_eq!(shown.row_count(), 0, "{why}: no constraint kept");
    }
}

/// Values compare as `=` does: numbers of one value are one key, a string and a number two.
#[test]
fn key_values_compare_as_equals() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    session
        .execute("CREATE CONSTRAINT n_key FOR (n:N) ON (n.k) NODE KEY")
        .unwrap();
    session.execute("INSERT (:N {k: 1})").unwrap();
    assert!(
        outcome(&session, "INSERT (:N {k: 1.0e0})").is_err(),
        "1 = 1.0"
    );
    assert!(
        outcome(&session, "INSERT (:N {k: 1.0})").is_err(),
        "1 = 1.0 as a decimal"
    );
    assert!(
        outcome(&session, "INSERT (:N {k: '1'})").is_ok(),
        "'1' is no number"
    );
}

/// It builds an index on its first property, named after it; dropping the constraint keeps the index.
#[test]
fn it_builds_an_index_on_its_first_property() {
    let db = keyed();
    let session = db.session();
    let indexes = format!("{:?}", session.execute("SHOW INDEXES").unwrap().rows());
    assert!(
        indexes.contains("person_key") && indexes.contains("first"),
        "{indexes}"
    );
    let constraints = format!("{:?}", session.execute("SHOW CONSTRAINTS").unwrap().rows());
    assert!(
        constraints.contains("person_key") && constraints.contains("NODE KEY"),
        "{constraints}"
    );
    session.execute("DROP CONSTRAINT person_key").unwrap();
    let indexes = format!("{:?}", session.execute("SHOW INDEXES").unwrap().rows());
    assert!(indexes.contains("person_key"), "the index stays: {indexes}");
    assert!(
        outcome(&session, "INSERT (:Person {first: 'Alix', last: 'Gus'})").is_ok(),
        "no longer a key"
    );
}

/// Two open transactions taking one key: the key is claimed when it is written, so the second
/// writer's insert fails with a write conflict (40000), both commits succeed after it, and one node
/// holds the key.
#[test]
fn two_transactions_cannot_both_take_a_key() {
    let db = keyed();
    let (mut a, mut b) = (db.session(), db.session());
    a.begin_transaction().unwrap();
    b.begin_transaction().unwrap();
    let one = outcome(&a, "INSERT (:Person {first: 'Mia', last: 'Hans'})");
    let two = outcome(&b, "INSERT (:Person {first: 'Mia', last: 'Hans'})");
    let three = a.commit().map_err(|error| error.to_string());
    let four = b.commit().map_err(|error| error.to_string());
    assert!(one.is_ok(), "{one:?}");
    let conflict = two.expect_err("the second insert of the key");
    assert!(
        conflict.starts_with("40000") && conflict.contains("Person"),
        "{conflict}"
    );
    assert!(three.is_ok(), "{three:?}");
    assert!(
        four.is_ok(),
        "the second commit, with nothing written: {four:?}"
    );
    let rows = db
        .session()
        .execute("MATCH (p:Person {first: 'Mia', last: 'Hans'}) RETURN count(p) AS c")
        .unwrap();
    assert_eq!(rows.rows()[0][0], Value::Int64(1));
}

/// The constraint and its effect survive a reopen.
#[test]
fn it_holds_after_a_reopen() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("key.grafeo");
    {
        let db = GrafeoDB::open(&path).unwrap();
        let session = db.session();
        session.execute(KEY).unwrap();
        session
            .execute("INSERT (:Person {first: 'Alix', last: 'Gus'})")
            .unwrap();
        drop(session);
        db.close().unwrap();
    }
    let db = GrafeoDB::open(&path).unwrap();
    let session = db.session();
    assert!(outcome(&session, "INSERT (:Person {first: 'Alix', last: 'Gus'})").is_err());
    assert!(outcome(&session, "INSERT (:Person {first: 'Mia'})").is_err());
    drop(session);
    db.close().unwrap();
}

/// Cypher writes the same constraint and it holds the same way.
#[test]
fn cypher_writes_the_same_key() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    session
        .execute_cypher(
            "CREATE CONSTRAINT person_key FOR (p:Person) REQUIRE (p.first, p.last) IS NODE KEY",
        )
        .unwrap();
    session
        .execute_cypher("CREATE (:Person {first: 'Alix', last: 'Gus'})")
        .unwrap();
    assert!(
        session
            .execute_cypher("CREATE (:Person {first: 'Alix', last: 'Gus'})")
            .is_err()
    );
    assert!(
        session
            .execute_cypher("CREATE (:Person {first: 'Mia'})")
            .is_err()
    );
    assert!(
        session
            .execute_cypher("MERGE (:Person {first: 'Alix', last: 'Gus'})")
            .is_ok(),
        "MERGE finds it"
    );
    assert_eq!(count(&session), 1);
}

/// Against UNIQUE plus NOT NULL on each property: UNIQUE takes several nodes without the property,
/// and a key of one property is that pair exactly.
#[test]
fn compared_with_unique_and_not_null() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    session
        .execute("CREATE CONSTRAINT email_unique FOR (u:User) ON (u.email) UNIQUE")
        .unwrap();
    assert!(
        outcome(
            &session,
            "INSERT (:User {name: 'Alix'}), (:User {name: 'Gus'})"
        )
        .is_ok(),
        "UNIQUE takes missing values"
    );
    session
        .execute("CREATE CONSTRAINT email_key FOR (k:Key) ON (k.email) NODE KEY")
        .unwrap();
    assert!(
        outcome(&session, "INSERT (:Key {name: 'Alix'})").is_err(),
        "NODE KEY requires it"
    );
    session
        .execute("CREATE CONSTRAINT both_unique FOR (b:Both) ON (b.email) UNIQUE")
        .unwrap();
    session
        .execute("CREATE CONSTRAINT both_required FOR (b:Both) ON (b.email) NOT NULL")
        .unwrap();
    for (statement, accepted) in [
        ("INSERT (:Key {email: 'alix@example.org'})", true),
        ("INSERT (:Both {email: 'alix@example.org'})", true),
        ("INSERT (:Key {email: 'alix@example.org'})", false),
        ("INSERT (:Both {email: 'alix@example.org'})", false),
        ("INSERT (:Key {name: 'Gus'})", false),
        ("INSERT (:Both {name: 'Gus'})", false),
    ] {
        assert_eq!(
            outcome(&session, statement).is_ok(),
            accepted,
            "{statement}"
        );
    }
}

/// A key holds in each graph on its own: the default graph's nodes do not take a key in another
/// graph, a graph made after the constraint gets it and its index, and within that graph the key
/// is required and unique.
#[test]
fn a_key_holds_in_each_graph_on_its_own() {
    let db = keyed();
    let session = db.session();
    session.execute("CREATE GRAPH branch").unwrap();
    session.execute("USE GRAPH branch").unwrap();
    for (statement, accepted, why) in [
        (
            "INSERT (:Person {first: 'Alix', last: 'Gus'})",
            true,
            "the default graph's key is free here",
        ),
        (
            "INSERT (:Person {first: 'Alix', last: 'Gus'})",
            false,
            "taken in this graph",
        ),
        (
            "INSERT (:Person {first: 'Mia'})",
            false,
            "required here too",
        ),
    ] {
        assert_eq!(outcome(&session, statement).is_ok(), accepted, "{why}");
    }
    assert_eq!(count(&session), 1, "this graph");
    let plan = format!(
        "{:?}",
        session
            .execute("EXPLAIN MATCH (p:Person {first: 'Alix', last: 'Gus'}) RETURN p")
            .unwrap()
            .rows()
    );
    assert!(plan.contains("[index: first]"), "the index here: {plan}");
    session.execute("USE GRAPH default").unwrap();
    assert_eq!(count(&session), 1, "the default graph");
}

/// The label decides: a node that holds it among others is held to the key, one without it is not,
/// and removing the label frees the node from the key and the key for another node.
#[test]
fn the_label_decides() {
    let db = keyed();
    let session = db.session();
    for (statement, accepted, why) in [
        (
            "INSERT (:Person:Employee {first: 'Alix', last: 'Gus'})",
            false,
            "another label besides",
        ),
        (
            "INSERT (:Employee {first: 'Alix', last: 'Gus'})",
            true,
            "without the label",
        ),
        (
            "MATCH (p:Person {first: 'Alix'}) REMOVE p:Person",
            true,
            "the label removed",
        ),
        (
            "MATCH (n {city: 'Amsterdam'}) REMOVE n.last",
            true,
            "no longer held to the key",
        ),
        (
            "INSERT (:Person {first: 'Alix', last: 'Gus'})",
            true,
            "the key is free again",
        ),
    ] {
        assert_eq!(outcome(&session, statement).is_ok(), accepted, "{why}");
    }
    assert_eq!(count(&session), 1);
}

/// Deleting a node frees its key, in the same transaction too; a rolled-back insert takes none.
#[test]
fn a_delete_frees_the_key() {
    let db = keyed();
    let mut session = db.session();
    session
        .execute("MATCH (p:Person {first: 'Alix'}) DELETE p")
        .unwrap();
    assert!(outcome(&session, "INSERT (:Person {first: 'Alix', last: 'Gus'})").is_ok());
    session.begin_transaction().unwrap();
    let deleted = outcome(&session, "MATCH (p:Person {first: 'Alix'}) DELETE p");
    let inserted = outcome(&session, "INSERT (:Person {first: 'Alix', last: 'Gus'})");
    let committed = session.commit().map_err(|error| error.to_string());
    assert!(deleted.is_ok(), "{deleted:?}");
    assert!(
        inserted.is_ok(),
        "deleted in this transaction: {inserted:?}"
    );
    assert!(committed.is_ok(), "{committed:?}");
    session.begin_transaction().unwrap();
    session
        .execute("INSERT (:Person {first: 'Mia', last: 'Hans'})")
        .unwrap();
    session.rollback().unwrap();
    assert!(
        outcome(&session, "INSERT (:Person {first: 'Mia', last: 'Hans'})").is_ok(),
        "the rolled-back insert took no key"
    );
    assert_eq!(count(&session), 2);
}

/// A key is checked at each write, not at the end of the statement: two nodes that swap their keys
/// in one `SET` are refused at the first assignment, for a NODE KEY as for a UNIQUE constraint,
/// and nothing changes. Moving one key through a free value first does the swap.
#[test]
fn a_key_is_checked_at_each_write() {
    for constraint in [
        KEY,
        "CREATE CONSTRAINT person_unique FOR (p:Person) ON (p.first, p.last) UNIQUE",
    ] {
        let db = GrafeoDB::new_in_memory();
        let session = db.session();
        session.execute(constraint).unwrap();
        session
            .execute("INSERT (:Person {first: 'Alix', last: 'Gus', n: 1}), (:Person {first: 'Alix', last: 'Mia', n: 2})")
            .unwrap();
        let swapped = outcome(
            &session,
            "MATCH (a:Person {n: 1}), (b:Person {n: 2}) SET a.last = 'Mia', b.last = 'Gus'",
        );
        assert!(
            matches!(&swapped, Err(text) if text.contains("GRAFEO-V001")),
            "{constraint}: {swapped:?}"
        );
        let lasts = "MATCH (p:Person) RETURN p.last AS last ORDER BY p.n";
        assert_eq!(
            session.execute(lasts).unwrap().rows(),
            [[Value::from("Gus")], [Value::from("Mia")]],
            "{constraint}: nothing changed"
        );
        for statement in [
            "MATCH (a:Person {n: 1}) SET a.last = 'Gus (moving)'",
            "MATCH (b:Person {n: 2}) SET b.last = 'Gus'",
            "MATCH (a:Person {n: 1}) SET a.last = 'Mia'",
        ] {
            session.execute(statement).unwrap();
        }
        assert_eq!(
            session.execute(lasts).unwrap().rows(),
            [[Value::from("Mia")], [Value::from("Gus")]],
            "{constraint}: swapped through a free value"
        );
    }
}

/// A label may have more than one key, and each holds.
#[test]
fn a_label_may_have_two_keys() {
    let db = keyed();
    let session = db.session();
    session
        .execute("CREATE CONSTRAINT person_city FOR (p:Person) ON (p.city) NODE KEY")
        .unwrap();
    for (statement, accepted, why) in [
        (
            "INSERT (:Person {first: 'Mia', last: 'Hans'})",
            false,
            "no city",
        ),
        (
            "INSERT (:Person {first: 'Mia', last: 'Hans', city: 'Amsterdam'})",
            false,
            "the city is taken",
        ),
        (
            "INSERT (:Person {first: 'Mia', last: 'Hans', city: 'Berlin'})",
            true,
            "both keys free",
        ),
    ] {
        assert_eq!(outcome(&session, statement).is_ok(), accepted, "{why}");
    }
}

/// A lookup by the key reads the index on its first property and finds the node; a lookup by the
/// other property alone has no index to read.
#[test]
fn a_lookup_by_the_key_reads_its_index() {
    let db = keyed();
    let session = db.session();
    let plan = |query: &str| {
        format!(
            "{:?}",
            session.execute(&format!("EXPLAIN {query}")).unwrap().rows()
        )
    };
    let by_key = "MATCH (p:Person {first: 'Alix', last: 'Gus'}) RETURN p.city AS city";
    assert!(plan(by_key).contains("[index: first]"), "{}", plan(by_key));
    assert_eq!(
        session.execute(by_key).unwrap().rows(),
        [[Value::from("Amsterdam")]]
    );
    let by_last = "MATCH (p:Person {last: 'Gus'}) RETURN p.city AS city";
    assert!(!plan(by_last).contains("[index:"), "{}", plan(by_last));
}

/// Every way a write breaks the key is refused with `GRAFEO-V001`, a schema violation.
#[test]
fn a_broken_key_is_a_schema_violation() {
    let db = keyed();
    let session = db.session();
    session
        .execute("INSERT (:Guest {first: 'Alix', last: 'Gus'})")
        .unwrap();
    for statement in [
        "INSERT (:Person {first: 'Alix', last: 'Gus'})",
        "INSERT (:Person {first: 'Mia'})",
        "MATCH (p:Person) SET p.last = null",
        "MATCH (p:Person) REMOVE p.first",
        "MATCH (g:Guest) SET g:Person",
    ] {
        let error = session.execute(statement).expect_err(statement).to_string();
        assert!(error.contains("GRAFEO-V001"), "{statement}: {error}");
    }
}

/// Lists are key values, compared element by element.
#[test]
fn a_list_is_a_key_value() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    session
        .execute("CREATE CONSTRAINT n_key FOR (n:N) ON (n.k) NODE KEY")
        .unwrap();
    for (statement, accepted) in [
        ("INSERT (:N {k: [1, 2]})", true),
        ("INSERT (:N {k: [1, 2]})", false),
        ("INSERT (:N {k: [2, 1]})", true),
        ("INSERT (:N {k: []})", true),
        ("INSERT (:N {k: []})", false),
    ] {
        assert_eq!(
            outcome(&session, statement).is_ok(),
            accepted,
            "{statement}"
        );
    }
}

/// The direct API is held to the key as a query is.
#[test]
fn the_direct_api_keeps_the_key() {
    let db = keyed();
    assert!(
        db.create_node_with_props(&["Person"], [("first", "Alix"), ("last", "Gus")])
            .is_err(),
        "a duplicate"
    );
    assert!(
        db.create_node_with_props(&["Person"], [("first", "Mia")])
            .is_err(),
        "a missing property"
    );
    assert!(db.create_node(&["Person"]).is_err(), "no properties");
    let mia = db
        .create_node_with_props(&["Person"], [("first", "Mia"), ("last", "Hans")])
        .unwrap();
    assert!(db.remove_node_property(mia, "last").is_err(), "a removal");
    assert!(
        db.set_node_property(mia, "last", Value::Null).is_err(),
        "a null"
    );
    assert!(
        db.set_node_property(mia, "first", Value::from("Alix"))
            .is_ok(),
        "Alix Hans is free"
    );
    assert!(
        db.set_node_property(mia, "last", Value::from("Gus"))
            .is_err(),
        "Alix Gus is taken"
    );
    let guest = db
        .create_node_with_props(&["Guest"], [("first", "Alix"), ("last", "Gus")])
        .unwrap();
    assert!(db.add_node_label(guest, "Person").is_err(), "a label added");
    assert_eq!(count(&db.session()), 2);
}

const CRASH_PATH_VAR: &str = "GRAFEO_NODE_KEY_CRASH_PATH";

/// Creates the key and one node in the database at `GRAFEO_NODE_KEY_CRASH_PATH`, then exits with
/// the database open: the reopen replays both from the log.
#[test]
fn key_crash_child() {
    let Some(path) = std::env::var_os(CRASH_PATH_VAR) else {
        return;
    };
    let db = GrafeoDB::open(std::path::PathBuf::from(path)).unwrap();
    let session = db.session();
    session.execute(KEY).unwrap();
    session
        .execute("INSERT (:Person {first: 'Alix', last: 'Gus'})")
        .unwrap();
    std::process::exit(0);
}

/// After a crash, as after a close, the key holds, is listed, and keeps its index.
#[test]
fn it_holds_after_a_crash() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("key.grafeo");
    let status = grafeo_common::testing::child_process::run(
        grafeo_common::child_test!("key_crash_child").env(CRASH_PATH_VAR, &path),
    )
    .unwrap();
    assert!(status.success(), "the child process failed");
    let db = GrafeoDB::open(&path).unwrap();
    let session = db.session();
    assert!(outcome(&session, "INSERT (:Person {first: 'Alix', last: 'Gus'})").is_err());
    assert!(outcome(&session, "INSERT (:Person {first: 'Mia'})").is_err());
    let constraints = format!("{:?}", session.execute("SHOW CONSTRAINTS").unwrap().rows());
    assert!(constraints.contains("NODE KEY"), "{constraints}");
    let plan = format!(
        "{:?}",
        session
            .execute("EXPLAIN MATCH (p:Person {first: 'Alix', last: 'Gus'}) RETURN p")
            .unwrap()
            .rows()
    );
    assert!(plan.contains("[index: first]"), "{plan}");
    assert_eq!(count(&session), 1);
    drop(session);
    db.close().unwrap();
}
