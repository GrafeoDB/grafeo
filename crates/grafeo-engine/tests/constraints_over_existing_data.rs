//! A rule is refused over data that already breaks it (#729).
//!
//! Tests first, from the ruling: `CREATE CONSTRAINT` over committed data that breaks it fails with
//! `GRAFEO-V001` and leaves nothing, for UNIQUE, NODE KEY, NOT NULL and EXISTS. Constraints name
//! node labels only, so that is what is covered. A file from a release that accepted such a rule
//! still opens, with the rule enforced on later writes.
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test main constraints_over_existing_data::
//! ```

#![cfg(all(feature = "gql", feature = "cypher"))]

use grafeo_common::utils::error::Result;
use grafeo_engine::GrafeoDB;
use grafeo_engine::session::Session;

#[derive(Clone, Copy, PartialEq, Debug)]
enum Language {
    Gql,
    Cypher,
}
use Language::{Cypher, Gql};

fn run(session: &Session, language: Language, statement: &str) -> Result<()> {
    match language {
        Gql => session.execute(statement).map(|_| ()),
        Cypher => session.execute_cypher(statement).map(|_| ()),
    }
}

fn constraints(session: &Session) -> Vec<String> {
    let mut rows: Vec<String> = session
        .execute("SHOW CONSTRAINTS")
        .unwrap()
        .rows()
        .iter()
        .map(|row| format!("{row:?}"))
        .collect();
    rows.sort();
    rows
}

/// `(name, language, the rule, data that breaks it, a later write it must not refuse)`.
const BROKEN: [(&str, Language, &str, &str, &str); 10] = [
    (
        "UNIQUE, two of one value",
        Gql,
        "CREATE CONSTRAINT c1 FOR (n:File) ON (n.id) UNIQUE",
        "INSERT (:File {id: 'a'}), (:File {id: 'a'})",
        "INSERT (:File {id: 'a'})",
    ),
    (
        "UNIQUE over two properties, one pair twice",
        Gql,
        "CREATE CONSTRAINT c1 FOR (n:File) ON (n.id, n.part) UNIQUE",
        "INSERT (:File {id: 'a', part: 1}), (:File {id: 'a', part: 1})",
        "INSERT (:File {id: 'a', part: 1})",
    ),
    (
        "NODE KEY, two of one value",
        Gql,
        "CREATE CONSTRAINT c1 FOR (n:File) ON (n.id) NODE KEY",
        "INSERT (:File {id: 'a'}), (:File {id: 'a'})",
        "INSERT (:File {id: 'a'})",
    ),
    (
        "NODE KEY, a missing property",
        Gql,
        "CREATE CONSTRAINT c1 FOR (n:File) ON (n.id) NODE KEY",
        "INSERT (:File {other: 1})",
        "INSERT (:File {other: 2})",
    ),
    (
        "NOT NULL, a missing property",
        Gql,
        "CREATE CONSTRAINT c1 FOR (n:File) ON (n.must) NOT NULL",
        "INSERT (:File {id: 'a'})",
        "INSERT (:File {id: 'b'})",
    ),
    (
        "EXISTS, a missing property",
        Gql,
        "CREATE CONSTRAINT c1 FOR (n:File) ON (n.must) EXISTS",
        "INSERT (:File {id: 'a'})",
        "INSERT (:File {id: 'b'})",
    ),
    (
        "openCypher UNIQUE",
        Cypher,
        "CREATE CONSTRAINT c1 FOR (n:File) REQUIRE n.id IS UNIQUE",
        "INSERT (:File {id: 'a'}), (:File {id: 'a'})",
        "INSERT (:File {id: 'a'})",
    ),
    (
        "openCypher NOT NULL",
        Cypher,
        "CREATE CONSTRAINT c1 FOR (n:File) REQUIRE n.must IS NOT NULL",
        "INSERT (:File {id: 'a'})",
        "INSERT (:File {id: 'b'})",
    ),
    (
        "openCypher NODE KEY, two of one value",
        Cypher,
        "CREATE CONSTRAINT c1 FOR (n:File) REQUIRE n.id IS NODE KEY",
        "INSERT (:File {id: 'a'}), (:File {id: 'a'})",
        "INSERT (:File {id: 'a'})",
    ),
    (
        "openCypher NODE KEY, a missing property",
        Cypher,
        "CREATE CONSTRAINT c1 FOR (n:File) REQUIRE n.id IS NODE KEY",
        "INSERT (:File {other: 1})",
        "INSERT (:File {other: 2})",
    ),
];

#[test]
fn a_rule_the_data_breaks_is_refused_and_leaves_nothing() {
    for (name, language, rule, data, later) in BROKEN {
        let db = GrafeoDB::new_in_memory();
        let session = db.session();
        session.execute(data).unwrap();
        let before = constraints(&session);
        let error = run(&session, language, rule)
            .err()
            .unwrap_or_else(|| panic!("{name}: accepted"));
        assert!(error.to_string().contains("GRAFEO-V001"), "{name}: {error}");
        assert_eq!(constraints(&session), before, "{name}: left a rule");
        assert_eq!(
            constraints(&db.session()),
            before,
            "{name}: another session"
        );
        // No rule stands: the write it would refuse passes.
        session
            .execute(later)
            .unwrap_or_else(|error| panic!("{name}: the later write: {error}"));
    }
}

#[test]
fn it_leaves_nothing_after_a_reopen() {
    let dir = tempfile::tempdir().unwrap();
    for (number, (name, language, rule, data, later)) in BROKEN.into_iter().enumerate() {
        let path = dir.path().join(format!("refused-{number}.grafeo"));
        let db = GrafeoDB::open(&path).unwrap();
        db.session().execute(data).unwrap();
        assert!(run(&db.session(), language, rule).is_err(), "{name}");
        db.close().unwrap();
        let db = GrafeoDB::open(&path).unwrap();
        assert!(constraints(&db.session()).is_empty(), "{name}: left a rule");
        db.session()
            .execute(later)
            .unwrap_or_else(|error| panic!("{name}: the later write: {error}"));
        db.close().unwrap();
    }
}

/// The name is free again, and the rule passes once the data meets it.
#[test]
fn the_rule_passes_once_the_data_meets_it() {
    for (name, cleanup) in [
        ("deleted", "MATCH (f:File) DELETE f"),
        ("fixed", "MATCH (f:File) SET f.id = id(f), f.must = 1"),
    ] {
        let db = GrafeoDB::new_in_memory();
        let session = db.session();
        session
            .execute("INSERT (:File {id: 'a'}), (:File {id: 'a'})")
            .unwrap();
        let unique = "CREATE CONSTRAINT c1 FOR (n:File) ON (n.id) UNIQUE";
        assert!(session.execute(unique).is_err(), "{name}");
        session.execute(cleanup).unwrap();
        session
            .execute(unique)
            .unwrap_or_else(|error| panic!("{name}: {error}"));
        assert_eq!(constraints(&session).len(), 1, "{name}");
    }
}

/// `(name, the rule, data that meets it)`: what does not count as breaking.
#[test]
fn data_that_meets_the_rule_is_accepted() {
    for (name, rule, data) in [
        (
            "no rows",
            "CREATE CONSTRAINT c1 FOR (n:File) ON (n.id) UNIQUE",
            "INSERT (:Other {id: 'a'})",
        ),
        (
            "another label shares the value",
            "CREATE CONSTRAINT c1 FOR (n:File) ON (n.id) UNIQUE",
            "INSERT (:File {id: 'a'}), (:Other {id: 'a'})",
        ),
        (
            "different values",
            "CREATE CONSTRAINT c1 FOR (n:File) ON (n.id) UNIQUE",
            "INSERT (:File {id: 'a'}), (:File {id: 'b'})",
        ),
        (
            "a unique rule does not require the property",
            "CREATE CONSTRAINT c1 FOR (n:File) ON (n.id) UNIQUE",
            "INSERT (:File {other: 1}), (:File {other: 2})",
        ),
        (
            "two properties differing in one",
            "CREATE CONSTRAINT c1 FOR (n:File) ON (n.id, n.part) UNIQUE",
            "INSERT (:File {id: 'a', part: 1}), (:File {id: 'a', part: 2})",
        ),
        (
            "the property on every row",
            "CREATE CONSTRAINT c1 FOR (n:File) ON (n.must) NOT NULL",
            "INSERT (:File {must: 1}), (:File {must: 1})",
        ),
        (
            "a key that holds",
            "CREATE CONSTRAINT c1 FOR (n:File) ON (n.id) NODE KEY",
            "INSERT (:File {id: 'a'}), (:File {id: 'b'})",
        ),
    ] {
        let db = GrafeoDB::new_in_memory();
        let session = db.session();
        session.execute(data).unwrap();
        session
            .execute(rule)
            .unwrap_or_else(|error| panic!("{name}: refused: {error}"));
        assert_eq!(constraints(&session).len(), 1, "{name}");
    }
}

/// Rows deleted earlier do not count, and neither do another session's uncommitted writes:
/// those meet the check at their own commit (see `rules_against_open_writers.rs`).
#[test]
fn only_what_is_committed_counts() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    session
        .execute("INSERT (:File {id: 'a', keep: 1}), (:File {id: 'a', keep: 0})")
        .unwrap();
    session
        .execute("MATCH (f:File {keep: 0}) DELETE f")
        .unwrap();
    let mut writer = db.session();
    writer.begin_transaction().unwrap();
    writer.execute("INSERT (:File {id: 'a', keep: 2})").unwrap();
    session
        .execute("CREATE CONSTRAINT c1 FOR (n:File) ON (n.id) UNIQUE")
        .unwrap_or_else(|error| panic!("{error}"));
    writer.rollback().unwrap();
    assert_eq!(constraints(&session).len(), 1);
}

/// In a transaction the check sees the transaction's own writes.
#[test]
fn a_transaction_is_checked_with_its_own_writes() {
    let db = GrafeoDB::new_in_memory();
    db.session().execute("INSERT (:File {id: 'a'})").unwrap();
    let mut session = db.session();
    session.begin_transaction().unwrap();
    session.execute("INSERT (:File {id: 'a'})").unwrap();
    let error = session
        .execute("CREATE CONSTRAINT c1 FOR (n:File) ON (n.id) UNIQUE")
        .expect_err("its own write breaks the rule");
    assert!(error.to_string().contains("GRAFEO-V001"), "{error}");
    session.rollback().unwrap();
    assert_eq!(constraints(&db.session()).len(), 0);

    // The other way round: it removes the duplicate itself, so the rule passes.
    let db = GrafeoDB::new_in_memory();
    db.session()
        .execute("INSERT (:File {id: 'a', keep: 1}), (:File {id: 'a', keep: 0})")
        .unwrap();
    let mut session = db.session();
    session.begin_transaction().unwrap();
    session
        .execute("MATCH (f:File {keep: 0}) DELETE f")
        .unwrap();
    session
        .execute("CREATE CONSTRAINT c1 FOR (n:File) ON (n.id) UNIQUE")
        .unwrap_or_else(|error| panic!("{error}"));
    session.commit().unwrap();
    assert_eq!(constraints(&db.session()).len(), 1);
}

// ---------------------------------------------------------------------------
// A file from a release that accepted such a rule
// ---------------------------------------------------------------------------

/// Written by the 0.6.1 release: two `:File {id: 'a'}` and a `:File {id: 'b'}` with no `must`,
/// under a UNIQUE on `id` and a NOT NULL on `must` created over them.
fn violating_file(dir: &std::path::Path) -> std::path::PathBuf {
    let from = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/released/0.6.1/violating-rules.grafeo");
    let to = dir.join("violating-rules.grafeo");
    std::fs::copy(from, &to).unwrap();
    to
}

#[test]
fn a_file_holding_a_rule_its_data_breaks_still_opens() {
    let dir = tempfile::tempdir().unwrap();
    let path = violating_file(dir.path());
    let db = GrafeoDB::open(&path).unwrap_or_else(|error| panic!("{error}"));
    let session = db.session();
    assert_eq!(constraints(&session).len(), 2, "the rules");
    assert_eq!(
        session
            .execute("MATCH (f:File) RETURN f.id")
            .unwrap()
            .row_count(),
        3,
        "the rows that break them"
    );
    drop(session);
    db.close().unwrap();
    // And again, from what this build wrote.
    let db = GrafeoDB::open(&path).unwrap();
    assert_eq!(constraints(&db.session()).len(), 2);
    db.close().unwrap();
}

#[test]
fn a_file_holding_a_rule_its_data_breaks_enforces_it_on_later_writes() {
    let dir = tempfile::tempdir().unwrap();
    let path = violating_file(dir.path());
    let db = GrafeoDB::open(&path).unwrap();
    let session = db.session();
    for refused in [
        "INSERT (:File {id: 'a', must: 3})",
        "INSERT (:File {id: 'c'})",
    ] {
        let error = session
            .execute(refused)
            .err()
            .unwrap_or_else(|| panic!("{refused}: accepted"));
        assert!(
            error.to_string().contains("GRAFEO-V001"),
            "{refused}: {error}"
        );
    }
    session
        .execute("INSERT (:File {id: 'c', must: 3})")
        .unwrap_or_else(|error| panic!("a write that meets both: {error}"));
    // A write that does not touch what breaks the rule goes on, so a broken file stays usable,
    // and one that repairs it passes.
    session
        .execute("MATCH (f:File {id: 'b'}) SET f.must = 4")
        .unwrap_or_else(|error| panic!("a repair: {error}"));
    session
        .execute("MATCH (f:File {must: 2}) DELETE f")
        .unwrap_or_else(|error| panic!("a repair: {error}"));
    drop(session);
    db.close().unwrap();
}

/// Read-only open of such a file works too.
#[test]
fn a_file_holding_a_rule_its_data_breaks_opens_read_only() {
    let dir = tempfile::tempdir().unwrap();
    let path = violating_file(dir.path());
    let db = GrafeoDB::open_read_only(&path).unwrap_or_else(|error| panic!("{error}"));
    assert_eq!(
        db.session()
            .execute("MATCH (f:File) RETURN f.id")
            .unwrap()
            .row_count(),
        3
    );
    db.close().unwrap();
}

// ---------------------------------------------------------------------------
// Every statement that adds a rule
// ---------------------------------------------------------------------------

fn status(error: &grafeo_common::utils::error::Error) -> String {
    grafeo_common::utils::gqlstatus::GqlStatus::from(error)
        .as_str()
        .to_string()
}

fn node_types(session: &Session) -> Vec<String> {
    let mut rows: Vec<String> = session
        .execute("SHOW NODE TYPES")
        .unwrap()
        .rows()
        .iter()
        .map(|row| format!("{row:?}"))
        .collect();
    rows.sort();
    rows
}

/// A rule in a type over elements that already break it, whether the type is new or a property
/// is added: `GRAFEO-V001`, and nothing changes. Required properties and property types, on nodes
/// and edges.
#[test]
fn a_type_rule_the_data_breaks_is_refused_and_leaves_nothing() {
    for (name, setup, read, statement, later) in [
        (
            "CREATE NODE TYPE, a missing property",
            vec!["INSERT (:City {size: 3})"],
            "SHOW NODE TYPES",
            "CREATE NODE TYPE City (name STRING NOT NULL)",
            "INSERT (:City {size: 4})",
        ),
        (
            "CREATE NODE TYPE, a value of another type",
            vec!["INSERT (:City {name: 'Amsterdam'})"],
            "SHOW NODE TYPES",
            "CREATE NODE TYPE City (name INT64)",
            "INSERT (:City {name: 'Paris'})",
        ),
        (
            "ALTER NODE TYPE, a new required property",
            vec![
                "CREATE NODE TYPE City (name STRING)",
                "INSERT (:City {name: 'Amsterdam'})",
            ],
            "SHOW NODE TYPES",
            "ALTER NODE TYPE City ADD PROPERTY size INTEGER NOT NULL",
            "INSERT (:City {name: 'Paris'})",
        ),
        (
            "CREATE EDGE TYPE, a missing property",
            vec!["INSERT (:A)-[:ROAD {toll: 1}]->(:B)"],
            "SHOW EDGE TYPES",
            "CREATE EDGE TYPE ROAD (km INTEGER NOT NULL)",
            "INSERT (:A)-[:ROAD {toll: 2}]->(:B)",
        ),
        (
            "CREATE EDGE TYPE, a value of another type",
            vec!["INSERT (:A)-[:ROAD {km: 'far'}]->(:B)"],
            "SHOW EDGE TYPES",
            "CREATE EDGE TYPE ROAD (km INTEGER)",
            "INSERT (:A)-[:ROAD {km: 'near'}]->(:B)",
        ),
        (
            "ALTER EDGE TYPE, a new required property",
            vec![
                "CREATE EDGE TYPE ROAD (km INTEGER)",
                "INSERT (:A)-[:ROAD {km: 1}]->(:B)",
            ],
            "SHOW EDGE TYPES",
            "ALTER EDGE TYPE ROAD ADD PROPERTY toll INTEGER NOT NULL",
            "INSERT (:A)-[:ROAD {km: 2}]->(:B)",
        ),
    ] {
        let db = GrafeoDB::new_in_memory();
        let session = db.session();
        for statement in setup {
            session.execute(statement).unwrap();
        }
        let shown = |session: &Session| {
            let mut rows: Vec<String> = session
                .execute(read)
                .unwrap()
                .rows()
                .iter()
                .map(|row| format!("{row:?}"))
                .collect();
            rows.sort();
            rows
        };
        let before = shown(&session);
        let error = session
            .execute(statement)
            .expect_err(&format!("{name}: accepted"));
        assert!(error.to_string().contains("GRAFEO-V001"), "{name}: {error}");
        assert_eq!(shown(&session), before, "{name}: left the rule");
        session
            .execute(later)
            .unwrap_or_else(|error| panic!("{name}: the later write: {error}"));
    }
}

/// In a transaction the check runs at the statement or under the commit lock: either way, what
/// another session committed since cannot get past it, and nothing stays.
#[test]
fn a_type_rule_in_a_transaction_checks_what_others_committed_meanwhile() {
    let db = GrafeoDB::new_in_memory();
    let before = node_types(&db.session());
    let mut session = db.session();
    session.begin_transaction().unwrap();
    session
        .execute("CREATE NODE TYPE City (name STRING NOT NULL)")
        .unwrap();
    db.session().execute("INSERT (:City {size: 3})").unwrap();
    assert!(
        session.commit().is_err(),
        "a rule the data breaks committed"
    );
    assert_eq!(node_types(&db.session()), before);
}

/// A type with no properties or constraints has nothing to check, over any data.
#[test]
fn a_type_with_no_rule_passes_over_any_data() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    session
        .execute("INSERT (:City {size: 3}), (:City {name: 1})")
        .unwrap();
    session
        .execute("CREATE NODE TYPE City")
        .unwrap_or_else(|error| panic!("{error}"));
}

/// A graph bound to a type whose data a change to the type would break: `G2000`, nothing changes.
#[test]
fn a_graph_type_change_a_bound_graph_breaks_is_refused_with_g2000() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    for statement in [
        "CREATE NODE TYPE City (name STRING)",
        "CREATE NODE TYPE Town (name STRING)",
        "CREATE GRAPH TYPE travel (NODE TYPE City, NODE TYPE Town)",
        "CREATE GRAPH trips TYPED travel",
        "USE GRAPH trips",
        "INSERT (:City {name: 'Amsterdam'})",
        "USE GRAPH default",
    ] {
        session.execute(statement).unwrap();
    }
    let shown = |session: &Session| {
        let mut rows: Vec<String> = session
            .execute("SHOW GRAPH TYPES")
            .unwrap()
            .rows()
            .iter()
            .map(|row| format!("{row:?}"))
            .collect();
        rows.sort();
        rows
    };
    let before = shown(&session);
    let error = session
        .execute("ALTER GRAPH TYPE travel DROP NODE TYPE City")
        .expect_err("the bound graph holds a City");
    assert_eq!(status(&error), "G2000", "{error}");
    assert_eq!(shown(&session), before, "the type changed");
    // A part the data does not use may go.
    session
        .execute("ALTER GRAPH TYPE travel DROP NODE TYPE Town")
        .unwrap_or_else(|error| panic!("a part no node uses: {error}"));
}

/// A copy into a type its source does not conform to, outside a transaction: `G2000`, no graph.
#[test]
fn a_copy_over_a_bound_type_its_source_breaks_is_refused_with_g2000() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    for statement in [
        "CREATE NODE TYPE City (name STRING)",
        "CREATE GRAPH TYPE travel (NODE TYPE City)",
        "CREATE GRAPH source",
        "USE GRAPH source",
        "INSERT (:Other {name: 'Jules'})",
        "USE GRAPH default",
    ] {
        session.execute(statement).unwrap();
    }
    let error = session
        .execute("CREATE GRAPH clone TYPED travel AS COPY OF source")
        .expect_err("the source holds what the type does not allow");
    assert_eq!(status(&error), "G2000", "{error}");
    assert!(
        session.execute("USE GRAPH clone").is_err(),
        "the refused copy left a graph"
    );
    // A source that conforms copies.
    for statement in [
        "CREATE GRAPH good",
        "USE GRAPH good",
        "INSERT (:City {name: 'Amsterdam'})",
        "USE GRAPH default",
        "CREATE GRAPH clone TYPED travel AS COPY OF good",
    ] {
        session
            .execute(statement)
            .unwrap_or_else(|error| panic!("{statement}: {error}"));
    }
    session.execute("USE GRAPH clone").unwrap();
    assert_eq!(
        session
            .execute("MATCH (c:City) RETURN c.name")
            .unwrap()
            .row_count(),
        1
    );
    session.execute("USE GRAPH default").unwrap();
    // A source with no type copies and takes a `LIKE`, as the copy keeps what it has.
    session
        .execute("CREATE GRAPH plain_copy AS COPY OF source")
        .unwrap_or_else(|error| panic!("{error}"));
    session
        .execute("CREATE GRAPH plain_like LIKE source")
        .unwrap_or_else(|error| panic!("{error}"));
}

/// The types of `atlas` over a bound graph holding `data`.
fn atlas_over(data: Option<&str>, open: bool) -> GrafeoDB {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    session
        .execute("CREATE NODE TYPE City (name STRING)")
        .unwrap();
    session
        .execute("CREATE EDGE TYPE ROAD (km INTEGER)")
        .unwrap();
    let graph_type = if open {
        "CREATE GRAPH TYPE atlas { node_types: [City], edge_types: [ROAD], open: true }"
    } else {
        "CREATE GRAPH TYPE atlas (NODE TYPE City, EDGE TYPE ROAD)"
    };
    session.execute(graph_type).unwrap();
    session.execute("CREATE GRAPH maps TYPED atlas").unwrap();
    if let Some(data) = data {
        session.execute("USE GRAPH maps").unwrap();
        session.execute(data).unwrap();
    }
    db
}

const ONE_NODE: &str = "INSERT (:City {name: 'Amsterdam'})";
const ONE_EDGE: &str =
    "INSERT (:City {name: 'Amsterdam'})-[:ROAD {km: 1}]->(:City {name: 'Paris'})";
const DROP_NODES: &str = "ALTER GRAPH TYPE atlas DROP NODE TYPE City";
const DROP_EDGES: &str = "ALTER GRAPH TYPE atlas DROP EDGE TYPE ROAD";

/// Dropping the last type of a side of a closed type over a graph holding elements on that side
/// would allow none of them: `G2000`, in those words, and nothing changes.
#[test]
fn the_last_type_of_a_side_is_refused_while_the_graph_uses_it() {
    for (name, data, drop_it, words) in [
        (
            "the last node type",
            ONE_NODE,
            DROP_NODES,
            "would allow no nodes",
        ),
        (
            "the last edge type",
            ONE_EDGE,
            DROP_EDGES,
            "would allow no edges",
        ),
    ] {
        let db = atlas_over(Some(data), false);
        let session = db.session();
        let error = session
            .execute(drop_it)
            .expect_err(&format!("{name}: dropped with the data there"));
        assert_eq!(status(&error), "G2000", "{name}: {error}");
        assert!(error.to_string().contains(words), "{name}: {error}");
        session.execute("USE GRAPH maps").unwrap();
        session
            .execute("INSERT (:City {name: 'Berlin'})")
            .unwrap_or_else(|error| panic!("{name}: the type changed: {error}"));
    }
}

/// With nothing on that side it passes, whatever the other side holds, and an open type passes.
#[test]
fn the_last_type_of_a_side_passes_with_nothing_on_it() {
    for (name, data, open, drop_it) in [
        ("no data, the last node type", None, false, DROP_NODES),
        ("no data, the last edge type", None, false, DROP_EDGES),
        (
            "nodes but no edges, the last edge type",
            Some(ONE_NODE),
            false,
            DROP_EDGES,
        ),
        (
            "an open type, the last node type",
            Some(ONE_NODE),
            true,
            DROP_NODES,
        ),
        (
            "an open type, the last edge type",
            Some(ONE_EDGE),
            true,
            DROP_EDGES,
        ),
    ] {
        let db = atlas_over(data, open);
        db.session()
            .execute(drop_it)
            .unwrap_or_else(|error| panic!("{name}: {error}"));
    }
}

/// A last type of a side goes when the elements have gone.
#[test]
fn the_last_type_goes_once_the_elements_have_gone() {
    let db = atlas_over(Some(ONE_NODE), false);
    let session = db.session();
    assert!(session.execute(DROP_NODES).is_err());
    session.execute("USE GRAPH maps").unwrap();
    session.execute("MATCH (c:City) DELETE c").unwrap();
    session.execute("USE GRAPH default").unwrap();
    session
        .execute(DROP_NODES)
        .unwrap_or_else(|error| panic!("{error}"));
}

/// A new graph type binds no graph, so it runs no data check: another graph's rows that would
/// break it do not stop it, and it leaves them as they are.
#[test]
fn a_new_graph_type_runs_no_data_check() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    session.execute("INSERT (:Person {age: 1})").unwrap();
    session
        .execute("CREATE GRAPH TYPE t (NODE TYPE Person (name STRING NOT NULL, age FLOAT64))")
        .unwrap_or_else(|error| panic!("{error}"));
    let rows = session
        .execute("MATCH (p:Person) RETURN p.age, p.name")
        .unwrap();
    assert_eq!(
        rows.rows(),
        &[vec![
            grafeo_common::types::Value::Int64(1),
            grafeo_common::types::Value::Null
        ]],
        "the rows are left alone"
    );
}

/// A graph type a graph is bound to cannot be replaced, whether its data would fit or not: the
/// error says to drop the graph first, and the old type stands.
#[test]
fn a_bound_graph_type_cannot_be_replaced() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    for statement in [
        "CREATE NODE TYPE City (name STRING)",
        "CREATE NODE TYPE Town (name STRING)",
        "CREATE GRAPH TYPE atlas (NODE TYPE City, NODE TYPE Town)",
        "CREATE GRAPH maps TYPED atlas",
        "USE GRAPH maps",
        "INSERT (:City {name: 'Amsterdam'})",
        "USE GRAPH default",
    ] {
        session.execute(statement).unwrap();
    }
    let shown = |session: &Session| {
        let mut rows: Vec<String> = session
            .execute("SHOW GRAPH TYPES")
            .unwrap()
            .rows()
            .iter()
            .map(|row| format!("{row:?}"))
            .collect();
        rows.sort();
        rows
    };
    let before = shown(&session);
    for (name, statement) in [
        (
            "data it would break",
            "CREATE OR REPLACE GRAPH TYPE atlas (NODE TYPE Town)",
        ),
        (
            "data it would fit",
            "CREATE OR REPLACE GRAPH TYPE atlas (NODE TYPE City)",
        ),
    ] {
        let error = session
            .execute(statement)
            .expect_err(&format!("{name}: replaced"));
        assert!(
            error.to_string().contains("drop the graph first"),
            "{name}: {error}"
        );
        assert_eq!(shown(&session), before, "{name}: the old type stands");
    }
    session.execute("USE GRAPH maps").unwrap();
    session
        .execute("INSERT (:Town {name: 'Delft'})")
        .unwrap_or_else(|error| panic!("the old type allows a Town: {error}"));
}

/// With no graph bound to it, a replacement checks nothing.
#[test]
fn replacing_an_unbound_graph_type_runs_no_data_check() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    session.execute("INSERT (:Town {name: 3})").unwrap();
    session
        .execute("CREATE NODE TYPE City (name STRING)")
        .unwrap();
    session
        .execute("CREATE GRAPH TYPE atlas (NODE TYPE City)")
        .unwrap();
    session
        .execute("CREATE OR REPLACE GRAPH TYPE atlas (NODE TYPE Town (name STRING NOT NULL))")
        .unwrap_or_else(|error| panic!("{error}"));
}
