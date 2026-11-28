//! A transaction's change events are its net changes (#679): later changes, a delete included,
//! fold into the create, and a create folded with its delete leaves nothing. So a node or an edge
//! created and deleted in one transaction reports no events (with the edges a `DETACH DELETE`
//! removes), an edge created and deleted beside a node that stays leaves only the node's create,
//! and an entity that existed before, updated and then deleted, reports its delete alone, with
//! what it held before the transaction. A triple inserted and deleted in one transaction, or an
//! existing one deleted and inserted again, reports nothing. Updates to an entity that exists
//! before and after the transaction are not folded: each write reports its own update.
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test main cdc_net_changes::
//! ```

#![cfg(all(feature = "cdc", feature = "gql"))]

use grafeo_common::types::{EpochId, NodeId, Value};
use grafeo_engine::cdc::{ChangeEvent, ChangeKind, EntityId};
use grafeo_engine::{Config, GrafeoDB};

/// Alix and Gus, with Alix knowing Gus, committed before each case.
const SETUP: &[&str] =
    &["INSERT (:Person {name: 'Alix'})-[:KNOWS {since: 2019}]->(:Person {name: 'Gus', age: 28})"];

/// A node by its name, and its id once the transaction committed (for a node created in it) or
/// before it ran.
type Ids = std::collections::HashMap<String, u64>;

/// An event a case expects, field for field: its kind and entity (a node by name, or an edge by
/// its endpoints' names), the node a `DETACH DELETE` removed it with, its labels and its before
/// and after properties.
struct Want {
    kind: ChangeKind,
    node: Option<&'static str>,
    edge: Option<(&'static str, &'static str)>,
    deleted_with: Option<&'static str>,
    labels: Option<&'static [&'static str]>,
    before: Option<Vec<(&'static str, Value)>>,
    after: Option<Vec<(&'static str, Value)>>,
}

impl Want {
    /// Whether `event` is this one, field for field.
    fn matches(&self, event: &ChangeEvent, ids: &Ids) -> bool {
        let id = |name: &str| ids[name];
        let map = |properties: &Option<Vec<(&'static str, Value)>>| {
            properties.as_ref().map(|properties| {
                properties
                    .iter()
                    .map(|(key, value)| ((*key).to_string(), value.clone()))
                    .collect::<std::collections::HashMap<_, _>>()
            })
        };
        let entity = match (self.node, self.edge) {
            (Some(name), _) => event.entity_id == EntityId::Node(NodeId::new(id(name))),
            (_, Some((src, dst))) => {
                matches!(event.entity_id, EntityId::Edge(_))
                    && (event.src_id, event.dst_id) == (Some(id(src)), Some(id(dst)))
            }
            _ => false,
        };
        entity
            && event.kind == self.kind
            && event.deleted_with == self.deleted_with.map(id)
            && event.labels
                == self
                    .labels
                    .map(|labels| labels.iter().map(ToString::to_string).collect())
            && event.before == map(&self.before)
            && event.after == map(&self.after)
    }
}

/// The five cases: a name, the transaction's statements, and its net events, exactly.
fn cases() -> Vec<(&'static str, Vec<&'static str>, Vec<Want>)> {
    vec![
        (
            "a node created and deleted",
            vec![
                "INSERT (:Temp {name: 'Hans'})",
                "MATCH (t:Temp {name: 'Hans'}) DELETE t",
            ],
            vec![],
        ),
        (
            "an edge created and deleted",
            vec![
                "MATCH (a:Person {name: 'Gus'}), (b:Person {name: 'Alix'}) INSERT (a)-[:LIKES {w: 1}]->(b)",
                "MATCH ()-[l:LIKES]->() DELETE l",
            ],
            vec![],
        ),
        (
            "a created node with created edges, detach deleted",
            vec![
                "INSERT (:Person {name: 'Mia'})",
                "MATCH (m:Person {name: 'Mia'}), (a:Person {name: 'Alix'}) \
                 INSERT (m)-[:KNOWS]->(a), (a)-[:KNOWS]->(m), (m)-[:LIKES]->(m)",
                "MATCH (m:Person {name: 'Mia'}) DETACH DELETE m",
            ],
            vec![],
        ),
        (
            "a created node whose created edge is deleted",
            vec![
                "INSERT (:Person {name: 'Butch', age: 52})",
                "MATCH (b:Person {name: 'Butch'}), (a:Person {name: 'Alix'}) INSERT (b)-[:KNOWS]->(a)",
                "MATCH (:Person {name: 'Butch'})-[k:KNOWS]->() DELETE k",
            ],
            vec![Want {
                kind: ChangeKind::Create,
                node: Some("Butch"),
                edge: None,
                deleted_with: None,
                labels: Some(&["Person"]),
                before: None,
                after: Some(vec![
                    ("name", Value::from("Butch")),
                    ("age", Value::from(52_i64)),
                ]),
            }],
        ),
        (
            "an existing node updated and deleted",
            vec![
                "MATCH (g:Person {name: 'Gus'}) SET g.age = 29, g:Pilot",
                "MATCH (g:Person {name: 'Gus'}) DETACH DELETE g",
            ],
            vec![
                Want {
                    kind: ChangeKind::Delete,
                    node: None,
                    edge: Some(("Alix", "Gus")),
                    deleted_with: Some("Gus"),
                    labels: None,
                    before: Some(vec![("since", Value::from(2019_i64))]),
                    after: None,
                },
                Want {
                    kind: ChangeKind::Delete,
                    node: Some("Gus"),
                    edge: None,
                    deleted_with: None,
                    labels: Some(&["Person"]),
                    before: Some(vec![
                        ("name", Value::from("Gus")),
                        ("age", Value::from(28_i64)),
                    ]),
                    after: None,
                },
            ],
        ),
    ]
}

/// The id of each named node `db` holds, by its name.
fn ids(db: &GrafeoDB) -> Ids {
    db.execute("MATCH (n) WHERE n.name IS NOT NULL RETURN n.name, id(n)")
        .unwrap()
        .rows()
        .iter()
        .map(|row| match (&row[0], &row[1]) {
            (Value::String(name), Value::Int64(id)) => {
                (name.to_string(), u64::try_from(*id).unwrap())
            }
            other => panic!("a name and an id, not {other:?}"),
        })
        .collect()
}

fn events_after(db: &GrafeoDB, epoch: EpochId) -> Vec<ChangeEvent> {
    db.changes_between(EpochId::new(epoch.as_u64() + 1), db.current_epoch())
        .unwrap()
}

fn run_transaction(db: &GrafeoDB, statements: &[&str]) {
    let mut session = db.session();
    session.begin_transaction().unwrap();
    for statement in statements {
        session
            .execute(statement)
            .unwrap_or_else(|error| panic!("{statement}: {error}"));
    }
    session.commit().unwrap();
}

/// A database with the setup committed, and the epoch after it.
fn set_up() -> (GrafeoDB, EpochId) {
    let db = GrafeoDB::with_config(Config::in_memory().with_cdc()).unwrap();
    for statement in SETUP {
        db.execute(statement).unwrap();
    }
    let before = db.current_epoch();
    (db, before)
}

/// Each case reports exactly its net events live, in order, each field for field: which entity,
/// the node a `DETACH DELETE` removed an edge with, the labels, and the before and after
/// properties.
#[test]
fn a_transaction_reports_its_net_changes() {
    let mut wrong = Vec::new();
    for (name, statements, want) in cases() {
        let (db, before) = set_up();
        let mut known = ids(&db);
        run_transaction(&db, &statements);
        known.extend(ids(&db));
        let events = events_after(&db, before);
        let exact = events.len() == want.len()
            && events
                .iter()
                .zip(&want)
                .all(|(event, want)| want.matches(event, &known));
        if !exact {
            wrong.push(format!("{name}: {events:?}"));
        }
    }
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}

/// An entity that existed before the transaction, updated and then deleted in it, reports its
/// delete with what it held before the transaction: the net change is the committed state gone,
/// not a state no commit had.
#[test]
fn a_delete_after_updates_shows_the_state_before_the_transaction() {
    let (db, before) = set_up();
    run_transaction(
        &db,
        &[
            "MATCH (:Person {name: 'Alix'})-[k:KNOWS]->() SET k.since = 2020, k.weight = 3",
            "MATCH (g:Person {name: 'Gus'}) SET g.age = 29, g.city = 'Utrecht', g:Pilot",
            "MATCH (g:Person {name: 'Gus'}) REMOVE g.name",
            "MATCH (g:Person {age: 29}) SET g.age = 30",
            "MATCH (g:Person {age: 30}) DETACH DELETE g",
        ],
    );
    let events = events_after(&db, before);
    assert_eq!(events.len(), 2, "{events:?}");
    let edge = &events[0];
    assert_eq!(edge.kind, ChangeKind::Delete);
    assert_eq!(
        edge.before,
        Some([("since".to_string(), Value::from(2019_i64))].into()),
        "the edge as committed"
    );
    let gus = &events[1];
    assert_eq!(gus.kind, ChangeKind::Delete);
    assert_eq!(
        gus.before,
        Some(
            [
                ("name".to_string(), Value::from("Gus")),
                ("age".to_string(), Value::from(28_i64)),
            ]
            .into()
        ),
        "Gus as committed"
    );
    assert_eq!(gus.labels, Some(vec!["Person".to_string()]));
    assert_eq!(gus.before_labels, None);
}

/// Gus's delete, the last event of a transaction of `statements` that ends with his delete.
fn gus_deleted_after(statements: &[&str]) -> ChangeEvent {
    let (db, before) = set_up();
    let mut all = statements.to_vec();
    all.push("MATCH (g {name: 'Gus'}) DETACH DELETE g");
    run_transaction(&db, &all);
    let events = events_after(&db, before);
    let gus = events.last().cloned().unwrap();
    assert!(
        gus.kind == ChangeKind::Delete && matches!(gus.entity_id, EntityId::Node(_)),
        "{events:?}"
    );
    gus
}

/// The committed state a delete after updates shows holds through every order of changes: the
/// labels before the first of two label changes, a key removed and set again, a key set to its
/// own value, and a label removed and added again.
#[test]
fn a_delete_shows_the_committed_state_through_every_order_of_changes() {
    let committed = Some(
        [
            ("name".to_string(), Value::from("Gus")),
            ("age".to_string(), Value::from(28_i64)),
        ]
        .into(),
    );
    let person = Some(vec!["Person".to_string()]);
    let two_labels = gus_deleted_after(&[
        "MATCH (g:Person {name: 'Gus'}) SET g:Pilot",
        "MATCH (g:Person {name: 'Gus'}) SET g:Boxer",
    ]);
    assert_eq!(
        two_labels.labels, person,
        "the labels before the first change"
    );
    assert_eq!(two_labels.before, committed);
    let removed_and_set = gus_deleted_after(&[
        "MATCH (g:Person {name: 'Gus'}) REMOVE g.age",
        "MATCH (g:Person {name: 'Gus'}) SET g.age = 40",
    ]);
    assert_eq!(
        removed_and_set.before, committed,
        "a key removed and set again"
    );
    let same_value = gus_deleted_after(&["MATCH (g:Person {name: 'Gus'}) SET g.age = 28"]);
    assert_eq!(same_value.before, committed, "a key set to its own value");
    let label_again = gus_deleted_after(&[
        "MATCH (g:Person {name: 'Gus'}) REMOVE g:Person",
        "MATCH (g {name: 'Gus'}) SET g:Person",
    ]);
    assert_eq!(
        label_again.labels, person,
        "a label removed and added again"
    );
    assert_eq!(label_again.before, committed);
}

/// An entity that exists before and after the transaction keeps one update per write, in order,
/// each with its own before and after: the transaction's updates are not folded.
#[test]
fn an_entity_that_lives_through_the_transaction_reports_each_update() {
    let (db, before) = set_up();
    run_transaction(
        &db,
        &[
            "MATCH (g:Person {name: 'Gus'}) SET g.age = 29",
            "MATCH (g:Person {name: 'Gus'}) SET g.age = 28",
        ],
    );
    let gus = NodeId::new(ids(&db)["Gus"]);
    let events = events_after(&db, before);
    let age = |age: i64| Some([("age".to_string(), Value::from(age))].into());
    let updates: Vec<_> = events
        .iter()
        .map(|event| {
            (
                event.kind.clone(),
                event.entity_id,
                event.before.clone(),
                event.after.clone(),
            )
        })
        .collect();
    assert_eq!(
        updates,
        [
            (ChangeKind::Update, EntityId::Node(gus), age(28), age(29)),
            (ChangeKind::Update, EntityId::Node(gus), age(29), age(28)),
        ]
    );
}

/// A triple inserted and deleted in one transaction, and an existing one deleted and inserted
/// again, were never a committed change: neither gives an event. A triple deleted, inserted and
/// deleted again gives its delete.
#[cfg(all(feature = "sparql", feature = "triple-store"))]
#[test]
fn a_triple_written_and_unwritten_in_one_transaction_gives_no_events() {
    const PREFIXES: &str = "PREFIX ex: <http://ex.org/> ";
    let triple_events = |db: &GrafeoDB| -> Vec<ChangeEvent> {
        db.changes_between(EpochId::new(0), db.current_epoch())
            .unwrap()
            .into_iter()
            .filter(|event| event.entity_id.is_triple())
            .collect()
    };
    let db = GrafeoDB::with_config(Config::in_memory().with_cdc()).unwrap();
    db.execute_sparql(&format!(
        "{PREFIXES}INSERT DATA {{ ex:alix ex:knows ex:gus . }}"
    ))
    .unwrap();
    let before = triple_events(&db).len();
    let mut session = db.session();
    let insert = "INSERT DATA { ex:hans ex:knows ex:gus . }";
    let delete = "DELETE DATA { ex:hans ex:knows ex:gus . }";
    let reinsert = "INSERT DATA { ex:alix ex:knows ex:gus . }";
    let undo = "DELETE DATA { ex:alix ex:knows ex:gus . }";
    for (updates, net) in [
        (vec![insert, delete], 0),
        (vec![undo, reinsert], 0),
        (vec![undo, reinsert, undo], 1),
    ] {
        session.begin_transaction().unwrap();
        for update in &updates {
            session
                .execute_sparql(&format!("{PREFIXES}{update}"))
                .unwrap();
        }
        session.commit().unwrap();
        let events = triple_events(&db);
        assert_eq!(events.len(), before + net, "{updates:?}: {events:?}");
        if net == 1 {
            assert_eq!(events.last().unwrap().kind, ChangeKind::Delete);
        }
    }
}

/// A delete that stays beside a triple deleted and inserted again names its own triple: the fold
/// removes the pair, not the delete of another triple.
#[cfg(all(feature = "sparql", feature = "triple-store"))]
#[test]
fn the_delete_that_stays_names_its_own_triple() {
    const PREFIXES: &str = "PREFIX ex: <http://ex.org/> ";
    let db = GrafeoDB::with_config(Config::in_memory().with_cdc()).unwrap();
    db.execute_sparql(&format!(
        "{PREFIXES}INSERT DATA {{ ex:alix ex:knows ex:gus . ex:alix ex:knows ex:mia . }}"
    ))
    .unwrap();
    let before = db.current_epoch();
    let mut session = db.session();
    session.begin_transaction().unwrap();
    for update in [
        "DELETE DATA { ex:alix ex:knows ex:gus . }",
        "INSERT DATA { ex:alix ex:knows ex:gus . }",
        "DELETE DATA { ex:alix ex:knows ex:mia . }",
    ] {
        session
            .execute_sparql(&format!("{PREFIXES}{update}"))
            .unwrap();
    }
    session.commit().unwrap();
    let events = events_after(&db, before);
    assert_eq!(events.len(), 1, "{events:?}");
    let delete = &events[0];
    assert_eq!(delete.kind, ChangeKind::Delete);
    assert_eq!(
        (
            delete.triple_subject.as_deref(),
            delete.triple_predicate.as_deref(),
            delete.triple_object.as_deref(),
        ),
        (
            Some("<http://ex.org/alix>"),
            Some("<http://ex.org/knows>"),
            Some("<http://ex.org/mia>"),
        )
    );
}
