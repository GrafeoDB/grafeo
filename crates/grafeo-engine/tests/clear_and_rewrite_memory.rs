//! A store that is cleared and written back with the same content stays the same size: deleted
//! rows' slots (edge ends, hot property values, adjacency offsets) are returned, across reopens at
//! load within one vector of slots per structure, and in a session at collect within twice the
//! first round's live slots plus one vector (a vector's capacity doubles as it grows and is kept
//! as it shrinks, so it swings from round to round without growing). Ids are never
//! reused, a reader that began before the clear keeps reading the old rows, history keeps what a
//! past epoch needs, the file stays within the copy-on-write bound, results equal a fresh
//! database's, and a crash during a checkpoint of a trimmed store reopens to the committed state.
//!
//! The estate is synthetic: 2,400 nodes in two namespaces and 4,800 edges with string
//! properties, cleared with one `DETACH DELETE` per namespace and written back through
//! `create_node` and `create_edge`.
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test main clear_and_rewrite_memory::
//! ```

#![cfg(all(feature = "gql", feature = "grafeo-file"))]

use std::collections::{BTreeMap, HashMap};
use std::path::Path;
use std::time::Duration;

use grafeo_common::types::{NodeId, PropertyKey, Value};
use grafeo_engine::config::HistoryRetention;
use grafeo_engine::{Config, GrafeoDB};

const ROUNDS: usize = 10;
const NODES: usize = 2_400;
const EDGES: usize = 4_800;
const NAMESPACES: [&str; 2] = ["Graph", "Model"];
const KINDS: [&str; 6] = [
    "Application",
    "Component",
    "Service",
    "Interface",
    "Process",
    "Data",
];
const CLEAR: &str =
    "MATCH (n) WHERE any(label IN labels(n) WHERE label STARTS WITH $namespace) DETACH DELETE n";
const CYPHER_CLEAR: &str =
    "MATCH (n) WHERE any(label IN labels(n) WHERE label STARTS WITH $namespace) DETACH DELETE n";

/// One vector of slots, the margin a group's start may leave below its first live row.
const VECTOR: usize = 2_048;
/// The keys of each table: the node properties and the edge properties.
const NODE_KEYS: usize = 4;
const EDGE_KEYS: usize = 2;

/// The four figures that grew, and the margin each may stay above the first round's.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Figures {
    edges: usize,
    node_properties: usize,
    edge_properties: usize,
    adjacency: usize,
}

impl Figures {
    fn of(db: &GrafeoDB) -> Self {
        let usage = db.memory_usage();
        Self {
            edges: usage.store.edges_bytes,
            node_properties: usage.store.node_properties_bytes,
            edge_properties: usage.store.edge_properties_bytes,
            adjacency: usage.indexes.forward_adjacency_bytes
                + usage.indexes.backward_adjacency_bytes,
        }
    }

    /// Where each figure may reach, `in_session` adding the property slots of a group's head.
    fn bound(self, in_session: bool) -> Self {
        let properties = |keys: usize| if in_session { keys * VECTOR * 8 } else { 0 };
        Self {
            edges: self.edges + VECTOR * 40,
            node_properties: self.node_properties + properties(NODE_KEYS),
            edge_properties: self.edge_properties + properties(EDGE_KEYS),
            adjacency: self.adjacency + VECTOR * 16,
        }
    }

    /// Where each figure may reach in a later round of the same session: a vector's capacity
    /// doubles when it grows and is kept when it shrinks, so from round to round it swings up
    /// to twice the first round's live slots without growing; then one vector more, and the
    /// property slots of a group's head.
    fn across_rounds(self) -> Self {
        Self {
            edges: 2 * self.edges + VECTOR * 40,
            node_properties: 2 * self.node_properties + NODE_KEYS * VECTOR * 8,
            edge_properties: 2 * self.edge_properties + EDGE_KEYS * VECTOR * 8,
            adjacency: 2 * self.adjacency + VECTOR * 16,
        }
    }

    fn within(self, bound: Self) -> bool {
        self.edges <= bound.edges
            && self.node_properties <= bound.node_properties
            && self.edge_properties <= bound.edge_properties
            && self.adjacency <= bound.adjacency
    }
}

type NodeRow = (String, Vec<(String, Value)>);
type EdgeRow = (usize, usize, String, Vec<(String, Value)>);

/// The estate, the same every time (a fixed sequence, not a random one).
fn estate() -> (Vec<NodeRow>, Vec<EdgeRow>) {
    let statuses = ["active", "planned", "retired"];
    let nodes = (0..NODES)
        .map(|i| {
            let label = format!("{}{}", NAMESPACES[i % 2], KINDS[i % KINDS.len()]);
            let props = vec![
                (
                    "identifier".to_string(),
                    Value::from(format!("{}-{i:05}", label.to_lowercase()).as_str()),
                ),
                (
                    "name".to_string(),
                    Value::from(format!("{label} number {i}").as_str()),
                ),
                (
                    "description".to_string(),
                    Value::from(
                        format!("a part of the estate, {}", "x".repeat(20 + i % 40)).as_str(),
                    ),
                ),
                ("status".to_string(), Value::from(statuses[i % 3])),
            ];
            (label, props)
        })
        .collect();
    let edges = (0..EDGES)
        .map(|j| {
            let src = (j * 7 + 3) % NODES;
            let dst = (j * 13 + 5) % NODES;
            (
                src,
                dst,
                format!("{}_RELATES", NAMESPACES[src % 2]),
                vec![
                    (
                        "note".to_string(),
                        Value::from(format!("relation {j}").as_str()),
                    ),
                    (
                        "strength".to_string(),
                        Value::from(if j % 2 == 0 { "low" } else { "high" }),
                    ),
                ],
            )
        })
        .collect();
    (nodes, edges)
}

fn params(namespace: &str) -> HashMap<String, Value> {
    HashMap::from([("namespace".to_string(), Value::from(namespace))])
}

/// Writes the estate through the direct API; the ids of the nodes, in order.
fn write(db: &GrafeoDB) -> Vec<NodeId> {
    let (nodes, edges) = estate();
    let ids: Vec<NodeId> = nodes
        .iter()
        .map(|(label, props)| {
            db.create_node_with_props(
                &[label.as_str()],
                props
                    .iter()
                    .map(|(k, v)| (PropertyKey::new(k.as_str()), v.clone())),
            )
            .unwrap()
        })
        .collect();
    for (src, dst, edge_type, props) in &edges {
        db.create_edge_with_props(
            ids[*src],
            ids[*dst],
            edge_type,
            props
                .iter()
                .map(|(k, v)| (PropertyKey::new(k.as_str()), v.clone())),
        )
        .unwrap();
    }
    ids
}

/// How a round clears the estate.
#[derive(Clone, Copy, Debug)]
enum Clear {
    Gql,
    Cypher,
}

fn clear(db: &GrafeoDB, how: Clear) {
    for namespace in NAMESPACES {
        match how {
            Clear::Gql => db.execute_with_params(CLEAR, params(namespace)),
            Clear::Cypher => db.execute_cypher_with_params(CYPHER_CLEAR, params(namespace)),
        }
        .unwrap();
    }
}

/// What a fixed set of queries answers, by `identifier` (ids differ between databases).
fn answers(db: &GrafeoDB) -> BTreeMap<&'static str, Vec<String>> {
    let rows = |query: &str| -> Vec<String> {
        let mut rows: Vec<String> = db
            .execute(query)
            .unwrap_or_else(|error| panic!("{query}: {error}"))
            .rows()
            .iter()
            .map(|row| format!("{row:?}"))
            .collect();
        rows.sort();
        rows
    };
    BTreeMap::from([
        (
            "per label",
            rows("MATCH (n) UNWIND labels(n) AS l RETURN l, count(*) AS c"),
        ),
        (
            "edges",
            rows("MATCH (a)-[r]->(b) RETURN a.identifier, type(r), r.note, b.identifier"),
        ),
        (
            "incoming",
            rows(
                "MATCH (b {identifier: 'graphapplication-00000'})<-[r]-(a) RETURN a.identifier, r.strength",
            ),
        ),
        (
            "values",
            rows("MATCH (n:ModelService) RETURN n.identifier, n.name, n.description, n.status"),
        ),
        (
            "degrees",
            rows("MATCH (n)-[r]->() RETURN n.identifier, count(r)"),
        ),
    ])
}

/// What a fresh database holding the estate answers.
fn fresh_answers() -> BTreeMap<&'static str, Vec<String>> {
    let db = GrafeoDB::new_in_memory();
    write(&db);
    answers(&db)
}

fn file_size(path: &Path) -> u64 {
    std::fs::metadata(path).map_or(0, |metadata| metadata.len())
}

// ---------------------------------------------------------------------------
// The measured shapes (statements 1 to 3 and 9)
// ---------------------------------------------------------------------------

/// Ten rounds with a close and reopen between: the figures at each open stay within one vector
/// of the first round's, the file within twice a fresh file's size, and the answers are a fresh
/// database's.
#[test]
fn across_reopens_the_store_stays_flat() {
    let dir = tempfile::tempdir().unwrap();
    let fresh = dir.path().join("fresh.grafeo");
    {
        let db = GrafeoDB::open(&fresh).unwrap();
        write(&db);
        db.close().unwrap();
    }
    let bound_file = 2 * file_size(&fresh);

    let path = dir.path().join("rounds.grafeo");
    {
        let db = GrafeoDB::open(&path).unwrap();
        write(&db);
        db.close().unwrap();
    }
    let mut first = None;
    let mut wrong = Vec::new();
    for round in 1..=ROUNDS {
        let db = GrafeoDB::open(&path).unwrap();
        let at_open = Figures::of(&db);
        let first = *first.get_or_insert(at_open);
        if !at_open.within(first.bound(false)) {
            wrong.push(format!(
                "round {round} at open: {at_open:?}, first {first:?}"
            ));
        }
        clear(&db, Clear::Gql);
        write(&db);
        db.close().unwrap();
        if file_size(&path) > bound_file {
            wrong.push(format!(
                "round {round}: the file is {} bytes, past {bound_file}",
                file_size(&path)
            ));
        }
    }
    let db = GrafeoDB::open(&path).unwrap();
    assert_eq!(
        answers(&db),
        fresh_answers(),
        "the answers after the rounds"
    );
    db.close().unwrap();
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}

/// The same ten rounds in one open session, cleared in GQL, in Cypher, and in one
/// transaction with the rewrite; and in a named graph. After each round the figures stay within
/// one vector (and the property slots of a group's head) of the first round's.
#[test]
fn within_a_session_the_store_stays_flat() {
    let mut wrong = Vec::new();
    for form in [
        "gql",
        "cypher",
        "one transaction",
        "named graph",
        "compact after",
    ] {
        let mut db = GrafeoDB::new_in_memory();
        if form == "named graph" {
            db.create_graph("estate").unwrap();
            db.set_current_graph(Some("estate")).unwrap();
        }
        write(&db);
        let mut first = None;
        for round in 1..=ROUNDS {
            match form {
                "cypher" => clear(&db, Clear::Cypher),
                "one transaction" => {
                    let mut session = db.session();
                    session.begin_transaction().unwrap();
                    for namespace in NAMESPACES {
                        session
                            .execute_with_params(CLEAR, params(namespace))
                            .unwrap();
                    }
                    session.commit().unwrap();
                }
                _ => clear(&db, Clear::Gql),
            }
            write(&db);
            db.gc();
            if form == "compact after" {
                db.compact().unwrap();
            }
            let after = Figures::of(&db);
            let first = *first.get_or_insert(after);
            if !after.within(first.across_rounds()) {
                wrong.push(format!("{form}, round {round}: {after:?}, first {first:?}"));
            }
        }
        if answers(&db) != fresh_answers() {
            wrong.push(format!(
                "{form}: the answers differ from a fresh database's"
            ));
        }
    }
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}

/// A clear rolled back changes nothing: the figures and the answers are what they were.
#[test]
fn a_clear_rolled_back_changes_nothing() {
    let db = GrafeoDB::new_in_memory();
    write(&db);
    let before = (Figures::of(&db), answers(&db));
    let mut session = db.session();
    session.begin_transaction().unwrap();
    for namespace in NAMESPACES {
        session
            .execute_with_params(CLEAR, params(namespace))
            .unwrap();
    }
    session.rollback().unwrap();
    db.gc();
    assert_eq!((Figures::of(&db), answers(&db)), before);
}

// ---------------------------------------------------------------------------
// What the trim must keep (statements 4 to 7)
// ---------------------------------------------------------------------------

/// Alix knows Gus, Gus knows Vincent, Jules knows Alix.
fn people(db: &GrafeoDB) {
    db.execute(
        "INSERT (a:Person {name: 'Alix', age: 30})-[:KNOWS {since: 2020}]->(g:Person {name: 'Gus', age: 25}), \
         (g)-[:KNOWS {since: 2021}]->(:Person {name: 'Vincent', age: 35}), \
         (:Person {name: 'Jules', age: 40})-[:KNOWS {since: 2019}]->(a)",
    )
    .unwrap();
}

fn people_view(session: &grafeo_engine::session::Session) -> Vec<String> {
    let mut rows: Vec<String> = session
        .execute(
            "MATCH (a:Person)-[k:KNOWS]->(b:Person) \
             RETURN labels(a), a.name, a.age, k.since, b.name, b.age",
        )
        .unwrap()
        .rows()
        .iter()
        .map(|row| format!("{row:?}"))
        .collect();
    rows.sort();
    rows
}

/// A reader that began before the clear keeps reading the old rows, both ways along the edges,
/// through collects and a checkpoint; after it ends and a collect runs, the bound holds again.
#[test]
fn a_reader_from_before_the_clear_reads_the_old_rows() {
    let dir = tempfile::tempdir().unwrap();
    let db = GrafeoDB::open(dir.path().join("reader.grafeo")).unwrap();
    people(&db);
    write(&db);
    let first = Figures::of(&db);
    let mut reader = db.session();
    reader.begin_transaction().unwrap();
    let before = people_view(&reader);
    let incoming_before = reader
        .execute("MATCH (a:Person {name: 'Alix'})<-[:KNOWS]-(j) RETURN j.name")
        .unwrap()
        .rows()
        .to_vec();

    db.execute("MATCH (p:Person) DETACH DELETE p").unwrap();
    clear(&db, Clear::Gql);
    write(&db);
    db.gc();
    assert_eq!(people_view(&reader), before, "after a clear and a collect");
    db.wal_checkpoint().unwrap();
    db.gc();
    assert_eq!(people_view(&reader), before, "after a checkpoint");
    assert_eq!(
        reader
            .execute("MATCH (a:Person {name: 'Alix'})<-[:KNOWS]-(j) RETURN j.name")
            .unwrap()
            .rows()
            .to_vec(),
        incoming_before
    );
    reader.rollback().unwrap();

    db.gc();
    let after = Figures::of(&db);
    assert!(
        after.within(first.bound(true)),
        "after the reader ended: {after:?}, first {first:?}"
    );
    db.close().unwrap();
}

/// With history kept, a read at an epoch before the clear sees the old rows, before and after a
/// checkpoint and reopen.
#[test]
fn history_keeps_what_a_past_epoch_needs() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("history.grafeo");
    let config = || {
        Config::persistent(&path)
            .with_history_retention(HistoryRetention::Duration(Duration::from_secs(3600)))
    };
    let (epoch, before) = {
        let db = GrafeoDB::with_config(config()).unwrap();
        people(&db);
        let epoch = db.current_epoch();
        let before = people_view(&db.session());
        db.execute("MATCH (p:Person) DETACH DELETE p").unwrap();
        people(&db);
        db.execute("MATCH (p:Person) DETACH DELETE p").unwrap();
        db.gc();
        let session = db.session();
        session.set_viewing_epoch(epoch);
        assert_eq!(people_view(&session), before, "in the session");
        db.close().unwrap();
        (epoch, before)
    };
    let db = GrafeoDB::with_config(config()).unwrap();
    let session = db.session();
    session.set_viewing_epoch(epoch);
    assert_eq!(
        people_view(&session),
        before,
        "after a checkpoint and reopen"
    );
    drop(session);
    db.close().unwrap();
}

/// Ids stay dead: an old id finds nothing in any way of looking, and no id is reused.
#[test]
fn the_cleared_ids_stay_dead() {
    let db = GrafeoDB::new_in_memory();
    let old = write(&db);
    clear(&db, Clear::Gql);
    let new = write(&db);
    db.gc();
    let highest_old = old.iter().map(|id| id.as_u64()).max().unwrap();
    assert!(
        new.iter().all(|id| id.as_u64() > highest_old),
        "an id was reused"
    );
    for id in [old[0], old[NODES / 2], old[NODES - 1]] {
        assert!(db.get_node(id).unwrap().is_none(), "{id:?}");
        let found = db
            .execute_with_params(
                "MATCH (n) WHERE id(n) = $id RETURN n",
                HashMap::from([(
                    "id".to_string(),
                    Value::Int64(i64::try_from(id.as_u64()).unwrap()),
                )]),
            )
            .unwrap()
            .rows()
            .len();
        assert_eq!(found, 0, "{id:?}");
    }
}

/// A create whose id was reserved before the clear, committed after the slots moved, is placed
/// and reads back, also after a reopen.
#[test]
fn a_late_create_below_the_start_reads_back() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("late.grafeo");
    {
        let db = GrafeoDB::open(&path).unwrap();
        write(&db);
        let mut late = db.session();
        late.begin_transaction().unwrap();
        late.execute("INSERT (:Person {name: 'Mia', age: 28})")
            .unwrap();
        clear(&db, Clear::Gql);
        write(&db);
        db.gc();
        late.commit().unwrap();
        let mia = |db: &GrafeoDB| {
            db.execute("MATCH (p:Person {name: 'Mia'}) RETURN p.age")
                .unwrap()
                .rows()
                .to_vec()
        };
        assert_eq!(mia(&db), [vec![Value::Int64(28)]]);
        db.close().unwrap();
        let db = GrafeoDB::open(&path).unwrap();
        assert_eq!(mia(&db), [vec![Value::Int64(28)]], "after a reopen");
        db.close().unwrap();
    }
}

/// The kinds of values a column above a moved start holds read back as written.
#[test]
fn every_kind_of_value_reads_back_above_a_moved_start() {
    let db = GrafeoDB::new_in_memory();
    write(&db);
    clear(&db, Clear::Gql);
    let values = [
        ("int", Value::Int64(-7)),
        ("float", Value::Float64(2.5)),
        ("bool", Value::Bool(true)),
        ("text", Value::from("Amsterdam")),
        (
            "list",
            Value::List(vec![Value::Int64(1), Value::from("a")].into()),
        ),
        ("vector", Value::Vector(vec![1.0_f32, 0.0, 0.5].into())),
    ];
    let id = db
        .create_node_with_props(
            &["GraphKinds"],
            values
                .iter()
                .map(|(k, v)| (PropertyKey::new(*k), v.clone())),
        )
        .unwrap();
    // A mixed column: the same key holding another kind on a second node.
    let other = db
        .create_node_with_props(
            &["GraphKinds"],
            [(PropertyKey::new("int"), Value::from("seven"))],
        )
        .unwrap();
    db.gc();
    for (key, value) in &values {
        assert_eq!(
            db.get_node(id)
                .unwrap()
                .unwrap()
                .properties
                .get(&PropertyKey::new(*key)),
            Some(value),
            "{key}"
        );
    }
    assert_eq!(
        db.get_node(other)
            .unwrap()
            .unwrap()
            .properties
            .get(&PropertyKey::new("int")),
        Some(&Value::from("seven"))
    );
}

/// Without backward edges an incoming match answers as with them, after the rounds.
#[test]
fn incoming_matches_without_backward_edges_answer_alike() {
    let with = GrafeoDB::new_in_memory();
    let without = GrafeoDB::with_config(Config::in_memory().without_backward_edges()).unwrap();
    for db in [&with, &without] {
        write(db);
        for _ in 0..3 {
            clear(db, Clear::Gql);
            write(db);
        }
        db.gc();
    }
    assert_eq!(answers(&with), answers(&without));
}

/// The oldest half deleted, the newer half live: the trim at collect keeps the answers right;
/// a long-lived node at row 0 keeps every row correct.
#[test]
fn partial_clears_keep_the_answers_right() {
    let db = GrafeoDB::new_in_memory();
    let ids = write(&db);
    for id in &ids[..NODES / 2] {
        db.detach_delete_node(*id).unwrap();
    }
    db.gc();
    let live = db.execute("MATCH (n) RETURN count(n)").unwrap().rows()[0][0].clone();
    assert_eq!(live, Value::Int64(i64::try_from(NODES / 2).unwrap()));
    for id in &ids[NODES / 2..NODES / 2 + 10] {
        assert!(
            db.get_node(*id).unwrap().is_some(),
            "a live node after the trim"
        );
    }

    let db = GrafeoDB::new_in_memory();
    db.execute("INSERT (:Anchor {name: 'Alix'})").unwrap();
    write(&db);
    for _ in 0..3 {
        clear(&db, Clear::Gql);
        write(&db);
    }
    db.gc();
    let fresh = GrafeoDB::new_in_memory();
    fresh.execute("INSERT (:Anchor {name: 'Alix'})").unwrap();
    write(&fresh);
    assert_eq!(answers(&db), answers(&fresh));
    assert_eq!(
        db.execute("MATCH (a:Anchor) RETURN a.name").unwrap().rows()[0][0],
        Value::from("Alix")
    );
}

// ---------------------------------------------------------------------------
// A crash during a checkpoint of a trimmed store (statement 10)
// ---------------------------------------------------------------------------

#[cfg(feature = "testing-crash-injection")]
const SWEEP_PATH: &str = "GRAFEO_CLEAR_AND_REWRITE_PATH";
#[cfg(feature = "testing-crash-injection")]
const SWEEP_POINT: &str = "GRAFEO_CLEAR_AND_REWRITE_POINT";
#[cfg(feature = "testing-crash-injection")]
const SWEEP_MARKER: &str = "GRAFEO_CLEAR_AND_REWRITE_MARKER";
#[cfg(feature = "testing-crash-injection")]
const SWEEP_TRIM: &str = "GRAFEO_CLEAR_AND_REWRITE_TRIM";

/// Writes, clears and rewrites (with a collect, so the groups start above row 0, or without,
/// so their dead head is not trimmed yet), then checkpoints up to a crash at the point given.
/// The database is built outside the crash closure, and the child ends with no close.
#[cfg(feature = "testing-crash-injection")]
#[test]
fn trimmed_checkpoint_crash_child() {
    use grafeo_common::testing::crash::{CrashResult, with_crash_at};
    use std::panic::AssertUnwindSafe;

    let (Some(path), Some(point), Some(marker), Some(trim)) = (
        std::env::var_os(SWEEP_PATH),
        std::env::var_os(SWEEP_POINT),
        std::env::var_os(SWEEP_MARKER),
        std::env::var_os(SWEEP_TRIM),
    ) else {
        return;
    };
    let point: u64 = point.to_str().unwrap().parse().unwrap();
    let db = GrafeoDB::open(std::path::PathBuf::from(path)).unwrap();
    write(&db);
    clear(&db, Clear::Gql);
    write(&db);
    if trim == "trimmed" {
        db.gc();
    }
    let outcome = with_crash_at(
        point,
        AssertUnwindSafe(|| {
            db.wal_checkpoint().unwrap();
        }),
    );
    if matches!(outcome, CrashResult::Completed(())) {
        std::fs::write(std::path::PathBuf::from(marker), b"done").unwrap();
    }
    std::process::exit(0);
}

/// A crash at each checkpoint point reopens to the committed state, whose answers are a fresh
/// database's, with the groups trimmed and with their dead heads still in place.
#[cfg(feature = "testing-crash-injection")]
#[test]
fn a_crash_while_checkpointing_a_trimmed_store_reopens_to_the_committed_state() {
    let fresh = fresh_answers();
    for trim in ["trimmed", "untrimmed"] {
        let mut finished = false;
        for point in 1..=400_u64 {
            let dir = tempfile::tempdir().unwrap();
            let path = dir.path().join("sweep.grafeo");
            let marker = dir.path().join("completed");
            let mut child = grafeo_common::child_test!("trimmed_checkpoint_crash_child");
            child
                .env(SWEEP_PATH, &path)
                .env(SWEEP_POINT, point.to_string())
                .env(SWEEP_MARKER, &marker)
                .env(SWEEP_TRIM, trim);
            let status = grafeo_common::testing::child_process::run(&mut child).unwrap();
            assert!(status.success(), "{trim}, {point}: the child failed");
            let db = GrafeoDB::open(&path).unwrap();
            assert_eq!(answers(&db), fresh, "{trim}, a crash at {point}");
            db.close().unwrap();
            if marker.exists() {
                finished = true;
                break;
            }
        }
        assert!(
            finished,
            "{trim}: the checkpoint never finished within the points tried"
        );
    }
}
