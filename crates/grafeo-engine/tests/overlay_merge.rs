//! A merge of a compacted database's overlay into its base keeps every
//! transaction's view and every commit.
//!
//! After `compact()` the database reads a columnar base under an overlay that
//! takes every write. Under memory pressure the overlay's memory consumer
//! merges the overlay into a new base and starts an empty overlay. The base
//! has no versions, so the merge waits until no transaction is open: memory
//! pressure while one is open merges nothing, the transaction keeps its
//! snapshot, its uncommitted changes (base deletes included) stay its own,
//! and its commit or rollback still works; the next pressure after it closes
//! merges, and the outcome stays. Sessions opened before a merge, and the
//! database itself, read and write the new overlay afterwards, and named
//! graphs stay, in memory and in the file.
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test overlay_merge
//! ```

#![cfg(all(feature = "compact-store", feature = "lpg", feature = "gql"))]

use std::sync::Arc;

use grafeo_common::types::Value;
use grafeo_engine::{GrafeoDB, Session};

/// The names of the people, sorted.
const PEOPLE: &str = "MATCH (p:Person) RETURN p.name AS name ORDER BY name";

/// Gus's age, if Gus is there.
const GUS_AGE: &str = "MATCH (g:Person {name: 'Gus'}) RETURN g.age AS age";

/// Who knows whom, sorted.
const KNOWS: &str = "MATCH (a:Person)-[:KNOWS]->(b:Person) RETURN a.name AS a, b.name AS b \
                     ORDER BY a";

/// The ids of the components of the current graph, sorted.
const COMPONENTS: &str = "MATCH (c:Component) RETURN c.id AS id ORDER BY id";

/// Alix knows Gus, who knows Mia.
const INSERT_PEOPLE: &str = "INSERT (:Person {name: 'Alix', age: 33})\
                             -[:KNOWS]->(:Person {name: 'Gus', age: 19})\
                             -[:KNOWS]->(:Person {name: 'Mia', age: 88})";

/// The name of the overlay's memory consumer.
const OVERLAY_CONSUMER: &str = "overlay:LpgStore";

/// A database or a session that runs read queries.
trait Reads {
    /// The rows `query` returns.
    fn rows(&self, query: &str) -> Vec<Vec<Value>>;
}

impl Reads for GrafeoDB {
    fn rows(&self, query: &str) -> Vec<Vec<Value>> {
        self.execute(query).unwrap().rows().to_vec()
    }
}

impl Reads for Session {
    fn rows(&self, query: &str) -> Vec<Vec<Value>> {
        self.execute(query).unwrap().rows().to_vec()
    }
}

/// One row per name.
fn names(names: &[&str]) -> Vec<Vec<Value>> {
    names.iter().map(|name| vec![Value::from(*name)]).collect()
}

/// The one-row result of an age.
fn age(years: i64) -> Vec<Vec<Value>> {
    vec![vec![Value::Int64(years)]]
}

/// Who knows whom in [`INSERT_PEOPLE`].
fn everyone_knows() -> Vec<Vec<Value>> {
    vec![
        vec![Value::from("Alix"), Value::from("Gus")],
        vec![Value::from("Gus"), Value::from("Mia")],
    ]
}

/// An in-memory database holding [`INSERT_PEOPLE`], compacted: all of it is
/// in the base.
fn compacted_people() -> GrafeoDB {
    let mut db = GrafeoDB::new_in_memory();
    db.execute(INSERT_PEOPLE).unwrap();
    db.compact().unwrap();
    db
}

/// Asks the overlay's memory consumer to free memory, as memory pressure
/// does, and returns whether it merged the overlay into the base: the
/// overlay holds no change afterwards.
fn merge_under_pressure(db: &GrafeoDB) -> bool {
    let layered = db.layered_store().expect("the database is compacted");
    assert!(
        layered.overlay_mutation_count() > 0,
        "the overlay holds changes to merge"
    );
    db.buffer_manager().spill_consumer_by_name(OVERLAY_CONSUMER);
    layered.overlay_mutation_count() == 0
}

/// A committed change in the overlay, so a merge has something to do: Gus
/// turns 3.
fn gus_turns_three(db: &GrafeoDB) {
    db.execute("MATCH (g:Person {name: 'Gus'}) SET g.age = 3")
        .unwrap();
}

/// Memory pressure while a transaction is open: the overlay's consumer
/// merges nothing, as the base has no versions, and the overlay keeps its
/// changes.
fn pressure_defers_the_merge(db: &GrafeoDB) {
    assert!(
        !merge_under_pressure(db),
        "memory pressure merges nothing while a transaction is open"
    );
}

/// Memory pressure once no transaction is open: the merge the open
/// transaction held back runs. A component, which no query here reads,
/// gives the overlay a change to merge when a rollback left it none.
fn the_deferred_merge_runs(db: &GrafeoDB) {
    let layered = db.layered_store().expect("the database is compacted");
    if layered.overlay_mutation_count() == 0 {
        db.execute("INSERT (:Component {id: 'c3'})").unwrap();
    }
    assert!(
        merge_under_pressure(db),
        "the merge runs once the transaction closed"
    );
}

/// An update of a base node, open while memory pressure asks for a merge,
/// stays with its transaction, and its rollback brings the old value back;
/// the merge after it keeps the old value. (Other sessions see an
/// uncommitted value today, merge or not: values are written in place.)
#[test]
fn an_open_update_stays_with_its_transaction_across_a_merge_and_rolls_back() {
    let db = compacted_people();
    let mut session = db.session();
    session.begin_transaction().unwrap();
    session
        .execute("MATCH (g:Person {name: 'Gus'}) SET g.age = 3")
        .unwrap();

    pressure_defers_the_merge(&db);
    assert_eq!(session.rows(GUS_AGE), age(3), "the transaction's own view");

    session.rollback().unwrap();
    assert_eq!(db.rows(GUS_AGE), age(19), "after the rollback");
    assert_eq!(session.rows(GUS_AGE), age(19), "the session after it");

    the_deferred_merge_runs(&db);
    assert_eq!(db.rows(GUS_AGE), age(19), "after the merge");
    assert_eq!(session.rows(GUS_AGE), age(19), "the session after it");
}

/// An update of a base node, open while memory pressure asks for a merge,
/// commits, and the merge after the commit keeps it.
#[test]
fn an_open_update_commits_after_a_merge() {
    let db = compacted_people();
    let mut session = db.session();
    session.begin_transaction().unwrap();
    session
        .execute("MATCH (g:Person {name: 'Gus'}) SET g.age = 3")
        .unwrap();

    pressure_defers_the_merge(&db);
    session.commit().unwrap();
    assert_eq!(db.rows(GUS_AGE), age(3), "a session after the commit");

    the_deferred_merge_runs(&db);
    assert_eq!(db.rows(GUS_AGE), age(3), "after the merge");
    let mut newer = db.session();
    newer.begin_transaction().unwrap();
    assert_eq!(
        newer.rows(GUS_AGE),
        age(3),
        "a transaction after the commit"
    );
    newer.rollback().unwrap();
}

/// A node created by a transaction open while memory pressure asks for a
/// merge is the transaction's own until it commits, and everyone's after,
/// also once the merge after the commit moved it into the base.
#[test]
fn an_open_insert_stays_private_across_a_merge_and_commits_after_it() {
    let db = compacted_people();
    let mut session = db.session();
    session.begin_transaction().unwrap();
    session
        .execute("INSERT (:Person {name: 'Vincent'})")
        .unwrap();

    pressure_defers_the_merge(&db);
    assert_eq!(
        session.rows(PEOPLE),
        names(&["Alix", "Gus", "Mia", "Vincent"]),
        "the transaction sees its node"
    );
    assert_eq!(db.rows(PEOPLE), names(&["Alix", "Gus", "Mia"]));

    session.commit().unwrap();
    assert_eq!(
        db.rows(PEOPLE),
        names(&["Alix", "Gus", "Mia", "Vincent"]),
        "after the commit"
    );

    the_deferred_merge_runs(&db);
    assert_eq!(
        db.rows(PEOPLE),
        names(&["Alix", "Gus", "Mia", "Vincent"]),
        "after the merge"
    );
}

/// A node created by a transaction open while memory pressure asks for a
/// merge goes with its rollback, and stays gone after the merge.
#[test]
fn an_open_insert_rolls_back_after_a_merge() {
    let db = compacted_people();
    let mut session = db.session();
    session.begin_transaction().unwrap();
    session
        .execute("INSERT (:Person {name: 'Vincent'})")
        .unwrap();

    pressure_defers_the_merge(&db);
    session.rollback().unwrap();
    assert_eq!(db.rows(PEOPLE), names(&["Alix", "Gus", "Mia"]));
    assert_eq!(session.rows(PEOPLE), names(&["Alix", "Gus", "Mia"]));

    the_deferred_merge_runs(&db);
    assert_eq!(
        db.rows(PEOPLE),
        names(&["Alix", "Gus", "Mia"]),
        "after the merge"
    );
}

/// The delete of a node created after `compact()` (an overlay node), open
/// while memory pressure asks for a merge, stays with its transaction, and
/// its rollback keeps the node, also after the merge. (Other sessions miss a
/// node whose delete is open today, merge or not.)
#[test]
fn an_open_delete_of_an_overlay_node_stays_with_its_transaction_across_a_merge_and_rolls_back() {
    let db = compacted_people();
    db.execute("INSERT (:Person {name: 'Jules'})").unwrap();
    let mut session = db.session();
    session.begin_transaction().unwrap();
    session
        .execute("MATCH (j:Person {name: 'Jules'}) DELETE j")
        .unwrap();

    pressure_defers_the_merge(&db);
    assert_eq!(
        session.rows(PEOPLE),
        names(&["Alix", "Gus", "Mia"]),
        "the transaction misses Jules"
    );

    session.rollback().unwrap();
    assert_eq!(
        db.rows(PEOPLE),
        names(&["Alix", "Gus", "Jules", "Mia"]),
        "after the rollback"
    );

    the_deferred_merge_runs(&db);
    assert_eq!(
        db.rows(PEOPLE),
        names(&["Alix", "Gus", "Jules", "Mia"]),
        "after the merge"
    );
}

/// The delete of an overlay node, open while memory pressure asks for a
/// merge, commits, and the merge after the commit keeps the node gone.
#[test]
fn an_open_delete_of_an_overlay_node_commits_after_a_merge() {
    let db = compacted_people();
    db.execute("INSERT (:Person {name: 'Jules'})").unwrap();
    let mut session = db.session();
    session.begin_transaction().unwrap();
    session
        .execute("MATCH (j:Person {name: 'Jules'}) DELETE j")
        .unwrap();

    pressure_defers_the_merge(&db);
    session.commit().unwrap();
    assert_eq!(db.rows(PEOPLE), names(&["Alix", "Gus", "Mia"]));

    the_deferred_merge_runs(&db);
    assert_eq!(
        db.rows(PEOPLE),
        names(&["Alix", "Gus", "Mia"]),
        "after the merge"
    );
}

/// A base delete (a pending tombstone in the overlay), open while memory
/// pressure asks for a merge, stays the transaction's own, commits, and the
/// merge after the commit leaves the node and its edges out of the base.
#[test]
fn an_open_base_delete_stays_private_across_a_merge_and_commits_after_it() {
    let db = compacted_people();
    let mut session = db.session();
    session.begin_transaction().unwrap();
    session
        .execute("MATCH (g:Person {name: 'Gus'}) DETACH DELETE g")
        .unwrap();

    pressure_defers_the_merge(&db);
    assert_eq!(
        session.rows(PEOPLE),
        names(&["Alix", "Mia"]),
        "the transaction misses Gus"
    );
    assert_eq!(
        session.rows(KNOWS),
        Vec::<Vec<Value>>::new(),
        "and his edges"
    );
    assert_eq!(db.rows(PEOPLE), names(&["Alix", "Gus", "Mia"]));
    assert_eq!(db.rows(KNOWS), everyone_knows());

    session.commit().unwrap();
    assert_eq!(db.rows(PEOPLE), names(&["Alix", "Mia"]), "after the commit");
    assert_eq!(db.rows(KNOWS), Vec::<Vec<Value>>::new(), "after the commit");

    the_deferred_merge_runs(&db);
    assert_eq!(db.rows(PEOPLE), names(&["Alix", "Mia"]), "after the merge");
    assert_eq!(db.rows(KNOWS), Vec::<Vec<Value>>::new(), "after the merge");
}

/// A base delete, open while memory pressure asks for a merge, rolls back,
/// and the merge after the rollback keeps the node and its edges.
#[test]
fn an_open_base_delete_rolls_back_after_a_merge() {
    let db = compacted_people();
    let mut session = db.session();
    session.begin_transaction().unwrap();
    session
        .execute("MATCH (g:Person {name: 'Gus'}) DETACH DELETE g")
        .unwrap();

    pressure_defers_the_merge(&db);
    session.rollback().unwrap();
    assert_eq!(db.rows(PEOPLE), names(&["Alix", "Gus", "Mia"]));
    assert_eq!(db.rows(KNOWS), everyone_knows());
    assert_eq!(session.rows(PEOPLE), names(&["Alix", "Gus", "Mia"]));

    the_deferred_merge_runs(&db);
    assert_eq!(
        db.rows(PEOPLE),
        names(&["Alix", "Gus", "Mia"]),
        "after the merge"
    );
    assert_eq!(db.rows(KNOWS), everyone_knows(), "after the merge");
}

/// A transaction open while memory pressure asks for a merge keeps its
/// snapshot: it misses a node another session created after it began, as
/// the merge, which would move the node into the base (which has no
/// versions), waits until it closes.
#[test]
fn a_transaction_open_across_a_merge_keeps_its_snapshot() {
    let db = compacted_people();
    let mut reader = db.session();
    reader.begin_transaction().unwrap();
    db.execute("INSERT (:Person {name: 'Vincent'})").unwrap();
    assert_eq!(
        reader.rows(PEOPLE),
        names(&["Alix", "Gus", "Mia"]),
        "the reader's snapshot before the merge"
    );

    pressure_defers_the_merge(&db);
    assert_eq!(
        reader.rows(PEOPLE),
        names(&["Alix", "Gus", "Mia"]),
        "the reader's snapshot after the pressure"
    );
    reader.rollback().unwrap();
    assert_eq!(db.rows(PEOPLE), names(&["Alix", "Gus", "Mia", "Vincent"]));

    the_deferred_merge_runs(&db);
    assert_eq!(
        db.rows(PEOPLE),
        names(&["Alix", "Gus", "Mia", "Vincent"]),
        "after the merge"
    );
}

/// With no transaction open, the merge runs and frees the overlay's memory:
/// nothing holds the old overlay afterwards, not even a session opened before
/// the merge. The data is all there afterwards.
#[test]
fn a_merge_with_no_open_transaction_frees_the_overlay() {
    let db = compacted_people();
    for number in 0..88 {
        db.execute(&format!("INSERT (:Person {{name: 'Vincent {number}'}})"))
            .unwrap();
    }
    gus_turns_three(&db);
    let layered = db.layered_store().unwrap();
    let before = layered.overlay_memory_bytes();
    let old_overlay = Arc::downgrade(&db.layered_store().unwrap().overlay_store());
    let session = db.session();
    assert_eq!(session.rows(GUS_AGE), age(3));

    assert!(merge_under_pressure(&db), "the overlay is merged");
    assert!(
        layered.overlay_memory_bytes() < before,
        "the merge frees overlay memory: {before} bytes before, {} after",
        layered.overlay_memory_bytes()
    );
    assert!(
        old_overlay.upgrade().is_none(),
        "nothing keeps the old overlay alive"
    );
    assert_eq!(session.rows(GUS_AGE), age(3), "the session reads on");
    assert_eq!(db.rows(PEOPLE).len(), 91, "the three people and 88 more");
    assert_eq!(db.rows(GUS_AGE), age(3));
    assert_eq!(db.rows(KNOWS), everyone_knows());
}

/// A session opened before a merge writes the new overlay afterwards: its
/// commits are seen by new sessions and its rollbacks undo its changes.
#[test]
fn a_session_opened_before_a_merge_writes_the_live_overlay() {
    let db = compacted_people();
    let mut session = db.session();
    gus_turns_three(&db);
    assert!(merge_under_pressure(&db), "no transaction is open");

    session
        .execute("INSERT (:Person {name: 'Vincent'})")
        .unwrap();
    assert_eq!(
        db.rows(PEOPLE),
        names(&["Alix", "Gus", "Mia", "Vincent"]),
        "a statement of the old session, committed"
    );

    session.begin_transaction().unwrap();
    session
        .execute("MATCH (m:Person {name: 'Mia'}) DETACH DELETE m")
        .unwrap();
    session.commit().unwrap();
    assert_eq!(
        db.rows(PEOPLE),
        names(&["Alix", "Gus", "Vincent"]),
        "a transaction of the old session, committed"
    );

    session.begin_transaction().unwrap();
    session
        .execute("MATCH (g:Person {name: 'Gus'}) SET g.age = 88")
        .unwrap();
    session.rollback().unwrap();
    assert_eq!(db.rows(GUS_AGE), age(3), "a rolled back transaction");
    assert_eq!(session.rows(GUS_AGE), age(3));
}

/// Named graphs stay after a merge, for new sessions, old sessions and the
/// database's own calls.
#[test]
fn named_graphs_stay_after_a_merge() {
    let mut db = GrafeoDB::new_in_memory();
    db.create_graph("model").unwrap();
    db.graph("model")
        .unwrap()
        .execute("INSERT (:Component {id: 'c0'})")
        .unwrap();
    db.execute(INSERT_PEOPLE).unwrap();
    db.compact().unwrap();
    let old = db.session();
    old.use_graph("model");
    gus_turns_three(&db);
    assert!(merge_under_pressure(&db), "no transaction is open");

    assert_eq!(db.list_graphs(), vec!["model".to_string()]);
    let model = db.graph("model").unwrap();
    assert_eq!(model.execute(COMPONENTS).unwrap().rows(), names(&["c0"]));
    old.execute("INSERT (:Component {id: 'c1'})").unwrap();
    model
        .create_node_with_props(&["Component"], [("id", Value::from("c3"))])
        .unwrap();
    assert_eq!(
        model.execute(COMPONENTS).unwrap().rows(),
        names(&["c0", "c1", "c3"]),
        "a new session"
    );
    assert_eq!(
        old.rows(COMPONENTS),
        names(&["c0", "c1", "c3"]),
        "a session opened before the merge"
    );
    assert_eq!(
        db.rows(COMPONENTS),
        Vec::<Vec<Value>>::new(),
        "the default graph has none"
    );
}

/// The indexes stay after a merge: the property index on `name`, and the
/// text and vector indexes built over the base, which find the nodes they
/// held but not the base node deleted before the merge (no tombstone hides
/// it after the merge), and the text index the node created after it.
#[cfg(all(feature = "text-index", feature = "vector-index"))]
#[test]
fn indexes_stay_after_a_merge() {
    let mut db = GrafeoDB::new_in_memory();
    for name in ["Alix", "Gus", "Mia"] {
        db.execute(&format!(
            "INSERT (:Person {{name: '{name}', bio: 'graph notes', embedding: vector([1.0, 0.0])}})"
        ))
        .unwrap();
    }
    db.compact().unwrap();
    db.create_property_index("name").unwrap();
    db.create_text_index("Person", "bio").unwrap();
    db.create_vector_index("Person", "embedding", Some(2), None, None, None, None)
        .unwrap();
    db.execute("MATCH (g:Person {name: 'Gus'}) DETACH DELETE g")
        .unwrap();
    assert!(merge_under_pressure(&db), "no transaction is open");
    db.execute("INSERT (:Person {name: 'Vincent', bio: 'graph notes'})")
        .unwrap();

    let names_found = |hits: Vec<(grafeo_common::types::NodeId, f64)>| {
        let mut found: Vec<Option<Value>> = hits
            .into_iter()
            .map(|(id, _)| {
                db.get_node(id)
                    .and_then(|node| node.get_property("name").cloned())
            })
            .collect();
        found.sort_by_key(|name| format!("{name:?}"));
        found
    };
    assert_eq!(
        names_found(db.text_search("Person", "bio", "graph", 19).unwrap()),
        vec![
            Some(Value::from("Alix")),
            Some(Value::from("Mia")),
            Some(Value::from("Vincent"))
        ],
        "the text index"
    );
    let vector = db
        .vector_search("Person", "embedding", &[1.0, 0.0], 19, None, None)
        .unwrap()
        .into_iter()
        .map(|(id, distance)| (id, f64::from(distance)))
        .collect();
    assert_eq!(
        names_found(vector),
        vec![Some(Value::from("Alix")), Some(Value::from("Mia"))],
        "the vector index"
    );
    assert!(db.has_property_index("name"), "the property index");
    assert_eq!(
        db.rows("MATCH (p:Person {name: 'Vincent'}) RETURN p.name"),
        names(&["Vincent"])
    );
}

/// `recompact()` merges as memory pressure does: it refuses while a
/// transaction is open, which then commits as if nothing happened.
#[test]
fn recompact_refuses_while_a_transaction_is_open() {
    let mut db = compacted_people();
    let mut session = db.session();
    session.begin_transaction().unwrap();
    session
        .execute("INSERT (:Person {name: 'Vincent'})")
        .unwrap();

    assert!(
        db.recompact().is_err(),
        "recompact() with a transaction open"
    );
    assert_eq!(
        session.rows(PEOPLE),
        names(&["Alix", "Gus", "Mia", "Vincent"]),
        "the transaction's own view"
    );
    session.commit().unwrap();
    assert_eq!(
        db.rows(PEOPLE),
        names(&["Alix", "Gus", "Mia", "Vincent"]),
        "after the commit"
    );
    db.recompact().unwrap();
    assert_eq!(db.rows(PEOPLE), names(&["Alix", "Gus", "Mia", "Vincent"]));
}

/// A session opened before `recompact()` writes the new overlay afterwards,
/// and named graphs stay.
#[test]
fn a_session_opened_before_recompact_writes_the_live_overlay() {
    let mut db = compacted_people();
    db.create_graph("model").unwrap();
    let session = db.session();
    gus_turns_three(&db);
    db.recompact().unwrap();

    session
        .execute("INSERT (:Person {name: 'Vincent'})")
        .unwrap();
    assert_eq!(db.rows(PEOPLE), names(&["Alix", "Gus", "Mia", "Vincent"]));
    assert_eq!(db.rows(GUS_AGE), age(3));
    assert_eq!(db.list_graphs(), vec!["model".to_string()]);
}

/// `compact()` frees the store it compacted: nothing keeps it alive next to
/// the compacted base.
#[test]
fn compact_frees_the_store_it_compacted() {
    let mut db = GrafeoDB::new_in_memory();
    db.execute(INSERT_PEOPLE).unwrap();
    let compacted = Arc::downgrade(&db.store());
    db.compact().unwrap();
    assert!(
        compacted.upgrade().is_none(),
        "nothing keeps the compacted store alive"
    );
    assert_eq!(db.rows(PEOPLE), names(&["Alix", "Gus", "Mia"]));
}

/// With the base spilled to a file under memory pressure, a merge of the
/// overlay replaces the base: a later spill or reload of the base keeps the
/// merged commits, it never brings back the base from before the merge.
#[cfg(feature = "mmap")]
mod base_tiers {
    use super::*;

    /// A compacted database that spills its base into `dir`.
    fn compacted_with_spill(dir: &std::path::Path) -> GrafeoDB {
        let mut db = GrafeoDB::with_config(
            grafeo_engine::Config::in_memory().with_spill_path(dir.to_path_buf()),
        )
        .unwrap();
        db.execute(INSERT_PEOPLE).unwrap();
        db.compact().unwrap();
        db
    }

    /// The name of the base's memory consumer.
    const BASE_CONSUMER: &str = "section:CompactStore";

    /// The base the database reads is the one the tier wrapper holds.
    fn assert_the_wrapper_holds_the_base(db: &GrafeoDB) {
        assert!(
            Arc::ptr_eq(
                &db.layered_store().unwrap().base_store_arc(),
                &db.compact_tiered().unwrap().store()
            ),
            "the base the database reads is the one the tier wrapper holds"
        );
    }

    #[test]
    fn a_base_spill_after_a_merge_keeps_the_merged_commits() {
        let dir = tempfile::tempdir().unwrap();
        let db = compacted_with_spill(dir.path());
        db.execute("INSERT (:Person {name: 'Vincent'})").unwrap();
        assert!(merge_under_pressure(&db), "no transaction is open");

        db.buffer_manager().spill_consumer_by_name(BASE_CONSUMER);
        assert!(
            db.compact_tiered().unwrap().is_on_disk(),
            "the merged base is spilled"
        );
        assert_the_wrapper_holds_the_base(&db);
        assert_eq!(db.rows(PEOPLE), names(&["Alix", "Gus", "Mia", "Vincent"]));
    }

    #[test]
    fn a_base_reload_after_a_merge_keeps_the_merged_commits() {
        let dir = tempfile::tempdir().unwrap();
        let db = compacted_with_spill(dir.path());
        db.buffer_manager().spill_consumer_by_name(BASE_CONSUMER);
        assert!(db.compact_tiered().unwrap().is_on_disk());
        db.execute("INSERT (:Person {name: 'Vincent'})").unwrap();
        assert!(merge_under_pressure(&db), "no transaction is open");

        db.buffer_manager().reload_eligible(1.0);
        assert_eq!(db.rows(PEOPLE), names(&["Alix", "Gus", "Mia", "Vincent"]));
        db.buffer_manager().spill_consumer_by_name(BASE_CONSUMER);
        assert_the_wrapper_holds_the_base(&db);
        assert_eq!(db.rows(PEOPLE), names(&["Alix", "Gus", "Mia", "Vincent"]));
    }
}

#[cfg(all(feature = "wal", feature = "grafeo-file"))]
mod file {
    use std::path::{Path, PathBuf};
    use std::process::Command;

    use grafeo_common::testing::child_process;

    use super::*;

    /// A database at `path` with a named graph `model` holding a component
    /// and [`INSERT_PEOPLE`], compacted and closed.
    fn compacted_file(path: &Path) {
        let mut db = GrafeoDB::open(path).unwrap();
        db.create_graph("model").unwrap();
        db.graph("model")
            .unwrap()
            .execute("INSERT (:Component {id: 'c0'})")
            .unwrap();
        db.execute(INSERT_PEOPLE).unwrap();
        db.compact().unwrap();
        db.close().unwrap();
    }

    /// Writes after a merge: sessions opened before it insert Vincent in the
    /// default graph and a component in `model`, a new session inserts
    /// Jules.
    fn write_across_a_merge(db: &GrafeoDB) {
        let old = db.session();
        let old_in_model = db.session();
        old_in_model.use_graph("model");
        gus_turns_three(db);
        assert!(merge_under_pressure(db), "no transaction is open");
        old.execute("INSERT (:Person {name: 'Vincent'})").unwrap();
        old_in_model
            .execute("INSERT (:Component {id: 'c1'})")
            .unwrap();
        db.execute("INSERT (:Person {name: 'Jules'})").unwrap();
    }

    /// The people [`write_across_a_merge`] left.
    fn assert_people_written_across_a_merge(db: &GrafeoDB, stage: &str) {
        assert_eq!(
            db.rows(PEOPLE),
            names(&["Alix", "Gus", "Jules", "Mia", "Vincent"]),
            "{stage}"
        );
        assert_eq!(db.rows(GUS_AGE), age(3), "{stage}");
        assert_eq!(db.rows(KNOWS), everyone_knows(), "{stage}");
    }

    /// The named graph [`write_across_a_merge`] left.
    fn assert_model_written_across_a_merge(db: &GrafeoDB, stage: &str) {
        assert_eq!(db.list_graphs(), vec!["model".to_string()], "{stage}");
        let model = db
            .graph("model")
            .unwrap_or_else(|error| panic!("{stage}: {error}"));
        assert_eq!(
            model.execute(COMPONENTS).unwrap().rows(),
            names(&["c0", "c1"]),
            "{stage}"
        );
    }

    /// Writes across a merge in a database at `path`, then a close and a
    /// reopen.
    fn reopened_after_writes_across_a_merge(path: &Path) -> GrafeoDB {
        compacted_file(path);
        {
            let db = GrafeoDB::open(path).unwrap();
            write_across_a_merge(&db);
            db.close().unwrap();
        }
        GrafeoDB::open(path).unwrap()
    }

    /// Commits made after a merge, also by a session opened before it,
    /// survive a close and a reopen.
    #[test]
    fn commits_after_a_merge_survive_a_reopen() {
        let dir = tempfile::tempdir().unwrap();
        let db = reopened_after_writes_across_a_merge(&dir.path().join("people.grafeo"));
        assert_people_written_across_a_merge(&db, "after the reopen");
        db.close().unwrap();
    }

    /// Named graphs survive a merge, a close and a reopen, with what was
    /// written to them after the merge.
    #[test]
    fn named_graphs_survive_a_merge_and_a_reopen() {
        let dir = tempfile::tempdir().unwrap();
        let db = reopened_after_writes_across_a_merge(&dir.path().join("people.grafeo"));
        assert_model_written_across_a_merge(&db, "after the reopen");
        db.close().unwrap();
    }

    /// After a reopen the loaded store is the overlay: a merge frees it, and
    /// the data and the indexes stay, also after the next reopen.
    #[test]
    fn a_merge_after_a_reopen_frees_the_loaded_overlay_and_keeps_the_indexes() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("people.grafeo");
        compacted_file(&path);
        {
            let db = GrafeoDB::open(&path).unwrap();
            db.create_property_index("name").unwrap();
            #[cfg(feature = "text-index")]
            db.create_text_index("Person", "name").unwrap();
            let loaded = Arc::downgrade(&db.layered_store().unwrap().overlay_store());
            gus_turns_three(&db);
            assert!(merge_under_pressure(&db), "no transaction is open");
            assert!(
                loaded.upgrade().is_none(),
                "nothing keeps the loaded overlay alive"
            );
            db.close().unwrap();
        }
        let db = GrafeoDB::open(&path).unwrap();
        assert_eq!(db.rows(PEOPLE), names(&["Alix", "Gus", "Mia"]));
        assert_eq!(db.rows(GUS_AGE), age(3));
        assert!(db.has_property_index("name"), "the property index");
        #[cfg(feature = "text-index")]
        assert_eq!(
            db.text_search("Person", "name", "Mia", 3).unwrap().len(),
            1,
            "the text index"
        );
        db.close().unwrap();
    }

    /// The epoch goes on across a merge: a close right after it records the
    /// database's epoch, and after the reopen commits continue above it.
    /// (Without `temporal` a reopen starts the epochs over, merge or not.)
    #[cfg(feature = "temporal")]
    #[test]
    fn the_epoch_survives_a_merge_and_a_reopen() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("people.grafeo");
        compacted_file(&path);
        let before = {
            let db = GrafeoDB::open(&path).unwrap();
            gus_turns_three(&db);
            let before = db.current_epoch();
            assert!(merge_under_pressure(&db), "no transaction is open");
            db.close().unwrap();
            before
        };
        let db = GrafeoDB::open(&path).unwrap();
        assert!(
            db.current_epoch() >= before,
            "the epoch after the reopen, {:?}, falls behind the one before, {before:?}",
            db.current_epoch()
        );
        db.close().unwrap();
    }

    /// The database path the child process works on.
    const PATH_VAR: &str = "GRAFEO_OVERLAY_MERGE_PATH";
    /// Exit code of a child that reached its end.
    const EXITED: i32 = 19;

    /// The WAL next to the database file.
    fn sidecar_wal(path: &Path) -> PathBuf {
        let mut sidecar = path.as_os_str().to_owned();
        sidecar.push(".wal");
        PathBuf::from(sidecar)
    }

    /// Child-process entry for
    /// [`a_checkpoint_after_a_merge_holds_its_commits_and_named_graphs`]; a
    /// no-op when run directly. It writes across a merge, checkpoints, and
    /// exits without `close()`, like a crash.
    #[test]
    fn overlay_merge_child() {
        let Some(path) = std::env::var_os(PATH_VAR) else {
            return;
        };
        let db = GrafeoDB::open(PathBuf::from(path)).unwrap();
        write_across_a_merge(&db);
        db.wal_checkpoint().unwrap();
        std::process::exit(EXITED);
    }

    /// A checkpoint after a merge, then a crash: the file holds the commits
    /// made after the merge and the named graphs.
    #[test]
    fn a_checkpoint_after_a_merge_holds_its_commits_and_named_graphs() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("people.grafeo");
        compacted_file(&path);
        let output = child_process::output(
            Command::new(std::env::current_exe().unwrap())
                .args(["--exact", "file::overlay_merge_child", "--nocapture"])
                .env(PATH_VAR, &path),
        )
        .unwrap();
        assert_eq!(
            output.status.code(),
            Some(EXITED),
            "the child exited early:\n{}",
            String::from_utf8_lossy(&output.stderr)
        );
        assert!(
            sidecar_wal(&path).exists(),
            "the child left its WAL, as a crash does"
        );
        let db = GrafeoDB::open(&path).unwrap();
        assert_people_written_across_a_merge(&db, "after the crash");
        assert_model_written_across_a_merge(&db, "after the crash");
        db.close().unwrap();
    }
}
