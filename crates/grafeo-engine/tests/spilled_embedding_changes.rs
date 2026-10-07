//! Changes to spilled embeddings (#522).
//!
//! A vector index spill moves its embeddings into a cache file that the
//! property column reads through (#594). A change made while spilled (an
//! update, a removal, a node delete, a rolled-back transaction) must win over
//! the spilled value on every reader: queries, the engine calls the bindings
//! make, vector search and the HNSW graph. Each case is checked while
//! spilled, after an in-process reload, after a reopen (closed after the
//! reload, closed while spilled, and spilled again after the reopen) and
//! after a crash.
//!
//! ```bash
//! cargo test -p grafeo-engine --features full --test spilled_embedding_changes
//! ```
//!
//! (`--all-features` turns on `temporal`, which has no vector spill.)

#![cfg(all(
    feature = "vector-index",
    feature = "mmap",
    feature = "grafeo-file",
    feature = "gql",
    feature = "wal",
    not(feature = "temporal"),
    not(miri)
))]

use std::collections::BTreeSet;
use std::path::Path;

use grafeo_common::storage::{SectionMemoryConfig, SectionType, TierOverride};
use grafeo_common::testing::child_process;
use grafeo_common::types::{NodeId, PropertyKey, Value};
use grafeo_engine::{Config, GrafeoDB};

const DIMENSIONS: usize = 4;

/// `config` with the vector section forced to disk.
fn force_disk(config: Config) -> Config {
    config.with_section_config(
        SectionType::VectorStore,
        SectionMemoryConfig {
            max_ram: None,
            tier: TierOverride::ForceDisk,
        },
    )
}

/// The embedding item `i` starts with: near the x axis, each its own.
fn original(i: u8) -> Vec<f32> {
    vec![1.0, f32::from(i) / 100.0, f32::from(i % 7) / 50.0, 0.0]
}

/// An embedding more than 2 away from every original, each `k` its own and
/// 0.25 from the next.
fn moved(k: u8) -> Vec<f32> {
    vec![-1.0, f32::from(k) / 4.0, 0.0, 1.0]
}

fn vector(value: &Value) -> Option<Vec<f32>> {
    match value {
        Value::Vector(v) => Some(v.to_vec()),
        _ => None,
    }
}

/// `embedding` as a GQL vector literal.
fn literal(embedding: &[f32]) -> String {
    let parts: Vec<String> = embedding.iter().map(|x| format!("{x:?}")).collect();
    format!("vector([{}])", parts.join(", "))
}

/// `statement` on the node `id`.
fn on(id: NodeId, statement: &str) -> String {
    format!("MATCH (n:Item) WHERE id(n) = {} {statement}", id.as_u64())
}

fn set(embedding: &[f32]) -> String {
    format!("SET n.embedding = {}", literal(embedding))
}

fn euclidean(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y) * (x - y))
        .sum::<f32>()
        .sqrt()
}

/// What a node holds after the change.
#[derive(Clone, Debug, PartialEq)]
enum Expected {
    Embedding(Vec<f32>),
    NoEmbedding,
    Deleted,
}

/// Every item with its original embedding.
fn unchanged(count: u8) -> Vec<Expected> {
    (0..count)
        .map(|i| Expected::Embedding(original(i)))
        .collect()
}

/// What the search checks may rely on.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Search {
    /// Nothing was taken out of the HNSW graph, so every node with an
    /// embedding is reachable: a search for its embedding finds it first and
    /// its true nearest neighbour second, and a search for everything finds
    /// exactly the nodes with an embedding.
    Connected,
    /// Nodes were taken out of the HNSW graph (a removal, a delete, or one
    /// rolled back). `HnswIndex::remove` does not reconnect the neighbours of
    /// the node it takes out, so that can cut other nodes off from the entry
    /// point, in memory as much as while spilled (a separate bug). Here the
    /// index holds exactly the nodes with an embedding, search returns no
    /// other node, and every distance it returns is to the node's current
    /// embedding.
    AfterRemovals,
}

/// A file database at `path` with `count` indexed `:Item` nodes (item `i`
/// holds `original(i)`), spilled.
fn spilled(path: &Path, count: u8) -> (GrafeoDB, Vec<NodeId>) {
    let db = GrafeoDB::with_config(force_disk(Config::persistent(path))).unwrap();
    let ids = (0..count)
        .map(|i| {
            db.create_node_with_props(
                &["Item"],
                [("embedding", Value::Vector(original(i).into()))],
            )
            .unwrap()
        })
        .collect();
    db.create_vector_index(
        "Item",
        "embedding",
        Some(DIMENSIONS),
        Some("euclidean"),
        None,
        None,
        None,
    )
    .unwrap();
    assert!(db.buffer_manager().spill_all() > 0, "nothing was spilled");
    assert_spilled(&db, true, "after the spill");
    (db, ids)
}

fn assert_spilled(db: &GrafeoDB, spilled: bool, phase: &str) {
    let expected = if spilled {
        vec![PropertyKey::new("embedding")]
    } else {
        Vec::new()
    };
    assert_eq!(
        db.store().spilled_node_property_columns(),
        expected,
        "{phase}: the spilled columns"
    );
}

fn close(db: GrafeoDB) {
    db.close().unwrap();
    drop(db);
}

/// Checks every reader of `db` against `expected` (item `i` is `ids[i]`):
/// GQL (`n.embedding`, `properties(n)`), `GrafeoDB::get_node`,
/// `Session::get_node`, the store's batch reads (the Python
/// `get_property_batch` and `get_nodes_by_label`), the index entries, and
/// vector search, which never finds an old embedding (see [`Search`] for the
/// rest).
fn check(db: &GrafeoDB, ids: &[NodeId], expected: &[Expected], search: Search, phase: &str) {
    assert_eq!(ids.len(), expected.len());
    let key = PropertyKey::new("embedding");
    let session = db.session();
    for (&id, expected) in ids.iter().zip(expected) {
        let result = db
            .execute(&on(id, "RETURN n.embedding, properties(n)"))
            .unwrap();
        if *expected == Expected::Deleted {
            assert!(result.rows().is_empty(), "{phase}: {id:?} was deleted");
            assert!(db.get_node(id).is_none(), "{phase}: get_node of {id:?}");
            assert!(
                session.get_node(id).is_none(),
                "{phase}: Session::get_node of {id:?}"
            );
            assert_eq!(
                db.store().get_node_property_batch(&[id], &key),
                vec![None],
                "{phase}: get_node_property_batch of {id:?}"
            );
            continue;
        }
        let want = match expected {
            Expected::Embedding(v) => Some(v.clone()),
            _ => None,
        };
        assert_eq!(result.rows().len(), 1, "{phase}: {id:?}");
        let row = &result.rows()[0];
        assert_eq!(
            vector(&row[0]),
            want,
            "{phase}: RETURN n.embedding of {id:?}"
        );
        let Value::Map(properties) = &row[1] else {
            panic!("{phase}: properties(n) of {id:?} is {:?}", row[1]);
        };
        assert_eq!(
            properties.get(&key).and_then(vector),
            want,
            "{phase}: properties(n) of {id:?}"
        );
        let node = db.get_node(id).expect("the node");
        assert_eq!(
            node.get_property("embedding").and_then(vector),
            want,
            "{phase}: get_node of {id:?}"
        );
        let node = session.get_node(id).expect("the node");
        assert_eq!(
            node.get_property("embedding").and_then(vector),
            want,
            "{phase}: Session::get_node of {id:?}"
        );
        assert_eq!(
            db.store().get_node_property_batch(&[id], &key)[0]
                .as_ref()
                .and_then(vector),
            want,
            "{phase}: get_node_property_batch of {id:?}"
        );
        assert_eq!(
            db.store().get_nodes_properties_batch(&[id])[0]
                .get(&key)
                .and_then(vector),
            want,
            "{phase}: get_nodes_properties_batch of {id:?}"
        );
    }

    let live: Vec<(NodeId, &Vec<f32>)> = ids
        .iter()
        .zip(expected)
        .filter_map(|(&id, expected)| match expected {
            Expected::Embedding(v) => Some((id, v)),
            _ => None,
        })
        .collect();
    let current = |id: NodeId| live.iter().find(|&&(n, _)| n == id).map(|&(_, v)| v);

    // The index holds exactly the nodes with an embedding.
    let index = db
        .store()
        .get_vector_index("Item", "embedding")
        .expect("the vector index");
    for &id in ids {
        assert_eq!(
            index.contains(id),
            current(id).is_some(),
            "{phase}: the index holds {id:?}"
        );
    }
    assert_eq!(index.len(), live.len(), "{phase}: the index size");

    // No search returns a node without an embedding or measures a node by
    // anything but its current embedding.
    let queries: Vec<Vec<f32>> = (0u8..)
        .zip(ids)
        .map(|(i, _)| original(i))
        .chain(live.iter().map(|&(_, v)| v.clone()))
        .collect();
    for query in &queries {
        let results = db
            .vector_search(
                "Item",
                "embedding",
                query,
                ids.len() + 1,
                Some(4 * ids.len()),
                None,
            )
            .unwrap();
        for &(id, distance) in &results {
            let embedding = current(id).unwrap_or_else(|| {
                panic!("{phase}: search for {query:?} returns {id:?}, which has no embedding")
            });
            assert!(
                (distance - euclidean(query, embedding)).abs() < 1e-4,
                "{phase}: search for {query:?} measures {id:?} at {distance}, \
                 not by its embedding {embedding:?}"
            );
        }
    }
    // No search finds an old embedding: no other node holds `original(i)`,
    // so only the old value of `ids[i]` would be found at distance 0.
    for ((i, &id), expected) in (0u8..).zip(ids).zip(expected) {
        if *expected == Expected::Embedding(original(i)) {
            continue;
        }
        let nearest = db
            .vector_search("Item", "embedding", &original(i), 1, None, None)
            .unwrap();
        assert!(
            nearest.first().is_none_or(|&(_, distance)| distance > 1e-3),
            "{phase}: the old embedding of {id:?} is found: {nearest:?}"
        );
    }
    if search == Search::AfterRemovals {
        return;
    }

    for &(id, embedding) in &live {
        let nearest = db
            .vector_search("Item", "embedding", embedding, 2, None, None)
            .unwrap();
        assert_eq!(
            nearest.first().map(|&(n, _)| n),
            Some(id),
            "{phase}: search for the embedding of {id:?}: {nearest:?}"
        );
        assert!(nearest[0].1 < 1e-6, "{phase}: {nearest:?}");
        // The second result is the true nearest other node (by distance:
        // two nodes may be as near).
        let second = live
            .iter()
            .filter(|&&(other, _)| other != id)
            .map(|&(_, v)| euclidean(embedding, v))
            .fold(f32::INFINITY, f32::min);
        if second.is_finite() {
            assert_eq!(nearest.len(), 2, "{phase}: {nearest:?}");
            assert!(
                (nearest[1].1 - second).abs() < 1e-4,
                "{phase}: the neighbour of {id:?} is at {second}: {nearest:?}"
            );
        }
    }
    let everything = db
        .vector_search(
            "Item",
            "embedding",
            &original(0),
            ids.len() + 1,
            Some(4 * ids.len()),
            None,
        )
        .unwrap();
    let found: BTreeSet<u64> = everything.iter().map(|&(n, _)| n.as_u64()).collect();
    let holding: BTreeSet<u64> = live.iter().map(|&(n, _)| n.as_u64()).collect();
    assert_eq!(found, holding, "{phase}: the nodes search finds");
    assert_eq!(everything.len(), live.len(), "{phase}: {everything:?}");
}

/// Makes `change` on a spilled database of `count` items and checks every
/// reader against `expected`: while spilled, after an in-process reload and
/// after a reopen; then on a second database closed while spilled, after a
/// reopen, when spilled again, and after that reload.
fn verify(count: u8, change: impl Fn(&GrafeoDB, &[NodeId]), expected: &[Expected], search: Search) {
    let dir = tempfile::tempdir().unwrap();

    let path = dir.path().join("amsterdam.grafeo");
    let (db, ids) = spilled(&path, count);
    change(&db, &ids);
    assert_spilled(&db, true, "after the change");
    check(&db, &ids, expected, search, "while spilled");
    assert!(db.reload_eligible(1.0) > 0, "nothing was reloaded");
    assert_spilled(&db, false, "after the reload");
    check(&db, &ids, expected, search, "after the reload");
    close(db);
    check(
        &GrafeoDB::open(&path).unwrap(),
        &ids,
        expected,
        search,
        "reopened after the reload",
    );

    let path = dir.path().join("berlin.grafeo");
    let (db, ids) = spilled(&path, count);
    change(&db, &ids);
    assert_spilled(&db, true, "after the change");
    close(db);
    let db = GrafeoDB::open(&path).unwrap();
    check(
        &db,
        &ids,
        expected,
        search,
        "reopened after a close while spilled",
    );
    close(db);
    let db = GrafeoDB::with_config(force_disk(Config::persistent(&path))).unwrap();
    db.buffer_manager().spill_all();
    assert_spilled(&db, true, "reopened with the vector section on disk");
    check(&db, &ids, expected, search, "reopened and spilled");
    assert!(db.reload_eligible(1.0) > 0, "nothing was reloaded");
    check(
        &db,
        &ids,
        expected,
        search,
        "reopened, spilled and reloaded",
    );
}

/// Runs `statements` on `id` in a transaction, checks that the transaction
/// sees `inside` (so the change happened), and rolls it back.
fn rolled_back(db: &GrafeoDB, id: NodeId, statements: &[String], inside: &Expected) {
    let mut session = db.session();
    session.begin_transaction().unwrap();
    for statement in statements {
        session.execute(&on(id, statement)).unwrap();
    }
    let result = session.execute(&on(id, "RETURN n.embedding")).unwrap();
    match inside {
        Expected::Deleted => assert!(result.rows().is_empty(), "{id:?} in the transaction"),
        Expected::NoEmbedding => assert_eq!(vector(&result.rows()[0][0]), None),
        Expected::Embedding(v) => assert_eq!(vector(&result.rows()[0][0]), Some(v.clone())),
    }
    session.rollback().unwrap();
}

// ── Item 1: an update while spilled wins ───────────────────────────

/// An embedding updated while spilled (through the API and through GQL) is
/// the one every reader returns, and search finds it with its new
/// neighbours and never at its old place.
#[test]
fn an_update_while_spilled_wins_on_every_reader() {
    let mut expected = unchanged(8);
    expected[0] = Expected::Embedding(moved(0));
    expected[3] = Expected::Embedding(moved(1));
    verify(
        8,
        |db, ids| {
            db.set_node_property(ids[0], "embedding", Value::Vector(moved(0).into()))
                .unwrap();
            db.execute(&on(ids[3], &set(&moved(1)))).unwrap();
        },
        &expected,
        Search::Connected,
    );
}

// ── Item 2: reads while spilled ────────────────────────────────────

/// Every reader returns the spilled values, not null.
#[test]
fn spilled_embeddings_read_on_every_path() {
    verify(8, |_, _| {}, &unchanged(8), Search::Connected);
}

// ── Item 3: removals and deletes while spilled ─────────────────────

/// `REMOVE n.embedding`, `SET n.embedding = NULL`, `remove_node_property`,
/// `DELETE n` and `delete_node` while spilled stay done on every reader:
/// the index drops those nodes and search never finds them again.
#[test]
fn removals_while_spilled_stay_removed_on_every_reader() {
    let mut expected = unchanged(8);
    expected[1] = Expected::NoEmbedding;
    expected[2] = Expected::NoEmbedding;
    expected[3] = Expected::NoEmbedding;
    expected[4] = Expected::Deleted;
    expected[5] = Expected::Deleted;
    verify(
        8,
        |db, ids| {
            db.execute(&on(ids[1], "REMOVE n.embedding")).unwrap();
            db.execute(&on(ids[2], "SET n.embedding = NULL")).unwrap();
            assert!(db.remove_node_property(ids[3], "embedding").unwrap());
            db.execute(&on(ids[4], "DELETE n")).unwrap();
            assert!(db.delete_node(ids[5]).unwrap());
        },
        &expected,
        Search::AfterRemovals,
    );
}

// ── Item 4: rollbacks while spilled ────────────────────────────────

/// A transaction that updates a spilled embedding and rolls back leaves the
/// spilled value (not null) on every reader, and search finds it in place.
#[test]
fn a_rolled_back_update_while_spilled_restores_the_spilled_value() {
    verify(
        8,
        |db, ids| {
            rolled_back(
                db,
                ids[0],
                &[set(&moved(0))],
                &Expected::Embedding(moved(0)),
            );
            rolled_back(
                db,
                ids[4],
                &[set(&moved(1)), set(&moved(2))],
                &Expected::Embedding(moved(2)),
            );
        },
        &unchanged(8),
        Search::Connected,
    );
}

/// A transaction that removes a spilled embedding (or deletes its node, or
/// updates then removes, or removes then updates) and rolls back leaves the
/// spilled value on every reader and in the index.
#[test]
fn a_rolled_back_removal_while_spilled_restores_the_spilled_value() {
    verify(
        8,
        |db, ids| {
            rolled_back(
                db,
                ids[1],
                &["REMOVE n.embedding".to_string()],
                &Expected::NoEmbedding,
            );
            rolled_back(
                db,
                ids[2],
                &["SET n.embedding = NULL".to_string()],
                &Expected::NoEmbedding,
            );
            rolled_back(db, ids[3], &["DELETE n".to_string()], &Expected::Deleted);
            rolled_back(
                db,
                ids[4],
                &[set(&moved(4)), "REMOVE n.embedding".to_string()],
                &Expected::NoEmbedding,
            );
            rolled_back(
                db,
                ids[5],
                &["REMOVE n.embedding".to_string(), set(&moved(5))],
                &Expected::Embedding(moved(5)),
            );
        },
        &unchanged(8),
        Search::AfterRemovals,
    );
}

// ── Item 5: an update while spilled is reachable in the HNSW graph ──

/// Embeddings updated while spilled are linked into the HNSW graph (the
/// re-insert reads its neighbours' spilled vectors): with 64 nodes, a search
/// for each new embedding finds its node first, and its nearest neighbour
/// second.
#[test]
fn an_embedding_updated_while_spilled_is_reachable_in_the_index() {
    const UPDATED: [usize; 8] = [0, 9, 21, 30, 33, 47, 52, 63];
    let mut expected = unchanged(64);
    for (k, &i) in (0u8..).zip(&UPDATED) {
        expected[i] = Expected::Embedding(moved(k));
    }
    verify(
        64,
        |db, ids| {
            for (k, &i) in (0u8..).zip(&UPDATED) {
                if k % 2 == 0 {
                    db.set_node_property(ids[i], "embedding", Value::Vector(moved(k).into()))
                        .unwrap();
                } else {
                    db.execute(&on(ids[i], &set(&moved(k)))).unwrap();
                }
            }
        },
        &expected,
        Search::Connected,
    );
}

// ── Mixed changes: copies and crashes ──────────────────────────────

/// Updates, removals, a delete and rolled-back changes, made while spilled.
fn mixed_change(db: &GrafeoDB, ids: &[NodeId]) {
    db.set_node_property(ids[0], "embedding", Value::Vector(moved(0).into()))
        .unwrap();
    db.execute(&on(ids[1], &set(&moved(1)))).unwrap();
    db.execute(&on(ids[2], "REMOVE n.embedding")).unwrap();
    assert!(db.remove_node_property(ids[3], "embedding").unwrap());
    db.execute(&on(ids[4], "DELETE n")).unwrap();
    rolled_back(
        db,
        ids[5],
        &[set(&moved(5))],
        &Expected::Embedding(moved(5)),
    );
    rolled_back(
        db,
        ids[6],
        &["REMOVE n.embedding".to_string()],
        &Expected::NoEmbedding,
    );
}

fn mixed_expected() -> Vec<Expected> {
    let mut expected = unchanged(8);
    expected[0] = Expected::Embedding(moved(0));
    expected[1] = Expected::Embedding(moved(1));
    expected[2] = Expected::NoEmbedding;
    expected[3] = Expected::NoEmbedding;
    expected[4] = Expected::Deleted;
    expected
}

// ── Item 6: copies ─────────────────────────────────────────────────

/// `to_memory()`, `save()` and a snapshot of a database changed while
/// spilled hold the changes, not the spilled values.
#[test]
fn copies_hold_the_changes_made_while_spilled() {
    let dir = tempfile::tempdir().unwrap();
    let (db, ids) = spilled(&dir.path().join("paris.grafeo"), 8);
    mixed_change(&db, &ids);
    let expected = mixed_expected();
    let search = Search::AfterRemovals;
    check(&db, &ids, &expected, search, "while spilled");

    check(
        &db.to_memory().unwrap(),
        &ids,
        &expected,
        search,
        "to_memory",
    );
    let copy = dir.path().join("paris-copy.grafeo");
    db.save(&copy).unwrap();
    let bytes = db.export_snapshot().unwrap();
    close(db);
    check(
        &GrafeoDB::open(&copy).unwrap(),
        &ids,
        &expected,
        search,
        "save",
    );
    check(
        &GrafeoDB::import_snapshot(&bytes).unwrap(),
        &ids,
        &expected,
        search,
        "snapshot",
    );
}

// ── Crashes ────────────────────────────────────────────────────────

const CHILD_PATH: &str = "GRAFEO_SPILLED_CHANGES_PATH";
const CHILD_CASE: &str = "GRAFEO_SPILLED_CHANGES_CASE";
const IDS_LINE: &str = "spilled-change-ids:";

/// Child-process entry for [`changes_while_spilled_survive_a_crash`]; a
/// no-op when run directly. Checkpoints the spilled database (a vector index
/// created since the last checkpoint does not survive a crash yet, #401),
/// makes the mixed change, maybe checkpoints and reloads, prints the ids and
/// exits without closing.
#[test]
fn crash_child() {
    let (Some(path), Ok(case)) = (std::env::var_os(CHILD_PATH), std::env::var(CHILD_CASE)) else {
        return;
    };
    let (db, ids) = spilled(Path::new(&path), 8);
    db.wal_checkpoint().unwrap();
    mixed_change(&db, &ids);
    if case != "changed" {
        db.wal_checkpoint().unwrap();
    }
    if case == "checkpointed_and_reloaded" {
        assert!(db.reload_eligible(1.0) > 0);
    }
    let ids: Vec<String> = ids.iter().map(|id| id.as_u64().to_string()).collect();
    println!("{IDS_LINE}{}", ids.join(","));
    // A crash: no close, no checkpoint, no destructors.
    std::process::exit(0);
}

/// Changes made while spilled survive a crash: with only the WAL, after a
/// checkpoint while spilled, and after a checkpoint and a reload.
#[test]
fn changes_while_spilled_survive_a_crash() {
    for case in ["changed", "checkpointed", "checkpointed_and_reloaded"] {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("prague.grafeo");
        let output = child_process::output(
            std::process::Command::new(std::env::current_exe().unwrap())
                .args(["--exact", "crash_child", "--nocapture"])
                .env(CHILD_PATH, &path)
                .env(CHILD_CASE, case),
        )
        .unwrap();
        let stdout = String::from_utf8_lossy(&output.stdout);
        assert!(
            output.status.success(),
            "{case}: the child failed: {stdout}\n{}",
            String::from_utf8_lossy(&output.stderr)
        );
        let ids: Vec<NodeId> = stdout
            .lines()
            .find_map(|line| line.strip_prefix(IDS_LINE))
            .unwrap_or_else(|| panic!("{case}: no ids from the child: {stdout}"))
            .split(',')
            .map(|id| NodeId::new(id.trim().parse().unwrap()))
            .collect();
        check(
            &GrafeoDB::open(&path).unwrap(),
            &ids,
            &mixed_expected(),
            Search::AfterRemovals,
            &format!("reopened after a crash ({case})"),
        );
    }
}
