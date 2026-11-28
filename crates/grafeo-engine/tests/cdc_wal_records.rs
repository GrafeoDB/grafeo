//! What a commit writes to the WAL with change data capture off and on (#679 patch 6).
//!
//! - **Off**, every group begins with kind 1 and no record is kind 2 or 26: the log is today's.
//! - **On**, every group of a commit that reported events begins with kind 2, holding its first
//!   change timestamp; each write with a before-image is kind 26 (a create stays kind 16 or 18).
//!   The timestamps a group reserves (one per write) are above those of every group before it.
//! - **Replay** applies kind 26 as the plain kind: the same workload logged with and without
//!   images reopens to the same store, the serial state.
//! - **The clock after an open** starts past the largest timestamp the open replayed.
//!
//! A child writes and exits with the database open, so the WAL holds every group.
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test main cdc_wal_records::
//! ```

#![cfg(all(
    feature = "cdc",
    feature = "gql",
    feature = "wal",
    feature = "grafeo-file"
))]

use std::path::{Path, PathBuf};

use grafeo_common::storage::log_record::{LogRecord, Origin, read_log_records};
use grafeo_common::testing::child_process;
use grafeo_common::types::{EpochId, HlcTimestamp, Value};
use grafeo_engine::cdc::EntityId;
use grafeo_engine::config::StorageFormat;
use grafeo_engine::{Config, GrafeoDB};
use grafeo_storage::file::v3::header::FileHeaderV3;
use grafeo_storage::wal::{ScanOptions, WalScan, list_wal_directory};

const MODE_VAR: &str = "GRAFEO_CDC_WAL_RECORDS_MODE";
const LIVE_VAR: &str = "GRAFEO_CDC_WAL_RECORDS_LIVE";
const PATH_VAR: &str = "GRAFEO_CDC_WAL_RECORDS_PATH";

/// Every LPG write form, one statement or transaction each.
const WORKLOAD: &[&str] = &[
    "INSERT (:Person {name: 'Alix', city: 'Amsterdam', age: 34})",
    "INSERT (:Person {name: 'Gus', city: 'Prague'})",
    "INSERT (:Person {name: 'Vincent'})",
    "MATCH (p:Person {name: 'Alix'}) SET p.city = 'Paris'",
    "MATCH (p:Person {name: 'Gus'}) SET p.age = 28",
    "MATCH (p:Person {name: 'Alix'}) REMOVE p.age",
    "MATCH (p:Person {name: 'Gus'}) SET p:Pilot",
    "MATCH (p:Person {name: 'Gus'}) REMOVE p:Pilot",
    "MATCH (a:Person {name: 'Alix'}), (b:Person {name: 'Gus'}) INSERT (a)-[:KNOWS {since: 2019}]->(b)",
    "MATCH (a:Person {name: 'Gus'}), (b:Person {name: 'Vincent'}) INSERT (a)-[:KNOWS {since: 2021}]->(b)",
    "MATCH (:Person {name: 'Alix'})-[k:KNOWS]->() SET k.since = 2020",
    "MATCH (:Person {name: 'Alix'})-[k:KNOWS]->() REMOVE k.since",
    "MATCH (:Person {name: 'Gus'})-[k:KNOWS]->(:Person {name: 'Vincent'}) DELETE k",
    "MATCH (p:Person {name: 'Vincent'}) DETACH DELETE p",
];

/// A transaction of several writes, after the statements.
const TRANSACTION: &[&str] = &[
    "INSERT (:Person {name: 'Mia', city: 'Lyon'})",
    "MATCH (p:Person {name: 'Mia'}) SET p.city = 'Lille'",
    "MATCH (a:Person {name: 'Mia'}), (b:Person {name: 'Alix'}) INSERT (a)-[:KNOWS]->(b)",
    "MATCH (p:Person {name: 'Gus'}) DETACH DELETE p",
];

fn config(path: &Path, cdc: bool) -> Config {
    let config = Config::persistent(path).with_storage_format(StorageFormat::Auto);
    if cdc { config.with_cdc() } else { config }
}

fn run_workload(db: &GrafeoDB) {
    for statement in WORKLOAD {
        db.execute(statement)
            .unwrap_or_else(|error| panic!("{statement}: {error}"));
    }
    let mut session = db.session();
    session.begin_transaction().unwrap();
    for statement in TRANSACTION {
        session
            .execute(statement)
            .unwrap_or_else(|error| panic!("{statement}: {error}"));
    }
    session.commit().unwrap();
}

fn wal_dir(path: &Path) -> PathBuf {
    let mut sidecar = path.as_os_str().to_owned();
    sidecar.push(".wal");
    PathBuf::from(sidecar)
}

/// Runs the workload in a child with CDC `mode` ("on", "off", or "ahead": on, with its clock an
/// hour ahead of the wall clock), which exits with the database open.
fn crash_after_workload(mode: &str, path: &Path) {
    let live = path.with_extension("live.json");
    let output = child_process::output_with_deadline(
        grafeo_common::child_test!("cdc_wal_records_child")
            .env(MODE_VAR, mode)
            .env(PATH_VAR, path)
            .env(LIVE_VAR, &live),
        std::time::Duration::from_secs(60),
    )
    .unwrap();
    assert!(
        output.status.success(),
        "the child failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(wal_dir(path).exists(), "the crash left the WAL");
}

#[test]
fn cdc_wal_records_child() {
    let (Ok(mode), Some(path)) = (std::env::var(MODE_VAR), std::env::var_os(PATH_VAR)) else {
        return;
    };
    let db = GrafeoDB::with_config(config(Path::new(&path), mode != "off")).unwrap();
    if mode == "ahead" {
        // A clock an hour ahead of the wall clock: the reopen's wall clock is then behind
        // every timestamp the log reserved, as after a clock set back.
        let hour_ahead = HlcTimestamp::now().physical_ms() + 3_600_000;
        db.cdc_log()
            .clock()
            .update(HlcTimestamp::new(hour_ahead, 0));
    }
    if mode == "toggle" {
        toggle_writes(&db);
    } else if mode == "oversized" {
        oversized_writes(&db);
    } else {
        run_workload(&db);
    }
    if mode == "direct" {
        direct_writes(&db);
    }
    if let Some(live) = std::env::var_os(LIVE_VAR) {
        let events = db
            .changes_between(EpochId::new(0), db.current_epoch())
            .unwrap();
        std::fs::write(live, serde_json::to_vec(&events).unwrap()).unwrap();
    }
    std::process::exit(0);
}

/// Writes through the direct API: a create, a set and a delete that commit at once, and a batch.
fn direct_writes(db: &GrafeoDB) {
    let id = db
        .create_node_with_props(&["Person"], [("name", Value::from("Hans"))])
        .unwrap();
    db.set_node_property(id, "city", Value::from("Berlin"))
        .unwrap();
    db.batch_create_nodes("Vector", "v", vec![vec![1.0, 2.0], vec![3.0, 4.0]])
        .unwrap();
    assert!(db.delete_node(id).unwrap());
}

/// The records of each group of the WAL of the database file at `path`, in order.
fn groups(path: &Path) -> Vec<Vec<LogRecord>> {
    let page = std::fs::read(path).unwrap();
    let database_id = FileHeaderV3::decode(&page[..4096]).unwrap().database_id;
    let listing = list_wal_directory(&wal_dir(path)).unwrap();
    let first_lsn = listing.segments.first().expect("a segment").0;
    let mut scan = WalScan::open(wal_dir(path), ScanOptions::new(database_id, first_lsn)).unwrap();
    let mut groups = Vec::new();
    while let Some(mut group) = scan.next_group().unwrap() {
        let mut records = Vec::new();
        while let Some(payload) = group.next_payload().unwrap() {
            read_log_records(payload, &mut |record| {
                records.push(record);
                Ok(())
            })
            .unwrap();
        }
        groups.push(records);
    }
    groups
}

fn kind_of(record: &LogRecord) -> u8 {
    record.kind()
}

/// The persons and edges a database holds, sorted, as text.
fn dump(db: &GrafeoDB) -> Vec<String> {
    let mut rows: Vec<String> = [
        "MATCH (p:Person) RETURN p.name, p.city, p.age, labels(p)",
        "MATCH (a)-[k:KNOWS]->(b) RETURN a.name, b.name, k.since",
    ]
    .iter()
    .flat_map(|query| {
        db.execute(query)
            .unwrap()
            .rows()
            .iter()
            .map(|row| format!("{row:?}"))
            .collect::<Vec<_>>()
    })
    .collect();
    rows.sort();
    rows
}

/// Off: every group begins with kind 1, and no record is kind 2 or 26.
#[test]
fn with_cdc_off_the_log_has_neither_clocks_nor_images() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("off.grafeo");
    crash_after_workload("off", &path);
    let groups = groups(&path);
    assert!(groups.len() >= WORKLOAD.len(), "{} groups", groups.len());
    for (number, group) in groups.iter().enumerate() {
        let kinds: Vec<u8> = group.iter().map(kind_of).collect();
        assert_eq!(kinds.first(), Some(&1), "group {number}: {kinds:?}");
        assert!(
            !kinds.contains(&2) && !kinds.contains(&26),
            "group {number}: {kinds:?}"
        );
        if let Some(LogRecord::GroupBegin(begin)) = group.first() {
            assert_eq!(begin.clock, None, "group {number}");
        }
    }
}

/// On: every group of the workload begins with kind 2; each write that replaced something is kind
/// 26 and each create a plain kind; the groups' timestamps, one per write, rise across groups.
#[test]
fn with_cdc_on_each_group_has_a_clock_and_each_replacing_write_an_image() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("on.grafeo");
    crash_after_workload("on", &path);
    let groups = groups(&path);
    let mut last_used: Option<u64> = None;
    let mut images = 0;
    let mut creates = 0;
    for (number, group) in groups.iter().enumerate() {
        let Some(LogRecord::GroupBegin(begin)) = group.first() else {
            panic!("group {number} does not begin with a begin: {group:?}");
        };
        let writes = group.len() - 1;
        if writes == 0 {
            continue;
        }
        let clock = begin
            .clock
            .unwrap_or_else(|| panic!("group {number} reported events without a clock"));
        assert_eq!(kind_of(&group[0]), 2, "group {number}");
        if let Some(last) = last_used {
            assert!(
                clock.as_u64() > last,
                "group {number}: its first timestamp {} is not above {last}",
                clock.as_u64()
            );
        }
        last_used = Some(clock.as_u64() + u64::try_from(writes).unwrap() - 1);
        for record in &group[1..] {
            match record {
                LogRecord::DataWithBefore { .. } => images += 1,
                LogRecord::Data { op, .. } => {
                    let kind = op.kind();
                    assert!(
                        kind == 16 || kind == 18,
                        "group {number}: a kind {kind} write without its image"
                    );
                    creates += 1;
                }
                other => panic!("group {number}: {other:?}"),
            }
        }
    }
    // The statements: a set over a value, a fresh set, a removal, a label added and removed, an
    // edge's set and removal, an edge delete, and Vincent's delete (his edge went before); the
    // transaction: Mia's set, then Gus's edge and Gus. Three people and two edges were created,
    // then Mia and her edge.
    assert_eq!(images, 12, "writes with images");
    assert_eq!(creates, 7, "creates");
}

/// Replay applies kind 26 as the plain kind: the workload logged with images and without reopens
/// to the same store, the serial state.
#[test]
fn a_log_with_images_replays_to_the_store_one_without_them_gives() {
    let dir = tempfile::tempdir().unwrap();
    let (on, off) = (dir.path().join("on.grafeo"), dir.path().join("off.grafeo"));
    crash_after_workload("on", &on);
    crash_after_workload("off", &off);
    let serial = {
        let db = GrafeoDB::new_in_memory();
        run_workload(&db);
        dump(&db)
    };
    for path in [&on, &off] {
        let db = GrafeoDB::with_config(config(path, false)).unwrap();
        assert_eq!(dump(&db), serial, "{}", path.display());
        db.close().unwrap();
    }
}

/// After an open that replayed clocked groups, the clock starts past the largest timestamp they
/// reserved, also when the wall clock is behind them (the child ran its clock an hour ahead):
/// the first event after the open is above it.
#[test]
fn the_clock_after_an_open_starts_past_what_it_replayed() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("clock.grafeo");
    crash_after_workload("ahead", &path);
    let replayed = groups(&path)
        .iter()
        .filter_map(|group| match group.first() {
            Some(LogRecord::GroupBegin(begin)) => begin
                .clock
                .map(|clock| clock.as_u64() + u64::try_from(group.len() - 2).unwrap_or(0)),
            _ => None,
        })
        .max()
        .expect("clocked groups");
    let db = GrafeoDB::with_config(config(&path, true)).unwrap();
    let id = db
        .create_node_with_props(
            &["Person"],
            [("name", grafeo_common::types::Value::from("Butch"))],
        )
        .unwrap();
    let events = db.cdc_log().history(EntityId::Node(id));
    let first: HlcTimestamp = events.first().expect("the create's event").timestamp;
    assert!(
        first.as_u64() > replayed,
        "the first timestamp after the open, {}, is not above the replayed {replayed}",
        first.as_u64()
    );
    db.close().unwrap();
}

/// The live events of the child that ran the workload on `path`.
fn live_of(path: &Path) -> Vec<grafeo_engine::cdc::ChangeEvent> {
    serde_json::from_slice(&std::fs::read(path.with_extension("live.json")).unwrap()).unwrap()
}

/// The timestamps the live log published are the ones each commit reserved with its group: an
/// event of a group's epoch lies in `clock .. clock + writes`, on the session paths and the
/// direct ones (calls that commit at once and a batch), and every group of the direct API has a
/// clock.
#[test]
fn the_live_timestamps_are_the_reservations_of_their_groups() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("direct.grafeo");
    crash_after_workload("direct", &path);
    let groups = groups(&path);
    let live = live_of(&path);
    assert!(live.len() > 20, "{} live events", live.len());
    for event in &live {
        let (clock, writes) = groups
            .iter()
            .find_map(|group| match group.first() {
                Some(LogRecord::GroupBegin(begin)) if begin.epoch == event.epoch => {
                    Some((begin.clock, u64::try_from(group.len() - 1).unwrap()))
                }
                _ => None,
            })
            .unwrap_or_else(|| panic!("no group at the epoch of {event:?}"));
        let clock = clock.unwrap_or_else(|| panic!("the group of {event:?} has no clock"));
        let at = event.timestamp.as_u64();
        assert!(
            (clock.as_u64()..clock.as_u64() + writes).contains(&at),
            "{event:?} is outside its group's reservation {clock:?} + {writes}"
        );
    }
    let direct: Vec<_> = groups
        .iter()
        .filter_map(|group| match group.first() {
            Some(LogRecord::GroupBegin(begin)) if begin.origin == Origin::Direct => Some(begin),
            _ => None,
        })
        .collect();
    assert!(direct.len() >= 4, "{} direct groups", direct.len());
    assert!(
        direct.iter().all(|begin| begin.clock.is_some()),
        "{direct:?}"
    );
}

/// The decision to capture is a commit's own, taken once: a session that began with CDC on
/// commits with a clock and reports its events though CDC was turned off before its commit, and
/// one that began with CDC off reports nothing and logs no clock though it was turned on.
#[test]
fn a_toggle_between_the_write_and_the_commit_keeps_the_commits_decision() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("toggle.grafeo");
    crash_after_workload("toggle", &path);
    let live = live_of(&path);
    assert_eq!(live.len(), 1, "{live:?}");
    assert!(format!("{live:?}").contains("Mia"));
    let clocks: Vec<bool> = groups(&path)
        .iter()
        .filter_map(|group| match group.first() {
            Some(LogRecord::GroupBegin(begin)) if group.len() > 1 => Some(begin.clock.is_some()),
            _ => None,
        })
        .collect();
    assert_eq!(clocks, [true, false], "Mia's commit clocked, Vincent's not");
}

/// The child's toggles: a session begun with CDC on commits after CDC went off, and one begun
/// with it off commits after it came back on.
#[expect(
    deprecated,
    reason = "the runtime toggle of this open is what the case turns"
)]
fn toggle_writes(db: &GrafeoDB) {
    let mut on = db.session();
    on.begin_transaction().unwrap();
    on.execute("INSERT (:Person {name: 'Mia'})").unwrap();
    db.set_cdc_enabled(false);
    on.commit().unwrap();
    let mut off = db.session();
    off.begin_transaction().unwrap();
    off.execute("INSERT (:Person {name: 'Vincent'})").unwrap();
    db.set_cdc_enabled(true);
    off.commit().unwrap();
}

/// The child's deletes of nodes too large for their whole before-image (more properties than
/// the list rule lets a record hold), in a session commit and on the direct API, beside a small
/// one.
fn oversized_writes(db: &GrafeoDB) {
    for name in ["Big", "Huge", "Small"] {
        db.execute(&format!("INSERT (:Person {{name: '{name}'}})"))
            .unwrap();
    }
    for name in ["Big", "Huge"] {
        // 140,000 properties of a few letters and a boolean: each takes fewer bytes than
        // its slot, so together they pass the list rule's allowance.
        for chunk in 0..140 {
            let sets: Vec<String> = (chunk * 1000..(chunk + 1) * 1000)
                .map(|i| format!("`{}`: true", tiny_key(i)))
                .collect();
            let sets = sets.join(", ");
            db.execute(&format!(
                "MATCH (n:Person {{name: '{name}'}}) SET n += {{{sets}}}"
            ))
            .unwrap();
        }
    }
    let mut session = db.session();
    session.begin_transaction().unwrap();
    session
        .execute("MATCH (n:Person {name: 'Big'}) DELETE n")
        .unwrap();
    session.commit().unwrap();
    let huge = db
        .execute("MATCH (n:Person {name: 'Huge'}) RETURN id(n)")
        .unwrap()
        .rows()[0][0]
        .clone();
    let Value::Int64(huge) = huge else {
        panic!("an id, not {huge:?}")
    };
    assert!(
        db.delete_node(grafeo_common::types::NodeId::new(
            u64::try_from(huge).unwrap()
        ))
        .unwrap()
    );
    db.execute("MATCH (n:Person {name: 'Small'}) DELETE n")
        .unwrap();
}

/// A distinct key of one to three letters for `i`, the shortest first (`cdc_durable_feed` uses
/// it too).
pub(crate) fn tiny_key(mut i: usize) -> String {
    const LETTERS: &[u8] = b"abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ";
    let base = LETTERS.len();
    let mut length = 1;
    while i >= base.pow(length) {
        i -= base.pow(length);
        length += 1;
    }
    (0..length)
        .rev()
        .map(|place| char::from(LETTERS[i / base.pow(place) % base]))
        .collect()
}

/// A delete whose whole before-image passes the record's limits succeeds with CDC on, in a
/// session commit and on the direct API: it logs as kind 26 with its identity alone (a node's
/// labels), and its live event keeps that identity and leaves the properties out (`before` is
/// `None`), as the log gives it back. A small node keeps its whole image (`before` its name).
/// An edge would need as many properties, which take minutes to set on one edge; its identity's
/// bytes are pinned in grafeo-common, and its live event is checked in memory (below).
#[test]
fn a_delete_too_large_for_its_image_logs_its_identity() {
    use grafeo_common::storage::log_record::BeforeImage;
    use grafeo_engine::cdc::ChangeKind;
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("oversized.grafeo");
    crash_after_workload("oversized", &path);
    let images: Vec<BeforeImage> = groups(&path)
        .into_iter()
        .flatten()
        .filter_map(|record| match record {
            LogRecord::Data { op, .. } if matches!(op.kind(), 17 | 19) => {
                panic!("a delete without its identity: {op:?}")
            }
            LogRecord::DataWithBefore { op, before, .. } if matches!(op.kind(), 17 | 19) => {
                Some(before)
            }
            _ => None,
        })
        .collect();
    let deleted: Vec<_> = live_of(&path)
        .into_iter()
        .filter(|event| event.kind == ChangeKind::Delete)
        .collect();
    assert_eq!(images.len(), 3, "{images:?}");
    assert_eq!(deleted.len(), 3, "{deleted:?}");
    let person = Some(vec!["Person".to_string()]);
    let mut identities = 0;
    for (image, event) in images.iter().zip(&deleted) {
        match image {
            BeforeImage::NodeIdentity { labels } => {
                identities += 1;
                assert_eq!(labels.as_slice(), ["Person"]);
                assert!(matches!(event.entity_id, EntityId::Node(_)), "{event:?}");
                assert_eq!(event.labels, person, "{event:?}");
                assert!(event.before.is_none(), "{event:?}");
            }
            BeforeImage::Node { labels, properties } => {
                assert_eq!(labels.as_slice(), ["Person"]);
                assert_eq!(properties.len(), 1, "the small node's name");
                assert_eq!(event.labels, person, "{event:?}");
                assert_eq!(event.before.as_ref().map(|before| before.len()), Some(1));
            }
            other => panic!("a delete's image: {other:?}"),
        }
    }
    assert_eq!(identities, 2, "Big and Huge by their identity");
    // The log with identity images replays to the deletes: Big, Huge and Small stay deleted.
    let reopened = GrafeoDB::with_config(config(&path, true)).unwrap();
    let left = reopened
        .execute("MATCH (n:Person) RETURN n.name")
        .unwrap()
        .rows()
        .len();
    assert_eq!(left, 0, "the deleted nodes stay deleted after the reopen");
}

/// A delete whose image passes the record's byte maximum (properties of 1.2 GB as the value
/// codec writes them, set one by one) succeeds with CDC on, in a session commit and on the
/// direct API, and its event keeps the node's labels, or the edge's type and endpoints, and
/// leaves the properties out by the same rule the WAL writer follows. In memory: writing the
/// properties to a WAL would take 1.2 GB of disk.
#[test]
fn a_delete_past_the_byte_maximum_reports_without_its_image() {
    // Three values of 412 MB each: past the default property size, which the case lifts.
    let db =
        GrafeoDB::with_config(Config::in_memory().with_cdc().without_max_property_size()).unwrap();
    let line = Value::from("Amsterdam ".repeat(103).as_str());
    let notes = Value::List(std::sync::Arc::from(vec![line; 400_000]));
    let nodes: Vec<_> = (0..2)
        .map(|_| {
            let id = db.create_node(&["Person"]).unwrap();
            for key in ["a", "b", "c"] {
                db.set_node_property(id, key, notes.clone()).unwrap();
            }
            id
        })
        .collect();
    let (jules, mia) = (
        db.create_node(&["Person"]).unwrap(),
        db.create_node(&["Person"]).unwrap(),
    );
    let edges: Vec<_> = (0..2)
        .map(|_| {
            let id = db.create_edge(jules, mia, "KNOWS").unwrap();
            for key in ["a", "b", "c"] {
                db.set_edge_property(id, key, notes.clone()).unwrap();
            }
            id
        })
        .collect();
    let mut session = db.session();
    session.begin_transaction().unwrap();
    session
        .execute(&format!(
            "MATCH ()-[k]->() WHERE id(k) = {} DELETE k",
            edges[0].as_u64()
        ))
        .unwrap();
    session
        .execute(&format!(
            "MATCH (n) WHERE id(n) = {} DELETE n",
            nodes[0].as_u64()
        ))
        .unwrap();
    session.commit().unwrap();
    assert!(db.delete_node(nodes[1]).unwrap());
    assert!(db.delete_edge(edges[1]).unwrap());
    for id in edges {
        let history = db.cdc_log().history(EntityId::Edge(id));
        let delete = history.last().unwrap();
        assert_eq!(delete.kind, grafeo_engine::cdc::ChangeKind::Delete);
        assert!(delete.before.is_none(), "{delete:?}");
        assert_eq!(
            (delete.edge_type.as_deref(), delete.src_id, delete.dst_id),
            (Some("KNOWS"), Some(jules.as_u64()), Some(mia.as_u64()))
        );
    }
    for id in nodes {
        let history = db.cdc_log().history(EntityId::Node(id));
        let delete = history.last().unwrap();
        assert_eq!(delete.kind, grafeo_engine::cdc::ChangeKind::Delete);
        assert!(delete.before.is_none(), "{delete:?}");
        assert_eq!(delete.labels, Some(vec!["Person".to_string()]));
    }
}
