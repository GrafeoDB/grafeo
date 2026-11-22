//! The in-memory change log is bounded in bytes (#679 patch 5, ruling Q6).
//!
//! `CdcRetentionConfig::with_max_bytes` bounds what the log's events hold, counted as they come
//! (the running count the memory account already keeps). When the log passes the bound, its
//! oldest whole commits are dropped until it is at or below three quarters of the bound; the
//! newest commit is always kept. So a prune keeps at most three quarters of the bound unless only
//! the newest commit remains, and drops no commit that did not need to go. Without an explicit
//! bound the database derives one from its memory limit: a sixteenth of it, capped at the limit.
//! `unlimited()` lifts epochs and events, not the byte bound. The count equals a walk of the log
//! after every step, and a large value is counted whole.
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test main cdc_byte_bound::
//! ```

#![cfg(all(feature = "cdc", feature = "gql"))]

use std::collections::{HashMap, VecDeque};

use grafeo_common::types::{EpochId, NodeId, Value};
use grafeo_engine::cdc::{CdcLog, CdcRetentionConfig, ChangeKind, EntityId};
use grafeo_engine::{Config, GrafeoDB};

const KIB: usize = 1 << 10;
const MIB: usize = 1 << 20;

fn database(config: Config) -> GrafeoDB {
    GrafeoDB::with_config(config.with_cdc()).unwrap()
}

/// Inserts `count` nodes, one statement each (so each is its own epoch), each holding a text of
/// `size` bytes; returns their ids in order.
fn insert(db: &GrafeoDB, count: usize, size: usize) -> Vec<EntityId> {
    (0..count)
        .map(|i| {
            let text = format!("{i:08}{}", "x".repeat(size));
            let id = db
                .create_node_with_props(&["Note"], [("text", Value::from(text.as_str()))])
                .unwrap();
            EntityId::Node(id)
        })
        .collect()
}

fn counted(db: &GrafeoDB) -> usize {
    db.cdc_log().counted_memory_bytes().0
}

fn assert_counted_is_walked(db: &GrafeoDB, what: &str) {
    assert_eq!(
        db.cdc_log().account_mismatches(),
        Vec::<(String, usize, usize)>::new(),
        "{what}: the count equals the walk"
    );
}

/// Past the bound the oldest events go: the count stays at or below it, the newest events stay
/// and the first ones are gone; the count equals the walk after every prune.
#[test]
fn the_oldest_events_go_past_the_byte_bound() {
    let bound = 256 * KIB;
    let db = database(
        Config::in_memory()
            .with_cdc_retention(CdcRetentionConfig::unlimited().with_max_bytes(bound)),
    );
    assert_eq!(db.cdc_log().max_bytes(), Some(bound));
    let ids = insert(&db, 200, 4 * KIB);
    assert!(
        counted(&db) <= bound,
        "{} counted past {bound}",
        counted(&db)
    );
    // At least three quarters stay, less the one commit (a 4 KiB text and its event) whose drop
    // brought the log to them.
    assert!(
        counted(&db) + 6 * KIB > bound - bound / 4,
        "the log holds up to its bound, not far below: {}",
        counted(&db)
    );
    assert!(
        !db.cdc_log().history(ids[199]).is_empty(),
        "the newest event stays"
    );
    assert!(
        db.cdc_log().history(ids[0]).is_empty(),
        "the first event went"
    );
    assert_counted_is_walked(&db, "after the prunes");
}

/// The count equals a walk of the log after every statement of a mixed sequence (creates of
/// several sizes, updates that grow and shrink, deletes, edges), with the bound pruning between.
#[test]
fn the_count_equals_the_walk_after_any_sequence() {
    let db = database(
        Config::in_memory()
            .with_cdc_retention(CdcRetentionConfig::unlimited().with_max_bytes(64 * KIB)),
    );
    let mut ids = Vec::new();
    for step in 0..300_usize {
        match step % 5 {
            0 | 1 => ids.extend(insert(&db, 1, 100 + (step * 97) % 3_000)),
            2 => {
                db.execute(&format!(
                    "MATCH (n:Note) WHERE n.text STARTS WITH '{:08}' SET n.text = '{}'",
                    step / 2,
                    "y".repeat((step * 31) % 2_000)
                ))
                .unwrap();
            }
            3 => {
                db.execute(&format!(
                    "MATCH (a:Note), (b:Note) WHERE a.text STARTS WITH '{:08}' AND b.text STARTS WITH '{:08}' \
                     INSERT (a)-[:LINKS {{w: {step}}}]->(b)",
                    step / 3,
                    step / 4
                ))
                .unwrap();
            }
            _ => {
                db.execute(&format!(
                    "MATCH (n:Note) WHERE n.text STARTS WITH '{:08}' DETACH DELETE n",
                    step / 5
                ))
                .unwrap();
            }
        }
        assert_counted_is_walked(&db, &format!("step {step}"));
        assert!(
            counted(&db) <= 64 * KIB,
            "step {step}: {} counted",
            counted(&db)
        );
    }
}

/// The same in a named graph, whose name each entity's key and each commit's list of keys own
/// on the heap: the count equals the walk after every statement of a mixed sequence there, with
/// the bound pruning between, and every event names the graph.
#[test]
fn the_count_equals_the_walk_in_a_named_graph() {
    let db = database(
        Config::in_memory()
            .with_cdc_retention(CdcRetentionConfig::unlimited().with_max_bytes(64 * KIB)),
    );
    db.execute("CREATE GRAPH trips").unwrap();
    let session = db.session();
    session.use_graph("trips");
    for step in 0..300_usize {
        let statement = match step % 5 {
            0 | 1 => format!(
                "INSERT (:Note {{text: '{step:08}{}'}})",
                "x".repeat(100 + (step * 97) % 3_000)
            ),
            2 => format!(
                "MATCH (n:Note) WHERE n.text STARTS WITH '{:08}' SET n.text = '{}'",
                step / 2,
                "y".repeat((step * 31) % 2_000)
            ),
            3 => format!(
                "MATCH (a:Note), (b:Note) WHERE a.text STARTS WITH '{:08}' AND b.text STARTS WITH '{:08}' \
                 INSERT (a)-[:LINKS {{w: {step}}}]->(b)",
                step / 3,
                step / 4
            ),
            _ => format!(
                "MATCH (n:Note) WHERE n.text STARTS WITH '{:08}' DETACH DELETE n",
                step / 5
            ),
        };
        session
            .execute(&statement)
            .unwrap_or_else(|error| panic!("step {step}: {error}"));
        assert_counted_is_walked(&db, &format!("step {step}"));
        assert!(
            counted(&db) <= 64 * KIB,
            "step {step}: {} counted",
            counted(&db)
        );
    }
    let events = db
        .cdc_log()
        .changes_between(EpochId::new(0), db.current_epoch());
    assert!(!events.is_empty(), "the log holds the newest commits");
    assert!(
        events
            .iter()
            .all(|event| event.graph.as_deref() == Some("trips")),
        "every event names the graph"
    );
}

/// A large value is counted whole: 100 events holding 1 MiB texts are counted at 100 MiB or more,
/// under a bound that holds them all.
#[test]
fn a_large_value_is_counted_whole() {
    let bound = 256 * MIB;
    let db = database(
        Config::in_memory()
            .with_memory_limit(1024 * MIB)
            .with_cdc_retention(CdcRetentionConfig::unlimited().with_max_bytes(bound)),
    );
    assert_eq!(db.cdc_log().max_bytes(), Some(bound));
    insert(&db, 100, MIB);
    assert!(
        counted(&db) >= 100 * MIB,
        "100 texts of 1 MiB counted at {} bytes",
        counted(&db)
    );
    assert_eq!(db.cdc_log().event_count(), 100, "nothing pruned");
    assert_counted_is_walked(&db, "100 large events");
}

/// Drops the front of `held` (the commits' nodes, oldest first) whose node has no events left;
/// how many went.
fn drop_gone(db: &GrafeoDB, held: &mut VecDeque<EntityId>) -> usize {
    let mut dropped = 0;
    while held
        .front()
        .is_some_and(|node| db.cdc_log().history(*node).is_empty())
    {
        held.pop_front();
        dropped += 1;
    }
    dropped
}

/// A prune runs only above the bound and drops the fewest oldest commits, oldest first: what it
/// keeps is at most three quarters of the bound unless only the newest commit remains, and before
/// it dropped its last commit the log was above three quarters (kept plus the size of the last
/// commit dropped). The log reports each prune's counts; which commits went is read from the log.
#[test]
fn a_prune_drops_the_oldest_commits_until_three_quarters_remain() {
    let bound = 256 * KIB;
    let target = bound - bound / 4;
    let db = database(
        Config::in_memory()
            .with_cdc_retention(CdcRetentionConfig::unlimited().with_max_bytes(bound)),
    );
    let mut held: VecDeque<EntityId> = VecDeque::new();
    let mut prunes = 0;
    for step in 0..400 {
        let prior = db.cdc_log().last_byte_prune();
        held.push_back(insert(&db, 1, 1 + (step * 37) % (8 * KIB))[0]);
        let dropped = drop_gone(&db, &mut held);
        let kept = counted(&db);
        if dropped == 0 {
            assert!(
                kept <= bound || held.len() == 1,
                "step {step}: {kept} past the bound"
            );
            continue;
        }
        prunes += 1;
        let report = db.cdc_log().last_byte_prune();
        assert_ne!(report, prior, "step {step}: the byte bound pruned");
        let (began, before_last, left) = report.unwrap();
        assert_eq!(left, kept, "step {step}");
        assert!(
            began > bound,
            "step {step}: a prune ran at {began}, not past the bound"
        );
        assert!(
            kept <= target || held.len() == 1,
            "step {step}: a prune kept {kept}, above three quarters of {bound}"
        );
        assert!(
            before_last > target,
            "step {step}: a prune kept {kept} and dropped a commit it did not need to"
        );
        assert_counted_is_walked(&db, &format!("step {step}"));
    }
    assert!(prunes > 2, "the log filled and pruned");
}

/// Commits of the given sizes (KiB of text, one node each) under `bound`; returns which of them
/// the log keeps, oldest first, and the bytes it holds.
fn kept_of(sizes: &[usize], bound: usize) -> (Vec<bool>, usize) {
    let db = database(
        Config::in_memory()
            .with_cdc_retention(CdcRetentionConfig::unlimited().with_max_bytes(bound)),
    );
    let nodes: Vec<EntityId> = sizes
        .iter()
        .map(|size| insert(&db, 1, size * KIB)[0])
        .collect();
    assert_counted_is_walked(&db, &format!("{sizes:?}"));
    let kept = nodes
        .iter()
        .map(|node| !db.cdc_log().history(*node).is_empty())
        .collect();
    (kept, counted(&db))
}

/// The three cases of the rule, at a 256 KiB bound (three quarters: 192 KiB). A prune drops
/// whole commits, so it can keep well under three quarters, but never more than it must.
#[test]
fn a_prune_keeps_what_minimality_gives() {
    let bound = 256 * KIB;
    // 100, 60 and 60 fit; 40 passes the bound; dropping the 100 is enough: 160 stay, under three
    // quarters, though the 60s are smaller than a quarter.
    let (kept, held) = kept_of(&[100, 60, 60, 40], bound);
    assert_eq!(kept, [false, true, true, true]);
    assert!(held <= bound - bound / 4, "{held}");
    // 60 and 40 and then 160 pass the bound; dropping the 60 leaves 200, above three quarters,
    // so the 40 goes too, though smaller than a quarter: 160 stay.
    let (kept, held) = kept_of(&[60, 40, 160], bound);
    assert_eq!(kept, [false, false, true]);
    assert!(held > 160 * KIB && held <= bound - bound / 4, "{held}");
    // A newest commit larger than the bound is kept alone, over the bound.
    let (kept, held) = kept_of(&[10, 300], bound);
    assert_eq!(kept, [false, true]);
    assert!(held > bound, "{held}");
}

/// A commit of many nodes goes whole or stays whole across prunes, with updates of older nodes
/// between them; the count equals the walk and stays within the bound.
#[test]
fn a_many_entity_commit_goes_whole() {
    let bound = 128 * KIB;
    let db = database(
        Config::in_memory()
            .with_cdc_retention(CdcRetentionConfig::unlimited().with_max_bytes(bound)),
    );
    for commit in 0..40 {
        let nodes: Vec<String> = (0..20)
            .map(|i| {
                format!(
                    "(:Batch {{commit: {commit}, i: {i}, text: '{}'}})",
                    "z".repeat(200)
                )
            })
            .collect();
        db.execute(&format!("INSERT {}", nodes.join(", "))).unwrap();
        if commit % 3 == 2 {
            db.execute(&format!(
                "MATCH (b:Batch {{commit: {}}}) SET b.text = 'short'",
                commit / 2
            ))
            .unwrap();
        }
        assert_counted_is_walked(&db, &format!("commit {commit}"));
        assert!(counted(&db) <= bound, "commit {commit}: {}", counted(&db));
        // Nodes get ids in creation order: commit `c` made nodes `20 c` to `20 c + 19`.
        for older in 0..=commit {
            let creates = (0..20)
                .filter(|i| {
                    let node = EntityId::Node(NodeId::new(older * 20 + i));
                    db.cdc_log()
                        .history(node)
                        .iter()
                        .any(|event| event.kind == ChangeKind::Create)
                })
                .count();
            assert!(
                creates == 0 || creates == 20,
                "commit {older} after {commit}: {creates} of 20 creates kept"
            );
        }
    }
}

/// A 1-byte bound keeps the newest commit alone; a bound of exactly one commit keeps that commit
/// (it is not past the bound) and then the newest alone.
#[test]
fn a_tiny_bound_keeps_the_newest_commit() {
    let db = database(
        Config::in_memory().with_cdc_retention(CdcRetentionConfig::unlimited().with_max_bytes(1)),
    );
    for step in 0..5 {
        let node = insert(&db, 1, 100)[0];
        assert_eq!(db.cdc_log().event_count(), 1, "step {step}");
        assert_eq!(db.cdc_log().history(node).len(), 1, "step {step}");
    }

    let measure = database(Config::in_memory());
    insert(&measure, 1, 2 * KIB);
    let one = counted(&measure);
    let db = database(
        Config::in_memory().with_cdc_retention(CdcRetentionConfig::unlimited().with_max_bytes(one)),
    );
    insert(&db, 1, 2 * KIB);
    assert_eq!(counted(&db), one, "one commit, exactly the bound, stays");
    assert_eq!(db.cdc_log().event_count(), 1);
    let newest = insert(&db, 1, 2 * KIB)[0];
    assert_eq!(db.cdc_log().event_count(), 1, "the older one went");
    assert_eq!(db.cdc_log().history(newest).len(), 1);
    // The commit queue may hold less room after the prune than in a log that never pruned.
    assert!(counted(&db) <= one, "{} of {one}", counted(&db));
}

/// With CDC off nothing is recorded and nothing is pruned; the report still names the bound.
#[test]
fn a_database_with_cdc_off_records_nothing_and_reports_the_bound() {
    let db = GrafeoDB::with_config(Config::in_memory().with_cdc_memory_limit(64 * KIB)).unwrap();
    insert(&db, 50, 4 * KIB);
    assert_eq!(db.cdc_log().event_count(), 0);
    assert_eq!(counted(&db), 0);
    assert_eq!(db.memory_usage().cdc.limit_bytes, 64 * KIB);
}

/// A log recorded one event at a time (`record`, as the helpers do) prunes as a database's does.
#[test]
fn single_event_records_prune_to_the_bound() {
    let bound = 32 * KIB;
    let log = CdcLog::with_retention(CdcRetentionConfig::unlimited().with_max_bytes(bound));
    for i in 1..=200_u64 {
        let count = log.event_count();
        let text = Value::from("t".repeat(usize::try_from(i * 13).unwrap()).as_str());
        log.record_create_node(
            NodeId::new(i),
            EpochId::new(i),
            Some(HashMap::from([("text".to_string(), text)])),
            None,
        );
        let held = log.counted_memory_bytes().0;
        let alone = log.event_count() == 1;
        if log.event_count() <= count {
            assert!(
                held <= bound - bound / 4 || alone,
                "event {i}: a prune kept {held}"
            );
        }
        assert!(held <= bound || alone, "event {i}: {held}");
        assert_eq!(
            log.account_mismatches(),
            Vec::<(String, usize, usize)>::new()
        );
    }
    assert!(
        log.history(EntityId::Node(NodeId::new(1))).is_empty(),
        "the first went"
    );
}

/// A retention set after the memory limit, without a byte bound of its own, keeps the limit; in
/// the other order too.
#[test]
fn a_retention_without_a_byte_bound_keeps_the_limit() {
    let after = Config::in_memory()
        .with_cdc_memory_limit(512 * KIB)
        .with_cdc_retention(CdcRetentionConfig::unlimited());
    assert_eq!(after.cdc_memory_limit(), Some(512 * KIB));
    let before = Config::in_memory()
        .with_cdc_retention(CdcRetentionConfig::unlimited())
        .with_cdc_memory_limit(512 * KIB);
    assert_eq!(before.cdc_memory_limit(), Some(512 * KIB));
    let own = Config::in_memory()
        .with_cdc_memory_limit(512 * KIB)
        .with_cdc_retention(CdcRetentionConfig::unlimited().with_max_bytes(MIB));
    assert_eq!(
        own.cdc_memory_limit(),
        Some(MIB),
        "a retention's own bound wins"
    );
    assert_eq!(database(after).cdc_log().max_bytes(), Some(512 * KIB));
}

/// A prune visits what it drops, not the log (a count, not a timing): on a log of 20,000
/// one-event commits, `prune_before` of two commits visits at most their events and keys, and a
/// byte prune at most twice the events it drops. The count equals the walk after each.
#[test]
fn a_prune_visits_only_what_it_drops() {
    let log = CdcLog::with_retention(CdcRetentionConfig::unlimited());
    for i in 1..=20_000_u64 {
        log.record_create_node(NodeId::new(i), EpochId::new(i), None, None);
    }
    let before = log.prune_visits();
    log.prune_before(EpochId::new(3));
    let visits = log.prune_visits() - before;
    assert_eq!(log.event_count(), 19_998);
    assert!(visits <= 4, "two events and their keys, not {visits}");
    assert_eq!(
        log.account_mismatches(),
        Vec::<(String, usize, usize)>::new()
    );

    let db = database(
        Config::in_memory()
            .with_cdc_retention(CdcRetentionConfig::unlimited().with_max_bytes(8 * MIB)),
    );
    let mut checked = false;
    for step in 0..100_000 {
        let (held, visits) = (db.cdc_log().event_count(), db.cdc_log().prune_visits());
        insert(&db, 1, 100);
        if db.cdc_log().event_count() <= held {
            let dropped = held + 1 - db.cdc_log().event_count();
            let visited = db.cdc_log().prune_visits() - visits;
            assert!(
                visited <= 2 * u64::try_from(dropped).unwrap(),
                "step {step}: {visited} visits for {dropped} events of {held}"
            );
            assert_counted_is_walked(&db, "after the byte prune");
            checked = true;
            break;
        }
    }
    assert!(checked, "the log reached its bound");
}

/// `unlimited()` lifts the epoch and event bounds, not the byte bound: the database derives it.
#[test]
fn unlimited_keeps_the_derived_byte_bound() {
    let limit = 32 * MIB;
    let db = database(
        Config::in_memory()
            .with_memory_limit(limit)
            .with_cdc_retention(CdcRetentionConfig::unlimited()),
    );
    assert_eq!(db.cdc_log().max_bytes(), Some(limit / 16));
}

/// A commit larger than the bound stays whole, as the newest commit always does, until a newer
/// commit arrives; then it goes whole and the newer one stays.
#[test]
fn a_commit_larger_than_the_bound_stays_until_a_newer_one() {
    let bound = 64 * KIB;
    let db = database(
        Config::in_memory()
            .with_cdc_retention(CdcRetentionConfig::unlimited().with_max_bytes(bound)),
    );
    let large = insert(&db, 1, 4 * bound);
    assert!(
        counted(&db) > bound,
        "the newest commit stays, over the bound"
    );
    assert_eq!(db.cdc_log().history(large[0]).len(), 1, "whole");
    let small = insert(&db, 1, KIB);
    assert!(
        db.cdc_log().history(large[0]).is_empty(),
        "the larger one went"
    );
    assert_eq!(
        db.cdc_log().history(small[0]).len(),
        1,
        "the newer one stays"
    );
    assert!(counted(&db) <= bound);
    assert_counted_is_walked(&db, "after the large commit went");
}

/// Without an explicit bound the database derives one: a sixteenth of its memory limit. Its log
/// fills up to that bound and no further; an explicit bound wins over the derived one.
#[test]
fn the_default_bound_is_a_sixteenth_of_the_memory_limit() {
    let limit = 32 * MIB;
    let db = database(Config::in_memory().with_memory_limit(limit));
    assert_eq!(db.cdc_log().max_bytes(), Some(limit / 16));
    insert(&db, 300, 16 * KIB);
    assert!(
        counted(&db) <= limit / 16,
        "{} counted past {}",
        counted(&db),
        limit / 16
    );
    assert!(counted(&db) > limit / 32, "the log fills up to its bound");
    assert_counted_is_walked(&db, "the derived bound");

    let explicit = database(
        Config::in_memory()
            .with_memory_limit(limit)
            .with_cdc_retention(CdcRetentionConfig::default().with_max_bytes(512 * KIB)),
    );
    assert_eq!(
        explicit.cdc_log().max_bytes(),
        Some(512 * KIB),
        "explicit wins"
    );
}

/// The default retention keeps its event and epoch bounds and adds no explicit byte bound: the
/// database derives that one.
#[test]
fn the_retention_config_reads_back() {
    let default = CdcRetentionConfig::default();
    assert_eq!(default.max_bytes, None, "derived by the database");
    assert_eq!(default.max_events, Some(100_000));
    let bounded = CdcRetentionConfig::unlimited().with_max_bytes(MIB);
    assert_eq!(bounded.max_bytes, Some(MIB));
    assert_eq!(bounded.max_events, None);
}

/// The bound as a setting (the amendment's binding surface): `Config::with_cdc_memory_limit` sets
/// the log's `max_bytes` and reads back through `cdc_memory_limit()` and the report's
/// `cdc.limit_bytes`. Unset, the report gives the derived sixteenth of the memory limit; above the
/// memory limit it is capped at it; 0 is refused at the open with `V001`.
#[test]
fn the_cdc_memory_limit_is_a_setting() {
    let limit = 64 * MIB;
    let set = Config::in_memory()
        .with_memory_limit(limit)
        .with_cdc_memory_limit(512 * KIB);
    assert_eq!(set.cdc_memory_limit(), Some(512 * KIB));
    assert_eq!(set.cdc_retention().max_bytes, Some(512 * KIB));
    let db = database(set);
    assert_eq!(db.cdc_log().max_bytes(), Some(512 * KIB));
    assert_eq!(db.memory_usage().cdc.limit_bytes, 512 * KIB);

    let unset = database(Config::in_memory().with_memory_limit(limit));
    assert_eq!(
        unset.memory_usage().cdc.limit_bytes,
        limit / 16,
        "the derived default"
    );

    let capped = database(
        Config::in_memory()
            .with_memory_limit(32 * MIB)
            .with_cdc_memory_limit(1024 * MIB),
    );
    assert_eq!(
        capped.memory_usage().cdc.limit_bytes,
        32 * MIB,
        "capped at the memory limit"
    );

    let error = GrafeoDB::with_config(Config::in_memory().with_cdc().with_cdc_memory_limit(0))
        .err()
        .expect("a zero bound is refused at the open");
    assert_eq!(error.error_code().as_str(), "GRAFEO-V001", "{error}");
}
