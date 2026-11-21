//! #673 part A, A4 made exact: the committed-claims index holds only the commits an open
//! transaction that can still commit began before. A reader that cannot commit (a pin outside
//! a transaction, such as a time-travel view or a held result stream, a transaction doomed by
//! a lost claim, an expired snapshot whose read still runs) never holds it back.
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --lib transaction::manager::claim_trim_tests::
//! ```

use super::*;

/// Commits behind the readers.
const COMMITS: usize = 10_000;

/// `count` transactions that each write node `from + i` alone and commit.
fn commits(mgr: &TransactionManager, from: u64, count: usize) {
    for i in 0..count {
        let tx = mgr.begin();
        mgr.record_write(tx, NodeId::new(from + u64::try_from(i).unwrap()))
            .unwrap();
        mgr.commit(tx).unwrap();
    }
}

/// With `pin` holding the oldest epoch, a transaction that began with it ends while a newer
/// one is open: the index trims to the newer one's start, keeps the commits after it, and is
/// empty once none is open, with the pin still held.
fn a_pin_does_not_hold_the_index(
    mgr: &TransactionManager,
    pin: impl FnOnce() -> Box<dyn std::any::Any>,
) {
    let held = pin();
    let older = mgr.begin();
    commits(mgr, 1, COMMITS);
    assert_eq!(mgr.claim_index_sizes(), (0, COMMITS, COMMITS));

    let newer = mgr.begin();
    mgr.abort(older).unwrap();
    assert_eq!(
        mgr.claim_index_sizes(),
        (0, 0, 0),
        "no transaction that can commit began before those commits"
    );
    assert_eq!(mgr.conflict_horizon(), mgr.start_epoch(newer));

    commits(mgr, 1 + u64::try_from(COMMITS).unwrap(), COMMITS);
    assert_eq!(
        mgr.claim_index_sizes(),
        (0, COMMITS, COMMITS),
        "the commits after the newer transaction began, which it needs"
    );
    mgr.abort(newer).unwrap();
    assert_eq!(mgr.claim_index_sizes(), (0, 0, 0), "none open");
    assert_eq!(mgr.conflict_horizon(), None);
    drop(held);
}

/// A time-travel pin (what an `AsOfView` holds) at an older epoch does not hold the index.
#[test]
fn a_time_travel_pin_does_not_hold_the_index() {
    let mgr = TransactionManager::new();
    mgr.set_history_retention(HistoryRetention::Forever);
    commits(&mgr, 1_000_000, 1);
    let at = mgr.current_epoch();
    a_pin_does_not_hold_the_index(&mgr, || Box::new(mgr.pin_read_at(at).unwrap()));
}

/// A read pin outside a transaction (what a held result stream holds) does not hold the
/// index.
#[test]
fn a_read_pin_does_not_hold_the_index() {
    let mgr = TransactionManager::new();
    a_pin_does_not_hold_the_index(&mgr, || Box::new(mgr.pin_read()));
}

/// Through the database: an `AsOfView` and a held result stream, each with 10,000 commits
/// behind it, keep nothing in the index once the transactions that began with them end.
#[cfg(feature = "gql")]
#[test]
fn an_as_of_view_and_a_held_stream_do_not_hold_the_index() {
    use crate::GrafeoDB;
    use crate::config::Config;

    for view in [true, false] {
        let db = GrafeoDB::with_config(
            Config::in_memory().with_history_retention(HistoryRetention::Forever),
        )
        .unwrap();
        db.execute("INSERT (:Person {name: 'Alix', n: 0}), (:Person {name: 'Gus', n: 0})")
            .unwrap();
        let mgr = Arc::clone(&db.live().unwrap().transaction_manager);
        let held: Box<dyn std::any::Any> = if view {
            Box::new(db.as_of(db.current_epoch()).unwrap())
        } else {
            Box::new(db.execute_streaming("MATCH (p:Person) RETURN p.n").unwrap())
        };
        let older = mgr.begin();
        let update = |name: &str, n: usize| {
            db.execute(&format!(
                "MATCH (p:Person) WHERE p.name = '{name}' SET p.n = {n}"
            ))
            .unwrap();
        };
        for n in 0..COMMITS {
            update("Alix", n);
        }
        assert_eq!(mgr.claim_index_sizes(), (0, 1, COMMITS), "view: {view}");

        let newer = mgr.begin();
        mgr.abort(older).unwrap();
        assert_eq!(mgr.claim_index_sizes(), (0, 0, 0), "view: {view}");
        for n in 0..COMMITS {
            update("Gus", n);
        }
        assert_eq!(
            mgr.claim_index_sizes(),
            (0, 1, COMMITS),
            "view: {view}: what the newer transaction needs"
        );
        mgr.abort(newer).unwrap();
        assert_eq!(
            mgr.claim_index_sizes(),
            (0, 0, 0),
            "view: {view}: none open"
        );
        drop(held);
    }
}

/// A transaction doomed by a lost key claim cannot commit: from its doom it holds the index
/// back no more, and the index trims past it while it is still open.
#[test]
fn a_doomed_transaction_does_not_hold_the_index() {
    let key = KeyClaim {
        graph: None,
        label: Arc::from("Person"),
        properties: vec!["email".to_string()],
        values: vec![HashableValue(grafeo_common::types::Value::from(
            "mia@example.org",
        ))],
    };
    let mgr = TransactionManager::new();
    let older = mgr.begin();
    commits(&mgr, 1, 1);
    let younger = mgr.begin();
    mgr.record_key(younger, key.clone()).unwrap();
    // The older transaction takes the key: the younger one is doomed.
    mgr.record_key(older, key).unwrap();
    assert!(mgr.given_up_on(younger).is_some());
    commits(&mgr, 2, COMMITS);
    assert_eq!(mgr.claim_index_sizes().2, COMMITS + 1);

    mgr.abort(older).unwrap();
    assert_eq!(mgr.state(younger), Some(TransactionState::Active));
    assert_eq!(
        mgr.claim_index_sizes().1,
        0,
        "trimmed past the doomed transaction before it ends"
    );
    assert_eq!(mgr.claim_index_sizes().2, 0);
    assert_eq!(mgr.conflict_horizon(), None);
    mgr.abort(younger).unwrap();
}

/// An expired snapshot whose read still runs keeps its versions (the read horizon) but holds
/// the index back no more: it cannot commit.
#[test]
fn an_expired_snapshot_with_a_read_running_does_not_hold_the_index() {
    let mgr = TransactionManager::new();
    mgr.set_snapshot_retention_limit(1000);
    let reader = mgr.begin();
    let at = mgr.read_horizon();
    let read = mgr.read_in(reader).unwrap().unwrap();
    commits(&mgr, 1, 1);
    let newer = mgr.begin();
    commits(&mgr, 2, 1);
    assert_eq!(mgr.claim_index_sizes().2, 2);

    // A commit that replaced past the limit expires the reader's snapshot.
    let writer = mgr.begin();
    mgr.start_commit(writer)
        .unwrap()
        .written()
        .complete_replacing(|| Ok(2000))
        .unwrap();
    assert!(mgr.given_up_on(reader).is_some(), "expired");
    assert_eq!(
        mgr.read_horizon(),
        at,
        "the read in progress holds the versions"
    );
    assert_eq!(
        mgr.claim_index_sizes().2,
        1,
        "only the commit after the newer transaction began"
    );
    assert_eq!(mgr.conflict_horizon(), mgr.start_epoch(newer));
    drop(read);
    mgr.abort(reader).unwrap();
    mgr.abort(newer).unwrap();
    assert_eq!(mgr.claim_index_sizes(), (0, 0, 0));
}

/// A fixed-seed generator (xorshift), so a failing sequence replays.
pub(super) struct Seeded(pub(super) u64);

impl Seeded {
    pub(super) fn below(&mut self, bound: usize) -> usize {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        usize::try_from(self.0 % u64::try_from(bound).unwrap()).unwrap()
    }
}

pub(super) const SEEDS: [u64; 4] = [0x9E37_79B9_7F4A_7C15, 7, 42, 2026];
const STEPS: usize = 3_000;

/// The oracle: the cached horizon equals the first epoch with a committable snapshot found by
/// a walk of the ledger, and the oldest start of the open transactions the database has not
/// given up on (`None` with none).
fn horizon_holds(mgr: &TransactionManager, open: &[TransactionId], step: &str) {
    let (cached, walked) = mgr.committable_horizons();
    assert_eq!(
        cached, walked,
        "{step}: the cache against a walk of the ledger"
    );
    let model = open
        .iter()
        .filter(|transaction| mgr.given_up_on(**transaction).is_none())
        .filter_map(|transaction| mgr.start_epoch(*transaction))
        .map(|epoch| epoch.as_u64())
        .min();
    assert_eq!(
        cached, model,
        "{step}: the cache against the open transactions"
    );
}

/// Seeded random sequences of committable begins, read pins (`pin_read`, `pin_read_at`), reads
/// in a transaction, publishes, aborts, expiries and dooms: after every step the cached
/// conflict horizon is the one a walk of the ledger finds, and `None` with nothing committable
/// open.
#[test]
fn the_cached_horizon_is_the_first_committable_epoch_after_every_step() {
    for seed in SEEDS {
        let mgr = TransactionManager::new();
        mgr.set_history_retention(HistoryRetention::Forever);
        mgr.set_snapshot_retention_limit(1000);
        let mut random = Seeded(seed);
        let mut open: Vec<TransactionId> = Vec::new();
        let mut pins: Vec<ReadPin> = Vec::new();
        let mut reads: Vec<SnapshotRead> = Vec::new();
        let mut keys = 0_i64;
        horizon_holds(&mgr, &open, "empty");
        for step in 0..STEPS {
            let pick = |random: &mut Seeded, open: &[TransactionId]| open[random.below(open.len())];
            let action = random.below(11);
            match action {
                0 | 1 => open.push(mgr.begin()),
                2 => pins.push(mgr.pin_read()),
                3 => {
                    let current = usize::try_from(mgr.current_epoch().as_u64()).unwrap();
                    let at = u64::try_from(random.below(current + 1)).unwrap();
                    pins.push(mgr.pin_read_at(EpochId::new(at)).unwrap());
                }
                4 if !pins.is_empty() => drop(pins.swap_remove(random.below(pins.len()))),
                5 if !open.is_empty() => {
                    let transaction = pick(&mut random, &open);
                    let node = NodeId::new(u64::try_from(random.below(64)).unwrap());
                    let _ = mgr.record_write(transaction, node);
                    if mgr.commit(transaction).is_err() {
                        mgr.abort(transaction).unwrap();
                    }
                    open.retain(|other| *other != transaction);
                }
                6 if !open.is_empty() => {
                    let transaction = pick(&mut random, &open);
                    mgr.abort(transaction).unwrap();
                    open.retain(|other| *other != transaction);
                }
                7 => {
                    // A commit that replaced past the limit: the oldest snapshots expire.
                    let writer = mgr.begin();
                    mgr.start_commit(writer)
                        .unwrap()
                        .written()
                        .complete_replacing(|| Ok(600))
                        .unwrap();
                }
                8 if open.len() >= 2 => {
                    // The older of two takes a key the younger holds: the younger is doomed.
                    let (first, second) = (pick(&mut random, &open), pick(&mut random, &open));
                    let (older, younger) = (first.min(second), first.max(second));
                    keys += 1;
                    let key = KeyClaim {
                        graph: None,
                        label: Arc::from("Person"),
                        properties: vec!["email".to_string()],
                        values: vec![HashableValue(grafeo_common::types::Value::Int64(keys))],
                    };
                    let _ = mgr.record_key(younger, key.clone());
                    let _ = mgr.record_key(older, key);
                }
                9 if !open.is_empty() => {
                    let transaction = pick(&mut random, &open);
                    if let Ok(Some(read)) = mgr.read_in(transaction) {
                        reads.push(read);
                    }
                }
                10 if !reads.is_empty() => drop(reads.swap_remove(random.below(reads.len()))),
                _ => {}
            }
            horizon_holds(
                &mgr,
                &open,
                &format!("seed {seed}, step {step}, action {action}"),
            );
        }
        drop(reads);
        drop(pins);
        for transaction in open.drain(..) {
            mgr.abort(transaction).unwrap();
        }
        horizon_holds(&mgr, &open, "all ended");
        assert_eq!(mgr.committable_horizons(), (None, None));
        assert_eq!(mgr.claim_index_sizes(), (0, 0, 0));
    }
}

/// The ledger alone, with committable snapshots counted at older epochs too (no begin does that
/// today: a begin is at the published epoch): after every step the cached horizon is the oldest
/// committable epoch, whatever the order.
#[test]
fn the_ledger_keeps_the_oldest_committable_epoch_in_any_order() {
    for seed in SEEDS {
        let mut random = Seeded(seed);
        let mut readers = Readers::default();
        let mut pins: Vec<u64> = Vec::new();
        let mut committable: Vec<u64> = Vec::new();
        for step in 0..STEPS {
            let epoch = u64::try_from(random.below(50)).unwrap();
            match random.below(4) {
                0 => {
                    ReadPins::hold_at(&mut readers, epoch);
                    pins.push(epoch);
                }
                1 => {
                    ReadPins::hold_at(&mut readers, epoch);
                    let index = readers
                        .pinned
                        .binary_search_by_key(&epoch, |pinned| pinned.epoch)
                        .unwrap();
                    ReadPins::count_committable(&mut readers, index);
                    committable.push(epoch);
                }
                2 if !committable.is_empty() => {
                    let epoch = committable.swap_remove(random.below(committable.len()));
                    ReadPins::uncount_committable(&mut readers, epoch);
                    ReadPins::release_in(&mut readers, EpochId::new(epoch));
                }
                3 if !pins.is_empty() => {
                    let epoch = pins.swap_remove(random.below(pins.len()));
                    ReadPins::release_in(&mut readers, EpochId::new(epoch));
                }
                _ => {}
            }
            let case = format!("seed {seed}, step {step}");
            assert_eq!(
                readers.committable_from,
                Readers::first_committable(&readers),
                "{case}"
            );
            assert_eq!(
                readers.committable_from,
                committable.iter().copied().min(),
                "{case}"
            );
        }
    }
}
