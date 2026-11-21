//! #673 part A: the claim indexes. Commit work is a constant number of lookups per claim the
//! committing transaction holds, whatever else is open and however long it has been open:
//!
//! - every transaction leaves the registry when it is published or aborted (A3);
//! - the committed-claims index holds only the commits an open, still-committable transaction
//!   began before, and trims when the oldest such transaction ends (A4);
//! - the open-claims index exists only while two transactions are open at once (A2), and
//!   conflict outcomes stay as they are (A2 of the decided behavior), at record time and at
//!   commit, after thousands of unrelated commits.
//!
//! The accessors read under `#[cfg(test)]`: `registered()` (the open transactions in the
//! registry), `claim_index_sizes()` (entries of the open index with its keys, entries of the
//! committed index with its keys, log entries) and `conflict_horizon()` (the oldest
//! still-committable start).
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --lib transaction::manager::claim_index_tests::
//! ```

use super::*;

/// A claim kind, as a session records it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Kind {
    Write,
    Delete,
    Endpoint,
    Key,
    Pattern,
}

const KINDS: [Kind; 5] = [
    Kind::Write,
    Kind::Delete,
    Kind::Endpoint,
    Kind::Key,
    Kind::Pattern,
];

/// Whether a claim of `first` and one of `second` on one entity (or one key value) conflict,
/// in either order: every write conflicts with every write (a delete is a write), an endpoint
/// conflicts with a delete only, a key with the same key, a MERGE pattern with the same pattern.
fn conflicts(first: Kind, second: Kind) -> bool {
    use Kind::{Delete, Endpoint, Key, Pattern, Write};
    matches!(
        (first, second),
        (Write | Delete, Write | Delete)
            | (Delete, Endpoint)
            | (Endpoint, Delete)
            | (Key, Key)
            | (Pattern, Pattern)
    )
}

fn key(value: i64) -> KeyClaim {
    KeyClaim {
        graph: None,
        label: Arc::from("Person"),
        properties: vec!["email".to_string()],
        values: vec![HashableValue(grafeo_common::types::Value::Int64(value))],
    }
}

/// Records a claim of `kind` on node `id` (or on key value `id`, or MERGE pattern fingerprint
/// `id`) for `tx`.
fn claim(mgr: &TransactionManager, tx: TransactionId, kind: Kind, id: u64) -> Result<()> {
    match kind {
        Kind::Write => mgr.record_write(tx, NodeId::new(id)),
        Kind::Delete => mgr.record_delete(tx, NodeId::new(id)),
        Kind::Endpoint => {
            // The edge's other end is a node of its own, which nothing else claims.
            mgr.record_endpoints(
                tx,
                [
                    GraphEntity::from(NodeId::new(id)),
                    GraphEntity::from(NodeId::new(id + 1_000_000)),
                ],
            )
        }
        Kind::Key => mgr.record_key(tx, key(i64::try_from(id).unwrap())),
        Kind::Pattern => mgr.record_pattern(tx, id, &"(:Person {name: 'Alix'})"),
    }
}

fn is_conflict(result: &Result<impl std::fmt::Debug>) -> bool {
    matches!(result, Err(error) if error.error_code() == grafeo_common::utils::error::ErrorCode::TransactionConflict)
}

/// `count` transactions that each write node `from + i` alone and commit.
fn unrelated_commits(mgr: &TransactionManager, from: u64, count: u64) {
    for i in 0..count {
        let tx = mgr.begin();
        mgr.record_write(tx, NodeId::new(from + i)).unwrap();
        mgr.commit(tx).unwrap();
    }
}

/// A3: every transaction, private or not, leaves the registry when it is published or
/// aborted, whatever else is open; `state` then knows nothing of it.
#[test]
fn every_transaction_leaves_the_registry_when_it_ends() {
    let mgr = TransactionManager::new();
    let reader = mgr.begin();
    assert_eq!(mgr.registered(), 1);

    let writer = mgr.begin();
    mgr.record_write(writer, NodeId::new(1)).unwrap();
    assert_eq!(mgr.registered(), 2);
    mgr.commit(writer).unwrap();
    assert_eq!(mgr.registered(), 1, "published, with a reader open");
    assert_eq!(mgr.state(writer), None);

    let aborted = mgr.begin();
    mgr.record_write(aborted, NodeId::new(2)).unwrap();
    mgr.abort(aborted).unwrap();
    assert_eq!(mgr.registered(), 1, "aborted");
    assert_eq!(mgr.state(aborted), None);

    let (private, _changes) = mgr.begin_private().unwrap();
    mgr.record_write(private, NodeId::new(3)).unwrap();
    assert_eq!(mgr.registered(), 2);
    mgr.commit(private).unwrap();
    assert_eq!(
        mgr.registered(),
        1,
        "a private transaction, with a reader open"
    );
    assert_eq!(mgr.state(private), None);

    assert_eq!(mgr.state(reader), Some(TransactionState::Active));
    mgr.commit(reader).unwrap();
    assert_eq!(mgr.registered(), 0);
    assert_eq!(mgr.state(reader), None);
}

/// A3, at scale: 5,000 commits and 500 aborts while a reader stays open leave one entry in the
/// registry, and `gc` finds nothing publish and abort left behind.
#[test]
fn the_registry_does_not_grow_with_commits_behind_a_reader() {
    let mgr = TransactionManager::new();
    let reader = mgr.begin();
    unrelated_commits(&mgr, 1, 5_000);
    for i in 0..500 {
        let tx = mgr.begin();
        mgr.record_write(tx, NodeId::new(100_000 + i)).unwrap();
        mgr.abort(tx).unwrap();
    }
    assert_eq!(mgr.registered(), 1);
    assert_eq!(mgr.gc(), 0, "publish and abort removed them already");
    mgr.abort(reader).unwrap();
    assert_eq!(mgr.registered(), 0);
    assert_eq!(mgr.gc(), 0);
}

/// A4: the committed index keeps one entry per entity, whatever the number of commits that
/// wrote it, plus a log entry per commit behind the oldest open start; it trims to the next
/// start when the oldest transaction ends, and is empty with none open.
#[test]
fn the_committed_index_holds_only_what_an_open_start_needs() {
    let mgr = TransactionManager::new();
    let older = mgr.begin();
    let older_start = mgr.start_epoch(older).unwrap();
    assert_eq!(mgr.conflict_horizon(), Some(older_start));

    // 300 commits of one node and 200 of nodes of their own, behind `older`.
    for _ in 0..300 {
        let tx = mgr.begin();
        mgr.record_write(tx, NodeId::new(7)).unwrap();
        mgr.commit(tx).unwrap();
    }
    unrelated_commits(&mgr, 1_000, 200);
    let (open, committed, log) = mgr.claim_index_sizes();
    assert_eq!(committed, 201, "one entry per entity");
    assert_eq!(log, 500, "one log entry per commit behind the oldest start");
    assert_eq!(open, 0, "a lone open reader that claimed nothing");

    let newer = mgr.begin();
    let newer_start = mgr.start_epoch(newer).unwrap();
    unrelated_commits(&mgr, 2_000, 50);
    assert_eq!(mgr.claim_index_sizes().2, 550);

    mgr.abort(older).unwrap();
    assert_eq!(mgr.conflict_horizon(), Some(newer_start));
    let (_, committed, log) = mgr.claim_index_sizes();
    assert_eq!(log, 50, "trimmed to the commits behind the newer start");
    assert_eq!(committed, 50);

    mgr.abort(newer).unwrap();
    assert_eq!(mgr.claim_index_sizes(), (0, 0, 0));
    assert_eq!(mgr.conflict_horizon(), None);
}

/// A lone transaction indexes nothing; the begin that makes two open indexes the one already
/// open (its claims, made while alone, included); the index clears when none is open.
#[test]
fn the_open_index_exists_only_while_two_transactions_are_open() {
    let mgr = TransactionManager::new();
    let first = mgr.begin();
    claim(&mgr, first, Kind::Write, 1).unwrap();
    claim(&mgr, first, Kind::Delete, 2).unwrap();
    claim(&mgr, first, Kind::Endpoint, 3).unwrap();
    claim(&mgr, first, Kind::Key, 4).unwrap();
    assert_eq!(mgr.claim_index_sizes().0, 0, "alone, nothing indexed");

    let second = mgr.begin();
    // Nodes 1, 2, 3, the endpoint's far end and one key value.
    assert_eq!(
        mgr.claim_index_sizes().0,
        5,
        "indexed when the second began"
    );
    claim(&mgr, second, Kind::Write, 10).unwrap();
    assert_eq!(mgr.claim_index_sizes().0, 6);
    mgr.commit(second).unwrap();
    assert_eq!(
        mgr.claim_index_sizes().0,
        5,
        "its own claims leave at publish"
    );

    mgr.abort(first).unwrap();
    assert_eq!(mgr.claim_index_sizes(), (0, 0, 0));

    let alone = mgr.begin();
    claim(&mgr, alone, Kind::Write, 1).unwrap();
    assert_eq!(
        mgr.claim_index_sizes().0,
        0,
        "cleared: a later lone writer indexes nothing"
    );
    mgr.commit(alone).unwrap();
}

/// First writer wins, for every pair of claim kinds, whether the first transaction claimed
/// while it was alone (indexed when the second began) or after the second began.
#[test]
fn first_writer_wins_for_every_pair_of_claim_kinds() {
    for first_kind in KINDS {
        for second_kind in KINDS {
            for alone in [true, false] {
                let mgr = TransactionManager::new();
                let first = mgr.begin();
                if alone {
                    claim(&mgr, first, first_kind, 5).unwrap();
                }
                let second = mgr.begin();
                if !alone {
                    claim(&mgr, first, first_kind, 5).unwrap();
                }
                let case = format!("{first_kind:?} then {second_kind:?}, alone: {alone}");
                let result = claim(&mgr, second, second_kind, 5);
                assert_eq!(
                    is_conflict(&result),
                    conflicts(first_kind, second_kind),
                    "{case}: {result:?}"
                );
                // Another entity is free in every case.
                claim(&mgr, second, second_kind, 6).unwrap();
                mgr.abort(first).unwrap();
                // With the first gone, the claim goes through.
                claim(&mgr, second, second_kind, 5).unwrap();
                mgr.commit(second).unwrap();
                assert_eq!(mgr.claim_index_sizes(), (0, 0, 0), "{case}");
            }
        }
    }
}

/// At commit, a transaction fails with `GRAFEO-T001` against a commit made after it began that
/// claimed the same entity in a conflicting way, for every pair of kinds, with 2,000 unrelated
/// commits before or after that commit; on another entity it commits.
#[test]
fn commit_conflicts_survive_thousands_of_unrelated_commits() {
    for committed_kind in KINDS {
        for own_kind in KINDS {
            for conflicting_last in [false, true] {
                let mgr = TransactionManager::new();
                let reader = mgr.begin();
                let bystander = mgr.begin();
                if conflicting_last {
                    unrelated_commits(&mgr, 10_000, 2_000);
                }
                let earlier = mgr.begin();
                claim(&mgr, earlier, committed_kind, 5).unwrap();
                mgr.commit(earlier).unwrap();
                if !conflicting_last {
                    unrelated_commits(&mgr, 10_000, 2_000);
                }
                let case = format!(
                    "committed {committed_kind:?}, own {own_kind:?}, conflicting commit last: \
                     {conflicting_last}"
                );
                claim(&mgr, reader, own_kind, 5).unwrap();
                let result = mgr.commit(reader);
                assert_eq!(
                    is_conflict(&result),
                    conflicts(committed_kind, own_kind),
                    "{case}: {result:?}"
                );
                if result.is_err() {
                    // The caller aborts a commit that lost its conflict, as a session does.
                    mgr.abort(reader).unwrap();
                    assert_eq!(mgr.state(reader), None, "{case}");
                }
                claim(&mgr, bystander, own_kind, 6).unwrap();
                mgr.commit(bystander).unwrap();
                assert_eq!(mgr.registered(), 0, "{case}");
                assert_eq!(mgr.claim_index_sizes(), (0, 0, 0), "{case}");
            }
        }
    }
}

/// The conflict horizon is exact: a commit published before a transaction began never
/// conflicts with it, one published after does, and the trim at the end of an older
/// transaction keeps what a newer one still needs.
#[test]
fn the_trim_keeps_exactly_the_commits_after_each_open_start() {
    let mgr = TransactionManager::new();
    let older = mgr.begin();
    let before_newer = mgr.begin();
    mgr.record_write(before_newer, NodeId::new(1)).unwrap();
    mgr.commit(before_newer).unwrap();
    // `newer` begins at the epoch the commit above published.
    let newer = mgr.begin();
    let after_newer = mgr.begin();
    mgr.record_write(after_newer, NodeId::new(2)).unwrap();
    mgr.commit(after_newer).unwrap();

    mgr.abort(older).unwrap();
    mgr.record_write(newer, NodeId::new(1)).unwrap();
    mgr.record_write(newer, NodeId::new(2)).unwrap();
    assert!(
        is_conflict(&mgr.commit(newer)),
        "node 2 was committed after `newer` began"
    );
    mgr.abort(newer).unwrap();

    let fresh = mgr.begin();
    mgr.record_write(fresh, NodeId::new(1)).unwrap();
    mgr.record_write(fresh, NodeId::new(2)).unwrap();
    mgr.commit(fresh)
        .expect("both commits were published before it began");
}

/// A savepoint restore takes the claims after the mark out of the open index: another open
/// transaction can claim them, and the index is back to its size at the mark.
#[test]
fn a_savepoint_restore_unindexes_the_claims_after_the_mark() {
    let mgr = TransactionManager::new();
    let first = mgr.begin();
    let second = mgr.begin();
    claim(&mgr, first, Kind::Write, 1).unwrap();
    let at_mark = mgr.claim_index_sizes().0;
    let mark = mgr.mark_claims(first).unwrap();
    for kind in KINDS {
        claim(&mgr, first, kind, 20).unwrap();
    }
    assert!(mgr.claim_index_sizes().0 > at_mark);
    mgr.restore_claims(first, mark).unwrap();
    assert_eq!(mgr.claim_index_sizes().0, at_mark);
    for kind in KINDS {
        claim(&mgr, second, kind, 20).unwrap();
    }
    assert!(is_conflict(&claim(&mgr, second, Kind::Write, 1)));
    mgr.abort(first).unwrap();
    mgr.commit(second).unwrap();
}

/// A transaction that begins between another's validation and its publish, while that one is
/// the only other open transaction, still loses to it at commit: the publish keeps the
/// writer's claims for it, since its snapshot is below the writer's epoch.
#[test]
fn a_begin_during_a_commit_in_progress_loses_to_it_after_the_publish() {
    let mgr = TransactionManager::new();
    let writer = mgr.begin();
    mgr.record_write(writer, NodeId::new(1)).unwrap();
    let publish = mgr.start_commit(writer).unwrap().written();
    let after = mgr.begin();
    assert_eq!(mgr.registered(), 2);
    assert!(mgr.start_epoch(after).unwrap() < publish.epoch());
    publish.complete().unwrap();
    assert_eq!(mgr.registered(), 1, "only `after` is open");

    mgr.record_write(after, NodeId::new(1))
        .expect("the writer is published: it holds no open claim");
    let result = mgr.commit(after);
    assert!(is_conflict(&result), "{result:?}");
    mgr.abort(after).unwrap();
    assert_eq!(mgr.claim_index_sizes(), (0, 0, 0));
}

/// The outcome of a claim or a commit.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Outcome {
    Ok,
    /// `GRAFEO-T001`.
    Conflict,
    /// Serializable isolation's read-write conflict.
    Serialization,
}

fn outcome<T>(result: &Result<T>) -> Outcome {
    use grafeo_common::utils::error::ErrorCode;
    match result {
        Ok(_) => Outcome::Ok,
        Err(error) => match error.error_code() {
            ErrorCode::TransactionConflict => Outcome::Conflict,
            ErrorCode::TransactionSerialization => Outcome::Serialization,
            _ => panic!("not a conflict: {error}"),
        },
    }
}

/// What the model knows of one transaction: its snapshot, its claims and its commit epoch.
#[derive(Default)]
struct Modeled {
    start: u64,
    serializable: bool,
    /// It lost a key or MERGE pattern claim to an older transaction.
    doomed: bool,
    /// Its commit epoch, from the start of its commit.
    epoch: Option<u64>,
    /// Its commit was taken back: active again with its claims, for the caller to roll back.
    taken_back: bool,
    /// The nodes it wrote, deletes included.
    writes: FxHashSet<u64>,
    deletes: FxHashSet<u64>,
    endpoints: FxHashSet<u64>,
    keys: FxHashSet<u64>,
    patterns: FxHashSet<u64>,
    reads: FxHashSet<u64>,
    /// The last write of each triple, by (in the named graph, subject): whether it is there.
    triples: FxHashMap<(bool, u64), bool>,
}

impl Modeled {
    fn exclusive(&mut self, kind: Kind) -> &mut FxHashSet<u64> {
        if kind == Kind::Key {
            &mut self.keys
        } else {
            &mut self.patterns
        }
    }
}

/// A brute-force model of conflict detection, as the scans before the claim indexes decided
/// it: every open transaction and every published commit, kept whole and walked per check.
#[derive(Default)]
struct Model {
    /// The active and committing transactions.
    open: Vec<(TransactionId, Modeled)>,
    /// Every published commit.
    published: Vec<Modeled>,
    published_epoch: u64,
    assigned_epoch: u64,
}

impl Model {
    fn get(&mut self, transaction: TransactionId) -> &mut Modeled {
        &mut self
            .open
            .iter_mut()
            .find(|(id, _)| *id == transaction)
            .unwrap()
            .1
    }

    fn remove(&mut self, transaction: TransactionId) -> Modeled {
        let index = self
            .open
            .iter()
            .position(|(id, _)| *id == transaction)
            .unwrap();
        self.open.remove(index).1
    }

    /// A write, delete or endpoint claim: first writer wins against every other open
    /// transaction, active or committing, doomed or not.
    fn entity_claim(&mut self, transaction: TransactionId, kind: Kind, nodes: &[u64]) -> Outcome {
        if self.get(transaction).doomed {
            return Outcome::Conflict;
        }
        let held = self
            .open
            .iter()
            .filter(|(id, _)| *id != transaction)
            .any(|(_, other)| {
                nodes.iter().any(|node| match kind {
                    Kind::Write => other.writes.contains(node),
                    Kind::Delete => other.writes.contains(node) || other.endpoints.contains(node),
                    Kind::Endpoint => other.deletes.contains(node),
                    Kind::Key | Kind::Pattern => unreachable!("not an entity claim"),
                })
            });
        if held {
            return Outcome::Conflict;
        }
        let ours = self.get(transaction);
        for node in nodes {
            match kind {
                Kind::Write => {
                    ours.writes.insert(*node);
                }
                Kind::Delete => {
                    ours.writes.insert(*node);
                    ours.deletes.insert(*node);
                }
                _ => {
                    ours.endpoints.insert(*node);
                }
            }
        }
        Outcome::Ok
    }

    /// A key or MERGE pattern claim: oldest claim wins. A holder that is committing or began
    /// first keeps it; every younger holder is doomed. A doomed holder's claims do not count.
    fn exclusive_claim(&mut self, transaction: TransactionId, kind: Kind, value: u64) -> Outcome {
        if self.get(transaction).doomed {
            return Outcome::Conflict;
        }
        if self.open.len() > 1 {
            let holders: Vec<(TransactionId, bool)> = self
                .open
                .iter_mut()
                .filter(|(id, other)| *id != transaction && !other.doomed)
                .filter_map(|(id, other)| {
                    let committing = other.epoch.is_some();
                    other
                        .exclusive(kind)
                        .contains(&value)
                        .then_some((*id, committing))
                })
                .collect();
            if holders
                .iter()
                .any(|(holder, committing)| *committing || *holder < transaction)
            {
                return Outcome::Conflict;
            }
            for (holder, _) in holders {
                self.get(holder).doomed = true;
            }
        }
        self.get(transaction).exclusive(kind).insert(value);
        Outcome::Ok
    }

    /// The commit of `transaction`: it loses to any commit, published or in progress, after
    /// its start that wrote what it wrote, deleted a node it links to, linked to a node it
    /// deletes, claimed one of its keys or MERGE patterns, or wrote one of its triples where
    /// either deletes it; then, serializable, to one that wrote what it read.
    fn commit(&self, transaction: TransactionId) -> Outcome {
        let ours = &self
            .open
            .iter()
            .find(|(id, _)| *id == transaction)
            .unwrap()
            .1;
        if ours.doomed {
            return Outcome::Conflict;
        }
        let after: Vec<&Modeled> = self
            .published
            .iter()
            .chain(
                self.open
                    .iter()
                    .filter(|(id, _)| *id != transaction)
                    .map(|(_, other)| other),
            )
            .filter(|other| other.epoch.is_some_and(|epoch| epoch > ours.start))
            .collect();
        let lost = |theirs: &&Modeled| {
            !ours.writes.is_disjoint(&theirs.writes)
                || !ours.endpoints.is_disjoint(&theirs.deletes)
                || !ours.deletes.is_disjoint(&theirs.endpoints)
                || !ours.keys.is_disjoint(&theirs.keys)
                || !ours.patterns.is_disjoint(&theirs.patterns)
                || ours.triples.iter().any(|(triple, present)| {
                    theirs
                        .triples
                        .get(triple)
                        .is_some_and(|there| !(*present && *there))
                })
        };
        if after.iter().any(lost) {
            Outcome::Conflict
        } else if ours.serializable
            && after
                .iter()
                .any(|theirs| !ours.reads.is_disjoint(&theirs.writes))
        {
            Outcome::Serialization
        } else {
            Outcome::Ok
        }
    }
}

/// The triple `<name> knows <mia>`.
#[cfg(feature = "triple-store")]
fn triple(name: &str) -> grafeo_core::graph::rdf::Triple {
    use grafeo_core::graph::rdf::{Term, Triple};

    Triple::new(
        Term::iri(format!("http://example.org/{name}")),
        Term::iri("http://example.org/knows"),
        Term::iri("http://example.org/mia"),
    )
}

/// The oracle of the claim indexes: seeded random interleavings of begins (some
/// serializable), claims of every kind (write, delete, edge endpoints, key, MERGE pattern,
/// serializable read, RDF triple in the default and a named graph), commits started and taken
/// back, commits written and published later in epoch order, and aborts. Every claim's and
/// every commit's outcome is the brute-force model's: at a claim, an open or committing holder
/// conflicts; at a commit, a commit published after the transaction's start, or one in
/// progress, does.
#[test]
fn conflict_outcomes_match_a_brute_force_model() {
    use super::claim_trim_tests::{SEEDS, Seeded};

    const STEPS: usize = 10_000;
    const NODES: usize = 16;
    const VALUES: usize = 4;
    const MAX_OPEN: usize = 5;

    for seed in SEEDS {
        let mgr = TransactionManager::new();
        let mut random = Seeded(seed);
        let mut model = Model::default();
        let mut in_progress: VecDeque<(TransactionId, Publish<'_>)> = VecDeque::new();
        // How many claims and commits came out ok, in conflict, in a serialization failure.
        let mut outcomes = [0_usize; 3];
        for step in 0..STEPS {
            let active: Vec<TransactionId> = model
                .open
                .iter()
                .filter(|(_, modeled)| modeled.epoch.is_none() && !modeled.taken_back)
                .map(|(id, _)| *id)
                .collect();
            let abortable: Vec<TransactionId> = model
                .open
                .iter()
                .filter(|(_, modeled)| modeled.epoch.is_none())
                .map(|(id, _)| *id)
                .collect();
            let action = random.below(15);
            let case = format!("seed {seed}, step {step}, action {action}");
            let node = |random: &mut Seeded| u64::try_from(random.below(NODES)).unwrap();
            let value = |random: &mut Seeded| u64::try_from(random.below(VALUES)).unwrap();
            let mut check = |expected: Outcome, got: Outcome, what: &str| {
                assert_eq!(got, expected, "{case}: {what}");
                outcomes[got as usize] += 1;
            };
            match action {
                0 | 1 if model.open.len() < MAX_OPEN => {
                    let serializable = random.below(2) == 0;
                    let transaction = if serializable {
                        mgr.begin_with_isolation(IsolationLevel::Serializable)
                    } else {
                        mgr.begin()
                    };
                    assert_eq!(
                        mgr.start_epoch(transaction).map(|epoch| epoch.as_u64()),
                        Some(model.published_epoch),
                        "{case}: a begin is at the published epoch"
                    );
                    let start = model.published_epoch;
                    model.open.push((
                        transaction,
                        Modeled {
                            start,
                            serializable,
                            ..Modeled::default()
                        },
                    ));
                }
                2..=9 | 14 if !active.is_empty() => {
                    let transaction = active[random.below(active.len())];
                    match action {
                        2 | 3 => {
                            let node = node(&mut random);
                            let got = outcome(&mgr.record_write(transaction, NodeId::new(node)));
                            let expected = model.entity_claim(transaction, Kind::Write, &[node]);
                            check(expected, got, &format!("write {node}"));
                        }
                        4 => {
                            let node = node(&mut random);
                            let got = outcome(&mgr.record_delete(transaction, NodeId::new(node)));
                            let expected = model.entity_claim(transaction, Kind::Delete, &[node]);
                            check(expected, got, &format!("delete {node}"));
                        }
                        5 => {
                            let ends = [node(&mut random), node(&mut random)];
                            let got = outcome(&mgr.record_endpoints(
                                transaction,
                                ends.map(|end| GraphEntity::from(NodeId::new(end))),
                            ));
                            let expected = model.entity_claim(transaction, Kind::Endpoint, &ends);
                            check(expected, got, &format!("endpoints {ends:?}"));
                        }
                        6 => {
                            let value = value(&mut random);
                            let got = outcome(
                                &mgr.record_key(transaction, key(i64::try_from(value).unwrap())),
                            );
                            let expected = model.exclusive_claim(transaction, Kind::Key, value);
                            check(expected, got, &format!("key {value}"));
                        }
                        7 => {
                            let value = value(&mut random);
                            let got = outcome(&mgr.record_pattern(
                                transaction,
                                value,
                                &"(:Person {name: 'Mia'})",
                            ));
                            let expected = model.exclusive_claim(transaction, Kind::Pattern, value);
                            check(expected, got, &format!("pattern {value}"));
                        }
                        8 | 14 => {
                            let node = node(&mut random);
                            mgr.record_read(transaction, NodeId::new(node)).unwrap();
                            model.get(transaction).reads.insert(node);
                        }
                        _ => {
                            #[cfg(feature = "triple-store")]
                            {
                                let named = random.below(2) == 1;
                                let subject = u64::try_from(random.below(4)).unwrap();
                                let insert = random.below(2) == 1;
                                mgr.changes(transaction)
                                    .unwrap()
                                    .record_triple(
                                        named.then_some("http://example.org/amsterdam"),
                                        triple(
                                            ["alix", "gus", "vincent", "jules"]
                                                [usize::try_from(subject).unwrap()],
                                        ),
                                        insert,
                                    )
                                    .unwrap();
                                model
                                    .get(transaction)
                                    .triples
                                    .insert((named, subject), insert);
                            }
                        }
                    }
                }
                10 | 11 if !active.is_empty() => {
                    let transaction = active[random.below(active.len())];
                    let expected = model.commit(transaction);
                    let result = mgr.start_commit(transaction);
                    check(expected, outcome(&result), "commit");
                    match result {
                        Err(_) => {
                            // The caller aborts a commit that lost, as a session does.
                            mgr.abort(transaction).unwrap();
                            model.remove(transaction);
                        }
                        Ok(guard) => {
                            model.assigned_epoch += 1;
                            assert_eq!(guard.epoch().as_u64(), model.assigned_epoch, "{case}");
                            if random.below(5) == 0 {
                                // Taken back (its log write failed): active again with its
                                // claims, which count until the caller rolls it back.
                                guard.abort();
                                model.assigned_epoch -= 1;
                                model.get(transaction).taken_back = true;
                            } else {
                                model.get(transaction).epoch = Some(model.assigned_epoch);
                                in_progress.push_back((transaction, guard.written()));
                            }
                        }
                    }
                }
                12 => {
                    if let Some((transaction, publish)) = in_progress.pop_front() {
                        publish.complete().unwrap();
                        let committed = model.remove(transaction);
                        model.published_epoch = committed.epoch.unwrap();
                        model.published.push(committed);
                        assert_eq!(
                            mgr.current_epoch().as_u64(),
                            model.published_epoch,
                            "{case}"
                        );
                    }
                }
                13 if !abortable.is_empty() => {
                    let transaction = abortable[random.below(abortable.len())];
                    mgr.abort(transaction).unwrap();
                    model.remove(transaction);
                }
                _ => {}
            }
            assert_eq!(mgr.registered(), model.open.len(), "{case}");
        }
        for (_, publish) in in_progress.drain(..) {
            publish.complete().unwrap();
        }
        for (transaction, modeled) in model.open.drain(..) {
            if modeled.epoch.is_none() {
                mgr.abort(transaction).unwrap();
            }
        }
        assert_eq!(mgr.registered(), 0, "seed {seed}");
        assert_eq!(mgr.claim_index_sizes(), (0, 0, 0), "seed {seed}");
        // Each outcome came up, so the run reached every branch of the model.
        assert!(
            outcomes.iter().all(|count| *count > 0),
            "seed {seed}: outcomes (ok, conflict, serialization) {outcomes:?}"
        );
    }
}

#[cfg(feature = "triple-store")]
const AMSTERDAM: &str = "http://example.org/amsterdam";

/// Records `<name> knows <mia>` in `graph` for `tx`: an insert, or a delete.
#[cfg(feature = "triple-store")]
fn record_triple(
    mgr: &TransactionManager,
    tx: TransactionId,
    graph: Option<&str>,
    name: &str,
    insert: bool,
) {
    mgr.changes(tx)
        .unwrap()
        .record_triple(graph, triple(name), insert)
        .unwrap();
}

/// Writes `<name> knows <mia>` in `graph` in a transaction of its own, and commits it.
#[cfg(feature = "triple-store")]
fn commit_triple(mgr: &TransactionManager, graph: Option<&str>, name: &str, insert: bool) {
    let tx = mgr.begin();
    record_triple(mgr, tx, graph, name, insert);
    mgr.start_commit(tx).unwrap().complete().unwrap();
}

/// RDF, in a named graph: two writes of one triple conflict at commit where either deletes it
/// (our insert against a later commit's delete included), with 2,000 unrelated commits in the
/// same graph before or after the conflicting one; the triple in another graph is free.
#[cfg(feature = "triple-store")]
#[test]
fn named_graph_triple_conflicts_survive_thousands_of_unrelated_commits() {
    for theirs in [false, true] {
        for ours in [false, true] {
            for conflicting_last in [false, true] {
                let mgr = TransactionManager::new();
                let reader = mgr.begin();
                let bystander = mgr.begin();
                let unrelated = || {
                    for i in 0..2_000 {
                        commit_triple(&mgr, Some(AMSTERDAM), &format!("person{i}"), true);
                    }
                };
                if conflicting_last {
                    unrelated();
                }
                commit_triple(&mgr, Some(AMSTERDAM), "alix", theirs);
                if !conflicting_last {
                    unrelated();
                }
                let case = format!(
                    "theirs insert: {theirs}, ours insert: {ours}, conflicting commit last: \
                     {conflicting_last}"
                );
                record_triple(&mgr, reader, Some(AMSTERDAM), "alix", ours);
                let result = mgr.start_commit(reader).and_then(CommitGuard::complete);
                assert_eq!(
                    is_conflict(&result),
                    !(theirs && ours),
                    "{case}: {result:?}"
                );
                if result.is_err() {
                    mgr.abort(reader).unwrap();
                }
                // The same triple in the default graph and in another named graph.
                record_triple(&mgr, bystander, None, "alix", ours);
                record_triple(
                    &mgr,
                    bystander,
                    Some("http://example.org/berlin"),
                    "alix",
                    ours,
                );
                mgr.start_commit(bystander)
                    .and_then(CommitGuard::complete)
                    .unwrap();
                assert_eq!(mgr.registered(), 0, "{case}");
                assert_eq!(mgr.claim_index_sizes(), (0, 0, 0), "{case}");
                assert_eq!(mgr.kept_rdf_writes_bytes(), 0, "{case}");
            }
        }
    }
}

/// RDF: the trim drops a named graph's triples with the commits that wrote them, once no open
/// transaction began before those commits, and keeps another graph's that a later commit
/// wrote, which a newer transaction still needs.
#[cfg(feature = "triple-store")]
#[test]
fn the_trim_drops_a_named_graphs_triples_with_the_commits_that_wrote_them() {
    let berlin = Some("http://example.org/berlin");
    let mgr = TransactionManager::new();
    let older = mgr.begin();
    commit_triple(&mgr, Some(AMSTERDAM), "alix", false);
    commit_triple(&mgr, Some(AMSTERDAM), "gus", true);
    let newer = mgr.begin();
    let newer_too = mgr.begin();
    commit_triple(&mgr, berlin, "vincent", false);
    assert_eq!(mgr.claim_index_sizes(), (0, 3, 3));
    let kept = mgr.kept_rdf_writes_bytes();

    mgr.abort(older).unwrap();
    assert_eq!(
        mgr.claim_index_sizes(),
        (0, 1, 1),
        "only Berlin's commit, after the newer transactions began"
    );
    assert!(mgr.kept_rdf_writes_bytes() < kept);

    // Amsterdam's commits were published before the newer transactions began.
    record_triple(&mgr, newer_too, Some(AMSTERDAM), "alix", true);
    record_triple(&mgr, newer_too, Some(AMSTERDAM), "gus", false);
    mgr.start_commit(newer_too)
        .and_then(CommitGuard::complete)
        .unwrap();
    // Berlin's delete was published after: our insert loses to it.
    record_triple(&mgr, newer, berlin, "vincent", true);
    let result = mgr.start_commit(newer).and_then(CommitGuard::complete);
    assert!(is_conflict(&result), "{result:?}");
    mgr.abort(newer).unwrap();
    assert_eq!(mgr.claim_index_sizes(), (0, 0, 0));
    assert_eq!(mgr.kept_rdf_writes_bytes(), 0);
}
