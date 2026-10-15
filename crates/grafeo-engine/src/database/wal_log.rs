//! A commit's group in the write-ahead log (WAL v2), written from what it
//! changed.
//!
//! A commit logs one group: a [`GroupBegin`] that names its epoch, then one
//! record per write of its change set, in the order the writes were applied
//! ([`ChangeSet::log_records`]); a change outside any commit logs its
//! standalone ops the same way. Each record is encoded into one scratch buffer
//! and handed to the group, which packs records into frames as they come, so
//! no list of records or of their bytes is built, however large the set.
//!
//! A record the log refuses (over [`MAX_RECORD_BYTES`]) is refused before
//! its bytes reach the group: the caller drops the group, which cuts the log
//! back to where it started, and nothing of the commit is in the log.
//!
//! [`MAX_RECORD_BYTES`]: grafeo_common::storage::MAX_RECORD_BYTES

use grafeo_common::change::ChangeSet;
use grafeo_common::storage::LogRecordRef;
use grafeo_common::storage::log_record::{GroupBegin, Origin};
use grafeo_common::types::EpochId;
use grafeo_common::utils::error::{Error, Result, TransactionError};
use grafeo_storage::wal::{GroupError, GroupWriter};

use crate::transaction::StandaloneChange;

/// Encodes records into a group one at a time, through one scratch buffer.
struct Encoder<'group, 'wal> {
    group: &'group mut GroupWriter<'wal>,
    /// The bytes of the record being pushed; reused for every record.
    scratch: Vec<u8>,
}

impl<'group, 'wal> Encoder<'group, 'wal> {
    fn new(group: &'group mut GroupWriter<'wal>) -> Self {
        Self {
            group,
            scratch: Vec::new(),
        }
    }

    /// Encodes `record` and pushes it into the group.
    ///
    /// # Errors
    ///
    /// The record's own error when it does not encode (over a limit): nothing
    /// of it reached the group. A commit error (see [`commit_error`]) when the
    /// group failed to write a frame.
    fn push(&mut self, record: &LogRecordRef<'_>) -> Result<()> {
        self.scratch.clear();
        record.encode_framed(&mut self.scratch)?;
        self.group.push(&self.scratch).map_err(commit_error)
    }

    /// The group's first record: the epoch its changes take, when it was
    /// written, and what wrote it.
    fn begin(&mut self, epoch: EpochId, origin: Origin) -> Result<()> {
        self.push(&LogRecordRef::GroupBegin(GroupBegin {
            epoch,
            timestamp_ms: now_ms(),
            origin,
        }))
    }
}

/// Writes the records of a commit at `epoch` into `group`: its
/// [`GroupBegin`], then one record per write of `set`, in the order the
/// writes were applied. The caller finishes the group (its LAST frame); on an
/// error it drops the group, which cuts the log back.
///
/// # Errors
///
/// As [`Encoder::push`]: a record that does not encode, or a commit error
/// when a frame write failed.
pub(crate) fn write_group(
    group: &mut GroupWriter<'_>,
    set: &ChangeSet,
    epoch: EpochId,
    origin: Origin,
) -> Result<()> {
    let mut encoder = Encoder::new(group);
    encoder.begin(epoch, origin)?;
    for record in set.log_records() {
        encoder.push(&record)?;
    }
    Ok(())
}

/// Writes the records of a change outside any commit into `group`: its
/// [`GroupBegin`] at `epoch`, then one record per op, in the order the ops
/// apply.
///
/// # Errors
///
/// As [`write_group`].
pub(crate) fn write_standalone(
    group: &mut GroupWriter<'_>,
    change: &StandaloneChange,
    epoch: EpochId,
    origin: Origin,
) -> Result<()> {
    let mut encoder = Encoder::new(group);
    encoder.begin(epoch, origin)?;
    for op in change.ops() {
        encoder.push(&LogRecordRef::Standalone(op))?;
    }
    Ok(())
}

/// The commit error of a log group that failed: a group that was not written
/// was cut back, so its commit can be rolled back and retried
/// ([`TransactionError::LogWriteFailed`]); a group written but not made
/// durable may or may not survive ([`TransactionError::CommitOutcomeUnknown`]);
/// a log that takes no more writes wrote nothing
/// ([`TransactionError::LogUnavailable`]).
pub(crate) fn commit_error(error: GroupError) -> Error {
    let error = match error {
        GroupError::NotWritten(cause) => TransactionError::LogWriteFailed(cause.to_string()),
        GroupError::OutcomeUnknown(cause) => {
            TransactionError::CommitOutcomeUnknown(cause.to_string())
        }
        GroupError::Unavailable { reason } => TransactionError::LogUnavailable(reason),
        other => TransactionError::LogWriteFailed(other.to_string()),
    };
    Error::Transaction(error)
}

/// Milliseconds since the Unix epoch, for a group's timestamp; 0 for a clock
/// set before it.
fn now_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |since| {
            u64::try_from(since.as_millis()).unwrap_or(u64::MAX)
        })
}

#[cfg(test)]
mod tests {
    use grafeo_common::change::{
        Before, ChangeSet, DataModel, DataOp, EdgeImage, GraphRef, Labels, NodeImage,
        PendingVersion, StandaloneOp,
    };
    use grafeo_common::storage::log_record::{GroupBegin, Origin, TermRecord, TripleRecord};
    use grafeo_common::storage::{LogRecord, read_log_records};
    use grafeo_common::types::{
        ArcStr, EdgeId, EpochId, NodeId, PropertyKey, TransactionId, Value,
    };
    use grafeo_common::utils::error::{Error, TransactionError};
    use grafeo_storage::wal::{
        DurabilityMode, GroupError, ScanOptions, Wal, WalError, WalOptions, WalScan,
    };

    use super::{commit_error, write_group, write_standalone};
    use crate::transaction::StandaloneChange;

    const DATABASE: u128 = 0x6772_6166_656f_0000_0000_0000_0000_0006;

    /// A WAL in `dir` with small frames, so a group spans several.
    fn wal(dir: &std::path::Path) -> Wal {
        Wal::open(
            dir,
            WalOptions {
                durability: DurabilityMode::NoSync,
                frame_target_bytes: 256,
                ..WalOptions::new(DATABASE, 0)
            },
        )
        .unwrap()
    }

    /// The groups of the WAL in `dir`, each as its transaction, its frame
    /// count and its records.
    fn groups(dir: &std::path::Path) -> Vec<(TransactionId, usize, Vec<LogRecord>)> {
        let mut scan = WalScan::open(dir, ScanOptions::new(DATABASE, 0)).unwrap();
        let mut groups = Vec::new();
        while let Some(mut group) = scan.next_group().unwrap() {
            let (transaction, frames) = (group.transaction_id(), group.frame_count());
            let mut records = Vec::new();
            while let Some(payload) = group.next_payload().unwrap() {
                read_log_records(payload, &mut |record| {
                    records.push(record);
                    Ok(())
                })
                .unwrap();
            }
            groups.push((transaction, frames, records));
        }
        scan.finish().unwrap();
        groups
    }

    /// The before-image a write of `op` that changed something has (see
    /// `Before::fits`).
    fn before_of(op: &DataOp) -> Before {
        match op {
            DataOp::SetNodeProperty { .. } | DataOp::SetEdgeProperty { .. } => Before::Value(None),
            DataOp::RemoveNodeProperty { .. } | DataOp::RemoveEdgeProperty { .. } => {
                Before::Value(Some(Value::from("Amsterdam")))
            }
            DataOp::AddNodeLabel { .. } => Before::Labels(Labels::new()),
            DataOp::RemoveNodeLabel { label, .. } => {
                Before::Labels([label.clone()].into_iter().collect())
            }
            DataOp::DeleteNode { .. } => Before::Node(Box::new(NodeImage {
                labels: Labels::new(),
                properties: Vec::new(),
            })),
            DataOp::DeleteEdge { .. } => Before::Edge(Box::new(EdgeImage {
                src: NodeId::new(3),
                dst: NodeId::new(19),
                edge_type: ArcStr::from("KNOWS"),
                properties: Vec::new(),
            })),
            _ => Before::Absent,
        }
    }

    fn triple(object: &str) -> Box<TripleRecord> {
        Box::new(TripleRecord {
            subject: TermRecord::Iri("http://example.org/alix".to_string()),
            predicate: TermRecord::Iri("http://example.org/livesIn".to_string()),
            object: TermRecord::Iri(format!("http://example.org/{object}")),
        })
    }

    /// A set with every kind of op, in the default graph, a named LPG graph
    /// and two RDF graphs, with the ops in the order they were pushed.
    fn every_op() -> (ChangeSet, Vec<(Option<String>, DataOp)>) {
        let mut set = ChangeSet::new();
        let lpg = set
            .slot(GraphRef {
                model: DataModel::Lpg,
                key: None,
            })
            .unwrap();
        let trips = set
            .slot(GraphRef {
                model: DataModel::Lpg,
                key: Some(ArcStr::from("trips")),
            })
            .unwrap();
        let rdf = set
            .slot(GraphRef {
                model: DataModel::Rdf,
                key: None,
            })
            .unwrap();
        let cities = set
            .slot(GraphRef {
                model: DataModel::Rdf,
                key: Some(ArcStr::from("http://example.org/cities")),
            })
            .unwrap();
        let (alix, gus, knows) = (NodeId::new(3), NodeId::new(19), EdgeId::new(88));
        let name = PropertyKey::new("name");
        let ops = [
            (
                lpg,
                None,
                DataOp::CreateNode {
                    id: alix,
                    labels: [ArcStr::from("Person")].into_iter().collect(),
                    properties: vec![(name.clone(), Value::from("Alix"))],
                },
            ),
            (
                trips,
                Some("trips"),
                DataOp::CreateEdge {
                    id: knows,
                    src: alix,
                    dst: gus,
                    edge_type: ArcStr::from("KNOWS"),
                    properties: vec![(PropertyKey::new("since"), Value::Int64(2019))],
                },
            ),
            (
                lpg,
                None,
                DataOp::SetNodeProperty {
                    id: gus,
                    key: PropertyKey::new("city"),
                    value: Value::from("Berlin"),
                },
            ),
            (
                lpg,
                None,
                DataOp::RemoveNodeProperty {
                    id: alix,
                    key: name,
                },
            ),
            (
                trips,
                Some("trips"),
                DataOp::SetEdgeProperty {
                    id: knows,
                    key: PropertyKey::new("weight"),
                    value: Value::Float64(0.88),
                },
            ),
            (
                trips,
                Some("trips"),
                DataOp::RemoveEdgeProperty {
                    id: knows,
                    key: PropertyKey::new("since"),
                },
            ),
            (
                lpg,
                None,
                DataOp::AddNodeLabel {
                    id: gus,
                    label: ArcStr::from("Driver"),
                },
            ),
            (
                lpg,
                None,
                DataOp::RemoveNodeLabel {
                    id: alix,
                    label: ArcStr::from("Person"),
                },
            ),
            (trips, Some("trips"), DataOp::DeleteEdge { id: knows }),
            (lpg, None, DataOp::DeleteNode { id: gus }),
            (
                rdf,
                None,
                DataOp::InsertTriple {
                    triple: triple("amsterdam"),
                },
            ),
            (
                cities,
                Some("http://example.org/cities"),
                DataOp::DeleteTriple {
                    triple: triple("paris"),
                },
            ),
        ];
        let mut expected = Vec::new();
        for (slot, key, op) in ops {
            expected.push((key.map(ToString::to_string), op.clone()));
            let before = before_of(&op);
            set.push(slot, op, before, PendingVersion::Created).unwrap();
        }
        (set, expected)
    }

    /// A commit's group decodes to its GroupBegin and then exactly the ops of
    /// its set, each with its graph, in the order they were applied, through
    /// the scanner the open replays with.
    #[test]
    fn a_written_group_decodes_to_the_set() {
        let dir = tempfile::tempdir().unwrap();
        let (set, expected) = every_op();
        {
            let wal = wal(dir.path());
            let mut group = wal.begin_group(TransactionId::new(7)).unwrap();
            write_group(&mut group, &set, EpochId::new(3), Origin::Transaction).unwrap();
            group.finish().unwrap();
        }

        let groups = groups(dir.path());
        assert_eq!(groups.len(), 1);
        let (transaction, frames, records) = &groups[0];
        assert_eq!(*transaction, TransactionId::new(7));
        assert!(*frames > 1, "small frames: the group spans several");
        let Some((LogRecord::GroupBegin(begin), data)) = records.split_first() else {
            panic!("no records: {records:?}");
        };
        assert!(matches!(
            begin,
            GroupBegin {
                epoch,
                origin: Origin::Transaction,
                timestamp_ms,
            } if *epoch == EpochId::new(3) && *timestamp_ms > 0
        ));
        let decoded: Vec<(Option<String>, DataOp)> = data
            .iter()
            .map(|record| match record {
                LogRecord::Data { graph, op } => (graph.clone(), op.clone()),
                other => panic!("not a data record: {other:?}"),
            })
            .collect();
        assert_eq!(decoded, expected);
    }

    /// A change outside a commit is a group of its standalone ops, in order.
    #[test]
    fn a_standalone_group_decodes_to_its_ops() {
        let dir = tempfile::tempdir().unwrap();
        let ops = [
            StandaloneOp::CreateGraph {
                name: "trips".to_string(),
            },
            StandaloneOp::DropGraph {
                name: "old trips".to_string(),
            },
        ];
        let mut change = StandaloneChange::new();
        for op in &ops {
            change.push(op.clone());
        }
        {
            let wal = wal(dir.path());
            let mut group = wal.begin_group(TransactionId::SYSTEM).unwrap();
            write_standalone(&mut group, &change, EpochId::new(5), Origin::Schema).unwrap();
            group.finish().unwrap();
        }

        let groups = groups(dir.path());
        assert_eq!(groups.len(), 1);
        let records = &groups[0].2;
        assert!(matches!(
            records[0],
            LogRecord::GroupBegin(GroupBegin {
                epoch,
                origin: Origin::Schema,
                ..
            }) if epoch == EpochId::new(5)
        ));
        let decoded: Vec<&StandaloneOp> = records[1..]
            .iter()
            .map(|record| match record {
                LogRecord::Standalone(op) => op,
                other => panic!("not a standalone record: {other:?}"),
            })
            .collect();
        assert_eq!(decoded, ops.iter().collect::<Vec<_>>());
    }

    /// A record over the log's limit is refused before any of its bytes reach
    /// the group, with its own error (not a log failure): the group, dropped,
    /// leaves nothing in the log, and the next group is written as usual.
    #[test]
    fn a_record_over_the_limit_fails_before_any_byte() {
        let dir = tempfile::tempdir().unwrap();
        let mut set = ChangeSet::new();
        let lpg = set
            .slot(GraphRef {
                model: DataModel::Lpg,
                key: None,
            })
            .unwrap();
        // About 1.01 GiB encoded, held as 2^20 references to one string of
        // 1 KiB: refused from its size, before any of it is encoded.
        let line = Value::from("Amsterdam ".repeat(103).as_str());
        let lines = Value::List(std::sync::Arc::from(vec![line; 1 << 20]));
        set.push(
            lpg,
            DataOp::SetNodeProperty {
                id: NodeId::new(3),
                key: PropertyKey::new("notes"),
                value: lines,
            },
            Before::Value(None),
            PendingVersion::Created,
        )
        .unwrap();
        let wal = wal(dir.path());
        let start = wal.end_lsn();
        {
            let mut group = wal.begin_group(TransactionId::new(7)).unwrap();
            let error =
                write_group(&mut group, &set, EpochId::new(1), Origin::Transaction).unwrap_err();
            assert!(
                !matches!(error, Error::Transaction(_)),
                "a record's own error, not a log failure: {error}"
            );
        }
        assert_eq!(wal.end_lsn(), start, "nothing of the group is in the log");

        let (small, _) = every_op();
        let mut group = wal.begin_group(TransactionId::new(8)).unwrap();
        write_group(&mut group, &small, EpochId::new(1), Origin::Transaction).unwrap();
        group.finish().unwrap();
        drop(wal);
        let groups = groups(dir.path());
        assert_eq!(groups.len(), 1);
        assert_eq!(groups[0].0, TransactionId::new(8));
    }

    /// A large set is encoded record by record into frames as it goes: the
    /// group spans many frames, each about the frame target, so neither the
    /// records nor their bytes were collected first.
    #[test]
    fn a_large_set_is_encoded_frame_by_frame() {
        let dir = tempfile::tempdir().unwrap();
        let mut set = ChangeSet::new();
        let lpg = set
            .slot(GraphRef {
                model: DataModel::Lpg,
                key: None,
            })
            .unwrap();
        let count = 5_000_u64;
        for id in 0..count {
            set.push(
                lpg,
                DataOp::SetNodeProperty {
                    id: NodeId::new(id),
                    key: PropertyKey::new("city"),
                    value: Value::from("Prague"),
                },
                Before::Value(None),
                PendingVersion::Created,
            )
            .unwrap();
        }
        {
            let wal = wal(dir.path());
            let mut group = wal.begin_group(TransactionId::new(7)).unwrap();
            write_group(&mut group, &set, EpochId::new(1), Origin::Bulk).unwrap();
            group.finish().unwrap();
        }
        let groups = groups(dir.path());
        let (_, frames, records) = &groups[0];
        assert_eq!(records.len(), usize::try_from(count).unwrap() + 1);
        // About 20 bytes a record, 256 bytes a frame.
        assert!(
            *frames > usize::try_from(count / 20).unwrap(),
            "{frames} frames for {count} records"
        );
    }

    /// Each way a group fails is the commit error that says what happened to
    /// the commit.
    #[test]
    fn a_failed_group_is_the_commit_error_of_its_outcome() {
        let io = || WalError::Io {
            path: "wal".into(),
            source: std::io::Error::other("disk full"),
        };
        assert!(matches!(
            commit_error(GroupError::NotWritten(io())),
            Error::Transaction(TransactionError::LogWriteFailed(reason)) if reason.contains("disk full")
        ));
        assert!(matches!(
            commit_error(GroupError::OutcomeUnknown(io())),
            Error::Transaction(TransactionError::CommitOutcomeUnknown(reason)) if reason.contains("disk full")
        ));
        assert!(matches!(
            commit_error(GroupError::Unavailable {
                reason: "an earlier sync failed".to_string()
            }),
            Error::Transaction(TransactionError::LogUnavailable(reason)) if reason == "an earlier sync failed"
        ));
    }
}
