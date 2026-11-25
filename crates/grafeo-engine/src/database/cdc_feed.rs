//! The durable change feed (#679): while change data capture is stored on
//! (`set_cdc(true)`), every commit's group in the WAL begins with the first
//! timestamp of its change events and holds each write with what it replaced
//! (log kinds 2 and 26), so the feed rebuilds the commit's events from the
//! log alone, as the live log recorded them, after a close or a crash.
//!
//! The default consumer's cursor (`cdc_cursor_default.json`, see
//! [`cdc_cursor`]) says where the feed reads from. A read does not move it;
//! an acknowledgement of a commit's position moves it past that commit, so
//! checkpoints keep the WAL from there on (within the retention limit).

use std::num::NonZeroUsize;
use std::path::PathBuf;

use grafeo_common::change::{Before, DataOp};
use grafeo_common::storage::log_record::{GroupBegin, ImageLogged};
use grafeo_common::storage::{LogRecord, read_log_records};
use grafeo_common::types::{EpochId, HlcTimestamp};
use grafeo_common::utils::error::{Error, QueryError, QueryErrorKind, Result, StorageError};
use grafeo_storage::file::GrafeoFileManager;
use grafeo_storage::wal::{GroupFrames, ScanOptions, WalError, WalScan, list_wal_directory};

use super::cdc_cursor::{self, CdcCursor};
use super::{Open, encryption};
use crate::cdc::{CdcBatch, CdcPosition, ChangeEvent, events_for_ops};

/// A group of the WAL, read whole.
struct Group {
    /// The log position it starts at.
    start: u64,
    /// The log position right after it.
    end: u64,
    begin: GroupBegin,
    records: Vec<LogRecord>,
}

impl Group {
    /// The group's change events, as the live log built them from the same
    /// writes, under the same verdicts of the log's image rule, and the
    /// timestamps from `first` on: an image as
    /// [`BeforeImage::into_logged`](grafeo_common::storage::log_record::BeforeImage::into_logged)
    /// reads it (a delete's identity alone as [`ImageLogged::Identity`]), and
    /// a write logged as its plain kind without the image it replaced as
    /// [`ImageLogged::Nothing`].
    fn events(self, first: HlcTimestamp) -> Vec<ChangeEvent> {
        let writes: Vec<(Option<String>, DataOp, Before, ImageLogged)> = self
            .records
            .into_iter()
            .filter_map(|record| match record {
                LogRecord::Data { graph, op } => {
                    Some((graph, op, Before::Absent, ImageLogged::Nothing))
                }
                LogRecord::DataWithBefore { graph, op, before } => {
                    let (before, logged) = before.into_logged();
                    Some((graph, op, before, logged))
                }
                // Graph and catalog operations take no timestamp, as live.
                LogRecord::GroupBegin(_) | LogRecord::Standalone(_) => None,
            })
            .collect();
        events_for_ops(
            writes
                .iter()
                .map(|(graph, op, before, logged)| (graph.as_deref(), op, before, *logged)),
            writes.len(),
            self.begin.epoch,
            first,
        )
    }
}

/// A group of the WAL, read whole from its frames.
fn read_group(mut frames: GroupFrames) -> Result<Group> {
    let (start, end) = (frames.start_lsn(), frames.end_lsn());
    let mut records = Vec::new();
    while let Some(payload) = frames.next_payload().map_err(Error::from)? {
        read_log_records(payload, &mut |record| {
            records.push(record);
            Ok(())
        })?;
    }
    let mut records = records.into_iter();
    let Some(LogRecord::GroupBegin(begin)) = records.next() else {
        return Err(Error::corruption(format!(
            "the group at log position {start} does not start with its GroupBegin"
        )));
    };
    Ok(Group {
        start,
        end,
        begin,
        records: records.collect(),
    })
}

/// The next group of `scan`, read whole.
fn next_group(scan: &mut WalScan) -> Result<Option<Group>> {
    scan.next_group()
        .map_err(Error::from)?
        .map(read_group)
        .transpose()
}

/// The WAL a database file's feed reads.
struct FeedLog<'a> {
    dir: PathBuf,
    file: &'a GrafeoFileManager,
    keys: encryption::DatabaseKeys,
}

impl FeedLog<'_> {
    /// A scan of the log from `lsn` on.
    fn scan(&self, lsn: u64) -> std::result::Result<WalScan, WalError> {
        WalScan::open(
            &self.dir,
            ScanOptions {
                from_lsn: lsn,
                ..Open::scan_options(self.file, &self.keys, false)
            },
        )
    }

    /// The first log position still in the WAL; `None` for a WAL without
    /// segments.
    fn first_lsn(&self) -> Result<Option<u64>> {
        Ok(list_wal_directory(&self.dir)
            .map_err(Error::from)?
            .segments
            .first()
            .map(|(lsn, _)| *lsn))
    }

    /// The position a feed whose log was cut goes on from: the first group
    /// still in the log, or the log's first position before any group.
    fn restart(&self) -> Result<CdcPosition> {
        let Some(first) = self.first_lsn()? else {
            return Ok(CdcPosition::new(0, EpochId::new(0)));
        };
        let mut scan = self.scan(first).map_err(Error::from)?;
        Ok(
            next_group(&mut scan)?.map_or(CdcPosition::new(first, EpochId::new(0)), |group| {
                CdcPosition::new(group.start, group.begin.epoch)
            }),
        )
    }

    /// The error of a feed whose cursor is behind the log that is left
    /// (`GRAFEO-S006`), naming where it goes on from.
    fn truncated(&self) -> Error {
        match self.restart() {
            Ok(restart) => Error::from(StorageError::CdcFeedTruncated {
                first_available: restart.lsn(),
                epoch: restart.epoch().as_u64(),
            }),
            Err(error) => error,
        }
    }
}

impl Open {
    /// The WAL of the feed: a database file's, with change data capture
    /// stored on.
    fn feed_log(&self) -> Result<FeedLog<'_>> {
        let stored = self
            .catalog
            .cdc_config()
            .is_some_and(|record| record.enabled);
        match self.file_manager.as_deref() {
            Some(file) if stored => Ok(FeedLog {
                dir: file.sidecar_wal_path(),
                file,
                keys: encryption::DatabaseKeys::from_config(&self.config),
            }),
            _ => Err(Error::InvalidValue(
                "the durable change feed reads the WAL of a database file with change data \
                 capture stored on: turn it on with set_cdc(true) on the file"
                    .to_string(),
            )),
        }
    }

    pub(crate) fn read_cdc_feed(&self, limit: NonZeroUsize) -> Result<Vec<CdcBatch>> {
        let _open = self.hold_open()?;
        let log = self.feed_log()?;
        let cursor = cdc_cursor::read_default(&log.dir)?;
        // No checkpoint removes the log the read goes through.
        let Some(_pin) = self.wal_retention.pin(cursor.lsn) else {
            return Err(log.truncated());
        };
        if log.first_lsn()?.is_some_and(|first| cursor.lsn < first) {
            return Err(log.truncated());
        }
        // A group written but not yet published may still fail to publish.
        let published = self.current_epoch();
        let mut scan = log.scan(cursor.lsn).map_err(Error::from)?;
        let (mut batches, mut held) = (Vec::new(), 0_usize);
        while let Some(group) = next_group(&mut scan)? {
            if group.begin.epoch > published {
                break;
            }
            // A group without a clock reported no events: an import, a
            // graph or catalog change.
            let Some(first) = group.begin.clock else {
                continue;
            };
            let position = CdcPosition::new(group.start, group.begin.epoch);
            let events = group.events(first);
            // Whole commits only: the first one even past the limit.
            if !batches.is_empty() && held + events.len() > limit.get() {
                break;
            }
            held += events.len();
            batches.push(CdcBatch { position, events });
            if held >= limit.get() {
                break;
            }
        }
        Ok(batches)
    }

    pub(crate) fn acknowledge_cdc_feed(&self, position: CdcPosition) -> Result<()> {
        let _open = self.hold_open()?;
        let log = self.feed_log()?;
        if self.read_only {
            return Err(Error::from(QueryError::unsupported(
                "a database opened read-only reads its change feed but cannot acknowledge it",
            )));
        }
        // Acknowledgements one at a time: the highest wins.
        let _ack = self.cdc_ack.lock();
        let cursor = cdc_cursor::read_default(&log.dir)?;
        let pin = self.wal_retention.pin(cursor.lsn.min(position.lsn()));
        if pin.is_none() || log.first_lsn()?.is_some_and(|first| cursor.lsn < first) {
            // The log at the cursor is gone: the position the error names
            // goes on from the first commit still there.
            let restart = log.restart()?;
            if position != restart {
                return Err(log.truncated());
            }
            return cdc_cursor::move_default(
                &log.dir,
                CdcCursor {
                    lsn: restart.lsn(),
                    epoch: EpochId::new(restart.epoch().as_u64().saturating_sub(1)),
                },
            );
        }
        // At or behind the cursor: acknowledged already.
        if position.lsn() < cursor.lsn {
            return Ok(());
        }
        let refused = || {
            Error::from(QueryError::new(
                QueryErrorKind::InvalidParameterValue,
                format!(
                    "the change feed has no commit at position {position}: acknowledge a \
                     position read_cdc_feed gave"
                ),
            ))
        };
        // One read at the position: a group must start there, at its epoch,
        // with change events, and be published.
        let mut scan = match log.scan(position.lsn()) {
            Ok(scan) => scan,
            Err(error @ WalError::Io { .. }) => return Err(Error::from(error)),
            Err(_) => return Err(refused()),
        };
        let frames = match scan.next_group() {
            Ok(Some(frames)) => frames,
            Err(error @ WalError::Io { .. }) => return Err(Error::from(error)),
            Ok(None) | Err(_) => return Err(refused()),
        };
        let group = read_group(frames).map_err(|_| refused())?;
        if group.start != position.lsn()
            || group.begin.epoch != position.epoch()
            || group.begin.clock.is_none()
            || group.begin.epoch > self.current_epoch()
        {
            return Err(refused());
        }
        cdc_cursor::move_default(
            &log.dir,
            CdcCursor {
                lsn: group.end,
                epoch: group.begin.epoch,
            },
        )
    }
}

impl crate::GrafeoDB {
    /// Reads the durable change feed from its cursor: the events of each
    /// commit since the last acknowledged one, a [`CdcBatch`] per commit, in
    /// commit order, up to `limit` events (whole commits only: the first one
    /// even when it alone holds more). The events equal the live log's of
    /// the same commits, field for field, also after a close or a crash. A
    /// read does not move the cursor: acknowledge a batch's position with
    /// [`acknowledge_cdc_feed`](Self::acknowledge_cdc_feed) once it is
    /// handled.
    ///
    /// The feed starts when change data capture is stored on
    /// ([`set_cdc(true)`](Self::set_cdc)); turning it off and on again
    /// starts a fresh one.
    ///
    /// # Stability: Experimental
    ///
    /// New in 0.6.4.
    ///
    /// # Errors
    ///
    /// `GRAFEO-V001` without a database file or without the setting stored
    /// on ([`Config::with_cdc`](crate::Config::with_cdc) captures for one
    /// open only and keeps no feed). `GRAFEO-S006`
    /// ([`StorageError::CdcFeedTruncated`]) when a checkpoint removed the log
    /// at the cursor (past [`Config::with_wal_retention_limit`](crate::Config::with_wal_retention_limit)):
    /// it names the position of the first commit still there, which
    /// [`acknowledge_cdc_feed`](Self::acknowledge_cdc_feed) accepts to go on
    /// from it. The errors of reading the WAL.
    pub fn read_cdc_feed(&self, limit: NonZeroUsize) -> Result<Vec<CdcBatch>> {
        self.live()?.read_cdc_feed(limit)
    }

    /// Moves the durable change feed past the commit at `position`, a
    /// position [`read_cdc_feed`](Self::read_cdc_feed) gave: the next read
    /// starts after it, and checkpoints keep the WAL from there on. A
    /// position at or behind the cursor is acknowledged already (a retry
    /// changes nothing); acknowledgements are serialized, and the highest
    /// wins. A crash during one leaves the old cursor or the new one.
    ///
    /// # Stability: Experimental
    ///
    /// New in 0.6.4.
    ///
    /// # Errors
    ///
    /// `22023` (an invalid parameter value) for a position the feed did not
    /// give: no published commit starts there at its epoch; the cursor stays.
    /// A database opened read-only refuses it. `GRAFEO-S006` when the log at
    /// the cursor was removed, unless `position` is the one that error names.
    /// As [`read_cdc_feed`](Self::read_cdc_feed) for the setting and the
    /// file, and the errors of writing the cursor.
    pub fn acknowledge_cdc_feed(&self, position: CdcPosition) -> Result<()> {
        self.live()?.acknowledge_cdc_feed(position)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use grafeo_common::change::{EdgeImage, Labels, NodeImage};
    use grafeo_common::storage::log_record::{BeforeImage, ImageLogged, Origin};
    use grafeo_common::types::{EdgeId, NodeId, Value};

    /// Each delete record of a group reads back as the live log reports the same write under
    /// the same verdict: a plain delete (no image: the identity itself did not fit) with neither
    /// labels nor properties; a node's or an edge's identity alone with its labels, or its type
    /// and endpoints, and no properties; a whole image with both.
    #[test]
    fn each_delete_record_reads_back_as_the_live_log_reports_it() {
        let labels: Labels = [grafeo_common::types::ArcStr::from("Person")]
            .into_iter()
            .collect();
        let (gone, alone, edge, whole) = (
            DataOp::DeleteNode { id: NodeId::new(3) },
            DataOp::DeleteNode { id: NodeId::new(4) },
            DataOp::DeleteEdge { id: EdgeId::new(9) },
            DataOp::DeleteNode { id: NodeId::new(5) },
        );
        let name = vec![("name".into(), Value::from("Mia"))];
        let first = HlcTimestamp::new(1_700_000_000_000, 0);
        let group = Group {
            start: 0,
            end: 0,
            begin: GroupBegin {
                epoch: EpochId::new(7),
                timestamp_ms: 1_700_000_000_000,
                origin: Origin::Transaction,
                clock: Some(first),
            },
            records: vec![
                LogRecord::Data {
                    graph: None,
                    op: gone.clone(),
                },
                LogRecord::DataWithBefore {
                    graph: None,
                    op: alone.clone(),
                    before: BeforeImage::NodeIdentity {
                        labels: labels.clone(),
                    },
                },
                LogRecord::DataWithBefore {
                    graph: None,
                    op: edge.clone(),
                    before: BeforeImage::EdgeIdentity {
                        edge_type: "KNOWS".into(),
                        src: NodeId::new(4),
                        dst: NodeId::new(5),
                        deleted_with: None,
                    },
                },
                LogRecord::DataWithBefore {
                    graph: None,
                    op: whole.clone(),
                    before: BeforeImage::Node {
                        labels: labels.clone(),
                        properties: name.clone(),
                    },
                },
            ],
        };
        let node = |properties| {
            Before::Node(Box::new(NodeImage {
                labels: labels.clone(),
                properties,
            }))
        };
        let live_ops = [
            (&gone, Before::Absent, ImageLogged::Nothing),
            (&alone, node(Vec::new()), ImageLogged::Identity),
            (
                &edge,
                Before::Edge(Box::new(EdgeImage {
                    src: NodeId::new(4),
                    dst: NodeId::new(5),
                    edge_type: "KNOWS".into(),
                    properties: Vec::new(),
                    deleted_with: None,
                })),
                ImageLogged::Identity,
            ),
            (&whole, node(name), ImageLogged::Whole),
        ];
        let live = events_for_ops(
            live_ops
                .iter()
                .map(|(op, before, logged)| (None, *op, before, *logged)),
            live_ops.len(),
            EpochId::new(7),
            first,
        );
        let durable = group.events(first);
        let json = |events: &[ChangeEvent]| serde_json::to_value(events).unwrap();
        assert_eq!(json(&durable), json(&live));
        assert_eq!(
            (durable[0].labels.is_none(), durable[0].before.is_none()),
            (true, true)
        );
        assert_eq!(durable[1].labels, Some(vec!["Person".to_string()]));
        assert!(durable[1].before.is_none());
        assert_eq!(
            (
                durable[2].edge_type.as_deref(),
                durable[2].src_id,
                durable[2].dst_id
            ),
            (Some("KNOWS"), Some(4), Some(5))
        );
        assert!(durable[2].before.is_none());
        assert_eq!(
            durable[3].before.as_ref().map(|before| before.len()),
            Some(1)
        );
    }

    #[test]
    fn a_token_round_trips_and_a_malformed_one_is_refused() {
        let position = CdcPosition::new(4_242, EpochId::new(19));
        assert_eq!(position.to_token(), "4242:19");
        assert_eq!(CdcPosition::from_token("4242:19").unwrap(), position);
        for malformed in ["", ":", "12", "a:b", "1:2:3", "-1:2", "+1:2", "1: 2"] {
            assert!(CdcPosition::from_token(malformed).is_err(), "{malformed:?}");
        }
    }
}
