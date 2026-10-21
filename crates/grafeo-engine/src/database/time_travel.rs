//! Time travel: reads of the database as it was at an earlier epoch or time
//! (H2c, the time-travel design).
//!
//! A read in the past is a read-only read at a past epoch, through the same
//! reads at a view as a transaction's snapshot: [`GrafeoDB::as_of`] gives an
//! [`AsOfView`] pinned at the epoch, [`GrafeoDB::execute_as_of`] runs one
//! query there, beside [`GrafeoDB::execute_at_epoch`]. How far back they
//! reach is the database's history retention
//! ([`HistoryRetention`](crate::HistoryRetention)); a read before what it
//! keeps fails with "history before ... is not retained", never with an
//! answer from versions garbage collection may have dropped. A view pins
//! its epoch while it lives, so what it reads stays.
//!
//! The names follow SQL:2011's system versioning (`FOR SYSTEM_TIME AS OF`),
//! as Oracle, CockroachDB and BigQuery do: ISO GQL has no time travel, only
//! the time values a point in time is written in (`ZONED DATETIME`). No
//! query syntax names a time yet.

use std::time::Duration;

use grafeo_common::types::{EdgeId, EpochId, NodeId, ReadView, Timestamp, Value, ZonedDatetime};
use grafeo_common::utils::error::{Error, Result};
use grafeo_core::graph::lpg::{Edge, Node};

use crate::session::Session;
use crate::transaction::ReadPin;

/// Where a time-travel read reads: an epoch, a time (the last epoch
/// committed at or before it, at the resolution of a millisecond), or a
/// duration before now.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TimePoint {
    /// An epoch, as [`GrafeoDB::current_epoch`](super::GrafeoDB::current_epoch)
    /// gives them.
    Epoch(EpochId),
    /// A time: the state the last commit at or before it left.
    Timestamp(Timestamp),
    /// A duration before now, on the clock commits take their times from
    /// ("five minutes ago"); see [`TimePoint::ago`].
    Ago(Duration),
}

impl TimePoint {
    /// The point `duration` before now: `TimePoint::ago(Duration::from_secs(300))`
    /// reads the database as it was five minutes ago.
    #[must_use]
    pub const fn ago(duration: Duration) -> Self {
        Self::Ago(duration)
    }
}

impl From<EpochId> for TimePoint {
    fn from(epoch: EpochId) -> Self {
        Self::Epoch(epoch)
    }
}

impl From<Timestamp> for TimePoint {
    fn from(timestamp: Timestamp) -> Self {
        Self::Timestamp(timestamp)
    }
}

/// A GQL `ZONED DATETIME` names its instant, whatever its offset.
impl From<ZonedDatetime> for TimePoint {
    fn from(datetime: ZonedDatetime) -> Self {
        Self::Timestamp(datetime.as_timestamp())
    }
}

/// A read-only view of the database as of a past epoch (see
/// [`GrafeoDB::as_of`](super::GrafeoDB::as_of)): its queries and reads see
/// the committed state of that epoch, and a write is refused. It pins its
/// epoch while it lives, so garbage collection keeps what it reads.
pub struct AsOfView {
    session: Session,
    epoch: EpochId,
    /// Keeps the collect horizon at the view's epoch or before.
    _pin: ReadPin,
}

impl std::fmt::Debug for AsOfView {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AsOfView")
            .field("epoch", &self.epoch)
            .finish_non_exhaustive()
    }
}

impl AsOfView {
    /// The epoch the view reads at.
    #[must_use]
    pub fn epoch(&self) -> EpochId {
        self.epoch
    }

    /// The view a read of the store reads at, the committed state of the
    /// view's epoch: for several reads through
    /// [`GrafeoDB::store`](super::GrafeoDB::store)'s `_at` methods that see
    /// one state, kept while this view lives. A view as of
    /// [`current_epoch`](super::GrafeoDB::current_epoch) reads the graph of
    /// now, also without history retention.
    #[must_use]
    pub fn read_view(&self) -> ReadView {
        ReadView::committed(self.epoch)
    }

    /// Runs a GQL query at the view's epoch.
    ///
    /// # Errors
    ///
    /// The query's errors, and a read-only error for a query that writes.
    #[cfg(feature = "gql")]
    pub fn execute(&self, query: &str) -> Result<crate::database::QueryResult> {
        self.session.execute(query)
    }

    /// Runs a GQL query with parameters at the view's epoch.
    ///
    /// # Errors
    ///
    /// As [`execute`](Self::execute).
    #[cfg(feature = "gql")]
    pub fn execute_with_params(
        &self,
        query: &str,
        params: std::collections::HashMap<String, Value>,
    ) -> Result<crate::database::QueryResult> {
        self.session.execute_with_params(query, params)
    }

    /// The node as it was at the view's epoch.
    #[must_use]
    pub fn get_node(&self, id: NodeId) -> Option<Node> {
        self.session.get_node(id)
    }

    /// The edge as it was at the view's epoch.
    #[must_use]
    pub fn get_edge(&self, id: EdgeId) -> Option<Edge> {
        self.session.get_edge(id)
    }

    /// A node's value of `key` as it was at the view's epoch.
    #[must_use]
    pub fn get_node_property(&self, id: NodeId, key: &str) -> Option<Value> {
        self.session.get_node_property(id, key)
    }
}

impl super::GrafeoDB {
    /// The epoch a read as of `point` reads at: the epoch itself (the
    /// current one for a later one), or the last epoch committed at or
    /// before a time, or before a duration ago.
    ///
    /// # Errors
    ///
    /// For a time while the database keeps no history, and for a time
    /// before what it keeps: "history before ... is not retained".
    pub fn epoch_as_of(&self, point: impl Into<TimePoint>) -> Result<EpochId> {
        match point.into() {
            TimePoint::Epoch(epoch) => Ok(epoch.min(self.transaction_manager.current_epoch())),
            TimePoint::Timestamp(timestamp) => self.transaction_manager.epoch_at(timestamp),
            TimePoint::Ago(duration) => {
                let back = u64::try_from(duration.as_millis()).unwrap_or(u64::MAX);
                let millis = self.transaction_manager.history_now().saturating_sub(back);
                let millis = i64::try_from(millis).map_err(|_| {
                    Error::InvalidValue(format!("a time {millis} ms after 1970 is out of range"))
                })?;
                self.transaction_manager
                    .epoch_at(Timestamp::from_millis(millis))
            }
        }
    }

    /// A read-only view of the database as of `point`, an epoch, a time or
    /// a duration ago (see [`AsOfView`]): what a reader at that epoch read,
    /// while the view lives.
    ///
    /// ```
    /// use grafeo_common::types::Value;
    /// use grafeo_engine::{Config, GrafeoDB, HistoryRetention};
    ///
    /// let db = GrafeoDB::with_config(
    ///     Config::in_memory().with_history_retention(HistoryRetention::Forever),
    /// )?;
    /// db.execute("INSERT (:Person {name: 'Alix'})")?;
    /// let then = db.current_epoch();
    /// db.execute("INSERT (:Person {name: 'Gus'})")?;
    /// let view = db.as_of(then)?;
    /// let people = view.execute("MATCH (p:Person) RETURN count(p)")?;
    /// assert_eq!(people.rows()[0][0], Value::Int64(1), "Gus came later");
    /// assert!(view.execute("INSERT (:Person {name: 'Vincent'})").is_err());
    /// # Ok::<(), grafeo_common::utils::error::Error>(())
    /// ```
    ///
    /// # Errors
    ///
    /// For a point before what the database keeps: "history before ... is
    /// not retained"; for a time while it keeps no history.
    pub fn as_of(&self, point: impl Into<TimePoint>) -> Result<AsOfView> {
        let epoch = self.epoch_as_of(point)?;
        // Pinned first, then read: the floor cannot pass the epoch.
        let pin = self.transaction_manager.pin_read_at(epoch)?;
        let session = self.session_with_role(crate::auth::Role::ReadOnly);
        session.set_viewing_epoch(epoch);
        Ok(AsOfView {
            session,
            epoch,
            _pin: pin,
        })
    }

    /// Runs a GQL query as the database was as of `point` (see
    /// [`epoch_as_of`](Self::epoch_as_of) and
    /// [`execute_at_epoch`](Self::execute_at_epoch)).
    ///
    /// # Errors
    ///
    /// The query's errors; a point before what the database keeps, or any
    /// time while it keeps no history: "history before ... is not
    /// retained".
    #[cfg(feature = "gql")]
    pub fn execute_as_of(
        &self,
        query: &str,
        point: impl Into<TimePoint>,
    ) -> Result<crate::database::QueryResult> {
        let epoch = self.epoch_as_of(point)?;
        self.execute_at_epoch(query, epoch)
    }
}
