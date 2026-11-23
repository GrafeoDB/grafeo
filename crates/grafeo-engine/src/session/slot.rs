//! A session as callers hold it: a slot the database's `close()` takes (#712).
//!
//! What a session holds of its database (its stores, catalog, logs and caches) lives in a
//! [`SessionInner`] inside a slot. The open database lists every slot it hands out, weakly,
//! and `close()` takes each one, so what the session held goes with the close however long the
//! caller keeps the [`Session`]; its later calls fail with `GRAFEO-T007`. A slot is made only by
//! [`Slots::register_session`], so none can escape the list.
//!
//! Lock order: `close()` first takes the database's open state and lets go of that lock, so no
//! new statement starts; then it takes each slot's write lock, which waits for the statement in
//! flight on that slot, if any; then it persists. A statement holds only its slot's read guard,
//! for the statement, and never waits on the open state's lock. No caller's code runs under
//! either lock: a caller's iterator is collected before the lock is taken.

use std::sync::{Arc, Weak};

use grafeo_common::utils::error::{Error, Result, TransactionError};
use parking_lot::{
    MappedRwLockReadGuard, MappedRwLockWriteGuard, Mutex, RwLock, RwLockReadGuard, RwLockWriteGuard,
};

use grafeo_core::execution::operators::Operator;
#[cfg(feature = "gql")]
use grafeo_core::execution::operators::OperatorResult;

use super::SessionInner;

/// A session's slot: `None` once `close()` took it.
type SessionSlot = RwLock<Option<SessionInner>>;

/// The slots an open database handed out, listed weakly: a dropped session leaves only a dead
/// entry, swept as the list grows.
#[derive(Default)]
pub(crate) struct Slots {
    sessions: Mutex<Listed<SessionSlot>>,
    streams: Mutex<Listed<StreamSlot>>,
}

/// The listed slots of one kind, and the length at which the dead ones are swept next.
struct Listed<T> {
    slots: Vec<Weak<T>>,
    sweep_at: usize,
}

impl<T> Default for Listed<T> {
    fn default() -> Self {
        Self {
            slots: Vec::new(),
            sweep_at: 0,
        }
    }
}

impl<T> Listed<T> {
    /// Lists `slot`, first sweeping the dead entries once the list has doubled since the last
    /// sweep.
    fn push(&mut self, slot: &Arc<T>) {
        if self.slots.len() >= self.sweep_at.max(FIRST_SWEEP) {
            self.slots.retain(|weak| weak.strong_count() > 0);
            self.sweep_at = 2 * self.slots.len();
        }
        self.slots.push(Arc::downgrade(slot));
    }
}

/// An open stream's operators: `None` once `close()` took them.
type StreamSlot = Mutex<Option<Box<dyn Operator>>>;

/// The operators of an open stream, in a slot the database's `close()` empties: what they
/// read (the stores, an operator's buffers) goes with the close, and the stream's next pull
/// fails with `GRAFEO-T007`. Made only by [`Slots::register_stream`].
#[cfg(feature = "gql")]
pub(crate) struct OperatorSlot(Arc<StreamSlot>);

#[cfg(feature = "gql")]
impl OperatorSlot {
    /// The next chunk of the stream's operators; `None` once the database is closed.
    pub(crate) fn next(&self) -> Option<OperatorResult> {
        self.0.lock().as_mut().map(|operator| operator.next())
    }
}

/// The length below which the list is never swept.
const FIRST_SWEEP: usize = 64;

impl Slots {
    /// A session over `inner`, listed so `close()` takes it: the one way to make a session's
    /// slot.
    pub(crate) fn register_session(&self, inner: SessionInner) -> Session {
        let identity = Arc::clone(&inner.identity);
        let slot = Arc::new(RwLock::new(Some(inner)));
        self.sessions.lock().push(&slot);
        Session { slot, identity }
    }

    /// Takes every listed slot: each waits for the statement in flight on it, and what the
    /// session held drops (an open transaction rolls back). Called by `close()` once the open
    /// state is taken, so no slot is listed after it.
    pub(crate) fn revoke(&self) {
        let sessions = std::mem::take(&mut self.sessions.lock().slots);
        for weak in sessions {
            if let Some(slot) = weak.upgrade() {
                let taken = slot.write().take();
                drop(taken);
            }
        }
        // A stream's lock waits for the pull in flight on it.
        let streams = std::mem::take(&mut self.streams.lock().slots);
        for weak in streams {
            if let Some(slot) = weak.upgrade() {
                let taken = slot.lock().take();
                drop(taken);
            }
        }
    }

    /// A slot holding a stream's `operator`, listed so `close()` empties it: the one way to
    /// make an [`OperatorSlot`].
    #[cfg(feature = "gql")]
    pub(crate) fn register_stream(&self, operator: Box<dyn Operator>) -> OperatorSlot {
        let slot = Arc::new(Mutex::new(Some(operator)));
        self.streams.lock().push(&slot);
        OperatorSlot(slot)
    }
}

/// Your handle to the database - execute queries and manage transactions.
///
/// Get one from [`GrafeoDB::session()`](crate::GrafeoDB::session). Each session
/// tracks its own transaction state, so you can have multiple concurrent
/// sessions without them interfering.
pub struct Session {
    slot: Arc<SessionSlot>,
    /// Who the session runs as: not data of the database, so it answers after `close()`.
    pub(super) identity: Arc<crate::auth::Identity>,
}

impl Session {
    /// A session of a closed database: every call fails with `GRAFEO-T007`.
    pub(crate) fn closed() -> Self {
        Self {
            slot: Arc::new(RwLock::new(None)),
            identity: Arc::new(crate::auth::Identity::anonymous()),
        }
    }

    /// The session's state for one call; `GRAFEO-T007` once the database is closed.
    pub(crate) fn inner(&self) -> Result<MappedRwLockReadGuard<'_, SessionInner>> {
        RwLockReadGuard::try_map(self.slot.read(), Option::as_ref).map_err(|_| closed())
    }

    /// [`inner`](Self::inner), for a call that changes the session.
    pub(crate) fn inner_mut(&self) -> Result<MappedRwLockWriteGuard<'_, SessionInner>> {
        RwLockWriteGuard::try_map(self.slot.write(), Option::as_mut).map_err(|_| closed())
    }
}

impl std::fmt::Debug for Session {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Session")
            .field("closed", &self.slot.read().is_none())
            .finish_non_exhaustive()
    }
}

/// The closed-database error.
fn closed() -> Error {
    Error::Transaction(TransactionError::DatabaseClosed)
}
