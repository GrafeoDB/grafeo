//! A pinned read of one graph: [`Snapshot`].

use std::sync::Arc;

use grafeo_common::types::{EpochId, ReadView};
use grafeo_core::graph::GraphStore;

use crate::transaction::ReadPin;

/// A pinned read of one graph: the graph's store, the view every read
/// through it takes, and a pin that keeps the versions that view reads while
/// the snapshot lives, so no garbage collection takes one of them between
/// two reads. Several reads through one snapshot see one state of the
/// graph, the committed state of its epoch, however many commits and
/// collects run meanwhile.
///
/// Get one from [`GrafeoDB::snapshot`](crate::GrafeoDB::snapshot) (the graph
/// the database's current graph and schema select),
/// [`GraphHandle::snapshot`](crate::GraphHandle::snapshot) (a named graph),
/// [`GrafeoDB::projection`](crate::GrafeoDB::projection) (a projection) or
/// [`AsOfView::snapshot`](crate::AsOfView::snapshot) (an earlier epoch), and
/// read with the [`GraphStore`] methods, passing [`view`](Self::view):
///
/// ```
/// # use grafeo_engine::{Direction, GrafeoDB};
/// let db = GrafeoDB::new_in_memory();
/// let alix = db.create_node(&["Person"])?;
/// let gus = db.create_node(&["Person"])?;
/// db.create_edge(alix, gus, "KNOWS")?;
///
/// let snapshot = db.snapshot()?;
/// let (store, view) = (snapshot.store(), snapshot.view());
/// assert_eq!(store.node_count(view)?, 2);
/// assert_eq!(store.edges(alix, Direction::Outgoing, &[], view)?.len(), 1);
/// # Ok::<(), grafeo_common::utils::error::Error>(())
/// ```
///
/// Holding a snapshot holds back garbage collection at its epoch: drop it
/// when the reads are done.
pub struct Snapshot {
    store: Arc<dyn GraphStore>,
    view: ReadView,
    /// Keeps the view's epoch at or after the collect horizon. A snapshot a
    /// query makes for the procedures it calls has none of its own: the
    /// query's statement or transaction pins its view for longer than the
    /// call that borrows the snapshot.
    _pin: Option<ReadPin>,
}

impl Snapshot {
    /// A snapshot of `store` at `view`, kept by `pin`.
    pub(crate) fn pinned(store: Arc<dyn GraphStore>, view: ReadView, pin: ReadPin) -> Self {
        Self {
            store,
            view,
            _pin: Some(pin),
        }
    }

    /// A snapshot of `store` at the view of a query that pins it itself, for
    /// the procedures the query calls.
    #[cfg(feature = "algos")]
    pub(crate) fn of_query(store: Arc<dyn GraphStore>, view: ReadView) -> Self {
        Self {
            store,
            view,
            _pin: None,
        }
    }

    /// The graph's store; read it at [`view`](Self::view).
    #[must_use]
    pub fn store(&self) -> &dyn GraphStore {
        &*self.store
    }

    /// The view the snapshot reads at: every read through it passes this.
    #[must_use]
    pub fn view(&self) -> ReadView {
        self.view
    }

    /// The epoch the snapshot reads at.
    #[must_use]
    pub fn epoch(&self) -> EpochId {
        self.view.epoch
    }

    /// The store as a shared handle, for an API that takes one (such as
    /// [`GrafeoDB::with_read_store`](crate::GrafeoDB::with_read_store) over a
    /// projection). It outlives the snapshot, and reads through it after the
    /// snapshot is dropped are not pinned.
    #[must_use]
    pub fn shared(&self) -> Arc<dyn GraphStore> {
        Arc::clone(&self.store)
    }
}

/// The store's reads, called on the snapshot: `snapshot.node(id,
/// snapshot.view())`.
impl std::ops::Deref for Snapshot {
    type Target = dyn GraphStore;

    fn deref(&self) -> &Self::Target {
        &*self.store
    }
}

impl std::fmt::Debug for Snapshot {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Snapshot")
            .field("view", &self.view)
            .finish_non_exhaustive()
    }
}
