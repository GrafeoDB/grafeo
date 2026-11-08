//! The memory account of a row-group store (#751): what one store holds,
//! counted where it grows, under the store's write lock, with one grant per
//! region it holds (the graph storage and the index buffers).
//!
//! Every part of a row group keeps a figure of its heap bytes that costs a
//! few loads to read (a capacity, or a count the part keeps as it changes),
//! and a write reads the figures of the row groups it touches before and
//! after it: the difference is what the account adds, line by line (see
//! [`Line`]). The parts' figures are only ever read as such differences, so
//! a figure needs to move exactly as the part does, not to start at its
//! part's size: a load builds the account from one walk of what it loaded
//! (`StoreAccount::reset`), and a row group dropped whole leaves the account
//! by its walk.
//!
//! A read may decode a cold column (a thaw) under the store's read lock,
//! where the account cannot be written: the column reports the bytes it
//! gained and freed to the store's thaws ([`Thaws`]), counters any thread
//! can add to, which the report adds to the account's lines and the next
//! write folds into them. A thaw takes a grant of the decoded bytes at once,
//! which the account adopts when it folds them.
//!
//! The walks of the parts (`diagnostics`) stay as the account's oracle: with
//! no reader running, the account of each line equals its walk.
//!
//! Admission ([`StoreAccount::admit`]) keeps headroom in the grants before
//! an op that may be refused: when the memory limit does not fit it, the op
//! is refused before it changes anything, and the store frees what it can
//! outside its lock and tries it once more (`RowGroupStore::apply`).

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, OnceLock};

use bytes::Bytes;
use grafeo_common::change::Table;
use grafeo_common::memory::buffer::{BufferManager, MemoryGrant, MemoryRegion};
use parking_lot::Mutex;

/// Whether admission refuses an op whose bytes the memory limit does not
/// fit (otherwise the grants grow whatever the limit).
const ADMISSION_REFUSES: bool = true;

/// The lines of the account: the byte fields of
/// [`RowGroupMemory`](super::RowGroupMemory) the walk reports, with the
/// cold chunks' bytes apart from their metadata (their own grants hold
/// them) and the store's indexes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Line {
    /// The node row groups' entries in their tables, with their checkpoint
    /// marks.
    NodeTables,
    /// The edge row groups' entries, with each row's ends and type and the
    /// groups' checkpoint marks.
    EdgeTables,
    /// The node property columns' decoded values and presence bits.
    NodeColumns,
    /// The edge property columns' decoded values and presence bits.
    EdgeColumns,
    /// The metadata of the node columns' cold chunks: the lists, the point
    /// views.
    NodeCold,
    /// The metadata of the edge columns' cold chunks.
    EdgeCold,
    /// The bytes of the node columns' cold chunks in memory.
    NodeColdChunks,
    /// The bytes of the edge columns' cold chunks in memory.
    EdgeColdChunks,
    /// The node rows' versions and update chains, with the list of node
    /// row groups a collect visits.
    NodeVersions,
    /// The edge rows' versions and update chains, with the list of edge
    /// row groups a collect visits.
    EdgeVersions,
    /// The nodes' outgoing edges.
    Outgoing,
    /// The nodes' incoming edges.
    Incoming,
    /// The bitmaps of rows per label.
    LabelBitmaps,
    /// The label dictionary and the counts per label.
    Labels,
    /// The edge type dictionary and the counts per edge type.
    EdgeTypes,
    /// The property key dictionary.
    Keys,
    /// The property indexes, on every label and on one.
    PropertyIndexes,
    /// The text indexes.
    TextIndexes,
    /// The vector indexes.
    VectorIndexes,
}

/// How many lines the account has.
pub(super) const LINES: usize = 19;

/// A figure per line, indexed by [`Line::index`].
pub(super) type Lines = [usize; LINES];

impl Line {
    /// The index of the line in [`Lines`].
    #[must_use]
    pub(super) const fn index(self) -> usize {
        self as usize
    }

    /// The region whose grant covers the line; `None` for the cold chunks'
    /// bytes, which their own grants cover.
    #[must_use]
    const fn region(self) -> Option<MemoryRegion> {
        match self {
            Self::NodeColdChunks | Self::EdgeColdChunks => None,
            Self::Outgoing
            | Self::Incoming
            | Self::PropertyIndexes
            | Self::TextIndexes
            | Self::VectorIndexes => Some(MemoryRegion::IndexBuffers),
            Self::NodeTables
            | Self::EdgeTables
            | Self::NodeColumns
            | Self::EdgeColumns
            | Self::NodeCold
            | Self::EdgeCold
            | Self::NodeVersions
            | Self::EdgeVersions
            | Self::LabelBitmaps
            | Self::Labels
            | Self::EdgeTypes
            | Self::Keys => Some(MemoryRegion::GraphStorage),
        }
    }

    /// The lines of a table's property columns: its columns, its cold
    /// chunks' metadata and their bytes.
    #[must_use]
    pub(super) const fn columns_of(table: Table) -> (Self, Self, Self) {
        match table {
            Table::Nodes => (Self::NodeColumns, Self::NodeCold, Self::NodeColdChunks),
            Table::Edges => (Self::EdgeColumns, Self::EdgeCold, Self::EdgeColdChunks),
        }
    }
}

/// The lines in their order, for the region of each index.
const ALL: [Line; LINES] = [
    Line::NodeTables,
    Line::EdgeTables,
    Line::NodeColumns,
    Line::EdgeColumns,
    Line::NodeCold,
    Line::EdgeCold,
    Line::NodeColdChunks,
    Line::EdgeColdChunks,
    Line::NodeVersions,
    Line::EdgeVersions,
    Line::Outgoing,
    Line::Incoming,
    Line::LabelBitmaps,
    Line::Labels,
    Line::EdgeTypes,
    Line::Keys,
    Line::PropertyIndexes,
    Line::TextIndexes,
    Line::VectorIndexes,
];

/// Which total a line goes to: the graph storage's (0), the index buffers'
/// (1), or none (2: the cold chunks' bytes, which their own grants cover).
const REGION_OF: [usize; LINES] = {
    let mut regions = [2; LINES];
    let mut index = 0;
    while index < LINES {
        regions[index] = match ALL[index].region() {
            Some(MemoryRegion::GraphStorage) => 0,
            Some(_) => 1,
            None => 2,
        };
        index += 1;
    }
    regions
};

/// The line at `index` of [`Lines`].
#[must_use]
pub(super) const fn line_at(index: usize) -> Line {
    ALL[index]
}

const _: () = {
    let mut index = 0;
    while index < LINES {
        assert!(ALL[index].index() == index, "the lines are in their order");
        index += 1;
    }
};

/// Adds `from` to `into`, line by line. The figures are differences of the
/// parts' counts, so the sums wrap: a line the report reads is the true
/// figure once every difference is in.
pub(super) fn add_lines(into: &mut Lines, from: &Lines) {
    for (line, bytes) in into.iter_mut().zip(from) {
        *line = line.wrapping_add(*bytes);
    }
}

/// The index of a table among the thaws' counters.
const fn table_index(table: Table) -> usize {
    match table {
        Table::Nodes => 0,
        Table::Edges => 1,
    }
}

/// The buffer manager a store takes its grants from, once a database
/// attaches one.
#[derive(Default)]
pub(super) struct Attached(OnceLock<Arc<BufferManager>>);

impl Attached {
    /// The manager, once attached.
    #[must_use]
    pub(super) fn get(&self) -> Option<&Arc<BufferManager>> {
        self.0.get()
    }

    /// Attaches `manager`; whether it was the first (a store keeps the
    /// first manager it gets).
    pub(super) fn set(&self, manager: Arc<BufferManager>) -> bool {
        self.0.set(manager).is_ok()
    }
}

impl std::fmt::Debug for Attached {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_tuple("Attached")
            .field(&self.0.get().is_some())
            .finish()
    }
}

/// The memory consumers registered for a store, each by its manager and
/// name: unregistered when the store drops, so a consumer that reads the
/// store (weakly) leaves the manager with it.
#[derive(Default)]
pub(super) struct Registered(Mutex<Vec<(Arc<BufferManager>, &'static str)>>);

impl Registered {
    /// Notes the consumer `name` of `manager`.
    pub(super) fn add(&self, manager: &Arc<BufferManager>, name: &'static str) {
        self.0.lock().push((Arc::clone(manager), name));
    }
}

impl Drop for Registered {
    fn drop(&mut self) {
        for (manager, name) in self.0.get_mut().drain(..) {
            manager.unregister_consumer(name);
        }
    }
}

impl std::fmt::Debug for Registered {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let names: Vec<&str> = self.0.lock().iter().map(|(_, name)| *name).collect();
        f.debug_tuple("Registered").field(&names).finish()
    }
}

/// The bytes the store's cold columns gained and freed as reads decoded
/// them, since the account last folded them in (see the module docs), with
/// the grant those thaws took.
#[derive(Debug, Default)]
pub(super) struct Thaws {
    /// Whether a thaw happened since the last fold.
    pending: AtomicBool,
    /// Per table, the decoded bytes the thaws gained.
    gained: [AtomicUsize; 2],
    /// Per table, the cold chunks' metadata the thaws freed.
    freed_meta: [AtomicUsize; 2],
    /// Per table, the cold chunks' bytes the thaws freed.
    freed_chunks: [AtomicUsize; 2],
    /// The buffer manager the thaws take their grants from.
    manager: Attached,
    /// The grant of the bytes the thaws gained since the last fold.
    grant: Mutex<Option<MemoryGrant>>,
    /// While above 0, a read's thaw takes its grant whatever the limit
    /// (`RowGroupStore::with_unrefused`: the index rebuilds at open).
    unrefused: AtomicUsize,
}

/// The grant a read's thaw asks for (see [`ThawSink::grant_thaw`]).
pub(super) enum ThawGrant {
    /// No buffer manager: nothing to grant.
    Unbounded,
    /// The bytes, granted.
    Granted(MemoryGrant),
    /// The memory limit has no room for them: the read decodes the chunk it
    /// needs instead of the column.
    Refused,
}

impl ThawGrant {
    /// The grant, when one was taken.
    pub(super) fn into_grant(self) -> Option<MemoryGrant> {
        match self {
            Self::Granted(grant) => Some(grant),
            Self::Unbounded | Self::Refused => None,
        }
    }
}

/// Holds a store's reads' thaws unrefused while it lives (see
/// `RowGroupStore::with_unrefused`).
pub(super) struct Unrefused(Vec<Arc<Thaws>>);

impl Drop for Unrefused {
    fn drop(&mut self) {
        for thaws in &self.0 {
            thaws.unrefused.fetch_sub(1, Ordering::AcqRel);
        }
    }
}

/// Where a cold column reports its thaw: its store's thaws, and its table.
#[derive(Debug, Clone)]
pub(super) struct ThawSink {
    thaws: Arc<Thaws>,
    table: Table,
}

impl ThawSink {
    /// The grant of `bytes` a read's thaw needs: refused when the memory
    /// limit has no room for them (no eviction runs under the store's
    /// lock), granted whatever the limit while the store is
    /// [unrefused](Unrefused).
    pub(super) fn grant_thaw(&self, bytes: usize) -> ThawGrant {
        let Some(manager) = self.thaws.manager.get() else {
            return ThawGrant::Unbounded;
        };
        if self.thaws.unrefused.load(Ordering::Acquire) > 0 {
            return ThawGrant::Granted(manager.reserve_forced(bytes, MemoryRegion::GraphStorage));
        }
        let mut grant = manager.empty_grant(MemoryRegion::GraphStorage);
        if grant.try_grow(bytes) {
            ThawGrant::Granted(grant)
        } else {
            ThawGrant::Refused
        }
    }

    /// Reports a thaw: the column gained `hot` decoded bytes and freed
    /// `meta` bytes of chunk metadata and `chunks` bytes of chunks. The
    /// decoded bytes hold `granted` (a read's, from
    /// [`grant_thaw`](Self::grant_thaw)), brought to `hot`, or a grant
    /// taken at once whatever the limit (a write's, which its admission
    /// covered).
    pub(super) fn thawed(
        &self,
        hot: usize,
        meta: usize,
        chunks: usize,
        granted: Option<MemoryGrant>,
    ) {
        let index = table_index(self.table);
        self.thaws.gained[index].fetch_add(hot, Ordering::Relaxed);
        self.thaws.freed_meta[index].fetch_add(meta, Ordering::Relaxed);
        self.thaws.freed_chunks[index].fetch_add(chunks, Ordering::Relaxed);
        self.thaws.pending.store(true, Ordering::Release);
        if hot > 0
            && let Some(manager) = self.thaws.manager.get()
        {
            let grant = match granted {
                Some(mut grant) => {
                    let size = grant.size();
                    if size < hot {
                        grant.grow_forced(hot - size);
                    } else {
                        grant.shrink(size - hot);
                    }
                    grant
                }
                None => manager.reserve_forced(hot, MemoryRegion::GraphStorage),
            };
            let mut held = self.thaws.grant.lock();
            match held.as_mut() {
                Some(held) => held.merge(grant),
                None => *held = Some(grant),
            }
        }
    }
}

/// A cold chunk's bytes, owning the grant that counts them: released when
/// the last clone (a reader's chunk, a point view's body) drops.
struct GrantedChunk {
    bytes: Bytes,
    _grant: MemoryGrant,
}

impl AsRef<[u8]> for GrantedChunk {
    fn as_ref(&self) -> &[u8] {
        &self.bytes
    }
}

/// `bytes`, a cold chunk's, holding a grant of their length from `memory`
/// for as long as any clone of them lives; as they are without a manager.
pub(super) fn granted_chunk(bytes: Bytes, memory: Option<&Arc<BufferManager>>) -> Bytes {
    match memory {
        Some(manager) if !bytes.is_empty() => {
            let grant = manager.reserve_forced(bytes.len(), MemoryRegion::GraphStorage);
            Bytes::from_owner(GrantedChunk {
                bytes,
                _grant: grant,
            })
        }
        _ => bytes,
    }
}

/// The buffer manager of an account, which shows as such in its debug form.
struct Manager(Arc<BufferManager>);

impl std::fmt::Debug for Manager {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("BufferManager")
    }
}

/// Bytes an op needs that the account could not admit.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct Needed(pub(super) usize);

/// What one row-group store holds, counted where it grows, under the
/// store's write lock (plain adds), with one grant per region it holds.
#[derive(Debug, Default)]
pub(super) struct StoreAccount {
    /// The counted bytes, line by line, without the thaws not folded in.
    counted: Lines,
    /// The counted bytes of the graph storage lines, of the index buffer
    /// lines and of the cold chunks (see `REGION_OF`), kept with them.
    totals: [usize; 3],
    /// The grant of the graph storage lines.
    storage: Option<MemoryGrant>,
    /// The grant of the index buffer lines.
    indexes: Option<MemoryGrant>,
    /// The buffer manager, `None` for a store no database holds (tests,
    /// tools): it counts and takes no grant.
    manager: Option<Manager>,
    /// What reads decoded since the last fold.
    thaws: Arc<Thaws>,
    /// Per table and property key, the row groups with a column of the key.
    key_groups: [Vec<u32>; 2],
    /// Per table, the property keys some row group has a column of.
    key_columns: [usize; 2],
    /// Per region, what the ops admitted since the last settle carry: not
    /// counted yet (the bracket of their writes is open, see `Bracket`).
    pending: [usize; 2],
    /// Per region, what the last admitted op asked for besides a step: the
    /// headroom a settle leaves in the grant.
    headroom: [usize; 2],
}

/// The multiple of `step` (the manager's grant step) at or above `bytes`,
/// saturating.
fn step_above(bytes: usize, step: usize) -> usize {
    bytes.checked_next_multiple_of(step).unwrap_or(usize::MAX)
}

/// `bytes`, a total of counted lines, which no store reaches past
/// `isize::MAX`: a wrapped total is a count that drifted below its parts, so
/// it is taken as 0 (and a debug build stops) rather than grown into a huge
/// grant.
fn sane(bytes: usize) -> usize {
    let sane = isize::try_from(bytes).is_ok();
    debug_assert!(sane, "a counted total wrapped: {bytes}");
    if sane { bytes } else { 0 }
}

impl StoreAccount {
    /// Takes its grants from `manager` from now on, covering what it counts
    /// already. A store attached once keeps its manager.
    pub(super) fn attach(&mut self, manager: Arc<BufferManager>) {
        if self.manager.is_some() {
            return;
        }
        // Set with the account's own: the thaws hold no other manager.
        self.thaws.manager.set(Arc::clone(&manager));
        self.manager = Some(Manager(manager));
        self.settle();
    }

    /// Where a cold column of `table` reports its thaw.
    #[must_use]
    /// Holds this store's reads' thaws unrefused (with `others`, its named
    /// graphs') until the guard drops.
    pub(super) fn unrefused(&self, others: impl IntoIterator<Item = Arc<Thaws>>) -> Unrefused {
        let all: Vec<Arc<Thaws>> = std::iter::once(Arc::clone(&self.thaws))
            .chain(others)
            .collect();
        for thaws in &all {
            thaws.unrefused.fetch_add(1, Ordering::AcqRel);
        }
        Unrefused(all)
    }

    /// The thaws of this store, for a guard of its root's
    /// ([`unrefused`](Self::unrefused)).
    pub(super) fn thaws(&self) -> Arc<Thaws> {
        Arc::clone(&self.thaws)
    }

    pub(super) fn sink(&self, table: Table) -> ThawSink {
        ThawSink {
            thaws: Arc::clone(&self.thaws),
            table,
        }
    }

    /// Adds what changed between two readings of the same parts: `after`
    /// less `before`, line by line.
    #[inline]
    pub(super) fn moved(&mut self, before: &Lines, after: &Lines) {
        for (index, (after, before)) in after.iter().zip(before).enumerate() {
            self.add_to(index, after.wrapping_sub(*before));
        }
    }

    /// The counted bytes, line by line, without the thaws not folded in: for
    /// a debug build's checks.
    #[cfg(debug_assertions)]
    pub(super) fn counted_lines(&self) -> Lines {
        self.counted
    }

    /// Adds `bytes` a part on `line` grew by, counted where it grew (a shared
    /// part: a dictionary, the counts, the collect's lists, a commit's marks).
    #[inline]
    pub(super) fn grown(&mut self, line: Line, bytes: usize) {
        if bytes != 0 {
            self.add_to(line.index(), bytes);
        }
    }

    /// Adds what one part on `line` changed by: from `before` to `after`.
    #[inline]
    pub(super) fn moved_line(&mut self, line: Line, before: usize, after: usize) {
        self.add_to(line.index(), after.wrapping_sub(before));
    }

    /// Takes `lines` out: what a row group dropped whole held, by its walk.
    pub(super) fn subtract(&mut self, lines: &Lines) {
        for (index, bytes) in lines.iter().enumerate() {
            self.add_to(index, 0_usize.wrapping_sub(*bytes));
        }
    }

    /// Adds `delta` (a difference, wrapping) to line `index` and its total.
    #[inline]
    fn add_to(&mut self, index: usize, delta: usize) {
        self.counted[index] = self.counted[index].wrapping_add(delta);
        let total = &mut self.totals[REGION_OF[index]];
        *total = total.wrapping_add(delta);
    }

    /// Counts `lines`, a walk of everything the store holds, in place of
    /// what it counted: a load, or a change of the whole store. The thaws
    /// before are in the walk.
    pub(super) fn reset(&mut self, lines: Lines) {
        self.counted = [0; LINES];
        self.totals = [0; 3];
        for (index, bytes) in lines.iter().enumerate() {
            self.add_to(index, *bytes);
        }
        self.take_thaws();
        self.adopt_thaw_grant();
        self.settle();
    }

    /// Notes that a row group of `table` made a column of key `key`.
    pub(super) fn column_made(&mut self, table: Table, key: u32) {
        let place = table_index(table);
        let groups = &mut self.key_groups[place];
        let index = key as usize;
        if index >= groups.len() {
            groups.resize(index + 1, 0);
        }
        if groups[index] == 0 {
            self.key_columns[place] += 1;
        }
        groups[index] += 1;
    }

    /// Notes that a row group of `table` with columns of `keys` went.
    pub(super) fn columns_gone(&mut self, table: Table, keys: impl IntoIterator<Item = u32>) {
        let place = table_index(table);
        for key in keys {
            let Some(groups) = self.key_groups[place].get_mut(key as usize) else {
                debug_assert!(false, "a column of key {key} no group made");
                continue;
            };
            debug_assert!(*groups > 0, "a column of key {key} gone twice");
            *groups = groups.saturating_sub(1);
            if *groups == 0 {
                self.key_columns[place] = self.key_columns[place].saturating_sub(1);
            }
        }
    }

    /// Counts the columns of `table` again: `keys` holds each row group's
    /// keys.
    pub(super) fn set_key_columns(&mut self, table: Table, keys: impl IntoIterator<Item = u32>) {
        let place = table_index(table);
        self.key_groups[place].clear();
        self.key_columns[place] = 0;
        for key in keys {
            self.column_made(table, key);
        }
    }

    /// The property keys of `table` some row group has a column of.
    #[must_use]
    pub(super) fn key_column_count(&self, table: Table) -> usize {
        self.key_columns[table_index(table)]
    }

    /// Whether it counts cold chunks, as of the last fold of the thaws: two
    /// loads, which spare a write the look for cold columns while the store
    /// holds none.
    #[must_use]
    pub(super) fn holds_cold(&self) -> bool {
        self.counted[Line::NodeCold.index()] != 0 || self.counted[Line::EdgeCold.index()] != 0
    }

    /// The counted bytes, line by line, with the thaws not folded in yet:
    /// what the report shows.
    #[must_use]
    pub(super) fn report(&self) -> Lines {
        let mut lines = self.counted;
        for table in [Table::Nodes, Table::Edges] {
            let index = table_index(table);
            let (columns, cold, chunks) = Line::columns_of(table);
            let thaws = &self.thaws;
            lines[columns.index()] =
                lines[columns.index()].wrapping_add(thaws.gained[index].load(Ordering::Relaxed));
            lines[cold.index()] =
                lines[cold.index()].wrapping_sub(thaws.freed_meta[index].load(Ordering::Relaxed));
            lines[chunks.index()] = lines[chunks.index()]
                .wrapping_sub(thaws.freed_chunks[index].load(Ordering::Relaxed));
        }
        lines
    }

    /// The bytes its grants hold: the two regions' grants, the thaws' and
    /// its cold chunks' (each chunk holds a grant of its bytes). Without a
    /// manager, which grants nothing, the bytes it counts.
    #[must_use]
    pub(super) fn granted(&self) -> usize {
        let report = self.report();
        if self.manager.is_none() {
            return report
                .iter()
                .fold(0, |total, bytes| total.wrapping_add(*bytes));
        }
        let held = |grant: &Option<MemoryGrant>| grant.as_ref().map_or(0, MemoryGrant::size);
        let thawed = self
            .thaws
            .grant
            .lock()
            .as_ref()
            .map_or(0, MemoryGrant::size);
        held(&self.storage)
            + held(&self.indexes)
            + thawed
            + report[Line::NodeColdChunks.index()]
            + report[Line::EdgeColdChunks.index()]
    }

    /// Admits a refusable op that carries about `payload` bytes of values
    /// and thaws about `thaw` bytes, and may grow the index buffers
    /// (`indexes`): keeps a step and that much headroom, past what it counts
    /// and what the ops admitted since the last settle carry, in each grant
    /// the op may grow, growing a short grant to the next step. Two compares
    /// when the grants have room. When admission refuses, it gives back what
    /// it grew for the op, nothing changed, and the op needs the returned
    /// bytes.
    pub(super) fn admit(
        &mut self,
        payload: usize,
        thaw: usize,
        indexes: bool,
    ) -> Result<(), Needed> {
        let need = payload.saturating_add(thaw);
        #[cfg(test)]
        if testing::refuses() {
            return Err(Needed(need.max(1)));
        }
        let Some(Manager(manager)) = &self.manager else {
            return Ok(());
        };
        let step = manager.grant_step();
        let [storage, index, _] = self.totals;
        let wants = [
            (MemoryRegion::GraphStorage, storage, need),
            (
                MemoryRegion::IndexBuffers,
                index,
                if indexes { payload } else { 0 },
            ),
        ];
        let mut grown = [0_usize; 2];
        for (place, (region, counted, op)) in wants.into_iter().enumerate() {
            if place == 1 && !indexes {
                continue;
            }
            let wanted = sane(counted)
                .saturating_add(self.pending[place])
                .saturating_add(step)
                .saturating_add(op);
            let grant = match place {
                0 => &mut self.storage,
                _ => &mut self.indexes,
            };
            let held = grant.as_ref().map_or(0, MemoryGrant::size);
            if held >= wanted {
                continue;
            }
            let grow = step_above(wanted, step).saturating_sub(held);
            let grant = grant.get_or_insert_with(|| manager.empty_grant(region));
            if ADMISSION_REFUSES && !grant.try_grow(grow) {
                // A refusal changes nothing: what this admission grew goes back.
                if place == 1
                    && let Some(storage) = self.storage.as_mut()
                {
                    storage.shrink(grown[0]);
                }
                return Err(Needed(grow));
            }
            if !ADMISSION_REFUSES {
                grant.grow_forced(grow);
            }
            grown[place] = grow;
        }
        self.pending[0] = self.pending[0].saturating_add(need);
        self.headroom[0] = need;
        if indexes {
            self.pending[1] = self.pending[1].saturating_add(payload);
            self.headroom[1] = payload;
        }
        Ok(())
    }

    /// Brings each grant up to the bytes its lines count (forced: the ops
    /// were admitted, or must not fail), after folding in the thaws, and
    /// gives back what a grant holds past two steps and the last admitted
    /// op's headroom over them: `counted <= grant < counted + 2 steps +
    /// headroom`.
    pub(super) fn settle(&mut self) {
        self.settle_to(true);
    }

    /// [`settle`](Self::settle) without the hysteresis: a grant keeps no
    /// more than the step its counted bytes round up to, so what a spill or
    /// a compression freed leaves the buffer manager's count at once (the
    /// eviction cycle relies on it).
    pub(super) fn settle_tight(&mut self) {
        self.settle_to(false);
    }

    /// [`settle`](Self::settle), with the hysteresis (`slack`) or without.
    fn settle_to(&mut self, slack: bool) {
        self.fold_thaws();
        self.pending = [0; 2];
        let Some(Manager(manager)) = &self.manager else {
            return;
        };
        let step = manager.grant_step();
        let [storage, index, _] = self.totals;
        for (place, (region, counted)) in [
            (MemoryRegion::GraphStorage, storage),
            (MemoryRegion::IndexBuffers, index),
        ]
        .into_iter()
        .enumerate()
        {
            let counted = sane(counted);
            let headroom = if slack { self.headroom[place] } else { 0 };
            let grant = match place {
                0 => &mut self.storage,
                _ => &mut self.indexes,
            };
            let held = grant.as_ref().map_or(0, MemoryGrant::size);
            if held < counted {
                grant
                    .get_or_insert_with(|| manager.empty_grant(region))
                    .grow_forced(step_above(counted, step).saturating_sub(held));
                continue;
            }
            let (bound, kept) = if slack {
                (
                    counted.saturating_add(2 * step).saturating_add(headroom),
                    step_above(counted.saturating_add(step).saturating_add(headroom), step),
                )
            } else {
                let kept = step_above(counted, step);
                (kept.saturating_add(1), kept)
            };
            if held >= bound
                && let Some(grant) = grant.as_mut()
            {
                grant.shrink(held.saturating_sub(kept));
            }
        }
    }

    /// Folds the thaws since the last fold into the lines, and their grant
    /// into the storage grant. Under the store's write lock: no read thaws
    /// meanwhile.
    fn fold_thaws(&mut self) {
        if !self.thaws.pending.load(Ordering::Acquire) {
            return;
        }
        let thawed = self.take_thaws();
        for (index, bytes) in thawed.iter().enumerate() {
            self.add_to(index, *bytes);
        }
        self.adopt_thaw_grant();
    }

    /// Takes the grant the thaws took into the storage grant.
    fn adopt_thaw_grant(&mut self) {
        let grant = self.thaws.grant.lock().take();
        if let Some(grant) = grant {
            match self.storage.as_mut() {
                Some(storage) => storage.merge(grant),
                None => self.storage = Some(grant),
            }
        }
    }

    /// Takes the thaws' counters, as lines to add to the account: zero
    /// again after.
    fn take_thaws(&mut self) -> Lines {
        let mut lines = [0; LINES];
        self.thaws.pending.store(false, Ordering::Release);
        for table in [Table::Nodes, Table::Edges] {
            let index = table_index(table);
            let (columns, cold, chunks) = Line::columns_of(table);
            lines[columns.index()] = self.thaws.gained[index].swap(0, Ordering::Relaxed);
            lines[cold.index()] =
                0_usize.wrapping_sub(self.thaws.freed_meta[index].swap(0, Ordering::Relaxed));
            lines[chunks.index()] =
                0_usize.wrapping_sub(self.thaws.freed_chunks[index].swap(0, Ordering::Relaxed));
        }
        lines
    }
}

/// A test hook: admission refuses ops on this thread (see
/// [`refuse_after`](testing::refuse_after)), so a test can refuse an op
/// partway through a row while admission refuses nothing yet.
#[cfg(test)]
pub(super) mod testing {
    use std::cell::Cell;

    thread_local! {
        /// The admissions to let through, then the ones to refuse.
        static REFUSALS: Cell<(u32, u32)> = const { Cell::new((0, 0)) };
    }

    /// Lets the next `admitted` admissions on this thread through, then
    /// refuses `refused` of them.
    pub(in crate::graph::rowgroup) fn refuse_after(admitted: u32, refused: u32) {
        REFUSALS.with(|refusals| refusals.set((admitted, refused)));
    }

    /// Whether this admission is refused.
    pub(super) fn refuses() -> bool {
        REFUSALS.with(|refusals| {
            let (admitted, refused) = refusals.get();
            if admitted > 0 {
                refusals.set((admitted - 1, refused));
                false
            } else if refused > 0 {
                refusals.set((0, refused - 1));
                true
            } else {
                false
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use grafeo_common::memory::buffer::{BufferManager, MemoryRegion};
    use grafeo_common::storage::{ChunkCaps, ImageSource, MemoryImage, SectionType};
    use grafeo_common::types::{EdgeId, NodeId, ReadView, Value};

    use crate::graph::conformance::{Tx, people};
    use crate::graph::rowgroup::RowGroupStore;
    use crate::graph::rowgroup::section::LPG_SECTION_VERSION;
    use crate::graph::traits::GraphStore;

    /// The account of `store` and its named graphs equals their walks.
    fn assert_counted(store: &RowGroupStore, what: &str) {
        assert_eq!(
            store.account_mismatches(),
            Vec::<(String, usize, usize)>::new(),
            "{what}: the lines that differ (counted, walked)"
        );
    }

    /// The grants of `manager`'s storage and index regions hold at least
    /// what `store` counts.
    fn assert_granted(store: &RowGroupStore, manager: &BufferManager, what: &str) {
        let stats = manager.stats();
        let granted = stats.region_usage(MemoryRegion::GraphStorage)
            + stats.region_usage(MemoryRegion::IndexBuffers);
        let held = store.memory().total_bytes();
        assert!(granted >= held, "{what}: {granted} granted for {held}");
    }

    fn attached() -> (RowGroupStore, Arc<BufferManager>) {
        let store = RowGroupStore::new();
        let manager = BufferManager::with_budget(1 << 30);
        store.attach_memory(Arc::clone(&manager));
        (store, manager)
    }

    /// Nodes `count` named `who`, each with a string and an integer, and an
    /// edge between each two in a row: the nodes and the edges.
    fn write_chain(
        store: &RowGroupStore,
        id: u64,
        who: &str,
        count: i64,
    ) -> (Vec<NodeId>, Vec<EdgeId>) {
        let mut tx = Tx::begin(store, id);
        let nodes: Vec<NodeId> = (0..count)
            .map(|i| {
                tx.create_node(
                    &["Person"],
                    &[
                        ("name", Value::from(format!("{who} {i}").as_str())),
                        ("i", Value::Int64(i)),
                    ],
                )
                .unwrap()
            })
            .collect();
        let edges = nodes
            .windows(2)
            .map(|pair| {
                tx.create_edge(pair[0], pair[1], "KNOWS", &[("w", Value::Int64(1))])
                    .unwrap()
            })
            .collect();
        tx.commit().unwrap();
        (nodes, edges)
    }

    /// A clear and a rewrite of the same rows: once the collect takes the old
    /// rows out, the slots below the new ones go back (edge ends, hot values,
    /// adjacency offsets, each at least one vector of rows), and the account
    /// follows the walk.
    #[test]
    fn a_dead_head_goes_back_at_the_collect_and_the_account_follows() {
        const ROWS: i64 = 2_100;
        const VECTOR: usize = 2_048;
        let (store, manager) = attached();
        let (old, old_edges) = write_chain(&store, 10, "Alix", ROWS);
        let (new, _) = write_chain(&store, 11, "Gus", ROWS);
        let both = store.memory();
        // The chain's edges, then its nodes: a detach delete would read the
        // whole store's image for each node.
        let mut tx = Tx::begin(&store, 12);
        for edge in old_edges {
            tx.delete_edge(edge).unwrap();
        }
        for node in &old {
            tx.delete_node(*node).unwrap();
        }
        tx.commit().unwrap();
        store.collect(store.current_epoch());
        assert_counted(&store, "the collect that trims");
        assert_granted(&store, &manager, "after the trim");
        let after = store.memory();
        // A slot of an edge's ends, of a hot integer or string, of an offset.
        let freed = [
            (
                "edge ends",
                both.edge_tables_bytes,
                after.edge_tables_bytes,
                VECTOR * 20,
            ),
            (
                "node values",
                both.node_columns_bytes,
                after.node_columns_bytes,
                VECTOR * 16,
            ),
            (
                "edge values",
                both.edge_columns_bytes,
                after.edge_columns_bytes,
                VECTOR * 8,
            ),
            (
                "outgoing",
                both.outgoing_bytes,
                after.outgoing_bytes,
                VECTOR * 8,
            ),
            (
                "incoming",
                both.incoming_bytes,
                after.incoming_bytes,
                VECTOR * 8,
            ),
        ];
        for (what, before, after, slots) in freed {
            assert!(
                after + slots <= before,
                "{what}: {after} after the trim, {before} with both chains"
            );
        }
        assert_eq!(new.len(), old.len());
        let names: Vec<Value> = new
            .iter()
            .take(3)
            .map(|node| {
                store
                    .node_property(*node, "name", ReadView::latest())
                    .unwrap()
                    .unwrap_or(Value::Null)
            })
            .collect();
        assert_eq!(
            names,
            [
                Value::from("Gus 0"),
                Value::from("Gus 1"),
                Value::from("Gus 2")
            ]
        );
    }

    /// Every kind of write, its commit, its rollback and the collect keep the
    /// account equal to the walk.
    #[test]
    fn the_account_follows_each_kind_of_write() {
        let (store, manager) = attached();
        let people = people(&store).unwrap();
        assert_counted(&store, "people");

        let mut tx = Tx::begin(&store, 10);
        let mut nodes = Vec::new();
        for i in 0..300_i64 {
            let value = match i % 4 {
                0 => Value::from(format!("Vincent {i}").as_str()),
                1 => Value::List(vec![Value::Int64(i), Value::from("Jules")].into()),
                2 => Value::Vector(vec![1.0_f32, 0.5, 0.25].into()),
                _ => Value::Float64(0.5),
            };
            nodes.push(
                tx.create_node(&["Person"], &[("v", value), ("i", Value::Int64(i))])
                    .unwrap(),
            );
        }
        for pair in nodes.windows(2) {
            tx.create_edge(pair[0], pair[1], "KNOWS", &[("w", Value::Int64(1))])
                .unwrap();
        }
        assert_counted(&store, "an open transaction's creates");
        tx.commit().unwrap();
        assert_counted(&store, "their commit");

        let mut tx = Tx::begin(&store, 11);
        for (i, node) in nodes.iter().enumerate().take(100) {
            tx.set_node(*node, "v", Value::from(format!("Mia {i}").as_str()))
                .unwrap();
            tx.add_label(*node, "Driver").unwrap();
        }
        tx.set_node(people.alix, "age", Value::Int64(40)).unwrap();
        assert_counted(&store, "values and labels kept for readers");
        tx.roll_back().unwrap();
        assert_counted(&store, "their rollback");

        let mut tx = Tx::begin(&store, 12);
        for node in nodes.iter().take(50) {
            tx.set_node(*node, "i", Value::from("Gus")).unwrap();
            tx.remove_label(*node, "Person").unwrap();
        }
        tx.commit().unwrap();
        assert_counted(&store, "a type change and a label removal");

        let mut tx = Tx::begin(&store, 13);
        for node in nodes.iter().take(20) {
            tx.detach_delete(*node).unwrap();
        }
        tx.commit().unwrap();
        assert_counted(&store, "deletes");
        store.collect(store.current_epoch());
        assert_counted(&store, "the collect");
        assert_granted(&store, &manager, "after the collect");
    }

    /// An op refused partway through a row (the third of five writes, refused
    /// again on its retry) changes nothing; the statement's undo restores
    /// the row, and the account equals the walk after it, and after the rest
    /// of the transaction commits.
    #[test]
    fn a_refusal_partway_through_a_row_leaves_the_account_on_the_walk() {
        let (store, manager) = attached();
        let people = people(&store).unwrap();
        store.collect(store.current_epoch());
        assert_counted(&store, "people");
        let mut tx = Tx::begin(&store, 10);
        let mark = tx.mark();
        super::testing::refuse_after(2, 2);
        let vincent = tx
            .create_node(&["Person"], &[("name", Value::from("Vincent"))])
            .unwrap();
        tx.set_node(vincent, "city", Value::from("Paris")).unwrap();
        let refused = tx.set_node(vincent, "age", Value::Int64(40));
        super::testing::refuse_after(0, 0);
        assert!(refused.is_err(), "the third write is refused: {refused:?}");
        tx.roll_back_to(mark).unwrap();
        assert_counted(&store, "the statement undone after the refusal");
        assert_granted(&store, &manager, "after the refusal");
        tx.set_node(people.alix, "age", Value::Int64(41)).unwrap();
        tx.set_node(people.alix, "city", Value::from("Berlin"))
            .unwrap();
        tx.commit().unwrap();
        assert_counted(&store, "the rest of the transaction committed");
        assert_granted(&store, &manager, "after the commit");
    }

    /// An index built over existing data and kept by the writes is counted.
    #[test]
    fn a_property_index_is_counted() {
        let (store, _manager) = attached();
        let mut tx = Tx::begin(&store, 10);
        let nodes: Vec<NodeId> = (0..200_i64)
            .map(|i| {
                tx.create_node(&["Person"], &[("id", Value::Int64(i % 37))])
                    .unwrap()
            })
            .collect();
        tx.commit().unwrap();
        store.create_property_index("id");
        store.create_label_index("Person", "id");
        assert_counted(&store, "the index build");
        let mut tx = Tx::begin(&store, 11);
        for node in &nodes {
            tx.set_node(*node, "id", Value::from("Amsterdam")).unwrap();
        }
        tx.commit().unwrap();
        store.collect(store.current_epoch());
        assert_counted(&store, "the index kept by the writes");
        assert!(store.drop_property_index("id"));
        assert_counted(&store, "the drop");
    }

    /// A load counts what it loaded from one walk, its cold chunks hold
    /// their grants, and a read that thaws a column is counted at once.
    #[test]
    fn a_load_and_a_thaw_are_counted() {
        let source = RowGroupStore::new();
        let mut tx = Tx::begin(&source, 10);
        for i in 0..500_i64 {
            tx.create_node(
                &["City"],
                &[
                    ("name", Value::from(format!("Paris {i}").as_str())),
                    ("i", Value::Int64(i)),
                ],
            )
            .unwrap();
        }
        tx.commit().unwrap();
        let mut image = MemoryImage::new();
        image
            .begin_section(SectionType::LpgStore, LPG_SECTION_VERSION)
            .unwrap();
        source
            .write_section(ChunkCaps::DEFAULT, &mut image)
            .unwrap();

        let (loaded, manager) = attached();
        loaded
            .read_section(&*image.section_source(SectionType::LpgStore).unwrap())
            .unwrap();
        assert!(loaded.cold_columns() > 0, "the load keeps its chunks cold");
        assert_counted(&loaded, "the load");
        assert_granted(&loaded, &manager, "the load");
        let name = loaded
            .node_property(NodeId::new(7), "name", ReadView::latest())
            .unwrap();
        assert_eq!(name, Some(Value::from("Paris 7")));
        assert_counted(&loaded, "a read that thawed the column");
        assert_granted(&loaded, &manager, "the thaw");
        let mut tx = Tx::begin(&loaded, 20);
        tx.set_node(NodeId::new(9), "i", Value::Int64(-9)).unwrap();
        tx.commit().unwrap();
        assert_counted(&loaded, "a write after the thaw");
        loaded.compress().unwrap();
        assert_counted(&loaded, "a compression");
        loaded.collect_all(loaded.current_epoch());
        assert_counted(&loaded, "a collect of every vector");
    }
}
