//! Degrees of a label's nodes, read from a sample and cached, for the planner.
//!
//! A sample reads the committed state only, so a plan does not depend on who plans
//! first, and it is a function of the data only: the same ranks on every platform,
//! whichever chunks are resident. It costs no write and no file format.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

use grafeo_common::memory::buffer::{BufferManager, MemoryGrant, MemoryRegion};
use grafeo_common::types::{ArcStr, NodeId, ReadView};
use grafeo_common::utils::hash::FxHashMap;
use parking_lot::Mutex;

use super::RowGroupStore;
use crate::graph::Direction;
use crate::graph::traits::{DEGREE_SAMPLE_NODES, GraphStore};

/// The most nodes a sample reads. A label of at most this many is read whole.
const SAMPLE_NODES: usize = DEGREE_SAMPLE_NODES;

/// The least change of a large label's counts that resamples it.
const LEAST_CHANGE: u64 = 16;

/// The most edges a sample of reached ends keeps.
const SAMPLE_EDGES: u64 = 256;

/// The longest list a first-stage sample counts: past it the entries it holds stand for
/// its edges, without going through them.
const COUNTED_LIST: u64 = 64;

/// The most values of reached ends a store holds: past it they are read again, so a
/// long-running process with new types does not grow for ever.
const MOST_REACHED_ENTRIES: usize = 256;

/// One sampled degree, with the counts it was sampled at.
#[derive(Debug, Clone, Copy)]
struct Sample {
    degree: f64,
    /// The largest count the sample saw.
    max: f64,
    label_nodes: u64,
    type_edges: u64,
}

/// What a reached degree is the degree of: the start's label, the edge types and way the
/// ends are reached by, the edge types and way counted there. Types sorted, as a pattern
/// may write them in any order.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct ReachedKey {
    start_label: Option<String>,
    via_types: Vec<String>,
    via_direction: Direction,
    on_types: Vec<String>,
    on_direction: Direction,
}

/// A reached degree, with the counts it was sampled at. `degree` is `None` for a read that
/// answers nothing (a 0 from part of the edges), which also costs nothing to hold.
#[derive(Debug, Clone, Copy)]
struct ReachedSample {
    degree: Option<f64>,
    start_nodes: u64,
    via_edges: u64,
    on_edges: u64,
}

/// The samples a store holds, by label, edge type and direction, and of reached ends. Not
/// persisted.
#[derive(Debug, Default)]
pub(super) struct DegreeSamples {
    samples: Mutex<FxHashMap<(String, String, Direction), Sample>>,
    reached: Mutex<FxHashMap<ReachedKey, ReachedSample>>,
    taken: AtomicU64,
    /// The reads samples of reached ends have spent.
    reached_reads: AtomicU64,
    /// What [`memory_bytes`](Self::memory_bytes) walks, kept as samples come
    /// and go.
    counted: AtomicUsize,
    /// The grant of those bytes, once a database attaches its memory: a
    /// bounded cache, with a grant of its own.
    grant: Mutex<Option<MemoryGrant>>,
}

/// What a sample of degrees holds on the heap.
fn sample_bytes((label, edge_type, _): &(String, String, Direction)) -> usize {
    label.len() + edge_type.len() + std::mem::size_of::<(String, String, Sample)>()
}

/// What a sample of reached ends holds on the heap.
fn reached_bytes(key: &ReachedKey) -> usize {
    let text = |texts: &[String]| texts.iter().map(String::len).sum::<usize>();
    key.start_label.as_ref().map_or(0, String::len)
        + text(&key.via_types)
        + text(&key.on_types)
        + std::mem::size_of::<(ReachedKey, ReachedSample)>()
}

impl DegreeSamples {
    /// About what the samples hold on the heap, by a walk of every sample:
    /// the oracle of [`counted_bytes`](Self::counted_bytes).
    pub(super) fn memory_bytes(&self) -> usize {
        let samples: usize = self.samples.lock().keys().map(sample_bytes).sum();
        let reached: usize = self.reached.lock().keys().map(reached_bytes).sum();
        samples + reached
    }

    /// [`memory_bytes`](Self::memory_bytes), as the samples keep it: one
    /// load.
    pub(super) fn counted_bytes(&self) -> usize {
        self.counted.load(Ordering::Relaxed)
    }

    /// Takes the grant of its bytes from `manager` from now on.
    pub(super) fn attach(&self, manager: &Arc<BufferManager>) {
        let mut grant = self.grant.lock();
        if grant.is_none() {
            *grant = Some(manager.empty_grant(MemoryRegion::IndexBuffers));
        }
        drop(grant);
        self.settle();
    }

    /// Notes `grown` more bytes, `freed` fewer: called with the lock of the
    /// map that changed held, so a clear never takes out bytes an insert has
    /// not added yet. Saturating: a count never below what it holds.
    fn moved(&self, grown: usize, freed: usize) {
        if grown == freed {
            return;
        }
        let _ = self
            .counted
            .try_update(Ordering::Relaxed, Ordering::Relaxed, |counted| {
                debug_assert!(counted + grown >= freed, "a degree sample count drifted");
                Some((counted + grown).saturating_sub(freed))
            });
    }

    /// Brings the grant to the counted bytes.
    fn settle(&self) {
        // Read under the grant's lock, so a settle that read an older count cannot shrink
        // what a newer one just grew.
        let mut grant = self.grant.lock();
        let counted = self.counted_bytes();
        if let Some(grant) = grant.as_mut() {
            let held = grant.size();
            if held < counted {
                grant.grow_forced(counted - held);
            } else {
                grant.shrink(held - counted);
            }
        }
    }
}

/// The ranks a sample of a label of `count` nodes reads, ascending: the fractional
/// parts of `i` times the golden ratio, scaled to the count, in integer arithmetic.
/// Evenly spread, without lining up with a load order that repeats.
fn ranks(count: u64) -> Vec<u64> {
    ranks_up_to(count, SAMPLE_NODES as u64)
}

/// [`ranks`] with `size` of them.
fn ranks_up_to(count: u64, size: u64) -> Vec<u64> {
    const GOLDEN: u64 = 0x9E37_79B9_7F4A_7C15;
    let mut ranks: Vec<u64> = (1..=size)
        .map(|i| {
            let fraction = u128::from(i.wrapping_mul(GOLDEN));
            u64::try_from((fraction * u128::from(count)) >> 64).unwrap_or(u64::MAX)
        })
        .collect();
    ranks.sort_unstable();
    ranks.dedup();
    ranks
}

/// Whether `then` has moved enough to `now` for a large label's sample to go stale.
fn moved(then: u64, now: u64) -> bool {
    then.abs_diff(now) >= (then / 10).max(LEAST_CHANGE)
}

impl ReachedSample {
    /// Whether the sample still stands at these counts, as [`Sample::stands`] says.
    fn stands(&self, start_nodes: u64, via_edges: u64, on_edges: u64) -> bool {
        if self.start_nodes <= SAMPLE_NODES as u64 {
            return self.start_nodes == start_nodes
                && self.via_edges == via_edges
                && self.on_edges == on_edges;
        }
        !moved(self.start_nodes, start_nodes)
            && !moved(self.via_edges, via_edges)
            && !moved(self.on_edges, on_edges)
    }
}

impl Sample {
    /// Whether the sample still stands at these counts: a label read whole resamples
    /// on any change, at most [`SAMPLE_NODES`] reads; a larger one when a count has
    /// moved by 10 percent or [`LEAST_CHANGE`], whichever is larger.
    fn stands(&self, label_nodes: u64, type_edges: u64) -> bool {
        if self.label_nodes <= SAMPLE_NODES as u64 {
            return self.label_nodes == label_nodes && self.type_edges == type_edges;
        }
        !moved(self.label_nodes, label_nodes) && !moved(self.type_edges, type_edges)
    }
}

impl RowGroupStore {
    /// About [`SAMPLE_NODES`] committed nodes of `label`, and whether they are all of them (see
    /// `sampled_nodes_with_label`): never the label's list.
    fn sample_of_label(&self, label: &str, view: ReadView) -> (Vec<NodeId>, bool) {
        let inner = self.inner.read();
        match inner.label_id(label) {
            Some(label) => {
                inner.sampled_nodes_with_label(label, SAMPLE_NODES, super::Read::view(view))
            }
            None => (Vec::new(), true),
        }
    }

    /// How many samples it has read, for a test to see that a plain query reads
    /// none and a small change reads no more.
    #[doc(hidden)]
    #[must_use]
    pub fn degree_samples_taken(&self) -> u64 {
        self.degree_samples.taken.load(Ordering::Relaxed)
    }

    /// How many reads the samples of reached ends have spent, for a test to see that a
    /// budget holds whatever the lists' lengths.
    #[doc(hidden)]
    #[must_use]
    pub fn degree_sample_reads(&self) -> u64 {
        self.degree_samples.reached_reads.load(Ordering::Relaxed)
    }

    /// The mean count of `edge_type`'s edges in `direction` over a sample of
    /// `label`'s nodes, `None` for a label without a node or a read that fails.
    pub(super) fn sample_degree(
        &self,
        label: &str,
        edge_type: &str,
        direction: Direction,
    ) -> Option<f64> {
        self.degree_sample(label, edge_type, direction)
            .map(|sample| sample.degree)
    }

    /// The largest count the sample of [`sample_degree`](Self::sample_degree) saw, from the same
    /// read.
    pub(super) fn sample_max_degree(
        &self,
        label: &str,
        edge_type: &str,
        direction: Direction,
    ) -> Option<f64> {
        self.degree_sample(label, edge_type, direction)
            .map(|sample| sample.max)
    }

    /// The sample of `label`'s degrees over `edge_type` in `direction`, kept while it stands.
    fn degree_sample(&self, label: &str, edge_type: &str, direction: Direction) -> Option<Sample> {
        let (label_nodes, type_edges) = self.committed_counts(label, edge_type);
        if label_nodes == 0 {
            return None;
        }
        let key = (label.to_string(), edge_type.to_string(), direction);
        if let Some(sample) = self.degree_samples.samples.lock().get(&key)
            && sample.stands(label_nodes, type_edges)
        {
            return Some(*sample);
        }
        let (degree, max) = self.read_sample(label, edge_type, direction)?;
        self.degree_samples.taken.fetch_add(1, Ordering::Relaxed);
        let sample = Sample {
            degree,
            max,
            label_nodes,
            type_edges,
        };
        let bytes = sample_bytes(&key);
        {
            let mut samples = self.degree_samples.samples.lock();
            if samples.insert(key, sample).is_none() {
                self.degree_samples.moved(bytes, 0);
            }
        }
        // Rare: a sample is kept until its counts move. Forced: a sample is a
        // read's, which never fails.
        self.degree_samples.settle();
        Some(sample)
    }

    /// See `GraphStore::sampled_reached_degree_within`.
    pub(super) fn sample_reached_degree(
        &self,
        start_label: Option<&str>,
        via_types: &[&str],
        via_direction: Direction,
        on_types: &[&str],
        on_direction: Direction,
        budget: &mut u64,
    ) -> Option<f64> {
        let sorted = |types: &[&str]| {
            let mut types: Vec<String> = types.iter().map(ToString::to_string).collect();
            types.sort_unstable();
            types
        };
        let key = ReachedKey {
            start_label: start_label.map(str::to_string),
            via_types: sorted(via_types),
            via_direction,
            on_types: sorted(on_types),
            on_direction,
        };
        let (start_nodes, via_edges, on_edges) =
            self.committed_reached_counts(start_label, via_types, on_types);
        if let Some(sample) = self.degree_samples.reached.lock().get(&key)
            && sample.stands(start_nodes, via_edges, on_edges)
        {
            return sample.degree;
        }
        if *budget == 0 {
            return None;
        }
        let (degree, reads) = self.read_reached(&key, *budget)?;
        *budget = budget.saturating_sub(reads);
        self.degree_samples
            .reached_reads
            .fetch_add(reads, Ordering::Relaxed);
        self.degree_samples.taken.fetch_add(1, Ordering::Relaxed);
        let mut reached = self.degree_samples.reached.lock();
        let mut freed = 0;
        if reached.len() >= MOST_REACHED_ENTRIES {
            freed = reached.keys().map(reached_bytes).sum();
            reached.clear();
        }
        let bytes = reached_bytes(&key);
        let replaced = reached.insert(
            key,
            ReachedSample {
                degree,
                start_nodes,
                via_edges,
                on_edges,
            },
        );
        let grown = if replaced.is_none() { bytes } else { 0 };
        self.degree_samples.moved(grown, freed);
        drop(reached);
        self.degree_samples.settle();
        degree
    }

    /// See `GraphStore::far_walk_back_within`. It holds nothing between plans: it reads list
    /// lengths alone, and only for a plan whose source's sample reached nothing.
    pub(super) fn sample_far_walk_back(
        &self,
        far_label: &str,
        types: &[&str],
        back: Direction,
        budget: &mut u64,
    ) -> Option<f64> {
        if types.is_empty() {
            return None;
        }
        let (far_nodes, type_edges, _) =
            self.committed_reached_counts(Some(far_label), types, types);
        if far_nodes == 0 {
            return None;
        }
        let view = ReadView::latest();
        let (sample, _) = self.sample_of_label(far_label, view);
        let reads = sample.len() as u64;
        if sample.is_empty() || *budget < reads {
            return None;
        }
        *budget -= reads;
        self.degree_samples
            .reached_reads
            .fetch_add(reads, Ordering::Relaxed);
        let names: Vec<ArcStr> = types.iter().map(|name| ArcStr::from(*name)).collect();
        let (mut entries, mut squares) = (0.0_f64, 0.0_f64);
        for node in &sample {
            let length = self.inner.read().entries_of_types_at_most(
                *node,
                back,
                &names,
                super::Read::view(view),
            )? as f64;
            entries += length;
            squares += length * length;
        }
        let seen = entries / sample.len() as f64 * far_nodes as f64;
        // Both ways, each edge is in two lists.
        let lists = if back == Direction::Both { 2.0 } else { 1.0 };
        let hidden = (type_edges as f64 * lists - seen).max(0.0);
        let weighted = if entries > 0.0 {
            squares / entries
        } else {
            0.0
        };
        Some(weighted + hidden)
    }

    /// The nodes a reached degree starts from and the edges of its two hops, as committed.
    fn committed_reached_counts(
        &self,
        start_label: Option<&str>,
        via_types: &[&str],
        on_types: &[&str],
    ) -> (u64, u64, u64) {
        let inner = self.inner.read();
        let count = |counts: &[i64], id: Option<u32>| {
            id.and_then(|id| counts.get(id as usize))
                .map_or(0, |count| u64::try_from(*count).unwrap_or(0))
        };
        let edges = |types: &[&str]| -> u64 {
            if types.is_empty() {
                return u64::try_from(inner.counts.edges).unwrap_or(0);
            }
            types
                .iter()
                .map(|name| count(&inner.counts.edge_types, inner.edge_types.get_id(name)))
                .sum()
        };
        let starts = match start_label {
            Some(label) => count(&inner.counts.labels, inner.label_id(label)),
            None => u64::try_from(inner.counts.nodes).unwrap_or(0),
        };
        (starts, edges(via_types), edges(on_types))
    }

    /// Reads a reached degree: the start nodes' lists as one sequence in rank order, at
    /// most [`SAMPLE_EDGES`] evenly spread positions of it, so every edge has one chance of
    /// being kept and a hub gets its share by length, and the next hop's count at each end
    /// kept, each distinct end read once. Its value, and the reads it took. Only lists that
    /// hold a kept position are read, one store call at a time.
    ///
    /// `budget` holds whatever the lists' lengths, past a read per start and per kept end: a
    /// quarter of it counts the lists and a quarter the ends' edges, and a list longer than
    /// its share counts as the entries it holds; the rest reads the lists, and a list longer
    /// than its share gives its first edges, a position past them wrapping round. So short of
    /// budget, a value reads high by what those entries hold of removed edges, and from the
    /// edges a hub got first.
    fn read_reached(&self, key: &ReachedKey, budget: u64) -> Option<(Option<f64>, u64)> {
        let view = ReadView::latest();
        let types_of = |names: &[String]| -> Vec<ArcStr> {
            names
                .iter()
                .map(|name| ArcStr::from(name.as_str()))
                .collect()
        };
        let (via, on) = (types_of(&key.via_types), types_of(&key.on_types));
        let (starts, every_start_read) = match &key.start_label {
            Some(label) => self.sample_of_label(label, view),
            // No label: every node is a start, sampled from the list of them.
            None => {
                let nodes = self.node_ids(view).ok()?;
                if nodes.len() <= SAMPLE_NODES {
                    (nodes, true)
                } else {
                    let sample = ranks(nodes.len() as u64)
                        .into_iter()
                        .filter_map(|rank| nodes.get(usize::try_from(rank).ok()?).copied())
                        .collect();
                    (sample, false)
                }
            }
        };
        let share = |spent_up_to: u64, reads: u64, parts: usize| {
            spent_up_to.saturating_sub(reads) / (parts.max(1) as u64)
        };
        let mut reads = 0_u64;
        let mut lengths = Vec::with_capacity(starts.len());
        for (index, start) in starts.iter().enumerate() {
            let within = share(budget / 4, reads, starts.len() - index);
            let (length, spent) =
                self.count_within(*start, key.via_direction, &via, within, view)?;
            lengths.push(length);
            reads += spent;
        }
        let total: u64 = lengths.iter().sum();
        if total == 0 {
            return Some((every_start_read.then_some(0.0), reads));
        }
        let every_edge_read = every_start_read && total <= SAMPLE_EDGES;
        let positions = if total <= SAMPLE_EDGES {
            (0..total).collect()
        } else {
            ranks_up_to(total, SAMPLE_EDGES)
        };
        let mut ends = Vec::with_capacity(positions.len());
        let mut left = positions.as_slice();
        let mut offset = 0_u64;
        for (start, length) in starts.iter().zip(&lengths) {
            let end = offset + length;
            let held = left.partition_point(|position| *position < end);
            if held > 0 {
                let (here, after) = left.split_at(held);
                let spare = share(budget - budget / 4, reads, left.len());
                let within = spare.saturating_mul(held as u64).max(held as u64);
                let wanted = here[held - 1] - offset + 1;
                let limit = usize::try_from(wanted.min(within)).unwrap_or(usize::MAX);
                let list = self.first_ends(*start, key.via_direction, &via, limit, view);
                reads += list.len() as u64;
                ends.extend(here.iter().filter_map(|position| {
                    let at = usize::try_from(position - offset).ok()? % limit;
                    list.get(at).copied()
                }));
                left = after;
            }
            offset = end;
        }
        let mut far_counts: FxHashMap<NodeId, u64> = FxHashMap::default();
        let mut sum = 0_u64;
        for (index, far) in ends.iter().enumerate() {
            let count = match far_counts.get(far) {
                Some(count) => *count,
                None => {
                    let within = share(budget, reads, ends.len() - index);
                    let (count, spent) =
                        self.count_within(*far, key.on_direction, &on, within, view)?;
                    reads += spent;
                    far_counts.insert(*far, count);
                    count
                }
            };
            sum += count;
        }
        let kept = ends.len() as u64;
        if kept == 0 || (sum == 0 && !every_edge_read) {
            return Some((None, reads));
        }
        Some((Some(sum as f64 / kept as f64), reads))
    }

    /// How many edges `node` has of `types`, and the reads that took: a list longer than
    /// `within` is not gone through, and counts as the entries it holds. A store that keeps
    /// no list to read counts them all, as one read.
    fn count_within(
        &self,
        node: NodeId,
        direction: Direction,
        types: &[ArcStr],
        within: u64,
        view: ReadView,
    ) -> Option<(u64, u64)> {
        let entries = self.inner.read().entries_of_types_at_most(
            node,
            direction,
            types,
            super::Read::view(view),
        );
        match entries.map(|entries| entries as u64) {
            Some(entries) if entries > within => Some((entries, 1)),
            entries => {
                let count = self.count_edges(node, direction, types, &[], view).ok()? as u64;
                Some((count, entries.unwrap_or(1).max(1)))
            }
        }
    }

    /// The other ends of the first `limit` edges of `node` of `types`.
    fn first_ends(
        &self,
        node: NodeId,
        direction: Direction,
        types: &[ArcStr],
        limit: usize,
        view: ReadView,
    ) -> Vec<NodeId> {
        self.inner
            .read()
            .first_ends_seen(node, direction, types, limit, super::Read::view(view))
    }

    /// The nodes of `label` and the edges of `edge_type`, as committed.
    fn committed_counts(&self, label: &str, edge_type: &str) -> (u64, u64) {
        let inner = self.inner.read();
        let count = |counts: &[i64], id: Option<u32>| {
            id.and_then(|id| counts.get(id as usize))
                .map_or(0, |count| u64::try_from(*count).unwrap_or(0))
        };
        (
            count(&inner.counts.labels, inner.label_id(label)),
            count(&inner.counts.edge_types, inner.edge_types.get_id(edge_type)),
        )
    }

    fn read_sample(
        &self,
        label: &str,
        edge_type: &str,
        direction: Direction,
    ) -> Option<(f64, f64)> {
        let view = ReadView::latest();
        let types = [ArcStr::from(edge_type)];
        let count = |node| {
            self.count_within(node, direction, &types, COUNTED_LIST, view)
                .map(|(count, _)| count)
        };
        let (sampled, _) = self.sample_of_label(label, view);
        let (mut edges, mut max) = (0, 0);
        for node in &sampled {
            let counted = count(*node)?;
            edges += counted;
            max = max.max(counted);
        }
        (!sampled.is_empty()).then(|| (edges as f64 / sampled.len() as f64, max as f64))
    }
}
