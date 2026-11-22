//! The segments of the paths a search follows, in one arena per search: each
//! path is its last segment, which names the segment before it, so paths
//! share their prefixes and an extension adds one segment. A segment is a
//! 24-byte slot named by a `u32` index; no segment has an allocation or an
//! `Arc` of its own.
//!
//! - **Blocks.** Slots live in blocks sized as a block of the search's queue
//!   (`PathQueue::block_items_within`, a quarter of what a budget under
//!   16 KiB leaves beside the index), each charged to the search's budget
//!   before it is allocated. Blocks never move.
//! - **Holders.** A slot's `children` counts every holder: each extension
//!   (a slot whose parent it is) and each open or kept path that names it.
//!   When the last holder lets go, the slot is freed and its parent loses a
//!   holder, back along the path while counts drop to 0.
//! - **What is charged.** A freed slot goes on a free list and stays charged,
//!   and the next segment takes a freed slot before a new block is allocated.
//!   So the arena's charge is the peak of its live slots, rounded up to a
//!   block, until the row ends: until the search is done with it and gives
//!   it back, or the budget is cleared for the next row. A wholly free block
//!   is not given back sooner.
//! - **Limits.** An index past `u32::MAX - 1` slots is refused as the budget
//!   refuses, never wrapped; at 24 bytes a slot the budget refuses long
//!   before.

use std::mem::size_of;

use grafeo_common::types::{EdgeId, NodeId};

use crate::execution::operators::OperatorError;
use crate::execution::path_budget::PathBudget;
use crate::execution::path_queue::PathQueue;

/// No segment: the parent of a path's first segment, the segment of a path
/// the search does not follow, the end of the free list.
pub(crate) const NO_SEGMENT: u32 = u32::MAX;

/// One segment of a followed path: the node it reaches, the edge it took
/// there (none for a path's first segment), the segment before it and its
/// holders (0 for a free slot, whose `parent` is the next free slot).
#[derive(Clone, Copy)]
struct Slot {
    node: NodeId,
    edge: EdgeId,
    parent: u32,
    children: u32,
}

const _: () = assert!(size_of::<Slot>() == 24);

/// The bytes one segment of a followed path holds: its slot (its earlier
/// segments are shared with the path it extends). Public for capacity
/// accounting and tests.
pub const SEGMENT_BYTES: usize = size_of::<Slot>();

/// The segments of one search's followed paths (see the module doc).
pub(crate) struct SegmentArena {
    /// The blocks, each of `per_block` slots once allocated.
    blocks: Vec<Vec<Slot>>,
    /// The slots a block holds.
    per_block: usize,
    /// The bits of an index that name the slot in its block: a block's
    /// slots fit a power of two, so a lookup is a shift and a mask.
    shift: u32,
    /// The first free slot, or [`NO_SEGMENT`].
    free: u32,
    /// The most slots it names: every index below [`NO_SEGMENT`].
    most: usize,
}

impl SegmentArena {
    /// An empty arena whose blocks are sized for a budget of `limit` bytes.
    pub(crate) fn within(limit: usize) -> Self {
        let per_block = PathQueue::<Slot>::block_items_within(limit);
        Self {
            blocks: Vec::new(),
            per_block,
            shift: per_block.next_power_of_two().trailing_zeros(),
            free: NO_SEGMENT,
            most: NO_SEGMENT as usize,
        }
    }

    /// The arena naming at most `most` slots, so a test reaches the limit.
    #[cfg(test)]
    pub(crate) fn with_most(mut self, most: usize) -> Self {
        self.most = most;
        self
    }

    /// A path's first segment, at `node`, held by the caller.
    pub(crate) fn root(
        &mut self,
        budget: &mut PathBudget,
        node: NodeId,
    ) -> Result<u32, OperatorError> {
        self.add(
            budget,
            Slot {
                node,
                edge: EdgeId::INVALID,
                parent: NO_SEGMENT,
                children: 1,
            },
        )
    }

    /// The segment that extends `parent` over `edge` to `node`, held by the
    /// caller; `parent` gains it as a holder. A refusal changes nothing.
    pub(crate) fn push(
        &mut self,
        budget: &mut PathBudget,
        parent: u32,
        node: NodeId,
        edge: EdgeId,
    ) -> Result<u32, OperatorError> {
        let segment = self.add(
            budget,
            Slot {
                node,
                edge,
                parent,
                children: 1,
            },
        )?;
        self.slot_mut(parent).children += 1;
        Ok(segment)
    }

    /// Puts `slot` in a freed slot, or in a new one, its block charged to
    /// `budget` before it is allocated.
    fn add(&mut self, budget: &mut PathBudget, slot: Slot) -> Result<u32, OperatorError> {
        if self.free != NO_SEGMENT {
            let segment = self.free;
            self.free = self.slot(segment).parent;
            *self.slot_mut(segment) = slot;
            return Ok(segment);
        }
        let room = self
            .blocks
            .last()
            .filter(|block| block.len() < self.per_block)
            .map(Vec::len);
        let (block, at) = room.map_or((self.blocks.len(), 0), |len| (self.blocks.len() - 1, len));
        let index = u32::try_from((block << self.shift) | at)
            .ok()
            .filter(|&index| (index as usize) < self.most)
            .ok_or_else(|| budget.exceeded(SEGMENT_BYTES))?;
        if room.is_none() {
            if self.blocks.len() == self.blocks.capacity() {
                self.grow_index(budget)?;
            }
            let mut new = Vec::new();
            budget.reserve_block(&mut new, self.per_block)?;
            self.blocks.push(new);
        }
        self.blocks[block].push(slot);
        Ok(index)
    }

    /// Doubles the index of blocks, from one, the new one charged before it
    /// is allocated and the old one given back after: no least growth that
    /// steps with the budget, as the queue's index has, which the queue's
    /// block rule makes room for and this second index would pass at 16 KiB.
    fn grow_index(&mut self, budget: &mut PathBudget) -> Result<(), OperatorError> {
        let old = self.blocks.capacity();
        let mut index = Vec::new();
        budget.reserve_block(&mut index, (2 * old).max(1))?;
        index.append(&mut self.blocks);
        self.blocks = index;
        budget.release(old * size_of::<Vec<Slot>>());
        Ok(())
    }

    /// Adds a holder of `segment`: one more path that names it.
    pub(crate) fn hold(&mut self, segment: u32) {
        self.slot_mut(segment).children += 1;
    }

    /// Lets go of one holder of `segment`: a segment with no holder left is
    /// freed, and lets go of its parent in turn. [`NO_SEGMENT`] is no segment.
    pub(crate) fn release(&mut self, mut segment: u32) {
        while segment != NO_SEGMENT {
            let free = self.free;
            let slot = self.slot_mut(segment);
            debug_assert!(
                slot.children > 0,
                "a segment let go of more often than held"
            );
            slot.children -= 1;
            if slot.children > 0 {
                return;
            }
            let parent = std::mem::replace(&mut slot.parent, free);
            self.free = segment;
            segment = parent;
        }
    }

    fn slot(&self, segment: u32) -> &Slot {
        let index = segment as usize;
        &self.blocks[index >> self.shift][index & ((1 << self.shift) - 1)]
    }

    fn slot_mut(&mut self, segment: u32) -> &mut Slot {
        let index = segment as usize;
        let mask = (1 << self.shift) - 1;
        &mut self.blocks[index >> self.shift][index & mask]
    }

    /// The node `segment` reaches.
    pub(crate) fn node(&self, segment: u32) -> NodeId {
        self.slot(segment).node
    }

    /// Whether `segment` is a path's first.
    pub(crate) fn is_first(&self, segment: u32) -> bool {
        self.slot(segment).parent == NO_SEGMENT
    }

    /// `segment` and the segments before it, back to the path's first: a
    /// loop, so a path as long as the graph allows needs no deep stack.
    fn back_to_the_root(&self, segment: u32) -> impl Iterator<Item = &Slot> {
        // A path names each slot once: a walk longer than the slots is a
        // cycle, which a slot freed while still named and then reused makes
        let slots = self.blocks.len() * self.per_block;
        std::iter::successors(Some(self.slot(segment)), |slot| {
            (slot.parent != NO_SEGMENT).then(|| self.slot(slot.parent))
        })
        .enumerate()
        .map(move |(step, slot)| {
            debug_assert!(step < slots, "a path's segments form a cycle");
            slot
        })
    }

    /// The nodes of the path that ends at `segment`, after `depth` edges,
    /// from its first.
    pub(crate) fn collect_nodes(&self, segment: u32, depth: u32) -> Vec<NodeId> {
        let mut nodes = Vec::with_capacity(depth as usize + 1);
        nodes.extend(self.back_to_the_root(segment).map(|slot| slot.node));
        nodes.reverse();
        nodes
    }

    /// The edges of the path that ends at `segment`, after `depth` edges,
    /// from its first.
    pub(crate) fn collect_edges(&self, segment: u32, depth: u32) -> Vec<EdgeId> {
        let mut edges = Vec::with_capacity(depth as usize);
        edges.extend(
            self.back_to_the_root(segment)
                .filter(|slot| slot.parent != NO_SEGMENT)
                .map(|slot| slot.edge),
        );
        edges.reverse();
        edges
    }

    /// Whether the path that ends at `segment` passes `node`.
    pub(crate) fn contains_node(&self, segment: u32, node: NodeId) -> bool {
        self.back_to_the_root(segment).any(|slot| slot.node == node)
    }

    /// Whether the path that ends at `segment` takes `edge`.
    pub(crate) fn contains_edge(&self, segment: u32, edge: EdgeId) -> bool {
        self.back_to_the_root(segment)
            .any(|slot| slot.parent != NO_SEGMENT && slot.edge == edge)
    }

    /// Gives back to `budget` what it was charged, as it goes with the search
    /// that is done with it.
    pub(crate) fn give_back(self, budget: &mut PathBudget) {
        budget.release(self.held_bytes());
    }

    /// The bytes it holds, block by block, with the index of its blocks:
    /// what its budget is charged for it.
    pub(crate) fn held_bytes(&self) -> usize {
        self.blocks
            .iter()
            .map(|block| block.capacity() * SEGMENT_BYTES)
            .sum::<usize>()
            + self.blocks.capacity() * size_of::<Vec<Slot>>()
    }

    /// The slots that hold a segment.
    #[cfg(test)]
    pub(crate) fn live(&self) -> usize {
        self.blocks
            .iter()
            .flatten()
            .filter(|slot| slot.children > 0)
            .count()
    }

    /// The slots a block holds.
    #[cfg(test)]
    pub(crate) fn per_block(&self) -> usize {
        self.per_block
    }

    /// The blocks allocated.
    #[cfg(test)]
    pub(crate) fn blocks(&self) -> usize {
        self.blocks.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::execution::operators::path_budget_message;
    use std::sync::{Arc, Weak};

    /// The `Arc` model of a segment: an `Arc`'s strong count is its holders,
    /// as a slot's `children` is.
    struct Model {
        node: NodeId,
        edge: EdgeId,
        parent: Option<Arc<Model>>,
        index: u32,
    }

    impl Model {
        fn nodes(&self) -> Vec<NodeId> {
            let mut nodes = vec![self.node];
            let mut at = self.parent.as_deref();
            while let Some(model) = at {
                nodes.push(model.node);
                at = model.parent.as_deref();
            }
            nodes.reverse();
            nodes
        }

        fn edges(&self) -> Vec<EdgeId> {
            let mut edges = Vec::new();
            let mut at = Some(self);
            while let Some(model) = at
                && model.parent.is_some()
            {
                edges.push(model.edge);
                at = model.parent.as_deref();
            }
            edges.reverse();
            edges
        }
    }

    /// The model of each slot by index: the last segment put there.
    #[derive(Default)]
    struct Oracle {
        slots: Vec<Weak<Model>>,
        peak: usize,
    }

    impl Oracle {
        fn record(&mut self, model: &Arc<Model>) {
            let index = model.index as usize;
            if self.slots.len() <= index {
                self.slots.resize_with(index + 1, Weak::new);
            }
            self.slots[index] = Arc::downgrade(model);
        }

        /// The counts, the live set, the budget's account, and reuse before
        /// growth: blocks for the peak of live slots, no more.
        fn check(&mut self, arena: &SegmentArena, budget: &PathBudget, step: usize) {
            let mut live = 0;
            for (index, weak) in self.slots.iter().enumerate() {
                let segment = u32::try_from(index).unwrap();
                let holders = weak.strong_count();
                // An index between a block's last slot and the next block's
                // first names no slot
                let Some(&slot) = arena
                    .blocks
                    .get(index >> arena.shift)
                    .and_then(|block| block.get(index & ((1 << arena.shift) - 1)))
                else {
                    assert_eq!(holders, 0, "step {step}: {segment} names no slot");
                    continue;
                };
                assert_eq!(
                    slot.children as usize, holders,
                    "step {step}: holders of {segment}"
                );
                if let Some(model) = weak.upgrade() {
                    live += 1;
                    assert_eq!(model.index, segment, "step {step}");
                    assert_eq!(slot.node, model.node, "step {step}");
                    assert_eq!(
                        slot.parent,
                        model
                            .parent
                            .as_ref()
                            .map_or(NO_SEGMENT, |parent| parent.index),
                        "step {step}: parent of {segment}"
                    );
                }
            }
            assert_eq!(arena.live(), live, "step {step}: the live set");
            self.peak = self.peak.max(live);
            assert_eq!(
                arena.blocks(),
                self.peak.div_ceil(arena.per_block()),
                "step {step}: blocks for a peak of {} live slots",
                self.peak
            );
            assert_eq!(
                budget.used(),
                arena.held_bytes(),
                "step {step}: the account"
            );
        }
    }

    /// A seeded sequence of new paths, extensions, paths held once more,
    /// completed paths and released paths against the `Arc` model, checked
    /// after every step.
    fn matches_the_arc_model(seed: u64, limit: usize, steps: usize) {
        let mut state = seed;
        let mut next = move |bound: usize| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            usize::try_from(state >> 33).unwrap() % bound
        };
        let mut budget = PathBudget::new(1 << 30, None);
        let mut arena = SegmentArena::within(limit);
        let mut oracle = Oracle::default();
        // The paths held: open, kept or being extended
        let mut held: Vec<Arc<Model>> = Vec::new();
        for step in 0..steps {
            // More growth than release for a while, then more release
            let grow = if step < steps / 2 { 6 } else { 3 };
            match next(10) {
                _ if held.is_empty() || next(40) == 0 => {
                    let node = NodeId(u64::try_from(next(1_000)).unwrap());
                    let index = arena.root(&mut budget, node).unwrap();
                    let model = Arc::new(Model {
                        node,
                        edge: EdgeId::INVALID,
                        parent: None,
                        index,
                    });
                    oracle.record(&model);
                    held.push(model);
                }
                roll if roll < grow => {
                    let parent = Arc::clone(&held[next(held.len())]);
                    let node = NodeId(u64::try_from(next(1_000)).unwrap());
                    let edge = EdgeId(u64::try_from(next(1_000)).unwrap());
                    let index = arena.push(&mut budget, parent.index, node, edge).unwrap();
                    let model = Arc::new(Model {
                        node,
                        edge,
                        parent: Some(parent),
                        index,
                    });
                    oracle.record(&model);
                    held.push(model);
                }
                roll if roll == grow => {
                    // A path named once more, as a segment queued for several
                    // states of a search over several edge patterns
                    let model = Arc::clone(&held[next(held.len())]);
                    arena.hold(model.index);
                    held.push(model);
                }
                roll if roll < grow + 3 => {
                    // A path done: read whole, then let go of
                    let model = held.swap_remove(next(held.len()));
                    let depth = u32::try_from(model.edges().len()).unwrap();
                    assert_eq!(arena.collect_nodes(model.index, depth), model.nodes());
                    assert_eq!(arena.collect_edges(model.index, depth), model.edges());
                    for &node in &model.nodes() {
                        assert!(arena.contains_node(model.index, node));
                    }
                    for &edge in &model.edges() {
                        assert!(arena.contains_edge(model.index, edge));
                    }
                    assert!(!arena.contains_edge(model.index, EdgeId::INVALID));
                    assert_eq!(arena.is_first(model.index), model.parent.is_none());
                    arena.release(model.index);
                }
                _ => {
                    let model = held.swap_remove(next(held.len()));
                    arena.release(model.index);
                }
            }
            oracle.check(&arena, &budget, step);
        }
        while let Some(model) = held.pop() {
            arena.release(model.index);
        }
        oracle.check(&arena, &budget, steps);
        assert_eq!(arena.live(), 0);
    }

    #[test]
    fn a_random_sequence_matches_the_arc_model() {
        // Blocks of 9 slots (a 1 KiB budget's), of 170 (a full block's)
        for (seed, limit) in [
            (0x2545_F491_4F6C_DD1D, 1024),
            (0x9E37_79B9_7F4A_7C15, 1024),
            (0xD1B5_4A32_D192_ED03, 1 << 30),
        ] {
            matches_the_arc_model(seed, limit, 8_000);
        }
    }

    #[test]
    fn a_block_is_sized_as_a_block_of_the_queue() {
        for limit in [0, 100, 1024, 4 * 1024, 16 * 1024, 1 << 30] {
            let arena = SegmentArena::within(limit);
            assert_eq!(
                arena.per_block,
                PathQueue::<Slot>::block_items_within(limit),
                "{limit}"
            );
            assert!(arena.per_block <= 1 << arena.shift, "{limit}");
        }
        assert_eq!(SegmentArena::within(1 << 30).per_block, 4096 / 24);
    }

    #[test]
    fn a_freed_slot_is_taken_before_a_new_block() {
        let mut budget = PathBudget::new(1 << 30, None);
        let mut arena = SegmentArena::within(1024);
        let per_block = arena.per_block;
        let roots: Vec<u32> = (0..per_block)
            .map(|node| arena.root(&mut budget, NodeId(node as u64)).unwrap())
            .collect();
        let held = budget.used();
        arena.release(roots[3]);
        assert_eq!(budget.used(), held, "a freed slot stays charged");
        let again = arena.root(&mut budget, NodeId(99)).unwrap();
        assert_eq!((again, arena.blocks(), budget.used()), (roots[3], 1, held));
        // The block is full: the next takes a new one
        arena.root(&mut budget, NodeId(100)).unwrap();
        assert_eq!(arena.blocks(), 2);
        assert_eq!(budget.used(), arena.held_bytes());
    }

    #[test]
    fn a_segment_is_freed_with_the_prefix_only_it_held() {
        // root <- a <- b, and root <- c: letting go of b frees b and a,
        // not the root, which c holds
        let mut budget = PathBudget::new(1 << 30, None);
        let mut arena = SegmentArena::within(1 << 30);
        let root = arena.root(&mut budget, NodeId(1)).unwrap();
        let a = arena
            .push(&mut budget, root, NodeId(2), EdgeId(10))
            .unwrap();
        let b = arena.push(&mut budget, a, NodeId(3), EdgeId(11)).unwrap();
        let c = arena
            .push(&mut budget, root, NodeId(4), EdgeId(12))
            .unwrap();
        arena.release(root);
        arena.release(a);
        assert_eq!(arena.live(), 4, "b holds a, a and c hold the root");
        assert_eq!(arena.collect_nodes(b, 2), [NodeId(1), NodeId(2), NodeId(3)]);
        assert_eq!(arena.collect_edges(b, 2), [EdgeId(10), EdgeId(11)]);
        arena.release(b);
        assert_eq!(arena.live(), 2, "the root and c");
        assert!(arena.contains_node(c, NodeId(1)) && !arena.contains_node(c, NodeId(2)));
        arena.release(c);
        assert_eq!(arena.live(), 0);
        assert_eq!(
            budget.used(),
            arena.held_bytes(),
            "freed slots stay charged"
        );
    }

    #[test]
    fn a_held_segment_is_freed_after_its_last_holder() {
        // root <- a, with a named by two queued paths: a and its prefix live
        // until both let go
        let mut budget = PathBudget::new(1 << 30, None);
        let mut arena = SegmentArena::within(1 << 30);
        let root = arena.root(&mut budget, NodeId(1)).unwrap();
        let a = arena
            .push(&mut budget, root, NodeId(2), EdgeId(10))
            .unwrap();
        arena.release(root);
        arena.hold(a);
        arena.release(a);
        assert_eq!(
            arena.live(),
            2,
            "one holder of a is left, and a holds the root"
        );
        assert_eq!(arena.collect_nodes(a, 1), [NodeId(1), NodeId(2)]);
        arena.release(a);
        assert_eq!(arena.live(), 0, "the last holder frees a and the root");
        let reused = arena.root(&mut budget, NodeId(3)).unwrap();
        assert!(reused == a || reused == root, "a freed slot is taken first");
    }

    #[test]
    fn past_the_most_slots_a_segment_is_refused() {
        let mut budget = PathBudget::new(1 << 30, None);
        // Full blocks of 170 slots, so the 20 slots are indices 0 to 19
        let mut arena = SegmentArena::within(1 << 30).with_most(20);
        let root = arena.root(&mut budget, NodeId(0)).unwrap();
        let mut last = root;
        for node in 1..20 {
            last = arena
                .push(&mut budget, root, NodeId(node), EdgeId(node))
                .unwrap();
        }
        assert_eq!(last, 19);
        let held = budget.used();
        let refused = arena
            .push(&mut budget, root, NodeId(20), EdgeId(20))
            .unwrap_err();
        assert!(path_budget_message(&refused).is_some(), "{refused}");
        assert_eq!(arena.live(), 20, "the refusal changes nothing");
        assert_eq!(budget.used(), held);
        assert!(arena.root(&mut budget, NodeId(21)).is_err());
        // A freed slot is still named within the limit
        arena.release(last);
        assert!(
            arena
                .push(&mut budget, root, NodeId(22), EdgeId(22))
                .is_ok()
        );
    }

    #[test]
    fn a_refused_block_is_not_allocated() {
        // Room for one block of 9 and its index, not for a second block
        let mut arena = SegmentArena::within(1024);
        let mut budget = PathBudget::new(9 * SEGMENT_BYTES + 200, None);
        let root = arena.root(&mut budget, NodeId(0)).unwrap();
        for node in 1..9 {
            arena
                .push(&mut budget, root, NodeId(node), EdgeId(node))
                .unwrap();
        }
        let held = budget.used();
        let refused = arena
            .push(&mut budget, root, NodeId(9), EdgeId(9))
            .unwrap_err();
        assert!(path_budget_message(&refused).is_some(), "{refused}");
        assert_eq!((arena.blocks(), arena.live()), (1, 9));
        assert!(budget.used() <= held + 4 * size_of::<Vec<Slot>>());
        assert_eq!(budget.used(), arena.held_bytes());
    }

    #[test]
    #[cfg(debug_assertions)]
    #[should_panic(expected = "a path's segments form a cycle")]
    fn a_path_whose_segments_form_a_cycle_fails_its_walk() {
        // What a slot freed while still named, then reused further down its
        // own path, leaves: a walk back that never reaches the first
        let mut budget = PathBudget::new(1 << 30, None);
        let mut arena = SegmentArena::within(1 << 30);
        let root = arena.root(&mut budget, NodeId(0)).unwrap();
        let next = arena.push(&mut budget, root, NodeId(1), EdgeId(1)).unwrap();
        arena.slot_mut(root).parent = next;
        arena.contains_node(next, NodeId(99));
    }

    #[test]
    fn a_long_path_is_read_and_checked_without_a_frame_per_segment() {
        let mut budget = PathBudget::new(usize::MAX, None);
        let mut arena = SegmentArena::within(usize::MAX);
        let mut segment = arena.root(&mut budget, NodeId(0)).unwrap();
        for i in 1..=300_000 {
            let next = arena
                .push(&mut budget, segment, NodeId(i), EdgeId(i))
                .unwrap();
            arena.release(segment);
            segment = next;
        }
        assert!(arena.contains_node(segment, NodeId(0)));
        assert!(!arena.contains_node(segment, NodeId(300_001)));
        assert!(arena.contains_edge(segment, EdgeId(1)));
        assert!(!arena.contains_edge(segment, EdgeId(0)));
        assert_eq!(arena.collect_edges(segment, 300_000).len(), 300_000);
        arena.release(segment);
        assert_eq!(arena.live(), 0);
    }
}
