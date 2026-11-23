//! The first-in first-out queue of a path search, grown in fixed-size blocks
//! within its [`PathBudget`]: a queue that doubles holds its old and its new
//! buffer at once while it grows, so a search charged that peak holds far
//! fewer paths than its budget has room for. Blocks never move: the queue
//! holds what it holds plus at most a few blocks.

use std::collections::VecDeque;
use std::mem::size_of;

use crate::execution::operators::OperatorError;
use crate::execution::path_budget::{PathBudget, capacity_within};

/// The bytes of one block of a path search's queue: measured, the smallest size at
/// which a search over many paths runs as fast as over one buffer.
pub const BLOCK_BYTES: usize = 4 * 1024;

/// The bytes of one slot of a path search queue's block index: a block, a `Vec`.
pub const INDEX_SLOT_BYTES: usize = size_of::<Vec<u8>>();

/// The budget whose blocks are full size once their index is paid for:
/// within it, and below it, a block is a quarter of what the budget leaves
/// beside the index of four blocks (see `PathQueue::block_items_within`).
pub const SMALL_BUDGET_BYTES: usize = SMALL_BUDGET_BLOCKS * BLOCK_BYTES;

/// The blocks of a small budget's queue: a block is a quarter of what the
/// budget leaves beside their index.
const SMALL_BUDGET_BLOCKS: usize = 4;

/// A first-in first-out queue of `T` in blocks of [`BLOCK_BYTES`], each
/// charged to the search's budget before it is allocated (see
/// [`reserve`](Self::reserve)); within a budget of about
/// [`SMALL_BUDGET_BYTES`] or less a block is smaller, in proportion (see
/// [`block_items_within`](Self::block_items_within)). Items come out in the
/// order they went in.
///
/// Items go into `tail`; a full tail moves to `full`. The front block,
/// `head`, is reversed once when it becomes the front, so it gives its items
/// from its end. A drained block is kept as the one spare, still charged, or
/// given back.
pub(crate) struct PathQueue<T> {
    /// The front block, reversed: the next item is its last.
    head: Vec<T>,
    /// The full blocks between the front and the back, in order.
    full: VecDeque<Vec<T>>,
    /// The back block, which takes the next item.
    tail: Vec<T>,
    /// A drained block kept for the next tail (empty, charged).
    spare: Option<Vec<T>>,
    /// The items held.
    len: usize,
}

impl<T> PathQueue<T> {
    /// An empty queue, which holds no block yet.
    pub(crate) fn new() -> Self {
        Self {
            head: Vec::new(),
            full: VecDeque::new(),
            tail: Vec::new(),
            spare: None,
            len: 0,
        }
    }

    /// The items a block holds.
    pub(crate) const fn block_items() -> usize {
        let item = if size_of::<T>() == 0 {
            1
        } else {
            size_of::<T>()
        };
        let items = BLOCK_BYTES / item;
        if items == 0 { 1 } else { items }
    }

    /// The items a new block holds within a budget of `limit` bytes: a
    /// quarter of what the budget leaves beside the index of four blocks
    /// (every block but the tail, grown as the budget grows any queue), in
    /// whole items, at least one and at most a full block. Four blocks and
    /// their index then always fit, and the block grows with the budget up
    /// to a full one, reached at [`SMALL_BUDGET_BYTES`] and its index: a
    /// larger budget does not hold fewer blocks.
    pub(crate) fn block_items_within(limit: usize) -> usize {
        let index = capacity_within(limit, SMALL_BUDGET_BLOCKS - 1) * INDEX_SLOT_BYTES;
        let entry = size_of::<T>().max(1);
        (limit.saturating_sub(index) / SMALL_BUDGET_BLOCKS / entry).clamp(1, Self::block_items())
    }

    /// The items held.
    pub(crate) fn len(&self) -> usize {
        self.len
    }

    /// Whether it holds no item.
    pub(crate) fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Makes room for one more item within `budget`, before the item is
    /// built: a new block (the spare, or one charged before it is
    /// allocated), and a slot for the full tail in the index of blocks, which
    /// grows as [`PathBudget::reserve_queue`] does. A refusal changes nothing.
    pub(crate) fn reserve(&mut self, budget: &mut PathBudget) -> Result<(), OperatorError> {
        if self.tail.len() < self.tail.capacity() {
            return Ok(());
        }
        let items = Self::block_items_within(budget.limit());
        self.add_block(budget, items)
    }

    /// Puts a block with room behind the full tail: the spare, or a new block
    /// of `items` items.
    fn add_block(&mut self, budget: &mut PathBudget, items: usize) -> Result<(), OperatorError> {
        if !self.tail.is_empty() {
            budget.reserve_queue(&mut self.full, 1)?;
        }
        let block = match self.spare.take() {
            Some(block) => block,
            None => {
                let mut block = Vec::new();
                budget.reserve_block(&mut block, items)?;
                block
            }
        };
        let filled = std::mem::replace(&mut self.tail, block);
        if !filled.is_empty() {
            self.full.push_back(filled);
        }
        Ok(())
    }

    /// Adds `item` at the back, into the room [`reserve`](Self::reserve)
    /// made.
    pub(crate) fn push(&mut self, item: T) {
        debug_assert!(
            self.tail.len() < self.tail.capacity(),
            "a push follows a reserve"
        );
        self.tail.push(item);
        self.len += 1;
    }

    /// Takes the item at the front; a block it drains is kept as the spare
    /// or given back to `budget`.
    pub(crate) fn pop(&mut self, budget: &mut PathBudget) -> Option<T> {
        if self.head.is_empty() {
            let next = match self.full.pop_front() {
                Some(block) => block,
                None if self.tail.is_empty() => return None,
                None => std::mem::take(&mut self.tail),
            };
            let drained = std::mem::replace(&mut self.head, next);
            self.recycle(drained, budget);
            self.head.reverse();
        }
        let item = self.head.pop()?;
        self.len -= 1;
        if self.head.is_empty() {
            let drained = std::mem::take(&mut self.head);
            self.recycle(drained, budget);
        }
        Some(item)
    }

    /// Keeps the drained `block` as the spare, or gives it back to `budget`
    /// when there is one.
    fn recycle(&mut self, block: Vec<T>, budget: &mut PathBudget) {
        debug_assert!(block.is_empty(), "only a drained block is recycled");
        if block.capacity() == 0 {
            return;
        }
        if self.spare.is_none() {
            self.spare = Some(block);
        } else {
            budget.release(block.capacity() * size_of::<T>());
        }
    }

    /// The bytes it holds, block by block: what its budget is charged for it.
    #[cfg(test)]
    pub(crate) fn held_bytes(&self) -> usize {
        let block = |block: &Vec<T>| block.capacity() * size_of::<T>();
        block(&self.head)
            + block(&self.tail)
            + self.full.iter().map(block).sum::<usize>()
            + self.spare.as_ref().map_or(0, block)
            + self.full.capacity() * size_of::<Vec<T>>()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::execution::operators::path_budget_message;

    /// Pushes `item` within `budget`.
    fn push(queue: &mut PathQueue<u64>, budget: &mut PathBudget, item: u64) {
        queue.reserve(budget).unwrap();
        queue.push(item);
    }

    #[test]
    fn items_come_out_in_order_across_blocks() {
        let mut budget = PathBudget::new(1 << 30, None);
        let mut queue = PathQueue::new();
        let block = PathQueue::<u64>::block_items();
        assert_eq!(block, 512);
        let count = 3 * block as u64 + 17;
        for item in 0..count {
            push(&mut queue, &mut budget, item);
        }
        assert_eq!(queue.len(), usize::try_from(count).unwrap());
        assert_eq!(budget.used(), queue.held_bytes());
        for item in 0..count {
            assert_eq!(queue.pop(&mut budget), Some(item));
            assert_eq!(budget.used(), queue.held_bytes(), "after popping {item}");
        }
        assert!(queue.is_empty());
        assert_eq!(queue.pop(&mut budget), None);
        // Drained: one spare block stays, charged, with the index
        assert!(queue.spare.is_some());
        assert_eq!(
            budget.used(),
            block * 8 + queue.full.capacity() * size_of::<Vec<u64>>()
        );
        assert_eq!(budget.used(), queue.held_bytes());
    }

    #[test]
    fn a_drained_queue_refills_from_its_spare() {
        let mut budget = PathBudget::new(1 << 30, None);
        let mut queue = PathQueue::new();
        let block = PathQueue::<u64>::block_items() as u64;
        for round in 0..5_u64 {
            for item in 0..=block {
                push(&mut queue, &mut budget, round * 1_000 + item);
            }
            let held = budget.used();
            for item in 0..=block {
                assert_eq!(queue.pop(&mut budget), Some(round * 1_000 + item));
            }
            assert!(queue.is_empty());
            assert_eq!(budget.used(), queue.held_bytes(), "round {round}");
            assert!(
                budget.used() <= held,
                "a drained block gives its charge back"
            );
        }
        // Two blocks at most, one of them the spare, however many rounds
        assert!(budget.used() <= 2 * BLOCK_BYTES + queue.full.capacity() * 24);
    }

    #[test]
    fn a_random_sequence_matches_a_vec_deque() {
        matches_a_vec_deque(PathQueue::<u64>::block_items());
    }

    #[test]
    fn a_random_sequence_matches_a_vec_deque_in_share_sized_blocks() {
        // The blocks of a search within a budget of 10 KiB
        let items = PathQueue::<u64>::block_items_within(10 * 1024);
        assert_eq!(items, (10 * 1024 - 4 * INDEX_SLOT_BYTES) / 4 / 8);
        matches_a_vec_deque(items);
    }

    /// A small LCG with a fixed seed: pushes and pops at random, more pushes
    /// than pops for a while, then more pops, in blocks of `items` items.
    fn matches_a_vec_deque(items: usize) {
        let mut state: u64 = 0x2545_F491_4F6C_DD1D;
        let mut next = || {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            state >> 33
        };
        let mut budget = PathBudget::new(1 << 30, None);
        let mut queue = PathQueue::new();
        let mut oracle = VecDeque::new();
        let mut item = 0_u64;
        for step in 0..60_000 {
            let push_odds = if step < 30_000 { 6 } else { 3 };
            if next() % 10 < push_odds {
                if queue.tail.len() == queue.tail.capacity() {
                    queue.add_block(&mut budget, items).unwrap();
                }
                queue.push(item);
                oracle.push_back(item);
                item += 1;
            } else {
                assert_eq!(queue.pop(&mut budget), oracle.pop_front(), "step {step}");
            }
            assert_eq!(queue.len(), oracle.len(), "step {step}");
            assert_eq!(budget.used(), queue.held_bytes(), "step {step}");
        }
        while let Some(want) = oracle.pop_front() {
            assert_eq!(queue.pop(&mut budget), Some(want));
        }
        assert_eq!(queue.pop(&mut budget), None);
        assert_eq!(budget.used(), queue.held_bytes());
    }

    #[test]
    fn a_refused_block_is_not_allocated() {
        // A budget whose blocks are full size, with room left for one block
        // and its index, not for a second block
        let limit = SMALL_BUDGET_BYTES + 4 * INDEX_SLOT_BYTES;
        assert_eq!(
            PathQueue::<u64>::block_items_within(limit),
            PathQueue::<u64>::block_items()
        );
        let charged = limit - BLOCK_BYTES - 64 * 24;
        let mut budget = PathBudget::new(limit, None);
        budget.charge(charged).unwrap();
        let mut queue = PathQueue::new();
        let block = PathQueue::<u64>::block_items() as u64;
        for item in 0..block {
            push(&mut queue, &mut budget, item);
        }
        let held = budget.used();
        let refused = queue.reserve(&mut budget).unwrap_err();
        assert!(path_budget_message(&refused).is_some(), "{refused}");
        assert_eq!(
            queue.tail.len(),
            queue.tail.capacity(),
            "the tail is as it was"
        );
        assert!(queue.spare.is_none() && queue.full.is_empty());
        assert!(budget.used() <= held + queue.full.capacity() * 24);
        assert_eq!(budget.used(), charged + queue.held_bytes());
        for item in 0..block {
            assert_eq!(queue.pop(&mut budget), Some(item));
        }
    }

    #[test]
    fn a_queue_holds_its_items_and_at_most_three_blocks_more() {
        // At its fullest a queue holds its items, a part-filled head and
        // tail, the spare and its index: never twice its items
        let mut budget = PathBudget::new(1 << 30, None);
        let mut queue = PathQueue::new();
        let mut most = 0;
        for item in 0..100_000_u64 {
            push(&mut queue, &mut budget, item);
            if item % 3 == 0 {
                queue.pop(&mut budget);
            }
            let items = queue.len() * size_of::<u64>();
            let index = queue.full.capacity() * size_of::<Vec<u64>>();
            assert!(
                budget.used() <= items + 3 * BLOCK_BYTES + index,
                "item {item}"
            );
            most = most.max(budget.used());
        }
        assert!(most > 0);
    }

    #[test]
    fn below_sixteen_kib_a_block_is_in_proportion() {
        // A quarter of what the budget leaves beside the index of four
        // blocks (4 slots, or 3 where the budget's least growth is 3), in
        // whole items, at least one, at most a full block
        let slot = INDEX_SLOT_BYTES;
        for (limit, items) in [
            (1 << 30, 512),
            (SMALL_BUDGET_BYTES + 4 * slot, 512),
            (SMALL_BUDGET_BYTES, (SMALL_BUDGET_BYTES - 4 * slot) / 4 / 8),
            (
                SMALL_BUDGET_BYTES - 1,
                (SMALL_BUDGET_BYTES - 1 - 3 * slot) / 4 / 8,
            ),
            (4 * 1024, (4 * 1024 - 4 * slot) / 4 / 8),
            (1024, (1024 - 4 * slot) / 4 / 8),
            (100, 1),
            (8, 1),
            (0, 1),
        ] {
            assert_eq!(
                PathQueue::<u64>::block_items_within(limit),
                items,
                "{limit}"
            );
        }
        // Every block of a 4 KiB budget, not only the first
        let mut budget = PathBudget::new(4 * 1024, None);
        let mut queue = PathQueue::new();
        for item in 0..=128 {
            push(&mut queue, &mut budget, item);
        }
        let items = PathQueue::<u64>::block_items_within(4 * 1024);
        assert_eq!((queue.tail.capacity(), queue.full.len()), (items, 1));
        assert_eq!(budget.used(), queue.held_bytes());
    }

    #[test]
    fn a_spare_keeps_its_size_and_its_charge() {
        // A small block drained becomes the spare, still charged; a full
        // block goes back once there is a spare
        let mut budget = PathBudget::new(1 << 30, None);
        let mut queue = PathQueue::new();
        let full = PathQueue::<u64>::block_items();
        for item in 0..7 + full {
            if queue.tail.len() == queue.tail.capacity() {
                let items = if item == 0 { 7 } else { full };
                queue.add_block(&mut budget, items).unwrap();
            }
            queue.push(item as u64);
        }
        for want in 0..7 {
            assert_eq!(queue.pop(&mut budget), Some(want));
        }
        assert_eq!(queue.spare.as_ref().map(Vec::capacity), Some(7));
        assert_eq!(budget.used(), queue.held_bytes());
        while queue.pop(&mut budget).is_some() {}
        assert_eq!(queue.spare.as_ref().map(Vec::capacity), Some(7));
        assert_eq!(budget.used(), queue.held_bytes());
        // It refills from the small spare first
        push(&mut queue, &mut budget, 0);
        assert_eq!(queue.tail.capacity(), 7);
        assert_eq!(budget.used(), queue.held_bytes());
    }
}
