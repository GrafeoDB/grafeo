//! The count of a chain of hops from a node, without listing its paths.
//!
//! The walks from a node over hops `k..` are the sum over its edges of hop `k` of
//! the walks from the far end over hops `k + 1..`, each worked out once per node. So
//! a hub with thousands of edges each way reads its lists, not the product of them.
//!
//! openCypher's patterns bind each edge once, which the plan states as pairs of
//! hops whose edges must differ. Those paths are counted by inclusion and exclusion:
//! a sum over the ways the hops may share edges, each with a sign. A term with no
//! shared edge is the walks above; one with shared edges walks up to the last hop
//! that shares, as a fixed edge prunes it, and counts the rest by the memo.

use std::sync::Arc;

use grafeo_common::types::{ArcStr, EdgeId, NodeId, ReadView};
use grafeo_common::utils::hash::FxHashMap;

use super::{OperatorError, ReadCounter};
use crate::graph::{Direction, GraphStore};

/// The most hops whose edges must differ that are counted without listing: the
/// terms of the sum grow as the Bell numbers (5, 15, 52 and 203 for 3 to 6 hops).
pub const MAX_COUNTED_DIFFERENT_HOPS: usize = 6;

/// The steps of a count between looks at the query's deadline: a list read or a memo
/// hit is far shorter than a clock read, and neither stops on its own.
const STEPS_PER_DEADLINE_CHECK: u32 = 1 << 12;

/// One hop of a counted chain.
#[derive(Debug, Clone)]
pub struct CountedHop {
    /// The way it goes from the node it starts at.
    pub direction: Direction,
    /// The edge types it follows, as stored (every type when empty).
    pub edge_types: Vec<ArcStr>,
    /// The labels its far end must have.
    pub end_labels: Vec<ArcStr>,
}

/// The paths of two hops through one middle node, from its two degrees: the first
/// hop read from the middle, times the second, without a list of either. A total over
/// every middle node reads two counts per node, where a walk from each first node
/// reads every edge.
#[derive(Debug, Clone)]
pub struct MiddleSum {
    /// The first hop, read from the middle node back to where it starts.
    pub back: CountedHop,
    /// The second hop.
    pub on: CountedHop,
    /// The labels the middle node must have, past the one its scan reads.
    pub middle_labels: Vec<ArcStr>,
    /// Whether one edge may not serve both hops: both read the edge the same way, so
    /// only a self-loop of the middle node could.
    pub different_edges: bool,
}

impl MiddleSum {
    /// The paths through `node`, counting a read per count.
    pub(super) fn count(
        &self,
        store: &dyn GraphStore,
        node: NodeId,
        view: ReadView,
        reads: Option<&ReadCounter>,
    ) -> Result<usize, OperatorError> {
        for label in &self.middle_labels {
            if store.node_has_label(node, label, view)? != Some(true) {
                return Ok(0);
            }
        }
        let count_hop = |hop: &CountedHop| {
            store.count_edges(node, hop.direction, &hop.edge_types, &hop.end_labels, view)
        };
        let back = count_hop(&self.back)?;
        super::read_counter::count(reads, 1);
        let on = if back == 0 {
            0
        } else {
            super::read_counter::count(reads, 1);
            count_hop(&self.on)?
        };
        let mut paths = (back as u128)
            .checked_mul(on as u128)
            .ok_or_else(too_many_paths)?;
        if self.different_edges && paths > 0 {
            paths -= self.self_loops(store, node, view)?.min(paths);
            super::read_counter::count(reads, 1);
        }
        let paths = i64::try_from(paths).map_err(|_| too_many_paths())?;
        usize::try_from(paths).map_err(|_| too_many_paths())
    }

    /// The edges of the middle node that serve both hops, as one edge may not: its
    /// self-loops, when the node has the labels of both far ends, which are itself.
    fn self_loops(
        &self,
        store: &dyn GraphStore,
        node: NodeId,
        view: ReadView,
    ) -> Result<u128, OperatorError> {
        for label in self.back.end_labels.iter().chain(&self.on.end_labels) {
            if store.node_has_label(node, label, view)? != Some(true) {
                return Ok(0);
            }
        }
        let loops =
            store.edges_between(node, node, self.on.direction, &self.on.edge_types, view)?;
        Ok(loops.len() as u128)
    }
}

/// One way the hops share edges: per hop the block it belongs to, hops of one block
/// on one edge, with the sign and multiplicity of the term.
#[derive(Debug, Clone)]
struct Term {
    blocks: Vec<usize>,
    coefficient: i64,
    /// The last hop in a block of more than one, which a walk must reach to know.
    last_shared: Option<usize>,
    /// Per hop, whether it shares its edge with no other: only its far end matters.
    alone: Vec<bool>,
    /// Per hop, whether every hop sharing its edge reads it the same way, with the
    /// same types and no labels: then parallel edges to one far end count alike.
    read_alike: Vec<bool>,
}

/// An edge a term has fixed: its ends as a hop read them.
#[derive(Debug, Clone, Copy)]
struct FixedEdge {
    edge: EdgeId,
    from: NodeId,
    to: NodeId,
    hop: usize,
}

/// A node's edges of one hop, as a list that keeps far ends with the hop's labels.
type HopList = Arc<[(NodeId, EdgeId)]>;

/// A node's far ends of one hop, each with the number of its edges there and the
/// first of them, in the order the list first reaches them.
type FarEnds = Arc<[(NodeId, u128, EdgeId)]>;

/// Counts chains of `hops`. It holds what it has read, for one statement's view.
pub(super) struct ChainCounter {
    store: Arc<dyn GraphStore>,
    hops: Vec<CountedHop>,
    terms: Vec<Term>,
    view: ReadView,
    reads: Option<Arc<ReadCounter>>,
    lists: Vec<FxHashMap<NodeId, HopList>>,
    /// A node's far ends of one hop with how many edges reach each.
    far_ends: Vec<FxHashMap<NodeId, FarEnds>>,
    walks: Vec<FxHashMap<NodeId, u128>>,
    /// The steps taken, to look at the deadline every so often.
    steps: u32,
}

impl ChainCounter {
    /// A counter of `hops`, where the edges of each pair of `different` hops differ.
    pub(super) fn new(
        store: Arc<dyn GraphStore>,
        hops: Vec<CountedHop>,
        different: &[(usize, usize)],
        view: ReadView,
        reads: Option<Arc<ReadCounter>>,
    ) -> Self {
        let levels = hops.len();
        let mut terms = terms_of(levels, different);
        for term in &mut terms {
            term.read_alike = (0..levels)
                .map(|level| {
                    (0..levels)
                        .filter(|&other| term.blocks[other] == term.blocks[level])
                        .all(|other| read_alike(&hops[level], &hops[other]))
                })
                .collect();
        }
        Self {
            store,
            terms,
            hops,
            view,
            reads,
            lists: vec![FxHashMap::default(); levels],
            far_ends: vec![FxHashMap::default(); levels],
            walks: vec![FxHashMap::default(); levels],
            steps: 0,
        }
    }

    /// One step of the sum: lists already read cost no read, so only here does a
    /// count on them notice its deadline.
    fn step(&mut self) -> Result<(), OperatorError> {
        self.steps = self.steps.wrapping_add(1);
        if self.steps.is_multiple_of(STEPS_PER_DEADLINE_CHECK) {
            crate::execution::deadline::check()?;
        }
        Ok(())
    }

    /// The count of the chain's paths from `node`.
    pub(super) fn count(&mut self, node: NodeId) -> Result<usize, OperatorError> {
        let mut total: i128 = 0;
        for index in 0..self.terms.len() {
            let term = self.terms[index].clone();
            let mut fixed = vec![None; self.hops.len()];
            let count = i128::try_from(self.term_from(&term, 0, node, &mut fixed)?)
                .map_err(|_| too_many_paths())?;
            total = i128::from(term.coefficient)
                .checked_mul(count)
                .and_then(|signed| total.checked_add(signed))
                .ok_or_else(too_many_paths)?;
        }
        // A count is a 64-bit integer, and so is what it sums to.
        let count = i64::try_from(total.max(0)).map_err(|_| too_many_paths())?;
        usize::try_from(count).map_err(|_| too_many_paths())
    }

    /// The walks of `term` from `node` over the hops from `level` on, with the
    /// edges earlier hops fixed, by block.
    fn term_from(
        &mut self,
        term: &Term,
        level: usize,
        node: NodeId,
        fixed: &mut Vec<Option<FixedEdge>>,
    ) -> Result<u128, OperatorError> {
        self.step()?;
        if term.last_shared.is_none_or(|last| level > last) {
            return self.walks_from(level, node);
        }
        let block = term.blocks[level];
        if let Some(edge) = fixed[block] {
            return match self.far_end_of(edge, level, node)? {
                Some(_) if level + 1 == self.hops.len() => Ok(1),
                Some(far) => self.term_from(term, level + 1, far, fixed),
                None => Ok(0),
            };
        }
        // A hop that shares its edge with none leaves only its far end to know:
        // parallel edges and a node's many self-loops are one branch.
        if term.alone[level] {
            let mut total: u128 = 0;
            for &(far, edges, _) in self.far_ends_of(level, node)?.iter() {
                let below = self.term_from(term, level + 1, far, fixed)?;
                total = below
                    .checked_mul(edges)
                    .and_then(|paths| total.checked_add(paths))
                    .ok_or_else(too_many_paths)?;
            }
            return Ok(total);
        }
        // Read alike, one of the parallel edges stands for them all.
        if term.read_alike[level] {
            let mut total: u128 = 0;
            for &(far, edges, edge) in self.far_ends_of(level, node)?.iter() {
                fixed[block] = Some(FixedEdge {
                    edge,
                    from: node,
                    to: far,
                    hop: level,
                });
                let below = if level + 1 == self.hops.len() {
                    1
                } else {
                    self.term_from(term, level + 1, far, fixed)?
                };
                total = below
                    .checked_mul(edges)
                    .and_then(|paths| total.checked_add(paths))
                    .ok_or_else(too_many_paths)?;
            }
            fixed[block] = None;
            return Ok(total);
        }
        let list = self.list_of(level, node)?;
        let mut total: u128 = 0;
        for &(far, edge) in list.iter() {
            fixed[block] = Some(FixedEdge {
                edge,
                from: node,
                to: far,
                hop: level,
            });
            let below = if level + 1 == self.hops.len() {
                1
            } else {
                self.term_from(term, level + 1, far, fixed)?
            };
            total = total.checked_add(below).ok_or_else(too_many_paths)?;
        }
        fixed[block] = None;
        Ok(total)
    }

    /// Where the hop at `level` from `node` goes over the edge `fixed`, when it can
    /// go over it: a hop of the same way and types reads its ends from how the edge
    /// was first read, any other from the node's list.
    fn far_end_of(
        &mut self,
        fixed: FixedEdge,
        level: usize,
        node: NodeId,
    ) -> Result<Option<NodeId>, OperatorError> {
        let first = &self.hops[fixed.hop];
        let hop = &self.hops[level];
        if first.direction == hop.direction
            && first.edge_types == hop.edge_types
            && first.end_labels.is_empty()
            && hop.end_labels.is_empty()
        {
            return Ok(match hop.direction {
                // Read the same way, the edge goes on only from its near end.
                Direction::Outgoing | Direction::Incoming => {
                    (fixed.from == node).then_some(fixed.to)
                }
                Direction::Both => {
                    if fixed.from == node {
                        Some(fixed.to)
                    } else if fixed.to == node {
                        Some(fixed.from)
                    } else {
                        None
                    }
                }
            });
        }
        let list = self.list_of(level, node)?;
        Ok(list
            .iter()
            .find(|(_, edge)| *edge == fixed.edge)
            .map(|(far, _)| *far))
    }

    /// The far ends of the hop at `level` from `node`, with their edges.
    fn far_ends_of(&mut self, level: usize, node: NodeId) -> Result<FarEnds, OperatorError> {
        if let Some(far_ends) = self.far_ends[level].get(&node) {
            return Ok(Arc::clone(far_ends));
        }
        let list = self.list_of(level, node)?;
        let mut positions: FxHashMap<NodeId, usize> = FxHashMap::default();
        let mut grouped: Vec<(NodeId, u128, EdgeId)> = Vec::new();
        for &(far, edge) in list.iter() {
            let position = *positions.entry(far).or_insert_with(|| {
                grouped.push((far, 0, edge));
                grouped.len() - 1
            });
            grouped[position].1 += 1;
        }
        let far_ends: FarEnds = grouped.into();
        self.far_ends[level].insert(node, Arc::clone(&far_ends));
        Ok(far_ends)
    }

    /// The walks from `node` over the hops from `level` on, each node once.
    fn walks_from(&mut self, level: usize, node: NodeId) -> Result<u128, OperatorError> {
        self.step()?;
        if let Some(&known) = self.walks[level].get(&node) {
            return Ok(known);
        }
        let walks = if level + 1 == self.hops.len() {
            self.last_hop_count(node)?
        } else {
            let mut total: u128 = 0;
            for &(far, edges, _) in self.far_ends_of(level, node)?.iter() {
                total = self
                    .walks_from(level + 1, far)?
                    .checked_mul(edges)
                    .and_then(|paths| total.checked_add(paths))
                    .ok_or_else(too_many_paths)?;
            }
            total
        };
        self.walks[level].insert(node, walks);
        Ok(walks)
    }

    /// The last hop's edges from `node`, counted without a list when no label
    /// asks for one to read each far end.
    fn last_hop_count(&mut self, node: NodeId) -> Result<u128, OperatorError> {
        let level = self.hops.len() - 1;
        let hop = &self.hops[level];
        if hop.end_labels.is_empty() {
            let count =
                self.store
                    .count_edges(node, hop.direction, &hop.edge_types, &[], self.view)?;
            super::read_counter::count(self.reads.as_deref(), count);
            return Ok(count as u128);
        }
        Ok(self.list_of(level, node)?.len() as u128)
    }

    /// The edges of the hop at `level` from `node` whose far end has its labels,
    /// each node's read once. Every edge read counts as a read, kept or not.
    fn list_of(&mut self, level: usize, node: NodeId) -> Result<HopList, OperatorError> {
        if let Some(list) = self.lists[level].get(&node) {
            return Ok(Arc::clone(list));
        }
        crate::execution::deadline::check()?;
        let hop = &self.hops[level];
        let mut edges = self
            .store
            .edges(node, hop.direction, &hop.edge_types, self.view)?;
        super::read_counter::count(self.reads.as_deref(), edges.len());
        if !hop.end_labels.is_empty() {
            let mut kept = Vec::with_capacity(edges.len());
            for (far, edge) in edges {
                let mut has_labels = true;
                for label in &hop.end_labels {
                    if self.store.node_has_label(far, label, self.view)? != Some(true) {
                        has_labels = false;
                        break;
                    }
                }
                if has_labels {
                    kept.push((far, edge));
                }
            }
            edges = kept;
        }
        let list: HopList = edges.into();
        self.lists[level].insert(node, Arc::clone(&list));
        Ok(list)
    }
}

/// The error of a count no 64-bit integer holds (22003): a wrong number would pass
/// for an answer.
fn too_many_paths() -> OperatorError {
    OperatorError::numeric_out_of_range("a count of paths past 64 bits")
}

/// Whether two hops read an edge the same way from its near end, so that which of two
/// parallel edges one has fixed makes no difference to the other.
fn read_alike(first: &CountedHop, second: &CountedHop) -> bool {
    first.direction != Direction::Both
        && first.direction == second.direction
        && first.edge_types == second.edge_types
        && first.end_labels.is_empty()
        && second.end_labels.is_empty()
}

/// The terms of the sum for `hops` hops where each pair of `different` hops differs:
/// over every subset of the pairs, the hops it joins share an edge, with the sign of
/// its size; subsets that join the same hops add up.
fn terms_of(hops: usize, different: &[(usize, usize)]) -> Vec<Term> {
    let mut coefficients: FxHashMap<Vec<usize>, i64> = FxHashMap::default();
    for subset in 0..1_u64 << different.len() {
        let mut blocks: Vec<usize> = (0..hops).collect();
        for (index, (a, b)) in different.iter().enumerate() {
            if subset & (1 << index) != 0 {
                let (keep, replace) = (blocks[*a].min(blocks[*b]), blocks[*a].max(blocks[*b]));
                for block in &mut blocks {
                    if *block == replace {
                        *block = keep;
                    }
                }
            }
        }
        let sign = if subset.count_ones() % 2 == 0 { 1 } else { -1 };
        *coefficients.entry(blocks).or_insert(0) += sign;
    }
    let mut terms: Vec<Term> = coefficients
        .into_iter()
        .filter(|(_, coefficient)| *coefficient != 0)
        .map(|(blocks, coefficient)| {
            let alone: Vec<bool> = (0..hops)
                .map(|hop| blocks.iter().filter(|block| **block == blocks[hop]).count() == 1)
                .collect();
            let last_shared = alone.iter().rposition(|alone| !alone);
            Term {
                blocks,
                coefficient,
                last_shared,
                alone,
                read_alike: Vec::new(),
            }
        })
        .collect();
    terms.sort_by(|a, b| a.blocks.cmp(&b.blocks));
    terms
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The partitions of 3 hops of one type: every walk, minus each pair, plus twice
    /// all three.
    #[test]
    fn three_hops_that_must_differ_have_the_five_terms() {
        let terms = terms_of(3, &[(0, 1), (0, 2), (1, 2)]);
        let mut found: Vec<(Vec<usize>, i64)> = terms
            .into_iter()
            .map(|term| (term.blocks, term.coefficient))
            .collect();
        found.sort();
        assert_eq!(
            found,
            [
                (vec![0, 0, 0], 2),
                (vec![0, 0, 2], -1),
                (vec![0, 1, 0], -1),
                (vec![0, 1, 1], -1),
                (vec![0, 1, 2], 1),
            ]
        );
    }

    #[test]
    fn the_terms_grow_as_the_bell_numbers() {
        let pairs = |hops: usize| -> Vec<(usize, usize)> {
            (0..hops)
                .flat_map(|a| (a + 1..hops).map(move |b| (a, b)))
                .collect()
        };
        assert_eq!(terms_of(4, &pairs(4)).len(), 15);
        assert_eq!(terms_of(5, &pairs(5)).len(), 52);
    }

    /// Hops of no pair stay apart: only the one pair's two terms.
    #[test]
    fn only_the_listed_pairs_may_share() {
        assert_eq!(terms_of(3, &[(0, 2)]).len(), 2);
    }
}
