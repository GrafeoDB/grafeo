//! Shortest path operator for finding paths between nodes.
//!
//! This operator computes shortest paths between source and target nodes
//! using BFS for unweighted graphs, and the paths a selective path pattern
//! keeps (ISO/IEC 39075:2024 16.6 `<path search prefix>`): the k shortest, or
//! every path of the k shortest lengths, under a path mode.

use super::expand::{EdgeWalk, stored_types, visible_edges_from};
use super::path_expansions::{Allowance, PathExpansions, path_expansion_error};
use super::variable_length_expand::{
    DEFAULT_PATH_SEARCH_BUDGET, PathMode, PathSegment, SEGMENT_BYTES, path_budget_error,
    release_segments, with_path_count,
};
use super::{Operator, OperatorError, OperatorResult, Predicate, ReadCounter};
use crate::execution::DataChunk;
use crate::execution::chunk::DataChunkBuilder;
use crate::execution::memory::OperatorMemory;
use crate::execution::path_budget::{PathBudget, PathRefusal};
use crate::execution::path_queue::PathQueue;
use crate::graph::Direction;
use crate::graph::GraphStore;
use grafeo_common::types::{ArcStr, EdgeId, LogicalType, NodeId, ReadView, Value};
use grafeo_common::utils::hash::{FxHashMap, FxHashSet};
use std::cell::RefCell;
use std::collections::VecDeque;
use std::hash::Hash;
use std::mem::size_of;
use std::sync::Arc;

/// What to do about a shortest-path search over its budget.
const SHORTEST_PATH_ADVICE: &str = "bound the quantifier (`*1..5` or `{1,5}`), \
     keep fewer paths (`ANY SHORTEST` or `SHORTEST 3` instead of `ALL SHORTEST`), or search \
     walks instead of trails or simple paths";

/// The refusal of a shortest-path search whose maps or queues would pass its
/// budget.
const SHORTEST_PATH_REFUSAL: PathRefusal = PathRefusal {
    what: "A shortest path search",
    advice: SHORTEST_PATH_ADVICE,
};

/// The paths a search for one input row keeps, charged to its budget.
#[derive(Default)]
struct KeptPaths {
    paths: Vec<FoundPath>,
}

impl KeptPaths {
    /// Keeps the path of `hops` edges that `build` makes, charged before it
    /// is built: refused when the search would then hold more than its
    /// budget, `others` paths being held besides the kept ones (the paths a
    /// restrictive search follows).
    fn keep(
        &mut self,
        budget: &mut PathBudget,
        hops: usize,
        others: usize,
        build: impl FnOnce() -> FoundPath,
    ) -> Result<(), OperatorError> {
        budget.reserve_vec(&mut self.paths, 1)?;
        charge_paths(budget, path_bytes(hops), self.paths.len() + others)?;
        self.paths.push(build());
        Ok(())
    }
}

/// The bytes the two lists of a path of `hops` edges hold.
fn path_bytes(hops: usize) -> usize {
    hops.saturating_mul(size_of::<NodeId>() + size_of::<EdgeId>())
        .saturating_add(size_of::<NodeId>())
}

/// Charges `bytes` of paths to `budget`, the search holding `paths` paths:
/// past the budget the refusal names the paths held, past the statement's
/// grant it is the grant's.
fn charge_paths(budget: &mut PathBudget, bytes: usize, paths: usize) -> Result<(), OperatorError> {
    if budget.used().saturating_add(bytes) > budget.limit() {
        return Err(path_budget_error(
            "A shortest path search",
            paths,
            (bytes, budget.used()),
            budget.limit(),
            SHORTEST_PATH_ADVICE,
        ));
    }
    budget.charge(bytes)
}

/// Pushes `value` onto `values`, charged to `budget` before it grows.
fn push<T>(budget: &mut PathBudget, values: &mut Vec<T>, value: T) -> Result<(), OperatorError> {
    budget.reserve_vec(values, 1)?;
    values.push(value);
    Ok(())
}

/// Gives back to `budget` what a list grown through [`push`] held, now that
/// it is dropped.
fn release_list<T>(budget: &mut PathBudget, values: Vec<T>) {
    budget.release(values.capacity().saturating_mul(size_of::<T>()));
}

/// Pushes `value` onto the back of `queue`, charged to `budget` before it
/// grows.
#[inline]
fn push_back<T>(
    budget: &mut PathBudget,
    queue: &mut VecDeque<T>,
    value: T,
) -> Result<(), OperatorError> {
    budget.reserve_queue(queue, 1)?;
    queue.push_back(value);
    Ok(())
}

/// Inserts `key` into `map`, charged to `budget` before it grows.
#[inline]
fn insert<K: Eq + Hash, V>(
    budget: &mut PathBudget,
    map: &mut FxHashMap<K, V>,
    key: K,
    value: V,
) -> Result<(), OperatorError> {
    budget.reserve_map(map, 1)?;
    map.insert(key, value);
    Ok(())
}

/// Records the step `step` into `node` in `steps`, charged to `budget` before
/// anything grows: every step with `all`, only the first one otherwise.
/// Returns whether `node` is new to `steps`.
fn record_step(
    budget: &mut PathBudget,
    steps: &mut Steps,
    node: NodeId,
    step: (NodeId, EdgeId),
    all: bool,
) -> Result<bool, OperatorError> {
    if let Some(into) = steps.get_mut(&node) {
        if all {
            push(budget, into, step)?;
        }
        return Ok(false);
    }
    budget.reserve_map(steps, 1)?;
    let mut into = Vec::new();
    push(budget, &mut into, step)?;
    steps.insert(node, into);
    Ok(true)
}

/// Which paths a search keeps for each pair of a source and a target: the
/// selection of a path search prefix (ISO/IEC 39075:2024 16.6), among the
/// paths that fit the hop bounds, the path mode and the edge condition.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum PathSelection {
    /// The `k` shortest paths (`ANY SHORTEST` and `shortestPath` are 1,
    /// `SHORTEST k` is k; `ANY k` takes these too). Paths of one length
    /// come in the order the search finds them.
    Shortest(usize),
    /// Every path of the `k` shortest lengths (`ALL SHORTEST` and
    /// `allShortestPaths` are 1, `SHORTEST k GROUPS` is k).
    ShortestGroups(usize),
}

impl PathSelection {
    /// The number of paths or groups.
    fn count(self) -> usize {
        match self {
            Self::Shortest(count) | Self::ShortestGroups(count) => count,
        }
    }
}

/// Operator that finds shortest paths between source and target nodes.
///
/// For each input row containing source and target nodes, this operator
/// computes the shortest path within the hop bounds and outputs its length:
/// one row per shortest path for `allShortestPaths`, one row otherwise, and no
/// row when no path fits (`OPTIONAL MATCH` adds the nulls with a left join).
/// Other selections keep more paths of each pair (see
/// [`Self::with_selection`]), a path mode restricts the paths (see
/// [`Self::with_path_mode`]), and [`Self::from_source`] searches from the
/// source to every node instead of to a target of the row. It can also
/// output the edges of the path for the edge variable (see
/// [`Self::with_edge_output`]) and the path itself (see
/// [`Self::with_path_output`]), and search only edges that meet a condition
/// (see [`Self::with_edge_condition`]).
pub struct ShortestPathOperator {
    /// The graph store.
    store: Arc<dyn GraphStore>,
    /// Input operator providing source/target node pairs.
    input: Box<dyn Operator>,
    /// Column index of the source node.
    source_column: usize,
    /// Column index of the target node; `None` when the search binds the
    /// target, to every node it reaches (see [`Self::from_source`]).
    target_column: Option<usize>,
    /// The edge types to follow, as stored (every type when empty).
    edge_types: Vec<ArcStr>,
    /// Direction of edge traversal.
    direction: Direction,
    /// The paths kept for each pair.
    selection: PathSelection,
    /// The paths the search follows: WALK, TRAIL, SIMPLE or ACYCLIC.
    path_mode: PathMode,
    /// Minimum number of edges in a path.
    min_hops: u32,
    /// Maximum number of edges in a path (`None` = unbounded).
    max_hops: Option<u32>,
    /// Who reads: the search walks only the edges (and reaches only the
    /// nodes) this view sees.
    view: ReadView,
    /// Counts the edges the expansion reads.
    reads: Option<Arc<ReadCounter>>,
    /// The path expansions the statement may still make (see
    /// [`Self::with_path_expansions`]); `None` outside a statement, without a limit.
    expansions: Option<Arc<PathExpansions>>,
    /// The columns written after the length.
    outputs: PathOutputs,
    /// A condition every edge of a path meets (see [`Self::with_edge_condition`]).
    edge_condition: Option<Box<dyn Predicate>>,
    /// The bytes the search for one input row may hold (see
    /// [`Self::with_memory_budget`]).
    budget: usize,
    /// The statement's memory, captured at construction: what the searches
    /// hold is a grant of it too, under the query memory limit.
    memory: OperatorMemory,
    /// What a row's search holds, against `budget` and as a grant of
    /// `memory` (made at the first search); kept across rows, so the grant is
    /// too, and given back when the operator drops.
    path_budget: Option<PathBudget>,
    /// The input chunk whose rows are being searched, when some are left.
    pending: Option<PendingInput>,
    /// Whether the operator has been exhausted.
    exhausted: bool,
}

/// An input chunk and the rows of it still to search: a chunk of output
/// ends when its paths hold enough ids, and the next call goes on from the
/// next row.
struct PendingInput {
    chunk: DataChunk,
    /// The chunk's selected rows.
    rows: Vec<usize>,
    /// The next of `rows` to search.
    next: usize,
    /// The row searched last, and its paths not yet written: one row's
    /// paths can take more than a chunk.
    found: Option<(usize, std::vec::IntoIter<FoundPath>)>,
}

/// The columns a shortest-path search writes after the path length.
#[derive(Debug, Clone, Copy, Default)]
struct PathOutputs {
    /// What the edge column holds, when there is one.
    edge: Option<EdgeOutput>,
    /// Whether to write the nodes, the edges and the value of each path.
    path: bool,
}

/// What the edge column of a shortest-path search holds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum EdgeOutput {
    /// The list of the path's edges: the variable of a quantified edge
    /// pattern is a group variable.
    List,
    /// The path's one edge: the variable of an edge pattern without a
    /// quantifier.
    Single,
}

/// A path a search found: its nodes from the source to the target, and the
/// edges between them.
#[derive(Debug, Clone, PartialEq, Eq)]
struct FoundPath {
    nodes: Vec<NodeId>,
    edges: Vec<EdgeId>,
}

impl FoundPath {
    /// The path of no edges at `node`.
    fn at(node: NodeId) -> Self {
        Self {
            nodes: vec![node],
            edges: Vec::new(),
        }
    }
}

impl ShortestPathOperator {
    /// Creates a shortest-path search between the source and the target of
    /// each input row, along the edges of `edge_types` (as stored, every
    /// type when empty) that `view` sees.
    pub fn new(
        store: Arc<dyn GraphStore>,
        input: Box<dyn Operator>,
        source_column: usize,
        target_column: usize,
        edge_types: Vec<String>,
        direction: Direction,
        view: ReadView,
    ) -> Self {
        Self::searching(
            store,
            input,
            source_column,
            Some(target_column),
            edge_types,
            direction,
            view,
        )
    }

    /// Creates a search from the source of each input row to every node it
    /// reaches: the paths to each node are a pair of their own, and the
    /// output gets a column after the input columns for the node, the
    /// target of the row's path.
    pub fn from_source(
        store: Arc<dyn GraphStore>,
        input: Box<dyn Operator>,
        source_column: usize,
        edge_types: Vec<String>,
        direction: Direction,
        view: ReadView,
    ) -> Self {
        Self::searching(
            store,
            input,
            source_column,
            None,
            edge_types,
            direction,
            view,
        )
    }

    fn searching(
        store: Arc<dyn GraphStore>,
        input: Box<dyn Operator>,
        source_column: usize,
        target_column: Option<usize>,
        edge_types: Vec<String>,
        direction: Direction,
        view: ReadView,
    ) -> Self {
        Self {
            store,
            input,
            source_column,
            target_column,
            edge_types: stored_types(edge_types),
            direction,
            selection: PathSelection::Shortest(1),
            path_mode: PathMode::Walk,
            min_hops: 0,
            max_hops: None,
            view,
            reads: None,
            expansions: None,
            outputs: PathOutputs::default(),
            edge_condition: None,
            budget: DEFAULT_PATH_SEARCH_BUDGET,
            memory: crate::execution::memory::capture(),
            path_budget: None,
            pending: None,
            exhausted: false,
        }
    }

    /// Sets the memory the search for one input row may hold, in bytes:
    /// its maps and queues, the paths a restrictive path mode follows, and
    /// the paths the search keeps (the default is
    /// [`DEFAULT_PATH_SEARCH_BUDGET`](super::DEFAULT_PATH_SEARCH_BUDGET)). A
    /// search that would hold more fails with `GRAFEO-S001`, whose message
    /// names the ways to need fewer paths and the setting.
    pub fn with_memory_budget(mut self, bytes: usize) -> Self {
        self.budget = bytes;
        self
    }

    /// Counts the edges the expansion reads in `reads`.
    #[must_use]
    pub fn with_read_counter(mut self, reads: Arc<ReadCounter>) -> Self {
        self.reads = Some(reads);
        self
    }

    /// Takes the path expansions the search makes from the statement's `expansions`: each
    /// edge it reads to extend a path, and each node and edge id of a path it returns in a
    /// row. Past the statement's limit it fails with `GRAFEO-Q009`. Without it the search is
    /// not limited.
    #[must_use]
    pub fn with_path_expansions(mut self, expansions: Arc<PathExpansions>) -> Self {
        self.expansions = Some(expansions);
        self
    }

    /// The error of a search past the statement's path expansion limit.
    fn expansions_out(&self) -> OperatorError {
        path_expansion_error(
            "A shortest path search",
            self.expansions.as_deref(),
            SHORTEST_PATH_ADVICE,
        )
    }

    /// Sets whether to find all shortest paths.
    pub fn with_all_paths(mut self, all_paths: bool) -> Self {
        self.selection = if all_paths {
            PathSelection::ShortestGroups(1)
        } else {
            PathSelection::Shortest(1)
        };
        self
    }

    /// Sets which paths of each pair the search keeps. The default is the
    /// one shortest path.
    pub fn with_selection(mut self, selection: PathSelection) -> Self {
        self.selection = selection;
        self
    }

    /// Restricts the paths to those of a path mode: TRAIL repeats no edge,
    /// ACYCLIC no node, SIMPLE no node but the first as the last. The
    /// default, WALK, allows every repeat.
    pub fn with_path_mode(mut self, path_mode: PathMode) -> Self {
        self.path_mode = path_mode;
        self
    }

    /// Limits the paths to between `min_hops` and `max_hops` edges (`None` =
    /// unbounded). The default is 0 to unbounded, so a node paired with
    /// itself has a zero-length path.
    pub fn with_hop_bounds(mut self, min_hops: u32, max_hops: Option<u32>) -> Self {
        self.min_hops = min_hops;
        self.max_hops = max_hops;
        self
    }

    /// Adds a column after the length for the variable of the edge pattern:
    /// the list of the path's edges with `as_list` (the group variable of a
    /// quantified edge pattern, as edge ids like a variable-length expand
    /// writes them), otherwise the path's one edge.
    pub fn with_edge_output(mut self, as_list: bool) -> Self {
        self.outputs.edge = Some(if as_list {
            EdgeOutput::List
        } else {
            EdgeOutput::Single
        });
        self
    }

    /// Adds three columns after the length (and the edge column): the ids of
    /// the path's nodes, the ids of its edges, and the path as a
    /// `Value::Path`, as a variable-length expand writes them for a named
    /// path.
    pub fn with_path_output(mut self) -> Self {
        self.outputs.path = true;
        self
    }

    /// Searches only the edges that meet `condition`, so that every edge of
    /// a path meets it before a shortest one is picked: the property map or
    /// the `WHERE` of the edge pattern, or the equality with an edge bound
    /// before. The condition reads one row: the columns of the input row,
    /// then the candidate edge in the column after them.
    pub fn with_edge_condition(mut self, condition: Box<dyn Predicate>) -> Self {
        self.edge_condition = Some(condition);
        self
    }

    /// The search for the input row `row` of `chunk`, taking its expansions from
    /// `allowance`.
    fn search_for_row(
        &self,
        chunk: &DataChunk,
        row: usize,
        allowance: Allowance,
    ) -> PathSearch<'_> {
        let condition_row = self.edge_condition.as_ref().map(|_| {
            let mut types: Vec<LogicalType> = (0..chunk.column_count())
                .map(|i| {
                    chunk
                        .column(i)
                        .map_or(LogicalType::Any, |c| c.data_type().clone())
                })
                .collect();
            types.push(LogicalType::Edge);
            let mut condition_row = DataChunk::with_capacity(&types, 1);
            for i in 0..chunk.column_count() {
                if let (Some(from), Some(to)) = (chunk.column(i), condition_row.column_mut(i)) {
                    from.copy_row_to(row, to);
                }
            }
            if let Some(edge) = condition_row.column_mut(chunk.column_count()) {
                edge.push_edge_id(EdgeId(0));
            }
            condition_row.set_count(1);
            RefCell::new(condition_row)
        });
        PathSearch {
            operator: self,
            condition_row,
            condition_error: RefCell::new(None),
            allowance: RefCell::new(allowance),
        }
    }
}

/// The search of one input row: the operator's settings, and the row the
/// edge condition reads (the input row and a candidate edge).
struct PathSearch<'a> {
    /// The operator whose search this is.
    operator: &'a ShortestPathOperator,
    /// The input row and, in its last column, the edge the condition checks.
    condition_row: Option<RefCell<DataChunk>>,
    /// The first error of the edge condition or of a read of the store,
    /// which fails the search (see [`paths`](Self::paths)).
    condition_error: RefCell<Option<OperatorError>>,
    /// The expansions the search may still make: past them it fails as on a
    /// failed read.
    allowance: RefCell<Allowance>,
}

/// The steps that reach each node of a layer of a search: from which node,
/// over which edge.
type Steps = FxHashMap<NodeId, Vec<(NodeId, EdgeId)>>;

impl PathSearch<'_> {
    /// The neighbors of `node` in `direction` and the edges to them: the
    /// edges of the wanted types this query sees (see [`visible_edges_from`])
    /// that meet the edge condition. A read that fails ends the search with
    /// its error (see [`paths`](Self::paths)); a self-loop is one step, also
    /// both ways.
    fn neighbors(&self, node: NodeId, direction: Direction) -> Vec<(NodeId, EdgeId)> {
        let op = self.operator;
        let found = visible_edges_from(
            op.store.as_ref(),
            node,
            direction,
            &EdgeWalk {
                edge_types: &op.edge_types,
                view: op.view,
                reads: op.reads.as_deref(),
            },
        );
        let mut edges = match found {
            Ok(edges) => edges,
            Err(error) => {
                self.condition_error.borrow_mut().get_or_insert(error);
                return Vec::new();
            }
        };
        // Each edge read is an expansion; past the limit the search reads no
        // more and ends with the error
        if !self.allowance.borrow_mut().take(edges.len()) {
            self.condition_error
                .borrow_mut()
                .get_or_insert_with(|| op.expansions_out());
            return Vec::new();
        }
        if let (Some(condition), Some(row)) = (&op.edge_condition, &self.condition_row) {
            let mut row = row.borrow_mut();
            let edge_column = row.column_count() - 1;
            edges.retain(|&(_, edge)| {
                let Some(column) = row.column_mut(edge_column) else {
                    return false;
                };
                column.clear();
                column.push_edge_id(edge);
                match condition.evaluate(&row, 0) {
                    Ok(passes) => passes,
                    Err(error) => {
                        self.condition_error.borrow_mut().get_or_insert(error);
                        false
                    }
                }
            });
        }
        edges
    }

    /// The paths the selection keeps from `source` to `target` within the
    /// hop bounds, or to every node the search reaches when there is no
    /// target; none when no path fits. What the search holds is charged to
    /// `budget` before it is allocated: it fails when the search would hold
    /// more than the operator's budget, or than the statement's grant.
    fn paths(
        &self,
        budget: &mut PathBudget,
        source: NodeId,
        target: Option<NodeId>,
    ) -> Result<Vec<FoundPath>, OperatorError> {
        let found = self.search_paths(budget, source, target);
        match self.condition_error.borrow_mut().take() {
            Some(error) => Err(error),
            None => found,
        }
    }

    /// The paths of [`paths`](Self::paths), before an error of the edge
    /// condition is checked.
    fn search_paths(
        &self,
        budget: &mut PathBudget,
        source: NodeId,
        target: Option<NodeId>,
    ) -> Result<Vec<FoundPath>, OperatorError> {
        let op = self.operator;
        if op.selection.count() == 0 {
            return Ok(Vec::new());
        }
        if op.path_mode != PathMode::Walk {
            return self.restricted_paths(budget, source, target);
        }
        let Some(target) = target else {
            return self.selected_walks(budget, source, None);
        };
        match op.selection {
            PathSelection::ShortestGroups(1) => self.shortest_walks(budget, source, target, true),
            PathSelection::Shortest(1) => {
                // Without a minimum, or with a minimum of one hop between two
                // different nodes, the plain shortest path has enough hops.
                let plain = op.min_hops == 0 || (op.min_hops == 1 && source != target);
                if !plain {
                    return self.shortest_walks(budget, source, target, false);
                }
                let mut kept = KeptPaths::default();
                self.shortest_path_bidirectional(budget, &mut kept, source, target)?;
                Ok(kept.paths)
            }
            _ => self.selected_walks(budget, source, Some(target)),
        }
    }

    /// The walks (WALK mode) the selection keeps from `source` to `target`,
    /// or to every node it reaches when there is no target, in the order of
    /// their lengths.
    ///
    /// The search goes layer by layer: layer `d` holds the nodes a walk of
    /// `d` edges ends at, with the steps into them. A node reached at `k`
    /// lengths of at least `min_hops`, `d1 < ... < dk`, is not expanded at
    /// a later length `d`: whatever a walk through it at `d` reaches in `r`
    /// more edges, the walks through it at the `d_i` reach in `k` different
    /// numbers of edges below `d + r`, so a walk on from `d` is beyond the
    /// `k` shortest lengths of its end, and beyond its `k` shortest walks.
    /// (Below `min_hops` every depth is a step of its own, which each layer
    /// keeps once.) So every node is expanded at `k` lengths at most, and
    /// the search ends without a maximum too. The walks are read back from
    /// each end along the steps: all of those of the `k` shortest lengths
    /// for groups, the first `k` otherwise.
    fn selected_walks(
        &self,
        budget: &mut PathBudget,
        source: NodeId,
        target: Option<NodeId>,
    ) -> Result<Vec<FoundPath>, OperatorError> {
        let op = self.operator;
        let count = op.selection.count();
        let min_hops = usize::try_from(op.min_hops).unwrap_or(usize::MAX);
        let max_hops = op
            .max_hops
            .map(|max| usize::try_from(max).unwrap_or(usize::MAX));
        // The steps into the nodes of each layer; layer 0 is the source
        let mut layers: Vec<Steps> = Vec::new();
        push(budget, &mut layers, Steps::default())?;
        let mut frontier = Vec::new();
        push(budget, &mut frontier, source)?;
        // How many lengths from `min_hops` on each node was expanded at
        let mut lengths: FxHashMap<NodeId, usize> = FxHashMap::default();
        // The ends of the walks to read back: a node and its depth
        let mut ends: Vec<(NodeId, usize)> = Vec::new();
        if min_hops == 0 {
            insert(budget, &mut lengths, source, 1)?;
            if target.is_none_or(|target| target == source) {
                push(budget, &mut ends, (source, 0))?;
            }
        }
        let mut depth = 0;
        while max_hops.is_none_or(|max| depth < max) {
            // The target has its lengths: every later walk is longer
            if let Some(target) = target
                && lengths
                    .get(&target)
                    .is_some_and(|&reached| reached >= count)
            {
                break;
            }
            depth += 1;
            let mut reached = Vec::new();
            let mut steps = Steps::default();
            for &node in &frontier {
                for (neighbor, edge) in self.neighbors(node, op.direction) {
                    if record_step(budget, &mut steps, neighbor, (node, edge), true)? {
                        push(budget, &mut reached, neighbor)?;
                    }
                }
            }
            if depth >= min_hops {
                let new = reached
                    .iter()
                    .filter(|&node| !lengths.contains_key(node))
                    .count();
                budget.reserve_map(&mut lengths, new)?;
                reached.retain(|node| {
                    let expanded = lengths.entry(*node).or_insert(0);
                    if *expanded >= count {
                        return false;
                    }
                    *expanded += 1;
                    true
                });
                let mut kept = FxHashSet::default();
                budget.reserve_set(&mut kept, reached.len())?;
                kept.extend(reached.iter().copied());
                // A dropped list gives its charge back
                steps.retain(|node, into| {
                    let keep = kept.contains(node);
                    if !keep {
                        release_list(budget, std::mem::take(into));
                    }
                    keep
                });
                budget.release(kept.allocation_size());
                for &node in &reached {
                    if target.is_none_or(|target| target == node) {
                        push(budget, &mut ends, (node, depth))?;
                    }
                }
            }
            if reached.is_empty() {
                release_list(budget, reached);
                break;
            }
            push(budget, &mut layers, steps)?;
            release_list(budget, std::mem::replace(&mut frontier, reached));
        }

        let groups = matches!(op.selection, PathSelection::ShortestGroups(_));
        let mut taken: FxHashMap<NodeId, usize> = FxHashMap::default();
        let mut kept = KeptPaths::default();
        for (end, depth) in ends {
            if !taken.contains_key(&end) {
                budget.reserve_map(&mut taken, 1)?;
            }
            let taken = taken.entry(end).or_insert(0);
            let limit = if groups {
                usize::MAX
            } else {
                count.saturating_sub(*taken)
            };
            if limit == 0 {
                continue;
            }
            *taken += read_back(budget, end, depth, limit, &mut kept, |node, depth| {
                layers
                    .get(depth)
                    .and_then(|layer| layer.get(&node))
                    .map_or(&[], Vec::as_slice)
            })?;
        }
        Ok(kept.paths)
    }

    /// The paths a restrictive path mode allows (TRAIL, SIMPLE, ACYCLIC)
    /// that the selection keeps from `source` to `target`, or to every node
    /// the search reaches when there is no target, in the order of their
    /// lengths.
    ///
    /// A breadth-first search over the allowed paths, which are finitely
    /// many: one path can need another to its own end that is not among
    /// the shortest there (a trail on from a node can need the edges a
    /// shorter trail to that node took), so every allowed path is followed,
    /// up to the maximum, and the selection takes the first of each end. A
    /// SIMPLE path back at its first node ends there.
    ///
    /// The paths followed and kept may hold at most the operator's budget:
    /// the allowed paths can multiply with every hop. A path's segment is
    /// charged when it is queued and given back when the last path through
    /// it is done.
    fn restricted_paths(
        &self,
        budget: &mut PathBudget,
        source: NodeId,
        target: Option<NodeId>,
    ) -> Result<Vec<FoundPath>, OperatorError> {
        /// What the selection kept of the paths to one end.
        #[derive(Default)]
        struct Kept {
            /// The number of paths kept.
            paths: usize,
            /// The number of lengths kept.
            lengths: usize,
            /// The length of the last path kept.
            last: u32,
        }

        const REFUSED: &str = "A shortest path search";
        let op = self.operator;
        let count = op.selection.count();
        let groups = matches!(op.selection, PathSelection::ShortestGroups(_));
        let mut kept: FxHashMap<NodeId, Kept> = FxHashMap::default();
        let mut paths = KeptPaths::default();
        // In blocks, as the variable-length search queues: a doubling queue
        // holds its old and new buffer at once while it grows
        let mut queue: PathQueue<(Arc<PathSegment>, u32)> = PathQueue::new();
        charge_paths(budget, SEGMENT_BYTES, 0)?;
        let root = Arc::new(PathSegment {
            node: source,
            edge: None,
            parent: None,
        });
        queue
            .reserve(budget)
            .map_err(|error| with_path_count(error, REFUSED, 0, budget, SHORTEST_PATH_ADVICE))?;
        queue.push((root, 0));
        // The path being expanded, out of the queue: held until the next is
        // taken, then given back with the segments only it held
        let mut expanded: Option<Arc<PathSegment>> = None;
        while let Some((segment, depth)) = queue.pop(budget) {
            if let Some(done) = expanded.replace(Arc::clone(&segment)) {
                release_segments(budget, done);
            }
            // The target has its paths: every later path is as long or longer
            if let Some(end) = target.and_then(|target| kept.get(&target)) {
                let done = if groups {
                    end.lengths >= count && depth > end.last
                } else {
                    end.paths >= count
                };
                if done {
                    break;
                }
            }
            if depth >= op.min_hops && target.is_none_or(|target| target == segment.node) {
                if !kept.contains_key(&segment.node) {
                    budget.reserve_map(&mut kept, 1)?;
                }
                let end = kept.entry(segment.node).or_default();
                let keep = if groups {
                    if end.paths > 0 && depth == end.last {
                        true
                    } else if end.lengths < count {
                        end.lengths += 1;
                        end.last = depth;
                        true
                    } else {
                        false
                    }
                } else {
                    end.paths < count
                };
                if keep {
                    end.paths += 1;
                    // The paths held besides the kept ones: those queued and
                    // the one expanded
                    let hops = usize::try_from(depth).unwrap_or(usize::MAX);
                    paths.keep(budget, hops, queue.len() + 1, || FoundPath {
                        nodes: segment.collect_nodes(depth),
                        edges: segment.collect_edges(depth),
                    })?;
                }
            }
            let closed = op.path_mode == PathMode::Simple && depth > 0 && segment.node == source;
            if closed || op.max_hops.is_some_and(|max| depth >= max) {
                continue;
            }
            for (neighbor, edge) in self.neighbors(segment.node, op.direction) {
                let allowed = match op.path_mode {
                    PathMode::Walk => true,
                    PathMode::Trail => !segment.contains_edge(edge),
                    PathMode::Simple => neighbor == source || !segment.contains_node(neighbor),
                    PathMode::Acyclic => !segment.contains_node(neighbor),
                };
                if allowed {
                    // The paths held: those queued, the one expanded and the
                    // kept ones
                    let held = queue.len() + 1 + paths.paths.len();
                    charge_paths(budget, SEGMENT_BYTES, held)?;
                    let next = PathSegment {
                        node: neighbor,
                        edge: Some(edge),
                        parent: Some(Arc::clone(&segment)),
                    };
                    queue.reserve(budget).map_err(|error| {
                        with_path_count(error, REFUSED, held, budget, SHORTEST_PATH_ADVICE)
                    })?;
                    queue.push((Arc::new(next), depth + 1));
                }
            }
        }
        Ok(paths.paths)
    }

    /// Keeps in `kept` the shortest path of `hops` edges that `build` makes,
    /// unless it is longer than the maximum.
    fn keep_shortest(
        &self,
        budget: &mut PathBudget,
        kept: &mut KeptPaths,
        hops: usize,
        build: impl FnOnce() -> FoundPath,
    ) -> Result<(), OperatorError> {
        if self
            .operator
            .max_hops
            .is_some_and(|max| usize::try_from(max).is_ok_and(|max| hops > max))
        {
            return Ok(());
        }
        kept.keep(budget, hops, 0, build)
    }

    /// Finds one shortest path with a breadth-first search from `source`,
    /// and keeps it in `kept`.
    fn shortest_path(
        &self,
        budget: &mut PathBudget,
        kept: &mut KeptPaths,
        source: NodeId,
        target: NodeId,
    ) -> Result<(), OperatorError> {
        if source == target {
            return self.keep_shortest(budget, kept, 0, || FoundPath::at(source));
        }

        // The step that first reached each node
        let mut parents: FxHashMap<NodeId, (NodeId, EdgeId)> = FxHashMap::default();
        let mut queue: VecDeque<NodeId> = VecDeque::new();
        push_back(budget, &mut queue, source)?;

        while let Some(current) = queue.pop_front() {
            for (neighbor, edge) in self.neighbors(current, self.operator.direction) {
                if neighbor == target {
                    insert(budget, &mut parents, target, (current, edge))?;
                    let mut hops = 0;
                    let mut node = target;
                    while node != source {
                        let Some(&(previous, _)) = parents.get(&node) else {
                            break;
                        };
                        hops += 1;
                        node = previous;
                    }
                    return self.keep_shortest(budget, kept, hops, || {
                        Self::path_back(&parents, source, target, hops)
                    });
                }
                if neighbor != source && !parents.contains_key(&neighbor) {
                    insert(budget, &mut parents, neighbor, (current, edge))?;
                    push_back(budget, &mut queue, neighbor)?;
                }
            }
        }

        Ok(()) // No path found
    }

    /// The path of `hops` edges from `source` to `node`, following the steps
    /// in `parents` back from `node`.
    fn path_back(
        parents: &FxHashMap<NodeId, (NodeId, EdgeId)>,
        source: NodeId,
        node: NodeId,
        hops: usize,
    ) -> FoundPath {
        let mut path = FoundPath {
            nodes: Vec::with_capacity(hops + 1),
            edges: Vec::with_capacity(hops),
        };
        path.nodes.push(node);
        let mut current = node;
        while current != source {
            let Some(&(previous, edge)) = parents.get(&current) else {
                break;
            };
            path.nodes.push(previous);
            path.edges.push(edge);
            current = previous;
        }
        path.nodes.reverse();
        path.edges.reverse();
        path
    }

    /// Finds the shortest paths from `source` to `target` within the hop
    /// bounds: all of them with `all`, otherwise the first one.
    ///
    /// A path of at least `min_hops` edges starts with exactly `min_hops`
    /// steps, so this first walks those steps from `source`, layer by layer,
    /// then runs one breadth-first search from all of their end nodes at once.
    /// With `min_hops` 0 that is a plain BFS from `source`; with `min_hops` 1
    /// and `source == target` it finds the shortest cycles through `source`.
    /// Every layer keeps the steps into each of its nodes (all of them with
    /// `all`), and the paths are read back from `target` along those steps.
    /// A shortest continuation never passes another end node of the first
    /// steps, so each path is found once.
    fn shortest_walks(
        &self,
        budget: &mut PathBudget,
        source: NodeId,
        target: NodeId,
        all: bool,
    ) -> Result<Vec<FoundPath>, OperatorError> {
        let op = self.operator;
        let min_hops = usize::try_from(op.min_hops).unwrap_or(usize::MAX);
        let max_hops = op
            .max_hops
            .map(|max| usize::try_from(max).unwrap_or(usize::MAX));
        if max_hops.is_some_and(|max| max < min_hops) {
            return Ok(Vec::new());
        }

        // Walks of exactly `min_hops` edges: `layers[i]` holds the steps into
        // the nodes the walks reach after `i + 1` edges (grown a layer at a
        // time: the minimum comes from the query, the layers from the graph)
        let mut layers: Vec<Steps> = Vec::new();
        let mut current: Vec<NodeId> = Vec::new();
        push(budget, &mut current, source)?;
        for _ in 0..min_hops {
            let mut steps = Steps::default();
            let mut next = Vec::new();
            for &node in &current {
                for (neighbor, edge) in self.neighbors(node, op.direction) {
                    if record_step(budget, &mut steps, neighbor, (node, edge), all)? {
                        push(budget, &mut next, neighbor)?;
                    }
                }
            }
            if next.is_empty() {
                return Ok(Vec::new());
            }
            push(budget, &mut layers, steps)?;
            release_list(budget, std::mem::replace(&mut current, next));
        }

        // Level-by-level BFS from all of them: `lengths` holds the length of
        // the shortest paths to every node reached so far, `steps` the steps
        // into the nodes past the first `min_hops` edges
        let mut lengths: FxHashMap<NodeId, usize> = FxHashMap::default();
        budget.reserve_map(&mut lengths, current.len())?;
        lengths.extend(current.iter().map(|&node| (node, min_hops)));
        let mut steps = Steps::default();
        let mut frontier = current;
        let mut length = min_hops;
        while !lengths.contains_key(&target) {
            if frontier.is_empty() || max_hops.is_some_and(|max| length >= max) {
                return Ok(Vec::new());
            }
            length += 1;

            let mut next_frontier = Vec::new();
            let expanded = frontier.capacity().saturating_mul(size_of::<NodeId>());
            for node in frontier {
                for (neighbor, edge) in self.neighbors(node, op.direction) {
                    match lengths.get(&neighbor) {
                        None => {
                            insert(budget, &mut lengths, neighbor, length)?;
                            push(budget, &mut next_frontier, neighbor)?;
                            record_step(budget, &mut steps, neighbor, (node, edge), all)?;
                        }
                        Some(&reached_at) if reached_at == length => {
                            record_step(budget, &mut steps, neighbor, (node, edge), all)?;
                        }
                        // Already reached by a shorter path
                        Some(_) => {}
                    }
                }
            }
            budget.release(expanded);
            frontier = next_frontier;
        }

        let limit = if all { usize::MAX } else { 1 };
        let mut kept = KeptPaths::default();
        read_back(budget, target, length, limit, &mut kept, |node, depth| {
            steps_into(&layers, &steps, min_hops, node, depth)
        })?;
        Ok(kept.paths)
    }

    /// Finds shortest path using bidirectional BFS, and keeps it in `kept`.
    ///
    /// Maintains forward and backward frontiers, alternating expansion of the
    /// smaller one. When a node is found in both visited sets, the shortest
    /// path is `forward_depth + backward_depth`. This reduces the search space
    /// from O(b^d) to O(b^(d/2)) where b is the branching factor and d is
    /// the path length.
    ///
    /// Falls back to unidirectional BFS if backward adjacency is unavailable.
    ///
    /// Each side keeps the step that first reached each of its nodes (the
    /// backward side the step toward `target`), and the best meeting, the
    /// edge between the two sides, so that the path can be read back.
    fn shortest_path_bidirectional(
        &self,
        budget: &mut PathBudget,
        kept: &mut KeptPaths,
        source: NodeId,
        target: NodeId,
    ) -> Result<(), OperatorError> {
        if source == target {
            return self.keep_shortest(budget, kept, 0, || FoundPath::at(source));
        }

        // Fall back to unidirectional if backward adjacency is unavailable
        let op = self.operator;
        if !op.store.has_backward_adjacency() {
            return self.shortest_path(budget, kept, source, target);
        }

        let reverse_dir = op.direction.reverse();

        // Forward BFS state: the depth of each node and the step into it
        let mut forward_visited: FxHashMap<NodeId, (i64, Option<(NodeId, EdgeId)>)> =
            FxHashMap::default();
        let mut forward_queue: VecDeque<(NodeId, i64)> = VecDeque::new();
        insert(budget, &mut forward_visited, source, (0, None))?;
        push_back(budget, &mut forward_queue, (source, 0))?;

        // Backward BFS state: the depth of each node and the step from it
        // toward the target
        let mut backward_visited: FxHashMap<NodeId, (i64, Option<(NodeId, EdgeId)>)> =
            FxHashMap::default();
        let mut backward_queue: VecDeque<(NodeId, i64)> = VecDeque::new();
        insert(budget, &mut backward_visited, target, (0, None))?;
        push_back(budget, &mut backward_queue, (target, 0))?;

        // Best known path: its length (upper bound) and where the two sides
        // meet, the edge from a forward node to a backward node
        let mut best: Option<i64> = None;
        let mut meeting: Option<(NodeId, EdgeId, NodeId)> = None;
        let mut last_forward = false;

        loop {
            // Expand the smaller frontier, the sides taking turns on a tie, so
            // that a side with nothing left is seen at once. A side that runs
            // out before the sides met has expanded everything it reaches,
            // which holds the node next to the other end on any path: there is
            // none.
            let expand_forward = match (forward_queue.front(), backward_queue.front()) {
                (Some(_), Some(_)) => match forward_queue.len().cmp(&backward_queue.len()) {
                    std::cmp::Ordering::Less => true,
                    std::cmp::Ordering::Greater => false,
                    std::cmp::Ordering::Equal => !last_forward,
                },
                _ if best.is_none() => break,
                (Some(_), None) => true,
                (None, Some(_)) => false,
                (None, None) => break,
            };
            last_forward = expand_forward;

            if expand_forward {
                let Some((current, depth)) = forward_queue.pop_front() else {
                    break;
                };

                // A node at this depth finds paths of depth + 1 or more, and a
                // better one must be shorter than the best: this side is done
                if let Some(b) = best
                    && depth + 1 >= b
                {
                    // Clear the queue; no further expansion can improve
                    forward_queue.clear();
                    continue;
                }

                for (neighbor, edge) in self.neighbors(current, op.direction) {
                    let new_depth = depth + 1;

                    // Check if backward frontier already visited this node
                    if let Some(&(backward_depth, _)) = backward_visited.get(&neighbor) {
                        let total = new_depth + backward_depth;
                        if best.is_none_or(|b| total < b) {
                            best = Some(total);
                            meeting = Some((current, edge, neighbor));
                        }
                    }

                    if !forward_visited.contains_key(&neighbor) {
                        let step = (new_depth, Some((current, edge)));
                        // Written out: this loop runs per visit, and the helpers out
                        // of line cost about 13 instructions a visit (Callgrind).
                        budget.reserve_map(&mut forward_visited, 1)?;
                        forward_visited.insert(neighbor, step);
                        if best.is_none_or(|b| new_depth < b) {
                            budget.reserve_queue(&mut forward_queue, 1)?;
                            forward_queue.push_back((neighbor, new_depth));
                        }
                    }
                }
            } else {
                let Some((current, depth)) = backward_queue.pop_front() else {
                    break;
                };

                if let Some(b) = best
                    && depth + 1 >= b
                {
                    backward_queue.clear();
                    continue;
                }

                // `edge` leads from `neighbor` to `current` in the direction
                // of the search
                for (neighbor, edge) in self.neighbors(current, reverse_dir) {
                    let new_depth = depth + 1;

                    // Check if forward frontier already visited this node
                    if let Some(&(forward_depth, _)) = forward_visited.get(&neighbor) {
                        let total = forward_depth + new_depth;
                        if best.is_none_or(|b| total < b) {
                            best = Some(total);
                            meeting = Some((neighbor, edge, current));
                        }
                    }

                    if !backward_visited.contains_key(&neighbor) {
                        let step = (new_depth, Some((current, edge)));
                        // Written out: this loop runs per visit, and the helpers out
                        // of line cost about 13 instructions a visit (Callgrind).
                        budget.reserve_map(&mut backward_visited, 1)?;
                        backward_visited.insert(neighbor, step);
                        if best.is_none_or(|b| new_depth < b) {
                            budget.reserve_queue(&mut backward_queue, 1)?;
                            backward_queue.push_back((neighbor, new_depth));
                        }
                    }
                }
            }
        }

        // The forward side back to the source, the meeting edge, then the
        // backward side on to the target
        let (Some((from, edge, to)), Some(best)) = (meeting, best) else {
            return Ok(());
        };
        let hops = usize::try_from(best).unwrap_or(usize::MAX);
        self.keep_shortest(budget, kept, hops, || {
            let mut path = FoundPath {
                nodes: Vec::with_capacity(hops.saturating_add(1)),
                edges: Vec::with_capacity(hops),
            };
            path.nodes.push(from);
            let mut node = from;
            while let Some(&(_, Some((previous, step)))) = forward_visited.get(&node) {
                path.nodes.push(previous);
                path.edges.push(step);
                node = previous;
            }
            path.nodes.reverse();
            path.edges.reverse();
            path.edges.push(edge);
            path.nodes.push(to);
            let mut node = to;
            while let Some(&(_, Some((next, step)))) = backward_visited.get(&node) {
                path.edges.push(step);
                path.nodes.push(next);
                node = next;
            }
            path
        })
    }
}

/// A node or edge id as the `Value::Int64` item of a path's node or edge
/// list, as a variable-length expand writes them.
fn entity_id(id: u64) -> Result<Value, OperatorError> {
    i64::try_from(id)
        .map(Value::Int64)
        .map_err(|_| OperatorError::Internal(format!("id {id} exceeds the i64 range")))
}

/// The paths of `length` edges from the source to `end`, read back from
/// `end` along the steps `steps_into` gives into a node at a depth (from
/// which node, over which edge; none at depth 0, the source): the first
/// `limit` of them, added to `kept`; returns how many. Iterative, so that a
/// long path does not run out of stack. Fails when the search would hold
/// more than its budget: the shortest paths between two nodes can be
/// exponentially many.
fn read_back<'a>(
    budget: &mut PathBudget,
    end: NodeId,
    length: usize,
    limit: usize,
    kept: &mut KeptPaths,
    steps_into: impl Fn(NodeId, usize) -> &'a [(NodeId, EdgeId)],
) -> Result<usize, OperatorError> {
    let mut found = 0;
    // One frame per node on the way back: the node, its depth, and the
    // next of its steps to follow
    let mut frames: Vec<(NodeId, usize, usize)> = Vec::new();
    push(budget, &mut frames, (end, length, 0))?;
    let mut nodes = Vec::new();
    push(budget, &mut nodes, end)?;
    let mut edges: Vec<EdgeId> = Vec::new();
    while let Some(&(node, depth, next)) = frames.last() {
        if found >= limit {
            break;
        }
        if depth == 0 {
            kept.keep(budget, edges.len(), 0, || FoundPath {
                nodes: nodes.iter().rev().copied().collect(),
                edges: edges.iter().rev().copied().collect(),
            })?;
            found += 1;
        } else if let Some(&(previous, edge)) = steps_into(node, depth).get(next) {
            if let Some(frame) = frames.last_mut() {
                frame.2 += 1;
            }
            push(budget, &mut frames, (previous, depth - 1, 0))?;
            push(budget, &mut nodes, previous)?;
            push(budget, &mut edges, edge)?;
            continue;
        }
        frames.pop();
        nodes.pop();
        if !frames.is_empty() {
            edges.pop();
        }
    }
    release_list(budget, frames);
    release_list(budget, nodes);
    release_list(budget, edges);
    Ok(found)
}

/// The steps that reach `node` after `depth` edges in a search for paths of
/// at least `min_hops` edges: from `layers` within the first `min_hops`
/// edges, from `steps` past them. Depth 0 is the source, reached by none.
fn steps_into<'a>(
    layers: &'a [Steps],
    steps: &'a Steps,
    min_hops: usize,
    node: NodeId,
    depth: usize,
) -> &'a [(NodeId, EdgeId)] {
    let layer = if depth > min_hops {
        Some(steps)
    } else {
        depth.checked_sub(1).and_then(|index| layers.get(index))
    };
    layer
        .and_then(|layer| layer.get(&node))
        .map_or(&[], Vec::as_slice)
}

impl ShortestPathOperator {
    /// [`Operator::next`], taking the expansions from `allowance`.
    fn next_within(&mut self, allowance: &mut Allowance) -> OperatorResult {
        if self.exhausted {
            return Ok(None);
        }

        // A pair without a path has no row, so a whole chunk can produce
        // nothing: keep reading until one produces rows or the input ends.
        loop {
            let mut pending = match self.pending.take() {
                Some(pending) => pending,
                None => {
                    let Some(chunk) = self.input.next()? else {
                        self.exhausted = true;
                        return Ok(None);
                    };
                    // The input's own path searches took from the statement's
                    // count meanwhile
                    allowance.refresh(self.expansions.as_deref());
                    let rows = chunk.selected_indices().collect();
                    PendingInput {
                        chunk,
                        rows,
                        next: 0,
                        found: None,
                    }
                }
            };
            let input_chunk = &pending.chunk;

            // Build output: input columns + path length
            let num_input_cols = input_chunk.column_count();
            let mut output_schema: Vec<LogicalType> = (0..num_input_cols)
                .map(|i| {
                    input_chunk
                        .column(i)
                        .map_or(LogicalType::Any, |c| c.data_type().clone())
                })
                .collect();
            // The target the search binds, before the length
            if self.target_column.is_none() {
                output_schema.push(LogicalType::Node);
            }
            output_schema.push(LogicalType::Any); // Path column (stores length as int)
            match self.outputs.edge {
                Some(EdgeOutput::List) => {
                    output_schema.push(LogicalType::List(Box::new(LogicalType::Edge)));
                }
                Some(EdgeOutput::Single) => output_schema.push(LogicalType::Edge),
                None => {}
            }
            if self.outputs.path {
                // Node ids, edge ids, and the `Value::Path`
                output_schema.extend([LogicalType::Any, LogicalType::Any, LogicalType::Any]);
            }

            // A selection of more than one path per pair, or a search to every
            // node, may need more rows than the input
            let initial_capacity =
                if self.selection == PathSelection::Shortest(1) && self.target_column.is_some() {
                    input_chunk.row_count()
                } else {
                    input_chunk.row_count() * 4 // Estimate 4x for multiple paths
                };
            let mut builder = DataChunkBuilder::with_capacity(&output_schema, initial_capacity);

            // The node and edge ids the paths of this output chunk hold: it
            // ends after the path that takes them to a sixty-fourth of the
            // budget, and the next call goes on from the next path, of this
            // row or the next
            let max_ids = (self.budget / 64 / std::mem::size_of::<NodeId>()).max(1);
            let mut ids = 0;
            while ids < max_ids {
                let Some((row, path)) = pending
                    .found
                    .as_mut()
                    .and_then(|(row, paths)| Some((*row, paths.next()?)))
                else {
                    let Some(&row) = pending.rows.get(pending.next) else {
                        break;
                    };
                    pending.next += 1;
                    // Get source and target nodes
                    let source = input_chunk
                        .column(self.source_column)
                        .and_then(|c| c.get_node_id(row));
                    // `Some(None)`: the search binds the target
                    let target = match self.target_column {
                        Some(column) => input_chunk
                            .column(column)
                            .and_then(|c| c.get_node_id(row))
                            .map(Some),
                        None => Some(None),
                    };

                    // A null endpoint (from an earlier OPTIONAL MATCH) has no path
                    let paths = match (source, target) {
                        (Some(s), Some(t)) => {
                            // One budget for every row: each search starts from
                            // nothing and grows within the grant the earlier
                            // ones took
                            let mut budget = self.path_budget.take().unwrap_or_else(|| {
                                PathBudget::new(self.budget, None)
                                    .with_memory(std::mem::take(&mut self.memory))
                                    .with_refusal(SHORTEST_PATH_REFUSAL)
                            });
                            budget.clear();
                            let search = self.search_for_row(input_chunk, row, *allowance);
                            let found = search.paths(&mut budget, s, t);
                            *allowance = search.allowance.into_inner();
                            self.path_budget = Some(budget);
                            found?
                        }
                        _ => Vec::new(),
                    };
                    pending.found = Some((row, paths.into_iter()));
                    continue;
                };

                // One row per path, and the node and edge ids it copies out of
                // the path are expansions
                if !allowance.take(path.nodes.len() + path.edges.len()) {
                    return Err(self.expansions_out());
                }
                ids += path.nodes.len() + path.edges.len();
                // Copy input columns
                for col_idx in 0..num_input_cols {
                    if let Some(in_col) = input_chunk.column(col_idx)
                        && let Some(out_col) = builder.column_mut(col_idx)
                    {
                        if let Some(node_id) = in_col.get_node_id(row) {
                            out_col.push_node_id(node_id);
                        } else if let Some(edge_id) = in_col.get_edge_id(row) {
                            out_col.push_edge_id(edge_id);
                        } else if let Some(value) = in_col.get_value(row) {
                            out_col.push_value(value);
                        } else {
                            out_col.push_value(Value::Null);
                        }
                    }
                }

                let mut column = num_input_cols;
                if self.target_column.is_none() {
                    if let (Some(out_col), Some(&end)) =
                        (builder.column_mut(column), path.nodes.last())
                    {
                        out_col.push_node_id(end);
                    }
                    column += 1;
                }

                // Add path length column
                let length = i64::try_from(path.edges.len()).map_err(|_| {
                    OperatorError::Internal(format!(
                        "a shortest path of {} edges is too long",
                        path.edges.len()
                    ))
                })?;
                if let Some(out_col) = builder.column_mut(column) {
                    out_col.push_value(Value::Int64(length));
                }
                column += 1;

                let edge_ids = || -> Result<Vec<Value>, OperatorError> {
                    path.edges.iter().map(|&edge| entity_id(edge.0)).collect()
                };
                if let Some(output) = self.outputs.edge {
                    if let Some(out_col) = builder.column_mut(column) {
                        match (output, path.edges.as_slice()) {
                            (EdgeOutput::List, _) => {
                                out_col.push_value(Value::List(edge_ids()?.into()));
                            }
                            (EdgeOutput::Single, [edge]) => out_col.push_edge_id(*edge),
                            // An edge pattern without a quantifier is one
                            // hop; a path of another length has no one edge
                            (EdgeOutput::Single, _) => out_col.push_value(Value::Null),
                        }
                    }
                    column += 1;
                }

                if self.outputs.path {
                    let node_ids: Vec<Value> = path
                        .nodes
                        .iter()
                        .map(|&node| entity_id(node.0))
                        .collect::<Result<_, _>>()?;
                    let edge_ids = edge_ids()?;
                    if let Some(out_col) = builder.column_mut(column) {
                        out_col.push_value(Value::List(node_ids.clone().into()));
                    }
                    if let Some(out_col) = builder.column_mut(column + 1) {
                        out_col.push_value(Value::List(edge_ids.clone().into()));
                    }
                    if let Some(out_col) = builder.column_mut(column + 2) {
                        out_col.push_value(Value::Path {
                            nodes: node_ids.into(),
                            edges: edge_ids.into(),
                        });
                    }
                }

                builder.advance_row();
                // Written: the path's lists go, and so does their charge
                if let Some(budget) = &mut self.path_budget {
                    budget.release(path_bytes(path.edges.len()));
                }
            }

            let chunk = builder.finish();
            let paths_left = pending
                .found
                .as_ref()
                .is_some_and(|(_, paths)| paths.len() > 0);
            if paths_left || pending.next < pending.rows.len() {
                self.pending = Some(pending);
            }
            if chunk.row_count() > 0 {
                return Ok(Some(chunk));
            }
        }
    }
}

impl Operator for ShortestPathOperator {
    fn next(&mut self) -> OperatorResult {
        // One read of the statement's count and one write of what the
        // searches took, also when they fail
        let mut allowance = Allowance::of(self.expansions.as_deref());
        let next = self.next_within(&mut allowance);
        allowance.settle(self.expansions.as_deref());
        next
    }

    fn reset(&mut self) {
        self.input.reset();
        self.pending = None;
        self.exhausted = false;
        if let Some(budget) = &mut self.path_budget {
            budget.clear();
        }
    }

    fn name(&self) -> &'static str {
        "ShortestPath"
    }

    fn into_any(self: Box<Self>) -> Box<dyn std::any::Any + Send> {
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::rowgroup::RowGroupStore;

    /// A mock operator that returns source/target node pairs, one chunk per
    /// inner vector (empty ones are skipped).
    struct MockPairOperator {
        chunks: Vec<Vec<(NodeId, NodeId)>>,
        position: usize,
    }

    impl MockPairOperator {
        fn new(pairs: Vec<(NodeId, NodeId)>) -> Self {
            Self::chunks(vec![pairs])
        }

        fn chunks(chunks: Vec<Vec<(NodeId, NodeId)>>) -> Self {
            Self {
                chunks,
                position: 0,
            }
        }
    }

    impl Operator for MockPairOperator {
        fn next(&mut self) -> OperatorResult {
            while let Some(pairs) = self.chunks.get(self.position) {
                self.position += 1;
                if pairs.is_empty() {
                    continue;
                }

                let schema = vec![LogicalType::Node, LogicalType::Node];
                let mut builder = DataChunkBuilder::with_capacity(&schema, pairs.len());
                for (source, target) in pairs {
                    builder.column_mut(0).unwrap().push_node_id(*source);
                    builder.column_mut(1).unwrap().push_node_id(*target);
                    builder.advance_row();
                }
                return Ok(Some(builder.finish()));
            }
            Ok(None)
        }

        fn reset(&mut self) {
            self.position = 0;
        }

        fn name(&self) -> &'static str {
            "MockPair"
        }

        fn into_any(self: Box<Self>) -> Box<dyn std::any::Any + Send> {
            self
        }
    }

    /// Runs the operator to the end: one `(source, target, length)` per row.
    fn rows(op: &mut ShortestPathOperator) -> Vec<(NodeId, NodeId, Value)> {
        let mut rows = Vec::new();
        while let Some(chunk) = op.next().unwrap() {
            for row in 0..chunk.row_count() {
                rows.push((
                    chunk.column(0).unwrap().get_node_id(row).unwrap(),
                    chunk.column(1).unwrap().get_node_id(row).unwrap(),
                    chunk.column(2).unwrap().get_value(row).unwrap(),
                ));
            }
        }
        rows
    }

    /// Searches outgoing paths of any edge type for the pairs in `chunks`.
    fn search(
        store: &Arc<RowGroupStore>,
        chunks: Vec<Vec<(NodeId, NodeId)>>,
        all_paths: bool,
        (min_hops, max_hops): (u32, Option<u32>),
    ) -> Vec<(NodeId, NodeId, Value)> {
        let mut op = ShortestPathOperator::new(
            Arc::clone(store) as Arc<dyn GraphStore>,
            Box::new(MockPairOperator::chunks(chunks)),
            0,
            1,
            vec![],
            Direction::Outgoing,
            ReadView::latest(),
        )
        .with_all_paths(all_paths)
        .with_hop_bounds(min_hops, max_hops);
        rows(&mut op)
    }

    #[test]
    fn test_find_shortest_path_direct() {
        let store = Arc::new(RowGroupStore::new());

        // a -> b (1 hop)
        let a = store.create_node(&["Node"]);
        let b = store.create_node(&["Node"]);
        store.create_edge(a, b, "KNOWS");

        let input = Box::new(MockPairOperator::new(vec![(a, b)]));
        let mut op = ShortestPathOperator::new(
            store.clone() as Arc<dyn GraphStore>,
            input,
            0, // source column
            1, // target column
            vec![],
            Direction::Outgoing,
            ReadView::latest(),
        );

        let chunk = op.next().unwrap().unwrap();
        assert_eq!(chunk.row_count(), 1);

        // Path length should be 1
        let path_col = chunk.column(2).unwrap();
        let path_len = path_col.get_value(0).unwrap();
        assert_eq!(path_len, Value::Int64(1));
    }

    #[test]
    fn test_find_shortest_path_same_node() {
        let store = Arc::new(RowGroupStore::new());
        let a = store.create_node(&["Node"]);

        let input = Box::new(MockPairOperator::new(vec![(a, a)]));
        let mut op = ShortestPathOperator::new(
            store.clone() as Arc<dyn GraphStore>,
            input,
            0,
            1,
            vec![],
            Direction::Outgoing,
            ReadView::latest(),
        );

        let chunk = op.next().unwrap().unwrap();
        assert_eq!(chunk.row_count(), 1);

        // Path length should be 0 (same node)
        let path_col = chunk.column(2).unwrap();
        let path_len = path_col.get_value(0).unwrap();
        assert_eq!(path_len, Value::Int64(0));
    }

    #[test]
    fn test_find_shortest_path_two_hops() {
        let store = Arc::new(RowGroupStore::new());

        // a -> b -> c (2 hops from a to c)
        let a = store.create_node(&["Node"]);
        let b = store.create_node(&["Node"]);
        let c = store.create_node(&["Node"]);
        store.create_edge(a, b, "KNOWS");
        store.create_edge(b, c, "KNOWS");

        let input = Box::new(MockPairOperator::new(vec![(a, c)]));
        let mut op = ShortestPathOperator::new(
            store.clone() as Arc<dyn GraphStore>,
            input,
            0,
            1,
            vec![],
            Direction::Outgoing,
            ReadView::latest(),
        );

        let chunk = op.next().unwrap().unwrap();
        assert_eq!(chunk.row_count(), 1);

        let path_col = chunk.column(2).unwrap();
        let path_len = path_col.get_value(0).unwrap();
        assert_eq!(path_len, Value::Int64(2));
    }

    #[test]
    fn test_find_shortest_path_no_path() {
        let store = Arc::new(RowGroupStore::new());

        // a and b are disconnected
        let a = store.create_node(&["Node"]);
        let b = store.create_node(&["Node"]);

        let input = Box::new(MockPairOperator::new(vec![(a, b)]));
        let mut op = ShortestPathOperator::new(
            store.clone() as Arc<dyn GraphStore>,
            input,
            0,
            1,
            vec![],
            Direction::Outgoing,
            ReadView::latest(),
        );

        // No path, no row: MATCH drops the pair, OPTIONAL MATCH adds the nulls
        assert!(op.next().unwrap().is_none());
    }

    #[test]
    fn test_find_shortest_path_prefers_shorter() {
        let store = Arc::new(RowGroupStore::new());

        // Create two paths: a -> d (1 hop) and a -> b -> c -> d (3 hops)
        let a = store.create_node(&["Node"]);
        let b = store.create_node(&["Node"]);
        let c = store.create_node(&["Node"]);
        let d = store.create_node(&["Node"]);

        // Long path
        store.create_edge(a, b, "KNOWS");
        store.create_edge(b, c, "KNOWS");
        store.create_edge(c, d, "KNOWS");

        // Short path (direct)
        store.create_edge(a, d, "KNOWS");

        let input = Box::new(MockPairOperator::new(vec![(a, d)]));
        let mut op = ShortestPathOperator::new(
            store.clone() as Arc<dyn GraphStore>,
            input,
            0,
            1,
            vec![],
            Direction::Outgoing,
            ReadView::latest(),
        );

        let chunk = op.next().unwrap().unwrap();
        let path_col = chunk.column(2).unwrap();
        let path_len = path_col.get_value(0).unwrap();
        assert_eq!(path_len, Value::Int64(1)); // Should find direct path
    }

    #[test]
    fn test_find_shortest_path_with_edge_type_filter() {
        let store = Arc::new(RowGroupStore::new());

        // a -KNOWS-> b -LIKES-> c
        let a = store.create_node(&["Node"]);
        let b = store.create_node(&["Node"]);
        let c = store.create_node(&["Node"]);
        store.create_edge(a, b, "KNOWS");
        store.create_edge(b, c, "LIKES");

        // Path with KNOWS filter should only reach b, not c
        let input = Box::new(MockPairOperator::new(vec![(a, c)]));
        let mut op = ShortestPathOperator::new(
            store.clone() as Arc<dyn GraphStore>,
            input,
            0,
            1,
            vec!["KNOWS".to_string()],
            Direction::Outgoing,
            ReadView::latest(),
        );

        assert!(op.next().unwrap().is_none()); // Can't reach c via KNOWS only
    }

    #[test]
    fn test_all_shortest_paths_single_path() {
        let store = Arc::new(RowGroupStore::new());

        // a -> b (single path)
        let a = store.create_node(&["Node"]);
        let b = store.create_node(&["Node"]);
        store.create_edge(a, b, "KNOWS");

        let input = Box::new(MockPairOperator::new(vec![(a, b)]));
        let mut op = ShortestPathOperator::new(
            store.clone() as Arc<dyn GraphStore>,
            input,
            0,
            1,
            vec![],
            Direction::Outgoing,
            ReadView::latest(),
        )
        .with_all_paths(true);

        let chunk = op.next().unwrap().unwrap();
        assert_eq!(chunk.row_count(), 1); // Only one path exists
    }

    #[test]
    fn test_all_shortest_paths_multiple_paths() {
        let store = Arc::new(RowGroupStore::new());

        // Create diamond: a -> b -> d and a -> c -> d (two paths of length 2)
        let a = store.create_node(&["Node"]);
        let b = store.create_node(&["Node"]);
        let c = store.create_node(&["Node"]);
        let d = store.create_node(&["Node"]);

        store.create_edge(a, b, "KNOWS");
        store.create_edge(a, c, "KNOWS");
        store.create_edge(b, d, "KNOWS");
        store.create_edge(c, d, "KNOWS");

        let input = Box::new(MockPairOperator::new(vec![(a, d)]));
        let mut op = ShortestPathOperator::new(
            store.clone() as Arc<dyn GraphStore>,
            input,
            0,
            1,
            vec![],
            Direction::Outgoing,
            ReadView::latest(),
        )
        .with_all_paths(true);

        let chunk = op.next().unwrap().unwrap();
        // Should return 2 rows (two paths of length 2)
        assert_eq!(chunk.row_count(), 2);

        // Both should have length 2
        let path_col = chunk.column(2).unwrap();
        assert_eq!(path_col.get_value(0).unwrap(), Value::Int64(2));
        assert_eq!(path_col.get_value(1).unwrap(), Value::Int64(2));
    }

    #[test]
    fn test_multiple_pairs_in_chunk() {
        let store = Arc::new(RowGroupStore::new());

        // Create: a -> b, c -> d
        let a = store.create_node(&["Node"]);
        let b = store.create_node(&["Node"]);
        let c = store.create_node(&["Node"]);
        let d = store.create_node(&["Node"]);

        store.create_edge(a, b, "KNOWS");
        store.create_edge(c, d, "KNOWS");

        // Test multiple pairs at once
        let input = Box::new(MockPairOperator::new(vec![(a, b), (c, d), (a, d)]));
        let mut op = ShortestPathOperator::new(
            store.clone() as Arc<dyn GraphStore>,
            input,
            0,
            1,
            vec![],
            Direction::Outgoing,
            ReadView::latest(),
        );

        // a->d has no path, so it has no row
        assert_eq!(
            rows(&mut op),
            vec![(a, b, Value::Int64(1)), (c, d, Value::Int64(1))]
        );
    }

    #[test]
    fn test_operator_reset() {
        let store = Arc::new(RowGroupStore::new());
        let a = store.create_node(&["Node"]);
        let b = store.create_node(&["Node"]);
        store.create_edge(a, b, "KNOWS");

        let input = Box::new(MockPairOperator::new(vec![(a, b)]));
        let mut op = ShortestPathOperator::new(
            store.clone() as Arc<dyn GraphStore>,
            input,
            0,
            1,
            vec![],
            Direction::Outgoing,
            ReadView::latest(),
        );

        // First iteration
        let chunk = op.next().unwrap();
        assert!(chunk.is_some());
        let chunk = op.next().unwrap();
        assert!(chunk.is_none());

        // After reset
        op.reset();
        let chunk = op.next().unwrap();
        assert!(chunk.is_some());
    }

    #[test]
    fn test_operator_name() {
        let store = Arc::new(RowGroupStore::new());
        let input = Box::new(MockPairOperator::new(vec![]));
        let op = ShortestPathOperator::new(
            store.clone() as Arc<dyn GraphStore>,
            input,
            0,
            1,
            vec![],
            Direction::Outgoing,
            ReadView::latest(),
        );

        assert_eq!(op.name(), "ShortestPath");
    }

    #[test]
    fn test_empty_input() {
        let store = Arc::new(RowGroupStore::new());
        let input = Box::new(MockPairOperator::new(vec![]));
        let mut op = ShortestPathOperator::new(
            store.clone() as Arc<dyn GraphStore>,
            input,
            0,
            1,
            vec![],
            Direction::Outgoing,
            ReadView::latest(),
        );

        // Empty input should return None
        let chunk = op.next().unwrap();
        assert!(chunk.is_none());
    }

    #[test]
    fn test_all_shortest_paths_no_path() {
        let store = Arc::new(RowGroupStore::new());

        // Disconnected nodes
        let a = store.create_node(&["Node"]);
        let b = store.create_node(&["Node"]);

        let input = Box::new(MockPairOperator::new(vec![(a, b)]));
        let mut op = ShortestPathOperator::new(
            store.clone() as Arc<dyn GraphStore>,
            input,
            0,
            1,
            vec![],
            Direction::Outgoing,
            ReadView::latest(),
        )
        .with_all_paths(true);

        assert!(op.next().unwrap().is_none());
    }

    #[test]
    fn test_all_shortest_paths_same_node() {
        let store = Arc::new(RowGroupStore::new());
        let a = store.create_node(&["Node"]);

        let input = Box::new(MockPairOperator::new(vec![(a, a)]));
        let mut op = ShortestPathOperator::new(
            store.clone() as Arc<dyn GraphStore>,
            input,
            0,
            1,
            vec![],
            Direction::Outgoing,
            ReadView::latest(),
        )
        .with_all_paths(true);

        let chunk = op.next().unwrap().unwrap();
        assert_eq!(chunk.row_count(), 1);

        let path_col = chunk.column(2).unwrap();
        assert_eq!(path_col.get_value(0).unwrap(), Value::Int64(0));
    }

    // === Bidirectional BFS Tests ===

    #[test]
    fn test_bidirectional_bfs_long_chain() {
        let store = Arc::new(RowGroupStore::new());

        // Chain: n0 -> n1 -> n2 -> ... -> n9 (9 hops)
        let nodes: Vec<NodeId> = (0..10).map(|_| store.create_node(&["Node"])).collect();
        for i in 0..9 {
            store.create_edge(nodes[i], nodes[i + 1], "NEXT");
        }

        let input = Box::new(MockPairOperator::new(vec![(nodes[0], nodes[9])]));
        let mut op = ShortestPathOperator::new(
            store.clone() as Arc<dyn GraphStore>,
            input,
            0,
            1,
            vec![],
            Direction::Outgoing,
            ReadView::latest(),
        );

        let chunk = op.next().unwrap().unwrap();
        let path_col = chunk.column(2).unwrap();
        assert_eq!(path_col.get_value(0).unwrap(), Value::Int64(9));
    }

    #[test]
    fn test_bidirectional_bfs_diamond() {
        let store = Arc::new(RowGroupStore::new());

        // Diamond: a -> b -> d, a -> c -> d (two paths of length 2)
        let a = store.create_node(&["Node"]);
        let b = store.create_node(&["Node"]);
        let c = store.create_node(&["Node"]);
        let d = store.create_node(&["Node"]);

        store.create_edge(a, b, "KNOWS");
        store.create_edge(a, c, "KNOWS");
        store.create_edge(b, d, "KNOWS");
        store.create_edge(c, d, "KNOWS");

        let input = Box::new(MockPairOperator::new(vec![(a, d)]));
        let mut op = ShortestPathOperator::new(
            store.clone() as Arc<dyn GraphStore>,
            input,
            0,
            1,
            vec![],
            Direction::Outgoing,
            ReadView::latest(),
        );

        let chunk = op.next().unwrap().unwrap();
        let path_col = chunk.column(2).unwrap();
        assert_eq!(path_col.get_value(0).unwrap(), Value::Int64(2));
    }

    #[test]
    fn test_bidirectional_bfs_no_path() {
        let store = Arc::new(RowGroupStore::new());

        let a = store.create_node(&["Node"]);
        let b = store.create_node(&["Node"]);
        // No edges between a and b

        let input = Box::new(MockPairOperator::new(vec![(a, b)]));
        let mut op = ShortestPathOperator::new(
            store.clone() as Arc<dyn GraphStore>,
            input,
            0,
            1,
            vec![],
            Direction::Outgoing,
            ReadView::latest(),
        );

        assert!(op.next().unwrap().is_none());
    }

    #[test]
    fn test_bidirectional_bfs_same_node() {
        let store = Arc::new(RowGroupStore::new());
        let a = store.create_node(&["Node"]);

        let input = Box::new(MockPairOperator::new(vec![(a, a)]));
        let mut op = ShortestPathOperator::new(
            store.clone() as Arc<dyn GraphStore>,
            input,
            0,
            1,
            vec![],
            Direction::Outgoing,
            ReadView::latest(),
        );

        let chunk = op.next().unwrap().unwrap();
        let path_col = chunk.column(2).unwrap();
        assert_eq!(path_col.get_value(0).unwrap(), Value::Int64(0));
    }

    #[test]
    fn test_bidirectional_bfs_prefers_shorter() {
        let store = Arc::new(RowGroupStore::new());

        let a = store.create_node(&["Node"]);
        let b = store.create_node(&["Node"]);
        let c = store.create_node(&["Node"]);
        let d = store.create_node(&["Node"]);

        // Long path: a -> b -> c -> d (3 hops)
        store.create_edge(a, b, "KNOWS");
        store.create_edge(b, c, "KNOWS");
        store.create_edge(c, d, "KNOWS");
        // Short path: a -> d (1 hop)
        store.create_edge(a, d, "KNOWS");

        let input = Box::new(MockPairOperator::new(vec![(a, d)]));
        let mut op = ShortestPathOperator::new(
            store.clone() as Arc<dyn GraphStore>,
            input,
            0,
            1,
            vec![],
            Direction::Outgoing,
            ReadView::latest(),
        );

        let chunk = op.next().unwrap().unwrap();
        let path_col = chunk.column(2).unwrap();
        assert_eq!(path_col.get_value(0).unwrap(), Value::Int64(1));
    }

    #[test]
    fn test_bidirectional_bfs_with_edge_type_filter() {
        let store = Arc::new(RowGroupStore::new());

        // a -KNOWS-> b -LIKES-> c
        let a = store.create_node(&["Node"]);
        let b = store.create_node(&["Node"]);
        let c = store.create_node(&["Node"]);
        store.create_edge(a, b, "KNOWS");
        store.create_edge(b, c, "LIKES");

        // Only KNOWS edges: a can reach b but not c
        let input = Box::new(MockPairOperator::new(vec![(a, c)]));
        let mut op = ShortestPathOperator::new(
            store.clone() as Arc<dyn GraphStore>,
            input,
            0,
            1,
            vec!["KNOWS".to_string()],
            Direction::Outgoing,
            ReadView::latest(),
        );

        assert!(op.next().unwrap().is_none());
    }

    #[test]
    fn test_bidirectional_bfs_has_backward_adjacency() {
        // Default store has backward adjacency enabled
        let store = Arc::new(RowGroupStore::new());
        assert!(store.has_backward_adjacency());

        let a = store.create_node(&["Node"]);
        let b = store.create_node(&["Node"]);
        let c = store.create_node(&["Node"]);
        store.create_edge(a, b, "KNOWS");
        store.create_edge(b, c, "KNOWS");

        // Bidirectional BFS should work and find shortest path
        let input = Box::new(MockPairOperator::new(vec![(a, c)]));
        let mut op = ShortestPathOperator::new(
            store.clone() as Arc<dyn GraphStore>,
            input,
            0,
            1,
            vec![],
            Direction::Outgoing,
            ReadView::latest(),
        );

        let chunk = op.next().unwrap().unwrap();
        let path_col = chunk.column(2).unwrap();
        assert_eq!(path_col.get_value(0).unwrap(), Value::Int64(2));
    }

    #[test]
    fn test_shortest_path_into_any() {
        let store = Arc::new(RowGroupStore::new());
        let input = Box::new(MockPairOperator::new(vec![]));
        let op = ShortestPathOperator::new(
            store.clone() as Arc<dyn GraphStore>,
            input,
            0,
            1,
            vec![],
            Direction::Outgoing,
            ReadView::latest(),
        );
        let any = Box::new(op).into_any();
        assert!(any.downcast::<ShortestPathOperator>().is_ok());
    }

    // === Pairs without a path and hop bounds (#514) ===

    #[test]
    fn a_chunk_without_paths_does_not_end_the_output() {
        // a -> b; c is isolated. No pair in the first input chunk has a path,
        // the pair in the second one does.
        let store = Arc::new(RowGroupStore::new());
        let a = store.create_node(&["Node"]);
        let b = store.create_node(&["Node"]);
        let c = store.create_node(&["Node"]);
        store.create_edge(a, b, "KNOWS");

        for all_paths in [false, true] {
            let chunks = vec![vec![(a, c), (c, b)], vec![(a, b)]];
            assert_eq!(
                search(&store, chunks, all_paths, (0, None)),
                vec![(a, b, Value::Int64(1))],
                "all_paths: {all_paths}"
            );
        }
    }

    #[test]
    fn zero_hop_minimum_keeps_the_zero_length_self_path() {
        let store = Arc::new(RowGroupStore::new());
        let a = store.create_node(&["Node"]);

        for all_paths in [false, true] {
            assert_eq!(
                search(&store, vec![vec![(a, a)]], all_paths, (0, None)),
                vec![(a, a, Value::Int64(0))],
                "all_paths: {all_paths}"
            );
        }
    }

    #[test]
    fn one_hop_minimum_excludes_the_zero_length_self_path() {
        // a has no edges, so no path of one or more hops leads back to it
        let store = Arc::new(RowGroupStore::new());
        let a = store.create_node(&["Node"]);

        for all_paths in [false, true] {
            assert_eq!(
                search(&store, vec![vec![(a, a)]], all_paths, (1, None)),
                vec![],
                "all_paths: {all_paths}"
            );
        }
    }

    #[test]
    fn one_hop_minimum_finds_the_shortest_cycle() {
        // a -> b -> a (2 hops) and a -> c -> d -> a (3 hops)
        let store = Arc::new(RowGroupStore::new());
        let a = store.create_node(&["Node"]);
        let b = store.create_node(&["Node"]);
        let c = store.create_node(&["Node"]);
        let d = store.create_node(&["Node"]);
        store.create_edge(a, b, "KNOWS");
        store.create_edge(b, a, "KNOWS");
        store.create_edge(a, c, "KNOWS");
        store.create_edge(c, d, "KNOWS");
        store.create_edge(d, a, "KNOWS");

        for all_paths in [false, true] {
            assert_eq!(
                search(&store, vec![vec![(a, a)]], all_paths, (1, None)),
                vec![(a, a, Value::Int64(2))],
                "all_paths: {all_paths}"
            );
        }
    }

    #[test]
    fn a_self_loop_is_a_one_hop_cycle() {
        let store = Arc::new(RowGroupStore::new());
        let a = store.create_node(&["Node"]);
        store.create_edge(a, a, "KNOWS");

        for all_paths in [false, true] {
            assert_eq!(
                search(&store, vec![vec![(a, a)]], all_paths, (1, None)),
                vec![(a, a, Value::Int64(1))],
                "all_paths: {all_paths}"
            );
        }
    }

    #[test]
    fn all_shortest_cycles_are_returned() {
        // a -> b -> d -> a and a -> c -> d -> a: two shortest cycles of 3 hops
        let store = Arc::new(RowGroupStore::new());
        let a = store.create_node(&["Node"]);
        let b = store.create_node(&["Node"]);
        let c = store.create_node(&["Node"]);
        let d = store.create_node(&["Node"]);
        store.create_edge(a, b, "KNOWS");
        store.create_edge(a, c, "KNOWS");
        store.create_edge(b, d, "KNOWS");
        store.create_edge(c, d, "KNOWS");
        store.create_edge(d, a, "KNOWS");

        assert_eq!(
            search(&store, vec![vec![(a, a)]], true, (1, None)),
            vec![(a, a, Value::Int64(3)); 2]
        );
        assert_eq!(
            search(&store, vec![vec![(a, a)]], false, (1, None)),
            vec![(a, a, Value::Int64(3))]
        );
    }

    #[test]
    fn minimum_above_the_shortest_distance() {
        // Triangle a -> b -> c -> a: b is 1 hop from a, and 4 hops when the
        // path has to go around the triangle once more.
        let store = Arc::new(RowGroupStore::new());
        let a = store.create_node(&["Node"]);
        let b = store.create_node(&["Node"]);
        let c = store.create_node(&["Node"]);
        store.create_edge(a, b, "KNOWS");
        store.create_edge(b, c, "KNOWS");
        store.create_edge(c, a, "KNOWS");

        for all_paths in [false, true] {
            let pairs = || vec![vec![(a, b)]];
            assert_eq!(
                search(&store, pairs(), all_paths, (1, None)),
                vec![(a, b, Value::Int64(1))],
                "all_paths: {all_paths}"
            );
            assert_eq!(
                search(&store, pairs(), all_paths, (2, None)),
                vec![(a, b, Value::Int64(4))],
                "all_paths: {all_paths}"
            );
            assert_eq!(
                search(&store, pairs(), all_paths, (2, Some(3))),
                vec![],
                "all_paths: {all_paths}"
            );
        }
    }

    #[test]
    fn maximum_drops_longer_paths() {
        // a -> b -> c
        let store = Arc::new(RowGroupStore::new());
        let a = store.create_node(&["Node"]);
        let b = store.create_node(&["Node"]);
        let c = store.create_node(&["Node"]);
        store.create_edge(a, b, "KNOWS");
        store.create_edge(b, c, "KNOWS");

        for all_paths in [false, true] {
            let pairs = || vec![vec![(a, b), (a, c)]];
            assert_eq!(
                search(&store, pairs(), all_paths, (1, Some(1))),
                vec![(a, b, Value::Int64(1))],
                "all_paths: {all_paths}"
            );
            assert_eq!(
                search(&store, pairs(), all_paths, (1, Some(2))),
                vec![(a, b, Value::Int64(1)), (a, c, Value::Int64(2))],
                "all_paths: {all_paths}"
            );
        }
    }

    /// A path as the operator writes it: node ids and edge ids.
    fn ids(nodes: &[NodeId], edges: &[EdgeId]) -> (Vec<Value>, Vec<Value>) {
        let id = |raw: u64| Value::Int64(i64::try_from(raw).unwrap());
        (
            nodes.iter().map(|node| id(node.0)).collect(),
            edges.iter().map(|edge| id(edge.0)).collect(),
        )
    }

    /// The paths an operator with path output writes, one per row: the node
    /// ids and the edge ids of its path value, checked against its node and
    /// edge columns and its length, after `extra` columns (the edge column).
    fn paths_of(op: &mut ShortestPathOperator, extra: usize) -> Vec<(Vec<Value>, Vec<Value>)> {
        let mut paths = Vec::new();
        while let Some(chunk) = op.next().unwrap() {
            for row in 0..chunk.row_count() {
                let value = |column: usize| chunk.column(column).unwrap().get_value(row).unwrap();
                let Value::Path { nodes, edges } = value(5 + extra) else {
                    panic!("expected a path value, got {:?}", value(5 + extra));
                };
                assert_eq!(value(3 + extra), Value::List(nodes.clone()), "node column");
                assert_eq!(value(4 + extra), Value::List(edges.clone()), "edge column");
                assert_eq!(
                    value(2),
                    Value::Int64(i64::try_from(edges.len()).unwrap()),
                    "length column"
                );
                assert_eq!(nodes.len(), edges.len() + 1, "a node more than edges");
                paths.push((nodes.to_vec(), edges.to_vec()));
            }
        }
        paths.sort_by_key(|(_, edges)| format!("{edges:?}"));
        paths
    }

    /// An operator over `pairs` that outputs paths.
    fn path_search(
        store: &Arc<RowGroupStore>,
        pairs: Vec<(NodeId, NodeId)>,
        all_paths: bool,
        (min_hops, max_hops): (u32, Option<u32>),
    ) -> ShortestPathOperator {
        ShortestPathOperator::new(
            Arc::clone(store) as Arc<dyn GraphStore>,
            Box::new(MockPairOperator::new(pairs)),
            0,
            1,
            vec![],
            Direction::Outgoing,
            ReadView::latest(),
        )
        .with_all_paths(all_paths)
        .with_hop_bounds(min_hops, max_hops)
        .with_path_output()
    }

    #[test]
    fn the_shortest_path_is_written_node_by_node_and_edge_by_edge() {
        // a -> b -> c -> d and the shortcut a -> c
        let store = Arc::new(RowGroupStore::new());
        let a = store.create_node(&["Node"]);
        let b = store.create_node(&["Node"]);
        let c = store.create_node(&["Node"]);
        let d = store.create_node(&["Node"]);
        let _ab = store.create_edge(a, b, "KNOWS");
        let _bc = store.create_edge(b, c, "KNOWS");
        let cd = store.create_edge(c, d, "KNOWS");
        let ac = store.create_edge(a, c, "KNOWS");

        for (all_paths, bounds) in [(false, (1, None)), (true, (1, None)), (false, (2, None))] {
            assert_eq!(
                paths_of(&mut path_search(&store, vec![(a, d)], all_paths, bounds), 0),
                vec![ids(&[a, c, d], &[ac, cd])],
                "all_paths: {all_paths}, bounds: {bounds:?}"
            );
        }
        assert_eq!(
            paths_of(&mut path_search(&store, vec![(a, a)], false, (0, None)), 0),
            vec![ids(&[a], &[])],
            "the zero-length path"
        );
    }

    #[test]
    fn the_one_way_search_finds_the_same_path() {
        // a -> b -> c -> d and the shortcut a -> c. A store without backward
        // adjacency searches one way only: the same path, edge by edge.
        let store = Arc::new(RowGroupStore::new());
        let a = store.create_node(&["Node"]);
        let b = store.create_node(&["Node"]);
        let c = store.create_node(&["Node"]);
        let d = store.create_node(&["Node"]);
        let ab = store.create_edge(a, b, "KNOWS");
        let bc = store.create_edge(b, c, "KNOWS");
        let cd = store.create_edge(c, d, "KNOWS");
        let ac = store.create_edge(a, c, "KNOWS");
        // The one path the one-way search keeps, under an unbounded budget
        let found = |search: &PathSearch<'_>, source, target| {
            let mut kept = KeptPaths::default();
            let mut budget = PathBudget::new(usize::MAX, None);
            search
                .shortest_path(&mut budget, &mut kept, source, target)
                .unwrap();
            assert!(kept.paths.len() <= 1, "one path at most");
            kept.paths.pop().map(|path| ids(&path.nodes, &path.edges))
        };

        let op = path_search(&store, vec![], false, (1, None));
        let search = op.search_for_row(
            &DataChunk::empty(),
            0,
            Allowance::of(op.expansions.as_deref()),
        );
        assert_eq!(found(&search, a, d), Some(ids(&[a, c, d], &[ac, cd])));
        assert_eq!(found(&search, a, a), Some(ids(&[a], &[])));
        assert_eq!(found(&search, d, a), None, "no way back");

        // The edge condition reads a row of the input
        let mut pairs = MockPairOperator::new(vec![(a, d)]);
        let row = pairs.next().unwrap().unwrap();
        let op = path_search(&store, vec![], false, (1, None))
            .with_edge_condition(Box::new(AvoidEdges(vec![ac])));
        assert_eq!(
            found(
                &op.search_for_row(&row, 0, Allowance::of(op.expansions.as_deref())),
                a,
                d
            ),
            Some(ids(&[a, b, c, d], &[ab, bc, cd]))
        );
    }

    #[test]
    fn all_shortest_paths_are_each_written_with_their_own_edges() {
        // Diamond a -> b -> d, a -> c -> d, and a second edge c -> d
        let store = Arc::new(RowGroupStore::new());
        let a = store.create_node(&["Node"]);
        let b = store.create_node(&["Node"]);
        let c = store.create_node(&["Node"]);
        let d = store.create_node(&["Node"]);
        let ab = store.create_edge(a, b, "KNOWS");
        let ac = store.create_edge(a, c, "KNOWS");
        let bd = store.create_edge(b, d, "KNOWS");
        let cd = store.create_edge(c, d, "KNOWS");
        let cd_again = store.create_edge(c, d, "KNOWS");

        let mut want = vec![
            ids(&[a, b, d], &[ab, bd]),
            ids(&[a, c, d], &[ac, cd]),
            ids(&[a, c, d], &[ac, cd_again]),
        ];
        want.sort_by_key(|(_, edges)| format!("{edges:?}"));
        assert_eq!(
            paths_of(&mut path_search(&store, vec![(a, d)], true, (1, None)), 0),
            want
        );
        let any = paths_of(&mut path_search(&store, vec![(a, d)], false, (1, None)), 0);
        assert_eq!(any.len(), 1, "ANY SHORTEST writes one path");
        assert!(want.contains(&any[0]), "{any:?} is one of {want:?}");
    }

    #[test]
    fn a_path_past_a_minimum_is_written_from_its_first_steps() {
        // Triangle a -> b -> c -> a: from a to b in at least 2 hops goes round
        // once more, a -> b -> c -> a -> b
        let store = Arc::new(RowGroupStore::new());
        let a = store.create_node(&["Node"]);
        let b = store.create_node(&["Node"]);
        let c = store.create_node(&["Node"]);
        let ab = store.create_edge(a, b, "KNOWS");
        let bc = store.create_edge(b, c, "KNOWS");
        let ca = store.create_edge(c, a, "KNOWS");

        for all_paths in [false, true] {
            assert_eq!(
                paths_of(
                    &mut path_search(&store, vec![(a, b)], all_paths, (2, None)),
                    0
                ),
                vec![ids(&[a, b, c, a, b], &[ab, bc, ca, ab])],
                "all_paths: {all_paths}"
            );
            assert_eq!(
                paths_of(
                    &mut path_search(&store, vec![(a, a)], all_paths, (1, None)),
                    0
                ),
                vec![ids(&[a, b, c, a], &[ab, bc, ca])],
                "the shortest cycle, all_paths: {all_paths}"
            );
        }
    }

    /// A condition that rejects the edges in its list; it reads the candidate
    /// edge from the column after the input row.
    struct AvoidEdges(Vec<EdgeId>);

    impl Predicate for AvoidEdges {
        fn evaluate(&self, chunk: &DataChunk, row: usize) -> Result<bool, OperatorError> {
            let edge = chunk
                .column(chunk.column_count() - 1)
                .and_then(|column| column.get_edge_id(row))
                .expect("the candidate edge is in the last column");
            Ok(!self.0.contains(&edge))
        }
    }

    #[test]
    fn the_edge_condition_holds_for_every_edge_of_the_path() {
        // a -> b -> c -> d and the shortcut a -> c
        let store = Arc::new(RowGroupStore::new());
        let a = store.create_node(&["Node"]);
        let b = store.create_node(&["Node"]);
        let c = store.create_node(&["Node"]);
        let d = store.create_node(&["Node"]);
        let ab = store.create_edge(a, b, "KNOWS");
        let bc = store.create_edge(b, c, "KNOWS");
        let cd = store.create_edge(c, d, "KNOWS");
        let ac = store.create_edge(a, c, "KNOWS");

        for (all_paths, bounds) in [(false, (1, None)), (true, (1, None)), (false, (2, None))] {
            let mut op = path_search(&store, vec![(a, d)], all_paths, bounds)
                .with_edge_condition(Box::new(AvoidEdges(vec![ac])));
            assert_eq!(
                paths_of(&mut op, 0),
                vec![ids(&[a, b, c, d], &[ab, bc, cd])],
                "all_paths: {all_paths}, bounds: {bounds:?}"
            );
        }
        let mut op = path_search(&store, vec![(a, d)], false, (1, None))
            .with_edge_condition(Box::new(AvoidEdges(vec![ac, bc])));
        assert!(paths_of(&mut op, 0).is_empty(), "no path avoids both");
    }

    /// A condition that only takes the edge in the input row's third column.
    struct RowEdge;

    impl Predicate for RowEdge {
        fn evaluate(&self, chunk: &DataChunk, row: usize) -> Result<bool, OperatorError> {
            let edge = |column: usize| chunk.column(column).and_then(|c| c.get_edge_id(row));
            Ok(edge(3).is_some() && edge(3) == edge(2))
        }
    }

    #[test]
    fn the_edge_condition_reads_the_input_row() {
        // Two parallel edges a -> b; each input row names one of them
        let store = Arc::new(RowGroupStore::new());
        let a = store.create_node(&["Node"]);
        let b = store.create_node(&["Node"]);
        let first = store.create_edge(a, b, "KNOWS");
        let second = store.create_edge(a, b, "KNOWS");

        struct RowsWithEdge(Vec<(NodeId, NodeId, EdgeId)>, bool);
        impl Operator for RowsWithEdge {
            fn next(&mut self) -> OperatorResult {
                if self.1 {
                    return Ok(None);
                }
                self.1 = true;
                let schema = [LogicalType::Node, LogicalType::Node, LogicalType::Edge];
                let mut builder = DataChunkBuilder::with_capacity(&schema, self.0.len());
                for &(source, target, edge) in &self.0 {
                    builder.column_mut(0).unwrap().push_node_id(source);
                    builder.column_mut(1).unwrap().push_node_id(target);
                    builder.column_mut(2).unwrap().push_edge_id(edge);
                    builder.advance_row();
                }
                Ok(Some(builder.finish()))
            }
            fn reset(&mut self) {
                self.1 = false;
            }
            fn name(&self) -> &'static str {
                "RowsWithEdge"
            }
            fn into_any(self: Box<Self>) -> Box<dyn std::any::Any + Send> {
                self
            }
        }

        let mut op = ShortestPathOperator::new(
            Arc::clone(&store) as Arc<dyn GraphStore>,
            Box::new(RowsWithEdge(vec![(a, b, second), (a, b, first)], false)),
            0,
            1,
            vec![],
            Direction::Outgoing,
            ReadView::latest(),
        )
        .with_hop_bounds(1, Some(1))
        .with_edge_output(false)
        .with_edge_condition(Box::new(RowEdge));
        let mut found = Vec::new();
        while let Some(chunk) = op.next().unwrap() {
            for row in 0..chunk.row_count() {
                let edge = |column: usize| chunk.column(column).unwrap().get_edge_id(row);
                found.push((edge(2), edge(4)));
            }
        }
        assert_eq!(
            found,
            vec![(Some(second), Some(second)), (Some(first), Some(first))],
            "each row's path takes the row's edge"
        );
    }

    #[test]
    fn the_edge_column_holds_the_list_or_the_one_edge() {
        // a -> b -> c
        let store = Arc::new(RowGroupStore::new());
        let a = store.create_node(&["Node"]);
        let b = store.create_node(&["Node"]);
        let c = store.create_node(&["Node"]);
        let ab = store.create_edge(a, b, "KNOWS");
        let bc = store.create_edge(b, c, "KNOWS");

        let mut op = path_search(&store, vec![(a, c)], false, (1, None)).with_edge_output(true);
        let chunk = op.next().unwrap().unwrap();
        assert_eq!(
            chunk.column(3).unwrap().get_value(0),
            Some(Value::List(ids(&[], &[ab, bc]).1.into())),
            "a quantified edge pattern binds the list of the path's edges"
        );

        let mut op = path_search(&store, vec![(a, b)], false, (1, Some(1))).with_edge_output(false);
        let chunk = op.next().unwrap().unwrap();
        assert_eq!(
            chunk.column(3).unwrap().get_edge_id(0),
            Some(ab),
            "an edge pattern without a quantifier binds the one edge"
        );
        assert_eq!(
            paths_of(
                &mut path_search(&store, vec![(a, b)], false, (1, Some(1))),
                0
            )
            .len(),
            1
        );
    }

    // -----------------------------------------------------------------------
    // Selections, path modes and the search to every node (ISO/IEC
    // 39075:2024 16.6 `<path search prefix>`)
    // -----------------------------------------------------------------------

    /// The paths an operator with path output writes, in its order: the
    /// target, the node ids and the edge ids. With `binds_target` the target
    /// column follows the two input columns, otherwise it is the second.
    fn found(op: &mut ShortestPathOperator, binds_target: bool) -> Vec<(NodeId, Vec<Value>)> {
        let base = 2 + usize::from(binds_target);
        let mut found = Vec::new();
        while let Some(chunk) = op.next().unwrap() {
            for row in 0..chunk.row_count() {
                let value = |column: usize| chunk.column(column).unwrap().get_value(row).unwrap();
                let Value::Path { nodes, edges } = value(base + 3) else {
                    panic!("expected a path value, got {:?}", value(base + 3));
                };
                let target = chunk
                    .column(if binds_target { 2 } else { 1 })
                    .unwrap()
                    .get_node_id(row)
                    .unwrap();
                assert_eq!(
                    nodes.last(),
                    Some(&Value::Int64(i64::try_from(target.0).unwrap())),
                    "the path ends at the target"
                );
                assert_eq!(
                    value(base),
                    Value::Int64(i64::try_from(edges.len()).unwrap()),
                    "length column"
                );
                found.push((target, edges.to_vec()));
            }
        }
        found
    }

    /// A search over `pairs` (from the first node of each to every node with
    /// `to_every_node`) that keeps `selection` among the paths of `mode`.
    fn selective_search(
        store: &Arc<RowGroupStore>,
        pairs: Vec<(NodeId, NodeId)>,
        to_every_node: bool,
        direction: Direction,
        (selection, mode): (PathSelection, PathMode),
        (min_hops, max_hops): (u32, Option<u32>),
    ) -> ShortestPathOperator {
        let store = Arc::clone(store) as Arc<dyn GraphStore>;
        let input = Box::new(MockPairOperator::new(pairs));
        let op = if to_every_node {
            ShortestPathOperator::from_source(
                store,
                input,
                0,
                vec![],
                direction,
                ReadView::latest(),
            )
        } else {
            ShortestPathOperator::new(store, input, 0, 1, vec![], direction, ReadView::latest())
        };
        op.with_selection(selection)
            .with_path_mode(mode)
            .with_hop_bounds(min_hops, max_hops)
            .with_path_output()
    }

    /// The lengths of the paths found, in order.
    fn lengths(found: &[(NodeId, Vec<Value>)]) -> Vec<usize> {
        found.iter().map(|(_, edges)| edges.len()).collect()
    }

    /// The edge ids of a path.
    fn edge_ids(edges: &[EdgeId]) -> Vec<Value> {
        ids(&[], edges).1
    }

    #[test]
    fn shortest_k_keeps_the_k_shortest_walks_of_a_pair() {
        // a -> b -> d, a -> c -> d, and b -> c: walks a to d of 2, 2 and 3
        let store = Arc::new(RowGroupStore::new());
        let a = store.create_node(&["Node"]);
        let b = store.create_node(&["Node"]);
        let c = store.create_node(&["Node"]);
        let d = store.create_node(&["Node"]);
        let ab = store.create_edge(a, b, "KNOWS");
        store.create_edge(a, c, "KNOWS");
        store.create_edge(b, d, "KNOWS");
        let cd = store.create_edge(c, d, "KNOWS");
        let bc = store.create_edge(b, c, "KNOWS");
        let walk = (PathMode::Walk, (1, None));
        for (selection, want) in [
            (PathSelection::Shortest(2), vec![2, 2]),
            (PathSelection::Shortest(3), vec![2, 2, 3]),
            (PathSelection::Shortest(19), vec![2, 2, 3]),
            (PathSelection::ShortestGroups(1), vec![2, 2]),
            (PathSelection::ShortestGroups(2), vec![2, 2, 3]),
        ] {
            let mut op = selective_search(
                &store,
                vec![(a, d)],
                false,
                Direction::Outgoing,
                (selection, walk.0),
                walk.1,
            );
            let found = found(&mut op, false);
            assert_eq!(lengths(&found), want, "{selection:?}");
            let mut edges: Vec<_> = found.into_iter().map(|(_, edges)| edges).collect();
            edges.sort_by_key(|edges| format!("{edges:?}"));
            edges.dedup();
            assert_eq!(edges.len(), want.len(), "{selection:?}: different paths");
        }
        // The third walk is the one over b -> c
        let mut op = selective_search(
            &store,
            vec![(a, d)],
            false,
            Direction::Outgoing,
            (PathSelection::Shortest(3), PathMode::Walk),
            (1, None),
        );
        assert_eq!(found(&mut op, false)[2].1, edge_ids(&[ab, bc, cd]));
    }

    #[test]
    fn a_walk_search_without_a_maximum_ends_on_a_cycle() {
        // a -> b -> c -> a: each node is reached once every three edges
        let store = Arc::new(RowGroupStore::new());
        let a = store.create_node(&["Node"]);
        let b = store.create_node(&["Node"]);
        let c = store.create_node(&["Node"]);
        store.create_edge(a, b, "KNOWS");
        store.create_edge(b, c, "KNOWS");
        store.create_edge(c, a, "KNOWS");
        for selection in [PathSelection::Shortest(3), PathSelection::ShortestGroups(3)] {
            let mut op = selective_search(
                &store,
                vec![(a, a)],
                true,
                Direction::Outgoing,
                (selection, PathMode::Walk),
                (1, None),
            );
            let found = found(&mut op, true);
            let of = |node: NodeId| -> Vec<usize> {
                found
                    .iter()
                    .filter(|(target, _)| *target == node)
                    .map(|(_, edges)| edges.len())
                    .collect()
            };
            assert_eq!(of(b), vec![1, 4, 7], "{selection:?}");
            assert_eq!(of(c), vec![2, 5, 8], "{selection:?}");
            assert_eq!(of(a), vec![3, 6, 9], "{selection:?}");
            assert_eq!(
                lengths(&found),
                vec![1, 2, 3, 4, 5, 6, 7, 8, 9],
                "in length order"
            );
        }
    }

    #[test]
    fn a_walk_search_takes_every_step_below_the_minimum() {
        // a <-> b and b -> c: walks a to c of at least 3 edges go back and
        // forth first, 4 and 6 edges long
        let store = Arc::new(RowGroupStore::new());
        let a = store.create_node(&["Node"]);
        let b = store.create_node(&["Node"]);
        let c = store.create_node(&["Node"]);
        let ab = store.create_edge(a, b, "KNOWS");
        let back = store.create_edge(b, a, "KNOWS");
        let bc = store.create_edge(b, c, "KNOWS");
        for (selection, want) in [
            (PathSelection::Shortest(1), vec![4]),
            (PathSelection::Shortest(2), vec![4, 6]),
            (PathSelection::ShortestGroups(2), vec![4, 6]),
        ] {
            let mut op = selective_search(
                &store,
                vec![(a, c)],
                false,
                Direction::Outgoing,
                (selection, PathMode::Walk),
                (3, Some(6)),
            );
            let found = found(&mut op, false);
            assert_eq!(lengths(&found), want, "{selection:?}");
            assert_eq!(found[0].1, edge_ids(&[ab, back, ab, bc]));
        }
    }

    /// Undirected: j - m, m - g, m - v, g - v. From j back to m in two or
    /// more edges, a walk goes back over j - m (3 edges), a trail around
    /// g - v (4 edges, two ways), and no acyclic or simple path exists.
    #[test]
    fn a_path_mode_restricts_the_paths_a_search_selects_among() {
        let store = Arc::new(RowGroupStore::new());
        let j = store.create_node(&["Node"]);
        let m = store.create_node(&["Node"]);
        let g = store.create_node(&["Node"]);
        let v = store.create_node(&["Node"]);
        let jm = store.create_edge(j, m, "KNOWS");
        let mg = store.create_edge(m, g, "KNOWS");
        let mv = store.create_edge(m, v, "KNOWS");
        let gv = store.create_edge(g, v, "KNOWS");
        let search = |selection, mode, pair| {
            let mut op = selective_search(
                &store,
                vec![pair],
                false,
                Direction::Both,
                (selection, mode),
                (2, None),
            );
            found(&mut op, false)
        };
        assert_eq!(
            lengths(&search(PathSelection::Shortest(1), PathMode::Walk, (j, m))),
            vec![3]
        );
        let trails = search(PathSelection::ShortestGroups(1), PathMode::Trail, (j, m));
        let mut trail_edges: Vec<_> = trails.iter().map(|(_, edges)| edges.clone()).collect();
        trail_edges.sort_by_key(|edges| format!("{edges:?}"));
        let mut want = vec![edge_ids(&[jm, mg, gv, mv]), edge_ids(&[jm, mv, gv, mg])];
        want.sort_by_key(|edges| format!("{edges:?}"));
        assert_eq!(trail_edges, want, "the two shortest trails");
        assert_eq!(
            lengths(&search(PathSelection::Shortest(1), PathMode::Trail, (j, m))),
            vec![4]
        );
        for mode in [PathMode::Acyclic, PathMode::Simple] {
            assert_eq!(
                search(PathSelection::Shortest(1), mode, (j, m)),
                Vec::new(),
                "{mode:?}"
            );
        }
        // A simple path may end where it starts, and ends there
        assert_eq!(
            search(PathSelection::Shortest(19), PathMode::Simple, (j, j)),
            vec![(j, edge_ids(&[jm, jm]))]
        );
        assert_eq!(
            search(PathSelection::Shortest(1), PathMode::Acyclic, (j, j)),
            Vec::new()
        );
    }

    #[test]
    fn a_simple_path_back_at_its_start_goes_no_further() {
        // Undirected triangle a - b - c - a and a - d: a, b, c, a, d is no
        // simple path, as a repeats before the end
        let store = Arc::new(RowGroupStore::new());
        let a = store.create_node(&["Node"]);
        let b = store.create_node(&["Node"]);
        let c = store.create_node(&["Node"]);
        let d = store.create_node(&["Node"]);
        store.create_edge(a, b, "KNOWS");
        store.create_edge(b, c, "KNOWS");
        store.create_edge(c, a, "KNOWS");
        let ad = store.create_edge(a, d, "KNOWS");
        let mut op = selective_search(
            &store,
            vec![(a, d)],
            false,
            Direction::Both,
            (PathSelection::Shortest(19), PathMode::Simple),
            (1, None),
        );
        assert_eq!(found(&mut op, false), vec![(d, edge_ids(&[ad]))]);
        // The cycles back to a are simple paths: there and back over each
        // edge of a, and both ways round the triangle
        let mut op = selective_search(
            &store,
            vec![(a, a)],
            false,
            Direction::Both,
            (PathSelection::Shortest(19), PathMode::Simple),
            (1, None),
        );
        assert_eq!(lengths(&found(&mut op, false)), vec![2, 2, 2, 3, 3]);
    }

    #[test]
    fn a_restricted_search_to_every_node_keeps_k_per_node() {
        // Undirected square a - b - c - d - a: two acyclic paths to c
        let store = Arc::new(RowGroupStore::new());
        let a = store.create_node(&["Node"]);
        let b = store.create_node(&["Node"]);
        let c = store.create_node(&["Node"]);
        let d = store.create_node(&["Node"]);
        store.create_edge(a, b, "KNOWS");
        store.create_edge(b, c, "KNOWS");
        store.create_edge(c, d, "KNOWS");
        store.create_edge(d, a, "KNOWS");
        let count = |selection| {
            let mut op = selective_search(
                &store,
                vec![(a, a)],
                true,
                Direction::Both,
                (selection, PathMode::Acyclic),
                (1, None),
            );
            let found = found(&mut op, true);
            [b, c, d].map(|node| found.iter().filter(|(target, _)| *target == node).count())
        };
        // b and d have a path of 1 and one of 3 edges, c two of 2
        assert_eq!(count(PathSelection::Shortest(1)), [1, 1, 1]);
        assert_eq!(count(PathSelection::Shortest(2)), [2, 2, 2]);
        assert_eq!(count(PathSelection::ShortestGroups(1)), [1, 2, 1]);
    }

    #[test]
    fn an_undirected_self_loop_is_one_step() {
        let store = Arc::new(RowGroupStore::new());
        let a = store.create_node(&["Node"]);
        let b = store.create_node(&["Node"]);
        let aa = store.create_edge(a, a, "KNOWS");
        store.create_edge(a, b, "KNOWS");
        let mut op = selective_search(
            &store,
            vec![(a, a)],
            false,
            Direction::Both,
            (PathSelection::ShortestGroups(1), PathMode::Walk),
            (1, None),
        );
        assert_eq!(found(&mut op, false), vec![(a, edge_ids(&[aa]))]);
    }

    #[test]
    fn a_search_from_the_source_binds_every_node_it_reaches() {
        // a -> b -> c, and d alone
        let store = Arc::new(RowGroupStore::new());
        let a = store.create_node(&["Node"]);
        let b = store.create_node(&["Node"]);
        let c = store.create_node(&["Node"]);
        let d = store.create_node(&["Node"]);
        let ab = store.create_edge(a, b, "KNOWS");
        let bc = store.create_edge(b, c, "KNOWS");
        for mode in [PathMode::Walk, PathMode::Trail] {
            let mut op = selective_search(
                &store,
                vec![(a, d), (d, a)],
                true,
                Direction::Outgoing,
                (PathSelection::Shortest(1), mode),
                (0, None),
            );
            assert_eq!(
                found(&mut op, true),
                vec![
                    (a, Vec::new()),
                    (b, edge_ids(&[ab])),
                    (c, edge_ids(&[ab, bc])),
                    (d, Vec::new()),
                ],
                "{mode:?}: the second input column is not read"
            );
        }
    }

    // --- The memory budget of a search ---

    /// `diamonds` diamonds in a row: from each joint, two edges to two nodes
    /// and from both on to the next joint, so 2^diamonds shortest paths lead
    /// from the first joint to the last. Returns the store and the joints.
    fn diamonds(diamonds: usize) -> (Arc<RowGroupStore>, Vec<NodeId>) {
        let store = Arc::new(RowGroupStore::new());
        let mut joints = vec![store.create_node(&["Joint"])];
        for _ in 0..diamonds {
            let from = *joints.last().unwrap();
            let to = store.create_node(&["Joint"]);
            for _ in 0..2 {
                let side = store.create_node(&["Side"]);
                store.create_edge(from, side, "NEXT");
                store.create_edge(side, to, "NEXT");
            }
            joints.push(to);
        }
        (store, joints)
    }

    /// The rows `op` returns, or the error it ends with.
    fn drain(op: &mut ShortestPathOperator) -> Result<(usize, usize), OperatorError> {
        let (mut rows, mut chunks) = (0, 0);
        while let Some(chunk) = op.next()? {
            rows += chunk.row_count();
            chunks += 1;
        }
        Ok((rows, chunks))
    }

    fn assert_over_budget(result: Result<(usize, usize), OperatorError>, context: &str) {
        let Some(message) = result
            .as_ref()
            .err()
            .and_then(super::super::variable_length_expand::path_budget_message)
        else {
            panic!("{context}: expected the path search budget's refusal, got {result:?}");
        };
        for advice in ["bound the quantifier", "ALL SHORTEST", "64 KiB"] {
            assert!(
                message.contains(advice),
                "{context}: the message names `{advice}`: {message}"
            );
        }
    }

    #[test]
    fn all_shortest_paths_over_the_budget_fail_with_an_error() {
        // 2^20 shortest paths of 40 edges between the ends: ALL SHORTEST
        // would keep them all
        let (store, joints) = diamonds(20);
        let ends = (joints[0], joints[20]);
        let search = |all: bool| {
            ShortestPathOperator::new(
                Arc::clone(&store) as Arc<dyn GraphStore>,
                Box::new(MockPairOperator::new(vec![ends])),
                0,
                1,
                vec![],
                Direction::Outgoing,
                ReadView::latest(),
            )
            .with_all_paths(all)
            .with_memory_budget(64 * 1024)
        };
        assert_over_budget(drain(&mut search(true)), "ALL SHORTEST");
        assert_eq!(drain(&mut search(false)).unwrap().0, 1, "ANY SHORTEST");
        // Every path of the two shortest lengths, from the walks too
        let mut groups = search(true).with_selection(PathSelection::ShortestGroups(2));
        assert_over_budget(drain(&mut groups), "SHORTEST 2 GROUPS");
    }

    #[test]
    fn a_trail_search_over_the_budget_fails_with_an_error() {
        // Two nodes with six edges each way: the trails from one to every
        // node, which the search follows all, are 6^2 * 5^2 * ... many
        let store = Arc::new(RowGroupStore::new());
        let alix = store.create_node(&["Node"]);
        let gus = store.create_node(&["Node"]);
        for _ in 0..6 {
            store.create_edge(alix, gus, "KNOWS");
            store.create_edge(gus, alix, "KNOWS");
        }
        for mode in [PathMode::Trail, PathMode::Simple] {
            let mut op = selective_search(
                &store,
                vec![(alix, alix)],
                true,
                Direction::Outgoing,
                (PathSelection::Shortest(1), mode),
                (1, None),
            )
            .with_memory_budget(64 * 1024);
            if mode == PathMode::Trail {
                assert_over_budget(drain(&mut op), "TRAIL");
            } else {
                // A simple path visits Gus once: two paths
                assert_eq!(drain(&mut op).unwrap().0, 2, "SIMPLE");
            }
        }
    }

    #[test]
    fn the_paths_of_many_input_rows_come_in_several_chunks() {
        // 300 pairs a diamond apart in one input chunk: a chunk of output
        // ends at a sixty-fourth of the budget of ids (32 here, a path of
        // two edges holds five), and the next call goes on from the next row
        let (store, joints) = diamonds(3);
        let pairs: Vec<(NodeId, NodeId)> = (0..300)
            .map(|i| (joints[i % 3], joints[i % 3 + 1]))
            .collect();
        let search = |budget: usize| {
            ShortestPathOperator::new(
                Arc::clone(&store) as Arc<dyn GraphStore>,
                Box::new(MockPairOperator::new(pairs.clone())),
                0,
                1,
                vec![],
                Direction::Outgoing,
                ReadView::latest(),
            )
            .with_all_paths(true)
            .with_path_output()
            .with_memory_budget(budget)
        };
        let (rows, chunks) = drain(&mut search(16 * 1024)).unwrap();
        assert_eq!(rows, 600, "two shortest paths per pair");
        assert!(chunks >= 600 / 8, "{chunks} chunks");
        assert_eq!(
            drain(&mut search(super::DEFAULT_PATH_SEARCH_BUDGET)).unwrap(),
            (600, 1),
            "one chunk within the default budget"
        );
    }

    #[test]
    fn the_paths_of_one_input_row_come_in_several_chunks() {
        // ALL SHORTEST between two nodes with 200 middles between them: 200
        // paths of two edges, five ids each. A chunk ends at a sixty-fourth
        // of the budget of ids (128 at 64 KiB) within one row's paths too,
        // so it holds 26 paths at most, and each path written gives its
        // charge back
        const MIDDLES: usize = 200;
        let store = Arc::new(RowGroupStore::new());
        let source = store.create_node(&["Node"]);
        let target = store.create_node(&["Node"]);
        for _ in 0..MIDDLES {
            let middle = store.create_node(&["Node"]);
            store.create_edge(source, middle, "KNOWS");
            store.create_edge(middle, target, "KNOWS");
        }
        let mut op = ShortestPathOperator::new(
            Arc::clone(&store) as Arc<dyn GraphStore>,
            Box::new(MockPairOperator::new(vec![(source, target)])),
            0,
            1,
            vec![],
            Direction::Outgoing,
            ReadView::latest(),
        )
        .with_all_paths(true)
        .with_path_output()
        .with_memory_budget(64 * 1024);
        let first = op.next().unwrap().expect("a first chunk").row_count();
        assert!(first <= 128 / 5 + 1, "{first} paths in the first chunk");
        let held = |op: &ShortestPathOperator| op.path_budget.as_ref().map_or(0, PathBudget::used);
        let after_first = held(&op);
        let mut rows = first;
        while let Some(chunk) = op.next().unwrap() {
            assert!(
                chunk.row_count() <= 128 / 5 + 1,
                "{} paths",
                chunk.row_count()
            );
            rows += chunk.row_count();
        }
        assert_eq!(rows, MIDDLES, "every shortest path");
        assert_eq!(
            after_first - held(&op),
            (MIDDLES - first) * path_bytes(2),
            "each path written gave its charge back"
        );
    }

    /// A ring of `size` nodes, each with an edge to the next (the last to the
    /// first), and one node without any edge: the ring's first node and the
    /// isolated one.
    fn ring_and_isolated(size: usize) -> (Arc<RowGroupStore>, NodeId, NodeId) {
        let store = Arc::new(RowGroupStore::new());
        let ring: Vec<NodeId> = (0..size).map(|_| store.create_node(&["Person"])).collect();
        for (i, &node) in ring.iter().enumerate() {
            store.create_edge(node, ring[(i + 1) % size], "KNOWS");
        }
        let isolated = store.create_node(&["Person"]);
        (store, ring[0], isolated)
    }

    /// The rows and the edges read of the shortest path from `source` to
    /// `target` in `direction`.
    fn counted_search(
        store: &Arc<RowGroupStore>,
        (source, target): (NodeId, NodeId),
        direction: Direction,
    ) -> (Vec<(NodeId, NodeId, Value)>, u64) {
        let reads = Arc::new(ReadCounter::default());
        let mut op = ShortestPathOperator::new(
            Arc::clone(store) as Arc<dyn GraphStore>,
            Box::new(MockPairOperator::new(vec![(source, target)])),
            0,
            1,
            vec![],
            direction,
            ReadView::latest(),
        )
        .with_read_counter(Arc::clone(&reads));
        let found = rows(&mut op);
        (found, reads.get())
    }

    #[test]
    fn a_search_to_an_isolated_target_stops_when_the_target_side_runs_out() {
        // The target has no edge, so no path exists once its side is read:
        // the search reads the source's own edges at most, not the ring.
        let (store, source, isolated) = ring_and_isolated(200);
        for (direction, degree) in [(Direction::Outgoing, 1), (Direction::Both, 2)] {
            let (found, reads) = counted_search(&store, (source, isolated), direction);
            assert!(found.is_empty(), "{direction:?}: {found:?}");
            assert!(
                reads <= degree,
                "{direction:?}: {reads} edges read for an isolated target"
            );
        }
    }

    #[test]
    fn a_search_from_an_isolated_source_stops_when_the_source_side_runs_out() {
        // The mirror: the source has no edge, the target sits in the ring.
        let (store, in_ring, isolated) = ring_and_isolated(200);
        for (direction, degree) in [(Direction::Incoming, 1), (Direction::Both, 2)] {
            let (found, reads) = counted_search(&store, (isolated, in_ring), direction);
            assert!(found.is_empty(), "{direction:?}: {found:?}");
            assert!(
                reads <= degree,
                "{direction:?}: {reads} edges read from an isolated source"
            );
        }
    }

    #[test]
    fn a_path_into_a_target_with_one_in_edge_is_still_found() {
        // The source is a hub (100 leaves), so the target's side, one in-edge
        // and then a chain, is the one expanded most; its frontier stays at
        // one node until the sides meet 41 hops from the source.
        let store = Arc::new(RowGroupStore::new());
        let source = store.create_node(&["Person"]);
        for _ in 0..100 {
            let leaf = store.create_node(&["Person"]);
            store.create_edge(source, leaf, "KNOWS");
        }
        let mut previous = source;
        for _ in 0..40 {
            let next = store.create_node(&["Person"]);
            store.create_edge(previous, next, "KNOWS");
            previous = next;
        }
        let target = store.create_node(&["Person"]);
        store.create_edge(previous, target, "KNOWS");
        for direction in [Direction::Outgoing, Direction::Both] {
            let (found, _) = counted_search(&store, (source, target), direction);
            assert_eq!(found, [(source, target, Value::Int64(41))], "{direction:?}");
        }
        // And the other way round, from the target's side.
        let (found, _) = counted_search(&store, (target, source), Direction::Incoming);
        assert_eq!(found, [(target, source, Value::Int64(41))]);
    }

    #[test]
    fn a_path_that_ends_one_side_after_the_sides_meet_is_still_found() {
        // a -> b -> c: the target's side has nothing past b, the source's
        // nothing before a; each side empties once, after the meeting.
        let store = Arc::new(RowGroupStore::new());
        let a = store.create_node(&["Person"]);
        let b = store.create_node(&["Person"]);
        let c = store.create_node(&["Person"]);
        store.create_edge(a, b, "KNOWS");
        store.create_edge(b, c, "KNOWS");
        let (found, _) = counted_search(&store, (a, c), Direction::Outgoing);
        assert_eq!(found, [(a, c, Value::Int64(2))]);
        let (found, _) = counted_search(&store, (c, a), Direction::Both);
        assert_eq!(found, [(c, a, Value::Int64(2))]);
    }

    /// Two trees joined by one bridge: the source reaches ten nodes, each ten
    /// more; ten nodes reach the target, each reached from ten more; every
    /// second-level node has twenty more edges (leaves). One bridge, from the
    /// source's first child to the target's first parent, makes the shortest
    /// path three edges long.
    fn bridged_trees() -> (Arc<RowGroupStore>, NodeId, NodeId) {
        let store = Arc::new(RowGroupStore::new());
        let node = || store.create_node(&["Person"]);
        let leaves = |from: NodeId, outgoing: bool| {
            for _ in 0..20 {
                let leaf = store.create_node(&["Person"]);
                if outgoing {
                    store.create_edge(from, leaf, "KNOWS");
                } else {
                    store.create_edge(leaf, from, "KNOWS");
                }
            }
        };
        let source = node();
        let target = node();
        let mut children = Vec::new();
        let mut parents = Vec::new();
        for _ in 0..10 {
            let child = node();
            store.create_edge(source, child, "KNOWS");
            for _ in 0..10 {
                let grandchild = node();
                store.create_edge(child, grandchild, "KNOWS");
                leaves(grandchild, true);
            }
            children.push(child);
            let parent = node();
            store.create_edge(parent, target, "KNOWS");
            for _ in 0..10 {
                let grandparent = node();
                store.create_edge(grandparent, parent, "KNOWS");
                leaves(grandparent, false);
            }
            parents.push(parent);
        }
        store.create_edge(children[0], parents[0], "KNOWS");
        (store, source, target)
    }

    #[test]
    fn a_search_stops_expanding_nodes_that_cannot_shorten_the_path() {
        // Once the sides meet at three edges, a node two edges from either
        // end can at best tie it: expanding the second levels (their 20
        // leaves each, 4,000 edges) finds nothing shorter. The search needs
        // the two first levels: about 10 + 10 + 10 x 11 + 10 x 11 edges.
        let (store, source, target) = bridged_trees();
        let (found, reads) = counted_search(&store, (source, target), Direction::Outgoing);
        assert_eq!(found, [(source, target, Value::Int64(3))]);
        assert!(reads <= 500, "{reads} edges read for a path of three edges");
    }

    // --- What a search holds: its budget and the statement's grant ---

    use crate::execution::memory::{
        self as statement_memory, OperatorMemoryContext, StatementMemory,
    };
    use grafeo_common::memory::buffer::BufferManager;
    use grafeo_common::utils::error::LimitRefusal;

    /// The middle nodes of a [`fan`].
    const WIDTH: usize = 1024;

    /// A fan: the source to `width` middle nodes, each on to the target.
    /// Returns the store, the source and the target.
    fn fan(width: usize) -> (Arc<RowGroupStore>, NodeId, NodeId) {
        let store = Arc::new(RowGroupStore::new());
        let source = store.create_node(&["Node"]);
        let target = store.create_node(&["Node"]);
        for _ in 0..width {
            let middle = store.create_node(&["Node"]);
            store.create_edge(source, middle, "KNOWS");
            store.create_edge(middle, target, "KNOWS");
        }
        (store, source, target)
    }

    /// A statement under a query memory limit of `cap`.
    fn statement(cap: usize) -> OperatorMemoryContext {
        OperatorMemoryContext::new(BufferManager::with_budget(1 << 30)).with_query_limit(cap)
    }

    /// The operator `build` makes, built in the statement of `ctx`: it takes
    /// the statement's memory at construction, as the planner builds it.
    fn built_in(
        ctx: &OperatorMemoryContext,
        build: impl FnOnce() -> ShortestPathOperator,
    ) -> ShortestPathOperator {
        let _scope = statement_memory::enter(StatementMemory::Budgeted(ctx.clone()));
        build()
    }

    /// The typed refusal behind `error`, when it is one.
    fn limit_refusal(error: &OperatorError) -> Option<&LimitRefusal> {
        match error {
            OperatorError::Wrapped(error) => error.limit_refusal(),
            _ => None,
        }
    }

    /// The paths a path budget refusal says the search held ("would hold
    /// more than N paths"), and the bytes it asked for; `None` for a
    /// refusal of a map or a queue, which names no paths.
    fn paths_named(error: &OperatorError) -> Option<(usize, u64)> {
        let message = super::super::variable_length_expand::path_budget_message(error)?;
        let count = message
            .split("would hold more than ")
            .nth(1)?
            .split_whitespace()
            .next()?
            .parse()
            .ok()?;
        match limit_refusal(error)? {
            LimitRefusal::Memory { requested, .. } => Some((count, *requested)),
            _ => None,
        }
    }

    /// Every selection, path mode, kind of search (to a target, or to every
    /// node) and pair of hop bounds: under the budget the rows of the
    /// unbounded search, past a small budget the budget's refusal, past a
    /// 64 KiB statement the query memory limit's, after which the statement
    /// holds nothing.
    #[test]
    fn every_search_holds_its_state_in_its_budget_and_the_statement() {
        let (store, source, target) = fan(WIDTH);
        let selections = [
            PathSelection::Shortest(1),
            PathSelection::Shortest(3),
            PathSelection::ShortestGroups(1),
        ];
        let modes = [
            PathMode::Walk,
            PathMode::Trail,
            PathMode::Simple,
            PathMode::Acyclic,
        ];
        for selection in selections {
            for mode in modes {
                for to_every_node in [false, true] {
                    for hops in [(1, None), (2, Some(5))] {
                        let case =
                            format!("{selection:?} {mode:?} every node {to_every_node} {hops:?}");
                        let op = || {
                            selective_search(
                                &store,
                                vec![(source, target)],
                                to_every_node,
                                Direction::Outgoing,
                                (selection, mode),
                                hops,
                            )
                        };
                        let want = found(&mut op(), to_every_node);
                        assert!(!want.is_empty(), "{case}: the fan has paths");

                        let ctx = statement(1 << 30);
                        let mut under = built_in(&ctx, || op().with_memory_budget(64 << 20));
                        assert_eq!(found(&mut under, to_every_node), want, "{case}: under");
                        assert!(ctx.query().used() > 0, "{case}: the search holds a grant");
                        drop(under);
                        assert_eq!(ctx.query().used(), 0, "{case}: given back");

                        let ctx = statement(1 << 30);
                        let mut small = built_in(&ctx, || op().with_memory_budget(4 << 10));
                        let error = drain(&mut small).expect_err(&case);
                        let message =
                            super::super::variable_length_expand::path_budget_message(&error)
                                .unwrap_or_else(|| panic!("{case}: the budget's refusal: {error}"));
                        assert!(message.contains("4 KiB"), "{case}: {message}");
                        assert!(
                            message.contains("A shortest path search"),
                            "{case}: {message}"
                        );
                        drop(small);
                        assert_eq!(ctx.query().used(), 0, "{case}: given back");

                        let ctx = statement(64 << 10);
                        let mut capped = built_in(&ctx, || op().with_memory_budget(1 << 30));
                        let error = drain(&mut capped).expect_err(&case);
                        assert!(
                            super::super::variable_length_expand::path_budget_message(&error)
                                .is_none()
                                && error.to_string().contains("query memory limit"),
                            "{case}: the statement's refusal: {error}"
                        );
                        drop(capped);
                        assert_eq!(ctx.query().used(), 0, "{case}: given back");
                    }
                }
            }
        }
    }

    /// `ANY SHORTEST` over a wide fan: its visited maps pass a 16 KiB budget,
    /// which before 0.6.4 they did not count, so the search answered.
    #[test]
    fn any_shortest_over_a_wide_graph_refuses_past_its_budget() {
        let (store, source, target) = fan(WIDTH);
        let mut op = path_search(&store, vec![(source, target)], false, (0, None))
            .with_memory_budget(16 << 10);
        let error = drain(&mut op).expect_err("over the budget");
        let message = super::super::variable_length_expand::path_budget_message(&error)
            .unwrap_or_else(|| panic!("the budget's refusal: {error}"));
        assert!(message.contains("16 KiB"), "{message}");
        let mut op = path_search(&store, vec![(source, target)], false, (0, None));
        assert_eq!(drain(&mut op).unwrap().0, 1, "within the default budget");
    }

    /// The paths a restrictive search's refusal names are the paths it held:
    /// the kept ones, those queued and the one being expanded. `SHORTEST 2`
    /// trails from s to t, over the edge s -> t and over a chain of 1,000
    /// edges whose last node also leads to 5 other nodes: keeping the second
    /// path (16 KB) is refused while the first is kept, the 5 others are
    /// queued and t is expanded, so 7.
    #[test]
    fn a_restrictive_search_names_the_paths_it_held() {
        const CHAIN: usize = 1000;
        const OTHERS: usize = 5;
        let store = Arc::new(RowGroupStore::new());
        let source = store.create_node(&["Node"]);
        let target = store.create_node(&["Node"]);
        store.create_edge(source, target, "KNOWS");
        let mut last = source;
        for _ in 1..CHAIN {
            let next = store.create_node(&["Node"]);
            store.create_edge(last, next, "KNOWS");
            last = next;
        }
        store.create_edge(last, target, "KNOWS");
        for _ in 0..OTHERS {
            let other = store.create_node(&["Node"]);
            store.create_edge(last, other, "KNOWS");
        }
        let mut refused = 0;
        for budget in (0..256 << 10).step_by(256) {
            let mut op = selective_search(
                &store,
                vec![(source, target)],
                false,
                Direction::Outgoing,
                (PathSelection::Shortest(2), PathMode::Trail),
                (0, None),
            )
            .with_memory_budget(budget);
            let Err(error) = drain(&mut op) else {
                break;
            };
            if let Some((paths, requested)) = paths_named(&error)
                && usize::try_from(requested).unwrap() == path_bytes(CHAIN)
            {
                refused += 1;
                assert_eq!(paths, 1 + OTHERS + 1, "{error}");
            }
        }
        assert!(refused > 0, "no budget refused the long path");
    }

    /// A restrictive search refused for a block of its queue names the paths
    /// it held, as one refused for a path does. From a source to 600 leaves,
    /// the last the target, every refusal names a count; one for a block
    /// names the leaves queued, whole blocks of them, and the source's path
    /// being expanded.
    #[test]
    fn a_restrictive_search_refused_for_a_block_names_its_paths() {
        let store = Arc::new(RowGroupStore::new());
        let source = store.create_node(&["Node"]);
        let mut target = source;
        for _ in 0..600 {
            target = store.create_node(&["Node"]);
            store.create_edge(source, target, "KNOWS");
        }
        let entry = size_of::<(Arc<PathSegment>, u32)>();
        let mut blocks = 0;
        for budget in (1024..48 << 10).step_by(64) {
            let mut op = selective_search(
                &store,
                vec![(source, target)],
                false,
                Direction::Outgoing,
                (PathSelection::Shortest(1), PathMode::Trail),
                (0, None),
            )
            .with_memory_budget(budget);
            let Err(error) = drain(&mut op) else {
                break;
            };
            let (paths, requested) =
                paths_named(&error).unwrap_or_else(|| panic!("{budget}: no path count: {error}"));
            let block = PathQueue::<(Arc<PathSegment>, u32)>::block_items_within(budget) * entry;
            if usize::try_from(requested).unwrap() == block {
                blocks += 1;
                assert_eq!((paths - 1) % (block / entry), 0, "{budget}: {error}");
            }
        }
        assert!(blocks > 0, "no budget refused a block");
    }

    /// A search that fails in the middle of a row, the row after one that
    /// answered: the statement's grant goes back when the operator drops,
    /// with no call to give it back.
    #[test]
    fn a_search_that_fails_mid_row_gives_the_statement_back() {
        let (store, source, target) = fan(WIDTH);
        let near = store.create_node(&["Node"]);
        let far = store.create_node(&["Node"]);
        store.create_edge(near, far, "KNOWS");
        let ctx = statement(64 << 10);
        let mut op = built_in(&ctx, || {
            path_search(&store, vec![(near, far), (source, target)], true, (0, None))
                .with_memory_budget(1 << 30)
        });
        let error = drain(&mut op).expect_err("the fan passes the statement's cap");
        assert!(error.to_string().contains("query memory limit"), "{error}");
        assert!(ctx.query().used() > 0, "the failed search holds its grant");
        drop(op);
        assert_eq!(ctx.query().used(), 0);
    }

    /// The bytes a hash table of `entries` entries of `T` holds at least:
    /// its smallest power of two of buckets with room for them, each with its
    /// control byte.
    fn table_bytes<T>(entries: usize) -> usize {
        (entries * 8 / 7 + 1).next_power_of_two() * (size_of::<T>() + 1)
    }

    /// The bytes a list of `len` items of `T` holds, grown one at a time
    /// through [`push`]: its capacity doubles from 4.
    fn list_bytes<T>(len: usize) -> usize {
        len.max(4).next_power_of_two() * size_of::<T>()
    }

    /// What the budget of `op` holds after its rows.
    fn held_after(op: &mut ShortestPathOperator) -> usize {
        drain(op).unwrap();
        op.path_budget.as_ref().map_or(0, PathBudget::used)
    }

    /// What each search holds on a fan of [`WIDTH`] is charged, structure by
    /// structure: what the budget holds after the row is at least what the
    /// structures it still has hold, so a structure that is not charged
    /// (or one given back that was not) makes it fall short.
    #[test]
    fn a_search_charges_every_map_and_queue_it_grows() {
        type Visited = (NodeId, (i64, Option<(NodeId, EdgeId)>));
        type Step = (NodeId, EdgeId);
        type Queued = (Arc<PathSegment>, u32);
        let (store, source, target) = fan(WIDTH);
        let n = WIDTH;
        let search = |selection, mode, to_every_node, hops| {
            let mut op = selective_search(
                &store,
                vec![(source, target)],
                to_every_node,
                Direction::Outgoing,
                (selection, mode),
                hops,
            );
            held_after(&mut op)
        };
        // The steps into the middle nodes: a table entry and a list each
        let middle_steps = table_bytes::<(NodeId, Vec<Step>)>(n) + n * list_bytes::<Step>(1);
        // The list of the paths of two edges, one per middle node, kept (each
        // path gives its own bytes back once it is written)
        let kept_two = list_bytes::<FoundPath>(n);
        // The list of a path to every node: the source's, the middle nodes',
        // the target's
        let kept_each = list_bytes::<FoundPath>(n + 2);

        // ANY SHORTEST: the two visited maps and the two queues
        let held = search(PathSelection::Shortest(1), PathMode::Walk, false, (0, None));
        let want = 2 * table_bytes::<Visited>(n + 1) + 2 * list_bytes::<(NodeId, i64)>(n);
        assert!(held >= want, "ANY SHORTEST: {held} < {want}");

        // The one-way search: the parents and the queue
        let op = path_search(&store, vec![], false, (0, None));
        let mut budget = PathBudget::new(usize::MAX, None);
        op.search_for_row(
            &DataChunk::empty(),
            0,
            Allowance::of(op.expansions.as_deref()),
        )
        .shortest_path(&mut budget, &mut KeptPaths::default(), source, target)
        .unwrap();
        let want = table_bytes::<(NodeId, Step)>(n + 1) + list_bytes::<NodeId>(n);
        assert!(budget.used() >= want, "one way: {} < {want}", budget.used());

        // ALL SHORTEST: the lengths, the steps (the target's list of n), and
        // the kept paths
        let held = search(
            PathSelection::ShortestGroups(1),
            PathMode::Walk,
            false,
            (0, None),
        );
        let want =
            table_bytes::<(NodeId, usize)>(n + 2) + middle_steps + list_bytes::<Step>(n) + kept_two;
        assert!(held >= want, "ALL SHORTEST: {held} < {want}");

        // ALL SHORTEST past a minimum: the first layer, the lengths from the
        // nodes it reaches, the steps on to the target, the kept paths
        let held = search(
            PathSelection::ShortestGroups(1),
            PathMode::Walk,
            false,
            (1, None),
        );
        let want =
            middle_steps + table_bytes::<(NodeId, usize)>(n + 1) + list_bytes::<Step>(n) + kept_two;
        assert!(held >= want, "ALL SHORTEST from one hop: {held} < {want}");

        // SHORTEST 3: the lengths and the steps
        let held = search(PathSelection::Shortest(3), PathMode::Walk, false, (0, None));
        let want = table_bytes::<(NodeId, usize)>(n + 2) + middle_steps + list_bytes::<Step>(n);
        assert!(held >= want, "SHORTEST 3: {held} < {want}");

        // To every node: the lengths, the steps, the ends, the paths taken
        // of each, and the kept paths
        let held = search(PathSelection::Shortest(1), PathMode::Walk, true, (0, None));
        let want = table_bytes::<(NodeId, usize)>(n + 2)
            + middle_steps
            + list_bytes::<Step>(n)
            + list_bytes::<(NodeId, usize)>(n + 2)
            + table_bytes::<(NodeId, usize)>(n + 2)
            + kept_each;
        assert!(held >= want, "to every node: {held} < {want}");

        // A trail to the target: the 2n - 1 segments still held when the
        // target has its path, and the queue
        let held = search(
            PathSelection::Shortest(1),
            PathMode::Trail,
            false,
            (0, None),
        );
        let want = (2 * n - 1) * SEGMENT_BYTES + list_bytes::<Queued>(n);
        assert!(held >= want, "TRAIL: {held} < {want}");

        // Trails to every node: what each end kept, the paths, the queue's
        // spare block once it drained
        let held = search(PathSelection::Shortest(1), PathMode::Trail, true, (0, None));
        let want = table_bytes::<(NodeId, (usize, usize, u32))>(n + 2)
            + kept_each
            + crate::execution::path_queue::BLOCK_BYTES;
        assert!(held >= want, "TRAIL to every node: {held} < {want}");
    }

    /// A chain of 1,023 edges: its lengths, its steps and the list of its one
    /// path held (the path gives its bytes back once it is written), and the
    /// read-back's frames and lists given back as charged.
    #[test]
    fn reading_a_long_path_back_is_charged() {
        let n = 1023;
        let store = Arc::new(RowGroupStore::new());
        let nodes: Vec<NodeId> = (0..=n).map(|_| store.create_node(&["Node"])).collect();
        for pair in nodes.windows(2) {
            store.create_edge(pair[0], pair[1], "KNOWS");
        }
        let mut op = path_search(&store, vec![(nodes[0], nodes[n])], true, (0, None));
        let held = held_after(&mut op);
        let want = table_bytes::<(NodeId, usize)>(n + 1)
            + table_bytes::<(NodeId, Vec<(NodeId, EdgeId)>)>(n)
            + n * list_bytes::<(NodeId, EdgeId)>(1)
            + list_bytes::<FoundPath>(1);
        assert!(held >= want, "{held} < {want}");

        // A walk search keeps a layer per edge, and so does the first part of
        // a search past a minimum: the list of layers, and each layer's steps
        // (the path gives its bytes back once it is written)
        let layer =
            grown_table(1, Vec::<(NodeId, EdgeId)>::new) + list_bytes::<(NodeId, EdgeId)>(1);
        let mut op = selective_search(
            &store,
            vec![(nodes[0], nodes[n])],
            false,
            Direction::Outgoing,
            (PathSelection::Shortest(2), PathMode::Walk),
            (0, None),
        );
        let held = held_after(&mut op);
        let want = list_bytes::<Steps>(n + 1) + n * layer + table_bytes::<(NodeId, usize)>(n + 1);
        assert!(held >= want, "SHORTEST 2: {held} < {want}");
        let minimum = u32::try_from(n).unwrap();
        let mut op = path_search(&store, vec![(nodes[0], nodes[n])], true, (minimum, None));
        let held = held_after(&mut op);
        let want = list_bytes::<Steps>(n) + n * layer;
        assert!(held >= want, "ALL SHORTEST from {n} hops: {held} < {want}");
    }

    /// What a table of `entries` entries of `V` holds, grown an entry at a
    /// time as the searches grow theirs.
    fn grown_table<V>(entries: usize, value: impl Fn() -> V) -> usize {
        let mut budget = PathBudget::new(usize::MAX, None);
        let mut table = FxHashMap::default();
        for id in 0..entries {
            budget.reserve_map(&mut table, 1).unwrap();
            table.insert(NodeId(u64::try_from(id).unwrap()), value());
        }
        table.allocation_size()
    }

    /// On one edge s -> t every search holds exactly what its structures
    /// hold, its first allocations too: each table of an entry or two, each
    /// list of four, the kept paths of one edge.
    #[test]
    fn each_first_allocation_of_a_search_is_charged() {
        type Visited = (i64, Option<(NodeId, EdgeId)>);
        let store = Arc::new(RowGroupStore::new());
        let source = store.create_node(&["Node"]);
        let target = store.create_node(&["Node"]);
        store.create_edge(source, target, "KNOWS");
        let steps =
            grown_table(1, Vec::<(NodeId, EdgeId)>::new) + list_bytes::<(NodeId, EdgeId)>(1);
        let lengths = grown_table(2, || 0_usize);
        // The list of the kept paths: each path gives its bytes back once it
        // is written
        let kept = list_bytes::<FoundPath>;
        let held = |selection, mode, to_every_node| {
            let mut op = selective_search(
                &store,
                vec![(source, target)],
                to_every_node,
                Direction::Outgoing,
                (selection, mode),
                (0, None),
            );
            held_after(&mut op)
        };

        // ANY SHORTEST: a visited table and a queue on each side
        let want =
            2 * grown_table(2, Visited::default) + 2 * list_bytes::<(NodeId, i64)>(1) + kept(1);
        let any = held(PathSelection::Shortest(1), PathMode::Walk, false);
        assert_eq!(any, want, "ANY SHORTEST");

        // The one-way search: the queue and the parents
        let op = path_search(&store, vec![], false, (0, None));
        let mut budget = PathBudget::new(usize::MAX, None);
        op.search_for_row(
            &DataChunk::empty(),
            0,
            Allowance::of(op.expansions.as_deref()),
        )
        .shortest_path(&mut budget, &mut KeptPaths::default(), source, target)
        .unwrap();
        let want = list_bytes::<NodeId>(1)
            + grown_table(1, <(NodeId, EdgeId)>::default)
            + list_bytes::<FoundPath>(1)
            + path_bytes(1);
        assert_eq!(budget.used(), want, "one way");

        // A trail: two segments, the queue's one block (kept as its spare),
        // the target's kept count
        let want = 2 * SEGMENT_BYTES
            + crate::execution::path_queue::BLOCK_BYTES
            + grown_table(1, <(usize, usize, u32)>::default)
            + kept(1);
        let trail = held(PathSelection::Shortest(1), PathMode::Trail, false);
        assert_eq!(trail, want, "TRAIL");

        // A walk to every node: the layers, the frontier (the target's), the
        // lengths, the steps, the ends, the paths taken of each, the paths
        let want = list_bytes::<Steps>(2)
            + list_bytes::<NodeId>(1)
            + lengths
            + steps
            + list_bytes::<(NodeId, usize)>(2)
            + grown_table(2, || 0_usize)
            + kept(2);
        let every = held(PathSelection::Shortest(1), PathMode::Walk, true);
        assert_eq!(every, want, "to every node");

        // ALL SHORTEST: the lengths, the next frontier, the steps, the path
        let want = lengths + list_bytes::<NodeId>(1) + steps + kept(1);
        let all = held(PathSelection::ShortestGroups(1), PathMode::Walk, false);
        assert_eq!(all, want, "ALL SHORTEST");
    }

    /// The step lists a layer drops (into nodes expanded at an earlier
    /// length) are given back: a walk search to every node holds the same
    /// with or without them. The source leads to 128 nodes, the first of
    /// them on to 2,048 more; with an edge from each of the 128 to each other
    /// one, the second layer also holds a list of 127 steps into each of the
    /// 128, which it drops (they were expanded at one edge). Grown through
    /// [`push`], each list has room for 128 steps of 16 bytes: 128 * 128 * 16
    /// is 262,144 bytes dropped. The only thing the dense graph keeps on top
    /// is its larger second frontier (2,176 nodes before the 128 are left
    /// out, room for 4,096 ids against 2,048: 16,384 bytes).
    #[test]
    fn a_walk_search_gives_back_the_steps_a_layer_drops() {
        const NEAR: usize = 128;
        const FAR: usize = 2_048;
        let held = |dense: bool| {
            let store = Arc::new(RowGroupStore::new());
            let source = store.create_node(&["Node"]);
            let near: Vec<NodeId> = (0..NEAR).map(|_| store.create_node(&["Node"])).collect();
            for &node in &near {
                store.create_edge(source, node, "KNOWS");
            }
            for _ in 0..FAR {
                let far = store.create_node(&["Node"]);
                store.create_edge(near[0], far, "KNOWS");
            }
            if dense {
                for &from in &near {
                    for &to in &near {
                        if from != to {
                            store.create_edge(from, to, "KNOWS");
                        }
                    }
                }
            }
            let mut op = selective_search(
                &store,
                vec![(source, source)],
                true,
                Direction::Outgoing,
                (PathSelection::Shortest(1), PathMode::Walk),
                (0, None),
            );
            assert_eq!(
                drain(&mut op).unwrap().0,
                1 + NEAR + FAR,
                "a path to each node"
            );
            op.path_budget.as_ref().map_or(0, PathBudget::used)
        };
        let dropped = NEAR * NEAR * size_of::<(NodeId, EdgeId)>();
        let (dense, sparse) = (held(true), held(false));
        assert!(
            dense < sparse + dropped / 2,
            "the dense graph holds {dense} bytes after the search, the sparse one {sparse}: \
             the {dropped} bytes of dropped steps are still charged"
        );
    }

    /// A restrictive search within a small budget queues as many paths as
    /// 0.6.3 did, less two: 0.6.3 charged a queued path 64 bytes (its entry
    /// and its segment), and a block of the queue is a quarter of what the
    /// budget leaves beside the index of four blocks: the entries' share less
    /// a quarter of the index, 1.5 entries at 96 bytes, so 2 in whole
    /// entries. From a source to each of
    /// `leaves` leaves, the last the target, the search queues every leaf at
    /// once, and answers as the unlimited run does; a budget under the
    /// source's 64 bytes refuses, as in 0.6.3. Below 16 KiB the kept map and
    /// the kept paths reserve what their first entries need, no more, so at
    /// 1 KiB too the search is short by those two only.
    #[test]
    fn a_small_budget_queues_as_many_paths_as_0_6_3() {
        let entry = size_of::<(Arc<PathSegment>, u32)>();
        let fan = |leaves: usize, budget: usize| {
            let store = Arc::new(RowGroupStore::new());
            let source = store.create_node(&["Node"]);
            let mut target = source;
            for _ in 0..leaves {
                target = store.create_node(&["Node"]);
                store.create_edge(source, target, "KNOWS");
            }
            selective_search(
                &store,
                vec![(source, target)],
                false,
                Direction::Outgoing,
                (PathSelection::Shortest(1), PathMode::Trail),
                (0, None),
            )
            .with_memory_budget(budget)
        };
        for kib in [1, 4, 8] {
            let budget = kib * 1024;
            let by_0_6_3 = budget / (entry + SEGMENT_BYTES);
            // One block holds them all: a quarter of what the budget leaves
            // beside the index of four blocks, which 0.6.3 did not count
            let index = crate::execution::path_budget::capacity_within(budget, 3)
                * crate::execution::path_queue::INDEX_SLOT_BYTES;
            let block = (budget - index) / 4 / entry;
            let answered = (1..=by_0_6_3 + 1)
                .take_while(|&leaves| drain(&mut fan(leaves, budget)).is_ok())
                .last()
                .unwrap_or(0);
            assert_eq!(
                (answered, by_0_6_3 - answered),
                (block, 2),
                "{kib} KiB: 0.6.3 queued {by_0_6_3}; one block holds {block}"
            );
            assert_eq!(
                found(&mut fan(answered, budget), false),
                found(&mut fan(answered, usize::MAX), false),
                "{kib} KiB: the paths of the unlimited run"
            );
        }
        assert_over_budget_bytes(drain(&mut fan(1, entry + SEGMENT_BYTES - 1)));
    }

    #[test]
    fn a_larger_small_budget_queues_at_most_a_block_fewer_paths() {
        // At every budget from 1 KiB to 32 KiB in steps of 8 bytes: at least
        // the most queued at any smaller budget less one block's entries, and
        // at 16 KiB no fewer than a byte below. Each count starts from the
        // last, and each fan is built once
        let mut fans: FxHashMap<usize, (Arc<RowGroupStore>, NodeId, NodeId)> = FxHashMap::default();
        let mut answers = |leaves: usize, budget: usize| {
            let (store, source, target) = fans.entry(leaves).or_insert_with(|| {
                let store = Arc::new(RowGroupStore::new());
                let source = store.create_node(&["Node"]);
                let mut target = source;
                for _ in 0..leaves {
                    target = store.create_node(&["Node"]);
                    store.create_edge(source, target, "KNOWS");
                }
                (store, source, target)
            });
            let mut op = selective_search(
                store,
                vec![(*source, *target)],
                false,
                Direction::Outgoing,
                (PathSelection::Shortest(1), PathMode::Trail),
                (0, None),
            )
            .with_memory_budget(budget);
            drain(&mut op).is_ok()
        };
        let mut count = |held: &mut usize, budget: usize| {
            while *held > 0 && !answers(*held, budget) {
                *held -= 1;
            }
            while answers(*held + 1, budget) {
                *held += 1;
            }
            *held
        };
        let (mut held, mut most) = (0, 0);
        for budget in (1024..=32 * 1024).step_by(8) {
            let paths = count(&mut held, budget);
            let block = PathQueue::<(Arc<PathSegment>, u32)>::block_items_within(budget);
            assert!(
                paths + block >= most,
                "{paths} at {budget} bytes, {most} below, blocks of {block}"
            );
            most = most.max(paths);
        }
        let mut held = 0;
        let below = count(&mut held, 16 * 1024 - 1);
        let at_16_kib = count(&mut held, 16 * 1024);
        assert!(
            at_16_kib >= below,
            "{at_16_kib} at 16 KiB, {below} a byte below"
        );
    }

    /// `result` is the path search budget's refusal.
    fn assert_over_budget_bytes(result: Result<(usize, usize), OperatorError>) {
        assert!(
            result.as_ref().is_err_and(|error| {
                super::super::variable_length_expand::path_budget_message(error).is_some()
            }),
            "expected the path search budget's refusal, got {result:?}"
        );
    }
}
