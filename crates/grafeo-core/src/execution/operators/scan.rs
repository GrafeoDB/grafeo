//! Scan operator for reading data from storage.

use super::{Operator, OperatorError, OperatorResult, ReadCounter};
use crate::execution::DataChunk;
use crate::execution::memory::OperatorMemory;
use crate::graph::{GraphStore, IdCursor};
use grafeo_common::types::{LogicalType, NodeId, ReadView};
use std::sync::Arc;

/// The ids a page may hold past what the scan asks for: the row-group store
/// stops at the end of a 64-row word.
const PAGE_SLACK: usize = 63;

/// The bytes of an id in the page.
const ID_BYTES: usize = std::mem::size_of::<NodeId>();

/// A scan operator that reads nodes from storage.
///
/// It reads the IDs it scans from the store a page at a time
/// ([`GraphStore::node_ids_after`]) into one buffer of a chunk and a word
/// more, all at the view it was built with, so it never lists a node its
/// statement creates. The buffer is counted exactly under the statement's
/// memory, charged before it is allocated: below one page the query memory
/// limit refuses it with `GRAFEO-S001`; against the memory limit it is
/// counted but never refused (see [`OperatorMemory::forced`]), so reads of a
/// database over its memory limit answer. A store that gives a longer page
/// (the trait's default lists every ID at once) grows the buffer, and the
/// grant follows it. The memory goes back at the end, on a reset and when
/// the scan drops; outside a budgeted statement it counts nothing.
pub struct ScanOperator {
    /// The store to scan from.
    store: Arc<dyn GraphStore>,
    /// The labels a node needs one of (empty: every node).
    labels: Vec<String>,
    /// Current position in the page.
    position: usize,
    /// The page of node IDs being emitted.
    batch: Vec<NodeId>,
    /// Where the store's paged read stands.
    cursor: IdCursor,
    /// Whether the store has listed every ID: its last read appended none.
    listed: bool,
    /// Whether the scan is exhausted.
    exhausted: bool,
    /// Chunk capacity.
    chunk_capacity: usize,
    /// Who reads: the scan lists the nodes this view sees.
    view: ReadView,
    /// Counts the node IDs the scan reads.
    reads: Option<Arc<ReadCounter>>,
    /// What the page holds under the statement's memory, captured at
    /// construction (see [`capture`](crate::execution::memory::capture)).
    memory: OperatorMemory,
}

impl ScanOperator {
    /// Creates a scan of every node `view` sees.
    pub fn new(store: Arc<dyn GraphStore>, view: ReadView) -> Self {
        Self::with_any_label(store, Vec::new(), view)
    }

    /// Creates a scan of the nodes with `label` that `view` sees.
    pub fn with_label(
        store: Arc<dyn GraphStore>,
        label: impl Into<String>,
        view: ReadView,
    ) -> Self {
        Self::with_any_label(store, vec![label.into()], view)
    }

    /// Creates a scan of the nodes with at least one of `labels` (every
    /// node for none) that `view` sees, each once, in ID order.
    pub fn with_any_label(store: Arc<dyn GraphStore>, labels: Vec<String>, view: ReadView) -> Self {
        Self {
            store,
            labels,
            position: 0,
            batch: Vec::new(),
            cursor: IdCursor::default(),
            listed: false,
            exhausted: false,
            chunk_capacity: 2048,
            view,
            reads: None,
            memory: crate::execution::memory::capture().forced(),
        }
    }

    /// Sets the chunk capacity.
    pub fn with_chunk_capacity(mut self, capacity: usize) -> Self {
        self.chunk_capacity = capacity;
        self
    }

    /// Counts the node IDs the scan reads in `reads`.
    #[must_use]
    pub fn with_read_counter(mut self, reads: Arc<ReadCounter>) -> Self {
        self.reads = Some(reads);
        self
    }

    /// Reads the next page when what is left of this one is short of a
    /// chunk: the rest moves to the front, and the store appends about a
    /// chunk's worth after it.
    fn read_page(&mut self) -> Result<(), OperatorError> {
        let left = self.batch.len() - self.position;
        if self.listed || left >= self.chunk_capacity {
            return Ok(());
        }
        if self.batch.capacity() == 0 {
            // Charged before it is allocated.
            let slots = self.chunk_capacity.saturating_add(PAGE_SLACK);
            if let Err(error) = self.memory.set(slots.saturating_mul(ID_BYTES)) {
                self.memory.release();
                return Err(error);
            }
            self.batch = Vec::with_capacity(slots);
        }
        self.batch.drain(..self.position);
        self.position = 0;
        let read = {
            // On the stack for up to four labels: no allocation per page.
            let labels: smallvec::SmallVec<[&str; 4]> =
                self.labels.iter().map(String::as_str).collect();
            self.store.node_ids_after(
                &labels,
                self.view,
                &mut self.cursor,
                self.chunk_capacity - left,
                &mut self.batch,
            )
        };
        let read = read.map_err(OperatorError::from).and_then(|()| {
            self.memory
                .set(self.batch.capacity().saturating_mul(ID_BYTES))
        });
        if let Err(error) = read {
            self.finish();
            return Err(error);
        }
        let appended = self.batch.len() - left;
        super::read_counter::count(self.reads.as_deref(), appended);
        self.listed = appended == 0;
        Ok(())
    }

    /// The scan emitted its last chunk: it frees the page and gives its
    /// memory back.
    fn finish(&mut self) {
        self.exhausted = true;
        self.position = 0;
        self.batch = Vec::new();
        self.memory.release();
    }
}

impl Operator for ScanOperator {
    fn next(&mut self) -> OperatorResult {
        if self.exhausted {
            return Ok(None);
        }
        self.read_page()?;
        if self.position >= self.batch.len() {
            self.finish();
            return Ok(None);
        }

        // Create output chunk with node IDs
        let schema = [LogicalType::Node];
        let mut chunk = DataChunk::with_capacity(&schema, self.chunk_capacity);

        let end = (self.position + self.chunk_capacity).min(self.batch.len());
        let count = end - self.position;

        {
            // Column 0 guaranteed to exist: chunk created with single-column schema above
            let col = chunk
                .column_mut(0)
                .expect("column 0 exists: chunk created with single-column schema");
            for i in self.position..end {
                col.push_node_id(self.batch[i]);
            }
        }

        chunk.set_count(count);
        self.position = end;

        Ok(Some(chunk))
    }

    fn reset(&mut self) {
        self.position = 0;
        self.batch = Vec::new();
        self.cursor = IdCursor::default();
        self.listed = false;
        self.exhausted = false;
        self.memory.release();
    }

    fn name(&self) -> &'static str {
        "Scan"
    }

    fn into_any(self: Box<Self>) -> Box<dyn std::any::Any + Send> {
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::rowgroup::RowGroupStore;

    #[test]
    fn test_scan_by_label() {
        let store: Arc<RowGroupStore> = Arc::new(RowGroupStore::new());

        store.create_node(&["Person"]);
        store.create_node(&["Person"]);
        store.create_node(&["Animal"]);

        let mut scan = ScanOperator::with_label(
            store.clone() as Arc<dyn GraphStore>,
            "Person",
            ReadView::latest(),
        );

        let chunk = scan.next().unwrap().unwrap();
        assert_eq!(chunk.row_count(), 2);

        // Should be exhausted
        let next = scan.next().unwrap();
        assert!(next.is_none());
    }

    /// A scan of several labels returns each node once, in ID order: a node
    /// with both labels once, a node with neither not at all.
    #[test]
    fn a_scan_of_several_labels_returns_each_node_once() {
        let store: Arc<RowGroupStore> = Arc::new(RowGroupStore::new());
        let director = store.create_node(&["Director"]);
        store.create_node(&["City"]);
        let both = store.create_node(&["Person", "Director"]);
        let person = store.create_node(&["Person"]);

        let mut scan = ScanOperator::with_any_label(
            store.clone() as Arc<dyn GraphStore>,
            vec![
                "Person".to_string(),
                "Director".to_string(),
                "Missing".to_string(),
            ],
            ReadView::latest(),
        );
        let chunk = scan.next().unwrap().unwrap();
        let column = chunk.column(0).unwrap();
        let found: Vec<NodeId> = (0..chunk.row_count())
            .map(|row| column.get_node_id(row).unwrap())
            .collect();
        assert_eq!(found, [director, both, person]);
        assert!(scan.next().unwrap().is_none());
    }

    #[test]
    fn test_scan_reset() {
        let store: Arc<RowGroupStore> = Arc::new(RowGroupStore::new());
        store.create_node(&["Person"]);

        let mut scan = ScanOperator::with_label(
            store.clone() as Arc<dyn GraphStore>,
            "Person",
            ReadView::latest(),
        );

        // First scan
        let chunk1 = scan.next().unwrap().unwrap();
        assert_eq!(chunk1.row_count(), 1);

        // Reset
        scan.reset();

        // Second scan should work
        let chunk2 = scan.next().unwrap().unwrap();
        assert_eq!(chunk2.row_count(), 1);
    }

    #[test]
    fn test_full_scan() {
        let store: Arc<RowGroupStore> = Arc::new(RowGroupStore::new());

        // Create nodes with different labels
        store.create_node(&["Person"]);
        store.create_node(&["Person"]);
        store.create_node(&["Animal"]);
        store.create_node(&["Place"]);

        // Full scan (no label filter) should return all nodes
        let mut scan = ScanOperator::new(store.clone() as Arc<dyn GraphStore>, ReadView::latest());

        let chunk = scan.next().unwrap().unwrap();
        assert_eq!(chunk.row_count(), 4, "Full scan should return all 4 nodes");

        // Should be exhausted
        let next = scan.next().unwrap();
        assert!(next.is_none());
    }

    #[test]
    fn test_scan_with_mvcc_context() {
        let store = Arc::new(RowGroupStore::new());

        // Two nodes, then a third: each create commits at its own epoch.
        store.create_node(&["Person"]);
        store.create_node(&["Person"]);
        let early = store.current_epoch();
        store.create_node(&["Person"]);
        let late = store.current_epoch();

        // A scan at the earlier epoch sees the first two nodes.
        let mut scan = ScanOperator::with_label(
            store.clone() as Arc<dyn GraphStore>,
            "Person",
            ReadView::new(early, None),
        );

        let chunk = scan.next().unwrap().unwrap();
        assert_eq!(
            chunk.row_count(),
            2,
            "Should see 2 nodes at the earlier epoch"
        );

        // A scan at the later epoch sees all three.
        let mut scan_all = ScanOperator::with_label(
            store.clone() as Arc<dyn GraphStore>,
            "Person",
            ReadView::new(late, None),
        );

        let chunk_all = scan_all.next().unwrap().unwrap();
        assert_eq!(
            chunk_all.row_count(),
            3,
            "Should see 3 nodes at the later epoch"
        );
    }

    #[test]
    fn test_scan_into_any() {
        let store: Arc<RowGroupStore> = Arc::new(RowGroupStore::new());
        let op = ScanOperator::with_label(
            store.clone() as Arc<dyn GraphStore>,
            "Person",
            ReadView::latest(),
        );
        let any = Box::new(op).into_any();
        assert!(any.downcast::<ScanOperator>().is_ok());
    }

    /// The scan reads its ids a page at a time into one buffer, counted
    /// exactly under the statement's memory.
    mod pages {
        use super::*;
        use crate::execution::memory::{self, OperatorMemoryContext, StatementMemory};
        use crate::graph::lpg::{Edge, Node};
        use crate::graph::{Direction, IdCursor};
        use crate::statistics::Statistics;
        use arcstr::ArcStr;
        use grafeo_common::memory::buffer::{BufferManager, MemoryRegion};
        use grafeo_common::types::{EdgeId, EpochId, TransactionId};
        use grafeo_common::utils::error::{LimitBound, Result};

        const KIB: usize = 1 << 10;
        const MIB: usize = 1 << 20;

        /// Persons in a store: 10,000 ids, four full chunks of 2,048 and one
        /// of 1,808; their list (80 KB) is more than five pages.
        const PERSONS: usize = 10_000;

        /// The bytes of the page a scan reserves: a chunk of ids and a word
        /// more.
        const PAGE: usize = (2048 + 63) * 8;

        /// A store of `count` committed Persons.
        fn persons(count: usize) -> Arc<RowGroupStore> {
            let store = Arc::new(RowGroupStore::new());
            for _ in 0..count {
                store.create_node(&["Person"]);
            }
            store
        }

        /// A statement's context with a large budget and a `limit` byte cap.
        fn statement(limit: usize) -> OperatorMemoryContext {
            OperatorMemoryContext::new(BufferManager::with_budget(256 * MIB))
                .with_query_limit(limit)
        }

        /// A scan of `labels` at `view`, built inside the statement `ctx`.
        fn scan_in(
            ctx: &OperatorMemoryContext,
            store: Arc<dyn GraphStore>,
            labels: &[&str],
            view: ReadView,
        ) -> ScanOperator {
            let _scope = memory::enter(StatementMemory::Budgeted(ctx.clone()));
            ScanOperator::with_any_label(
                store,
                labels.iter().map(|label| (*label).to_string()).collect(),
                view,
            )
        }

        /// What the statement's operators hold, under its cap and in the
        /// buffer manager.
        fn held(ctx: &OperatorMemoryContext) -> (usize, usize) {
            let region = ctx.buffer_manager().stats().region_allocated
                [MemoryRegion::ExecutionBuffers.index()];
            (ctx.query().used(), region)
        }

        /// `result` is the S001 refusal of the query memory limit, naming it.
        fn assert_refused(result: OperatorResult) {
            let Err(error) = result else {
                panic!("not refused at the query memory limit");
            };
            let OperatorError::Wrapped(inner) = &error else {
                panic!("not a refusal: {error}");
            };
            let text = error.to_string();
            assert_eq!(inner.error_code().as_str(), "GRAFEO-S001", "{text}");
            assert_eq!(
                inner.limit_refusal().map(|refusal| refusal.bound()),
                Some(LimitBound::QueryMemoryLimit),
                "{text}"
            );
            assert!(text.contains(LimitBound::QueryMemoryLimit.name()), "{text}");
        }

        /// What the scan's buffer holds: its grant counts it.
        fn page_bytes(scan: &ScanOperator) -> usize {
            scan.batch.capacity() * std::mem::size_of::<NodeId>()
        }

        /// The ids of every chunk the scan emits, `check` run on the scan and
        /// the chunk's size after each.
        fn drain(
            scan: &mut ScanOperator,
            mut check: impl FnMut(&ScanOperator, usize),
        ) -> Vec<NodeId> {
            let mut ids = Vec::new();
            while let Some(chunk) = scan.next().unwrap() {
                let column = chunk.column(0).unwrap();
                ids.extend((0..chunk.row_count()).map(|row| column.get_node_id(row).unwrap()));
                check(scan, chunk.row_count());
            }
            ids
        }

        #[test]
        fn a_scan_holds_one_page_at_a_time_counted_exactly() {
            let store = persons(PERSONS);
            // A 1 MiB grant step: rounding would show as more than the page.
            let ctx = statement(64 * MIB);
            let mut scan = scan_in(&ctx, store.clone(), &["Person"], ReadView::latest());
            let mut chunks = Vec::new();
            let ids = drain(&mut scan, |scan, rows| {
                chunks.push(rows);
                assert_eq!(scan.memory.granted(), page_bytes(scan), "the exact bytes");
                assert!(
                    [PAGE, 0].contains(&page_bytes(scan)),
                    "one page, or none at the end: {}",
                    page_bytes(scan)
                );
            });
            assert_eq!(
                ids,
                store.nodes_by_label("Person", ReadView::latest()).unwrap()
            );
            assert_eq!(chunks, [2048, 2048, 2048, 2048, 1808], "full chunks");
            assert_eq!(scan.memory.granted(), 0, "the end gives the grant back");
            assert_eq!(scan.batch.capacity(), 0, "and frees the page");
            assert_eq!(held(&ctx), (0, 0));
            assert!(scan.next().unwrap().is_none(), "and stays at its end");
        }

        #[test]
        fn a_scan_below_one_page_refuses_before_it_allocates() {
            let store = persons(PERSONS);
            let ctx = statement(8 * KIB);
            let mut scan = scan_in(&ctx, store, &["Person"], ReadView::latest());
            assert_refused(scan.next());
            assert_eq!(scan.batch.capacity(), 0, "nothing allocated");
            assert_eq!(scan.memory.granted(), 0);
            assert_eq!(held(&ctx), (0, 0));
        }

        /// A limit below the label's whole list (80 KB) and above a page.
        #[test]
        fn a_scan_under_a_limit_below_its_list_answers() {
            let store = persons(PERSONS);
            let ctx = statement(32 * KIB);
            let mut scan = scan_in(&ctx, store, &["Person"], ReadView::latest());
            assert_eq!(drain(&mut scan, |_, _| {}).len(), PERSONS);
            assert_eq!(held(&ctx), (0, 0));
        }

        /// A row group with label chains gives the rest of the group as one
        /// page: the buffer grows, and the grant follows it exactly.
        #[test]
        fn a_page_grown_by_chained_labels_is_counted() {
            let store = persons(PERSONS);
            let open = TransactionId::new(88);
            let epoch: EpochId = store.current_epoch();
            for id in [3, 4_000, 9_000] {
                assert!(store.remove_label_versioned(NodeId::new(id), "Person", open));
            }
            let view = ReadView::transaction(open, epoch);
            let ctx = statement(64 * MIB);
            let mut scan = scan_in(&ctx, store.clone(), &["Person"], view);
            let mut largest = 0;
            let ids = drain(&mut scan, |scan, _| {
                assert_eq!(scan.memory.granted(), page_bytes(scan), "the exact bytes");
                largest = largest.max(page_bytes(scan));
            });
            assert_eq!(ids, store.nodes_by_label("Person", view).unwrap());
            assert_eq!(ids.len(), PERSONS - 3);
            assert!(largest > PAGE, "the page grew: {largest}");
            assert_eq!(held(&ctx), (0, 0));
        }

        /// A buffer manager full of the store's memory: a scan still answers,
        /// its page counted in the manager while it runs and gone after.
        #[test]
        fn a_scan_over_a_full_manager_answers_and_is_counted() {
            let store = persons(PERSONS);
            let manager = BufferManager::with_budget(4 * MIB);
            let _store_memory = manager.reserve_forced(8 * MIB, MemoryRegion::GraphStorage);
            assert!(
                manager
                    .try_allocate(1, MemoryRegion::ExecutionBuffers)
                    .is_none(),
                "the manager is full"
            );
            let ctx = OperatorMemoryContext::new(manager);
            let mut scan = scan_in(&ctx, store, &["Person"], ReadView::latest());
            let mut rows = scan.next().unwrap().expect("a first chunk").row_count();
            assert_eq!(held(&ctx), (PAGE, PAGE), "counted in the manager");
            while let Some(chunk) = scan.next().unwrap() {
                rows += chunk.row_count();
            }
            assert_eq!(rows, PERSONS);
            assert_eq!(held(&ctx), (0, 0), "and gone after");
        }

        /// A drop gives the grant back; a reset does too, and the scan lists
        /// again from the first node.
        #[test]
        fn a_scan_dropped_early_or_reset_gives_its_grant_back() {
            let store = persons(PERSONS);
            let ctx = statement(MIB);
            let mut scan = scan_in(&ctx, store.clone(), &["Person"], ReadView::latest());
            scan.next().unwrap().unwrap();
            assert_eq!(held(&ctx), (PAGE, PAGE));
            drop(scan);
            assert_eq!(held(&ctx), (0, 0), "a drop gives it back");

            let mut scan = scan_in(&ctx, store, &["Person"], ReadView::latest());
            scan.next().unwrap().unwrap();
            scan.next().unwrap().unwrap();
            scan.reset();
            assert_eq!(scan.memory.granted(), 0, "a reset gives it back");
            assert_eq!(held(&ctx), (0, 0));
            let chunk = scan.next().unwrap().unwrap();
            assert_eq!(chunk.row_count(), 2048);
            assert_eq!(
                chunk.column(0).unwrap().get_node_id(0),
                Some(NodeId::new(0))
            );
            assert_eq!(
                drain(&mut scan, |_, _| {}).len(),
                PERSONS - 2048,
                "all again"
            );
        }

        /// Several labels and none page as one call lists, a page at a time.
        #[test]
        fn scans_of_several_labels_and_of_every_node_page() {
            let store = persons(PERSONS);
            for _ in 0..3_000 {
                store.create_node(&["City"]);
            }
            let ctx = statement(MIB);
            let view = ReadView::latest();
            for labels in [&["Person", "City"][..], &[]] {
                let mut scan = scan_in(&ctx, store.clone(), labels, view);
                let ids = drain(&mut scan, |scan, _| {
                    assert!([PAGE, 0].contains(&page_bytes(scan)), "{labels:?}");
                });
                assert_eq!(ids, store.node_ids(view).unwrap(), "{labels:?}");
            }
            let tight = statement(8 * KIB);
            let mut scan = scan_in(&tight, store, &[], view);
            assert_refused(scan.next());
        }

        /// Outside a budgeted statement the scan counts nothing.
        #[test]
        fn an_unbudgeted_scan_counts_nothing() {
            let store = persons(PERSONS);
            let mut scan = ScanOperator::with_label(store, "Person", ReadView::latest());
            assert_eq!(drain(&mut scan, |_, _| {}).len(), PERSONS);
            assert!(!scan.memory.counts());
            assert_eq!(scan.memory.granted(), 0);
        }

        /// The read counter counts each page as it is read.
        #[test]
        fn the_read_counter_counts_each_page() {
            let store = persons(PERSONS);
            let reads = Arc::new(ReadCounter::default());
            let mut scan = ScanOperator::with_label(store, "Person", ReadView::latest())
                .with_read_counter(Arc::clone(&reads));
            scan.next().unwrap().unwrap();
            assert!(reads.get() <= 2048 + 63, "one page read: {}", reads.get());
            drain(&mut scan, |_, _| {});
            assert_eq!(reads.get(), PERSONS as u64);
        }

        /// The row-group store behind the trait's defaults: its paged read
        /// is the default's, its label reads passed through.
        struct Defaults(Arc<RowGroupStore>);

        impl GraphStore for Defaults {
            fn node(&self, id: NodeId, view: ReadView) -> Result<Option<Node>> {
                self.0.node(id, view)
            }
            fn edge(&self, id: EdgeId, view: ReadView) -> Result<Option<Edge>> {
                self.0.edge(id, view)
            }
            fn edges(
                &self,
                node: NodeId,
                direction: Direction,
                types: &[ArcStr],
                view: ReadView,
            ) -> Result<Vec<(NodeId, EdgeId)>> {
                self.0.edges(node, direction, types, view)
            }
            fn has_backward_adjacency(&self) -> bool {
                self.0.has_backward_adjacency()
            }
            fn node_ids(&self, view: ReadView) -> Result<Vec<NodeId>> {
                self.0.node_ids(view)
            }
            fn nodes_by_label(&self, label: &str, view: ReadView) -> Result<Vec<NodeId>> {
                self.0.nodes_by_label(label, view)
            }
            fn nodes_with_any_label(&self, labels: &[&str], view: ReadView) -> Result<Vec<NodeId>> {
                self.0.nodes_with_any_label(labels, view)
            }
            fn all_labels(&self) -> Result<Vec<String>> {
                self.0.all_labels()
            }
            fn all_edge_types(&self) -> Result<Vec<String>> {
                self.0.all_edge_types()
            }
            fn all_property_keys(&self) -> Result<Vec<String>> {
                self.0.all_property_keys()
            }
            fn statistics(&self) -> Arc<Statistics> {
                self.0.statistics()
            }
            fn committed_epoch(&self) -> EpochId {
                self.0.committed_epoch()
            }
        }

        /// The default's paged read lists what the row-group store's pages
        /// list, in one call, and its next call nothing; a scan through it
        /// answers the same and counts its one page exactly.
        #[test]
        fn the_default_lists_what_the_row_group_pages_list() {
            let store = persons(3_000);
            for _ in 0..1_000 {
                store.create_node(&["City"]);
            }
            let defaults = Defaults(Arc::clone(&store));
            let view = ReadView::latest();
            for labels in [&["Person"][..], &["Person", "City"], &[], &["Missing"]] {
                let mut paged = Vec::new();
                let mut cursor = IdCursor::default();
                loop {
                    let before = paged.len();
                    store
                        .node_ids_after(labels, view, &mut cursor, 64, &mut paged)
                        .unwrap();
                    if paged.len() == before {
                        break;
                    }
                }
                let mut cursor = IdCursor::default();
                let mut listed = Vec::new();
                defaults
                    .node_ids_after(labels, view, &mut cursor, 64, &mut listed)
                    .unwrap();
                assert_eq!(listed, paged, "{labels:?}");
                let mut again = Vec::new();
                defaults
                    .node_ids_after(labels, view, &mut cursor, 64, &mut again)
                    .unwrap();
                assert!(again.is_empty(), "{labels:?}: listed once");
            }
            let ctx = statement(64 * MIB);
            let mut scan = scan_in(&ctx, Arc::new(defaults), &["Person"], view);
            let ids = drain(&mut scan, |scan, _| {
                assert_eq!(scan.memory.granted(), page_bytes(scan), "the exact bytes");
            });
            assert_eq!(ids.len(), 3_000);
            assert_eq!(held(&ctx), (0, 0));
        }

        /// A scan never lists the nodes its reader creates while it runs,
        /// through the row-group store's pages or the default: a chunk of one
        /// row, a create after each, emits the Persons there were.
        #[test]
        fn a_scan_never_lists_what_its_reader_creates() {
            for through_default in [false, true] {
                let store = persons(3_000);
                let open = TransactionId::new(77);
                let epoch = store.current_epoch();
                let view = ReadView::transaction(open, epoch);
                let source: Arc<dyn GraphStore> = if through_default {
                    Arc::new(Defaults(Arc::clone(&store)))
                } else {
                    store.clone()
                };
                let mut scan =
                    ScanOperator::with_label(source, "Person", view).with_chunk_capacity(1);
                let mut rows = 0;
                while let Some(chunk) = scan.next().unwrap() {
                    rows += chunk.row_count();
                    store.create_node_versioned(&["Person"], epoch, open);
                }
                assert_eq!(rows, 3_000, "through the default: {through_default}");
                assert_eq!(
                    store.nodes_by_label("Person", view).unwrap().len(),
                    6_000,
                    "the view sees the creates"
                );
            }
        }
    }
}
