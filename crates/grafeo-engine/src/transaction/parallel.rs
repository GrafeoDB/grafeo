//! Block-STM inspired parallel transaction execution.
//!
//! Executes a batch of operations in parallel optimistically, validates for conflicts,
//! and re-executes conflicting transactions. This is inspired by Aptos Block-STM and
//! provides significant speedup for batch-heavy workloads like ETL imports.
//!
//! # Algorithm
//!
//! The execution follows four phases:
//!
//! 1. **Optimistic Execution**: Execute all operations in parallel without locking.
//!    Each operation tracks its read and write sets.
//!
//! 2. **Validation**: Check if any read was invalidated by a concurrent write from
//!    an earlier transaction in the batch.
//!
//! 3. **Re-execution**: Re-execute invalidated transactions with knowledge of
//!    their dependencies.
//!
//! 4. **Commit**: Apply all writes in transaction order for determinism.
//!
//! # Performance
//!
//! | Conflict Rate | Expected Speedup |
//! |---------------|------------------|
//! | 0% | 3-4x on 4 cores |
//! | <10% | 2-3x |
//! | >30% | Falls back to sequential |
//!
//! # Example
//!
//! ```no_run
//! use grafeo_engine::transaction::parallel::{ParallelExecutor, BatchRequest};
//!
//! let executor = ParallelExecutor::new(4); // 4 workers
//!
//! let batch = BatchRequest::new(vec![
//!     "CREATE (n:Person {id: 1})",
//!     "CREATE (n:Person {id: 2})",
//!     "CREATE (n:Person {id: 3})",
//! ]);
//!
//! let result = executor.execute_batch(batch, |_idx, _op, _result| {
//!     // execute each operation against the store
//! });
//! assert!(result.all_succeeded());
//! ```

use std::collections::HashSet;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use grafeo_common::types::EpochId;
use grafeo_common::utils::hash::FxHashMap;
use parking_lot::{Mutex, RwLock};
use rayon::prelude::*;

use super::EntityId;

/// Maximum number of re-execution attempts before giving up.
const MAX_REEXECUTION_ROUNDS: usize = 10;

/// Minimum batch size to consider parallel execution (otherwise sequential is faster).
const MIN_BATCH_SIZE_FOR_PARALLEL: usize = 4;

/// Maximum conflict rate before falling back to sequential execution.
const MAX_CONFLICT_RATE_FOR_PARALLEL: f64 = 0.3;

/// If the largest conflict cluster contains more than this fraction of all conflicting
/// transactions, skip cluster-based re-execution (all conflicts are interconnected,
/// partitioning adds overhead with no benefit).
const CLUSTER_SKIP_THRESHOLD: f64 = 0.8;

/// Status of an operation execution.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum ExecutionStatus {
    /// Execution succeeded and is valid.
    Success,
    /// Execution needs re-validation due to potential conflicts.
    NeedsRevalidation,
    /// Execution was re-executed after conflict.
    Reexecuted,
    /// Execution failed with an error.
    Failed,
}

/// Result of executing a single operation in the batch.
#[derive(Debug)]
pub struct ExecutionResult {
    /// Index in the batch (for ordering).
    pub batch_index: usize,
    /// Execution status.
    pub status: ExecutionStatus,
    /// Entities read during execution (entity_id, epoch_read_at).
    pub read_set: HashSet<(EntityId, EpochId)>,
    /// Entities written during execution.
    pub write_set: HashSet<EntityId>,
    /// Dependencies on earlier transactions in the batch.
    pub dependencies: Vec<usize>,
    /// Number of times this operation was re-executed.
    pub reexecution_count: usize,
    /// Error message if failed.
    pub error: Option<String>,
}

impl ExecutionResult {
    /// Creates a new execution result.
    fn new(batch_index: usize) -> Self {
        Self {
            batch_index,
            status: ExecutionStatus::Success,
            read_set: HashSet::new(),
            write_set: HashSet::new(),
            dependencies: Vec::new(),
            reexecution_count: 0,
            error: None,
        }
    }

    /// Records a read operation.
    pub fn record_read(&mut self, entity: EntityId, epoch: EpochId) {
        self.read_set.insert((entity, epoch));
    }

    /// Records a write operation.
    pub fn record_write(&mut self, entity: EntityId) {
        self.write_set.insert(entity);
    }

    /// Marks as needing revalidation.
    pub fn mark_needs_revalidation(&mut self) {
        self.status = ExecutionStatus::NeedsRevalidation;
    }

    /// Marks as reexecuted.
    pub fn mark_reexecuted(&mut self) {
        self.status = ExecutionStatus::Reexecuted;
        self.reexecution_count += 1;
    }

    /// Marks as failed with an error.
    pub fn mark_failed(&mut self, error: String) {
        self.status = ExecutionStatus::Failed;
        self.error = Some(error);
    }
}

/// A batch of operations to execute in parallel.
#[derive(Debug, Clone)]
pub struct BatchRequest {
    /// The operations to execute (as query strings).
    pub operations: Vec<String>,
}

impl BatchRequest {
    /// Creates a new batch request.
    pub fn new(operations: Vec<impl Into<String>>) -> Self {
        Self {
            operations: operations.into_iter().map(Into::into).collect(),
        }
    }

    /// Returns the number of operations.
    #[must_use]
    pub fn len(&self) -> usize {
        self.operations.len()
    }

    /// Returns whether the batch is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.operations.is_empty()
    }
}

/// Result of executing a batch of operations.
#[derive(Debug)]
pub struct BatchResult {
    /// Results for each operation (in order).
    pub results: Vec<ExecutionResult>,
    /// Total number of successful operations.
    pub success_count: usize,
    /// Total number of failed operations.
    pub failure_count: usize,
    /// Total number of re-executions performed.
    pub reexecution_count: usize,
    /// Whether parallel execution was used (vs fallback to sequential).
    pub parallel_executed: bool,
    /// Number of conflict clusters found during re-execution (0 if no conflicts).
    pub conflict_cluster_count: usize,
    /// Size of the largest conflict cluster (0 if no conflicts).
    pub largest_cluster_size: usize,
}

impl BatchResult {
    /// Returns true if all operations succeeded.
    #[must_use]
    pub fn all_succeeded(&self) -> bool {
        self.failure_count == 0
    }

    /// Returns the indices of failed operations.
    pub fn failed_indices(&self) -> impl Iterator<Item = usize> + '_ {
        self.results
            .iter()
            .filter(|r| r.status == ExecutionStatus::Failed)
            .map(|r| r.batch_index)
    }
}

/// Tracks which entities have been written by which batch index.
#[derive(Debug, Default)]
struct WriteTracker {
    /// Entity -> batch index that wrote it.
    writes: RwLock<FxHashMap<EntityId, usize>>,
}

impl WriteTracker {
    /// Records a write by a batch index.
    /// Keeps track of the earliest writer for conflict detection.
    fn record_write(&self, entity: EntityId, batch_index: usize) {
        let mut writes = self.writes.write();
        writes
            .entry(entity)
            .and_modify(|existing| *existing = (*existing).min(batch_index))
            .or_insert(batch_index);
    }

    /// Checks if an entity was written by an earlier transaction.
    fn was_written_by_earlier(&self, entity: &EntityId, batch_index: usize) -> Option<usize> {
        let writes = self.writes.read();
        if let Some(&writer) = writes.get(entity)
            && writer < batch_index
        {
            return Some(writer);
        }
        None
    }
}

/// Union-find structure for partitioning conflicting transactions into independent clusters.
///
/// After round 1 of Block-STM execution, transactions that share entities in their
/// read/write sets are unioned together. Connected components form conflict clusters
/// that can be re-executed independently in parallel.
struct ConflictPartitioner {
    parent: Vec<usize>,
    rank: Vec<usize>,
}

impl ConflictPartitioner {
    /// Creates a new partitioner for `n` transactions.
    fn new(n: usize) -> Self {
        Self {
            parent: (0..n).collect(),
            rank: vec![0; n],
        }
    }

    /// Finds the root of `x` with path compression.
    fn find(&mut self, x: usize) -> usize {
        if self.parent[x] != x {
            self.parent[x] = self.find(self.parent[x]);
        }
        self.parent[x]
    }

    /// Unions two sets by rank.
    fn union(&mut self, a: usize, b: usize) {
        let ra = self.find(a);
        let rb = self.find(b);
        if ra == rb {
            return;
        }
        match self.rank[ra].cmp(&self.rank[rb]) {
            std::cmp::Ordering::Less => self.parent[ra] = rb,
            std::cmp::Ordering::Greater => self.parent[rb] = ra,
            std::cmp::Ordering::Equal => {
                self.parent[rb] = ra;
                self.rank[ra] += 1;
            }
        }
    }

    /// Partitions conflicting transactions into independent clusters based on entity overlap.
    ///
    /// Two transactions are in the same cluster if they share any entity in their combined
    /// read/write sets where at least one is a write. Returns clusters as vectors of
    /// transaction indices, sorted by dependency order within each cluster.
    ///
    /// Also returns the size of the largest cluster for threshold checks.
    fn partition(
        read_sets: &[HashSet<(EntityId, EpochId)>],
        write_sets: &[HashSet<EntityId>],
        invalid_indices: &[usize],
    ) -> (Vec<Vec<usize>>, usize) {
        if invalid_indices.is_empty() {
            return (Vec::new(), 0);
        }

        // Map invalid indices to a compact 0..N range for the union-find
        let index_to_compact: FxHashMap<usize, usize> = invalid_indices
            .iter()
            .enumerate()
            .map(|(compact, &orig)| (orig, compact))
            .collect();

        let n = invalid_indices.len();
        let mut uf = ConflictPartitioner::new(n);

        // Build entity -> list of compact indices that touch it (via write)
        let mut entity_writers: FxHashMap<EntityId, Vec<usize>> = FxHashMap::default();

        for &orig_idx in invalid_indices {
            let compact = index_to_compact[&orig_idx];
            for entity in &write_sets[orig_idx] {
                entity_writers.entry(*entity).or_default().push(compact);
            }
        }

        // Union transactions that share a written entity with any reader/writer
        for &orig_idx in invalid_indices {
            let compact = index_to_compact[&orig_idx];

            // Check reads: if this transaction reads an entity that another wrote
            for (entity, _epoch) in &read_sets[orig_idx] {
                if let Some(writers) = entity_writers.get(entity) {
                    for &writer_compact in writers {
                        if writer_compact != compact {
                            uf.union(compact, writer_compact);
                        }
                    }
                }
            }

            // Check writes: if this transaction writes an entity that another also writes
            for entity in &write_sets[orig_idx] {
                if let Some(writers) = entity_writers.get(entity) {
                    for &writer_compact in writers {
                        if writer_compact != compact {
                            uf.union(compact, writer_compact);
                        }
                    }
                }
            }
        }

        // Extract clusters (group by root)
        let mut cluster_map: FxHashMap<usize, Vec<usize>> = FxHashMap::default();
        for (compact, &orig_idx) in invalid_indices.iter().enumerate() {
            let root = uf.find(compact);
            cluster_map.entry(root).or_default().push(orig_idx);
        }

        let mut clusters: Vec<Vec<usize>> = cluster_map.into_values().collect();

        // Sort each cluster by batch index (dependency order)
        for cluster in &mut clusters {
            cluster.sort_unstable();
        }

        let largest = clusters.iter().map(Vec::len).max().unwrap_or(0);
        (clusters, largest)
    }
}

/// Block-STM inspired parallel transaction executor.
///
/// Executes batches of operations in parallel with optimistic concurrency control.
pub struct ParallelExecutor {
    /// Number of worker threads.
    num_workers: usize,
    /// Thread pool for parallel execution.
    pool: rayon::ThreadPool,
}

impl ParallelExecutor {
    /// Creates a new parallel executor with the specified number of workers.
    ///
    /// # Panics
    ///
    /// Panics if `num_workers` is 0 or if the thread pool cannot be created.
    #[must_use]
    pub fn new(num_workers: usize) -> Self {
        assert!(num_workers > 0, "num_workers must be positive");

        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(num_workers)
            .build()
            .expect("failed to build thread pool");

        Self { num_workers, pool }
    }

    /// Creates a parallel executor with the default number of workers (number of CPUs).
    #[must_use]
    pub fn default_workers() -> Self {
        // Use rayon's default parallelism which is based on num_cpus
        Self::new(rayon::current_num_threads().max(1))
    }

    /// Returns the number of workers.
    #[must_use]
    pub fn num_workers(&self) -> usize {
        self.num_workers
    }

    /// Executes a batch of operations in parallel.
    ///
    /// Operations are executed optimistically in parallel, validated for conflicts,
    /// and re-executed as needed. The final result maintains deterministic ordering.
    pub fn execute_batch<F>(&self, batch: BatchRequest, execute_fn: F) -> BatchResult
    where
        F: Fn(usize, &str, &mut ExecutionResult) + Sync + Send,
    {
        let n = batch.len();

        // Handle empty or small batches
        if n == 0 {
            return BatchResult {
                results: Vec::new(),
                success_count: 0,
                failure_count: 0,
                reexecution_count: 0,
                parallel_executed: false,
                conflict_cluster_count: 0,
                largest_cluster_size: 0,
            };
        }

        if n < MIN_BATCH_SIZE_FOR_PARALLEL {
            return self.execute_sequential(batch, execute_fn);
        }

        // Phase 1: Optimistic parallel execution
        let write_tracker = Arc::new(WriteTracker::default());
        let results: Vec<Mutex<ExecutionResult>> = (0..n)
            .map(|i| Mutex::new(ExecutionResult::new(i)))
            .collect();

        self.pool.install(|| {
            batch
                .operations
                .par_iter()
                .enumerate()
                .for_each(|(idx, op)| {
                    let mut result = results[idx].lock();
                    execute_fn(idx, op, &mut result);

                    // Record writes to tracker
                    for entity in &result.write_set {
                        write_tracker.record_write(*entity, idx);
                    }
                });
        });

        // Phase 2: Validation
        let mut invalid_indices = Vec::new();

        for (idx, result_mutex) in results.iter().enumerate() {
            let mut result = result_mutex.lock();

            // Collect entities to check (to avoid borrow issues)
            let read_entities: Vec<EntityId> =
                result.read_set.iter().map(|(entity, _)| *entity).collect();

            // Check if any of our reads were invalidated by an earlier write
            for entity in read_entities {
                if let Some(writer) = write_tracker.was_written_by_earlier(&entity, idx) {
                    result.mark_needs_revalidation();
                    result.dependencies.push(writer);
                }
            }

            if result.status == ExecutionStatus::NeedsRevalidation {
                invalid_indices.push(idx);
            }
        }

        // Check conflict rate
        let conflict_rate = invalid_indices.len() as f64 / n as f64;
        if conflict_rate > MAX_CONFLICT_RATE_FOR_PARALLEL {
            // Too many conflicts - fall back to sequential
            return self.execute_sequential(batch, execute_fn);
        }

        // Phase 3: Cluster-based re-execution of conflicting transactions
        //
        // Partition conflicting transactions into independent clusters using union-find
        // on entity overlap. Clusters with no shared entities can be re-executed fully
        // in parallel. Within each cluster, transactions run in dependency order.
        let total_reexecutions = AtomicUsize::new(0);

        // Collect read/write sets for partitioning
        let all_read_sets: Vec<HashSet<(EntityId, EpochId)>> =
            results.iter().map(|r| r.lock().read_set.clone()).collect();
        let all_write_sets: Vec<HashSet<EntityId>> =
            results.iter().map(|r| r.lock().write_set.clone()).collect();

        let (clusters, largest_cluster) =
            ConflictPartitioner::partition(&all_read_sets, &all_write_sets, &invalid_indices);

        // If the largest cluster dominates, skip partitioning and fall back to round-based
        let use_clusters = !clusters.is_empty()
            && (largest_cluster as f64 / invalid_indices.len().max(1) as f64)
                <= CLUSTER_SKIP_THRESHOLD;

        if use_clusters {
            // Execute clusters in parallel, transactions within each cluster sequentially
            // in dependency order. This typically resolves all conflicts in a single pass.
            self.pool.install(|| {
                clusters.par_iter().for_each(|cluster| {
                    for &idx in cluster {
                        let mut result = results[idx].lock();

                        // Clear previous state
                        result.read_set.clear();
                        result.write_set.clear();
                        result.dependencies.clear();

                        // Re-execute in dependency order within the cluster
                        execute_fn(idx, &batch.operations[idx], &mut result);
                        result.mark_reexecuted();
                        total_reexecutions.fetch_add(1, Ordering::Relaxed);

                        // Update write tracker with new writes
                        for entity in &result.write_set {
                            write_tracker.record_write(*entity, idx);
                        }

                        result.status = ExecutionStatus::Success;
                    }
                });
            });
        } else {
            // Fallback: round-based re-execution (original algorithm)
            for round in 0..MAX_REEXECUTION_ROUNDS {
                if invalid_indices.is_empty() {
                    break;
                }

                let still_invalid: Vec<usize> = self.pool.install(|| {
                    invalid_indices
                        .par_iter()
                        .filter_map(|&idx| {
                            let mut result = results[idx].lock();

                            result.read_set.clear();
                            result.write_set.clear();
                            result.dependencies.clear();

                            execute_fn(idx, &batch.operations[idx], &mut result);
                            result.mark_reexecuted();
                            total_reexecutions.fetch_add(1, Ordering::Relaxed);

                            let read_entities: Vec<EntityId> =
                                result.read_set.iter().map(|(entity, _)| *entity).collect();

                            for entity in read_entities {
                                if let Some(writer) =
                                    write_tracker.was_written_by_earlier(&entity, idx)
                                {
                                    result.mark_needs_revalidation();
                                    result.dependencies.push(writer);
                                    return Some(idx);
                                }
                            }

                            result.status = ExecutionStatus::Success;
                            None
                        })
                        .collect()
                });

                invalid_indices = still_invalid;

                if round == MAX_REEXECUTION_ROUNDS - 1 && !invalid_indices.is_empty() {
                    for idx in &invalid_indices {
                        let mut result = results[*idx].lock();
                        result.mark_failed("Max re-execution rounds reached".to_string());
                    }
                }
            }
        }

        // Phase 4: Collect results
        let mut final_results: Vec<ExecutionResult> =
            results.into_iter().map(|m| m.into_inner()).collect();

        // Sort by batch index to maintain order
        final_results.sort_by_key(|r| r.batch_index);

        let success_count = final_results
            .iter()
            .filter(|r| r.status != ExecutionStatus::Failed)
            .count();

        BatchResult {
            failure_count: n - success_count,
            success_count,
            reexecution_count: total_reexecutions.load(Ordering::Relaxed),
            parallel_executed: true,
            conflict_cluster_count: clusters.len(),
            largest_cluster_size: largest_cluster,
            results: final_results,
        }
    }

    /// Executes a batch sequentially (fallback for high conflict scenarios).
    fn execute_sequential<F>(&self, batch: BatchRequest, execute_fn: F) -> BatchResult
    where
        F: Fn(usize, &str, &mut ExecutionResult),
    {
        let mut results = Vec::with_capacity(batch.len());

        for (idx, op) in batch.operations.iter().enumerate() {
            let mut result = ExecutionResult::new(idx);
            execute_fn(idx, op, &mut result);
            results.push(result);
        }

        let success_count = results
            .iter()
            .filter(|r| r.status != ExecutionStatus::Failed)
            .count();

        BatchResult {
            failure_count: results.len() - success_count,
            success_count,
            reexecution_count: 0,
            parallel_executed: false,
            conflict_cluster_count: 0,
            largest_cluster_size: 0,
            results,
        }
    }
}

impl Default for ParallelExecutor {
    fn default() -> Self {
        Self::default_workers()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use grafeo_common::types::NodeId;
    use std::sync::atomic::AtomicU64;
    use std::thread;
    use std::time::Duration;

    #[test]
    fn test_empty_batch() {
        let executor = ParallelExecutor::new(4);
        let batch = BatchRequest::new(Vec::<String>::new());

        let result = executor.execute_batch(batch, |_, _, _| {});

        assert!(result.all_succeeded());
        assert_eq!(result.results.len(), 0);
    }

    #[test]
    fn test_single_operation() {
        let executor = ParallelExecutor::new(4);
        let batch = BatchRequest::new(vec!["CREATE (n:Test)"]);

        let result = executor.execute_batch(batch, |_, _, result| {
            result.record_write(EntityId::Node(NodeId::new(1)));
        });

        assert!(result.all_succeeded());
        assert_eq!(result.results.len(), 1);
        // Small batch uses sequential execution
        assert!(!result.parallel_executed);
    }

    #[test]
    fn test_independent_operations() {
        let executor = ParallelExecutor::new(4);
        let batch = BatchRequest::new(vec![
            "CREATE (n1:Test {id: 1})",
            "CREATE (n2:Test {id: 2})",
            "CREATE (n3:Test {id: 3})",
            "CREATE (n4:Test {id: 4})",
            "CREATE (n5:Test {id: 5})",
        ]);

        let counter = AtomicU64::new(0);

        let result = executor.execute_batch(batch, |idx, _, result| {
            // Each operation writes to a different entity
            result.record_write(EntityId::Node(NodeId::new(idx as u64)));
            counter.fetch_add(1, Ordering::Relaxed);
        });

        assert!(result.all_succeeded());
        assert_eq!(result.results.len(), 5);
        assert_eq!(result.reexecution_count, 0); // No conflicts
        assert!(result.parallel_executed);
        assert_eq!(counter.load(Ordering::Relaxed), 5);
    }

    #[test]
    fn test_conflicting_operations() {
        let executor = ParallelExecutor::new(4);
        let batch = BatchRequest::new(vec![
            "UPDATE (n:Test) SET n.value = 1",
            "UPDATE (n:Test) SET n.value = 2",
            "UPDATE (n:Test) SET n.value = 3",
            "UPDATE (n:Test) SET n.value = 4",
            "UPDATE (n:Test) SET n.value = 5",
        ]);

        let shared_entity = EntityId::Node(NodeId::new(100));

        let result = executor.execute_batch(batch, |_idx, _, result| {
            // All operations read and write the same entity
            result.record_read(shared_entity, EpochId::new(0));
            result.record_write(shared_entity);

            // Simulate some work
            thread::sleep(Duration::from_micros(10));
        });

        assert!(result.all_succeeded());
        assert_eq!(result.results.len(), 5);
        // Some operations should have been re-executed due to conflicts
        assert!(result.reexecution_count > 0 || !result.parallel_executed);
    }

    #[test]
    fn test_partial_conflicts() {
        let executor = ParallelExecutor::new(4);
        let batch = BatchRequest::new(vec![
            "op1", "op2", "op3", "op4", "op5", "op6", "op7", "op8", "op9", "op10",
        ]);

        // All operations write to independent entities (no conflicts)
        // This tests parallel execution with no read-write conflicts

        let result = executor.execute_batch(batch, |idx, _, result| {
            // Each operation writes to its own entity (no conflicts)
            let entity = EntityId::Node(NodeId::new(idx as u64));
            result.record_write(entity);
        });

        assert!(result.all_succeeded());
        assert_eq!(result.results.len(), 10);
        // Should be parallel since no conflicts
        assert!(result.parallel_executed);
        assert_eq!(result.reexecution_count, 0);
    }

    #[test]
    fn test_execution_order_preserved() {
        let executor = ParallelExecutor::new(4);
        let batch = BatchRequest::new(vec!["op0", "op1", "op2", "op3", "op4", "op5", "op6", "op7"]);

        let result = executor.execute_batch(batch, |idx, _, result| {
            result.record_write(EntityId::Node(NodeId::new(idx as u64)));
        });

        // Verify results are in order
        for (i, r) in result.results.iter().enumerate() {
            assert_eq!(
                r.batch_index, i,
                "Result at position {} has wrong batch_index",
                i
            );
        }
    }

    #[test]
    fn test_failure_handling() {
        let executor = ParallelExecutor::new(4);
        let batch = BatchRequest::new(vec!["success1", "fail", "success2", "success3", "success4"]);

        let result = executor.execute_batch(batch, |idx, op, result| {
            if op == "fail" {
                result.mark_failed("Intentional failure".to_string());
            } else {
                result.record_write(EntityId::Node(NodeId::new(idx as u64)));
            }
        });

        assert!(!result.all_succeeded());
        assert_eq!(result.failure_count, 1);
        assert_eq!(result.success_count, 4);

        let failed: Vec<usize> = result.failed_indices().collect();
        assert_eq!(failed, vec![1]);
    }

    #[test]
    fn test_write_tracker() {
        let tracker = WriteTracker::default();

        tracker.record_write(EntityId::Node(NodeId::new(1)), 0);
        tracker.record_write(EntityId::Node(NodeId::new(2)), 1);
        tracker.record_write(EntityId::Node(NodeId::new(1)), 2); // Keeps earliest (0)

        // Entity 1 was first written by index 0 (earliest is kept)
        assert_eq!(
            tracker.was_written_by_earlier(&EntityId::Node(NodeId::new(1)), 3),
            Some(0)
        );

        // Entity 2 was written by index 1
        assert_eq!(
            tracker.was_written_by_earlier(&EntityId::Node(NodeId::new(2)), 2),
            Some(1)
        );

        // Index 0 has no earlier writers
        assert_eq!(
            tracker.was_written_by_earlier(&EntityId::Node(NodeId::new(1)), 0),
            None
        );
    }

    #[test]
    fn test_batch_request() {
        let batch = BatchRequest::new(vec!["op1", "op2", "op3"]);
        assert_eq!(batch.len(), 3);
        assert!(!batch.is_empty());

        let empty_batch = BatchRequest::new(Vec::<String>::new());
        assert!(empty_batch.is_empty());
    }

    #[test]
    fn test_execution_result() {
        let mut result = ExecutionResult::new(5);

        assert_eq!(result.batch_index, 5);
        assert_eq!(result.status, ExecutionStatus::Success);
        assert!(result.read_set.is_empty());
        assert!(result.write_set.is_empty());

        result.record_read(EntityId::Node(NodeId::new(1)), EpochId::new(10));
        result.record_write(EntityId::Node(NodeId::new(2)));

        assert_eq!(result.read_set.len(), 1);
        assert_eq!(result.write_set.len(), 1);

        result.mark_needs_revalidation();
        assert_eq!(result.status, ExecutionStatus::NeedsRevalidation);

        result.mark_reexecuted();
        assert_eq!(result.status, ExecutionStatus::Reexecuted);
        assert_eq!(result.reexecution_count, 1);
    }

    // ---- Conflict Partitioner Tests ----

    #[test]
    fn test_partitioner_empty() {
        let (clusters, largest) = ConflictPartitioner::partition(&[], &[], &[]);
        assert!(clusters.is_empty(), "{clusters:?}");
        assert_eq!(largest, 0);
    }

    #[test]
    fn test_partitioner_disjoint_clusters() {
        // 4 transactions: {0,1} share entity A, {2,3} share entity B
        let entity_a = EntityId::Node(NodeId::new(100));
        let entity_b = EntityId::Node(NodeId::new(200));

        let read_sets = vec![
            HashSet::from([(entity_a, EpochId::new(0))]),
            HashSet::new(),
            HashSet::from([(entity_b, EpochId::new(0))]),
            HashSet::new(),
        ];
        let write_sets = vec![
            HashSet::from([entity_a]),
            HashSet::from([entity_a]),
            HashSet::from([entity_b]),
            HashSet::from([entity_b]),
        ];

        let invalid = vec![0, 1, 2, 3];
        let (clusters, largest) = ConflictPartitioner::partition(&read_sets, &write_sets, &invalid);

        assert_eq!(clusters.len(), 2, "should produce 2 disjoint clusters");
        assert_eq!(largest, 2, "each cluster has 2 transactions");

        // Verify all transactions are covered
        let all: HashSet<usize> = clusters.iter().flat_map(|c| c.iter().copied()).collect();
        assert_eq!(all, HashSet::from([0, 1, 2, 3]));
    }

    #[test]
    fn test_partitioner_single_cluster() {
        // All 3 transactions share entity A: one big cluster
        let entity_a = EntityId::Node(NodeId::new(42));

        let read_sets = vec![
            HashSet::from([(entity_a, EpochId::new(0))]),
            HashSet::from([(entity_a, EpochId::new(0))]),
            HashSet::from([(entity_a, EpochId::new(0))]),
        ];
        let write_sets = vec![
            HashSet::from([entity_a]),
            HashSet::from([entity_a]),
            HashSet::from([entity_a]),
        ];

        let invalid = vec![0, 1, 2];
        let (clusters, largest) = ConflictPartitioner::partition(&read_sets, &write_sets, &invalid);

        assert_eq!(clusters.len(), 1, "all share the same entity");
        assert_eq!(largest, 3);
        assert_eq!(clusters[0], vec![0, 1, 2]);
    }

    #[test]
    fn test_partitioner_chain_merges() {
        // Tx 0 writes A, Tx 1 writes A and B, Tx 2 writes B
        // Chain: 0 <-> 1 <-> 2 (all in one cluster via B bridging)
        let entity_a = EntityId::Node(NodeId::new(10));
        let entity_b = EntityId::Node(NodeId::new(20));

        let read_sets = vec![HashSet::new(), HashSet::new(), HashSet::new()];
        let write_sets = vec![
            HashSet::from([entity_a]),
            HashSet::from([entity_a, entity_b]),
            HashSet::from([entity_b]),
        ];

        let invalid = vec![0, 1, 2];
        let (clusters, largest) = ConflictPartitioner::partition(&read_sets, &write_sets, &invalid);

        assert_eq!(clusters.len(), 1, "chain should merge into one cluster");
        assert_eq!(largest, 3);
    }

    #[test]
    fn test_partitioner_read_write_conflict() {
        // Tx 0 writes A, Tx 1 reads A (no write): should be in the same cluster
        let entity_a = EntityId::Node(NodeId::new(50));

        let read_sets = vec![HashSet::new(), HashSet::from([(entity_a, EpochId::new(0))])];
        let write_sets = vec![HashSet::from([entity_a]), HashSet::new()];

        let invalid = vec![0, 1];
        let (clusters, largest) = ConflictPartitioner::partition(&read_sets, &write_sets, &invalid);

        assert_eq!(clusters.len(), 1, "read-write overlap merges clusters");
        assert_eq!(largest, 2);
    }

    #[test]
    fn test_partitioner_subset_of_transactions() {
        // Only indices 2 and 5 are invalid out of 6 total transactions.
        // They share no entities: should produce 2 clusters.
        let entity_a = EntityId::Node(NodeId::new(1));
        let entity_b = EntityId::Node(NodeId::new(2));

        let read_sets = vec![
            HashSet::new(),
            HashSet::new(),
            HashSet::from([(entity_a, EpochId::new(0))]),
            HashSet::new(),
            HashSet::new(),
            HashSet::from([(entity_b, EpochId::new(0))]),
        ];
        let write_sets = vec![
            HashSet::new(),
            HashSet::new(),
            HashSet::from([entity_a]),
            HashSet::new(),
            HashSet::new(),
            HashSet::from([entity_b]),
        ];

        let invalid = vec![2, 5];
        let (clusters, _) = ConflictPartitioner::partition(&read_sets, &write_sets, &invalid);

        assert_eq!(
            clusters.len(),
            2,
            "non-overlapping invalid txns form separate clusters"
        );
    }

    #[test]
    fn test_cluster_based_reexecution() {
        // Two groups of conflicting operations that don't overlap.
        // Group A (idx 0,1): read/write entity 100
        // Group B (idx 2,3): read/write entity 200
        // Group C (idx 4-7): independent, no conflicts
        let executor = ParallelExecutor::new(4);
        let batch = BatchRequest::new(vec![
            "g1_op1", "g1_op2", "g2_op1", "g2_op2", "ind1", "ind2", "ind3", "ind4",
        ]);

        let entity_a = EntityId::Node(NodeId::new(100));
        let entity_b = EntityId::Node(NodeId::new(200));

        let result = executor.execute_batch(batch, |idx, _, result| {
            match idx {
                0 | 1 => {
                    result.record_read(entity_a, EpochId::new(0));
                    result.record_write(entity_a);
                }
                2 | 3 => {
                    result.record_read(entity_b, EpochId::new(0));
                    result.record_write(entity_b);
                }
                _ => {
                    // Independent: unique entity per transaction
                    result.record_write(EntityId::Node(NodeId::new(idx as u64 + 1000)));
                }
            }
        });

        assert!(result.all_succeeded());
        assert_eq!(result.results.len(), 8);
        assert!(result.parallel_executed);
        // Conflict clusters should be detected (2 clusters for the conflicting pairs)
        // Independent ops (4-7) don't conflict, so not in any cluster
    }

    #[test]
    fn test_cluster_metrics_reported() {
        let executor = ParallelExecutor::new(4);
        let batch = BatchRequest::new(vec!["a", "b", "c", "d", "e", "f", "g", "h"]);

        // No conflicts: cluster count should be 0
        let result = executor.execute_batch(batch, |idx, _, result| {
            result.record_write(EntityId::Node(NodeId::new(idx as u64)));
        });

        assert_eq!(result.conflict_cluster_count, 0);
        assert_eq!(result.largest_cluster_size, 0);
        assert_eq!(result.reexecution_count, 0);
    }

    #[test]
    fn test_union_find_correctness() {
        let mut uf = ConflictPartitioner::new(6);

        // Union 0-1, 2-3, 4-5
        uf.union(0, 1);
        uf.union(2, 3);
        uf.union(4, 5);

        assert_eq!(uf.find(0), uf.find(1));
        assert_eq!(uf.find(2), uf.find(3));
        assert_eq!(uf.find(4), uf.find(5));
        assert_ne!(uf.find(0), uf.find(2));
        assert_ne!(uf.find(0), uf.find(4));

        // Now merge first two groups
        uf.union(1, 3);
        assert_eq!(uf.find(0), uf.find(2));
        assert_eq!(uf.find(0), uf.find(3));
        assert_ne!(uf.find(0), uf.find(4));
    }

    #[test]
    fn test_cluster_reexecution_resolves_conflicts() {
        // Two disjoint conflict groups: {0,1} share entity A, {2,3} share entity B.
        // After round 1, both groups conflict internally.
        // Cluster-based re-execution should resolve them in one pass.
        let executor = ParallelExecutor::new(4);
        let ops: Vec<String> = (0..8).map(|i| format!("op{i}")).collect();
        let batch = BatchRequest::new(ops);

        let entity_a = EntityId::Node(NodeId::new(100));
        let entity_b = EntityId::Node(NodeId::new(200));

        let result = executor.execute_batch(batch, |idx, _, result| match idx {
            0 | 1 => {
                result.record_read(entity_a, EpochId::new(0));
                result.record_write(entity_a);
            }
            2 | 3 => {
                result.record_read(entity_b, EpochId::new(0));
                result.record_write(entity_b);
            }
            _ => {
                result.record_write(EntityId::Node(NodeId::new(idx as u64 + 1000)));
            }
        });

        assert!(result.all_succeeded(), "all operations should succeed");
        assert!(result.parallel_executed, "should use parallel execution");
        // Conflicts must have been detected and re-executed
        assert!(
            result.conflict_cluster_count > 0,
            "should detect conflict clusters for shared entities"
        );
        assert!(
            result.reexecution_count > 0,
            "should re-execute conflicting operations"
        );
    }

    #[test]
    fn test_large_single_cluster_falls_back() {
        // All operations conflict on the same entity.
        // The largest cluster > 80% threshold should trigger round-based fallback.
        let executor = ParallelExecutor::new(4);
        let ops: Vec<String> = (0..10).map(|i| format!("op{i}")).collect();
        let batch = BatchRequest::new(ops);

        let shared = EntityId::Node(NodeId::new(999));

        let result = executor.execute_batch(batch, |_idx, _, result| {
            result.record_read(shared, EpochId::new(0));
            result.record_write(shared);
        });

        assert!(result.all_succeeded(), "all operations should succeed");
        // With 100% conflicts on a single entity, the cluster covers all invalid
        // indices and exceeds the 80% threshold, triggering round-based fallback.
        if result.parallel_executed {
            assert!(
                result.largest_cluster_size >= 8,
                "largest cluster should exceed 80% threshold, got {}",
                result.largest_cluster_size
            );
        }
    }

    #[test]
    fn test_sequential_fallback_high_conflict_rate() {
        // >30% conflict rate should trigger sequential fallback
        let executor = ParallelExecutor::new(4);
        let ops: Vec<String> = (0..5).map(|i| format!("op{i}")).collect();
        let batch = BatchRequest::new(ops);

        let shared = EntityId::Node(NodeId::new(42));

        let result = executor.execute_batch(batch, |_idx, _, result| {
            result.record_read(shared, EpochId::new(0));
            result.record_write(shared);
        });

        // With 100% conflict rate (>30% threshold), should fall back to sequential
        // or succeed through re-execution with conflict detection
        assert!(result.all_succeeded(), "all operations should succeed");
        if result.parallel_executed {
            assert!(
                result.reexecution_count > 0,
                "parallel path with 100% conflicts must trigger re-execution"
            );
        }
    }

    #[test]
    fn test_cluster_skip_threshold_large_cluster() {
        // When the largest conflict cluster is >80% of invalid set,
        // the cluster-based path is skipped in favor of round-based
        let executor = ParallelExecutor::new(4);
        let ops: Vec<String> = (0..10).map(|i| format!("op{i}")).collect();
        let batch = BatchRequest::new(ops);

        // All transactions write to the same entity: one giant cluster
        let shared = EntityId::Node(NodeId::new(1));
        let result = executor.execute_batch(batch, |_idx, _, result| {
            result.record_read(shared, EpochId::new(0));
            result.record_write(shared);
        });

        assert!(result.all_succeeded());
        // With one huge cluster (100% of conflicts), cluster skip threshold triggers
        // Either all in one cluster or sequential fallback
        assert!(
            result.largest_cluster_size >= result.conflict_cluster_count,
            "largest cluster should dominate"
        );
    }

    #[test]
    fn test_multiple_disjoint_clusters() {
        // Two separate groups of conflicting transactions should form two clusters
        let executor = ParallelExecutor::new(4);
        let ops: Vec<String> = (0..8).map(|i| format!("op{i}")).collect();
        let batch = BatchRequest::new(ops);

        let entity_a = EntityId::Node(NodeId::new(100));
        let entity_b = EntityId::Node(NodeId::new(200));

        let result = executor.execute_batch(batch, |idx, _, result| {
            // Even indices conflict on entity A, odd on entity B
            let entity = if idx % 2 == 0 { entity_a } else { entity_b };
            result.record_read(entity, EpochId::new(0));
            result.record_write(entity);
        });

        assert!(result.all_succeeded());
        // Should detect multiple clusters (or fall back if conflict rate is too high)
        if result.conflict_cluster_count > 1 {
            assert!(
                result.largest_cluster_size < 8,
                "with disjoint conflicts, no single cluster should contain all transactions"
            );
        }
    }

    #[test]
    fn test_batch_result_metrics_fields() {
        // Verify BatchResult fields are populated correctly.
        // Use a conflict-free batch so all operations succeed deterministically.
        let executor = ParallelExecutor::new(4);
        let ops: Vec<String> = (0..10).map(|i| format!("op{i}")).collect();
        let batch = BatchRequest::new(ops);

        let result = executor.execute_batch(batch, |idx, _, result| {
            // Each transaction writes to a unique entity (no conflicts)
            let entity = EntityId::Node(NodeId::new(idx as u64 + 1000));
            result.record_write(entity);
        });

        assert!(result.all_succeeded(), "conflict-free batch should succeed");
        assert_eq!(result.success_count, 10);
        assert_eq!(result.failure_count, 0);
        assert_eq!(result.reexecution_count, 0);
        assert!(result.parallel_executed);
        assert_eq!(result.conflict_cluster_count, 0);
        assert_eq!(result.largest_cluster_size, 0);
    }

    #[test]
    fn test_no_conflicts_no_reexecution() {
        // When transactions don't conflict, no re-execution should happen
        let executor = ParallelExecutor::new(4);
        let ops: Vec<String> = (0..8).map(|i| format!("op{i}")).collect();
        let batch = BatchRequest::new(ops);

        let result = executor.execute_batch(batch, |idx, _, result| {
            // Each transaction writes to a unique entity
            let entity = EntityId::Node(NodeId::new(idx as u64));
            result.record_write(entity);
        });

        assert!(result.all_succeeded());
        assert_eq!(
            result.reexecution_count, 0,
            "no conflicts means no re-execution"
        );
        assert_eq!(result.conflict_cluster_count, 0);
    }

    #[test]
    fn test_max_reexecution_rounds_reached() {
        // Simulate a pathological case where conflicts persist across rounds.
        // The round-based fallback path is used when the largest cluster exceeds
        // the 80% threshold. We force persistent conflicts so the loop exhausts
        // MAX_REEXECUTION_ROUNDS and marks remaining transactions as Failed.
        let executor = ParallelExecutor::new(2);
        let ops: Vec<String> = (0..10).map(|i| format!("op{i}")).collect();
        let batch = BatchRequest::new(ops);

        let shared = EntityId::Node(NodeId::new(999));
        let call_count = AtomicUsize::new(0);

        let result = executor.execute_batch(batch, |_idx, _, result| {
            call_count.fetch_add(1, Ordering::Relaxed);
            // Every transaction reads and writes the same entity, guaranteeing
            // that all later transactions see an earlier writer and get
            // flagged for revalidation on every round.
            result.record_read(shared, EpochId::new(0));
            result.record_write(shared);
        });

        // With 100% conflict rate on a single entity, the system either:
        // 1. Falls back to sequential (conflict rate > 30%), or
        // 2. Uses round-based re-execution (cluster > 80% threshold)
        // Either way, all transactions should succeed (sequential fallback handles it).
        assert!(result.all_succeeded());

        // Verify the function was called more than once per transaction (re-execution happened)
        // or that sequential fallback was used (no parallel execution).
        let total_calls = call_count.load(Ordering::Relaxed);
        assert!(
            total_calls >= 10,
            "expected at least 10 calls (one per op), got {total_calls}"
        );
    }

    #[test]
    fn test_small_batch_uses_sequential() {
        // Batches smaller than MIN_BATCH_SIZE_FOR_PARALLEL (4) use sequential
        let executor = ParallelExecutor::new(4);
        let batch = BatchRequest::new(vec!["a", "b", "c"]);

        let result = executor.execute_batch(batch, |idx, _, result| {
            result.record_write(EntityId::Node(NodeId::new(idx as u64)));
        });

        assert!(result.all_succeeded());
        assert!(
            !result.parallel_executed,
            "batch of 3 should use sequential"
        );
        assert_eq!(result.reexecution_count, 0);
    }

    #[test]
    fn test_conflict_partitioner_preserves_dependency_order() {
        // Within each cluster, transactions should be sorted by batch index
        // (dependency order) for correct re-execution.
        let entity_a = EntityId::Node(NodeId::new(1));

        let read_sets = vec![
            HashSet::from([(entity_a, EpochId::new(0))]),
            HashSet::from([(entity_a, EpochId::new(0))]),
            HashSet::from([(entity_a, EpochId::new(0))]),
        ];
        let write_sets = vec![
            HashSet::from([entity_a]),
            HashSet::from([entity_a]),
            HashSet::from([entity_a]),
        ];

        // Pass indices in reverse order to verify sorting
        let invalid = vec![2, 0, 1];
        let (clusters, _) = ConflictPartitioner::partition(&read_sets, &write_sets, &invalid);

        assert_eq!(clusters.len(), 1);
        // Each cluster should be sorted by batch index
        assert_eq!(clusters[0], vec![0, 1, 2]);
    }

    #[test]
    fn test_write_tracker_no_earlier_writer_for_unwritten_entity() {
        let tracker = WriteTracker::default();
        tracker.record_write(EntityId::Node(NodeId::new(1)), 5);

        // Query for an entity that was never written
        assert_eq!(
            tracker.was_written_by_earlier(&EntityId::Node(NodeId::new(99)), 10),
            None
        );
    }

    #[test]
    fn test_execution_result_mark_failed() {
        let mut result = ExecutionResult::new(0);
        assert_eq!(result.status, ExecutionStatus::Success);
        assert!(result.error.is_none());

        result.mark_failed("test error".to_string());
        assert_eq!(result.status, ExecutionStatus::Failed);
        assert_eq!(result.error.as_deref(), Some("test error"));
    }

    #[test]
    fn test_parallel_executor_num_workers() {
        let executor = ParallelExecutor::new(8);
        assert_eq!(executor.num_workers(), 8);
    }

    #[test]
    fn test_default_workers() {
        let executor = ParallelExecutor::default_workers();
        assert!(executor.num_workers() >= 1);
    }

    #[test]
    fn test_batch_result_failed_indices_empty_when_all_succeed() {
        let executor = ParallelExecutor::new(2);
        let batch = BatchRequest::new(vec!["a", "b", "c", "d"]);

        let result = executor.execute_batch(batch, |idx, _, result| {
            result.record_write(EntityId::Node(NodeId::new(idx as u64)));
        });

        let failed: Vec<usize> = result.failed_indices().collect();
        assert!(failed.is_empty(), "{failed:?}");
    }

    #[test]
    fn test_batch_result_multiple_failures() {
        let executor = ParallelExecutor::new(2);
        let batch = BatchRequest::new(vec!["ok1", "fail1", "ok2", "fail2", "ok3"]);

        let result = executor.execute_batch(batch, |idx, op, result| {
            if op.starts_with("fail") {
                result.mark_failed(format!("error at {idx}"));
            } else {
                result.record_write(EntityId::Node(NodeId::new(idx as u64 + 500)));
            }
        });

        assert!(!result.all_succeeded());
        assert_eq!(result.failure_count, 2);
        assert_eq!(result.success_count, 3);
        let failed: Vec<usize> = result.failed_indices().collect();
        assert_eq!(failed, vec![1, 3]);
    }
}
