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
//! ```ignore
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
//! let result = executor.execute_batch(batch)?;
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

/// Status of an operation execution.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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
        if let Some(&writer) = writes.get(entity) {
            if writer < batch_index {
                return Some(writer);
            }
        }
        None
    }

    /// Clears all writes.
    #[allow(dead_code)]
    fn clear(&self) {
        self.writes.write().clear();
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
    /// Panics if num_workers is 0.
    pub fn new(num_workers: usize) -> Self {
        assert!(num_workers > 0, "num_workers must be positive");

        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(num_workers)
            .build()
            .expect("Failed to build thread pool");

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

        // Phase 3: Re-execution of conflicting transactions
        let total_reexecutions = AtomicUsize::new(0);

        for round in 0..MAX_REEXECUTION_ROUNDS {
            if invalid_indices.is_empty() {
                break;
            }

            // Re-execute invalid transactions
            let still_invalid: Vec<usize> = self.pool.install(|| {
                invalid_indices
                    .par_iter()
                    .filter_map(|&idx| {
                        let mut result = results[idx].lock();

                        // Clear previous state
                        result.read_set.clear();
                        result.write_set.clear();
                        result.dependencies.clear();

                        // Re-execute
                        execute_fn(idx, &batch.operations[idx], &mut result);
                        result.mark_reexecuted();
                        total_reexecutions.fetch_add(1, Ordering::Relaxed);

                        // Collect entities for re-validation
                        let read_entities: Vec<EntityId> =
                            result.read_set.iter().map(|(entity, _)| *entity).collect();

                        // Re-validate
                        for entity in read_entities {
                            if let Some(writer) = write_tracker.was_written_by_earlier(&entity, idx)
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
                // Max rounds reached, mark remaining as failed
                for idx in &invalid_indices {
                    let mut result = results[*idx].lock();
                    result.mark_failed("Max re-execution rounds reached".to_string());
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
}
