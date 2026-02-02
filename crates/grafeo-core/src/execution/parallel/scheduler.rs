//! Morsel scheduler with work-stealing for parallel execution.
//!
//! The scheduler distributes morsels to worker threads using a work-stealing
//! strategy: workers try the global queue, then steal from other workers.
//!
//! # NUMA Awareness
//!
//! The scheduler supports NUMA-aware work stealing. Workers are assigned to
//! NUMA nodes and prefer to steal from workers on the same node to minimize
//! cross-node memory access latency.
//!
//! On systems without explicit NUMA support, workers are assigned to virtual
//! nodes based on their ID, approximating locality through ID proximity.

use super::morsel::Morsel;
use crossbeam::deque::{Injector, Steal, Stealer, Worker};
use parking_lot::Mutex;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

/// NUMA node identifier.
pub type NumaNode = usize;

/// Configuration for NUMA-aware scheduling.
#[derive(Debug, Clone)]
pub struct NumaConfig {
    /// Number of NUMA nodes.
    pub num_nodes: usize,
    /// Workers per NUMA node.
    pub workers_per_node: usize,
}

impl Default for NumaConfig {
    fn default() -> Self {
        // Default: assume uniform memory architecture (1 node)
        Self {
            num_nodes: 1,
            workers_per_node: usize::MAX,
        }
    }
}

impl NumaConfig {
    /// Creates a config for a specific NUMA topology.
    #[must_use]
    pub fn with_topology(num_nodes: usize, workers_per_node: usize) -> Self {
        Self {
            num_nodes,
            workers_per_node,
        }
    }

    /// Auto-detect NUMA topology (approximation based on worker count).
    ///
    /// Heuristic: assume 2 NUMA nodes for > 8 cores, 1 otherwise.
    #[must_use]
    pub fn auto_detect(num_workers: usize) -> Self {
        if num_workers > 8 {
            // Assume 2 NUMA nodes on larger systems
            Self {
                num_nodes: 2,
                workers_per_node: (num_workers + 1) / 2,
            }
        } else {
            Self::default()
        }
    }

    /// Returns the NUMA node for a worker ID.
    #[must_use]
    pub fn worker_node(&self, worker_id: usize) -> NumaNode {
        if self.workers_per_node == usize::MAX {
            0
        } else {
            worker_id / self.workers_per_node
        }
    }
}

/// Work-stealing morsel scheduler.
///
/// Distributes morsels to worker threads efficiently:
/// 1. Workers check the global injector queue
/// 2. If empty, steal from other workers via stealers
///
/// Supports NUMA-aware stealing to minimize cross-node memory access.
pub struct MorselScheduler {
    /// Number of worker threads.
    num_workers: usize,
    /// Global queue for morsel distribution.
    global_queue: Injector<Morsel>,
    /// Stealers for work-stealing (one per worker).
    stealers: Mutex<Vec<Stealer<Morsel>>>,
    /// Count of morsels still being processed.
    active_morsels: AtomicUsize,
    /// Total morsels submitted.
    total_submitted: AtomicUsize,
    /// Whether submission is complete.
    submission_done: AtomicBool,
    /// Whether all work is done.
    done: AtomicBool,
    /// NUMA configuration for locality-aware stealing.
    numa_config: NumaConfig,
}

impl MorselScheduler {
    /// Creates a new scheduler for the given number of workers.
    #[must_use]
    pub fn new(num_workers: usize) -> Self {
        Self::with_numa_config(num_workers, NumaConfig::auto_detect(num_workers))
    }

    /// Creates a scheduler with explicit NUMA configuration.
    #[must_use]
    pub fn with_numa_config(num_workers: usize, numa_config: NumaConfig) -> Self {
        Self {
            num_workers,
            global_queue: Injector::new(),
            stealers: Mutex::new(Vec::with_capacity(num_workers)),
            active_morsels: AtomicUsize::new(0),
            total_submitted: AtomicUsize::new(0),
            submission_done: AtomicBool::new(false),
            done: AtomicBool::new(false),
            numa_config,
        }
    }

    /// Returns the number of workers.
    #[must_use]
    pub fn num_workers(&self) -> usize {
        self.num_workers
    }

    /// Submits a morsel to the global queue.
    pub fn submit(&self, morsel: Morsel) {
        self.global_queue.push(morsel);
        self.active_morsels.fetch_add(1, Ordering::Relaxed);
        self.total_submitted.fetch_add(1, Ordering::Relaxed);
    }

    /// Submits multiple morsels to the global queue.
    pub fn submit_batch(&self, morsels: Vec<Morsel>) {
        let count = morsels.len();
        for morsel in morsels {
            self.global_queue.push(morsel);
        }
        self.active_morsels.fetch_add(count, Ordering::Relaxed);
        self.total_submitted.fetch_add(count, Ordering::Relaxed);
    }

    /// Signals that no more morsels will be submitted.
    pub fn finish_submission(&self) {
        self.submission_done.store(true, Ordering::Release);
        // Check if all work is already done
        if self.active_morsels.load(Ordering::Acquire) == 0 {
            self.done.store(true, Ordering::Release);
        }
    }

    /// Registers a worker's stealer for work-stealing.
    ///
    /// Returns the worker_id assigned.
    pub fn register_worker(&self, stealer: Stealer<Morsel>) -> usize {
        let mut stealers = self.stealers.lock();
        let worker_id = stealers.len();
        stealers.push(stealer);
        worker_id
    }

    /// Gets work from the global queue.
    pub fn get_global_work(&self) -> Option<Morsel> {
        loop {
            match self.global_queue.steal() {
                Steal::Success(morsel) => return Some(morsel),
                Steal::Empty => return None,
                Steal::Retry => continue,
            }
        }
    }

    /// Tries to steal work from other workers.
    ///
    /// Uses NUMA-aware stealing: prefers workers on the same NUMA node
    /// to minimize cross-node memory access latency.
    pub fn steal_work(&self, my_id: usize) -> Option<Morsel> {
        let stealers = self.stealers.lock();
        let num_stealers = stealers.len();

        if num_stealers <= 1 {
            return None;
        }

        // Get my NUMA node
        let my_node = self.numa_config.worker_node(my_id);

        // Phase 1: Try to steal from workers on the same NUMA node
        for i in 1..num_stealers {
            let victim = (my_id + i) % num_stealers;
            let victim_node = self.numa_config.worker_node(victim);

            // Skip workers on different nodes in first pass
            if victim_node != my_node {
                continue;
            }

            if let Some(morsel) = Self::try_steal_from(&stealers[victim]) {
                return Some(morsel);
            }
        }

        // Phase 2: Try workers on other NUMA nodes (cross-node stealing)
        for i in 1..num_stealers {
            let victim = (my_id + i) % num_stealers;
            let victim_node = self.numa_config.worker_node(victim);

            // Only try workers on different nodes now
            if victim_node == my_node {
                continue;
            }

            if let Some(morsel) = Self::try_steal_from(&stealers[victim]) {
                return Some(morsel);
            }
        }

        None
    }

    /// Attempts to steal from a single stealer.
    fn try_steal_from(stealer: &Stealer<Morsel>) -> Option<Morsel> {
        loop {
            match stealer.steal() {
                Steal::Success(morsel) => return Some(morsel),
                Steal::Empty => return None,
                Steal::Retry => continue,
            }
        }
    }

    /// Returns the NUMA node for a worker.
    #[must_use]
    pub fn worker_node(&self, worker_id: usize) -> NumaNode {
        self.numa_config.worker_node(worker_id)
    }

    /// Marks a morsel as completed.
    ///
    /// Must be called after processing each morsel.
    pub fn complete_morsel(&self) {
        let prev = self.active_morsels.fetch_sub(1, Ordering::Release);
        if prev == 1 && self.submission_done.load(Ordering::Acquire) {
            self.done.store(true, Ordering::Release);
        }
    }

    /// Returns whether all work is done.
    #[must_use]
    pub fn is_done(&self) -> bool {
        self.done.load(Ordering::Acquire)
    }

    /// Returns whether submission is complete.
    #[must_use]
    pub fn is_submission_done(&self) -> bool {
        self.submission_done.load(Ordering::Acquire)
    }

    /// Returns the number of active (in-progress) morsels.
    #[must_use]
    pub fn active_count(&self) -> usize {
        self.active_morsels.load(Ordering::Relaxed)
    }

    /// Returns the total number of morsels submitted.
    #[must_use]
    pub fn total_submitted(&self) -> usize {
        self.total_submitted.load(Ordering::Relaxed)
    }
}

impl std::fmt::Debug for MorselScheduler {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MorselScheduler")
            .field("num_workers", &self.num_workers)
            .field(
                "active_morsels",
                &self.active_morsels.load(Ordering::Relaxed),
            )
            .field(
                "total_submitted",
                &self.total_submitted.load(Ordering::Relaxed),
            )
            .field(
                "submission_done",
                &self.submission_done.load(Ordering::Relaxed),
            )
            .field("done", &self.done.load(Ordering::Relaxed))
            .finish()
    }
}

/// Handle for a worker to interact with the scheduler.
///
/// Provides a simpler API for workers with integrated work-stealing.
pub struct WorkerHandle {
    scheduler: Arc<MorselScheduler>,
    worker_id: usize,
    local_queue: Worker<Morsel>,
}

impl WorkerHandle {
    /// Creates a new worker handle and registers with the scheduler.
    #[must_use]
    pub fn new(scheduler: Arc<MorselScheduler>) -> Self {
        let local_queue = Worker::new_fifo();
        let worker_id = scheduler.register_worker(local_queue.stealer());
        Self {
            scheduler,
            worker_id,
            local_queue,
        }
    }

    /// Gets the next morsel to process.
    ///
    /// Tries: local queue -> global queue -> steal from others
    pub fn get_work(&self) -> Option<Morsel> {
        // Try local queue first
        if let Some(morsel) = self.local_queue.pop() {
            return Some(morsel);
        }

        // Try global queue
        if let Some(morsel) = self.scheduler.get_global_work() {
            return Some(morsel);
        }

        // Try stealing from others
        if let Some(morsel) = self.scheduler.steal_work(self.worker_id) {
            return Some(morsel);
        }

        // Check if we're done
        if self.scheduler.is_submission_done() && self.scheduler.active_count() == 0 {
            return None;
        }

        None
    }

    /// Pushes a morsel to this worker's local queue.
    pub fn push_local(&self, morsel: Morsel) {
        self.local_queue.push(morsel);
        self.scheduler
            .active_morsels
            .fetch_add(1, Ordering::Relaxed);
    }

    /// Marks the current morsel as complete.
    pub fn complete_morsel(&self) {
        self.scheduler.complete_morsel();
    }

    /// Returns the worker ID.
    #[must_use]
    pub fn worker_id(&self) -> usize {
        self.worker_id
    }

    /// Returns whether all work is done.
    #[must_use]
    pub fn is_done(&self) -> bool {
        self.scheduler.is_done()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scheduler_creation() {
        let scheduler = MorselScheduler::new(4);
        assert_eq!(scheduler.num_workers(), 4);
        assert_eq!(scheduler.active_count(), 0);
        assert!(!scheduler.is_done());
    }

    #[test]
    fn test_submit_and_get_work() {
        let scheduler = Arc::new(MorselScheduler::new(2));

        scheduler.submit(Morsel::new(0, 0, 0, 1000));
        scheduler.submit(Morsel::new(1, 0, 1000, 2000));
        assert_eq!(scheduler.total_submitted(), 2);
        assert_eq!(scheduler.active_count(), 2);

        // Get work from global queue
        let morsel = scheduler.get_global_work().unwrap();
        assert_eq!(morsel.id, 0);

        // Complete the morsel
        scheduler.complete_morsel();
        assert_eq!(scheduler.active_count(), 1);

        // Get more work
        let morsel = scheduler.get_global_work().unwrap();
        assert_eq!(morsel.id, 1);
        scheduler.complete_morsel();

        scheduler.finish_submission();
        assert!(scheduler.is_done());
    }

    #[test]
    fn test_submit_batch() {
        let scheduler = MorselScheduler::new(4);

        let morsels = vec![
            Morsel::new(0, 0, 0, 100),
            Morsel::new(1, 0, 100, 200),
            Morsel::new(2, 0, 200, 300),
        ];
        scheduler.submit_batch(morsels);

        assert_eq!(scheduler.total_submitted(), 3);
        assert_eq!(scheduler.active_count(), 3);
    }

    #[test]
    fn test_worker_handle() {
        let scheduler = Arc::new(MorselScheduler::new(2));

        let handle = WorkerHandle::new(Arc::clone(&scheduler));
        assert_eq!(handle.worker_id(), 0);
        assert!(!handle.is_done());

        scheduler.submit(Morsel::new(0, 0, 0, 100));

        let morsel = handle.get_work().unwrap();
        assert_eq!(morsel.id, 0);

        handle.complete_morsel();
        scheduler.finish_submission();

        assert!(handle.is_done());
    }

    #[test]
    fn test_worker_local_queue() {
        let scheduler = Arc::new(MorselScheduler::new(2));
        let handle = WorkerHandle::new(Arc::clone(&scheduler));

        // Push to local queue
        handle.push_local(Morsel::new(0, 0, 0, 100));

        // Should get it from local queue
        let morsel = handle.get_work().unwrap();
        assert_eq!(morsel.id, 0);
    }

    #[test]
    fn test_work_stealing() {
        let scheduler = Arc::new(MorselScheduler::new(2));

        // Create two workers
        let handle1 = WorkerHandle::new(Arc::clone(&scheduler));
        let handle2 = WorkerHandle::new(Arc::clone(&scheduler));

        // Push multiple items to worker 1's local queue
        for i in 0..5 {
            handle1.push_local(Morsel::new(i, 0, i * 100, (i + 1) * 100));
        }

        // Worker 1 takes one
        let _ = handle1.get_work().unwrap();

        // Worker 2 should be able to steal
        let stolen = handle2.get_work();
        assert!(stolen.is_some());
    }

    #[test]
    fn test_concurrent_workers() {
        use std::thread;

        let scheduler = Arc::new(MorselScheduler::new(4));
        let total_morsels = 100;

        // Submit morsels
        for i in 0..total_morsels {
            scheduler.submit(Morsel::new(i, 0, i * 100, (i + 1) * 100));
        }
        scheduler.finish_submission();

        // Spawn workers
        let completed = Arc::new(AtomicUsize::new(0));
        let mut handles = Vec::new();

        for _ in 0..4 {
            let sched = Arc::clone(&scheduler);
            let completed = Arc::clone(&completed);

            handles.push(thread::spawn(move || {
                let handle = WorkerHandle::new(sched);
                let mut count = 0;
                while let Some(_morsel) = handle.get_work() {
                    count += 1;
                    handle.complete_morsel();
                }
                completed.fetch_add(count, Ordering::Relaxed);
            }));
        }

        for handle in handles {
            handle.join().unwrap();
        }

        assert_eq!(completed.load(Ordering::Relaxed), total_morsels);
    }

    #[test]
    fn test_numa_config_default() {
        let config = NumaConfig::default();
        assert_eq!(config.num_nodes, 1);
        assert_eq!(config.worker_node(0), 0);
        assert_eq!(config.worker_node(100), 0);
    }

    #[test]
    fn test_numa_config_auto_detect() {
        // Small system: 1 NUMA node
        let config = NumaConfig::auto_detect(4);
        assert_eq!(config.num_nodes, 1);

        // Larger system: 2 NUMA nodes
        let config = NumaConfig::auto_detect(16);
        assert_eq!(config.num_nodes, 2);
        assert_eq!(config.workers_per_node, 8);
    }

    #[test]
    fn test_numa_config_worker_node() {
        let config = NumaConfig::with_topology(2, 4);

        // First 4 workers on node 0
        assert_eq!(config.worker_node(0), 0);
        assert_eq!(config.worker_node(1), 0);
        assert_eq!(config.worker_node(2), 0);
        assert_eq!(config.worker_node(3), 0);

        // Next 4 workers on node 1
        assert_eq!(config.worker_node(4), 1);
        assert_eq!(config.worker_node(5), 1);
        assert_eq!(config.worker_node(6), 1);
        assert_eq!(config.worker_node(7), 1);
    }

    #[test]
    fn test_scheduler_with_numa_config() {
        let config = NumaConfig::with_topology(2, 2);
        let scheduler = MorselScheduler::with_numa_config(4, config);

        assert_eq!(scheduler.num_workers(), 4);
        assert_eq!(scheduler.worker_node(0), 0);
        assert_eq!(scheduler.worker_node(1), 0);
        assert_eq!(scheduler.worker_node(2), 1);
        assert_eq!(scheduler.worker_node(3), 1);
    }

    #[test]
    fn test_numa_aware_stealing() {
        // Create scheduler with 2 NUMA nodes, 2 workers each
        let config = NumaConfig::with_topology(2, 2);
        let scheduler = Arc::new(MorselScheduler::with_numa_config(4, config));

        // Create 4 workers (0,1 on node 0; 2,3 on node 1)
        let handle0 = WorkerHandle::new(Arc::clone(&scheduler));
        let handle1 = WorkerHandle::new(Arc::clone(&scheduler));
        let handle2 = WorkerHandle::new(Arc::clone(&scheduler));
        let _handle3 = WorkerHandle::new(Arc::clone(&scheduler));

        // Worker 0 has work
        for i in 0..10 {
            handle0.push_local(Morsel::new(i, 0, i * 100, (i + 1) * 100));
        }

        // Worker 1 (same NUMA node) should be able to steal
        let stolen1 = handle1.get_work();
        assert!(stolen1.is_some(), "Same-node worker should steal first");

        // Worker 2 (different NUMA node) can also steal
        let stolen2 = handle2.get_work();
        assert!(stolen2.is_some(), "Cross-node worker can steal");
    }
}
