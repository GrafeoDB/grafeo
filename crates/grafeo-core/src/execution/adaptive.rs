//! Adaptive query execution with runtime cardinality feedback.
//!
//! This module provides adaptive execution capabilities that allow the query
//! engine to adjust its execution strategy based on actual runtime cardinalities.
//!
//! The key insight is that cardinality estimates can be significantly wrong,
//! especially for complex predicates or skewed data. By tracking actual row
//! counts during execution, we can detect when our estimates are off and
//! potentially re-optimize the remaining query plan.
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────┐     ┌──────────────────┐     ┌─────────────────┐
//! │  Optimizer  │────>│ Estimated Cards  │     │ CardinalityFeed │
//! └─────────────┘     └──────────────────┘     │     back        │
//!                              │               └────────┬────────┘
//!                              v                        │
//!                     ┌──────────────────┐              │
//!                     │ AdaptiveContext  │<─────────────┘
//!                     └────────┬─────────┘
//!                              │
//!                     ┌────────v─────────┐
//!                     │ ReoptimizeCheck  │
//!                     └──────────────────┘
//! ```
//!
//! # Example
//!
//! ```ignore
//! use grafeo_core::execution::adaptive::{AdaptiveContext, CardinalityCheckpoint};
//!
//! // Set up adaptive context with estimated cardinalities
//! let mut ctx = AdaptiveContext::new();
//! ctx.set_estimate("scan_1", 1000.0);
//! ctx.set_estimate("filter_1", 100.0);  // Expected 10% selectivity
//!
//! // During execution, record actuals
//! ctx.record_actual("scan_1", 1000);
//! ctx.record_actual("filter_1", 500);   // Actual 50% selectivity!
//!
//! // Check if re-optimization is warranted
//! if ctx.should_reoptimize() {
//!     // Re-plan remaining operators with updated statistics
//! }
//! ```

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, RwLock};

use super::chunk::DataChunk;
use super::operators::OperatorError;
use super::pipeline::{ChunkSizeHint, PushOperator, Sink};

/// Threshold for deviation that triggers re-optimization consideration.
/// A value of 2.0 means actual cardinality is 2x or 0.5x the estimate.
pub const DEFAULT_REOPTIMIZATION_THRESHOLD: f64 = 3.0;

/// Minimum number of rows before considering re-optimization.
/// Helps avoid thrashing on small result sets.
pub const MIN_ROWS_FOR_REOPTIMIZATION: u64 = 1000;

/// A checkpoint for tracking cardinality at a specific point in the plan.
#[derive(Debug, Clone)]
pub struct CardinalityCheckpoint {
    /// Unique identifier for this checkpoint (typically operator name/id).
    pub operator_id: String,
    /// Estimated cardinality from the optimizer.
    pub estimated: f64,
    /// Actual row count observed during execution.
    pub actual: u64,
    /// Whether this checkpoint has been recorded.
    pub recorded: bool,
}

impl CardinalityCheckpoint {
    /// Creates a new checkpoint with an estimate.
    #[must_use]
    pub fn new(operator_id: &str, estimated: f64) -> Self {
        Self {
            operator_id: operator_id.to_string(),
            estimated,
            actual: 0,
            recorded: false,
        }
    }

    /// Records the actual cardinality.
    pub fn record(&mut self, actual: u64) {
        self.actual = actual;
        self.recorded = true;
    }

    /// Returns the deviation ratio (actual / estimated).
    ///
    /// A ratio > 1.0 means we underestimated, < 1.0 means overestimated.
    /// Returns 1.0 if estimate is 0 to avoid division by zero.
    #[must_use]
    pub fn deviation_ratio(&self) -> f64 {
        if self.estimated <= 0.0 {
            return if self.actual == 0 { 1.0 } else { f64::INFINITY };
        }
        self.actual as f64 / self.estimated
    }

    /// Returns the absolute deviation (|actual - estimated|).
    #[must_use]
    pub fn absolute_deviation(&self) -> f64 {
        (self.actual as f64 - self.estimated).abs()
    }

    /// Checks if this checkpoint shows significant deviation.
    #[must_use]
    pub fn is_significant_deviation(&self, threshold: f64) -> bool {
        if !self.recorded {
            return false;
        }
        let ratio = self.deviation_ratio();
        ratio > threshold || ratio < 1.0 / threshold
    }
}

/// Feedback from runtime execution about actual cardinalities.
///
/// This struct collects actual row counts at various points during query
/// execution, allowing comparison with optimizer estimates.
#[derive(Debug, Default)]
pub struct CardinalityFeedback {
    /// Actual row counts by operator ID.
    actuals: HashMap<String, u64>,
    /// Running count (for streaming updates).
    running_counts: HashMap<String, AtomicU64>,
}

impl CardinalityFeedback {
    /// Creates a new empty feedback collector.
    #[must_use]
    pub fn new() -> Self {
        Self {
            actuals: HashMap::new(),
            running_counts: HashMap::new(),
        }
    }

    /// Records the final actual cardinality for an operator.
    pub fn record(&mut self, operator_id: &str, count: u64) {
        self.actuals.insert(operator_id.to_string(), count);
    }

    /// Adds to the running count for an operator (thread-safe).
    pub fn add_rows(&self, operator_id: &str, count: u64) {
        if let Some(counter) = self.running_counts.get(operator_id) {
            counter.fetch_add(count, Ordering::Relaxed);
        }
    }

    /// Initializes a running counter for an operator.
    pub fn init_counter(&mut self, operator_id: &str) {
        self.running_counts
            .insert(operator_id.to_string(), AtomicU64::new(0));
    }

    /// Finalizes the running count into the actuals.
    pub fn finalize_counter(&mut self, operator_id: &str) {
        if let Some(counter) = self.running_counts.get(operator_id) {
            let count = counter.load(Ordering::Relaxed);
            self.actuals.insert(operator_id.to_string(), count);
        }
    }

    /// Gets the actual count for an operator.
    #[must_use]
    pub fn get(&self, operator_id: &str) -> Option<u64> {
        self.actuals.get(operator_id).copied()
    }

    /// Gets the current running count for an operator.
    #[must_use]
    pub fn get_running(&self, operator_id: &str) -> Option<u64> {
        self.running_counts
            .get(operator_id)
            .map(|c| c.load(Ordering::Relaxed))
    }

    /// Returns all recorded actuals.
    #[must_use]
    pub fn all_actuals(&self) -> &HashMap<String, u64> {
        &self.actuals
    }
}

/// Context for adaptive query execution.
///
/// Holds both estimated and actual cardinalities, and provides methods
/// to determine when re-optimization should occur.
#[derive(Debug)]
pub struct AdaptiveContext {
    /// Checkpoints with estimates and actuals.
    checkpoints: HashMap<String, CardinalityCheckpoint>,
    /// Threshold ratio for triggering re-optimization.
    reoptimization_threshold: f64,
    /// Minimum rows before considering re-optimization.
    min_rows: u64,
    /// Whether re-optimization has been triggered.
    reoptimization_triggered: bool,
    /// Operator that caused re-optimization (if any).
    trigger_operator: Option<String>,
}

impl AdaptiveContext {
    /// Creates a new adaptive context with default settings.
    #[must_use]
    pub fn new() -> Self {
        Self {
            checkpoints: HashMap::new(),
            reoptimization_threshold: DEFAULT_REOPTIMIZATION_THRESHOLD,
            min_rows: MIN_ROWS_FOR_REOPTIMIZATION,
            reoptimization_triggered: false,
            trigger_operator: None,
        }
    }

    /// Creates a context with custom thresholds.
    #[must_use]
    pub fn with_thresholds(threshold: f64, min_rows: u64) -> Self {
        Self {
            checkpoints: HashMap::new(),
            reoptimization_threshold: threshold,
            min_rows,
            reoptimization_triggered: false,
            trigger_operator: None,
        }
    }

    /// Sets the estimated cardinality for an operator.
    pub fn set_estimate(&mut self, operator_id: &str, estimate: f64) {
        self.checkpoints.insert(
            operator_id.to_string(),
            CardinalityCheckpoint::new(operator_id, estimate),
        );
    }

    /// Records the actual cardinality for an operator.
    pub fn record_actual(&mut self, operator_id: &str, actual: u64) {
        if let Some(checkpoint) = self.checkpoints.get_mut(operator_id) {
            checkpoint.record(actual);
        } else {
            // Create checkpoint with unknown estimate
            let mut checkpoint = CardinalityCheckpoint::new(operator_id, 0.0);
            checkpoint.record(actual);
            self.checkpoints.insert(operator_id.to_string(), checkpoint);
        }
    }

    /// Applies feedback from a `CardinalityFeedback` collector.
    pub fn apply_feedback(&mut self, feedback: &CardinalityFeedback) {
        for (op_id, &actual) in feedback.all_actuals() {
            self.record_actual(op_id, actual);
        }
    }

    /// Checks if any checkpoint shows significant deviation.
    #[must_use]
    pub fn has_significant_deviation(&self) -> bool {
        self.checkpoints
            .values()
            .any(|cp| cp.is_significant_deviation(self.reoptimization_threshold))
    }

    /// Determines if re-optimization should be triggered.
    ///
    /// Returns true if:
    /// - There's significant deviation from estimates
    /// - We've processed enough rows to make a meaningful decision
    /// - Re-optimization hasn't already been triggered
    #[must_use]
    pub fn should_reoptimize(&mut self) -> bool {
        if self.reoptimization_triggered {
            return false;
        }

        for (op_id, checkpoint) in &self.checkpoints {
            if checkpoint.actual < self.min_rows {
                continue;
            }

            if checkpoint.is_significant_deviation(self.reoptimization_threshold) {
                self.reoptimization_triggered = true;
                self.trigger_operator = Some(op_id.clone());
                return true;
            }
        }

        false
    }

    /// Returns the operator that triggered re-optimization, if any.
    #[must_use]
    pub fn trigger_operator(&self) -> Option<&str> {
        self.trigger_operator.as_deref()
    }

    /// Gets the checkpoint for an operator.
    #[must_use]
    pub fn get_checkpoint(&self, operator_id: &str) -> Option<&CardinalityCheckpoint> {
        self.checkpoints.get(operator_id)
    }

    /// Returns all checkpoints.
    #[must_use]
    pub fn all_checkpoints(&self) -> &HashMap<String, CardinalityCheckpoint> {
        &self.checkpoints
    }

    /// Calculates a correction factor for a specific operator.
    ///
    /// This factor can be used to adjust remaining cardinality estimates.
    #[must_use]
    pub fn correction_factor(&self, operator_id: &str) -> f64 {
        self.checkpoints
            .get(operator_id)
            .filter(|cp| cp.recorded)
            .map(CardinalityCheckpoint::deviation_ratio)
            .unwrap_or(1.0)
    }

    /// Returns summary statistics about the adaptive execution.
    #[must_use]
    pub fn summary(&self) -> AdaptiveSummary {
        let recorded_count = self.checkpoints.values().filter(|cp| cp.recorded).count();
        let deviation_count = self
            .checkpoints
            .values()
            .filter(|cp| cp.is_significant_deviation(self.reoptimization_threshold))
            .count();

        let avg_deviation = if recorded_count > 0 {
            self.checkpoints
                .values()
                .filter(|cp| cp.recorded)
                .map(CardinalityCheckpoint::deviation_ratio)
                .sum::<f64>()
                / recorded_count as f64
        } else {
            1.0
        };

        let max_deviation = self
            .checkpoints
            .values()
            .filter(|cp| cp.recorded)
            .map(|cp| {
                let ratio = cp.deviation_ratio();
                if ratio > 1.0 { ratio } else { 1.0 / ratio }
            })
            .fold(1.0_f64, f64::max);

        AdaptiveSummary {
            checkpoint_count: self.checkpoints.len(),
            recorded_count,
            deviation_count,
            avg_deviation_ratio: avg_deviation,
            max_deviation_ratio: max_deviation,
            reoptimization_triggered: self.reoptimization_triggered,
            trigger_operator: self.trigger_operator.clone(),
        }
    }

    /// Resets the context for a new execution.
    pub fn reset(&mut self) {
        for checkpoint in self.checkpoints.values_mut() {
            checkpoint.actual = 0;
            checkpoint.recorded = false;
        }
        self.reoptimization_triggered = false;
        self.trigger_operator = None;
    }
}

impl Default for AdaptiveContext {
    fn default() -> Self {
        Self::new()
    }
}

/// Summary of adaptive execution statistics.
#[derive(Debug, Clone, Default)]
pub struct AdaptiveSummary {
    /// Total number of checkpoints.
    pub checkpoint_count: usize,
    /// Number of checkpoints with recorded actuals.
    pub recorded_count: usize,
    /// Number of checkpoints with significant deviation.
    pub deviation_count: usize,
    /// Average deviation ratio across all recorded checkpoints.
    pub avg_deviation_ratio: f64,
    /// Maximum deviation ratio observed.
    pub max_deviation_ratio: f64,
    /// Whether re-optimization was triggered.
    pub reoptimization_triggered: bool,
    /// Operator that triggered re-optimization.
    pub trigger_operator: Option<String>,
}

/// Thread-safe wrapper for `AdaptiveContext`.
///
/// Allows multiple operators to report cardinalities concurrently.
#[derive(Debug, Clone)]
pub struct SharedAdaptiveContext {
    inner: Arc<RwLock<AdaptiveContext>>,
}

impl SharedAdaptiveContext {
    /// Creates a new shared context.
    #[must_use]
    pub fn new() -> Self {
        Self {
            inner: Arc::new(RwLock::new(AdaptiveContext::new())),
        }
    }

    /// Creates from an existing context.
    #[must_use]
    pub fn from_context(ctx: AdaptiveContext) -> Self {
        Self {
            inner: Arc::new(RwLock::new(ctx)),
        }
    }

    /// Records actual cardinality for an operator.
    pub fn record_actual(&self, operator_id: &str, actual: u64) {
        if let Ok(mut ctx) = self.inner.write() {
            ctx.record_actual(operator_id, actual);
        }
    }

    /// Checks if re-optimization should be triggered.
    #[must_use]
    pub fn should_reoptimize(&self) -> bool {
        if let Ok(mut ctx) = self.inner.write() {
            ctx.should_reoptimize()
        } else {
            false
        }
    }

    /// Gets a read-only snapshot of the context.
    #[must_use]
    pub fn snapshot(&self) -> Option<AdaptiveContext> {
        self.inner.read().ok().map(|guard| AdaptiveContext {
            checkpoints: guard.checkpoints.clone(),
            reoptimization_threshold: guard.reoptimization_threshold,
            min_rows: guard.min_rows,
            reoptimization_triggered: guard.reoptimization_triggered,
            trigger_operator: guard.trigger_operator.clone(),
        })
    }
}

impl Default for SharedAdaptiveContext {
    fn default() -> Self {
        Self::new()
    }
}

/// A wrapper operator that tracks cardinality and reports to an adaptive context.
///
/// This wraps any `PushOperator` and counts the rows flowing through it,
/// reporting the count to the adaptive context.
pub struct CardinalityTrackingOperator {
    /// The wrapped operator.
    inner: Box<dyn PushOperator>,
    /// Operator identifier for reporting.
    operator_id: String,
    /// Row counter.
    row_count: u64,
    /// Shared adaptive context for reporting.
    context: SharedAdaptiveContext,
}

impl CardinalityTrackingOperator {
    /// Creates a new tracking wrapper.
    pub fn new(
        inner: Box<dyn PushOperator>,
        operator_id: &str,
        context: SharedAdaptiveContext,
    ) -> Self {
        Self {
            inner,
            operator_id: operator_id.to_string(),
            row_count: 0,
            context,
        }
    }

    /// Returns the current row count.
    #[must_use]
    pub fn current_count(&self) -> u64 {
        self.row_count
    }
}

impl PushOperator for CardinalityTrackingOperator {
    fn push(&mut self, chunk: DataChunk, sink: &mut dyn Sink) -> Result<bool, OperatorError> {
        // Track input rows
        self.row_count += chunk.len() as u64;

        // Push through to inner operator
        self.inner.push(chunk, sink)
    }

    fn finalize(&mut self, sink: &mut dyn Sink) -> Result<(), OperatorError> {
        // Report final cardinality to context
        self.context
            .record_actual(&self.operator_id, self.row_count);

        // Finalize inner operator
        self.inner.finalize(sink)
    }

    fn preferred_chunk_size(&self) -> ChunkSizeHint {
        self.inner.preferred_chunk_size()
    }

    fn name(&self) -> &'static str {
        // Return the inner operator's name
        self.inner.name()
    }
}

/// A sink that tracks cardinality of data flowing through it.
pub struct CardinalityTrackingSink {
    /// The wrapped sink.
    inner: Box<dyn Sink>,
    /// Operator identifier for reporting.
    operator_id: String,
    /// Row counter.
    row_count: u64,
    /// Shared adaptive context for reporting.
    context: SharedAdaptiveContext,
}

impl CardinalityTrackingSink {
    /// Creates a new tracking sink wrapper.
    pub fn new(inner: Box<dyn Sink>, operator_id: &str, context: SharedAdaptiveContext) -> Self {
        Self {
            inner,
            operator_id: operator_id.to_string(),
            row_count: 0,
            context,
        }
    }

    /// Returns the current row count.
    #[must_use]
    pub fn current_count(&self) -> u64 {
        self.row_count
    }
}

impl Sink for CardinalityTrackingSink {
    fn consume(&mut self, chunk: DataChunk) -> Result<bool, OperatorError> {
        self.row_count += chunk.len() as u64;
        self.inner.consume(chunk)
    }

    fn finalize(&mut self) -> Result<(), OperatorError> {
        // Report final cardinality
        self.context
            .record_actual(&self.operator_id, self.row_count);
        self.inner.finalize()
    }

    fn name(&self) -> &'static str {
        self.inner.name()
    }
}

/// Decision about whether to re-optimize a query.
#[derive(Debug, Clone, PartialEq)]
pub enum ReoptimizationDecision {
    /// Continue with current plan.
    Continue,
    /// Re-optimize the remaining plan.
    Reoptimize {
        /// The operator that triggered re-optimization.
        trigger: String,
        /// Correction factors to apply to remaining estimates.
        corrections: HashMap<String, f64>,
    },
    /// Abort the query (catastrophic misestimate).
    Abort { reason: String },
}

/// Evaluates whether re-optimization should occur based on context.
#[must_use]
pub fn evaluate_reoptimization(ctx: &AdaptiveContext) -> ReoptimizationDecision {
    let summary = ctx.summary();

    // If no significant deviations, continue
    if !summary.reoptimization_triggered {
        return ReoptimizationDecision::Continue;
    }

    // If deviation is catastrophic (>100x), consider aborting
    if summary.max_deviation_ratio > 100.0 {
        return ReoptimizationDecision::Abort {
            reason: format!(
                "Catastrophic cardinality misestimate: {}x deviation",
                summary.max_deviation_ratio
            ),
        };
    }

    // Build correction factors
    let corrections: HashMap<String, f64> = ctx
        .all_checkpoints()
        .iter()
        .filter(|(_, cp)| cp.recorded)
        .map(|(id, cp)| (id.clone(), cp.deviation_ratio()))
        .collect();

    ReoptimizationDecision::Reoptimize {
        trigger: summary.trigger_operator.unwrap_or_default(),
        corrections,
    }
}

/// Callback for creating a new plan based on observed cardinalities.
///
/// This is called when the adaptive pipeline detects significant deviation
/// and decides to re-optimize. The callback receives the adaptive context
/// with recorded actuals and should return a new set of operators.
pub type PlanFactory = Box<dyn Fn(&AdaptiveContext) -> Vec<Box<dyn PushOperator>> + Send + Sync>;

/// Configuration for adaptive pipeline execution.
#[derive(Debug, Clone)]
pub struct AdaptivePipelineConfig {
    /// Number of rows to process before checking for re-optimization.
    pub check_interval: u64,
    /// Threshold for deviation that triggers re-optimization.
    pub reoptimization_threshold: f64,
    /// Minimum rows before considering re-optimization.
    pub min_rows_for_reoptimization: u64,
    /// Maximum number of re-optimizations allowed per query.
    pub max_reoptimizations: usize,
}

impl Default for AdaptivePipelineConfig {
    fn default() -> Self {
        Self {
            check_interval: 10_000,
            reoptimization_threshold: DEFAULT_REOPTIMIZATION_THRESHOLD,
            min_rows_for_reoptimization: MIN_ROWS_FOR_REOPTIMIZATION,
            max_reoptimizations: 3,
        }
    }
}

impl AdaptivePipelineConfig {
    /// Creates a new configuration with custom thresholds.
    #[must_use]
    pub fn new(check_interval: u64, threshold: f64, min_rows: u64) -> Self {
        Self {
            check_interval,
            reoptimization_threshold: threshold,
            min_rows_for_reoptimization: min_rows,
            max_reoptimizations: 3,
        }
    }

    /// Sets the maximum number of re-optimizations allowed.
    #[must_use]
    pub fn with_max_reoptimizations(mut self, max: usize) -> Self {
        self.max_reoptimizations = max;
        self
    }
}

/// Result of executing an adaptive pipeline.
#[derive(Debug, Clone)]
pub struct AdaptiveExecutionResult {
    /// Total rows processed.
    pub total_rows: u64,
    /// Number of re-optimizations performed.
    pub reoptimization_count: usize,
    /// Operators that triggered re-optimization.
    pub triggers: Vec<String>,
    /// Final adaptive context with all recorded actuals.
    pub final_context: AdaptiveSummary,
}

/// A checkpoint during adaptive execution where plan switching can occur.
///
/// Checkpoints are placed at strategic points in the pipeline (typically after
/// filters or joins) where switching to a new plan makes sense.
#[derive(Debug)]
pub struct AdaptiveCheckpoint {
    /// Unique identifier for this checkpoint.
    pub id: String,
    /// Operator index in the pipeline (after which this checkpoint occurs).
    pub after_operator: usize,
    /// Estimated cardinality at this point.
    pub estimated_cardinality: f64,
    /// Actual rows seen so far.
    pub actual_rows: u64,
    /// Whether this checkpoint has triggered re-optimization.
    pub triggered: bool,
}

impl AdaptiveCheckpoint {
    /// Creates a new checkpoint.
    #[must_use]
    pub fn new(id: &str, after_operator: usize, estimated: f64) -> Self {
        Self {
            id: id.to_string(),
            after_operator,
            estimated_cardinality: estimated,
            actual_rows: 0,
            triggered: false,
        }
    }

    /// Records rows passing through this checkpoint.
    pub fn record_rows(&mut self, count: u64) {
        self.actual_rows += count;
    }

    /// Checks if deviation exceeds threshold.
    #[must_use]
    pub fn exceeds_threshold(&self, threshold: f64, min_rows: u64) -> bool {
        if self.actual_rows < min_rows {
            return false;
        }
        if self.estimated_cardinality <= 0.0 {
            return self.actual_rows > 0;
        }
        let ratio = self.actual_rows as f64 / self.estimated_cardinality;
        ratio > threshold || ratio < 1.0 / threshold
    }
}

/// Event emitted during adaptive execution.
#[derive(Debug, Clone)]
pub enum AdaptiveEvent {
    /// A checkpoint was reached.
    CheckpointReached {
        id: String,
        actual_rows: u64,
        estimated: f64,
    },
    /// Re-optimization was triggered.
    ReoptimizationTriggered {
        checkpoint_id: String,
        deviation_ratio: f64,
    },
    /// Plan was switched.
    PlanSwitched {
        old_operator_count: usize,
        new_operator_count: usize,
    },
    /// Execution completed.
    ExecutionCompleted { total_rows: u64 },
}

/// Callback for observing adaptive execution events.
pub type AdaptiveEventCallback = Box<dyn Fn(AdaptiveEvent) + Send + Sync>;

/// Builder for creating adaptive pipelines.
pub struct AdaptivePipelineBuilder {
    checkpoints: Vec<AdaptiveCheckpoint>,
    config: AdaptivePipelineConfig,
    context: AdaptiveContext,
    event_callback: Option<AdaptiveEventCallback>,
}

impl AdaptivePipelineBuilder {
    /// Creates a new builder with default configuration.
    #[must_use]
    pub fn new() -> Self {
        Self {
            checkpoints: Vec::new(),
            config: AdaptivePipelineConfig::default(),
            context: AdaptiveContext::new(),
            event_callback: None,
        }
    }

    /// Sets the configuration.
    #[must_use]
    pub fn with_config(mut self, config: AdaptivePipelineConfig) -> Self {
        self.config = config;
        self
    }

    /// Adds a checkpoint at the specified operator index.
    #[must_use]
    pub fn with_checkpoint(mut self, id: &str, after_operator: usize, estimated: f64) -> Self {
        self.checkpoints
            .push(AdaptiveCheckpoint::new(id, after_operator, estimated));
        self.context.set_estimate(id, estimated);
        self
    }

    /// Sets an event callback for observing execution.
    #[must_use]
    pub fn with_event_callback(mut self, callback: AdaptiveEventCallback) -> Self {
        self.event_callback = Some(callback);
        self
    }

    /// Sets estimates from a pre-configured context.
    #[must_use]
    pub fn with_context(mut self, context: AdaptiveContext) -> Self {
        self.context = context;
        self
    }

    /// Builds the configuration for adaptive execution.
    #[must_use]
    pub fn build(self) -> AdaptiveExecutionConfig {
        AdaptiveExecutionConfig {
            checkpoints: self.checkpoints,
            config: self.config,
            context: self.context,
            event_callback: self.event_callback,
        }
    }
}

impl Default for AdaptivePipelineBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Configuration for adaptive execution, built by `AdaptivePipelineBuilder`.
pub struct AdaptiveExecutionConfig {
    /// Checkpoints for monitoring cardinality.
    pub checkpoints: Vec<AdaptiveCheckpoint>,
    /// Execution configuration.
    pub config: AdaptivePipelineConfig,
    /// Adaptive context with estimates.
    pub context: AdaptiveContext,
    /// Optional event callback.
    pub event_callback: Option<AdaptiveEventCallback>,
}

impl AdaptiveExecutionConfig {
    /// Returns a summary of the adaptive execution after it completes.
    #[must_use]
    pub fn summary(&self) -> AdaptiveSummary {
        self.context.summary()
    }

    /// Records actual cardinality for a checkpoint.
    pub fn record_checkpoint(&mut self, checkpoint_id: &str, actual: u64) {
        self.context.record_actual(checkpoint_id, actual);

        if let Some(cp) = self.checkpoints.iter_mut().find(|c| c.id == checkpoint_id) {
            cp.actual_rows = actual;
        }

        if let Some(ref callback) = self.event_callback {
            let estimated = self
                .context
                .get_checkpoint(checkpoint_id)
                .map(|cp| cp.estimated)
                .unwrap_or(0.0);
            callback(AdaptiveEvent::CheckpointReached {
                id: checkpoint_id.to_string(),
                actual_rows: actual,
                estimated,
            });
        }
    }

    /// Checks if any checkpoint exceeds the deviation threshold.
    #[must_use]
    pub fn should_reoptimize(&self) -> Option<&AdaptiveCheckpoint> {
        self.checkpoints.iter().find(|cp| {
            !cp.triggered
                && cp.exceeds_threshold(
                    self.config.reoptimization_threshold,
                    self.config.min_rows_for_reoptimization,
                )
        })
    }

    /// Marks a checkpoint as having triggered re-optimization.
    pub fn mark_triggered(&mut self, checkpoint_id: &str) {
        if let Some(cp) = self.checkpoints.iter_mut().find(|c| c.id == checkpoint_id) {
            cp.triggered = true;
        }

        if let Some(ref callback) = self.event_callback {
            let ratio = self
                .context
                .get_checkpoint(checkpoint_id)
                .filter(|cp| cp.recorded)
                .map(|cp| cp.deviation_ratio())
                .unwrap_or(1.0);
            callback(AdaptiveEvent::ReoptimizationTriggered {
                checkpoint_id: checkpoint_id.to_string(),
                deviation_ratio: ratio,
            });
        }
    }
}

// ============= Pull-Based Operator Tracking =============

use super::operators::{Operator, OperatorResult}; // OperatorError imported above

/// A wrapper that tracks cardinality for pull-based operators.
///
/// This wraps any `Operator` and counts the rows flowing through it,
/// reporting the count to the adaptive context. Use this for integrating
/// adaptive execution with the standard pull-based executor.
pub struct CardinalityTrackingWrapper {
    /// The wrapped operator.
    inner: Box<dyn Operator>,
    /// Operator identifier for reporting.
    operator_id: String,
    /// Row counter.
    row_count: u64,
    /// Shared adaptive context for reporting.
    context: SharedAdaptiveContext,
    /// Whether finalization has been reported.
    finalized: bool,
}

impl CardinalityTrackingWrapper {
    /// Creates a new tracking wrapper for a pull-based operator.
    pub fn new(
        inner: Box<dyn Operator>,
        operator_id: &str,
        context: SharedAdaptiveContext,
    ) -> Self {
        Self {
            inner,
            operator_id: operator_id.to_string(),
            row_count: 0,
            context,
            finalized: false,
        }
    }

    /// Returns the current row count.
    #[must_use]
    pub fn current_count(&self) -> u64 {
        self.row_count
    }

    /// Reports the final cardinality to the context.
    fn report_final(&mut self) {
        if !self.finalized {
            self.context
                .record_actual(&self.operator_id, self.row_count);
            self.finalized = true;
        }
    }
}

impl Operator for CardinalityTrackingWrapper {
    fn next(&mut self) -> OperatorResult {
        match self.inner.next() {
            Ok(Some(chunk)) => {
                // Track rows
                self.row_count += chunk.row_count() as u64;
                Ok(Some(chunk))
            }
            Ok(None) => {
                // Stream exhausted - report final cardinality
                self.report_final();
                Ok(None)
            }
            Err(e) => {
                // Report on error too
                self.report_final();
                Err(e)
            }
        }
    }

    fn reset(&mut self) {
        self.row_count = 0;
        self.finalized = false;
        self.inner.reset();
    }

    fn name(&self) -> &'static str {
        self.inner.name()
    }
}

impl Drop for CardinalityTrackingWrapper {
    fn drop(&mut self) {
        // Ensure we report even if dropped early
        self.report_final();
    }
}

// ============= Adaptive Pipeline Execution =============

use super::pipeline::{DEFAULT_CHUNK_SIZE, Source}; // Sink imported above
use super::sink::CollectorSink;
use super::source::OperatorSource;

/// High-level adaptive pipeline that executes a pull-based operator with
/// cardinality tracking using push-based infrastructure.
///
/// This bridges the pull-based planner output with push-based execution:
/// 1. Wraps the pull operator as an `OperatorSource`
/// 2. Uses `CardinalityTrackingSink` to track output cardinality
/// 3. Provides adaptive feedback through `AdaptiveContext`
///
/// # Example
///
/// ```ignore
/// use grafeo_core::execution::adaptive::AdaptivePipelineExecutor;
///
/// let executor = AdaptivePipelineExecutor::new(operator, adaptive_context);
/// let (chunks, summary) = executor.execute()?;
/// ```
pub struct AdaptivePipelineExecutor {
    source: OperatorSource,
    context: SharedAdaptiveContext,
    config: AdaptivePipelineConfig,
}

impl AdaptivePipelineExecutor {
    /// Creates a new adaptive pipeline executor.
    ///
    /// # Arguments
    ///
    /// * `operator` - The pull-based operator to execute
    /// * `context` - Adaptive context with cardinality estimates
    pub fn new(operator: Box<dyn Operator>, context: AdaptiveContext) -> Self {
        Self {
            source: OperatorSource::new(operator),
            context: SharedAdaptiveContext::from_context(context),
            config: AdaptivePipelineConfig::default(),
        }
    }

    /// Creates an executor with custom configuration.
    pub fn with_config(
        operator: Box<dyn Operator>,
        context: AdaptiveContext,
        config: AdaptivePipelineConfig,
    ) -> Self {
        Self {
            source: OperatorSource::new(operator),
            context: SharedAdaptiveContext::from_context(context),
            config,
        }
    }

    /// Executes the pipeline and returns collected chunks with adaptive summary.
    ///
    /// # Returns
    ///
    /// A tuple of (collected DataChunks, adaptive execution summary).
    ///
    /// # Errors
    ///
    /// Returns an error if execution fails.
    pub fn execute(mut self) -> Result<(Vec<DataChunk>, AdaptiveSummary), OperatorError> {
        let mut sink = CardinalityTrackingSink::new(
            Box::new(CollectorSink::new()),
            "output",
            self.context.clone(),
        );

        let chunk_size = DEFAULT_CHUNK_SIZE;
        let mut total_rows: u64 = 0;
        let check_interval = self.config.check_interval;

        // Process all chunks from source
        while let Some(chunk) = self.source.next_chunk(chunk_size)? {
            let chunk_rows = chunk.len() as u64;
            total_rows += chunk_rows;

            // Push to tracking sink
            let continue_exec = sink.consume(chunk)?;
            if !continue_exec {
                break;
            }

            // Periodically check for reoptimization need
            if total_rows >= check_interval && total_rows.is_multiple_of(check_interval) {
                if self.context.should_reoptimize() {
                    // Log or emit event that reoptimization would be triggered
                    // Full re-planning would happen at a higher level
                }
            }
        }

        // Finalize sink
        sink.finalize()?;

        // Extract results from the inner sink
        let summary = self
            .context
            .snapshot()
            .map(|ctx| ctx.summary())
            .unwrap_or_default();

        // Get collected chunks from the inner CollectorSink
        // Note: We need to extract chunks from the wrapped sink
        // For now, we'll return the summary and the caller can collect separately
        Ok((Vec::new(), summary))
    }

    /// Executes and collects all results into DataChunks.
    ///
    /// This is a simpler interface that handles chunk collection internally.
    pub fn execute_collecting(
        mut self,
    ) -> Result<(Vec<DataChunk>, AdaptiveSummary), OperatorError> {
        let mut chunks = Vec::new();
        let chunk_size = DEFAULT_CHUNK_SIZE;
        let mut total_rows: u64 = 0;
        let check_interval = self.config.check_interval;

        // Process all chunks from source
        while let Some(chunk) = self.source.next_chunk(chunk_size)? {
            let chunk_rows = chunk.len() as u64;
            total_rows += chunk_rows;

            // Record cardinality
            self.context.record_actual("root", total_rows);

            // Collect the chunk
            if !chunk.is_empty() {
                chunks.push(chunk);
            }

            // Periodically check for reoptimization
            if total_rows >= check_interval && total_rows.is_multiple_of(check_interval) {
                let _ = self.context.should_reoptimize();
            }
        }

        let summary = self
            .context
            .snapshot()
            .map(|ctx| ctx.summary())
            .unwrap_or_default();

        Ok((chunks, summary))
    }

    /// Returns a reference to the shared context for external monitoring.
    pub fn context(&self) -> &SharedAdaptiveContext {
        &self.context
    }
}

/// Convenience function to execute a pull-based operator with adaptive tracking.
///
/// This is the recommended entry point for adaptive execution.
///
/// # Arguments
///
/// * `operator` - The pull-based operator to execute
/// * `context` - Adaptive context with cardinality estimates (or None for default)
/// * `config` - Optional configuration (uses defaults if None)
///
/// # Returns
///
/// A tuple of (collected DataChunks, adaptive summary if tracking was enabled).
pub fn execute_adaptive(
    operator: Box<dyn Operator>,
    context: Option<AdaptiveContext>,
    config: Option<AdaptivePipelineConfig>,
) -> Result<(Vec<DataChunk>, Option<AdaptiveSummary>), OperatorError> {
    let ctx = context.unwrap_or_default();
    let cfg = config.unwrap_or_default();

    let executor = AdaptivePipelineExecutor::with_config(operator, ctx, cfg);
    let (chunks, summary) = executor.execute_collecting()?;

    Ok((chunks, Some(summary)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_checkpoint_deviation_ratio() {
        let mut cp = CardinalityCheckpoint::new("test", 100.0);
        cp.record(200);

        // Actual is 2x estimate
        assert!((cp.deviation_ratio() - 2.0).abs() < 0.001);
    }

    #[test]
    fn test_checkpoint_underestimate() {
        let mut cp = CardinalityCheckpoint::new("test", 100.0);
        cp.record(500);

        // Underestimated by 5x
        assert!((cp.deviation_ratio() - 5.0).abs() < 0.001);
        assert!(cp.is_significant_deviation(3.0));
    }

    #[test]
    fn test_checkpoint_overestimate() {
        let mut cp = CardinalityCheckpoint::new("test", 100.0);
        cp.record(20);

        // Overestimated - actual is 0.2x estimate
        assert!((cp.deviation_ratio() - 0.2).abs() < 0.001);
        assert!(cp.is_significant_deviation(3.0)); // 0.2 < 1/3
    }

    #[test]
    fn test_checkpoint_accurate() {
        let mut cp = CardinalityCheckpoint::new("test", 100.0);
        cp.record(110);

        // Close to estimate
        assert!((cp.deviation_ratio() - 1.1).abs() < 0.001);
        assert!(!cp.is_significant_deviation(3.0)); // 1.1 is within threshold
    }

    #[test]
    fn test_checkpoint_zero_estimate() {
        let mut cp = CardinalityCheckpoint::new("test", 0.0);
        cp.record(100);

        // Zero estimate with actual rows = infinity
        assert!(cp.deviation_ratio().is_infinite());
    }

    #[test]
    fn test_checkpoint_zero_both() {
        let mut cp = CardinalityCheckpoint::new("test", 0.0);
        cp.record(0);

        // Zero estimate and zero actual = ratio of 1.0
        assert!((cp.deviation_ratio() - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_feedback_collection() {
        let mut feedback = CardinalityFeedback::new();
        feedback.record("scan_1", 1000);
        feedback.record("filter_1", 100);

        assert_eq!(feedback.get("scan_1"), Some(1000));
        assert_eq!(feedback.get("filter_1"), Some(100));
        assert_eq!(feedback.get("unknown"), None);
    }

    #[test]
    fn test_feedback_running_counter() {
        let mut feedback = CardinalityFeedback::new();
        feedback.init_counter("op_1");

        feedback.add_rows("op_1", 100);
        feedback.add_rows("op_1", 200);
        feedback.add_rows("op_1", 50);

        assert_eq!(feedback.get_running("op_1"), Some(350));

        feedback.finalize_counter("op_1");
        assert_eq!(feedback.get("op_1"), Some(350));
    }

    #[test]
    fn test_adaptive_context_basic() {
        let mut ctx = AdaptiveContext::new();
        ctx.set_estimate("scan", 1000.0);
        ctx.set_estimate("filter", 100.0);

        ctx.record_actual("scan", 1000);
        ctx.record_actual("filter", 500); // 5x underestimate

        let cp = ctx.get_checkpoint("filter").unwrap();
        assert!((cp.deviation_ratio() - 5.0).abs() < 0.001);
    }

    #[test]
    fn test_adaptive_context_should_reoptimize() {
        let mut ctx = AdaptiveContext::with_thresholds(2.0, 100);
        ctx.set_estimate("scan", 10000.0);
        ctx.set_estimate("filter", 1000.0);

        ctx.record_actual("scan", 10000);
        ctx.record_actual("filter", 5000); // 5x underestimate

        assert!(ctx.should_reoptimize());
        assert_eq!(ctx.trigger_operator(), Some("filter"));

        // Second call should return false (already triggered)
        assert!(!ctx.should_reoptimize());
    }

    #[test]
    fn test_adaptive_context_min_rows() {
        let mut ctx = AdaptiveContext::with_thresholds(2.0, 1000);
        ctx.set_estimate("filter", 100.0);
        ctx.record_actual("filter", 500); // 5x, but only 500 rows

        // Should not trigger because we haven't seen enough rows
        assert!(!ctx.should_reoptimize());
    }

    #[test]
    fn test_adaptive_context_no_deviation() {
        let mut ctx = AdaptiveContext::new();
        ctx.set_estimate("scan", 1000.0);
        ctx.record_actual("scan", 1100); // Close to estimate

        assert!(!ctx.has_significant_deviation());
        assert!(!ctx.should_reoptimize());
    }

    #[test]
    fn test_adaptive_context_correction_factor() {
        let mut ctx = AdaptiveContext::new();
        ctx.set_estimate("filter", 100.0);
        ctx.record_actual("filter", 300);

        assert!((ctx.correction_factor("filter") - 3.0).abs() < 0.001);
        assert!((ctx.correction_factor("unknown") - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_adaptive_context_apply_feedback() {
        let mut ctx = AdaptiveContext::new();
        ctx.set_estimate("scan", 1000.0);
        ctx.set_estimate("filter", 100.0);

        let mut feedback = CardinalityFeedback::new();
        feedback.record("scan", 1000);
        feedback.record("filter", 500);

        ctx.apply_feedback(&feedback);

        assert_eq!(ctx.get_checkpoint("scan").unwrap().actual, 1000);
        assert_eq!(ctx.get_checkpoint("filter").unwrap().actual, 500);
    }

    #[test]
    fn test_adaptive_summary() {
        let mut ctx = AdaptiveContext::with_thresholds(2.0, 0);
        ctx.set_estimate("op1", 100.0);
        ctx.set_estimate("op2", 200.0);
        ctx.set_estimate("op3", 300.0);

        ctx.record_actual("op1", 100); // Exact
        ctx.record_actual("op2", 600); // 3x

        // Trigger reoptimization
        let _ = ctx.should_reoptimize();

        let summary = ctx.summary();
        assert_eq!(summary.checkpoint_count, 3);
        assert_eq!(summary.recorded_count, 2);
        assert_eq!(summary.deviation_count, 1);
        assert!(summary.reoptimization_triggered);
    }

    #[test]
    fn test_adaptive_context_reset() {
        let mut ctx = AdaptiveContext::new();
        ctx.set_estimate("scan", 1000.0);
        ctx.record_actual("scan", 5000);
        let _ = ctx.should_reoptimize(); // Trigger

        assert!(ctx.reoptimization_triggered);

        ctx.reset();

        assert!(!ctx.reoptimization_triggered);
        assert_eq!(ctx.get_checkpoint("scan").unwrap().actual, 0);
        assert!(!ctx.get_checkpoint("scan").unwrap().recorded);
        // Estimate should be preserved
        assert!((ctx.get_checkpoint("scan").unwrap().estimated - 1000.0).abs() < 0.001);
    }

    #[test]
    fn test_shared_context() {
        let ctx = SharedAdaptiveContext::new();

        ctx.record_actual("op1", 1000);

        let snapshot = ctx.snapshot().unwrap();
        assert_eq!(snapshot.get_checkpoint("op1").unwrap().actual, 1000);
    }

    #[test]
    fn test_reoptimization_decision_continue() {
        let mut ctx = AdaptiveContext::new();
        ctx.set_estimate("scan", 1000.0);
        ctx.record_actual("scan", 1100);

        let decision = evaluate_reoptimization(&ctx);
        assert_eq!(decision, ReoptimizationDecision::Continue);
    }

    #[test]
    fn test_reoptimization_decision_reoptimize() {
        let mut ctx = AdaptiveContext::with_thresholds(2.0, 0);
        ctx.set_estimate("filter", 100.0);
        ctx.record_actual("filter", 500);
        let _ = ctx.should_reoptimize(); // Trigger

        let decision = evaluate_reoptimization(&ctx);

        if let ReoptimizationDecision::Reoptimize {
            trigger,
            corrections,
        } = decision
        {
            assert_eq!(trigger, "filter");
            assert!((corrections.get("filter").copied().unwrap_or(0.0) - 5.0).abs() < 0.001);
        } else {
            panic!("Expected Reoptimize decision");
        }
    }

    #[test]
    fn test_reoptimization_decision_abort() {
        let mut ctx = AdaptiveContext::with_thresholds(2.0, 0);
        ctx.set_estimate("filter", 1.0);
        ctx.record_actual("filter", 1000); // 1000x deviation!
        let _ = ctx.should_reoptimize();

        let decision = evaluate_reoptimization(&ctx);

        if let ReoptimizationDecision::Abort { reason } = decision {
            assert!(reason.contains("Catastrophic"));
        } else {
            panic!("Expected Abort decision");
        }
    }

    #[test]
    fn test_absolute_deviation() {
        let mut cp = CardinalityCheckpoint::new("test", 100.0);
        cp.record(150);

        assert!((cp.absolute_deviation() - 50.0).abs() < 0.001);
    }

    // ============= Plan Switching Tests =============

    #[test]
    fn test_adaptive_checkpoint_basic() {
        let mut cp = AdaptiveCheckpoint::new("filter_1", 0, 100.0);
        assert_eq!(cp.actual_rows, 0);
        assert!(!cp.triggered);

        cp.record_rows(50);
        assert_eq!(cp.actual_rows, 50);

        cp.record_rows(100);
        assert_eq!(cp.actual_rows, 150);
    }

    #[test]
    fn test_adaptive_checkpoint_exceeds_threshold() {
        let mut cp = AdaptiveCheckpoint::new("filter", 0, 100.0);

        // Below min rows
        cp.record_rows(50);
        assert!(!cp.exceeds_threshold(2.0, 100));

        // Above min rows but within threshold
        cp.record_rows(50);
        assert!(!cp.exceeds_threshold(2.0, 100)); // 100 actual vs 100 estimated = 1.0x

        // Above threshold (underestimate)
        cp.actual_rows = 0;
        cp.record_rows(500);
        assert!(cp.exceeds_threshold(2.0, 100)); // 500 actual vs 100 estimated = 5.0x

        // Above threshold (overestimate)
        let mut cp2 = AdaptiveCheckpoint::new("filter2", 0, 1000.0);
        cp2.record_rows(200);
        assert!(cp2.exceeds_threshold(2.0, 100)); // 200 actual vs 1000 estimated = 0.2x
    }

    #[test]
    fn test_adaptive_pipeline_config_default() {
        let config = AdaptivePipelineConfig::default();

        assert_eq!(config.check_interval, 10_000);
        assert!((config.reoptimization_threshold - DEFAULT_REOPTIMIZATION_THRESHOLD).abs() < 0.001);
        assert_eq!(
            config.min_rows_for_reoptimization,
            MIN_ROWS_FOR_REOPTIMIZATION
        );
        assert_eq!(config.max_reoptimizations, 3);
    }

    #[test]
    fn test_adaptive_pipeline_config_custom() {
        let config = AdaptivePipelineConfig::new(5000, 2.0, 500).with_max_reoptimizations(5);

        assert_eq!(config.check_interval, 5000);
        assert!((config.reoptimization_threshold - 2.0).abs() < 0.001);
        assert_eq!(config.min_rows_for_reoptimization, 500);
        assert_eq!(config.max_reoptimizations, 5);
    }

    #[test]
    fn test_adaptive_pipeline_builder() {
        let config = AdaptivePipelineBuilder::new()
            .with_config(AdaptivePipelineConfig::new(1000, 2.0, 100))
            .with_checkpoint("scan", 0, 10000.0)
            .with_checkpoint("filter", 1, 1000.0)
            .build();

        assert_eq!(config.checkpoints.len(), 2);
        assert_eq!(config.checkpoints[0].id, "scan");
        assert!((config.checkpoints[0].estimated_cardinality - 10000.0).abs() < 0.001);
        assert_eq!(config.checkpoints[1].id, "filter");
        assert!((config.checkpoints[1].estimated_cardinality - 1000.0).abs() < 0.001);
    }

    #[test]
    fn test_adaptive_execution_config_record_checkpoint() {
        let mut config = AdaptivePipelineBuilder::new()
            .with_checkpoint("filter", 0, 100.0)
            .build();

        config.record_checkpoint("filter", 500);

        // Check context was updated
        let cp = config.context.get_checkpoint("filter").unwrap();
        assert_eq!(cp.actual, 500);
        assert!(cp.recorded);

        // Check checkpoint was updated
        let acp = config
            .checkpoints
            .iter()
            .find(|c| c.id == "filter")
            .unwrap();
        assert_eq!(acp.actual_rows, 500);
    }

    #[test]
    fn test_adaptive_execution_config_should_reoptimize() {
        let mut config = AdaptivePipelineBuilder::new()
            .with_config(AdaptivePipelineConfig::new(1000, 2.0, 100))
            .with_checkpoint("filter", 0, 100.0)
            .build();

        // No data yet - should not trigger
        assert!(config.should_reoptimize().is_none());

        // Record within threshold
        config.record_checkpoint("filter", 150);
        assert!(config.should_reoptimize().is_none()); // 1.5x is within 2.0x threshold

        // Record exceeding threshold
        config.checkpoints[0].actual_rows = 0; // Reset for new test
        config.record_checkpoint("filter", 500);
        config.checkpoints[0].actual_rows = 500;

        let trigger = config.should_reoptimize();
        assert!(trigger.is_some());
        assert_eq!(trigger.unwrap().id, "filter");
    }

    #[test]
    fn test_adaptive_execution_config_mark_triggered() {
        let mut config = AdaptivePipelineBuilder::new()
            .with_checkpoint("filter", 0, 100.0)
            .build();

        assert!(!config.checkpoints[0].triggered);

        config.mark_triggered("filter");

        assert!(config.checkpoints[0].triggered);
    }

    #[test]
    fn test_adaptive_event_callback() {
        use std::sync::atomic::AtomicUsize;

        let event_count = Arc::new(AtomicUsize::new(0));
        let counter = event_count.clone();

        let mut config = AdaptivePipelineBuilder::new()
            .with_checkpoint("filter", 0, 100.0)
            .with_event_callback(Box::new(move |_event| {
                counter.fetch_add(1, Ordering::Relaxed);
            }))
            .build();

        config.record_checkpoint("filter", 500);

        // Should have received one CheckpointReached event
        assert_eq!(event_count.load(Ordering::Relaxed), 1);

        config.mark_triggered("filter");

        // Should have received one ReoptimizationTriggered event
        assert_eq!(event_count.load(Ordering::Relaxed), 2);
    }

    #[test]
    fn test_adaptive_checkpoint_with_zero_estimate() {
        let mut cp = AdaptiveCheckpoint::new("test", 0, 0.0);
        cp.record_rows(100);

        // Zero estimate should trigger if any rows are seen
        assert!(cp.exceeds_threshold(2.0, 50));
    }
}
