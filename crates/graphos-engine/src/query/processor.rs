//! Query processor that orchestrates the query pipeline.

use crate::database::QueryResult;
use graphos_common::utils::error::Result;

/// Processes queries through the full pipeline.
pub struct QueryProcessor {
    // TODO: Add references to catalog, optimizer, etc.
}

impl QueryProcessor {
    /// Creates a new query processor.
    #[must_use]
    pub fn new() -> Self {
        Self {}
    }

    /// Processes a query string and returns results.
    ///
    /// Pipeline:
    /// 1. Parse (AST)
    /// 2. Bind (semantic analysis)
    /// 3. Plan (logical plan)
    /// 4. Optimize (optimized logical plan)
    /// 5. Execute (physical execution)
    ///
    /// # Errors
    ///
    /// Returns an error if any stage of the pipeline fails.
    pub fn process(&self, _query: &str) -> Result<QueryResult> {
        // TODO: Implement full pipeline
        Ok(QueryResult::new(vec![]))
    }
}

impl Default for QueryProcessor {
    fn default() -> Self {
        Self::new()
    }
}
