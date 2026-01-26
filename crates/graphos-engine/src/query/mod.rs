//! Query processing pipeline.

pub mod binder;
pub mod executor;
pub mod optimizer;
pub mod planner;
pub mod processor;

pub use processor::QueryProcessor;
