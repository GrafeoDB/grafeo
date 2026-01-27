//! Plugin traits.

use graphos_common::utils::error::Result;
use std::collections::HashMap;

/// A Graphos plugin.
pub trait Plugin: Send + Sync {
    /// Returns the name of the plugin.
    fn name(&self) -> &str;

    /// Returns the version of the plugin.
    fn version(&self) -> &str;

    /// Called when the plugin is loaded.
    fn on_load(&self) -> Result<()> {
        Ok(())
    }

    /// Called when the plugin is unloaded.
    fn on_unload(&self) -> Result<()> {
        Ok(())
    }
}

/// A graph algorithm that can be invoked from queries.
pub trait Algorithm: Send + Sync {
    /// Returns the name of the algorithm.
    fn name(&self) -> &str;

    /// Returns a description of the algorithm.
    fn description(&self) -> &str;

    /// Returns the parameter definitions.
    fn parameters(&self) -> &[ParameterDef];

    /// Executes the algorithm.
    fn execute(&self, params: &Parameters) -> Result<AlgorithmResult>;
}

/// Definition of an algorithm parameter.
#[derive(Debug, Clone)]
pub struct ParameterDef {
    /// Parameter name.
    pub name: String,
    /// Parameter description.
    pub description: String,
    /// Parameter type.
    pub param_type: ParameterType,
    /// Whether the parameter is required.
    pub required: bool,
    /// Default value (if any).
    pub default: Option<String>,
}

/// Types of algorithm parameters.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParameterType {
    /// Integer parameter.
    Integer,
    /// Float parameter.
    Float,
    /// String parameter.
    String,
    /// Boolean parameter.
    Boolean,
    /// Node ID parameter.
    NodeId,
}

/// Parameters passed to an algorithm.
pub struct Parameters {
    /// Parameter values.
    values: HashMap<String, ParameterValue>,
}

impl Parameters {
    /// Creates a new empty parameter set.
    pub fn new() -> Self {
        Self {
            values: HashMap::new(),
        }
    }

    /// Sets an integer parameter.
    pub fn set_int(&mut self, name: impl Into<String>, value: i64) {
        self.values
            .insert(name.into(), ParameterValue::Integer(value));
    }

    /// Sets a float parameter.
    pub fn set_float(&mut self, name: impl Into<String>, value: f64) {
        self.values
            .insert(name.into(), ParameterValue::Float(value));
    }

    /// Sets a string parameter.
    pub fn set_string(&mut self, name: impl Into<String>, value: impl Into<String>) {
        self.values
            .insert(name.into(), ParameterValue::String(value.into()));
    }

    /// Sets a boolean parameter.
    pub fn set_bool(&mut self, name: impl Into<String>, value: bool) {
        self.values
            .insert(name.into(), ParameterValue::Boolean(value));
    }

    /// Gets an integer parameter.
    pub fn get_int(&self, name: &str) -> Option<i64> {
        match self.values.get(name) {
            Some(ParameterValue::Integer(v)) => Some(*v),
            _ => None,
        }
    }

    /// Gets a float parameter.
    pub fn get_float(&self, name: &str) -> Option<f64> {
        match self.values.get(name) {
            Some(ParameterValue::Float(v)) => Some(*v),
            _ => None,
        }
    }

    /// Gets a string parameter.
    pub fn get_string(&self, name: &str) -> Option<&str> {
        match self.values.get(name) {
            Some(ParameterValue::String(v)) => Some(v),
            _ => None,
        }
    }

    /// Gets a boolean parameter.
    pub fn get_bool(&self, name: &str) -> Option<bool> {
        match self.values.get(name) {
            Some(ParameterValue::Boolean(v)) => Some(*v),
            _ => None,
        }
    }
}

impl Default for Parameters {
    fn default() -> Self {
        Self::new()
    }
}

/// A parameter value.
#[derive(Debug, Clone)]
enum ParameterValue {
    Integer(i64),
    Float(f64),
    String(String),
    Boolean(bool),
}

/// Result of an algorithm execution.
pub struct AlgorithmResult {
    /// Result columns.
    pub columns: Vec<String>,
    /// Result rows.
    pub rows: Vec<Vec<graphos_common::types::Value>>,
}

impl AlgorithmResult {
    /// Creates a new empty result.
    pub fn new(columns: Vec<String>) -> Self {
        Self {
            columns,
            rows: Vec::new(),
        }
    }

    /// Adds a row to the result.
    pub fn add_row(&mut self, row: Vec<graphos_common::types::Value>) {
        self.rows.push(row);
    }

    /// Returns the number of rows.
    pub fn row_count(&self) -> usize {
        self.rows.len()
    }
}
