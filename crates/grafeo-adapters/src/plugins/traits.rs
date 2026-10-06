//! Plugin traits.

use grafeo_common::utils::error::Result;
use std::collections::HashMap;

/// A Grafeo plugin.
pub trait Plugin: Send + Sync {
    /// Returns the name of the plugin.
    fn name(&self) -> &str;

    /// Returns the version of the plugin.
    fn version(&self) -> &str;

    /// Called when the plugin is loaded.
    ///
    /// # Errors
    ///
    /// Returns an error if the plugin fails to initialize.
    fn on_load(&self) -> Result<()> {
        Ok(())
    }

    /// Called when the plugin is unloaded.
    ///
    /// # Errors
    ///
    /// Returns an error if the plugin fails to clean up.
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
    ///
    /// # Errors
    ///
    /// Returns an error if the algorithm fails (e.g., invalid parameters).
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
    /// List of arbitrary values (for vector queries, multi-value filters, etc.).
    List,
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

    /// Sets a list parameter.
    pub fn set_list(&mut self, name: impl Into<String>, value: Vec<grafeo_common::types::Value>) {
        self.values.insert(name.into(), ParameterValue::List(value));
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

    /// Gets a list parameter.
    pub fn get_list(&self, name: &str) -> Option<&[grafeo_common::types::Value]> {
        match self.values.get(name) {
            Some(ParameterValue::List(v)) => Some(v),
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
    List(Vec<grafeo_common::types::Value>),
}

/// Result of an algorithm execution.
pub struct AlgorithmResult {
    /// Result columns.
    pub columns: Vec<String>,
    /// Result rows.
    pub rows: Vec<Vec<grafeo_common::types::Value>>,
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
    pub fn add_row(&mut self, row: Vec<grafeo_common::types::Value>) {
        self.rows.push(row);
    }

    /// Returns the number of rows.
    pub fn row_count(&self) -> usize {
        self.rows.len()
    }

    /// Puts the rows in the order of their first `columns` integer columns
    /// (a node id, or the source and target of an edge), and rows with the
    /// same ids in the order of all their values, as `ORDER BY` orders them. A
    /// result built by iterating a hash map, whose order changes from call to
    /// call, then comes out the same every time, also with several rows per
    /// node or edge.
    pub fn sort_by_id_columns(&mut self, columns: usize) {
        use grafeo_common::types::Value;
        use grafeo_core::execution::operators::value_utils::compare_values_total;

        let id = |row: &[Value], column: usize| match row.get(column) {
            Some(Value::Int64(id)) => *id,
            _ => i64::MIN,
        };
        self.rows.sort_by(|a, b| {
            (0..columns)
                .map(|column| id(a, column).cmp(&id(b, column)))
                .find(|order| order.is_ne())
                .unwrap_or_else(|| {
                    a.iter()
                        .zip(b)
                        .map(|(a, b)| compare_values_total(a, b))
                        .find(|order| order.is_ne())
                        .unwrap_or_else(|| a.len().cmp(&b.len()))
                })
        });
    }
}

#[cfg(test)]
mod tests {
    use grafeo_common::types::Value;
    use grafeo_common::utils::hash::FxHashSet;

    use super::AlgorithmResult;

    /// Rows with several per id, added in the order of a hash set (which
    /// changes from set to set), then sorted.
    fn sorted_rows_from_a_hash_set() -> Vec<Vec<Value>> {
        let pairs: FxHashSet<(i64, u32)> = [3_i64, 19, 88]
            .into_iter()
            .flat_map(|id| (0..19).map(move |weight| (id, weight)))
            .collect();
        let mut result = AlgorithmResult::new(vec!["node_id".to_string(), "weight".to_string()]);
        for (id, weight) in pairs {
            result.add_row(vec![Value::Int64(id), Value::Float64(f64::from(weight))]);
        }
        result.sort_by_id_columns(1);
        result.rows
    }

    /// Rows with the same id are ordered by their other columns, so a result
    /// with several rows per node or edge comes out the same every time too
    /// (#592).
    #[test]
    fn rows_with_the_same_id_are_ordered_by_their_other_columns() {
        let first = sorted_rows_from_a_hash_set();
        let pairs: Vec<(i64, f64)> = first
            .iter()
            .map(|row| match (&row[0], &row[1]) {
                (Value::Int64(id), Value::Float64(weight)) => (*id, *weight),
                other => panic!("unexpected row {other:?}"),
            })
            .collect();
        assert!(
            pairs.windows(2).all(|w| w[0] < w[1]),
            "not ordered by id, then weight: {:?}",
            &pairs[..pairs.len().min(19)]
        );
        for _ in 0..3 {
            assert_eq!(sorted_rows_from_a_hash_set(), first);
        }
    }
}
