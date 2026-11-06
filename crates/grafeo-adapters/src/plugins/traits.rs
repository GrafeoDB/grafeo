//! Plugin traits.

use grafeo_common::utils::error::{Error, QueryError, QueryErrorKind, Result};
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
    /// The values a numeric parameter allows beyond its type, which every call is checked
    /// against (see [`check_parameters`]); [`Domain::OfType`] for a string, a boolean, a node id
    /// or a list.
    pub domain: Domain,
}

/// The values a number a procedure reads may take, beyond its type: a numeric argument, or an
/// edge property it reads as a weight, a capacity or a cost. A float must also be finite. A value
/// outside is refused with "invalid parameter value" (22023) before it is used.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Domain {
    /// Nothing beyond its type: a string, a boolean, a node id or a list, which the procedure
    /// reads itself; for a number, any finite one.
    OfType,
    /// Zero or more.
    NonNegative,
    /// More than zero.
    Positive,
    /// From 0 to 1, both included.
    Unit,
    /// At least the value of the parameter named, as the call gives it or by its default.
    AtLeast(&'static str),
}

impl Domain {
    /// Whether `value` lies inside, with `floor` the value of the parameter an
    /// [`AtLeast`](Self::AtLeast) names (`None`: not known, so no bound).
    #[must_use]
    pub fn admits(self, value: f64, floor: Option<f64>) -> bool {
        value.is_finite()
            && match self {
                Self::OfType => true,
                Self::NonNegative => value >= 0.0,
                Self::Positive => value > 0.0,
                Self::Unit => (0.0..=1.0).contains(&value),
                Self::AtLeast(_) => floor.is_none_or(|floor| value >= floor),
            }
    }

    /// The values it allows, for an error, with `floor` as in [`admits`](Self::admits).
    #[must_use]
    pub fn describe(self, floor: Option<f64>) -> String {
        match self {
            Self::OfType => "a finite number".to_string(),
            Self::NonNegative => "a finite number of 0 or more".to_string(),
            Self::Positive => "a finite number above 0".to_string(),
            Self::Unit => "a number from 0 to 1".to_string(),
            Self::AtLeast(other) => match floor {
                Some(floor) => format!("a finite number of at least '{other}' ({floor})"),
                None => format!("a finite number of at least '{other}'"),
            },
        }
    }
}

/// Refuses an argument of `params` outside the domain its definition in `defs` declares, with
/// "invalid parameter value" (22023) naming `procedure`, the parameter, the values it allows and
/// the value. The one check of every way into a procedure (`CALL` and the bindings' algorithm
/// methods), run once all the arguments are set, so a domain that names another parameter
/// (`fetch_k` at least `k`) reads that one as given, or else its default. A parameter left out
/// keeps its default, which lies inside.
///
/// # Errors
///
/// The invalid parameter value of the first argument outside its domain.
pub fn check_parameters(procedure: &str, defs: &[ParameterDef], params: &Parameters) -> Result<()> {
    for def in defs {
        let Some((value, given)) = params.number(&def.name) else {
            continue;
        };
        let floor = match def.domain {
            Domain::AtLeast(other) => params.number(other).map(|(floor, _)| floor).or_else(|| {
                defs.iter()
                    .find(|def| def.name == other)
                    .and_then(|def| def.default.as_deref())
                    .and_then(|default| default.parse().ok())
            }),
            _ => None,
        };
        if !def.domain.admits(value, floor) {
            return Err(Error::from(QueryError::new(
                QueryErrorKind::InvalidParameterValue,
                format!(
                    "argument '{}' of {procedure} must be {}, got {given}",
                    def.name,
                    def.domain.describe(floor)
                ),
            )));
        }
    }
    Ok(())
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

    /// An integer parameter that counts (iterations, a `k`, a threshold) as a `usize`; `None`
    /// when the call leaves it out.
    ///
    /// # Errors
    ///
    /// An invalid parameter value (22023) for a negative one, which only a caller that skips
    /// [`check_parameters`] can give.
    pub fn get_count(&self, name: &str) -> Result<Option<usize>> {
        match self.values.get(name) {
            Some(ParameterValue::Integer(v)) => usize::try_from(*v).map(Some).map_err(|_| {
                Error::from(QueryError::new(
                    QueryErrorKind::InvalidParameterValue,
                    format!("argument '{name}' must be a finite number of 0 or more, got {v}"),
                ))
            }),
            _ => Ok(None),
        }
    }

    /// A numeric parameter as a float, with its text as the call gave it (`-1`, `NaN`).
    fn number(&self, name: &str) -> Option<(f64, String)> {
        match self.values.get(name) {
            Some(ParameterValue::Integer(v)) => Some((*v as f64, v.to_string())),
            Some(ParameterValue::Float(v)) => Some((*v, v.to_string())),
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
