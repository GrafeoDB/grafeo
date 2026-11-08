//! Plugin traits.

use grafeo_common::utils::error::{Error, QueryError, QueryErrorKind, Result};
use grafeo_core::execution::operators::value_utils::compare_values;
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
    /// The values the parameter allows beyond its type, which every call is checked against
    /// (see [`prepare`]): a numeric range, a string's length or choices, or a list that takes
    /// one value as a list of one; [`Domain::OfType`] for nothing beyond the type.
    pub domain: Domain,
}

/// The values an argument may take, beyond its type: a numeric argument, or an edge property a
/// procedure reads as a weight, a capacity or a cost (a float must also be finite), a string of one
/// character or of a few choices, a list that also takes one value. A value outside is refused
/// with "invalid parameter value" (22023) before it is used.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Domain {
    /// Nothing beyond its type: a string, a boolean, a node id, a list or a map, which the
    /// procedure reads itself; for a number, any finite one.
    OfType,
    /// Zero or more.
    NonNegative,
    /// More than zero.
    Positive,
    /// From 0 to 1, both included.
    Unit,
    /// At least the value of the parameter named, as the call gives it or by its default.
    AtLeast(&'static str),
    /// A string of one ASCII character, such as a field delimiter.
    OneCharacter,
    /// A string that is one of these, such as `'error'` or `'skip'`.
    OneOf(&'static [&'static str]),
    /// A list that also takes one value, as a list of one: [`prepare`] makes the label `'Person'`
    /// the list `['Person']`. Declared on the parameter that reads it so, never taken for granted:
    /// one string where a vector is wanted stays a type error.
    ListOrOne,
}

impl Domain {
    /// Whether the number `value` lies inside, apart from the floor of an
    /// [`AtLeast`](Self::AtLeast): [`prepare`] compares a value with the
    /// parameter it names exactly, an integer as an integer. A domain of
    /// strings or lists bounds no number.
    #[must_use]
    pub fn admits(self, value: f64) -> bool {
        value.is_finite()
            && match self {
                Self::OfType
                | Self::AtLeast(_)
                | Self::OneCharacter
                | Self::OneOf(_)
                | Self::ListOrOne => true,
                Self::NonNegative => value >= 0.0,
                Self::Positive => value > 0.0,
                Self::Unit => (0.0..=1.0).contains(&value),
            }
    }

    /// Whether the string `value` lies inside: one ASCII character for
    /// [`OneCharacter`](Self::OneCharacter), one of the choices for
    /// [`OneOf`](Self::OneOf); any string for the other domains.
    #[must_use]
    pub fn admits_text(self, value: &str) -> bool {
        match self {
            Self::OneCharacter => value.len() == 1 && value.is_ascii(),
            Self::OneOf(choices) => choices.contains(&value),
            Self::OfType
            | Self::NonNegative
            | Self::Positive
            | Self::Unit
            | Self::AtLeast(_)
            | Self::ListOrOne => true,
        }
    }

    /// The values it allows, for an error, with `floor` the value of the
    /// parameter an [`AtLeast`](Self::AtLeast) names, as the call gave it
    /// (`None`: not known).
    #[must_use]
    pub fn describe(self, floor: Option<&str>) -> String {
        match self {
            Self::OfType => "a finite number".to_string(),
            Self::NonNegative => "a finite number of 0 or more".to_string(),
            Self::Positive => "a finite number above 0".to_string(),
            Self::Unit => "a number from 0 to 1".to_string(),
            Self::AtLeast(other) => match floor {
                Some(floor) => format!("a finite number of at least '{other}' ({floor})"),
                None => format!("a finite number of at least '{other}'"),
            },
            Self::OneCharacter => "one ASCII character".to_string(),
            Self::OneOf(choices) => format!(
                "one of {}",
                choices
                    .iter()
                    .map(|choice| format!("'{choice}'"))
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
            Self::ListOrOne => "a list, or one value as a list of one".to_string(),
        }
    }
}

/// The arguments of one call of a procedure, checked and completed by [`prepare`], the only
/// way to build one: every required parameter has a value, every argument lies inside its
/// domain, and every optional parameter that declares a default has it. A procedure's work
/// takes these, so none runs on arguments that were not checked, and none holds a default of its
/// own. They read through the getters of [`Parameters`].
#[derive(Debug, Clone)]
pub struct Prepared(Parameters);

impl std::ops::Deref for Prepared {
    type Target = Parameters;

    fn deref(&self) -> &Parameters {
        &self.0
    }
}

/// The arguments `params` of a call of `procedure`, whose parameters `defs` defines, prepared
/// for it: the one check of every way into a procedure (`CALL`, a direct call, the bindings'
/// algorithm methods), run once all the arguments are set.
///
/// A required parameter without a value, left out or given null, is refused with "invalid
/// parameter value" (22023) naming `procedure`, the parameter and its type; so is an argument
/// outside the domain its definition declares, naming the values it allows and the value. A
/// domain that names another parameter (`fetch_k` at least `k`) reads that one as given, or else
/// its default. One value for a parameter of [`Domain::ListOrOne`] becomes a list of one. Each
/// optional parameter without a value takes the default its definition declares, the one
/// `CALL grafeo.procedures()` lists (one that declares none stays without), and is held to its
/// domain as a given value is.
///
/// # Errors
///
/// The invalid parameter value of the first required parameter without a value, else of the
/// first argument outside its domain; an internal error for a declared default that is no value
/// of its parameter's type.
pub fn prepare(procedure: &str, defs: &[ParameterDef], params: &Parameters) -> Result<Prepared> {
    params.clone().prepared(procedure, defs)
}

/// The domain check of [`prepare`]: an argument outside its domain, a string's then a number's,
/// the defaults included.
fn check_domains(procedure: &str, defs: &[ParameterDef], params: &Parameters) -> Result<()> {
    for def in defs {
        if let Some(value) = params.get_string(&def.name)
            && !def.domain.admits_text(value)
        {
            return Err(Error::from(QueryError::new(
                QueryErrorKind::InvalidParameterValue,
                format!(
                    "argument '{}' of {procedure} must be {}, got '{value}'",
                    def.name,
                    def.domain.describe(None),
                ),
            )));
        }
    }
    check_numbers(procedure, defs, params)
}

/// A required parameter without a value, left out or given null.
fn check_required(procedure: &str, defs: &[ParameterDef], params: &Parameters) -> Result<()> {
    if let Some(def) = defs
        .iter()
        .find(|def| def.required && !params.has_value(&def.name))
    {
        let null = if params.given_null(&def.name) {
            ", got null"
        } else {
            ""
        };
        return Err(Error::from(QueryError::new(
            QueryErrorKind::InvalidParameterValue,
            format!(
                "{procedure}: missing required argument '{}' ({}){null}",
                def.name,
                def.param_type.name()
            ),
        )));
    }
    Ok(())
}

/// A numeric argument outside its domain.
fn check_numbers(procedure: &str, defs: &[ParameterDef], params: &Parameters) -> Result<()> {
    for def in defs {
        let Some(value) = params.number(&def.name) else {
            continue;
        };
        // The parameter an `AtLeast` names, as the call gives it or by its declared default.
        let floor = match def.domain {
            Domain::AtLeast(other) => params.number(other).or_else(|| {
                defs.iter()
                    .find(|def| def.name == other)
                    .and_then(|def| def.default.as_deref())
                    .and_then(declared_number)
            }),
            _ => None,
        };
        // An integer against an integer floor as integers, so a value one below a floor past
        // 2^53 is refused; an integer against a float exactly.
        let inside = def.domain.admits(number_as_float(&value))
            && floor.as_ref().is_none_or(|floor| {
                compare_values(&value, floor).is_some_and(std::cmp::Ordering::is_ge)
            });
        if !inside {
            let floor = floor.as_ref().map(number_text);
            return Err(Error::from(QueryError::new(
                QueryErrorKind::InvalidParameterValue,
                format!(
                    "argument '{}' of {procedure} must be {}, got {}",
                    def.name,
                    def.domain.describe(floor.as_deref()),
                    number_text(&value)
                ),
            )));
        }
    }
    Ok(())
}

/// A declared default of a numeric parameter as a number: an integer when it reads as one.
fn declared_number(default: &str) -> Option<grafeo_common::types::Value> {
    use grafeo_common::types::Value;
    default
        .parse()
        .map(Value::Int64)
        .or_else(|_| default.parse().map(Value::Float64))
        .ok()
}

/// A number of [`Parameters::number`] as a float, for the domains that bound its sign or its
/// range, which a float reads exactly enough.
fn number_as_float(value: &grafeo_common::types::Value) -> f64 {
    use grafeo_common::types::Value;
    match value {
        Value::Int64(v) => *v as f64,
        Value::Float64(v) => *v,
        _ => f64::NAN,
    }
}

/// A number of [`Parameters::number`] as the call gave it (`-1`, `NaN`), for an error.
fn number_text(value: &grafeo_common::types::Value) -> String {
    use grafeo_common::types::Value;
    match value {
        Value::Int64(v) => v.to_string(),
        Value::Float64(v) => v.to_string(),
        other => other.to_string(),
    }
}

/// The widening of [`prepare`]: one value for a parameter of [`Domain::ListOrOne`] becomes a
/// list of one.
fn widen_lists(defs: &[ParameterDef], params: &mut Parameters) {
    for def in defs.iter().filter(|def| def.domain == Domain::ListOrOne) {
        if let Some(value) = params.values.get_mut(&def.name)
            && let Some(one) = value.as_one_value()
        {
            *value = ParameterValue::List(vec![one]);
        }
    }
}

/// The defaults of [`prepare`]: each optional parameter of `defs` without a value in `params`
/// takes the default its definition declares.
fn apply_defaults(defs: &[ParameterDef], params: &mut Parameters) -> Result<()> {
    for def in defs {
        let Some(default) = def.default.as_deref() else {
            continue;
        };
        if def.required || params.has_value(&def.name) {
            continue;
        }
        let name = def.name.as_str();
        let parsed = match def.param_type {
            ParameterType::Integer | ParameterType::NodeId => default
                .parse()
                .map(|value| params.set_int(name, value))
                .ok(),
            ParameterType::Float => default
                .parse()
                .map(|value| params.set_float(name, value))
                .ok(),
            ParameterType::Boolean => default
                .parse()
                .map(|value| params.set_bool(name, value))
                .ok(),
            ParameterType::String => {
                params.set_string(name, default);
                Some(())
            }
            ParameterType::List | ParameterType::Map => None,
        };
        if parsed.is_none() {
            return Err(Error::Internal(format!(
                "the declared default '{default}' of '{name}' is no {}",
                def.param_type.name()
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
    /// A map of keys to values (the column types of a file, an edge end's key), which the
    /// procedure reads itself.
    Map,
}

impl ParameterType {
    /// The type's name in an error: `INTEGER`, `FLOAT`, `STRING`, `BOOLEAN`, `NODE ID`, `LIST`
    /// or `MAP`.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Integer => "INTEGER",
            Self::Float => "FLOAT",
            Self::String => "STRING",
            Self::Boolean => "BOOLEAN",
            Self::NodeId => "NODE ID",
            Self::List => "LIST",
            Self::Map => "MAP",
        }
    }
}

/// Parameters passed to an algorithm.
#[derive(Debug, Clone)]
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

    /// Sets a map parameter.
    pub fn set_map(
        &mut self,
        name: impl Into<String>,
        value: std::sync::Arc<
            std::collections::BTreeMap<
                grafeo_common::types::PropertyKey,
                grafeo_common::types::Value,
            >,
        >,
    ) {
        self.values.insert(name.into(), ParameterValue::Map(value));
    }

    /// Records that the call gave `name` a null: an optional parameter keeps its default, and
    /// [`prepare`] refuses a required one as given null.
    pub fn set_null(&mut self, name: impl Into<String>) {
        self.values.insert(name.into(), ParameterValue::Null);
    }

    /// These arguments of a call of `procedure`, whose parameters `defs` defines, prepared for
    /// it, as [`prepare`] prepares them, without a copy.
    ///
    /// # Errors
    ///
    /// The errors of [`prepare`].
    pub fn prepared(mut self, procedure: &str, defs: &[ParameterDef]) -> Result<Prepared> {
        check_required(procedure, defs, &self)?;
        widen_lists(defs, &mut self);
        apply_defaults(defs, &mut self)?;
        // After the defaults, so a declared default is held to its domain like a given value.
        check_domains(procedure, defs, &self)?;
        Ok(Prepared(self))
    }

    /// Whether the call gave `name` a value (not left out, not null).
    #[must_use]
    pub fn has_value(&self, name: &str) -> bool {
        self.values
            .get(name)
            .is_some_and(|value| !matches!(value, ParameterValue::Null))
    }

    /// Whether the call gave `name` a null.
    #[must_use]
    pub fn given_null(&self, name: &str) -> bool {
        matches!(self.values.get(name), Some(ParameterValue::Null))
    }

    /// The integer of `name`, which has a value: a required parameter, or an optional one that
    /// declares a default, once [`prepare`] has passed.
    ///
    /// # Errors
    ///
    /// An internal error naming it when it has none, which only a caller that skips those gets.
    pub fn required_int(&self, name: &str) -> Result<i64> {
        self.get_int(name).ok_or_else(|| unchecked(name))
    }

    /// The float of `name`, which has a value, as [`required_int`](Self::required_int).
    ///
    /// # Errors
    ///
    /// An internal error naming it when it has none, as [`required_int`](Self::required_int).
    pub fn required_float(&self, name: &str) -> Result<f64> {
        self.get_float(name).ok_or_else(|| unchecked(name))
    }

    /// The boolean of `name`, which has a value, as [`required_int`](Self::required_int).
    ///
    /// # Errors
    ///
    /// An internal error naming it when it has none, as [`required_int`](Self::required_int).
    pub fn required_bool(&self, name: &str) -> Result<bool> {
        self.get_bool(name).ok_or_else(|| unchecked(name))
    }

    /// The string of the required parameter `name`.
    ///
    /// # Errors
    ///
    /// An internal error naming it when it has none, as [`required_int`](Self::required_int).
    pub fn required_string(&self, name: &str) -> Result<&str> {
        self.get_string(name).ok_or_else(|| unchecked(name))
    }

    /// The count of the required parameter `name`, as [`get_count`](Self::get_count) reads it.
    ///
    /// # Errors
    ///
    /// An internal error naming it when it has none, as [`required_int`](Self::required_int),
    /// and the invalid parameter value of [`get_count`](Self::get_count).
    pub fn required_count(&self, name: &str) -> Result<usize> {
        self.get_count(name)?.ok_or_else(|| unchecked(name))
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
    /// [`prepare`] can give.
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

    /// A numeric parameter as the call gave it: an integer or a float.
    fn number(&self, name: &str) -> Option<grafeo_common::types::Value> {
        use grafeo_common::types::Value;
        match self.values.get(name) {
            Some(ParameterValue::Integer(v)) => Some(Value::Int64(*v)),
            Some(ParameterValue::Float(v)) => Some(Value::Float64(*v)),
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

    /// Gets a map parameter.
    pub fn get_map(
        &self,
        name: &str,
    ) -> Option<
        &std::collections::BTreeMap<grafeo_common::types::PropertyKey, grafeo_common::types::Value>,
    > {
        match self.values.get(name) {
            Some(ParameterValue::Map(v)) => Some(v),
            _ => None,
        }
    }
}

impl Default for Parameters {
    fn default() -> Self {
        Self::new()
    }
}

/// The error of a required parameter without a value that reached the procedure, which
/// [`prepare`] would have refused, or of an optional one it would have given its default.
fn unchecked(name: &str) -> Error {
    Error::Internal(format!(
        "argument '{name}' reached the procedure without a value"
    ))
}

/// A parameter value.
#[derive(Debug, Clone)]
enum ParameterValue {
    /// A null the call gave, which is no value.
    Null,
    Integer(i64),
    Float(f64),
    String(String),
    Boolean(bool),
    List(Vec<grafeo_common::types::Value>),
    Map(
        std::sync::Arc<
            std::collections::BTreeMap<
                grafeo_common::types::PropertyKey,
                grafeo_common::types::Value,
            >,
        >,
    ),
}

impl ParameterValue {
    /// One value as a list's element: `None` for a list, which needs no widening, and a null,
    /// which is no value.
    fn as_one_value(&self) -> Option<grafeo_common::types::Value> {
        use grafeo_common::types::Value;
        Some(match self {
            Self::Null | Self::List(_) => return None,
            Self::Integer(v) => Value::Int64(*v),
            Self::Float(v) => Value::Float64(*v),
            Self::String(v) => Value::from(v.as_str()),
            Self::Boolean(v) => Value::Bool(*v),
            Self::Map(v) => Value::Map(std::sync::Arc::clone(v)),
        })
    }
}

/// Result of an algorithm execution.
#[derive(Debug)]
pub struct AlgorithmResult {
    /// Result columns.
    pub columns: Vec<String>,
    /// Result rows.
    pub rows: Vec<Vec<grafeo_common::types::Value>>,
    /// The warnings of the run, which the statement that called it reports (a scores
    /// procedure that ran out of iterations, `NotConverged`).
    pub warnings: grafeo_common::utils::warning::WarningSet,
}

impl AlgorithmResult {
    /// Creates a new empty result.
    pub fn new(columns: Vec<String>) -> Self {
        Self {
            columns,
            rows: Vec::new(),
            warnings: grafeo_common::utils::warning::WarningSet::EMPTY,
        }
    }

    /// Returns the result with `warning` among its warnings.
    #[must_use]
    pub fn with_warning(mut self, warning: grafeo_common::utils::warning::Warning) -> Self {
        self.warnings.insert(warning);
        self
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
