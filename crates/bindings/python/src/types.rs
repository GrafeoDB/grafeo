//! Converts between Python and Grafeo value types automatically.
//!
//! | Python type | Grafeo type | Notes |
//! | ----------- | ----------- | ----- |
//! | `None` | `Null` | |
//! | `bool` | `Bool` | |
//! | `int` | `Int64` | |
//! | `float` | `Float64` | |
//! | `str` | `String` | |
//! | `list[float]` | `Vector` | Lists where every element is a Python float (not int) |
//! | `list` | `List` | All other lists converted recursively |
//! | `dict` | `Map` | Keys must be strings |
//! | `bytes` | `Bytes` | |
//! | `datetime` | `Timestamp` | Naive values are UTC; returned as naive UTC |

use std::collections::BTreeMap;
use std::sync::Arc;

use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDateTime, PyDict, PyFloat, PyList};

use grafeo_common::types::{PropertyKey, Timestamp, Value};

use crate::error::{PyGrafeoError, PyGrafeoResult};

/// Wraps a Grafeo value for explicit type handling.
///
/// Usually you don't need this - Python types convert automatically. Use this
/// when you need explicit control like `Value.null()` or type checking.
#[pyclass(name = "Value", skip_from_py_object)]
#[derive(Clone, Debug)]
pub struct PyValue {
    pub(crate) inner: Value,
}

#[pymethods]
impl PyValue {
    /// Create a null value.
    #[staticmethod]
    fn null() -> Self {
        Self { inner: Value::Null }
    }

    /// Create a boolean value.
    #[staticmethod]
    fn boolean(v: bool) -> Self {
        Self {
            inner: Value::Bool(v),
        }
    }

    /// Create an integer value.
    #[staticmethod]
    fn integer(v: i64) -> Self {
        Self {
            inner: Value::Int64(v),
        }
    }

    /// Create a float value.
    #[staticmethod]
    fn float(v: f64) -> Self {
        Self {
            inner: Value::Float64(v),
        }
    }

    /// Create a string value.
    #[staticmethod]
    fn string(v: String) -> Self {
        Self {
            inner: Value::String(v.into()),
        }
    }

    /// Check if value is null.
    fn is_null(&self) -> bool {
        matches!(self.inner, Value::Null)
    }

    /// Get boolean value.
    fn as_bool(&self) -> PyGrafeoResult<bool> {
        match &self.inner {
            Value::Bool(v) => Ok(*v),
            _ => Err(PyGrafeoError::Type("Value is not a boolean".into())),
        }
    }

    /// Get integer value.
    fn as_int(&self) -> PyGrafeoResult<i64> {
        match &self.inner {
            Value::Int64(v) => Ok(*v),
            _ => Err(PyGrafeoError::Type("Value is not an integer".into())),
        }
    }

    /// Get float value.
    fn as_float(&self) -> PyGrafeoResult<f64> {
        match &self.inner {
            Value::Float64(v) => Ok(*v),
            _ => Err(PyGrafeoError::Type("Value is not a float".into())),
        }
    }

    /// Get string value.
    fn as_str(&self) -> PyGrafeoResult<String> {
        match &self.inner {
            Value::String(v) => Ok(v.to_string()),
            _ => Err(PyGrafeoError::Type("Value is not a string".into())),
        }
    }

    fn __repr__(&self) -> String {
        format!("Value({:?})", self.inner)
    }

    fn __str__(&self) -> String {
        format!("{:?}", self.inner)
    }
}

impl PyValue {
    /// Converts a Python object to a Grafeo Value.
    pub fn from_py(obj: &Bound<'_, PyAny>) -> PyGrafeoResult<Value> {
        if obj.is_none() {
            return Ok(Value::Null);
        }

        if let Ok(v) = obj.extract::<bool>() {
            return Ok(Value::Bool(v));
        }

        if let Ok(v) = obj.extract::<i64>() {
            return Ok(Value::Int64(v));
        }

        if let Ok(v) = obj.extract::<f64>() {
            return Ok(Value::Float64(v));
        }

        if let Ok(v) = obj.extract::<String>() {
            return Ok(Value::String(v.into()));
        }

        if let Ok(v) = obj.extract::<Vec<Bound<'_, PyAny>>>() {
            // Only convert to Vector when ALL elements are Python floats (not ints
            // coerced to float). This prevents [1, 2, 3] from being stored as an
            // embedding vector instead of a general-purpose list.
            if !v.is_empty() && v.iter().all(|item| item.is_instance_of::<PyFloat>()) {
                let floats: Result<Vec<f32>, _> =
                    v.iter().map(|item| item.extract::<f32>()).collect();
                if let Ok(floats) = floats {
                    return Ok(Value::Vector(floats.into()));
                }
            }

            let mut items = Vec::new();
            for item in v {
                items.push(Self::from_py(&item)?);
            }
            return Ok(Value::List(items.into()));
        }

        if obj.is_instance_of::<PyDict>() {
            // SAFETY: We just checked it's a PyDict instance
            let dict: &Bound<'_, PyDict> = obj
                .cast()
                .map_err(|e| PyGrafeoError::Type(format!("Cannot cast to dict: {}", e)))?;
            let mut map = BTreeMap::new();
            for (key, value) in dict.iter() {
                let key_str: String = key
                    .extract()
                    .map_err(|e| PyGrafeoError::Type(format!("Dict key must be string: {}", e)))?;
                map.insert(PropertyKey::new(key_str), Self::from_py(&value)?);
            }
            return Ok(Value::Map(Arc::new(map)));
        }

        // Handle bytes
        if obj.is_instance_of::<PyBytes>() {
            let bytes: &Bound<'_, PyBytes> = obj
                .cast()
                .map_err(|e| PyGrafeoError::Type(format!("Cannot cast to bytes: {}", e)))?;
            let byte_slice: &[u8] = bytes.as_bytes();
            return Ok(Value::Bytes(byte_slice.into()));
        }

        // Handle datetime. Naive datetimes are UTC, the same convention used
        // when timestamps are returned, so values round-trip on any machine.
        if obj.is_instance_of::<PyDateTime>() {
            let micros = datetime_to_utc_micros(obj)
                .map_err(|e| PyGrafeoError::Type(format!("Failed to convert datetime: {}", e)))?;
            return Ok(Value::Timestamp(Timestamp::from_micros(micros)));
        }

        let type_name = obj
            .get_type()
            .name()
            .map_or_else(|_| "<unknown>".to_string(), |s| s.to_string());
        Err(PyGrafeoError::Type(format!(
            "Unsupported Python type: {}",
            type_name
        )))
    }

    /// Converts a Grafeo Value to a Python object.
    ///
    /// # Panics
    ///
    /// Panics on memory exhaustion during Python object allocation.
    pub fn to_py(value: &Value, py: Python<'_>) -> Py<PyAny> {
        use pyo3::conversion::IntoPyObjectExt;

        match value {
            Value::Null => py.None(),
            // PyO3 conversions for primitive types only fail on memory exhaustion
            Value::Bool(v) => (*v)
                .into_py_any(py)
                .expect("bool to Python conversion cannot fail"),
            Value::Int64(v) => (*v)
                .into_py_any(py)
                .expect("i64 to Python conversion cannot fail"),
            Value::Float64(v) => (*v)
                .into_py_any(py)
                .expect("f64 to Python conversion cannot fail"),
            Value::String(v) => {
                let s: &str = v.as_ref();
                s.into_py_any(py)
                    .expect("str to Python conversion cannot fail")
            }
            Value::List(items) => {
                let py_items: Vec<Py<PyAny>> = items.iter().map(|v| Self::to_py(v, py)).collect();
                PyList::new(py, py_items)
                    .expect("PyList creation only fails on memory exhaustion")
                    .unbind()
                    .into_any()
            }
            Value::Map(map) => {
                let dict = PyDict::new(py);
                for (k, v) in map.as_ref() {
                    dict.set_item(k.as_str(), Self::to_py(v, py))
                        .expect("dict.set_item only fails on memory exhaustion");
                }
                dict.unbind().into_any()
            }
            Value::Bytes(bytes) => PyBytes::new(py, bytes.as_ref()).unbind().into_any(),
            Value::Timestamp(ts) => utc_micros_to_datetime(py, ts.as_micros())
                .map_or_else(|_| py.None(), |dt| dt.unbind().into_any()),
            Value::Date(d) => {
                let datetime_mod = py.import("datetime").expect("datetime module should exist");
                let date_class = datetime_mod
                    .getattr("date")
                    .expect("datetime.date should exist");
                date_class
                    .call1((d.year(), d.month(), d.day()))
                    .map_or_else(|_| py.None(), |dt| dt.unbind().into_any())
            }
            Value::Time(t) => {
                let datetime_mod = py.import("datetime").expect("datetime module should exist");
                let time_class = datetime_mod
                    .getattr("time")
                    .expect("datetime.time should exist");
                let micros = t.nanosecond() / 1000;
                time_class
                    .call1((t.hour(), t.minute(), t.second(), micros))
                    .map_or_else(|_| py.None(), |dt| dt.unbind().into_any())
            }
            Value::Duration(d) => {
                use pyo3::conversion::IntoPyObjectExt;
                let dict = PyDict::new(py);
                dict.set_item("months", d.months())
                    .expect("dict.set_item only fails on memory exhaustion");
                dict.set_item("days", d.days())
                    .expect("dict.set_item only fails on memory exhaustion");
                dict.set_item("nanos", d.nanos())
                    .expect("dict.set_item only fails on memory exhaustion");
                dict.into_py_any(py)
                    .expect("dict to Python conversion cannot fail")
            }
            Value::ZonedDatetime(zdt) => {
                // Convert to Python datetime with fixed-offset timezone
                let datetime_mod = py.import("datetime").expect("datetime module is built-in");
                let local_date = zdt.to_local_date();
                let local_time = zdt.to_local_time();
                let micros = local_time.nanosecond() / 1000;
                let offset_secs = zdt.offset_seconds();

                // Build timezone using timedelta
                let td_class = datetime_mod
                    .getattr("timedelta")
                    .expect("datetime.timedelta should exist");
                let tz_class = datetime_mod
                    .getattr("timezone")
                    .expect("datetime.timezone should exist");
                let dt_class = datetime_mod
                    .getattr("datetime")
                    .expect("datetime.datetime should exist");

                let td = td_class
                    .call1((0, offset_secs))
                    .unwrap_or_else(|_| py.None().bind(py).clone());
                let tz = tz_class
                    .call1((td,))
                    .unwrap_or_else(|_| py.None().bind(py).clone());
                dt_class
                    .call1((
                        local_date.year(),
                        local_date.month(),
                        local_date.day(),
                        local_time.hour(),
                        local_time.minute(),
                        local_time.second(),
                        micros,
                        tz,
                    ))
                    .map_or_else(|_| py.None(), |dt| dt.unbind().into_any())
            }
            Value::Vector(v) => {
                // Convert vector to Python list of floats
                let py_floats: Vec<f32> = v.iter().copied().collect();
                PyList::new(py, py_floats)
                    .expect("PyList creation only fails on memory exhaustion")
                    .unbind()
                    .into_any()
            }
            Value::Path { nodes, edges } => {
                let dict = PyDict::new(py);
                let py_nodes: Vec<Py<PyAny>> = nodes.iter().map(|v| Self::to_py(v, py)).collect();
                let py_edges: Vec<Py<PyAny>> = edges.iter().map(|v| Self::to_py(v, py)).collect();
                dict.set_item(
                    "nodes",
                    PyList::new(py, py_nodes)
                        .expect("PyList creation only fails on memory exhaustion"),
                )
                .expect("dict.set_item only fails on memory exhaustion");
                dict.set_item(
                    "edges",
                    PyList::new(py, py_edges)
                        .expect("PyList creation only fails on memory exhaustion"),
                )
                .expect("dict.set_item only fails on memory exhaustion");
                dict.unbind().into_any()
            }
            Value::GCounter(counts) => {
                // Expose as {"$gcounter": {"replica": count, ...}, "$value": total}
                use pyo3::conversion::IntoPyObjectExt;
                let dict = PyDict::new(py);
                let replicas = PyDict::new(py);
                let mut total: u64 = 0;
                for (replica, count) in counts.iter() {
                    total += count;
                    replicas
                        .set_item(replica.as_str(), *count)
                        .expect("dict.set_item only fails on memory exhaustion");
                }
                dict.set_item("$gcounter", replicas)
                    .expect("dict.set_item only fails on memory exhaustion");
                dict.set_item("$value", total)
                    .expect("dict.set_item only fails on memory exhaustion");
                dict.into_py_any(py)
                    .expect("dict to Python conversion cannot fail")
            }
            Value::OnCounter { pos, neg } => {
                use pyo3::conversion::IntoPyObjectExt;
                let pos_sum: u128 = pos.values().copied().map(u128::from).sum();
                let neg_sum: u128 = neg.values().copied().map(u128::from).sum();
                let net: i128 = pos_sum.cast_signed() - neg_sum.cast_signed();
                let dict = PyDict::new(py);
                dict.set_item("$pncounter", true)
                    .expect("dict.set_item only fails on memory exhaustion");
                dict.set_item("$value", net)
                    .expect("dict.set_item only fails on memory exhaustion");
                dict.into_py_any(py)
                    .expect("dict to Python conversion cannot fail")
            }
            _ => {
                let s = value.to_string();
                s.into_py_any(py)
                    .expect("str to Python conversion cannot fail")
            }
        }
    }
}

const MICROS_PER_SECOND: i64 = 1_000_000;
const MICROS_PER_DAY: i64 = 86_400 * MICROS_PER_SECOND;

/// `datetime(1970, 1, 1)`, naive.
fn unix_epoch(py: Python<'_>) -> PyResult<Bound<'_, PyAny>> {
    py.import("datetime")?
        .getattr("datetime")?
        .call1((1970, 1, 1))
}

/// Microseconds since the Unix epoch for a Python datetime. A naive datetime
/// is read as UTC; an aware one is converted to UTC first.
///
/// Uses exact integer arithmetic (`datetime.timestamp()` reads naive values
/// as local time and goes through a float).
fn datetime_to_utc_micros(obj: &Bound<'_, PyAny>) -> PyResult<i64> {
    let py = obj.py();
    let naive_utc = if obj.call_method0("utcoffset")?.is_none() {
        obj.clone()
    } else {
        let utc = py.import("datetime")?.getattr("timezone")?.getattr("utc")?;
        let kwargs = PyDict::new(py);
        kwargs.set_item("tzinfo", py.None())?;
        obj.call_method1("astimezone", (utc,))?
            .call_method("replace", (), Some(&kwargs))?
    };
    let delta = naive_utc.call_method1("__sub__", (unix_epoch(py)?,))?;
    let days: i64 = delta.getattr("days")?.extract()?;
    let seconds: i64 = delta.getattr("seconds")?.extract()?;
    let micros: i64 = delta.getattr("microseconds")?.extract()?;
    Ok(days * MICROS_PER_DAY + seconds * MICROS_PER_SECOND + micros)
}

/// A naive UTC datetime for microseconds since the Unix epoch.
///
/// Built as `epoch + timedelta`, which is exact and, unlike
/// `datetime.fromtimestamp`, also works for dates before 1970 on Windows.
fn utc_micros_to_datetime(py: Python<'_>, micros: i64) -> PyResult<Bound<'_, PyAny>> {
    let timedelta = py.import("datetime")?.getattr("timedelta")?;
    let kwargs = PyDict::new(py);
    kwargs.set_item("microseconds", micros)?;
    let delta = timedelta.call((), Some(&kwargs))?;
    unix_epoch(py)?.call_method1("__add__", (delta,))
}

impl From<Value> for PyValue {
    fn from(inner: Value) -> Self {
        Self { inner }
    }
}

impl From<PyValue> for Value {
    fn from(py_val: PyValue) -> Self {
        py_val.inner
    }
}

/// Creates a vector value from a list of floats.
///
/// Use this for explicit vector construction:
/// ```python
/// import grafeo
/// vec = grafeo.vector([0.1, 0.2, 0.3])
/// db.create_node(['Doc'], {'embedding': vec})
/// ```
///
/// Note: All-float Python lists are automatically converted to vectors,
/// so `[0.1, 0.2, 0.3]` works directly in most cases.
#[pyfunction]
pub fn vector(values: Vec<f32>) -> PyResult<Vec<f32>> {
    if values.is_empty() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "vector() requires at least one element",
        ));
    }
    Ok(values)
}
