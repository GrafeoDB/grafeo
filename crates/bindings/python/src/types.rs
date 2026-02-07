//! Converts between Python and Grafeo value types automatically.
//!
//! | Python type | Grafeo type | Notes |
//! | ----------- | ----------- | ----- |
//! | `None` | `Null` | |
//! | `bool` | `Bool` | |
//! | `int` | `Int64` | |
//! | `float` | `Float64` | |
//! | `str` | `String` | |
//! | `list[float]` | `Vector` | All-float lists become vectors |
//! | `list` | `List` | Mixed-type lists converted recursively |
//! | `dict` | `Map` | Keys must be strings |
//! | `bytes` | `Bytes` | |
//! | `datetime` | `Timestamp` | Converted to/from UTC |

use std::collections::BTreeMap;
use std::sync::Arc;

use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDateTime, PyDict, PyList};

use grafeo_common::types::{PropertyKey, Timestamp, Value};

use crate::error::{PyGrafeoError, PyGrafeoResult};

/// Wraps a Grafeo value for explicit type handling.
///
/// Usually you don't need this - Python types convert automatically. Use this
/// when you need explicit control like `Value.null()` or type checking.
#[pyclass(name = "Value")]
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

        // All-float lists → Value::Vector (must come before generic list)
        if let Ok(v) = obj.extract::<Vec<f32>>()
            && !v.is_empty()
        {
            return Ok(Value::Vector(v.into()));
        }

        if let Ok(v) = obj.extract::<Vec<Bound<'_, PyAny>>>() {
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

        // Handle datetime
        if obj.is_instance_of::<PyDateTime>() {
            // Extract timestamp as float (seconds since epoch)
            let timestamp: f64 = obj
                .call_method0("timestamp")
                .and_then(|ts| ts.extract())
                .map_err(|e| {
                    PyGrafeoError::Type(format!("Failed to get datetime timestamp: {}", e))
                })?;
            // Convert to microseconds
            let micros = (timestamp * 1_000_000.0) as i64;
            return Ok(Value::Timestamp(Timestamp::from_micros(micros)));
        }

        let type_name = obj
            .get_type()
            .name()
            .map(|s| s.to_string())
            .unwrap_or_else(|_| "<unknown>".to_string());
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
            Value::Timestamp(ts) => {
                // Convert microseconds to seconds (as float for precision)
                let micros = ts.as_micros();
                let timestamp_float = micros as f64 / 1_000_000.0;

                // Import datetime module and create datetime from timestamp
                let datetime_mod = py.import("datetime").expect("datetime module should exist");
                let datetime_class = datetime_mod
                    .getattr("datetime")
                    .expect("datetime.datetime should exist");

                // Use utcfromtimestamp for UTC datetime
                datetime_class
                    .call_method1("utcfromtimestamp", (timestamp_float,))
                    .map(|dt| dt.unbind().into_any())
                    .unwrap_or_else(|_| py.None())
            }
            Value::Vector(v) => {
                // Convert vector to Python list of floats
                let py_floats: Vec<f32> = v.iter().copied().collect();
                PyList::new(py, py_floats)
                    .expect("PyList creation only fails on memory exhaustion")
                    .unbind()
                    .into_any()
            }
        }
    }
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
