//! Python type conversions.

use std::collections::BTreeMap;
use std::sync::Arc;

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use graphos_common::types::{PropertyKey, Value};

use crate::error::{PyGraphosError, PyGraphosResult};

/// Python-wrapped Value type.
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
    fn as_bool(&self) -> PyGraphosResult<bool> {
        match &self.inner {
            Value::Bool(v) => Ok(*v),
            _ => Err(PyGraphosError::Type("Value is not a boolean".into())),
        }
    }

    /// Get integer value.
    fn as_int(&self) -> PyGraphosResult<i64> {
        match &self.inner {
            Value::Int64(v) => Ok(*v),
            _ => Err(PyGraphosError::Type("Value is not an integer".into())),
        }
    }

    /// Get float value.
    fn as_float(&self) -> PyGraphosResult<f64> {
        match &self.inner {
            Value::Float64(v) => Ok(*v),
            _ => Err(PyGraphosError::Type("Value is not a float".into())),
        }
    }

    /// Get string value.
    fn as_str(&self) -> PyGraphosResult<String> {
        match &self.inner {
            Value::String(v) => Ok(v.to_string()),
            _ => Err(PyGraphosError::Type("Value is not a string".into())),
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
    /// Convert from Python object to Value.
    pub fn from_py(obj: &Bound<'_, PyAny>) -> PyGraphosResult<Value> {
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
                .map_err(|e| PyGraphosError::Type(format!("Cannot cast to dict: {}", e)))?;
            let mut map = BTreeMap::new();
            for (key, value) in dict.iter() {
                let key_str: String = key
                    .extract()
                    .map_err(|e| PyGraphosError::Type(format!("Dict key must be string: {}", e)))?;
                map.insert(PropertyKey::new(key_str), Self::from_py(&value)?);
            }
            return Ok(Value::Map(Arc::new(map)));
        }

        let type_name = obj
            .get_type()
            .name()
            .map(|s| s.to_string())
            .unwrap_or_else(|_| "<unknown>".to_string());
        Err(PyGraphosError::Type(format!(
            "Unsupported Python type: {}",
            type_name
        )))
    }

    /// Convert Value to Python object.
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
            // Handle other types as needed
            _ => py.None(),
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
