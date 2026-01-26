//! Python type conversions.

use pyo3::prelude::*;
use pyo3::types::{PyBool, PyDict, PyFloat, PyInt, PyList, PyString};

use graphos_common::types::Value;

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
            inner: Value::String(v),
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
            Value::String(v) => Ok(v.clone()),
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

        if let Ok(v) = obj.downcast::<PyBool>() {
            return Ok(Value::Bool(v.is_true()));
        }

        if let Ok(v) = obj.downcast::<PyInt>() {
            let val: i64 = v.extract().map_err(|e| {
                PyGraphosError::Type(format!("Cannot convert to i64: {}", e))
            })?;
            return Ok(Value::Int64(val));
        }

        if let Ok(v) = obj.downcast::<PyFloat>() {
            let val: f64 = v.extract().map_err(|e| {
                PyGraphosError::Type(format!("Cannot convert to f64: {}", e))
            })?;
            return Ok(Value::Float64(val));
        }

        if let Ok(v) = obj.downcast::<PyString>() {
            let val: String = v.extract().map_err(|e| {
                PyGraphosError::Type(format!("Cannot convert to string: {}", e))
            })?;
            return Ok(Value::String(val));
        }

        if let Ok(v) = obj.downcast::<PyList>() {
            let mut items = Vec::new();
            for item in v.iter() {
                items.push(Self::from_py(&item)?);
            }
            return Ok(Value::List(items));
        }

        if let Ok(v) = obj.downcast::<PyDict>() {
            let mut map = std::collections::HashMap::new();
            for (key, value) in v.iter() {
                let key_str: String = key.extract().map_err(|e| {
                    PyGraphosError::Type(format!("Dict key must be string: {}", e))
                })?;
                map.insert(key_str, Self::from_py(&value)?);
            }
            return Ok(Value::Map(map));
        }

        Err(PyGraphosError::Type(format!(
            "Unsupported Python type: {}",
            obj.get_type().name().unwrap_or("<unknown>")
        )))
    }

    /// Convert Value to Python object.
    pub fn to_py(value: &Value, py: Python<'_>) -> PyObject {
        match value {
            Value::Null => py.None(),
            Value::Bool(v) => v.into_pyobject(py).unwrap().into_any().unbind(),
            Value::Int64(v) => v.into_pyobject(py).unwrap().into_any().unbind(),
            Value::Float64(v) => v.into_pyobject(py).unwrap().into_any().unbind(),
            Value::String(v) => v.into_pyobject(py).unwrap().into_any().unbind(),
            Value::List(items) => {
                let list = PyList::new(
                    py,
                    items.iter().map(|v| Self::to_py(v, py)),
                ).unwrap();
                list.into()
            }
            Value::Map(map) => {
                let dict = PyDict::new(py);
                for (k, v) in map {
                    dict.set_item(k, Self::to_py(v, py)).unwrap();
                }
                dict.into()
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
