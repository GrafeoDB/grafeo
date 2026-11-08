//! Query warnings in Python: a frozen `QueryWarning` on `result.warnings`, and each raised
//! through `warnings.warn` from the caller's line, as a `DeprecationWarning` for a deprecation
//! and a `UserWarning` for any other kind.

use std::ffi::CString;

use grafeo_bindings_common::warning::Warning;
use pyo3::exceptions::{PyDeprecationWarning, PyUserWarning};
use pyo3::prelude::*;
use pyo3::types::PyType;

/// A warning a statement raised: a deprecated form it wrote or ran, or a notice.
///
/// `code` is its GQLSTATUS code (class `01`, for example `01X01`), `kind` is `"deprecation"`
/// or `"notice"`, and `message` names the replacement of a deprecated form.
#[pyclass(name = "QueryWarning", frozen, module = "grafeo")]
#[derive(Clone)]
pub struct PyQueryWarning {
    /// The GQLSTATUS code.
    #[pyo3(get)]
    code: String,
    /// `"deprecation"` or `"notice"`.
    #[pyo3(get)]
    kind: String,
    /// What the warning says.
    #[pyo3(get)]
    message: String,
}

#[pymethods]
impl PyQueryWarning {
    fn __repr__(&self) -> String {
        format!(
            "QueryWarning(code={:?}, kind={:?}, message={:?})",
            self.code, self.kind, self.message
        )
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.code == other.code && self.kind == other.kind && self.message == other.message
    }
}

impl From<&Warning> for PyQueryWarning {
    fn from(warning: &Warning) -> Self {
        Self {
            code: warning.code.clone(),
            kind: warning.kind.to_string(),
            message: warning.message.clone(),
        }
    }
}

/// `warnings` as Python objects, in their order.
pub(crate) fn to_py(warnings: &[Warning]) -> Vec<PyQueryWarning> {
    warnings.iter().map(PyQueryWarning::from).collect()
}

/// Raises each of `warnings` through `warnings.warn`, pointing at the Python line that called
/// the method returning them.
///
/// # Errors
///
/// The warning as an exception, when a filter turns it into one (`-W error`).
pub(crate) fn raise(py: Python<'_>, warnings: &[Warning]) -> PyResult<()> {
    for warning in warnings {
        let category: Bound<'_, PyType> = if warning.is_deprecation() {
            py.get_type::<PyDeprecationWarning>()
        } else {
            py.get_type::<PyUserWarning>()
        };
        // With its NULs replaced, the message always makes a C string.
        let message = CString::new(warning.message.replace('\0', " ")).unwrap_or_default();
        PyErr::warn(py, category.as_any(), &message, 1)?;
    }
    Ok(())
}
