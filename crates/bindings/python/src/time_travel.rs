//! Time travel: reads of the database as it was at an earlier epoch or time.
//!
//! A point in time is an `int` (an epoch), a timezone-aware `datetime` (a
//! time) or a `timedelta` (that long ago). A bare number is never read as a
//! time, and a naive `datetime` is refused: its instant depends on a zone it
//! does not name.

use std::sync::Arc;
use std::time::Duration;

use parking_lot::{Mutex, RwLock};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyDateTime, PyDelta, PyDict, PyInt, PyString};

use grafeo_common::types::{EdgeId, EpochId, NodeId, Timestamp};
use grafeo_engine::database::GrafeoDB;
use grafeo_engine::{AsOfView, HistoryRetention, TimePoint};

use crate::graph::{PyEdge, PyNode};
use crate::query::PyQueryResult;
use crate::types::PyValue;

/// The point in time `point` names: an `int` is an epoch, an aware
/// `datetime` a time, a `timedelta` a duration before now.
///
/// # Errors
///
/// `TypeError` for any other type (a `bool` too), `ValueError` for a
/// negative epoch or duration and for a naive `datetime`.
pub(crate) fn time_point(point: &Bound<'_, PyAny>) -> PyResult<TimePoint> {
    if point.is_instance_of::<PyBool>() {
        return Err(PyTypeError::new_err(
            "a point in time is an int (an epoch), a datetime or a timedelta, not a bool",
        ));
    }
    if point.is_instance_of::<PyInt>() {
        let epoch: u64 = point
            .extract()
            .map_err(|_| PyValueError::new_err(format!("an epoch is not negative: {point}")))?;
        return Ok(TimePoint::Epoch(EpochId::new(epoch)));
    }
    if point.is_instance_of::<PyDateTime>() {
        if point.getattr("tzinfo")?.is_none() {
            return Err(PyValueError::new_err(
                "a datetime point needs a timezone (datetime.now(datetime.UTC), or \
                 .replace(tzinfo=...)): a naive one names no instant",
            ));
        }
        // Whole milliseconds since 1970, the resolution of a time-travel
        // read, from the exact parts of the difference: `timestamp()` is a
        // float, and 0.339 s times 1000 floors to 338 ms.
        let datetime = point.py().import("datetime")?;
        let utc = datetime.getattr("timezone")?.getattr("utc")?;
        let epoch = datetime
            .getattr("datetime")?
            .call1((1970, 1, 1, 0, 0, 0, 0, utc))?;
        let (days, seconds, micros) = delta_parts(&point.call_method1("__sub__", (epoch,))?)?;
        let millis = days * 86_400_000 + seconds * 1000 + micros / 1000;
        return Ok(TimePoint::Timestamp(Timestamp::from_millis(millis)));
    }
    if point.is_instance_of::<PyDelta>() {
        return Ok(TimePoint::Ago(duration(point, "a timedelta point")?));
    }
    Err(PyTypeError::new_err(format!(
        "a point in time is an int (an epoch), a datetime or a timedelta, not {}",
        point.get_type().name()?
    )))
}

/// A `timedelta`'s exact parts: days (any sign), seconds (0 to 86399) and
/// microseconds (0 to 999999).
fn delta_parts(delta: &Bound<'_, PyAny>) -> PyResult<(i64, i64, i64)> {
    Ok((
        delta.getattr("days")?.extract()?,
        delta.getattr("seconds")?.extract()?,
        delta.getattr("microseconds")?.extract()?,
    ))
}

/// A non-negative `timedelta` as a `Duration`.
fn duration(delta: &Bound<'_, PyAny>, what: &str) -> PyResult<Duration> {
    let (days, seconds, micros) = delta_parts(delta)?;
    let seconds = u64::try_from(days * 86_400 + seconds)
        .map_err(|_| PyValueError::new_err(format!("{what} is not negative: {delta}")))?;
    Ok(Duration::from_secs(seconds) + Duration::from_micros(micros.unsigned_abs()))
}

/// The history retention `retention` names: `"off"`, `"forever"` or a
/// `timedelta` (how long before now the history reaches).
///
/// # Errors
///
/// `ValueError` for another string or a negative `timedelta`, `TypeError`
/// for another type.
pub(crate) fn history_retention(retention: &Bound<'_, PyAny>) -> PyResult<HistoryRetention> {
    if retention.is_instance_of::<PyString>() {
        let name: String = retention.extract()?;
        return match name.to_ascii_lowercase().as_str() {
            "off" => Ok(HistoryRetention::Off),
            "forever" => Ok(HistoryRetention::Forever),
            _ => Err(PyValueError::new_err(format!(
                "history_retention is \"off\", \"forever\" or a timedelta, not {name:?}"
            ))),
        };
    }
    if retention.is_instance_of::<PyDelta>() {
        return Ok(HistoryRetention::Duration(duration(
            retention,
            "history_retention",
        )?));
    }
    Err(PyTypeError::new_err(format!(
        "history_retention is \"off\", \"forever\" or a timedelta, not {}",
        retention.get_type().name()?
    )))
}

/// A read-only view of the database as it was at an earlier epoch, from
/// `GrafeoDB.as_of`.
///
/// Its queries and reads see the committed state of `epoch`; a write is
/// refused. It keeps the history of its epoch while it is open: close it
/// (or use it in a `with` block) when done.
///
/// Example:
///     with db.as_of(datetime.timedelta(minutes=5)) as then:
///         print(then.execute("MATCH (p:Person) RETURN count(p)"))
#[pyclass(name = "AsOfView", module = "grafeo")]
pub struct PyAsOfView {
    /// Keeps the database open while the view lives.
    _db: Arc<RwLock<GrafeoDB>>,
    view: Mutex<Option<AsOfView>>,
    epoch: u64,
}

impl PyAsOfView {
    pub(crate) fn new(db: Arc<RwLock<GrafeoDB>>, view: AsOfView) -> Self {
        let epoch = view.epoch().as_u64();
        Self {
            _db: db,
            view: Mutex::new(Some(view)),
            epoch,
        }
    }

    /// Runs `read` on the open view.
    fn with_view<T>(&self, read: impl FnOnce(&AsOfView) -> PyResult<T>) -> PyResult<T> {
        let view = self.view.lock();
        let view = view.as_ref().ok_or_else(|| {
            PyValueError::new_err("the AsOfView is closed: open another with GrafeoDB.as_of")
        })?;
        read(view)
    }
}

#[pymethods]
impl PyAsOfView {
    /// The epoch the view reads at.
    #[getter]
    fn epoch(&self) -> u64 {
        self.epoch
    }

    /// Runs a GQL query as the database was at the view's epoch; a query
    /// that writes is refused.
    #[pyo3(signature = (query, params=None))]
    fn execute(
        &self,
        query: &str,
        params: Option<&Bound<'_, PyDict>>,
        py: Python<'_>,
    ) -> PyResult<PyQueryResult> {
        let params = crate::direct::params(params)?;
        self.with_view(|view| {
            let result = crate::gil::without_gil(py, || {
                Ok(match params {
                    Some(params) => view.execute_with_params(query, params),
                    None => view.execute(query),
                }?)
            })?;
            Ok(crate::direct::query_result(result))
        })
    }

    /// The node as it was at the view's epoch, or None.
    fn get_node(&self, id: u64) -> PyResult<Option<PyNode>> {
        self.with_view(|view| Ok(view.get_node(NodeId(id)).map(crate::direct::node)))
    }

    /// The edge as it was at the view's epoch, or None.
    fn get_edge(&self, id: u64) -> PyResult<Option<PyEdge>> {
        self.with_view(|view| Ok(view.get_edge(EdgeId(id)).map(crate::direct::edge)))
    }

    /// A node's value of `key` as it was at the view's epoch, or None.
    fn get_node_property(&self, id: u64, key: &str, py: Python<'_>) -> PyResult<Option<Py<PyAny>>> {
        self.with_view(|view| {
            Ok(view
                .get_node_property(NodeId(id), key)
                .map(|value| PyValue::to_py(&value, py)))
        })
    }

    /// Releases the view and the history it keeps; later calls raise
    /// `ValueError`. Closing twice is fine.
    fn close(&self) {
        self.view.lock().take();
    }

    fn __enter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __exit__(
        &self,
        _exc_type: Option<&Bound<'_, PyAny>>,
        _exc_val: Option<&Bound<'_, PyAny>>,
        _exc_tb: Option<&Bound<'_, PyAny>>,
    ) -> bool {
        self.close();
        false
    }

    fn __repr__(&self) -> String {
        let open = if self.view.lock().is_some() {
            ""
        } else {
            ", closed"
        };
        format!("AsOfView(epoch={}{open})", self.epoch)
    }
}
