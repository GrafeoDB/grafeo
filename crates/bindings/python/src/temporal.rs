//! The Grafeo temporal types for values Python's own types cannot hold
//! exactly: a time with nanoseconds (`grafeo.Time`), a duration with months
//! or more than microseconds (`grafeo.Duration`), and a date or datetime
//! outside years 1 to 9999 (`grafeo.Date`, `grafeo.Timestamp`,
//! `grafeo.ZonedDateTime`). A result gives one only for such a value; every
//! other temporal comes back as Python's own type. Passed as a parameter, each
//! is the same engine value again.
//!
//! # Stability: Experimental
//!
//! New in 0.6.4: the classes, their names and their attributes may change in
//! any release, with a CHANGELOG entry.

use std::hash::{Hash, Hasher};

use grafeo_common::types::{Date, Duration, Time, Timestamp, Value, ZonedDatetime};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// The methods every Grafeo temporal class has: `str()` gives the engine's
/// text, `==` and `hash()` compare the engine values, and `to_native()` gives
/// Python's own type (see each class for what it drops or refuses).
macro_rules! temporal_class {
    ($class:ident, $name:literal, $variant:ident, { $($methods:tt)* }) => {
        #[pymethods]
        impl $class {
            $($methods)*

            fn __str__(&self) -> String {
                self.0.to_string()
            }

            fn __repr__(&self) -> String {
                format!("grafeo.{}('{}')", $name, self.0)
            }

            fn __eq__(&self, other: &Bound<'_, PyAny>) -> bool {
                other
                    .cast::<$class>()
                    .is_ok_and(|other| other.get().0 == self.0)
            }

            // The engine value's hash, which agrees with its `==`: equal
            // instants with other offsets hash alike (the text would not).
            fn __hash__(&self) -> u64 {
                let mut hasher = std::collections::hash_map::DefaultHasher::new();
                self.0.hash(&mut hasher);
                hasher.finish()
            }

            /// Python's own type for this value.
            ///
            /// # Errors
            ///
            /// `ValueError` when Python's type cannot hold it (a date outside
            /// years 1 to 9999, a duration with months); `OverflowError` for
            /// a duration past 999,999,999 days, as `timedelta` raises.
            fn to_native(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
                crate::types::native_temporal(py, &Value::$variant(self.0))
            }
        }

        impl $class {
            /// The engine value.
            pub(crate) fn value(&self) -> Value {
                Value::$variant(self.0)
            }
        }
    };
}

/// A time of day with nanoseconds and an optional UTC offset in seconds,
/// returned where `datetime.time` would drop the nanoseconds.
/// `grafeo.Time("12:31:14.645876123+01:00")` reads the engine's text.
/// `to_native()` gives the `datetime.time`, cut to microseconds.
///
/// Experimental, new in 0.6.4.
#[pyclass(name = "Time", module = "grafeo", frozen)]
pub struct PyGrafeoTime(pub(crate) Time);

temporal_class!(PyGrafeoTime, "Time", Time, {
    #[new]
    fn new(text: &str) -> PyResult<Self> {
        Time::parse(text)
            .map(Self)
            .ok_or_else(|| PyValueError::new_err(format!("not a time: {text:?}")))
    }

    #[getter]
    fn hour(&self) -> u32 {
        self.0.hour()
    }

    #[getter]
    fn minute(&self) -> u32 {
        self.0.minute()
    }

    #[getter]
    fn second(&self) -> u32 {
        self.0.second()
    }

    /// The nanoseconds within the second.
    #[getter]
    fn nanosecond(&self) -> u32 {
        self.0.nanosecond()
    }

    /// The UTC offset in seconds, `None` for a local time.
    #[getter]
    fn offset_seconds(&self) -> Option<i32> {
        self.0.offset_seconds()
    }
});

/// A calendar date outside years 1 to 9999, which `datetime.date` cannot
/// hold. `grafeo.Date("2024-02-29")` reads the engine's text.
///
/// Experimental, new in 0.6.4.
#[pyclass(name = "Date", module = "grafeo", frozen)]
pub struct PyGrafeoDate(pub(crate) Date);

temporal_class!(PyGrafeoDate, "Date", Date, {
    #[new]
    fn new(text: &str) -> PyResult<Self> {
        Date::parse(text)
            .map(Self)
            .ok_or_else(|| PyValueError::new_err(format!("not a date: {text:?}")))
    }

    #[getter]
    fn year(&self) -> i32 {
        self.0.year()
    }

    #[getter]
    fn month(&self) -> u32 {
        self.0.month()
    }

    #[getter]
    fn day(&self) -> u32 {
        self.0.day()
    }
});

/// A UTC instant in microseconds outside years 1 to 9999, which a naive
/// `datetime` cannot hold. `grafeo.Timestamp(micros)` takes microseconds since
/// the Unix epoch.
///
/// Experimental, new in 0.6.4.
#[pyclass(name = "Timestamp", module = "grafeo", frozen)]
pub struct PyGrafeoTimestamp(pub(crate) Timestamp);

temporal_class!(PyGrafeoTimestamp, "Timestamp", Timestamp, {
    #[new]
    fn new(micros: i64) -> Self {
        Self(Timestamp::from_micros(micros))
    }

    /// Microseconds since the Unix epoch.
    #[getter]
    fn micros(&self) -> i64 {
        self.0.as_micros()
    }
});

/// A datetime with a UTC offset outside years 1 to 9999, which an aware
/// `datetime` cannot hold. `grafeo.ZonedDateTime("...+01:00")` reads the
/// engine's text.
///
/// Experimental, new in 0.6.4.
#[pyclass(name = "ZonedDateTime", module = "grafeo", frozen)]
pub struct PyGrafeoZonedDateTime(pub(crate) ZonedDatetime);

temporal_class!(PyGrafeoZonedDateTime, "ZonedDateTime", ZonedDatetime, {
    #[new]
    fn new(text: &str) -> PyResult<Self> {
        ZonedDatetime::parse(text)
            .map(Self)
            .ok_or_else(|| PyValueError::new_err(format!("not a zoned datetime: {text:?}")))
    }

    /// The UTC offset in seconds.
    #[getter]
    fn offset_seconds(&self) -> i32 {
        self.0.offset_seconds()
    }
});

/// A duration of months, days and nanoseconds, kept apart, returned where a
/// `timedelta` would not hold it exactly (months, nanoseconds, or a time part
/// of a day or more or of the other sign than the days).
/// `grafeo.Duration("P1Y2M10D")` reads the engine's text. `to_native()` gives
/// the `timedelta` of a duration without months, cut to microseconds.
///
/// Experimental, new in 0.6.4.
#[pyclass(name = "Duration", module = "grafeo", frozen)]
pub struct PyGrafeoDuration(pub(crate) Duration);

temporal_class!(PyGrafeoDuration, "Duration", Duration, {
    #[new]
    fn new(text: &str) -> PyResult<Self> {
        Duration::parse(text)
            .map(Self)
            .ok_or_else(|| PyValueError::new_err(format!("not a duration: {text:?}")))
    }

    #[getter]
    fn months(&self) -> i64 {
        self.0.months()
    }

    #[getter]
    fn days(&self) -> i64 {
        self.0.days()
    }

    /// The time part in nanoseconds.
    #[getter]
    fn nanos(&self) -> i64 {
        self.0.nanos()
    }
});

/// The engine value of a Grafeo temporal object, `None` for any other object.
pub(crate) fn value_of(obj: &Bound<'_, PyAny>) -> Option<Value> {
    if let Ok(time) = obj.cast::<PyGrafeoTime>() {
        return Some(time.get().value());
    }
    if let Ok(date) = obj.cast::<PyGrafeoDate>() {
        return Some(date.get().value());
    }
    if let Ok(timestamp) = obj.cast::<PyGrafeoTimestamp>() {
        return Some(timestamp.get().value());
    }
    if let Ok(duration) = obj.cast::<PyGrafeoDuration>() {
        return Some(duration.get().value());
    }
    obj.cast::<PyGrafeoZonedDateTime>()
        .ok()
        .map(|zoned| zoned.get().value())
}
