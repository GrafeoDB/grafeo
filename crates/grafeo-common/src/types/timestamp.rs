//! Timestamps for temporal properties.
//!
//! Stored as microseconds since Unix epoch - plenty of precision for most uses.

use serde::{Deserialize, Serialize};
use std::fmt;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

/// A point in time, stored as microseconds since Unix epoch.
///
/// Microsecond precision, covering roughly 290,000 years in each direction
/// from 1970. Create with [`from_secs()`](Self::from_secs),
/// [`from_millis()`](Self::from_millis), or [`now()`](Self::now).
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize, Default)]
#[repr(transparent)]
pub struct Timestamp(i64);

impl Timestamp {
    /// The Unix epoch (1970-01-01 00:00:00 UTC).
    pub const EPOCH: Self = Self(0);

    /// The minimum representable timestamp.
    pub const MIN: Self = Self(i64::MIN);

    /// The maximum representable timestamp.
    pub const MAX: Self = Self(i64::MAX);

    /// Creates a timestamp from microseconds since the Unix epoch.
    #[inline]
    #[must_use]
    pub const fn from_micros(micros: i64) -> Self {
        Self(micros)
    }

    /// Creates a timestamp from milliseconds since the Unix epoch.
    #[inline]
    #[must_use]
    pub const fn from_millis(millis: i64) -> Self {
        Self(millis * 1000)
    }

    /// Creates a timestamp from seconds since the Unix epoch.
    #[inline]
    #[must_use]
    pub const fn from_secs(secs: i64) -> Self {
        Self(secs * 1_000_000)
    }

    /// Returns the current time as a timestamp.
    #[must_use]
    pub fn now() -> Self {
        let duration = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or(Duration::ZERO);
        Self::from_micros(duration.as_micros() as i64)
    }

    /// Returns the timestamp as microseconds since the Unix epoch.
    #[inline]
    #[must_use]
    pub const fn as_micros(&self) -> i64 {
        self.0
    }

    /// Returns the timestamp as milliseconds since the Unix epoch.
    #[inline]
    #[must_use]
    pub const fn as_millis(&self) -> i64 {
        self.0 / 1000
    }

    /// Returns the timestamp as seconds since the Unix epoch.
    #[inline]
    #[must_use]
    pub const fn as_secs(&self) -> i64 {
        self.0 / 1_000_000
    }

    /// Returns the timestamp as a `SystemTime`, if it's within the representable range.
    #[must_use]
    pub fn as_system_time(&self) -> Option<SystemTime> {
        if self.0 >= 0 {
            Some(UNIX_EPOCH + Duration::from_micros(self.0 as u64))
        } else {
            UNIX_EPOCH.checked_sub(Duration::from_micros((-self.0) as u64))
        }
    }

    /// Adds a duration to this timestamp.
    #[must_use]
    pub const fn add_micros(self, micros: i64) -> Self {
        Self(self.0.saturating_add(micros))
    }

    /// Subtracts a duration from this timestamp.
    #[must_use]
    pub const fn sub_micros(self, micros: i64) -> Self {
        Self(self.0.saturating_sub(micros))
    }

    /// Returns the duration between this timestamp and another.
    ///
    /// Returns a positive value if `other` is before `self`, negative otherwise.
    #[must_use]
    pub const fn duration_since(self, other: Self) -> i64 {
        self.0 - other.0
    }
}

impl fmt::Debug for Timestamp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Timestamp({}μs)", self.0)
    }
}

impl fmt::Display for Timestamp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Simple ISO 8601-ish format
        let secs = self.0 / 1_000_000;
        let micros = (self.0 % 1_000_000).unsigned_abs();

        // Calculate date/time components (simplified, doesn't handle all edge cases)
        const SECS_PER_DAY: i64 = 86400;
        const DAYS_PER_YEAR: i64 = 365;

        let days = secs / SECS_PER_DAY;
        let time_secs = (secs % SECS_PER_DAY + SECS_PER_DAY) % SECS_PER_DAY;

        let hours = time_secs / 3600;
        let minutes = (time_secs % 3600) / 60;
        let seconds = time_secs % 60;

        // Very rough year calculation (ignores leap years for display)
        let year = 1970 + days / DAYS_PER_YEAR;
        let day_of_year = days % DAYS_PER_YEAR;

        write!(
            f,
            "{:04}-{:03}T{:02}:{:02}:{:02}.{:06}Z",
            year, day_of_year, hours, minutes, seconds, micros
        )
    }
}

impl From<i64> for Timestamp {
    fn from(micros: i64) -> Self {
        Self::from_micros(micros)
    }
}

impl From<Timestamp> for i64 {
    fn from(ts: Timestamp) -> Self {
        ts.0
    }
}

impl TryFrom<SystemTime> for Timestamp {
    type Error = ();

    fn try_from(time: SystemTime) -> Result<Self, Self::Error> {
        match time.duration_since(UNIX_EPOCH) {
            Ok(duration) => Ok(Self::from_micros(duration.as_micros() as i64)),
            Err(e) => Ok(Self::from_micros(-(e.duration().as_micros() as i64))),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_timestamp_creation() {
        let ts = Timestamp::from_secs(1000);
        assert_eq!(ts.as_secs(), 1000);
        assert_eq!(ts.as_millis(), 1_000_000);
        assert_eq!(ts.as_micros(), 1_000_000_000);

        let ts = Timestamp::from_millis(1234);
        assert_eq!(ts.as_millis(), 1234);

        let ts = Timestamp::from_micros(1_234_567);
        assert_eq!(ts.as_micros(), 1_234_567);
    }

    #[test]
    fn test_timestamp_now() {
        let ts = Timestamp::now();
        // Should be after year 2020
        assert!(ts.as_secs() > 1_577_836_800);
    }

    #[test]
    fn test_timestamp_arithmetic() {
        let ts = Timestamp::from_secs(1000);

        let ts2 = ts.add_micros(1_000_000);
        assert_eq!(ts2.as_secs(), 1001);

        let ts3 = ts.sub_micros(1_000_000);
        assert_eq!(ts3.as_secs(), 999);

        assert_eq!(ts2.duration_since(ts), 1_000_000);
        assert_eq!(ts.duration_since(ts2), -1_000_000);
    }

    #[test]
    fn test_timestamp_ordering() {
        let ts1 = Timestamp::from_secs(100);
        let ts2 = Timestamp::from_secs(200);

        assert!(ts1 < ts2);
        assert!(ts2 > ts1);
        assert_eq!(ts1, Timestamp::from_secs(100));
    }

    #[test]
    fn test_timestamp_system_time_conversion() {
        let now = SystemTime::now();
        let ts: Timestamp = now.try_into().unwrap();
        let back = ts.as_system_time().unwrap();

        // Should be within 1 microsecond
        let diff = back
            .duration_since(now)
            .or_else(|e| Ok::<_, ()>(e.duration()))
            .unwrap();
        assert!(diff.as_micros() < 2);
    }

    #[test]
    fn test_timestamp_epoch() {
        assert_eq!(Timestamp::EPOCH.as_micros(), 0);
        assert_eq!(Timestamp::EPOCH.as_secs(), 0);
    }
}
