//! Timestamps for temporal properties.
//!
//! Stored as microseconds since Unix epoch - plenty of precision for most uses.

use super::date::civil_from_days;
use serde::{Deserialize, Serialize};
use std::fmt;
use std::time::{Duration as StdDuration, SystemTime, UNIX_EPOCH};

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
            .unwrap_or(StdDuration::ZERO);
        // reason: current wall-clock micros since epoch fit i64 for ~292,000 years
        #[allow(clippy::cast_possible_truncation)]
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
            // reason: self.0 is checked >= 0 on the line above
            #[allow(clippy::cast_sign_loss)]
            Some(UNIX_EPOCH + StdDuration::from_micros(self.0 as u64))
        } else {
            UNIX_EPOCH.checked_sub(StdDuration::from_micros(self.0.unsigned_abs()))
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

    /// Creates a timestamp from a date and time.
    #[must_use]
    pub fn from_date_time(date: super::Date, time: super::Time) -> Self {
        let day_micros = date.as_days() as i64 * 86_400_000_000;
        // reason: time nanos < 86.4e12, divided by 1000 is well within i64 range
        #[allow(clippy::cast_possible_wrap)]
        let time_micros = (time.as_nanos() / 1000) as i64;
        // If the time has an offset, subtract it to get UTC
        let offset_micros = time.offset_seconds().unwrap_or(0) as i64 * 1_000_000;
        Self(day_micros + time_micros - offset_micros)
    }

    /// Extracts the date component (UTC).
    #[must_use]
    pub fn to_date(self) -> super::Date {
        // reason: i64 micros / 86.4e9 yields a day count within i32 range for any valid timestamp
        #[allow(clippy::cast_possible_truncation)]
        let days = self.0.div_euclid(86_400_000_000) as i32;
        super::Date::from_days(days)
    }

    /// Extracts the time-of-day component (UTC).
    #[must_use]
    pub fn to_time(self) -> super::Time {
        let day_nanos = self.0.rem_euclid(86_400_000_000) as u64 * 1000;
        super::Time::from_nanos(day_nanos).unwrap_or_default()
    }

    /// Truncates this timestamp to the given unit.
    ///
    /// - `"year"`: truncates to midnight on January 1st
    /// - `"month"`: truncates to midnight on the 1st of the month
    /// - `"day"`: truncates to midnight (zeroes time component)
    /// - `"hour"`: zeroes minutes, seconds, microseconds
    /// - `"minute"`: zeroes seconds, microseconds
    /// - `"second"`: zeroes microseconds
    #[must_use]
    pub fn truncate(self, unit: &str) -> Option<Self> {
        match unit {
            "year" => {
                let date = self.to_date();
                let jan1 = super::Date::from_ymd(date.year(), 1, 1)?;
                Some(jan1.to_timestamp())
            }
            "month" => {
                let date = self.to_date();
                let first = super::Date::from_ymd(date.year(), date.month(), 1)?;
                Some(first.to_timestamp())
            }
            "day" => {
                let days = self.0.div_euclid(86_400_000_000);
                Some(Self(days * 86_400_000_000))
            }
            "hour" => {
                let days = self.0.div_euclid(86_400_000_000);
                let day_micros = self.0.rem_euclid(86_400_000_000);
                let hours = day_micros / 3_600_000_000;
                Some(Self(days * 86_400_000_000 + hours * 3_600_000_000))
            }
            "minute" => {
                let days = self.0.div_euclid(86_400_000_000);
                let day_micros = self.0.rem_euclid(86_400_000_000);
                let minutes = day_micros / 60_000_000;
                Some(Self(days * 86_400_000_000 + minutes * 60_000_000))
            }
            "second" => {
                let days = self.0.div_euclid(86_400_000_000);
                let day_micros = self.0.rem_euclid(86_400_000_000);
                let seconds = day_micros / 1_000_000;
                Some(Self(days * 86_400_000_000 + seconds * 1_000_000))
            }
            _ => None,
        }
    }

    /// Adds a temporal duration to this timestamp.
    #[must_use]
    pub fn add_duration(self, dur: &super::Duration) -> Self {
        // Add months via date arithmetic
        let date = self
            .to_date()
            .add_duration(&super::Duration::from_months(dur.months()));
        let time = self.to_time();
        let base = Self::from_date_time(date, time);
        // Add days and nanos directly
        let day_micros = dur.days() * 86_400_000_000;
        let nano_micros = dur.nanos() / 1000;
        Self(base.0 + day_micros + nano_micros)
    }
}

impl fmt::Debug for Timestamp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Timestamp({}μs)", self.0)
    }
}

impl fmt::Display for Timestamp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let micros = self.0;
        let micro_frac = micros.rem_euclid(1_000_000) as u64;

        // reason: i64 micros / 86.4e9 yields a day count within i32 range for any valid timestamp
        #[allow(clippy::cast_possible_truncation)]
        let total_days = micros.div_euclid(86_400_000_000) as i32;
        let day_micros = micros.rem_euclid(86_400_000_000);
        let day_secs = day_micros / 1_000_000;

        let hours = day_secs / 3600;
        let minutes = (day_secs % 3600) / 60;
        let seconds = day_secs % 60;

        let (year, month, day) = civil_from_days(total_days);

        write!(
            f,
            "{:04}-{:02}-{:02}T{:02}:{:02}:{:02}.{:06}Z",
            year, month, day, hours, minutes, seconds, micro_frac
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
            Ok(duration) => {
                let micros = i64::try_from(duration.as_micros()).map_err(|_| ())?;
                Ok(Self::from_micros(micros))
            }
            Err(e) => {
                let micros = i64::try_from(e.duration().as_micros()).map_err(|_| ())?;
                Ok(Self::from_micros(micros.checked_neg().ok_or(())?))
            }
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
    #[cfg(not(miri))] // SystemTime::now() requires clock_gettime, blocked by Miri isolation
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
    #[cfg(not(miri))] // SystemTime::now() requires clock_gettime, blocked by Miri isolation
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
    fn test_truncate() {
        // 2024-06-15T14:30:45.123456Z
        let date = crate::types::Date::from_ymd(2024, 6, 15).unwrap();
        let time = crate::types::Time::from_hms_nano(14, 30, 45, 123_456_000).unwrap();
        let ts = Timestamp::from_date_time(date, time);

        let year = ts.truncate("year").unwrap();
        assert_eq!(year.to_date().to_string(), "2024-01-01");
        assert_eq!(year.to_time().hour(), 0);

        let month = ts.truncate("month").unwrap();
        assert_eq!(month.to_date().to_string(), "2024-06-01");
        assert_eq!(month.to_time().hour(), 0);

        let day = ts.truncate("day").unwrap();
        assert_eq!(day.to_date().to_string(), "2024-06-15");
        assert_eq!(day.to_time().hour(), 0);

        let hour = ts.truncate("hour").unwrap();
        assert_eq!(hour.to_time().hour(), 14);
        assert_eq!(hour.to_time().minute(), 0);

        let minute = ts.truncate("minute").unwrap();
        assert_eq!(minute.to_time().hour(), 14);
        assert_eq!(minute.to_time().minute(), 30);
        assert_eq!(minute.to_time().second(), 0);

        let second = ts.truncate("second").unwrap();
        assert_eq!(second.to_time().second(), 45);
        assert_eq!(second.to_time().nanosecond(), 0);

        assert!(ts.truncate("invalid").is_none());
    }

    #[test]
    fn test_timestamp_epoch() {
        assert_eq!(Timestamp::EPOCH.as_micros(), 0);
        assert_eq!(Timestamp::EPOCH.as_secs(), 0);
    }

    #[test]
    fn test_add_duration_days_and_nanos() {
        use crate::types::Duration;
        let ts = Timestamp::from_secs(1_700_000_000); // 2023-11-14T22:13:20Z
        let dur = Duration::from_days(1);
        let result = ts.add_duration(&dur);
        // Adding 1 day = 86400 seconds = 86_400_000_000 microseconds
        assert_eq!(result.as_micros() - ts.as_micros(), 86_400_000_000);
    }

    #[test]
    fn test_add_duration_months() {
        use crate::types::Duration;
        let ts = Timestamp::from_secs(1_700_000_000); // 2023-11-14
        let dur = Duration::from_months(2);
        let result = ts.add_duration(&dur);
        let result_date = result.to_date();
        // Nov + 2 months = January (next year)
        assert_eq!(result_date.month(), 1);
        assert_eq!(result_date.year(), 2024);
    }
}
