//! Zoned datetime type: an instant in time with a fixed UTC offset.
//!
//! [`ZonedDatetime`] pairs a UTC microsecond timestamp with a numeric
//! offset (in seconds). Two values representing the same instant are
//! equal regardless of offset.

use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::fmt;
use std::hash::{Hash, Hasher};

use super::{Date, Time, Timestamp};

/// A datetime with a fixed UTC offset.
///
/// Internally stores the instant as microseconds since the Unix epoch
/// (same as [`Timestamp`]) plus an offset in seconds from UTC. Two
/// `ZonedDatetime` values are equal when they refer to the same instant,
/// even if their offsets differ.
///
/// # Examples
///
/// ```
/// use grafeo_common::types::ZonedDatetime;
///
/// let zdt = ZonedDatetime::parse("2024-06-15T10:30:00+05:30").unwrap();
/// assert_eq!(zdt.offset_seconds(), 19800);
/// assert_eq!(zdt.to_string(), "2024-06-15T10:30:00+05:30");
/// ```
#[derive(Clone, Copy, Serialize, Deserialize)]
pub struct ZonedDatetime {
    /// Microseconds since Unix epoch (UTC).
    utc_micros: i64,
    /// Fixed offset from UTC in seconds (e.g., +19800 for +05:30).
    offset_seconds: i32,
}

impl ZonedDatetime {
    /// Creates a `ZonedDatetime` from a UTC timestamp and an offset in seconds.
    #[inline]
    #[must_use]
    pub const fn from_timestamp_offset(ts: Timestamp, offset_seconds: i32) -> Self {
        Self {
            utc_micros: ts.as_micros(),
            offset_seconds,
        }
    }

    /// Creates a `ZonedDatetime` from a date and a time with a required offset.
    ///
    /// Returns `None` if the time has no offset.
    #[must_use]
    pub fn from_date_time(date: Date, time: Time) -> Option<Self> {
        let offset = time.offset_seconds()?;
        let ts = Timestamp::from_date_time(date, time);
        Some(Self {
            utc_micros: ts.as_micros(),
            offset_seconds: offset,
        })
    }

    /// Returns the underlying UTC timestamp.
    #[inline]
    #[must_use]
    pub const fn as_timestamp(&self) -> Timestamp {
        Timestamp::from_micros(self.utc_micros)
    }

    /// Returns the UTC offset in seconds.
    #[inline]
    #[must_use]
    pub const fn offset_seconds(&self) -> i32 {
        self.offset_seconds
    }

    /// Returns the date in the local (offset-adjusted) timezone.
    #[must_use]
    pub fn to_local_date(&self) -> Date {
        let local_micros = self.utc_micros + self.offset_seconds as i64 * 1_000_000;
        Timestamp::from_micros(local_micros).to_date()
    }

    /// Returns the time-of-day in the local (offset-adjusted) timezone,
    /// carrying the offset.
    #[must_use]
    pub fn to_local_time(&self) -> Time {
        let local_micros = self.utc_micros + self.offset_seconds as i64 * 1_000_000;
        Timestamp::from_micros(local_micros)
            .to_time()
            .with_offset(self.offset_seconds)
    }

    /// Truncates this zoned datetime to the given unit, preserving the offset.
    ///
    /// Truncation is performed on the local (offset-adjusted) time, then
    /// converted back to UTC. This ensures that truncating to "day" gives
    /// midnight in the local timezone, not midnight UTC.
    #[must_use]
    pub fn truncate(&self, unit: &str) -> Option<Self> {
        // Convert to local, truncate, convert back
        let local_micros = self.utc_micros + self.offset_seconds as i64 * 1_000_000;
        let local_ts = Timestamp::from_micros(local_micros);
        let truncated_local = local_ts.truncate(unit)?;
        let utc_micros = truncated_local.as_micros() - self.offset_seconds as i64 * 1_000_000;
        Some(Self {
            utc_micros,
            offset_seconds: self.offset_seconds,
        })
    }

    /// Parses an ISO 8601 datetime string with a mandatory UTC offset.
    ///
    /// Accepted formats:
    /// - `YYYY-MM-DDTHH:MM:SS+HH:MM`
    /// - `YYYY-MM-DDTHH:MM:SS.fff+HH:MM`
    /// - `YYYY-MM-DDTHH:MM:SSZ`
    ///
    /// Returns `None` if the string is not valid or has no offset.
    #[must_use]
    pub fn parse(s: &str) -> Option<Self> {
        let pos = s.find('T').or_else(|| s.find('t'))?;
        let date = Date::parse(&s[..pos])?;
        let time = Time::parse(&s[pos + 1..])?;
        // Require an explicit offset
        time.offset_seconds()?;
        Self::from_date_time(date, time)
    }
}

impl PartialEq for ZonedDatetime {
    fn eq(&self, other: &Self) -> bool {
        self.utc_micros == other.utc_micros
    }
}

impl Eq for ZonedDatetime {}

impl Hash for ZonedDatetime {
    fn hash<H: Hasher>(&self, state: &mut H) {
        // Hash by UTC instant only, consistent with PartialEq
        self.utc_micros.hash(state);
    }
}

impl Ord for ZonedDatetime {
    fn cmp(&self, other: &Self) -> Ordering {
        self.utc_micros.cmp(&other.utc_micros)
    }
}

impl PartialOrd for ZonedDatetime {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl fmt::Debug for ZonedDatetime {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ZonedDatetime({})", self)
    }
}

impl fmt::Display for ZonedDatetime {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let local_micros = self.utc_micros + self.offset_seconds as i64 * 1_000_000;
        let ts = Timestamp::from_micros(local_micros);
        let date = ts.to_date();
        let time = ts.to_time();

        let (year, month, day) = (date.year(), date.month(), date.day());
        let (h, m, s) = (time.hour(), time.minute(), time.second());
        let micro_frac = local_micros.rem_euclid(1_000_000) as u64;

        if micro_frac > 0 {
            // Trim trailing zeros from fractional seconds
            let frac = format!("{micro_frac:06}");
            let trimmed = frac.trim_end_matches('0');
            write!(
                f,
                "{year:04}-{month:02}-{day:02}T{h:02}:{m:02}:{s:02}.{trimmed}"
            )?;
        } else {
            write!(f, "{year:04}-{month:02}-{day:02}T{h:02}:{m:02}:{s:02}")?;
        }

        match self.offset_seconds {
            0 => write!(f, "Z"),
            off => {
                let sign = if off >= 0 { '+' } else { '-' };
                let abs = off.unsigned_abs();
                let oh = abs / 3600;
                let om = (abs % 3600) / 60;
                write!(f, "{sign}{oh:02}:{om:02}")
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_utc() {
        let zdt = ZonedDatetime::parse("2024-06-15T10:30:00Z").unwrap();
        assert_eq!(zdt.offset_seconds(), 0);
        assert_eq!(zdt.to_string(), "2024-06-15T10:30:00Z");
    }

    #[test]
    fn test_parse_positive_offset() {
        let zdt = ZonedDatetime::parse("2024-06-15T10:30:00+05:30").unwrap();
        assert_eq!(zdt.offset_seconds(), 19800);
        assert_eq!(zdt.to_string(), "2024-06-15T10:30:00+05:30");
    }

    #[test]
    fn test_parse_negative_offset() {
        let zdt = ZonedDatetime::parse("2024-06-15T10:30:00-04:00").unwrap();
        assert_eq!(zdt.offset_seconds(), -14400);
        assert_eq!(zdt.to_string(), "2024-06-15T10:30:00-04:00");
    }

    #[test]
    fn test_equality_same_instant() {
        // Same instant, different offsets: should be equal
        let z1 = ZonedDatetime::parse("2024-06-15T15:30:00+05:30").unwrap();
        let z2 = ZonedDatetime::parse("2024-06-15T10:00:00Z").unwrap();
        assert_eq!(z1, z2);
    }

    #[test]
    fn test_no_offset_fails() {
        assert!(ZonedDatetime::parse("2024-06-15T10:30:00").is_none());
    }

    #[test]
    fn test_local_date_time() {
        let zdt = ZonedDatetime::parse("2024-06-15T23:30:00+05:30").unwrap();
        let local_date = zdt.to_local_date();
        assert_eq!(local_date.year(), 2024);
        assert_eq!(local_date.month(), 6);
        assert_eq!(local_date.day(), 15);

        let local_time = zdt.to_local_time();
        assert_eq!(local_time.hour(), 23);
        assert_eq!(local_time.minute(), 30);
        assert_eq!(local_time.offset_seconds(), Some(19800));
    }

    #[test]
    fn test_from_timestamp_offset() {
        let ts = Timestamp::from_secs(1_718_444_400); // 2024-06-15T09:40:00Z
        let zdt = ZonedDatetime::from_timestamp_offset(ts, 3600); // +01:00
        assert_eq!(zdt.as_timestamp(), ts);
        assert_eq!(zdt.offset_seconds(), 3600);
        assert_eq!(zdt.to_string(), "2024-06-15T10:40:00+01:00");
    }

    #[test]
    fn test_ordering() {
        let earlier = ZonedDatetime::parse("2024-06-15T10:00:00Z").unwrap();
        let later = ZonedDatetime::parse("2024-06-15T12:00:00Z").unwrap();
        assert!(earlier < later);
    }

    #[test]
    fn test_truncate() {
        // 2024-06-15T14:30:45+02:00 (Amsterdam summer time)
        let zdt = ZonedDatetime::parse("2024-06-15T14:30:45+02:00").unwrap();

        let day = zdt.truncate("day").unwrap();
        assert_eq!(day.offset_seconds(), 7200);
        // Truncated to midnight local time
        assert_eq!(day.to_string(), "2024-06-15T00:00:00+02:00");

        let hour = zdt.truncate("hour").unwrap();
        assert_eq!(hour.to_string(), "2024-06-15T14:00:00+02:00");

        let minute = zdt.truncate("minute").unwrap();
        assert_eq!(minute.to_string(), "2024-06-15T14:30:00+02:00");
    }

    #[test]
    fn test_with_fractional_seconds() {
        let zdt = ZonedDatetime::parse("2024-06-15T10:30:00.123Z").unwrap();
        assert_eq!(zdt.to_string(), "2024-06-15T10:30:00.123Z");
    }
}
